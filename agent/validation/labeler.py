"""Background period-activity labeling: key frames -> labels.yaml activity.

The tracking path sees detection metadata only — no pixels. This worker owns
an MCP connection to the images server, so it can look at each closed
period's key frames and judge the cat's activity with one vision LLM call
(:meth:`MetaModelClient.predict_activity`) against the activity set in
``agent/agent_data/config/labels.yaml``. Results go to ``sink`` as
``(period_id, ActivityLabel, resolved_keyframes)``.

Requested keyframe names are synthesized from timestamps and may not exist
verbatim; each is resolved to the nearest real frame in its minute bucket
(via ``images__list_images``) before fetching. Everything is best-effort:
fetch/LLM failures deliver a degraded unknown label, never raise, and a
missing images backend degrades every job without blocking the run.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import queue
import threading

from agent.memory.activity import ActivityLabel

logger = logging.getLogger(__name__)

_SENTINEL = object()
_LIST_LIMIT = 500  # covers a full minute bucket at several fps


def _parse_image_time(name: str) -> datetime.datetime | None:
    try:
        return datetime.datetime.strptime(name[:26], "%Y-%m-%d-%H-%M-%S-%f")
    except (ValueError, TypeError):
        return None


def _minute_prefix(name: str) -> str | None:
    if len(name) >= 16 and _parse_image_time(name) is not None:
        return name[:16]
    return None


def _text_blocks(result) -> list[str]:
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            continue
        if type(block).__name__ == "TextContent" and isinstance(getattr(block, "text", None), str):
            texts.append(block.text)
    return texts


def _image_bytes_list(result) -> list[bytes]:
    out: list[bytes] = []
    for block in getattr(result, "content", None) or []:
        data = block.get("data") if isinstance(block, dict) else getattr(block, "data", None)
        if not isinstance(data, str) or not data:
            continue
        try:
            out.append(base64.b64decode(data))
        except Exception:
            continue
    return out


def _listed_names(result) -> list[str]:
    names: list[str] = []
    for text in _text_blocks(result):
        for line in text.splitlines():
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(payload, dict):
                for name in payload.get("images", []) or []:
                    if isinstance(name, str) and name not in names:
                        names.append(name)
    return names


class PeriodLabeler:
    """Label one period's activity per job, on a background thread.

    ``sink(period_id, label, resolved_keyframes)`` receives every result
    (degraded unknown on any failure). ``llm_client`` defaults to a shared
    ``MetaModelClient`` (role ``memory``); ``hub_factory`` builds the images
    MCP hub (injectable for tests).
    """

    def __init__(
        self,
        *,
        sink,
        llm_client=None,
        candidates: list[dict] | None = None,
        hub_factory=None,
        max_frames: int = 3,
    ):
        self._sink = sink
        self._llm_client = llm_client
        self._candidates = candidates
        self._hub_factory = hub_factory or _default_hub
        self._max_frames = max(1, int(max_frames))
        self._inbox: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._listings: dict[str, list[str]] = {}

    # -- control (tracking thread) --

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="period-labeler", daemon=True)
        self._thread.start()

    def label_period(
        self,
        period_id: int | None,
        *,
        cat_id: int | None,
        cat_name: str | None,
        span: str,
        keyframes: list[str],
        boxes: list,
    ) -> None:
        """Queue one period for labeling (thread-safe, non-blocking)."""
        self._inbox.put(
            {
                "period_id": period_id,
                "cat_id": cat_id,
                "cat_name": cat_name,
                "span": span,
                "keyframes": list(keyframes or [])[: self._max_frames],
                "boxes": list(boxes or []),
            }
        )

    def stop(self, timeout: float | None = None) -> None:
        """Finish queued jobs, stop the thread."""
        self._inbox.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- loop (worker thread) --

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception:
            logger.exception("period labeler crashed; queued periods stay unlabeled")

    async def _serve(self) -> None:
        if self._llm_client is None:
            try:
                from agent.memory import MetaModelClient, MemoryConfig

                cfg = MemoryConfig.from_yaml()
                self._llm_client = MetaModelClient(role=cfg.llm_model_role, trace_dir=cfg.trace_dir)
            except Exception as exc:
                logger.warning("period labeler LLM unavailable (%s); labels degrade to unknown", exc)
        if self._candidates is None:
            try:
                from agent.memory.activity import load_activity_labels

                self._candidates = load_activity_labels()["activity"]
            except Exception as exc:
                logger.warning("period labeler label set unreadable (%s); labels degrade to unknown", exc)
                self._candidates = []
        try:
            hub = self._hub_factory()
        except Exception as exc:
            logger.warning("period labeler images backend unavailable (%s); labels degrade to unknown", exc)
            hub = None
        if hub is None:
            await self._consume(None)
            return
        try:
            async with hub as connected:
                await self._consume(connected)
        except Exception:
            logger.exception("period labeler hub failed; remaining jobs degrade to unknown")
            await self._consume(None)

    async def _consume(self, hub) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job = await loop.run_in_executor(None, self._inbox.get)
            try:
                if job is _SENTINEL:
                    return
                label, resolved = await self._label_one(hub, job)
            except Exception as exc:  # noqa: BLE001 - one bad job never stops the loop
                logger.exception("period labeling failed; delivering unknown")
                label, resolved = ActivityLabel(activity=None, degraded=True), []
                job = job if isinstance(job, dict) else {}
            try:
                self._sink(job.get("period_id"), label, resolved)
            except Exception:
                logger.exception("period label sink failed; dropping result")
            finally:
                self._inbox.task_done()

    async def _label_one(self, hub, job: dict) -> tuple[ActivityLabel, list[str]]:
        pid = job.get("period_id")
        if hub is None or self._llm_client is None or not self._candidates:
            return ActivityLabel(activity=None, degraded=True), []
        frames: list[tuple[str, object, bytes]] = []  # (name, box, jpeg)
        names = [str(k) for k in job.get("keyframes", []) or []]
        boxes = list(job.get("boxes", []) or [])
        for i, name in enumerate(names):
            try:
                real = await self._resolve(hub, name)
                if real is None:
                    continue
                outcome = await hub.call("images__get_image", {"image_name": real})
                if outcome.error:
                    logger.warning("period %s keyframe %s fetch failed: %s", pid, real, outcome.error)
                    continue
                blobs = _image_bytes_list(outcome.result)
                if not blobs:
                    continue
                frames.append((real, boxes[i] if i < len(boxes) else None, blobs[0]))
            except Exception as exc:
                logger.warning("period %s keyframe %s failed: %s", pid, name, exc)
        if not frames:
            logger.warning("period %s: no key frames fetchable; label degrades to unknown", pid)
            return ActivityLabel(activity=None, degraded=True), []
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(
                None,
                lambda: self._llm_client.predict_activity(
                    images=[b for _, _, b in frames],
                    cat_name=job.get("cat_name"),
                    cat_id=job.get("cat_id"),
                    span=str(job.get("span") or ""),
                    boxes=[b for _, b, _ in frames],
                    keyframe_names=[n for n, _, _ in frames],
                    candidates=self._candidates,
                    label=f"period {pid}",
                ),
            )
        except Exception as exc:
            logger.warning("period %s activity LLM failed (%s); label degrades to unknown", pid, exc)
            return ActivityLabel(activity=None, degraded=True), [n for n, _, _ in frames]
        if not isinstance(out, dict):
            return ActivityLabel(activity=None, degraded=True), [n for n, _, _ in frames]
        try:
            confidence = float(out.get("activity_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        label = ActivityLabel(
            activity=out.get("activity") or None,
            confidence=min(max(confidence, 0.0), 1.0),
            appearance=str(out.get("appearance", "") or ""),
            background=str(out.get("background", "") or ""),
        )
        logger.info("period %s activity: %s (confidence %.2f)", pid, label.activity, label.confidence or 0.0)
        return label, [n for n, _, _ in frames]

    async def _resolve(self, hub, name: str) -> str | None:
        """Nearest real frame in the name's minute bucket (None when unresolvable)."""
        prefix = _minute_prefix(name)
        want = _parse_image_time(name)
        if prefix is None or want is None:
            return None
        if prefix not in self._listings:
            outcome = await hub.call("images__list_images", {"prefix": prefix, "limit": _LIST_LIMIT})
            if outcome.error:
                logger.warning("keyframe listing %s failed: %s", prefix, outcome.error)
                return None
            self._listings[prefix] = _listed_names(outcome.result)
        cands = [(abs((_parse_image_time(n) or want) - want), n) for n in self._listings[prefix]]
        cands = [(d, n) for d, n in cands if _parse_image_time(n) is not None]
        if not cands:
            return None
        return min(cands)[1]


def _default_hub():
    """Images-only MCP hub from the agent environment (None when unavailable)."""
    from agent.validation.worker import ensure_supervisor_importable

    ensure_supervisor_importable()
    from supervisor_agent.config import AgentConfig
    from supervisor_agent.mcp_hub import MCPHub

    cfg = AgentConfig.from_env()
    servers = [s for s in cfg.servers if s.name == "images"]
    if not servers:
        return None
    return MCPHub(servers)


__all__ = ["PeriodLabeler"]
