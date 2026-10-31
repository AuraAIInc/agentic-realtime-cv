"""Long-term spatial-temporal memory (phase 3: LLM-backed store/promote).

Writes (``store``/``promote``/``bootstrap``) enqueue worker jobs and return
immediately; reads (``invoke``/``invoke_for_escalation``/``semantic_search``)
are synchronous locked accesses. Store merges and promotion summaries go
through the configured ``LLMClient`` (real Meta Model API client or the
offline stub). Conflict judgement is the LLM's job; counting, caps, and
thresholds are programmatic. Any LLM failure degrades the affected unit to
pending-observation mode (raw text kept in ``recent``) and never raises into
callers.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from agent.memory.llm_client import (
    WORD_CAPS,
    LLMClient,
    StubLLMClient,
    extract_keywords,
    truncate_words,
)
from agent.memory.memory_config import MemoryConfig
from agent.memory.models import (
    JobReceipt,
    MemoryContext,
    MemoryEvent,
    MemoryHit,
    MemoryQuery,
    PromoteResult,
    StoreResult,
)
from agent.memory.spatial import SpatialIndex
from agent.memory.store import MemoryStore, SpatialUnit, TemporalUnit
from agent.memory.temporal import TemporalIndex
from agent.memory.worker import MemoryWorker

if TYPE_CHECKING:
    from agent.escalation.escalation import Escalation

logger = logging.getLogger(__name__)

_LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "slot_5min": 3, "hour": 4, "day": 5, "week": 6}

# Escalation code -> retrieval family (design doc mapping table).
ESCALATION_FAMILIES = {
    "T1": "identity",
    "T2": "identity",
    "T5": "identity",
    "T4": "motion",
    "T6": "motion",
    "T3": "continuity",
    "T7": "continuity",
    "P1": "routine",
    "P2": "routine",
    "P5": "routine",
    "P3": "conflict",
    "P4": "occlusion",
    "P6": "merge",
    "P7": "merge",
}


class LongTermMemory:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        store_path=None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config if config is not None else MemoryConfig.from_yaml()
        self.spatial = SpatialIndex(self.config)
        self.temporal = TemporalIndex(self.config)
        self.mem_store = MemoryStore(self.config, path=store_path)
        self.mem_store.load()
        self.llm: LLMClient = llm_client if llm_client is not None else StubLLMClient()
        self.worker = MemoryWorker(
            self.config,
            {
                "store": self._do_store,
                "promote": self._do_promote,
                "bootstrap": self._do_bootstrap,
            },
        )
        self.last_store: datetime.datetime | None = None
        self._dirty: set[tuple[str, str, str]] = set()
        self._bootstrap_pending = False

    # ---- async writes (enqueue and return) ----

    def store(self, event: MemoryEvent, image: bytes | None = None) -> JobReceipt:
        self.last_store = event.time  # observation arrival, not LLM completion
        if image is not None and not self.mem_store.bootstrapped and not self._bootstrap_pending:
            self._bootstrap_pending = True
            self.worker.submit("bootstrap", {"image": image})
        return self.worker.submit("store", {"event": event, "has_image": image is not None})

    def promote(self, force: bool = False) -> JobReceipt:
        return self.worker.submit("promote", {"force": force})

    def bootstrap(self, image: bytes) -> JobReceipt:
        self._bootstrap_pending = True
        return self.worker.submit("bootstrap", {"image": image})

    def tick(self, now: datetime.datetime) -> JobReceipt | None:
        """Enqueue an idle promotion when nothing was stored for ``idle_seconds``."""
        if self.last_store is None:
            return None
        if (now - self.last_store).total_seconds() >= self.config.idle_seconds:
            return self.promote()
        return None

    def drain(self, timeout: float | None = None) -> None:
        self.worker.drain(timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        self.worker.shutdown(timeout)

    def reset(self, keep_backup: bool = True) -> None:
        """Drain, clear the tree and YAML (optional timestamped ``.bak``), keep
        the worker running."""
        self.worker.drain()
        with self.worker.lock:
            path = self.mem_store.path
            if path.exists():
                if keep_backup:
                    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    path.rename(path.parent / f"{path.name}.{stamp}.bak")
                else:
                    path.unlink()
            self.mem_store.clear()
            self.mem_store.save()
            self.last_store = None
            self._dirty.clear()
            self._bootstrap_pending = False

    # ---- sync reads (locked) ----

    def invoke(self, query: MemoryQuery) -> MemoryContext:
        with self.worker.lock:
            spatial_hits: list[MemoryHit] = []
            if query.box is not None:
                hit = self.spatial.box_to_cell(query.box, 0)
                if hit is not None:
                    col, row, _cid = hit
                    for cc, cr, ncid in self.spatial.neighbors(
                        0, col, row, query.spatial_radius
                    ):
                        view, unobserved = self.mem_store.effective_l0(ncid, self.spatial)
                        text = view.get("activity") or view.get("environment") or ""
                        dist = max(abs(cc - col), abs(cr - row))
                        spatial_hits.append(
                            MemoryHit(
                                level="L0",
                                unit_id=ncid,
                                text=text,
                                score=1.0 / (1.0 + dist),
                                unobserved=unobserved,
                            )
                        )
                    parent = self.spatial.parent(0, col, row)
                    if parent is not None:
                        _, _pc, _pr, pid = parent
                        punit = self.mem_store.get_spatial("L1", pid)
                        if punit is not None and not punit.is_empty:
                            spatial_hits.append(
                                MemoryHit(
                                    level="L1",
                                    unit_id=pid,
                                    text=punit.activity or punit.environment or "",
                                    score=0.5,
                                )
                            )
                    l2 = self.mem_store.get_spatial("L2", "r0c0")
                    if l2 is not None and not l2.is_empty:
                        spatial_hits.append(
                            MemoryHit(
                                level="L2",
                                unit_id="r0c0",
                                text=l2.activity or l2.environment or "",
                                score=0.25,
                            )
                        )
            temporal_hits: list[MemoryHit] = []
            for sid in self.temporal.same_slot_history(query.time, query.history_days):
                unit = self.mem_store.get_temporal("slot_5min", sid)
                if unit is not None and not unit.is_empty:
                    temporal_hits.append(
                        MemoryHit(level="slot_5min", unit_id=sid, text=unit.summary, score=1.0)
                    )
            for lname, bid in (
                ("hour", self.temporal.hour_id(query.time)),
                ("day", self.temporal.day_id(query.time)),
                ("week", self.temporal.week_id(query.time)),
            ):
                unit = self.mem_store.get_temporal(lname, bid)
                if unit is not None and not unit.is_empty:
                    temporal_hits.append(
                        MemoryHit(level=lname, unit_id=bid, text=unit.summary, score=0.5)
                    )
            semantic = self.semantic_search(query.text) if query.text else []
            brief = self._render_brief(query, spatial_hits, temporal_hits, semantic)
            return MemoryContext(
                spatial=spatial_hits, temporal=temporal_hits, semantic=semantic, brief=brief
            )

    def invoke_for_escalation(
        self,
        esc: "Escalation",
        box: tuple[float, float, float, float] | None = None,
    ) -> MemoryContext:
        """Answer an escalation with the code family's spatial+temporal context.

        Families (design doc table): identity T1/T2/T5 (hit cell + neighbors),
        motion T4/T6 (wider cells + current slot/hour), continuity T3/T7
        (current slot + hour), routine P1/P2/P5 (day/hour + history), conflict
        P3 (same-slot history), occlusion P4 (environment + contrast), merge
        P6/P7 (gap span + place routine). Metrics carry no boxes, so T4/T6
        cover the path with a wider neighbor radius around ``box``.
        """
        code = str(getattr(esc, "code", "") or "").upper()
        family = ESCALATION_FAMILIES.get(code, "generic")
        when = getattr(esc, "time", None) or datetime.datetime.now()
        cat_id = getattr(esc, "cat_id", None)
        radius = self.config.spatial_chebyshev_radius
        history = self.config.temporal_history_days
        if family == "motion":
            query = MemoryQuery(
                time=when, box=box, cat_id=cat_id,
                spatial_radius=radius + 1, history_days=0,
            )
        elif family == "continuity":
            query = MemoryQuery(
                time=when, box=box, cat_id=cat_id,
                spatial_radius=0, history_days=0,
            )
        elif family in ("routine", "conflict"):
            query = MemoryQuery(
                time=when, box=box, cat_id=cat_id,
                spatial_radius=radius, history_days=history,
            )
        elif family == "merge":
            query = MemoryQuery(
                time=when, box=box, cat_id=cat_id,
                spatial_radius=radius, history_days=1,
            )
        else:  # identity, occlusion, generic
            query = MemoryQuery(
                time=when, box=box, cat_id=cat_id,
                spatial_radius=radius, history_days=1,
            )
        ctx = self.invoke(query)
        if family == "merge":
            end = getattr(esc, "end_time", None) or when
            with self.worker.lock:
                extra = self._span_temporal(when, end)
            seen = {(h.level, h.unit_id) for h in ctx.temporal}
            for hit in extra:
                if (hit.level, hit.unit_id) not in seen:
                    seen.add((hit.level, hit.unit_id))
                    ctx.temporal.append(hit)
        ctx.brief = f"[{code or '?'} {family}] {ctx.brief}"
        return ctx

    def _span_temporal(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> list[MemoryHit]:
        """Non-empty buckets covering ``[start, end]`` for gap-span queries."""
        if end < start:
            start, end = end, start
        hits: list[MemoryHit] = []
        seen: set[tuple[str, str]] = set()
        slot_minutes = self.temporal.minutes(self.temporal.level_names[0])
        steps = (
            ("slot_5min", datetime.timedelta(minutes=slot_minutes), 200),
            ("hour", datetime.timedelta(hours=1), 48),
            ("day", datetime.timedelta(days=1), 60),
        )
        for level, step, cap in steps:
            cur, n = start, 0
            while cur <= end and n < cap:
                bid = self.temporal.bucket_id(level, cur)
                if (level, bid) not in seen:
                    seen.add((level, bid))
                    unit = self.mem_store.get_temporal(level, bid)
                    if unit is not None and not unit.is_empty:
                        hits.append(
                            MemoryHit(
                                level=level, unit_id=bid,
                                text=unit.summary, score=0.5,
                            )
                        )
                cur += step
                n += 1
        return hits

    def semantic_search(self, text: str, top_k: int | None = None) -> list[MemoryHit]:
        """Keyword search over stored units: keyword-overlap score plus a
        full-phrase bonus, spatial levels before temporal on ties."""
        if top_k is None:
            top_k = self.config.semantic_top_k
        with self.worker.lock:
            tokens = set(extract_keywords(text))
            if not tokens:
                return []
            needle = text.lower()
            scored: list[MemoryHit] = []
            for label in self.mem_store.spatial_labels:
                for uid, unit in self.mem_store.spatial[label].items():
                    if unit.is_empty:
                        continue
                    hay = " ".join(
                        [unit.environment, unit.background, unit.cat_contrast, unit.activity]
                    )
                    scored.append(
                        self._score_unit(
                            label, uid, unit.activity or unit.environment,
                            hay, unit.keywords, tokens, needle,
                        )
                    )
            for lname in self.mem_store.temporal_names:
                for uid, unit in self.mem_store.temporal[lname].items():
                    if unit.is_empty:
                        continue
                    scored.append(
                        self._score_unit(
                            lname, uid, unit.summary, unit.summary,
                            unit.keywords, tokens, needle,
                        )
                    )
            hits = [h for h in scored if h.score > 0]
            hits.sort(key=lambda h: (-h.score, _LEVEL_RANK.get(h.level, 99), h.unit_id))
            return hits[: max(0, top_k)]

    @staticmethod
    def _score_unit(
        level: str,
        uid: str,
        display: str,
        haystack: str,
        keywords: list[str],
        tokens: set[str],
        needle: str,
    ) -> MemoryHit:
        unit_tokens = set(extract_keywords(haystack)) | {k.lower() for k in keywords}
        score = float(len(tokens & unit_tokens))
        if needle and needle in haystack.lower():
            score += 0.5
        return MemoryHit(level=level, unit_id=uid, text=display, score=score)

    def _render_brief(
        self,
        query: MemoryQuery,
        spatial: list[MemoryHit],
        temporal: list[MemoryHit],
        semantic: list[MemoryHit],
    ) -> str:
        parts: list[str] = []
        if spatial:
            observed = sum(1 for h in spatial if not h.unobserved)
            parts.append(f"{len(spatial)} spatial cells ({observed} observed)")
            first = next((h.text for h in spatial if h.text), "")
            if first:
                parts.append(f"place: {first[:120]}")
        else:
            parts.append("no spatial anchor")
        if temporal:
            ids = ", ".join(h.unit_id for h in temporal[:4])
            parts.append(f"{len(temporal)} temporal buckets ({ids})")
            if temporal[0].text:
                parts.append(f"recent: {temporal[0].text[:120]}")
        else:
            parts.append("no temporal history")
        if semantic:
            parts.append(f"{len(semantic)} semantic hits (top: {semantic[0].unit_id})")
        return "; ".join(parts) + "."

    # ---- worker jobs (run on the worker thread under the write lock) ----

    def _do_store(self, payload: dict) -> StoreResult:
        event: MemoryEvent = payload["event"]
        with self.worker.lock:
            spatial_ids: list[str] = []
            temporal_ids: list[str] = []
            if event.box is not None:
                hit = self.spatial.box_to_cell(event.box, 0)
                if hit is not None:
                    col, row, cid = hit
                    unit = self.mem_store.get_spatial("L0", cid)
                    if unit is None:
                        unit = SpatialUnit(
                            id=cid, bounds=self.spatial.cell_bounds(0, col, row)
                        )
                    self._merge_spatial_text(unit, event)
                    if not self.mem_store.bootstrapped:
                        self._mark_vision_unknown(unit)
                    self.mem_store.put_spatial("L0", unit)
                    spatial_ids.append(f"L0:{cid}")
                    self._dirty.add(("spatial", "L0", cid))
            slot = self.temporal.slot_id(event.time)
            tunit = self.mem_store.get_temporal("slot_5min", slot)
            if tunit is None:
                tunit = TemporalUnit(id=slot)
            self._merge_temporal_text(tunit, event)
            self.mem_store.put_temporal("slot_5min", tunit)
            temporal_ids.append(f"slot_5min:{slot}")
            self._dirty.add(("temporal", "slot_5min", slot))
            self.mem_store.save()
            return StoreResult(
                spatial_ids=spatial_ids, temporal_ids=temporal_ids, persisted=True
            )

    def _do_promote(self, payload: dict) -> PromoteResult:
        """Bottom-up promotion over dirty subtrees (``force``: whole tree).

        Stats roll up programmatically first; the LLM then rewrites each
        parent's stable text from children with support >=
        ``min_pattern_support``. At most ``max_units_per_call`` children per
        LLM call and the same number of LLM calls per promotion; parents past
        the budget stay dirty and count as deferred. Per-unit LLM errors keep
        the prior text and are reported, never raised.
        """
        force = bool(payload.get("force", False))
        budget = max(1, self.config.max_units_per_call)
        state = {"calls": 0}
        errors: list[str] = []
        with self.worker.lock:
            promoted_sp, deferred_sp = self._promote_spatial(force, budget, state, errors)
            promoted_tp, deferred_tp = self._promote_temporal(force, budget, state, errors)
            self.mem_store.save()
            return PromoteResult(
                promoted=promoted_sp + promoted_tp,
                deferred=deferred_sp + deferred_tp,
                errors=errors,
            )

    def _do_bootstrap(self, payload: dict) -> dict:
        with self.worker.lock:
            self._bootstrap_pending = False
            if self.mem_store.bootstrapped:
                return {"bootstrapped": True, "noop": True}
            now = datetime.datetime.now().isoformat(timespec="seconds")
            image = payload.get("image")
            try:
                if image is None:
                    raise ValueError("no image for vision bootstrap")
                res = self.llm.bootstrap(bytes(image))
                if not isinstance(res, dict):
                    raise ValueError("bootstrap LLM returned a non-dict")
            except Exception as exc:  # noqa: BLE001 - degrade, never raise
                logger.warning(
                    "memory bootstrap vision failed (%s); using text-only mode", exc
                )
                self._fill_vision_placeholder(now)
                self.mem_store.bootstrapped = True
                self.mem_store.save()
                return {"bootstrapped": True, "degraded": True, "error": str(exc)}
            full_view = res.get("full_view", {}) or {}
            l2 = self.mem_store.get_spatial("L2", "r0c0")
            if l2 is None:
                l2 = SpatialUnit(
                    id="r0c0",
                    bounds=[
                        0.0, 0.0,
                        float(self.config.frame_width),
                        float(self.config.frame_height),
                    ],
                )
            self._fill_vision_fields(
                l2,
                str(full_view.get("environment", "") or ""),
                str(full_view.get("background", "") or ""),
                str(full_view.get("cat_contrast", "") or ""),
                now,
            )
            self.mem_store.put_spatial("L2", l2)
            zones_written = 0
            cols, rows = self.spatial.grid(1)
            for zone in res.get("zones", []) or []:
                if not isinstance(zone, dict):
                    continue
                zid = str(zone.get("id", "") or "")
                try:
                    col, row = self.spatial.parse_cell_id(zid)
                except ValueError:
                    continue
                if not (0 <= col < cols and 0 <= row < rows):
                    continue
                unit = self.mem_store.get_spatial("L1", zid)
                if unit is None:
                    unit = SpatialUnit(
                        id=zid, bounds=self.spatial.cell_bounds(1, col, row)
                    )
                self._fill_vision_fields(
                    unit,
                    str(zone.get("environment", "") or ""),
                    str(zone.get("background", "") or ""),
                    str(zone.get("cat_contrast", "") or ""),
                    now,
                )
                self.mem_store.put_spatial("L1", unit)
                zones_written += 1
            self.mem_store.bootstrapped = True
            self.mem_store.save()
            return {"bootstrapped": True, "zones": zones_written}

    def _fill_vision_placeholder(self, now: str) -> None:
        l2 = self.mem_store.get_spatial("L2", "r0c0")
        if l2 is None:
            l2 = SpatialUnit(
                id="r0c0",
                bounds=[
                    0.0, 0.0,
                    float(self.config.frame_width),
                    float(self.config.frame_height),
                ],
            )
        self._fill_vision_fields(
            l2,
            "Unknown environment (vision bootstrap unavailable; text-only mode).",
            "Unknown.",
            "Unknown contrast.",
            now,
        )
        self.mem_store.put_spatial("L2", l2)

    @staticmethod
    def _mark_vision_unknown(unit: SpatialUnit) -> None:
        """Text-only mode: an unbootstrapped store marks vision fields unknown
        rather than guessing (bootstrap fills L2/L1 later; L0 inherits)."""
        if not unit.environment:
            unit.environment = (
                "Unknown environment (text-only mode; no bootstrap image yet)."
            )
        if not unit.background:
            unit.background = "Unknown."
        if not unit.cat_contrast:
            unit.cat_contrast = "Unknown contrast."

    @staticmethod
    def _fill_vision_fields(
        unit: SpatialUnit, environment: str, background: str, contrast: str, now: str
    ) -> None:
        """Fill empty vision fields without clobbering observed data."""
        if environment and not unit.environment:
            unit.environment = environment
        if background and not unit.background:
            unit.background = background
        if contrast and not unit.cat_contrast:
            unit.cat_contrast = contrast
        unit.updated = now

    # ---- LLM-backed merges with programmatic weighting ----

    def _source_label(self, event: MemoryEvent) -> str:
        if event.period_id is not None:
            return f"period {event.period_id}"
        if event.code:
            return str(event.code)
        return event.kind or "observation"

    def _bump_counters(self, unit, event: MemoryEvent) -> None:
        unit.stats["visits"] = int(unit.stats.get("visits", 0)) + 1
        if event.cat_id is not None:
            by_cat = unit.stats.setdefault("by_cat", {})
            key = str(event.cat_id)
            by_cat[key] = int(by_cat.get(key, 0)) + 1
        unit.weight = min(
            int(unit.stats["visits"]), self.config.stable_weight_cap
        )

    def _touch(self, unit, event: MemoryEvent) -> None:
        unit.updated = event.time.isoformat(timespec="seconds")
        label = self._source_label(event)
        if label not in unit.sources:
            unit.sources.append(label)

    def _append_recent(self, unit, stamped: str) -> None:
        unit.recent.append(stamped)
        if len(unit.recent) > self.config.recent_kept:
            del unit.recent[: -self.config.recent_kept]

    def _store_llm_update(
        self, stable: str, weight: int, event: MemoryEvent
    ) -> tuple[str, list[str], bool | None]:
        """Return ``(note, keywords, contradicts)``; ``None`` means the LLM
        failed and the caller must use pending-observation mode."""
        try:
            res = self.llm.store_update(
                {"stable": stable, "weight": weight}, event.text
            )
            if not isinstance(res, dict):
                raise ValueError("store LLM returned a non-dict")
            note = str(res.get("note", "") or "") or event.text
            keywords = [str(k) for k in res.get("keywords", []) or []]
            return note, keywords, bool(res.get("contradicts_stable", False))
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            logger.warning("memory store LLM failed (%s); keeping raw text", exc)
            return event.text, extract_keywords(event.text), None

    def _merge_spatial_text(self, unit: SpatialUnit, event: MemoryEvent) -> None:
        stamp = event.time.isoformat(timespec="minutes")
        self._bump_counters(unit, event)
        note, keywords, contradicts = self._store_llm_update(
            unit.activity, unit.weight, event
        )
        if contradicts is None:
            pass  # pending-observation mode: stable text and counter untouched
        elif not unit.activity:
            unit.activity = note
            unit.contradictions = 0
        elif contradicts:
            unit.contradictions = int(unit.contradictions) + 1
            if unit.contradictions >= self.config.flip_threshold:
                unit.activity = note
                unit.contradictions = 0
        else:
            unit.contradictions = 0
        self._append_recent(unit, f"{stamp} {note}")
        if keywords:
            unit.keywords = sorted(set(unit.keywords) | set(keywords))
        self._touch(unit, event)

    def _merge_temporal_text(self, unit: TemporalUnit, event: MemoryEvent) -> None:
        stamp = event.time.isoformat(timespec="minutes")
        self._bump_counters(unit, event)
        note, keywords, contradicts = self._store_llm_update(
            unit.summary, unit.weight, event
        )
        if contradicts is None:
            pass  # pending-observation mode: stable text and counter untouched
        elif not unit.summary:
            unit.summary = note
            unit.contradictions = 0
        elif contradicts:
            unit.contradictions = int(unit.contradictions) + 1
            if unit.contradictions >= self.config.flip_threshold:
                unit.summary = note
                unit.contradictions = 0
        else:
            unit.contradictions = 0
        self._append_recent(unit, f"{stamp} {note}")
        if keywords:
            unit.keywords = sorted(set(unit.keywords) | set(keywords))
        self._touch(unit, event)

    # ---- bottom-up promotion with compression ----

    def _spatial_child_view(self, child: SpatialUnit) -> dict:
        return {
            "id": child.id,
            "text": child.activity or child.environment,
            "keywords": list(child.keywords),
            "visits": int(child.stats.get("visits", 0)),
            "weight": int(child.weight),
        }

    def _temporal_child_view(self, child: TemporalUnit) -> dict:
        return {
            "id": child.id,
            "text": child.summary,
            "keywords": list(child.keywords),
            "visits": int(child.stats.get("visits", 0)),
            "weight": int(child.weight),
        }

    def _qualified(self, kids: list) -> list:
        """Children with support >= ``min_pattern_support``, dominant first."""
        support = self.config.min_pattern_support
        out = [k for k in kids if int(k.stats.get("visits", 0)) >= support]
        out.sort(key=lambda k: (-int(k.stats.get("visits", 0)), k.id))
        return out

    @staticmethod
    def _roll_stats(kids: list) -> tuple[int, dict[str, int]]:
        visits = 0
        by_cat: dict[str, int] = {}
        for kid in kids:
            visits += int(kid.stats.get("visits", 0))
            for key, val in (kid.stats.get("by_cat") or {}).items():
                by_cat[str(key)] = by_cat.get(str(key), 0) + int(val)
        return visits, by_cat

    def _summarize_parent(
        self,
        label: str,
        pid: str,
        child_dicts: list[dict],
        stats: dict,
        budget: int,
        state: dict,
        errors: list[str],
    ) -> dict | None:
        """One budgeted LLM summary; ``None`` on error or deferral (reported)."""
        if state["calls"] >= budget:
            return None
        try:
            out = self.llm.summarize(label, child_dicts, stats)
            if not isinstance(out, dict):
                raise ValueError("summarize LLM returned a non-dict")
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            errors.append(f"{label}:{pid}: {exc}")
            logger.warning(
                "memory promote summarize failed for %s:%s; keeping prior text",
                label,
                pid,
            )
            return None
        state["calls"] += 1
        return out

    def _promote_spatial(
        self, force: bool, budget: int, state: dict, errors: list[str]
    ) -> tuple[int, int]:
        promoted, deferred = 0, 0
        now = datetime.datetime.now().isoformat(timespec="seconds")
        changed: set[str] = set()  # L1 ids summarized this pass (drive L2)
        for level in (1, 2):
            label = self.spatial.level_label(level)
            child_label = self.spatial.level_label(level - 1)
            if force:
                pids: set[str] = set(self.mem_store.spatial.get(label, {}))
                for cid, child in self.mem_store.spatial.get(child_label, {}).items():
                    if child.is_empty:
                        continue
                    try:
                        col, row = self.spatial.parse_cell_id(cid)
                    except ValueError:
                        continue
                    parent = self.spatial.parent(level - 1, col, row)
                    if parent is not None:
                        pids.add(parent[3])
            elif level == 1:
                pids = set()
                for kind, lv, cid in self._dirty:
                    if kind != "spatial" or lv != "L0":
                        continue
                    try:
                        col, row = self.spatial.parse_cell_id(cid)
                    except ValueError:
                        continue
                    parent = self.spatial.parent(0, col, row)
                    if parent is not None:
                        pids.add(parent[3])
            else:
                # changed holds child-level ids: map each up to its parent.
                pids = set()
                for cid in changed:
                    try:
                        ccol, crow = self.spatial.parse_cell_id(cid)
                    except ValueError:
                        continue
                    parent = self.spatial.parent(level - 1, ccol, crow)
                    if parent is not None:
                        pids.add(parent[3])
            for pid in sorted(pids):
                col, row = self.spatial.parse_cell_id(pid)
                kids = []
                for _cc, _cr, cid in self.spatial.children(level, col, row):
                    child = self.mem_store.get_spatial(child_label, cid)
                    if child is not None and not child.is_empty:
                        kids.append(child)
                parent = self.mem_store.get_spatial(label, pid)
                if parent is None:
                    if not kids:
                        continue
                    parent = SpatialUnit(
                        id=pid, bounds=self.spatial.cell_bounds(level, col, row)
                    )
                visits, by_cat = self._roll_stats(kids)
                parent.stats["visits"] = visits
                parent.stats["by_cat"] = by_cat
                parent.weight = min(visits, self.config.stable_weight_cap)
                parent.updated = now
                qualified = self._qualified(kids)
                if not qualified:
                    # Stats rolled up; one-offs stay in children's recent by
                    # design, so the dirty flags are consumed without an LLM call.
                    # The parent's stats still changed, so it drives the level above.
                    for kid in kids:
                        self._dirty.discard(("spatial", child_label, kid.id))
                    self._store_or_drop_spatial(label, parent)
                    changed.add(pid)
                    continue
                if state["calls"] >= budget:
                    deferred += 1
                    self._store_or_drop_spatial(label, parent)
                    continue
                child_dicts = [
                    self._spatial_child_view(k)
                    for k in qualified[: self.config.max_units_per_call]
                ]
                out = self._summarize_parent(
                    label, pid, child_dicts,
                    {"visits": visits, "by_cat": by_cat},
                    budget, state, errors,
                )
                if out is None:
                    # Error: keep the prior text and stay dirty for a retry.
                    self._store_or_drop_spatial(label, parent)
                    continue
                text = truncate_words(
                    str(out.get("text", "") or ""), WORD_CAPS.get(label, 100)
                )
                if text:
                    parent.activity = text
                    promoted += 1
                kws = [str(k) for k in out.get("keywords", []) or []]
                if kws:
                    parent.keywords = sorted(set(parent.keywords) | set(kws))
                changed.add(pid)
                for kid in kids:
                    self._dirty.discard(("spatial", child_label, kid.id))
                self._store_or_drop_spatial(label, parent)
        return promoted, deferred

    def _store_or_drop_spatial(self, label: str, parent: SpatialUnit) -> None:
        if parent.is_empty:
            self.mem_store.spatial.get(label, {}).pop(parent.id, None)
        else:
            self.mem_store.put_spatial(label, parent)

    def _store_or_drop_temporal(self, level: str, parent: TemporalUnit) -> None:
        if parent.is_empty:
            self.mem_store.temporal.get(level, {}).pop(parent.id, None)
        else:
            self.mem_store.put_temporal(level, parent)

    def _promote_temporal(
        self, force: bool, budget: int, state: dict, errors: list[str]
    ) -> tuple[int, int]:
        promoted, deferred = 0, 0
        now = datetime.datetime.now().isoformat(timespec="seconds")
        pairs = (("hour", "slot_5min"), ("day", "hour"), ("week", "day"))
        changed: set[tuple[str, str]] = set()  # (level, id) summarized this pass
        for level, child_level in pairs:
            if force:
                bids: set[str] = set(self.mem_store.temporal.get(level, {}))
                for cid, child in self.mem_store.temporal.get(child_level, {}).items():
                    if child.is_empty:
                        continue
                    try:
                        start = self.temporal.bucket_start(child_level, cid)
                    except ValueError:
                        continue
                    bids.add(self.temporal.bucket_id(level, start))
            elif child_level == "slot_5min":
                bids = set()
                for kind, lv, cid in self._dirty:
                    if kind != "temporal" or lv != "slot_5min":
                        continue
                    try:
                        start = self.temporal.bucket_start(child_level, cid)
                    except ValueError:
                        continue
                    bids.add(self.temporal.bucket_id(level, start))
            else:
                bids = {bid for (lv, bid) in changed if lv == child_level}
            for bid in sorted(bids):
                kids = []
                for cid in self.temporal.child_ids(level, bid):
                    child = self.mem_store.get_temporal(child_level, cid)
                    if child is not None and not child.is_empty:
                        kids.append(child)
                parent = self.mem_store.get_temporal(level, bid)
                if parent is None:
                    if not kids:
                        continue
                    parent = TemporalUnit(id=bid)
                visits, by_cat = self._roll_stats(kids)
                parent.stats["visits"] = visits
                parent.stats["by_cat"] = by_cat
                parent.weight = min(visits, self.config.stable_weight_cap)
                parent.updated = now
                qualified = self._qualified(kids)
                if not qualified:
                    # Stats-only rollup (see spatial path): consume dirty flags
                    # but still drive the level above with the fresh stats.
                    for kid in kids:
                        self._dirty.discard(("temporal", child_level, kid.id))
                    self._store_or_drop_temporal(level, parent)
                    changed.add((level, bid))
                    continue
                if state["calls"] >= budget:
                    deferred += 1
                    self._store_or_drop_temporal(level, parent)
                    continue
                child_dicts = [
                    self._temporal_child_view(k)
                    for k in qualified[: self.config.max_units_per_call]
                ]
                out = self._summarize_parent(
                    level, bid, child_dicts,
                    {"visits": visits, "by_cat": by_cat},
                    budget, state, errors,
                )
                if out is None:
                    self._store_or_drop_temporal(level, parent)
                    continue
                text = truncate_words(
                    str(out.get("text", "") or ""), WORD_CAPS.get(level, 100)
                )
                if text:
                    parent.summary = text
                    promoted += 1
                kws = [str(k) for k in out.get("keywords", []) or []]
                if kws:
                    parent.keywords = sorted(set(parent.keywords) | set(kws))
                changed.add((level, bid))
                for kid in kids:
                    self._dirty.discard(("temporal", child_level, kid.id))
                self._store_or_drop_temporal(level, parent)
        return promoted, deferred

