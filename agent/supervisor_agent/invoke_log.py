"""Dump every agent invoke (MCPAgent.run) and its tool calls to MySQL.

Turned on by AGENT_INVOKE_DUMP=true in .env; uses the MYSQL_* settings the ingest server uses.
The agent only puts a snapshot on a queue (no I/O, no serialisation); a daemon thread builds
the rows and writes them, so the dump never adds to the agent's response time. When the DB is
down or the queue is full, records are dropped with a warning, never raised to the agent.

    agent_invoke  one row per MCPAgent.run
    tool_invoke   one row per tool call, agent_invoke_id -> agent_invoke.invoke_id
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import uuid

logger = logging.getLogger(__name__)

AGENT_INVOKE_DDL = """
CREATE TABLE IF NOT EXISTS agent_invoke (
    invoke_id            CHAR(36)      NOT NULL PRIMARY KEY,
    invoke_time          DATETIME(6)   NOT NULL,
    model                VARCHAR(128)  NOT NULL,
    tag                  VARCHAR(64)   NULL,
    input_query          MEDIUMTEXT    NULL,
    agent_response       MEDIUMTEXT    NULL,
    input_tokens         INT           NULL,
    output_tokens        INT           NULL,
    cached_tokens        INT           NULL,
    total_tokens         INT           NULL,
    est_cost_usd         DECIMAL(12,6) NULL,
    price_input_per_1m   DECIMAL(12,6) NULL,
    price_cached_per_1m  DECIMAL(12,6) NULL,
    price_output_per_1m  DECIMAL(12,6) NULL,
    images_sent          INT           NOT NULL DEFAULT 0,
    tool_count           INT           NOT NULL DEFAULT 0,
    tool_names           TEXT          NULL,
    tools_ms             DOUBLE        NULL,
    llm_ms               DOUBLE        NULL,
    total_ms             DOUBLE        NULL,
    turns                INT           NULL,
    stopped_reason       VARCHAR(255)  NULL,
    reasoning_effort     VARCHAR(32)   NULL,
    previous_response_id VARCHAR(128)  NULL,
    last_response_id     VARCHAR(128)  NULL,
    run_dir              VARCHAR(512)  NULL,
    KEY idx_invoke_time (invoke_time)
) DEFAULT CHARSET=utf8mb4
"""

TOOL_INVOKE_DDL = """
CREATE TABLE IF NOT EXISTS tool_invoke (
    invoke_id         CHAR(36)     NOT NULL PRIMARY KEY,
    agent_invoke_id   CHAR(36)     NOT NULL,
    invoke_time       DATETIME(6)  NOT NULL,
    turn              INT          NOT NULL,
    call_id           VARCHAR(128) NULL,
    tool_name         VARCHAR(128) NOT NULL,
    tool_input        MEDIUMTEXT   NULL,
    tool_output       MEDIUMTEXT   NULL,
    images_returned   INT          NOT NULL DEFAULT 0,
    response_ms       DOUBLE       NULL,
    server_elapsed_ms DOUBLE       NULL,
    server_steps_ms   TEXT         NULL,
    is_error          TINYINT(1)   NOT NULL DEFAULT 0,
    error             TEXT         NULL,
    KEY idx_agent_invoke (agent_invoke_id),
    KEY idx_tool_time (tool_name, invoke_time)
) DEFAULT CHARSET=utf8mb4
"""

_COLS_AGENT = ("invoke_id", "invoke_time", "model", "tag", "input_query", "agent_response", "input_tokens", "output_tokens",
               "cached_tokens", "total_tokens", "est_cost_usd", "price_input_per_1m", "price_cached_per_1m",
               "price_output_per_1m", "images_sent", "tool_count", "tool_names", "tools_ms", "llm_ms",
               "total_ms", "turns", "stopped_reason", "reasoning_effort", "previous_response_id", "last_response_id", "run_dir")
# Columns added after the first release: ALTER TABLE adds them to an existing agent_invoke.
_ADDED_AGENT_COLS = {
    "price_input_per_1m": "DECIMAL(12,6) NULL AFTER est_cost_usd",
    "price_cached_per_1m": "DECIMAL(12,6) NULL AFTER price_input_per_1m",
    "price_output_per_1m": "DECIMAL(12,6) NULL AFTER price_cached_per_1m",
}
_COLS_TOOL = ("invoke_id", "agent_invoke_id", "invoke_time", "turn", "call_id", "tool_name", "tool_input", "tool_output",
              "images_returned", "response_ms", "server_elapsed_ms", "server_steps_ms", "is_error", "error")
_STOP = object()


def enabled() -> bool:
    return os.getenv("AGENT_INVOKE_DUMP", "false").strip().lower() in ("1", "true", "yes", "on")


def count_images(obj) -> int:
    """input_image items anywhere in a Responses API input (query list or tool outputs)."""
    if isinstance(obj, dict):
        return (obj.get("type") == "input_image") + sum(count_images(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_images(v) for v in obj)
    return 0


def _strip_images(obj):
    """Replace base64 data URLs with a short placeholder so rows hold text, not pixels."""
    if isinstance(obj, dict):
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:") and ";base64," in obj:
        head, data = obj.split(",", 1)
        return f"{head},<{len(data)} base64 chars>"
    return obj


def _text(obj) -> str | None:
    if obj is None:
        return None
    return obj if isinstance(obj, str) else json.dumps(_strip_images(obj), ensure_ascii=False, default=str)


class InvokeLog:
    """One background writer thread per process; submit() is a non-blocking queue put."""

    def __init__(self, maxsize: int = 1000):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._conn = None
        self._thread = threading.Thread(target=self._worker, name="invoke-log", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def submit(self, snapshot: dict) -> None:
        try:
            self._q.put_nowait(snapshot)
        except queue.Full:
            logger.warning("invoke log queue full; dropping agent invoke record")

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued (up to timeout) on interpreter exit."""
        if self._thread.is_alive():
            try:
                self._q.put(_STOP, timeout=timeout)
            except queue.Full:
                return
            self._thread.join(timeout)

    # ------------------------------------------------------------------ writer thread
    def _worker(self) -> None:
        while True:
            snap = self._q.get()
            if snap is _STOP:
                break
            try:
                self._write(*_rows(snap))
            except Exception as e:  # never let a DB problem kill the thread
                logger.warning(f"invoke log write failed: {e}")
                self._drop_conn()
        self._drop_conn()

    def _connect(self):
        if self._conn is None:
            import pymysql

            self._conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DB"),
                charset="utf8mb4",
                connect_timeout=10,
                autocommit=False,
            )
            with self._conn.cursor() as cur:
                cur.execute(AGENT_INVOKE_DDL)
                cur.execute(TOOL_INVOKE_DDL)
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = 'agent_invoke'")
                have = {r[0].lower() for r in cur.fetchall()}
                for col, spec in _ADDED_AGENT_COLS.items():
                    if col not in have:
                        cur.execute(f"ALTER TABLE agent_invoke ADD COLUMN {col} {spec}")
            self._conn.commit()
        else:
            self._conn.ping(reconnect=True)
        return self._conn

    def _drop_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _write(self, agent_row: tuple, tool_rows: list[tuple]) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO agent_invoke ({', '.join(_COLS_AGENT)}) VALUES ({', '.join(['%s'] * len(_COLS_AGENT))})", agent_row)
                if tool_rows:
                    cur.executemany(f"INSERT INTO tool_invoke ({', '.join(_COLS_TOOL)}) VALUES ({', '.join(['%s'] * len(_COLS_TOOL))})", tool_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _rows(s: dict) -> tuple[tuple, list[tuple]]:
    """Build the DB rows from a snapshot taken by MCPAgent.run (runs in the writer thread)."""
    usage = s["usage"]
    price = s["pricing"] or {}  # USD per 1M tokens; cached falls back to input, as in estimate_cost_usd
    steps = {st.call_id: st for st in s["tool_steps"]}
    tool_rows, names, images = [], [], count_images(s["query"])
    for turn, started_at, calls, outputs in s["turns"]:
        for call, out in zip(calls, outputs):
            st = steps.get(call.call_id)
            items = out.get("output") or []
            n_img = count_images(items)
            images += n_img
            names.append(call.name)
            tool_rows.append((
                str(uuid.uuid4()), s["invoke_id"], started_at, turn, call.call_id, call.name,
                call.arguments,
                _text([it.get("text") if it.get("type") == "input_text" else _strip_images(it) for it in items]),
                n_img,
                st.round_trip_ms if st else None,
                st.server_elapsed_ms if st else None,
                _text(st.server_steps_ms) if st and st.server_steps_ms else None,
                int(bool(st and st.is_error)),
                st.error if st else None,
            ))
    agent_row = (
        s["invoke_id"], s["invoke_time"], s["model"], s["tag"], _text(s["query"]), s["answer"],
        usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cached_tokens"), usage.get("total_tokens"),
        s["est_cost_usd"], price.get("input"), price.get("cached_input", price.get("input")), price.get("output"), images, len(names), json.dumps(names), s["tools_ms"], s["llm_ms"], s["total_ms"],
        s["turns_count"], s["stopped_reason"], s["reasoning_effort"], s["previous_response_id"], s["last_response_id"], s["run_dir"],
    )
    return agent_row, tool_rows


_instance: InvokeLog | None = None
_lock = threading.Lock()


def get_invoke_log() -> InvokeLog | None:
    """The process-wide logger, or None when AGENT_INVOKE_DUMP is off."""
    global _instance
    if not enabled():
        return None
    try:
        import pymysql  # noqa: F401
    except ImportError:
        logger.error("AGENT_INVOKE_DUMP is on but pymysql is not installed (pip install -r requirements.txt); not dumping invokes")
        return None
    with _lock:
        if _instance is None:
            _instance = InvokeLog()
    return _instance
