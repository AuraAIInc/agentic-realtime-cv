"""Exercise the tracking tools without a live camera.

Replays a historical window from MySQL through fresh TrackCorrector / PeriodCorrector
instances (DB writes stubbed out, fake period ids far above the real ones), stops, then
calls every tool in-process and prints the replies. DB reads are real.

    python tracking_mcp/replay_check.py \
        --view-id 1 --start "2025-12-18 15:03" --end "2025-12-18 15:06"

Pick --end inside a visit so the cat is still on screen and its period is still open.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("LOG_LEVEL", "ERROR")

import pandas as pd

from model.db import MysqlDB
from model.period_corrector import PeriodCorrector
from model.track_corrector import TrackCorrector
from run_tracking_viz import MIN_CONFIDENCE, load_detections
from tracking_mcp.config import Settings
from tracking_mcp.history import History
from tracking_mcp.live_state import LiveState
from tracking_mcp.server import build_server

FAKE_PERIOD_BASE = 10**12  # never collides with real period ids


def stub_writes(corrector, counter):
    corrector.db.execute_insert = lambda *a, **k: next(counter)
    corrector.db.execute_update = lambda *a, **k: None
    corrector.db.execute_many_insert = lambda *a, **k: None
    corrector.db.execute_delete = lambda *a, **k: None
    return corrector


def replay(view_id: int, start: datetime.datetime, end: datetime.datetime) -> LiveState:
    detections = load_detections(MysqlDB(), view_id, start, end)
    print(f"Loaded {len(detections)} rows, {detections['create_time'].nunique()} frames")
    counter = iter(range(FAKE_PERIOD_BASE, FAKE_PERIOD_BASE + 10**6))
    tc = stub_writes(TrackCorrector(view_id=view_id), counter)
    pc = stub_writes(PeriodCorrector(view_id=view_id), counter)
    live = LiveState({view_id: tc}, {view_id: pc})
    for create_time in detections["create_time"].unique():
        create_time = pd.Timestamp(create_time).to_pydatetime()
        det = detections[(detections["create_time"] == create_time) & (detections["conf"] > MIN_CONFIDENCE)]
        img_detected = dict(zip(det["cat_id"].values.astype(int), det["box"].values))
        live.record_frame(
            {
                "view_id": view_id,
                "create_time": create_time.strftime("%Y-%m-%d-%H-%M-%S-%f"),
                "detections": [{"cat_id": c, "box": b[:4], "confidence": b[5]} for c, b in img_detected.items()],
            },
            create_time,
            MIN_CONFIDENCE,
        )
        for is_append, revised in tc.process_images(create_time, img_detected, is_grey=0, model_id=0, view_id=view_id, device_id=0, save_db=False):
            if is_append:
                for period in pc.process_detection(revised, view_id=view_id):
                    live.record_period(view_id, period)
            else:
                pc.unprocess_detection(revised)
    return live


async def call(mcp, name: str, args: dict) -> dict:
    result = await mcp.call_tool(name, args)
    blocks = result.content if hasattr(result, "content") else result
    text = next(b.text for b in blocks if getattr(b, "type", None) == "text")
    out = json.loads(text)
    status = "ERROR" if out.get("error") else "ok"
    print(f"\n=== {name}({args}) -> {status} ({out['timing']['elapsed_ms']} ms)")
    print(json.dumps(out, indent=1)[:2500])
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view-id", type=int, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args()
    live = replay(a.view_id, datetime.datetime.strptime(a.start, "%Y-%m-%d %H:%M"), datetime.datetime.strptime(a.end, "%Y-%m-%d %H:%M"))
    mcp = build_server(live, History(MysqlDB()), Settings())

    await call(mcp, "list_views", {})
    status = await call(mcp, "get_live_status", {"view_id": a.view_id})
    await call(mcp, "get_last_seen", {"view_id": a.view_id})
    await call(mcp, "get_current_periods", {"view_id": a.view_id})
    cats = [c for c in status["views"][0]["cats"] if c["last_period_id"]]
    for c in cats:
        await call(mcp, "get_period_detail", {"period_id": c["last_period_id"], "max_points": 10})
        await call(mcp, "get_arrival_path", {"cat_id": c["cat_id"], "view_id": a.view_id})
        await call(mcp, "get_last_movement", {"cat_id": c["cat_id"], "view_id": a.view_id, "lookback_hours": 1})
    await call(mcp, "get_periods", {"view_id": a.view_id, "time_to": a.end, "limit": 3})
    await call(mcp, "get_raw_detections", {"time_from": a.end, "time_to": a.end[:-2] + "59", "view_id": a.view_id, "limit": 3})


if __name__ == "__main__":
    asyncio.run(main())
