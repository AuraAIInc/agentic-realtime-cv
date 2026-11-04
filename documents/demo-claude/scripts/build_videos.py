"""Build the motion clips for the demo video (1920x1080, H.264, 30 fps unless noted).

    v01_cost_counter.mp4   frames/day and $/year counting up (s04 beat)
    v02_ball_push_in.mp4   slow push-in onto the ball tracked as "Qbo" (s02 beat)
    v03_replay_tracker.mp4 real frames 16:34:25-16:35:45 at ~3x, the Pi tracker's boxes,
                           doubts popping up as they fire, verdicts listed on the right

    python documents/demo-claude/scripts/build_videos.py [v01 v02 v03]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parents[1]
OUT = DEMO / "assets" / "video"
RUN = REPO / "output/tracking_runs/view3_20261028-1633_1640_replay_db_20261102-231613-423659"
HOUR = REPO / "image_server/retrieved_images/2026-10-28-16/images"

BG = (26, 26, 25)
PANEL = (35, 35, 33)
LINE = (58, 57, 53)
T1 = (242, 241, 237)
T2 = (195, 194, 183)
T3 = (142, 141, 132)
QB = (57, 135, 229)
QBO = (217, 89, 38)
AGENT = (144, 133, 233)
DOUBT = (250, 178, 25)
BAD = (230, 103, 103)
W, H = 1920, 1080

SF = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, weight: str = "Regular", mono: bool = False) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(MONO if mono else SF, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def encode(frames_dir: Path, out: Path, fps: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", "30", str(out)], check=True)


def ease(t: float) -> float:
    return 1 - (1 - t) ** 3


# ---------------------------------------------------------------- v01
def v01_cost_counter() -> Path:
    fps, secs = 30, 6
    out = OUT / "v01_cost_counter.mp4"
    with tempfile.TemporaryDirectory() as td:
        for i in range(fps * secs):
            t = min(1.0, i / (fps * 4.0))  # count for 4 s, then hold
            e = ease(t)
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            d.text((112, 250), "ONE CAMERA · 2 FRAMES A SECOND · EVERY FRAME TO A VISION LLM", font=font(26, "Semibold"), fill=T3)
            d.text((112, 330), f"{int(172800 * e):,}", font=font(170, "Bold"), fill=T1)
            d.text((112, 530), "frames a day", font=font(38), fill=T2)
            d.text((1000, 330), f"${int(63072 * e):,}", font=font(170, "Bold"), fill=BAD)
            d.text((1000, 530), "a year, at $0.001 a frame", font=font(38), fill=T2)
            # progress hairline
            d.rectangle((112, 640, 112 + int(1696 * e), 643), fill=BAD)
            if i > fps * 4.3:
                a = min(1.0, (i - fps * 4.3) / (fps * 0.5))
                c = tuple(int(BG[k] + (T1[k] - BG[k]) * a) for k in range(3))
                d.text((112, 720), "…for one camera.", font=font(52, "Semibold"), fill=c)
            im.save(Path(td) / f"{i:05d}.png")
        encode(Path(td), out, fps)
    return out


# ---------------------------------------------------------------- v02
def v02_ball_push_in() -> Path:
    fps, secs = 30, 5
    src = Image.open(DEMO / "assets/frames/2026-11-01-07-35-00-150851_ball_decoy.jpg").convert("RGB")
    sw, sh = src.size
    box = (351, 47, 407, 104)
    # end crop: 16:9 window centred on the ball, ~4.5x zoom
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    end_w = 180
    end_h = end_w * 9 / 16
    start = (0, (sh - sw * 9 / 16) / 2, sw, (sh + sw * 9 / 16) / 2)
    end = (cx - end_w / 2, max(0, cy - end_h / 2), cx + end_w / 2, max(0, cy - end_h / 2) + end_h)
    out = OUT / "v02_ball_push_in.mp4"
    with tempfile.TemporaryDirectory() as td:
        for i in range(fps * secs):
            t = ease(min(1.0, i / (fps * 4.2)))
            c = [start[k] + (end[k] - start[k]) * t for k in range(4)]
            frame = src.resize((W, H), Image.LANCZOS, box=tuple(c))
            d = ImageDraw.Draw(frame)
            sx, sy = W / (c[2] - c[0]), H / (c[3] - c[1])
            b = [(box[0] - c[0]) * sx, (box[1] - c[1]) * sy, (box[2] - c[0]) * sx, (box[3] - c[1]) * sy]
            lw = max(3, int(3 + 5 * t))
            d.rectangle(b, outline=QBO, width=lw)
            f = font(int(28 + 26 * t), "Bold")
            label = "Qbo · 26 min · 100% agreement"
            tw = d.textlength(label, font=f)
            d.rounded_rectangle((b[0], b[1] - f.size - 26, b[0] + tw + 28, b[1] - 6), radius=10, fill=QBO)
            d.text((b[0] + 14, b[1] - f.size - 20), label, font=f, fill=(17, 17, 17))
            frame.save(Path(td) / f"{i:05d}.png")
        encode(Path(td), out, fps)
    return out


# ---------------------------------------------------------------- v03
def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T"))


def v03_replay_tracker() -> Path:
    t_start, t_end = _ts("2026-10-28T16:34:25"), _ts("2026-10-28T16:35:45")
    files = {}
    for p in HOUR.glob("*.jpg"):
        stamp = p.stem.split("_", 1)[1]
        t = datetime.strptime(stamp, "%Y-%m-%d-%H-%M-%S-%f")
        if t_start <= t <= t_end:
            files[t] = p
    times = sorted(files)
    rows: dict[str, list[dict]] = {}
    for l in (RUN / "revised.jsonl").open():
        r = json.loads(l)
        rows.setdefault(r["create_time"], []).append(r)
    esc = sorted((json.loads(l) for l in (RUN / "escalations.jsonl").open()), key=lambda e: e["time"])
    verdicts = {}
    for l in (RUN / "validations.jsonl").open():
        d = json.loads(l)
        verdicts[d["escalation"]["name"]] = d["review"]["verdict"]
    reasons = {"T2": "relabeled by distance alone", "T4": "jump reset", "T6": "teleport split",
               "T7": "long guessed run", "P2": "weak winner", "P5": "low quality", "P6": "merge candidate"}
    corrected = {"keep_original", "cut_at_gap_start"}

    fw, fh = 1280, 960
    sx, sy = fw / 800, fh / 600
    out = OUT / "v03_replay_tracker.mp4"
    fps = 8  # ~2.8 source fps -> ~3x real time
    f_lbl, f_small, f_mono = font(26, "Bold"), font(24), font(24, "Semibold", mono=True)
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(times):
            canvas = Image.new("RGB", (W, H), BG)
            fr = Image.open(files[t]).convert("RGB").resize((fw, fh), Image.LANCZOS)
            d = ImageDraw.Draw(fr)
            key = t.isoformat()
            for r in rows.get(key, []):
                col = QB if r["cat_id"] == 25 else QBO
                b = (r["box_x1"] * sx, r["box_y1"] * sy, r["box_x2"] * sx, r["box_y2"] * sy)
                guessed = r.get("orig_cat_id") is None
                if guessed:  # dashed outline for interpolated rows
                    x1, y1, x2, y2 = b
                    for x in range(int(x1), int(x2), 22):
                        d.line((x, y1, min(x + 12, x2), y1), fill=col, width=4)
                        d.line((x, y2, min(x + 12, x2), y2), fill=col, width=4)
                    for y in range(int(y1), int(y2), 22):
                        d.line((x1, y, x1, min(y + 12, y2)), fill=col, width=4)
                        d.line((x2, y, x2, min(y + 12, y2)), fill=col, width=4)
                else:
                    d.rectangle(b, outline=col, width=4)
                name = ("Qb" if r["cat_id"] == 25 else "Qbo") + (" (guess)" if guessed else "")
                tw = d.textlength(name, font=f_lbl)
                d.rounded_rectangle((b[0], b[1] - 40, b[0] + tw + 20, b[1] - 4), radius=8, fill=col)
                d.text((b[0] + 10, b[1] - 38), name, font=f_lbl, fill=(17, 17, 17))
            # doubts that fire within +-1.2 s of this frame: flash a yellow banner on the frame
            live = [e for e in esc if abs((_ts(e["time"]) - t).total_seconds()) <= 1.2]
            for k, e in enumerate(live):
                txt = f"? {e['name']}  {reasons.get(e['code'], '')}"
                tw = d.textlength(txt, font=f_lbl)
                y = 24 + k * 52
                d.rounded_rectangle((24, y, 24 + tw + 32, y + 44), radius=10, fill=DOUBT)
                d.text((40, y + 6), txt, font=f_lbl, fill=(17, 17, 17))
            canvas.paste(fr, (60, 60))
            c = ImageDraw.Draw(canvas)
            c.rounded_rectangle((59, 59, 60 + fw, 60 + fh), radius=4, outline=LINE, width=2)
            # right panel: clock + escalation log
            px = 60 + fw + 48
            c.text((px, 66), "PI TRACKER · REPLAY", font=font(22, "Semibold"), fill=T3)
            c.text((px, 98), t.strftime("%H:%M:%S"), font=font(64, "Bold"), fill=T1)
            c.text((px, 176), "Oct 28 · ~3× speed", font=f_small, fill=T2)
            c.text((px, 236), "DOUBTS RAISED", font=font(22, "Semibold"), fill=T3)
            y = 272
            for e in esc:
                if _ts(e["time"]) > t or _ts(e["time"]) < t_start:
                    continue
                v = verdicts.get(e["name"], "")
                tag_col = AGENT if v in corrected else T3
                c.rounded_rectangle((px, y, px + 76, y + 36), radius=8, fill=DOUBT)
                c.text((px + 10, y + 5), e["name"].split("-")[0], font=f_mono, fill=(17, 17, 17))
                c.text((px + 92, y + 4), e["code"], font=f_mono, fill=T1)
                c.text((px + 142, y + 4), "→ " + v.replace("_", " "), font=f_small, fill=tag_col)
                y += 50
            c.text((px, 60 + fh - 124), "boxes = the Pi's own labels,", font=f_small, fill=T3)
            c.text((px, 60 + fh - 92), "mistakes included", font=f_small, fill=T3)
            c.text((px, 60 + fh - 52), "verdicts: returned later by Muse Spark", font=f_small, fill=AGENT)
            canvas.save(Path(td) / f"{i:05d}.png")
        encode(Path(td), out, fps)
    return out


CLIPS = {"v01": v01_cost_counter, "v02": v02_ball_push_in, "v03": v03_replay_tracker}

if __name__ == "__main__":
    for k in sys.argv[1:] or list(CLIPS):
        print(CLIPS[k]().relative_to(DEMO))
