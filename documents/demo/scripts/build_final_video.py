"""Build the final demo video, timed to the continuous voice-over.

1. build_voice.py renders assets/audio/voiceover_v2.wav (one take) + voiceover_v2_timings.json.
2. This script shows each beat's slide from just before its sentence starts until the next
   beat, plays the ball / counter / replay clips inside their beats, cross-fades between
   slides (slides themselves stay still: no zoom, no drift), lays the voice track under it,
   and writes captions from the same timings. Hard limit: 180 s.

    .venv/bin/python documents/demo/scripts/build_final_video.py
    -> documents/demo/assets/video/final_demo_v2.mp4 (+ assets/text/captions_final_v2.srt)
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
SL, VID, TXT, AUD = (DEMO / "assets" / d for d in ("slides", "video", "text", "audio"))
OUT = VID / "final_demo_v2.mp4"
FPS, W, H = 30, 1920, 1080
XF = 0.4         # cross-fade between segments
LEAD = 0.25      # a slide appears this long before its sentence starts
END_HOLD = 1.5   # silence + last slide after the final sentence
LIMIT = 180.0

# beat -> parts: (asset, seconds or None = rest of the beat, clip start offset)
PARTS = {
    1: [("s01_meet_qb_qbo", None, 0)],
    2: [("s03_simple_questions", None, 0)],
    3: [("v02_ball_push_in", 3.0, 0), ("s02_ball_decoy", None, 0)],
    4: [("v01_cost_counter", 4.0, 0), ("s04_cost_math", None, 0)],
    5: [("s05_two_halves", None, 0)],
    6: [("s05b_scope", 0.72, 0), ("s06_title_built_both", None, 0)],  # float < 1 = share of the beat
    7: [("s06b_bridge", None, 0)],
    8: [("s07_loop_diagram", None, 0)],
    9: [("v03_replay_tracker", 1.5, 3.0), ("s08_timeline", None, 0)],
    10: [("s09_doubt_taxonomy", None, 0)],
    11: [("s10_question_card", None, 0)],
    12: [("s10b_agent_work", None, 0)],
    13: [("s12_batching", None, 0)],
    14: [("s14_row_diff", None, 0)],
    15: [("s15_day_log", None, 0)],
    16: [("s16_memory_pyramid", None, 0)],
    17: [("s11_case_e5", None, 0)],
    18: [("s13_case_e10", None, 0)],
    19: [("s13b_detector_fixes", None, 0)],
    20: [("s17a_timeline_eval", None, 0)],
    21: [("s17_results", None, 0)],
    22: [("s19_beyond_cats", None, 0)],
    23: [("s18_roadmap", None, 0)],
    24: [("s20_end_card", None, 0)],
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def captions(beats: list[dict], path: Path) -> None:
    """One caption per sentence (split again past ~84 chars), timed by its share of the beat."""
    cues = []
    for b in beats:
        text = b["text"].replace("Q-bo", "Qbo").replace("Q-B", "Qb")
        sents = [s for s in re.split(r"(?<=[.?!:])\s+", text) if s]
        chunks = []
        for s in sents:
            while len(s) > 84:
                cut = s.rfind(" ", 0, 84)
                chunks.append(s[:cut]); s = s[cut + 1:]
            chunks.append(s)
        total = sum(len(c) for c in chunks)
        t = b["start"]
        for c in chunks:
            d = (b["end"] - b["start"]) * len(c) / total
            cues.append((t, t + d, c)); t += d
    path.write_text("".join(f"{i}\n{srt_time(a)} --> {srt_time(z)}\n{c}\n\n" for i, (a, z, c) in enumerate(cues, 1)))


def segment(asset: str, secs: float, offset: float, out: Path) -> None:
    if asset.startswith("v0"):
        run(["ffmpeg", "-y", "-ss", str(offset), "-i", str(VID / f"{asset}.mp4"),
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a19,"
                    f"fps={FPS},tpad=stop_mode=clone:stop_duration=3,format=yuv420p",
             "-t", f"{secs:.3f}", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(out)])
    else:  # still slide, no motion
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(SL / f"{asset}.png"),
             "-vf", f"scale={W}:{H},format=yuv420p", "-t", f"{secs:.3f}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "stillimage", str(out)])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_v2", help='voice/output suffix, e.g. "_v2": voiceover_v2.wav -> final_demo_v2.mp4')
    ap.add_argument("--end-hold", type=float, default=END_HOLD)
    args = ap.parse_args()
    out = VID / f"final_demo{args.tag}.mp4"
    tim = json.loads((AUD / f"voiceover{args.tag}_timings.json").read_text())
    beats = tim["beats"]
    total = round(tim["length"] + args.end_hold, 3)
    assert total <= LIMIT, f"{total} s > {LIMIT} s: shorten voiceover_v2.txt"
    assert [b["beat"] for b in beats] == sorted(PARTS), "voiceover_v2.txt beats must match PARTS"
    # beat windows: from LEAD before its sentence to LEAD before the next one
    edges = [0.0] + [max(0.0, b["start"] - LEAD) for b in beats[1:]] + [total]
    plan = []  # (asset, start, length, offset)
    for i, b in enumerate(beats):
        a, z = edges[i], edges[i + 1]
        t = a
        parts = PARTS[b["beat"]]
        for j, (asset, secs, off) in enumerate(parts):
            if secs is None or j == len(parts) - 1:
                length = z - t
            elif secs < 1:
                length = (z - a) * secs
            else:
                length = min(secs, (z - a) - 1.0)
            plan.append((asset, round(t, 3), round(length, 3), off))
            t += length
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        segs = []
        for k, (asset, start, length, off) in enumerate(plan):
            f = td / f"seg{k:02d}.mp4"
            segment(asset, length + (XF if k < len(plan) - 1 else 0), off, f)
            segs.append(f)
        inputs = sum((["-i", str(s)] for s in segs), [])
        fc, prev = [], "[0:v]"
        for k in range(1, len(segs)):
            fc.append(f"{prev}[{k}:v]xfade=transition=fade:duration={XF}:offset={plan[k][1]:.3f}[x{k}]")
            prev = f"[x{k}]"
        fc.append(f"{prev}fade=t=in:st=0:d=0.6,fade=t=out:st={total - 1.0:.3f}:d=1.0,format=yuv420p[v]")
        video = td / "video.mp4"
        run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[v]", "-t", str(total),
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS), str(video)])
        srt = TXT / f"captions_final{args.tag}.srt"
        captions(beats, srt)
        run(["ffmpeg", "-y", "-i", str(video), "-i", str(AUD / f"voiceover{args.tag}.wav"), "-i", str(srt),
             "-filter_complex", f"[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,apad,atrim=0:{total},"
                                f"loudnorm=I=-16:TP=-1.5:LRA=11[a]",
             "-map", "0:v", "-map", "[a]", "-map", "2:s", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-t", str(total), str(out)])
    for asset, start, length, _ in plan:
        print(f"{start:7.2f}s  {length:5.2f}s  {asset}")
    print(out, f"{duration(out):.2f}s (voice: {tim['voice']} {tim.get('blend', '')} @ {tim['speed']})")


if __name__ == "__main__":
    main()
