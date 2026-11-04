"""Derive narration.txt and captions.srt from demo-script-claude.md, so they never drift.

Each beat heading looks like  ### Beat 7 · Title · 0:44–0:55 (11 s)
followed by a                 > **VO:** "…"  line.
Captions split each VO line into chunks of <= 42 characters, timed by word share.
"""

from __future__ import annotations

import re
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
SCRIPT = DEMO / "demo-script-claude.md"
TEXT = DEMO / "assets" / "text"

HEAD = re.compile(r"^### Beat (\d+) · (.+?) · (\d+):(\d\d)–(\d+):(\d\d)")
VO = re.compile(r'^> \*\*VO:\*\* "(.+)"\s*$')


def stamp(s: float) -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def chunks(text: str, limit: int = 42) -> list[str]:
    """Sentence first, then wrap each sentence at <= limit characters."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.?!:])\s+", text):
        out += _wrap(sent, limit)
    return out


def _wrap(text: str, limit: int) -> list[str]:
    out, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > limit:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def main() -> None:
    beats, cur = [], None
    for line in SCRIPT.read_text().splitlines():
        if m := HEAD.match(line):
            n, title, m1, s1, m2, s2 = m.groups()
            cur = {"n": int(n), "title": title, "t0": int(m1) * 60 + int(s1), "t1": int(m2) * 60 + int(s2)}
            beats.append(cur)
        elif (m := VO.match(line)) and cur is not None:
            cur["vo"] = m.group(1)
    TEXT.mkdir(parents=True, exist_ok=True)
    words = sum(len(b["vo"].split()) for b in beats)
    lines = [f"# Narration · {words} words · read at ~2.5 words/s, ~1 s pause between beats", ""]
    lines += [f"[{b['n']:02d} · {b['t0'] // 60}:{b['t0'] % 60:02d}] {b['vo']}" for b in beats]
    (TEXT / "narration.txt").write_text("\n".join(lines) + "\n")

    srt, idx = [], 1
    for b in beats:
        parts = chunks(b["vo"])
        total = sum(len(p.split()) for p in parts)
        speak = min(b["t1"] - b["t0"] - 0.4, total / 2.5)  # speech ends before the beat does
        t = b["t0"] + 0.2
        for p in parts:
            d = speak * len(p.split()) / total
            srt += [str(idx), f"{stamp(t)} --> {stamp(t + d)}", p, ""]
            idx += 1
            t += d
    (TEXT / "captions.srt").write_text("\n".join(srt))
    print(f"{len(beats)} beats, {words} words, last beat ends {beats[-1]['t1'] // 60}:{beats[-1]['t1'] % 60:02d}")


if __name__ == "__main__":
    main()
