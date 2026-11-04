"""Build every slide PNG for the demo video.

Each slide is a small HTML page (shared dark theme, real frames, inline SVG charts) that
headless Chrome renders at 1920x1080. All numbers come from run 231613 (merged batches)
unless a slide says otherwise; boxes are read from the run's jsonl files, not typed in.

    python documents/demo-claude/scripts/build_slides.py            # all slides
    python documents/demo-claude/scripts/build_slides.py s11 s13    # just these
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parents[1]
HTML = HERE / "html"
OUT = DEMO / "assets" / "slides"
RUN = REPO / "output/tracking_runs/view3_20261028-1633_1640_replay_db_20261102-231613-423659"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

A = "../../assets"  # assets path as seen from scripts/html/*.html

# ---------------------------------------------------------------- theme
CSS = """
:root{
  --bg:#1a1a19; --panel:#232321; --panel2:#2b2b28; --line:#3a3935;
  --t1:#f2f1ed; --t2:#c3c2b7; --t3:#8e8d84;
  --qb:#3987e5; --qbo:#d95926; --edge:#199e70; --agent:#9085e9;
  --doubt:#fab219; --good:#0ca30c; --bad:#e66767;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;overflow:hidden;background:var(--bg);color:var(--t1);
  font-family:-apple-system,"SF Pro Display","Helvetica Neue",sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:"SF Mono",Menlo,monospace}
.slide{position:absolute;inset:0;padding:88px 112px 110px;display:flex;flex-direction:column;justify-content:center}
.kicker{font-size:26px;letter-spacing:.14em;text-transform:uppercase;color:var(--t3);font-weight:600}
h1{font-size:76px;line-height:1.06;font-weight:700;letter-spacing:-.02em}
h2{font-size:56px;line-height:1.1;font-weight:700;letter-spacing:-.015em}
.lead{font-size:34px;line-height:1.4;color:var(--t2)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:36px 40px}
.pill{display:inline-flex;align-items:center;gap:10px;padding:8px 18px;border-radius:999px;
  font-size:24px;font-weight:600;background:var(--panel2);border:1px solid var(--line)}
.dot{width:14px;height:14px;border-radius:50%;display:inline-block}
.frame{position:relative;border-radius:18px;overflow:hidden;border:1px solid var(--line)}
.frame img{display:block;width:100%;height:100%}
.box{position:absolute;border:4px solid;border-radius:6px}
.tag{position:absolute;padding:6px 14px;border-radius:10px;font-size:24px;font-weight:700;white-space:nowrap;color:#111}
.foot{position:absolute;left:112px;right:112px;bottom:48px;font-size:20px;color:var(--t3);display:flex;justify-content:space-between}
.stamp{display:inline-block;padding:10px 26px;border:4px solid;border-radius:14px;font-size:40px;font-weight:800;letter-spacing:.06em;transform:rotate(-4deg)}
"""


def page(body: str, extra_css: str = "", foot: str = "") -> str:
    footer = f'<div class="foot"><span>{foot}</span><span>Agentic CV · Meta Muse Spark</span></div>' if foot else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}{extra_css}</style></head>
<body>{body}{footer}</body></html>"""


def boxes_on(img: str, w: int, h: int, items: list[dict], src_w: int = 800, src_h: int = 600) -> str:
    """A frame at w x h with boxes in source-pixel coordinates."""
    sx, sy = w / src_w, h / src_h
    parts = [f'<div class="frame" style="width:{w}px;height:{h}px"><img src="{img}">']
    for it in items:
        x1, y1, x2, y2 = it["box"]
        style = it.get("style", "solid")
        parts.append(
            f'<div class="box" style="left:{x1*sx:.0f}px;top:{y1*sy:.0f}px;width:{(x2-x1)*sx:.0f}px;'
            f'height:{(y2-y1)*sy:.0f}px;border-color:{it["color"]};border-style:{style}"></div>'
        )
        if it.get("label"):
            pos = it.get("label_pos", "above")
            top = y1 * sy - 46 if pos == "above" else y2 * sy + 8
            parts.append(
                f'<div class="tag" style="left:{x1*sx:.0f}px;top:{top:.0f}px;background:{it["color"]}">{it["label"]}</div>'
            )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------- data
def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (RUN / f"{name}.jsonl").open()]


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T"))


# ---------------------------------------------------------------- slides
SLIDES: dict[str, tuple[str, str]] = {}


def slide(fn):
    SLIDES[fn.__name__.split("_", 1)[0]] = (fn.__name__, fn)
    return fn


@slide
def s01_meet_qb_qbo():
    css = """
    .wrap{display:flex;gap:56px;align-items:stretch;height:100%}
    .cat{flex:1;position:relative;border-radius:26px;overflow:hidden;border:1px solid var(--line)}
    .cat img{width:100%;height:100%;object-fit:cover}
    .cap{position:absolute;left:0;right:0;bottom:0;padding:120px 44px 40px;
      background:linear-gradient(transparent,rgba(10,10,10,.92))}
    .name{font-size:96px;font-weight:800;letter-spacing:-.02em;display:flex;align-items:center;gap:22px}
    .bar{width:16px;height:76px;border-radius:8px}
    .cue{font-size:32px;color:var(--t2);margin-top:8px}
    """
    body = f"""<div class="slide" style="padding:64px 88px"><div class="wrap">
      <div class="cat"><img src="{A}/reference/qb_standing_front.jpg" style="object-position:50% 30%">
        <div class="cap"><div class="name"><span class="bar" style="background:var(--qb)"></span>Qb</div>
        <div class="cue">cool white short hair · thin upright tail</div></div></div>
      <div class="cat"><img src="{A}/reference/qbo_on_back_face.jpg" style="object-position:50% 45%">
        <div class="cap"><div class="name"><span class="bar" style="background:var(--qbo)"></span>Qbo</div>
        <div class="cue">cream long hair · dark plume tail</div></div></div>
    </div></div>"""
    return page(body, css)


@slide
def s02_ball_decoy():
    css = """
    .grid{display:grid;grid-template-columns:1160px 1fr;gap:72px;align-items:center;height:100%}
    .big{font-size:124px;font-weight:800;letter-spacing:-.03em;line-height:1;white-space:nowrap}
    .unit{font-size:36px;color:var(--t2);font-weight:600;margin-top:10px}
    .stat{margin-bottom:48px}
    """
    frame = boxes_on(f"{A}/frames/2026-11-01-07-35-00-150851_ball_decoy.jpg", 1160, 870,
                     [{"box": [351, 47, 407, 104], "color": "var(--qbo)", "label": "Qbo · 100% agreement"}])
    body = f"""<div class="slide" style="padding:70px 112px 120px"><div class="grid">{frame}
      <div><div class="kicker" style="margin-bottom:40px">What the camera believed</div>
        <div class="stat"><div class="big">26 min</div><div class="unit">tracked as “Qbo”</div></div>
        <div class="stat"><div class="big">0.7 px</div><div class="unit">total movement</div></div>
        <div class="lead" style="font-size:30px">It was an exercise ball.</div>
      </div></div></div>"""
    return page(body, css, "Real frame 2026-11-01 07:35 · box [351,47,407,104] from trace review 2026-11-01")


@slide
def s03_simple_questions():
    css = """
    .bg{position:absolute;inset:0;background:url('%s/frames/2026-11-03-16-30-00-139271_morning_room.jpg') center/cover;filter:brightness(.42) saturate(.8)}
    .qs{display:flex;gap:36px;margin-top:64px}
    .q{flex:1;background:rgba(26,26,25,.82);backdrop-filter:blur(6px);border:1px solid var(--line);border-radius:24px;padding:40px}
    .q .t{font-size:54px;font-weight:700}
    .q .s{font-size:28px;color:var(--t2);margin-top:12px}
    .note{margin-top:56px;display:inline-flex;gap:18px;align-items:center;font-size:34px;
      background:rgba(26,26,25,.82);border:1px solid var(--line);border-left:8px solid var(--doubt);border-radius:16px;padding:24px 34px}
    """ % A
    body = """<div class="bg"></div><div class="slide">
      <div class="kicker">While I'm at work</div>
      <h1 style="margin-top:18px">Did my cats…</h1>
      <div class="qs">
        <div class="q"><div class="t">eat?</div><div class="s">when, and how often</div></div>
        <div class="q"><div class="t">sleep?</div><div class="s">where, and how long</div></div>
        <div class="q"><div class="t">play?</div><div class="s">which cat, how active</div></div>
      </div>
      <div class="note"><b style="color:var(--doubt)">!</b>Cats hide illness, so small changes in routine matter.</div>
    </div>"""
    return page(body, css, "Background: real camera frame, 2026-11-03 16:30 UTC · AAFP: cats are “masters of hiding signs of disease”")


@slide
def s04_cost_math():
    css = """
    .row{display:flex;align-items:center;gap:36px;margin-top:70px}
    .n{font-size:110px;font-weight:800;letter-spacing:-.03em;line-height:1}
    .l{font-size:28px;color:var(--t2);margin-top:12px}
    .op{font-size:80px;color:var(--t3);font-weight:300}
    .res{color:var(--bad)}
    """
    body = """<div class="slide">
      <div class="kicker">Send every frame to an AI?</div>
      <div class="row">
        <div><div class="n">172,800</div><div class="l">frames a day (2 fps, one camera)</div></div>
        <div class="op">×</div>
        <div><div class="n">$0.001</div><div class="l">per frame to a vision LLM</div></div>
        <div class="op">=</div>
        <div><div class="n res">$63,072</div><div class="l">a year, per camera</div></div>
      </div>
      <div class="lead" style="margin-top:90px">A tiny model on a Raspberry Pi is almost free, <b style="color:var(--t1)">but it makes mistakes.</b></div>
    </div>"""
    return page(body, css, "$0.001/frame ≈ 800 input tokens at $1.25/M (TokenCost, Muse Spark pricing)")


@slide
def s05_two_halves():
    css = """
    table{width:100%;border-collapse:separate;border-spacing:0;margin-top:56px;font-size:32px}
    th{font-size:34px;text-align:left;padding:26px 30px;font-weight:700}
    td{padding:24px 30px;border-top:1px solid var(--line);color:var(--t2)}
    td:first-child{color:var(--t3);font-size:26px;text-transform:uppercase;letter-spacing:.08em;width:300px}
    .hl{color:var(--t1)}
    """
    rows = [
        ("Speed", "~67 ms a frame", "30–80 s a review"),
        ("Cost, 24/7", "fixed hardware, ~6–8 W", "$63k a year"),
        ("Best at", "<span class='hl'>where</span> things are", "<span class='hl'>what</span> they mean, and why"),
        ("Fails by", "mixing up cats, chasing balls", "can't watch everything"),
        ("Memory", "seconds of tracking", "one context window"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    body = f"""<div class="slide">
      <div class="kicker">Two halves that fail in opposite ways</div>
      <table><tr><th></th>
        <th><span class="dot" style="background:var(--edge);margin-right:14px"></span>Fast eyes · YOLO on a Pi</th>
        <th><span class="dot" style="background:var(--agent);margin-right:14px"></span>Smart brain · Muse Spark</th></tr>{trs}</table>
    </div>"""
    return page(body, css, "Pi 5 CPU latency: Ultralytics guide · power: Sixfab Pi 5 AI HAT benchmark · review time: our traces")


@slide
def s06_title_built_both():
    body = """<div class="slide">
      <div class="kicker">So we built both</div>
      <h1 style="font-size:104px;margin-top:28px">Cheap eyes on <span style="color:var(--edge)">every frame</span>.<br>
      A smart brain <span style="color:var(--agent)">only when it’s unsure</span>.</h1>
    </div>"""
    return page(body)


@slide
def s07_loop_diagram():
    # Hand-placed SVG flow chart. Main path left->right, review loop below.
    def node(x, y, w, h, title, sub, color, fill="#232321"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="#3a3935" stroke-width="2"/>'
                f'<rect x="{x}" y="{y}" width="10" height="{h}" rx="5" fill="{color}"/>'
                f'<text x="{x+34}" y="{y+52}" font-size="32" font-weight="700" fill="#f2f1ed">{title}</text>'
                + "".join(f'<text x="{x+34}" y="{y+94+i*34}" font-size="24" fill="#c3c2b7">{s}</text>' for i, s in enumerate(sub)))

    def arrow(x1, y1, x2, y2, label="", color="#8e8d84", lx=None, ly=None):
        mid = f'<text x="{lx if lx is not None else (x1+x2)/2}" y="{ly if ly is not None else (y1+y2)/2-14}" font-size="22" fill="#8e8d84" text-anchor="middle">{label}</text>' if label else ""
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#ah)"/>' + mid

    W = 1696
    svg = [f'<svg width="{W}" height="800" viewBox="0 0 {W} 800" font-family="-apple-system,Helvetica Neue">',
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#8e8d84"/></marker></defs>']
    # row 1
    svg.append(node(0, 40, 360, 190, "Pi camera", ["YOLO on every frame", "2 fps · ~free"], "#199e70"))
    svg.append(node(445, 40, 380, 190, "Correctors", ["boxes → tracks → visits", "name their own doubt"], "#199e70"))
    # decision diamond
    cx, cy = 1010, 135
    svg.append(f'<polygon points="{cx},{cy-95} {cx+120},{cy} {cx},{cy+95} {cx-120},{cy}" fill="#2b2b28" stroke="#fab219" stroke-width="3"/>'
               f'<text x="{cx}" y="{cy+10}" font-size="30" font-weight="700" fill="#f2f1ed" text-anchor="middle">Unsure?</text>')
    svg.append(node(1290, 40, 406, 190, "Tracked rows", ["committed as-is", "99% of frames"], "#8e8d84"))
    svg.append(arrow(360, 135, 440, 135))
    svg.append(arrow(825, 135, 886, 135))
    svg.append(arrow(1130, 135, 1285, 135, "no"))
    # row 2: review loop
    svg.append(node(820, 400, 380, 220, "Plain-English", ["question + key frames", "+ strict JSON schema", "≤4 cases per call"], "#fab219"))
    svg.append(node(1290, 400, 406, 220, "Muse Spark agent", ["pulls frames via MCP", "compares the cats", "verdict + rationale"], "#9085e9"))
    svg.append(arrow(cx, cy + 95, cx, 395, "yes · ~1%", lx=cx + 20, ly=320).replace('text-anchor="middle">yes', 'text-anchor="start">yes'))
    svg.append(arrow(1200, 510, 1285, 510))
    # verdict back up to rows
    svg.append(f'<path d="M1493,400 L1493,236" stroke="#9085e9" stroke-width="3" fill="none" marker-end="url(#ah)"/>'
               f'<text x="1478" y="326" font-size="22" fill="#c3c2b7" text-anchor="end">verdict rewrites rows</text>')
    # memory
    svg.append(node(0, 400, 700, 220, "Spatial-temporal memory", ["trusted visits → day log (plain text)",
               "cells → zones → view · 5 min → hour → day → week", "brief attached to the next question"], "#3987e5"))
    svg.append(f'<path d="M1493,620 L1493,700 L350,700 L350,624" stroke="#3987e5" stroke-width="3" fill="none" marker-end="url(#ah)"/>'
               f'<text x="922" y="740" font-size="22" fill="#c3c2b7" text-anchor="middle">trusted visits + verdicts</text>')
    svg.append(arrow(700, 510, 815, 510, "brief", lx=757, ly=492))
    svg.append("</svg>")
    body = f"""<div class="slide"><div class="kicker">The loop</div>
      <div style="margin-top:56px">{''.join(svg)}</div></div>"""
    return page(body)


@slide
def s08_timeline():
    rev = load("revised")
    esc = load("escalations")
    verdicts = {}
    for l in (RUN / "validations.jsonl").open():
        d = json.loads(l)
        verdicts[d["escalation"]["name"]] = d["review"]["verdict"]
    corrected = {"keep_original", "cut_at_gap_start"}
    t0 = ts("2026-10-28T16:33:00")
    t1 = ts("2026-10-28T16:41:00")
    X0, X1 = 150, 1640
    span = (t1 - t0).total_seconds()

    def x(t):
        return X0 + (ts(t) - t0).total_seconds() / span * (X1 - X0)

    lanes = {25: ("Qb", 470, "#3987e5"), 26: ("Qbo", 590, "#d95926")}
    svg = ['<svg width="1696" height="800" viewBox="0 0 1696 800" font-family="-apple-system,Helvetica Neue">']
    # axis
    for m in range(0, 9):
        xx = X0 + m * 60 / span * (X1 - X0)
        svg.append(f'<line x1="{xx}" y1="400" x2="{xx}" y2="660" stroke="#2f2e2b" stroke-width="1"/>'
                   f'<text x="{xx}" y="700" font-size="22" fill="#8e8d84" text-anchor="middle">16:{33+m:02d}</text>')
    for cid, (name, y, col) in lanes.items():
        svg.append(f'<text x="{X0-24}" y="{y+10}" font-size="28" font-weight="700" fill="#f2f1ed" text-anchor="end">{name}</text>')
        for r in rev:
            if r["cat_id"] != cid:
                continue
            xx = x(r["create_time"])
            op = "1" if r.get("orig_cat_id") is not None else "0.35"
            svg.append(f'<rect x="{xx:.1f}" y="{y-26}" width="2" height="52" fill="{col}" opacity="{op}"/>')
    # escalation markers; stack labels that collide
    placed: list[tuple[float, int]] = []
    lines: list[str] = []
    chips: list[str] = []
    for e in sorted(esc, key=lambda e: e["time"]):
        xx = x(e["time"])
        level = 0
        while any(abs(xx - px) < 74 and lv == level for px, lv in placed):
            level += 1
        placed.append((xx, level))
        v = verdicts.get(e["name"], "")
        col = "#9085e9" if v in corrected else "#8e8d84"
        ly = 360 - level * 52
        lines.append(f'<line x1="{xx:.1f}" y1="{ly+12}" x2="{xx:.1f}" y2="650" stroke="#fab219" stroke-width="2" opacity=".9"/>')
        chips.append(f'<rect x="{xx-36:.1f}" y="{ly-26}" width="72" height="40" rx="8" fill="{col}"/>'
                   f'<text x="{xx:.1f}" y="{ly+2}" font-size="21" font-weight="700" fill="#111" text-anchor="middle">{e["name"].split("-")[1]}</text>')
    svg.extend(lines)
    svg.extend(chips)
    svg.append('<text x="1640" y="545" font-size="23" fill="#c3c2b7" text-anchor="end">↓ mostly Qb asleep in the bed, mislabeled · no doubt fired · open problem</text>')
    # legend
    svg.append('<g transform="translate(150,40)" font-size="24" fill="#c3c2b7">'
               '<rect x="0" y="-18" width="22" height="22" rx="4" fill="#fab219"/><text x="34" y="2">doubt raised by the Pi (12 in 8 min · 0.9% of frames)</text>'
               '<rect x="760" y="-18" width="22" height="22" rx="4" fill="#9085e9"/><text x="794" y="2">agent corrected the tracker (4)</text>'
               '<rect x="1200" y="-18" width="22" height="22" rx="4" fill="#8e8d84"/><text x="1234" y="2">agent upheld (8)</text>'
               '<rect x="0" y="34" width="22" height="22" rx="4" fill="#d95926" opacity=".35"/><text x="34" y="54">faint tick = guessed (interpolated) frame · solid = real detection</text></g>')
    svg.append("</svg>")
    body = f"""<div class="slide"><div class="kicker">8 real minutes · both cats on camera · 1,319 frames</div>
      <div style="margin-top:40px">{''.join(svg)}</div></div>"""
    return page(body, foot="Tracker output, run 231613 (view 3, 2026-10-28 16:33–16:41). Lane = tracker's label, before review.")


@slide
def s09_doubt_taxonomy():
    frame = [("T1", "multi-match relabel"), ("T2", "relabeled by distance alone"), ("T3", "duplicate tie"),
             ("T4", "jump reset"), ("T5", "target fallback"), ("T6", "teleport split"), ("T7", "long guessed run")]
    visit = [("P1", "close vote"), ("P2", "weak winner"), ("P3", "conflict reassign"), ("P4", "long unknown"),
             ("P5", "low quality"), ("P6", "merge candidate"), ("P7", "static link")]
    fired = {"T2": 3, "T4": 1, "T6": 1, "T7": 3, "P2": 2, "P5": 1, "P6": 1}
    css = """
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:48px;margin-top:44px}
    .colh{font-size:26px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:18px;font-weight:600}
    .cell{display:flex;align-items:center;gap:22px;padding:15px 24px;border-radius:14px;background:var(--panel);
      border:1px solid var(--line);margin-bottom:12px;font-size:28px;color:var(--t2)}
    .code{font-family:"SF Mono",Menlo,monospace;font-weight:700;width:56px;color:var(--t1)}
    .n{margin-left:auto;font-size:22px;font-weight:700;color:#111;background:var(--doubt);border-radius:999px;padding:3px 14px}
    .hero{border:3px solid var(--doubt);background:#2e2a1f;color:var(--t1)}
    .new{margin-top:22px;font-size:24px;color:var(--t3)}
    """
    def cells(items):
        out = []
        for c, t in items:
            cls = "cell hero" if c == "T2" else "cell"
            n = f'<span class="n">×{fired[c]}</span>' if c in fired else ""
            out.append(f'<div class="{cls}"><span class="code">{c}</span>{t}{n}</div>')
        return "".join(out)
    body = f"""<div class="slide">
      <div class="kicker">The Pi knows 14 kinds of doubt</div>
      <div class="cols"><div><div class="colh">One frame</div>{cells(frame)}</div>
      <div><div class="colh">A whole visit</div>{cells(visit)}</div></div>
      <div class="new">Badges = fired in this 8-minute window. Added since: P8 label-vs-vote disagreement, H1 quiet-window spot-check.</div>
    </div>"""
    return page(body, css)


@slide
def s10_question_card():
    css = """
    pre{font-family:"SF Mono",Menlo,monospace;font-size:29px;line-height:1.55;color:var(--t2);white-space:pre-wrap}
    .k{color:var(--doubt)} .h{color:var(--t1);font-weight:700}
    """
    text = """<span class="h">[E5-T2] A box was relabeled by nearness alone.</span>

<span class="k">Validate:</span>  is the photo really cat 26 (Qbo)?
<span class="k">Context:</span>   the detector's first guess was cat 25 (Qb). Qbo's last box
           was 129 px away, and that nearness is the entire basis
           for the switch. Nearness can steal a neighbor.
<span class="k">Look at:</span>   2026-10-28-16-34-36-154433.jpg  box [296,369,371,531]
<span class="k">Note:</span>      the Pi's YOLO is a hint, not evidence. Trust the images
           and cat profiles.

<span class="k">Answer:</span>    {"verdict": confirm | keep_original | reassign_other | drop,
            "rationale": "…"}"""
    body = f"""<div class="slide">
      <div class="kicker">Doubt becomes a plain-English question</div>
      <div class="card" style="margin-top:44px;padding:48px 56px"><pre>{text}</pre></div>
    </div>"""
    return page(body, css, "Condensed from the real E5-T2 prompt and T2 schema (agent run 20261102-231717, agent/escalation/review_schema.py)")


@slide
def s11_case_e5():
    frame = boxes_on(f"{A}/frames/2026-10-28-16-34-36-154433_E5.jpg", 1120, 840,
                     [{"box": [296, 369, 371, 531], "color": "var(--qbo)", "label": "tracker: Qbo"},
                      {"box": [296, 369, 371, 531], "color": "var(--qb)", "label": "agent: Qb", "label_pos": "below"}])
    css = """
    .grid{display:grid;grid-template-columns:1120px 1fr;gap:56px;height:100%;align-items:center}
    .refs{display:flex;gap:18px;margin:26px 0 30px}
    .ref{flex:1;height:250px;border-radius:14px;overflow:hidden;position:relative;border:1px solid var(--line)}
    .ref img{width:100%;height:100%;object-fit:cover}
    .ref span{position:absolute;left:12px;bottom:10px;font-size:22px;font-weight:700;padding:3px 12px;border-radius:8px;color:#111}
    blockquote{font-size:34px;line-height:1.4;color:var(--t1);border-left:6px solid var(--agent);padding-left:26px}
    """
    body = f"""<div class="slide" style="padding:120px 112px 100px"><div class="grid">{frame}
      <div><div class="kicker">Muse Spark reviews E5</div>
        <div class="refs">
          <div class="ref"><img src="{A}/reference/qb_standing_front.jpg" style="object-position:50% 40%"><span style="background:var(--qb)">Qb</span></div>
          <div class="ref"><img src="{A}/reference/qbo_lying_side.jpg" style="object-position:60% 50%"><span style="background:var(--qbo)">Qbo</span></div></div>
        <blockquote>“Cool white, smooth outline, thin upright tail. This is Qb, not Qbo.”</blockquote>
        <div style="margin-top:44px"><span class="stamp" style="color:var(--agent);border-color:var(--agent)">OVERTURNED</span></div>
      </div></div></div>"""
    return page(body, css, "Frame 2026-10-28 16:34:36 · verdict keep_original · rationale shortened from trace 20261102-231717")


@slide
def s12_batching():
    groups = [("Call 1", ["E1 T7", "E2 T6", "E3 P5", "E4 T7"]), ("Call 2", ["E5 T2", "E6 T2", "E7 T2", "E8 T4"]),
              ("Call 3", ["E9 P6", "E10 T7", "E11 P2", "E12 P2"])]
    metrics = [("Agent calls", 12, 3, "{:.0f}"), ("Tokens", 517.6, 229.6, "{:.0f}k"),
               ("Cost", 0.60, 0.26, "${:.2f}"), ("Minutes", 9.4, 3.8, "{:.1f}")]
    css = """
    .grid{display:grid;grid-template-columns:820px 1fr;gap:80px;margin-top:50px}
    .g{display:flex;align-items:center;gap:20px;margin-bottom:26px}
    .chips{display:flex;gap:10px;flex-wrap:wrap;width:500px}
    .chip{font-family:"SF Mono",Menlo,monospace;font-size:22px;padding:8px 12px;border-radius:10px;background:#2e2a1f;border:2px solid var(--doubt)}
    .call{font-size:26px;font-weight:700;padding:16px 22px;border-radius:14px;background:#26243a;border:2px solid var(--agent);white-space:nowrap}
    .m{margin-bottom:34px}
    .mh{font-size:26px;color:var(--t2);margin-bottom:12px;display:flex;justify-content:space-between}
    .b{height:30px;border-radius:0 8px 8px 0;margin-bottom:6px;display:flex;align-items:center}
    .b span{font-size:22px;margin-left:14px;color:var(--t2);white-space:nowrap}
    """
    left = "".join(
        f'<div class="g"><div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in cs)}</div>'
        f'<svg width="70" height="20"><line x1="0" y1="10" x2="58" y2="10" stroke="#8e8d84" stroke-width="3"/>'
        f'<path d="M56,3 L68,10 L56,17 z" fill="#8e8d84"/></svg><span class="call">{g}</span></div>'
        for g, cs in groups)
    right = []
    for name, solo, merged, fmt in metrics:
        pct = round((1 - merged / solo) * 100)
        right.append(f'<div class="m"><div class="mh"><span>{name}</span><b style="color:var(--t1)">−{pct}%</b></div>'
                     f'<div class="b" style="width:{560}px;background:#4a4944"><span style="position:relative;left:{560}px">one call each · {fmt.format(solo)}</span></div>'
                     f'<div class="b" style="width:{560*merged/solo:.0f}px;background:var(--agent)"><span style="position:relative;left:{560*merged/solo:.0f}px">merged · {fmt.format(merged)}</span></div></div>')
    body = f"""<div class="slide">
      <div class="kicker">Related doubts are reviewed together</div>
      <h2 style="margin-top:16px">12 questions, 3 calls. Each image fetched once.</h2>
      <div class="grid"><div>{left}</div><div>{''.join(right)}</div></div></div>"""
    return page(body, css, "Runs 012059 (one call per escalation) vs 231613 (merged batches ≤60 s, ≤4 cases) · same window, same tracker output")


@slide
def s13_case_e10():
    f1 = boxes_on(f"{A}/frames/2026-10-28-16-35-36-588746_E10.jpg", 1000, 750,
                  [{"box": [178, 465, 266, 585], "color": "var(--qbo)", "label": "tracker: Qbo (guessed 5.2 s)", "style": "dashed"},
                   {"box": [92, 268, 152, 338], "color": "var(--qbo)", "label": "Qbo is here", "label_pos": "above"}])
    css = """
    .grid{display:grid;grid-template-columns:1000px 1fr;gap:64px;margin-top:40px;align-items:center}
    blockquote{font-size:32px;line-height:1.42;color:var(--t1);border-left:6px solid var(--agent);padding-left:26px}
    .meta{font-size:26px;color:var(--t2);margin-top:30px;line-height:1.6}
    """
    body = f"""<div class="slide"><div class="kicker">Muse Spark reviews E10 · run of guessed frames</div>
      <div class="grid">{f1}
      <div><h2 style="font-size:50px">The tracker kept painting Qbo onto Qb’s spot.</h2>
        <blockquote style="margin-top:34px">“No cream Qbo at the guessed position at start, mid or end. The track imagined presence. Cut the whole guessed run.”</blockquote>
        <div class="meta">Qbo had walked into the cat house.<br>14 imagined rows removed.</div>
        <div style="margin-top:34px"><span class="stamp" style="color:var(--agent);border-color:var(--agent)">CUT</span></div>
      </div></div></div>"""
    return page(body, css, "Frame 2026-10-28 16:35:36 · verdict cut_at_gap_start · cat-house box drawn by hand for the viewer")


@slide
def s14_row_diff():
    css = """
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:48px;margin-top:56px}
    pre{font-family:"SF Mono",Menlo,monospace;font-size:27px;line-height:1.65;color:var(--t2)}
    .lbl{font-size:24px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);margin-bottom:18px;font-weight:600}
    .del{background:rgba(230,103,103,.16);color:#f2b0b0;border-radius:6px;padding:0 6px}
    .add{background:rgba(144,133,233,.2);color:#d6d0ff;border-radius:6px;padding:0 6px}
    """
    before = """{
  "create_time": "16:34:36.154",
  "box": [296, 369, 371, 531],
  "conf": 0.758,
  "orig_cat_id": 25,
  <span class="del">"cat_id": 26</span>
}"""
    after = """{
  "create_time": "16:34:36.154",
  "box": [296, 369, 371, 531],
  "conf": 0.758,
  "orig_cat_id": 25,
  <span class="add">"cat_id": 25</span>,
  <span class="add">"validated_by": "E5-T2"</span>,
  <span class="add">"verdict": "keep_original"</span>
}"""
    body = f"""<div class="slide"><div class="kicker">Every verdict rewrites the record</div>
      <h2 style="margin-top:16px">…and every change points back to its trace.</h2>
      <div class="cols"><div class="card"><div class="lbl">Tracker row (revised.jsonl)</div><pre>{before}</pre></div>
      <div class="card" style="border-color:var(--agent)"><div class="lbl">After review (validated.db)</div><pre>{after}</pre></div></div></div>"""
    return page(body, css, "Real row from run 231613 · production MySQL is read-only; validated rows go to a per-run SQLite file")


@slide
def s15_day_log():
    css = """
    .paper{margin-top:44px;background:#f4f1ea;color:#262521;border-radius:22px;padding:56px 72px;font-family:"New York","Georgia",serif}
    .d{font-family:-apple-system;font-size:24px;letter-spacing:.1em;text-transform:uppercase;color:#7a776d;font-weight:600}
    .e{font-size:40px;line-height:1.45;margin-top:26px}
    .t{font-family:"SF Mono",Menlo,monospace;font-size:30px;color:#3b6fb0;font-weight:700}
    .s{font-size:28px;color:#5c5a52;margin-top:10px;font-family:-apple-system}
    hr{border:none;border-top:1px solid #d8d3c6;margin:34px 0}
    """
    body = """<div class="slide"><div class="kicker">Memory: a diary, not boxes and timestamps</div>
      <div class="paper"><div class="d">View 3 · Tuesday, Oct 28</div>
        <div class="e"><span class="t">16:34–16:40</span> &nbsp;<b>Qb slept</b> curled up tight in the small green bed, paws tucked in.</div>
        <div class="s">sleeping · confidence 0.92 · 772 sightings · labeled by Muse Spark vision</div>
        <hr>
        <div class="d">Long-term memory · bottom-center zone</div>
        <div class="e" style="font-size:36px">“qbo visits twice on 2026-10-27 from 18:00 to 18:24 and sleeps curled in the green bed at bottom-center.”</div>
        <div class="s">rolled up by Muse Spark from trusted visits only</div>
      </div></div>"""
    return page(body, css, "Day entry: view_3_short_term_2026-10-28.md (latest session) · zone text: view_3_long_term.yaml")


@slide
def s16_memory_pyramid():
    css = """
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:72px;margin-top:56px}
    .lvl{margin:0 auto 14px;border-radius:14px;padding:20px 28px;display:flex;justify-content:space-between;align-items:center;font-size:28px}
    .lvl b{font-weight:700} .lvl span{color:var(--t2);font-size:23px}
    .h{font-size:26px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:22px;font-weight:600;text-align:center}
    .gate{margin-top:46px;display:flex;gap:28px;justify-content:center}
    """
    blues = ["#184f95", "#1c5cab", "#256abf", "#2a78d6"]
    sp = [("Whole view", "1 summary", 58), ("Zones", "12", 76), ("Cells", "48 × 100 px", 94)]
    tp = [("Weeks", "habits · kept forever", 58), ("Days", "routine · 180 days", 70), ("Hours", "activity mix · 30 days", 82), ("5-min slots", "events · 7 days", 94)]
    def stack(items, cols):
        return "".join(f'<div class="lvl" style="width:{w}%;background:{c}"><b>{a}</b><span>{b}</span></div>' for (a, b, w), c in zip(items, cols))
    body = f"""<div class="slide"><div class="kicker">Over weeks: what’s normal, for each cat, in each spot</div>
      <div class="cols"><div><div class="h">Where</div>{stack(sp, blues[1:])}</div>
      <div><div class="h">When</div>{stack(tp, blues)}</div></div>
      <div class="gate"><span class="pill"><span class="dot" style="background:var(--good)"></span>only trusted visits (≥ 0.7) enter</span>
        <span class="pill"><span class="dot" style="background:var(--doubt)"></span>a pattern needs 3 sightings</span>
        <span class="pill"><span class="dot" style="background:var(--bad)"></span>flips only after 3 contradictions</span></div></div>"""
    return page(body, css)


@slide
def s17_results():
    css = """
    .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:28px;margin-top:50px}
    .tile{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:34px 36px}
    .v{font-size:84px;font-weight:800;letter-spacing:-.03em;line-height:1}
    .l{font-size:25px;color:var(--t2);margin-top:14px;line-height:1.35}
    .bars{margin-top:56px}
    .r{display:grid;grid-template-columns:420px 1fr;align-items:center;margin-bottom:20px;font-size:27px;color:var(--t2)}
    .bar{height:44px;border-radius:0 10px 10px 0;display:flex;align-items:center}
    .bar span{margin-left:16px;font-weight:700;color:var(--t1);white-space:nowrap;position:relative}
    """
    W = 1100
    rows = [("Every frame to the LLM", 1.32, "#4a4944", "$1.32 · 100%"),
            ("One call per doubt", 0.60, "#6d6b64", "$0.60 · 45%"),
            ("Merged review (ours)", 0.26, "var(--agent)", "$0.26 · 19%")]
    bars = "".join(f'<div class="r"><span>{n}</span><div class="bar" style="width:{W*v/1.32:.0f}px;background:{c}"><span style="left:{W*v/1.32:.0f}px">{t}</span></div></div>' for n, v, c, t in rows)
    body = f"""<div class="slide"><div class="kicker">Results · 8 real minutes, both cats</div>
      <div class="tiles">
        <div class="tile"><div class="v">12</div><div class="l">doubts raised<br>0.9% of 1,319 frames</div></div>
        <div class="tile"><div class="v">12/12</div><div class="l">verdicts returned,<br>each with a reason</div></div>
        <div class="tile"><div class="v">$0.26</div><div class="l">to review them all<br>≈5× cheaper than all frames</div></div>
        <div class="tile"><div class="v">~1%</div><div class="l">of frames ever<br>leave the home</div></div>
      </div>
      <div class="bars"><div class="kicker" style="font-size:22px;margin-bottom:22px">Review cost for the same window</div>{bars}</div></div>"""
    return page(body, css, "Run 231613 vs 012059; all-frames baseline = 1,319 frames × $0.001 · memory adds ~$0.14 (activity labels, store, promote)")


@slide
def s18_roadmap():
    css = """
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:36px;margin-top:64px}
    .c{background:var(--panel);border:1px solid var(--line);border-radius:24px;padding:44px 42px;min-height:420px}
    .num{font-family:"SF Mono",Menlo,monospace;font-size:28px;color:var(--t3)}
    .t{font-size:46px;font-weight:700;margin-top:22px;line-height:1.12}
    .d{font-size:27px;color:var(--t2);margin-top:22px;line-height:1.45}
    """
    cards = [("01", "Catch more errors", "Escalate whole runs of nearness relabels, and visits whose label disagrees with the agent’s own verdicts.", "var(--doubt)"),
             ("02", "Run full days", "Replace the cost projection with a measured 24-hour bill.", "var(--edge)"),
             ("03", "Teach the Pi", "Verdicts → labeled dataset → retrain the edge model, so fewer doubts reach the agent.", "var(--agent)")]
    html = "".join(f'<div class="c" style="border-top:6px solid {col}"><div class="num">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d, col in cards)
    body = f"""<div class="slide"><div class="kicker">Next</div><div class="cards">{html}</div></div>"""
    return page(body, css)


@slide
def s19_beyond_cats():
    css = """
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:36px;margin-top:60px}
    .c{border-radius:24px;overflow:hidden;background:var(--panel);border:1px solid var(--line)}
    .ph{height:400px;background:repeating-linear-gradient(45deg,#262624 0 18px,#2b2b28 18px 36px);display:flex;align-items:center;justify-content:center;color:var(--t3);font-size:24px;text-align:center;padding:30px}
    .t{font-size:40px;font-weight:700;padding:30px 36px 8px}
    .d{font-size:25px;color:var(--t2);padding:0 36px 34px;line-height:1.4}
    """
    items = [("An elderly parent", "Consent-based routine-change alerts; footage stays home.", "B-ROLL 19a<br>hands, kettle, armchair<br>(no faces)"),
             ("Wildlife cameras", "Automated sorting; only the hard sightings get deep review.", "B-ROLL 19b<br>trail-cam deer at night"),
             ("A factory line", "Fast checks on every part; LLM review of the odd ones.", "B-ROLL 19c<br>conveyor close-up")]
    html = "".join(f'<div class="c"><div class="ph">{p}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for t, d, p in items)
    body = f"""<div class="slide"><div class="kicker">Cats are just the start</div>
      <h2 style="margin-top:16px">Any camera watching for rare moments has the same gap.</h2><div class="cards">{html}</div></div>"""
    return page(body, css)


@slide
def s20_end_card():
    css = """
    .three{display:grid;grid-template-columns:repeat(3,1fr);gap:40px;margin-top:70px}
    .x{border-top:6px solid;padding-top:26px}
    .x .t{font-size:76px;font-weight:700;line-height:1.1;letter-spacing:-.02em}
    .x .d{font-size:36px;color:var(--t2);margin-top:16px}
    .by{position:absolute;left:112px;bottom:96px;font-size:30px;color:var(--t2)}
    """
    body = """<div class="slide"><div class="kicker">Agentic computer vision with memory</div>
      <div class="three">
        <div class="x" style="border-color:var(--edge)"><div class="t">Cheap eyes</div><div class="d">on every frame</div></div>
        <div class="x" style="border-color:var(--agent)"><div class="t">A smart brain</div><div class="d">only when it matters</div></div>
        <div class="x" style="border-color:var(--qb)"><div class="t">A memory</div><div class="d">that keeps growing</div></div>
      </div>
      <div class="by">Built this week on <b style="color:var(--t1)">Meta Muse Spark</b> · Muyang Lu &amp; Terry Li · Meta Global AI Developer Hackathon 2026</div></div>"""
    return page(body, css)


# ---------------------------------------------------------------- render
def render(key: str) -> Path:
    name, fn = SLIDES[key]
    HTML.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    h = HTML / f"{name}.html"
    h.write_text(fn())
    png = OUT / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    "--allow-file-access-from-files", f"--screenshot={png}", "--window-size=1920,1080",
                    "--virtual-time-budget=3000", h.as_uri()], check=True, capture_output=True)
    return png


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SLIDES)
    for k in keys:
        print(render(k).relative_to(DEMO))
