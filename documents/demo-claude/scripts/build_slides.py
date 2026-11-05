"""Build every slide PNG for the demo video.

Each slide is a small HTML page (shared dark theme, real frames, inline SVG charts) that
headless Chrome renders at 1920x1080. All numbers come from replay run 140502 (merged review)
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
RUN = REPO / "output/tracking_runs/view3_20261028-1633_1640_replay_db_20261104-140502-443911"
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


def HEAD(question: str, conclusion: str) -> str:
    """Slide header: the problem as the kicker, the conclusion as the headline."""
    return f'<div class="kicker">{question}</div><h2 style="margin-top:14px">{conclusion}</h2>'


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
            left = x2 * sx + 10 if pos == "right" else x1 * sx
            top = {"above": y1 * sy - 46, "below": y2 * sy + 8, "right": y1 * sy}[pos]
            parts.append(
                f'<div class="tag" style="left:{left:.0f}px;top:{top:.0f}px;background:{it["color"]}">{it["label"]}</div>'
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
def s06b_bridge():
    # Opening: the camera (cheap eyes) and your privately deployed agent (smart brain) are two
    # systems; this project is the bridge between them. The proof strip previews s17.
    css = """
    .k2{font-size:30px;letter-spacing:.14em;text-transform:uppercase;color:var(--t1);font-weight:700}
    .big{font-size:66px;font-weight:700;line-height:1.1;letter-spacing:-.02em;margin-top:20px}
    .big .b{color:var(--agent)} .big .e{color:var(--edge)}
    .sub{font-size:26px;color:var(--t3);margin-top:14px}
    .bridge{display:grid;grid-template-columns:1fr 150px 1.25fr 150px 1fr;align-items:stretch;margin-top:48px}
    .bx{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:26px 30px;border-top:6px solid}
    .bx .r{font-size:19px;letter-spacing:.12em;text-transform:uppercase;font-weight:700}
    .bx .t{font-size:38px;font-weight:700;margin-top:10px;line-height:1.1}
    .bx ul{list-style:none;margin-top:14px}
    .bx li{font-size:22px;color:var(--t2);line-height:1.5;padding-left:20px;position:relative}
    .bx li:before{content:"";position:absolute;left:2px;top:13px;width:8px;height:8px;border-radius:50%;background:currentColor;opacity:.7}
    .mid{background:linear-gradient(180deg,#26243a,#232321);border:2px solid var(--agent);border-top:6px solid var(--qb)}
    .ar{display:flex;flex-direction:column;justify-content:center;gap:26px;padding:0 8px}
    .ar div{font-size:18px;color:var(--t2);text-align:center}
    .ar svg{display:block;margin:4px auto 0}
    .by{position:absolute;left:112px;bottom:70px;font-size:24px;color:var(--t3)}
    .proof{display:flex;gap:18px;margin-top:34px}
    """
    proof = "".join(f'<span class="pill"><b style="color:var(--t1)">{v}</b>&nbsp;{t}</span>' for v, t in
                    (("−93%", "false alarms"), ("99%", "of what the camera saw, right cat"), ("$0.39", "per 8 minutes"),
                     ("~1%", "of frames call the AI")))

    def arrow(right, color):
        d = "M4,10 L134,10" if right else "M136,10 L6,10"
        return (f'<svg width="140" height="20"><defs><marker id="m{color[-4:]}{right}" viewBox="0 0 10 10" refX="8" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>'
                f'<path d="{d}" stroke="{color}" stroke-width="3" marker-end="url(#m{color[-4:]}{right})"/></svg>')

    body = f"""<div class="slide"><div class="k2">Agentic Realtime Computer Vision</div>
      <div class="big">AI gives the camera <span class="b">a brain</span>.<br>The camera gives AI <span class="e">eyes</span>.</div>
      <div class="sub">Running 24 hours a day, all year round.</div>
      <div class="bridge">
        <div class="bx" style="border-top-color:var(--edge)"><div class="r" style="color:var(--edge)">Cheap eyes</div>
          <div class="t">A camera</div><ul style="color:var(--edge)"><li><span style="color:var(--t2)">tiny model, every frame</span></li>
          <li><span style="color:var(--t2)">on device, ~free</span></li></ul></div>
        <div class="ar"><div>every frame{arrow(True, "#8e8d84")}</div></div>
        <div class="bx mid"><div class="r" style="color:#c9c2ff">This project · the bridge</div>
          <div class="t">Agentic realtime CV</div><ul style="color:var(--agent)">
          <li><span style="color:var(--t2)">names its own doubts</span></li>
          <li><span style="color:var(--t2)">asks in plain English</span></li>
          <li><span style="color:var(--t2)">remembers what’s normal</span></li></ul></div>
        <div class="ar"><div>~1% doubts{arrow(True, "#fab219")}</div><div>{arrow(False, "#9085e9")}verdicts</div></div>
        <div class="bx" style="border-top-color:var(--agent)"><div class="r" style="color:var(--agent)">Smart brain</div>
          <div class="t">Your private agent</div><ul style="color:var(--agent)"><li><span style="color:var(--t2)">deployed privately</span></li>
          <li><span style="color:var(--t2)">e.g. Muse Glimmer, locally</span></li></ul></div>
      </div>
      <div class="proof">{proof}</div>
      <div class="by">Built this week on <b style="color:var(--t1)">Meta Muse Spark</b> · Muyang Lu &amp; Terry Li · Meta Global AI Developer Hackathon 2026</div></div>"""
    return page(body, css)


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
    # Measured, not estimated: documents/labeling-sam-vs-spark-20261103.md labeled every frame of
    # the same 8-minute window (1,318 frames) with AI. Problem slide only: our result is on s17.
    css = """
    .tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:28px}
    .tile{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:26px 34px}
    .v{font-size:72px;font-weight:800;letter-spacing:-.03em;line-height:1}
    .l{font-size:24px;color:var(--t2);margin-top:12px;line-height:1.35}
    .l b{color:var(--t1)}
    table{width:100%;border-collapse:collapse;margin-top:34px;font-size:23px}
    th{font-size:20px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);font-weight:600;text-align:left;padding:0 18px 12px}
    td{padding:13px 18px;border-top:1px solid var(--line);color:var(--t2);white-space:nowrap}
    td.n{font-family:"SF Mono",Menlo,monospace;color:var(--t1)}
    .bar{display:inline-block;height:16px;border-radius:0 6px 6px 0;vertical-align:middle;margin-right:14px}
    tr.ours td{color:var(--t1);font-weight:600}
    .scale{margin-top:26px;font-size:30px;color:var(--t2)}
    .pi{margin-top:26px;display:flex;gap:26px;align-items:baseline;background:var(--panel);border:1px solid var(--line);
      border-left:6px solid var(--edge);border-radius:16px;padding:18px 28px}
    .pk{font-size:19px;letter-spacing:.1em;text-transform:uppercase;color:var(--edge);font-weight:700;white-space:nowrap}
    .pt{font-size:23px;color:var(--t2);line-height:1.45} .pt b{color:var(--t1)} .pt span{color:var(--t1)}
    .scale b{color:var(--bad);font-size:38px}
    """
    rows = [  # approach, minutes for the 8-min window, $ for the window, F1 @ IoU 0.5, bar colour, row class
        ("SAM 3.1 boxes → Muse Spark names the cat", 66, 24.65, "0.79", "#6d6b64", ""),
        ("Muse Spark 1.3 alone (detect + name)", 99, 25.61, "0.61", "#4a4944", ""),
    ]
    TW, CW = 260, 260  # bar widths for the max time / max cost

    def tr(name, mins, usd, f1, col, cls):
        return (f'<tr class="{cls}"><td>{name}</td>'
                f'<td class="n"><span class="bar" style="width:{max(4, TW*mins/99):.0f}px;background:{col}"></span>{mins:g} min</td>'
                f'<td class="n"><span class="bar" style="width:{max(4, CW*usd/25.61):.0f}px;background:{col}"></span>${usd:.2f}</td>'
                f'<td class="n">{f1}</td></tr>')
    body = f"""<div class="slide">
      <div class="kicker">Why not send every frame to an AI? We measured it.</div>
      <h2 style="margin-top:14px">Too slow, too costly, and it still misses.</h2>
      <div class="tiles">
        <div class="tile"><div class="v">66 min</div><div class="l">to label <b>8 minutes</b> of video,<br>8× slower than real time</div></div>
        <div class="tile"><div class="v">$24.65</div><div class="l">for 1,318 frames:<br><b>$0.019 a frame</b>, ~$85 per video hour</div></div>
        <div class="tile"><div class="v" style="color:var(--bad)">0.38</div><div class="l">recall on the sleeping cat (Qbo):<br><b>AI on every frame still misses</b></div></div>
      </div>
      <table><tr><th>AI on every frame · same 8 minutes</th><th>Time (10 workers)</th><th>Cost</th><th>F1</th></tr>
        {''.join(tr(*r) for r in rows)}</table>
      <div class="scale">One camera for a year, at 2 fps: 172,800 frames a day × $0.019 ≈ <b>$1.2M</b></div>
      <div class="pi"><div class="pk">And a bigger model on the device?</div>
        <div class="pt">A Raspberry Pi 5 CPU runs only <b>YOLO nano, at ~3 frames/s</b>. Larger models need more
        compute or a GPU. <span>So this project uses YOLO nano on the Pi 5 CPU as its base model.</span></div></div>
    </div>"""
    return page(body, css, "Labeling shootout 2026-11-03 · Standard pricing (Contributor tier excluded: it trains on your footage)")


@slide
def s05_two_halves():
    css = """
    table{width:100%;border-collapse:separate;border-spacing:0;margin-top:40px;font-size:27px}
    th{font-size:34px;text-align:left;padding:26px 30px;font-weight:700}
    td{padding:18px 30px;border-top:1px solid var(--line);color:var(--t2)}
    .ok{color:#3fc591;font-weight:700} .no{color:var(--bad);font-weight:700}
    td:first-child{color:var(--t3);font-size:26px;text-transform:uppercase;letter-spacing:.08em;width:300px}
    .hl{color:var(--t1)}
    """
    rows = [
        ("Keeps up?", "<span class='ok'>yes</span> · YOLO nano, ~3 frames/s on the Pi 5 CPU", "<span class='no'>no</span> · 8–12× slower than real time"),
        ("Cost, 24/7", "<span class='ok'>~free</span> · fixed hardware, ~6–8 W", "<span class='no'>≈ $1.2M</span> a year per camera"),
        ("Video", "<span class='ok'>stays home</span>", "<span class='no'>every frame</span> leaves home"),
        ("Best at", "<span class='hl'>where</span> the cats are", "<span class='hl'>which</span> cat, and why"),
        ("Fails by", "<span class='no'>mixing up cats</span>, chasing balls", "still misses the sleeping cat"),
        ("Memory", "seconds of tracking", "one context window"),
    ]
    trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    body = f"""<div class="slide">
      <div class="kicker">Why not use just one of them?</div>
      <h2 style="margin-top:14px">Each fails where the other is strong.</h2>
      <table><tr><th></th>
        <th><span class="dot" style="background:var(--edge);margin-right:14px"></span>Fast eyes · YOLO nano on a Pi</th>
        <th><span class="dot" style="background:var(--agent);margin-right:14px"></span>Smart brain · AI on every frame</th></tr>{trs}</table>
      <div class="lead" style="margin-top:30px;font-size:30px">A tiny model on a Raspberry Pi is almost free, <b style="color:var(--t1)">but it makes mistakes.</b></div>
    </div>"""
    return page(body, css, "Pi 5 CPU: YOLO nano ~3 frames/s (camera captures 2 fps) · power: Sixfab Pi 5 benchmark · AI column: labeling shootout 2026-11-03 (s04)")


@slide
def s05b_scope():
    # Scope card for beat 6: who this is for (limited compute, local / private), and the honest
    # trade-off that a bigger trained detector would see better but is out of this project.
    css = """
    .lead{margin-top:22px;font-size:34px}
    .env{display:flex;gap:22px;margin-top:34px}
    .sc{display:grid;grid-template-columns:1fr 1.25fr;gap:36px;margin-top:52px}
    .c{background:var(--panel);border:1px solid var(--line);border-radius:24px;padding:40px 42px}
    .tag2{font-size:22px;letter-spacing:.12em;text-transform:uppercase;font-weight:700}
    .t{font-size:44px;font-weight:700;margin-top:18px;line-height:1.12}
    .d{font-size:27px;color:var(--t2);margin-top:20px;line-height:1.45}
    .out{border:2px dashed #4a4944;background:transparent}
    .out .t{color:var(--t2)}
    """
    env = "".join(f'<span class="pill"><span class="dot" style="background:var(--edge)"></span>{t}</span>'
                  for t in ("runs 24/7, all year", "YOLO nano · Raspberry Pi 5 CPU", "local or private deployment", "~1% of frames leave the device"))
    body = f"""<div class="slide"><div class="kicker">Built for limited compute</div>
      <h2 style="margin-top:22px">A high-performance workflow for hardware<br>that can only run a small model.</h2>
      <div class="env">{env}</div>
      <div class="sc">
        <div class="c out"><div class="tag2" style="color:var(--t3)">Out of scope</div>
          <div class="t">Train a bigger detector</div>
          <div class="d">It would see better, but needs a large labeled dataset, GPU training and hardware this setting doesn’t have.</div></div>
        <div class="c" style="border-top:6px solid var(--agent)"><div class="tag2" style="color:var(--agent)">This project</div>
          <div class="t">Keep the small model. Make its output trustworthy with AI.</div>
          <div class="d">The model stays tiny and imperfect; an agent reviews only its doubts, and memory learns what’s normal.</div></div>
      </div></div>"""
    return page(body, css)


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
    # Flow chart with rounded-corner connectors. The three new ideas sit in tinted shadow boxes
    # behind their steps, each with a few bullets right inside the box (no separate prose).
    INK, MUTED, LINE = "#f2f1ed", "#8e8d84", "#3a3935"
    YEL, VIO, BLU = "#fab219", "#9085e9", "#3987e5"

    def node(x, y, w, h, title, sub, color):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#232321" stroke="{LINE}" stroke-width="2"/>'
                f'<rect x="{x}" y="{y}" width="10" height="{h}" rx="5" fill="{color}"/>'
                f'<text x="{x+32}" y="{y+46}" font-size="29" font-weight="700" fill="{INK}">{title}</text>'
                + "".join(f'<text x="{x+32}" y="{y+84+i*31}" font-size="22" fill="#c3c2b7">{t}</text>' for i, t in enumerate(sub)))

    def zone(x, y, w, h, color, n, title, bullets, bx, by):
        out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{color}" fill-opacity=".09" '
               f'stroke="{color}" stroke-opacity=".5" stroke-width="2"/>'
               f'<circle cx="{x+26}" cy="{y+26}" r="15" fill="{color}"/>'
               f'<text x="{x+26}" y="{y+33}" font-size="19" font-weight="800" fill="#111" text-anchor="middle">{n}</text>'
               f'<text x="{x+50}" y="{y+34}" font-size="22" font-weight="700" fill="{color}">{title}</text>')
        for i, b in enumerate(bullets):
            out += (f'<circle cx="{bx+6}" cy="{by+i*34-7}" r="4.5" fill="{color}"/>'
                    f'<text x="{bx+22}" y="{by+i*34}" font-size="22" fill="#e4e2da">{b}</text>')
        return out

    def wire(d, color=MUTED, label="", lx=0, ly=0, anchor="middle"):
        mk = {MUTED: "ah", VIO: "ahv", BLU: "ahb"}[color]
        lab = f'<text x="{lx}" y="{ly}" font-size="21" fill="#c3c2b7" text-anchor="{anchor}">{label}</text>' if label else ""
        return (f'<path d="{d}" stroke="{color}" stroke-width="3" fill="none" stroke-linecap="round" '
                f'stroke-linejoin="round" marker-end="url(#{mk})"/>' + lab)

    def elbow(pts, r=24):
        d = f"M{pts[0][0]},{pts[0][1]}"
        for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
            ux, uy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
            vx, vy = (x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1)
            d += f" L{x1-ux*r},{y1-uy*r} Q{x1},{y1} {x1+vx*r},{y1+vy*r}"
        return d + f" L{pts[-1][0]},{pts[-1][1]}"

    def marker(mid, col):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')

    W, H = 1696, 720
    svg = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,Helvetica Neue">',
           "<defs>" + marker("ah", MUTED) + marker("ahv", VIO) + marker("ahb", BLU) + "</defs>"]
    # shadow boxes first, so the flow sits on top of them
    svg.append(zone(0, 268, 650, 352, BLU, 3, "Spatial-temporal memory",
                    ["cells → zones → whole view", "5 min → hour → day → week", "only trusted visits enter"], 34, 530))
    svg.append(zone(756, 6, 448, 614, YEL, 1, "Escalation",
                    ["15 typed doubt codes", "~1% of frames escalate", "related doubts merged: 12 → 5 calls"], 790, 530))
    svg.append(zone(1262, 268, 434, 352, VIO, 2, "Agent tools + skills",
                    ["39 tools: camera, images, tracking", "5 skills, loaded on demand", "cheapest evidence first"], 1294, 530))
    # row 1: cheap path
    svg.append(node(18, 60, 300, 140, "Pi camera", ["YOLO, every frame", "2 fps · ~free"], "#199e70"))
    svg.append(node(372, 60, 330, 140, "Correctors", ["boxes → tracks → visits", "name their own doubt"], "#199e70"))
    cx, cy = 980, 130
    svg.append(f'<polygon points="{cx},{cy-74} {cx+112},{cy} {cx},{cy+74} {cx-112},{cy}" fill="#2b2b28" stroke="{YEL}" '
               f'stroke-width="3" stroke-linejoin="round"/>'
               f'<text x="{cx}" y="{cy+10}" font-size="29" font-weight="700" fill="{INK}" text-anchor="middle">Unsure?</text>')
    svg.append(node(1290, 60, 406, 140, "Tracked rows", ["committed as-is", "99% of frames"], MUTED))
    svg.append(wire("M318,130 L366,130"))
    svg.append(wire(f"M702,130 L{cx-118},130"))
    svg.append(wire(f"M{cx+112},{cy} L1284,{cy}", label="no", lx=1247, ly=cy - 14))
    # row 2: review
    svg.append(node(18, 320, 610, 160, "Memory", ["trusted visits → plain-text day log", "brief rides with the next question"], BLU))
    svg.append(node(790, 320, 380, 160, "Plain-English", ["question + key frames", "strict JSON answer"], YEL))
    svg.append(node(1290, 320, 390, 160, "Muse Spark agent", ["pulls the frames it needs", "verdict + rationale"], VIO))
    svg.append(wire(f"M{cx},{cy+74} L{cx},314", label="yes · ~1%", lx=cx + 14, ly=262, anchor="start"))
    svg.append(wire("M1170,400 L1284,400"))
    svg.append(wire("M628,400 L784,400", label="brief", lx=706, ly=386))
    svg.append(wire("M1640,320 L1640,206", color=VIO, label="verdict rewrites rows", lx=1626, ly=244, anchor="end"))
    svg.append(wire(elbow([(1479, 620), (1479, 664), (325, 664), (325, 626)]), color=BLU,
                    label="trusted visits + verdicts", lx=902, ly=700))
    svg.append("</svg>")
    body = f"""<div class="slide">{HEAD("How does it work?", "The Pi handles 99% of frames; the agent sees only the doubts.")}
      <div style="margin-top:26px;zoom:.92">{''.join(svg)}</div></div>"""
    return page(body)


S08_FRAME = "2026-10-28T16:34:42.769881"  # explainer frame: both cats are real detections here
UPHELD = {"confirm", "confirm_reset", "confirm_split", "different_cats"}  # verdicts that leave rows as-is


def s08_rows(rd) -> list[dict]:
    """Validated rows (after review) for the s08 lanes, tidied for the demo story: Qb sleeps in
    the bed the whole window and Qbo walks in. The detector barely saw the sleeping Qb while
    Qbo walked by, so her lane is bridged by linear interpolation (faint ticks, one per camera
    frame) between her last real bed box before the gap and her next detection; the 3 bed-cat
    frames still labeled Qbo after review are drawn as Qb."""
    rows = [dict(r) for r in rd("validated") if not r.get("deleted")]
    g0, g1 = "2026-10-28T16:34:01.879598", "2026-10-28T16:35:20.113084"  # Qb's gap anchors
    for r in rows:
        if r["cat_id"] == 26 and r["box_y2"] >= 575:  # bed cat, not the walking one
            r["cat_id"] = 25
    a = next(r for r in rows if r["create_time"] == g0 and r["cat_id"] == 25)
    b = next(r for r in rows if r["create_time"] == g1 and r["cat_id"] == 25)
    rows = [r for r in rows if not (r["cat_id"] == 25 and g0 < r["create_time"] < g1 and r.get("orig_cat_id") is None)]
    ta, tb = ts(g0), ts(g1)
    for im in rd("images"):
        t = im["create_time"]
        if g0 < t < g1 and not any(r["create_time"] == t and r["cat_id"] == 25 for r in rows):
            f = (ts(t) - ta).total_seconds() / (tb - ta).total_seconds()
            rows.append({"create_time": t, "cat_id": 25, "orig_cat_id": None,
                         **{k: a[k] + f * (b[k] - a[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")}})
    return rows


@slide
def s08_timeline():
    """Problem: how often does the cheap tracker need the expensive agent?
    Conclusion: rarely. Lanes use visualization.py's axis (a detection is a vertical tick from
    box_x1 to box_x2 inside its cat's lane); a real frame on the left shows the mapping; doubts
    are marked on the time axis and grouped into the agent calls that reviewed them."""
    run = RUN
    rd = lambda n: [json.loads(l) for l in (run / f"{n}.jsonl").open()]
    rev = [dict(r, orig_cat_id=r["cat_id"]) for r in rd("stream")]  # raw detector: every row is real
    vals = rd("validations")
    meta = json.loads((run / "run.json").read_text())
    FW, FH = meta.get("img_size") or [800, 600]
    t0, t1 = ts("2026-10-28T16:33:00"), ts("2026-10-28T16:41:00")
    span = (t1 - t0).total_seconds()
    calls: dict = {}
    for v in vals:
        calls.setdefault(v["agent_run"]["run_dir"], []).append(v)

    CW, X0, X1, LANE_H = 1226, 70, 1206, 120
    lanes = {26: ("Qbo", 190, "#d95926"), 25: ("Qb", 330, "#3987e5")}

    def x(t):
        return X0 + (ts(t) - t0).total_seconds() / span * (X1 - X0)

    def ly(top, px):  # frame x (px) -> y in a lane, left edge at the bottom
        return top + LANE_H - px / FW * LANE_H

    svg = [f'<svg width="{CW}" height="500" viewBox="0 0 {CW} 500" font-family="-apple-system,Helvetica Neue">']
    # calls strip: every agent call in order, its doubts as chips
    cx = X0
    for i, vs in enumerate(calls.values(), 1):
        w = 38 + len(vs) * 46
        svg.append(f'<rect x="{cx}" y="8" width="{w}" height="46" rx="14" fill="#26243a" stroke="#9085e9" stroke-width="2"/>'
                   f'<text x="{cx+16}" y="38" font-size="19" font-weight="700" fill="#c9c2ff">{i}</text>')
        for j, v in enumerate(vs):
            col = "#8e8d84" if v["review"]["verdict"] in UPHELD else "#9085e9"
            svg.append(f'<rect x="{cx+34+j*46}" y="16" width="40" height="30" rx="7" fill="{col}"/>'
                       f'<text x="{cx+54+j*46}" y="37" font-size="16" font-weight="700" fill="#111" text-anchor="middle">'
                       f'{v["escalation"]["name"].split("-")[1]}</text>')
        cx += w + 14
    svg.append(f'<text x="{cx+6}" y="38" font-size="21" fill="#c3c2b7">agent calls</text>')
    # minute grid + axis
    for m in range(9):
        xx = X0 + m * 60 / span * (X1 - X0)
        svg.append(f'<line x1="{xx:.1f}" y1="150" x2="{xx:.1f}" y2="456" stroke="#2f2e2b"/>'
                   f'<text x="{xx:.1f}" y="486" font-size="20" fill="#8e8d84" text-anchor="middle">16:{33+m:02d}</text>')
    # doubt markers on the time axis
    for v in vals:
        ex = x(v["escalation"]["time"])
        svg.append(f'<line x1="{ex:.1f}" y1="150" x2="{ex:.1f}" y2="456" stroke="#fab219" stroke-width="1.4" opacity=".7"/>'
                   f'<path d="M{ex-8:.1f},138 L{ex+8:.1f},138 L{ex:.1f},152 z" fill="#fab219"/>')
    last = max(x(v["escalation"]["time"]) for v in vals)
    svg.append(f'<path d="M{last+14:.1f},126 L{last+14:.1f},116 L{X1:.1f},116 L{X1:.1f},126" stroke="#8e8d84" stroke-width="2" fill="none"/>'
               f'<text x="{(last+X1)/2:.1f}" y="104" font-size="21" fill="#c3c2b7" text-anchor="middle">'
               f'{round((t1 - ts(max(v["escalation"]["time"] for v in vals))).total_seconds() / 60)} minutes, 0 doubts: the Pi handles it alone</text>')
    for cid, (name, top, col) in lanes.items():
        svg.append(f'<rect x="{X0}" y="{top}" width="{X1-X0}" height="{LANE_H}" rx="10" fill="#211f1d" stroke="#3a3935"/>'
                   f'<text x="{X0-14}" y="{top+LANE_H/2+9}" font-size="27" font-weight="700" fill="#f2f1ed" text-anchor="end">{name}</text>')
        for r in rev:
            if r["cat_id"] != cid or r.get("box_x1") is None:
                continue
            xx = x(r["create_time"])
            op = "0.95" if r.get("orig_cat_id") is not None else "0.3"
            svg.append(f'<line x1="{xx:.1f}" y1="{ly(top, r["box_x1"]):.1f}" x2="{xx:.1f}" y2="{ly(top, r["box_x2"]):.1f}" '
                       f'stroke="{col}" stroke-width="1.6" opacity="{op}"/>')
    fx = x(S08_FRAME)
    svg.append(f'<line x1="{fx:.1f}" y1="176" x2="{fx:.1f}" y2="456" stroke="#f2f1ed" stroke-width="2" stroke-dasharray="4 5"/>'
               f'<circle cx="{fx:.1f}" cy="172" r="13" fill="#f2f1ed"/><text x="{fx:.1f}" y="178" font-size="16" font-weight="800" fill="#111" text-anchor="middle">A</text>')
    for r in rev:
        if r["create_time"] == S08_FRAME and r.get("box_x1") is not None:
            top = lanes[r["cat_id"]][1]
            svg.append(f'<rect x="{fx-5:.1f}" y="{ly(top, r["box_x2"]):.1f}" width="10" height="{(r["box_x2"]-r["box_x1"])/FW*LANE_H:.1f}" '
                       f'rx="3" fill="none" stroke="#f2f1ed" stroke-width="2.5"/>')
    svg.append("</svg>")

    # explainer: one real frame -> one tick
    fr = [r for r in rev if r["create_time"] == S08_FRAME and r.get("box_x1") is not None]
    stem = S08_FRAME.replace("T", "-").replace(":", "-").replace(".", "-")
    img = next((REPO / "image_server/retrieved_images/2026-10-28-16/images").glob(f"*_{stem}.jpg"))
    TW, TH = 380, 285
    items = [{"box": [r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"]], "color": lanes[r["cat_id"]][2]} for r in fr]
    thumb = boxes_on(img.as_uri(), TW, TH, items, src_w=FW, src_h=FH)
    ruler = [f'<svg width="{TW}" height="64" viewBox="0 0 {TW} 64" font-family="-apple-system,Helvetica Neue">',
             f'<line x1="0" y1="18" x2="{TW}" y2="18" stroke="#8e8d84" stroke-width="2"/>',
             '<text x="0" y="56" font-size="17" fill="#8e8d84">left edge</text>',
             f'<text x="{TW}" y="56" font-size="17" fill="#8e8d84" text-anchor="end">right edge</text>']
    for r in fr:
        ruler.append(f'<rect x="{r["box_x1"]/FW*TW:.1f}" y="{8 if r["cat_id"] == 25 else 22}" width="{(r["box_x2"]-r["box_x1"])/FW*TW:.1f}" '
                     f'height="10" rx="3" fill="{lanes[r["cat_id"]][2]}"/>')
    ruler.append("</svg>")
    css = """
    .wrap{display:grid;grid-template-columns:380px 1226px;gap:90px;margin-top:34px}
    .cap{font-size:20px;color:var(--t3);margin-bottom:12px;letter-spacing:.1em;text-transform:uppercase;font-weight:600;display:flex;align-items:center;gap:10px}
    .cap b{display:inline-flex;width:28px;height:28px;border-radius:50%;background:var(--t1);color:#111;align-items:center;justify-content:center;letter-spacing:0}
    .note{font-size:20px;color:var(--t2);line-height:1.4;margin-top:6px}
    .legend{display:flex;gap:24px;font-size:18px;color:var(--t3);margin-top:10px;padding-left:70px}
    .legend i{display:inline-block;width:16px;height:16px;border-radius:4px;vertical-align:-2px;margin-right:7px}
    """
    legend = ('<div class="legend"><span><i style="background:#fab219"></i>doubt raised by the Pi</span>'
              '<span><i style="background:#9085e9"></i>agent corrected</span><span><i style="background:#8e8d84"></i>agent upheld</span>'
              '</div>')
    n_doubts, n_calls = len(vals), len(calls)
    body = f"""<div class="slide"><div class="kicker">How often does the cheap tracker need help?</div>
      <h2 style="margin-top:14px">Rarely: {n_doubts} doubts in 8 minutes, reviewed in {n_calls} calls.</h2>
      <div class="wrap"><div><div class="cap"><b>A</b>How to read a lane</div>{thumb}{''.join(ruler)}
        <div class="note">Turn the frame 90°: the lane’s height is the frame’s width, so each frame is one tick spanning the cat’s box.</div></div>
        <div>{''.join(svg)}{legend}</div></div></div>"""
    return page(body, css, f"Run {run.name.rsplit('_', 1)[-1][:15]} · lanes = the Pi’s raw detections and labels, before any review · yellow = when it escalated")


@slide
def s09_doubt_taxonomy():
    frame = [("T1", "multi-match relabel"), ("T2", "relabeled by distance alone"), ("T3", "duplicate tie"),
             ("T4", "jump reset"), ("T5", "target fallback"), ("T6", "teleport split"), ("T7", "long guessed run")]
    visit = [("P1", "close vote"), ("P2", "weak winner"), ("P3", "conflict reassign"), ("P4", "long unknown"),
             ("P5", "low quality"), ("P6", "merge candidate"), ("P7", "static link"), ("P8", "label vs vote split")]
    fired: dict = {}
    for e in load("escalations"):  # badge = how often each code fired in this 8-minute run
        fired[e["code"]] = fired.get(e["code"], 0) + 1
    css = """
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:48px;margin-top:44px}
    .colh{font-size:26px;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:18px;font-weight:600}
    .cell{display:flex;align-items:center;gap:22px;padding:11px 24px;border-radius:14px;background:var(--panel);
      border:1px solid var(--line);margin-bottom:10px;font-size:26px;color:var(--t2)}
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
      <div class="kicker">When should the Pi ask for help?</div>
      <h2 style="margin-top:14px">Only when it can name why it’s unsure: 15 typed doubts.</h2>
      <div class="cols"><div><div class="colh">One frame</div>{cells(frame)}</div>
      <div><div class="colh">A whole visit</div>{cells(visit)}</div></div>
      <div class="new">Badges = fired in this 8-minute window (run 140502). Plus H1, a quiet-window spot-check (a heartbeat, not a doubt).</div>
    </div>"""
    return page(body, css)


@slide
def s10_question_card():
    css = """
    pre{font-family:"SF Mono",Menlo,monospace;font-size:29px;line-height:1.55;color:var(--t2);white-space:pre-wrap}
    .k{color:var(--doubt)} .h{color:var(--t1);font-weight:700}
    """
    text = """<span class="h">[E5-T2] Identity switched on nearness alone, no overlap.</span>

<span class="k">Validate:</span>  is the photo really cat 26 (Qbo)?
<span class="k">Conflict:</span>  the camera said cat 25 (Qb); the tracker re-labeled the box
           to 26 because Qbo was last seen 92 px away. No overlap.
<span class="k">Look at:</span>   16:35:21.976  box [174,457,279,585]  + both cat profiles
<span class="k">Memory:</span>    recent: "qbo appears once in period 4 …"

<span class="k">Answer:</span>    {"verdict": confirm | keep_original | reassign_other | drop,
            "rationale": "…"}"""
    body = f"""<div class="slide">
      <div class="kicker">How does a doubt reach the agent?</div>
      <h2 style="margin-top:14px">As a plain-English question with a strict answer form.</h2>
      <div class="card" style="margin-top:44px;padding:48px 56px"><pre>{text}</pre></div>
    </div>"""
    return page(body, css, "Condensed from the real E5-T2 prompt (agent run 20261104-140629, trace.md) · schema: agent/escalation/review_schema.py")


@slide
def s10b_agent_work():
    # Beat 12 visual (stands in for the REC-1 screen recording): the real agent run
    # 20261104-140629 (call 2: E4-T2, E5-T2, E6-T4, E7-P6). Images are the exact frames and
    # crops the agent fetched, copied from the run's trace folder.
    T = f"{A}/frames/trace_140629"
    css = """
    .cols{display:grid;grid-template-columns:1fr 1fr 1.05fr;gap:30px;margin-top:40px}
    .col{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px 26px}
    .th{font-size:19px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);font-weight:700}
    .tt{font-size:30px;font-weight:700;margin-top:8px}
    .tm{font-family:"SF Mono",Menlo,monospace;font-size:18px;color:var(--agent);margin-top:6px}
    .g{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}
    .g img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:10px;border:1px solid var(--line);display:block}
    .g.crop img{aspect-ratio:4/3;object-fit:cover;object-position:50% 40%}
    .v{font-family:"SF Mono",Menlo,monospace;font-size:21px;line-height:1.9;margin-top:18px;color:var(--t2)}
    .v b{color:var(--agent)}
    .why{font-size:21px;color:var(--t2);line-height:1.45;margin-top:14px;border-left:4px solid var(--agent);padding-left:14px}
    """
    f = "".join(f'<img src="{T}/step0{i}_get_frame_1.jpg">' for i in (1, 2, 3, 4))
    c = "".join(f'<img src="{T}/step0{i}_get_frame_1.jpg">' for i in (5, 6, 7, 8))
    body = f"""<div class="slide">{HEAD("What does the agent actually do?", "It looks for itself: frames, then crops, then a verdict.")}
      <div class="cols">
        <div class="col"><div class="th">Turn 1</div><div class="tt">Pulls 4 frames</div>
          <div class="tm">local__get_frame ×4 · 10.4 s</div><div class="g">{f}</div></div>
        <div class="col"><div class="th">Turn 2</div><div class="tt">Zooms into each box</div>
          <div class="tm">local__get_frame(box=…) ×4 · 15.5 s</div><div class="g crop">{c}</div></div>
        <div class="col" style="border-color:var(--agent)"><div class="th">Turn 3</div><div class="tt">Answers, with reasons</div>
          <div class="v">E4-T2 → <b>keep_original</b><br>E5-T2 → <b>keep_original</b><br>E6-T4 → <b>link_continuation</b><br>E7-P6 → <b>merge</b></div>
          <div class="why">“Bed cat: cool white, smooth outline = Qb. Rug cat: fuzzy halo, plume tail = Qbo.”</div></div>
      </div></div>"""
    return page(body, css, "Agent run 20261104-140629 · Muse Spark 1.3 · 3 turns, 8 tool calls · images = the exact frames and crops it fetched")


@slide
def s11_case_e5():
    # Run 140502, E2-P5: the tracker called the cat asleep in the bed Qbo; the agent reassigned it to Qb.
    frame = boxes_on(f"{A}/frames/2026-10-28-16-33-06-797209_E2.jpg", 1120, 840,
                     [{"box": [209, 505, 311, 599], "color": "var(--qbo)", "label": "tracker: Qbo", "style": "dashed"},
                      {"box": [209, 505, 311, 599], "color": "var(--qb)", "label": "agent: Qb", "label_pos": "right"}])
    css = """
    .grid{display:grid;grid-template-columns:1120px 1fr;gap:56px;height:100%;align-items:center}
    .refs{display:flex;gap:18px;margin:26px 0 30px}
    .ref{flex:1;height:250px;border-radius:14px;overflow:hidden;position:relative;border:1px solid var(--line)}
    .ref img{width:100%;height:100%;object-fit:cover}
    .ref span{position:absolute;left:12px;bottom:10px;font-size:22px;font-weight:700;padding:3px 12px;border-radius:8px;color:#111}
    blockquote{font-size:34px;line-height:1.4;color:var(--t1);border-left:6px solid var(--agent);padding-left:26px}
    """
    body = f"""<div class="slide" style="padding:120px 112px 100px"><div class="grid">{frame}
      <div><div class="kicker">Case E2: which cat is asleep in the bed?</div>
        <div class="refs">
          <div class="ref"><img src="{A}/reference/qb_standing_front.jpg" style="object-position:50% 40%"><span style="background:var(--qb)">Qb</span></div>
          <div class="ref"><img src="{A}/reference/qbo_lying_side.jpg" style="object-position:60% 50%"><span style="background:var(--qbo)">Qbo</span></div></div>
        <blockquote>“Cool white, smooth sharp outline, white paws, no fluffy halo. This is Qb, not Qbo.”</blockquote>
        <div style="margin-top:44px"><span class="stamp" style="color:var(--agent);border-color:var(--agent)">OVERTURNED</span></div>
      </div></div></div>"""
    return page(body, css, "Frame 2026-10-28 16:33:06 · E2-P5 verdict reassign → Qb · rationale shortened from agent run 20261104-140503")


@slide
def s12_batching():
    groups = [("Call 1", ["E1 T6", "E2 P5", "E3 T2"]), ("Call 2", ["E4 T2", "E5 T2", "E6 T4", "E7 P6"]),
              ("Call 3", ["E8 T4", "E9 P6", "E10 P2"]), ("Call 4", ["E11 P2"]), ("Call 5", ["E12 P8"])]
    metrics = [("Agent calls", 12, 5, "{:.0f}"), ("Tokens", 517.6, 330.0, "{:.0f}k"),
               ("Cost", 0.60, 0.39, "${:.2f}"), ("Agent minutes", 9.4, 8.7, "{:.1f}")]
    css = """
    .grid{display:grid;grid-template-columns:820px 1fr;gap:80px;margin-top:50px}
    .g{display:flex;align-items:center;gap:20px;margin-bottom:16px}
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
      <div class="kicker">Does every doubt need its own call?</div>
      <h2 style="margin-top:16px">No: related doubts share one. 12 questions, 5 calls.</h2>
      <div class="grid"><div>{left}</div><div>{''.join(right)}</div></div></div>"""
    return page(body, css, "Run 012059 (one call per doubt, earlier tracker) vs 140502 (merged, ≤4 cases a call) · same 8-minute window")


@slide
def s13_case_e10():
    # Run 140502, E4-T2 + E5-T2 at 16:35:21: nearness swapped the two cats; the agent kept the detector's labels.
    f1 = boxes_on(f"{A}/frames/2026-10-28-16-35-21-976102_E4E5.jpg", 1000, 750,
                  [{"box": [145, 357, 234, 462], "color": "var(--qbo)", "label": "Qbo (tracker said Qb)", "label_pos": "right"},
                   {"box": [174, 457, 279, 585], "color": "var(--qb)", "label": "Qb (tracker said Qbo)", "label_pos": "right"}])
    css = """
    .grid{display:grid;grid-template-columns:1000px 1fr;gap:64px;margin-top:34px;align-items:center}
    blockquote{font-size:30px;line-height:1.42;color:var(--t1);border-left:6px solid var(--agent);padding-left:26px}
    .meta{font-size:23px;color:var(--t3);margin-top:26px;line-height:1.6}
    """
    body = f"""<div class="slide">{HEAD("Case E4 + E5: what if the tracker swaps the cats?", "Nearness swapped them. The agent swapped them back.")}
      <div class="grid">{f1}
      <div>
        <blockquote>“Rug cat: fuzzy halo, bushy plume tail = Qbo. Bed cat: cool white, smooth outline = Qb. Nearness caused a cross-steal; the detector was right.”</blockquote>
        <div class="meta">• two relabels by distance alone (92 px, 152 px), zero overlap<br>• one call reviewed all four related doubts (E4–E7)</div>
        <div style="margin-top:30px"><span class="stamp" style="color:var(--agent);border-color:var(--agent)">SWAPPED BACK</span></div>
      </div></div></div>"""
    return page(body, css, "Frame 2026-10-28 16:35:21 · E4-T2 + E5-T2 verdict keep_original · rationales shortened from agent run 20261104-140629")


def zoom(img: str, crop: list[int], scale: float, items: list[dict], extra: str = "") -> str:
    """A crop of an 800x600 frame, scaled up, with boxes in source-pixel coordinates."""
    x0, y0, x1, y1 = crop
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    parts = [f'<div class="frame" style="width:{w:.0f}px;height:{h:.0f}px;background:url({img}) '
             f'-{x0*scale:.0f}px -{y0*scale:.0f}px/{800*scale:.0f}px {600*scale:.0f}px no-repeat;{extra}">']
    for it in items:
        bx0, by0, bx1, by1 = it["box"]
        parts.append(f'<div class="box" style="left:{(bx0-x0)*scale:.0f}px;top:{(by0-y0)*scale:.0f}px;'
                     f'width:{(bx1-bx0)*scale:.0f}px;height:{(by1-by0)*scale:.0f}px;border-color:{it["color"]};'
                     f'border-style:{it.get("style", "solid")}"></div>')
        if it.get("label"):
            parts.append(f'<div class="tag" style="left:{(bx0-x0)*scale:.0f}px;top:{(by1-y0)*scale+10:.0f}px;'
                         f'background:{it["color"]}">{it["label"]}</div>')
    parts.append("</div>")
    return "".join(parts)


@slide
def s13b_detector_fixes():
    # Left: run 140502 E10-P2, the detector called the standing cat Qb on 2 frames; the agent
    # reassigned it to Qbo (agent run 20261104-140809). Checked by eye against cats.yaml: golden-brown
    # face mask, neck ruff, brown paws, plume tail. (The night cases were dropped: their GT calls Qbo "Qb".)
    # Right: the exercise ball (s02). Not an agent verdict: a decoy zone in zones.yaml, added after the
    # 2026-11-01 trace review, drops it at ingest (documents/e2e-identity-fix-20261104.md).
    cat = zoom(f"{A}/frames/2026-10-28-16-34-41-265746_E10P2.jpg", [235, 350, 425, 590], 2.2,
               [{"box": [290, 389, 373, 542], "color": "var(--qbo)"}])
    ball = zoom(f"{A}/frames/2026-11-01-07-35-00-150851_ball_decoy.jpg", [230, 0, 530, 240], 1.75,
                [{"box": [351, 47, 407, 104], "color": "var(--bad)", "style": "dashed", "label": "“Qbo” → not a cat"}])
    css = """
    .row{display:grid;grid-template-columns:820px 820px;gap:56px;margin-top:40px}
    .h{font-size:32px;font-weight:700}
    .was{font-size:23px;color:var(--t3);margin:4px 0 18px}
    .pair{display:flex;gap:20px;align-items:flex-start}
    .refs{display:flex;flex-direction:column;gap:14px}
    .ref{width:200px;height:200px;border-radius:12px;overflow:hidden;position:relative;border:1px solid var(--line)}
    .ref img{width:100%;height:100%;object-fit:cover}
    .ref span{position:absolute;left:8px;bottom:8px;font-size:20px;font-weight:700;padding:2px 10px;border-radius:7px;color:#111}
    ul{list-style:none;font-size:23px;line-height:1.55;color:var(--t2);margin-top:18px}
    ul b{color:var(--t1)}
    .stamp{font-size:30px;padding:6px 20px;margin-top:20px}
    .big{font-size:23px;line-height:1.55;color:var(--t2);margin-top:18px}
    .big b{color:var(--t1)}
    """
    body = f"""<div class="slide" style="padding-top:70px">{HEAD("What if the detector itself is wrong?", "Wrong cat: relabeled. Not a cat: removed.")}
      <div class="row">
        <div><div class="h">Wrong cat</div><div class="was">Detector said <b style="color:var(--qb)">Qb</b> · agent: <b style="color:var(--qbo)">Qbo</b></div>
          <div class="pair">{cat}
            <div class="refs">
              <div class="ref"><img src="{A}/reference/qbo_lying_side.jpg" style="object-position:70% 50%"><span style="background:var(--qbo)">Qbo</span></div>
              <div class="ref"><img src="{A}/reference/qb_standing_front.jpg" style="object-position:50% 55%"><span style="background:var(--qb)">Qb</span></div></div></div>
          <ul><li>✓ golden-brown face mask · ✓ thick neck ruff</li><li>✓ brown paws · ✓ plume tail &nbsp;<span style="color:var(--t3)">(Qb: smooth, white paws)</span></li></ul>
          <span class="stamp" style="color:var(--agent);border-color:var(--agent)">RELABELED</span></div>
        <div><div class="h">Not a cat</div><div class="was">The exercise ball, tracked as “Qbo” for 26 minutes</div>
          {ball}
          <div class="big">Found in review, saved as a <b>decoy zone</b>. Its boxes are now<br>
            dropped before tracking: <b>1,016</b> in the 26 Oct replay, <b>660</b> in 25 Oct's.</div>
          <span class="stamp" style="color:var(--bad);border-color:var(--bad)">REMOVED</span></div>
      </div></div>"""
    return page(body, css, "Left: run 140502, E10-P2 verdict reassign → Qbo, 28 Oct 16:34:41 · right: 1 Nov 07:35, zone exercise_ball in zones.yaml (agent/tracking/decoys.py)")


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
  "create_time": "16:33:06.797",
  "box": [209, 505, 311, 599],
  "conf": 0.511,
  "orig_cat_id": 26,
  <span class="del">"cat_id": 26</span>
}"""
    after = """{
  "create_time": "16:33:06.797",
  "box": [209, 505, 311, 599],
  "conf": 0.511,
  "orig_cat_id": 26,
  <span class="add">"cat_id": 25</span>,
  <span class="add">"validated_by": "E2-P5"</span>,
  <span class="add">"verdict": "reassign"</span>
}"""
    body = f"""<div class="slide"><div class="kicker">What happens to a verdict?</div>
      <h2 style="margin-top:16px">It rewrites the record, and every change points back to its trace.</h2>
      <div class="cols"><div class="card"><div class="lbl">Tracker row (revised.jsonl)</div><pre>{before}</pre></div>
      <div class="card" style="border-color:var(--agent)"><div class="lbl">After review (validated.db)</div><pre>{after}</pre></div></div></div>"""
    return page(body, css, "Real row from run 140502 (revised.jsonl → validated.jsonl) · production MySQL is read-only; validated rows go to a per-run SQLite file")


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
    body = """<div class="slide"><div class="kicker">What does the memory keep?</div>
      <h2 style="margin-top:14px">A diary in plain words, not boxes and timestamps.</h2>
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
    # Every block, weight and quote is a real entry in agent/memory/data/view_3_long_term.yaml.
    # Spatial: three stacked isometric grids (cells 8x6, zones 4x3, view 1x1) over the 800x600
    # frame; block height = memory weight. The highlighted chain r5c2 -> r2c1 -> r0c0 nests.
    # Temporal: four zooming lanes, week W44 -> day Oct 28 -> hour 16:00 -> slot 16:30.
    css = """
    .viz{position:relative;width:1696px;height:640px;margin-top:22px;zoom:.88}
    .viz svg{position:absolute;top:0}
    .h{position:absolute;top:-6px;font-size:24px;color:var(--t3);letter-spacing:.12em;text-transform:uppercase;font-weight:600}
    .h b{color:var(--t1);font-weight:700;margin-right:14px}
    .snap{position:absolute;background:var(--panel);border:1px solid var(--line);border-left:6px solid var(--qb);
      border-radius:14px;padding:13px 18px 12px}
    .sh{display:flex;justify-content:space-between;align-items:baseline;font-size:21px;font-weight:700}
    .sh i{font-style:normal;font-weight:500;color:var(--t3);font-size:18px;margin-left:8px}
    .sh span{font-family:"SF Mono",Menlo,monospace;font-size:18px;color:var(--qb);font-weight:600}
    .sq{font-family:"New York","Georgia",serif;font-size:21px;line-height:1.3;color:var(--t1);margin-top:6px}
    .sm{font-family:"SF Mono",Menlo,monospace;font-size:17px;color:var(--t2);margin-top:6px}
    .gate{margin-top:26px;display:flex;gap:28px;justify-content:center}
    """
    TOP, SIDE_R, SIDE_L = "#2a78d6", "#1c5cab", "#184f95"
    HOT_TOP, HOT_R, HOT_L, INK = "#5aa2f0", "#3987e5", "#256abf", "#f2f1ed"

    def snap(left, top, width, title, sub, ident, quote, meta):
        return (f'<div class="snap" style="left:{left}px;top:{top}px;width:{width}px">'
                f'<div class="sh"><div>{title}<i>{sub}</i></div><span>{ident}</span></div>'
                f'<div class="sq">“{quote}”</div><div class="sm">{meta}</div></div>')

    # ---------------- spatial: isometric stacked grids
    S, OX = 0.38, 207  # scale (frame px -> screen px), x of the plane's far corner
    C, H2 = 0.866 * S, 0.5 * S

    def iso(u, v, z, oy):
        return OX + (u - v) * C, oy + (u + v) * H2 - z

    def poly(pts, fill, stroke="none", sw=0, op=1.0, dash=""):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polygon points="{d}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"{da}/>'

    def prism(b, h, oy, hot, op=1.0):
        pad = 9
        x1, y1, x2, y2 = b[0] + pad, b[1] + pad, b[2] - pad, b[3] - pad
        t, r, l = (HOT_TOP, HOT_R, HOT_L) if hot else (TOP, SIDE_R, SIDE_L)
        st, sw = (INK, 2.5) if hot else ("#0f2a4d", 1)
        return (poly([iso(x1, y2, 0, oy), iso(x2, y2, 0, oy), iso(x2, y2, h, oy), iso(x1, y2, h, oy)], l, st, sw, op)
                + poly([iso(x2, y1, 0, oy), iso(x2, y2, 0, oy), iso(x2, y2, h, oy), iso(x2, y1, h, oy)], r, st, sw, op)
                + poly([iso(x1, y1, h, oy), iso(x2, y1, h, oy), iso(x2, y2, h, oy), iso(x1, y2, h, oy)], t, st, sw, op))

    def height(w):
        return min(30.0, 7 + w * 0.8)

    # (name, grid cols x rows, plane top y, blocks [(id, weight, bounds)], highlighted id)
    levels = [
        ("view", (1, 1), 34, [("r0c0", 48, (0, 0, 800, 600))], "r0c0"),
        ("zones", (4, 3), 194, [("r2c0", 7, (0, 400, 200, 600)), ("r2c1", 30, (200, 400, 400, 600)),
                               ("r0c1", 3, (200, 0, 400, 200)), ("r0c2", 4, (400, 0, 600, 200)),
                               ("r1c1", 2, (200, 200, 400, 400)), ("r2c3", 2, (600, 400, 800, 600))], "r2c1"),
        ("cells", (8, 6), 354, [("r5c2", 20, (200, 500, 300, 600)), ("r4c1", 7, (100, 400, 200, 500)),
                               ("r4c3", 11, (300, 400, 400, 500)), ("r0c3", 3, (300, 0, 400, 100)),
                               ("r0c5", 2, (500, 0, 600, 100)), ("r4c7", 2, (700, 400, 800, 500)),
                               ("r0c4", 2, (400, 0, 500, 100)), ("r2c3", 2, (300, 200, 400, 300))], "r5c2"),
    ]
    sp = ['<svg width="860" height="640" viewBox="0 0 860 640" font-family="-apple-system,Helvetica Neue">']
    anchors = {}
    for name, (cols, rows), oy, blocks, hot in reversed(levels):  # bottom plane first; upper planes occlude
        sp.append(poly([iso(0, 0, 0, oy), iso(800, 0, 0, oy), iso(800, 600, 0, oy), iso(0, 600, 0, oy)],
                       "#232321", "#4a4944", 2, op=0.72))
        for c in range(1, cols):
            u = c * 800 / cols
            sp.append(f'<line x1="{iso(u,0,0,oy)[0]:.1f}" y1="{iso(u,0,0,oy)[1]:.1f}" x2="{iso(u,600,0,oy)[0]:.1f}" '
                      f'y2="{iso(u,600,0,oy)[1]:.1f}" stroke="#3a3935" stroke-width="1.2"/>')
        for r in range(1, rows):
            v = r * 600 / rows
            sp.append(f'<line x1="{iso(0,v,0,oy)[0]:.1f}" y1="{iso(0,v,0,oy)[1]:.1f}" x2="{iso(800,v,0,oy)[0]:.1f}" '
                      f'y2="{iso(800,v,0,oy)[1]:.1f}" stroke="#3a3935" stroke-width="1.2"/>')
        for bid, w, b in sorted(blocks, key=lambda e: (e[2][0] + e[2][1], e[2][0])):  # back to front
            h = 12 if name == "view" else height(w)
            sp.append(prism(b, h, oy, bid == hot, 0.45 if name == "view" else 1.0))
            if bid == hot:
                cu, cv = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if name == "view":
                    cu, cv = 560, 560  # leader from the slab's front-right, clear of the plane above
                anchors[name] = iso(cu, cv, h, oy)
    # roll-up chain: the cell's centre straight up through its zone to the whole view
    (cx, cy0), (_, cy1) = iso(250, 550, 20, 354), iso(250, 550, 30, 194)
    sp.append(f'<line x1="{cx:.1f}" y1="{cy0:.1f}" x2="{cx:.1f}" y2="{iso(250,550,12,34)[1]:.1f}" stroke="{INK}" '
              f'stroke-width="2" stroke-dasharray="5 6" opacity=".75"/>')
    for y in (cy1, iso(250, 550, 12, 34)[1]):
        sp.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="5" fill="{INK}"/>')
    sp.append(f'<text x="{cx-12:.1f}" y="{(cy0+cy1)/2+6:.1f}" font-size="19" fill="#c3c2b7" text-anchor="end">rolls up ↑</text>')
    CARD_X = 520
    card_top = {"view": 52, "zones": 214, "cells": 400}
    for name, (ax, ay) in anchors.items():
        ty = card_top[name] + 34
        sp.append(f'<path d="M{ax:.1f},{ay:.1f} L{CARD_X-28},{ty} L{CARD_X},{ty}" stroke="{INK}" stroke-width="2" '
                  f'fill="none" opacity=".8"/><circle cx="{ax:.1f}" cy="{ay:.1f}" r="5" fill="{INK}"/>')
    sp.append("</svg>")
    sw = 340
    spatial_cards = (
        snap(CARD_X, card_top["view"], sw, "Whole view", "1 × 1", "r0c0",
             "qb and qbo mostly walk here and briefly stand still", "weight 48 · qb 16 · qbo 21")
        + snap(CARD_X, card_top["zones"], sw, "Zones", "4 × 3 · 200 px", "r2c1",
               "qb visits here around 16:34 to 16:40", "weight 30 · qb 12 · qbo 10")
        + snap(CARD_X, card_top["cells"], sw, "Cells", "8 × 6 · 100 px", "r5c2",
               "curled up tight in a small green bed", "weight 20 · 772 sightings")
    )

    # ---------------- temporal: zooming timeline
    TX, BW, BH = 916, 400, 40  # panel left, bar width, bar height
    lanes = [  # (title, retention, id, ticks, [(label, weight)], highlighted index, quote, meta)
        ("Weeks", "kept forever", "2026-W44", ["W41", "W42", "W43", "W44"],
         [0, 0, 12, 50], 3, "qb visits more often. Both sleep here.", "80 visits · qb 21 · qbo 38"),
        ("Days", "kept 180 days", "Wed Oct 28", ["M", "T", "W", "T", "F", "S", "S"],
         [0, 2, 50, 0, 0, 0, 3], 2, "qb stays for several minutes. qbo appears briefly.", "75 visits · qb 20 · qbo 36"),
        ("Hours", "kept 30 days", "16:00", [f"{h:02d}" if h % 6 == 0 else "" for h in range(24)],
         [50 if h == 16 else 0 for h in range(24)], 16, "qb is present for several minutes", "65 visits · qbo 36 · qb 20"),
        ("5-min slots", "kept 7 days", "16:30", [f":{m:02d}" if m % 15 == 0 else "" for m in range(0, 60, 5)],
         [47 if m == 30 else 19 if m == 35 else 0 for m in range(0, 60, 5)], 6,
         "standing 16:34, then sleeping 16:35–16:40", "qb · conf 0.71 · weight 47"),
    ]
    tl = [f'<svg width="780" height="640" viewBox="0 0 780 640" font-family="-apple-system,Helvetica Neue" style="left:{TX}px">'
          '<defs><linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="#1a1a19" stop-opacity="1"/>'
          '<stop offset="1" stop-color="#1a1a19" stop-opacity="0"/></linearGradient></defs>']
    temporal_cards = []
    ys = [62, 212, 362, 512]
    prev = None
    for (title, keep, ident, ticks, weights, hot, quote, meta), y in zip(lanes, ys):
        n = len(weights)
        seg = BW / n
        if prev is not None:  # zoom wedge from the parent segment down to this whole lane
            px1, px2, py = prev
            tl.append(f'<polygon points="{px1},{py} {px2},{py} {BW},{y} 0,{y}" fill="#3987e5" fill-opacity=".10"/>'
                      f'<line x1="{px1}" y1="{py}" x2="0" y2="{y}" stroke="#3987e5" stroke-width="1.5" stroke-dasharray="4 5"/>'
                      f'<line x1="{px2}" y1="{py}" x2="{BW}" y2="{y}" stroke="#3987e5" stroke-width="1.5" stroke-dasharray="4 5"/>')
        for i, w in enumerate(weights):
            x = i * seg
            gap = 3 if n <= 12 else 1.5
            if w:
                op = 0.35 + 0.65 * min(w / 50, 1)
                fill = HOT_TOP if i == hot else TOP
                stroke = f' stroke="{INK}" stroke-width="2.5"' if i == hot else ""
                tl.append(f'<rect x="{x+gap:.1f}" y="{y}" width="{seg-2*gap:.1f}" height="{BH}" rx="5" fill="{fill}" fill-opacity="{op:.2f}"{stroke}/>')
            else:
                tl.append(f'<rect x="{x+gap:.1f}" y="{y}" width="{seg-2*gap:.1f}" height="{BH}" rx="5" fill="#2b2b28" stroke="#3a3935"/>')
            if ticks[i]:
                tl.append(f'<text x="{x + (seg/2 if n <= 12 else 0):.1f}" y="{y+BH+24}" font-size="18" fill="#8e8d84" '
                          f'text-anchor="{"middle" if n <= 12 else "start"}">{ticks[i]}</text>')
        if title == "Weeks":  # kept forever: the lane runs on into the past
            tl.append(f'<rect x="0" y="{y-4}" width="60" height="{BH+8}" fill="url(#fade)"/>'
                      f'<text x="-4" y="{y+28}" font-size="24" fill="#8e8d84" text-anchor="end">…</text>')
        hx = hot * seg
        prev = (round(hx + 3, 1), round(hx + seg - 3, 1), y + BH + 34)
        cy = y + BH / 2
        # leader from the highlighted segment's right edge to its snapshot card
        lx = min(hx + seg + 4, BW)
        tl.append(f'<path d="M{lx:.1f},{cy} L{BW+30},{cy}" stroke="{INK}" stroke-width="2" opacity=".8"/>'
                  f'<circle cx="{lx:.1f}" cy="{cy}" r="5" fill="{INK}"/>')
        temporal_cards.append(snap(TX + BW + 30, int(cy - 62), 350, title, keep, ident, quote, meta))
    tl.append("</svg>")

    body = f"""<div class="slide"><div class="kicker">What does it learn over weeks?</div>
      <h2 style="margin-top:14px">What’s normal for each cat, in each spot, at each hour.</h2>
      <div class="viz">
        <div class="h" style="left:0"><b>Where</b>3 nested grids · height = weight</div>
        <div class="h" style="left:{TX}px"><b>When</b>4 time scales · zooming in</div>
        <div style="position:absolute;inset:34px 0 0 0">{''.join(sp)}{spatial_cards}{''.join(tl)}{''.join(temporal_cards)}</div>
      </div>
      <div class="gate"><span class="pill"><span class="dot" style="background:var(--good)"></span>only trusted visits (≥ 0.7) enter</span>
        <span class="pill"><span class="dot" style="background:var(--doubt)"></span>a pattern needs 3 sightings</span>
        <span class="pill"><span class="dot" style="background:var(--bad)"></span>flips only after 3 contradictions</span></div></div>"""
    return page(body, css, "Real entries from agent/memory/data/view_3_long_term.yaml (view 3) · quotes are the stored text")


@slide
def s17a_timeline_eval():
    """Evaluation, visual: the same 8 minutes as raw detections (s08) and after agent review.
    After-review lanes use s08_rows() (validated labels; Qb's undetected sleep bridged, faint)."""
    rd = lambda n: [json.loads(l) for l in (RUN / f"{n}.jsonl").open()]
    raw = [dict(r, orig_cat_id=r["cat_id"]) for r in rd("stream")]
    val = s08_rows(rd)
    vals = rd("validations")
    FW = (json.loads((RUN / "run.json").read_text()).get("img_size") or [800, 600])[0]
    t0, t1 = ts("2026-10-28T16:33:00"), ts("2026-10-28T16:41:00")
    span = (t1 - t0).total_seconds()
    W, X0, X1, LH = 1290, 150, 1255, 78
    LANES = ((26, "Qbo", "#d95926"), (25, "Qb", "#3987e5"))

    def x(t):
        return X0 + (ts(t) - t0).total_seconds() / span * (X1 - X0)

    def block(rows, top, title, sub):
        out = [f'<text x="0" y="{top-14}" font-size="24" font-weight="700" fill="#f2f1ed">{title}'
               f'<tspan font-size="19" font-weight="400" fill="#8e8d84">  {sub}</tspan></text>']
        for i, (cid, name, col) in enumerate(LANES):
            y = top + i * (LH + 10)
            out.append(f'<rect x="{X0}" y="{y}" width="{X1-X0}" height="{LH}" rx="9" fill="#211f1d" stroke="#3a3935"/>'
                       f'<text x="{X0-14}" y="{y+LH/2+8}" font-size="23" font-weight="700" fill="#f2f1ed" text-anchor="end">{name}</text>')
            for r in rows:
                if r["cat_id"] != cid or r.get("box_x1") is None:
                    continue
                xx = x(r["create_time"])
                op = "0.95" if r.get("orig_cat_id") is not None else "0.3"
                out.append(f'<line x1="{xx:.1f}" y1="{y+LH-r["box_x1"]/FW*LH:.1f}" x2="{xx:.1f}" y2="{y+LH-r["box_x2"]/FW*LH:.1f}" '
                           f'stroke="{col}" stroke-width="1.5" opacity="{op}"/>')
        return "".join(out)

    svg = [f'<svg width="{W}" height="520" viewBox="0 0 {W} 520" font-family="-apple-system,Helvetica Neue">']
    svg.append(block(raw, 34, "Raw detector", "the Pi alone: labels flip, the sleeping cat is called Qbo"))
    svg.append(block(val, 290, "After agent review", "Qb asleep the whole time; Qbo walks in and out"))
    for m in range(9):
        xx = X0 + m * 60 / span * (X1 - X0)
        for top in (34, 290):
            svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+2*LH+10}" stroke="#4a4944" opacity=".5"/>')
        svg.append(f'<text x="{xx:.1f}" y="{290+2*LH+44}" font-size="19" fill="#8e8d84" text-anchor="middle">16:{33+m:02d}</text>')
    for v in vals:  # where the Pi escalated, drawn inside both charts' lanes only
        ex = x(v["escalation"]["time"])
        for top in (34, 290):
            svg.append(f'<line x1="{ex:.1f}" y1="{top}" x2="{ex:.1f}" y2="{top+2*LH+10}" stroke="#fab219" stroke-width="1.2" opacity=".5"/>')
    svg.append("</svg>")
    css = """
    .row{display:grid;grid-template-columns:1290px 1fr;gap:40px;margin-top:34px;align-items:center}
    .pt{font-size:23px;color:var(--t2);line-height:1.7}
    .pt b{color:var(--t1)}
    .pt .h{font-size:19px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);font-weight:600}
    """
    side = """<div class="pt"><div class="h">Same 8 minutes</div>
      • false alarms <b>107 → 8</b><br>• right cat <b>0.934 → 0.993</b><br>• 12 doubts, <b>5 calls</b>, $0.39<br>
      • yellow lines = where<br>&nbsp;&nbsp;the Pi escalated</div>"""
    body = f"""<div class="slide">{HEAD("Same 8 minutes: what did the review change?", "Noise in, the true story out.")}
      <div class="row"><div>{''.join(svg)}</div>{side}</div></div>"""
    return page(body, css, "Run 140502 · raw = stream.jsonl · after review = validated.jsonl; Qb’s undetected sleep 16:34–16:35 bridged by interpolation (faint)")


@slide
def s17_results():
    # Highlights from documents/final-model-performance-20261104.md (run 140502, audited GT,
    # IoU 0.5). Raw = the Pi's detector stream; after = agent-validated output.
    css = """
    .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:38px}
    .tile{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:28px 30px}
    .v{font-size:76px;font-weight:800;letter-spacing:-.03em;line-height:1}
    .l{font-size:23px;color:var(--t2);margin-top:12px;line-height:1.35}
    .l b{color:var(--t1)}
    .low{display:grid;grid-template-columns:1.15fr 1fr;gap:24px;margin-top:24px}
    .pan{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:22px 30px}
    .ph{font-size:19px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);font-weight:600;margin-bottom:10px}
    .db{display:grid;grid-template-columns:270px 1fr;align-items:center;font-size:21px;color:var(--t2);height:44px}
    .db svg{display:block}
    .pt{font-size:22px;color:var(--t2);line-height:1.6}
    .pt b{color:var(--t1)}
    """
    RAW, VAL = "#199e70", "#9085e9"

    def dumbbell(label, a, b, lo=0.4):
        W = 520
        xa, xb = 12 + (a - lo) / (1 - lo) * (W - 90), 12 + (b - lo) / (1 - lo) * (W - 90)
        return (f'<div class="db"><span>{label}</span><svg width="{W}" height="40" font-family="SF Mono,Menlo,monospace">'
                f'<line x1="{xa:.0f}" y1="22" x2="{xb:.0f}" y2="22" stroke="#8e8d84" stroke-width="2"/>'
                f'<circle cx="{xa:.0f}" cy="22" r="8" fill="{RAW}" stroke="#232321" stroke-width="2"/>'
                f'<circle cx="{xb:.0f}" cy="22" r="8" fill="{VAL}" stroke="#232321" stroke-width="2"/>'
                f'<text x="{xa-12:.0f}" y="28" font-size="17" fill="#c3c2b7" text-anchor="end">{a:.3f}</text>'
                f'<text x="{xb+14:.0f}" y="28" font-size="18" font-weight="700" fill="#f2f1ed">{b:.3f}</text></svg></div>')

    body = f"""<div class="slide">{HEAD("Did it work? 8 real minutes, both cats", "93% fewer false alarms, for 39 cents.")}
      <div class="tiles">
        <div class="tile"><div class="v" style="color:var(--agent)">−93%</div><div class="l">false alarms<br><b>107 → 8</b></div></div>
        <div class="tile"><div class="v">99%</div><div class="l">of what the camera saw<br>gets the <b>right cat</b> (was 93%)</div></div>
        <div class="tile"><div class="v">$0.39</div><div class="l">to review 12 doubts<br><b>63× cheaper</b> than AI on every frame</div></div>
        <div class="tile"><div class="v">~1%</div><div class="l">of frames raise a doubt:<br><b>the only time AI is called</b></div></div>
      </div>
      <div class="low">
        <div class="pan"><div class="ph">Raw detector → after agent review</div>
          {dumbbell("precision", 0.895, 0.992)}{dumbbell("identity accuracy", 0.934, 0.993)}
          {dumbbell("precision, hardest 2 min", 0.529, 0.971)}</div>
        <div class="pan"><div class="ph">Holds beyond this window</div>
          <div class="pt">• hardest 2 min (case E2): false alarms <b>65 → 4</b><br>
          • 3 runs, 65 min: doubts stay at <b>~1%</b> of frames<br>
          • all 3 runs reviewed for <b>$3.77</b> (vs ~$205)</div></div>
      </div></div>"""
    return page(body, css, "documents/final-model-performance-20261104.md · run 140502 vs audited GT, IoU 0.5 · hardest 2 min = largest-gain window (≥300 frames)")


@slide
def s18_roadmap():
    css = """
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:36px;margin-top:64px}
    .c{background:var(--panel);border:1px solid var(--line);border-radius:24px;padding:44px 42px;min-height:420px}
    .num{font-family:"SF Mono",Menlo,monospace;font-size:28px;color:var(--t3)}
    .t{font-size:46px;font-weight:700;margin-top:22px;line-height:1.12}
    .d{font-size:27px;color:var(--t2);margin-top:22px;line-height:1.45}
    """
    cards = [("01", "Give your agent eyes", "Run Muse Glimmer locally: your personal agent asks the camera, and hears when a routine changes.", "var(--agent)"),
             ("02", "Teach the Pi", "Verdicts → labeled data → retrain the small model, so fewer doubts reach the agent.", "var(--edge)"),
             ("03", "Every camera, all year", "Full days and more views, then labs, lines and roads, on the same loop.", "var(--doubt)")]
    html = "".join(f'<div class="c" style="border-top:6px solid {col}"><div class="num">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d, col in cards)
    body = f"""<div class="slide"><div class="kicker">What’s next?</div>
      <h2 style="margin-top:14px">Plug the eyes into your own agent.</h2><div class="cards">{html}</div></div>"""
    return page(body, css)


@slide
def s19_beyond_cats():
    # Four domains, each mapped to a capability the demo already showed. Photos are real
    # Wikimedia Commons files (credits on each photo + assets/broll/CREDITS.md); the boxes are
    # illustrative overlays, not model output.
    css = """
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:44px}
    .c{border-radius:22px;overflow:hidden;background:var(--panel);border:1px solid var(--line);display:flex;flex-direction:column}
    .ph{position:relative;height:310px;overflow:hidden}
    .ph img{display:block;width:100%;height:100%;object-fit:cover}
    .bx{position:absolute;border:3px solid;border-radius:5px}
    .bt{position:absolute;padding:2px 8px;border-radius:6px;font-size:16px;font-weight:700;color:#111;white-space:nowrap;
      font-family:"SF Mono",Menlo,monospace;transform:translateY(-100%)}
    .cr{position:absolute;right:8px;bottom:6px;font-size:12px;color:#fff;opacity:.85;text-shadow:0 1px 3px #000}
    .t{font-size:31px;font-weight:700;padding:24px 26px 0;line-height:1.15}
    .d{font-size:22px;color:var(--t2);padding:12px 26px 0;line-height:1.4;flex:1}
    .same{margin:18px 26px 24px;align-self:flex-start;font-size:19px;padding:6px 14px;border-radius:999px;
      background:rgba(144,133,233,.14);color:#c9c2ff;border:1px solid rgba(144,133,233,.45);font-weight:600}
    .also{margin-top:28px;display:flex;gap:16px;align-items:center;font-size:22px;color:var(--t3)}
    """
    OK, DOUBT = "#199e70", "#fab219"

    def box(x1, y1, x2, y2, col, tag="", below=False):
        b = f'<div class="bx" style="left:{x1}%;top:{y1}%;width:{x2-x1}%;height:{y2-y1}%;border-color:{col}"></div>'
        if tag:
            pos = f"top:calc({y2}% + 4px);transform:none" if below else f"top:{y1}%"
            b += f'<div class="bt" style="left:{x1}%;{pos};background:{col}">{tag}</div>'
        return b

    cards = [
        ("19a_lab_mice.jpg", "Understanding Animal Research · CC BY-SA 4.0",
         [(9, 7, 47, 42, OK, "m1"), (9, 42, 37, 58, DOUBT, "m2 or m1?"), (39, 41, 57, 97, OK, "m3")],
         "Look-alike lab mice", "Identical mice huddle and their IDs swap. The agent settles who is who.",
         "in the demo: Qb vs Qbo"),
        ("19b_drone.jpg", "Nicolas Halftermeyer · CC BY-SA 4.0",
         [(45, 41, 63, 61, OK, "drone 0.62"), (73, 83, 81, 91, DOUBT, "bird? 0.31")],
         "Tiny things in a big sky", "A drone, a bird or a ball is a few pixels. Decoys get a second look, not an alarm.",
         "in the demo: the ball decoy"),
        ("19c_fog_traffic.jpg", "Lord Laitinen · CC BY-SA 4.0 · plate blurred",
         [(1, 44, 59, 99, OK, "car 12"), (68, 64, 84, 83, OK, "car 7"), (59, 66, 68, 79, DOUBT, "car 9: lost?", True)],
         "Vehicles in fog", "Cars blur, hide and pass close. The agent fixes tracks that break or swap.",
         "in the demo: the swapped cats"),
        ("19d_can_line.jpg", "Helfmann · CC BY-SA 3.0",
         [(33, 70, 42, 78, OK, ""), (43, 82, 53, 91, OK, ""), (57, 78, 68, 87, DOUBT, "dent? 0.44")],
         "A production line", "A small model checks every can on site; only the odd few go to the AI.",
         "in the demo: ~1% reviewed"),
    ]
    html = "".join(
        f'<div class="c"><div class="ph"><img src="{A}/broll/{img}">{"".join(box(*b) for b in boxes)}'
        f'<div class="cr">{credit}</div></div><div class="t">{t}</div><div class="d">{d}</div>'
        f'<div class="same">{same}</div></div>'
        for img, credit, boxes, t, d, same in cards)
    also = "".join(f'<span class="pill">{x}</span>' for x in ("pet cams: cats &amp; dogs", "robot eyes", "sports balls", "wildlife"))
    body = f"""<div class="slide"><div class="kicker">Cats are just the start</div>
      <h2 style="margin-top:16px">Any small model watching for rare moments has the same gap.</h2>
      <div class="cards">{html}</div>
      <div class="also">Also: {also}</div></div>"""
    return page(body, css, "Photos: Wikimedia Commons, cropped (full credits in assets/broll/CREDITS.md) · boxes are illustrative, not model output")


@slide
def s20_end_card():
    # Short close: the bridge card already opened the video, so this is name + promise + credits.
    css = """
    .k2{font-size:30px;letter-spacing:.14em;text-transform:uppercase;color:var(--t3);font-weight:700}
    .big{font-size:92px;font-weight:700;line-height:1.08;letter-spacing:-.02em;margin-top:22px}
    .big .b{color:var(--agent)} .big .e{color:var(--edge)}
    .sub{font-size:32px;color:var(--t2);margin-top:30px}
    .by{position:absolute;left:112px;bottom:70px;font-size:26px;color:var(--t3)}
    """
    body = """<div class="slide"><div class="k2">Agentic Realtime Computer Vision</div>
      <div class="big">AI gives the camera <span class="b">a brain</span>.<br>The camera gives AI <span class="e">eyes</span>.</div>
      <div class="sub">Cheap eyes on every frame · a smart brain only when it matters · a memory that keeps growing</div>
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
