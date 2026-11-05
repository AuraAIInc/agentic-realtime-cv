"""YouTube thumbnails (1280x720) for the demo video, rendered from HTML by headless Chrome.

    python documents/demo/scripts/build_thumbnail.py
    -> documents/demo/assets/thumbnail.png      (A: the two cats + "AI gives the camera a brain")
    -> documents/demo/assets/thumbnail_alt.png  (B: the real overturned case + results)
    -> documents/demo/assets/thumbnail_c.png    (C: "107 -> 8" hero numbers + close-up)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
HTML = HERE / "html"
OUT = DEMO / "assets"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
A = "../../assets"  # assets path as seen from scripts/html/

BASE = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1280px;height:720px;overflow:hidden;background:#141413;color:#f2f1ed;
  font-family:-apple-system,"SF Pro Display","Helvetica Neue",sans-serif;-webkit-font-smoothing:antialiased}
.b{color:#9085e9} .e{color:#2fd093} .y{color:#fab219} .r{color:#ff7a7a}
.pill{display:inline-flex;align-items:baseline;gap:10px;padding:10px 18px;border-radius:14px;background:#24232a;
  border:2px solid #3a3946;font-size:26px;font-weight:700;white-space:nowrap}
.pill b{font-size:34px;font-weight:900;letter-spacing:-.02em}
.brand{position:absolute;left:44px;bottom:30px;font-size:20px;font-weight:700;color:#c3c2b7;letter-spacing:.08em;text-transform:uppercase}
.brand span{color:#9085e9}
"""


def thumb_a() -> str:
    css = BASE + """
    .photos{position:absolute;right:0;top:0;width:610px;height:720px;display:grid;grid-template-columns:1fr 1fr}
    .ph{position:relative;overflow:hidden}
    .ph img{width:100%;height:100%;object-fit:cover}
    .ph .tag{position:absolute;left:18px;bottom:66px;padding:6px 16px;border-radius:12px;font-size:34px;font-weight:900;color:#111}
    .fade{position:absolute;right:560px;top:0;width:140px;height:720px;background:linear-gradient(90deg,#141413,rgba(20,20,19,0))}
    .vs{position:absolute;right:270px;top:300px;width:96px;height:96px;border-radius:50%;background:#fab219;color:#111;
      font-size:58px;font-weight:900;display:flex;align-items:center;justify-content:center;box-shadow:0 8px 30px rgba(0,0,0,.6)}
    .t{position:absolute;left:44px;top:46px;width:640px}
    .k{font-size:24px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#c3c2b7}
    h1{font-size:86px;line-height:.98;font-weight:900;letter-spacing:-.035em;margin-top:16px}
    .sub{font-size:30px;font-weight:700;color:#c3c2b7;margin-top:18px;line-height:1.25}
    .pills{position:absolute;left:44px;top:470px;display:flex;flex-direction:column;gap:14px}
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
    <div class="photos">
      <div class="ph"><img src="{A}/reference/qb_standing_front.jpg" style="object-position:50% 35%"><span class="tag" style="background:#3987e5">Qb</span></div>
      <div class="ph"><img src="{A}/reference/qbo_on_back_face.jpg" style="object-position:50% 40%"><span class="tag" style="background:#d95926">Qbo</span></div>
    </div>
    <div class="fade"></div>
    <div class="vs">?</div>
    <div class="t"><div class="k">My camera couldn’t tell them apart</div>
      <h1>AI gives the camera a <span class="b">brain</span></h1>
      <div class="sub">A $100 Raspberry Pi + Meta Muse Spark agent</div></div>
    <div class="pills">
      <span class="pill"><b class="e">−93%</b> false alarms</span>
      <span class="pill"><b class="y">63×</b> cheaper than AI on every frame</span>
    </div>
    <div class="brand">Agentic Realtime Computer Vision · built on <span>Meta Muse Spark</span></div>
    </body></html>"""


def thumb_b() -> str:
    css = BASE + """
    .frame{position:absolute;left:0;top:0;width:700px;height:720px;overflow:hidden}
    .frame img{width:700px;height:720px}
    .box{position:absolute;border:7px solid}
    .lbl{position:absolute;padding:6px 14px;border-radius:10px;font-size:30px;font-weight:900;color:#111;white-space:nowrap}
    .strike{text-decoration:line-through;text-decoration-thickness:4px}
    .fade{position:absolute;left:560px;top:0;width:160px;height:720px;background:linear-gradient(270deg,#141413,rgba(20,20,19,0))}
    .t{position:absolute;left:690px;top:58px;width:560px}
    .k{font-size:24px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#c3c2b7}
    h1{font-size:72px;line-height:1.0;font-weight:900;letter-spacing:-.03em;margin-top:14px}
    .stamp{display:inline-block;margin-top:24px;padding:8px 22px;border:6px solid #9085e9;border-radius:14px;color:#9085e9;
      font-size:44px;font-weight:900;letter-spacing:.06em;transform:rotate(-4deg)}
    .pills{position:absolute;left:690px;top:440px;display:flex;flex-direction:column;gap:14px}
    .brand{left:auto;right:40px}
    """
    # E2 frame (16:33:06, run 140502): raw detector drew Qbo [209,505,311,599] @ 0.51
    # (GT: Qb [190,506,299,599]); E2-P5 verdict `reassign` corrected it to Qb.
    # thumb_b_crop.jpg = source crop (114,300,406,600) resized to exactly 700x720,
    # so the raw box maps to left:(209-114)*700/292=228 top:(505-300)*720/300=492
    # w:102*700/292=244 h:94*720/300=226. Labels sit side by side above the box.
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
    <div class="frame"><img src="{A}/thumb_b_crop.jpg">
      <div class="box" style="left:228px;top:492px;width:244px;height:226px;border-color:#3987e5"></div>
      <div class="lbl" style="left:228px;top:424px;background:#d95926"><span class="strike">Qbo</span></div>
      <div class="lbl" style="left:332px;top:424px;background:#3987e5">Qb ✓</div>
    </div>
    <div class="fade"></div>
    <div class="t"><div class="k">The cheap camera got it wrong</div>
      <h1>The AI agent <span class="b">caught it</span></h1>
      <span class="stamp">OVERTURNED</span></div>
    <div class="pills">
      <span class="pill"><b class="e">99%</b> right cat</span>
      <span class="pill"><b class="y">1%</b> of frames call the AI</span>
      <span class="pill"><b class="b">$0.39</b> per 8 minutes</span>
    </div>
    <div class="brand">Agentic Realtime CV · <span>Meta Muse Spark</span></div>
    </body></html>"""


def thumb_c() -> str:
    css = BASE + """
    .strike{text-decoration:line-through;text-decoration-thickness:7px}
    .photo{position:absolute;right:0;top:0;width:600px;height:720px;overflow:hidden}
    .photo img{width:100%;height:100%;object-fit:cover;object-position:62% 55%}
    .fade{position:absolute;right:550px;top:0;width:140px;height:720px;background:linear-gradient(90deg,#141413,rgba(20,20,19,0))}
    .t{position:absolute;left:44px;top:50px;width:650px}
    .k{font-size:24px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#c3c2b7}
    .hero{font-size:148px;line-height:1;font-weight:900;letter-spacing:-.04em;margin-top:12px;white-space:nowrap}
    .hero .arr{color:#c3c2b7}
    .cap{font-size:27px;font-weight:800;color:#c3c2b7;margin-top:10px;letter-spacing:.04em}
    .sub{font-size:29px;font-weight:700;color:#c3c2b7;margin-top:16px;line-height:1.3}
    .pills{position:absolute;left:44px;top:478px;display:flex;flex-direction:column;gap:14px}
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
    <div class="photo"><img src="{A}/reference/qb_lying_face.jpg"></div>
    <div class="fade"></div>
    <div class="t"><div class="k">My security camera cried wolf</div>
      <div class="hero"><span class="r strike">107</span> <span class="arr">&rarr;</span> <span class="e">8</span></div>
      <div class="cap">FALSE ALARMS &middot; 8 MINUTES OF FOOTAGE</div>
      <div class="sub">An AI agent re-checks only the 1% of frames the camera doubts</div></div>
    <div class="pills">
      <span class="pill"><b class="e">99%</b> gets the right cat</span>
      <span class="pill"><b class="b">$0.39</b> per 8 minutes</span>
    </div>
    <div class="brand">Agentic Realtime Computer Vision &middot; built on <span>Meta Muse Spark</span></div>
    </body></html>"""


def render(name: str, html: str) -> Path:
    HTML.mkdir(parents=True, exist_ok=True)
    h = HTML / f"{name}.html"
    h.write_text(html)
    png = OUT / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    "--allow-file-access-from-files", f"--screenshot={png}", "--window-size=1280,720",
                    "--virtual-time-budget=3000", h.as_uri()], check=True, capture_output=True)
    return png


if __name__ == "__main__":
    for name, fn in (("thumbnail", thumb_a), ("thumbnail_alt", thumb_b), ("thumbnail_c", thumb_c)):
        print(render(name, fn()).relative_to(DEMO))
