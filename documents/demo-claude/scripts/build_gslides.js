// Build a Google-Slides-ready .pptx from the demo slides: one full-bleed slide image per
// slide, in video order, grouped into the demo's acts, with each beat's narration and the
// slide's source line as speaker notes. Import: Google Drive -> New -> File upload -> open
// with Google Slides.   node documents/demo-claude/scripts/build_gslides.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const DEMO = path.resolve(__dirname, "..");
const meta = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(__dirname, "slide_meta.json"))).map(m => [m.name, m]));
const narration = fs.readFileSync(path.join(DEMO, "assets/text/narration.txt"), "utf8")
  .split("\n").filter(l => l.startsWith("[")).map(l => l.replace(/^\[\d+ · [\d:]+\]\s*/, ""));

// [act, [[beat number, slide name], ...]] -- same order as the final video
const ACTS = [
  ["My pain point", [[1, "s01_meet_qb_qbo"], [2, "s03_simple_questions"]]],
  ["The technical barrier", [[3, "s02_ball_decoy"], [4, "s04_cost_math"], [5, "s05_two_halves"]]],
  ["What we built", [[6, "s05b_scope"], [6, "s06_title_built_both"], [7, "s06b_bridge"]]],
  ["Architecture", [[8, "s07_loop_diagram"]]],
  ["How it works: escalation", [[9, "s08_timeline"], [10, "s09_doubt_taxonomy"], [11, "s10_question_card"]]],
  ["How it works: the agent", [[12, "s10b_agent_work"], [13, "s12_batching"], [14, "s14_row_diff"]]],
  ["How it works: memory", [[15, "s15_day_log"], [16, "s16_memory_pyramid"]]],
  ["Evaluation", [[17, "s11_case_e5"], [18, "s13_case_e10"], [19, "s13b_detector_fixes"], [20, "s17a_timeline_eval"], [21, "s17_results"]]],
  ["Beyond cats, and what's next", [[22, "s19_beyond_cats"], [23, "s18_roadmap"], [24, "s20_end_card"]]],
];
const TITLE_FIX = {
  s01_meet_qb_qbo: ["Meet the cats", "Qb and Qbo: my camera can't tell them apart"],
  s02_ball_decoy: ["What the camera believed", "An exercise ball, tracked as Qbo for 26 minutes"],
  s11_case_e5: ["Case E2: which cat is asleep in the bed?", "Overturned: this is Qb, not Qbo"],
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in, same 16:9 as the 1920x1080 slides
pres.title = "Agentic Realtime Computer Vision";
pres.author = "Muyang Lu and Terry Li";
pres.subject = "Meta Global AI Developer Hackathon 2026 demo";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };

const seenBeat = new Set();
for (const [act, items] of ACTS) {
  pres.addSection({ title: act });
  for (const [beat, name] of items) {
    const m = meta[name];
    const [kicker, title] = TITLE_FIX[name] || [m.kicker, m.title];
    const s = pres.addSlide({ sectionTitle: act });
    s.background = { color: "1A1A19" };
    s.addImage({ path: path.join(DEMO, "assets/slides", `${name}.png`), x: 0, y: 0, w: 10, h: 5.625,
                 altText: [kicker, title].filter(Boolean).join(": ") });
    const vo = seenBeat.has(beat) ? "(narration continues from the previous slide)" : narration[beat - 1];
    seenBeat.add(beat);
    s.addNotes([`Beat ${beat} · ${act}`, "", `VO: ${vo}`, "", m.foot ? `Source: ${m.foot}` : ""].join("\n").trim());
  }
}
const out = path.join(DEMO, "assets", "agentic-realtime-cv-demo.pptx");
pres.writeFile({ fileName: out }).then(f => console.log(f));
