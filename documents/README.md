# documents

Design notes, experiment reports and hackathon material. Files keep their original names
(other docs and the demo build scripts link to them); this page groups them.

## How the system works (start here)

| Doc | What it covers |
|---|---|
| [end-to-end-flow.md](end-to-end-flow.md) | Every stage of one `agent.main` run: ingest → TrackCorrector → PeriodCorrector → escalation → supervisor → apply → memory, with code pointers |
| [how-track-correctors-work.md](how-track-correctors-work.md) | The two correctors in depth: relabeling, interpolation, period voting, escalation hooks |
| [escalation-prompt-examples-20261101.md](escalation-prompt-examples-20261101.md) | A worked reviewer prompt for every escalation code |
| [escalation-merge-design.md](escalation-merge-design.md) | Merging close escalations into one supervisor call |
| [spatial-temporal-memory-design.md](spatial-temporal-memory-design.md) | Long-term memory: spatial grid × temporal buckets, trust gate, promotion |
| [short-term-memory-design.md](short-term-memory-design.md) | Short-term day memory and its merge into long-term memory |
| [tracking-viz.md](tracking-viz.md) | Reading the strip charts and per-frame process visualization |

## Results and experiments (dated)

| Doc | What it covers |
|---|---|
| [final-model-performance-20261104.md](final-model-performance-20261104.md) | Final numbers: raw detector vs agent-validated output across windows |
| [evaluation-report-20261104.md](evaluation-report-20261104.md) | Three replay runs scored against ground truth |
| [e2e-identity-fix-20261104.md](e2e-identity-fix-20261104.md) | Why the cats were mislabeled end to end, the fixes, the GT audit |
| [labeling-sam-vs-spark-20261103.md](labeling-sam-vs-spark-20261103.md) | Ground-truth labeling: Muse SAM 3.1 vs Muse Spark 1.3 |
| [agent-tool-invoke-report-20261103.md](agent-tool-invoke-report-20261103.md) | Agent and tool call statistics, Oct 27 – Nov 3 |
| [session-report-20261102-012059.md](session-report-20261102-012059.md) | One full-loop run, stage by stage |
| [agent-trace-review-20261101-003501.md](agent-trace-review-20261101-003501.md), [trace-review-20261101-003501.md](trace-review-20261101-003501.md) | Reviews of one agent trace: tools, prompts, skills |
| [experiment-20261031-191639-cat-false-negative.md](experiment-20261031-191639-cat-false-negative.md) | A missed-cat audit run |
| [model-api-chaining-probe.md](model-api-chaining-probe.md) | Model API behavior with image tool outputs |

Dated reports describe the code at that date; paths such as `agent/output/` in older ones
have since moved to the repo-root `output/`.

## Hackathon

| Doc | What it covers |
|---|---|
| [hackathon-project-report-claude.md](hackathon-project-report-claude.md), [hackathon-project-report-muse.md](hackathon-project-report-muse.md) | Project report (two drafts) |
| [demo-claude/](demo-claude/), [demo-muse/](demo-muse/) | Demo video script, slides, narration and build scripts |
| [project-plan.md](project-plan.md) ([中文](project-plan.zh.md)) | Original plan |
| [hackathon-guide.md](hackathon-guide.md) | Team guide and submission checklist |

`assets/` holds the images the reports embed.
