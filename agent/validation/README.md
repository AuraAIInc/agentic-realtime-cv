# validation

Turns escalations into verdicts and verdicts into corrected rows. Used by
`agent.main --validate` (worker + applier) and `--memory` (labeler).

| File | What it does |
|---|---|
| [`worker.py`](worker.py) | `ValidationWorker`: background thread. Takes escalation batches, builds the reviewer prompt (`agent/escalation/prompting.py`) + JSON output schema (`review_schema.py`), adds a short memory brief and the verdicts already given in this run, runs the supervisor agent with the validation tool set (`images__get_image`, `images__list_images`, `tracking__get_period_detail`, `local__get_frame`, `local__get_track_box`, `local__find_references`, `local__get_reference_image`), parses the answer, journals one op per escalation to `validations.jsonl`. A failed call journals an `error` verdict; it never stops the run. |
| [`batcher.py`](batcher.py) | Merges escalations that arrive close together (default within 60 s of the waiting batch, at most 4 cases and 12k prompt chars) into one joint agent call. Downstream still sees one op per escalation. Design: [`documents/escalation-merge-design.md`](../../documents/escalation-merge-design.md). |
| [`apply.py`](apply.py) | `materialize()`: applies the ops to `revised.jsonl` → `validated.jsonl` + `validated.db` (SQLite). Relabels, drops, splits, merges (with gap fill) and winner reassignments; verdicts it cannot execute exactly keep the rows and record a note. Every touched row carries `validated_by` and `verdict`. |
| [`labeler.py`](labeler.py) | `PeriodLabeler`: for each closed period, fetches its key frames from the images server and asks one vision LLM call for the activity (`agent_data/config/labels.yaml`), appearance and background. Feeds memory; degrades to "unknown" on any failure. |

Re-apply stored verdicts offline (no LLM) after changing `apply.py`:

```bash
python evaluation/rematerialize.py output/tracking_runs/<run> <out_dir>   # writes a re-applied copy
```

Tests: `python -m pytest agent/validation/tests -q` (fake backends, no API key).
