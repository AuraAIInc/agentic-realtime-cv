"""Re-apply a run's stored verdicts with the current apply code, into a copy of the run.

No LLM calls: validations.jsonl already holds every {escalation, review}. Use it to measure
apply-only changes. The copy gets new validated.jsonl / validated.db; everything else is linked.

    .venv/bin/python evaluation/rematerialize.py output/tracking_runs/<run> <out_dir>
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.validation.apply import materialize  # noqa: E402


def main(run: str, out: str) -> str:
    run = run.rstrip("/")
    dst = os.path.join(out, os.path.basename(run) + "_reapplied")
    os.makedirs(dst, exist_ok=True)
    for f in ("stream.jsonl", "revised.jsonl", "periods.jsonl", "images.jsonl", "escalations.jsonl", "validations.jsonl", "run.json"):
        if os.path.exists(os.path.join(run, f)):
            shutil.copy(os.path.join(run, f), dst)
    rows = [json.loads(l) for l in open(os.path.join(run, "revised.jsonl")) if l.strip()]
    ops = [json.loads(l) for l in open(os.path.join(run, "validations.jsonl")) if l.strip()]
    cat_ids = json.load(open(os.path.join(run, "run.json"))).get("cat_ids")
    for name in ("validated.jsonl", "validated.db"):
        if os.path.exists(os.path.join(dst, name)):
            os.remove(os.path.join(dst, name))
    materialize(dst, rows, ops, cat_ids=cat_ids)
    return dst


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2]))
