"""Label retrieved hour images with Muse Spark 1.3 Contributor in one call each.

Unlike the SAM script (propose regions, then classify), this script asks the
vision-language model to detect AND identify every cat in a single request.
The reply is constrained by a strict JSON schema generated from the
``SparkBoxes`` pydantic class below, so the output always parses; boxes are
reported in 0-1000 scale and converted to YOLO lines in spark_labels/.

WARNING - Contributor mode: the contributor tier is ~10x cheaper because you
grant Meta permission to train future models on your content (the images you
send and the labels you get back). Standard mode does not. Only run this on
footage you are willing to share for training. This warning is also printed
on every run.

Usage (Python >= 3.10, stdlib + pyyaml + pydantic):
    python image_server/grounding_truth/label_with_spark.py 2026-10-28-16
    python image_server/grounding_truth/label_with_spark.py --start 2026-10-28-16-33 --end 2026-10-28-16-40
    python image_server/grounding_truth/label_with_spark.py --start 2026-10-28-16-33 --end 2026-10-28-16-33 --limit 5 --dry-run

Time bounds are YYYY-MM-DD-HH-MM (the end minute is fully included) and may
span hours; every overlapped hour folder gets its own spark_labels/ beside
images/ and labels/. An empty .txt means "processed, nothing survived".
Existing spark_labels/ files are skipped unless --overwrite is passed.

Billing (Meta Model API, Contributor mode, verified 2026-11-03 at
https://dev.meta.ai/docs/pricing-rate-limits): $0.10/$0.20 per 1M
input/output tokens - roughly $0.001 per image. A Markdown + JSON session
report lands in grounding_truth/reports/ after every run, including dry runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import sys
import threading
import time

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_with_sam as sam_mod  # noqa: E402 (shared discovery/prompt/API helpers)

logger = logging.getLogger(__name__)

CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"
CONTRIBUTOR_FALLBACK_PRICING = {"input": 0.10, "output": 0.20}  # USD per 1M tokens
TYPICAL_TOKENS_EACH = {"input": 700, "output": 3500}  # measured; dry-run fallback
OUT_DIRNAME = "spark_labels"
COORD_MAX = 1000


class SparkBox(BaseModel):
    """One detected cat: class id + box in 0-1000 scale."""

    model_config = ConfigDict(extra="forbid")

    id: int
    x1: int
    y1: int
    x2: int
    y2: int


class SparkBoxes(BaseModel):
    """Strict structured-output schema sent to the API (via model_json_schema)."""

    model_config = ConfigDict(extra="forbid")

    boxes: list[SparkBox]


def load_contributor_settings(model_config=None):
    """Return (model, base_url, api_key, pricing) for the contributor tier."""
    cfg_path = model_config or os.getenv("MODEL_CONFIG") or sam_mod.DEFAULT_MODELS
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    base_url = (data.get("base_url", "") or "").rstrip("/")
    api_key = os.getenv(data.get("api_key_env", "MODEL_API_KEY"), "")
    pricing = ((data.get("pricing_usd_per_1m_tokens") or {}).get(CONTRIBUTOR_MODEL)
               or dict(CONTRIBUTOR_FALLBACK_PRICING))
    return (CONTRIBUTOR_MODEL, base_url, api_key.strip().strip("\"'"),
            {"input": float(pricing.get("input", 0.10)),
             "output": float(pricing.get("output", 0.20))})


def build_spark_detect_prompt(classes, ref_by_class, img_w, img_h):
    """Full input_text: detect + identify every cat; reference labels are context only."""
    lines = [
        f"Detect every cat in this {img_w}x{img_h} home-camera frame.",
        "Cats:",
    ]
    for q in classes:
        cat = q["cat"]
        cues = "; ".join(cat["cues"][:3])
        lines.append(f"- id {q['class_id']} {q['name']}: {cat['summary']} Cues: {cues}.")
    refs = [f"class {cid} at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            for cid, rbs in sorted(ref_by_class.items()) for x1, y1, x2, y2 in rbs]
    if refs:
        lines.append("Old-detector reference (may be wrong or missing; context "
                     "only, judge from the image): " + "; ".join(refs) + ".")
    else:
        lines.append("Old-detector reference: none (it found nothing; context "
                     "only, it may have missed a cat).")
    lines.append(f"Report each box as integers in 0-{COORD_MAX} scale "
                 f"({COORD_MAX} = full width/height). Skip anything you are unsure about.")
    return "\n".join(lines)


def build_spark_detect_payload(model, prompt, image_bytes):
    """Request body with the pydantic schema as a strict output format."""
    return {
        "model": model,
        "text": {"format": {"type": "json_schema", "name": "cat_boxes",
                            "schema": SparkBoxes.model_json_schema(), "strict": True}},
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt},
                        sam_mod._image_block(image_bytes)],
        }],
    }


def spark_boxes_to_yolo(parsed, valid_ids, img_w, img_h):
    """Validated SparkBoxes -> [(class_id, cx, cy, w, h)] YOLO; returns (kept, dropped)."""
    kept, dropped = [], 0
    for b in parsed.boxes:
        if b.id not in valid_ids:
            dropped += 1
            continue
        for v in (b.x1, b.y1, b.x2, b.y2):
            if not 0 <= v <= COORD_MAX:
                break
        else:
            if b.x2 > b.x1 and b.y2 > b.y1:
                cx = (b.x1 + b.x2) / 2 / COORD_MAX
                cy = (b.y1 + b.y2) / 2 / COORD_MAX
                w = (b.x2 - b.x1) / COORD_MAX
                h = (b.y2 - b.y1) / COORD_MAX
                kept.append((b.id, cx, cy, w, h))
                continue
        dropped += 1
    return kept, dropped


def estimate_cost_usd(calls, input_tokens, output_tokens, pricing):
    return (input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"])


def process_one(job, ctx):
    """Label one image with a single Contributor call. Never raises."""
    hour, images_dir, labels_dir, out_dir, name, stamp, classes = job
    started = time.monotonic()
    result = {"hour": hour, "image": name, "stamp": stamp, "status": "ok",
              "calls": 0, "boxes": {}, "dropped": 0, "seconds": 0.0,
              "usage": {}, "error": None}
    try:
        img_path = os.path.join(images_dir, name)
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        size = sam_mod.jpeg_size(img_path)
        if size is None:
            raise RuntimeError("unreadable JPEG header")
        img_w, img_h = size
        valid_ids = [c["class_id"] for c in classes]
        ref = sam_mod.read_all_reference_labels(labels_dir, name, valid_ids, img_w, img_h)
        prompt = build_spark_detect_prompt(classes, ref, img_w, img_h)
        with ctx["first_prompt_lock"]:
            if ctx["first_prompt"] is None:
                ctx["first_prompt"] = (f"{hour}/{name}", prompt)
        parsed = None
        for _ in range(2):  # one retry: strict mode usually makes this unnecessary
            output_text, usage = sam_mod.call_responses_with_retry(
                ctx["base_url"], ctx["api_key"],
                build_spark_detect_payload(ctx["model"], prompt, image_bytes))
            result["calls"] += 1
            sam_mod._tally_usage(result, usage)
            try:
                parsed = SparkBoxes.model_validate_json(output_text)
                break
            except ValueError:
                continue
        if parsed is None:
            raise RuntimeError("unparseable model reply after retry")
        kept, dropped = spark_boxes_to_yolo(parsed, valid_ids, img_w, img_h)
        result["dropped"] = dropped
        id_to_name = {c["class_id"]: c["name"] for c in classes}
        out_lines = []
        for cid, cx, cy, w, h in kept:
            out_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            result["boxes"][id_to_name[cid]] = result["boxes"].get(id_to_name[cid], 0) + 1
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, os.path.splitext(name)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    except Exception as e:  # per-image failure must not stop the session
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"[:300]
    result["seconds"] = time.monotonic() - started
    return result


def write_session_report(path_md, path_json, session):
    """Write the Markdown + JSON session report. Returns (path_md, path_json)."""
    s = session
    tokens = s["tokens"]
    md = [
        f"# Spark grounding-truth session — {s['session_id']} (CONTRIBUTOR tier)",
        "",
        f"- Started: {s['started']}  ·  finished: {s['finished']}  ·  "
        f"wall time: {s['wall_seconds']:.1f}s",
        f"- Range: {s['start_key']} .. {s['end_key'] or 'open'}"
        f"{' (DRY RUN - no API calls, no labels written)' if s['dry_run'] else ''}",
        f"- Hour folders: {', '.join(s['hours']) or 'none'}",
        f"- Model: {s['model']} (strict pydantic-structured output)  ·  "
        f"workers: {s['workers']}  ·  limit: {s['limit'] or 'none'}",
        f"- Pricing: ${s['pricing']['input']:.2f}/${s['pricing']['output']:.2f} "
        f"per 1M in/out tokens ([official pricing]({sam_mod.SAM_PRICING_URL}), "
        f"Contributor: content may train future Meta models)",
        "",
        "## Totals",
        "",
        f"- Images in range: {s['images_in_range']}  ·  processed: {s['processed']}  ·  "
        f"skipped (already labeled): {s['skipped']}  ·  failed: {s['failed']}",
        f"- Model calls: {s['calls']} (one per image)",
        f"- Boxes written: {s['boxes_total']} "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(s['boxes_by_class'].items())) or 'none'})"
        f"  ·  dropped (out-of-range/degenerate/unknown id): {s['dropped']}",
        f"- Tokens: {tokens['total']} total ({tokens['input']} in / {tokens['output']} out)",
        f"- Estimated cost: ${s['cost_usd']:.4f} "
        f"(${s['cost_usd'] / max(s['processed'], 1):.4f} per processed image)",
        "",
        "## Classes",
        "",
    ]
    for q in s["queries"]:
        md.append(f"- class {q['class_id']} = {q['name']}: {q['concept']}")
    md += [
        "",
        "## Full prompt of the first request",
        "",
        f"Image: {s['first_prompt_image'] or 'n/a (no images in range)'}"
        f"{' (illustrated; dry run makes no calls)' if s['dry_run'] else ''}",
        "",
        "```text",
        s["first_prompt"] or "(none)",
        "```",
        "",
        "## Per-hour breakdown",
        "",
        "| Hour | In range | Processed | Skipped | Failed | Boxes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for hour, row in sorted(s["per_hour"].items()):
        md.append(f"| {hour} | {row['in_range']} | {row['processed']} | "
                  f"{row['skipped']} | {row['failed']} | {row['boxes']} |")
    md += ["", "## Failures", ""]
    if s["failures"]:
        for f in s["failures"][:50]:
            md.append(f"- {f['hour']}/{f['image']}: {f['error']}")
        if len(s["failures"]) > 50:
            md.append(f"- ... and {len(s['failures']) - 50} more (see JSON report)")
    else:
        md.append("None.")
    md += [
        "",
        "## Notes",
        "",
        "- An empty spark_labels .txt means the image was processed and no box "
        "survived; it is not an error.",
        "- The labels/ boxes are prompt context only (marked as possibly "
        "wrong); written boxes come from the model reply alone.",
        "- Contributor tier: this session's images and replies may be used to "
        "train future Meta models.",
        "",
    ]
    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    return path_md, path_json


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] %(message)s")
    print("WARNING: Contributor tier - session images and replies may be used "
          "to train future Meta models. Standard mode does not do this.")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("hour", nargs="?", default=None,
                        help="YYYY-MM-DD-HH, e.g. 2026-10-28-16 (omit when --start/--end are given)")
    parser.add_argument("--start", default=None, help="label from this minute (YYYY-MM-DD-HH-MM)")
    parser.add_argument("--end", default=None, help="label through this minute, inclusive (YYYY-MM-DD-HH-MM)")
    parser.add_argument("--workers", type=int, default=10, help="parallel requests (default 10)")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N images in range (debug)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve files, print the first prompt and cost estimate; no API calls, no labels")
    parser.add_argument("--overwrite", action="store_true",
                        help="relabel images that already have a spark_labels/ file (default skips them)")
    parser.add_argument("--retrieved", default=None, help="retrieved_images root (default image_server/retrieved_images)")
    parser.add_argument("--cats", default=None, help="cat profiles YAML (default agent/agent_data/config/cats.yaml)")
    parser.add_argument("--models", default=None, help="models.json for base URL and pricing (default agent/supervisor_agent/models.json)")
    parser.add_argument("--report-dir", default=None, help="session report folder (default grounding_truth/reports)")
    args = parser.parse_args()

    try:
        start_key, end_key = sam_mod.resolve_bounds(args.hour, args.start, args.end)
    except ValueError as e:
        sys.exit(str(e))
    if args.workers < 1:
        sys.exit("--workers must be >= 1")
    sam_mod.load_env()
    model, base_url, api_key, pricing = load_contributor_settings(args.models)
    if not base_url:
        sys.exit("no base_url in models.json")
    if not api_key and not args.dry_run:
        sys.exit("set MODEL_API_KEY in the repo .env (or pass --dry-run to preview for free)")

    retrieved = args.retrieved or sam_mod.DEFAULT_RETRIEVED
    cats = sam_mod.load_cats(args.cats)
    targets = sam_mod.discover_images(retrieved, start_key, end_key, OUT_DIRNAME)
    if args.limit is not None:
        targets = targets[:max(args.limit, 0)]
    logger.info(f"{len(targets)} images in range {start_key} .. {end_key or 'open'}")

    classes_by_hour = {}
    for hour, *_ in targets:
        if hour not in classes_by_hour:
            classes_by_hour[hour] = sam_mod.load_hour_classes(os.path.join(retrieved, hour), cats)
            if not classes_by_hour[hour]:
                logger.warning(f"{hour}: no classes match the cat profiles, folder skipped")
    jobs, skipped = [], 0
    per_hour = {}
    for hour, images_dir, labels_dir, out_dir, name, stamp in targets:
        row = per_hour.setdefault(hour, {"in_range": 0, "processed": 0, "skipped": 0, "failed": 0, "boxes": 0})
        row["in_range"] += 1
        classes = classes_by_hour.get(hour) or []
        if not classes:
            continue
        label_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(label_path) and not args.overwrite and not args.dry_run:
            skipped += 1
            row["skipped"] += 1
            continue
        jobs.append((hour, images_dir, labels_dir, out_dir, name, stamp, classes))

    # full prompt of the first request (dry runs illustrate it directly).
    first_prompt, first_prompt_image = "", None
    if jobs and args.dry_run:
        hour, images_dir, labels_dir, _out, name, _stamp, classes = jobs[0]
        size = sam_mod.jpeg_size(os.path.join(images_dir, name)) or (0, 0)
        valid_ids = [c["class_id"] for c in classes]
        ref = sam_mod.read_all_reference_labels(labels_dir, name, valid_ids, *size)
        first_prompt = build_spark_detect_prompt(classes, ref, *size)
        first_prompt_image = f"{hour}/{name}"
        print("=" * 72)
        print(f"FIRST PROMPT ({first_prompt_image}):")
        print("=" * 72)
        print(first_prompt)
        print("=" * 72)

    ctx = {"base_url": base_url, "api_key": api_key, "model": model,
           "first_prompt_lock": threading.Lock(), "first_prompt": None}
    session_id = "spark-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    started = datetime.datetime.now()
    wall_start = time.monotonic()
    results = []
    if args.dry_run:
        est_cost = estimate_cost_usd(len(targets), len(targets) * TYPICAL_TOKENS_EACH["input"],
                                     len(targets) * TYPICAL_TOKENS_EACH["output"], pricing)
        print(f"DRY RUN: {len(targets)} images, {len(targets)} calls planned, "
              f"estimated cost ~${est_cost:.4f} (typical ~700/3500 tokens each; "
              f"time estimate unavailable without API calls)")
    elif jobs:
        # pilot: first image runs synchronously to calibrate the estimate
        pilot = process_one(jobs.pop(0), ctx)
        results.append(pilot)
        total = len(results) + len(jobs)
        per_call = pilot["usage"] or dict(TYPICAL_TOKENS_EACH)
        per_call_src = "pilot" if pilot["usage"] else "typical"
        est_cost = estimate_cost_usd(total, total * per_call.get("input", 0),
                                     total * per_call.get("output", 0), pricing)
        est_wall = pilot["seconds"] * (1 + (total - 1) / max(args.workers, 1))
        print(f"ESTIMATE for {total} images (pilot: {pilot['seconds']:.1f}s, "
              f"{per_call.get('input', 0)}/{per_call.get('output', 0)} in/out tokens, "
              f"{per_call_src}): ~{total} calls, ~${est_cost:.4f}, "
              f"~{sam_mod.format_duration(est_wall)} wall at {args.workers} workers "
              f"(rough: assumes linear scaling).")
        done = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, job, ctx): job for job in jobs}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"{done}/{total} images done")
    wall_seconds = time.monotonic() - wall_start
    if ctx["first_prompt"] is not None:
        first_prompt_image, first_prompt = ctx["first_prompt"]
        print("=" * 72)
        print(f"FIRST PROMPT ({first_prompt_image}):")
        print("=" * 72)
        print(first_prompt)
        print("=" * 72)

    calls = sum(r["calls"] for r in results)
    tokens = {"input": 0, "output": 0, "total": 0}
    boxes_by_class, boxes_total, dropped, failed = {}, 0, 0, []
    for r in results:
        for key in tokens:
            tokens[key] += r["usage"].get(key, 0)
        for name, count in r["boxes"].items():
            boxes_by_class[name] = boxes_by_class.get(name, 0) + count
            boxes_total += count
            per_hour[r["hour"]]["boxes"] += count
        dropped += r["dropped"]
        if r["status"] == "error":
            failed.append({"hour": r["hour"], "image": r["image"], "error": r["error"]})
            per_hour[r["hour"]]["failed"] += 1
        else:
            per_hour[r["hour"]]["processed"] += 1
    processed = len(results)
    cost_usd = estimate_cost_usd(calls, tokens["input"], tokens["output"], pricing)

    if not args.dry_run:
        for hour in {r["hour"] for r in results if r["status"] == "ok"}:
            src = os.path.join(retrieved, hour, "classes.txt")
            dst = os.path.join(retrieved, hour, OUT_DIRNAME, "classes.txt")
            if os.path.isfile(src) and not os.path.isfile(dst):
                try:
                    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                        f_out.write(f_in.read())
                except OSError as e:
                    logger.warning(f"{hour}: cannot copy classes.txt: {e}")

    queries_all = []
    seen = set()
    for hour in sorted(classes_by_hour):
        for q in classes_by_hour[hour]:
            key = (q["class_id"], q["name"])
            if key not in seen:
                seen.add(key)
                queries_all.append({"class_id": q["class_id"], "name": q["name"],
                                    "concept": sam_mod.short_concept(q["cat"])})
    session = {
        "session_id": session_id,
        "started": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(wall_seconds, 1),
        "dry_run": bool(args.dry_run),
        "start_key": start_key, "end_key": end_key,
        "hours": sorted(classes_by_hour),
        "model": model, "pricing": pricing, "workers": args.workers, "limit": args.limit,
        "images_in_range": len(targets), "processed": processed,
        "skipped": skipped, "failed": len(failed),
        "calls": calls, "boxes_total": boxes_total, "boxes_by_class": boxes_by_class,
        "dropped": dropped, "tokens": tokens, "cost_usd": round(cost_usd, 4),
        "queries": queries_all,
        "first_prompt_image": first_prompt_image, "first_prompt": first_prompt,
        "per_hour": per_hour, "failures": failed,
        "results": sorted(results, key=lambda r: (r["hour"], r["image"])),
    }
    report_dir = args.report_dir or sam_mod.REPORTS_DIR
    md_path, json_path = write_session_report(
        os.path.join(report_dir, f"session-{session_id}.md"),
        os.path.join(report_dir, f"session-{session_id}.json"), session)

    print(f"Done: {processed} images processed, {skipped} skipped, {len(failed)} failed, "
          f"{calls} calls, {boxes_total} boxes, "
          f"{tokens['total']} tokens ({tokens['input']} in / {tokens['output']} out), "
          f"{wall_seconds:.1f}s wall, estimated cost ${cost_usd:.4f}.")
    print(f"Session report: {md_path} (+ .json)")


if __name__ == "__main__":
    main()
