"""Download every image of one hour from the image MCP server and export a YOLO dataset.

Usage (Python >= 3.10):
    python image_server/fetch_hour_images.py 2025-05-05-19
    python image_server/fetch_hour_images.py 2025-05-05-19 --out ~/Downloads/cats --workers 8
    python image_server/fetch_hour_images.py 2025-05-05-19 --labels detection --view-id 3

Layout (default output image_server/retrieved_images/<hour>/, git-ignored):
    <hour>/
        classes.txt         one class name per line; line number = YOLO class id
        classes_map.json    cat_id -> class id, keeps indices stable across reruns
        data.yaml           ultralytics-style dataset file (names only)
        detections.png      detection chart for the hour (plot_detection_details)
        images/
            0000_2025-05-05-19-00-00-123456.jpg   (index prefix => name sort = time order)
            ...
        labels/
            0000_2025-05-05-19-00-00-123456.txt   (only for images that have detections)
            classes.txt     copy of the classes file, for labelImg
    Label lines are YOLO format: <class_id> <x_center> <y_center> <w> <h>, 0-1 normalized.

    Open the result in labelImg (https://pypi.org/project/labelImg/): switch to YOLO
    mode ("PascalVOC" button toggles to "YOLO"), open images/, and when asked for the
    annotation directory point it at labels/. Pre-labels come from MySQL
    (detection / revised_detection tables); existing label files are never
    overwritten unless --overwrite-labels is passed, so labelImg edits survive reruns.

Image download is stdlib-only. Labels need pymysql plus MYSQL_HOST/PORT/DB/USER/PASSWORD
from the repo-root .env (same file as IMAGE_MCP_URL/MCP_AUTH_TOKEN; it wins over the shell).
The chart needs matplotlib/seaborn/pandas on top and never fails the export: if its
dependencies are missing or the hour has no data, a warning is printed instead
(pass --no-viz to skip it).
Files already in the output folder are skipped (legacy flat files are moved into
images/ with their new indexed name), so an interrupted run can simply be restarted.
"""

import argparse
import base64
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HOUR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}$")
PREFIXED_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}\.jpg)$")
RETRIEVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieved_images")


REPO_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_env():
    # minimal reader for the repo-root .env (no python-dotenv needed), whatever the working
    # directory; its values win over shell exports, like the servers that share it
    if not os.path.isfile(REPO_ENV):
        print(f"warning: {REPO_ENV} not found, using the environment only", file=sys.stderr)
        return
    for line in open(REPO_ENV):
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


def token_from_env():
    """MCP_AUTH_TOKEN (shared), else IMAGE_MCP_TOKEN; same rule as mcp_image_server."""
    for name in ("MCP_AUTH_TOKEN", "IMAGE_MCP_TOKEN"):
        value = (os.getenv(name) or "").strip().strip("\"'").strip()
        if value:
            return value
    return None


class McpClient:
    """Streamable-HTTP MCP client: initialize once, then JSON-RPC tool calls on that session."""

    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.session_id = None
        self.next_id = 0
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "fetch_hour_images", "version": "1"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, body):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.session_id = resp.headers.get("Mcp-Session-Id") or self.session_id
            raw = resp.read().decode()
            content_type = resp.headers.get("Content-Type", "")
        if not raw.strip():
            return None
        if "event-stream" in content_type:
            raw = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")][-1]
        return json.loads(raw)

    def _rpc(self, method, params):
        self.next_id += 1
        resp = self._post({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        if "error" in resp:
            raise RuntimeError(f"{method}: {resp['error']}")
        return resp["result"]

    def call(self, tool, **arguments):
        result = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(f"{tool}: {result['content'][0].get('text', '')[:300]}")
        return result["content"]


def list_hour(client, hour):
    def names(prefix):
        data = json.loads(client.call("list_images", prefix=prefix, limit=100000)[0]["text"])
        return data["images"], data["truncated"]

    images, truncated = names(hour)
    if not truncated:
        return sorted(images)
    # the server caps a listing (max_list); minute prefixes are always small enough
    images = []
    for minute in range(60):
        images += names(f"{hour}-{minute:02d}")[0]
    return sorted(images)


def index_width(count):
    # 0000-9999 requested; an hour at ~3 fps can exceed 10k frames, so grow as needed
    # (a fixed width of 4 would break name-sort order past 9999).
    return max(4, len(str(max(0, count - 1))))


def prefixed_name(idx, width, old):
    return f"{idx:0{width}d}_{old}"


def parse_image_time(name):
    return datetime.datetime.strptime(name, "%Y-%m-%d-%H-%M-%S-%f.jpg")


def normalize_width(images_dir, labels_dir, width):
    """Rename already-downloaded files when the index width grew (e.g. 4 -> 5 digits)."""
    renamed = 0
    for old in sorted(os.listdir(images_dir)):
        m = PREFIXED_RE.match(old)
        if not m or len(m.group(1)) == width:
            continue
        new = f"{int(m.group(1)):0{width}d}_{m.group(2)}"
        os.replace(os.path.join(images_dir, old), os.path.join(images_dir, new))
        old_txt = os.path.join(labels_dir, old[:-4] + ".txt")
        if os.path.exists(old_txt):
            os.replace(old_txt, os.path.join(labels_dir, new[:-4] + ".txt"))
        renamed += 1
    if renamed:
        print(f"renamed {renamed} files to {width}-digit index width")
    return renamed


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path):
    """(width, height) from the JPEG header without decoding, or None if unreadable."""
    try:
        f = open(path, "rb")
    except OSError:
        return None
    with f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            while b == b"\xff":
                b = f.read(1)
            if not b:
                return None
            marker = b[0]
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if marker == 0xD9:
                return None
            raw_len = f.read(2)
            if len(raw_len) < 2:
                return None
            if marker in _SOF:
                d = f.read(6)
                if len(d) < 6:
                    return None
                return (int.from_bytes(d[3:5], "big"), int.from_bytes(d[1:3], "big"))
            f.seek(int.from_bytes(raw_len, "big") - 2, 1)


def to_yolo(x1, y1, x2, y2, space_w, space_h):
    """Clamp a pixel box to the coordinate space and normalize to YOLO cx,cy,w,h; None if degenerate."""
    x1 = min(max(float(x1), 0.0), space_w)
    x2 = min(max(float(x2), 0.0), space_w)
    y1 = min(max(float(y1), 0.0), space_h)
    y2 = min(max(float(y2), 0.0), space_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / space_w, (y1 + y2) / 2 / space_h, (x2 - x1) / space_w, (y2 - y1) / space_h)


def aspect_mismatch(space, actual, tol=0.03):
    sw, sh = space
    aw, ah = actual
    if not all((sw, sh, aw, ah)):
        return False
    return abs((sw / sh) / (aw / ah) - 1.0) > tol


# ------------------------------------------------------------------ labels from MySQL

def mysql_connect():
    try:
        import pymysql
    except ImportError:
        sys.exit("labels need the pymysql package (pip install pymysql), or pass --labels none")
    host = os.getenv("MYSQL_HOST")
    if not host:
        sys.exit("labels need MYSQL_HOST/PORT/DB/USER/PASSWORD in the environment or .env (or --labels none)")
    try:
        return pymysql.connect(
            host=host,
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
            connect_timeout=15,
            read_timeout=120,
        )
    except Exception as e:
        sys.exit(f"cannot reach MySQL at {host}: {str(e)[:200]} (or pass --labels none)")


def fetch_detections(table, start, end, view_id):
    """{create_time: [rows]} plus {view_id: (w, h)} and {cat_id: name}.

    Rows are normalized to (create_time, x1, y1, x2, y2, original_cat_id, conf, cat_id, view_id)
    for both tables; detection rows use cat_id as original_cat_id."""
    conn = mysql_connect()
    try:
        cur = conn.cursor()
        if table == "revised":
            cols = "create_time, box_x1, box_y1, box_x2, box_y2, original_cat_id, conf, cat_id, view_id"
            sql = f"SELECT {cols} FROM revised_detection WHERE create_time >= %s AND create_time < %s"
        else:
            cols = "create_time, box_x1, box_y1, box_x2, box_y2, cat_id, conf, cat_id, view_id"
            sql = f"SELECT {cols} FROM detection WHERE create_time >= %s AND create_time < %s"
        args: tuple = (start, end)
        if view_id is not None:
            sql += " AND view_id = %s"
            args = (start, end, view_id)
        try:
            cur.execute(sql, args)
            db_rows = cur.fetchall()
        except Exception as e:
            sys.exit(f"querying {table}_detection failed: {e}")
        by_time: dict = {}
        for r in db_rows:
            by_time.setdefault(r[0], []).append(r)
        try:
            cur.execute("SELECT view_id, width, height FROM view")
            view_sizes = {int(v): (int(w), int(h)) for v, w, h in cur.fetchall() if w and h}
        except Exception:
            view_sizes = {}
        try:
            cur.execute("SELECT cat_id, name FROM cats")
            cat_names = {int(c): n for c, n in cur.fetchall()}
        except Exception:
            cat_names = {}
    finally:
        conn.close()
    return by_time, view_sizes, cat_names


def load_class_map(out):
    path = os.path.join(out, "classes_map.json")
    if not os.path.isfile(path):
        return None
    try:
        data = json.load(open(path))
        order = [int(c) for c in data.get("order", [])]
        names = {int(k): v for k, v in data.get("names", {}).items()}
        if order and all(c in names for c in order):
            return order, names
    except (ValueError, OSError):
        pass
    return None


def write_classes(out, hour, order, names):
    """classes.txt (+ labelImg copy in labels/), classes_map.json and data.yaml."""
    lines = [names[c] for c in order]
    for path in (os.path.join(out, "classes.txt"), os.path.join(out, "labels", "classes.txt")):
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "classes_map.json"), "w") as f:
        json.dump({"order": order, "names": {str(c): names[c] for c in order}}, f, indent=2)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"# YOLO dataset exported by image_server/fetch_hour_images.py for hour {hour}\n")
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: images\nval: images\nnames:\n")
        for i, c in enumerate(order):
            f.write(f"  {i}: {json.dumps(names[c])}\n")


def write_labels(out, targets, by_time, view_sizes, cat_names, box_size, skip_interpolated, min_conf, overwrite):
    """Write one YOLO .txt per image that has detections. Returns (written, kept, empty, skipped)."""
    order_names = load_class_map(out)
    if order_names is None:
        cat_ids = sorted({int(r[7]) for rows in by_time.values() for r in rows if r[7] is not None})
        order_names = (cat_ids, {c: cat_names.get(c, f"cat_{c}") for c in cat_ids})
    order, names = order_names
    # never reorder existing indices; append unseen cats (names refresh from the cats table)
    seen = set(order)
    changed = False
    for c in sorted({int(r[7]) for rows in by_time.values() for r in rows if r[7] is not None}):
        if c not in seen:
            order.append(c)
            names[c] = cat_names.get(c, f"cat_{c}")
            seen.add(c)
            changed = True
    for c in order:
        if c in cat_names and names.get(c) != cat_names[c]:
            names[c] = cat_names[c]
            changed = True
    if changed or not os.path.isfile(os.path.join(out, "classes.txt")):
        write_classes(out, os.path.basename(out.rstrip("/")), order, names)
    elif not os.path.isfile(os.path.join(out, "labels", "classes.txt")):
        write_classes(out, os.path.basename(out.rstrip("/")), order, names)
    idx_of = {c: i for i, c in enumerate(order)}

    written = kept = empty = skipped = bad_boxes = 0
    warned_space = warned_aspect = False
    present_times = set()
    for old, new in targets:
        img_path = os.path.join(out, "images", new)
        if not os.path.isfile(img_path):
            continue
        try:
            ts = parse_image_time(old)
        except ValueError:
            skipped += 1
            continue
        present_times.add(ts)
        rows = [r for r in by_time.get(ts, []) if r[1] is not None]
        if skip_interpolated:
            rows = [r for r in rows if r[5] is not None]
        if min_conf is not None:
            rows = [r for r in rows if r[6] is None or float(r[6]) >= min_conf]
        if not rows:
            empty += 1
            continue
        label_path = os.path.join(out, "labels", new[:-4] + ".txt")
        if os.path.exists(label_path) and not overwrite:
            kept += 1
            continue
        actual = jpeg_size(img_path)
        lines = []
        for r in sorted(rows, key=lambda r: (int(r[7]), float(r[1]))):
            space = box_size or view_sizes.get(int(r[8])) if r[8] is not None else box_size
            space = space or actual
            if space is None:
                if not warned_space:
                    print("warning: no box coordinate space (view table empty?) and unreadable JPEG; skipping those boxes")
                    warned_space = True
                bad_boxes += 1
                continue
            if actual and aspect_mismatch(space, actual):
                if not warned_aspect:
                    print(f"warning: image is {actual[0]}x{actual[1]} but boxes use {space[0]}x{space[1]} space; normalizing by the box space")
                    warned_aspect = True
            yolo = to_yolo(r[1], r[2], r[3], r[4], space[0], space[1])
            if yolo is None:
                bad_boxes += 1
                continue
            lines.append(f"{idx_of[int(r[7])]} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
        if not lines:
            skipped += 1
            continue
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1
    orphan = sum(len(v) for t, v in by_time.items() if t not in present_times)
    if bad_boxes:
        print(f"  {bad_boxes} boxes skipped (no coordinate space or degenerate)")
    if orphan:
        print(f"  {orphan} detection rows had no matching downloaded image")
    return written, kept, empty, skipped


def write_viz(out, hour_start, revised, width, height):
    """Render the hour's detection chart into <out>/detections.png.

    Reuses plot_detection_details from existing_work_before_hackathon's visualization
    module (headless Agg backend, no interactive window). Returns True on success;
    missing dependencies, DB errors and empty hours only print a warning."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("viz skipped: matplotlib is not installed")
        return False
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "existing_work_before_hackathon"))
    try:
        from model.visualization import plot_detection_details
    except ImportError as e:
        print(f"viz skipped: cannot import visualization ({str(e)[:150]})")
        return False
    dest = os.path.join(out, "detections.png")
    try:
        ok = plot_detection_details(hour_start, hour_start + datetime.timedelta(hours=1),
                                    revised=revised, width=width, height=height,
                                    save_path=dest, show=False)
    except Exception as e:
        print(f"viz skipped: chart failed ({str(e)[:200]})")
        return False
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    if not ok:
        print("viz skipped: no data for this hour")
        return False
    print(f"chart: {dest}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("hour", help="YYYY-MM-DD-HH, 24-hour clock, e.g. 2025-05-05-19")
    parser.add_argument("--out", help="output folder (default image_server/retrieved_images/<hour>)")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--url", help="MCP endpoint (default $IMAGE_MCP_URL)")
    parser.add_argument("--labels", choices=["revised", "detection", "none"], default="revised",
                        help="MySQL table for pre-labels (default revised; none = images only)")
    parser.add_argument("--no-labels", action="store_true", help="shortcut for --labels none")
    parser.add_argument("--view-id", type=int, default=None, help="only use detections from this view (default: all)")
    parser.add_argument("--skip-interpolated", action="store_true",
                        help="revised only: skip tracker-interpolated rows (original_cat_id NULL)")
    parser.add_argument("--min-conf", type=float, default=None, help="only rows with conf >= this (rows with NULL conf are kept)")
    parser.add_argument("--overwrite-labels", action="store_true",
                        help="regenerate label files even if they exist (default keeps them, so labelImg edits survive)")
    parser.add_argument("--box-size", help="detection-box coordinate space as WxH (default: view width/height from MySQL)")
    parser.add_argument("--no-viz", action="store_true", help="skip the detections.png chart")
    parser.add_argument("--limit", type=int, default=None, help="fetch/label only the first N images (debug)")
    args = parser.parse_args()

    if not HOUR_RE.match(args.hour):
        sys.exit(f"hour must look like 2025-05-05-19, got {args.hour!r}")
    if args.no_labels:
        args.labels = "none"
    box_size = None
    if args.box_size:
        m = re.match(r"^(\d+)x(\d+)$", args.box_size)
        if not m:
            sys.exit("--box-size must look like 800x600")
        box_size = (int(m.group(1)), int(m.group(2)))
    load_env()
    url = args.url or os.getenv("IMAGE_MCP_URL", "http://10.0.0.165:10001/mcp")
    if not url:
        sys.exit("set IMAGE_MCP_URL (or pass --url), e.g. http://10.0.0.165:10001/mcp")
    token = token_from_env()
    out = os.path.expanduser(args.out) if args.out else os.path.join(RETRIEVED_DIR, args.hour)

    try:
        client = McpClient(url, token)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the image server at {url}: {e.reason} (is mcp_image_server running?)")

    names = list_hour(client, args.hour)
    if not names:
        sys.exit(f"no images for {args.hour}")
    print(f"{len(names)} images  {names[0]} .. {names[-1]}")

    width = index_width(len(names))
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    normalize_width(images_dir, labels_dir, width)

    targets = [(n, prefixed_name(i, width, n)) for i, n in enumerate(names)]
    if args.limit is not None:
        targets = targets[: max(0, args.limit)]

    todo, migrated, already = [], 0, 0
    for old, new in targets:
        if os.path.exists(os.path.join(images_dir, new)):
            already += 1
            continue
        # pick up files from the old flat layout (or unprefixed strays in images/)
        legacy = next((p for p in (os.path.join(out, old), os.path.join(images_dir, old)) if os.path.exists(p)), None)
        if legacy:
            os.replace(legacy, os.path.join(images_dir, new))
            migrated += 1
        else:
            todo.append((old, new))
    if already or migrated:
        print(f"skipping {already} already downloaded, migrated {migrated} legacy files")

    lock = threading.Lock()
    pending = iter(todo)
    failed = []
    done = [0]

    def worker():
        worker_client = McpClient(url, token)  # one MCP session per thread
        while True:
            with lock:
                item = next(pending, None)
            if item is None:
                return
            old, new = item
            try:
                images = [c for c in worker_client.call("get_image", image_name=old) if c["type"] == "image"]
                if not images:
                    raise RuntimeError("no image returned (deleted since listing?)")
                # write to a temp name first so a crash never leaves a half file that resume would skip
                path = os.path.join(images_dir, new)
                with open(path + ".part", "wb") as f:
                    f.write(base64.b64decode(images[0]["data"]))
                os.replace(path + ".part", path)
            except Exception as e:
                with lock:
                    failed.append((old, str(e)[:200]))
            with lock:
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(todo):
                    print(f"  {done[0]}/{len(todo)}", flush=True)

    start = time.time()
    if todo:
        threads = [threading.Thread(target=worker) for _ in range(max(1, min(args.workers, len(todo))))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(f"saved to {out} in {time.time() - start:.0f}s, {len(failed)} failed")
    for name, error in failed[:10]:
        print(f"  {name}: {error}")

    label_error = False
    hour_start = datetime.datetime.strptime(args.hour, "%Y-%m-%d-%H")
    view_sizes: dict = {}
    if args.labels != "none" and not failed:
        try:
            by_time, view_sizes, cat_names = fetch_detections(args.labels, hour_start, hour_start + datetime.timedelta(hours=1), args.view_id)
            print(f"{sum(len(v) for v in by_time.values())} rows from {args.labels}_detection"
                  + (f" (view {args.view_id})" if args.view_id is not None else ""))
            written, kept, empty, skipped = write_labels(
                out, targets, by_time, view_sizes, cat_names, box_size,
                args.skip_interpolated, args.min_conf, args.overwrite_labels)
            print(f"labels: {written} written, {kept} kept (existing), {empty} images without detections"
                  + (f", {skipped} skipped" if skipped else ""))
        except SystemExit as e:
            print(f"labels failed: {e}")
            label_error = True
    elif args.labels != "none":
        print("skipping labels: some downloads failed (rerun to resume, then labels will be written)")
        label_error = True

    if not args.no_viz:
        viz_w, viz_h = view_sizes.get(args.view_id, (640, 480)) if args.view_id is not None else (640, 480)
        write_viz(out, hour_start, args.labels == "revised", viz_w, viz_h)
    sys.exit(1 if failed or label_error else 0)


if __name__ == "__main__":
    main()
