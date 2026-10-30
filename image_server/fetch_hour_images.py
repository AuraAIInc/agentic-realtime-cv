"""Download every image of one hour from the image MCP server (list_images + get_image).

Usage (stdlib only, Python >= 3.10):
    python image_server/fetch_hour_images.py 2025-05-05-19
    python image_server/fetch_hour_images.py 2025-05-05-19 --out ~/Downloads/cats --workers 8

Images go to image_server/retrieved_images/<hour>/ by default (git-ignored).
Reads IMAGE_MCP_URL and MCP_AUTH_TOKEN from the environment or the nearest .env
(current folder, then this script's folder and its parents). Files already in the
output folder are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HOUR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}$")
RETRIEVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieved_images")


def load_env():
    # minimal .env reader so the script needs no python-dotenv; real env vars win
    folders = [os.getcwd()]
    d = os.path.dirname(os.path.abspath(__file__))
    while d not in folders:
        folders.append(d)
        d = os.path.dirname(d)
    for folder in folders:
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


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
        return images
    # the server caps a listing (max_list); minute prefixes are always small enough
    images = []
    for minute in range(60):
        images += names(f"{hour}-{minute:02d}")[0]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("hour", help="YYYY-MM-DD-HH, 24-hour clock, e.g. 2025-05-05-19")
    parser.add_argument("--out", help="output folder (default image_server/retrieved_images/<hour>)")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--url", help="MCP endpoint (default $IMAGE_MCP_URL)")
    args = parser.parse_args()

    if not HOUR_RE.match(args.hour):
        sys.exit(f"hour must look like 2025-05-05-19, got {args.hour!r}")
    load_env()
    url = args.url or os.getenv("IMAGE_MCP_URL", "http://10.0.0.165:10001/mcp")
    if not url:
        sys.exit("set IMAGE_MCP_URL (or pass --url), e.g. http://10.0.0.165:10001/mcp")
    token = os.getenv("MCP_AUTH_TOKEN")
    out = os.path.expanduser(args.out) if args.out else os.path.join(RETRIEVED_DIR, args.hour)

    try:
        client = McpClient(url, token)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the image server at {url}: {e.reason} (is mcp_image_server running?)")

    names = list_hour(client, args.hour)
    if not names:
        sys.exit(f"no images for {args.hour}")
    print(f"{len(names)} images  {names[0]} .. {names[-1]}")

    os.makedirs(out, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(out, n))]
    if len(todo) < len(names):
        print(f"skipping {len(names) - len(todo)} already downloaded")

    lock = threading.Lock()
    pending = iter(todo)
    failed = []
    done = [0]

    def worker():
        worker_client = McpClient(url, token)  # one MCP session per thread
        while True:
            with lock:
                name = next(pending, None)
            if name is None:
                return
            try:
                images = [c for c in worker_client.call("get_image", image_name=name) if c["type"] == "image"]
                if not images:
                    raise RuntimeError("no image returned (deleted since listing?)")
                # write to a temp name first so a crash never leaves a half file that resume would skip
                path = os.path.join(out, name)
                with open(path + ".part", "wb") as f:
                    f.write(base64.b64decode(images[0]["data"]))
                os.replace(path + ".part", path)
            except Exception as e:
                with lock:
                    failed.append((name, str(e)[:200]))
            with lock:
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(todo):
                    print(f"  {done[0]}/{len(todo)}", flush=True)

    start = time.time()
    threads = [threading.Thread(target=worker) for _ in range(max(1, min(args.workers, len(todo))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"saved to {out} in {time.time() - start:.0f}s, {len(failed)} failed")
    for name, error in failed[:10]:
        print(f"  {name}: {error}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
