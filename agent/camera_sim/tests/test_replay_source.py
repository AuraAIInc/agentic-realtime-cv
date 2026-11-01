"""scan_images reads both retrieved_images layouts: flat <stamp>.jpg files and
the fetch_hour_images layout <hour>/images/NNNN_<stamp>.jpg.

Run from the repo root:
    python -m pytest agent/camera_sim/tests/test_replay_source.py -q
"""

import datetime
import os

from agent.camera_sim.replay_source import parse_stamp, scan_images


def test_parse_stamp_accepts_indexed_and_plain_names():
    assert parse_stamp("0000_2026-10-27-18-00-00-327702.jpg") == datetime.datetime(2026, 10, 27, 18, 0, 0, 327702)
    assert parse_stamp("2026-10-27-18-00-00-327702.jpg") == datetime.datetime(2026, 10, 27, 18, 0, 0, 327702)
    assert parse_stamp("detections.png") is None
    assert parse_stamp("0000_2026-10-27-18-00-00-327702.jpg.part") is None
    assert parse_stamp("0000_2026-10-27-18-00-00-327702.txt") is None


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


def _indexed_hour(root):
    hour = os.path.join(root, "2026-10-27-18")
    _touch(os.path.join(hour, "images", "0000_2026-10-27-18-00-00-000001.jpg"))
    _touch(os.path.join(hour, "images", "0001_2026-10-27-18-00-01-000002.jpg"))
    _touch(os.path.join(hour, "labels", "0000_2026-10-27-18-00-00-000001.txt"))
    _touch(os.path.join(hour, "detections.png"))
    return hour


def test_scan_images_finds_indexed_layout(tmp_path):
    hour = _indexed_hour(str(tmp_path))

    found = scan_images(str(tmp_path))

    assert [p for _, p in found] == [
        os.path.join(hour, "images", "0000_2026-10-27-18-00-00-000001.jpg"),
        os.path.join(hour, "images", "0001_2026-10-27-18-00-01-000002.jpg"),
    ]
    assert [t for t, _ in found] == sorted(t for t, _ in found)


def test_scan_images_bounds_match_stamp_not_index(tmp_path):
    _indexed_hour(str(tmp_path))

    assert len(scan_images(str(tmp_path), start="2026-10-27-18-00-01")) == 1
    assert len(scan_images(str(tmp_path), end="2026-10-27-18-00-00-000001.jpg")) == 1
    # an indexed full name works as a bound too
    assert len(scan_images(str(tmp_path), start="0001_2026-10-27-18-00-01-000002.jpg")) == 1


def test_scan_images_still_reads_flat_layout(tmp_path):
    flat = os.path.join(str(tmp_path), "2026-10-26-02")
    _touch(os.path.join(flat, "2026-10-26-02-00-00-199755.jpg"))

    found = scan_images(str(tmp_path))

    assert [p for _, p in found] == [os.path.join(flat, "2026-10-26-02-00-00-199755.jpg")]
