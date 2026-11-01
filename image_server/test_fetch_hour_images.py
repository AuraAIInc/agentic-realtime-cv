"""Resume behavior for fetch_hour_images: already-downloaded images are skipped,
even when the hour listing grew since the last run and shifted index prefixes.

Run from the image_server/ folder:
    python -m pytest test_fetch_hour_images.py -q
(pytest from the repo root works too.)
"""

import datetime
import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_hour_images", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_hour_images.py")
)
fetch_hour_images = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_hour_images)


def _names(*micros):
    return [f"2026-10-27-18-00-00-{m:06d}.jpg" for m in micros]


def _targets(names, width=4):
    return [(n, fetch_hour_images.prefixed_name(i, width, n)) for i, n in enumerate(names)]


def _layout(tmp_path):
    out = tmp_path / "2026-10-27-18"
    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    return str(out), str(images_dir), str(labels_dir)


def _touch(path, content=b"fake-jpeg"):
    with open(path, "wb") as f:
        f.write(content)


def test_exact_prefixed_files_are_skipped(tmp_path):
    out, images_dir, labels_dir = _layout(tmp_path)
    names = _names(1, 2, 3)
    targets = _targets(names)
    for _, new in targets:
        _touch(os.path.join(images_dir, new))

    todo, already, migrated = fetch_hour_images.plan_downloads(images_dir, labels_dir, out, targets)

    assert todo == []
    assert already == 3
    assert migrated == 0


def test_misindexed_files_are_renamed_not_redownloaded(tmp_path):
    """A rerun after the listing grew shifts indices; existing files (and their
    labels) move to the new indexed name instead of downloading again."""
    out, images_dir, labels_dir = _layout(tmp_path)
    names = _names(1, 2)
    targets = _targets(names)  # 0000_...1.jpg, 0001_...2.jpg
    # last run had an extra earlier frame, so our frames were saved as 0001/0002
    _touch(os.path.join(images_dir, "0001_" + names[0]))
    _touch(os.path.join(images_dir, "0002_" + names[1]))
    _touch(os.path.join(labels_dir, "0001_" + names[0][:-4] + ".txt"), b"0 0.5 0.5 0.1 0.1\n")

    todo, already, migrated = fetch_hour_images.plan_downloads(images_dir, labels_dir, out, targets)

    assert todo == []
    assert migrated == 2
    assert os.path.isfile(os.path.join(images_dir, "0000_" + names[0]))
    assert os.path.isfile(os.path.join(images_dir, "0001_" + names[1]))
    assert not os.path.exists(os.path.join(images_dir, "0002_" + names[1]))
    assert os.path.isfile(os.path.join(labels_dir, "0000_" + names[0][:-4] + ".txt"))
    assert not os.path.exists(os.path.join(labels_dir, "0001_" + names[0][:-4] + ".txt"))


def test_legacy_flat_files_are_migrated(tmp_path):
    out, images_dir, labels_dir = _layout(tmp_path)
    names = _names(1)
    targets = _targets(names)
    _touch(os.path.join(out, names[0]))  # old flat layout, no index prefix

    todo, already, migrated = fetch_hour_images.plan_downloads(images_dir, labels_dir, out, targets)

    assert todo == []
    assert migrated == 1
    assert os.path.isfile(os.path.join(images_dir, targets[0][1]))
    assert not os.path.exists(os.path.join(out, names[0]))


def test_filter_range_keeps_global_indices():
    names = [f"2026-10-28-16-{m:02d}-00-000001.jpg" for m in (32, 33, 45, 59)]
    targets = _targets(names)
    got = fetch_hour_images.filter_range(targets, "2026-10-28-16-33", "2026-10-28-16-59")
    assert [old for old, _ in got] == names[1:]
    # indices still count from the hour's first frame, so the range merges cleanly
    assert [new for _, new in got] == [t[1] for t in targets[1:]]
    assert got[0][1].startswith("0001_")


def test_filter_range_end_minute_fully_included():
    names = ["2026-10-28-16-59-00-000001.jpg", "2026-10-28-16-59-59-999999.jpg"]
    got = fetch_hour_images.filter_range(_targets(names), "2026-10-28-16-00", "2026-10-28-16-59")
    assert [old for old, _ in got] == names


def test_resolve_range_defaults_to_hour():
    hour, start_b, end_b, start_dt, end_dt = fetch_hour_images.resolve_range("2026-10-28-16", None, None)
    assert hour == "2026-10-28-16"
    assert (start_b, end_b) == ("2026-10-28-16", "2026-10-28-16")
    assert (start_dt, end_dt) == (datetime.datetime(2026, 10, 28, 16), datetime.datetime(2026, 10, 28, 17))


def test_resolve_range_derives_hour_from_bounds():
    hour, start_b, end_b, start_dt, end_dt = fetch_hour_images.resolve_range(
        None, "2026-10-28-16-33", "2026-10-28-16-59")
    assert hour == "2026-10-28-16"
    assert (start_b, end_b) == ("2026-10-28-16-33", "2026-10-28-16-59")
    assert (start_dt, end_dt) == (datetime.datetime(2026, 10, 28, 16, 33), datetime.datetime(2026, 10, 28, 17, 0))


def test_resolve_range_rejects_bad_bounds():
    r = fetch_hour_images.resolve_range
    with pytest.raises(ValueError):
        r(None, None, None)  # need an hour or bounds
    with pytest.raises(ValueError):
        r(None, "2026-10-28-16-50", "2026-10-28-17-10")  # spans two hours
    with pytest.raises(ValueError):
        r("2026-10-28-16", "33", None)  # not a full minute stamp
    with pytest.raises(ValueError):
        r("2026-10-28-16", "2026-10-28-16-40", "2026-10-28-16-30")  # end before start


def test_missing_and_partial_files_are_redownloaded(tmp_path):
    out, images_dir, labels_dir = _layout(tmp_path)
    names = _names(1, 2, 3)
    targets = _targets(names)
    _touch(os.path.join(images_dir, targets[0][1]))  # done
    _touch(os.path.join(images_dir, targets[1][1] + ".part"))  # crashed mid-write: retry

    todo, already, migrated = fetch_hour_images.plan_downloads(images_dir, labels_dir, out, targets)

    assert [t[0] for t in todo] == names[1:]
    assert already == 1
    assert migrated == 0
