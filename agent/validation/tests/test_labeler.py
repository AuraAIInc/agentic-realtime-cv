"""PeriodLabeler resolves key frames, fetches them, and labels activity."""

import base64
import json
from types import SimpleNamespace

from agent.memory.activity import ActivityLabel
from agent.validation.labeler import PeriodLabeler

MINUTE = "2026-10-28-16-34"
REAL = [f"{MINUTE}-01-879598.jpg", f"{MINUTE}-05-908104.jpg", f"{MINUTE}-09-456481.jpg"]
JPEG = b"\xff\xd8fake-jpeg"


class _FakeHub:
    def __init__(self, listings=None, images=None, fail=()):
        self.listings = listings or {}
        self.images = images or {}
        self.fail = set(fail)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call(self, name, args):
        self.calls.append((name, args))
        if name in self.fail:
            return SimpleNamespace(error="boom", result=None)
        if name == "images__list_images":
            text = json.dumps({"images": self.listings.get(args["prefix"], [])})
            return SimpleNamespace(error=None, result=SimpleNamespace(content=[{"type": "text", "text": text}]))
        raw = self.images.get(args["image_name"])
        if raw is None:
            text = json.dumps({"exists": False})
            return SimpleNamespace(error=None, result=SimpleNamespace(content=[{"type": "text", "text": text}]))
        return SimpleNamespace(
            error=None,
            result=SimpleNamespace(
                content=[{"type": "image", "data": base64.b64encode(raw).decode(), "mimeType": "image/jpeg"}]
            ),
        )


class _FakeLLM:
    def __init__(self, out=None, exc=None):
        self.out = out or {"activity": "resting", "activity_confidence": 0.8, "appearance": "white", "background": "rug"}
        self.exc = exc
        self.calls = []

    def predict_activity(self, **kw):
        self.calls.append(kw)
        if self.exc is not None:
            raise self.exc
        return dict(self.out)


def _cands():
    return [{"name": "resting", "description": "lying still"}, {"name": "walking", "description": "moving"}]


def _run(labeler, jobs):
    labeler.start()
    try:
        for period_id, kwargs in jobs:
            labeler.label_period(period_id, **kwargs)
    finally:
        labeler.stop(timeout=30)


def test_happy_path_resolves_fetches_and_labels():
    hub = _FakeHub(listings={MINUTE: REAL}, images={REAL[0]: JPEG, REAL[2]: JPEG})
    llm = _FakeLLM()
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append((pid, label, names)), llm_client=llm, candidates=_cands(), hub_factory=lambda: hub)
    _run(labeler, [(1, {"cat_id": 25, "cat_name": "Qb", "span": "16:34 to 16:35", "keyframes": [f"{MINUTE}-02-000000.jpg", f"{MINUTE}-08-000000.jpg"], "boxes": [[1, 2, 3, 4]]})])
    assert len(got) == 1
    pid, label, resolved = got[0]
    assert pid == 1 and label.activity == "resting" and label.degraded is False
    assert resolved == [REAL[0], REAL[2]]  # nearest real frames
    assert llm.calls[0]["cat_id"] == 25 and llm.calls[0]["keyframe_names"] == resolved
    assert llm.calls[0]["boxes"] == [[1, 2, 3, 4], None]  # boxes parallel frames
    assert [c[0] for c in hub.calls].count("images__list_images") == 1  # minute cached


def test_unresolvable_names_degrade_without_fetch():
    hub = _FakeHub(listings={MINUTE: REAL}, images={n: JPEG for n in REAL})
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append((pid, label, names)), llm_client=_FakeLLM(), candidates=_cands(), hub_factory=lambda: hub)
    _run(labeler, [(2, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": ["bogus.jpg"], "boxes": []})])
    (pid, label, resolved) = got[0]
    assert label.activity is None and label.degraded is True and resolved == []
    assert hub.calls == []  # bad name: no listing, no fetch


def test_missing_frames_degrade():
    hub = _FakeHub(listings={MINUTE: REAL}, images={})  # listing ok, fetches empty
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append((pid, label, names)), llm_client=_FakeLLM(), candidates=_cands(), hub_factory=lambda: hub)
    _run(labeler, [(3, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg"], "boxes": []})])
    assert got[0][1].degraded is True and got[0][2] == []


def test_llm_failure_degrades_but_keeps_resolved_names():
    hub = _FakeHub(listings={MINUTE: REAL}, images={REAL[0]: JPEG})
    got = []
    labeler = PeriodLabeler(
        sink=lambda pid, label, names: got.append((pid, label, names)),
        llm_client=_FakeLLM(exc=RuntimeError("model down")),
        candidates=_cands(),
        hub_factory=lambda: hub,
    )
    _run(labeler, [(4, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg"], "boxes": []})])
    assert got[0][1].degraded is True and got[0][2] == [REAL[0]]


def test_no_hub_degrades_every_job():
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append((pid, label, names)), llm_client=_FakeLLM(), candidates=_cands(), hub_factory=lambda: None)
    _run(labeler, [(5, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": ["a.jpg"], "boxes": []})])
    assert got[0][1].degraded is True


def test_sink_failure_does_not_stop_later_jobs():
    hub = _FakeHub(listings={MINUTE: REAL}, images={n: JPEG for n in REAL})
    calls = []

    def sink(pid, label, names):
        calls.append(pid)
        if pid == 6:
            raise RuntimeError("sink down")

    labeler = PeriodLabeler(sink=sink, llm_client=_FakeLLM(), candidates=_cands(), hub_factory=lambda: hub)
    _run(
        labeler,
        [
            (6, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg"], "boxes": []}),
            (7, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-08-000000.jpg"], "boxes": []}),
        ],
    )
    assert calls == [6, 7]


def test_max_frames_bounds_fetches():
    hub = _FakeHub(listings={MINUTE: REAL}, images={n: JPEG for n in REAL})
    labeler = PeriodLabeler(sink=lambda *a: None, llm_client=_FakeLLM(), candidates=_cands(), hub_factory=lambda: hub, max_frames=1)
    _run(labeler, [(8, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg", f"{MINUTE}-08-000000.jpg"], "boxes": []})])
    assert [c[0] for c in hub.calls].count("images__get_image") == 1


def test_label_confidence_clamped():
    hub = _FakeHub(listings={MINUTE: REAL}, images={REAL[0]: JPEG})
    llm = _FakeLLM(out={"activity": "walking", "activity_confidence": 9.0, "appearance": "", "background": ""})
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append(label), llm_client=llm, candidates=_cands(), hub_factory=lambda: hub)
    _run(labeler, [(9, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg"], "boxes": []})])
    assert got[0].confidence == 1.0


def test_bad_label_payload_degrades():
    hub = _FakeHub(listings={MINUTE: REAL}, images={REAL[0]: JPEG})
    got = []
    labeler = PeriodLabeler(sink=lambda pid, label, names: got.append(label), llm_client=_FakeLLM(out="nope"), candidates=_cands(), hub_factory=lambda: hub)
    _run(labeler, [(10, {"cat_id": 25, "cat_name": "Qb", "span": "s", "keyframes": [f"{MINUTE}-02-000000.jpg"], "boxes": []})])
    assert got[0].degraded is True and got[0].activity is None


def test_activity_label_defaults():
    assert ActivityLabel(activity=None) == ActivityLabel(activity=None, confidence=None, appearance="", background="", degraded=False)
