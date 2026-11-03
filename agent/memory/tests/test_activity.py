"""Period activity aggregation: same-cat periods merge across small gaps."""

import datetime

from agent.memory.activity import (
    ActivityLabel,
    PeriodActivityAggregator,
    PeriodMember,
    load_activity_labels,
)

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)


def _member(pid, cat=25, name="Qb", start=0, end=60, det=10, conf=0.8, box=(1, 2, 3, 4), label=None):
    return PeriodMember(
        period_id=pid, cat_id=cat, cat_name=name,
        start=T0 + datetime.timedelta(seconds=start),
        end=T0 + datetime.timedelta(seconds=end),
        detections=det, mean_conf=conf, box=box,
        keyframes=[f"k{pid}-1.jpg", f"k{pid}-2.jpg"],
        label=label,
    )


def _label(activity="resting", conf=0.8):
    return ActivityLabel(activity=activity, confidence=conf, appearance="white cat", background="rug")


def test_labels_yaml_loads_activity_set():
    labels = load_activity_labels()
    names = [a["name"] for a in labels["activity"]]
    assert {"sleeping", "resting", "walking", "eating", "unknown"} <= set(names)
    assert all(a["description"] for a in labels["activity"] if a["name"] != "unknown")


def test_close_periods_combine_far_periods_split():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    assert agg.add(_member(1, start=0, end=60, label=_label())) == []
    assert agg.add(_member(2, start=200, end=260, label=_label())) == []  # gap 140 s: merge
    out = agg.add(_member(3, start=900, end=960, label=_label()))  # gap 640 s: split
    assert len(out) == 1 and out[0].period_ids == [1, 2]
    assert out[0].start == T0 and out[0].end == T0 + datetime.timedelta(seconds=260)
    assert out[0].detections == 20
    assert agg.flush()[0].period_ids == [3]


def test_exact_boundary_gap_splits():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60, label=_label()))
    out = agg.add(_member(2, start=360, end=420, label=_label()))  # gap exactly 300 s
    assert [r.period_ids for r in out] == [[1]]


def test_different_cats_never_merge():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, cat=25, name="Qb", start=0, end=60, label=_label()))
    agg.add(_member(2, cat=26, name="Qbo", start=30, end=90, label=_label("walking")))
    out = sorted(agg.flush(), key=lambda r: r.cat_id)
    assert [(r.cat_id, r.period_ids) for r in out] == [(25, [1]), (26, [2])]


def test_unlabeled_records_wait_for_labels_then_emit():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60))  # no label yet
    assert agg.add(_member(2, start=900, end=960)) == []  # closed but waiting
    out = agg.set_label(1, _label())
    assert [r.period_ids for r in out] == [[1]]
    assert agg.set_label(2, _label()) == []  # still open: not emitted
    assert agg.flush()[0].period_ids == [2]


def test_flush_degrades_unlabeled_members_to_unknown():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60))
    (record,) = agg.flush()
    assert record.ready and record.members[0].label.degraded is True
    assert "unknown activity" in record.narrative()


def test_zero_detection_periods_need_no_label():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60, det=0, conf=None))
    # Period 1 auto-labeled at add, so closing it emits immediately.
    assert [r.period_ids for r in agg.add(_member(2, start=900, end=960, det=0, conf=None, label=_label()))] == [[1]]
    assert [r.period_ids for r in agg.flush()] == [[2]]


def test_narrative_shape_single_and_mixed():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=3540, label=_label("walking", 0.9)))
    (solo,) = agg.flush()
    text = solo.narrative()
    assert text.startswith("Qb (cat 25) from 16:33 to 17:32 at [1, 2, 3, 4]")
    assert "doing walking (confidence 0.90" in text
    assert "Looks like: white cat." in text and "Background: rug." in text

    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60, label=_label("resting")))
    agg.add(_member(2, start=120, end=180, label=_label("walking")))
    (mixed,) = agg.flush()
    text = mixed.narrative()
    assert "(periods 1+2, 20 sightings" in text
    assert "doing resting 16:33–16:34, then walking 16:35–16:36" in text


def test_narrative_handles_missing_cat_box_and_conf():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(
        PeriodMember(
            period_id=None, cat_id=None, cat_name=None,
            start=T0, end=T0 + datetime.timedelta(seconds=30),
            detections=0, mean_conf=None, label=ActivityLabel(activity=None),
        )
    )
    (record,) = agg.flush()
    text = record.narrative()
    assert text.startswith("no cat assigned from 16:33 to 16:33")
    assert " at [" not in text and "unknown activity" in text


def test_weighted_conf_median_box_and_keyframe_bound():
    agg = PeriodActivityAggregator(gap_minutes=5.0, max_keyframes=3)
    agg.add(_member(1, start=0, end=60, det=10, conf=0.6, box=(0, 0, 10, 10), label=_label()))
    agg.add(_member(2, start=120, end=180, det=30, conf=0.8, box=(20, 20, 30, 30), label=_label()))
    (record,) = agg.flush()
    assert record.mean_conf == 0.75
    assert record.box in ((0, 0, 10, 10), (20, 20, 30, 30))
    assert len(record.keyframes) == 3


def test_duplicate_period_feed_ignored():
    agg = PeriodActivityAggregator(gap_minutes=5.0)
    agg.add(_member(1, start=0, end=60, label=_label()))
    agg.add(_member(1, start=0, end=60, label=_label()))
    (record,) = agg.flush()
    assert record.period_ids == [1] and record.detections == 10
