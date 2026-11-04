"""Tail P8: a track that carries on after a cat swap, while the whole-visit vote still agrees."""
import datetime

from agent.tracking.period_corrector import PeriodCorrector

T0 = datetime.datetime(2026, 10, 28, 16, 34, 0)


def _pc(labels, step_s=1.0, tracked=26):
    pc = PeriodCorrector.__new__(PeriodCorrector)
    # [time, x1, y1, x2, y2, orig_cat_id, conf, cat_id, period_id]
    pc.detections = [
        [T0 + datetime.timedelta(seconds=i * step_s), 10, 10, 50, 50, lab, 0.7, tracked, 2]
        for i, lab in enumerate(labels)
    ]
    return pc


def test_tail_of_other_cat_is_found():
    pc = _pc([26] * 40 + [26] + [25] * 12)
    other, when, n = pc._tail_disagreement(2, 26)
    assert other == 25 and n == 12
    assert when <= T0 + datetime.timedelta(seconds=41)


def test_no_tail_when_camera_agrees_or_tail_is_short():
    assert _pc([26] * 50)._tail_disagreement(2, 26) is None
    assert _pc([26] * 50 + [25] * 3)._tail_disagreement(2, 26) is None  # too few frames
    assert _pc([26] * 50 + [25] * 6, step_s=0.5)._tail_disagreement(2, 26) is None  # spans < 5 s


def test_scattered_mislabels_are_not_a_tail():
    assert _pc([26, 25] * 30)._tail_disagreement(2, 26) is None


def test_tail_that_is_most_of_the_visit_is_left_to_the_vote_check():
    assert _pc([26] * 5 + [25] * 20)._tail_disagreement(2, 26) is None
