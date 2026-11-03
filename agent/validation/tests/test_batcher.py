"""EscalationBatcher merges temporally close escalations, respects caps, drains cleanly."""

import datetime
import logging
import threading

from agent.escalation import Escalation
from agent.validation.batcher import EscalationBatcher

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)


def _esc(code="T2", seconds=0, end_seconds=None, seq=1, **kw):
    args = {
        "time": T0 + datetime.timedelta(seconds=seconds),
        "end_time": None if end_seconds is None else T0 + datetime.timedelta(seconds=end_seconds),
        "scope": "frame",
        "code": code,
        "reason": f"{code} reason",
        "seq": seq,
    }
    args.update(kw)
    return Escalation(**args)


def _names(batch):
    return [m.name for m in batch.members]


def test_close_new_batcher_take_returns_none():
    b = EscalationBatcher()
    b.close()
    assert b.take() is None


def test_same_frame_escalations_merge():
    b = EscalationBatcher()
    b.put(_esc("T2", seq=6))
    b.put(_esc("T2", seq=7))
    b.put(_esc("T4", seq=8))
    b.close()
    assert _names(b.take()) == ["E6-T2", "E7-T2", "E8-T4"]
    assert b.take() is None


def test_gap_within_window_joins_past_window_splits():
    b = EscalationBatcher(window_s=60.0)
    b.put(_esc("T7", seconds=0, seq=1))
    b.put(_esc("T6", seconds=59, seq=2))  # gap 59 s: joins
    b.put(_esc("T2", seconds=200, seq=3))  # gap 141 s past batch end: splits
    b.close()
    assert _names(b.take()) == ["E1-T7", "E2-T6"]
    assert _names(b.take()) == ["E3-T2"]
    assert b.take() is None


def test_exact_boundary_gap_joins():
    b = EscalationBatcher(window_s=60.0)
    b.put(_esc("T7", seconds=0, seq=1))
    b.put(_esc("T6", seconds=60, seq=2))
    b.close()
    assert _names(b.take()) == ["E1-T7", "E2-T6"]


def test_overlapping_period_span_joins():
    b = EscalationBatcher(window_s=60.0)
    b.put(_esc("P6", seconds=-100, end_seconds=10, seq=9))  # span overlaps the point below
    b.put(_esc("T7", seconds=0, seq=10))
    b.close()
    assert _names(b.take()) == ["E9-P6", "E10-T7"]


def test_size_cap_splits_batch():
    b = EscalationBatcher(max_batch=2)
    b.put(_esc("T2", seq=6))
    b.put(_esc("T2", seq=7))
    b.put(_esc("T4", seq=8))
    b.close()
    assert _names(b.take()) == ["E6-T2", "E7-T2"]
    assert _names(b.take()) == ["E8-T4"]


def test_char_budget_splits_batch():
    b = EscalationBatcher(max_chars=30, size_of=lambda esc: 10)
    b.put(_esc("T2", seq=6))  # 10 + 120 overhead alone exceeds the budget...
    b.put(_esc("T2", seq=7))
    b.close()
    # ...so nothing ever joins: the first batch holds one, the second holds one.
    assert _names(b.take()) == ["E6-T2"]
    assert _names(b.take()) == ["E7-T2"]


def test_char_budget_counts_when_it_fits():
    sizes = {"E6-T2": 10, "E7-T2": 10, "E8-T4": 5000}
    b = EscalationBatcher(max_chars=500, size_of=lambda esc: sizes[esc.name])
    b.put(_esc("T2", seq=6))
    b.put(_esc("T2", seq=7))  # 130 + 130 fits
    b.put(_esc("T4", seq=8))  # + 5120 does not: splits
    b.close()
    assert _names(b.take()) == ["E6-T2", "E7-T2"]
    assert _names(b.take()) == ["E8-T4"]


def test_broken_size_hook_falls_back_to_zero():
    def boom(esc):
        raise RuntimeError("prompt build failed")

    b = EscalationBatcher(size_of=boom)
    b.put(_esc("T2", seq=6))
    b.put(_esc("T2", seq=7))
    b.close()
    assert _names(b.take()) == ["E6-T2", "E7-T2"]


def test_in_flight_batch_gains_no_members():
    b = EscalationBatcher()
    b.put(_esc("T2", seq=6))
    first = b.take()  # worker took it: in flight
    b.put(_esc("T2", seq=7))  # same frame, but the batch is gone
    b.close()
    assert _names(first) == ["E6-T2"]
    assert _names(b.take()) == ["E7-T2"]


def test_put_after_close_drops_with_warning(caplog):
    b = EscalationBatcher()
    b.put(_esc("T2", seq=6))
    b.close()
    with caplog.at_level(logging.WARNING, logger="agent.validation.batcher"):
        b.put(_esc("T2", seq=7))
    assert "after queue close" in caplog.text
    assert _names(b.take()) == ["E6-T2"]
    assert b.take() is None


def test_batch_span_extends_with_members():
    b = EscalationBatcher(window_s=60.0)
    b.put(_esc("P6", seconds=0, end_seconds=100, seq=9))
    # 150 s past the batch start but only 50 s past its end: joins via the span.
    b.put(_esc("T7", seconds=150, seq=10))
    b.close()
    batch = b.take()
    assert _names(batch) == ["E9-P6", "E10-T7"]
    assert (batch.start, batch.end) == (T0, T0 + datetime.timedelta(seconds=150))


def test_producer_consumer_threads_drain_all():
    b = EscalationBatcher(window_s=60.0, max_batch=4)
    escs = [_esc("T2", seconds=i * 10, seq=i + 1) for i in range(10)]
    taken = []

    def produce():
        for esc in escs:
            b.put(esc)
        b.close()

    def consume():
        while (batch := b.take()) is not None:
            taken.extend(batch.members)

    producer = threading.Thread(target=produce)
    consumer = threading.Thread(target=consume)
    producer.start()
    consumer.start()
    producer.join(timeout=10)
    consumer.join(timeout=10)
    assert [m.seq for m in taken] == list(range(1, 11))


def test_merge_logs_names_and_gap(caplog):
    b = EscalationBatcher()
    b.put(_esc("T2", seq=6))
    with caplog.at_level(logging.INFO, logger="agent.validation.batcher"):
        b.put(_esc("T2", seconds=5, seq=7))
    assert "merged E7-T2 into batch [E6-T2] (gap 5.0 s)" in caplog.text
