import pytest

from app.modules.analysis.object_events import SustainedObjectEvents
from app.modules.detection.base import Detection


def phone(score: float = 0.5, label: str = "cell phone") -> list[Detection]:
    return [Detection(10, 10, 30, 40, score, 67, label)]


def feed(events: SustainedObjectEvents, present: set[int], n: int, fps: float = 5.0):
    closed = []
    for i in range(n):
        closed += events.update(i / fps, phone() if i in present else [])
    return closed + events.finish()


def test_brief_detection_is_not_an_event():
    assert feed(SustainedObjectEvents(), {0, 1}, 10) == []


def test_sustained_detection_becomes_one_event():
    out = feed(SustainedObjectEvents(), set(range(8)), 10)
    assert len(out) == 1
    assert out[0].duration_s == pytest.approx(1.4)
    assert out[0].frames == 8


def test_short_dropout_does_not_split_the_event():
    out = feed(SustainedObjectEvents(), {0, 1, 2, 3, 5, 6, 7}, 12)
    assert len(out) == 1
    assert out[0].duration_s == pytest.approx(1.4)
    assert out[0].frames == 7


def test_long_gap_splits_and_drops_the_short_part():
    out = feed(SustainedObjectEvents(), {0, 1, 2, 3} | set(range(7, 14)), 16)
    assert len(out) == 1  # the first 0.6 s burst is dropped
    assert out[0].start_s == pytest.approx(1.4)


def test_same_behaviour_at_different_frame_rates():
    at5 = feed(SustainedObjectEvents(), set(range(8)), 12, fps=5.0)
    at10 = feed(SustainedObjectEvents(), set(range(15)), 20, fps=10.0)
    assert len(at5) == len(at10) == 1
    assert at5[0].duration_s == pytest.approx(at10[0].duration_s)


def test_is_flagged_only_after_minimum_duration():
    events = SustainedObjectEvents()
    flags = []
    for i in range(8):
        events.update(i / 5, phone())
        flags.append(events.is_flagged)
    assert flags == [False, False, False, False, False, True, True, True]


def test_other_labels_are_ignored():
    events = SustainedObjectEvents()
    for i in range(10):
        events.update(i / 5, phone(label="book"))
    assert events.finish() == []


def test_event_scores():
    events = SustainedObjectEvents()
    for i, s in enumerate([0.3, 0.5, 0.4, 0.6, 0.2, 0.3]):
        events.update(i / 5, phone(s))
    (event,) = events.finish()
    assert event.max_score == pytest.approx(0.6)
    assert event.mean_score == pytest.approx(0.383333, abs=1e-4)