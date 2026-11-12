import numpy as np
import pytest

from app.modules.detection.base import Detection
from app.modules.tracking.bytetrack import ByteTracker, KalmanFilter, iou_matrix


def person(x: float, conf: float = 0.9, y: float = 100.0) -> Detection:
    """A 40 x 100 person box with its left edge at x."""
    return Detection(x, y, x + 40, y + 100, conf)


def test_iou_identical_disjoint_and_partial():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0
    assert iou_matrix(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)


def test_kalman_learns_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.4, 100.0]))
    for i in range(1, 16):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 5 * i, 100.0, 0.4, 100.0]))
    predicted, _ = kf.predict(mean, cov)
    assert predicted[0] == pytest.approx(100.0 + 5 * 16, abs=1.5)


def test_id_persists_for_moving_person():
    tracker = ByteTracker()
    ids, frames_with_output = set(), 0
    for i in range(20):
        out = tracker.update([person(i * 4)])
        ids |= {t.track_id for t in out}
        frames_with_output += bool(out)
    assert len(ids) == 1
    assert frames_with_output == 19  # frame 0 is tentative, so not reported yet


def test_two_people_keep_separate_ids():
    tracker = ByteTracker()
    left_ids, right_ids = set(), set()
    for i in range(20):
        out = sorted(tracker.update([person(i * 3), person(300 - i * 3)]), key=lambda t: t.x1)
        if len(out) == 2:
            left_ids.add(out[0].track_id)
            right_ids.add(out[1].track_id)
    assert len(left_ids) == 1 and len(right_ids) == 1
    assert left_ids != right_ids


def test_survives_short_occlusion():
    tracker = ByteTracker()
    ids = set()
    for i in range(20):
        present = not (10 <= i <= 14)
        out = tracker.update([person(i * 4)] if present else [])
        if not present:
            assert out == []
        ids |= {t.track_id for t in out}
    assert len(ids) == 1


def test_track_is_deleted_after_max_age():
    tracker = ByteTracker(max_age=3)
    before, after = set(), set()
    for i in range(20):
        out = tracker.update([person(100)] if (i < 5 or i >= 15) else [])
        (before if i < 5 else after).update(t.track_id for t in out)
    assert before and after
    assert before.isdisjoint(after)  # same spot, but the old track was deleted


def test_low_confidence_detections_keep_an_existing_track():
    tracker = ByteTracker()
    ids = set()
    for i in range(8):
        out = tracker.update([person(100, conf=0.9 if i < 5 else 0.3)])
        if i >= 1:
            assert len(out) == 1
        ids |= {t.track_id for t in out}
    assert len(ids) == 1


def test_low_confidence_alone_never_creates_a_track():
    tracker = ByteTracker()
    for _ in range(6):
        assert tracker.update([person(100, conf=0.3)]) == []


def test_single_frame_false_positive_is_never_reported():
    tracker = ByteTracker()
    assert tracker.update([person(100)]) == []
    for _ in range(5):
        assert tracker.update([]) == []


def test_reset_restarts_ids():
    tracker = ByteTracker(min_hits=1)
    assert tracker.update([person(100)])[0].track_id == 1
    tracker.reset()
    assert tracker.update([person(100)])[0].track_id == 1