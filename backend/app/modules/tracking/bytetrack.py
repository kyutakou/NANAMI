"""ByteTrack-style tracker, written from the paper's description.

Uses only box position, size and motion. No appearance features, no faces,
no re-identification, by design (privacy rule: no identity inference).
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.modules.detection.base import Detection
from app.modules.tracking.base import TrackedBox, Tracker


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection-over-union between every box in a (N,4) and b (M,4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class KalmanFilter:
    """Constant-velocity Kalman filter on [cx, cy, aspect, height] + velocities."""

    def __init__(self) -> None:
        self._wp = 1.0 / 20  # position noise, relative to box height
        self._wv = 1.0 / 160  # velocity noise, relative to box height
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, 4 + i] = 1.0  # position += velocity (1 frame)
        self._H = np.eye(4, 8)  # we only measure position-like values

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(z[3], 1.0)
        mean = np.concatenate([z, np.zeros(4)])
        std = [
            2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
            10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], 1.0)
        std = [
            self._wp * h, self._wp * h, 1e-2, self._wp * h,
            self._wv * h, self._wv * h, 1e-5, self._wv * h,
        ]
        q = np.diag(np.square(std))
        return self._F @ mean, self._F @ cov @ self._F.T + q

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], 1.0)
        r = np.diag(np.square([self._wp * h, self._wp * h, 1e-1, self._wp * h]))
        proj_mean = self._H @ mean
        proj_cov = self._H @ cov @ self._H.T + r
        # Kalman gain K = P H^T S^-1, computed with a solver instead of an inverse.
        gain = np.linalg.solve(proj_cov, self._H @ cov).T
        new_mean = mean + gain @ (z - proj_mean)
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov


class TrackState(Enum):
    TENTATIVE = 1  # seen too few times to report yet
    TRACKED = 2
    LOST = 3  # temporarily unmatched (occluded?)


def _to_xyah(det: Detection) -> np.ndarray:
    w = det.x2 - det.x1
    h = max(det.y2 - det.y1, 1e-3)
    return np.array([det.x1 + w / 2, det.y1 + h / 2, w / h, h], dtype=float)


class _Track:
    def __init__(self, track_id: int, det: Detection, kf: KalmanFilter, min_hits: int) -> None:
        self.id = track_id
        self.kf = kf
        self.min_hits = min_hits
        self.mean, self.cov = kf.initiate(_to_xyah(det))
        self.hits = 1
        self.time_since_update = 0
        self.score = det.confidence
        self.state = TrackState.TRACKED if min_hits <= 1 else TrackState.TENTATIVE

    def predict(self) -> None:
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0.0  # don't extrapolate height growth for unsure tracks
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, _to_xyah(det))
        self.hits += 1
        self.time_since_update = 0
        self.score = det.confidence
        if self.state == TrackState.LOST or (
            self.state == TrackState.TENTATIVE and self.hits >= self.min_hits
        ):
            self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    @property
    def tlbr(self) -> np.ndarray:
        cx, cy, a, h = self.mean[:4]
        h = max(h, 1e-3)
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _iou_cost(tracks: list[_Track], dets: list[Detection]) -> np.ndarray:
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    t = np.array([tr.tlbr for tr in tracks])
    d = np.array([[x.x1, x.y1, x.x2, x.y2] for x in dets])
    return 1.0 - iou_matrix(t, d)


def _associate(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal one-to-one matching; pairs costlier than max_cost are rejected."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    unmatched_r, unmatched_c = set(range(n)), set(range(m))
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            unmatched_r.discard(int(r))
            unmatched_c.discard(int(c))
    return matches, sorted(unmatched_r), sorted(unmatched_c)


class ByteTracker(Tracker):
    name = "bytetrack"

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 2,
        first_match_cost: float = 0.8,
        second_match_cost: float = 0.5,
        tentative_match_cost: float = 0.7,
    ) -> None:
        self._high = high_thresh
        self._low = low_thresh
        self._new = new_track_thresh
        self._max_age = max_age
        self._min_hits = min_hits
        self._cost1 = first_match_cost
        self._cost2 = second_match_cost
        self._cost3 = tentative_match_cost
        self._kf = KalmanFilter()
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedBox]:
        high = [d for d in detections if d.confidence >= self._high]
        low = [d for d in detections if self._low <= d.confidence < self._high]

        tentative = [t for t in self._tracks if t.state == TrackState.TENTATIVE]
        pool = [t for t in self._tracks if t.state != TrackState.TENTATIVE]
        for t in self._tracks:
            t.predict()

        # Stage 1: confident detections vs tracked + lost tracks.
        matches, unmatched_pool, unmatched_high = _associate(_iou_cost(pool, high), self._cost1)
        for ti, di in matches:
            pool[ti].update(high[di])

        # Stage 2: weak detections vs tracks that were tracked but are still unmatched.
        leftover = [pool[i] for i in unmatched_pool if pool[i].state == TrackState.TRACKED]
        matches2, unmatched_left, _ = _associate(_iou_cost(leftover, low), self._cost2)
        for ti, di in matches2:
            leftover[ti].update(low[di])
        for i in unmatched_left:
            leftover[i].mark_lost()

        # Stage 3: tentative tracks vs the confident detections nobody claimed.
        remaining = [high[i] for i in unmatched_high]
        matches3, unmatched_tent, unmatched_rem = _associate(
            _iou_cost(tentative, remaining), self._cost3
        )
        for ti, di in matches3:
            tentative[ti].update(remaining[di])
        dropped = {id(tentative[i]) for i in unmatched_tent}  # never confirmed: discard

        # Clean up, then start new tentative tracks from unclaimed confident detections.
        self._tracks = [
            t
            for t in self._tracks
            if id(t) not in dropped
            and not (t.state == TrackState.LOST and t.time_since_update > self._max_age)
        ]
        for i in unmatched_rem:
            det = remaining[i]
            if det.confidence >= self._new:
                self._tracks.append(_Track(self._next_id, det, self._kf, self._min_hits))
                self._next_id += 1

        return [
            TrackedBox(t.id, *map(float, t.tlbr), t.score)
            for t in self._tracks
            if t.state == TrackState.TRACKED and t.time_since_update == 0
        ]