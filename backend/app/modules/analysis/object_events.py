"""Turn per-frame object detections into sustained, reviewable events.

An event needs the object seen across at least `min_duration_s` (first to last
sighting), and short dropouts up to `max_gap_s` do not break it. Time is measured
in seconds, so behaviour does not depend on the processing frame rate.

An event is context for a human reviewer. It says an object looked present for a
while. It says nothing about what anyone was doing or intending.
Limitation: one signal per label. Two phones at once merge into one event.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.modules.detection.base import Detection

EPS = 1e-6


@dataclass(frozen=True)
class ObjectEvent:
    label: str
    start_s: float
    end_s: float
    frames: int
    max_score: float
    mean_score: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class SustainedObjectEvents:
    """Call update() once per processed frame (also frames with no detections)."""

    def __init__(
        self, label: str = "cell phone", min_duration_s: float = 1.0, max_gap_s: float = 0.5
    ) -> None:
        if min_duration_s < 0 or max_gap_s < 0:
            raise ValueError("durations must be non-negative")
        self._label = label
        self._min = min_duration_s
        self._gap = max_gap_s
        self._reset()

    def _reset(self) -> None:
        self._start: float | None = None
        self._last = 0.0
        self._scores: list[float] = []

    def _close(self) -> ObjectEvent | None:
        if self._start is None:
            return None
        event = None
        if self._last - self._start + EPS >= self._min:
            event = ObjectEvent(
                self._label, self._start, self._last, len(self._scores),
                max(self._scores), sum(self._scores) / len(self._scores),
            )
        self._reset()
        return event

    def update(self, timestamp_s: float, detections: list[Detection]) -> list[ObjectEvent]:
        """Returns events that closed on this frame (usually none)."""
        closed: list[ObjectEvent] = []
        if self._start is not None and timestamp_s - self._last > self._gap + EPS:
            event = self._close()
            if event:
                closed.append(event)
        scores = [d.confidence for d in detections if d.label == self._label]
        if scores:
            if self._start is None:
                self._start = timestamp_s
            self._last = timestamp_s
            self._scores.append(max(scores))
        return closed

    @property
    def is_flagged(self) -> bool:
        """True while a candidate has already lasted long enough (for live drawing)."""
        return self._start is not None and self._last - self._start + EPS >= self._min

    def finish(self) -> list[ObjectEvent]:
        """Call after the last frame to close an event still open at the end."""
        event = self._close()
        return [event] if event else []