"""Per-track time spans from pipeline results. Numbers only; no images, no faces."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.core.pipeline import FrameResult

FIELDS = ["track_id", "first_s", "last_s", "duration_s", "frames_seen", "gaps", "longest_gap_frames"]


@dataclass
class _Span:
    first_s: float
    last_s: float
    last_sample: int
    frames: int = 1
    gaps: int = 0
    longest_gap: int = 0


class TrackSpans:
    """Call add() once per processed frame (including frames with nobody in them)."""

    def __init__(self) -> None:
        self._spans: dict[int, _Span] = {}
        self._sample = -1

    def add(self, result: FrameResult) -> None:
        self._sample += 1
        for person in result.people:
            tid = person.box.track_id
            span = self._spans.get(tid)
            if span is None:
                self._spans[tid] = _Span(result.timestamp_s, result.timestamp_s, self._sample)
                continue
            missed = self._sample - span.last_sample - 1
            if missed > 0:
                span.gaps += 1
                span.longest_gap = max(span.longest_gap, missed)
            span.last_s = result.timestamp_s
            span.last_sample = self._sample
            span.frames += 1

    def rows(self) -> list[dict[str, float | int]]:
        rows = []
        for tid in sorted(self._spans):
            s = self._spans[tid]
            rows.append(
                {
                    "track_id": tid,
                    "first_s": round(s.first_s, 2),
                    "last_s": round(s.last_s, 2),
                    "duration_s": round(s.last_s - s.first_s, 2),
                    "frames_seen": s.frames,
                    "gaps": s.gaps,
                    "longest_gap_frames": s.longest_gap,
                }
            )
        return rows

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows())