"""Tracker interface. The pipeline depends on this, never on a specific tracker."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.modules.detection.base import Detection


@dataclass(frozen=True)
class TrackedBox:
    """A person box with a temporary track ID (meaningless outside this video)."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class Tracker(ABC):
    """Turns per-frame detections into boxes with IDs that persist across frames."""

    name: str

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[TrackedBox]:
        """Feed ALL detections of ONE frame (in order). Returns active tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks. Call this between videos."""