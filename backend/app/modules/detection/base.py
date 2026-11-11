"""Detector interface. The pipeline depends on this, never on a specific model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected person. Coordinates are pixels in the ORIGINAL frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    label: str = "person"


class Detector(ABC):
    """Anything that turns a BGR frame (H x W x 3, uint8) into person boxes."""

    name: str

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...