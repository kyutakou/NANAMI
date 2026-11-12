"""Pose estimator interface (top-down: person boxes in, keypoints out).

LICENSING NOTE: the interface and our own wrapper code are independent of any
model weights. Code licenses (e.g. Apache-2.0 for MMPose / rtmlib) do NOT
automatically cover pretrained weights, whose terms can depend on the datasets
they were trained on. Until each checkpoint and its training datasets are
reviewed (see docs/LICENSING.md), pose weights are for NON-COMMERCIAL research
experimentation only.

PRIVACY NOTE: keypoints describe body posture only. They must not be used to
identify anyone, and no output of this module is a conclusion about a person.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class Pose:
    """Keypoints for one person. Coordinates are pixels in the ORIGINAL frame."""

    keypoints: np.ndarray  # shape (17, 2): x, y
    scores: np.ndarray  # shape (17,): per-keypoint confidence


class PoseEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[Pose]:
        """One Pose per box (x1, y1, x2, y2), in the same order as `boxes`."""