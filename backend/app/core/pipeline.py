"""Video pipeline: frames -> detector -> tracker -> pose.

Each stage is injected, so any stage can be swapped without touching this file.
Track IDs are temporary labels inside one video; they identify no one.

Detections of class 0 (person) go to the tracker and then to pose. Detections of
any other class are returned as `objects`: context for a human reviewer only. They
never reach the tracker or the pose estimator.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.modules.detection.base import Detection, Detector
from app.modules.pose.base import Pose, PoseEstimator
from app.modules.tracking.base import TrackedBox, Tracker

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PERSON_CLASS = 0


@dataclass(frozen=True)
class TrackedPerson:
    box: TrackedBox
    pose: Pose


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp_s: float
    people: list[TrackedPerson]
    objects: list[Detection] = field(default_factory=list)


class Pipeline:
    def __init__(self, detector: Detector, tracker: Tracker, pose: PoseEstimator) -> None:
        self._detector = detector
        self._tracker = tracker
        self._pose = pose
        self.timings_ms: dict[str, list[float]] = {"detect": [], "track": [], "pose": []}

    def reset(self) -> None:
        """Call between videos so track IDs start fresh."""
        self._tracker.reset()
        self.timings_ms = {"detect": [], "track": [], "pose": []}

    def process(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> FrameResult:
        t0 = time.perf_counter()
        detections = self._detector.detect(frame)
        person_dets = [d for d in detections if d.class_id == PERSON_CLASS]
        objects = [d for d in detections if d.class_id != PERSON_CLASS]
        t1 = time.perf_counter()
        tracked = self._tracker.update(person_dets)
        t2 = time.perf_counter()
        boxes = [(t.x1, t.y1, t.x2, t.y2) for t in tracked]
        poses = self._pose.estimate(frame, boxes) if boxes else []
        t3 = time.perf_counter()

        self.timings_ms["detect"].append((t1 - t0) * 1000)
        self.timings_ms["track"].append((t2 - t1) * 1000)
        self.timings_ms["pose"].append((t3 - t2) * 1000)
        people = [TrackedPerson(b, p) for b, p in zip(tracked, poses, strict=True)]
        return FrameResult(frame_index, timestamp_s, people, objects)


def iter_video(
    path: str | Path, target_fps: float = 5.0, max_seconds: float | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, frame) at roughly target_fps."""
    p = Path(path)
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {p.suffix!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Could not open video (unsupported codec?): {p}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps <= 0:
            logger.warning("Video reports no frame rate; assuming 30 fps")
            src_fps = 30.0
        step = max(1, round(src_fps / target_fps))
        index = 0
        while cap.grab():
            timestamp = index / src_fps
            if max_seconds is not None and timestamp > max_seconds:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, timestamp, frame
            index += 1
    finally:
        cap.release()