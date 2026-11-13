"""Multi-class YOLOX-Tiny: people plus a few COCO object classes (phone, book).

Reuses the person detector's preprocessing, grids and NMS; changes nothing in it.

DESIGN RULES
- Objects are context for a human reviewer, never a conclusion about a person.
- Objects must NOT be fed to the person tracker or the pose estimator.
- COCO has no "sheet of paper" class. "book" is only a rough proxy for paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.modules.detection.base import Detection
from app.modules.detection.yolox import YoloxDetector, letterbox, nms

OBJECT_CLASSES = {0: "person", 67: "cell phone", 73: "book"}
DEFAULT_THRESHOLDS = {0: 0.35, 67: 0.25, 73: 0.25}


def postprocess_multiclass(
    pred: np.ndarray,
    grids: np.ndarray,
    strides: np.ndarray,
    ratio: float,
    frame_hw: tuple[int, int],
    decoded: bool,
    thresholds: dict[int, float],
    nms_threshold: float,
) -> list[Detection]:
    """(3549, 85) model output -> Detections for the classes in `thresholds`.

    Each class has its own confidence threshold, and NMS runs per class, so a
    phone lying on a person's box is not suppressed by the person.
    """
    ids = list(thresholds)
    cls_scores = pred[:, 4:5] * pred[:, [5 + c for c in ids]]  # (N, C)
    limits = np.array([thresholds[c] for c in ids], dtype=np.float32)
    rows, cols = np.nonzero(cls_scores >= limits)
    if rows.size == 0:
        return []

    scores = cls_scores[rows, cols]
    sel = pred[rows]
    if decoded:
        xy, wh = sel[:, 0:2], sel[:, 2:4]
    else:
        xy = (sel[:, 0:2] + grids[rows]) * strides[rows]
        wh = np.exp(sel[:, 2:4]) * strides[rows]

    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / ratio
    h, w = frame_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    class_of = np.array(ids)[cols]
    out: list[Detection] = []
    for c in ids:
        mask = class_of == c
        if not mask.any():
            continue
        b, s = boxes[mask], scores[mask]
        for i in nms(b, s, nms_threshold):
            out.append(
                Detection(
                    float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]),
                    float(s[i]), c, OBJECT_CLASSES.get(c, str(c)),
                )
            )
    return out


class YoloxObjectDetector(YoloxDetector):
    name = "yolox-tiny-objects"

    def __init__(
        self,
        model_path: str | Path,
        thresholds: dict[int, float] | None = None,
        nms_threshold: float = 0.45,
        input_size: int = 416,
    ) -> None:
        super().__init__(
            model_path, conf_threshold=0.1, nms_threshold=nms_threshold, input_size=input_size
        )
        self._thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        bad = [c for c in self._thresholds if not 0 <= c < 80]
        if bad:
            raise ValueError(f"Not COCO class ids (0-79): {bad}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 BGR image of shape (H, W, 3)")
        blob, ratio = letterbox(frame, self._size)
        pred = self._session.run(None, {self._input_name: blob})[0][0]
        return postprocess_multiclass(
            pred, self._grids, self._strides, ratio, frame.shape[:2],
            self._decoded, self._thresholds, self._nms,
        )