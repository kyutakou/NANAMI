"""YOLOX-Tiny person detector running on onnxruntime (CPU only)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.modules.detection.base import Detection, Detector

logger = logging.getLogger(__name__)

STRIDES = (8, 16, 32)
PAD_VALUE = 114  # grey padding used by YOLOX preprocessing


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad to size x size.

    Returns a (1, 3, size, size) float32 blob (BGR, 0-255) and the scale ratio.
    """
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return blob, ratio


def make_grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell coordinates and stride for each of the candidate boxes."""
    grids, strides = [], []
    for stride in STRIDES:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(
        np.float32
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression: keep the best box, drop heavy overlaps, repeat."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(
    pred: np.ndarray,
    grids: np.ndarray,
    strides: np.ndarray,
    ratio: float,
    frame_hw: tuple[int, int],
    decoded: bool,
    conf_threshold: float,
    nms_threshold: float,
    class_id: int = 0,
) -> list[Detection]:
    """Turn the (3549, 85) model output into person Detections."""
    scores = pred[:, 4] * pred[:, 5 + class_id]  # objectness x class score
    mask = scores >= conf_threshold
    if not mask.any():
        return []

    pred, scores = pred[mask], scores[mask]
    if decoded:
        xy, wh = pred[:, 0:2], pred[:, 2:4]
    else:
        xy = (pred[:, 0:2] + grids[mask]) * strides[mask]
        wh = np.exp(pred[:, 2:4]) * strides[mask]

    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / ratio
    h, w = frame_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    keep = nms(boxes, scores, nms_threshold)
    return [
        Detection(
            float(boxes[i, 0]),
            float(boxes[i, 1]),
            float(boxes[i, 2]),
            float(boxes[i, 3]),
            float(scores[i]),
            class_id,
        )
        for i in keep
    ]


class YoloxDetector(Detector):
    name = "yolox-tiny"

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 416,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        # CPU only, explicitly: never let onnxruntime pick another provider.
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._size = input_size
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._grids, self._strides = make_grids(input_size)
        self._decoded = self._probe_decoded()
        logger.info("YOLOX loaded: %s (decoded output: %s)", path.name, self._decoded)

    def _probe_decoded(self) -> bool:
        """Heuristic: on a blank image, decoded centers reach hundreds of pixels."""
        dummy = np.zeros((1, 3, self._size, self._size), dtype=np.float32)
        out = self._session.run(None, {self._input_name: dummy})[0][0]
        return bool(out[:, 0].max() > 100)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 BGR image of shape (H, W, 3)")
        blob, ratio = letterbox(frame, self._size)
        pred = self._session.run(None, {self._input_name: blob})[0][0]
        return postprocess(
            pred,
            self._grids,
            self._strides,
            ratio,
            frame.shape[:2],
            self._decoded,
            self._conf,
            self._nms,
        )