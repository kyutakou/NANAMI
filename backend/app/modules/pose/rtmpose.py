"""RTMPose-t (body7, 17 COCO keypoints) top-down estimator on onnxruntime (CPU only).

LICENSING NOTE: this wrapper is our own code. The weights are NOT cleared:
code licenses (MMPose / rtmlib, Apache-2.0) do not automatically cover pretrained
weights, which depend on their training datasets. Non-commercial research
experimentation only, until docs/LICENSING.md says otherwise.

Settings below come from the model's own pipeline.json (padding 1.25, 192x256
input, RGB mean/std, SimCC split ratio 2.0, no flip test, no DARK refinement).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.modules.pose.base import Pose, PoseEstimator

logger = logging.getLogger(__name__)

INPUT_W, INPUT_H = 192, 256
PADDING = 1.25
SPLIT_RATIO = 2.0
NUM_KEYPOINTS = 17
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

Box = tuple[float, float, float, float]


def bbox_to_center_scale(
    box: Box, padding: float = PADDING, aspect: float = INPUT_W / INPUT_H
) -> tuple[np.ndarray, np.ndarray]:
    """Box -> (center, scale) with the scale grown to the model's aspect ratio."""
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float64)


def affine_matrix(
    center: np.ndarray, scale: np.ndarray, out_w: int = INPUT_W, out_h: int = INPUT_H
) -> np.ndarray:
    """2x3 matrix mapping the (center, scale) window of the frame onto the crop."""
    sx, sy = out_w / scale[0], out_h / scale[1]
    return np.array(
        [[sx, 0.0, out_w / 2 - sx * center[0]], [0.0, sy, out_h / 2 - sy * center[1]]],
        dtype=np.float64,
    )


def preprocess_crop(
    frame: np.ndarray, box: Box
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop one person: warp, BGR->RGB, normalise, to CHW float32."""
    center, scale = bbox_to_center_scale(box)
    crop = cv2.warpAffine(
        frame, affine_matrix(center, scale), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR
    )
    rgb = crop[:, :, ::-1].astype(np.float32)
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), center, scale


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = SPLIT_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """(N,K,Wx), (N,K,Wy) -> locations (N,K,2) in crop pixels and scores (N,K)."""
    x_idx, y_idx = simcc_x.argmax(axis=2), simcc_y.argmax(axis=2)
    scores = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
    locs = np.stack([x_idx, y_idx], axis=-1).astype(np.float64) / split_ratio
    return locs, np.maximum(scores, 0.0)


def to_frame_coords(locs: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Crop pixel coordinates (K,2) -> original frame pixel coordinates."""
    out = np.empty_like(locs, dtype=np.float64)
    out[:, 0] = (locs[:, 0] / INPUT_W - 0.5) * scale[0] + center[0]
    out[:, 1] = (locs[:, 1] / INPUT_H - 0.5) * scale[1] + center[1]
    return out


class RtmPoseEstimator(PoseEstimator):
    name = "rtmpose-t"

    def __init__(self, model_path: str | Path, max_batch: int = 8) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        # CPU only, explicitly: never let onnxruntime pick another provider.
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._max_batch = max(1, max_batch)
        self._check_signature()
        logger.info("RTMPose loaded: %s (max batch %d)", path.name, self._max_batch)

    def _check_signature(self) -> None:
        in_shape = list(self._session.get_inputs()[0].shape)
        if in_shape[1:] != [3, INPUT_H, INPUT_W]:
            raise ValueError(f"Unexpected model input shape: {in_shape}")
        outs = {o.name: list(o.shape) for o in self._session.get_outputs()}
        if outs.get("simcc_x", [0])[-1] != int(INPUT_W * SPLIT_RATIO) or outs.get(
            "simcc_y", [0]
        )[-1] != int(INPUT_H * SPLIT_RATIO):
            raise ValueError(f"Unexpected model outputs: {outs}")

    def estimate(self, frame: np.ndarray, boxes: list[Box]) -> list[Pose]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 BGR image of shape (H, W, 3)")
        frame = np.ascontiguousarray(frame)

        empty = Pose(np.zeros((NUM_KEYPOINTS, 2), np.float32), np.zeros(NUM_KEYPOINTS, np.float32))
        poses: list[Pose] = [empty] * len(boxes)
        valid = [i for i, b in enumerate(boxes) if b[2] - b[0] >= 1 and b[3] - b[1] >= 1]

        for start in range(0, len(valid), self._max_batch):
            idx = valid[start : start + self._max_batch]
            prepped = [preprocess_crop(frame, boxes[i]) for i in idx]
            batch = np.stack([p[0] for p in prepped])
            simcc_x, simcc_y = self._session.run(["simcc_x", "simcc_y"], {self._input_name: batch})
            if simcc_x.shape[1] != NUM_KEYPOINTS:
                raise ValueError(f"Expected {NUM_KEYPOINTS} keypoints, got {simcc_x.shape[1]}")
            locs, scores = decode_simcc(simcc_x, simcc_y)
            for j, i in enumerate(idx):
                kp = to_frame_coords(locs[j], prepped[j][1], prepped[j][2])
                poses[i] = Pose(kp.astype(np.float32), scores[j].astype(np.float32))
        return poses