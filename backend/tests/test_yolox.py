from pathlib import Path

import numpy as np
import pytest

from app.modules.detection.yolox import (
    YoloxDetector,
    letterbox,
    make_grids,
    nms,
    postprocess,
)

MODEL = Path(__file__).resolve().parents[2] / "models" / "yolox_tiny" / "yolox_tiny.onnx"


def test_grid_count_matches_model_output():
    grids, strides = make_grids(416)
    assert grids.shape == (3549, 2)
    assert strides.shape == (3549, 1)


def test_letterbox_keeps_aspect_ratio_and_pads():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    blob, ratio = letterbox(frame, 416)
    assert blob.shape == (1, 3, 416, 416)
    assert ratio == pytest.approx(0.65)
    assert blob[0, 0, 0, 0] == 0  # image area
    assert blob[0, 0, 300, 0] == 114  # padding area


def test_nms_removes_overlapping_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_postprocess_decodes_raw_output_and_merges_duplicates():
    grids, strides = make_grids(416)
    pred = np.zeros((3549, 85), dtype=np.float32)
    pred[:, 4] = 0.0
    idx = 10 * 52 + 10  # stride-8 grid cell at column 10, row 10 -> center (80, 80)
    for i, score in ((idx, 0.9), (idx + 1, 0.8)):
        pred[i, 2:4] = np.log(5.0)  # width/height = 5 * 8 = 40 px
        pred[i, 4] = score
        pred[i, 5] = 1.0  # person class
    result = postprocess(pred, grids, strides, 1.0, (416, 416), False, 0.35, 0.45)
    assert len(result) == 1
    d = result[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((60, 60, 100, 100), abs=0.5)


@pytest.mark.skipif(not MODEL.is_file(), reason="model file not present")
def test_real_model_runs_on_blank_frame():
    detector = YoloxDetector(MODEL)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []