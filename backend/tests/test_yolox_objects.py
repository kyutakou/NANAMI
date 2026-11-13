from pathlib import Path

import numpy as np
import pytest

from app.modules.detection.yolox import make_grids
from app.modules.detection.yolox_objects import (
    YoloxObjectDetector,
    postprocess_multiclass,
)

MODEL = Path(__file__).resolve().parents[2] / "models" / "yolox_tiny" / "yolox_tiny.onnx"
THRESHOLDS = {0: 0.35, 67: 0.25}


def blank():
    grids, strides = make_grids(416)
    return grids, strides, np.zeros((3549, 85), dtype=np.float32)


def put(pred, idx, cls, score):
    pred[idx, 2:4] = np.log(5.0)  # width/height = 5 * stride 8 = 40 px
    pred[idx, 4] = score
    pred[idx, 5 + cls] = 1.0


def run(pred, grids, strides, thresholds=THRESHOLDS):
    return postprocess_multiclass(pred, grids, strides, 1.0, (416, 416), False, thresholds, 0.45)


def test_person_and_phone_are_both_found_with_labels():
    grids, strides, pred = blank()
    put(pred, 10 * 52 + 10, 0, 0.9)  # person, centre (80, 80)
    put(pred, 30 * 52 + 30, 67, 0.8)  # phone, centre (240, 240)
    result = sorted(run(pred, grids, strides), key=lambda d: d.class_id)
    assert [d.label for d in result] == ["person", "cell phone"]
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == pytest.approx(
        (60, 60, 100, 100), abs=0.5
    )
    assert (result[1].x1, result[1].y1, result[1].x2, result[1].y2) == pytest.approx(
        (220, 220, 260, 260), abs=0.5
    )


def test_each_class_has_its_own_threshold():
    grids, strides, pred = blank()
    put(pred, 530, 67, 0.2)  # below the phone threshold (0.25)
    assert run(pred, grids, strides) == []
    assert len(run(pred, grids, strides, {67: 0.15})) == 1


def test_nms_is_per_class():
    grids, strides, pred = blank()
    idx = 530
    put(pred, idx, 0, 0.9)
    pred[idx, 5 + 67] = 1.0  # same box also scores as a phone
    assert {d.label for d in run(pred, grids, strides)} == {"person", "cell phone"}


def test_duplicate_boxes_of_one_class_merge():
    grids, strides, pred = blank()
    put(pred, 530, 67, 0.9)
    put(pred, 531, 67, 0.8)
    assert len(run(pred, grids, strides)) == 1


@pytest.mark.skipif(not MODEL.is_file(), reason="model file not present")
def test_real_model_runs_and_returns_a_list():
    detector = YoloxObjectDetector(MODEL)
    assert isinstance(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)), list)