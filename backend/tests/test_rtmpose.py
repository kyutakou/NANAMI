from pathlib import Path

import numpy as np
import pytest

from app.modules.pose.rtmpose import (
    RtmPoseEstimator,
    affine_matrix,
    bbox_to_center_scale,
    decode_simcc,
    preprocess_crop,
    to_frame_coords,
)

MODEL = (
    Path(__file__).resolve().parents[2]
    / "models" / "rtmpose_t" / "20230831" / "rtmpose_onnx"
    / "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504" / "end2end.onnx"
)


def test_center_scale_pads_and_fixes_aspect():
    center, scale = bbox_to_center_scale((100, 100, 140, 200))
    assert center == pytest.approx((120, 150))
    assert scale == pytest.approx((93.75, 125.0))


def test_wide_box_grows_height_to_match_aspect():
    _, scale = bbox_to_center_scale((0, 0, 200, 100))
    assert scale == pytest.approx((250.0, 250.0 / 0.75))


def test_affine_sends_box_center_to_crop_center():
    center, scale = bbox_to_center_scale((100, 100, 140, 200))
    point = affine_matrix(center, scale) @ np.array([120.0, 150.0, 1.0])
    assert point == pytest.approx((96, 128))


def test_decode_simcc_peaks_and_min_score():
    x, y = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
    x[0, :, 100], y[0, :, 200] = 1.0, 2.0
    locs, scores = decode_simcc(x, y)
    assert locs[0, 0] == pytest.approx((50, 100))
    assert scores[0, 0] == pytest.approx(1.0)


def test_to_frame_coords_round_trip():
    center, scale = bbox_to_center_scale((100, 100, 140, 200))
    locs = np.array([[96, 128], [0, 0], [192, 256]], dtype=float)
    out = to_frame_coords(locs, center, scale)
    assert out == pytest.approx(np.array([[120, 150], [73.125, 87.5], [166.875, 212.5]]))


def test_preprocess_swaps_bgr_to_rgb_and_normalises():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[..., 0] = 255  # pure blue in BGR
    blob, _, _ = preprocess_crop(frame, (300, 200, 340, 300))
    assert blob.shape == (3, 256, 192) and blob.dtype == np.float32
    assert blob[2].mean() == pytest.approx((255 - 103.53) / 57.375, abs=1e-3)
    assert blob[0].mean() == pytest.approx((0 - 123.675) / 58.395, abs=1e-3)


@pytest.mark.skipif(not MODEL.is_file(), reason="model file not present")
def test_real_model_shapes_batching_and_degenerate_box():
    estimator = RtmPoseEstimator(MODEL)
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    box = (200.0, 100.0, 280.0, 300.0)
    poses = estimator.estimate(frame, [box, box, (10.0, 10.0, 10.0, 10.0)])
    assert len(poses) == 3
    assert poses[0].keypoints.shape == (17, 2) and poses[0].scores.shape == (17,)
    np.testing.assert_allclose(poses[0].keypoints, poses[1].keypoints, atol=1e-3)
    assert not poses[2].scores.any()  # degenerate box -> zero pose