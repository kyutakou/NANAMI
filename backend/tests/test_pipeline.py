import cv2
import numpy as np
import pytest

from app.core.pipeline import Pipeline, iter_video
from app.modules.detection.base import Detection, Detector
from app.modules.pose.base import Pose, PoseEstimator
from app.modules.tracking.bytetrack import ByteTracker


class FakeDetector(Detector):
    name = "fake"

    def __init__(self, present: bool = True) -> None:
        self.present = present

    def detect(self, frame):
        return [Detection(100, 100, 140, 200, 0.9)] if self.present else []


class FakePose(PoseEstimator):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, frame, boxes):
        self.calls += 1
        return [Pose(np.zeros((17, 2), np.float32), np.ones(17, np.float32)) for _ in boxes]


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_pipeline_gives_one_pose_per_tracked_person():
    pose = FakePose()
    pipeline = Pipeline(FakeDetector(), ByteTracker(), pose)
    results = [pipeline.process(FRAME, i, i / 5) for i in range(5)]
    assert results[0].people == []  # first sighting is tentative
    for r in results[1:]:
        assert len(r.people) == 1 and r.people[0].box.track_id == 1
        assert r.people[0].pose.keypoints.shape == (17, 2)
    assert len(pipeline.timings_ms["detect"]) == 5


def test_pose_not_called_when_nobody_is_tracked():
    pose = FakePose()
    pipeline = Pipeline(FakeDetector(present=False), ByteTracker(), pose)
    for i in range(3):
        assert pipeline.process(FRAME, i, i).people == []
    assert pose.calls == 0


def test_reset_restarts_track_ids():
    pipeline = Pipeline(FakeDetector(), ByteTracker(), FakePose())
    for i in range(3):
        pipeline.process(FRAME, i, i)
    pipeline.reset()
    ids = {p.box.track_id for i in range(3) for p in pipeline.process(FRAME, i, i).people}
    assert ids == {1}


def test_iter_video_samples_frames(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("cannot write test video on this machine")
    for _ in range(30):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    frames = list(iter_video(path, target_fps=5.0))
    assert [f[0] for f in frames] == [0, 6, 12, 18, 24]
    assert frames[1][1] == pytest.approx(0.2)


def test_iter_video_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        list(iter_video(tmp_path / "notes.txt"))
    with pytest.raises(FileNotFoundError):
        list(iter_video(tmp_path / "missing.mp4"))