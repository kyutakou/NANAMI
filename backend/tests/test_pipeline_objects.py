import numpy as np

from app.core.pipeline import Pipeline
from app.modules.detection.base import Detection, Detector
from app.modules.pose.base import Pose, PoseEstimator
from app.modules.tracking.bytetrack import ByteTracker

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class MixedDetector(Detector):
    name = "mixed"

    def detect(self, frame):
        return [
            Detection(100, 100, 140, 200, 0.9, 0, "person"),
            Detection(300, 300, 320, 330, 0.8, 67, "cell phone"),
        ]


class SpyTracker(ByteTracker):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[Detection]] = []

    def update(self, detections):
        self.seen.append(list(detections))
        return super().update(detections)


class SpyPose(PoseEstimator):
    name = "spy"

    def __init__(self) -> None:
        self.box_counts: list[int] = []

    def estimate(self, frame, boxes):
        self.box_counts.append(len(boxes))
        return [Pose(np.zeros((17, 2), np.float32), np.ones(17, np.float32)) for _ in boxes]


def test_objects_are_returned_but_never_tracked_or_posed():
    tracker, pose = SpyTracker(), SpyPose()
    pipeline = Pipeline(MixedDetector(), tracker, pose)
    results = [pipeline.process(FRAME, i, i / 5) for i in range(4)]

    for seen in tracker.seen:
        assert [d.class_id for d in seen] == [0]  # only the person reaches the tracker
    assert set(pose.box_counts) <= {1}  # pose only ever receives the person's box
    for r in results:
        assert [d.label for d in r.objects] == ["cell phone"]
    assert all(len(r.people) <= 1 for r in results)
    assert results[-1].people[0].box.track_id == 1


def test_frame_result_objects_default_is_empty():
    from app.core.pipeline import FrameResult

    assert FrameResult(0, 0.0, []).objects == []