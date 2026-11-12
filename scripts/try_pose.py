"""Detect people, estimate pose, print timing, save an annotated copy."""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modules.detection.yolox import YoloxDetector  # noqa: E402
from app.modules.pose.rtmpose import RtmPoseEstimator  # noqa: E402

POSE_MODEL = (
    ROOT / "models" / "rtmpose_t" / "20230831" / "rtmpose_onnx"
    / "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504" / "end2end.onnx"
)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

logging.basicConfig(level=logging.INFO, format="%(message)s")


def timed(fn, runs=10):
    fn()  # warm-up, excluded
    start = time.perf_counter()
    for _ in range(runs):
        result = fn()
    return result, (time.perf_counter() - start) / runs * 1000


def main(image_path: str, threshold: float, labels: bool) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"Could not read image: {image_path}")

    detector = YoloxDetector(ROOT / "models" / "yolox_tiny" / "yolox_tiny.onnx")
    estimator = RtmPoseEstimator(POSE_MODEL)

    detections, det_ms = timed(lambda: detector.detect(frame))
    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
    poses, pose_ms = timed(lambda: estimator.estimate(frame, boxes))
    print(f"People: {len(boxes)} | detect {det_ms:.1f} ms | pose {pose_ms:.1f} ms")

    for d, pose in zip(detections, poses, strict=True):
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        s = pose.scores
        print(f"  keypoint scores: min {s.min():.2f} median {sorted(s)[8]:.2f} max {s.max():.2f}")
        visible = s >= threshold
        for a, b in EDGES:
            if visible[a] and visible[b]:
                cv2.line(frame, tuple(map(int, pose.keypoints[a])),
                         tuple(map(int, pose.keypoints[b])), (255, 160, 0), 2)
        for k, (x, y) in enumerate(pose.keypoints):
            if visible[k]:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                if labels:
                    cv2.putText(frame, str(k), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 255), 1)

    out_dir = ROOT / "data" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pose_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out), frame)
    print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--labels", action="store_true", help="draw keypoint indices")
    args = parser.parse_args()
    main(args.image, args.threshold, args.labels)