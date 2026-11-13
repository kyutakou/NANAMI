"""Show what the multi-class detector sees in ONE frame (an image, or a time in a video).

Prints the top scores per class (even weak ones) so we can judge whether phones and
books are detectable at all. The saved image is footage-derived: keep it local.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modules.detection.yolox_objects import OBJECT_CLASSES, YoloxObjectDetector  # noqa: E402

COLORS = {0: (0, 255, 0), 67: (255, 0, 255), 73: (0, 255, 255)}  # BGR
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_frame(path: str, at: float):
    p = Path(path)
    if p.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read a frame at {at} s")
        return frame
    frame = cv2.imread(str(p))
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="image or video")
    parser.add_argument("--at", type=float, default=0.0, help="second to grab, for videos")
    parser.add_argument("--min", type=float, default=0.25, help="draw boxes at or above this score")
    args = parser.parse_args()

    frame = load_frame(args.path, args.at)
    print("Frame size (w x h):", frame.shape[1], "x", frame.shape[0])

    # Very low thresholds on purpose, so we can see how weak the phone/book scores are.
    detector = YoloxObjectDetector(
        ROOT / "models" / "yolox_tiny" / "yolox_tiny.onnx",
        thresholds={c: 0.03 for c in OBJECT_CLASSES},
    )
    detector.detect(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(5):
        detections = detector.detect(frame)
    print(f"Average over 5 runs: {(time.perf_counter() - start) / 5 * 1000:.1f} ms")

    for cid, name in OBJECT_CLASSES.items():
        top = sorted((d for d in detections if d.class_id == cid), key=lambda d: -d.confidence)[:5]
        scores = ", ".join(f"{d.confidence:.2f}" for d in top) or "none above 0.03"
        print(f"  {name:10s} top scores: {scores}")

    for d in detections:
        if d.confidence < args.min:
            continue
        color = COLORS.get(d.class_id, (255, 255, 255))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (p1[0], max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_dir = ROOT / "data" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"objects_{Path(args.path).stem}.jpg"
    cv2.imwrite(str(out), frame)
    print("Saved:", out)


if __name__ == "__main__":
    main()