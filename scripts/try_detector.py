"""Run the YOLOX detector on one image, print timing, save an annotated copy."""
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modules.detection.yolox import YoloxDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(image_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"Could not read image: {image_path}")
    print("Image size (w x h):", frame.shape[1], "x", frame.shape[0])

    detector = YoloxDetector(ROOT / "models" / "yolox_tiny" / "yolox_tiny.onnx")
    detector.detect(frame)  # warm-up run, excluded from timing

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        detections = detector.detect(frame)
    ms = (time.perf_counter() - start) / runs * 1000
    print(f"Average over {runs} runs: {ms:.1f} ms per frame")
    print(f"People found: {len(detections)}")

    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f"{d.confidence:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_dir = ROOT / "data" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"detected_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out), frame)
    print("Saved:", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/try_detector.py <image>")
    main(sys.argv[1])