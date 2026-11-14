"""Run detector -> tracker -> pose over a video, plus phone/book object context.

Saves an annotated copy and a numbers-only track summary (CSV) under data/evidence/.
Both outputs are derived from footage: keep them local and delete them on the
same schedule as the source clip.

Objects (phone, book) are context for a human reviewer. They are not tracked,
get no pose, and say nothing about anyone's behaviour.
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.pipeline import Pipeline, iter_video  # noqa: E402
from app.core.track_summary import TrackSpans  # noqa: E402
from app.modules.detection.yolox_objects import YoloxObjectDetector  # noqa: E402
from app.modules.pose.rtmpose import RtmPoseEstimator  # noqa: E402
from app.modules.tracking.bytetrack import ByteTracker  # noqa: E402

POSE_MODEL = (
    ROOT / "models" / "rtmpose_t" / "20230831" / "rtmpose_onnx"
    / "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504" / "end2end.onnx"
)
# Upper body only (COCO indices 0-10). Hips/knees/ankles are often hidden by desks
# and the model still "guesses" them, so we neither draw nor use them.
UPPER_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
OBJECT_COLORS = {67: (255, 0, 255), 73: (0, 255, 255)}  # BGR: phone magenta, book yellow

logging.basicConfig(level=logging.INFO, format="%(message)s")


def draw(frame, result, threshold):
    for person in result.people:
        b = person.box
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"#{b.track_id}", (int(b.x1), max(int(b.y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        kp, sc = person.pose.keypoints, person.pose.scores
        for a, c in UPPER_EDGES:
            if sc[a] >= threshold and sc[c] >= threshold:
                cv2.line(frame, tuple(map(int, kp[a])), tuple(map(int, kp[c])), (255, 160, 0), 2)
        for k in range(11):
            if sc[k] >= threshold:
                cv2.circle(frame, (int(kp[k][0]), int(kp[k][1])), 4, (0, 0, 255), -1)
    for d in result.objects:
        color = OBJECT_COLORS.get(d.class_id, (255, 255, 255))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (p1[0], max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=5.0, help="frames per second to process")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.3, help="keypoint draw threshold")
    parser.add_argument("--object-threshold", type=float, default=0.25,
                        help="minimum score for phone/book boxes")
    args = parser.parse_args()

    # People use a low threshold on purpose: ByteTrack uses weak detections to keep
    # tracks alive. Objects use their own, higher threshold.
    thresholds = {0: 0.1, 67: args.object_threshold, 73: args.object_threshold}
    detector = YoloxObjectDetector(
        ROOT / "models" / "yolox_tiny" / "yolox_tiny.onnx", thresholds=thresholds
    )
    pipeline = Pipeline(detector, ByteTracker(), RtmPoseEstimator(POSE_MODEL))

    out_dir = ROOT / "data" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    out_path = out_dir / f"tracked_{stem}.mp4"
    csv_path = out_dir / f"tracks_{stem}.csv"

    writer, frames, ids = None, 0, Counter()
    spans = TrackSpans()
    obj_frames: Counter = Counter()
    obj_best: dict[str, float] = {}

    for index, ts, frame in iter_video(args.video, args.fps, args.max_seconds):
        result = pipeline.process(frame, index, ts)
        frames += 1
        ids.update(p.box.track_id for p in result.people)
        spans.add(result)
        obj_frames.update({d.label for d in result.objects})
        for d in result.objects:
            obj_best[d.label] = max(obj_best.get(d.label, 0.0), d.confidence)
        draw(frame, result, args.threshold)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            if not writer.isOpened():
                print("WARNING: could not open the video writer; no output video will be saved")
        if writer.isOpened():
            writer.write(frame)
    if writer is not None:
        writer.release()

    spans.to_csv(csv_path)

    print(f"\nFrames processed: {frames} (target {args.fps} fps)")
    for stage, values in pipeline.timings_ms.items():
        if values:
            print(f"  {stage:7s} avg {sum(values) / len(values):6.1f} ms")
    total = sum(sum(v) for v in pipeline.timings_ms.values())
    if frames:
        print(f"  total   avg {total / frames:6.1f} ms/frame")
    print("Track IDs seen (id: frames):", dict(sorted(ids.items())))
    print("\nTrack summary (first to last second, frames seen, gaps, longest gap in frames):")
    for r in spans.rows():
        print(f"  #{r['track_id']}: {r['first_s']}s to {r['last_s']}s, {r['frames_seen']} frames, "
              f"{r['gaps']} gaps, longest gap {r['longest_gap_frames']}")
    print(f"\nObjects at score >= {args.object_threshold} (frames with a detection / frames, best score):")
    for label in ("cell phone", "book"):
        print(f"  {label:10s} {obj_frames[label]}/{frames} frames, best {obj_best.get(label, 0.0):.2f}")
    print("Saved:", out_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()