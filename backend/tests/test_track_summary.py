import numpy as np

from app.core.pipeline import FrameResult, TrackedPerson
from app.core.track_summary import TrackSpans
from app.modules.pose.base import Pose
from app.modules.tracking.base import TrackedBox


def result(i: int, ids: list[int]) -> FrameResult:
    pose = Pose(np.zeros((17, 2), np.float32), np.zeros(17, np.float32))
    people = [TrackedPerson(TrackedBox(t, 0, 0, 10, 10, 0.9), pose) for t in ids]
    return FrameResult(i, i * 0.2, people)


def test_spans_count_frames_and_gaps():
    spans = TrackSpans()
    for i, ids in enumerate([[1], [1, 2], [1, 2], [2], [2], [1, 2]]):
        spans.add(result(i, ids))
    rows = {r["track_id"]: r for r in spans.rows()}
    assert rows[1]["frames_seen"] == 4
    assert rows[1]["gaps"] == 1 and rows[1]["longest_gap_frames"] == 2
    assert rows[2]["first_s"] == 0.2 and rows[2]["frames_seen"] == 5 and rows[2]["gaps"] == 0


def test_csv_has_header_and_one_row_per_track(tmp_path):
    spans = TrackSpans()
    spans.add(result(0, [1]))
    spans.add(result(1, [1]))
    path = tmp_path / "tracks.csv"
    spans.to_csv(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("track_id,")
    assert len(lines) == 2