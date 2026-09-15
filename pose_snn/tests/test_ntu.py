from pathlib import Path

import numpy as np
import pytest

from pose_temporal_snn.ntu import (
    build_manifest,
    parse_skeleton_file,
    select_primary_body,
    split_for_subject,
)
from pose_temporal_snn.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v1-development-preflight.json"


def _joint_line(x: float, tracking: int = 2) -> str:
    values = [x, 0.1, 0.2, 10.0, 20.0, 30.0, 40.0, 1.0, 0.0, 0.0, 0.0, tracking]
    return " ".join(str(value) for value in values)


def _body(body_id: int, *, offset: float = 0.0, tracked: int = 25) -> list[str]:
    lines = [f"{body_id} 0 2 2 2 2 0 0.0 0.0 2", "25"]
    lines.extend(_joint_line(offset + joint / 100.0, 2 if joint < tracked else 1) for joint in range(25))
    return lines


def _skeleton_text(frames: list[list[list[str]]]) -> str:
    lines = [str(len(frames))]
    for bodies in frames:
        lines.append(str(len(bodies)))
        for body in bodies:
            lines.extend(body)
    return "\n".join(lines) + "\n"


def _write(path: Path, frames: list[list[list[str]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_skeleton_text(frames), encoding="utf-8")
    return path


def test_parse_canonical_skeleton_preserves_joint_tracks(tmp_path):
    path = _write(
        tmp_path / "S001C001P056R001A023.skeleton",
        [[_body(42, offset=0.0)], [_body(42, offset=1.0)]],
    )
    sample = parse_skeleton_file(path)
    assert sample.sample_id == "S001C001P056R001A023"
    assert sample.subject == 56
    assert sample.action == 23
    assert sample.frame_count == 2
    assert len(sample.bodies) == 1
    body = sample.bodies[0]
    assert body.body_id == 42
    assert body.positions.shape == (2, 25, 3)
    assert body.tracking_state.shape == (2, 25)
    assert body.present.tolist() == [True, True]
    assert np.all(body.tracking_state == 2)


def test_primary_body_uses_fully_tracked_count_then_lowest_id(tmp_path):
    path = _write(
        tmp_path / "S001C001P056R001A023.skeleton",
        [
            [_body(20, tracked=20), _body(10, tracked=25)],
            [_body(20, tracked=25), _body(10, tracked=20)],
        ],
    )
    sample = parse_skeleton_file(path)
    assert select_primary_body(sample).body_id == 10


def test_absent_body_frame_does_not_invent_tracking(tmp_path):
    path = _write(
        tmp_path / "S001C001P056R001A023.skeleton",
        [[_body(10)], [_body(20)]],
    )
    sample = parse_skeleton_file(path)
    body10 = next(body for body in sample.bodies if body.body_id == 10)
    assert body10.present.tolist() == [True, False]
    assert np.count_nonzero(body10.positions[1]) == 0
    assert np.count_nonzero(body10.tracking_state[1]) == 0


def test_subject_split_is_frozen_by_protocol():
    protocol = load_protocol(PROTOCOL)
    assert split_for_subject(56, protocol) == "train"
    assert split_for_subject(14, protocol) == "validation"
    assert split_for_subject(3, protocol) == "final_test"
    with pytest.raises(ValueError, match="subject"):
        split_for_subject(0, protocol)


def test_manifest_is_byte_backed_and_deterministic(tmp_path):
    protocol = load_protocol(PROTOCOL)
    _write(tmp_path / "S001C001P056R001A023.skeleton", [[_body(10)], [_body(10, offset=1.0)]])
    _write(tmp_path / "S001C001P014R001A023.skeleton", [[_body(20)], [_body(20, offset=2.0)]])
    _write(tmp_path / "S001C001P003R001A023.skeleton", [[_body(30)], [_body(30, offset=3.0)]])
    first = build_manifest(tmp_path, protocol)
    second = build_manifest(tmp_path, protocol)
    assert first == second
    assert first["kind"] == "pose_temporal_snn_ntu120_skeleton_inventory"
    assert first["body_selection_rule"] == "max-fully-tracked-joints-then-lowest-body-id"
    assert len(first["dataset_content_hash"]) == 64
    assert len(first["split_hash"]) == 64
    assert [row["split"] for row in first["samples"]] == ["final_test", "validation", "train"]
    assert all(len(row["sha256"]) == 64 for row in first["samples"])


def test_parser_rejects_noncanonical_or_malformed_input(tmp_path):
    path = _write(tmp_path / "sample.skeleton", [[_body(1)]])
    with pytest.raises(ValueError, match="filename"):
        parse_skeleton_file(path)

    bad = tmp_path / "S001C001P056R001A023.skeleton"
    bad.write_text("1\n1\n42 0 2 2 2 2 0 0.0 0.0 2\n24\n", encoding="utf-8")
    with pytest.raises(ValueError, match="25 joints"):
        parse_skeleton_file(bad)
