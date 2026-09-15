from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yolo_flywire.ntu_skeleton import (
    build_gate1_skeleton_manifest,
    parse_skeleton_file,
    select_primary_body,
    split_for_subject_gate1,
)


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
    path.write_text(_skeleton_text(frames), encoding="utf-8")
    return path


def test_parse_canonical_skeleton_preserves_25_joint_tracks(tmp_path):
    path = _write(
        tmp_path / "S001C001P001R001A023.skeleton",
        [[_body(42, offset=0.0)], [_body(42, offset=1.0)]],
    )
    sample = parse_skeleton_file(path)
    assert sample.sample_id == "S001C001P001R001A023"
    assert sample.subject == 1
    assert sample.action == 23
    assert sample.frame_count == 2
    assert len(sample.bodies) == 1
    body = sample.bodies[0]
    assert body.body_id == 42
    assert body.positions.shape == (2, 25, 3)
    assert body.tracking_state.shape == (2, 25)
    assert body.present.tolist() == [True, True]
    assert np.all(body.tracking_state == 2)
    assert body.tracked_joint_count == 50


def test_primary_body_uses_tracked_joint_count_then_numeric_body_id(tmp_path):
    path = _write(
        tmp_path / "S001C001P001R001A023.skeleton",
        [
            [_body(20, tracked=20), _body(10, tracked=25)],
            [_body(20, tracked=25), _body(10, tracked=20)],
        ],
    )
    sample = parse_skeleton_file(path)
    # Both accumulate 45 fully tracked joints; numeric body id breaks the tie.
    assert select_primary_body(sample).body_id == 10


def test_parser_fills_absent_body_frames_without_inventing_tracking(tmp_path):
    path = _write(
        tmp_path / "S001C001P001R001A023.skeleton",
        [[_body(10)], [_body(20)]],
    )
    sample = parse_skeleton_file(path)
    body10 = next(body for body in sample.bodies if body.body_id == 10)
    assert body10.present.tolist() == [True, False]
    assert np.count_nonzero(body10.positions[1]) == 0
    assert np.count_nonzero(body10.tracking_state[1]) == 0


def test_parser_rejects_noncanonical_name(tmp_path):
    path = _write(tmp_path / "sample.skeleton", [[_body(1)]])
    with pytest.raises(ValueError, match="filename"):
        parse_skeleton_file(path)


def test_parser_rejects_wrong_joint_count(tmp_path):
    lines = ["1", "1", "42 0 2 2 2 2 0 0.0 0.0 2", "24"]
    lines.extend(_joint_line(joint / 100.0) for joint in range(24))
    path = tmp_path / "S001C001P001R001A023.skeleton"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="25 joints"):
        parse_skeleton_file(path)


def test_parser_rejects_nonfinite_joint_and_trailing_records(tmp_path):
    frames = [[_body(42)]]
    text = _skeleton_text(frames).replace("0.0 0.1", "nan 0.1", 1)
    path = tmp_path / "S001C001P001R001A023.skeleton"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        parse_skeleton_file(path)

    path.write_text(_skeleton_text(frames) + "unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trailing"):
        parse_skeleton_file(path)


def test_gate1_xsub_outer_split_is_independent_and_frozen():
    assert split_for_subject_gate1(1) == "train"
    assert split_for_subject_gate1(56) == "train"
    assert split_for_subject_gate1(3) == "final_test"
    assert split_for_subject_gate1(106) == "final_test"
    with pytest.raises(ValueError, match="subject"):
        split_for_subject_gate1(0)


def test_manifest_is_byte_backed_deterministic_and_records_body_selection(tmp_path):
    _write(
        tmp_path / "S001C001P001R001A023.skeleton",
        [[_body(10, offset=0.0)], [_body(10, offset=1.0)]],
    )
    _write(
        tmp_path / "S001C001P003R001A023.skeleton",
        [[_body(30, offset=2.0)], [_body(30, offset=3.0)]],
    )
    first = build_gate1_skeleton_manifest(tmp_path, actions=(23,))
    second = build_gate1_skeleton_manifest(tmp_path, actions=(23,))
    assert first == second
    assert first["kind"] == "cx_gate1_ntu120_skeleton_input_inventory"
    assert first["body_selection_rule"] == "max-fully-tracked-joints-then-lowest-body-id"
    assert len(first["dataset_content_hash"]) == 64
    assert len(first["split_hash"]) == 64
    assert [row["split"] for row in first["samples"]] == ["train", "final_test"]
    assert [row["selected_body_id"] for row in first["samples"]] == [10, 30]
    assert all(len(row["sha256"]) == 64 for row in first["samples"])


def test_manifest_filters_actions_before_parsing_unselected_files(tmp_path):
    _write(tmp_path / "S001C001P001R001A023.skeleton", [[_body(10)]])
    # Malformed but unselected action must not become evidence or break the selected inventory.
    (tmp_path / "S001C001P001R001A024.skeleton").write_text("broken\n", encoding="utf-8")
    manifest = build_gate1_skeleton_manifest(tmp_path, actions=(23,))
    assert [row["action"] for row in manifest["samples"]] == [23]


def test_manifest_rejects_empty_action_selection_and_missing_selected_data(tmp_path):
    with pytest.raises(ValueError, match="actions"):
        build_gate1_skeleton_manifest(tmp_path, actions=())
    with pytest.raises(ValueError, match="no selected"):
        build_gate1_skeleton_manifest(tmp_path, actions=(23,))
