from __future__ import annotations

from pathlib import Path

import pytest

from pose_graph_ssm.ntu import build_manifest, parse_skeleton_file, select_primary_body, split_for_subject
from pose_graph_ssm.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(ROOT / "protocols" / "v1-development-preflight.json")


def _joint_line(x: float, y: float, z: float, tracking: int = 2) -> str:
    floats = [x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return " ".join(str(v) for v in floats) + f" {tracking}"


def _body_lines(body_id: int, frame: int, *, offset: float = 0.0, tracking: int = 2) -> list[str]:
    lines = [f"{body_id} 0 0 0 0 0 0 0 0 0", "25"]
    for joint in range(25):
        x = offset + frame * 0.1 + joint * 0.01
        y = joint * 0.02
        z = 0.5
        if joint == 20:
            y += 1.0
        lines.append(_joint_line(x, y, z, tracking))
    return lines


def write_skeleton(path: Path, *, body_ids=(10,), frames: int = 4, action: int = 8) -> Path:
    sample = path / f"S001C001P056R001A{action:03d}.skeleton"
    lines = [str(frames)]
    for frame in range(frames):
        lines.append(str(len(body_ids)))
        for index, body_id in enumerate(body_ids):
            lines.extend(_body_lines(body_id, frame, offset=float(index)))
    sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sample


def test_parse_canonical_skeleton_and_select_primary_body(tmp_path):
    sample_path = write_skeleton(tmp_path, body_ids=(20, 10))
    sample = parse_skeleton_file(sample_path)
    assert sample.sample_id == "S001C001P056R001A008"
    assert sample.subject == 56
    assert sample.action == 8
    assert sample.frame_count == 4
    assert len(sample.bodies) == 2
    assert select_primary_body(sample).body_id == 10


def test_subject_split_is_frozen():
    assert split_for_subject(56, PROTOCOL) == "train"
    assert split_for_subject(14, PROTOCOL) == "validation"
    assert split_for_subject(3, PROTOCOL) == "final_test"


def test_parser_rejects_wrong_joint_count(tmp_path):
    path = tmp_path / "S001C001P056R001A008.skeleton"
    path.write_text("1\n1\n10 0 0 0 0 0 0 0 0 0\n24\n", encoding="utf-8")
    with pytest.raises(ValueError, match="25 joints"):
        parse_skeleton_file(path)


def test_manifest_is_byte_backed_and_records_selected_body(tmp_path):
    path = write_skeleton(tmp_path, body_ids=(20, 10))
    manifest = build_manifest(tmp_path, PROTOCOL)
    assert len(manifest["samples"]) == 1
    row = manifest["samples"][0]
    assert row["selected_body_id"] == 10
    assert row["split"] == "train"
    assert row["size_bytes"] == path.stat().st_size
    assert len(row["sha256"]) == 64
    assert len(manifest["dataset_content_hash"]) == 64
    assert len(manifest["split_hash"]) == 64


def test_manifest_rejects_duplicate_content(tmp_path):
    first = write_skeleton(tmp_path, body_ids=(10,), action=8)
    duplicate = tmp_path / "S001C001P056R001A009.skeleton"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="duplicate.*content"):
        build_manifest(tmp_path, PROTOCOL)
