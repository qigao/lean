from __future__ import annotations

from pathlib import Path

from pose_graph_ssm.data import prepare_data
from pose_graph_ssm.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(ROOT / "protocols" / "v1-development-preflight.json")


def _joint_line(x: float, y: float, z: float, tracking: int) -> str:
    floats = [x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return " ".join(str(value) for value in floats) + f" {tracking}"


def _write_sample(
    root: Path,
    *,
    subject: int,
    action: int,
    scale: float,
    tracking: int = 2,
    frames: int = 30,
) -> Path:
    path = root / f"S001C001P{subject:03d}R001A{action:03d}.skeleton"
    lines = [str(frames)]
    for frame in range(frames):
        lines.extend(["1", "10 0 0 0 0 0 0 0 0 0", "25"])
        for joint in range(25):
            x = scale * (frame * 0.01 + joint * 0.001)
            y = scale * joint * 0.002
            z = scale * 0.5
            if joint == 20:
                y += scale
            lines.append(_joint_line(x, y, z, tracking))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dataset(root: Path, *, validation_scale: float) -> None:
    train_specs = (
        (56, 8, 10),
        (57, 9, 20),
        (58, 22, 30),
        (59, 23, 40),
        (70, 26, 50),
        (78, 27, 60),
        (80, 31, 70),
        (81, 34, 80),
    )
    for index, (subject, action, frames) in enumerate(train_specs, start=1):
        _write_sample(root, subject=subject, action=action, scale=float(index), frames=frames)
    _write_sample(root, subject=14, action=35, scale=validation_scale, frames=30)
    # Subject 3 is sealed final-test. Every joint is untracked so normalization
    # would fail if final-test coordinates leaked into preparation.
    _write_sample(root, subject=3, action=36, scale=1.0, tracking=0, frames=30)


def test_validation_coordinate_change_does_not_refit_feature_statistics(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _dataset(first_root, validation_scale=1.0)
    _dataset(second_root, validation_scale=1000.0)

    first = prepare_data(first_root, PROTOCOL)
    second = prepare_data(second_root, PROTOCOL)

    assert first.binding.dataset_content_hash != second.binding.dataset_content_hash
    assert first.binding.feature_stats_hash == second.binding.feature_stats_hash
    assert first.standardizer.fingerprint == second.standardizer.fingerprint
    assert len(first.train_samples) == 8
    assert len(first.validation_samples) == 1


def test_sealed_final_test_is_inventory_only_and_never_normalized(tmp_path):
    _dataset(tmp_path, validation_scale=1.0)
    prepared = prepare_data(tmp_path, PROTOCOL)
    assert len(prepared.final_test_inventory) == 1
    row = prepared.final_test_inventory[0]
    assert row["split"] == "final_test"
    assert row["sample_id"].endswith("A036")
    assert "features" not in row
    assert all(sample.sample_id != row["sample_id"] for sample in prepared.train_samples)
    assert all(sample.sample_id != row["sample_id"] for sample in prepared.validation_samples)


def test_preparation_binds_all_train_derived_and_architecture_invariants(tmp_path):
    _dataset(tmp_path, validation_scale=1.0)
    prepared = prepare_data(tmp_path, PROTOCOL)
    assert prepared.binding.feature_stats_hash == prepared.standardizer.fingerprint
    assert len(prepared.binding.dataset_content_hash) == 64
    assert len(prepared.binding.split_hash) == 64
    assert len(prepared.binding.label_map_hash) == 64
    assert len(prepared.binding.adjacency_hash) == 64
    assert len(prepared.binding.ssm_spec_hash) == 64
    assert prepared.binding.length_quartiles == (20, 40, 60)
