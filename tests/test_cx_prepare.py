from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from yolo_flywire.cx_development import CxDevelopmentBudget
from yolo_flywire.cx_prepare import (
    CX_NORMALIZATION_ID,
    CxPrepareConfig,
    prepare_development_input,
)
from yolo_flywire.cx_protocol import load_cx_protocol
from yolo_flywire.models.cx_lif import CxDynamics


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"


def _joint_line(x: float, y: float, z: float, tracking: int = 2) -> str:
    values = [x, y, z, 10.0, 20.0, 30.0, 40.0, 1.0, 0.0, 0.0, 0.0, tracking]
    return " ".join(str(value) for value in values)


def _body(body_id: int, *, motion: float, zero_joint: int | None = None) -> list[str]:
    lines = [f"{body_id} 0 2 2 2 2 0 0.0 0.0 2", "25"]
    for joint in range(25):
        if joint == 0:
            x, y, z = 0.0, 0.0, 0.0
        elif joint == 20:
            x, y, z = 0.0, 1.0, 0.0
        else:
            x, y, z = 0.01 * joint, 0.02 * joint, 0.0
        if joint == 5:
            x += motion
        tracking = 0 if zero_joint == joint else 2
        lines.append(_joint_line(x, y, z, tracking))
    return lines


def _write_sample(
    root: Path,
    *,
    subject: int,
    action: int = 23,
    motions: tuple[float, ...] = (0.0, 0.1, 0.2),
    zero_joint: int | None = None,
) -> None:
    lines = [str(len(motions))]
    for motion in motions:
        lines.extend(("1", *_body(10, motion=motion, zero_joint=zero_joint)))
    name = f"S001C001P{subject:03d}R001A{action:03d}.skeleton"
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _root(tmp_path: Path, *, validation_motion: float = 5.0) -> Path:
    root = tmp_path / "ntu"
    root.mkdir(parents=True)
    # P056 is development train, P031 is frozen inner validation, P003 is final test.
    _write_sample(root, subject=56, motions=(0.0, 0.1, 0.2))
    _write_sample(root, subject=31, motions=(0.0, validation_motion, 0.0))
    # Deliberately impossible to normalize because joint 24 has no usable sample.
    # prepare must still succeed because sealed final-test coordinates are never normalized.
    _write_sample(root, subject=3, motions=(0.0, 100.0, 0.0), zero_joint=24)
    return root


def _budget() -> CxDevelopmentBudget:
    return CxDevelopmentBudget(
        optimizer="adam",
        learning_rate=1e-3,
        epochs=3,
        max_updates=100,
        batch_size=1,
        early_stopping="best-validation-macro-f1-over-all-epochs",
        retention_horizon=4,
        perturbation_fraction=0.25,
        recovery_horizon=3,
    )


def _config() -> CxPrepareConfig:
    return CxPrepareConfig(
        actions=(23,),
        event_quantile=0.75,
        dynamics=CxDynamics(
            tau_membrane=20.0,
            synaptic_decay=0.8,
            refractory_steps=2,
            threshold=0.5,
            reset=0.0,
            recurrent_delay_steps=1,
            recurrent_gain=0.1,
            magnitude_policy="log1p",
        ),
        budget=_budget(),
    )


def test_prepare_binds_raw_bytes_inner_split_encoder_dynamics_and_graph_pins(tmp_path):
    protocol = load_cx_protocol(PROTOCOL)
    output = tmp_path / "prepared"
    prepared = prepare_development_input(protocol, _config(), _root(tmp_path), output)

    assert prepared.manifest["dataset_content_hash"] == prepared.binding.dataset_content_hash
    assert prepared.development_split_hash == prepared.binding.split_hash
    assert prepared.manifest["split_hash"] != prepared.development_split_hash
    assert prepared.encoder.fingerprint == prepared.binding.event_encoder_hash
    assert prepared.dynamics_hash == prepared.binding.dynamics_hash
    assert prepared.binding.input_node_fingerprint == protocol.input_node_fingerprint
    assert prepared.binding.output_node_fingerprint == protocol.output_node_fingerprint
    assert prepared.binding.real_graph_fingerprint == protocol.cx_artifact_fingerprint
    assert prepared.counts == (("train", 1), ("validation", 1), ("final_test", 1))
    assert prepared.normalization_id == CX_NORMALIZATION_ID
    assert {path.name for path in output.iterdir()} == {
        "binding.json",
        "encoder.json",
        "input_manifest.json",
        "prepare.json",
    }
    with pytest.raises(FileExistsError):
        prepare_development_input(protocol, _config(), _root(tmp_path / "again"), output)


def test_validation_coordinates_do_not_change_training_fitted_encoder(tmp_path):
    protocol = load_cx_protocol(PROTOCOL)
    a_root = _root(tmp_path / "a", validation_motion=5.0)
    b_root = _root(tmp_path / "b", validation_motion=5000.0)
    a = prepare_development_input(protocol, _config(), a_root, tmp_path / "a-out")
    b = prepare_development_input(protocol, _config(), b_root, tmp_path / "b-out")
    assert a.encoder.fingerprint == b.encoder.fingerprint
    assert a.binding.event_encoder_hash == b.binding.event_encoder_hash
    assert a.binding.dataset_content_hash != b.binding.dataset_content_hash


def test_prepare_requires_train_and_validation_but_never_opens_final_test(tmp_path):
    protocol = load_cx_protocol(PROTOCOL)
    root = tmp_path / "only-train"
    root.mkdir()
    _write_sample(root, subject=56)
    with pytest.raises(ValueError, match="validation"):
        prepare_development_input(protocol, _config(), root, tmp_path / "out")
    assert protocol.final_test_enabled is False


def test_prepare_rejects_dynamics_magnitude_policy_drift(tmp_path):
    protocol = load_cx_protocol(PROTOCOL)
    config = _config()
    changed = replace(config, dynamics=replace(config.dynamics, magnitude_policy="raw"))
    with pytest.raises(ValueError, match="magnitude_policy"):
        prepare_development_input(protocol, changed, _root(tmp_path), tmp_path / "out")
