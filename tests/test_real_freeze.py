"""Real-input freeze binds verified bytes and policies; it never executes final-test inference."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from test_ntu_io import _dataset


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {
    "ultralytics": "8.4.146",
    "torch": "2.14.0+cu130",
    "numpy": "2.4.6",
    "av": "15.1.0",
    "opencv-python": "5.0.0.93",
}


def _api():
    return importlib.import_module("yolo_flywire.real_freeze")


def _protocol():
    return json.loads((ROOT / "protocols" / "v0-real-ntu120-preflight.json").read_text(encoding="utf-8"))


def _case(tmp_path):
    ntu = importlib.import_module("yolo_flywire.ntu_io")
    root = _dataset(tmp_path / "rgb")
    inventory = ntu.build_rgb_manifest(root)
    weights = tmp_path / "yolo26n-pose.pt"
    weights.write_bytes(b"trusted local fixture weights\n")
    return root, inventory, weights


def _forbidden(*args, **kwargs):
    raise AssertionError("real-input freeze reached decode, predictor construction, training or comparison")


def test_freeze_binds_verified_dataset_weights_runtime_schema_and_encoder(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    monkeypatch.setattr("yolo_flywire.pose_extract.decode_video", _forbidden)
    monkeypatch.setattr("yolo_flywire.pose_backend.load_predictor", _forbidden)
    monkeypatch.setattr("yolo_flywire.pose_indexed_training.train_indexed_model", _forbidden)
    monkeypatch.setattr("yolo_flywire.pose_comparison.run_pose_comparison", _forbidden)

    frozen = api.freeze_real_inputs(_protocol(), root=root, inventory=inventory, weights=weights)

    from yolo_flywire.ntu_io import _hash_json
    from yolo_flywire.pose_extract import _schema
    from yolo_flywire.pose_features import PoseFeatureSpec, pose_encoder_hash

    assert frozen["dataset_content_hash"] == inventory["dataset_content_hash"]
    assert frozen["split_hash"] == inventory["split_hash"]
    assert frozen["input_inventory_hash"] == _hash_json(inventory)
    assert frozen["ultralytics_package_version"] == "8.4.146"
    assert frozen["yolo_weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert frozen["observation_schema_hash"] == _hash_json(_schema())
    feature_spec = PoseFeatureSpec(**frozen["pose_feature_spec"])
    assert frozen["pose_encoder_hash"] == pose_encoder_hash(feature_spec)
    assert frozen["extraction_spec"]["identity"]["ultralytics_version"] == "8.4.146"
    assert frozen["extraction_spec"]["identity"]["weights_sha256"] == frozen["yolo_weights_sha256"]
    assert frozen["extraction_spec_hash"] == _hash_json(frozen["extraction_spec"])
    assert frozen["final_test_used_for_selection"] is False
    assert frozen["final_test_decoded"] is False
    assert frozen["classifier_evaluated"] is False
    assert frozen["evidence_scope"] == "real_input_frozen_no_final_test_execution"
    assert api.freeze_real_inputs(_protocol(), root=root, inventory=inventory, weights=weights) == frozen


def test_changed_rgb_bytes_or_manifest_are_rejected(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    row = inventory["samples"][0]
    path = root / row["relative_path"]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError):
        api.freeze_real_inputs(_protocol(), root=root, inventory=inventory, weights=weights)


def test_conflicting_existing_real_input_pin_is_rejected_not_repaired(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    protocol = _protocol()
    protocol["dataset_content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="dataset_content_hash"):
        api.freeze_real_inputs(protocol, root=root, inventory=inventory, weights=weights)


@pytest.mark.parametrize("problem", ("wrong-name", "empty", "symlink"))
def test_weights_must_be_explicit_nonempty_local_yolo_checkpoint(tmp_path, monkeypatch, problem):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    if problem == "wrong-name":
        changed = tmp_path / "other.pt"
        changed.write_bytes(weights.read_bytes())
        weights = changed
    elif problem == "empty":
        weights.write_bytes(b"")
    else:
        target = tmp_path / "real-yolo26n-pose.pt"
        target.write_bytes(weights.read_bytes())
        weights.unlink()
        weights.symlink_to(target)
    with pytest.raises(ValueError):
        api.freeze_real_inputs(_protocol(), root=root, inventory=inventory, weights=weights)


def test_wrong_installed_ultralytics_version_fails_closed(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    changed = dict(VERSIONS)
    changed["ultralytics"] = "8.4.145"
    monkeypatch.setattr(api, "runtime_versions", lambda: changed)
    with pytest.raises(ValueError, match="ultralytics"):
        api.freeze_real_inputs(_protocol(), root=root, inventory=inventory, weights=weights)


@pytest.mark.parametrize("feature_spec", (None, {}, {"confidence_threshold": 0.0, "scale_epsilon": 1e-6},
                                                    {"confidence_threshold": 0.05, "scale_epsilon": 0.0}))
def test_pose_feature_policy_must_be_explicit_and_valid(tmp_path, monkeypatch, feature_spec):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    protocol = _protocol()
    protocol["pose_feature_spec"] = feature_spec
    with pytest.raises(ValueError):
        api.freeze_real_inputs(protocol, root=root, inventory=inventory, weights=weights)


def test_freeze_cli_publishes_canonical_snapshot_exclusively(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    protocol_path = ROOT / "protocols" / "v0-real-ntu120-preflight.json"
    output = tmp_path / "frozen" / "protocol.json"
    argv = ["freeze", "--protocol", str(protocol_path), "--root", str(root),
            "--inventory", str(inventory_path), "--weights", str(weights), "--output", str(output)]
    assert api.main(argv) == 0
    raw = output.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert raw == json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        api.main(argv)
    assert exc.value.code != 0
    assert output.read_bytes() == before


def test_freeze_does_not_mutate_caller_protocol_or_inventory(tmp_path, monkeypatch):
    api = _api()
    root, inventory, weights = _case(tmp_path)
    monkeypatch.setattr(api, "runtime_versions", lambda: dict(VERSIONS))
    protocol = _protocol()
    protocol_before, inventory_before = deepcopy(protocol), deepcopy(inventory)
    api.freeze_real_inputs(protocol, root=root, inventory=inventory, weights=weights)
    assert protocol == protocol_before
    assert inventory == inventory_before
