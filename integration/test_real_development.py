"""Development-only frozen-input orchestration on generated media, NOT recognition evidence."""
from __future__ import annotations

import hashlib
import importlib
import json

from test_pose_backend import _video

CLASSES = (
    "A8:sitting_down", "A9:standing_up", "A22:cheer_up", "A23:hand_waving",
    "A26:hopping", "A27:jump_up", "A31:pointing", "A34:rub_two_hands_together",
    "A35:nod_head_or_bow", "A36:shake_head",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
                    encoding="utf-8")


def test_frozen_real_development_revalidates_bytes_and_never_decodes_final_test(tmp_path, monkeypatch):
    api = importlib.import_module("yolo_flywire.real_development")
    freeze = importlib.import_module("yolo_flywire.real_freeze")
    extraction = importlib.import_module("yolo_flywire.pose_extract")
    comparison = importlib.import_module("yolo_flywire.pose_comparison")
    from ultralytics import YOLO
    from yolo_flywire.ntu_io import build_rgb_manifest
    from yolo_flywire.pose_backend import runtime_versions

    weights = tmp_path / "yolo26n-pose.pt"
    model = YOLO("yolo26n-pose.yaml", task="pose")
    assert model.model.yaml["kpt_shape"] == [17, 3]
    model.model.kpt_shape = [17, 3]
    model.save(str(weights))

    root = tmp_path / "rgb"
    root.mkdir()
    for subject_index, subject in enumerate((1, 14, 3)):
        for index, label in enumerate(CLASSES):
            action = int(label.split(":")[0][1:])
            _video(root / f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi",
                   value=subject_index * 60 + index * 3, frame_count=1 + index % 3)
    inventory = build_rgb_manifest(root)

    template = json.loads((api.REPOSITORY_ROOT / "protocols" / "v0-real-ntu120-preflight.json")
                          .read_text(encoding="utf-8"))
    frozen = freeze.freeze_real_inputs(template, root=root, inventory=inventory, weights=weights)
    protocol = tmp_path / "frozen-protocol.json"
    inventory_path = tmp_path / "inventory.json"
    config_path = tmp_path / "development-config.json"
    controls = tmp_path / "controls.json"
    connectivity = tmp_path / "connectivity.csv"
    _write_json(protocol, frozen)
    _write_json(inventory_path, inventory)
    _write_json(config_path, {
        "seeds": frozen["seeds"], "epochs": 20, "lr": 0.01, "batch_size": 8,
        "max_updates": 40, "parameter_ceiling": 50000,
        "gru_hidden_dim": 4, "graph_node_dim": 2,
    })
    controls.write_text("{}\n", encoding="utf-8")
    connectivity.write_text("generated fixture; comparison is intercepted\n", encoding="utf-8")

    actual_decode, decoded = extraction.decode_video, []
    def guarded_decode(path):
        assert "P003" not in path.name, "final-test fixture was decoded"
        decoded.append(path.name)
        yield from actual_decode(path)
    monkeypatch.setattr(extraction, "decode_video", guarded_decode)

    calls = []
    def fake_comparison(bundle, **kwargs):
        calls.append((bundle, kwargs))
        spec = kwargs["config"]
        assert isinstance(spec, comparison.PoseComparisonSpec)
        assert spec.seeds == tuple(frozen["seeds"])
        assert (spec.epochs, spec.max_updates, spec.parameter_ceiling) == (20, 40, 50000)
        assert kwargs["expected_topology_sha256"] == _sha(protocol)
        assert kwargs["expected_controls_sha256"] == _sha(controls)
        return {
            "format_version": 1,
            "evidence_scope": "development_validation_only",
            "final_test_evaluated": False,
            "topology_claim_evaluated": False,
            "prepared_binding_sha256": kwargs["expected_binding_sha256"],
            "arms": [],
            "paired_validation": [],
        }
    monkeypatch.setattr(api, "run_pose_comparison", fake_comparison)

    report = api.run_real_development(
        protocol, protocol_sha256=_sha(protocol), root=root, inventory=inventory,
        weights=weights, execution_config=json.loads(config_path.read_text(encoding="utf-8")),
        connectivity=connectivity, controls=controls, controls_sha256=_sha(controls),
        output=tmp_path / "run",
    )

    assert len(decoded) == 20
    assert len(calls) == 1
    assert report["evidence_scope"] == "real_development_validation_only"
    assert report["final_test_decoded"] is False
    assert report["final_test_evaluated"] is False
    assert report["classifier_evaluated_on_final_test"] is False
    assert report["protocol_sha256"] == _sha(protocol)
    assert report["controls_sha256"] == _sha(controls)
    assert report["comparison"]["final_test_evaluated"] is False
    assert (tmp_path / "run" / "development_report.json").is_file()


def test_real_development_rejects_execution_budget_or_seed_drift_before_extraction(tmp_path, monkeypatch):
    api = importlib.import_module("yolo_flywire.real_development")
    protocol = json.loads((api.REPOSITORY_ROOT / "protocols" / "v0-real-ntu120-preflight.json")
                          .read_text(encoding="utf-8"))
    protocol.update({
        "dataset_content_hash": "1" * 64, "split_hash": "2" * 64,
        "yolo_weights_sha256": "3" * 64, "observation_schema_hash": "4" * 64,
        "pose_encoder_hash": "5" * 64, "extraction_spec_hash": "6" * 64,
    })
    config = {
        "seeds": [7, 11], "epochs": 20, "lr": 0.01, "batch_size": 8,
        "max_updates": 40, "parameter_ceiling": 50000,
        "gru_hidden_dim": 4, "graph_node_dim": 2,
    }
    monkeypatch.setattr(api, "extract_development", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("drift reached extraction")))
    try:
        api._comparison_spec(protocol, config)
    except ValueError as exc:
        assert "seeds" in str(exc)
    else:
        raise AssertionError("seed drift was accepted")

    config["seeds"] = protocol["seeds"]
    config["max_updates"] = 39
    try:
        api._comparison_spec(protocol, config)
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("budget drift was accepted")


def test_real_development_cli_exposes_no_final_test_option():
    api = importlib.import_module("yolo_flywire.real_development")
    parser = api._build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--final-test" not in option_strings
    assert "--test" not in option_strings
