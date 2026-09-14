import hashlib
import json
from pathlib import Path

from yolo_flywire.hosted_aggregate import aggregate_remote_shards
from yolo_flywire.hosted_extract import hosted_extractor_code_hash
from yolo_flywire.hosted_freeze import HostedExtractionSpec
from yolo_flywire.ntu_io import _ACTION_LABELS, _canonical_json, _hash_json, _split_policy, split_for_subject
from yolo_flywire.pose_extract import _schema
from yolo_flywire.pose_features import PoseFeatureSpec, pose_encoder_hash
from yolo_flywire.remote_indexed_development import load_remote_indexed_pose_development
from yolo_flywire.remote_pose_index import index_remote_development_bundle
from yolo_flywire.remote_rgb import validate_remote_manifest


ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    subjects = {
        split: next(subject for subject in range(1, 107) if split_for_subject(subject) == split)
        for split in ("train", "validation", "final_test")
    }
    rows = []
    for split, subject in subjects.items():
        for action, label in _ACTION_LABELS.items():
            filename = f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi"
            sample_id = filename.removesuffix("_rgb.avi")
            payload = sample_id.encode()
            rows.append({
                "filename": filename, "sample_id": sample_id,
                "setup": 1, "camera": 1, "subject": subject, "repetition": 1,
                "action": action, "label": label, "split": split,
                "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "locator": f"https://example.invalid/{sample_id}",
            })
    return {
        "format_version": 1, "kind": "ntu120_rgb_remote_transport",
        "task_labels": [_ACTION_LABELS[action] for action in sorted(_ACTION_LABELS)],
        "split_policy": _split_policy(), "samples": rows,
    }


def _protocol(remote):
    protocol = json.loads((ROOT / "protocols/v1-hosted-yolo11n-preflight.json").read_text())
    spec = HostedExtractionSpec(
        weights_sha256="a" * 64,
        ultralytics_version="8.4.146",
        torch_version="2.14.0",
        numpy_version="2.4.6",
        av_version="15.1.0",
        opencv_version="5.0.0.93",
    )
    feature = PoseFeatureSpec(**protocol["pose_feature_spec"])
    descriptor = spec.descriptor()
    protocol.update({
        "dataset_content_hash": remote.dataset_content_hash,
        "split_hash": remote.split_hash,
        "input_inventory_hash": remote.input_inventory_hash,
        "source_manifest_hash": remote.source_manifest_hash,
        "yolo_weights_sha256": spec.weights_sha256,
        "observation_schema_hash": _hash_json(_schema()),
        "pose_encoder_hash": pose_encoder_hash(feature),
        "extraction_spec": descriptor,
        "extraction_spec_hash": _hash_json(descriptor),
        "final_test_decoded": False,
        "classifier_evaluated": False,
    })
    return protocol, spec, feature


def _write_sidecar(path: Path, row):
    raw = (_canonical_json(row) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _shard(tmp_path, remote, protocol, spec):
    shard = tmp_path / "shard"
    samples_root = shard / "samples"
    samples_root.mkdir(parents=True)
    records = []
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for source in remote.samples:
        if source["split"] == "final_test":
            records.append({
                "sample_id": source["sample_id"], "split": "final_test",
                "source_size_bytes": source["size_bytes"], "source_sha256": source["sha256"],
                "verified_bytes": True, "decoded": False, "output_sha256": None,
            })
            continue
        directory = samples_root / source["sample_id"]
        directory.mkdir()
        geometry_hash = _write_sidecar(directory / "observations.jsonl", {
            "sample_id": source["sample_id"], "frame_index": 0, "label": source["label"],
            "body_keypoints": points, "detector_confidence": 0.0,
        })
        timing_hash = _write_sidecar(directory / "timing.jsonl", {
            "sample_id": source["sample_id"], "frame_index": 0,
            "pts": 1, "time_base": [1, 30],
        })
        records.append({
            "sample_id": source["sample_id"], "split": source["split"],
            "source_size_bytes": source["size_bytes"], "source_sha256": source["sha256"],
            "verified_bytes": True, "decoded": True, "frame_count": 1,
            "missing_person_frames": 1, "height": 32, "width": 32,
            "output_sha256": {"observations.jsonl": geometry_hash, "timing.jsonl": timing_hash},
        })
    report = {
        "format_version": 1,
        "kind": "ntu120_yolo11_pose_remote_shard",
        "evidence_scope": "hosted_shard_extraction_only",
        "final_test_decoded": False,
        "classifier_evaluated": False,
        "shard_index": 0,
        "shard_count": 1,
        "dataset_content_hash": remote.dataset_content_hash,
        "split_hash": remote.split_hash,
        "input_inventory_hash": remote.input_inventory_hash,
        "source_manifest_hash": remote.source_manifest_hash,
        "transport_manifest_hash": remote.transport_manifest_hash,
        "versions": spec.expected_versions(),
        "python_version": "3.11.0",
        "platform": "test-platform",
        "extractor_code_hash": hosted_extractor_code_hash(),
        "extraction_spec": spec.descriptor(),
        "extraction_spec_hash": protocol["extraction_spec_hash"],
        "observation_schema": _schema(),
        "observation_schema_hash": protocol["observation_schema_hash"],
        "samples": records,
    }
    (shard / "shard-report.json").write_text(_canonical_json(report) + "\n")
    return shard


def test_hosted_aggregate_and_remote_index_need_no_rgb_root(tmp_path):
    manifest = _manifest()
    remote = validate_remote_manifest(manifest)
    protocol, spec, feature = _protocol(remote)
    shard = _shard(tmp_path, remote, protocol, spec)
    bundle = tmp_path / "aggregate"
    report = aggregate_remote_shards(manifest, protocol=protocol, shard_paths=(shard,), output=bundle)
    assert len(report["samples"]) == 20
    assert report["final_test_decoded"] is False
    assert not any("final_test" in row for row in report["samples"])

    manifest_pin = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    indexed = index_remote_development_bundle(
        bundle, remote_manifest=manifest, protocol=protocol,
        expected_manifest_sha256=manifest_pin,
    )
    assert len(indexed.samples) == 20
    source = load_remote_indexed_pose_development(
        bundle,
        remote_manifest=manifest,
        protocol=protocol,
        feature_spec=feature,
        classes=tuple(protocol["task_labels"]),
        expected_manifest_sha256=manifest_pin,
        expected_encoder_hash=protocol["pose_encoder_hash"],
    )
    source.verify(expected_binding_sha256=source.binding_sha256)
    part = source.read_partition(
        split="train", indices=(0, 1), expected_binding_sha256=source.binding_sha256,
    )
    assert part.observations.features.shape == (2, 1, 121)
    assert part.split == "train"
