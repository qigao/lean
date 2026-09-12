"""Actual PyAV/YOLO-to-four-arm development smoke on generated data only."""
import hashlib
import importlib

from test_pose_backend import offline, _video
from integration.pose_comparison_fixture import graph_inputs

CLASSES = (
    "A8:sitting_down", "A9:standing_up", "A22:cheer_up", "A23:hand_waving",
    "A26:hopping", "A27:jump_up", "A31:pointing", "A34:rub_two_hands_together",
    "A35:nod_head_or_bow", "A36:shake_head",
)


def test_actual_libraries_through_pinned_binding_and_all_four_arms(tmp_path, monkeypatch):
    comparison = importlib.import_module("yolo_flywire.pose_comparison")
    from ultralytics import YOLO
    from yolo_flywire.ntu_io import build_rgb_manifest
    from yolo_flywire.pose_backend import runtime_versions
    from yolo_flywire.pose_extract import ExtractionSpec, extract_development
    from yolo_flywire.pose_development import load_pose_development
    from yolo_flywire.pose_features import PoseFeatureSpec, pose_encoder_hash
    import yolo_flywire.pose_extract as extraction

    weights = tmp_path / "yolo26n-pose.pt"
    model = YOLO("yolo26n-pose.yaml", task="pose")
    assert model.model.yaml["kpt_shape"] == [17, 3]
    model.model.kpt_shape = [17, 3]  # Complete untrained fixture before its byte pin.
    model.save(str(weights))
    versions = runtime_versions()
    extraction_spec = ExtractionSpec(
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        ultralytics_version=versions["ultralytics"], torch_version=versions["torch"],
        numpy_version=versions["numpy"], av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )
    root = tmp_path / "rgb"
    root.mkdir()
    for subject_index, subject in enumerate((1, 14, 3)):
        for index, label in enumerate(CLASSES):
            action = int(label.split(":")[0][1:])
            _video(root / f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi",
                   value=subject_index * 60 + index * 3, frame_count=1 + index % 3)
    inventory = build_rgb_manifest(root)
    actual_decode, decoded = extraction.decode_video, []
    def guarded_decode(path):
        assert "P003" not in path.name, "final-test fixture was decoded"
        decoded.append(path.name)
        yield from actual_decode(path)
    monkeypatch.setattr(extraction, "decode_video", guarded_decode)
    feature_spec = PoseFeatureSpec()
    graphs = graph_inputs(tmp_path / "graphs", seeds=(7,))
    config = comparison.PoseComparisonSpec(seeds=(7,), epochs=2, lr=.01, batch_size=4,
        max_updates=6, parameter_ceiling=10000, gru_hidden_dim=4, graph_node_dim=2)
    reports = []
    for name in ("first", "second"):
        output = tmp_path / name
        extract_development(root, inventory, weights, extraction_spec, output)
        options = dict(root=root, inventory=inventory, extraction_spec=extraction_spec,
            feature_spec=feature_spec, classes=CLASSES,
            expected_manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
            expected_encoder_hash=pose_encoder_hash(feature_spec))
        # Generated fixtures pin after separate preparation, not a real-data freeze.
        prepared = load_pose_development(output, **options)
        assert set(prepared.train.observations.lengths.tolist()) == {1, 2, 3}
        # Indexed reads are checked against the existing eager binding, not used
        # to imply that the trainer itself is already streaming.
        from yolo_flywire.pose_index import index_development_bundle
        from yolo_flywire.pose_features import encode_timed_pose
        from yolo_flywire.pose_batches import collate_pose_features
        import torch
        indexed = index_development_bundle(output, root=root, inventory=inventory,
            spec=extraction_spec, expected_manifest_sha256=options["expected_manifest_sha256"])
        for split in ("train", "validation"):
            part = getattr(prepared, split)
            for order in ((7, 0, 4), (9,)):
                samples = indexed.read_samples(split=split, indices=order)
                assert tuple(sample.sample_id for sample in samples) == tuple(part.sample_ids[i] for i in order)
                batch = collate_pose_features(tuple(encode_timed_pose(
                    sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
                    detector_confidences=sample.detector_confidences, spec=feature_spec,
                ) for sample in samples))
                rows = torch.tensor(order, dtype=torch.int64)
                steps = int(batch.lengths.max().item())
                assert torch.equal(batch.features, part.observations.features.index_select(0, rows)[:, :steps])
                assert torch.equal(batch.lengths, part.observations.lengths.index_select(0, rows))
                assert torch.equal(batch.time_mask, part.observations.time_mask.index_select(0, rows)[:, :steps])
        indexed.verify()
        reports.append(comparison.run_pose_comparison(output, **options, **graphs,
            expected_binding_sha256=prepared.binding_sha256, config=config))
        prepared.verify()
    assert len(decoded) == 40  # Runner/binding read only; no repeated inference.
    assert reports[0] == reports[1]
    assert len(reports[0]["arms"]) == 4
    assert all(row["optimizer_steps"] == 6 for row in reports[0]["arms"])
    assert reports[0]["final_test_evaluated"] is False
    assert reports[0]["topology_claim_evaluated"] is False
    print("Actual-library four-arm smoke: two generated video bundles, pinned input binding, "
          "24 Adam steps per comparison; untrained YOLO and generated graph, NOT recognition evidence")
