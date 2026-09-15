from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pose_graph_ssm.data import PreparedData, PreparedSample, PreparationBinding
from pose_graph_ssm.features import FeatureStandardizer
from pose_graph_ssm.graph import adjacency_fingerprint
from pose_graph_ssm.models.selective_ssm import ssm_spec_fingerprint
from pose_graph_ssm.protocol import load_protocol
from pose_graph_ssm.training import architecture_fingerprint, train_validation_arm


PROTOCOL = load_protocol("pose_graph_ssm/protocols/v1-development-preflight.json")


def _features(length: int, offset: float) -> np.ndarray:
    value = np.zeros((length, 25, 15), dtype=np.float64)
    time = np.arange(length, dtype=np.float64)
    value[:, 0, 0] = offset + time * 0.05
    value[:, 1, 3] = offset * 0.1 + time * 0.02
    return value


def _prepared() -> PreparedData:
    standardizer = FeatureStandardizer(
        mean=np.zeros((25, 15), dtype=np.float64),
        std=np.ones((25, 15), dtype=np.float64),
    )
    label_map = tuple((action, index) for index, action in enumerate(PROTOCOL.actions))
    binding = PreparationBinding(
        dataset_content_hash="1" * 64,
        split_hash="2" * 64,
        feature_stats_hash=standardizer.fingerprint,
        label_map=label_map,
        label_map_hash=PreparationBinding.hash_label_map(label_map),
        adjacency_hash=adjacency_fingerprint(),
        ssm_spec_hash=ssm_spec_fingerprint(),
        length_quartiles=(20, 40, 60),
    )
    train = tuple(
        PreparedSample(sample_id=f"train-{index}", label=index % 2, features=_features(6 + index, float(index)))
        for index in range(4)
    )
    validation = (
        PreparedSample(sample_id="validation-0", label=0, features=_features(30, 0.25)),
        PreparedSample(sample_id="validation-1", label=1, features=_features(30, 1.25)),
    )
    return PreparedData(
        binding=binding,
        standardizer=standardizer,
        train_samples=train,
        validation_samples=validation,
        final_test_inventory=(),
    )


def test_same_seed_all_arms_use_identical_sample_order_and_update_budget():
    prepared = _prepared()
    runs = [train_validation_arm(kind, 7, prepared, PROTOCOL) for kind in PROTOCOL.model_kinds]
    assert len({run.sample_order_fingerprint for run in runs}) == 1
    assert len({run.update_count for run in runs}) == 1
    assert runs[0].update_count == len(prepared.train_samples) * PROTOCOL.training.epochs
    assert all(run.update_count <= PROTOCOL.training.max_updates for run in runs)


def test_repeated_same_arm_fixture_training_is_cpu_deterministic():
    prepared = _prepared()
    first = train_validation_arm("gru", 11, prepared, PROTOCOL)
    second = train_validation_arm("gru", 11, prepared, PROTOCOL)
    assert first == second


def test_training_rejects_binding_drift_before_model_construction():
    prepared = _prepared()
    drifted = replace(prepared.binding, adjacency_hash="0" * 64)
    with pytest.raises(ValueError, match="adjacency"):
        train_validation_arm("graph_ssm", 7, replace(prepared, binding=drifted), PROTOCOL)


def test_training_rejects_model_above_shared_parameter_ceiling():
    prepared = _prepared()
    tiny_ceiling = replace(PROTOCOL, parameter_ceiling=1)
    with pytest.raises(ValueError, match="parameter ceiling"):
        train_validation_arm("gru", 7, prepared, tiny_ceiling)


def test_architecture_fingerprints_are_stable_distinct_and_result_bound():
    fingerprints = {kind: architecture_fingerprint(kind, PROTOCOL) for kind in PROTOCOL.model_kinds}
    assert all(len(value) == 64 for value in fingerprints.values())
    assert len(set(fingerprints.values())) == 4
    run = train_validation_arm("ssm_only", 19, _prepared(), PROTOCOL)
    assert run.architecture_fingerprint == fingerprints["ssm_only"]
    assert np.isfinite(run.best_validation_macro_f1)
    assert np.isfinite(run.early_prediction_auc)
    assert np.isfinite(run.retention_auc)
    assert np.isfinite(run.dropout_recovery_rate)
