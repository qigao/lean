from __future__ import annotations

import pytest

from yolo_flywire.graph_topology_corrected import reduce_records


SEEDS = (7, 11, 19, 23, 31)


def _record(seed: int, *, rewired: float, flywire: float) -> dict:
    common = {
        "format_version": 1,
        "kind": "kth_corrected_topology_seed",
        "evidence_scope": "kth_real_development_corrected_topology_only",
        "final_test_evaluated": False,
        "topology_claim_evaluated": False,
        "protocol_sha256": "a" * 64,
        "controls_sha256": "b" * 64,
        "source_manifest_hash": "c" * 64,
        "prepared_binding_sha256": "d" * 64,
        "aggregate_manifest_sha256": "e" * 64,
        "execution_config_hash": "f" * 64,
        "control_identity_hash": "0" * 64,
        "normalization": {"weight_policy": "log1p_incoming_l1", "diagonal_policy": "drop"},
        "architecture": {
            "input_policy": "dense_all_nodes",
            "readout_policy": "flatten",
            "node_dim": 2,
            "parameter_count": 47886,
        },
    }
    return {
        **common,
        "seed": seed,
        "arms": [
            {"seed": seed, "family": "rewired", "optimizer_steps": 40,
             "validation_metrics": {"macro_f1": rewired}},
            {"seed": seed, "family": "flywire", "optimizer_steps": 40,
             "validation_metrics": {"macro_f1": flywire}},
        ],
        "paired_validation": {
            "seed": seed,
            "flywire_macro_f1": flywire,
            "rewired_macro_f1": rewired,
            "difference": flywire - rewired,
        },
    }


def test_corrected_topology_reducer_preserves_exact_five_seed_pairing() -> None:
    records = tuple(
        _record(seed, rewired=0.40 + i * 0.01, flywire=0.43 + i * 0.01)
        for i, seed in enumerate(SEEDS)
    )
    report = reduce_records(records, expected_seeds=SEEDS)
    assert report["seeds"] == list(SEEDS)
    assert report["mean_macro_f1"]["flywire"] == pytest.approx(0.45)
    assert report["mean_macro_f1"]["rewired"] == pytest.approx(0.42)
    assert report["paired_mean_difference"] == pytest.approx(0.03)
    assert report["positive_difference_seeds"] == 5
    assert report["final_test_evaluated"] is False
    assert report["topology_claim_evaluated"] is False


def test_corrected_topology_reducer_rejects_missing_seed() -> None:
    records = tuple(_record(seed, rewired=0.4, flywire=0.5) for seed in SEEDS[:-1])
    with pytest.raises(ValueError, match="seed roster"):
        reduce_records(records, expected_seeds=SEEDS)


def test_corrected_topology_reducer_rejects_wrong_arm_roster() -> None:
    records = [_record(seed, rewired=0.4, flywire=0.5) for seed in SEEDS]
    records[0]["arms"] = records[0]["arms"][:1]
    with pytest.raises(ValueError, match="arm roster"):
        reduce_records(tuple(records), expected_seeds=SEEDS)
