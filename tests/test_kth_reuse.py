from pathlib import Path

import pytest

from yolo_flywire.kthreuse import _reduce_seed_records


SEEDS = (7, 11, 19, 23, 31)
FAMILIES = ("gru", "random_graph", "rewired", "flywire")


def _record(seed: int, *, binding: str = "a" * 64) -> dict:
    arms = []
    for index, family in enumerate(FAMILIES):
        arms.append({
            "seed": seed,
            "family": family,
            "input_binding_sha256": binding,
            "optimizer_steps": 40,
            "validation_metrics": {"macro_f1": 0.1 * (index + 1)},
        })
    scores = {arm["family"]: arm["validation_metrics"]["macro_f1"] for arm in arms}
    return {
        "format_version": 1,
        "kind": "kth_real_seed_execution",
        "seed": seed,
        "prepared_binding_sha256": binding,
        "protocol_sha256": "b" * 64,
        "controls_sha256": "c" * 64,
        "config_hash": "d" * 64,
        "aggregate_manifest_sha256": "e" * 64,
        "dataset_content_hash": "f" * 64,
        "split_hash": "1" * 64,
        "source_manifest_hash": "2" * 64,
        "execution": {"device": "cpu", "dtype": "float32"},
        "graph_provenance": {"identity": {"graph": "flywire"}, "node_types": ["a", "b"]},
        "arms": arms,
        "paired_validation": {
            "seed": seed,
            "flywire_macro_f1": scores["flywire"],
            "rewired_macro_f1": scores["rewired"],
            "difference": scores["flywire"] - scores["rewired"],
        },
    }


def test_reduce_seed_records_requires_complete_five_seed_four_arm_roster():
    reduced = _reduce_seed_records(tuple(_record(seed) for seed in SEEDS), expected_seeds=SEEDS)
    assert [row["seed"] for row in reduced["paired_validation"]] == list(SEEDS)
    assert [(row["seed"], row["family"]) for row in reduced["arms"]] == [
        (seed, family) for seed in SEEDS for family in FAMILIES
    ]

    with pytest.raises(ValueError, match="seed roster"):
        _reduce_seed_records(tuple(_record(seed) for seed in SEEDS[:-1]), expected_seeds=SEEDS)


def test_reduce_seed_records_rejects_cross_seed_binding_drift():
    records = [_record(seed) for seed in SEEDS]
    records[-1] = _record(SEEDS[-1], binding="9" * 64)
    with pytest.raises(ValueError, match="prepared binding"):
        _reduce_seed_records(tuple(records), expected_seeds=SEEDS)


def test_reuse_workflow_downloads_cached_run_and_never_redownloads_rgb_or_yolo():
    text = Path(".github/workflows/yolo-flywire-kth-reuse.yml").read_text(encoding="utf-8")
    assert "source_run_id" in text
    assert "SOURCE_RUN_ID" in text
    assert "run-id: ${{ env.SOURCE_RUN_ID }}" in text
    assert "kth-*-${{ env.SOURCE_RUN_ID }}" in text
    assert "matrix:" in text and "seed:" in text
    for seed in SEEDS:
        assert f"- {seed}" in text
    assert "KTH_BASE_URL" not in text
    assert "YOLO11_URL" not in text
    assert "yolo11n-pose.pt" not in text
