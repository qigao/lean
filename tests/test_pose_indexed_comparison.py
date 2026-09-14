"""Four-arm orchestration must consume the bound indexed development source."""
from __future__ import annotations

import importlib

import pytest

from integration.pose_comparison_fixture import graph_inputs
from test_pose_bundle import extracted
from test_pose_development import _options
from test_pose_indexed_development import _prepare as _indexed_prepare


@pytest.fixture
def indexed_comparison_case(extracted, tmp_path):
    source = _indexed_prepare(extracted)
    options = {
        **_options(extracted),
        **graph_inputs(tmp_path / "graphs"),
        "expected_binding_sha256": source.binding_sha256,
    }
    return extracted[1], options, source


def test_four_arm_runner_uses_bound_indexed_source_without_eager_partitions(
    indexed_comparison_case,
):
    api = importlib.import_module("yolo_flywire.pose_comparison")
    assert not hasattr(api, "load_pose_development")
    bundle, options, source = indexed_comparison_case
    config = api.PoseComparisonSpec(
        seeds=(7, 11), epochs=2, lr=.01, batch_size=4, max_updates=6,
        parameter_ceiling=10000, gru_hidden_dim=4, graph_node_dim=2,
    )
    report = api.run_pose_comparison(bundle, **options, config=config)
    assert report["prepared_binding_sha256"] == source.binding_sha256
    assert len(report["arms"]) == 8
    assert all(row["input_binding_sha256"] == source.binding_sha256 for row in report["arms"])
    source.verify(expected_binding_sha256=source.binding_sha256)
