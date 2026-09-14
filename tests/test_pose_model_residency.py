"""Model lifetime and indexed/eager equivalence on generated development fixtures only."""
from dataclasses import asdict
import gc
import weakref

import pytest
import torch

from test_pose_comparison import comparison_case, extracted, _api, _run, _spec, FAMILIES
from integration.pose_comparison_fixture import graph_inputs
from yolo_flywire.pose_development import load_pose_development
from yolo_flywire.pose_training import evaluate_padded, train_padded_model


@pytest.mark.parametrize("seeds", ((7,), (7, 11), (7, 11, 19, 23, 31)))
def test_only_one_model_is_resident_independent_of_seed_count(comparison_case, tmp_path, monkeypatch, seeds):
    api = _api()
    model_refs, tensor_refs, samples, per_model_bytes = [], [], [], []

    def observe(original):
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            model_refs.append(weakref.ref(self))
            tensors = tuple(self.parameters()) + tuple(self.buffers())
            per_model_bytes.append(sum(t.numel() * t.element_size() for t in tensors))
            tensor_refs.extend((weakref.ref(t), t.numel() * t.element_size()) for t in tensors)
            # Ignore unreachable cycles; count reachable models/tensors, not RSS or allocator caches.
            del tensors
            gc.collect()
            samples.append((sum(ref() is not None for ref in model_refs),
                            sum(size for ref, size in tensor_refs if ref() is not None)))
        return initialize

    for kind in (api.GRUClassifier, api.GraphRecurrentClassifier):
        monkeypatch.setattr(kind, "__init__", observe(kind.__init__))
    assets = graph_inputs(tmp_path / "residency-graphs", seeds=seeds)
    report = _run(comparison_case, config=_spec(seeds=seeds), **assets)
    assert len(report["arms"]) == 4 * len(seeds)
    assert max(count for count, _ in samples) == 1, f"initialized-model residency samples: {samples}"
    assert max(size for _, size in samples) <= max(per_model_bytes)
    gc.collect()
    assert all(ref() is None for ref in model_refs)
    assert all(ref() is None for ref, _ in tensor_refs)
    print(f"Model residency: seeds={len(seeds)} peak_models=1 "
          f"peak_parameter_and_buffer_bytes={max(size for _, size in samples)}; not process RSS")


def _eager_reference_rows(case, config):
    """Previous eager training values as an independent oracle for the indexed runner."""
    api = _api()
    bundle, options, indexed = case
    prepared = load_pose_development(
        bundle, root=options["root"], inventory=options["inventory"],
        extraction_spec=options["extraction_spec"], feature_spec=options["feature_spec"],
        classes=options["classes"], expected_manifest_sha256=options["expected_manifest_sha256"],
        expected_encoder_hash=options["expected_encoder_hash"],
    )
    selection, _, rewired = api._graph_inputs(
        options["topology_protocol"], options["connectivity"], options["controls"],
        options["expected_topology_sha256"], options["expected_controls_sha256"], config,
    )
    graph = selection.graph
    pending = []
    for seed in config.seeds:
        variants = {"random_graph": api.random_sparse_graph(graph.num_nodes, graph.num_edges, seed),
                    "rewired": rewired[seed], "flywire": graph}
        training = api.TrainConfig(seed=seed, epochs=config.epochs, lr=config.lr,
                                  batch_size=config.batch_size, parameter_ceiling=config.parameter_ceiling)
        for family in FAMILIES:
            with api._seeded_cpu(seed), torch.device("cpu"):
                model = (api.GRUClassifier(121, config.gru_hidden_dim, len(prepared.train.classes))
                         if family == "gru" else api.GraphRecurrentClassifier(
                             121, variants[family], config.graph_node_dim, len(prepared.train.classes)))
            fingerprint = "none" if family == "gru" else api.graph_fingerprint(variants[family])
            pending.append((seed, family, model, training, fingerprint))
    rows = []
    for seed, family, model, training, fingerprint in pending:
        run = train_padded_model(model, prepared.train, prepared.validation, training)
        rows.append({
            "seed": seed, "family": family, "input_binding_sha256": indexed.binding_sha256,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "topology_fingerprint": fingerprint, "adjacency_sha256": api._adjacency_hash(model),
            "state_hash": run.state_hash, "best_epoch": run.best_epoch,
            "validation_metrics": asdict(evaluate_padded(
                model, prepared.validation, batch_size=config.batch_size)),
            "optimizer_steps": run.optimizer_steps, "epoch_order_hashes": list(run.epoch_order_hashes),
            "training_config": asdict(training), "training_config_hash": api._hash_json(asdict(training)),
        })
    prepared.verify()
    indexed.verify(expected_binding_sha256=indexed.binding_sha256)
    return rows


def test_all_arm_outputs_match_eager_reference_exactly(comparison_case):
    reference = _eager_reference_rows(comparison_case, _spec())
    report = _run(comparison_case)
    assert report["arms"] == reference
    assert report["final_test_evaluated"] is False
    assert report["topology_claim_evaluated"] is False


def test_last_preflight_model_failure_still_prevents_all_training(comparison_case, monkeypatch):
    api = _api()
    original, constructions, updates = api.GraphRecurrentClassifier.__init__, [], []

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        constructions.append(1)
        if len(constructions) == 6:  # Last of three graph arms x two seeds.
            with torch.no_grad():
                self.readout.bias[0] = float("nan")

    def forbidden(*args, **kwargs):
        updates.append(1)
        raise AssertionError("late preflight failure must precede every optimization")

    monkeypatch.setattr(api.GraphRecurrentClassifier, "__init__", initialize)
    monkeypatch.setattr(api, "train_indexed_model", forbidden)
    with pytest.raises(ValueError):
        _run(comparison_case)
    assert len(constructions) == 6 and updates == []


@pytest.mark.parametrize("field,completed_arms", (("learned_state", 0), ("adjacency", 1)))
def test_reconstruction_drift_is_rejected_before_affected_arm_training(
        comparison_case, monkeypatch, field, completed_arms):
    api = _api()
    constructions, trained = [], []

    def alter(original):
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            constructions.append(1)
            if len(constructions) > 8:  # Eight models must have passed the complete preflight first.
                with torch.no_grad():
                    if field == "learned_state":
                        self.readout.bias[0] += 1
                    elif hasattr(self, "adjacency"):
                        self.adjacency[0, 0] += 1
        return initialize

    actual = api.train_indexed_model
    def train(*args, **kwargs):
        result = actual(*args, **kwargs)
        trained.append(1)
        return result

    for kind in (api.GRUClassifier, api.GraphRecurrentClassifier):
        monkeypatch.setattr(kind, "__init__", alter(kind.__init__))
    monkeypatch.setattr(api, "train_indexed_model", train)
    with pytest.raises(ValueError, match="preflight"):
        _run(comparison_case)
    assert len(trained) == completed_arms
