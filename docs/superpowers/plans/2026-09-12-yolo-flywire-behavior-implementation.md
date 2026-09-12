# YOLO + FlyWire Behavior Recognition V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible V0 experiment that tests whether a FlyWire-derived connectome topology provides topology-specific value for pedestrian behavior and hand-gesture recognition from frozen YOLO/Pose point-set sequences.

**Architecture:** Lean 4 defines the experiment contract, comparison admissibility, and evidence boundaries. Python owns synthetic and real point-sequence ingestion, temporal baselines, graph models, rewiring, FlyWire graph provenance, training, evaluation, and evidence manifests. The primary causal comparison is FlyWire-derived topology versus a topology-matched rewired control under the same observations, split, seeds, capacity constraints, and training budget.

**Tech Stack:** Lean 4 + Lake; Python 3.11+; pytest; NumPy; PyTorch (preferred for V0); JSON evidence manifests; YOLO/Pose outputs as frozen input data.

**Spec:** `docs/superpowers/specs/2026-09-12-yolo-flywire-behavior-design.md`

## Global Constraints

- V0 excludes BB/social simulation, multi-person identity tracking, ByteTrack replacement, metric-depth replacement, end-to-end RGB training, and full-brain simulation.
- The detector is not the object under test; main comparisons use the same frozen point-set observations.
- A claim of topology-specific FlyWire benefit is inadmissible without the rewired FlyWire control.
- Rewired controls must preserve node count, edge count, in/out-degree sequence as closely as the selected algorithm permits, and the declared edge-weight distribution policy.
- Final-test data must not participate in fitting, hyperparameter selection, rewiring selection, or success-threshold selection.
- Negative results are valid outcomes and must not cause post-hoc broadening of the claim.
- Lean formalizes contracts and admissible conclusions; it does not prove empirical performance or biological validity.
- All empirical runs must record code commit, split identity, seed, model family, topology identity, training budget, and configuration hash.

---

## Repository Structure

```text
lean/
  lakefile.lean
  lean-toolchain
  YoloFlywire/
    Basic.lean
    Contracts.lean
    Protocol.lean
    Claims.lean
  YoloFlywire.lean
  python/
    pyproject.toml
    yolo_flywire/
      __init__.py
      schema.py
      synthetic.py
      features.py
      graphs.py
      models/
        __init__.py
        gru.py
        graph_rnn.py
      train.py
      eval.py
      manifests.py
      flywire.py
      cli.py
  tests/
    test_schema.py
    test_synthetic.py
    test_features.py
    test_graphs.py
    test_models.py
    test_protocol_manifest.py
    test_flywire.py
  data/
    README.md
  docs/
    superpowers/
      specs/
      plans/
```

`YoloFlywire/*` contains only formal experiment semantics. `python/yolo_flywire/*` contains empirical runtime code. `data/` contains documentation and ignored local datasets only; no large YOLO or FlyWire artifacts are committed.

---

### Task 1: Bootstrap Lean/Python project and formal protocol vocabulary

**Files:**
- Create: `lean-toolchain`
- Create: `lakefile.lean`
- Create: `YoloFlywire/Basic.lean`
- Create: `YoloFlywire/Contracts.lean`
- Create: `YoloFlywire/Protocol.lean`
- Create: `YoloFlywire/Claims.lean`
- Create: `YoloFlywire.lean`
- Create: `python/pyproject.toml`
- Create: `python/yolo_flywire/__init__.py`

**Interfaces:**
- Produces Lean types: `Split`, `ModelFamily`, `TopologyKind`, `TrainingBudget`, `RunProtocol`, `EvidenceClaim`.
- Produces predicates: `sameObservationBoundary`, `sameBudget`, `hasRequiredRewiredControl`, `claimAdmissible`.
- Later tasks serialize Python manifests that correspond to these concepts.

- [ ] **Step 1: Write the Lean RED assertions**

Create `YoloFlywire/Claims.lean` with theorem statements that cannot compile until the protocol types exist:

```lean
import YoloFlywire.Protocol

namespace YoloFlywire

example (p : RunProtocol) :
    claimAdmissible p EvidenceClaim.topologySpecificAdvantage →
    hasRequiredRewiredControl p := by
  intro h
  exact topologyClaim_requires_rewired p h

example (p : RunProtocol) :
    p.finalTestUsedForSelection = false := by
  exact protocol_final_test_sealed p

end YoloFlywire
```

- [ ] **Step 2: Run Lean to confirm RED**

Run:

```bash
lake build
```

Expected: FAIL because `RunProtocol`, `EvidenceClaim`, and the theorems are undefined.

- [ ] **Step 3: Add minimal Lean project and protocol model**

Use a pinned Lean toolchain in `lean-toolchain` and define:

```lean
namespace YoloFlywire

inductive Split where
  | train | validation | test
  deriving DecidableEq, Repr

inductive ModelFamily where
  | gru | tcn | transformer | randomGraph | rewiredFlywire | flywire
  deriving DecidableEq, Repr

inductive TopologyKind where
  | dense | randomSparse | rewired | connectome
  deriving DecidableEq, Repr

structure TrainingBudget where
  epochs : Nat
  maxUpdates : Nat
  parameterCeiling : Nat
  deriving DecidableEq, Repr

inductive EvidenceClaim where
  | temporalModelUseful
  | topologySpecificAdvantage
  | robustnessAdvantage
  deriving DecidableEq, Repr

structure RunArm where
  family : ModelFamily
  topology : TopologyKind
  observationSchemaHash : String
  splitHash : String
  budget : TrainingBudget
  seeds : List Nat
  deriving Repr

structure RunProtocol where
  arms : List RunArm
  finalTestUsedForSelection : Bool
  primaryMetric : String
  successThreshold : Float
  deriving Repr
```

Define `hasRequiredRewiredControl`, `claimAdmissible`, and theorem bodies so that `topologySpecificAdvantage` requires both a FlyWire arm and a rewired arm with matching observation/split/budget fields, and `protocol_final_test_sealed` follows from a constructor-side validity witness rather than asserting arbitrary `RunProtocol` values are valid. If this requires separating raw `RunProtocol` from `ValidProtocol`, do so now rather than encoding a false theorem.

- [ ] **Step 4: Run Lean to confirm GREEN**

Run:

```bash
lake build
```

Expected: PASS.

- [ ] **Step 5: Add Python package metadata**

Create `python/pyproject.toml` with Python `>=3.11`, runtime dependencies on `numpy` and `torch`, and dev dependency on `pytest`. Configure pytest to discover `../tests` when run from `python/`.

- [ ] **Step 6: Commit**

```bash
git add lean-toolchain lakefile.lean YoloFlywire YoloFlywire.lean python/pyproject.toml python/yolo_flywire/__init__.py
git commit -m "feat: define yolo flywire experiment contract"
```

---

### Task 2: Define point-sequence schema and synthetic behavior dataset

**Files:**
- Create: `python/yolo_flywire/schema.py`
- Create: `python/yolo_flywire/synthetic.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_synthetic.py`

**Interfaces:**
- Produces immutable dataclasses: `KeypointFrame`, `PointSequence`, `LabeledSequence`.
- Produces `make_synthetic_dataset(seed: int, samples_per_class: int, frames: int) -> list[LabeledSequence]`.
- Synthetic classes: `standing`, `walking`, `wave_left`, `wave_right`, `push`, `pull`.

- [ ] **Step 1: Write schema RED tests**

```python
from yolo_flywire.schema import KeypointFrame, PointSequence


def test_point_sequence_rejects_inconsistent_keypoint_count():
    f0 = KeypointFrame(points=((0.0, 0.0, 1.0),))
    f1 = KeypointFrame(points=((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)))
    try:
        PointSequence(frames=(f0, f1))
    except ValueError as exc:
        assert "keypoint count" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_schema.py -q
```

Expected: FAIL because schema module is missing.

- [ ] **Step 3: Implement immutable schema**

`KeypointFrame.points` is a tuple of `(x, y, confidence)` triples. `PointSequence` validates non-empty frames and a constant keypoint count. `LabeledSequence` stores `sequence`, `label`, and `sample_id`.

- [ ] **Step 4: Write synthetic RED tests**

Test determinism and temporal distinguishability:

```python
from yolo_flywire.synthetic import make_synthetic_dataset


def test_synthetic_dataset_is_seed_deterministic():
    a = make_synthetic_dataset(seed=7, samples_per_class=2, frames=12)
    b = make_synthetic_dataset(seed=7, samples_per_class=2, frames=12)
    assert a == b


def test_left_and_right_wave_have_opposite_wrist_displacement():
    data = make_synthetic_dataset(seed=1, samples_per_class=1, frames=12)
    by_label = {x.label: x for x in data}
    left = by_label["wave_left"].sequence.frames
    right = by_label["wave_right"].sequence.frames
    assert left[-1].points[-1][0] - left[0].points[-1][0] < 0
    assert right[-1].points[-1][0] - right[0].points[-1][0] > 0
```

- [ ] **Step 5: Implement deterministic synthetic generator**

Use a fixed minimal skeleton of six points: torso anchor, left/right shoulder, left/right wrist, pelvis. Add bounded Gaussian jitter from `numpy.random.Generator`. Ensure class identity is encoded by temporal motion, not by different static point IDs.

- [ ] **Step 6: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_schema.py ../tests/test_synthetic.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/schema.py python/yolo_flywire/synthetic.py tests/test_schema.py tests/test_synthetic.py
git commit -m "feat: add synthetic pose sequence dataset"
```

---

### Task 3: Implement frozen feature extraction with uncertainty masks

**Files:**
- Create: `python/yolo_flywire/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Produces `FeatureSpec` and `encode_sequence(sequence: PointSequence, spec: FeatureSpec) -> np.ndarray`.
- Output shape: `[time, feature_dim]`.
- All models receive exactly the same encoded tensor in the main comparison.

- [ ] **Step 1: Write RED tests**

```python
import numpy as np
from yolo_flywire.features import FeatureSpec, encode_sequence
from yolo_flywire.synthetic import make_synthetic_dataset


def test_encoding_is_translation_invariant_when_enabled():
    sample = make_synthetic_dataset(3, 1, 8)[0].sequence
    shifted = sample.shifted(dx=100.0, dy=-40.0)
    spec = FeatureSpec(normalize_translation=True, normalize_scale=True)
    np.testing.assert_allclose(encode_sequence(sample, spec), encode_sequence(shifted, spec), atol=1e-6)


def test_encoding_contains_velocity_and_confidence():
    sample = make_synthetic_dataset(4, 1, 8)[0].sequence
    x = encode_sequence(sample, FeatureSpec())
    assert x.shape[0] == 8
    assert np.isfinite(x).all()
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_features.py -q
```

Expected: FAIL because feature module and/or `shifted` helper are missing.

- [ ] **Step 3: Implement normalization and motion features**

For each frame: center points on the torso anchor, normalize scale using shoulder span with epsilon floor, retain confidence values, append first-order velocity, and emit explicit missing/low-confidence masks. Do not encode class labels or sample IDs.

- [ ] **Step 4: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_features.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/yolo_flywire/features.py python/yolo_flywire/schema.py tests/test_features.py
git commit -m "feat: add frozen temporal pose features"
```

---

### Task 4: Add graph contract and degree-preserving rewiring

**Files:**
- Create: `python/yolo_flywire/graphs.py`
- Create: `tests/test_graphs.py`

**Interfaces:**
- Produces `DirectedGraph(num_nodes, src, dst, weight)`.
- Produces `random_sparse_graph(num_nodes, num_edges, seed)`.
- Produces `rewire_degree_preserving(graph, seed, swaps)`.
- Produces `graph_fingerprint(graph) -> str`.

- [ ] **Step 1: Write RED invariance tests**

```python
from collections import Counter
from yolo_flywire.graphs import DirectedGraph, rewire_degree_preserving


def degrees(g):
    return Counter(g.src), Counter(g.dst)


def test_rewiring_preserves_directed_degree_sequence():
    g = DirectedGraph(
        num_nodes=5,
        src=(0,0,1,2,3,4),
        dst=(1,2,2,3,4,0),
        weight=(1,2,1,1,3,1),
    )
    r = rewire_degree_preserving(g, seed=9, swaps=50)
    assert degrees(r) == degrees(g)
    assert sorted(r.weight) == sorted(g.weight)
    assert set(zip(r.src, r.dst)) != set(zip(g.src, g.dst))
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_graphs.py -q
```

Expected: FAIL because graph module is missing.

- [ ] **Step 3: Implement graph validation and rewiring**

Use directed double-edge swaps `(a->b, c->d) -> (a->d, c->b)` while rejecting self-loops and duplicate edges. Carry the two original edge weights with the swapped source edges so the global weight multiset is preserved. Raise a clear error if the requested number of successful swaps cannot be achieved within a bounded attempt budget.

- [ ] **Step 4: Add deterministic fingerprint test**

Fingerprint must hash canonicalized `(num_nodes, sorted(src,dst,weight))` content and be independent of input edge order.

- [ ] **Step 5: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_graphs.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/graphs.py tests/test_graphs.py
git commit -m "feat: add graph controls and degree preserving rewiring"
```

---

### Task 5: Implement GRU and graph-recurrent temporal models

**Files:**
- Create: `python/yolo_flywire/models/__init__.py`
- Create: `python/yolo_flywire/models/gru.py`
- Create: `python/yolo_flywire/models/graph_rnn.py`
- Create: `tests/test_models.py`

**Interfaces:**
- `GRUClassifier(input_dim, hidden_dim, num_classes)` -> logits `[batch, classes]`.
- `GraphRecurrentClassifier(input_dim, graph, node_dim, num_classes)` -> logits `[batch, classes]`.
- Both consume `[batch, time, feature_dim]`.

- [ ] **Step 1: Write RED shape and recurrence tests**

```python
import torch
from yolo_flywire.graphs import random_sparse_graph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier


def test_models_share_batch_time_feature_input_contract():
    x = torch.randn(4, 12, 18)
    gru = GRUClassifier(18, 16, 6)
    graph = random_sparse_graph(12, 24, seed=1)
    grnn = GraphRecurrentClassifier(18, graph, node_dim=8, num_classes=6)
    assert gru(x).shape == (4, 6)
    assert grnn(x).shape == (4, 6)
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_models.py -q
```

Expected: FAIL because model modules are missing.

- [ ] **Step 3: Implement GRU baseline**

Use `torch.nn.GRU(batch_first=True)` and classify from final hidden state. Expose `parameter_count()`.

- [ ] **Step 4: Implement graph recurrent model**

Use fixed sparse adjacency derived from `DirectedGraph`. At each step, project input features to node inputs, perform recurrent message passing using the fixed adjacency mask and learnable gains/edge scale parameters permitted by the spec, update node states, pool, and classify. Do not learn new edges.

- [ ] **Step 5: Add topology-sensitivity test**

With fixed parameters and the same input, two non-isomorphic graph masks must produce different hidden trajectories when weights are initialized deterministically to nonzero values.

- [ ] **Step 6: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_models.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/models tests/test_models.py
git commit -m "feat: add temporal baseline and graph recurrent model"
```

---

### Task 6: Add deterministic training/evaluation harness and evidence manifests

**Files:**
- Create: `python/yolo_flywire/train.py`
- Create: `python/yolo_flywire/eval.py`
- Create: `python/yolo_flywire/manifests.py`
- Create: `tests/test_protocol_manifest.py`

**Interfaces:**
- `TrainConfig(seed, epochs, lr, batch_size, parameter_ceiling)`.
- `train_model(model, train, validation, config) -> TrainedRun`.
- `evaluate(model, test) -> MetricBundle` with macro F1 and balanced accuracy.
- `RunManifest` stores all fields required by the spec and emits canonical JSON + content hash.

- [ ] **Step 1: Write RED manifest tests**

```python
from yolo_flywire.manifests import RunManifest


def test_manifest_hash_changes_when_split_changes():
    a = RunManifest.example(split_hash="a")
    b = RunManifest.example(split_hash="b")
    assert a.content_hash() != b.content_hash()


def test_manifest_rejects_missing_rewired_control_for_topology_claim():
    m = RunManifest.example(claim="topology_specific_advantage", compared_families=("flywire",))
    try:
        m.validate_claim_boundary()
    except ValueError as exc:
        assert "rewired" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_protocol_manifest.py -q
```

Expected: FAIL because harness modules are missing.

- [ ] **Step 3: Implement canonical manifest and metrics**

Canonical JSON must sort keys and use stable separators. Record: code commit, dataset ID, split hash, observation schema hash, model family, topology fingerprint, seed, budget, training config hash, metric values, and compared families.

- [ ] **Step 4: Implement deterministic training seed plumbing**

Seed Python, NumPy, and PyTorch from one declared run seed. Validation chooses the best checkpoint; test evaluation is a separate function that is not called by training.

- [ ] **Step 5: Add synthetic overfit smoke test**

Train a tiny GRU on a tiny deterministic synthetic split and assert macro F1 exceeds a deliberately low smoke threshold, such as `0.60`, to validate plumbing rather than establish research evidence.

- [ ] **Step 6: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_protocol_manifest.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/train.py python/yolo_flywire/eval.py python/yolo_flywire/manifests.py tests/test_protocol_manifest.py
git commit -m "feat: add reproducible training and evidence manifests"
```

---

### Task 7: Add FlyWire graph ingestion with explicit provenance

**Files:**
- Create: `python/yolo_flywire/flywire.py`
- Create: `tests/test_flywire.py`
- Create: `data/README.md`

**Interfaces:**
- `FlyWireEdge(pre_id, post_id, synapse_count, region=None, cell_type=None)`.
- `FlyWireSelection(release_id, selection_rule, neuron_ids, edges)`.
- `selection_to_graph(selection) -> DirectedGraph`.
- `load_flywire_csv(path, release_id, selection_rule) -> FlyWireSelection`.

- [ ] **Step 1: Write RED provenance test**

```python
from yolo_flywire.flywire import load_flywire_csv


def test_flywire_selection_requires_release_and_rule(tmp_path):
    p = tmp_path / "edges.csv"
    p.write_text("pre_id,post_id,synapse_count\n1,2,4\n", encoding="utf-8")
    try:
        load_flywire_csv(p, release_id="", selection_rule="")
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_flywire.py -q
```

Expected: FAIL because loader is missing.

- [ ] **Step 3: Implement local CSV ingestion only**

V0 ingestion reads a user-supplied exported edge table; it does not embed credentials or silently query remote services. Validate positive synapse counts, no unknown neuron references after selection, deterministic neuron-ID remapping, and provenance strings.

- [ ] **Step 4: Document expected data contract**

`data/README.md` must specify required columns, optional annotations, recommended bounded visual/motion subgraph size, and the requirement to record exact FlyWire release identity and extraction rule.

- [ ] **Step 5: Add rewired-control pairing test**

Load a small fixture graph, derive its rewired control using `rewire_degree_preserving`, and assert node count, edge count, degree sequence, and weight multiset match while graph fingerprints differ.

- [ ] **Step 6: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_flywire.py ../tests/test_graphs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/flywire.py tests/test_flywire.py data/README.md
git commit -m "feat: add provenance bound flywire graph ingestion"
```

---

### Task 8: Add CLI for one frozen comparison run

**Files:**
- Create: `python/yolo_flywire/cli.py`
- Modify: `python/pyproject.toml`
- Create: `tests/test_cli.py`

**Interfaces:**
- Command: `yolo-flywire synthetic-run --seed 7 --output out/`.
- Command: `yolo-flywire compare --config protocol.json --output out/`.
- Every command writes a manifest before or alongside metrics.

- [ ] **Step 1: Write RED CLI test**

Use `subprocess.run` against the module entrypoint and assert `synthetic-run` creates `manifest.json` and `metrics.json` in a temporary directory.

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_cli.py -q
```

Expected: FAIL because CLI is missing.

- [ ] **Step 3: Implement synthetic-run**

Create deterministic synthetic data, split by sample ID, train the GRU smoke baseline, evaluate once, and write canonical outputs.

- [ ] **Step 4: Implement compare**

`compare` must require an explicit protocol config listing model arms. Reject a topology-specific claim if the config omits a rewired control before training starts.

- [ ] **Step 5: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/cli.py python/pyproject.toml tests/test_cli.py
git commit -m "feat: add protocol bound experiment cli"
```

---

### Task 9: Freeze V0 evaluation protocol before real final-test runs

**Files:**
- Create: `protocols/v0-synthetic.json`
- Create: `protocols/v0-real-template.json`
- Create: `docs/v0-evaluation-protocol.md`
- Modify: `YoloFlywire/Protocol.lean`
- Modify: `YoloFlywire/Claims.lean`

**Interfaces:**
- Protocol JSON includes split hash, observation schema hash, seeds, budgets, model families, rewiring algorithm, primary metric, secondary metrics, success threshold, and test-seal flag.
- Lean checks a corresponding simplified protocol witness for claim admissibility.

- [ ] **Step 1: Write protocol RED theorem**

Add an example that a topology-specific protocol containing only `flywire` and `gru` cannot produce a valid topology-specific claim witness.

- [ ] **Step 2: Run RED**

```bash
lake build
```

Expected: FAIL until the stronger validity predicate is encoded.

- [ ] **Step 3: Strengthen Lean validity predicate**

Require explicit rewired arm matching the FlyWire arm's observation schema, split hash, and budget. Keep empirical thresholds as data, not theorems.

- [ ] **Step 4: Create frozen synthetic protocol**

Declare at least five seeds, macro F1 as primary, balanced accuracy and robustness metrics as secondary, a fixed epoch/update budget, and the exact rewiring algorithm identifier.

- [ ] **Step 5: Create real-data template without invented dataset values**

The template must contain null/empty fields that make validation fail until dataset ID, YOLO version, split hash, FlyWire release, selection rule, and success threshold are explicitly filled and frozen.

- [ ] **Step 6: Run Lean and Python validation GREEN**

```bash
lake build
cd python && python -m pytest ../tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add protocols docs/v0-evaluation-protocol.md YoloFlywire/Protocol.lean YoloFlywire/Claims.lean
git commit -m "docs: freeze v0 evaluation protocol boundary"
```

---

### Task 10: Add robustness and lesion evaluation without changing claim semantics

**Files:**
- Modify: `python/yolo_flywire/eval.py`
- Modify: `python/yolo_flywire/graphs.py`
- Create: `tests/test_robustness.py`

**Interfaces:**
- `add_keypoint_noise(x, sigma, seed)`.
- `mask_keypoints(x, probability, seed)`.
- `lesion_graph(graph, node_ids=(), edge_indices=())`.
- `RobustnessReport` contains clean and perturbed metric bundles.

- [ ] **Step 1: Write RED deterministic perturbation tests**

Assert repeated perturbations with the same seed are identical and that lesioning removes only declared nodes/edges.

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_robustness.py -q
```

Expected: FAIL because robustness functions are missing.

- [ ] **Step 3: Implement perturbations and lesion utilities**

Perturb only observations or graph structure as declared. Never mutate the original arrays/graph in place.

- [ ] **Step 4: Add report contract test**

A robustness result may support `robustnessAdvantage`; it must not automatically imply `topologySpecificAdvantage` unless the rewired comparison is present.

- [ ] **Step 5: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_robustness.py ../tests/test_protocol_manifest.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/eval.py python/yolo_flywire/graphs.py tests/test_robustness.py
git commit -m "feat: add robustness and topology lesion evaluation"
```

---

### Task 11: Integrate real frozen YOLO/Pose sequences

**Files:**
- Create: `python/yolo_flywire/yolo_io.py`
- Create: `tests/test_yolo_io.py`
- Modify: `data/README.md`

**Interfaces:**
- `load_pose_jsonl(path) -> list[LabeledSequence]` for exported/frozen pose observations.
- Required record fields: `sample_id`, `frame_index`, `label`, `body_keypoints`; optional `hand_keypoints`, `bbox`, `detector_confidence`.
- The loader never invokes YOLO during experiment comparison.

- [ ] **Step 1: Write RED fixture test**

Create a temporary JSONL with two sequences and assert stable frame ordering, schema validation, and rejection of duplicate `(sample_id, frame_index)` records.

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_yolo_io.py -q
```

Expected: FAIL because loader is missing.

- [ ] **Step 3: Implement frozen observation loader**

Sort records by sample/frame, validate constant keypoint schema per dataset, preserve confidence/missingness, and compute an observation-schema hash from declared field layout rather than sample values.

- [ ] **Step 4: Add leakage guard**

Reject dataset records whose label is included inside the model feature payload beyond the dedicated target field.

- [ ] **Step 5: Run GREEN**

```bash
cd python && python -m pytest ../tests/test_yolo_io.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/yolo_io.py tests/test_yolo_io.py data/README.md
git commit -m "feat: ingest frozen yolo pose sequences"
```

---

### Task 12: End-to-end exact-protocol comparison and evidence gate

**Files:**
- Modify: `python/yolo_flywire/cli.py`
- Modify: `python/yolo_flywire/manifests.py`
- Create: `tests/test_end_to_end.py`
- Create: `docs/v0-result-template.md`

**Interfaces:**
- One command executes every protocol arm over every declared seed and writes one aggregate report.
- Aggregate report states exactly one of: `topology_advantage_supported`, `topology_advantage_not_established`, or `protocol_invalid`.

- [ ] **Step 1: Write RED end-to-end test**

Using the synthetic protocol and a tiny graph fixture, run GRU, random graph, rewired graph, and FlyWire-labeled graph arms over two tiny seeds. Assert all arms share the same split hash, observation schema hash, and budget; assert aggregate conclusion generation does not crash when FlyWire loses.

- [ ] **Step 2: Run RED**

```bash
cd python && python -m pytest ../tests/test_end_to_end.py -q
```

Expected: FAIL because aggregate comparison is missing.

- [ ] **Step 3: Implement aggregate evidence gate**

Compute per-seed metrics, paired differences for FlyWire minus rewired, and deterministic summary statistics. Only emit `topology_advantage_supported` if the frozen success rule is satisfied; otherwise emit `topology_advantage_not_established`. Never rewrite the protocol threshold based on results.

- [ ] **Step 4: Add negative-result regression**

Construct fixture metrics where FlyWire underperforms rewired and assert the aggregate report explicitly returns `topology_advantage_not_established`.

- [ ] **Step 5: Add result template**

`docs/v0-result-template.md` must include protocol identity, data provenance, arm table, seed-level metrics, robustness results, paired topology comparison, limitations, and a fixed wording section for negative results.

- [ ] **Step 6: Run full GREEN verification**

```bash
lake build
cd python && python -m pytest ../tests -q
```

Expected: all Lean targets and Python tests PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/cli.py python/yolo_flywire/manifests.py tests/test_end_to_end.py docs/v0-result-template.md
git commit -m "feat: gate topology specific behavior evidence"
```

---

## Final Verification Gate

Before claiming V0 implementation complete:

- [ ] Run `lake build` from repository root and record exact output.
- [ ] Run `cd python && python -m pytest ../tests -q` and record exact pass/fail count.
- [ ] Run one `synthetic-run` from a clean output directory.
- [ ] Validate the generated manifest contains commit SHA, protocol hash, split hash, observation schema hash, seed, budget, model family, topology fingerprint, and metrics.
- [ ] Run one end-to-end synthetic comparison including GRU, random sparse, rewired, and FlyWire-labeled graph arms.
- [ ] Confirm the pipeline accepts a negative topology result without changing thresholds or throwing an error.
- [ ] Do not run a real final-test dataset until `protocols/v0-real-template.json` is fully populated, reviewed, and frozen in a commit.

## Completion Definition

V0 is implementation-complete when the repository can reproduce a frozen temporal behavior-recognition experiment across a conventional temporal baseline, random sparse graph, rewired FlyWire control, and FlyWire-derived topology; Lean rejects inadmissible topology-specific claims at the contract level; Python emits provenance-bound evidence manifests; and both positive and negative empirical outcomes are handled without changing the predeclared claim boundary.
