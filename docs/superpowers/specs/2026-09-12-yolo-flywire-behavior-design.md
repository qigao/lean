# YOLO + FlyWire Behavior Recognition Study Design

Date: 2026-09-12
Branch: `research/yolo-flywire-behavior-v0`
Status: Draft for review

## 1. Research question

Can a FlyWire-derived connectome-constrained temporal network improve recognition of pedestrian behavior and hand gestures from YOLO/Pose point-set sequences, beyond what is explained by simply adding temporal capacity or sparsity?

The target claim is intentionally narrow:

> Given the same YOLO/Pose observations, training budget, evaluation protocol, and comparable model capacity, does the biological connectome topology provide a reproducible inductive bias for temporal behavior recognition?

This study does **not** attempt to prove that FlyWire reproduces human cognition, that fruit-fly vision is a direct model of human gesture semantics, or that connectome dynamics should replace YOLO, depth estimation, ByteTrack, or re-identification systems.

## 2. Scope

### In scope

- Single-person pedestrian behavior recognition.
- Hand gesture recognition from keypoint motion.
- Joint body + hand temporal pattern recognition.
- YOLO/Pose outputs as the primary observations.
- FlyWire-derived recurrent/sparse graph dynamics as a temporal model.
- Controlled comparison against conventional and topology-matched baselines.
- Lean formalization of experiment contracts, invariants, and admissible conclusions.
- Python/JAX/PyTorch execution for empirical training and evaluation.

### Out of scope for V0

- BB world / social simulation.
- Multi-person identity tracking.
- Replacing ByteTrack.
- Replacing monocular or stereo metric depth estimation.
- End-to-end RGB-to-behavior training through the connectome.
- Claims about human social cognition.
- Full 139k-neuron whole-brain simulation.

## 3. Input representation

For frame `t`, the detector supplies a structured observation:

```text
O_t = {
  body_keypoints,
  hand_keypoints,
  bbox,
  detector_confidence
}
```

The temporal sample is:

```text
S_t = [O_(t-T+1), ..., O_t]
```

The raw detector class label is not treated as the target behavior explanation. The temporal model consumes geometry and motion features derived from the point sets.

### 3.1 Normalization

Before temporal modeling, coordinates must be normalized to reduce nuisance variation:

- translation normalization around a stable body/hand anchor;
- scale normalization using body or palm extent;
- optional rotation normalization when appropriate;
- missing-keypoint masks retained explicitly;
- detector confidence retained as an uncertainty feature.

### 3.2 Motion features

For each keypoint `p_i(t)`:

```text
velocity_i(t)     = p_i(t) - p_i(t-1)
acceleration_i(t) = velocity_i(t) - velocity_i(t-1)
```

Pairwise and articulated features may include:

- relative distances;
- joint angles;
- palm/finger geometry;
- body orientation;
- bbox expansion/contraction;
- normalized motion direction;
- confidence and visibility masks.

The baseline protocol must distinguish features available to all models from model-specific internal representations.

## 4. Tasks

V0 should use simple, falsifiable classes before complex interactions.

### 4.1 Pedestrian state classes

Candidate initial classes:

- standing
- walking
- running
- sitting
- falling
- turning

### 4.2 Hand gesture classes

Candidate initial classes:

- wave
- point
- stop
- beckon
- push
- pull

### 4.3 Compositional classes

Only after the independent body and hand tasks are stable:

- walking + waving
- standing + pointing
- approaching + stop gesture

The compositional task tests whether one temporal latent state can represent simultaneous body and hand dynamics.

## 5. Model families

All model families consume the same frozen input representation unless an ablation explicitly says otherwise.

### 5.1 Conventional baselines

At minimum:

- GRU
- TCN
- a small temporal Transformer or equivalent attention baseline

These establish whether gains arise simply from adding temporal memory.

### 5.2 Sparse random graph baseline

A recurrent graph with comparable node/edge budget but randomly generated topology.

Purpose: test whether generic sparse recurrence is sufficient.

### 5.3 Rewired FlyWire control

Construct a topology control that preserves as much low-order graph structure as practical, such as:

- node count;
- edge count;
- in/out degree distribution;
- edge-weight or synapse-count distribution;
- optionally cell-type or regional partitions when the experiment requires them.

Edges are otherwise rewired.

This is the critical falsification control.

### 5.4 FlyWire-derived model

Use a bounded visual/motion-related connectome subgraph rather than the entire fly brain for V0.

The exact subgraph extraction rule must be frozen before final evaluation and recorded with provenance.

The network may learn dynamic parameters, readout weights, gains, normalization terms, and other explicitly declared parameters, but the topology under test must remain identifiable as the FlyWire-derived condition.

## 6. Architecture boundary

The core data flow is:

```text
RGB/video
  -> YOLO/Pose
  -> normalized point-set sequence
  -> temporal model
  -> behavior / gesture prediction
```

For the FlyWire condition:

```text
point-set sequence
  -> input encoder
  -> connectome-constrained dynamics
  -> latent state
  -> readout head
  -> class probabilities
```

The study tests the temporal model, not the detector.

YOLO/Pose outputs should therefore be precomputed or otherwise frozen for the main comparison wherever practical.

## 7. What counts as evidence

A result of the form:

```text
YOLO + FlyWire > YOLO alone
```

is insufficient, because any temporal model may improve over framewise detection.

The minimum meaningful comparison is:

```text
FlyWire-derived > rewired FlyWire
```

under a protocol that controls model capacity, data split, training budget, preprocessing, and evaluation.

Secondary comparisons against GRU/TCN/Transformer are also required.

### 7.1 Primary evidence criteria

Before the final test set is opened, freeze:

- primary metric;
- secondary metrics;
- data split;
- seeds;
- model capacity constraints;
- training budget;
- early-stopping policy;
- rewiring procedure;
- statistical summary procedure;
- success threshold.

A positive result must be reproducible across the declared seeds and must not depend on a single lucky split.

### 7.2 Negative result

If the FlyWire-derived topology does not outperform its rewired/topology-matched control under the frozen protocol, the study must accept the result as evidence that this experiment did not establish a useful topology-specific advantage.

The implementation must not silently broaden the claim afterward.

## 8. Metrics

Candidate metrics:

- macro F1 as the primary classification metric;
- balanced accuracy;
- per-class recall;
- calibration error or proper scoring loss;
- sample efficiency / learning curve area;
- robustness to keypoint noise;
- robustness to missing keypoints / short occlusion;
- latency and parameter count as engineering measurements.

Accuracy alone is not sufficient when classes are imbalanced.

## 9. Robustness experiments

After the clean baseline is stable:

### 9.1 Observation noise

Perturb keypoint coordinates with controlled noise.

### 9.2 Missing observations

Mask individual keypoints and short frame intervals.

### 9.3 Temporal ambiguity

Construct classes that are framewise similar but temporally distinct, for example left-vs-right wave or push-vs-pull.

### 9.4 Topology lesion

Remove selected nodes/edges or subcircuits from the FlyWire-derived graph and measure performance change.

Lesion evidence is only interpretive if the lesion rule is declared before inspecting the final test results.

## 10. Lean responsibilities

Lean is used for the formal experiment boundary, not for large-scale neural training.

Lean should model the experiment objects needed to state and verify protocol invariants, including concepts such as:

```text
ObservationSequence
DatasetSplit
ModelFamily
TopologyKind
TrainingBudget
EvaluationProtocol
MetricResult
EvidenceClaim
```

Candidate invariants:

- train/validation/test partitions are disjoint;
- final-test samples are not used for fitting or hyperparameter selection;
- compared models receive the same admissible observation fields;
- a FlyWire-specific claim cannot be derived from a comparison that lacks the rewired control;
- a run cannot satisfy a frozen protocol if its declared seed/split/budget differs;
- empirical performance claims remain empirical claims, not formal proofs of biological validity.

Lean should not encode fabricated numerical performance.

## 11. Runtime responsibilities

Python/JAX/PyTorch owns:

- dataset loading;
- YOLO/Pose feature ingestion;
- feature normalization;
- graph loading and rewiring;
- temporal model execution;
- optimization;
- metrics;
- robustness experiments;
- result manifests.

The runtime emits machine-readable evidence artifacts that can be checked against the Lean-side protocol contract.

## 12. Provenance requirements

The experiment must record:

- FlyWire dataset/connectome release identity;
- exact extraction rule for the chosen subgraph;
- source neuron/edge identifiers when available;
- transformation from synapse/connectivity data to model adjacency/weights;
- rewiring algorithm and seed;
- YOLO/Pose model/version;
- dataset identity/version;
- code commit;
- experiment configuration hash.

Without provenance, topology comparisons are not reproducible.

## 13. Initial repository structure

Proposed structure after this design is approved:

```text
lean/
  docs/
    superpowers/
      specs/
      plans/
  YoloFlywire/
    Contracts.lean
    Protocol.lean
    Claims.lean
  python/
    yolo_flywire/
      data.py
      features.py
      graphs.py
      models/
      eval.py
      manifests.py
  tests/
    lean/
    python/
```

Names may be adjusted during planning, but the Lean/runtime separation is architectural.

## 14. Implementation sequence after approval

1. Formalize the protocol and evidence boundary in Lean.
2. Add a minimal synthetic point-sequence dataset to establish RED/GREEN integration without external data.
3. Implement conventional temporal baselines.
4. Implement graph runtime with random and rewired controls.
5. Add FlyWire graph ingestion with provenance.
6. Add real YOLO/Pose sequence ingestion.
7. Freeze the evaluation protocol.
8. Run clean and robustness comparisons.
9. Report either positive or negative topology-specific evidence without changing the claim boundary.

## 15. Acceptance boundary for V0

V0 is complete when the repository can reproducibly run the same frozen temporal behavior-recognition task across:

- at least one conventional temporal baseline;
- a random sparse graph baseline;
- a rewired FlyWire control;
- the FlyWire-derived topology;

and emit a protocol-bound comparison report whose admissible conclusion is enforced by the formal contract.

V0 does not require a positive FlyWire result. A rigorous negative result is an acceptable outcome.
