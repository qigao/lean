# Pose Temporal SNN V1 Design

Date: 2026-09-15
Branch: `research/pose-temporal-snn-v1`
Status: design approved in chat; implementation not started

## 1. Research question

V1 asks one narrow question:

> Does a spiking temporal model provide a useful advantage for streaming skeleton behavior understanding when preprocessing, data splits, trainable parameter budget, seeds, and evaluation are controlled?

The target advantages are temporal, not biological:

- early action prediction;
- continual frame-by-frame state update;
- state retention under temporary input loss;
- recovery from short perturbations;
- sparse event/spike activity.

V1 does not make or test any FlyWire/connectome claim.

## 2. Hard scope boundary

### In scope

- Native NTU RGB+D skeleton sequences as the first benchmark.
- Tracking-aware skeleton normalization.
- Signed motion-event encoding from joint/bone dynamics.
- Three temporal model families under one shared protocol:
  1. dense/sparse recurrent LIF SNN (`RSNN`);
  2. skeleton-topology spiking graph network (`SpikingGraph`);
  3. strong conventional recurrent baseline (`GRU`).
- Early-prediction observation ratios: 10%, 20%, 40%, 60%, 80%, 100%.
- Full-sequence macro F1.
- State retention after zero external input.
- Perturbation recovery.
- Streaming activity/compute accounting.
- Five frozen seeds and paired development comparisons.

### Out of scope for V1

- FlyWire, central complex, connectome graphs, neurotransmitter annotations, biological synapse weights.
- RTMW133 / WholeBody production extraction.
- RGB end-to-end training.
- S3T-Former reproduction or code vendoring.
- Transformer-scale models.
- Neuromorphic hardware benchmarking.
- Few-shot/online plasticity.
- Final-test claims before development protocol is frozen.

RTMW133 becomes Gate 2 only if V1 shows the temporal SNN route is worth keeping. Online/continual local learning becomes Gate 3.

## 3. Isolation from the abandoned FlyWire work

The existing FlyWire/CX branches are historical evidence only.

This branch starts from `research/yolo-flywire-behavior-v0` at `7b5de8176adf4beff5e2a46c8ff021caad6404ae`, before the CX Gate 1 implementation work.

New production code must live in a separate Python project and must not import `yolo_flywire.cx_*`, FlyWire graph builders, null-graph code, or connectome provenance modules.

The new project root is:

```text
pose_snn/
    pyproject.toml
    src/pose_temporal_snn/
    tests/
```

The old `python/yolo_flywire` project remains untouched in V1. This keeps rollback simple: the entire new SNN line can be removed without editing the old experiment package, and the old FlyWire package can later be archived independently.

## 4. Data boundary

V1 starts from native NTU skeleton sequences to avoid pose-estimator error as a confound.

The first benchmark uses the same 10-class motion subset already used by the prior pose study:

```text
A8   sitting_down
A9   standing_up
A22  cheer_up
A23  hand_waving
A26  hopping
A27  jump_up
A31  pointing
A34  rub_two_hands_together
A35  nod_head_or_bow
A36  shake_head
```

The official NTU120 X-Sub boundary remains the outer split.

V1 defines one frozen development-validation subject subset entirely inside the outer training side. The held-out outer side stays sealed until development choices are frozen.

Raw NTU data are not committed or redistributed. Runs consume a legally obtained local skeleton root and create only hashes/manifests plus model/result artifacts.

## 5. Skeleton preparation

The reusable behavior from the CX spike experiment is retained as a specification, not as a runtime dependency.

For a selected primary body:

1. A joint observation is usable when `tracking_state >= 1` and the body is present.
2. Missing joint observations are filled deterministically along time by linear interpolation, with endpoint hold outside the first/last usable frame.
3. Each joint must have at least one usable observation or the sample fails closed.
4. Joint 0 is the root center:

```text
P'_j(t) = P_j(t) - P_0(t)
```

5. Scale is the median valid torso length between joints 0 and 20:

```text
s = median_t ||P_20(t) - P_0(t)||
P_norm = P' / s
```

6. Degenerate/nonfinite scale fails closed.

This explicitly prevents body-tracking dropouts from becoming artificial high-velocity events.

## 6. Event representation

V1 is motion-driven rather than rate-encoding static pose values.

From normalized positions derive:

```text
JM_t = P_t - P_(t-1)              joint motion
B_t  = P_child(t) - P_parent(t)    bone vectors
BM_t = B_t - B_(t-1)              bone motion
A_t  = JM_t - JM_(t-1)            joint acceleration
```

The event feature vector is:

```text
X_t = concat(JM_t, BM_t, A_t)
```

Per-feature thresholds are fitted from development-training sequences only using one frozen absolute-motion quantile `q`.

Signed events are:

```text
E_i(t) = +1  if X_i(t) > +theta_i
       = -1  if X_i(t) < -theta_i
       =  0  otherwise
```

Validation/final-test data may only transform through an already-fitted encoder.

No Poisson encoding is used in V1.

## 7. Temporal cores

All cores expose one shared interface:

```python
initial_state(batch_size, ...)
step(events_t, state) -> state
logits(state) -> [batch, classes]
forward(events) -> logits
```

Streaming evaluation must use `step`; batch `forward` is convenience only.

### 7.1 RSNN

A conventional recurrent LIF model with trainable recurrent weights.

V1 freezes:

- hidden size chosen before final-test exposure;
- one-step recurrent delay;
- membrane/synaptic decay;
- refractory behavior;
- threshold/reset;
- input projection and classifier readout.

The recurrent matrix may use a fixed sparsity mask selected independently of task outcomes. Real-valued trainable recurrent weights are allowed because the research question is about SNN temporal computation, not biological fixed connectivity.

### 7.2 SpikingGraph

A graph-structured SNN using only the human 25-joint anatomical skeleton.

Spatial propagation is restricted to the frozen NTU kinematic adjacency plus self-state; temporal state remains spiking/recurrent.

The graph answers a different question from RSNN:

> Does explicit human-body topology help a spiking temporal model beyond an unstructured recurrent SNN?

No non-human graph is admitted.

### 7.3 GRU baseline

A conventional streaming GRU with the same event input, outer/inner split, seeds, update budget, and readout class count.

GRU is the decisive ANN baseline for V1 because it also maintains incremental state. Comparing against only a sliding-window ANN would over-credit SNN for avoiding repeated history computation.

A TCN/SSM baseline may be added only in a later protocol revision before final-test exposure; it is not required for the first V1 development gate.

## 8. Fairness contract

For every seed, the model arms must share:

- identical train/validation sample inventory;
- identical event encoder bytes/fingerprint;
- identical label mapping;
- identical sample order per epoch;
- identical optimizer family where applicable;
- identical maximum update count;
- identical observation-ratio evaluation;
- identical checkpoint-selection rule.

Exact parameter count equality is not required because model families differ structurally. Instead V1 freezes a parameter ceiling and reports trainable parameter counts. Any model exceeding the ceiling is inadmissible.

Checkpoint selection is development-validation macro F1 only. Early-AUC, retention, and recovery are evaluated on that same selected checkpoint; they do not participate in model selection.

## 9. Primary and secondary endpoints

### Primary endpoint

Normalized area under macro F1 versus observation ratio:

```text
10%, 20%, 40%, 60%, 80%, 100%
```

Call this `early_prediction_auc`.

The primary comparison is:

```text
RSNN - GRU
```

paired by seed.

SpikingGraph versus GRU and SpikingGraph versus RSNN are secondary comparisons.

### Secondary endpoints

- full-sequence macro F1;
- prediction at each observation ratio;
- zero-input retention after the 40% boundary;
- confidence/state half-life;
- deterministic perturbation recovery;
- spike rate;
- active-neuron fraction;
- recurrent synaptic events per observed frame;
- wall-clock streaming latency as an engineering metric only;
- trainable parameter count.

## 10. Success / stop rule

V1 is intentionally easier to kill than to extend.

The first development success criterion is:

```text
mean paired (RSNN - GRU) early_prediction_auc >= 0.02
AND RSNN - GRU > 0 in at least 4/5 seeds
```

At least one temporal secondary metric must also show a reproducible benefit:

```text
retention improvement >= 0.05 in mean paired normalized retention AUC
OR
recovery-rate improvement >= 0.05
```

with positive direction in at least 4/5 seeds.

If RSNN is null/negative versus GRU, do not proceed to RTMW133 merely because full-sequence accuracy is acceptable.

If SpikingGraph wins while RSNN does not, continue the human-skeleton graph line but do not infer a generic SNN advantage.

If neither spiking arm shows a temporal advantage, stop the SNN research line and retain only the generic streaming skeleton/evaluation utilities.

## 11. Online learning is a separate gate

No online plasticity is part of V1.

If V1 passes, Gate 3 may add a small plastic associative readout or local recurrent rule and test sequential class addition / catastrophic forgetting.

The online-learning question is kept separate because mixing plasticity into V1 would make it impossible to attribute any result to temporal SNN dynamics versus continual-learning machinery.

## 12. Protocol artifacts

Before any outer final-test run, freeze:

- input dataset-content hash;
- outer and inner split hash;
- selected action IDs;
- normalization ID;
- event-encoder quantile and fitted threshold fingerprint;
- label-map fingerprint;
- each model architecture/dynamics fingerprint;
- optimizer and training budget;
- seeds;
- parameter ceiling;
- observation ratios;
- retention and perturbation protocol;
- primary metric and success threshold.

Final test remains sealed until every required pin is non-null.

## 13. Package/component layout

```text
pose_snn/
  pyproject.toml
  src/pose_temporal_snn/
    ntu.py
    normalization.py
    events.py
    protocol.py
    data.py
    evaluation.py
    training.py
    models/
      rsnn.py
      spiking_graph.py
      gru.py
    cli.py
  tests/
    test_ntu.py
    test_normalization.py
    test_events.py
    test_protocol.py
    test_rsnn.py
    test_spiking_graph.py
    test_gru.py
    test_evaluation.py
    test_training_fairness.py
    test_final_test_seal.py
```

Each module has one narrow dependency direction:

```text
ntu -> normalization -> events -> data
models ---------------------------> training -> evaluation
protocol -------------------------> all experiment entry points
```

No module in the new project imports from `yolo_flywire`.

## 14. CI

Create a new independent workflow:

```text
.github/workflows/pose-temporal-snn.yml
```

Jobs:

1. lightweight data/protocol tests using NumPy + pytest;
2. CPU-Torch model/training tests;
3. full `pose_snn` test suite;
4. existing Lean build as a repository regression check.

The workflow does not run FlyWire/Codex provenance and does not download NTU data.

## 15. Evidence boundary

The following are not evidence that SNN is better:

- lower operation count with lower predictive quality;
- better final-sequence accuracy with no early-prediction benefit;
- one favorable seed;
- different train/validation inventories;
- different encoder thresholds by model arm;
- post-hoc hidden size, sparsity, retention horizon, or perturbation tuning on final-test outcomes;
- comparison only against a non-streaming sliding-window model.

V1 supports a claim only about the frozen skeleton task and temporal metrics. It does not imply superiority on RGB behavior recognition, RTMW133, robotics, neuromorphic hardware, or online learning.
