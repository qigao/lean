# Pose Graph + SSM V1 Design

Date: 2026-09-15
Branch: `research/pose-graph-ssm-v1`
Status: design approved in chat; implementation not started

## 1. Research question

V1 asks one narrow, falsifiable question:

> Does preserving human skeletal graph structure while maintaining a causal state-space temporal state improve streaming action understanding over matched non-graph and non-SSM baselines?

The intended advantage is practical temporal modeling, not biological plausibility:

- early action prediction;
- frame-by-frame streaming inference;
- robustness to temporary pose loss;
- recovery after tracking dropouts;
- stable behavior state on longer sequences.

V1 makes no FlyWire, connectome, SNN, LIF, spike-efficiency, or neuromorphic claim.

## 2. Hard scope boundary

### In scope

- Native NTU RGB+D 120 skeleton sequences as the first benchmark.
- The existing 10-class motion subset:
  - A8 sitting_down
  - A9 standing_up
  - A22 cheer_up
  - A23 hand_waving
  - A26 hopping
  - A27 jump_up
  - A31 pointing
  - A34 rub_two_hands_together
  - A35 nod_head_or_bow
  - A36 shake_head
- Tracking-aware skeleton repair and normalization.
- Continuous kinematic features; no spike/event quantization.
- Four causal streaming model arms:
  1. `GRU` — no explicit human graph, recurrent ANN baseline;
  2. `SSMOnly` — causal selective state-space model without graph propagation;
  3. `GraphTCN` — human graph plus causal temporal convolution;
  4. `GraphSSM` — human graph plus causal selective state-space model.
- Early prediction at 10%, 20%, 40%, 60%, 80%, 100% observation.
- Full-sequence macro F1.
- Zero-input state retention.
- Controlled pose-dropout recovery.
- Natural sequence-length stratified reporting.
- Five frozen seeds and paired model comparisons.

### Out of scope for V1

- FlyWire, Codex, central complex, biological connectivity, neurotransmitters.
- SNN/LIF/surrogate-gradient training.
- RTMW133/WholeBody production extraction.
- RGB end-to-end training.
- Mamba package dependency or claim of reproducing Mamba exactly.
- Transformer-scale models.
- Online/continual learning.
- Neuromorphic or physical-energy claims.
- Final-test use before all development choices are frozen.

RTMW133 becomes Gate 2 only if GraphSSM passes the V1 temporal gate. A hierarchical body/hands/face/feet graph is therefore explicitly deferred.

## 3. Isolation from abandoned FlyWire/SNN work

This branch starts from commit `7b5de8176adf4beff5e2a46c8ff021caad6404ae`, before the FlyWire-CX and pose-SNN branches.

New production code lives in a new independent Python project:

```text
pose_graph_ssm/
  pyproject.toml
  src/pose_graph_ssm/
  tests/
  protocols/
```

No source file in the new project may import or refer to:

```text
yolo_flywire
flywire
codex
connectome
cx_
spiking
lif
surrogate_spike
```

The earlier branches remain historical experiments only and are not runtime dependencies.

## 4. Data boundary and split

Use the official NTU120 X-Sub outer split.

Frozen outer-training subjects:

```text
1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38,
45,46,47,49,50,52,53,54,55,56,57,58,59,70,74,78,80,81,82,
83,84,85,86,89,91,92,93,94,95,97,98,100,103
```

Frozen inner-validation subjects, all inside outer train:

```text
14,28,35,46,50,54,74,83,84,86,103
```

All subjects in `1..106` outside the official outer-training set are `final_test`.

Final-test skeleton bytes may be inventoried and hashed while sealed, but their coordinates may not be normalized, feature-transformed, used for statistics, used for model selection, or scored before a separate unseal commit.

Raw NTU data are never committed or redistributed.

## 5. Skeleton preparation

For each selected primary body:

1. Select the body maximizing the total count of fully tracked (`tracking_state == 2`) joints; break ties by lowest numeric body ID.
2. A joint observation is usable iff the body is present and `tracking_state >= 1`.
3. Missing joint coordinates are repaired independently along time by linear interpolation; leading/trailing gaps use endpoint hold.
4. A joint with no usable observation fails closed.
5. Center each frame on joint 0:

```text
P_centered[j,t] = P[j,t] - P[0,t]
```

6. Compute subject scale as the median positive finite distance between joints 0 and 20:

```text
s = median_t ||P[20,t] - P[0,t]||
```

7. Reject nonfinite or degenerate `s <= 1e-6`.
8. Normalize positions by `s`.

This prevents body-tracking dropouts from becoming artificial velocity spikes and makes translation/uniform scale irrelevant.

## 6. Continuous kinematic feature representation

V1 removes the SNN event threshold entirely.

For each of 25 joints derive five XYZ feature groups:

```text
P_t   = normalized joint position
JM_t  = P_t - P_(t-1)
B_t   = P_child(t) - P_parent(t)
BM_t  = B_t - B_(t-1)
A_t   = JM_t - JM_(t-1)
```

The root joint has zero bone and bone-motion vectors.

Per-joint feature vector:

```text
X[j,t] = concat(P, JM, B, BM, A) in R^15
```

Tensor shape is therefore:

```text
[T, 25, 15]
```

Do not collapse the joint dimension before a graph model has processed it.

### 6.1 Train-only feature standardization

Fit mean and standard deviation independently for each `(joint, feature-channel)` using development-training coordinates only:

```text
mu.shape    = [25,15]
sigma.shape = [25,15]
```

Transform:

```text
X_norm = (X - mu) / max(sigma, 1e-6)
```

Validation and final test may only use frozen statistics. Validation may not refit them. Final test may not transform at all while sealed.

The feature-statistics artifact has a canonical SHA-256 fingerprint.

## 7. Frozen human graph

Use only the canonical NTU25 parent map:

```text
(-1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
  0,12,13,14, 0,16,17,18, 1, 7, 7,11,11)
```

Construct an undirected adjacency plus self loops, then row-normalize it.

No learned edges, attention-created non-anatomical edges, or dataset-dependent topology selection are admitted in V1.

The graph is an architectural prior describing human body structure only.

## 8. Shared streaming model interface

Every model exposes:

```python
initial_state(batch_size, device, dtype)
step(frame_t, state) -> new_state
logits(state) -> [batch, classes]
forward(sequence) -> [batch, classes]
```

`forward` is defined only as repeated causal `step` calls.

Evaluation of prefixes, retention, and dropout recovery must use `step`; no arm may use bidirectional context, full-history attention, or hidden future frames.

### 8.1 Zero-input semantics

After feature standardization, an all-zero frame is the canonical no-observation input used by retention and controlled-dropout tests.

All direct frame-entry projections in `SSMOnly`, `GraphTCN`, and `GraphSSM` use `bias=False`; graph self/neighbor projections and causal temporal-convolution input transforms also use `bias=False`. In the selective SSM, `W_B` and `W_C` are bias-free, while `b_delta` is allowed because it is an intrinsic learned time-scale parameter rather than input-dependent drive.

`GRUCell` retains its standard recurrent/input biases; those are treated as part of the GRU's autonomous learned dynamics. Zero input therefore means no data-dependent external term, not that every model state must numerically decay to zero.

A retention/dropout implementation may not substitute raw coordinate zeros before standardization, because raw zero would mean anatomical origin rather than neutral standardized input.

## 9. Model arms

All models consume the same normalized continuous features and labels.

### 9.1 GRU

Purpose: decisive conventional recurrent baseline.

Input per frame is flattened:

```text
25 * 15 = 375 dimensions
```

Architecture:

```text
GRUCell(375 -> 64)
Linear(64 -> 10)
```

No bidirectionality, attention, temporal window cache, or graph structure.

### 9.2 SSMOnly

Purpose: isolate the value of a causal state-space temporal core without a human graph.

Per frame:

```text
flatten [25,15] -> Linear(375,64,bias=False) -> RMSNorm
```

Then apply two stacked `SelectiveSSMBlock(64)` blocks.

This arm uses exactly the same SSM recurrence definition as GraphSSM.

### 9.3 GraphTCN

Purpose: isolate the value of SSM dynamics relative to a conventional causal temporal operator while retaining the same human graph.

Per joint input projection:

```text
Linear(15,64,bias=False)
```

Two blocks:

```text
HumanGraphBlock(64)
CausalTemporalConvBlock(64, kernel=3, dilation=1 or 2)
```

The temporal convolution is strictly causal and exposes a fixed streaming cache rather than recomputing history.

### 9.4 GraphSSM

Purpose: primary proposed architecture.

Per joint input projection:

```text
Linear(15,64,bias=False)
```

Preserve shape `[B,25,64]` at every streaming step.

Two alternating blocks:

```text
HumanGraphBlock(64)
SelectiveSSMBlock(64)   # shared across joints, temporal state per joint
```

The SSM parameterization is shared across joints, but each joint owns an independent recurrent state vector.

After the second block, apply learned joint pooling and a linear classifier.

## 10. HumanGraphBlock

Given joint states `H in R^[B,25,C]` and frozen row-normalized adjacency `A`:

```text
M = A @ H
Y = W_self(H) + W_neighbor(M)
H' = RMSNorm(H + GELU(Y))
```

`W_self` and `W_neighbor` are bias-free. `A` is a non-trainable buffer.

This block is intentionally simple: graph attention and dynamic topology are excluded so the V1 comparison remains attributable.

## 11. SelectiveSSMBlock

V1 implements a small native causal selective diagonal SSM instead of depending on `mamba_ssm`.

This is an SSM experiment, not a claim to reproduce Mamba.

For input `u_t in R^C`:

```text
A = -softplus(A_log)                         # stable negative diagonal
Delta_t = softplus(W_delta u_t + b_delta)
B_t = tanh(W_B u_t)
C_t = sigmoid(W_C u_t)
Decay_t = exp(clamp(Delta_t * A, -20, 0))
S_t = Decay_t * S_(t-1) + (1 - Decay_t) * B_t
Y_t = C_t * S_t + D * u_t
Output_t = RMSNorm(u_t + W_out(Y_t))
```

`W_B`, `W_C`, and `W_out` are bias-free. `W_delta` may use only the explicitly declared `b_delta` time-scale bias shown above.

All operations are elementwise in the state dimension except learned linear projections.

Properties:

- causal;
- stable by construction;
- one fixed-size state vector per stream/joint;
- `O(1)` state memory per new frame;
- no history replay in streaming mode;
- input-dependent state update rates.

## 12. Joint pooling

GraphTCN and GraphSSM use the same learned pooling:

```text
score_j = Linear(H_j -> 1)
alpha = softmax(score over 25 joints)
pooled = sum_j alpha_j * H_j
```

Then:

```text
Linear(64 -> 10)
```

Pooling occurs only after all graph/temporal blocks, so fine-grained joint state remains available through the temporal core.

## 13. Training fairness

Frozen seeds:

```text
7,11,19,23,31
```

For each seed, all four arms share:

- identical train/validation sample inventory;
- identical feature-statistics fingerprint;
- identical label map;
- identical per-epoch sample ordering;
- identical optimizer family;
- identical maximum update count;
- identical checkpoint-selection metric;
- identical observation-ratio evaluation;
- identical retention/dropout protocol.

Training V1:

```text
optimizer       = AdamW
learning_rate   = 3e-4
weight_decay    = 1e-4
batch_size      = 1
max_epochs      = 20
max_updates     = 4000
checkpoint_rule = best validation full-sequence macro F1
parameter_ceiling = 120000
```

Parameter counts need not be equal because the architectures differ, but every arm must be below the same ceiling and must report its exact trainable parameter count.

Checkpoint selection may use only validation full-sequence macro F1. Early-AUC, retention, dropout recovery, and length-stratified scores may not select checkpoints.

## 14. Primary and secondary metrics

### 14.1 Primary endpoint

Evaluate macro F1 after observing:

```text
10%,20%,40%,60%,80%,100%
```

Primary statistic:

```text
early_prediction_auc = normalized trapezoidal AUC of macro F1 versus observation ratio
```

Primary paired comparison:

```text
GraphSSM - GRU
```

### 14.2 Attribution comparisons

To understand why GraphSSM changes performance:

```text
GraphSSM - SSMOnly   # value of explicit human graph structure
GraphSSM - GraphTCN  # value of SSM temporal state versus causal temporal convolution
```

These are secondary but mandatory reports.

### 14.3 Robustness metrics

At the frozen 40% observation boundary:

1. **Zero-input retention**: supply 20 standardized zero-input frames and measure normalized true-class margin retention AUC.
2. **Pose-dropout recovery**: replace all standardized pose features with zero for a frozen burst of `8` frames, then restore the real observed stream; measure first stable return to the matched unperturbed predicted class and recovery rate within 10 restored frames.
3. **Natural sequence-length strata**: report full-sequence and early-AUC metrics by validation frame-count quartile. Quartile boundaries are computed from development-training frame counts and frozen before final test.

## 15. Development success and stop rule

GraphSSM V1 passes the primary gate only if:

```text
mean paired (GraphSSM - GRU) early_prediction_auc >= 0.02
AND GraphSSM - GRU > 0 in at least 4/5 seeds
```

and at least one temporal robustness condition holds:

```text
mean paired retention_auc improvement >= 0.05
OR
mean paired dropout_recovery_rate improvement >= 0.05
```

with positive direction in at least 4/5 seeds.

Attribution evidence is reported separately:

```text
Graph contribution:
mean paired (GraphSSM - SSMOnly) early_prediction_auc >= 0.01
and >0 in at least 4/5 seeds

SSM contribution:
mean paired (GraphSSM - GraphTCN) early_prediction_auc >= 0.01
and >0 in at least 4/5 seeds
```

Interpretation:

- GraphSSM passes primary + both attribution comparisons: continue to RTMW133 Gate 2.
- GraphSSM passes primary but not GraphSSM > SSMOnly: temporal SSM may matter, graph does not add enough value; do not claim a Graph+SSM-specific benefit.
- GraphSSM passes primary but not GraphSSM > GraphTCN: graph matters, but SSM does not add enough over causal temporal convolution.
- GraphSSM null/negative versus GRU: stop this route before RTMW133.
- Better 100% accuracy alone cannot turn a temporal null into a pass.

## 16. Gate 2 only after V1 passes

Gate 2 replaces native NTU25 input with:

```text
RGB -> YOLO person detection -> RTMW133
```

WholeBody is not represented as one homogeneous 133-node graph.

The intended later hierarchy is:

```text
body graph
left-hand graph
right-hand graph
face graph
feet graph
    -> cross-part fusion
    -> temporal SSM
```

No Gate 2 code belongs in V1.

## 17. Protocol provenance

Before any outer final-test run, freeze:

- raw dataset-content hash;
- split hash;
- selected actions;
- primary-body rule;
- normalization ID;
- continuous feature specification;
- train-only `[25,15]` mean/std fingerprint;
- label-map fingerprint;
- human adjacency fingerprint;
- every model architecture fingerprint;
- SSM recurrence specification fingerprint;
- optimizer/training budget;
- seeds;
- observation ratios;
- retention/dropout protocol;
- train-derived sequence-length quartile boundaries;
- primary and attribution success thresholds.

Final test remains sealed until every required pin is non-null and byte-verified.

## 18. Package layout

```text
pose_graph_ssm/
  pyproject.toml
  protocols/
    v1-development-preflight.json
  src/pose_graph_ssm/
    protocol.py
    ntu.py
    normalization.py
    features.py
    graph.py
    evaluation.py
    data.py
    training.py
    results.py
    cli.py
    models/
      base.py
      gru.py
      selective_ssm.py
      ssm_only.py
      graph_tcn.py
      graph_ssm.py
  tests/
    test_protocol.py
    test_ntu.py
    test_normalization.py
    test_features.py
    test_graph.py
    test_selective_ssm.py
    test_models.py
    test_streaming.py
    test_evaluation.py
    test_training_fairness.py
    test_final_test_seal.py
    test_development_runner.py
```

Dependency direction:

```text
ntu -> normalization -> features -> data
                       graph -----> models -> training -> evaluation/results
protocol ------------------------------------> experiment entry points
```

## 19. CI and evidence boundary

Create one independent workflow:

```text
.github/workflows/pose-graph-ssm.yml
```

Jobs:

1. lightweight protocol/data/feature tests with NumPy;
2. CPU-Torch graph/SSM/model tests;
3. full `pose_graph_ssm` suite;
4. source-isolation scan;
5. repository Lean build as a regression check.

CI downloads no NTU data and runs no FlyWire/Codex work.

The following do **not** support a Graph+SSM benefit claim:

- fixture/synthetic tests;
- architecture correctness alone;
- one favorable seed;
- better full-sequence F1 without early/robustness advantage;
- changing feature statistics by model arm;
- post-hoc hidden-size, graph, SSM recurrence, dropout burst, or success-threshold tuning after final-test exposure;
- comparison only against a non-streaming baseline.

V1 supports only claims scoped to the frozen skeleton task and metrics.
