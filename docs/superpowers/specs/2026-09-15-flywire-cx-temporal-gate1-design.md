# FlyWire Central-Complex Temporal Gate 1 Design

Date: 2026-09-15
Branch: `research/flywire-cx-temporal-gate1`
Status: Approved design, pre-implementation

## 1. Research question

Gate 1 asks one falsifiable question:

> Does biologically organized FlyWire central-complex topology improve formation, retention, and recovery of temporal state relative to matched null graphs, when observation encoding, neuron dynamics, trainable parameter budget, seeds, and evaluation are held fixed?

The primary claim is deliberately narrower than action-recognition accuracy. The experiment is designed to detect a topology-specific dynamical advantage, not to establish that FlyWire is a general-purpose classifier.

Gate 1 does **not** claim that human joints map to individual fly neurons, that central-complex semantics transfer directly to human behavior, or that a whole-fly-brain simulation is needed for useful temporal modeling.

## 2. Relationship to the existing YOLO/FlyWire study

This work is a separate Gate 1 branch derived from `research/yolo-flywire-behavior-v0`.

The existing Issue #68 / PR #69 experiment freezes a FAFB v783 **visual-system type graph** around T4/T5 motion types and evaluates FlyWire versus a degree-/weight-matched rewired graph on a YOLO/Pose behavior-recognition protocol.

This Gate 1 does not modify that frozen visual-system graph, its measured fingerprints, its statistical threshold, or its real-data evidence boundary. It reuses generic graph/provenance infrastructure only where semantics remain identical.

## 3. Scope

### In scope

- FAFB v783 central-complex graph extraction with frozen provenance.
- Central-complex core and interface-population annotations.
- Fixed biological input and output node sets.
- Directed weighted graph dynamics using synapse-count magnitude and neurotransmitter-derived sign where provenance supports it.
- A minimal event/spike encoder for skeleton kinematics.
- A fixed-topology recurrent LIF temporal core.
- Degree-preserving and block-preserving null controls.
- Early-action prediction at fixed observation ratios.
- State-retention and perturbation-recovery diagnostics.
- Spike/event-efficiency measurements.
- Five-seed paired comparisons and fail-closed provenance.

### Out of scope

- RTMW 133-point WholeBody extraction.
- YOLO or pose-estimator fine-tuning.
- Mushroom-body online learning.
- Descending-neuron/VNC motor control.
- Whole-brain 139k-neuron simulation.
- End-to-end RGB training.
- Selecting graph thresholds, regions, or null models using final-test performance.
- Broad claims that FlyWire improves generic action recognition.

## 4. Data boundary

Gate 1 starts from skeleton time series so that pose-extraction error cannot confound the topology test.

The intended first benchmark is NTU RGB+D 60/120 native skeleton data under a frozen official outer split. Skeleton data are used only for Gate 1; this does not alter Issue #68's rule that official NTU skeletons are not classifier input in that separate YOLO/Pose confirmatory experiment.

A sample is represented as a sequence of joint coordinates plus visibility/validity information. The same normalized kinematic representation is supplied to every graph arm.

No final-test sequence may be used to choose the FlyWire subgraph, synapse threshold, input/output populations, spike thresholds, observation ratios, rewiring policy, or success criterion.

## 5. Kinematic event representation

For normalized joint positions `P_t`, derive:

```text
J_t  = P_t
B_t  = child(P_t) - parent(P_t)
JM_t = P_t - P_(t-1)
BM_t = B_t - B_(t-1)
A_t  = JM_t - JM_(t-1)
```

The minimal Gate 1 encoder converts motion changes into signed sparse events rather than Poisson-encoding static coordinates.

For feature component `i`:

```text
S_pos[i,t] = 1  if delta[i,t] > +theta_i else 0
S_neg[i,t] = 1  if delta[i,t] < -theta_i else 0
```

Thresholds are fitted on training data only and then frozen. The encoder may include a small learned linear projection from the shared kinematic feature vector into fixed CX input channels, but it may not inject directly into arbitrary recurrent nodes.

## 6. Central-complex graph artifact

### 6.1 Source

Use the public Codex FAFB v783 static-download products, not live page scraping. Gate 1 requires exactly these source families:

```text
consolidated_cell_types
connections_princeton
```

The artifact records at minimum:

- dataset: FAFB;
- release: v783;
- exact source product names;
- source content SHA-256 values;
- extraction code/version identifier;
- connection threshold: 5 aggregate synapses per directed neuron pair, matching the FAFB Codex default minimum;
- canonical node ordering;
- graph fingerprint.

Codex connection data may contain multiple rows for one neuron pair when synapses occur in multiple neuropils. Gate 1 first aggregates those rows by directed neuron pair, then applies the fixed 5-synapse threshold. The threshold is not tuned for model performance.

### 6.2 Canonical core

The initial core is restricted to neurons belonging to the canonical central-complex system centered on:

```text
PB  protocerebral bridge
EB  ellipsoid body
FB  fan-shaped body
NO  noduli
```

Selection is cell-type/annotation driven, not merely "has one arbor in a CX neuropil". The extraction code must emit the exact accepted type annotations and a deterministic selection report.

### 6.3 Interface populations

Nodes are partitioned into explicit roles:

```text
INPUT     biological CX-afferent/interface population
CORE      recurrent CX population
OUTPUT    biological CX-efferent/action-selection population
```

Input routing targets only `INPUT` nodes. Model readout consumes only `OUTPUT` nodes.

The initial output population includes well-annotated PFL-family CX output pathways when present in the frozen dataset. Exact type lists are emitted by the artifact builder and frozen before training.

No human joint is assigned semantic identity to a particular fly neuron.

## 7. Graph weights and neuron dynamics

Gate 1 R0 is a fixed biological reservoir.

For edge `i -> j`:

```text
magnitude_ij = f(synapse_count_ij)
sign_i       = neurotransmitter-derived excitatory/inhibitory sign under the frozen policy
weight_ij    = global_gain * sign_i * magnitude_ij
```

The implementation supports `raw` and `log1p` magnitude policies for development diagnostics, but the confirmatory protocol freezes one policy before final evaluation. There is no per-edge learned recurrent weight in R0.

Use a minimal deterministic LIF recurrence with explicit parameters:

```text
tau_membrane
synaptic_decay
refractory_steps
threshold
reset
recurrent_delay_steps
```

Initial values follow the Shiu-style whole-brain LIF regime when units are compatible; any discretization or deviation is recorded in the protocol.

Trainable parameters in R0 are limited to:

- input projection into biological input nodes;
- output readout from biological output nodes;
- explicitly declared global gain/scale parameters.

The recurrent graph mask and relative biological recurrent weights remain fixed.

## 8. Null-graph ladder

All null graphs share node count, input/output role assignments, neuron dynamics, trainable parameter shapes, and training budget with Real CX.

### 8.1 Degree-preserving rewired

Directed double-edge swaps preserve:

- node count;
- edge count;
- exact per-node in-degree;
- exact per-node out-degree;
- global recurrent edge-weight multiset;
- any explicitly frozen diagonal/self representation policy.

Input/output role membership remains attached to node identity; only eligible recurrent connectivity is rewired.

### 8.2 Block-preserving rewired

A stricter null additionally preserves declared mesoscale blocks such as neuropil/type-class and interface/core/output roles. Swaps may occur only among edge pairs whose reassignment preserves the frozen block-to-block edge-count matrix.

This distinguishes precise biological wiring from coarse architecture.

Interpretation:

```text
Real > Degree, Real ~= Block  => mesoscale organization is sufficient.
Real > Block                   => finer biological wiring adds evidence.
Real ~= both                   => no topology-specific benefit detected.
```

### 8.3 Random sparse reference

A simple edge-count-matched sparse random graph may be reported as a weak reference, but it is not the decisive null.

## 9. Temporal tasks

### 9.1 Early prediction

For each sequence, evaluate fixed observation ratios:

```text
10%, 20%, 40%, 60%, 80%, 100%
```

The model receives only the observed prefix. For each ratio compute macro F1. The primary early-prediction statistic is trapezoidal area under macro-F1 versus observation-ratio curve, normalized by the ratio range so the result remains in `[0, 1]`.

### 9.2 State retention

At the 40% observation boundary, set subsequent external events to zero while allowing recurrent state to evolve for a fixed horizon frozen in the protocol.

Measure:

- `retention_auc`: area under the true-class logit-margin curve over the zero-input horizon, normalized to the margin at the zero-input boundary and clipped only by a frozen numeric-stability rule;
- output-confidence half-life `T50`;
- duration for which the predicted class remains unchanged.

`retention_auc` is the Gate 1 retention endpoint; `T50` and stable-duration are descriptive secondary metrics.

### 9.3 Perturbation recovery

At the same frozen 40% observation boundary, apply one frozen temporary node-silencing perturbation to recurrent state and then continue with the same subsequent observed events as the unperturbed matched run.

Measure:

- `recovery_rate`: fraction of eligible samples whose original unperturbed predicted class is recovered within the frozen recovery horizon;
- median steps to recovery among recovered samples;
- degradation versus the unperturbed matched trajectory.

Node-silencing fraction and recovery horizon are selected using training/validation only and then frozen.

## 10. Scoreboard

Gate 1 reports five axes:

```text
1. full-sequence predictive quality
2. early-prediction AUC
3. state retention
4. perturbation recovery
5. spike/event efficiency
```

Efficiency measures include:

- spike rate;
- active-neuron fraction;
- recurrent synaptic events per observed frame;
- peak recurrent state memory footprint;
- inference latency as an engineering measurement, not a neuroscience claim.

## 11. Statistical protocol

Use five frozen model seeds:

```text
7, 11, 19, 23, 31
```

The decisive comparisons are paired by seed and data split:

```text
Real CX - Degree-preserving CX
Real CX - Block-preserving CX
```

Primary endpoint:

```text
early_prediction_auc = normalized AUC of macro F1 at 10/20/40/60/80/100%
```

Gate 1A practical-effect rule for biological CX organization:

```text
mean paired (Real - Degree) early_prediction_auc >= 0.02
AND Real - Degree > 0 in at least 4/5 seeds
AND at least one temporal-state secondary condition holds:
    mean paired retention_auc difference >= 0.05 with >0 in at least 4/5 seeds
    OR
    mean paired recovery_rate difference >= 0.05 with >0 in at least 4/5 seeds
```

Gate 1B fine-wiring rule uses the same thresholds for `Real - Block`. Gate 1B is reported separately and is not silently substituted for Gate 1A.

Before final-test execution, freeze:

- dataset/split fingerprints;
- graph artifact fingerprint;
- input/output node fingerprints;
- all null-graph fingerprints;
- kinematic encoder identity and fitted thresholds;
- LIF parameters;
- recurrent magnitude/sign policy;
- observation ratios;
- retention horizon;
- perturbation fraction and recovery horizon;
- training budget;
- seeds;
- primary metric and practical-effect thresholds above.

No retry may replace a failed or unfavorable seed result after final-test exposure.

## 12. Gate decision

Gate 1A passes only if the frozen Real-CX arm satisfies the exact Real-versus-Degree rule in Section 11. A pass means biological CX organization produced a reproducible temporal-state advantage over a degree-/weight-matched null under this task.

Gate 1B passes only if the frozen Real-CX arm independently satisfies the exact Real-versus-Block rule. A Gate 1A pass with Gate 1B null means the evidence supports mesoscale organization, not precise fine wiring.

If Gate 1A is null or negative, stop the exact-CX branch before RTMW133 or mushroom-body work. A null result is a valid completed result.

A Gate 1A pass licenses Gate 2 only:

```text
RTMW133 vs reduced-body skeleton on fine-grained early prediction
```

Mushroom-body online learning remains Gate 3 and is not introduced by Gate 1.

## 13. Implementation boundaries

Create new focused modules rather than changing the semantics of existing visual-system functions:

```text
python/yolo_flywire/cx_artifact.py
python/yolo_flywire/cx_nulls.py
python/yolo_flywire/cx_events.py
python/yolo_flywire/models/cx_lif.py
python/yolo_flywire/cx_evaluation.py
```

Reuse `DirectedGraph`, deterministic graph fingerprinting, and generic validation helpers when their contracts match exactly.

Do not change:

- existing visual type graph selection;
- Issue #68 frozen fingerprints;
- current visual-graph rewiring algorithm semantics;
- existing GRU/GraphRecurrentClassifier behavior.

New behavior receives new protocol identifiers and tests.

## 14. Fail-closed requirements

The implementation must reject rather than infer or fall back when:

- source hashes do not match;
- required neuron annotations are missing;
- neurotransmitter sign is required but unavailable under the frozen policy;
- input/output role sets are empty or overlap in a forbidden way;
- graph/null fingerprints differ from the protocol;
- degree or block invariants fail;
- a final-test run is requested before provenance is complete;
- a requested observation ratio, seed, or perturbation setting differs from the frozen protocol.

There is no compatibility fallback to the old visual graph, reduced rewiring budget, all-node input injection, or all-node mean readout.

## 15. Evidence boundary

The following are **not** evidence of a topology-specific benefit:

- Real CX beating a GRU but not matched rewired CX;
- a synthetic-data result;
- graph provenance alone;
- a single favorable seed;
- better 100% sequence accuracy with no temporal-state advantage;
- post-hoc threshold, region, or perturbation tuning.

A positive conclusion must remain exactly scoped to the frozen task and metrics. A null result must be reported as null rather than followed by an outcome-driven graph redesign on the same final-test data.
