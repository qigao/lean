# Versioned Observational-Data Protocol Design

## Status

Approved by the user's explicit `go` after the finite prison POMDP adequacy layer was merged. This increment is Python-only. It freezes empirical inputs and comparison rules around the existing model; it does not add prison states, actions, horizons, parameters, Lean definitions, or solver capabilities.

## Purpose

The current runtime can execute, calibrate, validate, and diagnose a model, but observed targets can still arrive as ad hoc dictionaries and thresholds can still be chosen after seeing results. V1 adds a reproducible empirical protocol with five boundaries:

1. immutable, versioned observational datasets;
2. explicit train, selection-validation, and final-test partitions with source-record disjointness;
3. target construction that preserves the raw counts, source observation ids, grouping rule, seed plan, and dataset lineage;
4. content-hashed adequacy thresholds and comparison registration fixed before final evaluation;
5. a fair alternative-model comparison report that evaluates every frozen model specification on the exact same final targets, loss, metric extractor, and seed plan.

These boundaries make leakage and post-hoc configuration observable. They do not prove that the dataset is representative, that the preregistration existed before data access, or that any model is empirically true.

## Architecture

Create a focused `narrative_dynamics.observations` package:

```text
observations/
  dataset.py          immutable persisted dataset and partitions
  targets.py          categorical target construction and suite conversion
  preregistration.py  frozen model specs and adequacy/comparison registration
  comparison.py       exact registered final-test comparison
```

The package reuses existing `Scenario`, `CategoricalMetricGroup`, `HeldOutCase`, `HeldOutSuite`, `SelectionValidationReport`, `FinalTestReport`, metric/loss identities, manifests, and aggregate report attestation. It does not duplicate calibration or validation engines.

## Dataset Protocol

### Partition roles

```python
class ObservationPartitionRole(str, Enum):
    TRAIN = "train"
    SELECTION_VALIDATION = "selection_validation"
    FINAL_TEST = "final_test"
```

A dataset must contain at least one case in every role. The roles are immutable and are not inferred from file location or naming conventions.

### Observation cases

```python
@dataclass(frozen=True)
class ObservationCase:
    name: str
    role: ObservationPartitionRole
    scenario: Scenario
    counts: Mapping[str, int]
    observation_ids: tuple[str, ...]
    provenance: Mapping[str, object] = field(default_factory=dict)
```

`counts` are raw non-negative categorical count coordinates, not probabilities. `observation_ids` identify the underlying observations that contributed to the case. Each case requires at least one observation id and positive total count mass. All values are recursively detached and immutable.

Observation ids must be globally unique across the complete dataset. This is the leakage boundary: the same source observation cannot appear in train, selection, or final partitions. Reusing the same scenario mechanics across roles is allowed because controlled generalization studies may intentionally do so.

### Dataset identity

```python
@dataclass(frozen=True)
class ObservationDataset:
    name: str
    version: str
    source: Mapping[str, object]
    cases: tuple[ObservationCase, ...]
    schema_version: int = 1
```

Cases are canonicalized by `(role, name)`. Dataset identity includes schema version, name, version, source metadata, every scenario content hash, counts, observation ids, provenance, and role. `content_hash` is a typed SHA-256 produced by the existing canonical hash function.

The persisted representation is strict JSON-compatible data. `to_payload()`, `from_payload()`, and `load_observation_dataset()` support round trips and optionally verify a declared `content_hash`. Unknown top-level or case fields fail closed.

### Partitions

`dataset.partition(role)` returns an immutable `ObservationPartition` carrying the parent dataset hash and its own content hash. A partition cannot be assembled from cases belonging to another dataset or role.

## Target Construction Provenance

### Construction plan

```python
@dataclass(frozen=True)
class CategoricalTargetPlan:
    name: str
    version: str
    role: ObservationPartitionRole
    groups: tuple[CategoricalMetricGroup, ...]
    seeds_by_case: Mapping[str, tuple[int, ...]]
```

The plan is immutable and content-hashed. Its categorical groups must be nonempty, have unique names, and cover each selected case's count schema exactly once. Per-key loss weights are not used during target construction; group weights are retained only because the same group declaration is shared with the scoring rule.

The seed plan must contain exactly the selected partition's case names. Every case receives a nonempty ordered integer seed tuple. Seeds are simulation design inputs, not observational data.

### Constructed targets

For every selected case and categorical group:

```text
target[key] = count[key] / sum(group counts)
```

Every group total must equal the number of source observation ids in the case. This ensures each categorical coordinate is derived from the declared observations rather than an unrelated mass table.

`construct_categorical_targets(dataset, plan)` returns a `TargetConstructionReport` containing:

- dataset and partition hashes;
- plan identity;
- raw counts and normalized targets;
- source observation ids and provenance;
- scenarios and seed plans;
- an `ExperimentManifest` with stage `target_construction`.

The report can be converted to an existing `HeldOutSuite` only for selection-validation or final-test roles. Training targets remain available as immutable constructed cases but are not mislabeled as held-out evaluation data.

## Frozen Models and Preregistration

### Frozen model specification

```python
@dataclass(frozen=True)
class FrozenModelSpec:
    name: str
    model_identity: Mapping[str, object]
    parameters: tuple[tuple[str, float], ...]
    selection_manifest_hash: str
```

`FrozenModelSpec.from_selection(...)` requires a real `SelectionValidationReport`, captures its selected parameters and manifest hash, and captures the runtime model identity through `component_identity()`. Final comparison never calibrates or mutates these parameters.

### Adequacy thresholds

```python
@dataclass(frozen=True)
class AdequacyThresholds:
    max_mean_loss: float
    max_worst_loss: float
```

Both limits must be finite and non-negative. Threshold identity is content-hashed.

### Comparison registration

```python
@dataclass(frozen=True)
class ComparisonPreregistration:
    name: str
    version: str
    dataset_hash: str
    final_partition_hash: str
    target_construction_manifest_hash: str
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    thresholds: AdequacyThresholds
    models: tuple[FrozenModelSpec, ...]
```

The registration binds the exact final dataset partition, target construction, metric extractor, scoring rule, thresholds, model set, fixed parameters, and selection lineage. Model names are unique. The fixed ranking rule is:

```text
mean final loss, then worst-case loss, then model name
```

The registration content hash is written into the final comparison manifest. V1 records a preregistered configuration identity but cannot prove when or by whom it was created; a trusted external registry or signature layer remains future work.

## Fair Alternative-Model Comparison

```python
compare_registered_models(
    *,
    runner: SimulationRunner,
    registration: ComparisonPreregistration,
    target_report: TargetConstructionReport,
    models: Mapping[str, ModelSource],
    extractor: MetricExtractor,
    loss: MetricLoss,
) -> AlternativeModelComparisonReport
```

Before any model executes, the function verifies:

- the target report is final-test data;
- dataset, partition, and target-construction hashes match the registration;
- metric and loss identities match exactly;
- the supplied model-name set matches exactly;
- every runtime model identity matches its frozen specification.

Only after the complete preflight succeeds does evaluation begin. Each model is evaluated once using the existing `evaluate_on_final_test_suite()` and its fixed parameters. All models receive the same suite, targets, seed plan, metric extractor, and loss.

Each `AlternativeModelEvaluation` ords fixed parameters, mean and worst loss, threshold adequacy, and the child final-test manifest hash. The report ranks evaluations by the preregistered rule, exposes `best`, carries stage `alternative_model_comparison`, and is compatible with `attest_report()`.

Failure of one model is not silently converted into a favorable or unfavorable score; existing typed execution/schema failures propagate.

## Error Boundaries

The implementation fails closed on:

- unsupported dataset schema versions or unknown payload fields;
- empty dataset roles, duplicate case names, duplicate observation ids, non-integer/negative counts, or mutable/opaque provenance values;
- target groups that omit or duplicate count coordinates;
- group count totals that disagree with source observation count;
- missing, extra, empty, or non-integer seed plans;
- train targets being converted to held-out suites;
- selection reports or model identities that do not match a frozen model spec;
- invalid threshold values;
- registration/model/target/loss/metric mismatch;
- missing or extra alternative models.

## Manifests and Artifacts

Add two experiment stages:

- `target_construction`;
- `alternative_model_comparison`.

Target reports and comparison reports are immutable dataclasses with existing `ExperimentManifest` lineage. `attest_report()` must bind each report's canonical payload to its manifest without special cases.

## Testing

Strict RED -> GREEN:

1. dataset tests fail because the observations package does not exist;
2. target-construction tests fail because there is no provenance-preserving constructor;
3. preregistration tests fail because selected models and thresholds cannot be frozen;
4. comparison tests fail because there is no exact preflight or common final-test evaluation;
5. implement the minimum production code;
6. add review REDs for any discovered leakage or identity defect;
7. require feature and PR #2 merge contexts to pass Lean-generated conformance, full Lean build, every theorem test, and the complete Python suite.

## Non-goals

- no raw event-level statistical model beyond categorical count coordinates;
- no database, dataframe, or external dependency;
- no timestamp-based proof that registration preceded final-data access;
- no digital signature or public registry for datasets or preregistrations;
- no automatic train-set calibration orchestration;
- no bootstrap, confidence interval, or posterior predictive coverage;
- no modification to the prison model or Lean kernel;
- no claim that the chosen dataset or alternatives are representative or scientifically adequate.
