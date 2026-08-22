# Simulation–Calibration Boundary Design

## Purpose

Turn the existing formal cognition and goal-selection core into a research platform that can run, calibrate, validate, and diagnose non-ideal or third-party models without reimplementing those algorithms in Lean.

## Architectural boundary

Lean remains the formal specification and theorem layer. It proves model-level identities, admissibility conditions, provenance constraints, and world invariants. It does not reimplement mature numerical algorithms such as POMDP solvers, MCTS, HTN, GFlowNet, approximate Bayesian computation, or mean-field solvers.

Python is the research runtime. Existing or black-box models are wrapped behind one simulation protocol, executed with explicit seeds, summarized by metric extractors, calibrated against target summaries, and validated by synthetic recovery and held-out experiments.

The integration rule is:

```text
untrusted model / solver
        ↓
canonical SimulationTrace
        ↓
metrics and calibration
        ↓
validation evidence
```

A model may use any internal algorithm, but it may not bypass the canonical trace, seed, parameter, and scenario metadata recorded by the runner.

## First milestone

The first milestone is deliberately dependency-free and contains five pieces:

1. A `SimulatorModel` protocol whose implementations receive a scenario, numeric parameter mapping, and a seeded `random.Random` instance.
2. A canonical immutable `SimulationTrace` containing model name, scenario id, sorted parameters, seed, ordered events, and outcome data.
3. A `SimulationRunner` that provides deterministic single runs and ordered batch runs.
4. Metric aggregation and a deterministic exhaustive grid calibrator using weighted squared error.
5. Synthetic recovery that generates target summaries from known parameters and checks whether calibration recovers them.

The milestone also includes a `GroundedGoalDecisionModel` adapter that reuses the existing grounded finite-goal softmax implementation. It samples a selected goal from the existing grounded choice distribution and emits a canonical trace. No second goal-scoring formula is introduced.

## Data flow

```text
Scenario + Parameters + Seed
        ↓
SimulationRunner
        ↓
SimulatorModel.simulate(...)
        ↓
ModelRun(events, outcome)
        ↓
SimulationTrace
        ↓
MetricExtractor
        ↓
Aggregated metrics
        ↓
Grid calibration / synthetic recovery
```

## Determinism and provenance

The runner constructs the RNG and owns run metadata. A model receives the RNG but does not choose or rewrite the seed. Equal model, scenario, parameter mapping, and seed must produce equal traces for deterministic model implementations.

Batch output order follows the input seed order. Parameter mappings are canonicalized as sorted `(name, value)` tuples. Numeric parameters, metrics, targets, weights, and losses must be finite.

## Calibration semantics

For candidate parameters `θ`, seeds `ξ₁…ξₙ`, trace metric extractor `Φ`, target summary `y`, and optional nonnegative weights `w`, calibration minimizes:

```text
L(θ) = Σₖ wₖ · ( mean_j Φₖ(Sim(θ, ξⱼ)) - yₖ )²
```

The first implementation performs exhaustive finite grid search. Ties are resolved lexicographically by canonical parameter tuples, making results reproducible.

This is a research baseline, not the final inference method. ABC-SMC, Bayesian optimization, neural SBI, and surrogate models can later implement the same calibration interface.

## Synthetic recovery

Synthetic recovery uses a known parameter set to generate target metrics over observation seeds, then calibrates over a candidate grid using separate calibration seeds. The report preserves:

- true parameters;
- synthetic target metrics;
- ranked candidate evaluations;
- whether the best candidate exactly matches the canonical true parameter tuple.

Failure to recover does not automatically mean an implementation bug. It may reveal insufficient metrics, stochastic uncertainty, parameter non-identifiability, or a grid that excludes the true parameters.

## Error handling

The platform rejects:

- empty seed sets;
- empty parameter grids;
- empty grid dimensions;
- non-finite numeric values;
- negative metric weights;
- metric key drift between traces;
- targets or weights referring to unknown metric names;
- grounded-goal scenarios with empty goal maps;
- grounded-goal adapter parameters other than the documented finite `beta` value.

## Testing

The milestone uses standard-library `unittest`, matching the repository CI. Tests cover:

- deterministic replay;
- batch order and metadata canonicalization;
- metric aggregation and validation;
- deterministic grid ranking and parameter recovery;
- a non-identifiable example that is reported rather than hidden;
- grounded-goal adapter normalization and seeded sampling;
- synthetic recovery of a grounded-goal inverse-temperature candidate from repeated choice frequencies.

## Out of scope

This milestone does not add online adaptation, MCMC, ABC-SMC, neural SBI, multiprocessing, persistent experiment storage, Go runtime integration, action execution, or narrative extraction. Those remain later independently testable milestones.