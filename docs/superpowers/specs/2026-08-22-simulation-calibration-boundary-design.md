# Simulation–Calibration Boundary Design

## Purpose

Turn the formal cognition and goal-selection core into a research platform that can run, calibrate, externally validate, and diagnose non-ideal or third-party models without reimplementing those algorithms in Lean.

## Architectural boundary

Lean remains the formal specification and theorem layer. It proves model-level identities, admissibility conditions, provenance constraints, and world invariants. It does not reimplement mature numerical algorithms such as POMDP solvers, MCTS, HTN, GFlowNet, approximate Bayesian computation, or mean-field solvers.

Python is the research runtime. Existing or black-box models are wrapped behind one simulation protocol, executed with explicit seeds, summarized by metric extractors, calibrated against target summaries, and validated by synthetic recovery, uncertainty analysis, held-out scenarios, and local sensitivity diagnostics.

The integration rule is:

```text
untrusted model / solver
        ↓
canonical SimulationTrace
        ↓
metrics and calibration
        ↓
parameter acceptance set
        ↓
external validation and diagnostics
```

A model may use any internal algorithm, but it may not bypass the canonical trace, seed, parameter, and scenario metadata recorded by the runner.

## Implemented platform

The dependency-free Python platform contains:

1. A `SimulatorModel` protocol whose implementations receive a scenario, numeric parameter mapping, and a seeded `random.Random` instance.
2. A canonical immutable `SimulationTrace` containing model name, scenario id, sorted parameters, seed, ordered events, and outcome data.
3. A `SimulationRunner` that provides deterministic single runs and ordered batch runs.
4. Metric aggregation and a deterministic exhaustive grid calibrator using weighted squared error.
5. Synthetic recovery that generates target summaries from known parameters and checks whether calibration recovers them.
6. Repeated calibration over independent seed blocks, with per-candidate wins, accepted-block count, mean loss, and population loss standard deviation.
7. Canonical finite parameter acceptance sets and coordinate-wise identifiability reports.
8. Whole-block seed translations with accepted-set union and intersection.
9. Held-out scenario validation with per-case, mean, and worst loss.
10. External validation and optional second-stage filtering of every retained parameter candidate.
11. Common-seed central-difference metric sensitivity and held-out loss slope/curvature reports.
12. An explicit model registry for generic, planner, POMDP, rule-engine, and population adapters.

The platform also includes a `GroundedGoalDecisionModel` adapter that reuses the existing grounded finite-goal softmax implementation. It samples a selected goal from the existing grounded choice distribution and emits a canonical trace. No second goal-scoring formula is introduced.

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
Calibration / synthetic recovery
        ↓
Accepted parameter set
        ↓
Held-out validation / sensitivity / identifiability
```

## Determinism and provenance

The runner constructs the RNG and owns run metadata. A model receives the RNG but does not choose or rewrite the seed. Equal model, scenario, parameter mapping, and seed must produce equal traces for deterministic model implementations.

Batch output order follows the input seed order. Parameter mappings are canonicalized as sorted `(name, value)` tuples. Numeric parameters, metrics, targets, weights, losses, perturbations, and diagnostics must be finite.

The registry stores already-constructed adapters only. It performs no dynamic imports and exposes no execution method. Registered models still run exclusively through `SimulationRunner`.

## Calibration semantics

For candidate parameters `θ`, seeds `ξ₁…ξₙ`, trace metric extractor `Φ`, target summary `y`, and optional nonnegative weights `w`, calibration minimizes:

```text
L(θ) = Σₖ wₖ · ( mean_j Φₖ(Sim(θ, ξⱼ)) - yₖ )²
```

The baseline implementation performs exhaustive finite grid search. Ties are resolved lexicographically by canonical parameter tuples, making results reproducible.

Repeated calibration partitions or supplies seeds as independent blocks. A candidate is accepted in one block when its loss is at most the block optimum plus an explicit non-negative tolerance. A final accepted set requires an explicit minimum fraction of accepted blocks.

This is a research baseline, not the final inference method. ABC-SMC, Bayesian optimization, neural SBI, and surrogate models can later implement the same calibration interface.

## Synthetic recovery and identifiability

Synthetic recovery uses a known parameter set to generate target metrics over observation seeds, then calibrates over a candidate grid using separate calibration seeds. The report preserves true parameters, synthetic target metrics, ranked candidate evaluations, and exact recovery status.

Failure to recover does not automatically mean an implementation bug. It may reveal insufficient metrics, stochastic uncertainty, parameter non-identifiability, or a grid that excludes the true parameters.

Identifiability is reported only relative to a finite accepted set. A coordinate is marked identified only when every retained candidate agrees on that parameter value. This does not claim global structural identifiability.

## External validation

Held-out validation evaluates fixed parameters on scenarios and seed sets excluded from calibration. It never recalibrates the candidate. Reports retain per-case error as well as mean and worst-case error so averaging cannot hide a failed scenario.

An accepted set can be ranked and optionally filtered by external mean and worst-loss thresholds. Seed-block variation reports expose the union and intersection of candidates retained under deterministic translations of all seed blocks.

## Sensitivity

Two local diagnostics are supported:

- common-seed central differences of arbitrary trace metrics;
- central differences of held-out mean and worst loss, including slope and curvature.

These are local numerical diagnostics. They do not prove global sensitivity or global identifiability.

## Error handling

The platform rejects empty seed sets, empty grids or dimensions, non-finite values, negative weights, metric schema drift, unknown target metrics, malformed accepted parameter schemas, invalid perturbation steps, duplicate held-out case identifiers, duplicate registry names, unsupported registration categories, and adapters without callable `simulate()` methods.

## Testing

The repository CI runs the full Lean build, every Lean theorem test, and the complete Python suite. Python tests cover deterministic replay, calibration, synthetic recovery, accepted sets, seed-block robustness, held-out validation, local sensitivity, explicit non-identifiability, registry behavior, grounded-goal adaptation, and evidence-retraction gating.

## Out of scope

The current milestone does not add online adaptation, MCMC, ABC-SMC, neural SBI, multiprocessing, persistent experiment storage, Go runtime integration, action execution, or narrative extraction. Those remain later independently testable milestones.
