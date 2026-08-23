# Finite Prison-Scenario POMDP Adapter Design

## Status

Approved by the user’s explicit `go` after the trusted Python runtime gates were established. This change is Python-only and does not modify `NarrativeDynamics/Core` or any Lean theorem.

## Purpose

Add the first finite POMDP adapter as a concrete consumer of the existing runtime boundary. The adapter models a prisoner choosing whether to inspect guard conditions and then attempt escape through a gate or tunnel under a hidden guard-alert state.

The adapter is an empirical/research hypothesis. It is not a claim that the finite state, observation, reward, or policy equations are a true description of human behavior.

## Finite Model

Hidden state:

- `quiet`
- `alert`

Observations after the optional `inspect` action:

- `clear`
- `alarm`

Actions:

- `inspect`
- `gate`
- `tunnel`

The hidden state is static during one short episode. The scenario supplies the true alert prior, sensor accuracy, route success probabilities, inspection cost, success reward, and capture cost. Model parameters supply a belief log-odds bias, risk-aversion multiplier, and inverse temperature.

The solver performs exact finite enumeration for the one-information-step horizon:

1. compute route expected utilities under the current belief;
2. compute the value of inspection by enumerating both observations and their Bayesian posteriors;
3. form a Boltzmann policy over `inspect`, `gate`, and `tunnel`;
4. if inspection is selected, update the belief from the sampled observation and form a second Boltzmann policy over the two routes;
5. sample the terminal escape outcome from the scenario’s state-conditional success probability.

The outcome also includes deterministic expected policy/escape/utility coordinates. Calibration uses these expectation coordinates, avoiding Monte Carlo noise in parameter-identification tests while preserving seeded stochastic episode traces.

## Runtime Boundary

The adapter exposes:

- an importable model factory for `SubprocessModel`;
- a declared `ModelContract` with parameter, scenario, event, and outcome schemas;
- an unpinned contract that must be bound by `TrustedExecutionPolicy` outside the adapter module;
- a stable metric extractor for calibration and held-out validation.

Execution flow:

```text
external policy pin
  -> PolicyBoundSource
  -> ModelRegistry(kind=POMDP)
  -> SimulationRunner
  -> implementation measurement match
  -> parameter/scenario validation
  -> fresh subprocess
  -> event/outcome validation
  -> ResultArtifact + ExperimentManifest
```

The adapter never embeds or self-generates its trusted expected implementation hash.

## Events and Outcome

Events are fixed-schema records for prior belief, initial decision, optional observation, posterior belief, optional route decision, and terminal outcome. The outcome contains the sampled episode result plus deterministic expectation coordinates used by metrics.

## Validation Workflow

Tests exercise:

- exact Bayesian updates and finite policy normalization;
- same-seed replay and different-seed stochastic variation;
- production-ready policy-bound POMDP registration;
- subprocess isolation and all four schema attestations;
- finite-grid synthetic recovery;
- repeated seed-block acceptance;
- selection-validation and untouched final-test APIs;
- aggregate final-report artifact integrity.

## Non-goals

- no generic POMDP framework;
- no infinite horizon or approximate solver;
- no Lean formalization of the POMDP;
- no claim of causal or psychological validity;
- no OS sandbox expansion;
- no relaxation of legacy/runtime gates.
