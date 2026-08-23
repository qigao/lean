# Finite Prison-Scenario POMDP Adapter Design

## Status

Approved by the user’s explicit `go` after the trusted Python runtime gates were established. This change is Python-only and does not modify `NarrativeDynamics/Core` or any Lean theorem.

## Purpose

Add the first finite POMDP adapter as a concrete consumer of the existing runtime boundary. The adapter models a prisoner choosing whether to scout guard conditions, attempt escape, or submit under a hidden weak/strong guard state.

The adapter is an empirical/research hypothesis. It is not a claim that the finite state, observation, reward, or policy equations are a true description of human behavior.

## Finite Model

Hidden state:

- `weak`
- `strong`

Observations after the optional `scout` action:

- `clear`
- `alarm`

Initial actions:

- `scout`
- `escape`
- `submit`

Terminal actions after scouting:

- `escape`
- `submit`

The scenario supplies the prior probability of a weak guard, signal accuracy, symmetric guard-state persistence, escape reward, capture cost, submit reward, scouting cost, discount, and horizon. `guard_persistence = p` means a weak guard remains weak with probability `p`, while a strong guard becomes weak with probability `1 - p`.

The only fitted model parameter is a positive inverse temperature `beta`.

For horizon 1, scouting is unavailable and the solver forms a softmax policy over direct escape and submit values. For horizon 2, it performs exact finite enumeration:

1. compute direct escape and submit values under the current belief;
2. enumerate clear/alarm observation masses and Bayesian posteriors;
3. propagate each posterior through the guard-persistence transition;
4. form a terminal softmax over escape and submit;
5. integrate the discounted value of scouting;
6. form the initial softmax over scout, escape, and submit;
7. sample one episode using only the runner-supplied seeded RNG.

The outcome also includes deterministic expected terminal-action, escape-success, and utility coordinates. Those expectations integrate the true hidden-state prior, observation process, transition, and selected terminal policy, allowing finite-grid calibration without Monte Carlo noise while preserving stochastic episode traces.

Softmax evaluation subtracts the maximum action value before exponentiation. The floating-point normalization residual is assigned to the largest policy coordinate, preventing an underflowed tail action from receiving a tiny negative mass.

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

Events use the fixed vocabulary:

- `belief_state`;
- `initial_decision`;
- optional `observation_received`;
- optional `terminal_decision`;
- `episode_ended`.

The outcome contains the sampled initial and terminal actions, optional signal and posterior, sampled escape/utility result, and deterministic expectation coordinates used by the metric extractor.

## Validation Workflow

Tests exercise:

- finite-horizon information value and policy normalization;
- horizon-1 removal of scouting;
- exact same-seed replay;
- non-negative normalized policy mass under an underflow edge case;
- production-ready policy-bound POMDP registration;
- fresh-process isolation and all four schema attestations;
- invalid-scenario rejection before worker launch;
- finite-grid recovery of `beta`;
- selection-validation and untouched final-test APIs;
- aggregate final-report artifact integrity.

## Non-goals

- no generic POMDP framework;
- no infinite horizon or approximate solver;
- no Lean formalization of the POMDP;
- no claim of causal or psychological validity;
- no OS sandbox expansion;
- no relaxation of legacy/runtime gates.
