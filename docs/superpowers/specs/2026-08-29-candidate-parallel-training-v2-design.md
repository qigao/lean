# Candidate-Parallel Training V2 Design

Date: 2026-08-29

## Status

Proposed architectural design for the next study revision after `de09518d5d034b14a7a22545af02117769297517`.

This design is performance-only. It does not change the scientific protocol, model families, parameter grids, train/selection/final partitions, seeds, metrics, losses, adequacy thresholds, separation rules, or claim scope.

The current Real External Validation Study V1 remains frozen and unchanged.

## Problem

The current training and selection path evaluates finite parameter candidates sequentially inside one Python process.

For the Feher/Hare two-stage study this means:

- reactive: 4 candidate parameter tuples;
- intentional: 16 candidate parameter tuples;
- planning: 16 candidate parameter tuples;
- every candidate is evaluated across all TRAIN cases with seeds `(101, 102)`;
- every accepted candidate is then evaluated across all SELECTION cases with seeds `(201, 202)`.

The work is embarrassingly parallel across candidate parameter tuples, but the current implementation leaves most CPU cores idle. GPU execution is not a natural fit because the runtime is composed of many Python model simulations, state transitions, metric extraction calls, and manifest-attested runs rather than large tensor kernels.

The goal of V2 is to reduce wall-clock time by parallelizing candidate evaluation while preserving exact scientific identity and deterministic report construction.

## Non-goals

V2 does not:

- change any parameter grid value;
- change any simulation seed;
- change model semantics;
- change numerical loss definitions;
- change ranking or tie-break rules;
- change the participant split;
- change the dataset or target construction;
- change the external-validation claim firewall;
- execute FINAL in the training executor;
- introduce GPU-specific code;
- make worker scheduling, process IDs, timing, or completion order part of scientific identity.

## Design choice

Use a two-layer architecture:

1. deterministic candidate shards;
2. canonical report assembly.

Parallel execution is an optional operational backend over the same shard contract. The existing public sequential API remains valid and can be implemented as the same shard evaluator with `max_workers=1`.

This is preferred over embedding an ad-hoc `ProcessPoolExecutor` directly into the existing training loop because the shard boundary provides an independently testable scientific contract and can later be reused by GitHub Actions matrix jobs or self-hosted multi-core runners.

## Core invariants

### Scientific equivalence

For identical repository identity, model source, parameter grid, targets, seeds, extractor, and loss, sequential and parallel execution must produce the same canonical scientific outputs.

At minimum, equivalence requires equality of:

- candidate parameter tuples;
- per-case losses;
- per-case run manifest hashes and their canonical ordering;
- mean loss;
- worst loss;
- candidate ranking;
- selected parameters;
- retained parameter sets;
- training manifest content hash;
- selection manifest content hash;
- frozen candidate content hash.

Parallel worker completion order must not change any of these values.

### Candidate isolation

One shard evaluates exactly one canonical candidate parameter tuple across a complete declared case set and seed set.

A shard may not:

- substitute another candidate;
- omit or duplicate a case;
- omit, add, or reorder scientific seeds in its reported lineage;
- aggregate results from another candidate;
- partially succeed after a worker failure.

### Fail closed

Assembly rejects:

- missing candidate shards;
- duplicate candidate shards;
- undeclared candidate shards;
- mismatched target/report identity;
- mismatched model identity;
- mismatched extractor or loss identity;
- mismatched repository identity;
- malformed or incomplete case coverage;
- malformed run-manifest lineage;
- any worker exception or cancelled shard.

No partial `TrainingFitReport` or `SelectionValidationReport` may be returned.

### FINAL isolation

The parallel executor introduced by this design is only for TRAIN and SELECTION workflows.

For the Feher/Hare study:

- TRAIN seeds remain exactly `(101, 102)`;
- SELECTION seeds remain exactly `(201, 202)`;
- FINAL seeds `(301, 302)` are forbidden at this executor boundary.

The locked FINAL workflow remains a separate release-gated execution path.

## Components

### 1. TrainingCandidateShard

Introduce an immutable training shard record representing evaluation of one parameter tuple across all TRAIN target cases.

Conceptual fields:

- canonical parameters;
- target report hash;
- model component identity;
- repository identity;
- simulation seeds;
- metric identity;
- loss identity;
- ordered per-case results;
- per-case calibration manifest hashes;
- ordered run-manifest hashes;
- mean loss;
- worst loss;
- shard content hash.

The shard contains scientific results and lineage only. Operational metadata such as worker PID or elapsed time does not enter the shard identity.

### 2. evaluate_training_candidate

A pure scientific orchestration function evaluates one declared candidate against every TRAIN case using the existing `SimulationRunner`, metric extractor, and loss implementation.

It must preserve the same model lifecycle and run semantics as the existing sequential calibration path.

No ranking occurs inside this function.

### 3. assemble_training_fit_report

The assembler receives:

- the original target report;
- the original declared finite candidate set;
- model, metric, loss, and seed identities;
- one complete shard for each candidate.

It validates the shard set and sorts candidates canonically by the existing ranking rule:

1. `mean_loss`;
2. `worst_loss`;
3. canonical parameter tuple.

It then constructs `TrainingFitReport` using deterministic parent-hash ordering independent of worker completion order.

The resulting report must be content-equivalent to the sequential reference implementation.

### 4. SelectionCandidateShard

SELECTION has the same parallelism opportunity because the accepted parameter set is currently evaluated candidate by candidate.

Introduce an immutable selection shard representing one accepted candidate evaluated over the complete selection-validation suite.

It binds:

- candidate parameters;
- accepted-parameter-set identity;
- suite identity;
- model/repository identity;
- seeds `(201, 202)`;
- extractor and Brier loss identity;
- per-case held-out evaluations and run lineage;
- mean and worst loss;
- shard content hash.

### 5. assemble_selection_validation_report

The selection assembler verifies complete accepted-candidate coverage, reconstructs the same canonical evaluation ordering and retained set, and applies the existing selection rule without modification.

It returns the same `SelectionValidationReport` and therefore the same `FrozenModelSpec` as sequential execution.

### 6. CandidateExecutor

Introduce a small operational scheduling interface whose responsibility is only to execute independent shard callables and return all successful shard values.

Required backends:

- sequential backend;
- bounded local process backend.

The bounded local process backend chooses no more workers than:

`min(requested_max_workers, available_cpu_count, candidate_count)`.

No worker count is embedded in scientific manifests.

A future GitHub Actions matrix backend may consume the same shard serialization contract without changing training or selection scientific logic.

## Process model

Candidate-level process isolation is preferred over case-level or seed-level sharding.

A worker owns one complete candidate because this:

- minimizes inter-process communication;
- keeps case and seed ordering local and deterministic;
- avoids splitting one candidate's lineage across many process boundaries;
- makes worker failure correspond to one atomic shard;
- aligns directly with the finite-grid scientific unit.

The parent process performs all canonical merge and report construction.

## Multiprocessing constraints

The implementation must support Linux GitHub-hosted runners and ordinary local development.

The design must not rely on mutable global state inherited only through `fork`.

Worker inputs must be explicitly reconstructable or safely serializable. If the existing model source cannot be reliably pickled, the process backend must pass a stable family/source descriptor and reconstruct the model source inside the worker rather than weaken process isolation.

The sequential backend is the portability fallback and scientific reference path.

## Repository and implementation attestation

Every simulation trace must continue to bind the intended repository identity and existing implementation attestation.

Parallel workers must not accidentally measure:

- a helper workflow commit instead of the frozen study revision;
- an untracked generated file as production code;
- a different installed package copy;
- a different model source implementation.

The parent validates that shard lineage agrees on repository identity and model implementation identity before assembly.

Operational executor code may have its own implementation identity, but worker count and scheduling order do not affect the scientific result hash.

## Deterministic ordering

The following orderings are scientific and remain canonical:

- parameter names and candidate tuples;
- target case order;
- seed order;
- run-manifest hashes within a case;
- candidate ranking and lexical tie-breaks;
- accepted and retained parameter sets.

The following are operational and must not affect scientific identity:

- task submission order;
- worker assignment;
- process ID;
- CPU core;
- start/finish timestamp;
- completion order;
- queue depth;
- retry bookkeeping outside a failed scientific attempt.

## Error semantics

A worker exception is propagated as a typed/structured candidate execution failure at the parent boundary.

The parent:

1. cancels remaining not-yet-required operational work where possible;
2. returns no assembled scientific report;
3. records no partial candidate ranking as authoritative;
4. permits an operational retry only with exactly the same scientific inputs and repository revision.

A retry with changed model code, parameter grid, seeds, targets, loss, or scientific protocol is a new study revision.

## Testing strategy

Implementation follows strict RED to GREEN.

### Training equivalence tests

Add tests that prove:

- sequential reference output equals candidate-shard assembly;
- `max_workers=1` equals the reference path;
- `max_workers=4` equals `max_workers=1`;
- deliberately reversed/randomized completion order leaves the report hash unchanged;
- every candidate's per-case losses and run-manifest hashes are identical;
- duplicate/missing/extra shards fail closed;
- one worker exception produces no `TrainingFitReport`.

### Selection equivalence tests

Add the same properties for selection:

- accepted candidate evaluations are identical;
- retained set is identical;
- selected parameters are identical;
- `SelectionValidationReport` hash is identical;
- `FrozenModelSpec` hash is identical;
- incomplete or forged shards fail closed.

### Seed firewall tests

For the Feher/Hare wrapper, prove:

- training requests only `101, 102`;
- selection requests only `201, 202`;
- `301, 302` cannot enter the parallel training/selection executor;
- no FINAL model execution occurs as a side effect of protocol construction.

### Cross-platform/process tests

At least one CI test must exercise a real process backend rather than a thread-only fake.

Tests must avoid depending on process completion order or wall-clock timing.

### Performance acceptance

Performance is a secondary acceptance criterion after scientific equivalence.

Use a deterministic medium-size benchmark fixture to compare sequential and bounded-process execution. The benchmark should report speedup but should not make fragile wall-clock ratios a required ordinary CI gate.

For the real 16-candidate families, the operational target is substantial wall-clock reduction on multi-core hardware, with no scientific identity drift.

## Public API strategy

Do not break the existing public training/selection APIs.

The existing sequential functions remain valid defaults. Parallelism should be opt-in through an explicit executor/backend or a new wrapper, rather than silently changing process behavior for all callers.

This allows:

- old tests and small workloads to remain simple;
- exact sequential replay;
- explicit study-level choice of execution backend;
- future matrix/sharded orchestration without changing scientific report types.

## GitHub Actions strategy

Phase 1 uses the bounded local process backend on one runner.

Phase 2 may add matrix sharding for expensive real studies:

- one matrix job evaluates one or more candidate shards;
- jobs upload metadata-only shard artifacts;
- a merge job verifies exact shard-set completeness and canonical identities;
- the merge job constructs the same report as the local sequential/process backends.

Matrix execution must reuse the production shard schema and assembler. Scientific merge logic must not be duplicated in YAML or helper-only scripts.

## Migration plan

1. Branch from the frozen V1 revision.
2. Add test-only RED for training shard equivalence and failure semantics.
3. Add minimal training shard/assembler implementation.
4. Add bounded process executor behind explicit opt-in.
5. Add test-only RED for selection shard equivalence.
6. Add selection shard/assembler implementation.
7. Add Feher/Hare seed-firewall integration tests.
8. Verify full Python and Lean suites.
9. Run an exact-head synthetic sequential-vs-parallel evidence comparison.
10. Only after all hashes match, create a new real-study revision and rerun source/readiness/preregistration gates before using parallel execution on real TRAIN/SELECTION.

## Success criteria

V2 is complete when:

- sequential and parallel scientific reports are hash-identical on locked fixtures;
- worker completion order cannot change outputs;
- failures cannot produce partial authoritative reports;
- TRAIN/SELECTION seed boundaries remain exact;
- FINAL remains isolated;
- public sequential behavior remains supported;
- real multi-core execution shows a meaningful wall-clock improvement;
- no claim-scope or scientific-protocol change is introduced.
