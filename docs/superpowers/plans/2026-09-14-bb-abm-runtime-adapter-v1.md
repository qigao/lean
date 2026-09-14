# BB-driven ABM Runtime Adapter V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute authored finite BB birth/idle traces in Python, creating fresh agents, retaining old states, running one existing V1 propagation round per tick, and reporting exact BB trace mass with inspectable model boundaries.

**Architecture:** Add frozen raw/result records and one checked Fraction-based BB kernel in two new modules. A private tick adapter projects growth into an ordinary V1 model epoch; one public replay function owns validation, history, and atomic failure. Compare complete finite histories with a separate corpus computed by the existing Lean replay.

**Tech Stack:** Python 3.13, standard-library dataclasses/Fraction/unittest, existing V1 contracts and `simulate_round`, Lean 4.32.0/mathlib v4.32.0, existing trust auditor and GitHub Actions.

**Spec:** [BB-driven ABM Runtime Adapter V1 Design](../specs/2026-09-14-bb-abm-runtime-adapter-v1-design.md), pinned at [`aa10d64694205f20a497432e779c22239d7e896b`](https://github.com/qigao/lean/blob/aa10d64694205f20a497432e779c22239d7e896b/docs/superpowers/specs/2026-09-14-bb-abm-runtime-adapter-v1-design.md). Source foundation: [`fb17c2b7d5ac721fdae90051832c9bb703a7c230`](https://github.com/qigao/lean/commit/fb17c2b7d5ac721fdae90051832c9bb703a7c230), [PR #73](https://github.com/qigao/lean/pull/73).

**Planning status (2026-09-14):** Detailed plan following the user's instruction to continue [PR #74](https://github.com/qigao/lean/pull/74). The design remains a proposed design; this document does not claim an independent design approval. No implementation task below has run. Code blocks prescribe interfaces, algorithms and tests; they are not execution evidence. The earlier foundation plan is complete and must not be rerun as unfinished work.

## Global Constraints

The following requirements are copied from the pinned design. Task-local refinements below do not weaken them.

- The first runtime accepts an authored sequence of ordered target choices.
- The Python BB kernel is a new implementation checked against the existing Lean reference.
- Fitness and BB trace masses use `fractions.Fraction`.
- Accept exact integer or Fraction fitness values, rejecting booleans and implicit float-to-fitness conversion.
- Never silently drop a duplicate.
- Generate exactly two active `SocialEdge` channels with influence `1.0` and one fixed relation type `"bb"` for each undirected edge.
- Keep every old `NetworkAgentSpec` field fixed, including role and adoption threshold.
- A newborn request contains no exposure-count override.
- Broadcasting is derived from the actual V1 belief and threshold, never accepted as an independent raw flag.
- Call the existing `simulate_round(model, post_growth_population)` exactly once.
- The first public execution API is a finite, side-effect-free `replay_bb_population(seed, m, ticks)`.
- No later malformed request can displace the first failure.
- Failure returns no successful prefix, partial topology, population, allocated ID or accumulated mass, and input records remain unchanged.
- Hashes identify content; they are not evidence that a transition was computed correctly.
- Preserve the existing eight-vector `bb_abm_v1.json` unchanged.
- Keep existing V1/V5/V19 contracts, production persistence, all 35 foundation audit requirements, toolchain/dependencies and workflow selection/checkout behavior unchanged.
- Any new Lean fixture must retain bounded execution and the existing trust allowlist; do not disable old gates to admit a new exporter.

---

## File ownership and execution order

| Task | Files created | Existing files modified | Depends on |
| --- | --- | --- | --- |
| 1: Records | `narrative_dynamics/abm/bb_runtime_contracts.py`, `tests/bb_runtime_fixtures.py`, `tests/test_network_abm_bb_runtime_contracts.py` | None | Pinned foundation |
| 2: Exact BB kernel | `narrative_dynamics/abm/bb_runtime.py`, `tests/test_network_abm_bb_runtime_kernel.py` | None | 1 |
| 3: One tick | `tests/test_network_abm_bb_runtime_round.py` | New runtime module from 2 | 1, 2 |
| 4: Finite replay | `tests/test_network_abm_bb_runtime_replay.py` | New runtime module from 2 | 1–3 |
| 5: Reference and permanent gate | `NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean`, `NarrativeDynamics/Tests/FitnessABMRuntime.lean`, `conformance/bb_abm_runtime_v1.json`, `tests/test_network_abm_bb_runtime_conformance.py`, `tools/check_fitness_abm_runtime.sh` | `.github/workflows/proof.yml` | 1–4 |
| 6: Example and acceptance | `examples/bb_abm_runtime.py` | `README.md`, this plan's execution record | 1–5 |

These are sequential review units. Task 1 tests construct records using existing V1 APIs; they do not import the unfinished replay function. Task 2 tests exercise private BB helpers directly. Task 3 tests exercise the private one-tick helper. Only Task 4 makes the public execution API available. Keep each intermediate test command limited to completed units, then run full discovery at acceptance.

Use direct module imports for this first adapter. Leave `narrative_dynamics/abm/__init__.py` and its exact V21.3 export test unchanged. There is no new public advance/resume function. No new package, service, database, sampler or command-line framework is needed.

## Preparation and evidence

- [ ] Read the pinned design and the current source files listed in its section 2. Check the working tree before editing, use an isolated implementation branch, and record its base SHA. Keep planning and implementation scope distinguishable in the PR description.
- [ ] Record actual environment versions and a focused existing baseline:

```bash
git status --short --branch
git rev-parse HEAD
python3 --version
cat lean-toolchain
lake env lean --version
git -C .lake/packages/mathlib rev-parse HEAD
python3 -m unittest tests.test_network_abm_contracts tests.test_network_abm_simulation tests.test_network_abm_bb_conformance -v
```

A missing Lean executable/cache is an environment prerequisite, not a RED test and not a pass. Use the pinned ordinary setup or actual CI, preserving dependencies. Do not substitute an old #73 run for new implementation evidence. Capture each real RED diagnostic, GREEN command and checked SHA in the task execution record. Do not check boxes on the strength of code inspection alone.

## Shared interfaces and error vocabulary

All new data classes use `@dataclass(frozen=True)`. Raw input records deliberately have no eager domain validation; their fields are annotated `object` so malformed values can reach the ordered checked boundary. Raw records are request containers, not certificates. Valid execution accepts tuple containers only, following the existing V1 contract style; reject lists at the field's validation position rather than coercing them. Invalid nested objects are never mutated or recursively serialized on an error path.

| Record | Exact fields in constructor order |
| --- | --- |
| `BBRuntimeRawSeed` | `node_count: object`, `fitness: object`, `edges: object` |
| `BBRuntimeAgent` | `agent_id: object`, `role: object`, `receptivity: object`, `broadcast_threshold: object`, `belief: object`, `exposure_count: object = 0`, `adoption_threshold: object = 0.5` |
| `BBRuntimeNewborn` | `agent_id: object`, `role: object`, `receptivity: object`, `broadcast_threshold: object`, `belief: object`, `adoption_threshold: object = 0.5` |
| `BBRuntimeBirth` | `fitness: object`, `targets: object`, `agent: object` |
| `BBRuntimeTick` | `kind: object = "idle"`, `birth: object = None` |
| `BBRuntimeSeed` | `network: object`, `agents: object`, `model_id: object = "bb-runtime"`, `version: object = "1"` |
| `BBRuntimeConfig` | `model_id: str`, `version: str`, `m: int`, `seed_node_count: int`, `seed_edge_count: int` |
| `BBRuntimeTopology` | `fitness: tuple[Fraction, ...]`, `edges: tuple[tuple[int, int], ...]`; property `node_count = len(fitness)` |
| `BBRuntimeFrame` | `config: BBRuntimeConfig`, `tick_count: int`, `birth_count: int`, `agent_ids: tuple[str, ...]`, `topology: BBRuntimeTopology`, `model: NetworkABMModel`, `population: PopulationState`, `trace_mass: Fraction`, `parent_frame_hash: str | None`, `applied_tick: BBRuntimeTick | None` |
| `BBRuntimeTransition` | `prior: BBRuntimeFrame`, `tick: BBRuntimeTick`, `post_growth_population: PopulationState`, `round_result: NetworkRoundResult`, `tick_mass: Fraction`, `next_frame: BBRuntimeFrame` |
| `BBRuntimeReplay` | `initial: BBRuntimeFrame`, `transitions: tuple[BBRuntimeTransition, ...]`, `final: BBRuntimeFrame` |
| `BBRuntimeError` | `stage: str`, `code: str`, `field: str | None = None`, `agent_index: int | None = None`, `tick_index: int | None = None`, `birth_index: int | None = None`, `bb_cause: str | None = None`, `expected: int | None = None`, `actual: int | None = None` |

The schema discriminator in serialized wrapper records is `"bb-abm-runtime-v1"`, distinct from the stable caller's V1 `version`. A frame's `epoch` property is its `birth_count`. A transition's `tick_index`, `birth_index` and `global_round` properties are respectively `prior.tick_count`, `prior.birth_count`, and `prior.tick_count + 1`; no independently stored counters.

Result records provide `to_dict() -> dict[str, object]` and `content_hash` using `stable_content_hash`. Topology/config/error also provide `to_dict`; an error has no state/mass/hash-of-prefix fields. Raw tick serialization is used only for a normalized validated applied tick. No `from_dict`, resume API, public probability argument or generic object serializer is introduced.

Stable error mapping:

| Stage | Codes and fields | Lean relation |
| --- | --- | --- |
| `seed_network` | `invalidNodeCount`, `fitnessSizeMismatch`, `nonpositiveFitness`, `invalidEdge`, `duplicateEdge`, `disconnectedSeed` | `.seedNetwork cause`; `bb_cause == code` |
| `initial_m` | `initialM`, field `m` | `.initialM`; `bb_cause = None` |
| `seed_agents` | `seedAgentCount`, field `agents`, expected/actual counts | `.seedAgentCount` |
| `seed_agent` | `invalidAgentValue`, field `receptivity`, `broadcast_threshold`, or `belief`; numeric `agent_index` | `.seedAgent index field`; map `threshold` to `broadcast_threshold` |
| `tick_network` | `invalidM`, `nonpositiveFitness`, `targetCountMismatch`, `targetOutOfRange`, `duplicateTarget` | `.tickNetwork tickIndex birthIndex cause`; `bb_cause == code` |
| `tick_agent` | `invalidAgentValue` and same three scalar fields | `.tickAgent`; same threshold mapping |
| At the field's owning stage | `invalidType`, `invalidExposure`, `invalidText`, `invalidAdoptionThreshold`, `duplicateAgentId` | Python-specific, `bb_cause = None` |
| `seed_identity` | `invalidText`/`invalidType`, field `model_id` then `version` | Python-specific; after all seed agents |
| `ticks` | `invalidType`, field `ticks` | Python-specific; after complete genesis validation |
| `tick` | `invalidType`, `invalidTickKind`, `unexpectedBirthData`, `missingBirthData` | Python-specific; current tick/birth indices |

Use `invalidType` for a wrong Python domain (including booleans where numbers are required), semantic codes for a correctly typed value outside its domain. Nonfinite agent floats and finite out-of-range values use `invalidAgentValue`; invalid natural exposure values use `invalidExposure`. Negative numeric endpoints/targets use `invalidEdge`/`targetOutOfRange`; bool or non-int endpoints/targets use `invalidType`. Preserve exact failure stage/field, not exception-message parsing. Never catch arbitrary implementation exceptions and label them invalid user input.

## Task 1: Frozen records, numeric identity and structural lineage

**Files:** Create the three Task 1 files in the ownership table.

**Consumes:** Existing `NetworkABMModel`, `NetworkAgentSpec`, `NetworkAgentState`, `SocialNetwork`, `SocialEdge`, `PopulationState`, `NetworkRoundResult`, `stable_content_hash`.

**Produces:** All records above; in `tests/bb_runtime_fixtures.py`, `seed(*, fitness=(1, 1), ids=("0", "1"), exposures=(0, 0)) -> BBRuntimeSeed`, `birth(agent_id, targets, *, fitness=1, belief=0, receptivity=1, threshold=0.5) -> BBRuntimeTick`, and `genesis_frame(*, ids=("0", "1"), exposures=(0, 0)) -> BBRuntimeFrame`. The last helper constructs known-valid V1 objects directly, without future runtime helpers.

- [ ] Write the raw fixture helpers and these initial tests with explicit imports from the new contracts module. `genesis_frame` uses unit fitness, edge `(0, 1)`, profiles `(1, 0.5, 0.5)`, beliefs `(1, 0)`, direct `NetworkAgentState` construction, V1 local round zero, mass one, and no wrapper parent/applied tick.

```python
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import unittest

from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeError, BBRuntimeReplay, BBRuntimeTopology,
)
from tests.bb_runtime_fixtures import genesis_frame


class BBRuntimeContractsTests(unittest.TestCase):
    def test_empty_wrapper_and_numeric_registry(self):
        frame = genesis_frame(ids=("z", "a"), exposures=(7, 3))
        replay = BBRuntimeReplay(frame, (), frame)
        self.assertEqual(replay.final, frame)
        self.assertEqual(frame.agent_ids, ("z", "a"))
        self.assertEqual(tuple(a.agent_id for a in frame.model.agents), ("a", "z"))
        self.assertEqual(frame.trace_mass, Fraction(1))
        self.assertEqual(frame.population.round_index, 0)
        with self.assertRaises(FrozenInstanceError):
            frame.tick_count = 1

    def test_rejects_forged_model_and_roster_bindings(self):
        frame = genesis_frame()
        wrong_hash = replace(frame.population, model_hash="sha256:" + "0" * 64)
        with self.assertRaises(ValueError):
            replace(frame, population=wrong_hash)
        with self.assertRaises(ValueError):
            replace(frame, agent_ids=("0", "ghost"))
        with self.assertRaises(ValueError):
            replace(frame, trace_mass=Fraction(1, 2))
        with self.assertRaises(ValueError):
            replace(frame, parent_frame_hash="sha256:" + "1" * 64)

    def test_exact_canonical_topology_and_detached_serialization(self):
        frame = genesis_frame()
        self.assertEqual(frame.topology.to_dict()["fitness"], ["1", "1"])
        with self.assertRaises(ValueError):
            BBRuntimeTopology((Fraction(1), Fraction(1)), ((1, 0),))
        before = frame.content_hash
        detached = frame.to_dict()
        detached["agent_ids"].append("foreign")
        self.assertEqual(frame.content_hash, before)
        self.assertEqual(genesis_frame().content_hash, before)

    def test_error_has_no_success_prefix_payload(self):
        error = BBRuntimeError(
            "tick_network", "targetOutOfRange", field="targets",
            tick_index=2, birth_index=0, bb_cause="targetOutOfRange",
        )
        self.assertEqual((error.tick_index, error.birth_index), (2, 0))
        self.assertFalse({"final", "transitions", "trace_mass", "population"}
                         & error.to_dict().keys())
```

- [ ] Run RED: `python3 -m unittest tests.test_network_abm_bb_runtime_contracts -v`. The first expected failure is the absent new module. Once records exist, retain the same behavioral assertions; do not replace them with construction-only smoke tests.
- [ ] Implement records and serialization. For canonical rationals use this one formatter in `bb_runtime_contracts.py`:

```python
def _fraction_text(value: Fraction) -> str:
    if not isinstance(value, Fraction):
        raise TypeError("canonical rational must be Fraction")
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
```

Implement `BBRuntimeConfig` validation with nonempty ID/version, natural non-bool counts, `2 <= seed_node_count`, `1 <= m <= seed_node_count`, and `seed_node_count - 1 <= seed_edge_count <= seed_node_count * (seed_node_count - 1) // 2`. Topology validates exact positive Fraction values, `n >= 2`, and a sorted tuple of unique canonical pairs `0 <= u < v < n`. Connectivity is computed by Task 2's seed parser; a structural record is not a connectivity certificate or an executable checkpoint.

Frame validation must check exact numeric registry length/uniqueness, topology/model bidirectional unit-edge equality by **ID mapping**, config/model identity, V1 model hash and roster, broadcasting/profile consistency, natural counts with `birth_count <= tick_count`, `n = n0 + birth_count`, and `E = E0 + m * birth_count`. Require Fraction `0 < trace_mass <= 1`. Genesis requires both counts zero, V1 round zero/parent `None`, mass one, wrapper parent/applied tick `None`. Later frames require a valid wrapper parent hash, a normalized applied tick, and positive V1 local round. A birth frame has local round one. Transition validation below supplies predecessor-dependent checks; a standalone frame cannot verify its own historical probability.

Frame `to_dict` keys are `schema`, `config`, `tick_count`, `birth_count`, `agent_ids`, `topology`, `model`, `population`, `trace_mass`, `parent_frame_hash`, `applied_tick`. Use complete nested model/population `to_dict` values and new lists/dicts, not references to internal mutable containers. Hash exactly this dictionary with the existing typed helper.

Transition validation checks:

1. `next_frame.config == prior.config`; global ticks increment by one, births by one iff birth; `next_frame.parent_frame_hash == prior.content_hash`; applied tick equals the transition's normalized tick.
2. `round_result.prior_state == post_growth_population`; `round_result.next_state == next_frame.population`; round model identity binds `next_frame.model`; local round and V1 parent chain obey existing contracts.
3. Idle retains model/topology/registry, uses `prior.population` as input, and has tick mass one. Birth appends exactly one new ID/fitness, retains old profiles and all old agent states by ID, adds precisely the supplied target edges, and uses local-round-zero input with newborn exposure zero and derived broadcasting.
4. `next_frame.trace_mass == prior.trace_mass * tick_mass`, exact Fraction range, and normalized birth metadata matches the new profile/state. Do not recompute propagation or add a second BB probability algorithm to constructors; the executor owns numerical correctness.

Transition serialization uses `schema`, `prior_frame_hash`, `tick`, `post_growth_population`, `round_result`, `tick_mass`, `next_frame_hash`. Replay serialization includes the initial frame and an ordered list pairing each transition dictionary with its next-frame dictionary, plus `final_frame_hash`; thus it binds every complete frame without recursive history duplication. Replay validates a genesis initial frame, exact transition chaining, and the final tail; zero transitions require `final == initial`.

- [ ] Add record tests using a manually constructed birth transition and one existing `simulate_round` call: reject mismatched round input, changed old belief/exposure/profile, nonzero newborn exposure, missing/reused ID, swapped target metadata, an idle with changed topology, wrong cumulative multiplication, and a discontinuous replay tail. For equal-fitness `m=2` births in target orders `(0,1)` and `(1,0)`, require equal final model/population/mass and distinct frame/replay hashes. Reverse **derived** model/state tuple enumeration and require stable hashes; never reverse the authored numeric registry or seed-agent array to claim enumeration invariance.
- [ ] Run GREEN and commit the completed record unit:

```bash
python3 -m unittest tests.test_network_abm_bb_runtime_contracts -v
git diff --check
git add narrative_dynamics/abm/bb_runtime_contracts.py tests/bb_runtime_fixtures.py tests/test_network_abm_bb_runtime_contracts.py
git commit -m "feat: define BB runtime records and model-boundary lineage"
```

## Task 2: One exact checked BB birth kernel

**Files:** Create `narrative_dynamics/abm/bb_runtime.py` and `tests/test_network_abm_bb_runtime_kernel.py`.

**Consumes:** Task 1 raw seed/birth/topology/error records; `Fraction`.

**Produces (private, in the runtime module):**

| Function/record | Signature/fields |
| --- | --- |
| `_parse_bb_seed` | `(raw: object) -> BBRuntimeTopology | BBRuntimeError` |
| `_check_m` | `(m: object, n: int, *, stage: str, tick_index: int | None = None, birth_index: int | None = None) -> int | BBRuntimeError` |
| `_CheckedBBBirth` | frozen fields `fitness: Fraction`, `targets: tuple[int, ...]`, `tick_mass: Fraction` |
| `_check_birth` | `(topology: BBRuntimeTopology, m: object, raw: BBRuntimeBirth, *, tick_index: int, birth_index: int) -> _CheckedBBBirth | BBRuntimeError` |
| `_apply_birth` | `(topology: BBRuntimeTopology, checked: _CheckedBBBirth) -> BBRuntimeTopology` |

`_check_birth` validates only BB fields and computes mass; it does not allocate a successor model, append an ID, or inspect newborn profile fields. Task 3 validates the rest of the request before calling `_apply_birth`. Do not introduce a second weighted-choice helper with different semantics.

- [ ] Write these RED cases, plus the validation matrix below:

```python
from fractions import Fraction
import unittest

from narrative_dynamics.abm.bb_runtime import _apply_birth, _check_birth, _parse_bb_seed
from narrative_dynamics.abm.bb_runtime_contracts import BBRuntimeError
from tests.bb_runtime_fixtures import birth, seed


class BBRuntimeKernelTests(unittest.TestCase):
    def test_ordered_weighted_birth_uses_frozen_undirected_degrees(self):
        topology = _parse_bb_seed(seed(fitness=(1, 3)).network)
        checks = [
            _check_birth(topology, 2, birth("2", order).birth,
                         tick_index=0, birth_index=0)
            for order in ((0, 1), (1, 0))
        ]
        self.assertEqual([c.tick_mass for c in checks],
                         [Fraction(1, 4), Fraction(3, 4)])
        left, right = (_apply_birth(topology, c) for c in checks)
        self.assertEqual(left, right)
        self.assertEqual(left.edges, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(topology.edges, ((0, 1),))

    def test_second_birth_uses_new_degrees_only_on_the_next_tick(self):
        topology = _parse_bb_seed(seed().network)
        first = _check_birth(topology, 1, birth("2", (1,)).birth,
                             tick_index=0, birth_index=0)
        grown = _apply_birth(topology, first)
        second = _check_birth(grown, 1, birth("3", (2,)).birth,
                              tick_index=1, birth_index=1)
        self.assertEqual((first.tick_mass, second.tick_mass),
                         (Fraction(1, 2), Fraction(1, 4)))
        self.assertEqual(first.tick_mass * second.tick_mass, Fraction(1, 8))

    def test_network_fields_fail_before_any_newborn_agent_field(self):
        topology = _parse_bb_seed(seed().network)
        raw = birth("2", (0, 0), fitness=0, belief=2).birth
        error = _check_birth(topology, 2, raw, tick_index=4, birth_index=1)
        self.assertIsInstance(error, BBRuntimeError)
        self.assertEqual((error.stage, error.code, error.bb_cause),
                         ("tick_network", "nonpositiveFitness", "nonpositiveFitness"))
        self.assertEqual((error.tick_index, error.birth_index), (4, 1))
```

- [ ] Run RED: `python3 -m unittest tests.test_network_abm_bb_runtime_kernel -v`; expect missing private kernel declarations, then real arithmetic/precedence failures as implementation proceeds.
- [ ] Implement seed validation in this order: raw-record shape; non-bool integer `node_count >= 2`; tuple fitness and exact dimension; fitness entries in numeric order, each int/Fraction but not bool, then positive; tuple edges and each pair's shape/endpoints/self-edge check over the **whole edge list**; canonical duplicate detection over the accepted pairs; connectivity by BFS from node zero. Reject a later invalid edge before an earlier duplicate because Lean checks all endpoint validity before distinctness. Only after validation return sorted canonical edges and Fraction fitness.

For BFS, construct `adjacency = [set() for _ in range(n)]`, add both endpoints for each checked undirected edge, maintain a visited set and stack starting at zero, and require `len(visited) == n`. Degree is the length of each undirected adjacency set. No directed-channel count, supplied degree, connectivity flag or cached normalizer is accepted.

At a birth, `_check_m` precedes positive exact newborn fitness, tuple target dimension, all bounds/type checks, then distinctness. A duplicate target followed by an out-of-range target reports the bounds error. Retain the authored tuple order. Compute mass only after those checks, using:

```python
degree = [0] * topology.node_count
for u, v in topology.edges:
    degree[u] += 1
    degree[v] += 1
weights = tuple(f * d for f, d in zip(topology.fitness, degree))
remaining = sum(weights, Fraction(0))
mass = Fraction(1)
for target in targets:
    mass *= weights[target] / remaining
    remaining -= weights[target]
return _CheckedBBBirth(new_fitness, targets, mass)
```

Here `targets` and `new_fitness` are the validated tuple/Fraction locals. For `m=n`, the final subtraction may yield zero; there is no subsequent division. A connected seed with positive fitness makes each required denominator positive. An impossible zero denominator after checked internal input is an invariant failure, not a fabricated successful mass.

Implement extension only as:

```python
def _apply_birth(topology: BBRuntimeTopology,
                 checked: _CheckedBBBirth) -> BBRuntimeTopology:
    newborn = topology.node_count
    return BBRuntimeTopology(
        topology.fitness + (checked.fitness,),
        tuple(sorted(topology.edges + tuple((t, newborn) for t in checked.targets))),
    )
```

- [ ] Complete table-driven tests with explicit expected stage/code/field: node count before fitness size; fitness size before positivity; zero/negative fitness before bad edges; edge bounds before reversed duplicates; duplicate edges before connectedness; disconnected seed; `m=0`, `m>n`, `m=True`; float and bool fitness rejected; Fraction fitness retained exactly; `m=n`; wrong target count before bounds; bounds before duplicates; reversed seed orientation canonicalized only on success. Test nonuniform three-node degrees with fitness `(2,3,5)` on path `0--1--2`: weights `(2,6,5)`, `m=2`, `(1,0)` mass `12/91` and `(0,1)` mass `12/143`, while extension gives the same graph. This distinguishes frozen weighted sampling without replacement from unweighted or partially updated alternatives.
- [ ] Run GREEN and commit:

```bash
python3 -m unittest tests.test_network_abm_bb_runtime_contracts tests.test_network_abm_bb_runtime_kernel -v
git diff --check
git add narrative_dynamics/abm/bb_runtime.py tests/test_network_abm_bb_runtime_kernel.py
git commit -m "feat: implement exact checked BB birth arithmetic"
```

## Task 3: Validate an entire tick, preserve state, then propagate once

**Files:** Modify `narrative_dynamics/abm/bb_runtime.py`; create `tests/test_network_abm_bb_runtime_round.py`.

**Consumes:** Tasks 1–2; the unchanged `simulate_round` from `narrative_dynamics.abm.simulation`.

**Produces (private, in the runtime module):**

| Function | Exact signature |
| --- | --- |
| `_parse_agent` | `(raw: object, *, newborn: bool, used_ids: frozenset[str], stage: str, agent_index: int | None = None, tick_index: int | None = None, birth_index: int | None = None) -> tuple[NetworkAgentSpec, NetworkAgentState] | BBRuntimeError` |
| `_model_for` | `(config: BBRuntimeConfig, agent_ids: tuple[str, ...], topology: BBRuntimeTopology, profiles: tuple[NetworkAgentSpec, ...]) -> NetworkABMModel` |
| `_advance_tick` | `(prior: BBRuntimeFrame, raw: object) -> BBRuntimeTransition | BBRuntimeError` |

Import the existing operator as `from narrative_dynamics.abm.simulation import simulate_round`; the call-count tests patch `narrative_dynamics.abm.bb_runtime.simulate_round` with `wraps=simulate_round`. Constructors and hashing must not call it again.

- [ ] Write these one-tick RED tests before the helper implementation:

```python
from fractions import Fraction
import unittest
from unittest.mock import patch

from narrative_dynamics.abm.bb_runtime import _advance_tick
from narrative_dynamics.abm.bb_runtime_contracts import BBRuntimeTick
from narrative_dynamics.abm.simulation import simulate_round
from tests.bb_runtime_fixtures import birth, genesis_frame


class BBRuntimeRoundTests(unittest.TestCase):
    def test_birth_retains_old_state_before_exactly_one_v1_round(self):
        initial = genesis_frame(exposures=(7, 3))
        with patch("narrative_dynamics.abm.bb_runtime.simulate_round",
                   wraps=simulate_round) as operator:
            step = _advance_tick(initial, birth("2", (1,)))
        operator.assert_called_once_with(step.next_frame.model,
                                         step.post_growth_population)
        post = {a.agent_id: a for a in step.post_growth_population.agents}
        old = {a.agent_id: a for a in initial.population.agents}
        self.assertEqual({k: post[k] for k in old}, old)
        self.assertEqual((post["2"].belief, post["2"].exposure_count), (0.0, 0))
        self.assertEqual(step.post_growth_population.round_index, 0)
        self.assertIsNone(step.post_growth_population.parent_state_hash)
        self.assertEqual(step.round_result.round_index, 1)
        self.assertEqual(step.next_frame.trace_mass, Fraction(1, 2))
        self.assertEqual(step.next_frame.population.parent_state_hash,
                         step.post_growth_population.content_hash)
        self.assertEqual(step.next_frame.parent_frame_hash, initial.content_hash)

    def test_new_receiver_relays_only_after_next_global_tick(self):
        first = _advance_tick(genesis_frame(), birth("2", (1,)))
        self.assertEqual([a.belief for a in first.next_frame.population.agents],
                         [1.0, 1.0, 0.0])
        second = _advance_tick(first.next_frame, BBRuntimeTick())
        self.assertEqual(second.post_growth_population, first.next_frame.population)
        self.assertEqual([a.belief for a in second.next_frame.population.agents],
                         [1.0, 1.0, 1.0])
        self.assertEqual([a.exposure_count for a in second.next_frame.population.agents],
                         [1, 2, 1])
        self.assertEqual(second.round_result.round_index, 2)
        self.assertEqual(second.global_round, 2)
        self.assertEqual(second.tick_mass, Fraction(1))
```

- [ ] Run RED: `python3 -m unittest tests.test_network_abm_bb_runtime_round -v`; expect missing `_advance_tick`, followed by state-retention/clock failures until the actual operator is composed correctly.
- [ ] Implement `_parse_agent` with exact order: expected raw record type; receptivity, broadcast threshold, belief; seed exposure (newborn has no such field); external ID, duplicate ID, role, adoption threshold. Shape/type failures occur at the same field position. Use the existing finite float normalization and text semantics, with explicit domain checks and structured errors; catch only the expected conversion/domain failures, including huge-int float overflow at that field. Construct `NetworkAgentSpec` only after all those checks, so its eager metadata validation cannot displace an earlier scalar/network error. Construct state directly, with broadcasting `belief >= broadcast_threshold`. Never call `initialize_population`.

The normalized birth applied to a result is `BBRuntimeTick("birth", BBRuntimeBirth(checked.fitness, checked.targets, normalized_newborn))`, where `normalized_newborn` contains validated strings and the V1-normalized float values. Do not store a raw mutable list or an unvalidated metadata object in a successful frame. Normalize idle as `BBRuntimeTick()`.

Implement `_model_for` exactly from the append-only registry and checked topology:

```python
def _model_for(config: BBRuntimeConfig, agent_ids: tuple[str, ...],
               topology: BBRuntimeTopology,
               profiles: tuple[NetworkAgentSpec, ...]) -> NetworkABMModel:
    channels = tuple(
        SocialEdge(agent_ids[source], agent_ids[target], "bb", 1.0, True)
        for u, v in topology.edges
        for source, target in ((u, v), (v, u))
    )
    return NetworkABMModel(
        config.model_id, config.version, profiles,
        SocialNetwork(agent_ids, channels),
    )
```

Implement `_advance_tick` sequentially:

1. Validate only the current tick record/tag. Idle with birth data fails; birth without a birth record fails. Attach `prior.tick_count` and `prior.birth_count` to its errors.
2. For birth call `_check_birth(prior.topology, prior.config.m, raw.birth, tick_index=prior.tick_count, birth_index=prior.birth_count)`, then `_parse_agent(raw.birth.agent, newborn=True, used_ids=frozenset(prior.agent_ids), stage="tick_agent", tick_index=prior.tick_count, birth_index=prior.birth_count)`. Return either error immediately. Only after both succeed call `_apply_birth` and append the new external ID. Copy old profiles and old states by ID, not by model tuple position.
3. Build the new model and `PopulationState(model.model_id, model.content_hash, 0, None, old_states + (new_state,))`. Old belief, exposure and broadcasting values remain exactly equal at this boundary. For idle use the prior model/topology/registry/population with mass one.
4. Use a single common operator call and result construction after the branch:

```python
round_result = simulate_round(model, post_growth)
next_frame = BBRuntimeFrame(
    config=prior.config,
    tick_count=prior.tick_count + 1,
    birth_count=prior.birth_count + int(tick.kind == "birth"),
    agent_ids=agent_ids,
    topology=topology,
    model=model,
    population=round_result.next_state,
    trace_mass=prior.trace_mass * tick_mass,
    parent_frame_hash=prior.content_hash,
    applied_tick=tick,
)
return BBRuntimeTransition(prior, tick, post_growth, round_result,
                           tick_mass, next_frame)
```

The branch locals are the validated normalized `tick`, selected `model`, `post_growth`, `agent_ids`, `topology` and exact `tick_mass`. There is no post-propagation birth or second behavioral probability factor.

- [ ] Extend tests: attach-source gives beliefs `[1,1,1]` and exposures `[0,1,1]`; zero receptivity still counts incoming exposure; silent population emits no transmissions; zero threshold broadcasts belief zero; initially broadcasting newborn emits immediately; newly informed receivers do not relay within the same tick. Preserve an old non-default role/adoption threshold exactly. Reject a reused external ID without calling `simulate_round` or `_apply_birth`; for conflicting BB/profile errors return the BB error first.
- [ ] Add a non-dyadic/threshold-near V1 consistency case using seed beliefs `(0.1, 0.2)`, receptivity `0.3`, threshold `0.1 + 0.2`, and a newborn with belief `0.3`/threshold `0.3`. Build the expected V1 model and post-growth snapshot independently in the test, call the existing operator, and compare complete `NetworkRoundResult` values with exact equality. This tests delegation to the existing floating operator; it is not a Lean rational threshold assertion. Also test NaN, both infinities and booleans at each scalar field with earlier conflicting fields to pin precedence.
- [ ] Run GREEN and commit:

```bash
python3 -m unittest tests.test_network_abm_bb_runtime_contracts tests.test_network_abm_bb_runtime_kernel tests.test_network_abm_bb_runtime_round -v
git diff --check
git add narrative_dynamics/abm/bb_runtime.py tests/test_network_abm_bb_runtime_round.py
git commit -m "feat: compose BB growth with one retained-state V1 round"
```

## Task 4: Atomic finite replay and complete lineage

**Files:** Modify `narrative_dynamics/abm/bb_runtime.py`; create `tests/test_network_abm_bb_runtime_replay.py`.

**Consumes:** Tasks 1–3.

**Produces:** Private `_initial_frame(seed: object, m: object) -> BBRuntimeFrame | BBRuntimeError`; sole public execution function `replay_bb_population(seed: BBRuntimeSeed, m: int, ticks: tuple[BBRuntimeTick, ...]) -> BBRuntimeReplay | BBRuntimeError`. Runtime type checks remain mandatory despite annotations. Set the execution module's `__all__ = ["replay_bb_population"]`; do not re-export it from the existing package root.

- [ ] Write the multi-tick RED:

```python
from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeError, BBRuntimeReplay, BBRuntimeTick,
)
from tests.bb_runtime_fixtures import birth, seed


class BBRuntimeReplayTests(unittest.TestCase):
    def test_successive_births_and_idle_keep_two_clocks(self):
        out = replay_bb_population(seed(), 1,
                                   (birth("2", (1,)), birth("3", (2,)), BBRuntimeTick()))
        self.assertIsInstance(out, BBRuntimeReplay)
        frames = [t.next_frame for t in out.transitions]
        self.assertEqual([f.tick_count for f in frames], [1, 2, 3])
        self.assertEqual([f.birth_count for f in frames], [1, 2, 2])
        self.assertEqual([f.population.round_index for f in frames], [1, 1, 2])
        self.assertEqual([a.belief for a in frames[1].population.agents], [1, 1, 1, 0])
        self.assertEqual([a.exposure_count for a in frames[1].population.agents],
                         [1, 2, 1, 0])
        self.assertEqual([a.belief for a in out.final.population.agents], [1, 1, 1, 1])
        self.assertEqual(out.final.trace_mass, Fraction(1, 8))
        self.assertEqual(out.final.agent_ids, ("0", "1", "2", "3"))
        self.assertEqual(out.final.topology.edges, ((0, 1), (1, 2), (2, 3)))

    def test_empty_still_validates_and_does_not_propagate(self):
        with patch("narrative_dynamics.abm.bb_runtime.simulate_round") as operator:
            out = replay_bb_population(seed(), 1, ())
            invalid = replay_bb_population(seed(), 0, ())
        operator.assert_not_called()
        self.assertEqual(out.initial, out.final)
        self.assertEqual(out.transitions, ())
        self.assertEqual(out.final.trace_mass, Fraction(1))
        self.assertEqual((invalid.stage, invalid.code), ("initial_m", "initialM"))

    def test_first_error_and_two_indices_are_atomic(self):
        original = seed()
        bad = birth("2", (2,))
        requests = (BBRuntimeTick(), BBRuntimeTick(), bad)
        expected = replay_bb_population(original, 1, requests)
        actual = replay_bb_population(original, 1, requests + (object(),))
        self.assertIsInstance(actual, BBRuntimeError)
        self.assertEqual(actual, expected)
        self.assertEqual((actual.tick_index, actual.birth_index), (2, 0))
        self.assertEqual(actual.bb_cause, "targetOutOfRange")
        self.assertEqual(original, seed())
        self.assertEqual(requests[-1], bad)
        self.assertFalse(hasattr(actual, "transitions"))
```

- [ ] Run RED: `python3 -m unittest tests.test_network_abm_bb_runtime_replay -v`; expect the absent public replay, then the multi-tick/atomicity assertions to exercise its real result.
- [ ] Implement `_initial_frame`: check outer seed record shape, `_parse_bb_seed`, initial `_check_m` even for empty/idle schedules, tuple seed-agent array and exact count, then `_parse_agent` left-to-right with each accepted ID added to an immutable used-ID set. After all agents, validate model ID then version. Build config, numeric registry, `_model_for`, direct local-round-zero population, and a mass-one genesis frame. Do not sort the raw agent array, call `_advance_tick`, or call any population simulator at genesis.
- [ ] Implement the complete finite loop with no full-schedule prescan:

```python
def replay_bb_population(seed: BBRuntimeSeed, m: int,
                         ticks: tuple[BBRuntimeTick, ...]
                         ) -> BBRuntimeReplay | BBRuntimeError:
    initial = _initial_frame(seed, m)
    if isinstance(initial, BBRuntimeError):
        return initial
    if not isinstance(ticks, tuple):
        return BBRuntimeError("ticks", "invalidType", field="ticks")
    current = initial
    transitions = []
    for raw in ticks:
        step = _advance_tick(current, raw)
        if isinstance(step, BBRuntimeError):
            return step
        transitions.append(step)
        current = step.next_frame
    return BBRuntimeReplay(initial, tuple(transitions), current)
```

Earlier successful local computations on an eventually failing trace are allowed; returning or publishing them is not. Do not add callbacks, mutation of input records, lazy result iterators, output files or external ID allocation. A failed replay does not consume an ID globally; a subsequent replay from the same seed can use it.

- [ ] Complete the acceptance matrix as public-API tests:

| Test | Exact assertion |
| --- | --- |
| Registry crosses 9 to 10 | Start with a ten-node unit-fitness path, IDs `str(i)` in numeric order, distinguishable beliefs `i/16`, exposures `i`, receptivity zero; birth ID `"10"` at target 9. Registry ends `(…,"9","10")`; each old post-growth state equals its input by ID even though the V1 sorted tuple puts `"10"` before `"2"`. Mass is `1/18`. |
| Repeated execution | Same normalized inputs produce equal replay content and hashes. Integer fitness and equivalent Fraction fitness normalize to the same hashes. |
| Ordered history | With unit fitness, `m=2`, target permutations have equal final model/population and mass `1/2`, but different applied ticks and frame/replay hashes. With fitness `(1,3)`, require exact masses `1/4` and `3/4`. |
| Idle-only | Two idles keep topology/fitness/model, mass one, local rounds 1 then 2, global ticks 1 then 2, birth count zero. Invalid initial `m` fails before the ticks container or later request. |
| Birth after idles | Two idles then valid birth: global round 3, local round 1, new epoch 1; old exposures retained. |
| Separate late indices | Idle, valid birth, idle, invalid target 3: error tick 3/birth 1. Two idles then first invalid birth: tick 2/birth 0. |
| First failure | Invalid newborn threshold followed by bad fitness or an arbitrary malformed object keeps the original tick-agent error; reversing the two births yields the first network error. |
| Seed precedence | Bad seed before bad `m`; bad `m` before wrong roster count; wrong count before malformed record; record 0 scalar error before record 1 error; ID/role/adoption failures occur after that record's scalars/exposure. |
| Shape and freshness | Tuple-only fitness/edges/agents/targets/ticks, expected raw types, idle tag constraints, nonempty/duplicate ID, role/adoption validation, and no newborn exposure field. Do not use `dataclasses.replace` with a nonexistent newborn exposure field as an execution-boundary test; test its constructor contract separately. |
| Atomicity | Deep-copy valid requests before replay and compare afterwards; for malformed nested list fields verify no mutation. After failure replay the valid prefix plus a corrected request and require success with the same newborn ID. No partial result attributes. |
| V1 regressions | Existing fixed-roster `NetworkRoundResult` still rejects mixed models/rosters and `PopulationTrajectory` still rejects an empty round tuple; no edits to those constructors. |

- [ ] Run GREEN and commit:

```bash
python3 -m unittest discover -s tests -p 'test_network_abm_bb_runtime_*.py' -v
python3 -m unittest tests.test_network_abm_contracts tests.test_network_abm_simulation tests.test_network_abm_public_api -v
git diff --check
git add narrative_dynamics/abm/bb_runtime.py tests/test_network_abm_bb_runtime_replay.py
git commit -m "feat: expose atomic finite BB population replay"
```

## Task 5: Actual Lean histories, Python conformance and permanent bounded CI

**Files:** Create all six Task 5 files in the ownership table; add one step to `.github/workflows/proof.yml`. Preserve `tools/check_fitness_abm.sh`, its 35 required names, the old exporter, old corpus and old consumer exactly.

**Consumes:** Public Python replay; existing Lean `FitnessABM.replay`, `FitnessABM.checkedBirth`, `FitnessABM.advance`, `NetworkPropagation.transmissions`, `NetworkPropagation.broadcasting`, and `FitnessAttachment.actualEdgeCount`. The new exporter imports `NarrativeDynamics.Core.FitnessABMReplay` only; the new fixture imports the new exporter. Do not import tests or executable exporters into `NarrativeDynamics.lean`.

**Produces:** The `bb-abm-runtime-v1` corpus with exactly 16 successful cases and 20 shared-error cases; three required kernel dependency reports; one additional bounded verification step.

### Corpus schema and producer interfaces

Use namespace `NarrativeDynamics.Conformance.BBRuntime`. Fix these data declarations and names in `FitnessABMRuntimeVectors.lean`:

```lean
structure RuntimeCaseInput where
  id : String
  seed : FitnessAttachment.RawSeed
  m : Nat
  agents : Array FitnessABM.RawAgent
  ticks : List FitnessABM.RawTick

structure StateObservation where
  nodeCount : Nat
  edges : List (Nat × Nat)
  fitness : List Rat
  receptivity : List Rat
  thresholds : List Rat
  beliefs : List Rat
  exposures : List Nat
  broadcasting : List Bool
  deriving DecidableEq, Repr

structure PrefixObservation where
  state : StateObservation
  tickCount : Nat
  birthCount : Nat
  traceMass : Rat
  deriving DecidableEq, Repr

structure TransitionObservation where
  tickIndex : Nat
  birthIndex : Nat
  tickMass : Rat
  postGrowth : StateObservation
  transmissions : List (Nat × Nat × Rat)
  deriving DecidableEq, Repr

structure HistoryObservation where
  prefixes : List PrefixObservation
  transitions : List TransitionObservation
  deriving DecidableEq, Repr

def replayPrefix (input : RuntimeCaseInput) (count : Nat) :
    Except FitnessABM.JointError FitnessABM.Result :=
  FitnessABM.replay input.seed input.m input.agents (input.ticks.take count)
```

Additional exact interfaces:

| Declaration | Type and behavior |
| --- | --- |
| `observeState` | `{n : Nat} -> FitnessABM.JointState n -> StateObservation`; enumerate numeric IDs and actual adjacency, profiles and states |
| `observePrefix` | `(input : RuntimeCaseInput) -> (count : Nat) -> Except FitnessABM.JointError PrefixObservation`; map actual `replayPrefix`; use result's actual `final.roundIndex` and probability, birth count from `inputBirths (ticks.take count)` |
| `observeTransition` | `(input : RuntimeCaseInput) -> (index : Fin input.ticks.length) -> Except FitnessABM.JointError TransitionObservation`; use actual prior prefix plus current checked birth/idle |
| `observeCase` | `RuntimeCaseInput -> Except FitnessABM.JointError HistoryObservation`; validate full replay first, then every prefix 0 through T and every transition 0 through T-1 |
| `successInputs`, `errorInputs` | `List RuntimeCaseInput`; exact case manifests below |
| `renderCorpus` | `Except String Lean.Json`; evaluate inputs, reject an unexpected success/error classification, and serialize actual observations |
| namespace/global `main` | `IO Unit`; print one compressed JSON line or fail nonzero on renderer error, matching the existing exporter entry convention |

For birth transitions, call `replayPrefix input index.val`, then `FitnessABM.checkedBirth prior.final.state input.m raw`; map `.network`/`.agent` errors to `.tickNetwork`/`.tickAgent` with authored indices. Its successful joint state is `postGrowth`, and its exact returned probability is `tickMass`. For idle, `postGrowth` is the prior joint state and mass is one. Enumerate `NetworkPropagation.transmissions` on that state's actual adjacency/population, with signal taken from the source's prior belief. The next prefix is independently computed by the real `FitnessABM.replay`; do not recreate the propagation formula in the exporter.

Serialize exactly:

| JSON object | Keys/value shape |
| --- | --- |
| Corpus | `schema: "bb-abm-runtime-v1"`, `success` array of 16 cases, `errors` array of 20 cases |
| Successful case | `id`, `input`, `prefixes`, `transitions` |
| Error case | `id`, `input`, `expected_error` |
| Input | `seed: {node_count, fitness, edges}`, `m`, `agents: [{receptivity, threshold, belief, exposures}]`, `ticks: [null or {fitness, targets, receptivity, threshold, belief}]` |
| State observation | `node_count`, `edges`, `fitness`, `receptivity`, `thresholds`, `beliefs`, `exposures`, `broadcasting` |
| Prefix | `state`, `tick_count`, `birth_count`, `trace_mass` |
| Transition | `tick_index`, `birth_index`, `tick_mass`, `post_growth`, `transmissions: [{source, target, signal}]` |
| Expected error | The full `BBRuntimeError.to_dict()` field set, including `None` as JSON null for absent indices/counts/cause; no extra state fields |

All Rat values, including invalid raw inputs, serialize with the canonical integer-or-fraction string. Nat values serialize as JSON numbers and Bool as JSON booleans. Edges are ascending canonical numeric pairs; transmission rows sort by numeric source then target. Runtime-only IDs, roles, adoption metadata, V1 hashes and local clocks are not invented as Lean outputs.

For semantic BB errors use field `node_count`, `fitness`, `edges`, `m` or `targets` for the owning group; do not invent a particular bad element index that Lean does not return. Python-specific element-type errors may retain fields such as `fitness[0]`. Map every `JointError` constructor explicitly using the shared error table, and map every `Internal.Error` constructor to its stable name, including unreachable-in-this-replay `negativeWeight` and `zeroMass`. `seedAgentCount` alone supplies expected/actual counts; all other shared errors leave those fields null.

### Exact finite cases and literal obligations

Defaults are the design's unit seed edge, agents `[<1,1/2,1,0>, <1,1/2,0,0>]`, `m=1`, and newborn fitness one/profile `<1,1/2>`/belief zero. In the table, `B(j)` is a raw birth targeting numeric ID `j`, `I` is `none`, and every row has its own input record; no expected observations are used to construct raw inputs.

| Success ID | Input override/calendar | Required literal final `(nodes, ticks, edges; beliefs; exposures; mass)` |
| --- | --- | --- |
| `empty` | `[]` | `(2,0,1; [1,0]; [0,0]; 1)` |
| `idle-two` | `[I,I]` | `(2,2,1; [1,1]; [1,2]; 1)` |
| `attach-source` | `[B(0)]` | `(3,1,2; [1,1,1]; [0,1,1]; 1/2)` |
| `attach-relay` | `[B(1)]` | `(3,1,2; [1,1,0]; [0,1,0]; 1/2)` |
| `relay-idle` | `[B(1),I]` | `(3,2,2; [1,1,1]; [1,2,1]; 1/2)` |
| `successive-births` | `[B(1),B(2)]` | `(4,2,3; [1,1,1,0]; [1,2,1,0]; 1/8)` |
| `successive-idle` | `[B(1),B(2),I]` | `(4,3,3; [1,1,1,1]; [2,4,2,1]; 1/8)` |
| `weighted-source` | seed fitness `[1,3]`, `[B(0)]` | `(3,1,2; [1,1,1]; [0,1,1]; 1/4)` |
| `ordered-01` | fitness `[1,3]`, `m=2`, targets `[0,1]` | `(3,1,3; [1,1,1]; [0,1,1]; 1/4)` |
| `ordered-10` | fitness `[1,3]`, `m=2`, targets `[1,0]` | `(3,1,3; [1,1,1]; [0,1,1]; 3/4)` |
| `zero-receptive` | seed agent 1 receptivity zero, `[B(0)]` | `(3,1,2; [1,0,1]; [0,1,1]; 1/2)` |
| `silent` | both seed beliefs zero, `[B(0)]` | `(3,1,2; [0,0,0]; [0,0,0]; 1/2)` |
| `zero-threshold` | both seed beliefs zero, seed agent 0 threshold zero, `[B(0)]` | `(3,1,2; [0,0,0]; [0,1,1]; 1/2)` |
| `broadcast-newborn` | `[B(1)]` with newborn belief one | `(3,1,2; [1,1,1]; [0,2,0]; 1/2)` |
| `retained-exposures` | seed exposures `[7,3]`, `[B(1)]` | `(3,1,2; [1,1,0]; [7,4,0]; 1/2)` |
| `half-receptive` | seed agent 1 receptivity `1/2`, `[B(0)]` | `(3,1,2; [1,1/2,1]; [0,1,1]; 1/2)` |

| Error ID | Input override | Exact Lean error |
| --- | --- | --- |
| `seed-nodes` | seed `<0,#[],#[]>`, conflicting invalid agent scalar | `.seedNetwork .invalidNodeCount` |
| `seed-size` | node count 2, fitness `#[0]` | `.seedNetwork .fitnessSizeMismatch` |
| `seed-fitness` | fitness `#[0,1]`, invalid edge `(0,2)` | `.seedNetwork .nonpositiveFitness` |
| `seed-edge` | edges `#[(0,1),(1,0),(0,2)]` | `.seedNetwork .invalidEdge` |
| `seed-duplicate` | edges `#[(0,1),(1,0)]` | `.seedNetwork .duplicateEdge` |
| `seed-disconnected` | node count 3, fitness `#[1,1,1]`, only edge `(0,1)` | `.seedNetwork .disconnectedSeed` |
| `initial-m-zero` | `m=0`, empty calendar | `.initialM` |
| `initial-m-too-large` | `m=3`, idle-only calendar | `.initialM` |
| `seed-agent-count` | one agent for two-node seed | `.seedAgentCount 2 1` |
| `seed-agent-r` | agent 0 receptivity `-1`, threshold 2 | `.seedAgent 0 .receptivity` |
| `seed-agent-threshold` | agent 0 threshold 2, belief 2 | `.seedAgent 0 .threshold` |
| `seed-agent-belief` | agent 1 belief 2 | `.seedAgent 1 .belief` |
| `birth-fitness` | first birth fitness 0, belief 2 | `.tickNetwork 0 0 .nonpositiveFitness` |
| `birth-target-count` | `m=1`, targets `[0,0]` | `.tickNetwork 0 0 .targetCountMismatch` |
| `birth-target-range` | first target `[2]` | `.tickNetwork 0 0 .targetOutOfRange` |
| `birth-target-duplicate` | `m=2`, targets `[0,0]` | `.tickNetwork 0 0 .duplicateTarget` |
| `birth-agent-threshold` | first newborn threshold 2, belief 2 | `.tickAgent 0 0 .threshold` |
| `late-birth` | `[I,B(1),I,B(3)]` | `.tickNetwork 3 1 .targetOutOfRange` |
| `first-failure` | invalid threshold birth, then zero fitness birth | `.tickAgent 0 0 .threshold` |
| `late-first-birth` | `[I,I,B(2)]` | `.tickNetwork 2 0 .targetOutOfRange` |

All other error calendars are empty for seed errors or contain the one stated invalid birth for birth errors. Where a birth overrides only fitness or scalar fields, its targets are `[0]`; this applies to both births in `first-failure`. Default agent arrays remain unchanged unless overridden; for `seed-nodes` use the one conflicting raw agent `<-1,2,2,0>`. For the idle-only invalid-m case use `[I,I]`. These refinements fix the corpus inputs without relying on a later author to choose examples.

### RED, implementation and consumer

- [ ] Create the Python consumer first. Its first RED must be the absent `conformance/bb_abm_runtime_v1.json`, using `Path(__file__).resolve().parents[1]`. Assert schema, exact success/error ID sets, and counts 16/20 so missing cases cannot silently pass. Implement this raw-input conversion in the test, importing the named contracts and `Fraction`:

```python
def decode_input(raw):
    network = raw["seed"]
    agents = tuple(
        BBRuntimeAgent(str(i), "peer", float(Fraction(a["receptivity"])),
                       float(Fraction(a["threshold"])), float(Fraction(a["belief"])),
                       a["exposures"], 0.5)
        for i, a in enumerate(raw["agents"])
    )
    seed = BBRuntimeSeed(
        BBRuntimeRawSeed(network["node_count"],
                         tuple(Fraction(x) for x in network["fitness"]),
                         tuple(tuple(e) for e in network["edges"])),
        agents, "bb-runtime-conformance", "1",
    )
    ticks = []
    births = 0
    for tick in raw["ticks"]:
        if tick is None:
            ticks.append(BBRuntimeTick())
        else:
            newborn = BBRuntimeNewborn(
                str(network["node_count"] + births), "peer",
                float(Fraction(tick["receptivity"])),
                float(Fraction(tick["threshold"])), float(Fraction(tick["belief"])),
                0.5,
            )
            ticks.append(BBRuntimeTick("birth", BBRuntimeBirth(
                Fraction(tick["fitness"]), tuple(tick["targets"]), newborn)))
            births += 1
    return seed, raw["m"], tuple(ticks)
```

This decoder handles only the known Lean corpus domain; it is not a new production JSON parser. Raw invalid scalar values remain invalid float values and must reach the checked replay in their original order.

The main successful-case assertion loop is:

```python
out = replay_bb_population(*decode_input(case["input"]))
self.assertIsInstance(out, BBRuntimeReplay)
frames = (out.initial,) + tuple(t.next_frame for t in out.transitions)
self.assertEqual(len(frames), len(case["prefixes"]))
self.assertEqual(len(out.transitions), len(case["transitions"]))
for frame, expected in zip(frames, case["prefixes"]):
    self.assertEqual(frame.tick_count, expected["tick_count"])
    self.assertEqual(frame.birth_count, expected["birth_count"])
    self.assertEqual(frame.trace_mass, Fraction(expected["trace_mass"]))
    self.assert_state(frame, frame.population, expected["state"])
for step, expected in zip(out.transitions, case["transitions"]):
    self.assertEqual(step.tick_index, expected["tick_index"])
    self.assertEqual(step.birth_index, expected["birth_index"])
    self.assertEqual(step.tick_mass, Fraction(expected["tick_mass"]))
    self.assert_state(step.next_frame, step.post_growth_population,
                      expected["post_growth"])
    ids = {agent_id: i for i, agent_id in enumerate(step.next_frame.agent_ids)}
    actual = sorted((ids[t.source_agent_id], ids[t.target_agent_id], t.signal)
                    for t in step.round_result.transmissions)
    wanted = [(t["source"], t["target"], float(Fraction(t["signal"])))
              for t in expected["transmissions"]]
    self.assertEqual(actual, wanted)
```

Define `assert_state(self, frame, population, expected)` in the test class. It compares actual node count, exact topology edge list and Fraction fitness; it creates profile/state maps by ID and enumerates `frame.agent_ids` to compare receptivity, thresholds, beliefs, exposures and broadcasting with the expected arrays. Convert only scalar Rat strings to float; use exact equality for this dyadic manifest. Also check each actual transmission's local round equals its containing `round_result.round_index`, its relation is `"bb"`, and influence is `1.0`. Error cases compare `out.to_dict()` with `expected_error` and require `BBRuntimeError`; no expected-success prefix is returned. Put the case ID in `subTest` so mutation diagnostics identify the corrupted case.

- [ ] Run and retain the actual consumer RED:

```bash
python3 -m unittest tests.test_network_abm_bb_runtime_conformance -v
```

- [ ] Write the Lean case-input definitions and literal fixture tests before implementing the new observation functions. Put fixtures in namespace `NarrativeDynamics.Tests.FitnessABMRuntime`. Define an observable final summary returning `Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat)`, matching the table and existing fixture convention; a local decidable-equality instance can pattern-match the `Except` constructors and decide only the observable data. Target named per-case equalities, combined into these required theorems:

| Required theorem | Literal obligation |
| --- | --- |
| `success_literals` | Every one of the 16 actual replay final summaries equals its explicit table row, in manifest order |
| `shared_error_literals` | Every one of the 20 actual replay outcomes is the specified constructor/indices, in manifest order |
| `transition_literals` | In `successive-births`, the second post-growth beliefs/exposures are `[1,1,0,0]`/`[0,1,0,0]`, tick mass `1/4`, and transmissions `[(0,1,1),(1,0,1),(1,2,1)]`; in `retained-exposures`, first post-growth exposures are `[7,3,0]`; in `broadcast-newborn`, first transmissions are `[(0,1,1),(2,1,1)]` |

For example, the required second-growth assertion is about the actual `observeTransition` result, not a separately constructed input snapshot. All case equalities must use the producer's actual input definitions. This ties the new full-trace reference to literals independently of Python. Existing anonymous examples in `Tests/FitnessABMReplay.lean` are source guidance, not callable public lemmas or evidence that an unexecuted new fixture passed.

Run `timeout --kill-after=10s 240s lake build NarrativeDynamics.Conformance.FitnessABMRuntimeVectors` to build the input-only module, then `timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMRuntime.lean` and record the missing observation declaration RED after the toolchain is ready. Dependency installation failure is not this RED.

- [ ] Implement the Lean observations/producer and prove the literal fixtures. Use actual graph adjacency to enumerate edges; do not reuse raw edges after births. `Array.ofFn`/`List.ofFn` enumerate finite agent IDs; use the state's `adjDec` for executable adjacency/transmissions. Serialize actual `observeCase` outputs with `Lean.Json.mkObj`, not manually authored expected JSON. Ensure the renderer rejects a success entry returning an error and an error entry returning success.

Use symbolic `replay`/`checkedBirth_spec`/`grow`/`advance` equations before finite reduction, as the existing fixtures do. `by decide_cbv` is appropriate for small remaining scalar/finite equalities; expanding an entire dependent raw replay and all validity proofs is avoidable. Keep default heartbeats; if needed, use only fixture-scoped finite `set_option maxRecDepth 4096`, already used by the foundation. A timeout or proof error remains a failure: simplify the proof obligation and rerun before changing any resource budget. Do not use admitted proofs, native evaluation as proof, unlimited settings, or fabricated dependency output.

End the fixture with actual reports:

```lean
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.success_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.shared_error_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.transition_literals
```

- [ ] Build the exporter before the importing fixture, prove the literals, and perform the **first** real corpus generation:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Conformance.FitnessABMRuntimeVectors
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMRuntime.lean
timeout --kill-after=10s 240s lake env lean --run NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean > conformance/bb_abm_runtime_v1.json
python3 -m unittest tests.test_network_abm_bb_runtime_conformance -v
```

Run these sequentially with failure stopping the sequence (`set -euo pipefail` in a script). If the exporter fails, discard its partial new file; do not commit it or fabricate missing fields. After this first creation all verification generates to a temporary file and compares unconditionally.

### Permanent gate and mutation/restoration

- [ ] Create `tools/check_fitness_abm_runtime.sh` with this complete gate:

```bash
#!/usr/bin/env bash
set -euo pipefail
bb_runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bb_runtime_root"
bb_runtime_log="$(mktemp)"
bb_runtime_vectors="$(mktemp)"
trap 'rm -f "$bb_runtime_log" "$bb_runtime_vectors"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean \
  NarrativeDynamics/Tests/FitnessABMRuntime.lean

/usr/bin/time -f 'FitnessABMRuntimeVectors build elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake build NarrativeDynamics.Conformance.FitnessABMRuntimeVectors

/usr/bin/time -f 'FitnessABMRuntime literals elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMRuntime.lean \
  2>&1 | tee "$bb_runtime_log"

python3 tools/audit_fitness_trust.py log "$bb_runtime_log" \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.success_literals \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.shared_error_literals \
  --require NarrativeDynamics.Tests.FitnessABMRuntime.transition_literals

/usr/bin/time -f 'FitnessABMRuntimeVectors run elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean --run NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean \
  > "$bb_runtime_vectors"
cmp conformance/bb_abm_runtime_v1.json "$bb_runtime_vectors"
```

Add this step immediately after the existing `BB ABM joint evolution contracts` step:

```yaml
      - name: BB ABM runtime replay conformance
        timeout-minutes: 15
        run: bash tools/check_fitness_abm_runtime.sh
```

The existing 25-minute foundation step remains unchanged. The new 15-minute step contains three independently bounded 240-second phases plus auditing/comparison; it does not enlarge an old phase's timeout. No event selector, branch filter, checkout ref, dependency lock or workflow permission changes are needed. The existing Python full-discovery job and World Studio `test_network_abm*.py` discovery already collect the new consumer. The Lean job needs only standard-library Python for the trust auditor.

- [ ] Check the gate's failure behavior before recording acceptance: missing new required report is rejected by the existing auditor, and a wrong committed corpus is rejected by `cmp`. Keep the existing trust-audit regression suite; do not write a second auditor or duplicate its implementation tests.
- [ ] Run this deliberate semantic corpus mutation, retain its actual diagnostic, and restore the exact bytes even on failure:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

path = Path("conformance/bb_abm_runtime_v1.json")
original = path.read_bytes()
try:
    corpus = json.loads(original)
    case = next(c for c in corpus["success"] if c["id"] == "successive-births")
    case["prefixes"][2]["trace_mass"] = "1/7"
    path.write_text(json.dumps(corpus, separators=(",", ":")) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "unittest",
         "tests.test_network_abm_bb_runtime_conformance", "-v"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout)
    if result.returncode == 0:
        raise AssertionError("wrong ordered-trace mass was accepted")
    if "successive-births" not in result.stdout or "AssertionError" not in result.stdout:
        raise AssertionError("mutation failed for an unrelated reason")
finally:
    path.write_bytes(original)
PY
bash tools/check_fitness_abm_runtime.sh
python3 -m unittest tests.test_network_abm_bb_runtime_conformance tests.test_network_abm_bb_conformance -v
```

The restored gate must **freshly regenerate** the Lean corpus, audit real reports and compare exact bytes. Restoring a Python test alone or accepting an unrelated ImportError does not complete the mutation check. Never commit the mutant file. Keep the original eight-vector corpus byte-for-byte unchanged.

- [ ] Run GREEN and commit this complete integration unit:

```bash
bash -n tools/check_fitness_abm_runtime.sh
python3 -m unittest tests.test_fitness_trust_audit -v
bash tools/check_fitness_abm.sh
bash tools/check_fitness_abm_runtime.sh
python3 -m unittest discover -s tests -p 'test_network_abm_bb_runtime_*.py' -v
git diff --check
git add NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean NarrativeDynamics/Tests/FitnessABMRuntime.lean conformance/bb_abm_runtime_v1.json tests/test_network_abm_bb_runtime_conformance.py tools/check_fitness_abm_runtime.sh .github/workflows/proof.yml
git commit -m "test: compare complete BB runtime traces with Lean replay"
```

Task 5 is complete only with actual proof/exporter execution and consumer results on the recorded revision. The plan's CI snippets and prescribed expected failures are not themselves evidence.

## Task 6: Executable example, documented boundary and complete acceptance

**Files:** Create `examples/bb_abm_runtime.py`; modify the BB–ABM section of `README.md` and this plan's execution record. Do not change the example's imports to depend on test helpers.

**Consumes:** Completed public replay API and current-revision reference/conformance evidence.

**Produces:** `python3 -m examples.bb_abm_runtime`, a documented finite result, and a reviewed implementation with exact CI revision evidence.

- [ ] Add the example as a consumer of the actual public API:

```python
from fractions import Fraction

from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent, BBRuntimeBirth, BBRuntimeError, BBRuntimeNewborn,
    BBRuntimeRawSeed, BBRuntimeSeed, BBRuntimeTick,
)


def main() -> int:
    seed = BBRuntimeSeed(
        BBRuntimeRawSeed(2, (1, 1), ((0, 1),)),
        (BBRuntimeAgent("0", "peer", 1, 0.5, 1),
         BBRuntimeAgent("1", "peer", 1, 0.5, 0)),
        "bb-successive-births", "1",
    )
    ticks = (
        BBRuntimeTick("birth", BBRuntimeBirth(
            1, (1,), BBRuntimeNewborn("2", "peer", 1, 0.5, 0))),
        BBRuntimeTick("birth", BBRuntimeBirth(
            1, (2,), BBRuntimeNewborn("3", "peer", 1, 0.5, 0))),
        BBRuntimeTick(),
    )
    out = replay_bb_population(seed, 1, ticks)
    if isinstance(out, BBRuntimeError):
        print(f"replay failed: {out.stage}/{out.code}")
        return 1
    for frame in (out.initial,) + tuple(t.next_frame for t in out.transitions):
        states = {a.agent_id: a for a in frame.population.agents}
        beliefs = [states[i].belief for i in frame.agent_ids]
        exposures = [states[i].exposure_count for i in frame.agent_ids]
        print(f"tick={frame.tick_count} epoch={frame.epoch} "
              f"local_round={frame.population.round_index} "
              f"nodes={frame.topology.node_count} edges={len(frame.topology.edges)} "
              f"beliefs={beliefs} exposures={exposures} mass={frame.trace_mass}")
    return 0 if out.final.trace_mass == Fraction(1, 8) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Run `python3 -m examples.bb_abm_runtime`. It must exit zero and print these observations from actual frames:

| Global tick | Epoch | V1 local round | Nodes/edges | Beliefs | Exposures | Trace mass |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 2/1 | `[1,0]` | `[0,0]` | 1 |
| 1 | 1 | 1 | 3/2 | `[1,1,0]` | `[0,1,0]` | 1/2 |
| 2 | 2 | 1 | 4/3 | `[1,1,1,0]` | `[1,2,1,0]` | 1/8 |
| 3 | 2 | 2 | 4/3 | `[1,1,1,1]` | `[2,4,2,1]` | 1/8 |

This is an executable acceptance check; no new snapshot test mirroring print formatting is needed. The actual behavioral and reference tests already own these results.

- [ ] Add a README subsection after the existing Lean BB-driven evolution description. Show the module command and short output table; link the new example, design, public module and full-trace corpus. State that authored ordered targets drive real Python births, exact mass is conditional on fixed inputs/calendar, existing V1 floating propagation is reused, and finite conformance is not a proof of all Python executions. Explain numeric-ID registry, epoch-local versus global clocks, and atomic error return sufficiently for an API caller. State the supported finite boundary without claiming a stochastic sampler, adaptive fitness, V19 integration, universal small-world behavior, a fixed hop bound or consensus result.
- [ ] Run the complete required acceptance checks, recording actual results and skips:

```bash
git diff --check
python3 -m unittest discover -s tests -v
lake build
bash tools/check_fitness_abm.sh
bash tools/check_fitness_abm_runtime.sh
python3 -m examples.bb_abm_runtime
```

The foundation's 1,646-test result is historical baseline evidence, not the expected count for new discovery. Record the actual new count. The existing optional Blender smoke skip, if still present for its documented missing executable, must be reported accurately rather than counted as a pass. Review newly introduced warnings; do not expand this change solely to fix inherited unrelated warnings.

- [ ] Commit the example and documentation, then publish the implementation PR/branch through the authorized GitHub connection. Use the verification-before-completion and requesting-code-review skills for actual acceptance/review. Preserve task commits and identify the reviewed source SHA; a planning self-review does not replace the later implementation review.

```bash
git add examples/bb_abm_runtime.py README.md docs/superpowers/plans/2026-09-14-bb-abm-runtime-adapter-v1.md
git commit -m "docs: demonstrate and bound authored BB population replay"
```

- [ ] Verify the latest implementation revision in actual CI. The proof workflow's Lean and Python jobs explicitly check out the PR head; inspect their recorded `git rev-parse HEAD`. World Studio's PR workflow uses GitHub's merge checkout; inspect the actual merge commit, its parents and tree before describing its coverage. Confirm full Python discovery, all old Lean gates, all 35 unchanged foundation reports, all three new reports, both corpus generations/comparisons, and applicable World Studio checks. A selector skipping a duplicate push job is not failure when the corresponding actual PR proof job covers this head; an absent or skipped proof job alone is not acceptance.
- [ ] Complete the implementation review and fix any material findings with scoped re-verification. Update the execution record with concrete commits, actual RED/GREEN diagnostics, mutation/restoration evidence and current CI links. Keep uncompleted boxes unchecked. Integrate or merge only according to the user's current instruction; writing this plan does not itself merge either stacked PR.

## Design coverage and planning self-review

| Pinned design requirement | Owning task / concrete gate |
| --- | --- |
| Separate wrapper and unchanged fixed-roster contracts (§§1–2, 4) | 1 structural records; 3 V1 boundary; 4 fixed-roster/empty-trajectory regressions |
| Append-only ID mapping, fitness, unit channels and metadata (§3) | 1 registry/model binding; 2 exact topology; 3 metadata retention; 4 ID 9→10 case |
| Two clocks, model hashes, old-state retention and one propagation (§§4–5) | 3 pre-growth/input/round checks and one-call assertion; 4 multi-epoch trace; 6 printed example |
| Frozen weighted ordered choices and counts (§5) | 2 weighted `m=2`, three-node degrees and second-birth tests; 4 mass/count formulas |
| BB-first validation, two indices, first failure and atomic empty replay (§6) | 2 full-category validation order; 3 delayed agent validation; 4 public failure matrix; 5 shared-error corpus |
| Canonical lineage and content identity, no trusted caller history (§7) | 1 constructors/serialization; 4 equal-outcome ordered-history hashes and sole finite API |
| Exact BB versus floating V1 boundary (§8) | 2 Fraction math; 3 non-dyadic V1 comparison; 5 exact dyadic reference checks; 6 README scope |
| Every acceptance row and actual full histories (§9) | 1–4 focused tests; 5 16 success/20 error manifest, all prefixes and transition inputs/transmissions |
| Kernel literals, fresh generation, mutation/restoration (§9) | 5 three required dependency reports, unconditional byte comparison and wrong-mass rejection |
| File units, unchanged gates and bounded resources (§10) | Ownership table; 5 separate 15-minute step, unchanged 25-minute foundation step and 35 requirements |
| Current revision acceptance and deferred scope (§11) | 6 full discovery/proof/World Studio evidence and implementation review |

Planning review is performed by the plan author, without a subagent dispatch. Before publishing this plan, verify the coverage table, exact cross-task names/constructor order, sequential GREEN dependencies, all named paths and existing signatures. Scan for incomplete instructions, unresolved types and circular imports/hash construction. Documentation-only checks establish the quality and scope of this plan, not the future code's correctness.

Planning self-review completed on 2026-09-14: all 17 copied constraints match the pinned design, all six tasks map to the design coverage table, the case manifests contain 16 success and 20 shared-error entries, all 13 Python snippets parse as syntax, all 12 shell blocks pass `bash -n`, and the relative document links resolve. These are documentation checks; no new Python behavior or Lean proof was executed. The review corrected error-field mapping and the producer/fixture RED dependency order before publication.

The module dependency direction is `bb_runtime_contracts -> existing contracts/simulation` and `bb_runtime -> bb_runtime_contracts + existing simulation`. The new Lean fixture imports its exporter, which imports core; build the exporter before the fixture. There is no import from the new contracts back into the new executor, and no frame contains a transition hash that hashes that same frame.

## Execution record

| Unit | Current status | Evidence required before completion |
| --- | --- | --- |
| Preparation | Not started | Implementation base SHA, versions and focused baseline |
| Task 1 | Not started | Record RED/GREEN and commit; immutable/forged-binding/lineage assertions |
| Task 2 | Not started | Exact arithmetic and precedence RED/GREEN and commit |
| Task 3 | Not started | Retention/delayed-relay/one-call RED/GREEN and commit |
| Task 4 | Not started | Finite replay/identity/error atomicity RED/GREEN and commit |
| Task 5 | Not started | Actual Lean literals/reports, generated corpus, consumer RED/GREEN, mutation/restoration and commit |
| Task 6 | Not started | Executed example, complete current-revision CI, implementation review and final commit |

The next execution unit is preparation followed by Task 1. This document is the detailed handoff; it does not claim any runtime implementation or a new mathematical proof result.
