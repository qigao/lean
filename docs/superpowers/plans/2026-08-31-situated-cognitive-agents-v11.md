# Situated Cognitive Agents V11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every V10 embodied agent autonomously update private beliefs and select
physical actions with an auditable finite-horizon POMDP policy.

**Architecture:** Immutable cognitive contracts bind finite hypotheses, observation
likelihoods/rules, physical action templates, transitions, goals, rewards, and private
mind state to one exact V10 model/story. A deterministic runtime admits only each
agent's new perspective events, performs Bayesian updates and soft Bellman planning,
compiles lexical MAP actions to V10 intents, resolves them synchronously, and records
decision explanations.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V10 situated runtime,
existing `PlanningBeliefState`, shared `finite_softmax`, stable content hashes,
`unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-situated-cognitive-agents-v11-design.md`

## Global Constraints

- No agent decision receives objective world state, complete event ledger, or another
  agent's mind.
- Observations affect decisions only in the following round.
- Beliefs are exact finite probability distributions with `1e-12` simplex tolerance.
- Action selection is deterministic lexical MAP over a complete softmax policy.
- All model, state, decision, round, explanation, and trajectory artifacts are
  immutable, canonical, content-hashed, and exact-parent-linked.
- V10 remains backward-compatible.

---

### Task 1: Cognitive model and private-state contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_cognition_contracts.py`
- Create: `tests/situated_cognition_fixtures.py`
- Create: `tests/test_network_abm_situated_cognition_contracts.py`

**Interfaces:**
- Produces `SituatedHypothesis`, `SituatedObservationSymbol`,
  `SituatedObservationRule`, `SituatedObservationLikelihood`,
  `SituatedActionSpec`, `SituatedHypothesisTransition`, `SituatedGoalSpec`,
  `SituatedGoalReward`, `SituatedAgentCognitiveModel`, `SituatedCognitiveModel`,
  `SituatedAgentMindState`, `SituatedCognitiveState`, and
  `initialize_situated_cognition(model, story)`.

- [ ] Write failing tests proving exact agent/hypothesis/likelihood/reward coverage,
  canonical ordering, prior coverage, private initial place, state/story binding, and
  stable hashes.
- [ ] Run `python -m unittest tests.test_network_abm_situated_cognition_contracts -v`;
  expected RED is missing-module import failure.
- [ ] Implement minimal immutable contracts. A missing transition table means exact
  identity; a supplied table must cover every `(action, prior, next)` cell and every
  row must sum to one.
- [ ] Re-run the contract tests to GREEN and commit `feat: define situated cognitive agents`.

### Task 2: Private Bayesian admission and finite POMDP decisions

**Files:**
- Create: `narrative_dynamics/abm/situated_cognition.py`
- Create: `tests/test_network_abm_situated_cognition.py`

**Interfaces:**
- Produces `SituatedCognitiveDecision`, `SituatedCognitiveRoundResult`,
  `admit_situated_observations`, `decide_situated_action`, and
  `simulate_situated_cognitive_round`.
- Consumes Task 1 contracts, `perspective_timeline`, `advance_situated_story`,
  `PlanningBeliefState`, and `finite_softmax`.

- [ ] Write failing tests where an inspection updates only Alice from `0.5/0.5` to a
  hand-derived `0.9/0.1`, Bob remains unchanged, and re-admission is idempotent.
- [ ] Confirm RED, then implement sequential Bayesian admission from unmatched/new
  private observations with zero-evidence rejection.
- [ ] Write failing planning tests proving two-step information value, lexical MAP
  ties, own-place/non-repeat/source feasibility, and goal-contribution audit values.
- [ ] Implement finite soft Bellman recursion and intent compilation; future schedule
  rows use declared action IDs while root feasibility comes only from private mind.
- [ ] Write a failing synchronous-round test proving decisions use pre-round evidence,
  new tellings update recipients only next round, and reversed model declaration order
  preserves exact result/hash identity.
- [ ] Implement cognitive round execution and next private-state commit; re-run all
  V11 tests to GREEN and commit `feat: run situated cognitive agents`.

### Task 3: Explanations, trajectory, autonomous office case, and delivery

**Files:**
- Modify: `narrative_dynamics/abm/situated_cognition.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Create: `tests/test_network_abm_situated_cognitive_story.py`
- Modify: `README.md`

**Interfaces:**
- Produces `SituatedDecisionExplanation`, `SituatedCognitiveTrajectory`,
  `explain_situated_decision`, and `simulate_situated_cognition`.

- [ ] Write the failing four-agent office test: Alice inspects, moves, tells with her
  inspection source; Bob admits Alice's telling then retells; Dana remains unchanged.
- [ ] Implement multi-round trajectory and structured explanations containing selected
  action, most-likely hypothesis, probability, value, goal contributions, and admitted
  evidence IDs.
- [ ] Assert exact replay and that V10 `information_chain` reconstructs inspection →
  Alice telling → Bob telling.
- [ ] Export the exact V11 API and add an independently executable README example.
- [ ] Run every README Python block, all `test_network_abm*.py`, the 40 targeted
  narrative regressions, `compileall`, diff check, unfinished-marker scan, and status.
- [ ] Use `superpowers:verification-before-completion`, commit
  `docs: expose situated cognitive agents V11`, push, and update PR #44.

## Plan self-review

- Spec coverage: private admission, belief update, planning, physical compilation,
  delayed observation timing, explanations, replay, and office propagation each map
  to an explicit task and test.
- Type consistency: the runtime consumes only Task 1 public types and returns the Task
  2/3 records named above.
- Placeholder scan: no production placeholder or unspecified implementation step is
  permitted; every behavior is tied to a named test and exact verification command.
