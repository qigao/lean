# Situated Cognitive Agents V11 Design

## Objective

Turn V10's independent embodied actors into autonomous cognitive agents. Each agent
must update a private probabilistic belief from only its own projected observations,
choose a physical action by finite-horizon POMDP planning against declared goals,
and retain an auditable decision record explaining what it knew and why the selected
action had the highest policy probability.

V11 connects to the existing planning semantics by reusing
`PlanningBeliefState` and the shared `finite_softmax`. It does not pass the objective
`SituatedWorldState`, another agent's mind, or the complete event ledger into an
agent decision context.

## Cognitive model

A `SituatedCognitiveModel` binds the exact V10 world model and one
`SituatedAgentCognitiveModel` for every embodied agent. Each agent model declares:

- a finite hidden-hypothesis space;
- a prior `PlanningBeliefState` over exactly those hypotheses;
- observation symbols and declarative rules mapping private V10 event views to them;
- likelihoods `P(symbol | hypothesis)` for Bayesian admission;
- physical action specifications and a finite action schedule;
- optional hypothesis transitions `P(h' | h, action)`, defaulting to identity;
- weighted goals and per-goal state/action rewards;
- discount and inverse-temperature values.

An action specification compiles to one V10 `SituatedActionIntent`. It may constrain
the actor's known current place, be non-repeatable, and require a previously observed
`INSPECT` or `TELL` event to cite as the source of a local telling. Constraints use
only the agent's own mind state.

## Private mind state

`SituatedAgentMindState` contains only:

- the agent's current belief distribution;
- its known own place;
- observation IDs already admitted;
- event IDs it personally observed;
- actions it previously selected;
- decision count.

`SituatedCognitiveState` is the simulator-owned collection of private minds. It binds
the exact cognitive model, the exact current story hash, and one parent state. The
runtime presents only one mind plus its new private event views to that agent's
belief/planning calculation; collection storage does not create shared knowledge.

## Round order

One cognitive round executes deterministically:

1. Validate that model, story, and cognitive state bind exactly.
2. For each agent, select only observations in its V10 perspective that have not yet
   been admitted.
3. Apply matching observation rules in chronological order and update its belief by
   Bayes' rule. Unmatched physical events remain remembered but do not change belief.
4. Derive feasible root actions from own-place, repeatability, and private-source
   constraints.
5. Solve finite-horizon soft Bellman recursion over declared hypotheses, transitions,
   observation likelihoods, rewards, and action schedule.
6. Produce a complete root policy with shared `finite_softmax`; choose lexical MAP.
7. Compile one intent per agent and resolve them together through V10.
8. Update each mind's known own place from its own successful movement, append only
   its projected event IDs, and commit the next parent-linked cognitive state/story.

New observations created in step 7 affect planning in the following round. This
prevents same-round circular causality.

## Decision explanations

Every `SituatedCognitiveDecision` records:

- newly admitted observation and symbol IDs;
- prior and posterior belief;
- feasible actions;
- root action values and policy;
- selected action and exact compiled V10 intent;
- the selected action's weighted contribution from every declared goal.

`explain_situated_decision` returns the selected action, most likely hypothesis,
policy probability, expected value, goal contributions, and admitted evidence IDs.
This is a faithful explanation of the declared model computation, not a post-hoc
invention of hidden motives.

## Office acceptance scenario

The V10 office is run without authored round actions:

1. Alice initially selects `INSPECT memo` under uncertainty.
2. Her private inspection maps to `approved`, updating only her belief.
3. Her non-repeatable inspection becomes unavailable; she moves to the open office.
4. At the open office she tells colleagues, citing her own inspection event.
5. Bob hears the telling, updates his own belief next round, and independently
   selects a retelling action citing Alice's event.
6. Dana, isolated in the manager office, neither updates nor acts on the claim.

The objective story, private perspectives, information chain, beliefs, actions, and
decision explanations replay identically.

## Deferred work

- continuous or learned state/action models;
- natural-language generation of goals, hypotheses, or rewards;
- theory of mind and beliefs about another agent's beliefs;
- bounded-memory forgetting and belief revision under contradictory testimony;
- stochastic action sampling (V11 uses deterministic lexical MAP);
- population birth/death integration and group-level policy learning.
