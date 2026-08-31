# Agent Autonomy V6 Design

## Objective

Add a deterministic decision layer to the unified evolving network so active agents use local information, post-propagation belief, learned source trust, fixed role policy, and finite verification resources to choose whether to share, remain silent, verify one received transmission, or exit. Every choice is emitted as an auditable intent and affects later rounds through one autonomy state chain.

V6 keeps entry invitations and death as environment events. Voluntary `EXIT` is no longer accepted from the environment; it is produced only by agent policy. Objective truth remains an environment observation, but it is accepted only for the exact transmission an agent selected for verification.

## Role decision policy

Every catalogued role has exactly one `RoleDecisionPolicy`:

- `sharing_enabled`: whether the role may share at all;
- `share_belief_threshold`: minimum post-round belief for choosing `SHARE`;
- `verify_trust_threshold`: maximum prior source trust eligible for verification;
- `exit_belief_threshold`: optional belief level below which the agent exits;
- `initial_verification_budget`: non-negative number of verifications available to each agent with that role.

An agent chooses `SHARE` only when sharing is enabled, its post-round belief meets both the role threshold and the base profile broadcast threshold, and it is not exiting. Otherwise it chooses `SILENT`.

An active agent may verify at most one same-round incoming transmission. It chooses the source edge with the lowest prior trust, using canonical edge identity as the tie-breaker, when that trust is at or below the role verification threshold and budget remains. Verification decrements budget exactly once.

An agent exits when `exit_belief_threshold` is not `None` and its post-round belief is strictly below that value. It becomes inactive after its current-round observation and optional verification, and cannot transmit in later rounds until an environment `ENTER` event returns it.

## Autonomous round order

Round `t + 1` executes exactly:

1. Apply environment `ENTER` and `DEATH` events. Reject environment `EXIT`.
2. Recompute the sharing choice of entering agents from their entry belief and role policy.
3. Build the active induced graph from prior topology/trust and prior sharing choices.
4. Propagate synchronously. A source transmits only when active, topology-active, V1 broadcast-eligible, and currently choosing to share.
5. Each active agent observes only transmissions targeted to itself plus the prior trust of those sources.
6. Emit one canonical `AgentActionIntent` per active agent: `SHARE` or `SILENT`, optional verification edge, optional exit.
7. Require truth observations to cover exactly the selected verification edges, then update trust.
8. Apply autonomous exits.
9. Rewire only candidate edges whose endpoints remain active, using post-propagation beliefs.
10. Commit evolving state, resource counters, choices, and parent identities to one autonomy next state.

Trust learning, sharing choices, exits, and topology changes affect only later propagation. No current-round decision can retroactively change a transmission.

## Contracts

`AutonomousNetworkModel` binds one exact `EvolvingNetworkModel` and a canonical complete role-policy tuple. Unknown, missing, or duplicate role policies are rejected.

`AutonomousAgentState` stores agent id, current sharing choice, remaining verification budget, cumulative verification count, cumulative decision count, and cumulative silent decision count.

`AutonomousPopulationState` stores one exact `EvolvingPopulationState` and one resource state per catalogued agent. Round indices and parent chains must agree across wrapper and embedded states.

`TruthObservation` identifies an exact candidate edge and objective truth in `[0, 1]`.

`AgentActionIntent` records round, agent id, role, observed belief, incoming transmission count, mean incoming trust or `None`, sharing decision, optional verification edge, exit choice, and verification budget before/after. The budget delta must match whether verification was selected.

`AutonomousRoundResult` records environment events, transmissions, action intents, truth observations, trust updates, rewiring updates, generated autonomous exit events, and the exact next state. `AutonomousTrajectory` validates paired environment-event and truth-observation schedules.

## Resource and inactivity semantics

Budgets and cumulative counters persist while an agent is inactive or dead. Inactive/dead agents have `sharing=False` and produce no intent. An entering agent retains its remaining budget and history; its sharing choice is recomputed before propagation from the entry belief. A dead agent remains unable to enter under V5 lifecycle rules.

## Behavior metrics

`AgentAutonomyMetrics` reports:

- cumulative decision count;
- cumulative verification count and verification rate;
- cumulative silent decision count and silence rate;
- cumulative autonomous exit count and exit rate;
- total remaining verification budget;
- verification budget utilization against role-derived initial total budget;
- current active sharing count and active sharing rate, defined as zero when no agent is active.

These state-derived metrics remain comparable across trajectories without retaining every prior result in memory.

## Acceptance scenarios

1. A relay receives a low-trust transmission, spends one budget unit, emits a verification intent, and learns trust only after propagation.
2. The same relay cannot verify after its budget reaches zero.
3. A role with sharing disabled remains silent even when belief is `1` and produces no outgoing transmission in the following round.
4. A low-belief active agent emits an exit intent, may finish a selected verification, becomes inactive, and does not transmit later.
5. Environment `EXIT` events are rejected; environment `ENTER` and `DEATH` remain valid.
6. Truth observations missing a selected edge or covering an unselected edge are rejected before a next state is returned.
7. Agents observe only incoming transmissions and prior trust; unrelated edges cannot influence their verification choice.
8. Lowest-trust verification selection and all intent/result hashes are invariant to catalog, event, and truth input order.
9. Zero active agents produce no intents, observations, trust updates, or rewiring updates and advance deterministically.
10. Behavior metrics match hand-computed decisions, budgets, silence, verification, sharing, and autonomous exits.

## Deferred work

- endogenous entry and mortality hazards;
- dynamic role transitions;
- utility optimization or learned decision policies;
- delayed/noisy verification results;
- verification cost heterogeneity beyond integer budget;
- multi-topic beliefs and claim-specific decisions.
