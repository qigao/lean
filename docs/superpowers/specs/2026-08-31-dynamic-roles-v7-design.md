# Dynamic Roles V7 Design

## Objective

Make role membership an endogenous, auditable part of the evolving network state. An active agent uses its role at the start of a round to govern V6 sharing, verification, and exit decisions; after that round, it may transition to another configured role from its own post-round belief, accumulated actions, and time spent in the role. The new role affects behavior only from the following round.

V7 changes role assignment, not immutable agent identity. Susceptibility and the base broadcast threshold remain properties of the catalogued agent profile. Role decision policies remain model configuration. Verification budget remains an agent-owned finite resource: changing roles neither replenishes nor removes it.

## Transition policy

`RoleTransitionRule` defines one directed transition with:

- `from_role` and `to_role` from the complete V6 role-policy catalog;
- a non-negative `priority`, unique among rules leaving one role;
- optional inclusive `minimum_belief` and exclusive `maximum_belief` bounds;
- non-negative minimum completed rounds in the current role;
- non-negative minimum cumulative decision and verification counts.

A rule must contain at least one substantive condition. If both belief bounds exist, the minimum must be strictly lower than the maximum. Self-transitions are rejected.

After an agent has completed the current round, all rules leaving its current role are evaluated against only that agent's post-round belief, cumulative V6 counters, and completed role tenure. The matching rule with the numerically smallest priority wins. Rules and results are canonically ordered, so input order cannot affect replay.

## Round order

Round `t + 1` executes exactly:

1. Validate the dynamic-role wrapper, embedded V6 state, exact agent-role catalog, and role values.
2. Apply V6 environment entry/death and propagation using every agent's role from dynamic state at `t`.
3. Produce V6 local intents, selected verification, trust learning, autonomous exits, and rewiring with the same timing as V6.
4. For each agent still active after V6, evaluate outgoing role rules from its prior role using post-round local evidence.
5. Emit at most one canonical `RoleTransitionRecord` per active agent.
6. Commit new role assignments and tenure counters in one dynamic-role state whose parent is the exact prior dynamic state. The embedded V6 chain remains exact.

The old role governs the whole current round. A transition cannot retroactively change current propagation, verification eligibility, exit policy, or rewiring.

## State and audit contracts

`DynamicRoleModel` binds one exact `AutonomousNetworkModel` and a canonical transition-rule tuple. Every referenced role must exist in the V6 policy catalog. Roles may have no outgoing rule.

`DynamicRoleAgentState` stores agent id, current role, completed active rounds in that role, and cumulative transition count. Initial roles come from immutable base profiles with zero tenure and zero transitions.

`DynamicRolePopulationState` embeds one exact `AutonomousPopulationState` plus exactly one role state for every catalogued agent. Wrapper and embedded rounds must agree. Inactive and dead agents retain their current role, tenure, and transition history.

`RoleTransitionRecord` stores the round, agent, source/destination roles, matched priority, post-round belief, completed source-role tenure, cumulative decisions, and cumulative verifications. These fields make every transition independently explainable.

`DynamicRoleRoundResult` embeds the exact V6 result, canonical transition records, and exact next wrapper state. `DynamicRoleTrajectory` validates paired V6 schedules and the complete state chain.

## V6 integration boundary

V6 gains an internal role-override path used only by V7. The public V6 functions continue to use immutable profile roles and preserve their hashes and behavior. The override must cover the exact agent catalog and contain only configured roles. It influences entrant sharing and local decision-policy lookup, including the role written into `AgentActionIntent`; it does not alter base cognitive parameters or budget-history validation.

## Role-level emergence metrics

`RoleDynamicsMetrics` reports:

- current active population;
- one count and share for every configured role, including zero-count roles;
- normalized Shannon entropy of the active role distribution, or zero for an empty population or one-role catalog;
- cumulative transition count and transition rate per cumulative V6 decision;
- mean completed tenure among currently active agents.

The exact role-count tuple makes composition comparable between treatments. Entropy measures diversity, not desirability. Zero-denominator rates are `0.0`.

## Acceptance scenarios

1. A relay that verifies a source and reaches the configured belief threshold becomes a source at the end of the round.
2. The transition record contains the exact local evidence that matched the rule.
3. The relay policy governs the transition round; the source policy first governs the next round.
4. A role transition does not replenish verification budget.
5. An agent that autonomously exits or dies during the round does not transition and retains its role state while inactive/dead.
6. A later environment entry uses the retained current role for initial sharing and current-round decisions.
7. Multiple matching rules select the smallest unique priority; rule and schedule input order do not alter hashes.
8. Agents with no matching rule remain in role and gain one completed active tenure round.
9. Zero active agents produce no transitions and advance deterministically without changing roles or tenure.
10. Role composition, normalized entropy, transition rate, and active tenure match hand-computed examples.

## Deferred work

- transition rules using neighborhood-level or population-level signals;
- role-specific replenishment, income, or resource exchange;
- learned or stochastic transition policies;
- simultaneous multi-role membership;
- creation and retirement of roles during a run;
- empirical estimation of transition thresholds, addressed by V8 calibration.
