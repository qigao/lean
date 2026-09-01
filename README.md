# Human–Social Narrative Dynamics

A formalization and simulation research spike for a human–social–historical narrative engine.

## Boundary

- **Lean** defines and proves the semantic kernel: typed world structure, epistemic isolation, provenance, admissibility, motivational identities, and invariants.
- **Python** runs and evaluates existing, learned, stochastic, or black-box models through canonical scenarios, seeds, traces, metrics, calibration, uncertainty, held-out validation, sensitivity, and adapter registration.
- Mature algorithms such as POMDP solvers, MCTS, HTN, GFlowNet, ASP, SBI, and mean-field solvers are wrapped rather than reimplemented in Lean.

## Python research runtime

The dependency-free `narrative_dynamics` package currently provides:

- immutable simulation contracts and deterministic seeded replay;
- metric aggregation and finite-grid calibration;
- synthetic parameter recovery;
- repeated calibration across seed blocks;
- explicit accepted parameter sets and coordinate-wise identifiability diagnostics;
- seed-block variation with accepted-set union/intersection;
- held-out scenario validation and accepted-set external filtering;
- common-seed central-difference sensitivity;
- explicit model categories and registry discovery;
- a grounded-goal adapter that reuses the formally aligned finite-goal softmax.

Registered models still execute only through `SimulationRunner`; registration does not make a third-party model trusted or empirically valid.

## Network interaction and emergence V1

The dependency-free `narrative_dynamics.abm` package adds a fixed-population
agent-based runtime. Directed social edges constrain information delivery;
belief, exposure, and broadcasting state persist across synchronous rounds;
macro metrics and paired network interventions make population effects explicit.

```python
from narrative_dynamics.abm import (
    EdgeSelector,
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkIntervention,
    SocialEdge,
    SocialNetwork,
    compare_intervention,
    initialize_population,
    measure_emergence,
    simulate_population,
)

agents = tuple(
    NetworkAgentSpec(agent_id, role, 1.0, 0.5, 0.5)
    for agent_id, role in (("a", "source"), ("b", "relay"), ("c", "recipient"))
)
model = NetworkABMModel(
    "line-network",
    "1",
    agents,
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0),
            SocialEdge("b", "c", "peer", 1.0),
        ),
    ),
)
initial = initialize_population(model, beliefs={"a": 1.0})
trajectory = simulate_population(model, initial, rounds=2)
metrics = measure_emergence(model, trajectory.final_state)
assert metrics.adoption_rate == 1.0

cut_bridge = NetworkIntervention(
    "cut-bridge",
    disabled_edges=(EdgeSelector("b", "c", "peer"),),
)
comparison = compare_intervention(model, initial, cut_bridge, rounds=2)
assert abs(comparison.metric_deltas["adoption_rate"] + 1 / 3) < 1e-12
```

V1 deliberately keeps identities, roles, and the population roster fixed. It
models information-driven behavioral change; population lifecycle, network
rewiring, learned parameters, and stochastic contact remain later phases.

### Adaptive source trust V2

Agents can also learn which neighboring sources are reliable. Current trust
scales edge influence; truth feedback updates trust after propagation, so the
new value affects the following round only.

```python
from narrative_dynamics.abm import (
    AdaptiveTrustModel,
    EdgeSelector,
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
    TruthFeedback,
    initialize_adaptive_population,
    measure_adaptive_trust,
    simulate_adaptive_round,
)

agents = (
    NetworkAgentSpec("accurate", "source", 1.0, 0.5, 0.5),
    NetworkAgentSpec("inaccurate", "source", 1.0, 0.5, 0.0),
    NetworkAgentSpec("target", "recipient", 1.0, 0.5, 0.5),
)
base = NetworkABMModel(
    "two-sources",
    "1",
    agents,
    SocialNetwork(
        ("accurate", "inaccurate", "target"),
        (
            SocialEdge("accurate", "target", "report", 1.0),
            SocialEdge("inaccurate", "target", "report", 1.0),
        ),
    ),
)
adaptive = AdaptiveTrustModel("adaptive-sources", "1", base, 0.5, 0.5)
state = initialize_adaptive_population(
    adaptive,
    beliefs={"accurate": 1.0, "inaccurate": 0.0},
)
feedback = (
    TruthFeedback(EdgeSelector("accurate", "target", "report"), 1.0),
    TruthFeedback(EdgeSelector("inaccurate", "target", "report"), 1.0),
)
learned = simulate_adaptive_round(adaptive, state, feedback=feedback)
assert measure_adaptive_trust(adaptive, learned.next_state).max_trust == 0.75

next_round = simulate_adaptive_round(adaptive, learned.next_state)
target = next(item for item in next_round.next_state.agents if item.agent_id == "target")
assert target.belief == 0.75
```

### Population lifecycle V3

The effective agent set can now change at round boundaries. Entry, exit, and
death apply before information propagation; only active agents and the edges
between them participate in that round. The complete identity catalog remains
fixed so every lifecycle transition and state hash stays auditable.

```python
from narrative_dynamics.abm import (
    LifecycleEventKind,
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    SocialEdge,
    SocialNetwork,
    initialize_lifecycle_population,
    measure_population_lifecycle,
    simulate_lifecycle_round,
)

catalog = NetworkABMModel(
    "line-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "entrant", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(("a", "b"), (SocialEdge("a", "b", "peer", 1.0),)),
)
lifecycle = PopulationLifecycleModel(
    "changing-population",
    "1",
    catalog,
    initial_active_agent_ids=("a",),
)
state = initialize_lifecycle_population(lifecycle, beliefs={"a": 1.0})
entered = simulate_lifecycle_round(
    lifecycle,
    state,
    events=(PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),),
)
b = next(item for item in entered.next_state.members if item.agent_id == "b")
metrics = measure_population_lifecycle(lifecycle, entered.next_state)
assert b.belief == 1.0
assert metrics.active_population == 2
assert metrics.cumulative_entries == 2
```

### Endogenous network rewiring V4

Relationships can now respond to the information-driven beliefs of their
endpoints. Each round propagates over the prior topology, then applies
hysteretic similarity thresholds; formed or dissolved edges affect the next
round only. Candidate relationship identities remain fixed and auditable.

```python
from narrative_dynamics.abm import (
    EndogenousRewiringModel,
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
    initialize_rewiring_population,
    measure_network_structure,
    simulate_rewiring_population,
)

catalog = NetworkABMModel(
    "rewiring-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0, active=True),
            SocialEdge("b", "c", "peer", 1.0, active=False),
        ),
    ),
)
rewiring = EndogenousRewiringModel(
    "belief-driven-network",
    "1",
    catalog,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
initial = initialize_rewiring_population(
    rewiring,
    beliefs={"a": 1.0, "b": 0.0, "c": 1.0},
)
trajectory = simulate_rewiring_population(rewiring, initial, rounds=2)
first_round_edges = trajectory.rounds[0].transmissions
second_round_edges = trajectory.rounds[1].transmissions
metrics = measure_network_structure(rewiring, trajectory.final_state)
assert len(first_round_edges) == 1
assert len(second_round_edges) == 2
assert metrics.active_edge_count == 2
assert metrics.cumulative_rewirings == 1
```

### Unified evolving network V5

Lifecycle, learned source trust, and endogenous topology can run in one atomic
state transition. Lifecycle events determine current participants; prior trust
and topology determine current propagation; feedback learning and rewiring are
committed together for the next round.

```python
from narrative_dynamics.abm import (
    EdgeSelector,
    EvolvingNetworkModel,
    LifecycleEventKind,
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationLifecycleEvent,
    SocialEdge,
    SocialNetwork,
    TruthFeedback,
    initialize_evolving_population,
    measure_evolving_system,
    simulate_evolving_round,
)

catalog = NetworkABMModel(
    "unified-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0, active=True),
            SocialEdge("b", "c", "peer", 1.0, active=False),
        ),
    ),
)
evolving = EvolvingNetworkModel(
    "unified-evolution",
    "1",
    catalog,
    initial_active_agent_ids=("a", "c"),
    learning_rate=0.5,
    initial_trust=0.5,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
initial = initialize_evolving_population(
    evolving,
    beliefs={"a": 1.0, "c": 0.5},
)
result = simulate_evolving_round(
    evolving,
    initial,
    events=(PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),),
    feedback=(TruthFeedback(EdgeSelector("a", "b", "peer"), 1.0),),
)
b = next(item for item in result.next_state.members if item.agent_id == "b")
ab_trust = next(
    item.trust
    for item in result.next_state.edge_trust
    if item.edge.source_agent_id == "a"
)
bc_active = next(
    item.active
    for item in result.next_state.edge_topology
    if item.edge.source_agent_id == "b"
)
metrics = measure_evolving_system(evolving, result.next_state)
assert b.belief == 0.5
assert ab_trust == 0.75
assert bc_active
assert metrics.active_population == 3
assert metrics.mean_trust == 0.625
```

### Agent autonomy V6

Active agents can make deterministic local decisions from post-round belief,
received transmissions, prior source trust, role policy, and finite verification
budget. Decisions to share, remain silent, verify, or exit are stored as
auditable intents and affect later rounds only.

```python
from narrative_dynamics.abm import (
    AutonomousNetworkModel,
    EdgeSelector,
    EvolvingNetworkModel,
    LifecycleEventKind,
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationLifecycleEvent,
    RoleDecisionPolicy,
    SocialEdge,
    SocialNetwork,
    TruthObservation,
    initialize_autonomous_population,
    measure_agent_autonomy,
    simulate_autonomous_round,
)

catalog = NetworkABMModel(
    "autonomy-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0, active=True),
            SocialEdge("b", "c", "peer", 1.0, active=False),
        ),
    ),
)
evolving = EvolvingNetworkModel(
    "autonomous-evolution",
    "1",
    catalog,
    initial_active_agent_ids=("a", "c"),
    learning_rate=0.5,
    initial_trust=0.5,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
autonomous = AutonomousNetworkModel(
    "local-decisions",
    "1",
    evolving,
    (
        RoleDecisionPolicy("source", True, 0.5, 0.4, None, 1),
        RoleDecisionPolicy("relay", True, 0.5, 0.6, 0.2, 2),
        RoleDecisionPolicy("recipient", False, 0.5, 0.6, 0.2, 1),
    ),
)
initial = initialize_autonomous_population(
    autonomous,
    beliefs={"a": 1.0, "c": 0.5},
)
result = simulate_autonomous_round(
    autonomous,
    initial,
    environment_events=(
        PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
    ),
    truth_observations=(
        TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),
    ),
)
relay_intent = next(item for item in result.intents if item.agent_id == "b")
relay_resource = next(item for item in result.next_state.agents if item.agent_id == "b")
learned_trust = next(
    item.trust
    for item in result.next_state.evolving_state.edge_trust
    if item.edge.source_agent_id == "a"
)
metrics = measure_agent_autonomy(autonomous, result.next_state)
assert relay_intent.verification_edge == EdgeSelector("a", "b", "peer")
assert relay_resource.remaining_verification_budget == 1
assert learned_trust == 0.75
assert metrics.verification_count == 1
assert metrics.active_sharing_count == 2
```

### Dynamic roles V7

Roles can evolve from each agent's own post-round belief, action history, and
time in its current role. The old role governs the complete current round; a
matched transition is auditable and changes policy lookup only in later rounds.
Changing role never replenishes the agent's finite verification budget.

```python
from narrative_dynamics.abm import (
    AutonomousNetworkModel,
    DynamicRoleModel,
    EdgeSelector,
    EvolvingNetworkModel,
    LifecycleEventKind,
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationLifecycleEvent,
    RoleDecisionPolicy,
    RoleTransitionRule,
    SocialEdge,
    SocialNetwork,
    TruthObservation,
    initialize_dynamic_role_population,
    measure_role_dynamics,
    simulate_dynamic_role_round,
)

catalog = NetworkABMModel(
    "role-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0, active=True),
            SocialEdge("b", "c", "peer", 1.0, active=False),
        ),
    ),
)
evolving = EvolvingNetworkModel(
    "role-evolution",
    "1",
    catalog,
    initial_active_agent_ids=("a", "c"),
    learning_rate=0.5,
    initial_trust=0.5,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
autonomy = AutonomousNetworkModel(
    "role-autonomy",
    "1",
    evolving,
    (
        RoleDecisionPolicy("source", True, 0.5, 0.4, None, 1),
        RoleDecisionPolicy("relay", True, 0.5, 0.6, 0.2, 2),
        RoleDecisionPolicy("recipient", False, 0.5, 0.6, 0.2, 1),
    ),
)
roles = DynamicRoleModel(
    "endogenous-roles",
    "1",
    autonomy,
    (
        RoleTransitionRule(
            "relay",
            "source",
            priority=0,
            minimum_belief=0.5,
            minimum_rounds_in_role=1,
            minimum_verification_count=1,
        ),
    ),
)
initial = initialize_dynamic_role_population(
    roles,
    beliefs={"a": 1.0, "c": 0.5},
)
first = simulate_dynamic_role_round(
    roles,
    initial,
    environment_events=(
        PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
    ),
    truth_observations=(
        TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),
    ),
)
transition = next(item for item in first.transitions if item.agent_id == "b")
relay_resource = next(
    item for item in first.next_state.autonomy_state.agents if item.agent_id == "b"
)
second = simulate_dynamic_role_round(
    roles,
    first.next_state,
    truth_observations=(
        TruthObservation(EdgeSelector("b", "c", "peer"), 1.0),
    ),
)
next_intent = next(
    item for item in second.autonomy_result.intents if item.agent_id == "b"
)
metrics = measure_role_dynamics(roles, first.next_state)
source_count = next(
    item.count for item in metrics.active_role_counts if item.role == "source"
)
assert transition.from_role == "relay"
assert transition.to_role == "source"
assert relay_resource.remaining_verification_budget == 1
assert next_intent.role == "source"
assert metrics.transition_count == 1
assert source_count == 2
```

### Empirical calibration V8

A frozen empirical dataset separates training cases from untouched holdout
cases. Calibration exhaustively replays every declared candidate, minimizes
weighted trajectory error on training cases only, and retains per-case snapshot
hashes and holdout loss for audit.

```python
from narrative_dynamics.abm import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    AutonomousNetworkModel,
    CalibrationSplit,
    DynamicRoleModel,
    EdgeSelector,
    EmpiricalABMCase,
    EmpiricalABMDataset,
    EvolvingNetworkModel,
    LifecycleEventKind,
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationLifecycleEvent,
    RoleDecisionPolicy,
    RoleTransitionRule,
    SocialEdge,
    SocialNetwork,
    TruthObservation,
    apply_calibration_candidate,
    calibrate_dynamic_role_model,
    initialize_dynamic_role_population,
    observe_dynamic_role_state,
    simulate_dynamic_role_population,
)

catalog = NetworkABMModel(
    "calibration-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b", "c"),
        (
            SocialEdge("a", "b", "peer", 1.0, active=True),
            SocialEdge("b", "c", "peer", 1.0, active=False),
        ),
    ),
)
evolving = EvolvingNetworkModel(
    "calibration-evolution",
    "1",
    catalog,
    initial_active_agent_ids=("a", "c"),
    learning_rate=0.5,
    initial_trust=0.5,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
autonomy = AutonomousNetworkModel(
    "calibration-autonomy",
    "1",
    evolving,
    (
        RoleDecisionPolicy("source", True, 0.5, 0.4, None, 1),
        RoleDecisionPolicy("relay", True, 0.5, 0.6, 0.2, 2),
        RoleDecisionPolicy("recipient", False, 0.5, 0.6, 0.2, 1),
    ),
)
base = DynamicRoleModel(
    "calibration-roles",
    "1",
    autonomy,
    (
        RoleTransitionRule(
            "relay",
            "source",
            priority=0,
            minimum_belief=0.5,
            minimum_rounds_in_role=1,
            minimum_verification_count=1,
        ),
    ),
)
truth = ABMCalibrationCandidate(0.5, 0.2, 0.8)
alternative = ABMCalibrationCandidate(0.2, 0.6, 0.7)
environment_schedule = ((
    PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
),)
truth_schedule = ((
    TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),
),)


def observed_case(case_id, split, beliefs):
    generating_model = apply_calibration_candidate(base, truth)
    initial = initialize_dynamic_role_population(
        generating_model,
        beliefs=dict(beliefs),
    )
    trajectory = simulate_dynamic_role_population(
        generating_model,
        initial,
        environment_event_schedule=environment_schedule,
        truth_observation_schedule=truth_schedule,
    )
    observations = tuple(
        observe_dynamic_role_state(generating_model, item.next_state)
        for item in trajectory.rounds
    )
    return EmpiricalABMCase(
        case_id,
        split,
        beliefs,
        environment_schedule,
        truth_schedule,
        observations,
    )


dataset = EmpiricalABMDataset(
    "observed-network",
    "1",
    (
        observed_case(
            "train",
            CalibrationSplit.TRAIN,
            (("a", 1.0), ("c", 0.0)),
        ),
        observed_case(
            "holdout",
            CalibrationSplit.HOLDOUT,
            (("a", 0.8), ("c", 0.2)),
        ),
    ),
)
report = calibrate_dynamic_role_model(
    base,
    dataset,
    candidates=(alternative, truth),
    weights=ABMCalibrationWeights.uniform(),
)
selected = next(
    item for item in report.candidate_fits if item.candidate == report.selected_candidate
)
assert report.selected_candidate == truth
assert selected.training_loss == 0.0
assert selected.holdout_loss == 0.0
assert all(
    fit.observed_snapshot_hashes == fit.predicted_snapshot_hashes
    for fit in selected.case_fits
)
```

### Calibrated experiment suite V9

Experiment protocols lock the V8-selected candidate as baseline, replay every
arm on the same holdout cases, and report signed effects and objective-specific
ranking without reusing training cases as outcomes.

```python
from narrative_dynamics.abm import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    ABMExperimentArm,
    ABMExperimentProtocol,
    AutonomousNetworkModel,
    CalibrationSplit,
    DynamicRoleModel,
    EdgeSelector,
    EmpiricalABMCase,
    EmpiricalABMDataset,
    EvolvingNetworkModel,
    ExperimentObjective,
    NetworkABMModel,
    NetworkAgentSpec,
    RoleDecisionPolicy,
    RoleTransitionRule,
    SocialEdge,
    SocialNetwork,
    TruthObservation,
    apply_calibration_candidate,
    calibrate_dynamic_role_model,
    initialize_dynamic_role_population,
    observe_dynamic_role_state,
    run_calibrated_abm_experiment,
    simulate_dynamic_role_population,
)

catalog = NetworkABMModel(
    "experiment-catalog",
    "1",
    (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "recipient", 1.0, 0.5, 0.5),
    ),
    SocialNetwork(
        ("a", "b"),
        (SocialEdge("a", "b", "peer", 1.0, active=True),),
    ),
)
evolving = EvolvingNetworkModel(
    "experiment-evolution",
    "1",
    catalog,
    initial_active_agent_ids=("a", "b"),
    learning_rate=0.5,
    initial_trust=0.5,
    dissolution_similarity=0.2,
    formation_similarity=0.8,
)
autonomy = AutonomousNetworkModel(
    "experiment-autonomy",
    "1",
    evolving,
    (
        RoleDecisionPolicy("source", True, 0.5, 0.4, None, 0),
        RoleDecisionPolicy("recipient", False, 0.5, 1.0, None, 1),
    ),
)
base = DynamicRoleModel(
    "experiment-roles",
    "1",
    autonomy,
    (
        RoleTransitionRule(
            "recipient",
            "source",
            priority=0,
            minimum_belief=0.4,
            minimum_rounds_in_role=1,
            minimum_verification_count=1,
        ),
    ),
)
baseline_candidate = ABMCalibrationCandidate(0.5, 0.2, 0.8)
treatment_candidate = ABMCalibrationCandidate(0.2, 0.2, 0.8)
environment_schedule = ((),)
truth_schedule = ((
    TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),
),)


def experiment_case(case_id, split, b_belief):
    generator = apply_calibration_candidate(base, baseline_candidate)
    beliefs = (("a", 1.0), ("b", b_belief))
    initial = initialize_dynamic_role_population(generator, beliefs=dict(beliefs))
    trajectory = simulate_dynamic_role_population(
        generator,
        initial,
        environment_event_schedule=environment_schedule,
        truth_observation_schedule=truth_schedule,
    )
    return EmpiricalABMCase(
        case_id,
        split,
        beliefs,
        environment_schedule,
        truth_schedule,
        tuple(
            observe_dynamic_role_state(generator, item.next_state)
            for item in trajectory.rounds
        ),
    )


dataset = EmpiricalABMDataset(
    "experiment-observations",
    "1",
    (
        experiment_case("train", CalibrationSplit.TRAIN, 0.0),
        experiment_case("holdout", CalibrationSplit.HOLDOUT, 0.2),
    ),
)
calibration = calibrate_dynamic_role_model(
    base,
    dataset,
    candidates=(treatment_candidate, baseline_candidate),
    weights=ABMCalibrationWeights.uniform(),
)
protocol = ABMExperimentProtocol(
    "trust-treatment",
    "1",
    "baseline",
    "mean_trust",
    ExperimentObjective.MAXIMIZE,
    (
        ABMExperimentArm("baseline", baseline_candidate),
        ABMExperimentArm("treatment", treatment_candidate),
    ),
)
experiment = run_calibrated_abm_experiment(
    base,
    dataset,
    calibration,
    protocol,
)
arm_results = {item.arm.arm_id: item for item in experiment.arm_results}
assert calibration.selected_candidate == baseline_candidate
assert experiment.ranking == ("baseline", "treatment")
assert arm_results["baseline"].delta_from_baseline == 0.0
assert arm_results["treatment"].delta_from_baseline < 0.0
assert tuple(
    item.case_id for item in arm_results["baseline"].case_results
) == ("holdout",)
```

### Situated story world V10

V10 places independent agents in a physical topology. Actions change one objective
world, but each agent receives only local observations. The analyst can reconstruct
the grounded event and information chain without giving agents access to global
truth.

```python
from narrative_dynamics.abm import (
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageSpec,
    PlaceSpec,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedWorldModel,
    WorldObjectSpec,
    advance_situated_story,
    information_chain,
    initialize_situated_story,
    initialize_situated_world,
    perspective_timeline,
)

office = SituatedWorldModel(
    "office-story",
    "1",
    (
        PlaceSpec("records", "Records room"),
        PlaceSpec("open", "Open office"),
        PlaceSpec("manager", "Manager office"),
    ),
    (PassageSpec("records-open", "records", "open"),),
    (
        EmbodiedAgentSpec("alice", "analyst", "records", 1),
        EmbodiedAgentSpec("bob", "engineer", "open", 1),
        EmbodiedAgentSpec("dana", "manager", "manager", 1),
    ),
    (
        WorldObjectSpec(
            "memo",
            "official memo",
            "records",
            portable=True,
            evidence=(EvidenceFact("restructuring", "approved"),),
        ),
    ),
)
initial = initialize_situated_world(office)
story = initialize_situated_story(office, initial)
story = advance_situated_story(office, story, (
    SituatedActionIntent(
        "inspect-memo", "alice", SituatedActionKind.INSPECT, "memo"
    ),
))
inspection = next(
    event for event in story.rounds[-1].events if event.actor_agent_id == "alice"
)
story = advance_situated_story(office, story, (
    SituatedActionIntent(
        "move-to-open", "alice", SituatedActionKind.MOVE, "records-open"
    ),
))
story = advance_situated_story(office, story, (
    SituatedActionIntent(
        "tell-bob",
        "alice",
        SituatedActionKind.TELL,
        message="The restructuring is approved.",
        source_event_ids=(inspection.event_id,),
    ),
))
telling = next(
    event for event in story.rounds[-1].events if event.actor_agent_id == "alice"
)

alice_view = perspective_timeline(story, "alice")
bob_view = perspective_timeline(story, "bob")
dana_view = perspective_timeline(story, "dana")
assert inspection.event_id in {item.event.event_id for item in alice_view}
assert inspection.event_id not in {item.event.event_id for item in bob_view}
assert telling.event_id in {item.event.event_id for item in bob_view}
assert telling.event_id not in {item.event.event_id for item in dana_view}
assert tuple(item.event_id for item in information_chain(story, telling.event_id)) == (
    inspection.event_id,
    telling.event_id,
)
```

### Perceptual environment graph V15

V15 adds a deterministic analyst/query projection over an exact situated-world
round. Perception edges are directed: an open meeting-room door permits visual,
clear auditory, and direct interaction access, while a closed door leaves only a
weaker auditory edge. The closed-door projection below detects that something was
heard without exposing its actor, action, outcome, or details.

```python
from narrative_dynamics.abm import (
    EmbodiedAgentSpec,
    PassageSpec,
    PassageState,
    PlaceSpec,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
    SituatedPerceptFidelity,
    SituatedWorldModel,
    SituatedWorldState,
    can_situated_agents_interact,
    derive_situated_perception_reach,
    initialize_situated_world,
    percepts_for_agent,
    project_situated_percepts,
    resolve_situated_round,
)

office = SituatedWorldModel(
    "perception-office",
    "1",
    (PlaceSpec("corridor", "Corridor"), PlaceSpec("meeting", "Meeting room")),
    (PassageSpec("meeting-door", "corridor", "meeting", initially_open=False),),
    (
        EmbodiedAgentSpec("alice", "analyst", "corridor"),
        EmbodiedAgentSpec("bob", "manager", "meeting"),
    ),
)
perception_model = SituatedPerceptionModel(
    "office-perception",
    "1",
    office,
    (
        SituatedPerceptionEdge(
            "door-visual-open", SituatedPerceptionLayer.VISIBILITY,
            "corridor", "meeting", 1.0,
            SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door",
        ),
        SituatedPerceptionEdge(
            "door-audio-open", SituatedPerceptionLayer.AUDITORY,
            "corridor", "meeting", 5.0,
            SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door",
        ),
        SituatedPerceptionEdge(
            "door-audio-closed", SituatedPerceptionLayer.AUDITORY,
            "corridor", "meeting", 25.0,
            SituatedEdgeActivation.PASSAGE_CLOSED, "meeting-door",
        ),
        SituatedPerceptionEdge(
            "door-interaction-open", SituatedPerceptionLayer.INTERACTION,
            "corridor", "meeting", 0.0,
            SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door",
        ),
    ),
    (
        SituatedAgentPerceptionProfile("alice", 1.0, 20.0, 45.0),
        SituatedAgentPerceptionProfile("bob", 1.0, 20.0, 45.0),
    ),
    (SituatedEventSignalProfile(SituatedActionKind.TELL, True, 60.0),),
)
closed_state = initialize_situated_world(office)
open_state = SituatedWorldState(
    closed_state.model_id,
    closed_state.model_hash,
    closed_state.round_index,
    closed_state.parent_state_hash,
    closed_state.agents,
    closed_state.objects,
    (PassageState("meeting-door", True),),
)
open_reach = derive_situated_perception_reach(
    perception_model, open_state, source_place_id="corridor"
)
assert open_reach.visual_costs["meeting"] == 1.0
assert open_reach.auditory_losses["meeting"] == 5.0

tell_round = resolve_situated_round(
    office,
    closed_state,
    (SituatedActionIntent(
        "alice-tells-secret", "alice", SituatedActionKind.TELL,
        message="The meeting is confidential.",
    ),),
)
closed = project_situated_percepts(perception_model, tell_round)
bob = percepts_for_agent(closed, "bob")
assert bob[0].fidelity is SituatedPerceptFidelity.DETECTED
assert bob[0].details == ()
assert bob[0].actor_agent_id is None
assert can_situated_agents_interact(
    perception_model, tell_round.prior_state, "alice", "bob"
) is False
```

The base V15 projection remains query-only and preserves every V10-V14 API. The
explicit V15.1 entry points below integrate those percepts with cognition and memory
without silently changing legacy behavior. See the [perceptual-environment
architecture spec](docs/superpowers/specs/2026-08-31-simulated-story-production-architecture-design.md).

### Situated cognitive agents V11

V11 lets those embodied agents update private probability distributions and choose
their own physical actions with finite-horizon planning. Decision records preserve
the evidence, belief, action values, policy, and weighted goal contributions that
actually produced each action.

```python
from narrative_dynamics.abm import (
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageSpec,
    PlaceSpec,
    SituatedActionKind,
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedCognitiveModel,
    SituatedGoalReward,
    SituatedGoalSpec,
    SituatedHypothesis,
    SituatedObservationLikelihood,
    SituatedObservationRule,
    SituatedObservationSymbol,
    SituatedWorldModel,
    WorldObjectSpec,
    explain_situated_decision,
    information_chain,
    initialize_situated_cognition,
    initialize_situated_story,
    initialize_situated_world,
    simulate_situated_cognition,
)
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState

world = SituatedWorldModel(
    "autonomous-office",
    "1",
    (PlaceSpec("records", "Records"), PlaceSpec("open", "Open office")),
    (PassageSpec("records-open", "records", "open"),),
    (
        EmbodiedAgentSpec("alice", "analyst", "records", 1),
        EmbodiedAgentSpec("bob", "engineer", "open", 1),
    ),
    (WorldObjectSpec(
        "memo", "official memo", "records", True,
        (EvidenceFact("restructuring", "approved"),),
    ),),
)
actions = (
    SituatedActionSpec("inspect", SituatedActionKind.INSPECT, "memo", required_place_ids=("records",), repeatable=False),
    SituatedActionSpec("move", SituatedActionKind.MOVE, "records-open", required_place_ids=("records",), repeatable=False),
    SituatedActionSpec(
        "tell", SituatedActionKind.TELL,
        message="The restructuring is approved.",
        required_place_ids=("open",),
        repeatable=False,
        source_event_kinds=(SituatedActionKind.INSPECT, SituatedActionKind.TELL),
    ),
    SituatedActionSpec("wait", SituatedActionKind.WAIT),
)
hypotheses = (
    SituatedHypothesis("approved", "approved"),
    SituatedHypothesis("denied", "denied"),
)
symbols = (
    SituatedObservationSymbol("approved", "supports approval"),
    SituatedObservationSymbol("denied", "opposes approval"),
)
rules = (
    SituatedObservationRule(
        "memo-approved", "approved", "inspect",
        SituatedActionKind.INSPECT, "inspected", "restructuring", "approved",
    ),
    SituatedObservationRule(
        "heard-approved", "approved", "tell",
        SituatedActionKind.TELL, "told", "message",
        "The restructuring is approved.",
    ),
)
likelihoods = tuple(
    SituatedObservationLikelihood(
        action.action_id,
        hypothesis.hypothesis_id,
        symbol.symbol_id,
        (0.9 if hypothesis.hypothesis_id == symbol.symbol_id else 0.1)
        if action.action_id in {"inspect", "tell"} else 0.5,
    )
    for action in actions
    for hypothesis in hypotheses
    for symbol in symbols
)
reward_values = {
    ("approved", "inspect"): 3.0, ("denied", "inspect"): 3.0,
    ("approved", "move"): 2.0, ("denied", "move"): 0.0,
    ("approved", "tell"): 4.0, ("denied", "tell"): -4.0,
    ("approved", "wait"): 0.0, ("denied", "wait"): 0.0,
}


def cognitive_agent(agent_id):
    return SituatedAgentCognitiveModel(
        agent_id,
        hypotheses,
        PlanningBeliefState({"approved": 0.5, "denied": 0.5}),
        symbols,
        rules,
        likelihoods,
        actions,
        (("inspect", "move", "tell", "wait"), ("move", "tell", "wait")),
        (),
        (SituatedGoalSpec("inform", "discover and share reliable information"),),
        tuple(
            SituatedGoalReward("inform", hypothesis, action, value)
            for (hypothesis, action), value in reward_values.items()
        ),
        0.8,
        4.0,
    )


cognition = SituatedCognitiveModel(
    "autonomous-office-cognition", "1", world,
    (cognitive_agent("alice"), cognitive_agent("bob")),
)
initial_story = initialize_situated_story(world, initialize_situated_world(world))
initial_minds = initialize_situated_cognition(cognition, initial_story)
trajectory = simulate_situated_cognition(
    cognition, initial_story, initial_minds, round_count=4
)
choices = {
    (result.next_state.round_index, decision.agent_id): decision.selected_action_id
    for result in trajectory.rounds
    for decision in result.decisions
}
assert choices[(1, "alice")] == "inspect"
assert choices[(2, "alice")] == "move"
assert choices[(3, "alice")] == "tell"
assert choices[(4, "bob")] == "tell"
bob_explanation = explain_situated_decision(trajectory.rounds[-1], "bob")
assert bob_explanation.most_likely_hypothesis_id == "approved"
bob_tell = next(
    event for event in trajectory.final_story.rounds[-1].events
    if event.actor_agent_id == "bob"
)
assert tuple(
    event.kind.value
    for event in information_chain(trajectory.final_story, bob_tell.event_id)
) == ("inspect", "tell", "tell")
```

### Agent-private long-term memory V12

V12 can persist those private perspectives across process restarts. Ordinary SQLite
rows retain the structured event provenance; FTS5 is only a rebuildable text index.
Every ingestion and query is explicitly scoped to one agent.

```python
from tempfile import TemporaryDirectory

from narrative_dynamics.abm import (
    EmbodiedAgentSpec,
    EvidenceFact,
    PlaceSpec,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedMemoryQuery,
    SituatedWorldModel,
    WorldObjectSpec,
    advance_situated_story,
    ingest_situated_story,
    initialize_situated_story,
    initialize_situated_world,
    search_situated_memories,
)

world = SituatedWorldModel(
    "memory-office",
    "1",
    (PlaceSpec("records", "Records"), PlaceSpec("manager", "Manager office")),
    (),
    (
        EmbodiedAgentSpec("alice", "analyst", "records", 1),
        EmbodiedAgentSpec("bob", "engineer", "records", 1),
        EmbodiedAgentSpec("dana", "manager", "manager", 1),
    ),
    (WorldObjectSpec(
        "memo", "official memo", "records", False,
        (EvidenceFact("restructuring", "approved"),),
    ),),
)
story = initialize_situated_story(world, initialize_situated_world(world))
story = advance_situated_story(world, story, (
    SituatedActionIntent(
        "inspect-memo", "alice", SituatedActionKind.INSPECT, "memo"
    ),
))
inspection = next(
    event for event in story.rounds[-1].events if event.actor_agent_id == "alice"
)
story = advance_situated_story(world, story, (
    SituatedActionIntent(
        "tell-bob",
        "alice",
        SituatedActionKind.TELL,
        message="The restructuring is approved.",
        source_event_ids=(inspection.event_id,),
    ),
))

with TemporaryDirectory() as temporary:
    database = f"{temporary}/memories.sqlite3"
    for agent_id in ("alice", "bob", "dana"):
        ingest_situated_story(database, story, agent_id)

    alice = search_situated_memories(
        database, SituatedMemoryQuery("alice", text="restructuring")
    )
    bob = search_situated_memories(
        database, SituatedMemoryQuery("bob", text="restructuring")
    )
    dana = search_situated_memories(
        database, SituatedMemoryQuery("dana", text="restructuring")
    )
    assert inspection.event_id in {hit.memory.event_id for hit in alice}
    assert inspection.event_id not in {hit.memory.event_id for hit in bob}
    assert len(bob) == 1  # Bob remembers Alice's audible telling.
    assert dana == ()     # Dana observed neither event.
```

### Memory-augmented situated cognition V13

V13 closes that loop without introducing a shared chat context. Each embodied agent
restarts from the same physical-story checkpoint, searches only its own SQLite/FTS5
records, and admits an eligible memory once before choosing its next POMDP action.
This executable comparison uses the repository's deterministic office fixture:

```python
from tempfile import TemporaryDirectory

from narrative_dynamics.abm import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
    advance_situated_story,
    initialize_situated_memory_cognition,
    initialize_situated_story,
    initialize_situated_world,
    simulate_situated_memory_cognitive_round,
    standard_situated_memory_policy,
)
from tests.situated_cognition_fixtures import cognitive_office_model

cognition = cognitive_office_model()
history = initialize_situated_story(
    cognition.world_model, initialize_situated_world(cognition.world_model)
)
history = advance_situated_story(cognition.world_model, history, (
    SituatedActionIntent(
        "historical-inspect", "alice", SituatedActionKind.INSPECT, "memo"
    ),
))
inspection = next(
    event for event in history.rounds[-1].events
    if event.actor_agent_id == "alice"
)
history = advance_situated_story(cognition.world_model, history, (
    SituatedActionIntent(
        "historical-move", "alice", SituatedActionKind.MOVE, "records-open"
    ),
))
cue = SituatedMemoryRecallCue(
    "restructuring",
    "restructuring",
    required_place_ids=("open",),
    event_kinds=(SituatedActionKind.INSPECT,),
    channels=(ObservationChannel.INSPECTION,),
)


def memory_model(enable_recall):
    return SituatedMemoryCognitiveModel(
        "office-memory-cognition",
        "1",
        cognition,
        standard_situated_memory_policy(),
        tuple(
            SituatedAgentRecallPolicy(
                agent.agent_id,
                (cue,) if enable_recall and agent.agent_id == "alice" else (),
            )
            for agent in cognition.agents
        ),
    )


control_model = memory_model(False)
recall_model = memory_model(True)
control_state = initialize_situated_memory_cognition(control_model, history)
recall_state = initialize_situated_memory_cognition(recall_model, history)

with TemporaryDirectory() as temporary:
    control = simulate_situated_memory_cognitive_round(
        f"{temporary}/control.sqlite3",
        control_model,
        history,
        control_state,
    )
    recalled = simulate_situated_memory_cognitive_round(
        f"{temporary}/recall.sqlite3",
        recall_model,
        history,
        recall_state,
    )

control_alice = next(item for item in control.decisions if item.agent_id == "alice")
recalled_alice = next(item for item in recalled.decisions if item.agent_id == "alice")
assert control_alice.posterior_belief.probabilities["approved"] == 0.5
assert control_alice.selected_action_id == "wait"
assert recalled_alice.posterior_belief.probabilities["approved"] == 0.9
assert recalled_alice.selected_action_id == "tell"
assert recalled_alice.intent.source_event_ids == (inspection.event_id,)
assert len(recalled_alice.recalled_memory_ids) == 1
```

The checkpoint prevents old story observations from being magically replayed. Exact
agent/world/round filters prevent privacy and future leaks, and recalled IDs prevent
repeated Bayesian reinforcement.

### Social memory revision V14

V14 gives recalled testimony a social lifecycle. Repetition consolidates into one
claim, a source's opposite claim supersedes but does not erase its history, later
private evidence confirms or contradicts active claims, and only the observer's
directed trust/affinity toward that source changes.

```python
from narrative_dynamics.abm import (
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    advance_situated_social_memory,
    initialize_situated_social_memory,
)
from tests.test_network_abm_situated_social_memory import cognitive_checkpoint
from tests.test_network_abm_situated_social_memory_contracts import social_model

model = social_model()
cognition = cognitive_checkpoint(model, 3)
social = initialize_situated_social_memory(model, cognition)
evidence = (
    SituatedSocialEvidence(
        "heard-1", SituatedSocialEvidenceKind.TESTIMONY,
        "bob", "restructuring", "approved", 1, "tell-1",
        source_agent_id="alice", memory_id="heard-1",
    ),
    SituatedSocialEvidence(
        "heard-2", SituatedSocialEvidenceKind.TESTIMONY,
        "bob", "restructuring", "approved", 2, "tell-2",
        source_agent_id="alice", memory_id="heard-2",
    ),
    SituatedSocialEvidence(
        "inspected", SituatedSocialEvidenceKind.VERIFICATION,
        "bob", "restructuring", "approved", 3, "inspect-1",
    ),
)
updated = advance_situated_social_memory(
    model, cognition, social, cognition, evidence
).next_state
claim = updated.claims[0]
bob_to_alice = next(
    item for item in updated.relationships
    if item.observer_agent_id == "bob" and item.source_agent_id == "alice"
)
assert claim.support_count == 2
assert claim.status.value == "confirmed"
assert bob_to_alice.trust == 0.6
assert bob_to_alice.affinity == 0.1
```

The symbol vocabulary and contradiction groups are declared rather than inferred
from arbitrary prose. Unresolved claims expire by deterministic age/capacity rules;
forgetting changes only the V14 index and never deletes SQLite history. Learned trust
tempers later recalled external testimony, while direct inspection keeps its full
V13 evidence weight.

### Percept-driven cognition, memory, and social evidence V15.1

V15.1 makes each agent's sanitized percept the evidence boundary for cognition and
long-term memory. The same objective office event can therefore produce different
private histories: an open door gives Bob exact testimony, while a closed door gives
him only an unidentified sound. SQLite stores only what Bob actually perceived.

```python
from tempfile import TemporaryDirectory

from narrative_dynamics.abm import (
    ObservationChannel,
    SituatedActionKind,
    SituatedMemoryRecallCue,
    SituatedPerceptMemoryQuery,
    initialize_situated_percept_memory_cognition,
    initialize_situated_social_memory,
    search_situated_percept_memories,
    simulate_situated_percept_social_cognitive_round,
)
from tests.test_network_abm_situated_percept_cognition import SECRET
from tests.test_network_abm_situated_percept_memory_cognition import (
    mind,
    recall_model,
)
from tests.test_network_abm_situated_percept_social_cognition import (
    bound_social_model,
    tell_checkpoint,
)


def run_branch(database, door_open):
    perception, cognition, story = tell_checkpoint(door_open=door_open)
    cue = SituatedMemoryRecallCue(
        "heard-tell",
        SECRET if door_open else "detected",
        event_kinds=(SituatedActionKind.TELL,) if door_open else (),
        channels=(ObservationChannel.AUDITORY,),
    )
    memory_model = recall_model(
        {"bob": (cue,)}, cognition=cognition, perception=perception
    )
    social_model = bound_social_model(memory_model)
    cognitive_state = initialize_situated_percept_memory_cognition(
        memory_model, story
    )
    social_state = initialize_situated_social_memory(
        social_model, cognitive_state
    )
    result = simulate_situated_percept_social_cognitive_round(
        database,
        memory_model,
        social_model,
        story,
        cognitive_state,
        social_state,
    )
    secret_hits = search_situated_percept_memories(
        database,
        SituatedPerceptMemoryQuery("bob", text=SECRET),
    )
    bob = mind(result.next_cognitive_state, "bob")
    objective_tell = story.rounds[0].events[0]
    return objective_tell, bob, secret_hits, result.next_social_state


with TemporaryDirectory() as temporary:
    exact = run_branch(f"{temporary}/open.sqlite3", True)
    detected = run_branch(f"{temporary}/closed.sqlite3", False)

assert exact[0] == detected[0]  # One objective TELL under the same world rules.
assert exact[1].belief.probabilities["approved"] > 0.5
assert detected[1].belief.probabilities["approved"] == 0.5
assert len(exact[2]) == 1
assert detected[2] == ()       # The closed-door row contains no secret text.
assert len(exact[3].claims) == 1
assert detected[3].claims == ()
```

This percept/cognition path remains deterministic. V16 adds a separate provider
boundary for interpreting natural language without weakening that evidence model.

### Private RAG and grounded language V16

V16 treats an LLM (or any other structured-language provider) as a semantic
compiler, never as a world transition. A provider may propose bounded FTS5 query
expansions and candidate claims, but deterministic runtime code chooses the private
memory rows, validates every ID and enum against a finite vocabulary, and creates
the accepted content-addressed artifact.

```python
from narrative_dynamics.abm import (
    SituatedGroundingProviderIdentity,
    SituatedGroundingRequest,
    build_situated_grounding_prompt,
    compile_situated_semantic_grounding,
    grounded_claims_to_situated_social_evidence,
    replay_situated_semantic_grounding,
)


class JsonLanguageProvider:
    identity = SituatedGroundingProviderIdentity(
        "my-provider", "1", "my-structured-language-model"
    )

    def __init__(self, complete_json):
        self._complete_json = complete_json

    def complete_json(self, *, task, payload):
        # The injected callable may use a hosted LLM, a local model, or rules.
        return self._complete_json(task=task, payload=payload)


provider = JsonLanguageProvider(application_json_completion)
request = SituatedGroundingRequest(
    "interpret-bob-7",
    "bob",
    bob_exact_tell_memory_id,
    "Alice 对重组决定到底是什么意思？",
    maximum_memories=6,
)
prompt = build_situated_grounding_prompt(
    "office.sqlite3",
    grounding_model,
    request,
    retrieval_planner=provider,
)
artifact = compile_situated_semantic_grounding(prompt, provider)
social_evidence = grounded_claims_to_situated_social_evidence(
    grounding_model, artifact
)
assert replay_situated_semantic_grounding(grounding_model, artifact) is artifact
```

In the office case, Bob's prompt can contain only Bob's sanitized SQLite rows. An
exact TELL memory may support a grounded testimony claim; an exact private INSPECT
memory may support verification. A closed-door `detected` sound has no speaker,
message, outcome, or details and therefore cannot support a status/message
predicate that declares `exact` as its minimum fidelity. It may still support a
separate declared low-fidelity predicate such as “the observer detected a sound”;
that qualified result remains in the grounding layer and does not enter V14.

Every accepted claim must cite the request's primary memory. The artifact retains
the provider identity captured before invocation, schema hash, prompt-template
hash, private-context hash, raw response hash, validation result, and optional
retrieval-planner provenance. V14 bridging accepts only present, affirmed,
asserted claims whose declared social subject/topic matches an exact TELL or
INSPECT. Repeated equivalent interpretations of one event deduplicate to one
stable evidence ID; conflicting symbols for the same event/topic fail closed.

The repository intentionally bundles no vendor SDK or network client. Applications
inject a provider through `complete_json`; accepted artifacts replay with no model
call. FTS5 retrieval works without an LLM, while optional provider query expansion
helps with paraphrases and synonyms. V16 does not yet add embeddings/vector search,
open-world predicates, free-form action creation, or direct model mutation of the
world or agent state.

### V17 deterministic narrative projection

V17 turns accepted world, cognitive, and social history into deterministic beats,
scenes, and a cut. It supports objective authority, one-agent limited POV, and a
declared multi-POV cut. Limited POV uses only sanitized percepts and private
cognitive, memory, claim, and relationship state owned by that POV agent; objective
projection may carry those private records for multiple owners, but keeps every
record private and owner-tagged.

Chronological projection preserves accepted event time. Authored order can present
flashbacks, while each beat retains its original round and sequence instead of
rewriting history to place causes beside effects. A cut's scene IDs and each
scene's beat IDs define the one canonical presentation order; the top-level scene
and beat tuples follow that same order. An undisclosed percept location is `None`
and starts its own scene, while an actual world place whose ID is `undisclosed`
retains normal known-place grouping. The result is a content-addressed
truth-and-entitlement packet for V18: it is not generated prose, and it does not
mutate the world or any agent state.

## Verification

GitHub Actions runs:

```text
lake build
all Lean theorem tests
python3 -m unittest discover -s tests -v
```

The formal and simulation layers follow a RED → GREEN workflow. See `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md` for the current design and modeling limitations.
