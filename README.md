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

## Verification

GitHub Actions runs:

```text
lake build
all Lean theorem tests
python3 -m unittest discover -s tests -v
```

The formal and simulation layers follow a RED → GREEN workflow. See `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md` for the current design and modeling limitations.
