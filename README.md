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

### V18 entitlement-bound narrative realization

V18 turns a V17 projection into replayable prose or screenplay passages without
changing the projected truth. The provider sees one scene-local packet per call:
the scene's canonical beats, only their exact entitlement closure, the selected
`prose` or `screenplay` policy, and bounded response instructions. It receives no
automatic cross-scene transcript or hidden global context.

Any application can supply the small provider protocol directly:

```python
from narrative_dynamics.abm import (
    NarrativeRealizationFormat,
    NarrativeRealizationPolicy,
    NarrativeRealizationProviderIdentity,
    NarrativeRealizationRequest,
    build_narrative_realization_prompt,
    compile_narrative_realization,
    replay_narrative_realization,
)


class JsonNarrativeProvider:
    identity = NarrativeRealizationProviderIdentity(
        "my-provider", "1", "my-json-model"
    )

    def complete_json(self, *, task, payload):
        return application_json_completion(task=task, payload=payload)


provider = JsonNarrativeProvider()
realization_policy = NarrativeRealizationPolicy(
    "screenplay-en",
    "1",
    NarrativeRealizationFormat.SCREENPLAY,  # PROSE is also supported.
    "en",
    tone_tags=("restrained",),
)
realization_request = NarrativeRealizationRequest(
    "render-42", projection.content_hash
)
prompt = build_narrative_realization_prompt(
    projection, realization_policy, realization_request, provider
)
artifact = compile_narrative_realization(prompt, provider)
assert artifact.assurance.value == "citation_bound"
assert replay_narrative_realization(projection, artifact) is artifact
```

`citation_bound` means deterministic validation proved scene and beat coverage,
ordering, size limits, provider identity, and exact entitlement citations. It does
not claim that arbitrary natural-language prose is formally entailed. Provider
wording remains presentation-only and cannot become V17 evidence or mutate the
simulation. In both assurance modes, `accepted` means structurally accepted
presentation, not independently authoritative truth. Any authoritative consumer
must call `replay_narrative_realization` with the exact V17 projection first.

For literal output with stronger, honest assurance, use the provider-free
fallback. It emits one passage per beat containing only sorted entitlement
`<JSON string>=<JSON string>` facts; beat IDs and beat kinds remain passage
metadata. The canonical text `[]` denotes an entitlement with zero facts and does
not introduce a fact.

```python
from narrative_dynamics.abm import realize_narrative_exact_facts

exact_artifact = realize_narrative_exact_facts(
    projection, realization_policy, realization_request
)
assert exact_artifact.assurance.value == "exact_facts"
assert replay_narrative_realization(projection, exact_artifact) is exact_artifact
```

An optional OpenAI adapter is isolated in the integrations package:

```python
from narrative_dynamics.integrations import OpenAINarrativeProvider

openai_provider = OpenAINarrativeProvider.from_env(".env")
openai_prompt = build_narrative_realization_prompt(
    projection, realization_policy, realization_request, openai_provider
)
openai_artifact = compile_narrative_realization(openai_prompt, openai_provider)
```

The dotenv path, `OPENAI_API_KEY`, and optional `OPENAI_BASE_URL` are
construction-only configuration: they never enter provider identity, prompts,
artifacts, hashes, representations, or errors. The adapter reads `OPENAI_MODEL`
and optional `OPENAI_PROVIDER` for its public identity and imports the OpenAI and
dotenv packages only when constructing the optional integration. Adapter identity
version `2` authenticates its effective message semantics: a fixed instruction to
return exactly one JSON object matching the supplied response schema, followed by
the exact scene task/payload JSON. Importing `narrative_dynamics.abm` remains
provider-neutral.

### V19 atomic situated-network runtime

V19 unifies observation and orchestration for the existing situated office model.
One atomic call advances the V15.1 percept/social/cognitive round exactly once, then
binds its returned story, cognitive state, and social state to one privacy-preserving
multiplex-network snapshot and one emergence-metric record. Snapshot transmissions
identify events, agents, fidelity, and disclosed channels only when that observer's
sanitized V15 percept explicitly discloses `kind=TELL` and a distinct non-null actor.
They never copy private message payloads or recover identity from an anonymous sound.

```python
from dataclasses import replace
from tempfile import TemporaryDirectory

from narrative_dynamics.abm import (
    EmbodiedAgentSpec,
    PassageSpec,
    PassageState,
    PlaceSpec,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedAgentPerceptionProfile,
    SituatedAgentRecallPolicy,
    SituatedClaimTopic,
    SituatedCognitiveModel,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedGoalReward,
    SituatedGoalSpec,
    SituatedHypothesis,
    SituatedMemoryCognitiveModel,
    SituatedNetworkRuntimeModel,
    SituatedObservationLikelihood,
    SituatedObservationSymbol,
    SituatedPerceptFidelity,
    SituatedPerceptMemoryCognitiveModel,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryPolicy,
    SituatedWorldModel,
    advance_situated_story,
    hash_situated_percept_memory_store,
    initialize_situated_network_runtime,
    initialize_situated_percept_memory_cognition,
    initialize_situated_social_memory,
    initialize_situated_story,
    initialize_situated_world,
    project_situated_network_snapshot,
    project_situated_percepts,
    simulate_situated_network_round,
    standard_situated_memory_policy,
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState


office = SituatedWorldModel(
    "v19-office",
    "1",
    (PlaceSpec("corridor", "Corridor"), PlaceSpec("meeting", "Meeting room")),
    (PassageSpec("door", "corridor", "meeting", initially_open=False),),
    (
        EmbodiedAgentSpec("alice", "analyst", "corridor"),
        EmbodiedAgentSpec("bob", "manager", "meeting"),
    ),
)
perception = SituatedPerceptionModel(
    "v19-office-perception",
    "1",
    office,
    (
        SituatedPerceptionEdge(
            "audio-open", SituatedPerceptionLayer.AUDITORY,
            "corridor", "meeting", 5.0,
            SituatedEdgeActivation.PASSAGE_OPEN, "door",
        ),
        SituatedPerceptionEdge(
            "audio-closed", SituatedPerceptionLayer.AUDITORY,
            "corridor", "meeting", 25.0,
            SituatedEdgeActivation.PASSAGE_CLOSED, "door",
        ),
    ),
    (
        SituatedAgentPerceptionProfile("alice", 1.0, 20.0, 45.0),
        SituatedAgentPerceptionProfile("bob", 1.0, 20.0, 45.0),
    ),
    (SituatedEventSignalProfile(SituatedActionKind.TELL, False, 60.0),),
)


def cognitive_agent(agent_id):
    hypotheses = (
        SituatedHypothesis("approved", "The proposal is approved"),
        SituatedHypothesis("denied", "The proposal is denied"),
    )
    symbols = (
        SituatedObservationSymbol("approved", "Approval evidence"),
        SituatedObservationSymbol("denied", "Denial evidence"),
    )
    return SituatedAgentCognitiveModel(
        agent_id,
        hypotheses,
        PlanningBeliefState({"approved": 0.5, "denied": 0.5}),
        symbols,
        (),
        tuple(
            SituatedObservationLikelihood(
                "wait", hypothesis.hypothesis_id, symbol.symbol_id, 0.5
            )
            for hypothesis in hypotheses
            for symbol in symbols
        ),
        (SituatedActionSpec("wait", SituatedActionKind.WAIT),),
        (("wait",),),
        (),
        (SituatedGoalSpec("idle", "Wait deterministically", 1.0),),
        tuple(
            SituatedGoalReward("idle", hypothesis.hypothesis_id, "wait", 0.0)
            for hypothesis in hypotheses
        ),
        0.0,
        1.0,
    )


cognition = SituatedCognitiveModel(
    "v19-office-cognition",
    "1",
    office,
    tuple(cognitive_agent(agent.agent_id) for agent in office.agents),
)
recall_policies = tuple(
    SituatedAgentRecallPolicy(agent.agent_id) for agent in office.agents
)
percept_memory = SituatedPerceptMemoryCognitiveModel(
    "v19-office-percept-memory",
    "1",
    perception,
    cognition,
    standard_situated_percept_memory_policy(),
    recall_policies,
)
social_memory = SituatedSocialMemoryModel(
    "v19-office-social",
    "1",
    SituatedMemoryCognitiveModel(
        "v19-office-memory",
        "1",
        cognition,
        standard_situated_memory_policy(),
        recall_policies,
    ),
    (SituatedClaimTopic("decision", ("approved", "denied")),),
    SituatedSocialMemoryPolicy(),
)
network = SituatedNetworkRuntimeModel(
    "v19-office-network",
    "1",
    percept_memory,
    social_memory,
    "approved",
    0.7,
    0.5,
)


def tell_checkpoint(door_open):
    world_state = initialize_situated_world(office)
    if door_open:
        world_state = replace(
            world_state,
            passages=(PassageState("door", True),),
        )
    story = initialize_situated_story(
        office, world_state, perception_model=perception
    )
    story = advance_situated_story(
        office,
        story,
        (SituatedActionIntent(
            "alice-tell",
            "alice",
            SituatedActionKind.TELL,
            message="The proposal is approved.",
        ),),
    )
    cognitive_state = initialize_situated_percept_memory_cognition(
        percept_memory, story
    )
    social_state = initialize_situated_social_memory(
        social_memory, cognitive_state
    )
    return story, cognitive_state, social_state


closed_story, closed_cognition, closed_social = tell_checkpoint(False)
closed_percepts = project_situated_percepts(
    perception, closed_story.rounds[-1]
)
bob_closed = next(
    item for item in closed_percepts.percepts if item.agent_id == "bob"
)
closed_snapshot = project_situated_network_snapshot(
    network, closed_story, closed_cognition, closed_social
)
assert bob_closed.fidelity is SituatedPerceptFidelity.DETECTED
assert bob_closed.actor_agent_id is None and bob_closed.kind is None
assert closed_snapshot.transmissions == ()
assert closed_snapshot.latest_tell_event_count == 1

open_story, open_cognition, open_social = tell_checkpoint(True)
open_snapshot = project_situated_network_snapshot(
    network, open_story, open_cognition, open_social
)
assert len(open_snapshot.transmissions) == 1
assert open_snapshot.transmissions[0].source_agent_id == "alice"
assert open_snapshot.transmissions[0].fidelity is SituatedPerceptFidelity.EXACT

with TemporaryDirectory() as temporary:
    database = f"{temporary}/office.sqlite3"
    runtime = initialize_situated_network_runtime(
        database, network, open_story, open_cognition, open_social
    )
    round_result = simulate_situated_network_round(database, network, runtime)
    assert round_result.next_state.parent_state_hash == runtime.content_hash
    assert (
        round_result.next_state.memory_store_hash
        == hash_situated_percept_memory_store(database)
    )
```

In the office case, each runtime state presents agent nodes alongside directed social
relationship edges, situated visual/auditory access edges, sanitized latest-round
transmissions, and aggregate adoption, trust, claim, and reach measures. A successful
objective TELL still increments the payload-free `latest_tell_event_count` when a
closed door leaves Bob with only an anonymous DETECTED percept, but that percept does
not increment any V19 source-labelled transmission bucket. The exact story prefix,
cognitive/social parent hashes, logical SQLite memory hash, and outer V19 parent make
the resulting trajectory replay-auditable without merging these projections back
into older network models.

V19 hashes canonical logical memory rows and metadata, including activation state;
database paths, row IDs, FTS/index state, and raw SQLite bytes stay outside the
contract. A round runs V15.1 against a staged database and publishes it only after
the complete result validates, so Python transition/validation errors leave the
original logical store unchanged. SQLite publication and caller-managed durable
persistence of the returned state are not one cross-resource transaction: a process
crash after publication but before the caller persists that state leaves the prior
state stale, requiring recovery from a matching subsystem/store checkpoint. The
runtime therefore requires a file-backed database.

V19 unifies observability and orchestration, but it does not yet implement physical
lifecycle or V1-V9 rewiring feedback; both remain future work. Physical map geometry
and Blender visualization remain presentation/integration layers rather than part of
the atomic runtime, and a production authoring UI remains future work.

### V20 Blender graybox replay export

V20 turns a completed situated-network trajectory into one editable Blender file.
The `.blend` contains the spatial map, labeled graybox places and passages, independent
stick-figure agents, movement and door keyframes, objective event timeline markers,
belief/claim custom properties, an overview camera, and replay provenance hashes.

```python
from narrative_dynamics.abm import load_tiled_situated_spatial_map
from narrative_dynamics.integrations import export_situated_network_blend

# `network` and `trajectory` are the V19 model and completed trajectory above.
world = network.percept_memory_model.cognitive_model.world_model
spatial_map = load_tiled_situated_spatial_map(
    "office.tmj",
    world,
    meters_per_pixel=0.05,
)
report = export_situated_network_blend(
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    "office-replay.blend",
    network,
    trajectory,
    spatial_map,
)
print(report.replay_hash)
```

The Tiled map must be orthogonal JSON. Plain rectangle objects classified as `place`
or `passage` use their object `name` as the matching world place/passage ID; the map
must cover every world place and passage exactly once. Group and object-layer pixel
offsets are inherited. Omitting `spatial_map` selects a deterministic grid layout, so
physical coordinates are optional. The exporter runs
Blender headlessly, stages the result, validates its `.blend` header, and atomically
replaces the requested output only after success. It exports no message payloads,
private details, database paths, renderer output, video, player, or Blender add-on.

### V21.1 data-authored situated scenarios

V21.1 adds a strict JSON input boundary over the existing V20 runtime. The complete
data-only law-firm package in `examples/law_firm_scenario` can be loaded, compiled,
and initialized directly; it does not import or execute a Python generator:

```python
from narrative_dynamics.abm import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
    load_situated_scenario_package,
)

source = load_situated_scenario_package("examples/law_firm_scenario")
scenario = compile_situated_scenario_package(source)
state = initialize_compiled_scenario("law-firm-memory.sqlite3", scenario)
```

The package directory and manifest form a tree of ownership: `scenario.json`
declares each document's role, relative `path`, and raw-byte `sha256`; `run.json`
contains run policy, agent logical IDs come from their agent documents, and singleton
roles are their logical identities. The document contents form graphs. Places connect
through directed passages and perception edges; institutions use an acyclic parent
forest; social relationships are directed; story scenes use an acyclic dependency
graph; and grants connect agents, roles, institutions, or the public to catalog
resources. Cross-document references always use stable authored IDs, never array
positions, filesystem paths, Python names, or object identity. Authored act order,
scene order within an act, and action-schedule horizons retain sequence semantics;
unordered catalogs and graph members are canonicalized for compiled identity.

Physical input defines the world roster, perception graph, authoritative round-zero
agent/object/passage state, and either a real Tiled `map.tmj` or the declared
`physical.map=auto_grid` fallback, both compiled by the public V20 spatial authority.
Social input separately defines institutions, memberships, multiplex directed typed
relationships, one unambiguous trust/affinity runtime seed per directed agent pair,
and supported enforceable norms.
The compiler requires exact fixed-roster and cross-document coverage and constructs
the existing public V10-V20 world, cognition, perception, memory, social-memory,
network-runtime, story, and spatial-map values. The compiled scenario binds the
authoritative initial story, cognitive checkpoint, and social checkpoint; opening
the SQLite percept-memory store is deferred until initialization.

Story mode is `authored`, `hybrid`, or `sandbox`, and must match the run policy.
Authored mode assigns every round to an active scene contract; hybrid mode keeps
agents autonomous while permitting only declared typed interventions; sandbox mode
supplies initial conditions and terminal limits without steering actions. These
modes are input policy only in V21.1. Knowledge and asset catalogs carry content
hashes, metadata, typed per-resource entitlements, and access grants. Every catalog
or direct agent grant must be covered by an agent, role, institution, or public
entitlement. V21.1 never fetches their URIs, builds a retrieval index, calls an LLM,
or loads an asset. Catalog metadata is therefore an authorization and identity
boundary, not an execution hook. The committed acceptance package has three legal
professionals, one client, and a private contract-scan resource.

Loading is deliberately inert: it reads only UTF-8 JSON documents named by the
manifest, enforces size limits and exact schemas, recomputes every declared content
hash, and rejects absolute paths, parent traversal, symlink escape, duplicate paths
or file identities, duplicate keys, non-finite numbers, and undeclared source
dependencies. Public raw manifest/document hashes remain provenance evidence but are
excluded from equality and semantic identity. Compilation uses schema-semantic package/document
identity, so moving the package or reordering semantically unordered input does not
change the compiled hash. Compilation fails closed with `ScenarioCompilationError`;
its stable document role (and logical ID for agent documents), RFC 6901 JSON pointer,
and code identify malformed or unresolved input without exposing machine-local paths
or source values. The planned deterministic `ScenarioCompilationReport` is explicitly
deferred beyond V21.1; this release exposes only the sanitized exception boundary.

V21.1 stops after safe load, semantic compilation, and exact runtime-state
initialization. It does not advance the scenario through a new V21 coordinator and
does not publish output records, an output bus, live transport, or live Blender
projection. Those are the separate V21.2 phase; LLM/retrieval and production asset
adapters remain later V21 phases. A run policy may reserve output kinds or a Blender
mode for those future consumers, but V21.1 does not silently stub or execute them.

### Lean social-mesh feasibility V23.0

V23.0 begins with a formal graph-theoretic foundation rather than a networking or
NetworkX prototype. `NarrativeDynamics.Core.SocialMesh` defines exact-length walks
and bounded reachability over a directed mesh snapshot. Lean proves that every path
inside a local-society projection lifts to the global mesh; if the society is closed
under outgoing edges, every global path starting inside remains in that projection
and no bounded path reaches an outsider.

One explicit bridge composes bounded paths on both sides. In particular, a path of
at most two hops to the bridge, the bridge edge itself, and a path of at most three
hops from it imply reachability within six hops. This is a conditional six-degree
theorem, not a claim that every empirical society has diameter six. A separate
placement structure makes authoritative physical residence a function while social
membership remains a relation, formally permitting one Agent to belong to multiple
societies without occupying multiple physical worlds.

This proof layer does not yet formalize Watts--Strogatz clustering or expected path
length, alternate attachment-model preferential attachment or power-law asymptotics, temporal
mesh transitions, automatic community formation, Python execution, NetworkX, P2P,
WSS, or Raft. Those models can be added only after their assumptions and required
invariants are stated explicitly.

### Lean deterministic small-world certificate V23.1

V23.1 turns the initial bridge result into a composable two-level network theorem.
`NarrativeDynamics.Core.SmallWorld` maps an exact walk in the society graph to an
exact Agent-level walk between designated society gateways. Agent entry, society
travel, and Agent exit then compose with an additive hop budget. Consequently, if
every Agent can reach and be reached from its assigned society gateway within one
hop, and every pair of societies is connected within four hops, every ordered Agent
pair is connected within six hops: `1 + 4 + 1 = 6`.

The module also defines a deterministic `SmallWorldCertificate`: symmetric edges,
a uniform global hop bound, and perfect local clustering, meaning that every pair
of distinct neighbors of a center is directly connected. The certificate is
stronger than merely having a high empirical clustering coefficient. Adding
shortcuts provably preserves its existing global hop bound, but the proof does not
claim arbitrary new edges preserve perfect clustering because they may introduce
new open wedges.

All conclusions remain conditional on explicit graph witnesses. V23.1 does not
claim that real societies satisfy six degrees, calculate a numerical clustering
coefficient or average shortest path, sample a Watts--Strogatz distribution, or
prove an alternate attachment-model preferential-attachment or power-law asymptotic theorem. Those
finite-metric and probabilistic results remain separate proof phases.

### Lean finite small-world metrics V23.2

V23.2 supplies exact finite-network measurements for the V23.1 certificate.
`NarrativeDynamics.Core.SmallWorldMetrics` enumerates each node's outgoing
neighbors, the distinct ordered neighbor pairs, and the subset closed by an edge.
Their rational cardinality ratio is the local clustering coefficient. A node with
fewer than two distinct neighbors has denominator zero and coefficient zero;
perfect local clustering with a nonempty candidate-pair set is proved to have
coefficient one. For symmetric networks, ordered-pair duplication changes both
counts equally and leaves the ratio unchanged.

The same module defines exact shortest hop count and mesh diameter as the least
natural numbers satisfying the existing `ReachWithin` and `GlobalHopBound`
predicates. The specification theorems prove those minima are reachable, while the
minimality theorems rule out any smaller witnessed bound. On finite node types,
average shortest-path length is the rational mean over all ordered distinct node
pairs, with empty and singleton types evaluating to zero.

A `SmallWorldCertificate g limit` now proves both that the exact mesh diameter is
at most `limit` and that the average shortest-path length is at most `limit`. These
are deterministic consequences of supplied graph witnesses, not empirical
estimates. V23.2 still does not sample a Watts--Strogatz network, prove an expected
clustering/path-length law, construct an alternate attachment process, or establish a
power-law degree distribution.

### Lean Watts--Strogatz foundation V23.3

V23.3 constructs the deterministic initial graph needed by a later stochastic
Watts--Strogatz model. `NarrativeDynamics.Core.WattsStrogatz` places `Fin n` nodes
on a modular ring and connects distinct nodes whose clockwise or counterclockwise
difference is within a configured radius. The relation is decidable, symmetric,
and loopless. Valid parameter packages require a positive radius and
`2 * radius < nodeCount`. This still allows the complete ring at
`nodeCount = 2 * radius + 1`.

Mathlib's finite `cycleGraph` is formally embedded into every positive-radius
ring. A converter preserves exact walk length from `SimpleGraph.Walk` to the
project's `MeshWalk`; finite simple-path length then proves the regular ring has a
global hop bound of `nodeCount - 1`. The existing executable metric definitions
also check a concrete baseline: the six-node radius-two ring has local clustering
coefficient `2 / 3` at node zero.

An explicit undirected-shortcut operation retains every old edge and adds both
orientations of one new connection. Distinct shortcut endpoints preserve
looplessness, and a symmetric base remains symmetric. More importantly, Lean now
proves that retaining edges cannot increase any pair's exact shortest hop count,
the exact mesh diameter, or finite average shortest-path length. No analogous
global clustering monotonicity is asserted because a shortcut may create a new
open wedge even while shortening paths.

V23.3 is still not the random WS model: it defines no probability space, rewiring
sampler, expectation, concentration bound, or asymptotic small-world theorem. It
also makes no alternate attachment-model or power-law claim.

### Lean finite fitness attachment and replay V23.5

V23.5 formalizes a finite Bianconi–Barabási fitness-attachment model in
`NarrativeDynamics.Core.FitnessAttachment`, `FitnessBirth`, `FitnessValidation`,
and `FitnessReplay`. States contain connected simple graphs with at least two
vertices and immutable, strictly positive rational fitness. Degrees are computed
from actual adjacency. For a fixed positive attachment count bounded by the seed
size, each birth selects an ordered list of distinct existing targets. All choices
within one birth use the same old graph and degrees. Each choice masks the selected targets and renormalizes the remaining fitness-times-degree
weights. The newborn becomes eligible in subsequent births.

Lean proves normalization, support, ordered and unordered target-mass laws, and
preservation of connectivity through actual graph updates. Each birth adds one
vertex and the specified number of undirected edges while preserving old edges
and stored fitness. Common fitness gives the finite degree-weighted
constant-fitness BB law with selection without replacement. Multiplying all seed and
newborn fitness values by one positive factor preserves the topology and trace
probabilities.

The raw API validates seed data and every supplied birth before applying it.
Finite replay returns the actual final state and exact rational probability of
the supplied ordered trace; it draws no random numbers. A failed birth reports
its first zero-based index and cause without returning partial state or
probability. Successful replay preserves stable vertex IDs, extends the complete
fitness list, and has positive trace probability. The finite continuation law
sums to one for a fixed schedule of newborn fitness values.

`NarrativeDynamics.Tests.FitnessScope` checks an eight-vertex path obtained through
six actual checked births from a two-vertex edge, with unit fitness and one target
per birth. The supplied target sequence has probability `1 / 46080`. Its actual
graph has seven edges and degrees `[1, 2, 2, 2, 2, 2, 2, 1]`; Lean proves endpoint
shortest distance seven, absence of a six-hop route, and diameter seven. The lower
bound follows by induction on every possible walk, using the fact that each edge
changes the vertex label by one.

This positive-probability example rules out an unconditional six-hop guarantee
for the model. Power-law statistics, typical distances, and asymptotic
small-world behavior require separate probabilistic analysis; further six-hop
results require additional assumptions and proofs.

### V21.2 typed simulation output and public journal

V21.2 projects an accepted situated-network round into one deterministic typed batch.
Every record retains the exact stream, scenario, round, state, sequence, payload, and
source-artifact identities. Audience capabilities filter typed views before any
serialization; retained records keep their global source sequences (including gaps)
and the source batch hash.

```python
from narrative_dynamics.abm import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
    filter_simulation_output,
    project_simulation_output,
    replay_public_simulation_journal,
    write_public_simulation_journal,
)

batch = project_simulation_output(
    scenario,
    round_result,
    stream_id="law-firm-run",
)
public_view = filter_simulation_output(
    batch,
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
)
journal = write_public_simulation_journal("law-firm.jsonl", batch)
assert replay_public_simulation_journal("law-firm.jsonl") == journal
```

The JSONL journal contains public records only. It strictly reconstructs typed
payloads and verifies payload, record, view, and journal hashes during replay. Each
append stages the complete replacement beside the destination, flushes and fsyncs it,
then publishes it with an atomic replace, so a failed publication preserves the prior
journal. Replay is local and deterministic: it performs no Agent or provider call.

V21.2 itself does not import or fabricate a bus, coordinator, or commands; those are
the V21.3 layer below. JSON-RPC/H2/WSS, the Web editor, live Blender, and remote
Agents remain later phases, and an output policy containing only unsupported kinds
fails closed at the V21.2 projection boundary.

### V21.3 synchronous scenario coordinator and exact forks

V21.3 turns one compiled scenario into a controllable local run. The coordinator is
the sole writer for its SQLite memory store, advances exactly one existing V19 round
per `step`, keeps capability-scoped output and command audit history, and can create
an exact checkpoint whose state and SQLite snapshot seed an independent paused run.
This complete law-firm outline uses only the public API:

```python
from narrative_dynamics.abm import (
    InMemoryScenarioStateStore,
    LocalScenarioCheckpointStore,
    ScenarioCommandCapability,
    ScenarioCommandKind,
    ScenarioCommandRequest,
    ScenarioCoordinator,
    ScenarioForkRequest,
    SimulationAudienceCapability,
    SimulationOutputAudience,
    SimulationOutputBus,
    SimulationOutputKind,
    compile_situated_scenario_package,
    load_situated_scenario_package,
)

scenario = compile_situated_scenario_package(
    load_situated_scenario_package("examples/law_firm_scenario")
)
bus = SimulationOutputBus()
bus.subscribe(
    "public-preview",
    tuple(SimulationOutputKind),
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
    lambda view: print(view.next_state_hash, len(view.records)),
)
checkpoints = LocalScenarioCheckpointStore("law-firm-checkpoints")
coordinator = ScenarioCoordinator.create(
    "law-firm-memory.sqlite3",
    scenario,
    run_id="law-firm-run",
    stream_id="law-firm-stream",
    state_store=InMemoryScenarioStateStore(),
    publisher=bus,
    checkpoint_store=checkpoints,
)
operator = ScenarioCommandCapability(
    "operator",
    "law-firm-run",
    tuple(sorted(ScenarioCommandKind, key=lambda kind: kind.value)),
    can_fork=True,
    can_read_all_audit=True,
)

start_request = ScenarioCommandRequest(
    "start-1", "start-key-1", "law-firm-run", scenario.content_hash, 1,
    coordinator.state.content_hash, "operator", ScenarioCommandKind.START,
)
started = coordinator.submit_command(start_request, operator)
step_request = ScenarioCommandRequest(
    "step-1", "step-key-1", "law-firm-run", scenario.content_hash, 1,
    coordinator.state.content_hash, "operator", ScenarioCommandKind.STEP,
)
stepped = coordinator.submit_command(step_request, operator)
public_state = coordinator.public_state_view()
public_output = coordinator.output_view(
    stepped.output_batch_hash,
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
)

checkpoint_request = ScenarioCommandRequest(
    "checkpoint-1", "checkpoint-key-1", "law-firm-run", scenario.content_hash, 1,
    coordinator.state.content_hash, "operator", ScenarioCommandKind.CHECKPOINT,
    requested_checkpoint_id="after-first-round",
)
checkpointed = coordinator.submit_command(checkpoint_request, operator)
child, forked = coordinator.fork(
    ScenarioForkRequest(
        "fork-1", "fork-key-1", "law-firm-run", scenario.content_hash, 1,
        checkpointed.checkpoint_hash, "law-firm-branch", "law-firm-branch-stream",
    ),
    operator,
    "law-firm-branch-memory.sqlite3",
)
assert child.run_view().status.value == "paused"
assert coordinator.command_result(start_request.command_id, operator) is started
```

`start` and `resume` are synchronous status changes; they never launch a background
loop. Subscriber callbacks observe already committed state and cannot issue commands
reentrantly. Commands, forks, and coherent queries are serialized by one coordinator
`RLock`; the output bus similarly serializes cross-thread publication and subscription
mutation while allowing same-thread callback unsubscribe for the next publication.
Non-`step` results remain in the capability-scoped command audit ledger and do not
fabricate V19 output batches. Output retention uses `maximum_output_records`, while
command-attempt and fork-attempt maps each use the finite derived bound
`maximum_output_records + maximum_rounds + len(ScenarioCommandKind)`, so a one-record
budget cannot block mandatory `start` plus the declared steps. Checkpoint/restore
publication uses same-filesystem hard-link create-if-absent semantics and physical file
ownership checks; it never replaces a concurrent artifact or target. JSON-RPC over
H2/WSS, the pure-Web editor, story interventions, live Blender updates, LLM/retrieval
providers, and remote Agents remain later phases; V21.3 adds none of those transports
or execution paths.

## World Studio V22

World Studio is the pure-Web authoring and run console described by the
[V22 design](docs/superpowers/specs/2026-09-02-world-studio-v22-design.md). Project
documents and revisions persist in the configured SQLite workspace. Run coordinators,
their in-memory registry, and released stream subscriptions are process-local: a browser
can reconnect and resume within the configured retention window, but after a server
restart it reopens the persisted project and creates a new run.

Install the locked server and browser inputs, install the single declared Playwright
browser for end-to-end verification, and build the hashed static bundle:

```text
python -m pip install -r requirements-world-studio.txt
cd world_studio_web
npm ci
npx playwright install chromium
npm run build
cd ..
```

The launcher accepts only configured roots and identifiers. It never reads `.env`,
loads a provider, or accepts a filesystem path from the browser. Create the workspace
and export directories first. A development-only loopback launch is:

```text
python -m tools.run_world_studio \
  --workspace-root .world-studio/workspace \
  --import-root examples \
  --import-source law-firm=examples/law_firm_scenario \
  --export-root .world-studio/exports \
  --static-root world_studio_web/dist \
  --bind-host 127.0.0.1 --bind-port 8443 \
  --origin http://127.0.0.1:8443 \
  --development-trust-all \
  --authority-id local-operator \
  --project-id law-firm --run-id law-firm-run \
  --agent-id alice
```

Open `http://127.0.0.1:8443/studio/`. The compatible root entry also serves the
application, while bundled assets and safe SPA deep links live under `/studio/`.

Development trust-all refuses non-loopback addresses. For a non-loopback deployment,
omit that flag, use one non-empty bearer token in a protected file, and supply both TLS
files. Hypercorn negotiates HTTP/2 for HTTPS RPC and the browser uses same-origin WSS:

```text
python -m tools.run_world_studio \
  --workspace-root /srv/world-studio/workspace \
  --import-root /srv/world-studio/imports \
  --import-source law-firm=/srv/world-studio/imports/law_firm_scenario \
  --export-root /srv/world-studio/exports \
  --static-root world_studio_web/dist \
  --bind-host 0.0.0.0 --bind-port 8443 \
  --origin https://studio.example.test:8443 \
  --auth-token-file /run/secrets/world-studio-token \
  --tls-certificate /run/secrets/world-studio.crt \
  --tls-private-key /run/secrets/world-studio.key \
  --maximum-sessions 128 --session-lifetime-seconds 3600 \
  --authority-id operator \
  --project-id law-firm --run-id law-firm-run \
  --agent-id alice
```

Open `https://studio.example.test:8443/studio/` and enter the configured token in
the bootstrap form. The token is sent once in the `Authorization` header to
`POST /session`; it is never placed in a URL, static asset, browser storage, or cookie.
The server instead sets a random, expiring, capacity-bounded `HttpOnly`, `Secure`,
`SameSite=Strict`, `Path=/` session cookie used by same-origin RPC and WSS. `DELETE
/session` logs out immediately; deterministic oldest-session eviction applies at the
configured capacity.

In the browser, import a configured source identifier, edit graph/map/property or raw
JSON views, repair pointer-specific diagnostics, validate, and compile an immutable
revision. Then create a run, start/step it, inspect only the public/Agent/network views
authorized by the session capability, checkpoint it, and fork from that checkpoint.
Lifecycle state and hashes change only after authoritative RPC results. Stream resume
recreates the identical released subscription, acknowledges committed source batches,
and uses the active audience's existing scoped-state RPC if retention cannot fill a
gap. Switching audience clears the old private view before loading the new one.

The server sends a restrictive CSP without `unsafe-inline` or `unsafe-eval`, revalidates
the HTML entry point, caches content-hashed assets immutably, and rejects static-path
traversal. V22 deliberately adds no React or React Flow, Protobuf or gRPC, provider/LLM
integration, remote worker/executor, live Blender mutation, or arbitrary state-set RPC.

## Verification

GitHub Actions runs:

```text
lake build
all Lean theorem tests
python3 -m unittest discover -s tests -v
```

The formal and simulation layers follow a RED → GREEN workflow. See `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md` for the current design and modeling limitations.

### Lean exact finite BB trace distribution V23.6

V23.6 lifts the finite Bianconi–Barabási replay kernel into an executable exact
probability law over all legal ordered target traces for a fixed typed initial
state, attachment count, and positive newborn-fitness schedule.
`NarrativeDynamics.Core.FitnessDistribution` defines the finite `TargetTrace`
carrier, exact rational `traceProbability`, authoritative `traceFinal`, finite
`eventProbability`, and exact rational `expectation`. The trace law is normalized
through the existing `continuationMass = 1` theorem rather than a second stochastic
kernel. `FitnessDistributionInvariance` proves that common positive fitness scaling
preserves every trace probability and final topology, and therefore preserves any
explicitly topology-invariant event probability.

The exact finite network fixtures connect this distribution back to the existing
small-world metrics. Starting from a two-node unit-fitness edge with `m = 1` and two
births, the six ordered traces have masses `1/4, 1/8, 1/8, 1/8, 1/4, 1/8`; exactly
the two star outcomes have final mesh diameter at most two, so
`P(meshDiameter ≤ 2) = 1/2`. Starting from a unit-fitness triangle with `m = 2`, all
six ordered one-birth traces have mass `1/6`. The final-graph event that the newborn
is adjacent to stable IDs 1 and 2 has probability `1/3` because the distinct ordered
traces `(1,2)` and `(2,1)` each contribute `1/6`; final states are not quotient- or
deduplicated. The two-birth edge experiment also retains the exact result
`E[degree(0)] = 15/8`.

The V23.6 proof workflow includes dedicated distribution, network-event, BB-only
naming, and trust gates. The maintained implementation adds no RNG, Monte Carlo,
PMF/Measure migration, random-fitness generator, asymptotic power-law or condensation
claim, or empirical/high-probability six-hop claim. The source trust audit rejects
`sorry`/`admit`, `native_decide`, user-axiom and unsafe declarations (including
private declarations), and zero proof-resource limits across the fitness Lean
modules and tests. As a conservative source lint, it also inspects braces in
ordinary strings as possible interpolation terms; raw strings remain literal data.
The log audit requires the expected theorem reports and accepts
only `propext`, `Classical.choice`, and `Quot.sound` as their transitive axioms.
The contract steps retain bounded timeouts and fail on proof or audit errors.
The audit's positive and negative regression cases run in the Lean proof job.

When Python discovery is enabled outside the existing fitness-branch exclusions,
its job checks out the same exact head as Lean and installs the existing pinned
`requirements-world-studio.txt` environment before running the full unittest suite.

### Lean BB-driven agent evolution V23.7

The BB agent model composes finite fitness attachment with one concrete rational
scalar-belief ABM. Fitness stays fixed in the BB state, agent receptivity and
broadcast thresholds stay fixed after birth, and every undirected BB edge supplies
unit influence in both directions. A birth retains every old agent's stable ID,
profile, belief, and exposure before propagation, initializes the newborn with zero
exposure, and then performs one synchronous propagation round from the complete
post-birth snapshot. Idle ticks perform the same propagation without changing the
network or multiplying the trace probability. The definitions and generic contracts
are in [`NetworkPropagation`](NarrativeDynamics/Core/NetworkPropagation.lean),
[`FitnessABM`](NarrativeDynamics/Core/FitnessABM.lean),
[`FitnessABMReplay`](NarrativeDynamics/Core/FitnessABMReplay.lean), and
[`FitnessABMDistribution`](NarrativeDynamics/Core/FitnessABMDistribution.lean).

The exact probability experiment fixes the seed, initial agent state, positive
attachment count, birth calendar, newborn fitness, profiles, and initial beliefs;
only legal ordered BB target traces vary. For a two-node unit-fitness seed with
beliefs `[1, 0]` and one zero-belief newborn, attaching the newborn to source `0`
has mass `1/2` and produces beliefs `[1, 1, 1]`, while attaching it to relay `1`
has mass `1/2` and produces `[1, 1, 0]`. Thus the newborn-broadcast event has exact
probability `1/2`. Changing only the seed fitness to `[1, 3]` changes that event
probability to `1/4`. The kernel-checked branches, delayed relay, idle tick, exact
event masses, and diameter witnesses are in the
[`FitnessABMDistribution` fixtures](NarrativeDynamics/Tests/FitnessABMDistribution.lean).

The checked replay is atomic: it returns either the complete joint state and exact
existing BB trace mass or one indexed validation error. The shared corpus compares
eight finite, single-round Lean examples with the existing fixed-roster Python V1
runtime; this is example agreement, not a proof of Python floating-point arithmetic
or a production birth adapter. The proof does not add V19 population creation,
adaptive fitness, full semantic cognition, asymptotic small-world results, or a
universal fixed-hop guarantee.

### Python authored BB population replay V1

The public [`replay_bb_population`](narrative_dynamics/abm/bb_runtime.py) API
executes a finite tuple of caller-authored ticks. Each birth's fixed, ordered
target tuple addresses the current append-only numeric-ID registry; a newborn's
external string ID is appended at the next numeric index and never recovered by
sorting V1 agents. The adapter computes BB fitness weights, each ordered birth
mass, and the cumulative trace mass exactly with `Fraction`. That mass is
conditional on the supplied seed, attachment count, birth calendar, fitness,
profiles, initial beliefs, and authored target choices. The behavioral step reuses
the existing V1 floating-point `simulate_round` propagation.

Run the production-imported [example](examples/bb_abm_runtime.py):

```bash
python3 -m examples.bb_abm_runtime
```

It prints the actual registry-ordered frames for two births followed by one idle
tick:

| Global tick | Epoch | V1 local round | Nodes/edges | Beliefs | Exposures | Exact trace mass |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 2/1 | `[1.0, 0.0]` | `[0, 0]` | `1` |
| 1 | 1 | 1 | 3/2 | `[1.0, 1.0, 0.0]` | `[0, 1, 0]` | `1/2` |
| 2 | 2 | 1 | 4/3 | `[1.0, 1.0, 1.0, 0.0]` | `[1, 2, 1, 0]` | `1/8` |
| 3 | 2 | 2 | 4/3 | `[1.0, 1.0, 1.0, 1.0]` | `[2, 4, 2, 1]` | `1/8` |

The global tick advances for every authored tick and the epoch advances only on a
birth. A birth creates a new V1 model epoch whose post-growth input is local round
zero and whose propagation result is local round one; an idle continues the
current epoch's local clock. Validation either returns a complete replay or one
atomic `BBRuntimeError`, with no successful prefix, partial topology, allocated ID,
or accumulated mass exposed on failure.

The [full-trace corpus](conformance/bb_abm_runtime_v1.json) covers 16 finite success
cases and 20 shared error cases, including every recorded prefix and transition,
against the Lean replay described by the
[design](docs/superpowers/specs/2026-09-14-bb-abm-runtime-adapter-v1-design.md).
This finite conformance establishes agreement for those checked inputs; it is not
a proof of all executions of the Python floating-point propagation. The supported
runtime boundary is the authored finite schedule exposed by the
[`bb_runtime` module](narrative_dynamics/abm/bb_runtime.py).
