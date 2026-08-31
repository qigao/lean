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

## Verification

GitHub Actions runs:

```text
lake build
all Lean theorem tests
python3 -m unittest discover -s tests -v
```

The formal and simulation layers follow a RED → GREEN workflow. See `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md` for the current design and modeling limitations.
