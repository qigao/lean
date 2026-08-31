from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedCognitiveModel,
    SituatedGoalReward,
    SituatedGoalSpec,
    SituatedHypothesis,
    SituatedObservationLikelihood,
    SituatedObservationRule,
    SituatedObservationSymbol,
)
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from tests.situated_fixtures import office_model


def agent_model(agent_id: str, *, prior_approved: float = 0.5):
    actions = (
        SituatedActionSpec("inspect", SituatedActionKind.INSPECT, target_id="memo", required_place_ids=("records",), repeatable=False),
        SituatedActionSpec("move", SituatedActionKind.MOVE, target_id="records-open", required_place_ids=("records",), repeatable=False),
        SituatedActionSpec(
            "tell",
            SituatedActionKind.TELL,
            message="The restructuring is approved.",
            required_place_ids=("open",),
            source_event_kinds=(SituatedActionKind.INSPECT, SituatedActionKind.TELL),
            repeatable=False,
        ),
        SituatedActionSpec("wait", SituatedActionKind.WAIT),
    )
    hypotheses = (
        SituatedHypothesis("approved", "restructuring approved"),
        SituatedHypothesis("denied", "restructuring denied"),
    )
    symbols = (
        SituatedObservationSymbol("approved", "evidence supports approval"),
        SituatedObservationSymbol("denied", "evidence opposes approval"),
    )
    rules = (
        SituatedObservationRule("inspect-approved", "approved", "inspect", SituatedActionKind.INSPECT, "inspected", "restructuring", "approved"),
        SituatedObservationRule("tell-approved", "approved", "tell", SituatedActionKind.TELL, "told", "message", "The restructuring is approved."),
    )
    likelihoods = tuple(
        SituatedObservationLikelihood(action.action_id, hypothesis.hypothesis_id, symbol.symbol_id, (
            0.9 if hypothesis.hypothesis_id == symbol.symbol_id else 0.1
        ) if action.action_id in {"inspect", "tell"} else 0.5)
        for action in actions
        for hypothesis in hypotheses
        for symbol in symbols
    )
    goals = (SituatedGoalSpec("inform", "discover and share reliable information", 1.0),)
    reward_values = {
        ("approved", "inspect"): 3.0,
        ("approved", "move"): 2.0,
        ("approved", "tell"): 4.0,
        ("approved", "wait"): 0.0,
        ("denied", "inspect"): 3.0,
        ("denied", "move"): 0.0,
        ("denied", "tell"): -4.0,
        ("denied", "wait"): 0.0,
    }
    rewards = tuple(
        SituatedGoalReward("inform", hypothesis, action, value)
        for (hypothesis, action), value in reward_values.items()
    )
    return SituatedAgentCognitiveModel(
        agent_id,
        hypotheses,
        PlanningBeliefState({"approved": prior_approved, "denied": 1.0 - prior_approved}),
        symbols,
        rules,
        likelihoods,
        actions,
        (("inspect", "move", "tell", "wait"), ("move", "tell", "wait")),
        (),
        goals,
        rewards,
        discount=0.8,
        beta=4.0,
    )


def cognitive_office_model():
    world = office_model()
    return SituatedCognitiveModel(
        "office-cognition",
        "1.0",
        world,
        tuple(agent_model(item.agent_id) for item in world.agents),
    )
