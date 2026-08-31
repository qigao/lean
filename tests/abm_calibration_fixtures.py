from narrative_dynamics.abm.autonomy_contracts import TruthObservation
from narrative_dynamics.abm.calibration_contracts import (
    CalibrationSplit,
    EmpiricalABMCase,
    ObservedABMSnapshot,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
)


def snapshot(round_index: int = 1, *, mean_trust: float = 0.625):
    return ObservedABMSnapshot(
        round_index,
        active_share=1.0,
        mean_active_belief=0.5,
        mean_trust=mean_trust,
        learned_edge_rate=0.5,
        effective_active_edge_rate=0.5,
        rewired_edge_rate=0.0,
        verification_rate=1 / 3,
        active_sharing_rate=2 / 3,
        transition_rate=1 / 3,
        role_entropy=0.5,
    )


def empirical_case(
    case_id: str,
    split: CalibrationSplit,
    *,
    reverse_beliefs: bool = False,
):
    beliefs = (("a", 1.0), ("c", 0.0))
    if reverse_beliefs:
        beliefs = tuple(reversed(beliefs))
    return EmpiricalABMCase(
        case_id,
        split,
        beliefs,
        ((PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),),),
        ((TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),),),
        (snapshot(),),
    )
