from narrative_dynamics.abm.situated_contracts import (
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageSpec,
    PlaceSpec,
    SituatedWorldModel,
    WorldObjectSpec,
    initialize_situated_world,
)


def office_model(*, memo_portable: bool = True, alice_capacity: int = 1):
    return SituatedWorldModel(
        model_id="small-office",
        version="1.0",
        places=(
            PlaceSpec("records", "Records room"),
            PlaceSpec("open", "Open office"),
            PlaceSpec("lobby", "Lobby"),
            PlaceSpec("manager", "Manager office"),
        ),
        passages=(
            PassageSpec("lobby-open", "lobby", "open", initially_open=True),
            PassageSpec("open-lobby", "open", "lobby", initially_open=True),
            PassageSpec("open-records", "open", "records", initially_open=True),
            PassageSpec("records-open", "records", "open", initially_open=True),
            PassageSpec("open-manager", "open", "manager", initially_open=False),
            PassageSpec("manager-open", "manager", "open", initially_open=False),
        ),
        agents=(
            EmbodiedAgentSpec("alice", "analyst", "records", alice_capacity),
            EmbodiedAgentSpec("bob", "engineer", "open", 1),
            EmbodiedAgentSpec("carol", "designer", "open", 1),
            EmbodiedAgentSpec("dana", "manager", "manager", 1),
        ),
        objects=(
            WorldObjectSpec(
                "memo",
                "official memo",
                "records",
                portable=memo_portable,
                evidence=(EvidenceFact("restructuring", "approved"),),
            ),
            WorldObjectSpec("badge", "access badge", "lobby", portable=True),
        ),
    )


def office_state(**kwargs):
    model = office_model(**kwargs)
    return model, initialize_situated_world(model)
