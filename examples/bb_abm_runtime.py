from fractions import Fraction

from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent,
    BBRuntimeBirth,
    BBRuntimeError,
    BBRuntimeNewborn,
    BBRuntimeRawSeed,
    BBRuntimeSeed,
    BBRuntimeTick,
)


def main() -> int:
    seed = BBRuntimeSeed(
        BBRuntimeRawSeed(2, (1, 1), ((0, 1),)),
        (
            BBRuntimeAgent("0", "peer", 1, 0.5, 1),
            BBRuntimeAgent("1", "peer", 1, 0.5, 0),
        ),
        "bb-successive-births",
        "1",
    )
    ticks = (
        BBRuntimeTick(
            "birth",
            BBRuntimeBirth(
                1,
                (1,),
                BBRuntimeNewborn("2", "peer", 1, 0.5, 0),
            ),
        ),
        BBRuntimeTick(
            "birth",
            BBRuntimeBirth(
                1,
                (2,),
                BBRuntimeNewborn("3", "peer", 1, 0.5, 0),
            ),
        ),
        BBRuntimeTick(),
    )
    out = replay_bb_population(seed, 1, ticks)
    if isinstance(out, BBRuntimeError):
        print(f"replay failed: {out.stage}/{out.code}")
        return 1
    for frame in (out.initial,) + tuple(t.next_frame for t in out.transitions):
        states = {a.agent_id: a for a in frame.population.agents}
        beliefs = [states[i].belief for i in frame.agent_ids]
        exposures = [states[i].exposure_count for i in frame.agent_ids]
        print(
            f"tick={frame.tick_count} epoch={frame.epoch} "
            f"local_round={frame.population.round_index} "
            f"nodes={frame.topology.node_count} edges={len(frame.topology.edges)} "
            f"beliefs={beliefs} exposures={exposures} mass={frame.trace_mass}"
        )
    return 0 if out.final.trace_mass == Fraction(1, 8) else 1


if __name__ == "__main__":
    raise SystemExit(main())
