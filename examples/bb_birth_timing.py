"""Compare authored birth schedules at one fixed global horizon.

Run with ``python -m examples.bb_birth_timing`` for a JSON report.
Both a birth tick and an idle tick include exactly one V1 propagation round.
This is a finite runtime experiment, not a new Lean theorem or random sampler.
"""

from dataclasses import asdict
import json

from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent, BBRuntimeBirth, BBRuntimeError, BBRuntimeNewborn,
    BBRuntimeRawSeed, BBRuntimeSeed, BBRuntimeTick,
)


def run_experiment(*, extra_idle_rounds: int = 0) -> dict[str, object]:
    """Replay the original schedules followed by a shared idle-only tail."""
    if type(extra_idle_rounds) is not int or extra_idle_rounds < 0:
        raise ValueError("extra_idle_rounds must be a nonnegative integer")
    seed = BBRuntimeSeed(
        BBRuntimeRawSeed(2, (1, 1), ((0, 1),)),
        (
            BBRuntimeAgent("0", "peer", 0.5, 0.5, 1),
            BBRuntimeAgent("1", "peer", 0.5, 0.5, 0),
        ),
        "bb-birth-timing", "1",
    )
    births = (
        BBRuntimeTick("birth", BBRuntimeBirth(
            1, (1,), BBRuntimeNewborn("2", "peer", 0.5, 0.5, 0),
        )),
        BBRuntimeTick("birth", BBRuntimeBirth(
            1, (2,), BBRuntimeNewborn("3", "peer", 0.5, 0.5, 0),
        )),
    )
    scenarios = []
    for schedule in ("BBII", "BIBI", "IIBB"):
        ordered_births = iter(births)
        ticks = tuple(next(ordered_births) if kind == "B" else BBRuntimeTick()
                      for kind in schedule)
        ticks += (BBRuntimeTick(),) * extra_idle_rounds
        replay = replay_bb_population(seed, 1, ticks)
        if isinstance(replay, BBRuntimeError):
            raise RuntimeError(f"{schedule}: {replay.stage}/{replay.code}")
        frames = (replay.initial,) + tuple(t.next_frame for t in replay.transitions)
        frame_rows = []
        for frame in frames:
            states = {a.agent_id: a for a in frame.population.agents}
            frame_rows.append({
                "tick": frame.tick_count,
                "epoch": frame.epoch,
                "local_round": frame.population.round_index,
                "agent_ids": list(frame.agent_ids),
                "beliefs": [states[i].belief for i in frame.agent_ids],
                "exposures": [states[i].exposure_count for i in frame.agent_ids],
            })
        scenarios.append({
            "schedule": schedule,
            "birth_targets": [list(t.birth.targets) for t in ticks if t.kind == "birth"],
            "final_edges": [list(e) for e in replay.final.topology.edges],
            "fitness": [str(f) for f in replay.final.topology.fitness],
            "final_profiles": [asdict(a) for a in replay.final.model.agents],
            "trace_mass": str(replay.final.trace_mass),
            "propagation_rounds_by_agent": [
                sum(i in f.agent_ids for f in frames[1:])
                for i in replay.final.agent_ids
            ],
            "frames": frame_rows,
        })
    return {
        "experiment": "bb-birth-timing-v1",
        "scope": "authored finite histories; fixed fitness; floating V1 propagation",
        "schedule_key": "B = birth plus one round; I = one idle propagation round",
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, sort_keys=True, allow_nan=False))
