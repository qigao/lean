"""Observe a common fixed-network tail; finite values do not prove convergence."""

import json

from examples.bb_birth_timing import run_experiment


def run_tail_experiment() -> dict[str, object]:
    horizons = (0, 1, 2, 4, 8, 16, 32, 64)
    histories = run_experiment(extra_idle_rounds=max(horizons))["scenarios"]
    checkpoints = []
    for extra_rounds in horizons:
        scenarios = []
        for history in histories:
            baseline = history["frames"][4]
            frame = history["frames"][4 + extra_rounds]
            profiles = {p["agent_id"]: p for p in history["final_profiles"]}
            beliefs = frame["beliefs"]
            scenarios.append({
                "schedule": history["schedule"],
                "agent_ids": frame["agent_ids"],
                "beliefs": beliefs,
                "within_history_spread": max(beliefs) - min(beliefs),
                # Derived from the unchanged V1 broadcast-threshold rule.
                "broadcasting": [
                    value >= profiles[agent_id]["broadcast_threshold"]
                    for agent_id, value in zip(frame["agent_ids"], beliefs)
                ],
                "new_exposures": [
                    current - prior for current, prior in
                    zip(frame["exposures"], baseline["exposures"])
                ],
                "birth_count": frame["epoch"],
                "edges": history["final_edges"],
                "trace_mass": history["trace_mass"],
            })
        checkpoints.append({
            "extra_rounds": extra_rounds,
            "global_tick": 4 + extra_rounds,
            "max_between_history_gap": max(
                max(values) - min(values)
                for values in zip(*(s["beliefs"] for s in scenarios))
            ),
            "scenarios": scenarios,
        })
    return {
        "experiment": "bb-birth-timing-tail-v1",
        "scope": "finite floating runtime observations; no asymptotic theorem",
        "baseline_global_tick": 4,
        "checkpoints": checkpoints,
    }


if __name__ == "__main__":
    print(json.dumps(run_tail_experiment(), indent=2, sort_keys=True, allow_nan=False))
