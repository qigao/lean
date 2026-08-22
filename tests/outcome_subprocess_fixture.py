from __future__ import annotations

import sys

from narrative_dynamics.contracts import ModelRun, TraceEvent


class InvalidOutcomeProcessModel:
    name = "invalid-outcome-process"

    def simulate(self, scenario, parameters, rng):
        value = float(parameters["value"])
        print("invalid-outcome-stdout")
        print("invalid-outcome-stderr", file=sys.stderr)
        return ModelRun(
            events=(TraceEvent(tick=0, kind="emitted", data={"value": value}),),
            outcome={"value": "not-a-number"},
        )


def create_invalid_outcome_model():
    return InvalidOutcomeProcessModel()
