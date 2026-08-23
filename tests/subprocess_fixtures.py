from __future__ import annotations

import os
import subprocess
import sys
import time

from narrative_dynamics.contracts import ModelRun, TraceEvent


_IMPORT_INSTANCE_COUNT = 0


class EchoProcessModel:
    name = "process-echo"

    def __init__(self) -> None:
        global _IMPORT_INSTANCE_COUNT
        _IMPORT_INSTANCE_COUNT += 1
        self.instance_count = _IMPORT_INSTANCE_COUNT

    def simulate(self, scenario, parameters, rng):
        draw = rng.random()
        print(f"echo stdout {scenario.id}", flush=True)
        print(f"echo stderr {scenario.id}", file=sys.stderr, flush=True)
        return ModelRun(
            events=(
                TraceEvent(
                    tick=0,
                    kind="echo",
                    data={"draw": draw},
                ),
            ),
            outcome={
                "value": float(parameters["value"]),
                "draw": draw,
                "instance_count": self.instance_count,
            },
        )


class SleepProcessModel:
    name = "process-sleep"

    def simulate(self, scenario, parameters, rng):
        print("sleep started", flush=True)
        time.sleep(float(parameters["seconds"]))
        return ModelRun(events=(), outcome={"completed": True})


class FailingProcessModel:
    name = "process-failure"

    def simulate(self, scenario, parameters, rng):
        print("before remote failure", flush=True)
        print("remote failure stderr", file=sys.stderr, flush=True)
        raise RuntimeError("fixture boom")


class NoisyProcessModel:
    name = "process-noisy"

    def simulate(self, scenario, parameters, rng):
        size = int(parameters["bytes"])
        os.write(sys.stdout.fileno(), b"x" * size)
        return ModelRun(events=(), outcome={"bytes": size})


class HugeTraceProcessModel:
    name = "process-huge-trace"

    def simulate(self, scenario, parameters, rng):
        size = int(parameters["bytes"])
        return ModelRun(events=(), outcome={"blob": "x" * size})


class DescendantProcessModel:
    name = "process-descendant"

    def simulate(self, scenario, parameters, rng):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ]
        )
        print(f"descendant started {child.pid}", flush=True)
        return ModelRun(events=(), outcome={"child_pid": child.pid})


class BusyProcessModel:
    name = "process-busy"

    def simulate(self, scenario, parameters, rng):
        print("busy started", flush=True)
        value = 0
        while True:
            value = (value + 1) % 1_000_003


def create_echo_model():
    return EchoProcessModel()


def create_sleep_model():
    return SleepProcessModel()


def create_failing_model():
    return FailingProcessModel()


def create_noisy_model():
    return NoisyProcessModel()


def create_huge_trace_model():
    return HugeTraceProcessModel()


def create_descendant_model():
    return DescendantProcessModel()


def create_busy_model():
    return BusyProcessModel()
