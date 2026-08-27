from __future__ import annotations

import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import calibrate_seed_block_variants
from narrative_dynamics.narrative.world import advance_world_step, world_state_from_story
from tests.test_narrative_stochastic_world import (
    _selected_outcome,
    _stochastic_phase_model,
)
from tests.test_narrative_world_transition import (
    intent,
    make_world_domain,
    make_world_story,
)


def run_test_narrative_once(*, root_seed: int, level: float):
    domain = make_world_domain()
    story = make_world_story()
    model = _stochastic_phase_model(domain)
    prior = world_state_from_story(story, domain, seed=root_seed)
    result = advance_world_step(
        story,
        domain,
        prior,
        model,
        (intent("d-bob-phase", "bob-ready"),),
    )
    record = next(item for item in result.transitions if item.actor_id == "bob")
    assert record.stochastic_sample is not None
    outcome = _selected_outcome(result, "bob")
    sign = 1.0 if outcome == "ready" else -1.0
    return level + sign, record.stochastic_sample.content_hash


class NarrativeAleatoricAdapter:
    name = "narrative-aleatoric-adapter"

    def simulate(self, scenario, parameters, rng):
        root_seed = rng.getrandbits(64)
        value, lineage_hash = run_test_narrative_once(
            root_seed=root_seed,
            level=float(parameters["level"]),
        )
        return ModelRun(
            events=(),
            outcome={
                "value": value,
                "narrative_root_seed": root_seed,
                "narrative_lineage_hash": lineage_hash,
            },
        )


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class NarrativeStochasticSeedDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = NarrativeAleatoricAdapter()
        self.scenario = Scenario(id="narrative-aleatoric", payload={})

    def test_outer_seed_deterministically_maps_to_narrative_root_seed_and_manifest(self):
        first = self.runner.run_once(
            self.model,
            self.scenario,
            {"level": 0.0},
            seed=11,
        )
        second = self.runner.run_once(
            self.model,
            self.scenario,
            {"level": 0.0},
            seed=11,
        )
        self.assertEqual(first.seed, 11)
        self.assertEqual(second.seed, 11)
        self.assertEqual(
            first.outcome["narrative_root_seed"],
            second.outcome["narrative_root_seed"],
        )
        self.assertEqual(
            first.outcome["narrative_lineage_hash"],
            second.outcome["narrative_lineage_hash"],
        )
        self.assertEqual(first.manifest.content_hash, second.manifest.content_hash)

        different = None
        for outer_seed in range(12, 257):
            candidate = self.runner.run_once(
                self.model,
                self.scenario,
                {"level": 0.0},
                seed=outer_seed,
            )
            if (
                candidate.outcome["narrative_root_seed"]
                != first.outcome["narrative_root_seed"]
            ):
                different = candidate
                break
        self.assertIsNotNone(different)
        assert different is not None
        self.assertNotEqual(different.seed, first.seed)
        self.assertNotEqual(
            different.outcome["narrative_root_seed"],
            first.outcome["narrative_root_seed"],
        )

    def test_existing_seed_block_variants_detect_aleatoric_acceptance_drift(self):
        by_sign = {}
        for outer_seed in range(1, 257):
            trace = self.runner.run_once(
                self.model,
                self.scenario,
                {"level": 0.0},
                seed=outer_seed,
            )
            sign = 1 if float(trace.outcome["value"]) > 0.0 else -1
            by_sign.setdefault(sign, outer_seed)
            if set(by_sign) == {-1, 1}:
                break
        self.assertEqual(set(by_sign), {-1, 1})
        seed_a = by_sign[-1]
        seed_b = by_sign[1]

        report = calibrate_seed_block_variants(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (-1.0, 1.0)},
            seed_blocks=((seed_a,),),
            seed_offsets=(0, seed_b - seed_a),
            extractor=value_metrics,
            target={"value": 0.0},
        )
        accepted = tuple(
            variant.calibration.acceptance_set.parameters
            for variant in report.variants
        )
        self.assertEqual(len(accepted), 2)
        self.assertNotEqual(accepted[0], accepted[1])
        self.assertEqual(report.accepted_intersection.parameters, ())


if __name__ == "__main__":
    unittest.main()
