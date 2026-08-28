from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from narrative_dynamics.external_validation import ExternalScoreRole
from narrative_dynamics.simulation import SimulationRunner
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout

_PROTOCOL_IMPORT_ERROR: Exception | None = None
try:
    import narrative_dynamics.studies.feher_hare_two_stage_v1 as study
    from narrative_dynamics.studies.two_stage_source import TwoStageSourceManifest
except Exception as error:
    _PROTOCOL_IMPORT_ERROR = error


class FeherHareProtocolTests(unittest.TestCase):
    def require_protocol(self):
        self.assertIsNone(
            _PROTOCOL_IMPORT_ERROR,
            f"Feher-Hare study protocol boundary is missing: {_PROTOCOL_IMPORT_ERROR}",
        )

    def _prepared(self):
        self.require_protocol()
        temp = tempfile.TemporaryDirectory()
        root, revision = build_synthetic_two_stage_checkout(Path(temp.name), magic_n=6, spaceship_n=6)
        manifest = TwoStageSourceManifest(
            name="synthetic-feher-hare",
            version="1",
            repository="test/synthetic-two-stage",
            revision=revision,
            license_reference="test-only",
            files=_files(root),
        )
        prepared = study.prepare_feher_hare_two_stage_v1(root=root, manifest=manifest)
        return temp, prepared

    def test_parameter_grids_are_exact_and_global_across_tasks(self):
        self.require_protocol()
        self.assertEqual(study.REACTIVE_GRID, {"beta": (0.5, 1.0, 2.0, 4.0)})
        self.assertEqual(
            study.HISTORY_GRID,
            {
                "beta": (0.5, 1.0, 2.0, 4.0),
                "memory_decay": (0.5, 0.75, 0.9, 1.0),
            },
        )

    def test_training_and_selection_use_brier_only(self):
        temp, prepared = self._prepared()
        self.addCleanup(temp.cleanup)
        with mock.patch.object(study, "two_stage_log_loss", side_effect=AssertionError("Log used before FINAL")):
            frozen = study.fit_and_freeze_feher_hare_models(
                runner=SimulationRunner(),
                prepared=prepared,
            )
        self.assertEqual(len(frozen.frozen_candidates), 3)

    def test_log_protocol_freezes_the_same_selected_candidates_without_reselection(self):
        temp, prepared = self._prepared()
        self.addCleanup(temp.cleanup)
        frozen = study.fit_and_freeze_feher_hare_models(runner=SimulationRunner(), prepared=prepared)
        bundle = study.build_feher_hare_protocol_bundle(prepared=prepared, frozen_models=frozen)
        self.assertEqual(
            tuple(candidate.content_hash for candidate in bundle.brier_protocol.candidates),
            tuple(candidate.content_hash for candidate in bundle.log_protocol.candidates),
        )

    def test_seed_plans_are_exact(self):
        self.require_protocol()
        self.assertEqual(study.TRAIN_SEEDS, (101, 102))
        self.assertEqual(study.SELECTION_SEEDS, (201, 202))
        self.assertEqual(study.FINAL_SEEDS, (301, 302))

    def test_planning_is_bookkeeping_baseline_but_all_pairs_are_scored(self):
        temp, prepared = self._prepared()
        self.addCleanup(temp.cleanup)
        frozen = study.fit_and_freeze_feher_hare_models(runner=SimulationRunner(), prepared=prepared)
        bundle = study.build_feher_hare_protocol_bundle(prepared=prepared, frozen_models=frozen)
        self.assertEqual(bundle.brier_protocol.baseline_name, "planning")
        self.assertEqual(bundle.log_protocol.baseline_name, "planning")
        self.assertEqual(len(bundle.preregistration.separation_rule.identity_payload()), 3)
        self.assertEqual(len(bundle.brier_protocol.candidates), 3)

    def test_magic_and_spaceship_final_cases_form_exact_task_strata(self):
        temp, prepared = self._prepared()
        self.addCleanup(temp.cleanup)
        frozen = study.fit_and_freeze_feher_hare_models(runner=SimulationRunner(), prepared=prepared)
        bundle = study.build_feher_hare_protocol_bundle(prepared=prepared, frozen_models=frozen)
        self.assertEqual({stratum.name for stratum in bundle.preregistration.strata}, {"magic_carpet", "spaceship"})
        assigned = {case for stratum in bundle.preregistration.strata for case in stratum.final_case_names}
        self.assertEqual(assigned, {case.name for case in prepared.final_targets.cases})

    def test_external_preregistration_has_no_constraint_plans(self):
        temp, prepared = self._prepared()
        self.addCleanup(temp.cleanup)
        frozen = study.fit_and_freeze_feher_hare_models(runner=SimulationRunner(), prepared=prepared)
        bundle = study.build_feher_hare_protocol_bundle(prepared=prepared, frozen_models=frozen)
        self.assertEqual(bundle.preregistration.constraint_plans, ())
        self.assertEqual(
            tuple(role for role, _ in bundle.preregistration.adequacy_thresholds_by_score),
            (ExternalScoreRole.BRIER, ExternalScoreRole.LOG),
        )
        self.assertEqual(prepared.evidence.claim_scope, "external_observational_predictive_only")


if __name__ == "__main__":
    unittest.main()
