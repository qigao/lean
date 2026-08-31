from __future__ import annotations

from dataclasses import replace
import math
import unittest

from narrative_dynamics.cross_dataset_candidates import (
    GridCandidateEvaluation,
    TransferFamily,
    freeze_refit_candidates,
    freeze_zero_shot_candidates,
)
from tests.cross_dataset_transfer_fixtures import (
    builder_identities,
    complete_36_point_evaluations,
    digest,
    final_unlock_grant,
    new_source_lineage,
    prepared_transfer,
    r3_anchor,
)


class CrossDatasetCandidateTests(unittest.TestCase):
    def test_zero_shot_freeze_matches_r3_exactly(self) -> None:
        freeze = freeze_zero_shot_candidates(r3_anchor(), new_source_lineage())
        self.assertEqual(
            tuple(
                (row.family.value, row.parameters, row.source_candidate_hash)
                for row in freeze.candidates
            ),
            (
                (
                    "reactive",
                    (("beta", 0.5),),
                    "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
                ),
                (
                    "intentional",
                    (("beta", 2.0), ("memory_decay", 0.5)),
                    "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
                ),
                (
                    "planning",
                    (("beta", 4.0), ("memory_decay", 0.75)),
                    "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
                ),
            ),
        )
        self.assertFalse(freeze.new_data_training)
        self.assertFalse(freeze.new_data_selection)

    def test_zero_shot_rejects_anchor_or_lineage_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "R3 anchor"):
            freeze_zero_shot_candidates(
                replace(r3_anchor(), lock_commit="a" * 40),
                new_source_lineage(),
            )
        lineage = new_source_lineage()
        with self.assertRaisesRegex(ValueError, "lineage fields"):
            freeze_zero_shot_candidates(
                r3_anchor(),
                {**lineage, "final_projection_hash": digest("forbidden")},
            )

    def test_refit_freeze_requires_every_grid_point_and_lexical_tie_break(self) -> None:
        prepared, _backend = prepared_transfer()
        freeze = freeze_refit_candidates(
            train=prepared.train,
            selection=prepared.selection_validation,
            evaluations=complete_36_point_evaluations(with_tie=True),
            brier_loss_identity=digest("brier"),
            builder_identities=builder_identities(),
            simulation_identity=digest("simulation"),
            tie_break_identity="lexical_parameters_v1",
            target_spec_hash=digest("transfer-target-spec"),
            split_manifest_hash=prepared.split_manifest.content_hash,
        )
        self.assertEqual(len(freeze.evaluations), 36)
        self.assertFalse(freeze.final_projection_opened)
        self.assertFalse(freeze.final_model_execution)
        self.assertEqual(
            tuple(row.parameters for row in freeze.selected_candidates),
            (
                (("beta", 0.5),),
                (("beta", 0.5), ("memory_decay", 0.5)),
                (("beta", 0.5), ("memory_decay", 0.5)),
            ),
        )

    def _freeze(self, evaluations=None, **overrides: object):
        prepared, _backend = prepared_transfer()
        values: dict[str, object] = {
            "train": prepared.train,
            "selection": prepared.selection_validation,
            "evaluations": (
                complete_36_point_evaluations()
                if evaluations is None
                else evaluations
            ),
            "brier_loss_identity": digest("brier"),
            "builder_identities": builder_identities(),
            "simulation_identity": digest("simulation"),
            "tie_break_identity": "lexical_parameters_v1",
            "target_spec_hash": digest("transfer-target-spec"),
            "split_manifest_hash": prepared.split_manifest.content_hash,
        }
        values.update(overrides)
        return freeze_refit_candidates(**values)

    def test_refit_rejects_duplicate_missing_and_extra_grid_points(self) -> None:
        rows = complete_36_point_evaluations()
        cases = (
            (rows + (rows[0],), "duplicate"),
            (rows[:-1], "complete exact grid"),
            (
                rows
                + (
                    replace(
                        rows[0],
                        parameters=(("beta", 8.0),),
                        evaluation_receipt_hash=digest("extra-grid-point"),
                    ),
                ),
                "complete exact grid",
            ),
        )
        for mutated, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._freeze(mutated)

    def test_refit_rejects_score_seed_and_callable_drift(self) -> None:
        rows = complete_36_point_evaluations()
        mutations = (
            (replace(rows[0], score="LOG"), "BRIER"),
            (replace(rows[0], train_seeds=(101,)), "TRAIN seeds"),
            (replace(rows[0], selection_seeds=(201,)), "SELECTION seeds"),
            (
                replace(rows[0], brier_loss_identity=digest("wrong-loss")),
                "Brier loss identity",
            ),
            (
                replace(rows[0], builder_identity=digest("wrong-builder")),
                "builder identity",
            ),
            (
                replace(rows[0], simulation_identity=digest("wrong-simulation")),
                "simulation identity",
            ),
        )
        for replacement, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._freeze((replacement,) + rows[1:])

    def test_refit_rejects_projection_target_split_and_tie_break_drift(self) -> None:
        rows = complete_36_point_evaluations()
        mutations = (
            (
                replace(rows[0], train_projection_hash=digest("wrong-train")),
                "TRAIN projection",
            ),
            (
                replace(
                    rows[0],
                    selection_projection_hash=digest("wrong-selection"),
                ),
                "SELECTION projection",
            ),
            (
                replace(rows[0], target_spec_hash=digest("wrong-target")),
                "target spec",
            ),
            (
                replace(rows[0], split_manifest_hash=digest("wrong-split")),
                "split manifest",
            ),
        )
        for replacement, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._freeze((replacement,) + rows[1:])
        with self.assertRaisesRegex(ValueError, "tie-break"):
            self._freeze(tie_break_identity="numeric_parameters_v1")

    def test_non_finite_loss_is_rejected(self) -> None:
        row = complete_36_point_evaluations()[0]
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    replace(row, selection_brier_loss=value)

    def test_refit_rejects_any_final_projection_or_prior_result(self) -> None:
        prepared, backend = prepared_transfer()
        worker = backend.unlock(prepared.final_vault_handle, final_unlock_grant())
        with self.assertRaisesRegex(ValueError, "FINAL input"):
            self._freeze(final_projection=worker)
        with self.assertRaisesRegex(ValueError, "FINAL input"):
            self._freeze(prior_final_result={"status": "known"})

    def test_evaluation_receipts_are_durable_and_strict(self) -> None:
        row = complete_36_point_evaluations()[0]
        payload = row.to_payload()
        self.assertEqual(GridCandidateEvaluation.from_payload(payload), row)
        with self.assertRaises(ValueError):
            GridCandidateEvaluation.from_payload({**payload, "unknown": True})


if __name__ == "__main__":
    unittest.main()
