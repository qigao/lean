from __future__ import annotations

import unittest

import narrative_dynamics


class CrossDatasetPublicApiTests(unittest.TestCase):
    def test_root_exports_only_durable_non_final_contracts(self) -> None:
        required = {
            "CROSS_DATASET_TRANSFER_CLAIM_SCOPE",
            "DatasetCandidateCatalog",
            "DatasetCatalogEntry",
            "DatasetSearchProtocol",
            "DatasetSelectionDecision",
            "DatasetSelectionStatus",
            "select_dataset_candidate",
            "DatasetSourceFile",
            "DatasetSourceManifest",
            "SemanticInvarianceEvidence",
            "TransformReceipt",
            "VerifiedSourceSnapshot",
            "PublicSplitManifest",
            "ZeroShotCandidateFreeze",
            "RefitCandidateFreeze",
            "TransferInferenceEvidence",
            "TransferScore",
            "TransferProtocol",
            "TransferScoreRelease",
            "DualTransferPreflight",
            "TransferAuthorizationReceipt",
            "TransferStudyTerminal",
            "CrossDatasetTransferReport",
        }
        forbidden = {
            "RestrictedStudySecret",
            "RoleAssignmentIndex",
            "FinalVaultBackend",
            "FinalWorkerProjection",
            "_ParticipantLossBlock",
            "SealedTransferPredictionArtifact",
            "GitTransferAttemptStore",
            "parse_transfer_authorization",
            "execute_transfer_final_predictions",
            "score_and_report_transfer",
            "run_locked_cross_dataset_transfer_final",
        }
        exported = set(narrative_dynamics.__all__)
        self.assertTrue(required.issubset(exported), required - exported)
        self.assertTrue(forbidden.isdisjoint(exported), forbidden & exported)
        self.assertEqual(len(narrative_dynamics.__all__), len(exported))
        self.assertTrue(all(hasattr(narrative_dynamics, name) for name in exported))

    def test_no_root_public_callable_can_open_final(self) -> None:
        forbidden_fragments = ("unlock", "open_final", "vault", "locked_final")
        for name in narrative_dynamics.__all__:
            with self.subTest(name=name):
                self.assertFalse(
                    any(fragment in name.lower() for fragment in forbidden_fragments)
                )


if __name__ == "__main__":
    unittest.main()
