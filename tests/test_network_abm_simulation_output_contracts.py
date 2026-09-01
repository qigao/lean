from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationCommandResultPayload,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputRecord,
    SimulationOutputView,
    SimulationPrivatePerceptPayload,
    SimulationStateDeltaPayload,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    ObservationChannel,
    SituatedPercept,
    SituatedPerceptFidelity,
)


SCENARIO_HASH = "sha256:" + "1" * 64
STATE_HASH = "sha256:" + "2" * 64
PRIOR_STATE_HASH = "sha256:" + "3" * 64
ROUND_RESULT_HASH = "sha256:" + "4" * 64


class SimulationOutputContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.percept = SituatedPercept(
            "percept-1",
            1,
            "alice",
            "event-1",
            "sha256:" + "5" * 64,
            (ObservationChannel.AUDITORY,),
            SituatedPerceptFidelity.DETECTED,
        )

    @staticmethod
    def _public_record(sequence: int = 1) -> SimulationOutputRecord:
        return SimulationOutputRecord(
            "stream-1",
            SCENARIO_HASH,
            sequence,
            1,
            STATE_HASH,
            SimulationOutputKind.COMMAND_RESULT,
            SimulationOutputAudience.PUBLIC,
            None,
            (),
            SimulationCommandResultPayload("command-1", True, "accepted"),
        )

    @staticmethod
    def _batch(records: tuple[SimulationOutputRecord, ...]) -> SimulationOutputBatch:
        return SimulationOutputBatch(
            "stream-1",
            SCENARIO_HASH,
            PRIOR_STATE_HASH,
            STATE_HASH,
            ROUND_RESULT_HASH,
            records[0].sequence,
            records[-1].sequence,
            records,
        )

    def test_payloads_are_frozen_hashable_and_canonical(self):
        payload = SimulationStateDeltaPayload(
            PRIOR_STATE_HASH,
            STATE_HASH,
            ("bob", "alice"),
            ("passage-2", "passage-1"),
            (),
        )

        # Mutation caught: payload identity depends on caller-provided tuple order.
        self.assertEqual(payload.changed_agent_ids, ("alice", "bob"))
        # Mutation caught: payload dictionaries omit canonical fields used for transport.
        self.assertEqual(
            payload.to_dict(),
            {
                "prior_snapshot_hash": PRIOR_STATE_HASH,
                "next_snapshot_hash": STATE_HASH,
                "changed_agent_ids": ["alice", "bob"],
                "changed_passage_ids": ["passage-1", "passage-2"],
                "changed_object_ids": [],
            },
        )
        # Mutation caught: payload hashes are unstable or payload instances are unhashable.
        self.assertEqual(hash(payload), hash(payload))
        # Mutation caught: frozen payloads can be altered after their hash is observed.
        with self.assertRaises(FrozenInstanceError):
            payload.changed_agent_ids = ()  # type: ignore[misc]

    def test_blender_delta_rejects_duplicate_and_mapping_shapes(self):
        # Mutation caught: duplicate Blender identities overwrite each other downstream.
        with self.assertRaisesRegex(ValueError, "unique"):
            SimulationBlenderDeltaPayload(
                (("alice", "lobby"), ("alice", "office")),
                (),
                (),
            )
        # Mutation caught: arbitrary mappings bypass the canonical tuple wire shape.
        with self.assertRaisesRegex(TypeError, "tuple"):
            SimulationBlenderDeltaPayload(  # type: ignore[arg-type]
                {"alice": "lobby"},
                (),
                (),
            )

    def test_agent_record_requires_matching_owner(self):
        payload = SimulationPrivatePerceptPayload(self.percept)
        # Mutation caught: private records can be addressed to a different Agent.
        with self.assertRaisesRegex(ValueError, "owner"):
            SimulationOutputRecord(
                "stream-1", SCENARIO_HASH, 1, 1, STATE_HASH,
                SimulationOutputKind.PERCEPT_PRIVATE,
                SimulationOutputAudience.AGENT,
                "bob",
                (),
                payload,
            )

    def test_public_record_rejects_private_payload(self):
        payload = SimulationPrivatePerceptPayload(self.percept)
        # Mutation caught: a private percept can cross the public audience boundary.
        with self.assertRaisesRegex(ValueError, "audience"):
            SimulationOutputRecord(
                "stream-1", SCENARIO_HASH, 1, 1, STATE_HASH,
                SimulationOutputKind.PERCEPT_PRIVATE,
                SimulationOutputAudience.PUBLIC,
                None,
                (),
                payload,
            )

    def test_record_rejects_payload_kind_mismatch(self):
        # Mutation caught: record kind can disagree with the typed payload schema.
        with self.assertRaisesRegex(ValueError, "kind"):
            SimulationOutputRecord(
                "stream-1", SCENARIO_HASH, 1, 1, STATE_HASH,
                SimulationOutputKind.STATE_DELTA,
                SimulationOutputAudience.PUBLIC,
                None,
                (),
                SimulationCommandResultPayload("command-1", True, "accepted"),
            )

    def test_batch_rejects_duplicate_sequences(self):
        first = self._public_record(1)
        duplicate = replace(
            first,
            payload=SimulationCommandResultPayload("command-2", True, "accepted"),
        )
        # Mutation caught: two source records can claim the same global sequence.
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._batch((first, duplicate))

    def test_batch_rejects_gapped_sequences(self):
        first = self._public_record(1)
        third = replace(
            first,
            sequence=3,
            payload=SimulationCommandResultPayload("command-2", True, "accepted"),
        )
        # Mutation caught: a complete batch can silently omit a source sequence.
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._batch((first, third))

    def test_batch_rejects_mixed_scenario_hashes(self):
        first = self._public_record(1)
        second = replace(
            first,
            scenario_hash="sha256:" + "6" * 64,
            sequence=2,
            payload=SimulationCommandResultPayload("command-2", True, "accepted"),
        )
        # Mutation caught: records from another scenario can enter an atomic batch.
        with self.assertRaisesRegex(ValueError, "scenario"):
            self._batch((first, second))

    def test_batch_rejects_wrong_state_hash(self):
        record = replace(self._public_record(), state_hash=PRIOR_STATE_HASH)
        # Mutation caught: records can bind the prior state instead of the accepted next state.
        with self.assertRaisesRegex(ValueError, "state"):
            self._batch((record,))

    def test_batch_rejects_mixed_rounds(self):
        first = self._public_record(1)
        second = replace(
            first,
            sequence=2,
            round_index=2,
            payload=SimulationCommandResultPayload("command-2", True, "accepted"),
        )
        # Mutation caught: one batch can combine records from different accepted rounds.
        with self.assertRaisesRegex(ValueError, "round"):
            self._batch((first, second))

    def test_batch_rejects_noncanonical_record_order(self):
        command = self._public_record(2)
        state = SimulationOutputRecord(
            "stream-1",
            SCENARIO_HASH,
            1,
            1,
            STATE_HASH,
            SimulationOutputKind.STATE_DELTA,
            SimulationOutputAudience.PUBLIC,
            None,
            (),
            SimulationStateDeltaPayload(PRIOR_STATE_HASH, STATE_HASH, (), (), ()),
        )
        # Mutation caught: sequence order can override the fixed semantic kind order.
        with self.assertRaisesRegex(ValueError, "canonical"):
            self._batch((state, command))

    def test_batch_rejects_record_payload_kind_mismatch(self):
        record = self._public_record()
        object.__setattr__(record, "kind", SimulationOutputKind.STATE_DELTA)
        # Mutation caught: a batch trusts a corrupted record's mismatched kind and payload.
        with self.assertRaisesRegex(ValueError, "kind"):
            self._batch((record,))

    def test_audience_views_filter_ownership_and_retain_source_gaps(self):
        public = self._public_record(1)
        alice = SimulationOutputRecord(
            "stream-1", SCENARIO_HASH, 2, 1, STATE_HASH,
            SimulationOutputKind.PERCEPT_PRIVATE,
            SimulationOutputAudience.AGENT,
            "alice",
            (),
            SimulationPrivatePerceptPayload(self.percept),
        )
        bob_percept = replace(self.percept, percept_id="percept-2", agent_id="bob")
        bob = replace(
            alice,
            sequence=3,
            owner_agent_id="bob",
            payload=SimulationPrivatePerceptPayload(bob_percept),
        )
        batch = self._batch((public, alice, bob))

        agent_view = SimulationOutputView.from_batch(
            batch,
            SimulationAudienceCapability(SimulationOutputAudience.AGENT, "bob"),
        )
        public_view = SimulationOutputView.from_batch(
            batch,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
        )

        # Mutation caught: Agent filtering renumbers records or leaks another Agent's record.
        self.assertEqual(tuple(record.sequence for record in agent_view.records), (1, 3))
        # Mutation caught: public filtering exposes any non-public record.
        self.assertEqual(tuple(record.sequence for record in public_view.records), (1,))
        # Mutation caught: a filtered view loses the exact source batch identity.
        self.assertEqual(agent_view.source_batch_hash, batch.content_hash)


if __name__ == "__main__":
    unittest.main()
