from __future__ import annotations

from copy import copy
from dataclasses import fields, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.simulation_output import project_simulation_output
from narrative_dynamics.abm.simulation_output_contracts import (
    SIMULATION_OUTPUT_VIEW_SCHEMA,
    SimulationNetworkMetricsPayload,
    SimulationOutputAudience,
    SimulationOutputKind,
)
from narrative_dynamics.abm.simulation_output_journal import (
    SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA,
    SIMULATION_PUBLIC_JOURNAL_SCHEMA,
    replay_public_simulation_journal,
    write_public_simulation_journal,
)
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


class SimulationOutputJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        package_root = write_law_firm_package(cls.root / "scenario")
        mutate_json(
            package_root / "run.json",
            "/allowed_output_kinds",
            [kind.value for kind in SimulationOutputKind],
        )
        refresh_manifest_hash(package_root, "run", "run")
        cls.scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )
        database = cls.root / "memory.sqlite3"
        initial = initialize_compiled_scenario(database, cls.scenario)
        cls.first_round = simulate_situated_network_round(
            database,
            cls.scenario.runtime_model,
            initial,
        )
        cls.first_batch = project_simulation_output(
            cls.scenario,
            cls.first_round,
            stream_id="law-firm-run",
            first_sequence=41,
        )
        cls.second_round = simulate_situated_network_round(
            database,
            cls.scenario.runtime_model,
            cls.first_round.next_state,
        )
        cls.second_batch = project_simulation_output(
            cls.scenario,
            cls.second_round,
            stream_id="law-firm-run",
            first_sequence=cls.first_batch.last_sequence + 1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.case = TemporaryDirectory(dir=self.root)
        self.path = Path(self.case.name) / "law-firm.jsonl"

    def tearDown(self) -> None:
        self.case.cleanup()

    def _write_two_batches(self) -> None:
        write_public_simulation_journal(self.path, self.first_batch)
        write_public_simulation_journal(self.path, self.second_batch)

    def _documents(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]

    def _store_documents(self, documents: list[dict[str, object]]) -> None:
        text = "".join(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
            for document in documents
        )
        self.path.write_text(text, encoding="utf-8", newline="")

    @staticmethod
    def _refresh_record_hashes(batch: dict[str, object]) -> None:
        records = batch["records"]
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, dict)
            record_body = {
                key: value for key, value in record.items() if key != "record_hash"
            }
            record["record_hash"] = stable_content_hash(record_body)

    @classmethod
    def _refresh_integrity(cls, documents: list[dict[str, object]]) -> None:
        for batch in documents[1:]:
            cls._refresh_record_hashes(batch)
            records = batch["records"]
            assert isinstance(records, list)
            view_body = {
                "schema": SIMULATION_OUTPUT_VIEW_SCHEMA,
                "stream_id": batch["stream_id"],
                "scenario_hash": batch["scenario_hash"],
                "prior_state_hash": batch["prior_state_hash"],
                "next_state_hash": batch["next_state_hash"],
                "round_result_hash": batch["round_result_hash"],
                "first_sequence": batch["first_sequence"],
                "last_sequence": batch["last_sequence"],
                "record_hashes": [record["record_hash"] for record in records],
                "source_batch_hash": batch["source_batch_hash"],
                "checkpoint": batch["checkpoint"],
            }
            batch["view_hash"] = stable_content_hash(view_body)
        header = documents[0]
        header_body = {
            "schema": header["schema"],
            "stream_id": header["stream_id"],
            "scenario_hash": header["scenario_hash"],
            "parent_journal_hash": header["parent_journal_hash"],
            "view_hashes": [batch["view_hash"] for batch in documents[1:]],
        }
        header["content_hash"] = stable_content_hash(header_body)

    def _assert_sanitized_failure(self, action) -> None:
        private_value = "private-payload-do-not-expose"
        with self.assertRaises((TypeError, ValueError)) as raised:
            action()
        message = str(raised.exception)
        self.assertNotIn(private_value, message)
        self.assertNotIn(str(self.path.parent), message)

    def test_two_real_batches_append_atomically_and_replay_exact_public_views(self) -> None:
        first = write_public_simulation_journal(self.path, self.first_batch)
        second = write_public_simulation_journal(self.path, self.second_batch)
        replayed = replay_public_simulation_journal(self.path)

        # Replacing instead of appending, or decoding to dictionaries, breaks equality.
        self.assertEqual(replayed, second)
        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertEqual(len(replayed.batches), 2)
        self.assertTrue(all(
            record.audience is SimulationOutputAudience.PUBLIC
            for view in replayed.batches
            for record in view.records
        ))
        original_public = tuple(
            record
            for batch in (self.first_batch, self.second_batch)
            for record in batch.records
            if record.audience is SimulationOutputAudience.PUBLIC
        )
        replayed_records = tuple(
            record for view in replayed.batches for record in view.records
        )
        self.assertEqual(replayed_records, original_public)
        self.assertEqual(
            tuple(type(record.payload) for record in replayed_records),
            tuple(type(record.payload) for record in original_public),
        )
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("percept.private", text)
        self.assertNotIn("agent.decision", text)
        self.assertNotIn('"audience":"agent"', text)
        self.assertTrue(text.endswith("\n"))
        documents = self._documents()
        self.assertEqual(documents[0]["schema"], SIMULATION_PUBLIC_JOURNAL_SCHEMA)
        self.assertTrue(all(
            document["schema"] == SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA
            for document in documents[1:]
        ))

    def test_filtered_views_keep_global_source_sequence_gaps_and_batch_identity(self) -> None:
        journal = write_public_simulation_journal(self.path, self.first_batch)
        view = journal.batches[0]

        # Renumbering public records would erase their global source identities.
        self.assertEqual(view.first_sequence, self.first_batch.first_sequence)
        self.assertEqual(view.last_sequence, self.first_batch.last_sequence)
        self.assertEqual(view.source_batch_hash, self.first_batch.content_hash)
        self.assertEqual(
            tuple(record.sequence for record in view.records),
            tuple(
                record.sequence
                for record in self.first_batch.records
                if record.audience is SimulationOutputAudience.PUBLIC
            ),
        )
        self.assertGreater(
            view.last_sequence - view.first_sequence + 1,
            len(view.records),
        )

    def test_private_only_projected_batches_replay_empty_public_views_with_continuity(
        self,
    ) -> None:
        private_scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(
                    SimulationOutputKind.PERCEPT_PRIVATE.value,
                    SimulationOutputKind.AGENT_DECISION.value,
                    SimulationOutputKind.MEMORY_UPDATE.value,
                    SimulationOutputKind.SOCIAL_UPDATE.value,
                ),
            ),
        )
        first_batch = project_simulation_output(
            private_scenario,
            self.first_round,
            stream_id="law-firm-private-only",
            first_sequence=301,
        )
        second_batch = project_simulation_output(
            private_scenario,
            self.second_round,
            stream_id="law-firm-private-only",
            first_sequence=first_batch.last_sequence + 1,
        )
        for batch in (first_batch, second_batch):
            self.assertTrue(batch.records)
        self.assertTrue(all(
            record.audience is SimulationOutputAudience.AGENT
            for batch in (first_batch, second_batch)
            for record in batch.records
        ))

        write_public_simulation_journal(self.path, first_batch)
        written = write_public_simulation_journal(self.path, second_batch)
        replayed = replay_public_simulation_journal(self.path)

        # Requiring a retained public record would lose private-only source continuity.
        self.assertEqual(replayed, written)
        self.assertEqual(len(replayed.batches), 2)
        for view, source in zip(
            replayed.batches,
            (first_batch, second_batch),
            strict=True,
        ):
            self.assertEqual(view.records, ())
            self.assertEqual(view.first_sequence, source.first_sequence)
            self.assertEqual(view.last_sequence, source.last_sequence)
            self.assertEqual(view.source_batch_hash, source.content_hash)
            self.assertEqual(view.prior_state_hash, source.prior_state_hash)
            self.assertEqual(view.next_state_hash, source.next_state_hash)
        self.assertEqual(
            replayed.batches[1].first_sequence,
            replayed.batches[0].last_sequence + 1,
        )
        self.assertEqual(
            replayed.batches[1].prior_state_hash,
            replayed.batches[0].next_state_hash,
        )

        text = self.path.read_text(encoding="utf-8")
        documents = self._documents()
        self.assertTrue(all(document["records"] == [] for document in documents[1:]))
        for batch in (first_batch, second_batch):
            for record in batch.records:
                self.assertNotIn(record.content_hash, text)
                self.assertNotIn(record.payload.content_hash, text)
                self.assertNotIn(record.kind.value, text)

    def test_write_revalidates_selected_records_before_any_serialization(self) -> None:
        private_index = next(
            index
            for index, record in enumerate(self.first_batch.records)
            if record.audience is SimulationOutputAudience.AGENT
        )
        corrupted_record = copy(self.first_batch.records[private_index])
        object.__setattr__(
            corrupted_record,
            "audience",
            SimulationOutputAudience.PUBLIC,
        )
        object.__setattr__(corrupted_record, "owner_agent_id", None)
        records = list(self.first_batch.records)
        records[private_index] = corrupted_record
        corrupted_batch = copy(self.first_batch)
        object.__setattr__(corrupted_batch, "records", tuple(records))

        # Trusting only the mutated audience would serialize the private payload.
        with self.assertRaises((TypeError, ValueError)):
            write_public_simulation_journal(self.path, corrupted_batch)
        self.assertFalse(self.path.exists())

    def test_write_rejects_corrupted_nested_public_payload_before_replacement(self) -> None:
        write_public_simulation_journal(self.path, self.first_batch)
        previous = self.path.read_bytes()
        private_value = "private-payload-do-not-expose"
        metric_index = next(
            index
            for index, record in enumerate(self.second_batch.records)
            if record.kind is SimulationOutputKind.NETWORK_METRICS
        )
        corrupted_metrics = copy(
            self.second_batch.records[metric_index].payload.metrics
        )
        object.__setattr__(
            corrupted_metrics,
            "tracked_belief_mean",
            private_value,
        )
        corrupted_payload = copy(self.second_batch.records[metric_index].payload)
        object.__setattr__(corrupted_payload, "metrics", corrupted_metrics)
        corrupted_record = copy(self.second_batch.records[metric_index])
        object.__setattr__(corrupted_record, "payload", corrupted_payload)
        records = list(self.second_batch.records)
        records[metric_index] = corrupted_record
        corrupted_batch = copy(self.second_batch)
        object.__setattr__(corrupted_batch, "records", tuple(records))

        # Reusing a nested frozen value bypasses its constructor and can publish it.
        with self.assertRaises((TypeError, ValueError)):
            write_public_simulation_journal(self.path, corrupted_batch)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertNotIn(private_value.encode("utf-8"), self.path.read_bytes())

    def test_write_ignores_overridden_instance_serialization_in_real_bytes(self) -> None:
        private_value = "private-payload-do-not-expose"
        record_index = next(
            index
            for index, record in enumerate(self.first_batch.records)
            if record.kind is SimulationOutputKind.STATE_DELTA
        )
        original_record = self.first_batch.records[record_index]
        corrupted_payload = copy(original_record.payload)
        payload_document = original_record.payload.to_dict()
        payload_document["private_override"] = private_value
        object.__setattr__(
            corrupted_payload,
            "to_dict",
            lambda: payload_document,
        )
        corrupted_record = copy(original_record)
        object.__setattr__(corrupted_record, "payload", corrupted_payload)
        record_document = original_record.to_dict()
        record_document["payload"] = payload_document
        record_document["private_override"] = private_value
        object.__setattr__(
            corrupted_record,
            "to_dict",
            lambda: record_document,
        )
        records = list(self.first_batch.records)
        records[record_index] = corrupted_record
        corrupted_batch = copy(self.first_batch)
        object.__setattr__(corrupted_batch, "records", tuple(records))

        written = write_public_simulation_journal(self.path, corrupted_batch)

        # Caller-owned serializers must never participate in the journal wire bytes.
        encoded = self.path.read_bytes()
        self.assertNotIn(private_value.encode("utf-8"), encoded)
        self.assertEqual(
            written.batches[0].source_batch_hash,
            self.first_batch.content_hash,
        )
        self.assertEqual(replay_public_simulation_journal(self.path), written)

    def test_replay_rejects_tampered_payload_hash(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[1]["records"][0]["payload_hash"] = "sha256:" + "0" * 64
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_tampered_record_hash(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[1]["records"][0]["record_hash"] = "sha256:" + "0" * 64
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_tampered_view_hash(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[1]["view_hash"] = "sha256:" + "0" * 64
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_tampered_source_batch_hash(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[1]["source_batch_hash"] = "sha256:" + "0" * 64
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_tampered_header_hash(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[0]["content_hash"] = "sha256:" + "0" * 64
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_source_sequence_regression_even_with_fresh_hashes(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[2]["first_sequence"] -= 1
        self._refresh_integrity(documents)
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_mixed_round_view_even_with_fresh_hashes(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        records = documents[1]["records"]
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 2)
        target = next(
            record
            for record in records
            if record["kind"] != SimulationOutputKind.NETWORK_METRICS.value
        )
        target["round_index"] += 1
        self._refresh_integrity(documents)
        self._store_documents(documents)

        # Refreshed hashes must not make a multi-round atomic view replayable.
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_state_chain_break_even_with_fresh_hashes(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[2]["prior_state_hash"] = "sha256:" + "1" * 64
        self._refresh_integrity(documents)
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_scenario_break_even_with_fresh_hashes(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[2]["scenario_hash"] = "sha256:" + "2" * 64
        for record in documents[2]["records"]:
            record["scenario_hash"] = documents[2]["scenario_hash"]
        self._refresh_integrity(documents)
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_stream_break_even_with_fresh_hashes(self) -> None:
        self._write_two_batches()
        documents = self._documents()
        documents[2]["stream_id"] = "foreign-stream"
        for record in documents[2]["records"]:
            record["stream_id"] = documents[2]["stream_id"]
        self._refresh_integrity(documents)
        self._store_documents(documents)
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_invalid_utf8_without_exposing_path(self) -> None:
        self.path.write_bytes(b'{"schema":"private-payload-do-not-expose"}\xff\n')
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_sanitizes_missing_file_errors_that_contain_local_paths(self) -> None:
        # Letting the filesystem exception escape would disclose the absolute path.
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_duplicate_json_keys_without_exposing_value(self) -> None:
        self.path.write_text(
            '{"schema":"private-payload-do-not-expose",'
            '"schema":"private-payload-do-not-expose"}\n',
            encoding="utf-8",
        )
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_nonfinite_json_constants_without_exposing_value(self) -> None:
        self.path.write_text(
            '{"schema":"private-payload-do-not-expose","value":NaN}\n',
            encoding="utf-8",
        )
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_sanitizes_deeply_nested_json_parser_failure(self) -> None:
        depth = 5_000
        self.path.write_text(
            '{"private-payload-do-not-expose":'
            + "[" * depth
            + "0"
            + "]" * depth
            + "}\n",
            encoding="utf-8",
        )

        # A parser recursion failure must not bypass the sanitized replay boundary.
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_replay_rejects_truncated_final_line(self) -> None:
        self._write_two_batches()
        self.path.write_bytes(self.path.read_bytes()[:-1])
        self._assert_sanitized_failure(
            lambda: replay_public_simulation_journal(self.path)
        )

    def test_failed_atomic_replace_preserves_previous_bytes_and_cleans_stage(self) -> None:
        write_public_simulation_journal(self.path, self.first_batch)
        previous = self.path.read_bytes()
        entries = set(self.path.parent.iterdir())

        # os.replace is the external atomic-publication boundary; file assertions stay real.
        with patch(
            "narrative_dynamics.abm.simulation_output_journal.os.replace",
            side_effect=OSError("publish failed"),
        ):
            with self.assertRaises(OSError):
                write_public_simulation_journal(self.path, self.second_batch)

        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(set(self.path.parent.iterdir()), entries)

    def test_replay_reconstructs_network_metrics_as_the_exact_typed_contract(self) -> None:
        journal = write_public_simulation_journal(self.path, self.first_batch)
        replayed = replay_public_simulation_journal(self.path)
        expected = next(
            record.payload
            for record in journal.batches[0].records
            if record.kind is SimulationOutputKind.NETWORK_METRICS
        )
        actual = next(
            record.payload
            for record in replayed.batches[0].records
            if record.kind is SimulationOutputKind.NETWORK_METRICS
        )

        # A generic mapping or partial metrics decoder loses the typed payload contract.
        self.assertIsInstance(actual, SimulationNetworkMetricsPayload)
        self.assertEqual(actual, expected)
        self.assertEqual(
            set(actual.metrics.to_dict()),
            {field.name for field in fields(actual.metrics)},
        )


if __name__ == "__main__":
    unittest.main()
