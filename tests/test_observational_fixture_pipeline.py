from __future__ import annotations

from pathlib import Path
import unittest

from narrative_dynamics.observations import (
    ObservationPartitionRole,
    load_observation_dataset,
)


FIXTURE = Path("fixtures/observations/prison_initial_choice_v1.json")


class PrisonObservationFixtureTests(unittest.TestCase):
    def test_committed_fixture_is_synthetic_versioned_and_role_complete(self):
        self.assertTrue(FIXTURE.is_file(), "versioned synthetic prison fixture is missing")
        dataset = load_observation_dataset(FIXTURE)

        self.assertEqual(dataset.name, "prison-initial-choice")
        self.assertEqual(dataset.version, "1.0.0")
        self.assertEqual(dataset.source["kind"], "synthetic_fixture")
        self.assertEqual(dataset.source["purpose"], "protocol_integration_test")
        self.assertFalse(dataset.provenance["empirical_human_data"])
        self.assertFalse(dataset.provenance["population_representative"])
        self.assertEqual(
            {partition.role for partition in dataset.partitions},
            set(ObservationPartitionRole),
        )
        for partition in dataset.partitions:
            self.assertTrue(partition.records)
            for record in partition.records:
                self.assertEqual(
                    set(record.counts),
                    {"scout", "escape", "submit"},
                )

    def test_committed_counts_are_fixed_and_persistence_pair_is_discriminating(self):
        self.assertTrue(FIXTURE.is_file(), "versioned synthetic prison fixture is missing")
        dataset = load_observation_dataset(FIXTURE)
        final_partition = dataset.partition(ObservationPartitionRole.FINAL_TEST)
        final_by_id = {record.id: record for record in final_partition.records}

        self.assertEqual(
            final_by_id["final-1"].count_map,
            {"scout": 18, "escape": 66, "submit": 16},
        )
        self.assertEqual(
            final_by_id["final-2"].count_map,
            {"scout": 43, "escape": 45, "submit": 12},
        )
        first = dict(final_by_id["final-1"].scenario.payload)
        second = dict(final_by_id["final-2"].scenario.payload)
        self.assertEqual(
            {key: value for key, value in first.items() if key != "guard_persistence"},
            {key: value for key, value in second.items() if key != "guard_persistence"},
        )
        self.assertEqual(first["guard_persistence"], 0.2)
        self.assertEqual(second["guard_persistence"], 0.9)


if __name__ == "__main__":
    unittest.main()
