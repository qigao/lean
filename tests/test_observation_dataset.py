from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.contracts import Scenario
from narrative_dynamics.observations import (
    OBSERVATION_DATASET_SCHEMA_VERSION,
    ObservationCase,
    ObservationDataset,
    ObservationPartitionRole,
    load_observation_dataset,
)


def make_dataset(*, reverse: bool = False) -> ObservationDataset:
    mutable_source = {"study": "prison-choice", "release": [1, 0]}
    cases = [
        ObservationCase(
            name="train-case",
            role=ObservationPartitionRole.TRAIN,
            scenario=Scenario(id="train", payload={"condition": "train"}),
            counts={"choice.a": 8, "choice.b": 2},
            observation_ids=tuple(f"train-{index}" for index in range(10)),
            provenance={"collector": {"site": "A"}},
        ),
        ObservationCase(
            name="selection-case",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            scenario=Scenario(id="selection", payload={"condition": "selection"}),
            counts={"choice.a": 8, "choice.b": 2},
            observation_ids=tuple(f"selection-{index}" for index in range(10)),
            provenance={"collector": {"site": "B"}},
        ),
        ObservationCase(
            name="final-case",
            role=ObservationPartitionRole.FINAL_TEST,
            scenario=Scenario(id="final", payload={"condition": "final"}),
            counts={"choice.a": 9, "choice.b": 1},
            observation_ids=tuple(f"final-{index}" for index in range(10)),
            provenance={"collector": {"site": "C"}},
        ),
    ]
    if reverse:
        cases.reverse()
    return ObservationDataset(
        name="prison-observations",
        version="2026.08.23",
        source=mutable_source,
        cases=tuple(cases),
    )


class ObservationDatasetTests(unittest.TestCase):
    def test_dataset_is_versioned_immutable_roundtrippable_and_order_stable(self):
        dataset = make_dataset()
        reversed_dataset = make_dataset(reverse=True)

        self.assertEqual(dataset.schema_version, OBSERVATION_DATASET_SCHEMA_VERSION)
        self.assertEqual(dataset.content_hash, reversed_dataset.content_hash)
        self.assertEqual(
            tuple(part.role for part in dataset.partitions),
            (
                ObservationPartitionRole.TRAIN,
                ObservationPartitionRole.SELECTION_VALIDATION,
                ObservationPartitionRole.FINAL_TEST,
            ),
        )
        payload = dataset.to_payload()
        rebuilt = ObservationDataset.from_payload(json.loads(json.dumps(payload)))
        self.assertEqual(rebuilt, dataset)
        self.assertEqual(rebuilt.content_hash, dataset.content_hash)

        payload["source"]["study"] = "mutated"
        payload["cases"][0]["counts"]["choice.a"] = 0
        self.assertEqual(dataset.source["study"], "prison-choice")
        self.assertEqual(dataset.cases[0].count_map["choice.a"], 8)

    def test_loader_reads_exact_payload_and_verifies_declared_hash(self):
        dataset = make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            path.write_text(json.dumps(dataset.to_payload()), encoding="utf-8")
            loaded = load_observation_dataset(path)
            self.assertEqual(loaded.content_hash, dataset.content_hash)

            payload = dataset.to_payload()
            payload["content_hash"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_observation_dataset(path)

    def test_roles_names_observations_counts_and_payload_schema_fail_closed(self):
        cases = list(make_dataset().cases)
        with self.assertRaises(ValueError):
            ObservationDataset(
                name="missing-final",
                version="1",
                source={},
                cases=tuple(case for case in cases if case.role is not ObservationPartitionRole.FINAL_TEST),
            )

        duplicate = ObservationCase(
            name="duplicate-observation",
            role=ObservationPartitionRole.FINAL_TEST,
            scenario=Scenario(id="duplicate", payload={}),
            counts={"choice.a": 1},
            observation_ids=(cases[0].observation_ids[0],),
        )
        with self.assertRaises(ValueError):
            ObservationDataset(
                name="duplicate",
                version="1",
                source={},
                cases=tuple(cases[:-1]) + (duplicate,),
            )

        for bad_counts in (
            {"choice.a": -1},
            {"choice.a": 1.5},
            {"choice.a": True},
            {"": 1},
        ):
            with self.subTest(bad_counts=bad_counts):
                with self.assertRaises((TypeError, ValueError)):
                    ObservationCase(
                        name="bad",
                        role=ObservationPartitionRole.TRAIN,
                        scenario=Scenario(id="bad", payload={}),
                        counts=bad_counts,
                        observation_ids=("bad-1",),
                    )

        payload = make_dataset().to_payload()
        payload["unexpected"] = 1
        with self.assertRaises(ValueError):
            ObservationDataset.from_payload(payload)

        payload = make_dataset().to_payload()
        payload["cases"][0]["unexpected"] = 1
        with self.assertRaises(ValueError):
            ObservationDataset.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
