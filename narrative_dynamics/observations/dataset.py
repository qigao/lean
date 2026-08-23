from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from types import MappingProxyType

from narrative_dynamics.contracts import Scenario, stable_content_hash


OBSERVATION_DATASET_SCHEMA_VERSION = 1
_ROLE_ORDER = {
    "train": 0,
    "selection_validation": 1,
    "final_test": 2,
}


class ObservationPartitionRole(str, Enum):
    TRAIN = "train"
    SELECTION_VALIDATION = "selection_validation"
    FINAL_TEST = "final_test"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _role(value: ObservationPartitionRole | str) -> ObservationPartitionRole:
    try:
        return value if isinstance(value, ObservationPartitionRole) else ObservationPartitionRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError("observation partition role must be supported") from error


def _freeze(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{label} keys must be non-empty strings")
        return MappingProxyType({
            key: _freeze(value[key], label=f"{label}.{key}")
            for key in sorted(keys)
        })
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _freeze_mapping(value: Mapping[str, object] | None, *, label: str) -> Mapping[str, object]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _counts(value: Mapping[str, int], *, label: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    canonical: dict[str, int] = {}
    for key in sorted(value):
        _text(key, label=f"{label} key")
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{label} values must be integers")
        if count < 0:
            raise ValueError(f"{label} values must be non-negative")
        canonical[key] = count
    if sum(canonical.values()) <= 0:
        raise ValueError(f"{label} must have positive total mass")
    return MappingProxyType(canonical)


def _identifiers(values: tuple[str, ...] | list[str], *, label: str) -> tuple[str, ...]:
    identifiers = tuple(_text(value, label=label) for value in values)
    if not identifiers:
        raise ValueError(f"{label} must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} values must be unique")
    return identifiers


@dataclass(frozen=True)
class ObservationCase:
    name: str
    role: ObservationPartitionRole | str
    scenario: Scenario
    counts: Mapping[str, int]
    observation_ids: tuple[str, ...]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="observation case name"))
        object.__setattr__(self, "role", _role(self.role))
        if not isinstance(self.scenario, Scenario):
            raise TypeError("observation case scenario must be a Scenario")
        object.__setattr__(self, "counts", _counts(self.counts, label="observation case counts"))
        object.__setattr__(self, "observation_ids", _identifiers(self.observation_ids, label="observation source id"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, label="observation case provenance"))

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "scenario": {"id": self.scenario.id, "content_hash": self.scenario.content_hash},
            "counts": self.counts,
            "observation_ids": self.observation_ids,
            "provenance": self.provenance,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ObservationRecord:
    id: str
    scenario: Scenario
    counts: Mapping[str, int]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="observation record id"))
        if not isinstance(self.scenario, Scenario):
            raise TypeError("observation record scenario must be a Scenario")
        object.__setattr__(self, "counts", _counts(self.counts, label="observation record counts"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, label="observation record metadata"))

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    def identity_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scenario": {"id": self.scenario.id, "content_hash": self.scenario.content_hash},
            "counts": self.counts,
            "metadata": self.metadata,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @property
    def observation_fingerprint(self) -> str:
        return stable_content_hash({
            "scenario": {"id": self.scenario.id, "content_hash": self.scenario.content_hash},
            "counts": self.counts,
        })


@dataclass(frozen=True)
class ObservationPartition:
    name: str
    role: ObservationPartitionRole | str
    records: tuple[ObservationRecord, ...] = ()
    cases: tuple[ObservationCase, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="observation partition name"))
        role = _role(self.role)
        object.__setattr__(self, "role", role)
        records = tuple(self.records)
        cases = tuple(self.cases)
        if bool(records) == bool(cases):
            raise ValueError("observation partition requires exactly one of records or cases")
        if records:
            if any(not isinstance(record, ObservationRecord) for record in records):
                raise TypeError("observation partition records must be ObservationRecord values")
            records = tuple(sorted(records, key=lambda record: record.id))
            ids = tuple(record.id for record in records)
            scenarios = tuple(record.scenario.id for record in records)
            if len(set(ids)) != len(ids):
                raise ValueError("observation record ids must be unique within a partition")
            if len(set(scenarios)) != len(scenarios):
                raise ValueError("observation scenario ids must be unique within a partition")
        else:
            if any(not isinstance(case, ObservationCase) for case in cases):
                raise TypeError("observation partition cases must be ObservationCase values")
            if any(case.role is not role for case in cases):
                raise ValueError("observation partition cases must match the partition role")
            cases = tuple(sorted(cases, key=lambda case: case.name))
            names = tuple(case.name for case in cases)
            if len(set(names)) != len(names):
                raise ValueError("observation case names must be unique within a partition")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "cases", cases)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "records": tuple(record.identity_payload() for record in self.records),
            "cases": tuple(case.identity_payload() for case in self.cases),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ObservationDataset:
    name: str
    version: str
    source: object
    cases: tuple[ObservationCase, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)
    partitions: tuple[ObservationPartition, ...] = ()
    schema_version: int = OBSERVATION_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="observation dataset name"))
        object.__setattr__(self, "version", _text(self.version, label="observation dataset version"))
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != OBSERVATION_DATASET_SCHEMA_VERSION
        ):
            raise ValueError("unsupported observation dataset schema version")
        object.__setattr__(self, "source", _freeze(self.source, label="observation dataset source"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, label="observation dataset provenance"))

        cases = tuple(self.cases)
        partitions = tuple(self.partitions)
        if bool(cases) == bool(partitions):
            raise ValueError("observation dataset requires exactly one of cases or partitions")

        expected_roles = set(ObservationPartitionRole)
        if cases:
            if any(not isinstance(case, ObservationCase) for case in cases):
                raise TypeError("observation dataset cases must be ObservationCase values")
            cases = tuple(sorted(cases, key=lambda case: (_ROLE_ORDER[case.role.value], case.name)))
            names = tuple(case.name for case in cases)
            if len(set(names)) != len(names):
                raise ValueError("observation case names must be globally unique")
            if {case.role for case in cases} != expected_roles:
                raise ValueError("observation dataset must contain train, selection-validation, and final-test cases")
            observation_ids = tuple(
                observation_id for case in cases for observation_id in case.observation_ids
            )
            if len(set(observation_ids)) != len(observation_ids):
                raise ValueError("observation source ids must be globally unique across partitions")
            partitions = tuple(
                ObservationPartition(
                    name=role.value,
                    role=role,
                    cases=tuple(case for case in cases if case.role is role),
                )
                for role in ObservationPartitionRole
            )
        else:
            if any(not isinstance(partition, ObservationPartition) for partition in partitions):
                raise TypeError("observation dataset partitions must be ObservationPartition values")
            partitions = tuple(sorted(partitions, key=lambda item: _ROLE_ORDER[item.role.value]))
            roles = tuple(partition.role for partition in partitions)
            if set(roles) != expected_roles or len(roles) != len(expected_roles):
                raise ValueError("observation dataset must contain exactly one partition for each required role")
            names = tuple(partition.name for partition in partitions)
            if len(set(names)) != len(names):
                raise ValueError("observation partition names must be unique")
            if any(not partition.records for partition in partitions):
                raise ValueError("record-oriented observation partitions must be non-empty")
            record_ids = tuple(record.id for partition in partitions for record in partition.records)
            if len(set(record_ids)) != len(record_ids):
                raise ValueError("observation record ids must be globally unique across partitions")
            fingerprints = tuple(
                record.observation_fingerprint
                for partition in partitions
                for record in partition.records
            )
            if len(set(fingerprints)) != len(fingerprints):
                raise ValueError("identical source observations cannot cross dataset partitions")
            cases = ()

        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "partitions", partitions)

    def partition(self, role: ObservationPartitionRole | str) -> ObservationPartition:
        selected = _role(role)
        return next(partition for partition in self.partitions if partition.role is selected)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "provenance": self.provenance,
            "cases": tuple(case.identity_payload() for case in self.cases),
            "partitions": tuple(
                partition.identity_payload() for partition in self.partitions if partition.records
            ),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "source": _thaw(self.source),
            "provenance": _thaw(self.provenance),
        }
        if self.cases:
            payload["cases"] = [
                {
                    "name": case.name,
                    "role": case.role.value,
                    "scenario": {"id": case.scenario.id, "payload": _thaw(case.scenario.payload)},
                    "counts": dict(case.counts),
                    "observation_ids": list(case.observation_ids),
                    "provenance": _thaw(case.provenance),
                }
                for case in self.cases
            ]
        else:
            payload["partitions"] = [
                {
                    "name": partition.name,
                    "role": partition.role.value,
                    "records": [
                        {
                            "id": record.id,
                            "scenario": {"id": record.scenario.id, "payload": _thaw(record.scenario.payload)},
                            "counts": dict(record.counts),
                            "metadata": _thaw(record.metadata),
                        }
                        for record in partition.records
                    ],
                }
                for partition in self.partitions
            ]
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ObservationDataset":
        if not isinstance(payload, Mapping):
            raise TypeError("observation dataset payload must be a mapping")
        allowed = {
            "schema_version", "name", "version", "source", "provenance",
            "cases", "partitions", "content_hash",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"observation dataset payload has unknown fields: {tuple(sorted(unexpected))}")
        required = {"schema_version", "name", "version", "source", "content_hash"}
        if not required <= set(payload):
            raise ValueError("observation dataset payload is missing required fields")
        has_cases = "cases" in payload
        has_partitions = "partitions" in payload
        if has_cases == has_partitions:
            raise ValueError("observation dataset payload requires exactly one representation")

        provenance = payload.get("provenance", {})
        cases: tuple[ObservationCase, ...] = ()
        partitions: tuple[ObservationPartition, ...] = ()
        if has_cases:
            raw_cases = payload["cases"]
            if not isinstance(raw_cases, list):
                raise TypeError("observation dataset cases payload must be a list")
            case_fields = {"name", "role", "scenario", "counts", "observation_ids", "provenance"}
            built_cases: list[ObservationCase] = []
            for raw_case in raw_cases:
                if not isinstance(raw_case, Mapping) or set(raw_case) != case_fields:
                    raise ValueError("observation case payload fields must match the schema exactly")
                raw_scenario = raw_case["scenario"]
                if not isinstance(raw_scenario, Mapping) or set(raw_scenario) != {"id", "payload"}:
                    raise ValueError("observation scenario payload fields must match the schema")
                built_cases.append(ObservationCase(
                    name=raw_case["name"],
                    role=raw_case["role"],
                    scenario=Scenario(id=raw_scenario["id"], payload=raw_scenario["payload"]),
                    counts=raw_case["counts"],
                    observation_ids=tuple(raw_case["observation_ids"]),
                    provenance=raw_case["provenance"],
                ))
            cases = tuple(built_cases)
        else:
            raw_partitions = payload["partitions"]
            if not isinstance(raw_partitions, list):
                raise TypeError("observation dataset partitions payload must be a list")
            partition_fields = {"name", "role", "records"}
            record_fields = {"id", "scenario", "counts", "metadata"}
            built_partitions: list[ObservationPartition] = []
            for raw_partition in raw_partitions:
                if not isinstance(raw_partition, Mapping) or set(raw_partition) != partition_fields:
                    raise ValueError("observation partition payload fields must match the schema")
                raw_records = raw_partition["records"]
                if not isinstance(raw_records, list):
                    raise TypeError("observation records payload must be a list")
                records: list[ObservationRecord] = []
                for raw_record in raw_records:
                    if not isinstance(raw_record, Mapping) or set(raw_record) != record_fields:
                        raise ValueError("observation record payload fields must match the schema")
                    raw_scenario = raw_record["scenario"]
                    if not isinstance(raw_scenario, Mapping) or set(raw_scenario) != {"id", "payload"}:
                        raise ValueError("observation scenario payload fields must match the schema")
                    records.append(ObservationRecord(
                        id=raw_record["id"],
                        scenario=Scenario(id=raw_scenario["id"], payload=raw_scenario["payload"]),
                        counts=raw_record["counts"],
                        metadata=raw_record["metadata"],
                    ))
                built_partitions.append(ObservationPartition(
                    name=raw_partition["name"],
                    role=raw_partition["role"],
                    records=tuple(records),
                ))
            partitions = tuple(built_partitions)

        dataset = cls(
            name=payload["name"],
            version=payload["version"],
            source=payload["source"],
            cases=cases,
            provenance=provenance,
            partitions=partitions,
            schema_version=payload["schema_version"],
        )
        if payload["content_hash"] != dataset.content_hash:
            raise ValueError("observation dataset declared content hash does not match payload")
        return dataset


def load_observation_dataset(path: str | Path) -> ObservationDataset:
    return ObservationDataset.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "OBSERVATION_DATASET_SCHEMA_VERSION",
    "ObservationCase",
    "ObservationDataset",
    "ObservationPartition",
    "ObservationPartitionRole",
    "ObservationRecord",
    "load_observation_dataset",
]
