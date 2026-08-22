from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from narrative_dynamics.contracts import _freeze_mapping
from narrative_dynamics.manifest import stable_content_hash


class ModelLifecycle(str, Enum):
    """How a registered source supplies model state to one runner batch."""

    SHARED_INSTANCE = "shared_instance"
    FRESH_PER_BATCH = "fresh_per_batch"


def _validated_metadata_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


@dataclass(frozen=True)
class ModelSchema:
    """Immutable identity for one opaque adapter schema definition.

    The runtime hashes and versions the supplied definition but does not claim
    to interpret or enforce a particular schema language.
    """

    name: str
    version: str
    definition: Mapping[str, object]
    specified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_metadata_text(self.name, label="model schema name"),
        )
        object.__setattr__(
            self,
            "version",
            _validated_metadata_text(self.version, label="model schema version"),
        )
        if not isinstance(self.specified, bool):
            raise TypeError("model schema specified flag must be boolean")
        object.__setattr__(
            self,
            "definition",
            _freeze_mapping(self.definition, label="model schema definition"),
        )

    @classmethod
    def unspecified(cls, role: str) -> "ModelSchema":
        canonical_role = _validated_metadata_text(
            role,
            label="unspecified model schema role",
        )
        return cls(
            name=f"unspecified:{canonical_role}",
            version="unversioned",
            definition={},
            specified=False,
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "name": self.name,
                "version": self.version,
                "definition": self.definition,
                "specified": self.specified,
            }
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "specified": self.specified,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ModelContract:
    """Version, implementation revision, and boundary schemas for one adapter."""

    version: str = "unversioned"
    implementation_revision: str = "unversioned"
    parameter_schema: ModelSchema = field(
        default_factory=lambda: ModelSchema.unspecified("parameters")
    )
    scenario_schema: ModelSchema = field(
        default_factory=lambda: ModelSchema.unspecified("scenario")
    )
    event_schema: ModelSchema = field(
        default_factory=lambda: ModelSchema.unspecified("events")
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "version",
            _validated_metadata_text(self.version, label="model version"),
        )
        object.__setattr__(
            self,
            "implementation_revision",
            _validated_metadata_text(
                self.implementation_revision,
                label="model implementation revision",
            ),
        )
        for label, schema in (
            ("parameter", self.parameter_schema),
            ("scenario", self.scenario_schema),
            ("event", self.event_schema),
        ):
            if not isinstance(schema, ModelSchema):
                raise TypeError(f"{label} schema must be a ModelSchema")

    @classmethod
    def from_source(cls, source: object) -> "ModelContract":
        declared = getattr(source, "contract", None)
        if declared is not None:
            if not isinstance(declared, cls):
                raise TypeError("source contract must be a ModelContract")
            return declared

        version = getattr(source, "version", "unversioned")
        if version is None:
            version = "unversioned"
        revision = getattr(source, "implementation_revision", None)
        if revision is None:
            revision = getattr(source, "implementation_hash", "unversioned")
        if revision is None:
            revision = "unversioned"

        schemas: dict[str, ModelSchema] = {}
        for attribute, role in (
            ("parameter_schema", "parameters"),
            ("scenario_schema", "scenario"),
            ("event_schema", "events"),
        ):
            value = getattr(source, attribute, None)
            if value is None:
                schemas[attribute] = ModelSchema.unspecified(role)
            elif isinstance(value, ModelSchema):
                schemas[attribute] = value
            else:
                raise TypeError(f"source {attribute} must be a ModelSchema")

        return cls(
            version=version,
            implementation_revision=revision,
            parameter_schema=schemas["parameter_schema"],
            scenario_schema=schemas["scenario_schema"],
            event_schema=schemas["event_schema"],
        )

    @property
    def complete(self) -> bool:
        return (
            self.version != "unversioned"
            and self.implementation_revision != "unversioned"
            and self.parameter_schema.specified
            and self.scenario_schema.specified
            and self.event_schema.specified
        )

    @property
    def schema_identities(self) -> dict[str, object]:
        return {
            "parameters": self.parameter_schema.manifest_identity(),
            "scenario": self.scenario_schema.manifest_identity(),
            "events": self.event_schema.manifest_identity(),
        }

    def manifest_identity(self) -> dict[str, object]:
        return {
            "version": self.version,
            "implementation_revision": self.implementation_revision,
            "schemas": self.schema_identities,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.manifest_identity())


def _declared_source_identity(source: object, attribute: str) -> str | None:
    value = getattr(source, attribute, None)
    if value is None and attribute == "implementation_revision":
        value = getattr(source, "implementation_hash", None)
    if value is None:
        return None
    return _validated_metadata_text(value, label=f"source {attribute}")


def validate_contract_matches_source(
    source: object,
    contract: ModelContract,
) -> None:
    """Reject explicit metadata that contradicts identity declared by a source."""

    declared_version = _declared_source_identity(source, "version")
    if (
        declared_version not in (None, "unversioned")
        and contract.version != declared_version
    ):
        raise ValueError("model contract version disagrees with source version")

    declared_revision = _declared_source_identity(
        source,
        "implementation_revision",
    )
    if (
        declared_revision not in (None, "unversioned")
        and contract.implementation_revision != declared_revision
    ):
        raise ValueError(
            "model contract implementation revision disagrees with source"
        )

    declared_contract = getattr(source, "contract", None)
    if declared_contract is not None and declared_contract != contract:
        raise ValueError("explicit model contract disagrees with source contract")


__all__ = [
    "ModelContract",
    "ModelLifecycle",
    "ModelSchema",
    "validate_contract_matches_source",
]
