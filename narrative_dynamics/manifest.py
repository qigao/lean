from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from narrative_dynamics.contracts import (
    MANIFEST_SCHEMA_VERSION,
    ExperimentManifest,
    ExperimentStage,
    Scenario,
    stable_content_hash,
)


RUNTIME_IDENTITY = MappingProxyType(
    {
        "name": "narrative_dynamics",
        "version": "manifest-v1",
    }
)


def _type_name(value: object) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def _normalized_enum_or_text(value: object, *, label: str) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty string")
    return raw


def component_identity(
    component: object,
    *,
    fallback_name: str | None = None,
) -> dict[str, object]:
    """Describe one model or runtime component without executing it."""

    custom_identity = getattr(component, "manifest_identity", None)
    if callable(custom_identity):
        identity = custom_identity()
        if not isinstance(identity, Mapping):
            raise TypeError("custom component identity must be a mapping")
        stable_content_hash(identity)
        return dict(identity)

    raw_name = getattr(component, "name", None)
    if isinstance(raw_name, str) and raw_name:
        name = raw_name
    elif isinstance(fallback_name, str) and fallback_name:
        name = fallback_name
    else:
        raise ValueError("manifest component name must be a non-empty string")

    raw_version = getattr(component, "version", "unversioned")
    if raw_version is None:
        raw_version = "unversioned"
    if not isinstance(raw_version, str) or not raw_version:
        raise ValueError("manifest component version must be a non-empty string")

    identity: dict[str, object] = {
        "name": name,
        "version": raw_version,
        "type": _type_name(component),
    }

    implementation_revision = getattr(
        component,
        "implementation_revision",
        None,
    )
    if implementation_revision is not None:
        if (
            not isinstance(implementation_revision, str)
            or not implementation_revision
        ):
            raise ValueError(
                "component implementation revision must be a non-empty string"
            )
        identity["implementation_revision"] = implementation_revision

    implementation_hash = getattr(component, "implementation_hash", None)
    if implementation_hash is not None:
        if not isinstance(implementation_hash, str) or not implementation_hash:
            raise ValueError("component implementation hash must be a non-empty string")
        identity["implementation_hash"] = implementation_hash

    lifecycle = getattr(component, "lifecycle", None)
    if lifecycle is not None:
        identity["lifecycle"] = _normalized_enum_or_text(
            lifecycle,
            label="component lifecycle",
        )

    return identity


def callable_identity(value: Callable[..., object]) -> dict[str, object]:
    """Describe a metric extractor or other callable with a stable explicit name."""

    module = getattr(value, "__module__", value.__class__.__module__)
    qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
    return component_identity(
        value,
        fallback_name=f"{module}.{qualname}",
    )


def scenario_identity(scenario: object) -> dict[str, object]:
    """Hash a canonical scenario fully or mark a custom scenario identity fallback."""

    scenario_id = getattr(scenario, "id", None)
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("manifest scenario id must be a non-empty string")

    scenario_type = _type_name(scenario)
    if isinstance(scenario, Scenario):
        return {
            "id": scenario.id,
            "type": scenario_type,
            "content_scope": "full",
            "content_hash": scenario.content_hash,
        }

    manifest_payload = getattr(scenario, "manifest_payload", None)
    if callable(manifest_payload):
        payload = manifest_payload()
        return {
            "id": scenario_id,
            "type": scenario_type,
            "content_scope": "full",
            "content_hash": stable_content_hash(
                {
                    "id": scenario_id,
                    "type": scenario_type,
                    "payload": payload,
                }
            ),
        }

    version = getattr(scenario, "version", "unversioned")
    if not isinstance(version, str) or not version:
        version = "unversioned"
    identity = {
        "id": scenario_id,
        "type": scenario_type,
        "version": version,
    }
    return {
        **identity,
        "content_scope": "identity_only",
        "content_hash": stable_content_hash(identity),
    }


def required_manifest_hash(value: object, *, label: str) -> str:
    manifest = getattr(value, "manifest", None)
    if not isinstance(manifest, ExperimentManifest):
        raise RuntimeError(f"{label} is missing an experiment manifest")
    return manifest.content_hash


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "RUNTIME_IDENTITY",
    "ExperimentManifest",
    "ExperimentStage",
    "callable_identity",
    "component_identity",
    "required_manifest_hash",
    "scenario_identity",
    "stable_content_hash",
]
