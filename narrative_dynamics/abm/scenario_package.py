"""Strict loader for trusted, declarative situated scenario packages."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.scenario_package_contracts import (
    SCENARIO_DOCUMENT_SCHEMA,
    SCENARIO_PACKAGE_SCHEMA,
    ScenarioDocumentLocator,
    ScenarioDocumentRole,
    ScenarioPackageManifest,
    ScenarioPackageSource,
    ScenarioSourceDocument,
    _validate_optional_fallbacks,
)


_MANIFEST_NAME = "scenario-package.json"
_JSON_DOCUMENT_LIMIT = 1024 * 1024
_TILED_MAP_LIMIT = 16 * 1024 * 1024


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("scenario JSON object keys must be unique")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("scenario JSON constants must be finite")


def _resolve_document(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("scenario document path must stay under the package root")
    try:
        resolved = (root / relative).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("scenario document file is missing") from error
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("scenario document path must stay under the package root")
    return resolved


def _read_json(path: Path, *, size_limit: int, label: str) -> Mapping[str, object]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"scenario {label} file is missing") from error
    if len(data) > size_limit:
        raise ValueError(f"scenario {label} file exceeds the permitted size")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"scenario {label} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"scenario {label} must contain valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"scenario {label} JSON root must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"scenario {label} has an unsupported shape")


def _load_manifest(root: Path) -> ScenarioPackageManifest:
    try:
        manifest_path = (root / _MANIFEST_NAME).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("scenario package manifest file is missing") from error
    if not manifest_path.is_relative_to(root.resolve(strict=True)):
        raise ValueError("scenario package manifest path must stay under the package root")
    value = _read_json(manifest_path, size_limit=_JSON_DOCUMENT_LIMIT, label="package manifest")
    _require_exact_keys(value, {"schema", "scenario_id", "version", "documents"}, label="package manifest")
    if value["schema"] != SCENARIO_PACKAGE_SCHEMA:
        raise ValueError("scenario package manifest schema is not supported")
    if not isinstance(value["documents"], list):
        raise ValueError("scenario package manifest documents must be an array")
    locators: list[ScenarioDocumentLocator] = []
    for locator in value["documents"]:
        if not isinstance(locator, Mapping):
            raise ValueError("scenario package manifest document locator must be an object")
        _require_exact_keys(locator, {"role", "logical_id", "relative_path", "expected_hash"}, label="package manifest document locator")
        locators.append(ScenarioDocumentLocator(**locator))
    return ScenarioPackageManifest(
        scenario_id=value["scenario_id"],
        version=value["version"],
        documents=tuple(locators),
        schema=value["schema"],
    )


def _load_document(root: Path, locator: ScenarioDocumentLocator) -> ScenarioSourceDocument:
    path = _resolve_document(root, locator.relative_path)
    limit = _TILED_MAP_LIMIT if locator.role is ScenarioDocumentRole.PHYSICAL_MAP else _JSON_DOCUMENT_LIMIT
    value = _read_json(path, size_limit=limit, label="document")
    _require_exact_keys(value, {"schema", "value"}, label="document")
    if value["schema"] != SCENARIO_DOCUMENT_SCHEMA:
        raise ValueError("scenario document schema is not supported")
    if not isinstance(value["value"], Mapping):
        raise ValueError("scenario document value must be an object")
    actual_hash = stable_content_hash(value)
    if actual_hash != locator.expected_hash:
        raise ValueError("scenario document hash does not match the manifest")
    return ScenarioSourceDocument(
        role=locator.role,
        logical_id=locator.logical_id,
        schema=value["schema"],
        value=value["value"],
    )


def load_situated_scenario_package(root: Path) -> ScenarioPackageSource:
    """Load declared source JSON only; never invoke or fetch package content."""

    package_root = Path(root)
    try:
        package_root = package_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("scenario package root is missing") from error
    if not package_root.is_dir():
        raise ValueError("scenario package root must be a directory")
    manifest = _load_manifest(package_root)
    documents = tuple(_load_document(package_root, locator) for locator in manifest.documents)
    return ScenarioPackageSource(
        scenario_id=manifest.scenario_id,
        version=manifest.version,
        documents=documents,
        manifest_hash=manifest.content_hash,
    )
