"""Strict, inert loader for declarative situated scenario packages."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

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


_MANIFEST_NAME = "scenario.json"
_JSON_DOCUMENT_LIMIT = 1024 * 1024
_TILED_MAP_LIMIT = 16 * 1024 * 1024


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _NonFiniteNumberError


def _parse_finite_float(value: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise _NonFiniteNumberError from None
    if not math.isfinite(result):
        raise _NonFiniteNumberError
    return result


def _raw_sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _resolve_document(root: Path, document_path: str) -> Path:
    relative = Path(document_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("scenario document path must stay under the package root")
    resolved: Path | None = None
    try:
        resolved = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError):
        pass
    if resolved is None:
        raise ValueError("scenario document file is unavailable")
    if not resolved.is_relative_to(root):
        raise ValueError("scenario document path must stay under the package root")
    if not resolved.is_file():
        raise ValueError("scenario document file must be a regular file")
    return resolved


def _read_json(
    path: Path,
    *,
    size_limit: int,
    label: str,
) -> tuple[Mapping[str, object], str]:
    data: bytes | None = None
    try:
        data = path.read_bytes()
    except OSError:
        pass
    if data is None:
        raise ValueError(f"scenario {label} file is unavailable")
    if len(data) > size_limit:
        raise ValueError(f"scenario {label} file exceeds the permitted size")
    text: str | None = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise ValueError(f"scenario {label} must be UTF-8")
    value: object | None = None
    failure = ""
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except _DuplicateKeyError:
        failure = "keys"
    except _NonFiniteNumberError:
        failure = "numbers"
    except (json.JSONDecodeError, RecursionError, OverflowError, ValueError):
        failure = "syntax"
    if failure == "keys":
        raise ValueError("scenario JSON object keys must be unique")
    if failure == "numbers":
        raise ValueError("scenario JSON numbers must be finite")
    if failure:
        raise ValueError(f"scenario {label} must contain valid bounded JSON")
    if not isinstance(value, Mapping):
        raise ValueError(f"scenario {label} JSON root must be an object")
    return value, _raw_sha256(data)


def _require_exact_keys(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"scenario {label} has an unsupported shape")


def _load_manifest(root: Path) -> tuple[ScenarioPackageManifest, str]:
    manifest_path: Path | None = None
    try:
        manifest_path = (root / _MANIFEST_NAME).resolve(strict=True)
    except (OSError, RuntimeError):
        pass
    if manifest_path is None:
        raise ValueError("scenario package manifest file is unavailable")
    if not manifest_path.is_relative_to(root):
        raise ValueError("scenario package manifest path must stay under the package root")
    if not manifest_path.is_file():
        raise ValueError("scenario package manifest must be a regular file")
    value, raw_hash = _read_json(
        manifest_path,
        size_limit=_JSON_DOCUMENT_LIMIT,
        label="package manifest",
    )
    _require_exact_keys(value, {"schema", "scenario_id", "version", "documents"}, label="package manifest")
    if value["schema"] != SCENARIO_PACKAGE_SCHEMA:
        raise ValueError("scenario package manifest schema is not supported")
    if not isinstance(value["documents"], list):
        raise ValueError("scenario package manifest documents must be an array")
    locators: list[ScenarioDocumentLocator] = []
    for locator in value["documents"]:
        if not isinstance(locator, Mapping):
            raise ValueError("scenario package manifest document locator must be an object")
        _require_exact_keys(
            locator,
            {"role", "path", "sha256"},
            label="package manifest document locator",
        )
        locators.append(ScenarioDocumentLocator(**locator))
    manifest = ScenarioPackageManifest(
        scenario_id=value["scenario_id"],
        version=value["version"],
        documents=tuple(locators),
        schema=value["schema"],
    )
    run_locator = next(
        locator for locator in manifest.documents
        if locator.role is ScenarioDocumentRole.RUN
    )
    if run_locator.path != "run.json":
        raise ValueError("scenario run policy must use the canonical run.json path")
    map_locator = next(
        (
            locator for locator in manifest.documents
            if locator.role is ScenarioDocumentRole.PHYSICAL_MAP
        ),
        None,
    )
    if map_locator is not None and Path(map_locator.path).suffix.casefold() != ".tmj":
        raise ValueError("scenario physical map must be a Tiled .tmj document")
    return manifest, raw_hash


def _load_document(root: Path, locator: ScenarioDocumentLocator) -> ScenarioSourceDocument:
    path = _resolve_document(root, locator.path)
    limit = _TILED_MAP_LIMIT if locator.role is ScenarioDocumentRole.PHYSICAL_MAP else _JSON_DOCUMENT_LIMIT
    value, raw_hash = _read_json(path, size_limit=limit, label="document")
    if locator.role is ScenarioDocumentRole.PHYSICAL_MAP:
        document_value = value
    else:
        _require_exact_keys(value, {"schema", "value"}, label="document")
        if value["schema"] != SCENARIO_DOCUMENT_SCHEMA:
            raise ValueError("scenario document schema is not supported")
        if not isinstance(value["value"], Mapping):
            raise ValueError("scenario document value must be an object")
        document_value = value["value"]
    if raw_hash != locator.sha256:
        raise ValueError("scenario document hash does not match the manifest")
    logical_id = locator.role.value
    if locator.role is ScenarioDocumentRole.AGENT:
        candidate = document_value.get("agent_id")
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("scenario agent document must declare an agent ID")
        logical_id = candidate
    return ScenarioSourceDocument(
        role=locator.role,
        logical_id=logical_id,
        schema=SCENARIO_DOCUMENT_SCHEMA,
        value=document_value,
        raw_content_hash=raw_hash,
    )


def load_situated_scenario_package(root: Path) -> ScenarioPackageSource:
    """Load declared source JSON only; never invoke or fetch package content."""

    package_root = Path(root)
    resolved_root: Path | None = None
    try:
        resolved_root = package_root.resolve(strict=True)
    except (OSError, RuntimeError):
        pass
    if resolved_root is None:
        raise ValueError("scenario package root is unavailable")
    package_root = resolved_root
    if not package_root.is_dir():
        raise ValueError("scenario package root must be a directory")
    manifest, raw_manifest_hash = _load_manifest(package_root)
    resolved_documents = tuple(
        _resolve_document(package_root, locator.path)
        for locator in manifest.documents
    )
    file_identities: set[tuple[int, int]] = set()
    for path in resolved_documents:
        identity: tuple[int, int] | None = None
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
        except OSError:
            pass
        if identity is None:
            raise ValueError("scenario document file is unavailable")
        if identity in file_identities:
            raise ValueError("scenario document locators must not resolve to the same file")
        file_identities.add(identity)
    documents = tuple(
        _load_document(package_root, locator)
        for locator in manifest.documents
    )
    return ScenarioPackageSource(
        scenario_id=manifest.scenario_id,
        version=manifest.version,
        documents=documents,
        raw_manifest_hash=raw_manifest_hash,
    )
