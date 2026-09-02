"""Orchestration for the sole editable scenario-draft authority."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import ctypes
import errno
import os
import re
import secrets
import shutil
import sys
from typing import cast

from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.scenario_package_contracts import (
    SCENARIO_DOCUMENT_SCHEMA,
    SCENARIO_PACKAGE_SCHEMA,
    ScenarioDocumentRole,
    ScenarioPackageSource,
    ScenarioSourceDocument,
)
from narrative_dynamics.studio.contracts import (
    DraftOperationKind,
    JsonValue,
    ScenarioDiagnostic,
    ScenarioDiagnosticReport,
    ScenarioDiagnosticSeverity,
    ScenarioDraftDocument,
    ScenarioDraftOperation,
    ScenarioDraftOperationResult,
    ScenarioDraftSnapshot,
    ScenarioProjectExport,
    ScenarioWorkspaceLimits,
    canonical_json_bytes,
    canonical_json_hash,
    scenario_document_raw_value,
    thaw_json,
)
from narrative_dynamics.studio.diagnostics import (
    compile_with_report,
    manifest_missing_report,
)
from narrative_dynamics.studio.project_store import (
    SQLiteScenarioProjectStore,
    ScenarioProjectConflictError,
    ScenarioProjectValidationError,
)


_EXPORT_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SINGLETON_EXPORT_PATHS = {
    "physical.world": "physical/world.json",
    "physical.perception": "physical/perception.json",
    "physical.initial_state": "physical/initial-state.json",
    "physical.map": "physical/map.tmj",
    "social.institutions": "social/institutions.json",
    "social.relationships": "social/relationships.json",
    "social.norms": "social/norms.json",
    "story.outline": "story/outline.json",
    "story.interventions": "story/interventions.json",
    "knowledge.catalog": "knowledge/catalog.json",
    "knowledge.access": "knowledge/access.json",
    "asset.catalog": "assets/catalog.json",
    "run": "run.json",
}


def _publish_stage(stage: Path, target: Path) -> None:
    """Private publication seam: atomically rename without replacing a target."""

    if stage.parent != target.parent:
        raise ValueError("scenario publication requires a same-parent stage")
    if sys.platform == "win32":
        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file.restype = ctypes.c_int
        if move_file(str(stage), str(target), 0):
            return
        error = ctypes.get_last_error()
        if error in {80, 183}:
            raise FileExistsError(error, "scenario export target exists")
        raise OSError(error, "scenario package publication failed")
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename_at2 = getattr(libc, "renameat2", None)
        if rename_at2 is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
        rename_at2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_at2.restype = ctypes.c_int
        if rename_at2(-100, os.fsencode(stage), -100, os.fsencode(target), 1) == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, "scenario export target exists")
        raise OSError(error, "scenario package publication failed")
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is None:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
        rename_exclusive.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_exclusive.restype = ctypes.c_int
        if rename_exclusive(os.fsencode(stage), os.fsencode(target), 4) == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, "scenario export target exists")
        raise OSError(error, "scenario package publication failed")
    raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")


def _write_fsynced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _discard_stage(stage: Path) -> None:
    try:
        if stage.exists():
            shutil.rmtree(stage)
    except OSError:
        pass


def _export_document_path(document: ScenarioDraftDocument) -> str:
    if document.role == ScenarioDocumentRole.AGENT.value:
        assert document.logical_id is not None
        return f"agents/{document.logical_id}.json"
    return _SINGLETON_EXPORT_PATHS[document.role]


def _canonical_manifest(
    scenario_id: str,
    version: str,
    documents: tuple[ScenarioDraftDocument, ...],
) -> dict[str, object]:
    return {
        "schema": SCENARIO_PACKAGE_SCHEMA,
        "scenario_id": scenario_id,
        "version": version,
        "documents": [
            {
                "role": document.role,
                "path": _export_document_path(document),
                "sha256": document.content_hash,
            }
            for document in documents
        ],
    }


def _export_interleave(snapshot: ScenarioDraftSnapshot) -> None:
    """Private deterministic seam for export concurrency regression coverage."""


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(item.is_symlink() for item in (absolute, *absolute.parents))


def _draft_document(source: ScenarioSourceDocument) -> ScenarioDraftDocument:
    role = source.role.value
    logical_id = source.logical_id if source.role is ScenarioDocumentRole.AGENT else None
    value = thaw_json(source.value)  # type: ignore[arg-type]
    return ScenarioDraftDocument(
        role,
        logical_id,
        value,  # type: ignore[arg-type]
        canonical_json_hash(scenario_document_raw_value(role, value)),
    )


def _source_from_snapshot(snapshot: ScenarioDraftSnapshot) -> ScenarioPackageSource:
    if snapshot.scenario_id is None or snapshot.version is None or not snapshot.documents:
        raise ScenarioProjectValidationError(
            "scenario project cannot compile",
            manifest_missing_report(snapshot.project_id, snapshot.revision),
        )
    documents = tuple(
        ScenarioSourceDocument(
            ScenarioDocumentRole(document.role),
            document.logical_id or document.role,
            SCENARIO_DOCUMENT_SCHEMA,
            cast(Mapping[str, object], document.value),
            document.content_hash,
        )
        for document in snapshot.documents
    )
    raw_manifest_hash = canonical_json_hash(
        _canonical_manifest(
            snapshot.scenario_id,
            snapshot.version,
            snapshot.documents,
        )
    )
    return ScenarioPackageSource(
        snapshot.scenario_id,
        snapshot.version,
        documents,
        raw_manifest_hash,
    )


def _decode_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def _array_index(token: str, length: int, *, insertion: bool) -> int:
    if token == "-" and insertion:
        return length
    if not token or (len(token) > 1 and token.startswith("0")) or not token.isascii() or not token.isdigit():
        raise ScenarioProjectValidationError("draft operation JSON pointer is invalid")
    index = int(token)
    maximum = length if insertion else length - 1
    if index > maximum:
        raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
    return index


def _parent(root: object, tokens: list[str]) -> tuple[object, str]:
    if not tokens:
        raise ScenarioProjectValidationError("draft operation cannot address the root")
    current = root
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[_array_index(token, len(current), insertion=False)]
        elif isinstance(current, dict):
            if token not in current:
                raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
            current = current[token]
        else:
            raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
    return current, tokens[-1]


def _set_value(
    root: object,
    pointer: str,
    value: object,
    *,
    insert: bool,
    upsert: bool = False,
) -> object:
    tokens = _decode_pointer(pointer)
    if not tokens:
        if insert:
            raise ScenarioProjectValidationError("draft insert cannot replace the root")
        return value
    parent, token = _parent(root, tokens)
    if isinstance(parent, list):
        index = _array_index(token, len(parent), insertion=insert)
        if insert:
            parent.insert(index, value)
        else:
            parent[index] = value
    elif isinstance(parent, dict):
        if insert:
            if token in parent:
                raise ScenarioProjectConflictError("draft insert target already exists")
            parent[token] = value
        else:
            if token not in parent and not upsert:
                raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
            parent[token] = value
    else:
        raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
    return root


def _remove_value(root: object, pointer: str) -> tuple[object, object]:
    tokens = _decode_pointer(pointer)
    parent, token = _parent(root, tokens)
    if isinstance(parent, list):
        removed = parent.pop(_array_index(token, len(parent), insertion=False))
    elif isinstance(parent, dict):
        if token not in parent:
            raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
        removed = parent.pop(token)
    else:
        raise ScenarioProjectValidationError("draft operation JSON pointer is unknown")
    return root, removed


def _move_array_value(root: object, from_pointer: str, pointer: str) -> object:
    source_parent, source_token = _parent(root, _decode_pointer(from_pointer))
    target_parent, target_token = _parent(root, _decode_pointer(pointer))
    if not isinstance(source_parent, list) or not isinstance(target_parent, list):
        raise ScenarioProjectValidationError(
            "draft move source and target must be array positions"
        )
    source_index = _array_index(source_token, len(source_parent), insertion=False)
    target_length = len(target_parent) - (1 if source_parent is target_parent else 0)
    target_index = _array_index(target_token, target_length, insertion=True)
    moved = source_parent.pop(source_index)
    target_parent.insert(target_index, moved)
    return root


class ScenarioProjectWorkspace:
    def __init__(
        self,
        store: SQLiteScenarioProjectStore,
        *,
        import_roots: tuple[Path, ...],
        export_root: Path | None,
        limits: ScenarioWorkspaceLimits,
    ) -> None:
        self._store = store
        self._import_roots = import_roots
        self._export_root = export_root
        self._limits = limits

    @classmethod
    def open(
        cls,
        database_path: str | Path,
        *,
        import_roots: tuple[str | Path, ...] = (),
        export_root: str | Path | None = None,
        limits: ScenarioWorkspaceLimits | None = None,
    ) -> "ScenarioProjectWorkspace":
        effective_limits = limits or ScenarioWorkspaceLimits()
        roots: list[Path] = []
        for root in import_roots:
            try:
                resolved = Path(root).resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError("scenario import root is unavailable") from None
            if not resolved.is_dir():
                raise ValueError("scenario import root must be a directory")
            roots.append(resolved)
        resolved_export = None
        if export_root is not None:
            lexical_export = Path(export_root)
            if _has_symlink_component(lexical_export):
                raise ValueError("scenario export root must not be a symlink")
            try:
                resolved_export = lexical_export.resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError("scenario export root is unavailable") from None
            if not resolved_export.is_dir() or resolved_export.is_symlink():
                raise ValueError("scenario export root must be a regular directory")
        return cls(
            SQLiteScenarioProjectStore(database_path, effective_limits),
            import_roots=tuple(roots),
            export_root=resolved_export,
            limits=effective_limits,
        )

    def create_project(self, project_id: str) -> ScenarioDraftSnapshot:
        report = manifest_missing_report(project_id, 1)
        snapshot = ScenarioDraftSnapshot(
            project_id,
            1,
            (),
            {},
            report.content_hash,
            None,
        )
        self._store.create(project_id, snapshot)
        return snapshot

    def _allowed_import(self, source_root: str | Path) -> Path:
        if not self._import_roots:
            raise ScenarioProjectValidationError("scenario package import is disabled")
        try:
            source = Path(source_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ScenarioProjectValidationError("scenario package import failed") from None
        if not any(source == root or source.is_relative_to(root) for root in self._import_roots):
            raise ScenarioProjectValidationError("scenario package import is outside its root")
        return source

    def import_package(
        self,
        project_id: str,
        source_root: str | Path,
        *,
        expected_revision: int,
        expected_snapshot_hash: str,
    ) -> ScenarioDraftSnapshot:
        current = self._store.load(project_id)
        if (
            current.revision != expected_revision
            or current.content_hash != expected_snapshot_hash
        ):
            raise ScenarioProjectConflictError("scenario project revision is stale")
        source_path = self._allowed_import(source_root)
        try:
            source = load_situated_scenario_package(source_path)
        except ValueError:
            raise ScenarioProjectValidationError("scenario package import failed") from None
        revision = current.revision + 1
        canonical_documents = tuple(_draft_document(item) for item in source.documents)
        for document in canonical_documents:
            size_limit = (
                self._limits.tiled_map_bytes
                if document.role == ScenarioDocumentRole.PHYSICAL_MAP.value
                else self._limits.json_document_bytes
            )
            if (
                len(
                    canonical_json_bytes(
                        scenario_document_raw_value(document.role, document.value)
                    )
                )
                > size_limit
            ):
                raise ScenarioProjectValidationError(
                    "scenario draft document exceeds the permitted size"
                )
        canonical_source = ScenarioPackageSource(
            source.scenario_id,
            source.version,
            tuple(
                ScenarioSourceDocument(
                    item.role,
                    item.logical_id,
                    item.schema,
                    item.value,
                    canonical_documents[index].content_hash,
                )
                for index, item in enumerate(source.documents)
            ),
            canonical_json_hash(
                _canonical_manifest(
                    source.scenario_id,
                    source.version,
                    canonical_documents,
                )
            ),
        )
        compiled, report = compile_with_report(project_id, revision, canonical_source)
        snapshot = ScenarioDraftSnapshot(
            project_id,
            revision,
            canonical_documents,
            current.layout,
            report.content_hash,
            None if compiled is None else compiled.content_hash,
            scenario_id=source.scenario_id,
            version=source.version,
        )
        return self._store.import_snapshot(
            snapshot,
            report,
            expected_revision,
            expected_snapshot_hash,
        )

    def snapshot(self, project_id: str) -> ScenarioDraftSnapshot:
        return self._store.load(project_id)

    @staticmethod
    def _document_index(
        snapshot: ScenarioDraftSnapshot, operation: ScenarioDraftOperation
    ) -> int:
        for index, document in enumerate(snapshot.documents):
            if (
                document.role == operation.document_role
                and document.logical_id == operation.logical_id
            ):
                return index
        raise ScenarioProjectValidationError("scenario draft document is unknown")

    def apply(self, operation: ScenarioDraftOperation) -> ScenarioDraftOperationResult:
        if not isinstance(operation, ScenarioDraftOperation):
            raise TypeError("scenario draft apply requires ScenarioDraftOperation")
        existing = self._store.operation_result(
            operation.project_id, operation.idempotency_key
        )
        if existing is not None:
            if existing.operation_hash == operation.content_hash:
                return existing
            raise ScenarioProjectConflictError(
                "scenario project idempotency key conflicts"
            )
        operation_collision = self._store.operation_by_id(
            operation.project_id, operation.operation_id
        )
        if operation_collision is not None:
            raise ScenarioProjectConflictError("scenario project operation ID conflicts")
        current = self._store.load(operation.project_id)
        if (
            current.revision != operation.expected_revision
            or current.content_hash != operation.expected_snapshot_hash
        ):
            raise ScenarioProjectConflictError("scenario project revision is stale")
        if (
            len(
                canonical_json_bytes(
                    {
                        "pointer": operation.pointer,
                        "from_pointer": operation.from_pointer,
                        "value": thaw_json(operation.value),
                    }
                )
            )
            > self._limits.operation_payload_bytes
        ):
            raise ScenarioProjectValidationError(
                "scenario draft operation payload exceeds the permitted size"
            )
        revision = current.revision + 1
        documents = list(current.documents)
        layout = thaw_json(current.layout)
        if operation.kind is DraftOperationKind.SET_LAYOUT:
            layout = _set_value(
                layout,
                operation.pointer,
                thaw_json(operation.value),
                insert=False,
                upsert=True,
            )
            report = ScenarioDiagnosticReport(
                current.project_id,
                revision,
                self._store.diagnostic_report(current.project_id).diagnostics,
            )
            compiled_hash = current.compiled_scenario_hash
        else:
            document_index = self._document_index(current, operation)
            document = documents[document_index]
            value = thaw_json(document.value)
            if operation.kind is DraftOperationKind.REPLACE_DOCUMENT:
                value = thaw_json(operation.value)
            elif operation.kind is DraftOperationKind.SET_VALUE:
                value = _set_value(
                    value, operation.pointer, thaw_json(operation.value), insert=False
                )
            elif operation.kind is DraftOperationKind.INSERT_VALUE:
                value = _set_value(
                    value, operation.pointer, thaw_json(operation.value), insert=True
                )
            elif operation.kind is DraftOperationKind.REMOVE_VALUE:
                value, _ = _remove_value(value, operation.pointer)
            elif operation.kind is DraftOperationKind.MOVE_VALUE:
                assert operation.from_pointer is not None
                value = _move_array_value(
                    value, operation.from_pointer, operation.pointer
                )
            if not isinstance(value, dict):
                raise ScenarioProjectValidationError(
                    "scenario draft document root must be an object"
                )
            raw_value = scenario_document_raw_value(document.role, value)
            size_limit = (
                self._limits.tiled_map_bytes
                if document.role == ScenarioDocumentRole.PHYSICAL_MAP.value
                else self._limits.json_document_bytes
            )
            if len(canonical_json_bytes(raw_value)) > size_limit:
                raise ScenarioProjectValidationError(
                    "scenario draft document exceeds the permitted size"
                )
            documents[document_index] = ScenarioDraftDocument(
                document.role,
                document.logical_id,
                value,
                canonical_json_hash(raw_value),
            )
            provisional = ScenarioDraftSnapshot(
                current.project_id,
                revision,
                tuple(documents),
                layout,  # type: ignore[arg-type]
                ScenarioDiagnosticReport(current.project_id, revision).content_hash,
                None,
                scenario_id=current.scenario_id,
                version=current.version,
            )
            try:
                source = _source_from_snapshot(provisional)
            except ValueError:
                report = ScenarioDiagnosticReport(
                    current.project_id,
                    revision,
                    (
                        ScenarioDiagnostic(
                            ScenarioDiagnosticSeverity.ERROR,
                            "contract_violation",
                            operation.document_role,
                            operation.logical_id,
                            operation.pointer,
                        ),
                    ),
                )
                compiled = None
            else:
                compiled, report = compile_with_report(
                    current.project_id, revision, source
                )
            compiled_hash = None if compiled is None else compiled.content_hash
        next_snapshot = ScenarioDraftSnapshot(
            current.project_id,
            revision,
            tuple(documents),
            layout,  # type: ignore[arg-type]
            report.content_hash,
            compiled_hash,
            scenario_id=current.scenario_id,
            version=current.version,
        )
        return self._store.apply(operation, next_snapshot, report)

    def undo(
        self,
        project_id: str,
        expected_revision: int,
        expected_snapshot_hash: str,
    ) -> ScenarioDraftSnapshot:
        return self._store.undo(
            project_id, expected_revision, expected_snapshot_hash
        )

    def redo(
        self,
        project_id: str,
        expected_revision: int,
        expected_snapshot_hash: str,
    ) -> ScenarioDraftSnapshot:
        return self._store.redo(
            project_id, expected_revision, expected_snapshot_hash
        )

    def validate(self, project_id: str) -> ScenarioDiagnosticReport:
        return self._store.diagnostic_report(project_id)

    def compile(self, project_id: str):
        snapshot = self._store.load(project_id)
        report = self._store.diagnostic_report(project_id)
        if report.has_errors:
            raise ScenarioProjectValidationError(
                "scenario project cannot compile", report
            )
        return self._compile_snapshot(snapshot)

    @staticmethod
    def _compile_snapshot(snapshot: ScenarioDraftSnapshot):
        source = _source_from_snapshot(snapshot)
        compiled, report = compile_with_report(
            snapshot.project_id,
            snapshot.revision,
            source,
        )
        if compiled is None:
            raise ScenarioProjectValidationError(
                "scenario project cannot compile", report
            )
        if snapshot.compiled_scenario_hash != compiled.content_hash:
            raise ScenarioProjectValidationError("scenario project compile identity failed")
        return compiled

    def export(self, project_id: str, target_name: str) -> ScenarioProjectExport:
        if self._export_root is None:
            raise ScenarioProjectValidationError("scenario package export is disabled")
        if (
            not isinstance(target_name, str)
            or _EXPORT_TARGET.fullmatch(target_name) is None
            or len(target_name) > 128
            or target_name in {".", ".."}
            or ":" in target_name
        ):
            raise ScenarioProjectValidationError(
                "scenario export target must be a path-safe single name"
            )
        export_root = self._export_root
        if export_root.is_symlink() or not export_root.is_dir():
            raise ScenarioProjectValidationError("scenario package publication failed")
        snapshot = self._store.load(project_id)
        _export_interleave(snapshot)
        compiled = self._compile_snapshot(snapshot)
        published_name = compiled.package_hash.removeprefix("sha256:")
        target = export_root / published_name
        if target.exists() or target.is_symlink():
            raise ScenarioProjectConflictError("scenario export target already exists")
        stage = export_root / f".{target_name}.stage-{secrets.token_hex(12)}"
        try:
            stage.mkdir()
            for document in snapshot.documents:
                relative_path = _export_document_path(document)
                raw_value = scenario_document_raw_value(document.role, document.value)
                data = canonical_json_bytes(raw_value)
                raw_hash = canonical_json_hash(raw_value)
                if raw_hash != document.content_hash:
                    raise ValueError
                _write_fsynced(stage / Path(relative_path), data)
            if snapshot.scenario_id is None or snapshot.version is None:
                raise ValueError
            manifest = _canonical_manifest(
                snapshot.scenario_id,
                snapshot.version,
                snapshot.documents,
            )
            _write_fsynced(stage / "scenario.json", canonical_json_bytes(manifest))
            _publish_stage(stage, target)
        except FileExistsError:
            _discard_stage(stage)
            raise ScenarioProjectConflictError(
                "scenario export target already exists"
            ) from None
        except (OSError, KeyError, TypeError, ValueError):
            _discard_stage(stage)
            raise ScenarioProjectValidationError(
                "scenario package publication failed"
            ) from None
        return ScenarioProjectExport(
            project_id,
            snapshot.revision,
            snapshot.content_hash,
            published_name,
            compiled.package_hash,
            compiled.content_hash,
        )


__all__ = ("ScenarioProjectWorkspace",)
