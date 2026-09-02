"""Explicit local composition and Hypercorn launcher for World Studio V22."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from threading import RLock

from narrative_dynamics.abm.scenario_checkpoint_store import (
    LocalScenarioCheckpointStore,
    _PhysicalFileOwnershipToken,
)
from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.situated_percept_memory import (
    initialize_situated_percept_memory,
)
from narrative_dynamics.abm.scenario_coordinator_contracts import (
    ScenarioCommandCapability,
    ScenarioForkRequest,
)
from narrative_dynamics.abm.simulation_output_bus import SimulationDeliveryReport
from narrative_dynamics.studio import (
    STUDIO_PERMISSIONS,
    InMemoryScenarioRunRegistry,
    JsonRpcDispatcher,
    ScenarioWorkspaceLimits,
    ScenarioProjectWorkspace,
    StudioCapability,
    WorldStudioService,
)
from narrative_dynamics.studio.streaming import StudioOutputLimits, StudioOutputRouter
from narrative_dynamics.integrations.world_studio_server import (
    WorldStudioServerLimits,
    create_world_studio_asgi_app,
)


def _existing_directory(value: Path, *, label: str) -> Path:
    try:
        resolved = Path(value).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"World Studio {label} is unavailable") from None
    if not resolved.is_dir():
        raise ValueError(f"World Studio {label} must be a directory")
    return resolved


def _existing_file(value: Path, *, label: str) -> Path:
    try:
        resolved = Path(value).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"World Studio {label} is unavailable") from None
    if not resolved.is_file():
        raise ValueError(f"World Studio {label} must be a file")
    return resolved


@dataclass(frozen=True)
class LauncherSettings:
    workspace_root: Path
    import_roots: tuple[Path, ...]
    import_sources: tuple[tuple[str, Path], ...]
    export_root: Path
    static_root: Path
    bind_host: str
    bind_port: int
    allowed_origins: tuple[str, ...]
    development_trust_all: bool
    auth_token: str | None
    tls_certificate: Path | None
    tls_private_key: Path | None
    authority_id: str
    project_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]
    permissions: tuple[str, ...]
    workspace_limits: ScenarioWorkspaceLimits = field(default_factory=ScenarioWorkspaceLimits)
    output_limits: StudioOutputLimits = field(default_factory=StudioOutputLimits)
    server_limits: WorldStudioServerLimits = field(default_factory=WorldStudioServerLimits)

    def __post_init__(self) -> None:
        workspace = _existing_directory(self.workspace_root, label="workspace root")
        export = _existing_directory(self.export_root, label="export root")
        static = _existing_directory(self.static_root, label="static root")
        if not (static / "index.html").is_file():
            raise ValueError("World Studio static root must contain index.html")
        roots = tuple(_existing_directory(root, label="import root") for root in self.import_roots)
        if not roots:
            raise ValueError("World Studio requires at least one explicit import root")
        sources: list[tuple[str, Path]] = []
        source_ids: set[str] = set()
        for source_id, source_path in self.import_sources:
            if source_id in source_ids:
                raise ValueError("World Studio import source IDs must be unique")
            StudioCapability("source-validator", project_ids=(source_id,))
            source = _existing_directory(source_path, label="import source")
            if not any(source == root or source.is_relative_to(root) for root in roots):
                raise ValueError("World Studio import source is outside its explicit roots")
            source_ids.add(source_id)
            sources.append((source_id, source))
        try:
            address = ipaddress.ip_address(self.bind_host)
        except ValueError:
            raise ValueError("World Studio bind host must be an explicit IP address") from None
        if not isinstance(self.bind_port, int) or isinstance(self.bind_port, bool) or not 1 <= self.bind_port <= 65535:
            raise ValueError("World Studio bind port must be between 1 and 65535")
        if not isinstance(self.development_trust_all, bool):
            raise TypeError("World Studio development trust-all must be boolean")
        if self.development_trust_all and not address.is_loopback:
            raise ValueError("World Studio development trust-all requires loopback binding")
        if self.development_trust_all and self.auth_token is not None:
            raise ValueError("World Studio development trust-all cannot accept an auth token")
        if not self.development_trust_all and (
            not isinstance(self.auth_token, str) or not self.auth_token or len(self.auth_token) > 4096
        ):
            raise ValueError("World Studio explicit authentication token is required")
        if (self.tls_certificate is None) != (self.tls_private_key is None):
            raise ValueError("World Studio TLS certificate and private key must be configured together")
        certificate = None
        private_key = None
        if self.tls_certificate is not None and self.tls_private_key is not None:
            certificate = _existing_file(self.tls_certificate, label="TLS certificate")
            private_key = _existing_file(self.tls_private_key, label="TLS private key")
        if not address.is_loopback and (
            self.development_trust_all or self.auth_token is None or certificate is None
        ):
            raise ValueError("World Studio non-loopback binding requires explicit authentication and TLS")
        if not isinstance(self.workspace_limits, ScenarioWorkspaceLimits):
            raise TypeError("World Studio workspace limits are invalid")
        if not isinstance(self.output_limits, StudioOutputLimits):
            raise TypeError("World Studio output limits are invalid")
        if not isinstance(self.server_limits, WorldStudioServerLimits):
            raise TypeError("World Studio server limits are invalid")
        capability = StudioCapability(
            self.authority_id,
            project_ids=self.project_ids,
            run_ids=self.run_ids,
            agent_ids=self.agent_ids,
            permissions=self.permissions,
        )
        object.__setattr__(self, "workspace_root", workspace)
        object.__setattr__(self, "export_root", export)
        object.__setattr__(self, "static_root", static)
        object.__setattr__(self, "import_roots", roots)
        object.__setattr__(self, "import_sources", tuple(sources))
        object.__setattr__(self, "tls_certificate", certificate)
        object.__setattr__(self, "tls_private_key", private_key)
        object.__setattr__(self, "project_ids", capability.project_ids)
        object.__setattr__(self, "run_ids", capability.run_ids)
        object.__setattr__(self, "agent_ids", capability.agent_ids)
        object.__setattr__(self, "permissions", capability.permissions)


class _RouterPublisher:
    def __init__(self, router: StudioOutputRouter, run_id: str) -> None:
        self._router = router
        self._run_id = run_id

    def publish(self, batch) -> SimulationDeliveryReport:
        delivered = self._router.publish(self._run_id, batch)
        return SimulationDeliveryReport(batch.content_hash, delivered, ())


@dataclass(frozen=True)
class _OwnedArtifact:
    path: Path
    device: int
    inode: int
    is_directory: bool


@dataclass
class _RunArtifactClaim:
    run_id: str
    marker: _OwnedArtifact
    database: _OwnedArtifact | None = None
    checkpoints: _OwnedArtifact | None = None


class _LocalCoordinatorFactory:
    _METADATA_SCHEMA = "narrative-dynamics.local-run-metadata/v1"

    def __init__(self, root: Path, router: StudioOutputRouter) -> None:
        self._root = root.resolve(strict=True)
        self._router = router
        self._lock = RLock()
        self._owned: dict[str, ScenarioCoordinator] = {}
        self._claims: dict[str, _RunArtifactClaim] = {}
        self._metadata_path = self._root / "run-metadata.json"
        self._persisted_run_ids = self._load_run_ids()

    @property
    def unavailable_run_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._persisted_run_ids))

    def _database(self, run_id: str) -> Path:
        return self._root / f"{run_id}.sqlite3"

    def _checkpoints(self, run_id: str) -> Path:
        return self._root / f"{run_id}-checkpoints"

    def _marker(self, run_id: str) -> Path:
        return self._root / f".{run_id}.owner"

    @staticmethod
    def _run_id(run_id: str) -> str:
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        filename_base = run_id.split(".", 1)[0].upper() if isinstance(run_id, str) else ""
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in {".", ".."}
            or Path(run_id).name != run_id
            or "/" in run_id
            or "\\" in run_id
            or run_id[-1] in {".", " "}
            or filename_base in reserved
            or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in run_id)
        ):
            raise ValueError("local coordinator run ID must be a filename-safe identity")
        return run_id

    @staticmethod
    def _artifact(path: Path, *, directory: bool) -> _OwnedArtifact:
        details = path.stat(follow_symlinks=False)
        if directory:
            valid = stat.S_ISDIR(details.st_mode)
        else:
            valid = stat.S_ISREG(details.st_mode)
        if not valid:
            raise RuntimeError("local coordinator artifact type changed")
        return _OwnedArtifact(path, details.st_dev, details.st_ino, directory)

    @staticmethod
    def _matches(artifact: _OwnedArtifact) -> bool:
        try:
            current = artifact.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        expected_type = stat.S_ISDIR if artifact.is_directory else stat.S_ISREG
        return (
            expected_type(current.st_mode)
            and current.st_dev == artifact.device
            and current.st_ino == artifact.inode
        )

    def _load_run_ids(self) -> set[str]:
        persisted: set[str] = set()
        if self._metadata_path.exists():
            try:
                payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise ValueError("World Studio run metadata is unavailable") from None
            if (
                not isinstance(payload, dict)
                or set(payload) != {"schema", "run_ids"}
                or payload.get("schema") != self._METADATA_SCHEMA
                or not isinstance(payload.get("run_ids"), list)
                or any(not isinstance(item, str) for item in payload["run_ids"])
            ):
                raise ValueError("World Studio run metadata is invalid")
            persisted.update(self._run_id(item) for item in payload["run_ids"])
        discovered: set[str] = set()
        with os.scandir(self._root) as entries:
            for entry in entries:
                name = entry.name
                if name.endswith(".sqlite3"):
                    discovered.add(name.removesuffix(".sqlite3"))
                elif name.endswith("-checkpoints"):
                    discovered.add(name.removesuffix("-checkpoints"))
                elif name.startswith(".") and name.endswith(".owner"):
                    discovered.add(name[1:].removesuffix(".owner"))
        persisted.update(self._run_id(item) for item in discovered)
        if persisted or self._metadata_path.exists():
            self._write_run_ids(persisted)
        return persisted

    def _write_run_ids(self, run_ids: set[str]) -> None:
        payload = json.dumps(
            {"schema": self._METADATA_SCHEMA, "run_ids": sorted(run_ids)},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = self._root / f".run-metadata-{secrets.token_hex(16)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _claim(self, run_id: str, *, checkpoints: bool) -> _RunArtifactClaim:
        exact_run_id = self._run_id(run_id)
        database = self._database(exact_run_id)
        checkpoint_root = self._checkpoints(exact_run_id)
        marker = self._marker(exact_run_id)
        with self._lock:
            if exact_run_id in self._claims:
                raise FileExistsError("local coordinator run is already claimed")
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            claim: _RunArtifactClaim | None = None
            marker_artifact: _OwnedArtifact | None = None
            try:
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(secrets.token_hex(32))
                    handle.flush()
                    os.fsync(handle.fileno())
                marker_artifact = self._artifact(marker, directory=False)
                if os.path.lexists(database) or os.path.lexists(checkpoint_root):
                    if self._matches(marker_artifact):
                        marker.unlink()
                    raise FileExistsError("local coordinator artifacts already exist")
                database_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
                database_flags |= getattr(os, "O_NOFOLLOW", 0)
                database_descriptor = os.open(database, database_flags, 0o600)
                os.close(database_descriptor)
                database_artifact = self._artifact(database, directory=False)
                claim = _RunArtifactClaim(
                    exact_run_id,
                    marker_artifact,
                    database=database_artifact,
                )
                if checkpoints:
                    checkpoint_root.mkdir(mode=0o700)
                    claim.checkpoints = self._artifact(
                        checkpoint_root, directory=True
                    )
                self._claims[exact_run_id] = claim
                self._persisted_run_ids.add(exact_run_id)
                try:
                    self._write_run_ids(self._persisted_run_ids)
                except Exception:
                    self._persisted_run_ids.discard(exact_run_id)
                    self._claims.pop(exact_run_id, None)
                    if self._matches(marker_artifact):
                        marker.unlink()
                    if self._matches(database_artifact):
                        database.unlink()
                    if claim.checkpoints is not None and self._matches(claim.checkpoints):
                        claim.checkpoints.path.rmdir()
                    raise
                return claim
            except Exception:
                if claim is not None and claim.database is not None and self._matches(claim.database):
                    try:
                        claim.database.path.unlink()
                    except OSError:
                        pass
                if claim is not None and claim.checkpoints is not None and self._matches(claim.checkpoints):
                    try:
                        claim.checkpoints.path.rmdir()
                    except OSError:
                        pass
                if (
                    marker_artifact is not None
                    and self._matches(marker_artifact)
                    and exact_run_id not in self._claims
                ):
                    try:
                        marker_artifact.path.unlink()
                    except OSError:
                        pass
                raise

    def _refresh_claim(self, claim: _RunArtifactClaim, *, checkpoints: bool) -> None:
        with self._lock:
            database = self._database(claim.run_id)
            checkpoint_root = self._checkpoints(claim.run_id)
            if claim.database is None or not self._matches(claim.database):
                raise RuntimeError("local coordinator database ownership changed")
            if checkpoints and (
                claim.checkpoints is None or not self._matches(claim.checkpoints)
            ):
                raise RuntimeError("local coordinator checkpoint ownership changed")
            elif not checkpoints and os.path.lexists(checkpoint_root):
                raise RuntimeError("local coordinator checkpoint ownership changed")

    def _abort(self, run_id: str) -> None:
        with self._lock:
            claim = self._claims.get(run_id)
            if claim is None:
                return
            if not self._matches(claim.marker):
                raise RuntimeError("local coordinator ownership marker changed")
            for artifact in (claim.database, claim.checkpoints):
                if artifact is None:
                    continue
                if not self._matches(artifact):
                    raise RuntimeError("local coordinator owned artifact changed")
            if claim.database is not None:
                claim.database.path.unlink()
            if claim.checkpoints is not None:
                shutil.rmtree(claim.checkpoints.path)
            claim.marker.path.unlink()
            self._owned.pop(run_id, None)
            self._claims.pop(run_id, None)
            self._persisted_run_ids.discard(run_id)
            self._write_run_ids(self._persisted_run_ids)

    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        del project_id
        claim = self._claim(run_id, checkpoints=True)
        try:
            checkpoint_store = LocalScenarioCheckpointStore(self._checkpoints(run_id))
            initialize_situated_percept_memory(self._database(run_id))
            coordinator = ScenarioCoordinator.create(
                self._database(run_id),
                scenario,
                run_id=run_id,
                stream_id=stream_id,
                publisher=_RouterPublisher(self._router, run_id),
                checkpoint_store=checkpoint_store,
            )
        finally:
            self._refresh_claim(claim, checkpoints=True)
        with self._lock:
            self._owned[run_id] = coordinator
        return coordinator

    def fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
    ):
        claim = self._claim(request.child_run_id, checkpoints=False)
        try:
            child, result = coordinator.fork(
                request,
                capability,
                self._database(request.child_run_id),
                child_publisher=_RouterPublisher(self._router, request.child_run_id),
                child_database_ownership_token=_PhysicalFileOwnershipToken(
                    claim.database.device,
                    claim.database.inode,
                ) if claim.database is not None else None,
            )
        finally:
            self._refresh_claim(claim, checkpoints=False)
        with self._lock:
            self._owned[request.child_run_id] = child
        return child, result

    def abort_create(self, *, project_id: str, run_id: str, stream_id: str) -> None:
        del project_id, stream_id
        self._abort(run_id)

    def abort_fork(self, coordinator: ScenarioCoordinator, request: ScenarioForkRequest) -> None:
        del coordinator
        self._abort(request.child_run_id)


class _ConfiguredAuthenticator:
    def __init__(self, capability: StudioCapability, *, trust_all: bool, token: str | None) -> None:
        self._capability = capability
        self._trust_all = trust_all
        self._token = token

    def __call__(self, request) -> StudioCapability | None:
        if self._trust_all:
            return self._capability
        header = request.headers.get("authorization", "")
        expected = f"Bearer {self._token}"
        return self._capability if secrets.compare_digest(header, expected) else None


def build_application(settings: LauncherSettings):
    """Compose real SQLite/project/coordinator/stream authorities from explicit roots."""
    if not isinstance(settings, LauncherSettings):
        raise TypeError("World Studio launcher settings are required")
    runs_root = settings.workspace_root / "runs"
    runs_root.mkdir(exist_ok=True)
    workspace = ScenarioProjectWorkspace.open(
        settings.workspace_root / "projects.sqlite3",
        import_roots=settings.import_roots,
        export_root=settings.export_root,
        limits=settings.workspace_limits,
    )
    router = StudioOutputRouter(limits=settings.output_limits)
    factory = _LocalCoordinatorFactory(runs_root, router)
    registry = InMemoryScenarioRunRegistry(
        unavailable_run_ids=factory.unavailable_run_ids,
    )
    service = WorldStudioService(
        workspace,
        registry,
        factory,
        import_sources=dict(settings.import_sources),
    )
    capability = StudioCapability(
        settings.authority_id,
        project_ids=settings.project_ids,
        run_ids=settings.run_ids,
        agent_ids=settings.agent_ids,
        permissions=settings.permissions,
    )
    authenticate = _ConfiguredAuthenticator(
        capability,
        trust_all=settings.development_trust_all,
        token=settings.auth_token,
    )
    app = create_world_studio_asgi_app(
        JsonRpcDispatcher(service),
        router,
        authenticate,
        authenticate,
        allowed_origins=settings.allowed_origins,
        limits=settings.server_limits,
        static_root=settings.static_root,
        allow_ambient_authentication=settings.development_trust_all,
    )
    app.state.studio_service = service
    app.state.output_router = router
    app.state.coordinator_factory = factory
    return app


def _source(value: str) -> tuple[str, Path]:
    source_id, separator, path = value.partition("=")
    if not separator or not source_id or not path:
        raise argparse.ArgumentTypeError("import source must be ID=PATH")
    return source_id, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local World Studio V22 server")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--import-root", required=True, action="append", type=Path)
    parser.add_argument("--import-source", required=True, action="append", type=_source)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--static-root", required=True, type=Path)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--bind-port", required=True, type=int)
    parser.add_argument("--origin", required=True, action="append")
    parser.add_argument("--development-trust-all", action="store_true")
    parser.add_argument("--auth-token-file", type=Path)
    parser.add_argument("--tls-certificate", type=Path)
    parser.add_argument("--tls-private-key", type=Path)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--project-id", required=True, action="append")
    parser.add_argument("--run-id", required=True, action="append")
    parser.add_argument("--agent-id", action="append", default=[])
    parser.add_argument("--permission", action="append", choices=STUDIO_PERMISSIONS)
    parser.add_argument("--maximum-http-body-bytes", type=int, default=1_048_576)
    parser.add_argument("--maximum-websocket-frame-bytes", type=int, default=1_048_576)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-websocket-connections", type=int, default=128)
    parser.add_argument("--maximum-subscriptions-per-connection", type=int, default=16)
    parser.add_argument("--maximum-sessions", type=int, default=128)
    parser.add_argument("--session-lifetime-seconds", type=int, default=3600)
    parser.add_argument("--maximum-retained-batches", type=int, default=128)
    parser.add_argument("--maximum-released-subscriptions", type=int, default=64)
    return parser


def _read_token(path: Path | None) -> str | None:
    if path is None:
        return None
    token_file = _existing_file(path, label="auth token file")
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except OSError:
        raise ValueError("World Studio auth token file is unavailable") from None
    if not token or "\n" in token or "\r" in token:
        raise ValueError("World Studio auth token file must contain one token")
    return token


def settings_from_args(arguments: list[str] | None = None) -> LauncherSettings:
    values = _parser().parse_args(arguments)
    return LauncherSettings(
        workspace_root=values.workspace_root,
        import_roots=tuple(values.import_root),
        import_sources=tuple(values.import_source),
        export_root=values.export_root,
        static_root=values.static_root,
        bind_host=values.bind_host,
        bind_port=values.bind_port,
        allowed_origins=tuple(values.origin),
        development_trust_all=values.development_trust_all,
        auth_token=_read_token(values.auth_token_file),
        tls_certificate=values.tls_certificate,
        tls_private_key=values.tls_private_key,
        authority_id=values.authority_id,
        project_ids=tuple(values.project_id),
        run_ids=tuple(values.run_id),
        agent_ids=tuple(values.agent_id),
        permissions=tuple(values.permission or STUDIO_PERMISSIONS),
        output_limits=StudioOutputLimits(
            maximum_retained_batches=values.maximum_retained_batches,
            maximum_released_subscriptions=values.maximum_released_subscriptions,
        ),
        server_limits=WorldStudioServerLimits(
            maximum_http_body_bytes=values.maximum_http_body_bytes,
            maximum_websocket_frame_bytes=values.maximum_websocket_frame_bytes,
            request_timeout_seconds=values.request_timeout_seconds,
            maximum_websocket_connections=values.maximum_websocket_connections,
            maximum_subscriptions_per_connection=values.maximum_subscriptions_per_connection,
            maximum_sessions=values.maximum_sessions,
            session_lifetime_seconds=values.session_lifetime_seconds,
        ),
    )


def main(arguments: list[str] | None = None) -> None:
    settings = settings_from_args(arguments)
    app = build_application(settings)
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    host = f"[{settings.bind_host}]" if ":" in settings.bind_host else settings.bind_host
    config.bind = [f"{host}:{settings.bind_port}"]
    config.alpn_protocols = ["h2", "http/1.1"]
    config.use_reloader = False
    config.workers = 1
    if settings.tls_certificate is not None:
        config.certfile = str(settings.tls_certificate)
        config.keyfile = str(settings.tls_private_key)
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
