"""Explicit local composition and Hypercorn launcher for World Studio V22."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import ipaddress
from pathlib import Path
import secrets
import shutil
from threading import RLock

from narrative_dynamics.abm.scenario_checkpoint_store import LocalScenarioCheckpointStore
from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
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


class _LocalCoordinatorFactory:
    def __init__(self, root: Path, router: StudioOutputRouter) -> None:
        self._root = root.resolve(strict=True)
        self._router = router
        self._lock = RLock()
        self._owned: dict[str, ScenarioCoordinator] = {}

    def _database(self, run_id: str) -> Path:
        return self._root / f"{run_id}.sqlite3"

    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        del project_id
        checkpoint_store = LocalScenarioCheckpointStore(self._root / f"{run_id}-checkpoints")
        coordinator = ScenarioCoordinator.create(
            self._database(run_id),
            scenario,
            run_id=run_id,
            stream_id=stream_id,
            publisher=_RouterPublisher(self._router, run_id),
            checkpoint_store=checkpoint_store,
        )
        with self._lock:
            self._owned[run_id] = coordinator
        return coordinator

    def fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
    ):
        child, result = coordinator.fork(
            request,
            capability,
            self._database(request.child_run_id),
            child_publisher=_RouterPublisher(self._router, request.child_run_id),
        )
        with self._lock:
            self._owned[request.child_run_id] = child
        return child, result

    def abort_create(self, *, project_id: str, run_id: str, stream_id: str) -> None:
        del project_id, stream_id
        with self._lock:
            self._owned.pop(run_id, None)
        self._database(run_id).unlink(missing_ok=True)
        shutil.rmtree(self._root / f"{run_id}-checkpoints", ignore_errors=True)

    def abort_fork(self, coordinator: ScenarioCoordinator, request: ScenarioForkRequest) -> None:
        del coordinator
        with self._lock:
            self._owned.pop(request.child_run_id, None)
        self._database(request.child_run_id).unlink(missing_ok=True)


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
    registry = InMemoryScenarioRunRegistry()
    factory = _LocalCoordinatorFactory(runs_root, router)
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
