from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

from narrative_dynamics.attestation._common import (
    validated_hash,
    validated_text,
)
from narrative_dynamics.contracts import stable_content_hash


IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION = 1
_MAX_IMPLEMENTATION_ARTIFACT_BYTES = 16 * 1024 * 1024
_UNUSABLE_MODULE_ORIGINS = {"built-in", "frozen", "namespace"}


class ImplementationAttestationUnavailable(RuntimeError):
    """The trusted parent cannot map an execution target to readable module bytes."""


@dataclass(frozen=True)
class ImplementationArtifact:
    """Digest of one logical module without exposing host-local paths."""

    locator: str
    sha256: str
    byte_count: int
    kind: str = "python_source"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locator",
            validated_text(self.locator, label="implementation artifact locator"),
        )
        object.__setattr__(
            self,
            "sha256",
            validated_hash(self.sha256, label="implementation artifact hash"),
        )
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or self.byte_count < 0
        ):
            raise ValueError("implementation artifact byte count must be non-negative")
        object.__setattr__(
            self,
            "kind",
            validated_text(self.kind, label="implementation artifact kind"),
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "locator": self.locator,
            "kind": self.kind,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class ImplementationAttestation:
    """Measured identity of directly located module files.

    V1 deliberately does not claim a transitive dependency closure, package
    signature verification, native-library attestation, or proof that a worker
    loaded the same bytes after this trusted-parent snapshot.
    """

    target: str
    artifacts: tuple[ImplementationArtifact, ...]
    schema_version: int = IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target",
            validated_text(self.target, label="implementation target"),
        )
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError(
                "implementation attestation schema version must be positive"
            )
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("implementation attestation requires at least one artifact")
        if any(not isinstance(item, ImplementationArtifact) for item in artifacts):
            raise TypeError(
                "implementation attestation artifacts must be ImplementationArtifact values"
            )
        artifacts = tuple(sorted(artifacts, key=lambda item: item.locator))
        if len({item.locator for item in artifacts}) != len(artifacts):
            raise ValueError("implementation artifact locators must be unique")
        object.__setattr__(self, "artifacts", artifacts)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "scope": "direct_python_modules",
            "artifacts": tuple(item.manifest_identity() for item in self.artifacts),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._payload(), "content_hash": self.content_hash}


def _source_candidate(path: Path) -> Path:
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except (ValueError, NotImplementedError):
            pass
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ImplementationAttestationUnavailable(
            "loaded implementation module has no readable backing file"
        ) from error
    if not resolved.is_file():
        raise ImplementationAttestationUnavailable(
            "loaded implementation module backing path is not a file"
        )
    return resolved


def _loaded_module_path(module_name: str, module: object) -> Path:
    loaded_path = getattr(module, "__file__", None)
    if not isinstance(loaded_path, str) or not loaded_path:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if (
            not isinstance(origin, str)
            or not origin
            or origin in _UNUSABLE_MODULE_ORIGINS
        ):
            raise ImplementationAttestationUnavailable(
                f"loaded implementation module {module_name!r} has no file identity"
            )
        loaded_path = origin
    return _source_candidate(Path(loaded_path))


def _existing_source(path: Path) -> Path | None:
    """Return one readable source candidate without raising for a search miss."""

    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() else None


def _unloaded_module_path(module_name: str) -> Path:
    """Resolve an unloaded module using regular/namespace package precedence.

    The resolver intentionally avoids importing external code. It mirrors the
    parts of Python import selection that matter for ordinary filesystem source:
    an earlier module or regular package blocks later same-name entries, while
    namespace-package portions are combined only until a regular package wins.
    """

    parts = module_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ImplementationAttestationUnavailable(
            f"implementation module {module_name!r} is not a valid module name"
        )

    search_locations = tuple(Path(entry or os.curdir) for entry in sys.path)
    for index, part in enumerate(parts):
        is_final = index == len(parts) - 1
        namespace_locations: list[Path] = []
        resolved_regular_package: Path | None = None

        for location in search_locations:
            package_directory = location / part
            package_init = _existing_source(package_directory / "__init__.py")
            module_file = _existing_source(location / f"{part}.py")

            # FileFinder gives a regular package in this path entry precedence
            # over the same-name module file and over all later sys.path entries.
            if package_init is not None:
                if is_final:
                    return package_init
                resolved_regular_package = package_directory.resolve()
                break

            # A module in an earlier path entry blocks later packages. It cannot
            # supply a child module when this is an intermediate dotted segment.
            if module_file is not None:
                if is_final:
                    return module_file
                raise ImplementationAttestationUnavailable(
                    f"implementation module prefix {part!r} is not a package"
                )

            try:
                namespace_directory = package_directory.resolve(strict=True)
            except OSError:
                continue
            if namespace_directory.is_dir():
                namespace_locations.append(namespace_directory)

        if is_final:
            raise ImplementationAttestationUnavailable(
                f"implementation module {module_name!r} has no readable source file"
            )
        if resolved_regular_package is not None:
            search_locations = (resolved_regular_package,)
        elif namespace_locations:
            search_locations = tuple(namespace_locations)
        else:
            raise ImplementationAttestationUnavailable(
                f"implementation module {module_name!r} has no importable package prefix"
            )

    raise ImplementationAttestationUnavailable(
        f"implementation module {module_name!r} has no readable source file"
    )


def _module_file(module_name: str) -> Path:
    """Resolve the loaded module or import-faithful path for an unloaded module.

    A loaded or explicitly blocked module never falls back to sys.path. Doing so
    could attest a same-name shadow file that was not selected by Python.
    """

    if module_name in sys.modules:
        loaded = sys.modules[module_name]
        if loaded is None:
            raise ImplementationAttestationUnavailable(
                f"implementation module {module_name!r} is blocked in sys.modules"
            )
        return _loaded_module_path(module_name, loaded)
    return _unloaded_module_path(module_name)


def _artifact_kind(path: Path) -> str:
    return "python_source" if path.suffix in {".py", ".pyw"} else "python_module_file"


def _measure_module(module_name: str) -> ImplementationArtifact:
    module_name = validated_text(module_name, label="implementation module name")
    path = _module_file(module_name)
    try:
        byte_count = path.stat().st_size
        if byte_count > _MAX_IMPLEMENTATION_ARTIFACT_BYTES:
            raise ImplementationAttestationUnavailable(
                f"implementation module {module_name!r} exceeds the measurement limit"
            )
        payload = path.read_bytes()
    except ImplementationAttestationUnavailable:
        raise
    except OSError as error:
        raise ImplementationAttestationUnavailable(
            f"cannot read implementation module {module_name!r}"
        ) from error
    if len(payload) > _MAX_IMPLEMENTATION_ARTIFACT_BYTES:
        raise ImplementationAttestationUnavailable(
            f"implementation module {module_name!r} exceeds the measurement limit"
        )
    return ImplementationArtifact(
        locator=f"python-module:{module_name}",
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        byte_count=len(payload),
        kind=_artifact_kind(path),
    )


def _unwrap_registered_source(source: object) -> object:
    nested = getattr(source, "source", None)
    contract = getattr(source, "contract", None)
    return nested if nested is not None and contract is not None else source


def measure_implementation(source: object) -> ImplementationAttestation:
    """Measure directly located module bytes for a model, factory, or process source."""

    source = _unwrap_registered_source(source)
    factory_path = getattr(source, "factory", None)
    worker_module = getattr(source, "worker_module", None)
    if isinstance(factory_path, str) and isinstance(worker_module, str):
        factory_module, separator, factory_attribute = factory_path.partition(":")
        if not separator or not factory_module or not factory_attribute:
            raise ImplementationAttestationUnavailable(
                "subprocess implementation factory is not a module:attribute path"
            )
        artifacts = {
            item.locator: item
            for item in (
                _measure_module(factory_module),
                _measure_module(worker_module),
            )
        }
        return ImplementationAttestation(
            target=f"python-subprocess-factory:{factory_path}",
            artifacts=tuple(artifacts.values()),
        )

    create = getattr(source, "create", None)
    if callable(create):
        module_name = getattr(create, "__module__", None)
        qualname = getattr(create, "__qualname__", None)
        if not isinstance(module_name, str) or not module_name:
            raise ImplementationAttestationUnavailable(
                "model factory callable has no module identity"
            )
        if not isinstance(qualname, str) or not qualname:
            qualname = create.__class__.__qualname__
        return ImplementationAttestation(
            target=f"python-callable:{module_name}.{qualname}",
            artifacts=(_measure_module(module_name),),
        )

    implementation_type = source if isinstance(source, type) else source.__class__
    module_name = getattr(implementation_type, "__module__", None)
    qualname = getattr(implementation_type, "__qualname__", None)
    if not isinstance(module_name, str) or not module_name:
        raise ImplementationAttestationUnavailable(
            "implementation class has no module identity"
        )
    if not isinstance(qualname, str) or not qualname:
        raise ImplementationAttestationUnavailable(
            "implementation class has no qualified name"
        )
    return ImplementationAttestation(
        target=f"python-class:{module_name}.{qualname}",
        artifacts=(_measure_module(module_name),),
    )


def implementation_attestation_identity(source: object) -> dict[str, object]:
    """Return an explicit measured or unavailable manifest identity."""

    try:
        measured = measure_implementation(source)
    except ImplementationAttestationUnavailable as error:
        return {
            "schema_version": IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": str(error),
        }
    return {**measured.manifest_identity(), "status": "measured"}


__all__ = [
    "IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION",
    "ImplementationArtifact",
    "ImplementationAttestation",
    "ImplementationAttestationUnavailable",
    "implementation_attestation_identity",
    "measure_implementation",
]
