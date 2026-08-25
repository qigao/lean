from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec


_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _hook_hashes(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("trusted domain hook hashes must be a mapping")
    frozen: dict[str, str] = {}
    for name, content_hash in value.items():
        key = _text(name, label="trusted domain hook name")
        frozen[key] = _content_hash(
            content_hash,
            label=f"trusted domain hook {key!r} implementation hash",
        )
    return MappingProxyType(dict(sorted(frozen.items())))


@dataclass(frozen=True)
class TrustedDomainPin:
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    hook_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_id", _text(self.domain_id, label="trusted domain id"))
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="trusted domain version"),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _content_hash(self.domain_spec_hash, label="trusted domain spec hash"),
        )
        object.__setattr__(self, "hook_hashes", _hook_hashes(self.hook_hashes))

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "hook_hashes": dict(self.hook_hashes),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TrustedDomainPolicy:
    policy_id: str
    version: str
    pins: tuple[TrustedDomainPin, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, label="trusted domain policy id"))
        object.__setattr__(self, "version", _text(self.version, label="trusted domain policy version"))
        pins = tuple(self.pins)
        if not pins or any(not isinstance(item, TrustedDomainPin) for item in pins):
            raise TypeError("trusted domain policy requires TrustedDomainPin values")
        keys = tuple((item.domain_id, item.domain_version) for item in pins)
        if len(set(keys)) != len(keys):
            raise ValueError("trusted domain policy pins must be unique by domain/version")
        object.__setattr__(
            self,
            "pins",
            tuple(sorted(pins, key=lambda item: (item.domain_id, item.domain_version))),
        )

    def pin_for(self, domain_id: str, domain_version: str) -> TrustedDomainPin:
        domain_id = _text(domain_id, label="trusted domain lookup id")
        domain_version = _text(domain_version, label="trusted domain lookup version")
        matches = tuple(
            item
            for item in self.pins
            if item.domain_id == domain_id and item.domain_version == domain_version
        )
        if len(matches) != 1:
            raise ValueError("trusted domain is not allowed by policy")
        return matches[0]

    def bind(self, domain: DomainSpec) -> TrustedDomainBinding:
        if not isinstance(domain, DomainSpec):
            raise TypeError("trusted domain binding requires DomainSpec")
        pin = self.pin_for(domain.domain_id, domain.version)
        actual_hooks = {
            hook.name: hook.implementation_hash for hook in domain.semantic_hooks
        }
        if pin.domain_spec_hash != domain.content_hash or dict(pin.hook_hashes) != actual_hooks:
            raise ValueError("trusted domain identity does not match policy")
        return TrustedDomainBinding(domain, self, pin)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "pins": [item.to_dict() for item in self.pins],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TrustedDomainBinding:
    domain: DomainSpec
    policy: TrustedDomainPolicy
    pin: TrustedDomainPin

    def __post_init__(self) -> None:
        if not isinstance(self.domain, DomainSpec):
            raise TypeError("trusted domain binding requires DomainSpec")
        if not isinstance(self.policy, TrustedDomainPolicy):
            raise TypeError("trusted domain binding requires TrustedDomainPolicy")
        if not isinstance(self.pin, TrustedDomainPin):
            raise TypeError("trusted domain binding requires TrustedDomainPin")
        expected = self.policy.pin_for(self.domain.domain_id, self.domain.version)
        if expected != self.pin:
            raise ValueError("trusted domain binding pin does not match policy")
        actual_hooks = {
            hook.name: hook.implementation_hash for hook in self.domain.semantic_hooks
        }
        if self.pin.domain_spec_hash != self.domain.content_hash or dict(self.pin.hook_hashes) != actual_hooks:
            raise ValueError("trusted domain identity does not match policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "trust_status": "pinned",
            "domain_id": self.domain.domain_id,
            "domain_version": self.domain.version,
            "domain_spec_hash": self.domain.content_hash,
            "policy_id": self.policy.policy_id,
            "policy_version": self.policy.version,
            "policy_hash": self.policy.content_hash,
            "pin_hash": self.pin.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeAnalysisArtifact:
    canonical_narrative_hash: str
    domain_spec_hash: str
    decision_model_identity: str
    analysis_scope_hash: str
    intervention_manifest_hash: str
    analysis_payload_hash: str
    compilation_artifact_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "canonical_narrative_hash",
            "domain_spec_hash",
            "decision_model_identity",
            "analysis_scope_hash",
            "intervention_manifest_hash",
            "analysis_payload_hash",
        ):
            object.__setattr__(
                self,
                name,
                _content_hash(getattr(self, name), label=name.replace("_", " ")),
            )
        if self.compilation_artifact_hash is not None:
            object.__setattr__(
                self,
                "compilation_artifact_hash",
                _content_hash(
                    self.compilation_artifact_hash,
                    label="compilation artifact hash",
                ),
            )

    def identity_dict(self) -> dict[str, str]:
        return {
            "canonical_narrative_hash": self.canonical_narrative_hash,
            "domain_spec_hash": self.domain_spec_hash,
            "decision_model_identity": self.decision_model_identity,
            "analysis_scope_hash": self.analysis_scope_hash,
            "intervention_manifest_hash": self.intervention_manifest_hash,
            "analysis_payload_hash": self.analysis_payload_hash,
        }

    @property
    def analysis_identity_hash(self) -> str:
        return stable_content_hash(self.identity_dict())

    def lineage_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "analysis_identity_hash": self.analysis_identity_hash,
            "compilation_artifact_hash": self.compilation_artifact_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.lineage_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self.lineage_dict(), "content_hash": self.content_hash}
