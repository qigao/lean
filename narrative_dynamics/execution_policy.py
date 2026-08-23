from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.model_contract import ModelContract


TRUSTED_EXECUTION_POLICY_SCHEMA_VERSION = 1
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class TrustedExecutionPolicyViolation(ValueError):
    """A source cannot be bound to the independently supplied execution policy."""

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.expected = expected
        self.actual = actual
        prefix = "trusted execution policy violation"
        if model_name is not None:
            prefix += f" for {model_name!r}"
        super().__init__(f"{prefix}: {message}")


def _validated_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _validated_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


@dataclass(frozen=True)
class TrustedModelPin:
    """Independent allow-list entry for one exact declared model contract."""

    model_name: str
    declared_contract_hash: str
    expected_implementation_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_name",
            _validated_text(self.model_name, label="trusted model name"),
        )
        object.__setattr__(
            self,
            "declared_contract_hash",
            _validated_hash(
                self.declared_contract_hash,
                label="trusted declared contract hash",
            ),
        )
        object.__setattr__(
            self,
            "expected_implementation_hash",
            _validated_hash(
                self.expected_implementation_hash,
                label="trusted expected implementation hash",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "declared_contract_hash": self.declared_contract_hash,
            "expected_implementation_hash": self.expected_implementation_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class TrustedExecutionPolicy:
    """Versioned policy supplied by a trusted deployment/configuration channel.

    The policy is deliberately separate from model-owned metadata. Binding does
    not execute or measure the source; it injects the independently selected pin
    into the immutable model contract. The existing runner then measures and
    verifies that pin immediately before execution.
    """

    name: str
    version: str
    pins: tuple[TrustedModelPin, ...]
    schema_version: int = TRUSTED_EXECUTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_text(self.name, label="trusted execution policy name"),
        )
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, label="trusted execution policy version"),
        )
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("trusted execution policy schema version must be positive")
        pins = tuple(self.pins)
        if any(not isinstance(pin, TrustedModelPin) for pin in pins):
            raise TypeError("trusted execution policy pins must be TrustedModelPin values")
        pins = tuple(sorted(pins, key=lambda pin: pin.model_name))
        names = tuple(pin.model_name for pin in pins)
        if len(set(names)) != len(names):
            raise ValueError("trusted execution policy model names must be unique")
        object.__setattr__(self, "pins", pins)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "pins": tuple(pin.manifest_identity() for pin in self.pins),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}

    def pin_for(self, model_name: str) -> TrustedModelPin:
        canonical_name = _validated_text(model_name, label="policy model name")
        for pin in self.pins:
            if pin.model_name == canonical_name:
                return pin
        raise TrustedExecutionPolicyViolation(
            "no allow-list entry exists",
            model_name=canonical_name,
        )

    def bind(
        self,
        source: object,
        *,
        contract: ModelContract | None = None,
    ) -> "PolicyBoundSource":
        model_name = getattr(source, "name", None)
        if not isinstance(model_name, str) or not model_name:
            raise TrustedExecutionPolicyViolation(
                "source does not expose a non-empty model name"
            )
        pin = self.pin_for(model_name)
        declared = ModelContract.from_source(source) if contract is None else contract
        if not isinstance(declared, ModelContract):
            raise TypeError("trusted execution policy contract must be ModelContract")

        actual_contract_hash = declared.content_hash
        if actual_contract_hash != pin.declared_contract_hash:
            raise TrustedExecutionPolicyViolation(
                "declared contract hash is not allow-listed",
                model_name=model_name,
                expected=pin.declared_contract_hash,
                actual=actual_contract_hash,
            )

        source_pin = getattr(declared, "expected_implementation_hash", None)
        if source_pin is not None and source_pin != pin.expected_implementation_hash:
            raise TrustedExecutionPolicyViolation(
                "model-owned implementation pin disagrees with trusted policy",
                model_name=model_name,
                expected=pin.expected_implementation_hash,
                actual=source_pin,
            )

        bound_contract = replace(
            declared,
            expected_implementation_hash=pin.expected_implementation_hash,
        )
        return PolicyBoundSource(
            source=source,
            contract=bound_contract,
            policy=self,
            pin=pin,
            declared_contract_hash=actual_contract_hash,
        )


@dataclass(frozen=True)
class PolicyBoundSource:
    """Transparent execution source carrying an externally pinned contract."""

    source: object
    contract: ModelContract
    policy: TrustedExecutionPolicy
    pin: TrustedModelPin
    declared_contract_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ModelContract):
            raise TypeError("policy-bound source contract must be ModelContract")
        if not isinstance(self.policy, TrustedExecutionPolicy):
            raise TypeError("policy-bound source policy must be TrustedExecutionPolicy")
        if not isinstance(self.pin, TrustedModelPin):
            raise TypeError("policy-bound source pin must be TrustedModelPin")
        object.__setattr__(
            self,
            "declared_contract_hash",
            _validated_hash(
                self.declared_contract_hash,
                label="policy-bound declared contract hash",
            ),
        )
        if self.pin.declared_contract_hash != self.declared_contract_hash:
            raise ValueError("policy-bound source pin does not match declared contract")
        if (
            getattr(self.contract, "expected_implementation_hash", None)
            != self.pin.expected_implementation_hash
        ):
            raise ValueError("policy-bound contract does not contain the trusted pin")

    @property
    def name(self) -> str:
        value = getattr(self.source, "name", None)
        if not isinstance(value, str) or not value:
            raise TrustedExecutionPolicyViolation(
                "bound source lost its model name",
                model_name=self.pin.model_name,
            )
        return value

    def __getattr__(self, attribute: str) -> object:
        return getattr(self.source, attribute)

    def manifest_identity(self) -> dict[str, object]:
        identity_method = getattr(self.source, "manifest_identity", None)
        if callable(identity_method):
            raw = identity_method()
            if not isinstance(raw, Mapping):
                raise TypeError("source manifest identity must be a mapping")
            identity: dict[str, object] = dict(raw)
        else:
            identity = {
                "name": self.name,
                "version": self.contract.version,
                "implementation_revision": self.contract.implementation_revision,
            }

        identity.update(
            {
                "name": self.name,
                "version": self.contract.version,
                "implementation_revision": self.contract.implementation_revision,
                "contract": self.contract.manifest_identity(),
                "contract_hash": self.contract.content_hash,
                "trusted_execution_policy": {
                    "schema_version": self.policy.schema_version,
                    "name": self.policy.name,
                    "version": self.policy.version,
                    "content_hash": self.policy.content_hash,
                    "pin": self.pin.manifest_identity(),
                    "declared_contract_hash": self.declared_contract_hash,
                    "bound_contract_hash": self.contract.content_hash,
                    "verification": "matched",
                },
            }
        )
        stable_content_hash(identity)
        return identity


__all__ = [
    "TRUSTED_EXECUTION_POLICY_SCHEMA_VERSION",
    "PolicyBoundSource",
    "TrustedExecutionPolicy",
    "TrustedExecutionPolicyViolation",
    "TrustedModelPin",
]
