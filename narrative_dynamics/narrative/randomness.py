from __future__ import annotations

from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash


RANDOM_DERIVATION_VERSION = "narrative-hash-categorical-v1"
_NAMESPACES = frozenset({"world.transition", "observation.projection"})
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_U64 = 1 << 64


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _trimmed_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def validate_root_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("narrative root seed must be an integer")
    return seed


def derive_stream_hash(
    *,
    root_seed: int,
    namespace: str,
    step_index: int,
    source_hash: str,
    component_hash: str,
    component_key: tuple[str, ...],
) -> str:
    root_seed = validate_root_seed(root_seed)
    namespace = _trimmed_text(namespace, label="narrative random namespace")
    if namespace not in _NAMESPACES:
        raise ValueError("narrative random namespace is unsupported")
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or step_index < 0
    ):
        raise ValueError("narrative random step index must be non-negative")
    source_hash = _hash(source_hash, label="narrative random source hash")
    component_hash = _hash(
        component_hash,
        label="narrative random component hash",
    )
    if not isinstance(component_key, tuple) or not component_key:
        raise TypeError(
            "narrative random component key must be a non-empty tuple"
        )
    canonical_key = tuple(
        _trimmed_text(item, label="narrative random component key item")
        for item in component_key
    )
    return stable_content_hash(
        {
            "derivation_version": RANDOM_DERIVATION_VERSION,
            "root_seed": root_seed,
            "namespace": namespace,
            "step_index": step_index,
            "source_hash": source_hash,
            "component_hash": component_hash,
            "component_key": list(canonical_key),
        }
    )


def draw_u64_for_stream(stream_hash: str, *, draw_index: int = 0) -> int:
    stream_hash = _hash(stream_hash, label="narrative random stream hash")
    if (
        not isinstance(draw_index, int)
        or isinstance(draw_index, bool)
        or draw_index != 0
    ):
        raise ValueError("narrative random V1 draw index must be zero")
    draw_hash = stable_content_hash(
        {
            "derivation_version": RANDOM_DERIVATION_VERSION,
            "stream_hash": stream_hash,
            "draw_index": draw_index,
        }
    )
    return int(draw_hash.removeprefix("sha256:")[:16], 16)


@dataclass(frozen=True)
class RandomSampleRecord:
    derivation_version: str
    root_seed: int
    namespace: str
    step_index: int
    source_hash: str
    component_hash: str
    component_key: tuple[str, ...]
    stream_hash: str
    draw_index: int
    draw_u64: int
    distribution_hash: str
    selected_outcome_id: str
    selected_outcome_hash: str

    def __post_init__(self) -> None:
        if self.derivation_version != RANDOM_DERIVATION_VERSION:
            raise ValueError("narrative random derivation version mismatch")
        root_seed = validate_root_seed(self.root_seed)
        namespace = _trimmed_text(
            self.namespace,
            label="narrative random namespace",
        )
        if namespace not in _NAMESPACES:
            raise ValueError("narrative random namespace is unsupported")
        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 0
        ):
            raise ValueError("narrative random step index must be non-negative")
        source_hash = _hash(
            self.source_hash,
            label="narrative random source hash",
        )
        component_hash = _hash(
            self.component_hash,
            label="narrative random component hash",
        )
        if not isinstance(self.component_key, tuple) or not self.component_key:
            raise TypeError(
                "narrative random component key must be a non-empty tuple"
            )
        component_key = tuple(
            _trimmed_text(
                item,
                label="narrative random component key item",
            )
            for item in self.component_key
        )
        if (
            not isinstance(self.draw_index, int)
            or isinstance(self.draw_index, bool)
            or self.draw_index != 0
        ):
            raise ValueError("narrative random V1 draw index must be zero")
        if (
            not isinstance(self.draw_u64, int)
            or isinstance(self.draw_u64, bool)
            or self.draw_u64 < 0
            or self.draw_u64 >= _MAX_U64
        ):
            raise ValueError("narrative random draw must be an unsigned 64-bit integer")
        distribution_hash = _hash(
            self.distribution_hash,
            label="narrative random distribution hash",
        )
        selected_outcome_id = _trimmed_text(
            self.selected_outcome_id,
            label="narrative random selected outcome id",
        )
        selected_outcome_hash = _hash(
            self.selected_outcome_hash,
            label="narrative random selected outcome hash",
        )
        expected_stream = derive_stream_hash(
            root_seed=root_seed,
            namespace=namespace,
            step_index=self.step_index,
            source_hash=source_hash,
            component_hash=component_hash,
            component_key=component_key,
        )
        if self.stream_hash != expected_stream:
            raise ValueError("narrative random stream hash mismatch")
        expected_draw = draw_u64_for_stream(
            expected_stream,
            draw_index=self.draw_index,
        )
        if self.draw_u64 != expected_draw:
            raise ValueError("narrative random draw mismatch")
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "component_hash", component_hash)
        object.__setattr__(self, "component_key", component_key)
        object.__setattr__(self, "stream_hash", expected_stream)
        object.__setattr__(self, "distribution_hash", distribution_hash)
        object.__setattr__(self, "selected_outcome_id", selected_outcome_id)
        object.__setattr__(self, "selected_outcome_hash", selected_outcome_hash)

    def to_dict(self) -> dict[str, object]:
        return {
            "derivation_version": self.derivation_version,
            "root_seed": self.root_seed,
            "namespace": self.namespace,
            "step_index": self.step_index,
            "source_hash": self.source_hash,
            "component_hash": self.component_hash,
            "component_key": list(self.component_key),
            "stream_hash": self.stream_hash,
            "draw_index": self.draw_index,
            "draw_u64": self.draw_u64,
            "distribution_hash": self.distribution_hash,
            "selected_outcome_id": self.selected_outcome_id,
            "selected_outcome_hash": self.selected_outcome_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def sample_categorical(
    *,
    root_seed: int,
    namespace: str,
    step_index: int,
    source_hash: str,
    component_hash: str,
    component_key: tuple[str, ...],
    distribution_hash: str,
    outcomes: tuple[tuple[str, float, str], ...],
) -> RandomSampleRecord:
    if not isinstance(outcomes, tuple):
        raise TypeError("narrative categorical outcomes must be a tuple")
    if len(outcomes) < 2:
        raise ValueError("narrative categorical distribution requires at least two outcomes")

    canonical: list[tuple[str, float, str]] = []
    for raw in outcomes:
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise TypeError(
                "narrative categorical outcomes must be id/probability/hash triples"
            )
        outcome_id = _trimmed_text(
            raw[0],
            label="narrative categorical outcome id",
        )
        probability = raw[1]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
        ):
            raise TypeError("narrative categorical probability must be numeric")
        probability = float(probability)
        if (
            not math.isfinite(probability)
            or probability <= 0.0
            or probability > 1.0
        ):
            raise ValueError(
                "narrative categorical probability must be finite and in (0, 1]"
            )
        outcome_hash = _hash(
            raw[2],
            label="narrative categorical outcome hash",
        )
        canonical.append((outcome_id, probability, outcome_hash))

    canonical.sort(key=lambda item: item[0])
    if len({item[0] for item in canonical}) != len(canonical):
        raise ValueError("narrative categorical outcome ids must be unique")
    probabilities = tuple(item[1] for item in canonical)
    if math.fsum(probabilities) != 1.0:
        raise ValueError("narrative categorical probabilities must sum exactly to one")

    distribution_hash = _hash(
        distribution_hash,
        label="narrative categorical distribution hash",
    )
    stream_hash = derive_stream_hash(
        root_seed=root_seed,
        namespace=namespace,
        step_index=step_index,
        source_hash=source_hash,
        component_hash=component_hash,
        component_key=component_key,
    )
    draw_u64 = draw_u64_for_stream(stream_hash)
    variate = draw_u64 / _MAX_U64
    cumulative = 0.0
    selected = canonical[-1]
    for item in canonical[:-1]:
        cumulative = math.fsum((cumulative, item[1]))
        if variate < cumulative:
            selected = item
            break

    return RandomSampleRecord(
        derivation_version=RANDOM_DERIVATION_VERSION,
        root_seed=root_seed,
        namespace=namespace,
        step_index=step_index,
        source_hash=source_hash,
        component_hash=component_hash,
        component_key=component_key,
        stream_hash=stream_hash,
        draw_index=0,
        draw_u64=draw_u64,
        distribution_hash=distribution_hash,
        selected_outcome_id=selected[0],
        selected_outcome_hash=selected[2],
    )


__all__ = [
    "RANDOM_DERIVATION_VERSION",
    "RandomSampleRecord",
    "derive_stream_hash",
    "draw_u64_for_stream",
    "sample_categorical",
    "validate_root_seed",
]
