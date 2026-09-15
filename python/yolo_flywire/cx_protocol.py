from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


_PROTOCOL_ID = "v2-cx-temporal-gate1-preflight"
_DATASET = "FAFB"
_RELEASE = "v783"
_CELL_TYPES_PRODUCT = "consolidated_cell_types"
_CONNECTIONS_PRODUCT = "connections_princeton"
_CONNECTION_THRESHOLD = 5
_INPUT_FAMILIES = ("ER", "ExR", "PFN")
_CORE_FAMILIES = ("EPG", "PEG", "PEN", "Delta7", "hDelta", "vDelta")
_OUTPUT_FAMILIES = ("PFL",)
_NT_SIGNS = {
    "ACH": 1,
    "OCT": 1,
    "SER": 1,
    "DA": 1,
    "GABA": -1,
    "GLUT": -1,
}
_SEEDS = (7, 11, 19, 23, 31)
_OBSERVATION_RATIOS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
_EARLY_AUC_EFFECT = 0.02
_TEMPORAL_EFFECT = 0.05
_POSITIVE_SEED_COUNT = 4
_DEGREE_REWIRING_ALGORITHM = "cx-directed-double-edge-swap-v1"
_BLOCK_REWIRING_ALGORITHM = "cx-block-preserving-double-edge-swap-v1"
_REWIRING_SWAPS_PER_EDGE = 10
_MAGNITUDE_POLICIES = frozenset({"raw", "log1p"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = frozenset(
    {
        "protocol_id",
        "dataset",
        "release",
        "cell_types_product",
        "connections_product",
        "connection_threshold",
        "input_families",
        "core_families",
        "output_families",
        "nt_signs",
        "seeds",
        "observation_ratios",
        "early_auc_effect",
        "temporal_effect",
        "positive_seed_count",
        "degree_rewiring_algorithm",
        "block_rewiring_algorithm",
        "rewiring_successful_swaps_per_offdiagonal_edge",
        "recurrent_magnitude_policy",
        "retention_observation_ratio",
        "final_test_enabled",
        "source_hashes",
        "dataset_content_hash",
        "split_hash",
        "event_encoder_hash",
        "dynamics_hash",
        "input_node_fingerprint",
        "output_node_fingerprint",
        "cx_artifact_fingerprint",
        "degree_null_fingerprints",
        "block_null_fingerprints",
    }
)


@dataclass(frozen=True)
class CxProtocol:
    protocol_id: str
    dataset: str
    release: str
    cell_types_product: str
    connections_product: str
    connection_threshold: int
    input_families: tuple[str, ...]
    core_families: tuple[str, ...]
    output_families: tuple[str, ...]
    nt_signs: tuple[tuple[str, int], ...]
    seeds: tuple[int, ...]
    observation_ratios: tuple[float, ...]
    early_auc_effect: float
    temporal_effect: float
    positive_seed_count: int
    degree_rewiring_algorithm: str
    block_rewiring_algorithm: str
    rewiring_successful_swaps_per_offdiagonal_edge: int
    recurrent_magnitude_policy: str
    retention_observation_ratio: float
    final_test_enabled: bool
    source_hashes: tuple[tuple[str, str | None], ...]
    dataset_content_hash: str | None
    split_hash: str | None
    event_encoder_hash: str | None
    dynamics_hash: str | None
    input_node_fingerprint: str | None
    output_node_fingerprint: str | None
    cx_artifact_fingerprint: str | None
    degree_null_fingerprints: tuple[tuple[str, str], ...] | None
    block_null_fingerprints: tuple[tuple[str, str], ...] | None

    @property
    def all_families(self) -> tuple[str, ...]:
        return self.input_families + self.core_families + self.output_families

    def role_for_family(self, family: str) -> str:
        if family in self.input_families:
            return "input"
        if family in self.core_families:
            return "core"
        if family in self.output_families:
            return "output"
        raise ValueError(f"unknown CX family: {family!r}")

    def transmitter_sign(self, nt_type: str) -> int:
        for name, sign in self.nt_signs:
            if name == nt_type:
                return sign
        raise ValueError(f"unknown neurotransmitter sign: {nt_type!r}")


def _require_exact_keys(value: dict[str, Any]) -> None:
    keys = set(value)
    missing = sorted(_REQUIRED_FIELDS - keys)
    extra = sorted(keys - _REQUIRED_FIELDS)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError("CX protocol schema mismatch: " + " ".join(details))


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name.replace('_', ' ')} must be a non-empty list")
    if any(type(item) is not str or not item.strip() or item != item.strip() for item in value):
        raise ValueError(f"{name.replace('_', ' ')} must contain canonical non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name.replace('_', ' ')} must be unique")
    return result


def _digest(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be null or a lowercase SHA-256 digest")
    return value


def _fingerprint_map(
    name: str,
    value: Any,
    seeds: tuple[int, ...],
) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be null or a seed-keyed object")
    expected = {str(seed) for seed in seeds}
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly the frozen seed keys")
    return tuple((str(seed), _digest(f"{name}[{seed}]", value[str(seed)])) for seed in seeds)  # type: ignore[arg-type,return-value]


def _exact_float(name: str, value: Any, expected: float) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or float(value) != expected:
        raise ValueError(f"{name.replace('_', ' ')} must remain frozen at {expected}")
    return float(value)


def validate_cx_protocol_dict(value: dict[str, Any]) -> CxProtocol:
    if not isinstance(value, dict):
        raise ValueError("CX protocol must be a JSON object")
    _require_exact_keys(value)

    input_families = _string_tuple("input_families", value["input_families"])
    core_families = _string_tuple("core_families", value["core_families"])
    output_families = _string_tuple("output_families", value["output_families"])
    role_sets = tuple(map(set, (input_families, core_families, output_families)))
    if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("CX family roles must be disjoint")

    frozen_strings = {
        "protocol_id": _PROTOCOL_ID,
        "dataset": _DATASET,
        "release": _RELEASE,
        "cell_types_product": _CELL_TYPES_PRODUCT,
        "connections_product": _CONNECTIONS_PRODUCT,
        "degree_rewiring_algorithm": _DEGREE_REWIRING_ALGORITHM,
        "block_rewiring_algorithm": _BLOCK_REWIRING_ALGORITHM,
    }
    for name, expected in frozen_strings.items():
        if value[name] != expected:
            raise ValueError(f"{name.replace('_', ' ')} must remain frozen at {expected!r}")

    if input_families != _INPUT_FAMILIES:
        raise ValueError("input families must remain frozen")
    if core_families != _CORE_FAMILIES:
        raise ValueError("core families must remain frozen")
    if output_families != _OUTPUT_FAMILIES:
        raise ValueError("output families must remain frozen")

    if type(value["connection_threshold"]) is not int or value["connection_threshold"] != _CONNECTION_THRESHOLD:
        raise ValueError("connection threshold must remain frozen at 5")

    raw_signs = value["nt_signs"]
    if not isinstance(raw_signs, dict) or raw_signs != _NT_SIGNS:
        raise ValueError("neurotransmitter sign policy must remain frozen")
    nt_signs = tuple((name, _NT_SIGNS[name]) for name in sorted(_NT_SIGNS))

    raw_seeds = value["seeds"]
    if (
        not isinstance(raw_seeds, list)
        or any(type(seed) is not int for seed in raw_seeds)
        or tuple(raw_seeds) != _SEEDS
    ):
        raise ValueError("seeds must remain frozen at 7,11,19,23,31")
    seeds = tuple(raw_seeds)

    raw_ratios = value["observation_ratios"]
    if (
        not isinstance(raw_ratios, list)
        or any(type(ratio) not in (int, float) or isinstance(ratio, bool) for ratio in raw_ratios)
        or tuple(float(ratio) for ratio in raw_ratios) != _OBSERVATION_RATIOS
    ):
        raise ValueError("observation ratios must remain frozen")
    observation_ratios = tuple(float(ratio) for ratio in raw_ratios)

    early_auc_effect = _exact_float("early_auc_effect", value["early_auc_effect"], _EARLY_AUC_EFFECT)
    temporal_effect = _exact_float("temporal_effect", value["temporal_effect"], _TEMPORAL_EFFECT)
    if type(value["positive_seed_count"]) is not int or value["positive_seed_count"] != _POSITIVE_SEED_COUNT:
        raise ValueError("positive seed count must remain frozen at 4")
    if (
        type(value["rewiring_successful_swaps_per_offdiagonal_edge"]) is not int
        or value["rewiring_successful_swaps_per_offdiagonal_edge"] != _REWIRING_SWAPS_PER_EDGE
    ):
        raise ValueError("rewiring successful swaps per offdiagonal edge must remain frozen at 10")

    magnitude_policy = value["recurrent_magnitude_policy"]
    if type(magnitude_policy) is not str or magnitude_policy not in _MAGNITUDE_POLICIES:
        raise ValueError("recurrent magnitude policy must be raw or log1p")
    retention_ratio = _exact_float("retention_observation_ratio", value["retention_observation_ratio"], 0.40)

    if type(value["final_test_enabled"]) is not bool:
        raise ValueError("final_test_enabled must be a boolean")

    raw_sources = value["source_hashes"]
    source_names = (_CELL_TYPES_PRODUCT, _CONNECTIONS_PRODUCT)
    if not isinstance(raw_sources, dict) or set(raw_sources) != set(source_names):
        raise ValueError("source_hashes must contain exactly the two frozen Codex products")
    source_hashes = tuple((name, _digest(f"source_hashes[{name}]", raw_sources[name])) for name in source_names)

    dataset_content_hash = _digest("dataset_content_hash", value["dataset_content_hash"])
    split_hash = _digest("split_hash", value["split_hash"])
    event_encoder_hash = _digest("event_encoder_hash", value["event_encoder_hash"])
    dynamics_hash = _digest("dynamics_hash", value["dynamics_hash"])
    input_node_fingerprint = _digest("input_node_fingerprint", value["input_node_fingerprint"])
    output_node_fingerprint = _digest("output_node_fingerprint", value["output_node_fingerprint"])
    cx_artifact_fingerprint = _digest("cx_artifact_fingerprint", value["cx_artifact_fingerprint"])
    degree_null_fingerprints = _fingerprint_map("degree_null_fingerprints", value["degree_null_fingerprints"], seeds)
    block_null_fingerprints = _fingerprint_map("block_null_fingerprints", value["block_null_fingerprints"], seeds)

    if value["final_test_enabled"]:
        required = {
            "dataset_content_hash": dataset_content_hash,
            "split_hash": split_hash,
            "event_encoder_hash": event_encoder_hash,
            "dynamics_hash": dynamics_hash,
            "input_node_fingerprint": input_node_fingerprint,
            "output_node_fingerprint": output_node_fingerprint,
            "cx_artifact_fingerprint": cx_artifact_fingerprint,
            "degree_null_fingerprints": degree_null_fingerprints,
            "block_null_fingerprints": block_null_fingerprints,
        }
        missing = [name for name, item in required.items() if item is None]
        missing.extend(name for name, digest in source_hashes if digest is None)
        if missing:
            raise ValueError("final test provenance is incomplete: " + ", ".join(missing))

    return CxProtocol(
        protocol_id=_PROTOCOL_ID,
        dataset=_DATASET,
        release=_RELEASE,
        cell_types_product=_CELL_TYPES_PRODUCT,
        connections_product=_CONNECTIONS_PRODUCT,
        connection_threshold=_CONNECTION_THRESHOLD,
        input_families=input_families,
        core_families=core_families,
        output_families=output_families,
        nt_signs=nt_signs,
        seeds=seeds,
        observation_ratios=observation_ratios,
        early_auc_effect=early_auc_effect,
        temporal_effect=temporal_effect,
        positive_seed_count=_POSITIVE_SEED_COUNT,
        degree_rewiring_algorithm=_DEGREE_REWIRING_ALGORITHM,
        block_rewiring_algorithm=_BLOCK_REWIRING_ALGORITHM,
        rewiring_successful_swaps_per_offdiagonal_edge=_REWIRING_SWAPS_PER_EDGE,
        recurrent_magnitude_policy=magnitude_policy,
        retention_observation_ratio=retention_ratio,
        final_test_enabled=value["final_test_enabled"],
        source_hashes=source_hashes,
        dataset_content_hash=dataset_content_hash,
        split_hash=split_hash,
        event_encoder_hash=event_encoder_hash,
        dynamics_hash=dynamics_hash,
        input_node_fingerprint=input_node_fingerprint,
        output_node_fingerprint=output_node_fingerprint,
        cx_artifact_fingerprint=cx_artifact_fingerprint,
        degree_null_fingerprints=degree_null_fingerprints,
        block_null_fingerprints=block_null_fingerprints,
    )


def load_cx_protocol(path: str | Path) -> CxProtocol:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read CX protocol: {source}") from exc
    return validate_cx_protocol_dict(raw)
