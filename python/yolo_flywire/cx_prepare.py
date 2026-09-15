from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .cx_development import (
    CxDevelopmentBinding,
    CxDevelopmentBudget,
    development_split_for_subject,
)
from .cx_events import KinematicEventEncoder
from .cx_protocol import CxProtocol
from .cx_skeleton_prep import normalize_ntu_body
from .models.cx_lif import CxDynamics
from .ntu_skeleton import (
    build_gate1_skeleton_manifest,
    parse_skeleton_file,
    select_primary_body,
)


CX_NORMALIZATION_ID = "ntu25-tracking-ge1-linear-interp-root0-center-median-torso0-20-v1"
_DEVELOPMENT_SPLIT_POLICY_ID = "cx-gate1-inner-subject-validation-v1"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CxPrepareConfig:
    actions: tuple[int, ...]
    event_quantile: float
    dynamics: CxDynamics
    budget: CxDevelopmentBudget

    def __post_init__(self) -> None:
        if type(self.actions) is not tuple or not self.actions:
            raise ValueError("actions must be a non-empty tuple")
        if any(type(action) is not int or not 1 <= action <= 120 for action in self.actions):
            raise ValueError("actions must contain integers in 1..120")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if type(self.event_quantile) not in (int, float) or isinstance(self.event_quantile, bool):
            raise ValueError("event_quantile must be numeric")
        quantile = float(self.event_quantile)
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("event_quantile must be finite and inside (0,1)")
        if type(self.dynamics) is not CxDynamics:
            raise ValueError("dynamics must be a CxDynamics")
        if type(self.budget) is not CxDevelopmentBudget:
            raise ValueError("budget must be a CxDevelopmentBudget")


@dataclass(frozen=True)
class CxPreparedDevelopment:
    manifest: dict[str, Any]
    development_split_hash: str
    encoder: KinematicEventEncoder
    dynamics_hash: str
    binding: CxDevelopmentBinding
    counts: tuple[tuple[str, int], ...]
    normalization_id: str
    preparation_hash: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dynamics_payload(dynamics: CxDynamics) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "cx_lif_dynamics",
        "tau_membrane": float(dynamics.tau_membrane),
        "synaptic_decay": float(dynamics.synaptic_decay),
        "refractory_steps": dynamics.refractory_steps,
        "threshold": float(dynamics.threshold),
        "reset": float(dynamics.reset),
        "recurrent_delay_steps": dynamics.recurrent_delay_steps,
        "recurrent_gain": float(dynamics.recurrent_gain),
        "magnitude_policy": dynamics.magnitude_policy,
    }


def _required_protocol_pins(protocol: CxProtocol) -> None:
    scalar = {
        "input_node_fingerprint": protocol.input_node_fingerprint,
        "output_node_fingerprint": protocol.output_node_fingerprint,
        "cx_artifact_fingerprint": protocol.cx_artifact_fingerprint,
    }
    missing = [name for name, value in scalar.items() if value is None]
    if protocol.degree_null_fingerprints is None:
        missing.append("degree_null_fingerprints")
    if protocol.block_null_fingerprints is None:
        missing.append("block_null_fingerprints")
    if missing:
        raise ValueError("CX development preparation requires measured graph pins: " + ", ".join(missing))


def _development_assignments(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[tuple[str, int], ...]]:
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("NTU skeleton manifest contains no samples")

    assignments: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0, "final_test": 0}
    for row in raw_samples:
        if not isinstance(row, dict):
            raise ValueError("NTU skeleton manifest contains a malformed sample")
        try:
            sample_id = str(row["sample_id"])
            subject = int(row["subject"])
            body_id = int(row["selected_body_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("NTU skeleton manifest sample identity is malformed") from exc
        split = development_split_for_subject(subject)
        counts[split] += 1
        assignments.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "development_split": split,
                "selected_body_id": body_id,
            }
        )

    if counts["train"] == 0:
        raise ValueError("CX development preparation requires at least one train sample")
    if counts["validation"] == 0:
        raise ValueError("CX development preparation requires at least one validation sample")

    ordered_counts = tuple((name, counts[name]) for name in ("train", "validation", "final_test"))
    return assignments, ordered_counts


def _selected_normalized_sequence(root: Path, row: dict[str, Any]) -> Any:
    relative = row.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("NTU skeleton manifest sample has invalid relative_path")
    source = root / relative
    sample = parse_skeleton_file(source)
    expected_sample_id = row.get("sample_id")
    if sample.sample_id != expected_sample_id:
        raise ValueError("NTU skeleton sample identity changed after manifest construction")
    selected = select_primary_body(sample)
    expected_body_id = row.get("selected_body_id")
    if selected.body_id != expected_body_id:
        raise ValueError("NTU primary-body selection changed after manifest construction")
    return normalize_ntu_body(selected)


def _write_atomic(output: Path, payloads: dict[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"CX prepared output already exists: {output}")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"CX prepared temporary output already exists: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in payloads.items():
            (temporary / name).write_bytes(data)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_development_input(
    protocol: CxProtocol,
    config: CxPrepareConfig,
    skeleton_root: str | Path,
    output_dir: str | Path,
) -> CxPreparedDevelopment:
    if type(protocol) is not CxProtocol:
        raise ValueError("protocol must be a CxProtocol")
    if type(config) is not CxPrepareConfig:
        raise ValueError("config must be a CxPrepareConfig")
    if config.dynamics.magnitude_policy != protocol.recurrent_magnitude_policy:
        raise ValueError(
            "dynamics magnitude_policy does not match the frozen protocol recurrent_magnitude_policy"
        )
    _required_protocol_pins(protocol)

    root = Path(skeleton_root).absolute()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"CX prepared output already exists: {output}")

    manifest = build_gate1_skeleton_manifest(root, actions=config.actions)
    assignments, counts = _development_assignments(manifest)
    development_split_payload = {
        "format_version": 1,
        "kind": "cx_gate1_development_split",
        "policy_id": _DEVELOPMENT_SPLIT_POLICY_ID,
        "outer_split_hash": manifest["split_hash"],
        "dataset_content_hash": manifest["dataset_content_hash"],
        "actions": list(config.actions),
        "validation_subjects": [
            subject
            for subject in range(1, 107)
            if development_split_for_subject(subject) == "validation"
        ],
        "assignments": assignments,
    }
    development_split_hash = _hash_json(development_split_payload)

    assignment_by_sample = {row["sample_id"]: row["development_split"] for row in assignments}
    train_sequences = []
    validation_rows: list[dict[str, Any]] = []
    for row in manifest["samples"]:
        split = assignment_by_sample[row["sample_id"]]
        if split == "train":
            train_sequences.append(_selected_normalized_sequence(root, row))
        elif split == "validation":
            validation_rows.append(row)
        elif split != "final_test":
            raise ValueError(f"unknown CX development split: {split!r}")
        # Sealed final-test rows are deliberately not normalized or transformed.

    encoder = KinematicEventEncoder.fit(train_sequences, quantile=float(config.event_quantile))
    # Validation may be inspected for shape/normalization compatibility, but it is
    # strictly downstream of the frozen training-only threshold fit.
    for row in validation_rows:
        encoder.transform(_selected_normalized_sequence(root, row))

    dynamics_payload = _dynamics_payload(config.dynamics)
    dynamics_hash = _hash_json(dynamics_payload)

    degree = tuple((int(seed), digest) for seed, digest in protocol.degree_null_fingerprints or ())
    block = tuple((int(seed), digest) for seed, digest in protocol.block_null_fingerprints or ())
    binding = CxDevelopmentBinding(
        dataset_content_hash=manifest["dataset_content_hash"],
        split_hash=development_split_hash,
        event_encoder_hash=encoder.fingerprint,
        dynamics_hash=dynamics_hash,
        input_node_fingerprint=protocol.input_node_fingerprint or "",
        output_node_fingerprint=protocol.output_node_fingerprint or "",
        real_graph_fingerprint=protocol.cx_artifact_fingerprint or "",
        degree_graph_fingerprints=degree,
        block_graph_fingerprints=block,
        budget=config.budget,
        seeds=protocol.seeds,
    )

    prepare_payload = {
        "format_version": _FORMAT_VERSION,
        "kind": "cx_gate1_development_input_preparation",
        "protocol_id": protocol.protocol_id,
        "normalization_id": CX_NORMALIZATION_ID,
        "actions": list(config.actions),
        "event_quantile": float(config.event_quantile),
        "dataset_content_hash": manifest["dataset_content_hash"],
        "outer_split_hash": manifest["split_hash"],
        "development_split_hash": development_split_hash,
        "event_encoder_hash": encoder.fingerprint,
        "dynamics_hash": dynamics_hash,
        "counts": {name: count for name, count in counts},
        "final_test_normalized": False,
        "final_test_transformed": False,
        "dynamics": dynamics_payload,
        "budget": asdict(config.budget),
    }
    preparation_hash = _hash_json(prepare_payload)
    prepare_payload["preparation_hash"] = preparation_hash

    payloads = {
        "input_manifest.json": (_canonical_json(manifest) + "\n").encode("utf-8"),
        "encoder.json": encoder.to_json().encode("utf-8"),
        "binding.json": (_canonical_json(asdict(binding)) + "\n").encode("utf-8"),
        "prepare.json": (_canonical_json(prepare_payload) + "\n").encode("utf-8"),
    }
    _write_atomic(output, payloads)

    return CxPreparedDevelopment(
        manifest=manifest,
        development_split_hash=development_split_hash,
        encoder=encoder,
        dynamics_hash=dynamics_hash,
        binding=binding,
        counts=counts,
        normalization_id=CX_NORMALIZATION_ID,
        preparation_hash=preparation_hash,
    )
