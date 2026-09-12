from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from statistics import mean


@dataclass(frozen=True)
class RunManifest:
    code_commit: str
    dataset_id: str
    split_hash: str
    observation_schema_hash: str
    model_family: str
    topology_fingerprint: str
    seed: int
    budget: dict[str, int]
    training_config_hash: str
    metrics: dict[str, float]
    compared_families: tuple[str, ...]
    claim: str

    @classmethod
    def example(cls, **changes: object) -> "RunManifest":
        values: dict[str, object] = {
            "code_commit": "example",
            "dataset_id": "synthetic-v0",
            "split_hash": "split",
            "observation_schema_hash": "schema",
            "model_family": "flywire",
            "topology_fingerprint": "topology",
            "seed": 7,
            "budget": {"epochs": 1, "max_updates": 1, "parameter_ceiling": 10000},
            "training_config_hash": "config",
            "metrics": {},
            "compared_families": ("flywire", "rewired"),
            "claim": "topology_specific_advantage",
        }
        values.update(changes)
        return cls(**values)  # type: ignore[arg-type]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def validate_claim_boundary(self) -> None:
        if self.claim == "topology_specific_advantage":
            families = set(self.compared_families)
            if "flywire" not in families or "rewired" not in families:
                raise ValueError("topology-specific claim requires matched flywire and rewired controls")


@dataclass(frozen=True)
class TopologyEvidenceReport:
    conclusion: str
    mean_paired_difference: float
    paired_differences: dict[int, float]
    success_threshold: float


def aggregate_topology_evidence(
    *,
    seed_metrics: dict[int, dict[str, float]],
    success_threshold: float,
) -> TopologyEvidenceReport:
    """Aggregate only paired FlyWire-vs-rewired evidence across identical seeds."""
    if success_threshold < 0.0:
        raise ValueError("success_threshold must be non-negative")
    if not seed_metrics:
        raise ValueError("topology evidence requires at least one paired seed")

    paired: dict[int, float] = {}
    for seed, metrics in sorted(seed_metrics.items()):
        if "flywire" not in metrics or "rewired" not in metrics:
            raise ValueError(f"seed {seed} is missing paired flywire/rewired metrics")
        flywire = float(metrics["flywire"])
        rewired = float(metrics["rewired"])
        paired[int(seed)] = flywire - rewired

    mean_difference = mean(paired.values())
    conclusion = (
        "topology_advantage_supported"
        if mean_difference >= success_threshold
        else "topology_advantage_not_established"
    )
    return TopologyEvidenceReport(
        conclusion=conclusion,
        mean_paired_difference=float(mean_difference),
        paired_differences=paired,
        success_threshold=float(success_threshold),
    )
