"""R2.M4c.0 native metric units: a read-only view, not another model owner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json

from narrative_dynamics.abm.model_family_comparison import NativeModelFamilyComparison

SCHEMA = "r2-m4c0-native-metric-eligibility-v1"
CONTRACTS = {
    "bb_fitness_growth": {
        "metric_id": "undirected_birth_topology_density",
        "owner": "BBRuntimeReplay",
        "unit": "occupied_unordered_pair_per_eligible_unordered_pair",
        "denominator_kind": "ALL_POSSIBLE_UNORDERED_PAIRS_IN_CURRENT_BB_ROSTER",
        "population_scope": "BORN_BB_NODES_AT_NATIVE_TICK",
        "phase": "AFTER_BB_TICK",
        "clock_kind": "GLOBAL_TICK_NOT_BIRTH_INDEX",
    },
    "directed_rewiring": {
        "metric_id": "directed_candidate_activation_fraction",
        "owner": "RewiringPopulationState",
        "unit": "active_directed_candidate_per_declared_directed_candidate",
        "denominator_kind": "FIXED_DIRECTED_CANDIDATE_CATALOG",
        "population_scope": "REWIRING_FIXED_AGENT_CATALOG",
        "phase": "AFTER_REWIRING_NOT_PROPAGATION",
        "clock_kind": "NATIVE_REWIRING_ROUND",
    },
    "lifecycle_feedback": {
        "metric_id": "active_catalog_membership_fraction",
        "owner": "EvolvingPopulationState",
        "unit": "active_agent_per_catalog_member",
        "denominator_kind": "ALL_NATIVE_CATALOG_MEMBERS_INCLUDING_INACTIVE",
        "population_scope": "EVOLVING_AGENT_CATALOG_ALL_STATUSES",
        "phase": "AFTER_LIFECYCLE_FEEDBACK_REWIRING",
        "clock_kind": "NATIVE_EVOLVING_ROUND",
    },
}
PAIR_REASONS = {
    "bb_vs_rewiring": "UNDIRECTED_BIRTH_PAIRS_VS_DIRECTED_CANDIDATE_CATALOG",
    "bb_vs_evolving": "UNDIRECTED_BIRTH_PAIRS_VS_ACTIVE_MEMBERSHIP",
    "rewiring_vs_evolving": "DIRECTED_CANDIDATE_ACTIVATION_VS_MEMBER_LIFECYCLE",
    "any_vs_v19": "NO_REGISTERED_V19_SOURCE_RELATION_ROSTER_OR_OBSERVER_MAPPING",
}


@dataclass(frozen=True)
class NativeMetric:
    family: str
    metric_id: str
    owner: str
    unit: str
    denominator_kind: str
    population_scope: str
    phase: str
    clock_kind: str
    native_time: int
    run_source_hash: str
    frame_source_hash: str
    numerator: int
    denominator: int
    exact_fraction: str

    def __post_init__(self) -> None:
        contract = CONTRACTS.get(self.family)
        if contract is None or any(getattr(self, k) != v for k, v in contract.items()):
            raise ValueError("native metric unit/owner/scope retyping is forbidden")
        if (type(self.native_time) is not int or self.native_time < 1
                or type(self.numerator) is not int or type(self.denominator) is not int
                or self.denominator <= 0 or not 0 <= self.numerator <= self.denominator
                or self.exact_fraction != str(Fraction(self.numerator, self.denominator))):
            raise ValueError("native metric denominator, clock or exact fraction is invalid")
        for digest in (self.run_source_hash, self.frame_source_hash):
            if type(digest) is not str or len(digest) != 71 or not digest.startswith("sha256:"):
                raise ValueError("native metric requires exact source revision hash")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def metric_pair_eligibility(
    left: NativeMetric, right: NativeMetric, *,
    mapping_version: str | None = None,
) -> str:
    """A SUPPORTED result is descriptive within-run only, not causal."""
    if not isinstance(left, NativeMetric) or not isinstance(right, NativeMetric):
        raise TypeError("metric eligibility requires typed native records")
    if mapping_version is not None:
        raise ValueError("MAPPED_WITH_ASSUMPTIONS requires an approved mapping; none exists")
    if (left.family == right.family and left.run_source_hash == right.run_source_hash
            and all(getattr(left, k) == getattr(right, k) for k in CONTRACTS[left.family])):
        return "SUPPORTED"
    return "NOT_COMPARABLE"


def _make(family: str, time: int, run: str, frame: str,
          numerator: int, denominator: int) -> NativeMetric:
    return NativeMetric(family=family, **CONTRACTS[family], native_time=time,
                        run_source_hash=run, frame_source_hash=frame,
                        numerator=numerator, denominator=denominator,
                        exact_fraction=str(Fraction(numerator, denominator)))


def _unique_directed(edges: object) -> int:
    if type(edges) is not list:
        raise ValueError("directed candidate list is required")
    found = set()
    for edge in edges:
        if (type(edge) is not list or len(edge) != 3
                or any(type(x) is not str or not x for x in edge)
                or edge[0] == edge[1]):
            raise ValueError("malformed directed candidate edge")
        found.add(tuple(edge))
    if len(found) != len(edges):
        raise ValueError("duplicate directed candidate edge")
    return len(found)


def _bb_frames(f: dict) -> list[NativeMetric]:
    records = []
    for step in f["steps"]:
        roster = step["native_agent_ids"]
        edges = step["native_undirected_topology"]
        if (type(roster) is not list or len(roster) < 2
                or any(type(x) is not str or not x for x in roster)
                or len(set(roster)) != len(roster) or type(edges) is not list):
            raise ValueError("invalid BB native roster/topology")
        unique = set()
        for edge in edges:
            if (type(edge) is not list or len(edge) != 2 or edge[0] == edge[1]
                    or any(type(x) is not str or x not in roster for x in edge)):
                raise ValueError("invalid BB undirected edge")
            unique.add(frozenset(edge))
        if len(unique) != len(edges):
            raise ValueError("reversed or repeated BB undirected edge")
        records.append(_make("bb_fitness_growth", step["native_tick"],
                             f["initial_frame_hash"], step["next_frame_hash"],
                             len(unique), len(roster) * (len(roster) - 1) // 2))
    return records


def _rewiring_frames(f: dict) -> list[NativeMetric]:
    denominator = f["candidate_count"]
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("candidate catalog denominator must be positive")
    records = []
    for step in f["steps"]:
        prior = step["prior_active_candidate_edges"]
        post = step["next_active_candidate_edges"]
        if (_unique_directed(prior) > denominator
                or _unique_directed(post) > denominator):
            raise ValueError("active rewiring edges exceed candidate catalog")
        before = {tuple(edge) for edge in prior}
        if any(tuple(tx) not in before for tx in step["native_transmissions"]):
            raise ValueError("rewiring transmitted outside prior directed topology")
        records.append(_make("directed_rewiring", step["round"],
                             f["initial_state_hash"], step["next_state_hash"],
                             len(post), denominator))
    return records


def _evolving_frames(f: dict) -> list[NativeMetric]:
    records = []
    for step in f["steps"]:
        members = step["native_members_after"]
        if (type(members) is not list or not members
                or any(type(m) is not dict or type(m.get("agent_id")) is not str
                       or type(m.get("status")) is not str for m in members)):
            raise ValueError("evolving metric requires actual native catalog")
        ids = [m["agent_id"] for m in members]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate evolving catalog member")
        active = sum(m["status"] == "active" for m in members)
        records.append(_make("lifecycle_feedback", step["round"],
                             f["initial_state_hash"], step["next_state_hash"],
                             active, len(members)))
    return records


@dataclass(frozen=True)
class NativeMetricProtocol:
    """Audited immutable metric registration; no universal conversion."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("protocol must store immutable canonical JSON")
        d = json.loads(self._canonical_json)
        expected = {k: {"status": "NOT_COMPARABLE", "reason": v}
                    for k, v in PAIR_REASONS.items()}
        if (type(d) is not dict or d.get("schema") != SCHEMA
                or d.get("source_owner") != "READ_ONLY_R2_M4B_PROJECTION"
                or d.get("normalization") != "NONE"
                or d.get("cross_family_numeric_ranking") != "FORBIDDEN"
                or d.get("causal_attribution") != "UNATTRIBUTED"
                or d.get("sample_size_per_native_run") != 1
                or d.get("confidence_interval") != "NOT_ESTIMABLE"
                or d.get("external_validation") != "NO_EXTERNAL_VALIDATION"
                or type(d.get("source_report_hash")) is not str
                or not d["source_report_hash"].startswith("sha256:")
                or type(d.get("families")) is not dict
                or set(d["families"]) != set(CONTRACTS)
                or d.get("comparison_eligibility") != expected
                or type(d.get("v19_reference")) is not dict
                or d["v19_reference"].get("metric_status") !=
                   "NOT_MEASURABLE_FROM_M4B_REFERENCE"):
            raise ValueError("cross-family metric claim or provenance is invalid")
        for family, rows in d["families"].items():
            if type(rows) is not list or not rows:
                raise ValueError("native family has no metric frames")
            parsed = [NativeMetric(**row) for row in rows]
            times = [m.native_time for m in parsed]
            if (any(m.family != family for m in parsed)
                    or times != sorted(set(times))
                    or len({m.run_source_hash for m in parsed}) != 1):
                raise ValueError("native metric frames lost run/clock/family identity")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)


def build_native_metric_protocol(source: NativeModelFamilyComparison) -> NativeMetricProtocol:
    if not isinstance(source, NativeModelFamilyComparison):
        raise TypeError("real native R2.M4b source replay is required")
    data = source.to_dict()
    if data["normalization"] != "NONE" or data["cross_family_numeric_ranking"] != "FORBIDDEN":
        raise ValueError("native source disallows numeric ranking")
    fam = data["families"]
    metrics = {
        "bb_fitness_growth": _bb_frames(fam["bb_fitness_growth"]),
        "directed_rewiring": _rewiring_frames(fam["directed_rewiring"]),
        "lifecycle_feedback": _evolving_frames(fam["lifecycle_feedback"]),
    }
    v19 = data["v19_reference"]
    root = {
        "schema": SCHEMA,
        "source_owner": "READ_ONLY_R2_M4B_PROJECTION",
        "source_report_hash": source.content_hash,
        "normalization": "NONE",
        "cross_family_numeric_ranking": "FORBIDDEN",
        "causal_attribution": "UNATTRIBUTED",
        "sample_size_per_native_run": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "families": {k: [m.to_dict() for m in rows] for k, rows in metrics.items()},
        "comparison_eligibility": {
            k: {"status": "NOT_COMPARABLE", "reason": v}
            for k, v in PAIR_REASONS.items()
        },
        "v19_reference": {
            "source_owner": v19["source_owner"],
            "story_lineage": v19["story_lineage"],
            "control_story_hash": v19["control_story_hash"],
            "intervention_story_hash": v19["intervention_story_hash"],
            "metric_status": "NOT_MEASURABLE_FROM_M4B_REFERENCE",
        },
    }
    return NativeMetricProtocol(
        json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
