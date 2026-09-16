"""Claim-by-claim orchestration for proof-backed analysis."""

from __future__ import annotations

from fractions import Fraction
import hashlib
from types import MappingProxyType

from .certificate import Certificate, CertificateBuilder, candidate_global_interior
from .model import ConstantSchedule, NamedSchedule, PathModel, PiecewiseSchedule
from .named_schedules import fixed_fixture_route, resolve_named_schedule
from .result import AnalysisResult, ClaimResult, ClaimStatus, ProvenanceMismatchError
from .runner import LeanCertificateRunner


_CLAIM_ORDER = (
    "parameters_valid",
    "initial_all_broadcast",
    "exposure_law",
    "effective_alpha_lookup",
    "reachable_interior",
    "path2_consensus",
    "pathn_consensus_exists",
    "consensus_value_known",
)


def _fraction(value) -> Fraction:
    return Fraction(value.numerator, value.denominator)


def _parameters_valid_candidate(model: PathModel) -> bool:
    threshold = _fraction(model.threshold)
    if not (0 <= threshold <= 1):
        return False
    schedule = model.schedule
    if isinstance(schedule, ConstantSchedule):
        values = (schedule.value,)
    elif isinstance(schedule, PiecewiseSchedule):
        values = (schedule.default, *(value for _, value in schedule.points))
    elif isinstance(schedule, NamedSchedule):
        resolve_named_schedule(schedule.schedule_id)
        return True
    else:  # closed AST; defensive only
        return False
    return all(0 <= _fraction(value) <= 1 for value in values)


def _all_broadcast_candidate(model: PathModel) -> bool:
    threshold = _fraction(model.threshold)
    return all(threshold <= _fraction(belief) <= 1 for belief in model.beliefs)


def _unknown(claim_id: str, note: str) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=ClaimStatus.UNKNOWN,
        theorem=None,
        assumptions=(),
        exact_values={},
        note=note,
    )


def _compile_claims(certificate: Certificate, runner: LeanCertificateRunner) -> tuple[ClaimResult, ...]:
    evidence = runner.compile(certificate)
    expected = hashlib.sha256(certificate.source.encode("utf-8")).hexdigest()
    if evidence.certificate_digest != expected:
        raise ProvenanceMismatchError(
            "compiled certificate digest does not match generated source"
        )
    return tuple(
        ClaimResult(
            claim_id=claim.claim_id,
            status=claim.expected_status,
            theorem=claim.theorem,
            assumptions=claim.assumptions,
            exact_values=claim.exact_values,
            note=None,
        )
        for claim in certificate.claims
    )


def _merge_claims(target: dict[str, ClaimResult], claims: tuple[ClaimResult, ...]) -> None:
    for claim in claims:
        existing = target.get(claim.claim_id)
        if existing is not None and existing != claim:
            raise ProvenanceMismatchError(
                f"conflicting compiled evidence for claim {claim.claim_id}"
            )
        target[claim.claim_id] = claim


def analyze_model(
    model: PathModel,
    *,
    runner: LeanCertificateRunner,
) -> AnalysisResult:
    """Analyze one normalized model without treating Python as proof authority."""
    builder = CertificateBuilder()
    claims: dict[str, ClaimResult] = {}

    parameters_valid = _parameters_valid_candidate(model)
    all_broadcast = _all_broadcast_candidate(model)

    if parameters_valid and all_broadcast:
        structural = builder.build_structural_positive(model, (
            "parameters_valid",
            "initial_all_broadcast",
            "exposure_law",
            "effective_alpha_lookup",
        ))
        _merge_claims(claims, _compile_claims(structural, runner))
    else:
        if parameters_valid:
            certificate = builder.build_structural_positive(model, ("parameters_valid",))
        else:
            certificate = builder.build_structural_negative(model, ("parameters_valid",))
        _merge_claims(claims, _compile_claims(certificate, runner))

        if all_broadcast:
            certificate = builder.build_structural_positive(model, ("initial_all_broadcast",))
        else:
            certificate = builder.build_structural_negative(model, ("initial_all_broadcast",))
        _merge_claims(claims, _compile_claims(certificate, runner))

        claims["exposure_law"] = _unknown(
            "exposure_law",
            "all-broadcast exposure theorem prerequisites are not all certified",
        )
        claims["effective_alpha_lookup"] = _unknown(
            "effective_alpha_lookup",
            "post-incoming lookup theorem prerequisites are not all certified",
        )

    if parameters_valid and all_broadcast:
        eps = candidate_global_interior(model)
        if eps is not None:
            certificate = builder.build_pathn_consensus(model, eps)
            _merge_claims(claims, _compile_claims(certificate, runner))
        else:
            claims["reachable_interior"] = _unknown(
                "reachable_interior",
                "no supported positive global-interior certificate route",
            )
            claims["pathn_consensus_exists"] = _unknown(
                "pathn_consensus_exists",
                "no supported theorem route establishes finite-Path consensus",
            )
    else:
        claims["reachable_interior"] = _unknown(
            "reachable_interior",
            "generic interior theorem prerequisites are not all certified",
        )
        claims["pathn_consensus_exists"] = _unknown(
            "pathn_consensus_exists",
            "generic consensus theorem prerequisites are not all certified",
        )

    if model.n == 2:
        route = fixed_fixture_route(model)
        if route is not None:
            certificate = builder.build_named_path2(model, route)
            _merge_claims(claims, _compile_claims(certificate, runner))
        else:
            claims["path2_consensus"] = _unknown(
                "path2_consensus",
                "no supported exact Path2 product-limit or fixed-fixture theorem route",
            )

    if "consensus_value_known" not in claims:
        claims["consensus_value_known"] = _unknown(
            "consensus_value_known",
            "no applicable theorem identifies the common limit in closed form",
        )

    ordered = tuple(
        claims[claim_id]
        for claim_id in _CLAIM_ORDER
        if claim_id in claims and (claim_id != "path2_consensus" or model.n == 2)
    )
    schedule_kind = (
        "constant"
        if isinstance(model.schedule, ConstantSchedule)
        else "piecewise"
        if isinstance(model.schedule, PiecewiseSchedule)
        else f"named:{model.schedule.schedule_id}"
    )
    return AnalysisResult(
        model_summary=MappingProxyType({
            "topology": "path",
            "n": model.n,
            "schedule": schedule_kind,
            "exposure_semantics": "post-incoming lookup",
        }),
        claims=ordered,
    )
