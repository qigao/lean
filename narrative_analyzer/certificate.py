"""Deterministic closed-template Lean certificate generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType

from .model import (
    ConstantSchedule,
    ExactRat,
    NamedSchedule,
    PathModel,
    PiecewiseSchedule,
)
from .named_schedules import LEAN_NAMESPACE, resolve_named_schedule
from .result import (
    CertificateGenerationError,
    ClaimStatus,
)


_PRODUCTION_IMPORT = "NarrativeDynamics.Core.FitnessABMPathNExposureConvergence"
_EXPOSURE_NS = "NarrativeDynamics.FitnessABMPathNExposure"
_CONVERGENCE_NS = "NarrativeDynamics.FitnessABMPathNExposureConvergence"
_STRUCTURAL_ORDER = (
    "parameters_valid",
    "initial_all_broadcast",
    "exposure_law",
    "effective_alpha_lookup",
)
_NEGATIVE_STRUCTURAL = ("parameters_valid", "initial_all_broadcast")
_NAMED_VALID_THEOREMS = {
    "slowZeroSchedule": f"{LEAN_NAMESPACE}.slowZeroSchedule_valid",
    "nearOneSchedule": f"{LEAN_NAMESPACE}.nearOneSchedule_valid",
    "harmonicSchedule": f"{LEAN_NAMESPACE}.harmonicSchedule_valid",
}


@dataclass(frozen=True)
class CertificateClaim:
    claim_id: str
    expected_status: ClaimStatus
    theorem: str
    assumptions: tuple[str, ...]
    exact_values: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("certificate claim id must not be empty")
        if self.expected_status is ClaimStatus.UNKNOWN:
            raise ValueError("certificates cannot claim UNKNOWN")
        if not self.theorem.strip():
            raise ValueError("certificate claim requires theorem provenance")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(
            self, "exact_values", MappingProxyType(dict(self.exact_values))
        )


@dataclass(frozen=True)
class Certificate:
    source: str
    claims: tuple[CertificateClaim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))


def _fraction(q: ExactRat) -> Fraction:
    return Fraction(q.numerator, q.denominator)


def _render_rat(q: ExactRat) -> str:
    if q.denominator == 1:
        return f"({q.numerator} : Rat)"
    return f"(({q.numerator} : Rat) / {q.denominator})"


def _schedule_data(model: PathModel) -> object:
    schedule = model.schedule
    if isinstance(schedule, ConstantSchedule):
        return ["constant", schedule.value.numerator, schedule.value.denominator]
    if isinstance(schedule, PiecewiseSchedule):
        return [
            "piecewise",
            [schedule.default.numerator, schedule.default.denominator],
            [
                [exposure, value.numerator, value.denominator]
                for exposure, value in schedule.points
            ],
        ]
    if isinstance(schedule, NamedSchedule):
        entry = resolve_named_schedule(schedule.schedule_id)
        return ["named", entry.schedule_id]
    raise CertificateGenerationError(f"unsupported schedule AST: {schedule!r}")


def _normalized_bytes(model: PathModel) -> bytes:
    payload = {
        "n": model.n,
        "beliefs": [[q.numerator, q.denominator] for q in model.beliefs],
        "exposures": list(model.exposures),
        "threshold": [model.threshold.numerator, model.threshold.denominator],
        "schedule": _schedule_data(model),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _model_digest(model: PathModel) -> str:
    return hashlib.sha256(_normalized_bytes(model)).hexdigest()[:12]


def _render_schedule(model: PathModel) -> str:
    schedule = model.schedule
    if isinstance(schedule, ConstantSchedule):
        return f"fun _ => {_render_rat(schedule.value)}"
    if isinstance(schedule, PiecewiseSchedule):
        body = _render_rat(schedule.default)
        for exposure, value in reversed(schedule.points):
            body = f"if e = {exposure} then {_render_rat(value)} else {body}"
        return f"fun e => {body}"
    if isinstance(schedule, NamedSchedule):
        entry = resolve_named_schedule(schedule.schedule_id)
        return f"{entry.lean_definition}.receptivityAt"
    raise CertificateGenerationError(f"unsupported schedule AST: {schedule!r}")


def _render_state(model: PathModel) -> str:
    entries = ", ".join(
        f"⟨{_render_rat(belief)}, {exposure}⟩"
        for belief, exposure in zip(model.beliefs, model.exposures, strict=True)
    )
    return f"![{entries}]"


def _header() -> list[str]:
    return [
        f"import {_PRODUCTION_IMPORT}",
        "",
        "namespace NarrativeAnalyzerCertificate",
        "",
        "open NarrativeDynamics",
        "open NarrativeDynamics.FitnessABMPathNExposure",
        "open NarrativeDynamics.FitnessABMPathNExposureConvergence",
        "",
    ]


def _definitions(model: PathModel, digest: str) -> tuple[list[str], str, str]:
    params = f"AnalyzerParams_{digest}"
    state = f"AnalyzerState_{digest}"
    lines = [
        f"private def {params} : ExposureParameters :=",
        f"  ⟨{_render_schedule(model)}, {_render_rat(model.threshold)}⟩",
        "",
        f"private def {state} : State {model.n} := {_render_state(model)}",
        "",
    ]
    return lines, params, state


def _claim_order(claims: Iterable[str], allowed: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(claims)
    if len(requested) != len(set(requested)):
        raise ValueError("certificate claim ids must be unique")
    unknown = [claim for claim in requested if claim not in allowed]
    if unknown:
        raise ValueError(f"unsupported certificate claim: {unknown[0]}")
    return requested


def _validity_lines(model: PathModel, params: str, lemma: str) -> list[str]:
    schedule = model.schedule
    lines = [f"private theorem {lemma} : {params}.Valid := by", "  constructor"]
    if isinstance(schedule, ConstantSchedule):
        lines += [
            "  · intro e",
            f"    norm_num [{params}]",
        ]
    elif isinstance(schedule, PiecewiseSchedule):
        lines += [
            "  · intro e",
            f"    simp only [{params}]",
            "    split_ifs <;> norm_num",
        ]
    elif isinstance(schedule, NamedSchedule):
        entry = resolve_named_schedule(schedule.schedule_id)
        valid_theorem = _NAMED_VALID_THEOREMS[entry.schedule_id]
        lines += [
            "  · intro e",
            f"    exact ({valid_theorem}.1 e)",
        ]
    else:
        raise CertificateGenerationError(f"unsupported schedule AST: {schedule!r}")
    lines += [
        f"  · norm_num [{params}]",
        "",
    ]
    return lines


def _all_broadcast_lines(params: str, state: str, lemma: str) -> list[str]:
    return [
        f"private theorem {lemma} : allBroadcast {params} {state} := by",
        "  intro i",
        f"  fin_cases i <;> norm_num [allBroadcast, {params}, {state}]",
        "",
    ]


def _schedule_values(model: PathModel) -> tuple[tuple[int | None, ExactRat], ...]:
    schedule = model.schedule
    if isinstance(schedule, ConstantSchedule):
        return ((0, schedule.value),)
    if isinstance(schedule, PiecewiseSchedule):
        values: list[tuple[int | None, ExactRat]] = list(schedule.points)
        used = {exposure for exposure, _ in schedule.points}
        default_witness = 0
        while default_witness in used:
            default_witness += 1
        values.append((default_witness, schedule.default))
        return tuple(values)
    if isinstance(schedule, NamedSchedule):
        resolve_named_schedule(schedule.schedule_id)
        return ()
    raise CertificateGenerationError(f"unsupported schedule AST: {schedule!r}")


def _parameter_invalid_witness(model: PathModel) -> tuple[str, int | None] | None:
    threshold = _fraction(model.threshold)
    if threshold < 0 or threshold > 1:
        return ("threshold", None)
    for exposure, value in _schedule_values(model):
        q = _fraction(value)
        if q < 0 or q > 1:
            return ("schedule", exposure)
    return None


def _all_broadcast_invalid_index(model: PathModel) -> int | None:
    threshold = _fraction(model.threshold)
    for index, belief in enumerate(model.beliefs):
        q = _fraction(belief)
        if threshold > q or q > 1:
            return index
    return None


class CertificateBuilder:
    def build_structural_positive(
        self,
        model: PathModel,
        claims: Iterable[str],
    ) -> Certificate:
        requested = _claim_order(claims, _STRUCTURAL_ORDER)
        digest = _model_digest(model)
        lines = _header()
        definitions, params, state = _definitions(model, digest)
        lines += definitions

        valid_lemma = f"AnalyzerParamsValid_{digest}"
        broadcast_lemma = f"AnalyzerAllBroadcast_{digest}"
        exposure_lemma = f"AnalyzerExposureLaw_{digest}"
        alpha_lemma = f"AnalyzerEffectiveAlphaLookup_{digest}"

        needs_valid = any(
            claim in requested
            for claim in ("parameters_valid", "exposure_law", "effective_alpha_lookup")
        )
        needs_broadcast = any(
            claim in requested
            for claim in (
                "initial_all_broadcast",
                "exposure_law",
                "effective_alpha_lookup",
            )
        )
        if needs_valid:
            lines += _validity_lines(model, params, valid_lemma)
        if needs_broadcast:
            lines += _all_broadcast_lines(params, state, broadcast_lemma)

        if "exposure_law" in requested:
            lines += [
                f"private theorem {exposure_lemma} (k : Nat) (i : Fin {model.n}) :",
                f"    (((step {params} {model.n})^[k] {state}) i).exposure =",
                f"      ({state} i).exposure + k * FitnessABMPathN.degree {model.n} i := by",
                "  exact exposure_iterate",
                f"    {params} {valid_lemma} {model.n} (by norm_num) {state}",
                f"    {broadcast_lemma} k i",
                "",
            ]

        if "effective_alpha_lookup" in requested:
            lines += [
                f"private theorem {alpha_lemma} (k : Nat) (i : Fin {model.n}) :",
                f"    {params}.receptivityAt",
                f"        ((((step {params} {model.n})^[k] {state}) i).exposure +",
                f"          FitnessABMPathN.degree {model.n} i) =",
                f"      {params}.receptivityAt",
                f"        (({state} i).exposure +",
                f"          (k + 1) * FitnessABMPathN.degree {model.n} i) := by",
                f"  rw [exposure_iterate {params} {valid_lemma} {model.n} (by norm_num)",
                f"    {state} {broadcast_lemma} k i]",
                "  congr 1",
                "  omega",
                "",
            ]

        lines += ["end NarrativeAnalyzerCertificate", ""]

        metadata = {
            "parameters_valid": CertificateClaim(
                claim_id="parameters_valid",
                expected_status=ClaimStatus.PROVED,
                theorem=(
                    f"{valid_lemma}; definition "
                    f"{_EXPOSURE_NS}.ExposureParameters.Valid"
                ),
                assumptions=(),
                exact_values={},
            ),
            "initial_all_broadcast": CertificateClaim(
                claim_id="initial_all_broadcast",
                expected_status=ClaimStatus.PROVED,
                theorem=f"{broadcast_lemma}; definition {_CONVERGENCE_NS}.allBroadcast",
                assumptions=(),
                exact_values={},
            ),
            "exposure_law": CertificateClaim(
                claim_id="exposure_law",
                expected_status=ClaimStatus.PROVED,
                theorem=f"{exposure_lemma}; {_CONVERGENCE_NS}.exposure_iterate",
                assumptions=("parameters_valid", "initial_all_broadcast", "n >= 2"),
                exact_values={},
            ),
            "effective_alpha_lookup": CertificateClaim(
                claim_id="effective_alpha_lookup",
                expected_status=ClaimStatus.PROVED,
                theorem=f"{alpha_lemma}; {_CONVERGENCE_NS}.exposure_iterate",
                assumptions=("parameters_valid", "initial_all_broadcast", "n >= 2"),
                exact_values={"lookup": "e0(i) + (k+1) * degree(i)"},
            ),
        }
        return Certificate(
            source="\n".join(lines),
            claims=tuple(metadata[claim] for claim in requested),
        )

    def build_structural_negative(
        self,
        model: PathModel,
        claims: Iterable[str],
    ) -> Certificate:
        requested = _claim_order(claims, _NEGATIVE_STRUCTURAL)
        digest = _model_digest(model)
        lines = _header()
        definitions, params, state = _definitions(model, digest)
        lines += definitions
        claim_records: list[CertificateClaim] = []

        for claim_id in requested:
            if claim_id == "parameters_valid":
                witness = _parameter_invalid_witness(model)
                if witness is None:
                    raise ValueError("model has no exact parameter-validity counter-witness")
                lemma = f"AnalyzerParamsInvalid_{digest}"
                kind, exposure = witness
                lines += [f"private theorem {lemma} : ¬ {params}.Valid := by", "  intro h"]
                if kind == "threshold":
                    lines += [
                        "  have hbad := h.2",
                        f"  norm_num [{params}] at hbad",
                    ]
                else:
                    assert exposure is not None
                    lines += [
                        f"  have hbad := h.1 {exposure}",
                        f"  norm_num [{params}] at hbad",
                    ]
                lines += [""]
                claim_records.append(
                    CertificateClaim(
                        claim_id=claim_id,
                        expected_status=ClaimStatus.DISPROVED,
                        theorem=(
                            f"{lemma}; negation of "
                            f"{_EXPOSURE_NS}.ExposureParameters.Valid"
                        ),
                        assumptions=(),
                        exact_values={},
                    )
                )
            elif claim_id == "initial_all_broadcast":
                index = _all_broadcast_invalid_index(model)
                if index is None:
                    raise ValueError("model has no exact all-broadcast counter-witness")
                lemma = f"AnalyzerAllBroadcastInvalid_{digest}"
                lines += [
                    f"private theorem {lemma} : ¬ allBroadcast {params} {state} := by",
                    "  intro h",
                    f"  have hbad := h ({index} : Fin {model.n})",
                    f"  norm_num [allBroadcast, {params}, {state}] at hbad",
                    "",
                ]
                claim_records.append(
                    CertificateClaim(
                        claim_id=claim_id,
                        expected_status=ClaimStatus.DISPROVED,
                        theorem=f"{lemma}; negation of {_CONVERGENCE_NS}.allBroadcast",
                        assumptions=(),
                        exact_values={"witness_index": str(index)},
                    )
                )

        lines += ["end NarrativeAnalyzerCertificate", ""]
        return Certificate(source="\n".join(lines), claims=tuple(claim_records))

    def build_pathn_consensus(self, model: PathModel, eps: ExactRat) -> Certificate:
        raise CertificateGenerationError("PathN consensus certificates are Task 5")

    def build_named_path2(self, model: PathModel, route: object) -> Certificate:
        raise CertificateGenerationError("named Path2 certificates are Task 6")
