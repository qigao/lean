"""Command-line rendering for the proof-backed analyzer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, TextIO

from .analyze import analyze_model
from .model import ModelInputError, load_model
from .result import (
    AnalysisResult,
    CertificateCompileError,
    CertificateGenerationError,
    CertificateTimeoutError,
    ClaimStatus,
    ProvenanceMismatchError,
)
from .runner import LeanCertificateRunner


def render_result(result: AnalysisResult) -> str:
    summary = result.model_summary
    lines = [
        f"Topology: Path {summary.get('n', '?')}",
        f"Exposure semantics: {summary.get('exposure_semantics', 'post-incoming lookup')}",
        "",
    ]
    for claim in result.claims:
        lines.append(f"{claim.claim_id:<28} {claim.status.value}")
        if claim.status in (ClaimStatus.PROVED, ClaimStatus.DISPROVED):
            lines.append(f"  theorem: {claim.theorem}")
            if claim.assumptions:
                lines.append("  assumptions: " + ", ".join(claim.assumptions))
            if claim.exact_values:
                exact = ", ".join(
                    f"{key}={value}" for key, value in sorted(claim.exact_values.items())
                )
                lines.append(f"  exact: {exact}")
        elif claim.note:
            lines.append(f"  note: {claim.note}")
    lines += ["", f"Consensus: {result.consensus_status.value}"]
    value_claim = next(
        (c for c in result.claims if c.claim_id == "consensus_value_known"), None
    )
    if value_claim is not None and value_claim.status is ClaimStatus.PROVED:
        value = value_claim.exact_values.get("value", "proved")
        lines.append(f"Consensus value: {value}")
    else:
        lines.append("Consensus value: unknown in closed form")
    return "\n".join(lines) + "\n"


def _default_runner_factory(repo_root: Path) -> LeanCertificateRunner:
    return LeanCertificateRunner(repo_root)


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    runner_factory: Callable[[Path], object] | None = None,
    analyze: Callable[..., AnalysisResult] = analyze_model,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    if len(args) != 1:
        print("usage: narrative-analyze model.toml", file=stderr)
        return 2

    try:
        model = load_model(Path(args[0]))
    except (ModelInputError, OSError) as exc:
        print(f"input error: {exc}", file=stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    factory = _default_runner_factory if runner_factory is None else runner_factory
    try:
        result = analyze(model, runner=factory(repo_root))
    except ProvenanceMismatchError as exc:
        print(f"internal analyzer error: {exc}", file=stderr)
        return 4
    except (
        CertificateGenerationError,
        CertificateCompileError,
        CertificateTimeoutError,
    ) as exc:
        print(f"certificate error: {exc}", file=stderr)
        return 3
    except (AssertionError, ValueError) as exc:
        print(f"internal analyzer error: {exc}", file=stderr)
        return 4

    stdout.write(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
