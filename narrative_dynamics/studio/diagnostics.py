"""Deterministic translation of the existing scenario compiler's one diagnostic."""

from __future__ import annotations

from narrative_dynamics.abm.scenario_compiler import (
    ScenarioCompilationError,
    compile_situated_scenario_package,
)
from narrative_dynamics.abm.scenario_package_contracts import ScenarioPackageSource
from narrative_dynamics.studio.contracts import (
    ScenarioDiagnostic,
    ScenarioDiagnosticReport,
    ScenarioDiagnosticSeverity,
)


def manifest_missing_report(project_id: str, revision: int) -> ScenarioDiagnosticReport:
    return ScenarioDiagnosticReport(
        project_id,
        revision,
        (
            ScenarioDiagnostic(
                ScenarioDiagnosticSeverity.ERROR,
                "manifest_missing",
                "package",
                None,
                "",
            ),
        ),
    )


def compiler_error_report(
    project_id: str,
    revision: int,
    error: ScenarioCompilationError,
) -> ScenarioDiagnosticReport:
    role = error.document_role
    logical_id = None
    if role.startswith("agent:"):
        role, logical_id = "agent", role.removeprefix("agent:")
    return ScenarioDiagnosticReport(
        project_id,
        revision,
        (
            ScenarioDiagnostic(
                ScenarioDiagnosticSeverity.ERROR,
                error.code,
                role,
                logical_id,
                error.json_pointer,
            ),
        ),
    )


def compile_with_report(
    project_id: str,
    revision: int,
    source: ScenarioPackageSource,
):
    """Return ``(compiled, report)`` without exposing compiler exception text."""

    try:
        compiled = compile_situated_scenario_package(source)
    except ScenarioCompilationError as error:
        return None, compiler_error_report(project_id, revision, error)
    return compiled, ScenarioDiagnosticReport(project_id, revision)


__all__ = (
    "compile_with_report",
    "compiler_error_report",
    "manifest_missing_report",
)
