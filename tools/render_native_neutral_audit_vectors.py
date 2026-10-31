"""R2.M5g0 one original V19 source-owned neutral audit trust contrast."""
import json

from narrative_dynamics.abm.mesh_neutral_audit_trust import (
    NeutralAuditSourceReport, observe_neutral_training_trust,
)


def render_source_vector(report: NeutralAuditSourceReport | None = None) -> str:
    if report is None:
        report = observe_neutral_training_trust()
    return json.dumps(
        {
            "schema": "r2-m5g0-v19-neutral-audit-trust-learning-v1",
            "report_hash": report.content_hash,
            "report": report.to_dict(),
        },
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
