"""M5g.1 original V19 neutral trust → new source target TELL/action vector."""
import json

from narrative_dynamics.abm.mesh_neutral_target_decision import (
    NeutralAuditTargetDecisionReport, observe_neutral_audit_then_new_target_choice,
)


def render_source_vector(report:NeutralAuditTargetDecisionReport|None=None)->str:
    if report is None:
        report=observe_neutral_audit_then_new_target_choice()
    return json.dumps({
        "schema":"r2-m5g1-native-neutral-audit-post-trust-target-choice-v1",
        "report_hash":report.content_hash,
        "report":report.to_dict(),
    },sort_keys=True,indent=2,ensure_ascii=False)+"\n"


if __name__=="__main__":
    print(render_source_vector(),end="")
