"""R2.M5g.2 original V19 complete 2x2 direct-TELL policy grid."""
import json
from narrative_dynamics.abm.mesh_native_trust_ablation import (
    NativeTrustChannelAblationReport,
    observe_native_trust_channel_ablation,
)


def render_source_vector(report:NativeTrustChannelAblationReport|None=None)->str:
    if report is None:
        report=observe_native_trust_channel_ablation()
    return json.dumps({
        "schema":"r2-m5g2-original-v19-native-direct-tell-trust-ablation-v1",
        "report_hash":report.content_hash,
        "report":report.to_dict(),
    },sort_keys=True,indent=2,ensure_ascii=False)+"\n"


if __name__=="__main__":
    print(render_source_vector(),end="")
