"""R2.M5f original V19 complete response grid, no result selection."""
import json

from narrative_dynamics.abm.mesh_response_cost_grid import (
    NativeResponseGridReport, observe_native_speaking_cost_grid,
)


def render_source_vector(report: NativeResponseGridReport | None = None) -> str:
    if report is None:
        report = observe_native_speaking_cost_grid()
    return json.dumps(
        {
            "schema": "r2-m5f-exploratory-native-speaking-cost-grid-v1",
            "report_hash": report.content_hash,
            "report": report.to_dict(),
        },
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
