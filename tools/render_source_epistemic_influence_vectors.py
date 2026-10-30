"""R2.M5c reviewable V19 source-owned epistemic paired experiment corpus."""
from __future__ import annotations

import json
from tests.test_network_epistemic_influence import build_source_epistemic_case


def render_source_vector() -> str:
    report = build_source_epistemic_case()
    return json.dumps(
        {
            "schema": "r2-m5c-source-epistemic-role-authority-v1",
            "report_hash": report.content_hash,
            "report": report.to_dict(),
        },
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
