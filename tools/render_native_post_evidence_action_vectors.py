"""R2.M5d exact V19 post-evidence actual Agent decision/actuation vector."""
from __future__ import annotations

import json
from tests.test_network_post_evidence_actions import build_post_evidence_action_case


def render_source_vector() -> str:
    report = build_post_evidence_action_case()
    return json.dumps(
        {
            "schema": "r2-m5d-v19-post-evidence-native-actions-v1",
            "report_hash": report.content_hash,
            "report": report.to_dict(),
        },
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
