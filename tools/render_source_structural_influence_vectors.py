"""R2.M5b reviewed structural V19/V24 source-linked directed observations."""
from __future__ import annotations

import json
from tests.test_network_structural_influence import build_native_structural_case


def render_source_vector() -> str:
    report = build_native_structural_case()
    return json.dumps(
        {"schema": "r2-m5b-source-directed-key-player-v1",
         "report_hash": report.content_hash,
         "report": report.to_dict()},
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
