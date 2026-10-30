"""R2.M4c1 reviewed native control/intervention dossier from existing models."""
from __future__ import annotations

import json
from tests.test_network_native_paired import build_native_paired_case


def render_vector_corpus() -> str:
    report = build_native_paired_case()
    return json.dumps(
        {"schema": "r2-m4c1-native-within-family-paired-v1",
         "report_hash": report.content_hash,
         "report": report.to_dict()},
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_vector_corpus(), end="")
