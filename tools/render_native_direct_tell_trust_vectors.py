"""Canonical native V19 opt-in direct testimony trust source experiment."""
import json
from tests.test_network_direct_tell_trust import build_direct_tell_trust_case


def render_source_vector() -> str:
    report = build_direct_tell_trust_case()
    return json.dumps(
        {"schema": "r2-m5e0-v19-source-direct-tell-trust-v1",
         "report_hash": report.content_hash, "report": report.to_dict()},
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
