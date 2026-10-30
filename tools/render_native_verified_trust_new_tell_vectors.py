"""R2.M5e.1 original V19 source histories + new native TELL."""
import json
from tests.test_network_trust_history_new_tell import build_history_new_tell_case


def render_source_vector() -> str:
    report = build_history_new_tell_case()
    return json.dumps(
        {"schema": "r2-m5e1-v19-confirmation-contradiction-new-tell-v1",
         "report_hash": report.content_hash,
         "report": report.to_dict()},
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
