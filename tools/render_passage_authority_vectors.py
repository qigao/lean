"""Source-replay V19 passage controller actions; do not invent event records."""
import json

from tests.test_network_power_authority import build_authority_case


def render_source_vector() -> str:
    report = build_authority_case()
    return json.dumps(
        {"schema": "r2-m5a-v19-source-passage-control-v1",
         "report_hash": report.content_hash,
         "report": report.to_dict()},
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
