"""R2.M5e.2 original V19 native decision with two feasible actions."""
import json
from tests.test_network_native_two_choice import build_source_native_choice_case


def render_source_vector() -> str:
    report = build_source_native_choice_case()
    return json.dumps(
        {
            "schema": "r2-m5e2-v19-two-native-choices-v1",
            "report_hash": report.content_hash,
            "report": report.to_dict(),
        },
        sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_source_vector(), end="")
