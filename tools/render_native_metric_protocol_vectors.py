"""Render M4c0 model-local metric evidence from original M4b native models."""
import json
from narrative_dynamics.abm.native_metric_protocol import build_native_metric_protocol
from tests.test_network_model_family_comparison import build_family_fixture


def render_vector_corpus() -> str:
    source = build_family_fixture()
    return json.dumps(build_native_metric_protocol(source).to_dict(),
                      sort_keys=True, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    print(render_vector_corpus(), end="")
