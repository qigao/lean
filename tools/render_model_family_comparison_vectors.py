"""Native source-derived BB / rewiring / evolving / V19 qualification dossier."""
from __future__ import annotations
import json
from tests.test_network_model_family_comparison import build_family_fixture

SCHEMA="v19-abm-model-family-qualification-v1"


def build_vector_manifest()->dict[str,object]:
    report=build_family_fixture()
    data=report.to_dict()
    if (
        data.get("schema") != SCHEMA
        or data.get("cross_family_numeric_ranking") != "FORBIDDEN"
        or data.get("normalization") != "NONE"
        or data.get("causal_attribution") != "UNATTRIBUTED"
    ):
        raise AssertionError("native model qualification attempted an invalid comparison")
    return {
        "schema":SCHEMA,
        "notice":("Three independent native ABM models alongside real V19 branch "
                  "without edge-count normalization, trust equivalence or inferred causality"),
        "report_hash":report.content_hash,
        "report":data,
    }


def render_vector_corpus()->str:
    return json.dumps(build_vector_manifest(),sort_keys=True,
                      indent=2,ensure_ascii=False)+"\n"


if __name__=="__main__":
    print(render_vector_corpus(),end="")
