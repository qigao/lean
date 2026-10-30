"""Canonical real V19 same-origin boundary→MOVE→TELL→social-learning contrast."""
from __future__ import annotations
import json
from tests.test_network_coupled_evolution import build_paired_coupled_case


def render_vector_corpus() -> str:
    data=build_paired_coupled_case()
    if (data["schema"]!="v19-controlled-coupled-evolution-v1"
        or data["cross_branch"]["story_lineage"]!="DIVERGENT_STORY_BRANCH"
        or data["cross_branch"]["causal_attribution"]!="UNATTRIBUTED"
    ):
        raise AssertionError("paired V19 baseline/counterfactual is not independently grounded")
    return json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+"\n"


if __name__=="__main__":
    print(render_vector_corpus(),end="")
