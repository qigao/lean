"""Reviewed, real-source V19 MOVE/TAKE/DROP and subsequent TELL vectors."""
from __future__ import annotations

import json

from narrative_dynamics.abm.mesh_environment_observation import (
    observe_environment_action_trace,
)
from tests.test_network_environment_actions import build_environment_scenarios

SCHEMA="v19-environment-action-trace-v1"


def build_vector_manifest()->dict[str,object]:
    cases=[]
    for name,before,after in build_environment_scenarios():
        trace=observe_environment_action_trace(before,after)
        if trace.prior_source_hash!=before.mesh.source_snapshot.content_hash:
            raise AssertionError("R2 environment source baseline drift")
        if trace.next_source_hash!=after.mesh.source_snapshot.content_hash:
            raise AssertionError("R2 environment source revision drift")
        if trace.to_dict()["causal_attribution"]!="UNATTRIBUTED":
            raise AssertionError("R2 environment cannot infer external cause")
        cases.append({"name":name,"trace_hash":trace.content_hash,"trace":trace.to_dict()})
    return {
        "schema":SCHEMA,
        "notice":("Real re-executable V19 environment events and selected V24 edges; "
                  "does not disclose private messages or prove historical causality"),
        "cases":cases,
    }


def render_vector_corpus()->str:
    return json.dumps(build_vector_manifest(),ensure_ascii=False,
                      sort_keys=True,indent=2)+"\n"


if __name__=="__main__":
    print(render_vector_corpus(),end="")
