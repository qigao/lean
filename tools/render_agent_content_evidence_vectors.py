"""Reviewed V19 source-derived exact percept / claim / directed-trust vectors."""
from __future__ import annotations
import json
from tests.test_agent_content_evidence import build_content_cases

SCHEMA="v19-agent-content-evidence-v1"


def build_vector_manifest()->dict[str,object]:
    result=[]
    for name,(_,_,story,cognition,social,round_result,trace) in build_content_cases():
        data=trace.to_dict()
        if (data["causal_attribution"]!="UNATTRIBUTED"
            or data["prior_story_hash"]!=story.content_hash
            or data["prior_cognitive_state_hash"]!=cognition.content_hash
            or data["prior_social_state_hash"]!=social.content_hash
            or data["transition_hash"]!=round_result.content_hash
            or data["next_social_state_hash"]!=round_result.next_social_state.content_hash
        ):
            raise AssertionError("content source/evidence witness is stale or untrusted")
        result.append({"name":name,"trace_hash":trace.content_hash,"trace":data})
    return {
        "schema":SCHEMA,
        "notice":(
            "Real V19 private perception and source-owned social claims; "
            "no private content payload, external actor authentication, "
            "automatic trust or inferred universal graph property"
        ),
        "cases":result,
    }


def render_vector_corpus()->str:
    return json.dumps(build_vector_manifest(),sort_keys=True,indent=2,
                      ensure_ascii=False)+"\n"


if __name__=="__main__":
    print(render_vector_corpus(),end="")
