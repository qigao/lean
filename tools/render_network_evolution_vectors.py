"""R2.M0 canonical source-derived three-driver network comparison vectors.

No independent graph algorithm or mutable network source is introduced. Each
case uses the same exact V19 model/story/cognition/social and V24 MeshSnapshot
fixtures as the focused runtime regression.
"""
from __future__ import annotations

import json

from narrative_dynamics.abm.mesh_evolution_observation import compare_network_layers
from tests.test_network_evolution_observation import build_review_scenarios

SCHEMA = "v19-three-driver-network-observation-v1"


def build_vector_manifest() -> dict[str, object]:
    cases = []
    seen = set()
    for name, before, after in build_review_scenarios():
        if name in seen:
            raise AssertionError("duplicate R2 source-derived case")
        seen.add(name)
        report = compare_network_layers(before, after)
        if report.before_source_hash != before.mesh.source_snapshot.content_hash:
            raise AssertionError("R2 before V19 source changed while reporting")
        if report.after_source_hash != after.mesh.source_snapshot.content_hash:
            raise AssertionError("R2 after V19 source changed while reporting")
        if report.to_dict()["causal_attribution"] != "UNATTRIBUTED":
            raise AssertionError("R2 cannot fabricate causal attribution")
        cases.append({"name": name, "report_hash": report.content_hash,
                      "report": report.to_dict()})
    return {
        "schema": SCHEMA,
        "notice": ("Source-derived finite V19/V24 comparisons: observations and "
                   "story-prefix evidence, never unverified causal attribution "
                   "or disclosure of private message payloads"),
        "cases": cases,
    }


def render_vector_corpus() -> str:
    return json.dumps(
        build_vector_manifest(), indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"


if __name__ == "__main__":
    print(render_vector_corpus(), end="")
