from __future__ import annotations

from dataclasses import replace
import json

import pytest

from narrative_dynamics.abm.situated_projection import project_situated_narrative
from narrative_dynamics.abm.situated_realization import (
    build_narrative_realization_prompt,
    compile_narrative_realization,
    render_narrative_realization_text,
    replay_narrative_realization,
)
from narrative_dynamics.abm.situated_realization_contracts import (
    NarrativePassage,
    NarrativeRealizationAssurance,
    NarrativeRealizationFormat,
    NarrativeRealizationPolicy,
    NarrativeRealizationProviderIdentity,
    NarrativeRealizationRequest,
    NarrativeRealizedScene,
)
from narrative_dynamics.contracts import stable_content_hash
from tests.test_network_abm_situated_projection import (
    limited_policy,
    objective_policy,
    one_place_story,
    private_inspection_then_public_clue_story,
)


class RecordingProvider:
    def __init__(self, responder):
        self.identity = NarrativeRealizationProviderIdentity(
            "recording-provider", "1.0", "test-model"
        )
        self.payloads: list[dict[str, object]] = []
        self.tasks: list[str] = []
        self._responder = responder

    def complete_json(self, *, task, payload):
        self.tasks.append(task)
        self.payloads.append(payload)
        return self._responder(payload)


def valid_one_passage_per_scene_response(payload):
    return {
        "passages": [
            {
                "beat_ids": payload["scene"]["beat_ids"],
                "text": "The scene unfolds.",
            }
        ]
    }


def policy(**kwargs):
    return NarrativeRealizationPolicy(
        "realization", "1.0", NarrativeRealizationFormat.PROSE, "en", **kwargs
    )


def request(projection):
    return NarrativeRealizationRequest("render-1", projection.content_hash)


def compile_multi_scene_fixture():
    projection = project_situated_narrative(
        one_place_story(round_count=3),
        objective_policy(maximum_scene_beats=1),
    )
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)
    return artifact, provider, projection


def single_scene_projection():
    return project_situated_narrative(
        one_place_story(round_count=4),
        objective_policy(maximum_scene_beats=4),
    )


def compile_response(response, *, realization_policy=None):
    projection = single_scene_projection()
    provider = RecordingProvider(lambda payload: response)
    selected_policy = realization_policy or policy()
    prompt = build_narrative_realization_prompt(
        projection, selected_policy, request(projection), provider
    )
    return compile_narrative_realization(prompt, provider)


def test_limited_realization_never_sends_hidden_private_facts():
    projection = project_situated_narrative(
        private_inspection_then_public_clue_story(), limited_policy("bob")
    )
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    compile_narrative_realization(prompt, provider)
    serialized = json.dumps(provider.payloads, sort_keys=True)
    assert "restructuring" not in serialized
    assert "approved" not in serialized


def test_each_provider_call_contains_only_one_scene_context():
    artifact, provider, projection = compile_multi_scene_fixture()
    assert artifact.projection_hash == projection.content_hash
    assert provider.tasks == [
        "situated_narrative_scene_realization_v1"
    ] * len(projection.scenes)
    assert len(provider.payloads) == len(projection.scenes)
    for payload, scene in zip(provider.payloads, projection.scenes):
        assert payload["scene"]["scene_id"] == scene.scene_id
        assert {item["scene_id"] for item in [payload["scene"]]} == {
            scene.scene_id
        }


def test_scene_packet_contains_only_prompt_policy_schema_and_limits():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    realization_policy = policy(
        tone_tags=("spare", "calm"),
        maximum_passages_per_scene=3,
        maximum_passage_characters=123,
    )
    prompt = build_narrative_realization_prompt(
        projection, realization_policy, request(projection), provider
    )

    compile_narrative_realization(prompt, provider)

    payload = provider.payloads[0]
    assert set(payload) == {
        "scene",
        "beats",
        "entitlements",
        "policy",
        "response_schema",
        "limits",
    }
    assert payload["policy"] == realization_policy.to_dict()
    assert payload["response_schema"] == {
        "type": "object",
        "required": ["passages"],
        "additional_properties": False,
        "passage": {
            "type": "object",
            "required": ["beat_ids", "text"],
            "additional_properties": False,
        },
    }
    assert payload["limits"] == {
        "maximum_passages": 3,
        "maximum_passage_characters": 123,
    }


def test_compile_derives_passage_id_from_accepted_runtime_fields():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )

    artifact = compile_narrative_realization(prompt, provider)

    passage = artifact.scenes[0].passages[0]
    assert passage.passage_id == stable_content_hash(
        {
            "scene_id": projection.scenes[0].scene_id,
            "beat_ids": list(projection.scenes[0].beat_ids),
            "entitlement_ids": list(passage.entitlement_ids),
            "text": "The scene unfolds.",
        }
    )


def test_provider_cannot_mutate_the_fixed_schema_for_later_scene_packets():
    projection = project_situated_narrative(
        one_place_story(round_count=2),
        objective_policy(maximum_scene_beats=1),
    )

    class SchemaMutatingProvider(RecordingProvider):
        def __init__(self, responder):
            super().__init__(responder)
            self.required_before_mutation = []

        def complete_json(self, *, task, payload):
            self.required_before_mutation.append(
                list(payload["response_schema"]["required"])
            )
            response = super().complete_json(task=task, payload=payload)
            payload["response_schema"]["required"].append("provider_state")
            return response

    provider = SchemaMutatingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )

    compile_narrative_realization(prompt, provider)

    assert provider.required_before_mutation == [["passages"], ["passages"]]


def test_prompt_uses_exact_projection_objects_in_cut_presentation_order():
    projection = project_situated_narrative(
        one_place_story(round_count=4), objective_policy(maximum_scene_beats=2)
    )
    provider = RecordingProvider(valid_one_passage_per_scene_response)

    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )

    assert tuple(item.scene for item in prompt.scenes) == projection.scenes
    assert all(
        scene_prompt.scene is scene
        for scene_prompt, scene in zip(prompt.scenes, projection.scenes)
    )
    beat_by_id = {beat.beat_id: beat for beat in projection.beats}
    entitlement_by_id = {
        item.entitlement_id: item for item in projection.cut.entitlements
    }
    for scene_prompt in prompt.scenes:
        assert all(
            beat is beat_by_id[beat_id]
            for beat, beat_id in zip(scene_prompt.beats, scene_prompt.scene.beat_ids)
        )
        assert all(
            entitlement is entitlement_by_id[entitlement.entitlement_id]
            for entitlement in scene_prompt.entitlements
        )


def test_prompt_rejects_request_for_another_projection():
    projection = project_situated_narrative(
        one_place_story(round_count=2), objective_policy()
    )
    other = project_situated_narrative(
        one_place_story(round_count=3), objective_policy()
    )
    provider = RecordingProvider(valid_one_passage_per_scene_response)

    with pytest.raises(ValueError, match="request.*projection hash"):
        build_narrative_realization_prompt(
            projection, policy(), request(other), provider
        )


@pytest.mark.parametrize(
    "provider",
    (
        object(),
        type("InvalidIdentityProvider", (), {"identity": {"provider_id": "x"}})(),
        type(
            "MissingCompletionProvider",
            (),
            {
                "identity": NarrativeRealizationProviderIdentity(
                    "provider", "1.0", "model"
                )
            },
        )(),
    ),
)
def test_prompt_rejects_an_invalid_provider_protocol(provider):
    projection = project_situated_narrative(
        one_place_story(round_count=1), objective_policy()
    )

    with pytest.raises(TypeError, match="provider"):
        build_narrative_realization_prompt(
            projection, policy(), request(projection), provider
        )


@pytest.mark.parametrize(
    "response",
    (
        None,
        [],
        {},
        {"passages": [], "extra": True},
        {"passages": [None]},
        {"passages": [{"beat_ids": []}]},
        {"passages": [{"beat_ids": [], "text": "Text.", "extra": True}]},
    ),
)
def test_compile_rejects_non_objects_and_non_exact_response_keys(response):
    with pytest.raises(ValueError, match="exact.*response"):
        compile_response(response)


def test_compile_rejects_empty_or_excessive_passage_lists():
    projection = single_scene_projection()
    beat_ids = list(projection.scenes[0].beat_ids)
    responses = (
        {"passages": []},
        {
            "passages": [
                {"beat_ids": beat_ids[:2], "text": "First."},
                {"beat_ids": beat_ids[2:], "text": "Second."},
            ]
        },
    )
    one_passage_policy = policy(maximum_passages_per_scene=1)

    for response in responses:
        with pytest.raises(ValueError, match="passage count"):
            compile_response(response, realization_policy=one_passage_policy)


def test_compile_rejects_unknown_beat_ids():
    response = {
        "passages": [{"beat_ids": ["beat:unknown"], "text": "Unknown."}]
    }
    with pytest.raises(ValueError, match="unknown beat"):
        compile_response(response)


def test_compile_rejects_duplicate_beat_ids():
    projection = single_scene_projection()
    beat_ids = list(projection.scenes[0].beat_ids)
    response = {
        "passages": [
            {"beat_ids": beat_ids + [beat_ids[-1]], "text": "Repeated."}
        ]
    }
    with pytest.raises(ValueError, match="duplicate beat"):
        compile_response(response)


def test_compile_rejects_omitted_beat_ids():
    projection = single_scene_projection()
    response = {
        "passages": [
            {"beat_ids": list(projection.scenes[0].beat_ids[:-1]), "text": "Short."}
        ]
    }
    with pytest.raises(ValueError, match="complete beat coverage"):
        compile_response(response)


def test_compile_rejects_reordered_beat_ids():
    projection = single_scene_projection()
    response = {
        "passages": [
            {
                "beat_ids": list(reversed(projection.scenes[0].beat_ids)),
                "text": "Backward.",
            }
        ]
    }
    with pytest.raises(ValueError, match="canonical beat order"):
        compile_response(response)


def test_compile_rejects_non_contiguous_passage_beat_ids():
    projection = single_scene_projection()
    beat_ids = projection.scenes[0].beat_ids
    response = {
        "passages": [
            {"beat_ids": [beat_ids[0], beat_ids[2]], "text": "First."},
            {"beat_ids": [beat_ids[1], beat_ids[3]], "text": "Second."},
        ]
    }
    with pytest.raises(ValueError, match="contiguous"):
        compile_response(response)


def test_compile_rejects_cross_scene_beat_ids():
    projection = project_situated_narrative(
        one_place_story(round_count=2),
        objective_policy(maximum_scene_beats=1),
    )
    response = {
        "passages": [
            {"beat_ids": [projection.scenes[1].beat_ids[0]], "text": "Foreign."}
        ]
    }
    provider = RecordingProvider(lambda payload: response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )

    with pytest.raises(ValueError, match="scene-local beat"):
        compile_narrative_realization(prompt, provider)


@pytest.mark.parametrize("text", ("", "\x00", "x" * 17))
def test_compile_rejects_empty_nul_or_oversized_passage_text(text):
    projection = single_scene_projection()
    response = {
        "passages": [
            {"beat_ids": list(projection.scenes[0].beat_ids), "text": text}
        ]
    }
    short_policy = policy(maximum_passage_characters=16)
    with pytest.raises(ValueError, match="provider passage text"):
        compile_response(response, realization_policy=short_policy)


def test_compile_rejects_provider_identity_mutation_during_a_scene_call():
    projection = single_scene_projection()

    class DriftingProvider(RecordingProvider):
        def complete_json(self, *, task, payload):
            response = super().complete_json(task=task, payload=payload)
            self.identity = NarrativeRealizationProviderIdentity(
                "changed-provider", "2.0", "changed-model"
            )
            return response

    provider = DriftingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )

    with pytest.raises(ValueError, match="identity changed"):
        compile_narrative_realization(prompt, provider)


def test_compile_rejects_in_place_identity_mutation_without_rewriting_prompt_provenance():
    projection = single_scene_projection()

    class InPlaceDriftingProvider(RecordingProvider):
        def complete_json(self, *, task, payload):
            response = super().complete_json(task=task, payload=payload)
            object.__setattr__(
                self.identity,
                "provider_id",
                "changed-provider",
            )
            return response

    provider = InPlaceDriftingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    prompt_hash_before_call = prompt.content_hash

    with pytest.raises(ValueError, match="identity changed"):
        compile_narrative_realization(prompt, provider)

    assert prompt.provider == NarrativeRealizationProviderIdentity(
        "recording-provider", "1.0", "test-model"
    )
    assert prompt.provider is not provider.identity
    assert prompt.content_hash == prompt_hash_before_call


def test_artifact_retains_pre_invocation_identity_snapshot_from_structural_prompt():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    structural_prompt = replace(prompt, provider=provider.identity)
    prompt_hash_before_call = structural_prompt.content_hash

    artifact = compile_narrative_realization(structural_prompt, provider)
    artifact_hash_before_mutation = artifact.content_hash
    object.__setattr__(provider.identity, "provider_id", "changed-provider")

    assert artifact.provider == NarrativeRealizationProviderIdentity(
        "recording-provider", "1.0", "test-model"
    )
    assert artifact.provider is not provider.identity
    assert artifact.prompt_hash == prompt_hash_before_call
    assert artifact.content_hash == artifact_hash_before_mutation


def test_artifact_records_citation_bound_provenance_and_replays_without_provider():
    projection = single_scene_projection()
    response = {
        "passages": [
            {
                "beat_ids": list(projection.scenes[0].beat_ids),
                "text": "Literal expected text.",
            }
        ]
    }
    provider = RecordingProvider(lambda payload: response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)

    assert artifact.projection_hash == projection.content_hash
    assert artifact.assurance is NarrativeRealizationAssurance.CITATION_BOUND
    assert artifact.validation_result == "accepted"
    assert replay_narrative_realization(projection, artifact) is artifact
    assert render_narrative_realization_text(artifact) == "Literal expected text."


def test_replay_rejects_wrong_projection_prompt_and_scene_prompt_hashes():
    artifact, provider, projection = compile_multi_scene_fixture()
    wrong_hash = "sha256:" + "f" * 64
    wrong_scene = replace(artifact.scenes[0], scene_prompt_hash=wrong_hash)
    corruptions = (
        replace(artifact, projection_hash=wrong_hash),
        replace(artifact, prompt_hash=wrong_hash),
        replace(artifact, scenes=(wrong_scene,) + artifact.scenes[1:]),
    )

    for corrupted in corruptions:
        with pytest.raises(ValueError, match="replay.*hash"):
            replay_narrative_realization(projection, corrupted)


def test_replay_rejects_wrong_schema_template_and_response_hashes():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)
    wrong_hash = "sha256:" + "f" * 64

    corrupted_schema = replace(artifact)
    object.__setattr__(corrupted_schema, "schema_hash", wrong_hash)
    corrupted_template = replace(artifact)
    object.__setattr__(corrupted_template, "prompt_template_hash", wrong_hash)
    corrupted_scene = replace(artifact.scenes[0])
    object.__setattr__(corrupted_scene, "provider_response_hash", wrong_hash)
    corrupted_response = replace(artifact, scenes=(corrupted_scene,))

    for corrupted in (corrupted_schema, corrupted_template, corrupted_response):
        with pytest.raises(ValueError, match="replay.*hash"):
            replay_narrative_realization(projection, corrupted)


def test_replay_rejects_reordered_scenes():
    artifact, provider, projection = compile_multi_scene_fixture()
    reordered = replace(artifact, scenes=tuple(reversed(artifact.scenes)))

    with pytest.raises(ValueError, match="scene order"):
        replay_narrative_realization(projection, reordered)


def test_replay_rejects_incomplete_beat_coverage():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)
    source = artifact.scenes[0].passages[0]
    selected_beat_ids = source.beat_ids[:-1]
    beat_by_id = {beat.beat_id: beat for beat in projection.beats}
    entitlement_ids = tuple(
        sorted(
            {
                entitlement_id
                for beat_id in selected_beat_ids
                for entitlement_id in beat_by_id[beat_id].entitlement_ids
            }
        )
    )
    shortened = NarrativePassage(
        source.passage_id,
        source.scene_id,
        selected_beat_ids,
        entitlement_ids,
        source.text,
    )
    response_hash = stable_content_hash(
        {
            "provider_response": {
                "passages": [
                    {"beat_ids": list(shortened.beat_ids), "text": shortened.text}
                ]
            }
        }
    )
    scene = NarrativeRealizedScene(
        artifact.scenes[0].scene_id,
        artifact.scenes[0].scene_prompt_hash,
        response_hash,
        (shortened,),
    )
    corrupted = replace(artifact, scenes=(scene,))

    with pytest.raises(ValueError, match="complete beat coverage"):
        replay_narrative_realization(projection, corrupted)


def test_replay_rejects_wrong_entitlement_closure_and_passage_id():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)
    scene = artifact.scenes[0]
    passage = scene.passages[0]
    wrong_entitlement = replace(passage, entitlement_ids=("entitlement:wrong",))
    wrong_passage_id = replace(passage, passage_id="passage:wrong")
    corruptions = (
        replace(scene, passages=(wrong_entitlement,)),
        replace(scene, passages=(wrong_passage_id,)),
    )

    for corrupted_scene in corruptions:
        corrupted = replace(artifact, scenes=(corrupted_scene,))
        with pytest.raises(ValueError, match="entitlement closure|passage id"):
            replay_narrative_realization(projection, corrupted)


def test_replay_rejects_structurally_tampered_acceptance_and_passage_mapping():
    projection = single_scene_projection()
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)

    rejected = replace(artifact)
    object.__setattr__(rejected, "validation_result", "rejected")

    passage = replace(artifact.scenes[0].passages[0])
    object.__setattr__(passage, "scene_id", "scene:foreign")
    scene = replace(artifact.scenes[0])
    object.__setattr__(scene, "passages", (passage,))
    wrong_mapping = replace(artifact, scenes=(scene,))

    for corrupted in (rejected, wrong_mapping):
        with pytest.raises(ValueError, match="accepted|passage mapping"):
            replay_narrative_realization(projection, corrupted)


def test_render_preserves_scene_and_passage_order_with_blank_line_boundaries():
    projection = single_scene_projection()
    beat_ids = projection.scenes[0].beat_ids
    response = {
        "passages": [
            {"beat_ids": list(beat_ids[:2]), "text": "First passage."},
            {"beat_ids": list(beat_ids[2:]), "text": "Second passage."},
        ]
    }
    provider = RecordingProvider(lambda payload: response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)

    assert render_narrative_realization_text(artifact) == (
        "First passage.\n\nSecond passage."
    )
