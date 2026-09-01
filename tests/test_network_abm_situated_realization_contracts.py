from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeat,
    NarrativeBeatKind,
    NarrativeCut,
    NarrativeEntitlement,
    NarrativeEntitlementScope,
    NarrativeFact,
    NarrativeProjection,
    NarrativeProjectionPolicy,
    NarrativeScene,
    NarrativeSupportRef,
)
from narrative_dynamics.abm.situated_realization_contracts import (
    NarrativePassage,
    NarrativeRealizationArtifact,
    NarrativeRealizationAssurance,
    NarrativeRealizationFormat,
    NarrativeRealizationPolicy,
    NarrativeRealizationPrompt,
    NarrativeRealizationProviderIdentity,
    NarrativeRealizationRequest,
    NarrativeRealizedScene,
    NarrativeSceneRealizationPrompt,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _entitlement(entitlement_id: str, event_id: str) -> NarrativeEntitlement:
    return NarrativeEntitlement(
        entitlement_id,
        NarrativeEntitlementScope.OBJECTIVE,
        None,
        1,
        (NarrativeFact("event", event_id),),
        (NarrativeSupportRef("world_event", event_id, HASH_A),),
    )


def _beat(beat_id: str, entitlement_id: str, event_id: str, round_index: int) -> NarrativeBeat:
    return NarrativeBeat(
        beat_id,
        NarrativeBeatKind.PHYSICAL,
        round_index,
        round_index,
        "office",
        None,
        ("alice",),
        1.0,
        (NarrativeSupportRef("world_event", event_id, HASH_A),),
        (entitlement_id,),
        (),
    )


@pytest.fixture
def projection() -> NarrativeProjection:
    first_entitlement = _entitlement("entitlement-1", "event-1")
    second_entitlement = _entitlement("entitlement-2", "event-2")
    first_beat = _beat("beat-1", first_entitlement.entitlement_id, "event-1", 1)
    second_beat = _beat("beat-2", second_entitlement.entitlement_id, "event-2", 2)
    first_scene = NarrativeScene("scene-1", (first_beat.beat_id,), 1, 1, "office", None)
    second_scene = NarrativeScene("scene-2", (second_beat.beat_id,), 2, 2, "office", None)
    return NarrativeProjection(
        HASH_A,
        None,
        NarrativeProjectionPolicy("objective", "1.0", NarrativeAuthority.OBJECTIVE),
        (first_beat, second_beat),
        (first_scene, second_scene),
        NarrativeCut(
            "cut-1",
            (first_scene.scene_id, second_scene.scene_id),
            (first_entitlement, second_entitlement),
        ),
    )


@pytest.fixture
def policy() -> NarrativeRealizationPolicy:
    return NarrativeRealizationPolicy(
        "realization", "1.0", NarrativeRealizationFormat.PROSE, "en"
    )


@pytest.fixture
def provider() -> NarrativeRealizationProviderIdentity:
    return NarrativeRealizationProviderIdentity("provider", "1.0", "model")


def _scene_prompt(projection: NarrativeProjection, scene_index: int = 0) -> NarrativeSceneRealizationPrompt:
    scene = projection.scenes[scene_index]
    beat_by_id = {beat.beat_id: beat for beat in projection.beats}
    entitlement_by_id = {
        entitlement.entitlement_id: entitlement for entitlement in projection.cut.entitlements
    }
    beats = tuple(beat_by_id[beat_id] for beat_id in scene.beat_ids)
    entitlement_ids = {
        entitlement_id for beat in beats for entitlement_id in beat.entitlement_ids
    }
    return NarrativeSceneRealizationPrompt(
        scene,
        beats,
        tuple(entitlement_by_id[entitlement_id] for entitlement_id in entitlement_ids),
    )


def _prompt(
    projection: NarrativeProjection,
    policy: NarrativeRealizationPolicy,
    provider: NarrativeRealizationProviderIdentity,
) -> NarrativeRealizationPrompt:
    return NarrativeRealizationPrompt(
        NarrativeRealizationRequest("render-1", projection.content_hash),
        policy,
        provider,
        projection.content_hash,
        tuple(_scene_prompt(projection, index) for index in range(len(projection.scenes))),
    )


def _response_hash(*passages: NarrativePassage) -> str:
    return stable_content_hash(
        {
            "provider_response": {
                "passages": [
                    {"beat_ids": list(passage.beat_ids), "text": passage.text}
                    for passage in passages
                ]
            }
        }
    )


def test_request_rejects_a_non_content_addressed_projection():
    with pytest.raises(ValueError, match="projection hash"):
        NarrativeRealizationRequest("render-1", "not-a-hash")


def test_scene_prompt_rejects_an_entitlement_outside_its_beats(projection):
    scene = projection.scenes[0]
    beats = tuple(beat for beat in projection.beats if beat.beat_id in scene.beat_ids)
    unrelated = next(
        item
        for item in projection.cut.entitlements
        if item.entitlement_id
        not in {
            entitlement_id for beat in beats for entitlement_id in beat.entitlement_ids
        }
    )
    with pytest.raises(ValueError, match="exact entitlement closure"):
        NarrativeSceneRealizationPrompt(scene, beats, (unrelated,))


def test_passage_rejects_empty_text():
    with pytest.raises(ValueError, match="text"):
        NarrativePassage(
            "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), ""
        )


@pytest.mark.parametrize("format", ("prose", "unsupported", 1))
def test_policy_requires_a_realization_format(format):
    with pytest.raises(TypeError, match="format"):
        NarrativeRealizationPolicy("realization", "1.0", format, "en")


@pytest.mark.parametrize("assurance", ("citation_bound", "unsupported", 1))
def test_artifact_requires_a_realization_assurance(
    projection, policy, provider, assurance
):
    prompt = _prompt(projection, policy, provider)
    scene_prompt = prompt.scenes[0]
    passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "Text."
    )
    realized_scene = NarrativeRealizedScene(
        "scene-1", scene_prompt.content_hash, _response_hash(passage), (passage,)
    )
    with pytest.raises(TypeError, match="assurance"):
        NarrativeRealizationArtifact(
            "render-1",
            projection.content_hash,
            policy,
            provider,
            prompt.content_hash,
            prompt.schema_hash,
            prompt.prompt_template_hash,
            assurance,
            "accepted",
            (realized_scene,),
        )


@pytest.mark.parametrize("fields", (("", "1.0", "model"), ("provider", "", "model"), ("provider", "1.0", "")))
def test_provider_identity_requires_each_public_component(fields):
    with pytest.raises(ValueError, match="provider"):
        NarrativeRealizationProviderIdentity(*fields)


def test_policy_enforces_language_tone_and_passage_bounds():
    invalid_values = (
        {"language": "x" * 65},
        {"tone_tags": tuple(f"tag-{index}" for index in range(9))},
        {"tone_tags": ("x" * 65,)},
        {"tone_tags": ("calm", "calm")},
        {"maximum_passages_per_scene": 0},
        {"maximum_passages_per_scene": 65},
        {"maximum_passage_characters": 0},
        {"maximum_passage_characters": 65537},
    )
    for kwargs in invalid_values:
        language = kwargs.pop("language", "en")
        with pytest.raises(ValueError):
            NarrativeRealizationPolicy(
                "realization", "1.0", NarrativeRealizationFormat.PROSE, language, **kwargs
            )


def test_policy_sorts_set_like_tone_tags_and_is_frozen():
    policy = NarrativeRealizationPolicy(
        "realization",
        "1.0",
        NarrativeRealizationFormat.SCREENPLAY,
        "en",
        tone_tags=("spare", "calm"),
    )
    assert policy.tone_tags == ("calm", "spare")
    with pytest.raises(FrozenInstanceError):
        policy.language = "fr"


def test_scene_prompt_requires_the_scene_beat_presentation_order(projection):
    first_scene = projection.scenes[0]
    first_beat = projection.beats[0]
    with pytest.raises(ValueError, match="beat.*order"):
        NarrativeSceneRealizationPrompt(first_scene, (first_beat, first_beat), (projection.cut.entitlements[0],))


def test_scene_prompt_rejects_a_dangling_entitlement_id(projection):
    source = projection.beats[0]
    dangling = NarrativeBeat(
        source.beat_id,
        source.kind,
        source.round_index,
        source.sequence,
        source.place_id,
        source.active_pov_agent_id,
        source.agent_ids,
        source.salience,
        source.supporting_artifacts,
        ("missing-entitlement",),
        source.cause_beat_ids,
        source.source_event_id,
        source.phase,
    )
    with pytest.raises(ValueError, match="exact entitlement closure"):
        NarrativeSceneRealizationPrompt(projection.scenes[0], (dangling,), ())


def test_scene_prompt_rejects_a_same_id_beat_with_changed_scene_metadata(projection):
    source = projection.beats[0]
    forged = NarrativeBeat(
        source.beat_id,
        source.kind,
        2,
        source.sequence,
        "lobby",
        source.active_pov_agent_id,
        source.agent_ids,
        source.salience,
        source.supporting_artifacts,
        source.entitlement_ids,
        source.cause_beat_ids,
        source.source_event_id,
        source.phase,
    )
    with pytest.raises(ValueError, match="scene metadata"):
        NarrativeSceneRealizationPrompt(
            projection.scenes[0], (forged,), (projection.cut.entitlements[0],)
        )


def test_scene_prompt_rejects_a_same_id_entitlement_without_beat_support(projection):
    source = projection.cut.entitlements[0]
    forged = NarrativeEntitlement(
        source.entitlement_id,
        source.scope,
        source.owner_agent_id,
        source.round_index,
        source.facts,
        (NarrativeSupportRef("world_event", "event-1", HASH_B),),
    )
    with pytest.raises(ValueError, match="entitlement support"):
        NarrativeSceneRealizationPrompt(projection.scenes[0], (projection.beats[0],), (forged,))


def test_realized_scene_rejects_duplicate_passage_ids(projection):
    prompt = _scene_prompt(projection)
    passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "Text."
    )
    with pytest.raises(ValueError, match="passage ids"):
        NarrativeRealizedScene(
            "scene-1", prompt.content_hash, _response_hash(passage, passage), (passage, passage)
        )


def test_realized_scene_requires_a_hash_of_its_exact_response_payload(projection):
    prompt = _scene_prompt(projection)
    passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "Text."
    )
    with pytest.raises(ValueError, match="provider response hash"):
        NarrativeRealizedScene("scene-1", prompt.content_hash, HASH_B, (passage,))

    realized_scene = NarrativeRealizedScene(
        "scene-1", prompt.content_hash, _response_hash(passage), (passage,)
    )
    assert realized_scene.provider_response_hash == _response_hash(passage)


def test_realized_scene_rejects_a_beat_reused_by_multiple_passages(projection):
    prompt = _scene_prompt(projection)
    first = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "First."
    )
    second = NarrativePassage(
        "passage-2", "scene-1", ("beat-1",), ("entitlement-1",), "Second."
    )
    with pytest.raises(ValueError, match="beat ids"):
        NarrativeRealizedScene(
            "scene-1", prompt.content_hash, _response_hash(first, second), (first, second)
        )


def test_realized_scene_rejects_passages_for_another_scene(projection):
    prompt = _scene_prompt(projection)
    foreign = NarrativePassage(
        "passage-1", "scene-2", ("beat-2",), ("entitlement-2",), "Text."
    )
    with pytest.raises(ValueError, match="reference their realized scene"):
        NarrativeRealizedScene(
            "scene-1", prompt.content_hash, _response_hash(foreign), (foreign,)
        )


def test_prompt_rejects_a_beat_in_multiple_scene_contexts(projection, policy, provider):
    original = _scene_prompt(projection)
    duplicate_scene = NarrativeScene("scene-duplicate", ("beat-1",), 1, 1, "office", None)
    duplicate = NarrativeSceneRealizationPrompt(
        duplicate_scene, (projection.beats[0],), (projection.cut.entitlements[0],)
    )
    with pytest.raises(ValueError, match="beat ids"):
        NarrativeRealizationPrompt(
            NarrativeRealizationRequest("render-1", projection.content_hash),
            policy,
            provider,
            projection.content_hash,
            (original, duplicate),
        )


def test_artifact_preserves_the_supplied_realized_scene_order(projection, policy, provider):
    prompt = _prompt(projection, policy, provider)
    first_passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "First."
    )
    second_passage = NarrativePassage(
        "passage-2", "scene-2", ("beat-2",), ("entitlement-2",), "Second."
    )
    first_scene = NarrativeRealizedScene(
        "scene-1", prompt.scenes[0].content_hash, _response_hash(first_passage), (first_passage,)
    )
    second_scene = NarrativeRealizedScene(
        "scene-2", prompt.scenes[1].content_hash, _response_hash(second_passage), (second_passage,)
    )
    artifact = NarrativeRealizationArtifact(
        "render-1",
        projection.content_hash,
        policy,
        provider,
        prompt.content_hash,
        prompt.schema_hash,
        prompt.prompt_template_hash,
        NarrativeRealizationAssurance.CITATION_BOUND,
        "accepted",
        (second_scene, first_scene),
    )
    assert tuple(scene.scene_id for scene in artifact.scenes) == ("scene-2", "scene-1")


def test_artifact_rejects_distinct_scenes_with_the_same_prompt_hash(projection, policy, provider):
    prompt = _prompt(projection, policy, provider)
    first_passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "First."
    )
    second_passage = NarrativePassage(
        "passage-2", "scene-2", ("beat-2",), ("entitlement-2",), "Second."
    )
    first_scene = NarrativeRealizedScene(
        "scene-1", prompt.scenes[0].content_hash, _response_hash(first_passage), (first_passage,)
    )
    second_scene = NarrativeRealizedScene(
        "scene-2", prompt.scenes[0].content_hash, _response_hash(second_passage), (second_passage,)
    )
    with pytest.raises(ValueError, match="scene prompt hashes"):
        NarrativeRealizationArtifact(
            "render-1",
            projection.content_hash,
            policy,
            provider,
            prompt.content_hash,
            prompt.schema_hash,
            prompt.prompt_template_hash,
            NarrativeRealizationAssurance.CITATION_BOUND,
            "accepted",
            (first_scene, second_scene),
        )


@pytest.mark.parametrize("value", ("not-a-hash", "sha256:ABC", "sha1:" + "a" * 64))
def test_provenance_fields_require_content_hashes(value, projection, policy, provider):
    prompt = _prompt(projection, policy, provider)
    with pytest.raises(ValueError, match="hash"):
        NarrativeRealizationPrompt(
            NarrativeRealizationRequest("render-1", projection.content_hash),
            policy,
            provider,
            value,
            prompt.scenes,
        )
    with pytest.raises(ValueError, match="hash"):
        NarrativeRealizedScene("scene-1", value, HASH_B, ())


def test_artifact_requires_an_accepted_validation_result(projection, policy, provider):
    prompt = _prompt(projection, policy, provider)
    with pytest.raises(ValueError, match="accepted"):
        NarrativeRealizationArtifact(
            "render-1",
            projection.content_hash,
            policy,
            provider,
            prompt.content_hash,
            prompt.schema_hash,
            prompt.prompt_template_hash,
            NarrativeRealizationAssurance.CITATION_BOUND,
            "rejected",
            (),
        )


def test_contract_hashes_are_stable_under_equivalent_construction(projection, provider):
    left = NarrativeRealizationPolicy(
        "realization", "1.0", NarrativeRealizationFormat.PROSE, "en", tone_tags=("calm", "spare")
    )
    right = NarrativeRealizationPolicy(
        "realization", "1.0", NarrativeRealizationFormat.PROSE, "en", tone_tags=("spare", "calm")
    )
    assert left.content_hash == right.content_hash
    left_prompt = _prompt(projection, left, provider)
    right_prompt = _prompt(projection, right, provider)
    assert left_prompt.content_hash == right_prompt.content_hash
    assert left_prompt.schema_hash == right_prompt.schema_hash
    assert left_prompt.prompt_template_hash == right_prompt.prompt_template_hash


def test_prompt_exposes_defensive_authenticated_task_and_schema_provenance(
    projection, policy, provider
):
    prompt = _prompt(projection, policy, provider)

    schema = prompt.response_schema
    template = prompt.prompt_template

    assert prompt.task == "situated_narrative_scene_realization_v1"
    assert template == {
        "task": "situated_narrative_scene_realization_v1",
        "context": "single_scene_exact_entitlement_closure",
    }
    assert schema == {
        "type": "object",
        "required": ["passages"],
        "additional_properties": False,
        "passage": {
            "type": "object",
            "required": ["beat_ids", "text"],
            "additional_properties": False,
        },
    }
    assert stable_content_hash(template) == prompt.prompt_template_hash
    assert stable_content_hash(schema) == prompt.schema_hash

    template["task"] = "forged-task"
    schema["required"].append("forged-field")
    assert prompt.task == "situated_narrative_scene_realization_v1"
    assert prompt.prompt_template["task"] == "situated_narrative_scene_realization_v1"
    assert prompt.response_schema["required"] == ["passages"]
