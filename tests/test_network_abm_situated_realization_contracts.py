from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

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
        "scene-1", scene_prompt.content_hash, HASH_B, (passage,)
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


def test_realized_scene_rejects_duplicate_passage_ids(projection):
    prompt = _scene_prompt(projection)
    passage = NarrativePassage(
        "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), "Text."
    )
    with pytest.raises(ValueError, match="passage ids"):
        NarrativeRealizedScene(
            "scene-1", prompt.content_hash, HASH_B, (passage, passage)
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
