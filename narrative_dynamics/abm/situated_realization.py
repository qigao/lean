"""Scene-local provider compilation for situated narrative projections."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_projection_contracts import NarrativeProjection
from narrative_dynamics.abm.situated_realization_contracts import (
    NarrativePassage,
    NarrativeRealizationArtifact,
    NarrativeRealizationAssurance,
    NarrativeRealizationPolicy,
    NarrativeRealizationPrompt,
    NarrativeRealizationProviderIdentity,
    NarrativeRealizationRequest,
    NarrativeRealizedScene,
    NarrativeSceneRealizationPrompt,
)


_TASK = "situated_narrative_scene_realization_v1"
_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["passages"],
    "additional_properties": False,
    "passage": {
        "type": "object",
        "required": ["beat_ids", "text"],
        "additional_properties": False,
    },
}


def _provider_protocol(
    provider: object,
) -> tuple[NarrativeRealizationProviderIdentity, Callable[..., object]]:
    identity = getattr(provider, "identity", None)
    completion = getattr(provider, "complete_json", None)
    if not isinstance(identity, NarrativeRealizationProviderIdentity):
        raise TypeError(
            "narrative realization provider identity must be an immutable "
            "NarrativeRealizationProviderIdentity"
        )
    if not callable(completion):
        raise TypeError("narrative realization provider must expose complete_json")
    identity_snapshot = NarrativeRealizationProviderIdentity(
        identity.provider_id,
        identity.version,
        identity.model_name,
    )
    return identity_snapshot, completion


def build_narrative_realization_prompt(
    projection: NarrativeProjection,
    policy: NarrativeRealizationPolicy,
    request: NarrativeRealizationRequest,
    provider: object,
) -> NarrativeRealizationPrompt:
    """Bind exact projected scene values to an immutable provider identity."""

    if not isinstance(projection, NarrativeProjection):
        raise TypeError("narrative realization requires a NarrativeProjection")
    if not isinstance(policy, NarrativeRealizationPolicy):
        raise TypeError("narrative realization requires a NarrativeRealizationPolicy")
    if not isinstance(request, NarrativeRealizationRequest):
        raise TypeError("narrative realization requires a NarrativeRealizationRequest")
    if request.projection_hash != projection.content_hash:
        raise ValueError(
            "narrative realization request projection hash must match the supplied projection"
        )
    provider_identity, _ = _provider_protocol(provider)

    return _build_prompt(projection, policy, request, provider_identity)


def _build_prompt(
    projection: NarrativeProjection,
    policy: NarrativeRealizationPolicy,
    request: NarrativeRealizationRequest,
    provider_identity: NarrativeRealizationProviderIdentity,
) -> NarrativeRealizationPrompt:

    scene_by_id = {scene.scene_id: scene for scene in projection.scenes}
    beat_by_id = {beat.beat_id: beat for beat in projection.beats}
    entitlement_by_id = {
        entitlement.entitlement_id: entitlement
        for entitlement in projection.cut.entitlements
    }
    scene_prompts = []
    for scene_id in projection.cut.scene_ids:
        scene = scene_by_id[scene_id]
        beats = tuple(beat_by_id[beat_id] for beat_id in scene.beat_ids)
        entitlement_ids = {
            entitlement_id
            for beat in beats
            for entitlement_id in beat.entitlement_ids
        }
        entitlements = tuple(
            entitlement_by_id[entitlement_id]
            for entitlement_id in sorted(entitlement_ids)
        )
        scene_prompts.append(NarrativeSceneRealizationPrompt(scene, beats, entitlements))

    return NarrativeRealizationPrompt(
        request,
        policy,
        provider_identity,
        projection.content_hash,
        tuple(scene_prompts),
    )


def _scene_payload(
    scene_prompt: NarrativeSceneRealizationPrompt,
    policy: NarrativeRealizationPolicy,
) -> dict[str, object]:
    payload = scene_prompt.to_dict()
    payload["policy"] = policy.to_dict()
    payload["response_schema"] = deepcopy(_RESPONSE_SCHEMA)
    payload["limits"] = {
        "maximum_passages": policy.maximum_passages_per_scene,
        "maximum_passage_characters": policy.maximum_passage_characters,
    }
    return payload


def _passage_id(
    scene_id: str,
    beat_ids: tuple[str, ...],
    entitlement_ids: tuple[str, ...],
    text: str,
) -> str:
    return stable_content_hash(
        {
            "scene_id": scene_id,
            "beat_ids": list(beat_ids),
            "entitlement_ids": list(entitlement_ids),
            "text": text,
        }
    )


def _validated_response_passages(
    response: object,
    scene_prompt: NarrativeSceneRealizationPrompt,
    policy: NarrativeRealizationPolicy,
    all_prompt_beat_ids: frozenset[str],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    if not isinstance(response, dict) or set(response) != {"passages"}:
        raise ValueError("provider must return an exact realization response object")
    raw_passages = response["passages"]
    if not isinstance(raw_passages, list):
        raise ValueError("provider must return an exact realization response passage list")
    if not 1 <= len(raw_passages) <= policy.maximum_passages_per_scene:
        raise ValueError("provider realization response passage count is outside policy")

    canonical_beat_ids = scene_prompt.scene.beat_ids
    local_beat_ids = frozenset(canonical_beat_ids)
    canonical_index = {
        beat_id: index for index, beat_id in enumerate(canonical_beat_ids)
    }
    validated: list[tuple[tuple[str, ...], str]] = []
    flattened: list[str] = []
    for item in raw_passages:
        if not isinstance(item, dict) or set(item) != {"beat_ids", "text"}:
            raise ValueError("provider must return exact passage response objects")
        raw_beat_ids = item["beat_ids"]
        if (
            not isinstance(raw_beat_ids, list)
            or not raw_beat_ids
            or any(not isinstance(beat_id, str) or not beat_id for beat_id in raw_beat_ids)
        ):
            raise ValueError("provider passage beat_ids must be a non-empty string list")
        beat_ids = tuple(raw_beat_ids)
        foreign = set(beat_ids).difference(local_beat_ids)
        if foreign:
            if foreign.intersection(all_prompt_beat_ids):
                raise ValueError("provider passage must cite only scene-local beat ids")
            raise ValueError("provider passage cites an unknown beat id")
        if len(set(beat_ids)) != len(beat_ids) or set(beat_ids).intersection(flattened):
            raise ValueError("provider response contains a duplicate beat id")

        positions = tuple(canonical_index[beat_id] for beat_id in beat_ids)
        if positions != tuple(sorted(positions)):
            raise ValueError("provider response must preserve canonical beat order")
        if positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError("provider passage beat ids must be contiguous")

        text = item["text"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or "\x00" in text
            or len(text) > policy.maximum_passage_characters
        ):
            raise ValueError("provider passage text violates realization policy")
        flattened.extend(beat_ids)
        validated.append((beat_ids, text))

    if set(flattened) != local_beat_ids:
        raise ValueError("provider response must provide complete beat coverage")
    if tuple(flattened) != canonical_beat_ids:
        raise ValueError("provider response must preserve canonical beat order")
    return tuple(validated)


def compile_narrative_realization(
    prompt: NarrativeRealizationPrompt,
    provider: object,
) -> NarrativeRealizationArtifact:
    """Call the provider once for each scene-local prompt packet."""

    if not isinstance(prompt, NarrativeRealizationPrompt):
        raise TypeError("narrative realization compilation requires a prompt")
    provider_identity, _ = _provider_protocol(provider)
    if provider_identity != prompt.provider:
        raise ValueError("narrative realization provider identity must match the prompt")
    provider_identity_hash = provider_identity.content_hash
    bound_prompt = NarrativeRealizationPrompt(
        prompt.request,
        prompt.policy,
        provider_identity,
        prompt.projection_hash,
        prompt.scenes,
    )

    all_prompt_beat_ids = frozenset(
        beat.beat_id for scene in bound_prompt.scenes for beat in scene.beats
    )
    realized_scenes = []
    for scene_prompt in bound_prompt.scenes:
        current_identity, current_completion = _provider_protocol(provider)
        if current_identity.content_hash != provider_identity_hash:
            raise ValueError("narrative realization provider identity changed during compilation")
        response = current_completion(
            task=_TASK,
            payload=_scene_payload(scene_prompt, bound_prompt.policy),
        )
        current_identity, _ = _provider_protocol(provider)
        if current_identity.content_hash != provider_identity_hash:
            raise ValueError("narrative realization provider identity changed during compilation")

        beat_by_id = {beat.beat_id: beat for beat in scene_prompt.beats}
        passages = []
        for beat_ids, text in _validated_response_passages(
            response,
            scene_prompt,
            bound_prompt.policy,
            all_prompt_beat_ids,
        ):
            entitlement_ids = tuple(
                sorted(
                    {
                        entitlement_id
                        for beat_id in beat_ids
                        for entitlement_id in beat_by_id[beat_id].entitlement_ids
                    }
                )
            )
            passages.append(
                NarrativePassage(
                    _passage_id(
                        scene_prompt.scene.scene_id,
                        beat_ids,
                        entitlement_ids,
                        text,
                    ),
                    scene_prompt.scene.scene_id,
                    beat_ids,
                    entitlement_ids,
                    text,
                )
            )
        passages_tuple = tuple(passages)
        response_hash = stable_content_hash(
            {
                "provider_response": {
                    "passages": [
                        {"beat_ids": list(passage.beat_ids), "text": passage.text}
                        for passage in passages_tuple
                    ]
                }
            }
        )
        realized_scenes.append(
            NarrativeRealizedScene(
                scene_prompt.scene.scene_id,
                scene_prompt.content_hash,
                response_hash,
                passages_tuple,
            )
        )

    return NarrativeRealizationArtifact(
        bound_prompt.request.request_id,
        bound_prompt.projection_hash,
        bound_prompt.policy,
        bound_prompt.provider,
        bound_prompt.content_hash,
        bound_prompt.schema_hash,
        bound_prompt.prompt_template_hash,
        NarrativeRealizationAssurance.CITATION_BOUND,
        "accepted",
        tuple(realized_scenes),
    )


def replay_narrative_realization(
    projection: NarrativeProjection,
    artifact: NarrativeRealizationArtifact,
) -> NarrativeRealizationArtifact:
    """Verify an artifact solely against its supplied projection provenance."""

    if not isinstance(projection, NarrativeProjection):
        raise TypeError("narrative realization replay requires a NarrativeProjection")
    if not isinstance(artifact, NarrativeRealizationArtifact):
        raise TypeError("narrative realization replay requires a realization artifact")
    if artifact.validation_result != "accepted":
        raise ValueError("narrative realization replay requires an accepted artifact")
    if artifact.projection_hash != projection.content_hash:
        raise ValueError("narrative realization replay projection hash mismatch")
    if artifact.assurance is not NarrativeRealizationAssurance.CITATION_BOUND:
        raise ValueError("narrative realization replay requires citation-bound assurance")

    request = NarrativeRealizationRequest(
        artifact.request_id,
        projection.content_hash,
    )
    prompt = _build_prompt(
        projection,
        artifact.policy,
        request,
        artifact.provider,
    )
    if artifact.prompt_hash != prompt.content_hash:
        raise ValueError("narrative realization replay prompt hash mismatch")
    if artifact.schema_hash != prompt.schema_hash:
        raise ValueError("narrative realization replay schema hash mismatch")
    if artifact.prompt_template_hash != prompt.prompt_template_hash:
        raise ValueError("narrative realization replay prompt template hash mismatch")

    expected_scene_ids = tuple(
        scene_prompt.scene.scene_id for scene_prompt in prompt.scenes
    )
    artifact_scene_ids = tuple(scene.scene_id for scene in artifact.scenes)
    if artifact_scene_ids != expected_scene_ids:
        raise ValueError("narrative realization replay scene order mismatch")

    all_prompt_beat_ids = frozenset(
        beat.beat_id for scene in prompt.scenes for beat in scene.beats
    )
    for scene_prompt, realized_scene in zip(prompt.scenes, artifact.scenes):
        if realized_scene.scene_prompt_hash != scene_prompt.content_hash:
            raise ValueError("narrative realization replay scene prompt hash mismatch")
        if any(
            passage.scene_id != realized_scene.scene_id
            for passage in realized_scene.passages
        ):
            raise ValueError("narrative realization replay passage mapping mismatch")
        response = {
            "passages": [
                {"beat_ids": list(passage.beat_ids), "text": passage.text}
                for passage in realized_scene.passages
            ]
        }
        validated = _validated_response_passages(
            response,
            scene_prompt,
            artifact.policy,
            all_prompt_beat_ids,
        )
        expected_response_hash = stable_content_hash(
            {"provider_response": response}
        )
        if realized_scene.provider_response_hash != expected_response_hash:
            raise ValueError("narrative realization replay provider response hash mismatch")

        beat_by_id = {beat.beat_id: beat for beat in scene_prompt.beats}
        for passage, (beat_ids, text) in zip(realized_scene.passages, validated):
            expected_entitlement_ids = tuple(
                sorted(
                    {
                        entitlement_id
                        for beat_id in beat_ids
                        for entitlement_id in beat_by_id[beat_id].entitlement_ids
                    }
                )
            )
            if passage.entitlement_ids != expected_entitlement_ids:
                raise ValueError(
                    "narrative realization replay passage entitlement closure mismatch"
                )
            expected_passage_id = _passage_id(
                realized_scene.scene_id,
                beat_ids,
                expected_entitlement_ids,
                text,
            )
            if passage.passage_id != expected_passage_id:
                raise ValueError("narrative realization replay passage id mismatch")
    return artifact


def render_narrative_realization_text(
    artifact: NarrativeRealizationArtifact,
) -> str:
    """Flatten accepted passages in artifact presentation order."""

    if not isinstance(artifact, NarrativeRealizationArtifact):
        raise TypeError("narrative realization rendering requires a realization artifact")
    return "\n\n".join(
        passage.text
        for scene in artifact.scenes
        for passage in scene.passages
    )


__all__ = (
    "build_narrative_realization_prompt",
    "compile_narrative_realization",
    "replay_narrative_realization",
    "render_narrative_realization_text",
)
