"""Immutable, content-addressed contracts for V18 narrative realization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeBeat,
    NarrativeEntitlement,
    NarrativeScene,
)


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")

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
_PROMPT_TEMPLATE = {
    "task": "situated_narrative_scene_realization_v1",
    "context": "single_scene_exact_entitlement_closure",
}
_SCHEMA_HASH = stable_content_hash(_RESPONSE_SCHEMA)
_PROMPT_TEMPLATE_HASH = stable_content_hash(_PROMPT_TEMPLATE)


def _text(value: object, *, label: str, maximum_length: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain NUL")
    if maximum_length is not None and len(value) > maximum_length:
        raise ValueError(f"{label} must be at most {maximum_length} characters")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _bounded_integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _tuple_of_text(value: object, *, label: str, require_nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in value)
    if require_nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _provider_response_hash(passages: tuple[NarrativePassage, ...]) -> str:
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


class NarrativeRealizationFormat(str, Enum):
    PROSE = "prose"
    SCREENPLAY = "screenplay"


class NarrativeRealizationAssurance(str, Enum):
    CITATION_BOUND = "citation_bound"
    EXACT_FACTS = "exact_facts"


@dataclass(frozen=True)
class NarrativeRealizationProviderIdentity:
    provider_id: str
    version: str
    model_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, label="realization provider id"))
        object.__setattr__(self, "version", _text(self.version, label="realization provider version"))
        object.__setattr__(self, "model_name", _text(self.model_name, label="realization provider model name"))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "version": self.version,
            "model_name": self.model_name,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeRealizationPolicy:
    policy_id: str
    version: str
    format: NarrativeRealizationFormat
    language: str
    tone_tags: tuple[str, ...] = ()
    maximum_passages_per_scene: int = 8
    maximum_passage_characters: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, label="realization policy id"))
        object.__setattr__(self, "version", _text(self.version, label="realization policy version"))
        if not isinstance(self.format, NarrativeRealizationFormat):
            raise TypeError("realization policy format must be NarrativeRealizationFormat")
        object.__setattr__(self, "language", _text(self.language, label="realization policy language", maximum_length=64))
        tone_tags = _tuple_of_text(self.tone_tags, label="realization policy tone tags")
        if len(tone_tags) > 8:
            raise ValueError("realization policy permits at most eight tone tags")
        if len(set(tone_tags)) != len(tone_tags):
            raise ValueError("realization policy tone tags must be unique")
        if any(len(tag) > 64 for tag in tone_tags):
            raise ValueError("realization policy tone tags must be at most 64 characters")
        object.__setattr__(self, "tone_tags", tuple(sorted(tone_tags)))
        object.__setattr__(
            self,
            "maximum_passages_per_scene",
            _bounded_integer(
                self.maximum_passages_per_scene,
                label="realization policy maximum passages per scene",
                minimum=1,
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "maximum_passage_characters",
            _bounded_integer(
                self.maximum_passage_characters,
                label="realization policy maximum passage characters",
                minimum=1,
                maximum=65536,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "format": self.format.value,
            "language": self.language,
            "tone_tags": list(self.tone_tags),
            "maximum_passages_per_scene": self.maximum_passages_per_scene,
            "maximum_passage_characters": self.maximum_passage_characters,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeRealizationRequest:
    request_id: str
    projection_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, label="realization request id"))
        object.__setattr__(self, "projection_hash", _hash(self.projection_hash, label="realization request projection hash"))

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, "projection_hash": self.projection_hash}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeSceneRealizationPrompt:
    scene: NarrativeScene
    beats: tuple[NarrativeBeat, ...]
    entitlements: tuple[NarrativeEntitlement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene, NarrativeScene):
            raise TypeError("realization scene prompt scene must be a NarrativeScene")
        if not isinstance(self.beats, tuple) or any(not isinstance(item, NarrativeBeat) for item in self.beats):
            raise TypeError("realization scene prompt beats must be a tuple of NarrativeBeat values")
        beat_ids = tuple(item.beat_id for item in self.beats)
        if beat_ids != self.scene.beat_ids:
            raise ValueError("realization scene prompt beats must match scene beat presentation order")
        if (
            self.scene.start_round_index != min(item.round_index for item in self.beats)
            or self.scene.end_round_index != max(item.round_index for item in self.beats)
            or any(item.place_id != self.scene.place_id for item in self.beats)
            or any(item.active_pov_agent_id != self.scene.active_pov_agent_id for item in self.beats)
        ):
            raise ValueError("realization scene prompt scene metadata must match its beats")
        if not isinstance(self.entitlements, tuple) or any(not isinstance(item, NarrativeEntitlement) for item in self.entitlements):
            raise TypeError("realization scene prompt entitlements must be a tuple of NarrativeEntitlement values")
        expected_entitlement_ids = {
            entitlement_id for beat in self.beats for entitlement_id in beat.entitlement_ids
        }
        entitlement_ids = tuple(item.entitlement_id for item in self.entitlements)
        if len(set(entitlement_ids)) != len(entitlement_ids) or set(entitlement_ids) != expected_entitlement_ids:
            raise ValueError("realization scene prompt requires the exact entitlement closure")
        entitlement_by_id = {
            item.entitlement_id: item for item in self.entitlements
        }
        for beat in self.beats:
            entitled_support = {
                (
                    support.artifact_kind,
                    support.artifact_id,
                    support.artifact_hash,
                )
                for entitlement_id in beat.entitlement_ids
                for support in entitlement_by_id[entitlement_id].supporting_artifacts
            }
            if any(
                (
                    support.artifact_kind,
                    support.artifact_id,
                    support.artifact_hash,
                )
                not in entitled_support
                for support in beat.supporting_artifacts
            ):
                raise ValueError("realization scene prompt entitlement support must back every beat support")
        object.__setattr__(self, "beats", tuple(self.beats))
        object.__setattr__(
            self,
            "entitlements",
            tuple(sorted(self.entitlements, key=lambda item: item.entitlement_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene.to_dict(),
            "beats": [item.to_dict() for item in self.beats],
            "entitlements": [item.to_dict() for item in self.entitlements],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeRealizationPrompt:
    request: NarrativeRealizationRequest
    policy: NarrativeRealizationPolicy
    provider: NarrativeRealizationProviderIdentity
    projection_hash: str
    scenes: tuple[NarrativeSceneRealizationPrompt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, NarrativeRealizationRequest):
            raise TypeError("realization prompt request must be a NarrativeRealizationRequest")
        if not isinstance(self.policy, NarrativeRealizationPolicy):
            raise TypeError("realization prompt policy must be a NarrativeRealizationPolicy")
        if not isinstance(self.provider, NarrativeRealizationProviderIdentity):
            raise TypeError("realization prompt provider must be a NarrativeRealizationProviderIdentity")
        projection_hash = _hash(self.projection_hash, label="realization prompt projection hash")
        if self.request.projection_hash != projection_hash:
            raise ValueError("realization prompt projection hash must match request projection hash")
        object.__setattr__(self, "projection_hash", projection_hash)
        if not isinstance(self.scenes, tuple) or any(not isinstance(item, NarrativeSceneRealizationPrompt) for item in self.scenes):
            raise TypeError("realization prompt scenes must be a tuple of NarrativeSceneRealizationPrompt values")
        scene_ids = tuple(item.scene.scene_id for item in self.scenes)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("realization prompt scene ids must be unique")
        beat_ids = tuple(
            beat.beat_id
            for scene in self.scenes
            for beat in scene.beats
        )
        if len(set(beat_ids)) != len(beat_ids):
            raise ValueError("realization prompt beat ids must be unique across scenes")
        object.__setattr__(self, "scenes", tuple(self.scenes))

    @property
    def schema_hash(self) -> str:
        return _SCHEMA_HASH

    @property
    def prompt_template_hash(self) -> str:
        return _PROMPT_TEMPLATE_HASH

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "policy": self.policy.to_dict(),
            "provider": self.provider.to_dict(),
            "projection_hash": self.projection_hash,
            "schema_hash": self.schema_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "scenes": [item.to_dict() for item in self.scenes],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativePassage:
    passage_id: str
    scene_id: str
    beat_ids: tuple[str, ...]
    entitlement_ids: tuple[str, ...]
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "passage_id", _text(self.passage_id, label="realization passage id"))
        object.__setattr__(self, "scene_id", _text(self.scene_id, label="realization passage scene id"))
        beat_ids = _tuple_of_text(self.beat_ids, label="realization passage beat ids", require_nonempty=True)
        if len(set(beat_ids)) != len(beat_ids):
            raise ValueError("realization passage beat ids must be unique")
        object.__setattr__(self, "beat_ids", beat_ids)
        entitlement_ids = _tuple_of_text(
            self.entitlement_ids,
            label="realization passage entitlement ids",
            require_nonempty=True,
        )
        if len(set(entitlement_ids)) != len(entitlement_ids):
            raise ValueError("realization passage entitlement ids must be unique")
        object.__setattr__(self, "entitlement_ids", tuple(sorted(entitlement_ids)))
        object.__setattr__(self, "text", _text(self.text, label="realization passage text"))

    def to_dict(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "scene_id": self.scene_id,
            "beat_ids": list(self.beat_ids),
            "entitlement_ids": list(self.entitlement_ids),
            "text": self.text,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeRealizedScene:
    scene_id: str
    scene_prompt_hash: str
    provider_response_hash: str
    passages: tuple[NarrativePassage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _text(self.scene_id, label="realized scene id"))
        object.__setattr__(self, "scene_prompt_hash", _hash(self.scene_prompt_hash, label="realized scene prompt hash"))
        object.__setattr__(self, "provider_response_hash", _hash(self.provider_response_hash, label="realized scene provider response hash"))
        if not isinstance(self.passages, tuple) or any(not isinstance(item, NarrativePassage) for item in self.passages):
            raise TypeError("realized scene passages must be a tuple of NarrativePassage values")
        passage_ids = tuple(item.passage_id for item in self.passages)
        if len(set(passage_ids)) != len(passage_ids):
            raise ValueError("realized scene passage ids must be unique")
        if any(item.scene_id != self.scene_id for item in self.passages):
            raise ValueError("realized scene passages must reference their realized scene")
        beat_ids = tuple(
            beat_id for passage in self.passages for beat_id in passage.beat_ids
        )
        if len(set(beat_ids)) != len(beat_ids):
            raise ValueError("realized scene passage beat ids must be unique")
        if self.provider_response_hash != _provider_response_hash(self.passages):
            raise ValueError("realized scene provider response hash must match its exact response payload")
        object.__setattr__(self, "passages", tuple(self.passages))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "scene_prompt_hash": self.scene_prompt_hash,
            "provider_response_hash": self.provider_response_hash,
            "passages": [item.to_dict() for item in self.passages],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeRealizationArtifact:
    request_id: str
    projection_hash: str
    policy: NarrativeRealizationPolicy
    provider: NarrativeRealizationProviderIdentity
    prompt_hash: str
    schema_hash: str
    prompt_template_hash: str
    assurance: NarrativeRealizationAssurance
    validation_result: str
    scenes: tuple[NarrativeRealizedScene, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, label="realization artifact request id"))
        object.__setattr__(self, "projection_hash", _hash(self.projection_hash, label="realization artifact projection hash"))
        if not isinstance(self.policy, NarrativeRealizationPolicy):
            raise TypeError("realization artifact policy must be a NarrativeRealizationPolicy")
        if not isinstance(self.provider, NarrativeRealizationProviderIdentity):
            raise TypeError("realization artifact provider must be a NarrativeRealizationProviderIdentity")
        object.__setattr__(self, "prompt_hash", _hash(self.prompt_hash, label="realization artifact prompt hash"))
        schema_hash = _hash(self.schema_hash, label="realization artifact schema hash")
        if schema_hash != _SCHEMA_HASH:
            raise ValueError("realization artifact schema hash must be the fixed realization schema hash")
        object.__setattr__(self, "schema_hash", schema_hash)
        prompt_template_hash = _hash(self.prompt_template_hash, label="realization artifact prompt template hash")
        if prompt_template_hash != _PROMPT_TEMPLATE_HASH:
            raise ValueError("realization artifact prompt template hash must be the fixed realization prompt template hash")
        object.__setattr__(self, "prompt_template_hash", prompt_template_hash)
        if not isinstance(self.assurance, NarrativeRealizationAssurance):
            raise TypeError("realization artifact assurance must be NarrativeRealizationAssurance")
        if self.validation_result != "accepted":
            raise ValueError("realization artifact validation result must be accepted")
        if not isinstance(self.scenes, tuple) or any(not isinstance(item, NarrativeRealizedScene) for item in self.scenes):
            raise TypeError("realization artifact scenes must be a tuple of NarrativeRealizedScene values")
        scene_ids = tuple(item.scene_id for item in self.scenes)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("realization artifact scene ids must be unique")
        scene_prompt_hashes = tuple(item.scene_prompt_hash for item in self.scenes)
        if len(set(scene_prompt_hashes)) != len(scene_prompt_hashes):
            raise ValueError("realization artifact scene prompt hashes must be unique")
        object.__setattr__(self, "scenes", tuple(self.scenes))

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "projection_hash": self.projection_hash,
            "policy": self.policy.to_dict(),
            "provider": self.provider.to_dict(),
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "assurance": self.assurance.value,
            "validation_result": self.validation_result,
            "scenes": [item.to_dict() for item in self.scenes],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
