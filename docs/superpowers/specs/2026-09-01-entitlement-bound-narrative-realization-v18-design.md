# Entitlement-Bound Narrative Realization V18 Design

## Goal

Turn a V17 `NarrativeProjection` into replayable prose or screenplay passages while preserving the projection's canonical scene/beat order, point-of-view authority, exact entitlement boundary, and content-addressed provenance.

## Boundary

V17 remains the only source of narrative truth. V18 realizes an accepted cut; it never creates or mutates world events, cognition, memory, claims, relationships, beats, scenes, or entitlements.

An injected language provider may choose wording and group adjacent beats into passages. Deterministic runtime code chooses the context exposed to each call, derives every entitlement citation, validates scene and beat coverage, bounds output size, captures provider identity and response hashes, and constructs the accepted artifact.

Structural acceptance is not a formal natural-language entailment proof. A citation-bound provider passage proves which facts the provider was allowed to see, not that every phrase is semantically entailed. V18 therefore reports `citation_bound` assurance honestly. A provider-free exact-fact renderer reports `exact_facts` assurance because its text is assembled only from literal entitlement key/value pairs.

## Non-Goals

- No world generation or natural-language-to-world compiler.
- No new action, dialogue, fact, beat, scene, POV, or causal edge creation.
- No vendor SDK, network client, embeddings, image generation, audio generation, or video rendering.
- No cross-scene provider context, hidden chat transcript, or global story prompt.
- No claim that arbitrary provider prose is formally hallucination-free.
- No V1-V9/V10-V17 runtime unification or graphical authoring UI; those are separate V19 and V20 projects.

## Public Contracts

`narrative_dynamics/abm/situated_realization_contracts.py` defines immutable, content-addressed values:

- `NarrativeRealizationFormat`: `prose` or `screenplay`.
- `NarrativeRealizationAssurance`: `citation_bound` or `exact_facts`.
- `NarrativeRealizationProviderIdentity(provider_id, version, model_name)`.
- `NarrativeRealizationPolicy(policy_id, version, format, language, tone_tags=(), maximum_passages_per_scene=8, maximum_passage_characters=4096)`.
- `NarrativeRealizationRequest(request_id, projection_hash)`.
- `NarrativeSceneRealizationPrompt(scene, beats, entitlements)` containing exactly one V17 scene and only the entitlements referenced by its beats.
- `NarrativeRealizationPrompt(request, policy, provider, projection_hash, scenes)` containing canonical scene prompts and fixed schema/template hashes.
- `NarrativePassage(passage_id, scene_id, beat_ids, entitlement_ids, text)`.
- `NarrativeRealizedScene(scene_id, scene_prompt_hash, provider_response_hash, passages)`.
- `NarrativeRealizationArtifact(request_id, projection_hash, policy, provider, prompt_hash, schema_hash, prompt_template_hash, assurance, validation_result, scenes)`.

All tuples preserve presentation order unless their semantics are explicitly set-like. IDs for provider-created passages are derived by runtime code from the scene, ordered beat IDs, exact entitlement IDs, and accepted text. Machine-local paths and provider secrets never enter any contract.

`language` is limited to 64 characters. A policy permits at most eight unique tone tags of at most 64 characters each, at most 64 passages per scene, and at most 65,536 characters per passage. The defaults remain eight passages and 4,096 characters.

## Provider Interface

Applications inject an object with:

```python
identity: NarrativeRealizationProviderIdentity

def complete_json(self, *, task: str, payload: dict[str, object]) -> object:
    ...
```

The task is `situated_narrative_scene_realization_v1`. The provider is called once per scene. A payload contains:

- structured style policy;
- scene ID, place, active POV, and round bounds;
- ordered beats with kind, phase, agents, cause beat IDs, and entitlement IDs;
- only the exact entitlements reachable from those beats, including literal fact key/value pairs;
- the strict output schema and limits.

The response schema is exactly:

```json
{
  "passages": [
    {
      "beat_ids": ["one or more adjacent scene beat IDs"],
      "text": "non-empty realized text"
    }
  ]
}
```

The provider does not submit entitlement IDs. Runtime derives a passage's entitlement IDs as the exact union referenced by its beat IDs, so a response cannot widen authority by naming an unrelated entitlement.

## Validation

`build_narrative_realization_prompt(projection, policy, request, provider)` rejects a request whose projection hash does not equal the supplied V17 projection. It produces one prompt packet per scene in `projection.cut.scene_ids` order. Every packet contains the scene's exact beat sequence and only their exact entitlements.

`compile_narrative_realization(prompt, provider)` captures provider identity before the first call and rejects identity mutation. For every scene it requires:

- an object with the exact `passages` key;
- between one and `maximum_passages_per_scene` passages;
- exact passage keys `beat_ids` and `text`;
- a non-empty contiguous slice of that scene's beat sequence;
- flattened passage beat IDs exactly equal the scene beat IDs, with no omission, duplication, or reorder;
- non-empty text without NUL and no more than `maximum_passage_characters` characters.

The compiler derives entitlement citations and passage IDs, hashes the raw provider response, retains no hidden provider state, and returns an artifact with `citation_bound` assurance and validation result `accepted`.

`realize_narrative_exact_facts(projection, policy, request)` is provider-free. It emits one passage per beat whose text is a deterministic serialization of only that beat's exact entitlement `key=value` facts, and returns `exact_facts` assurance under a fixed built-in provider identity. Beat IDs and kinds remain passage metadata and are not inserted into exact-fact text.

## Replay and Rendering

`replay_narrative_realization(projection, artifact)` performs no provider call. It requires the exact projection hash, reconstructs scene/beat/entitlement closure, verifies canonical coverage and derived passage IDs, then returns the same artifact.

`render_narrative_realization_text(artifact)` joins accepted passage text in canonical scene/passage order. It is a convenience view and never becomes a world fact.

## Privacy and Provenance Invariants

- A scene provider call receives no entitlement absent from that scene's selected beats.
- Limited and multi-POV realization inherits V17's sanitized, owner-private entitlement boundary.
- Cross-scene context is never sent automatically, even when the same provider instance handles every scene.
- Provider identity is captured before invocation and must remain stable.
- Schema hash, prompt-template hash, prompt hash, per-scene context hash, raw-response hash, projection hash, and policy remain in the artifact.
- Repeated exact-fact realization is byte-equivalent and content-address stable.
- Provider prose is presentation data, never admissible evidence for later world or cognitive transitions.

## Public API and Documentation

All V18 contracts and functions are exported from `narrative_dynamics.abm`. README documentation shows an injected JSON provider, exact-fact fallback, replay, and the explicit distinction between citation-bound prose and formally exact fact rendering.
