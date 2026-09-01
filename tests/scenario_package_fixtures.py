from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from narrative_dynamics.abm.scenario_package_contracts import (
    SCENARIO_DOCUMENT_SCHEMA,
    SCENARIO_PACKAGE_SCHEMA,
)


RESOURCE_HASH = "sha256:" + "a" * 64
PREVIEW_HASH = "sha256:" + "b" * 64


def _canonical_hash_value(value: object) -> object:
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (key, _canonical_hash_value(value[key]))
                for key in sorted(value)
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_canonical_hash_value(item) for item in value))
    raise TypeError("test canonical hash accepts JSON values only")


def _canonical_content_hash(value: object) -> str:
    encoded = json.dumps(
        _canonical_hash_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _document(value: dict[str, object]) -> dict[str, object]:
    return {"schema": SCENARIO_DOCUMENT_SCHEMA, "value": value}


def _likelihoods(action_ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "action_id": action_id,
            "hypothesis_id": hypothesis_id,
            "symbol_id": symbol_id,
            "probability": 0.8 if hypothesis_id == expected else 0.2,
        }
        for action_id in action_ids
        for hypothesis_id in ("file-found", "file-missing")
        for symbol_id, expected in (
            ("evidence-found", "file-found"),
            ("evidence-missing", "file-missing"),
        )
    ]


def _rewards(action_ids: tuple[str, ...]) -> list[dict[str, object]]:
    values = {
        "inspect": 3.0,
        "move": 1.0,
        "take": 2.0,
        "tell": 2.0,
        "wait": 0.0,
    }
    return [
        {
            "goal_id": "resolve-case",
            "hypothesis_id": hypothesis_id,
            "action_id": action_id,
            "value": (
                -values[action_kind]
                if hypothesis_id == "file-missing" and action_kind == "tell"
                else values[action_kind]
            ),
        }
        for hypothesis_id in ("file-found", "file-missing")
        for action_id, action_kind in (
            (item, next(kind for prefix, kind in (
                ("inspect", "inspect"),
                ("move", "move"),
                ("take", "take"),
                ("tell", "tell"),
                ("wait", "wait"),
            ) if f"-{prefix}" in item))
            for item in action_ids
        )
    ]


def _agent_value(
    agent_id: str,
    role: str,
    initial_place: str,
    direct_grants: tuple[str, ...],
) -> dict[str, object]:
    action_ids = (
        f"{agent_id}-inspect-file",
        f"{agent_id}-move-archive",
        f"{agent_id}-take-file",
        f"{agent_id}-tell-status",
        f"{agent_id}-wait",
    )
    return {
        "agent_id": agent_id,
        "body": {
            "role": role,
            "initial_place": initial_place,
            "inventory_capacity": 1,
        },
        "perception": {
            "max_visual_cost": 5.0,
            "minimum_detectable_sound": 10.0,
            "minimum_clear_sound": 30.0,
        },
        "cognition": {
            "hypotheses": [
                {"hypothesis_id": "file-found", "description": "The missing file has been found."},
                {"hypothesis_id": "file-missing", "description": "The file remains missing."},
            ],
            "prior_belief": {"file-found": 0.4, "file-missing": 0.6},
            "observation_symbols": [
                {"symbol_id": "evidence-found", "description": "Evidence supports finding the file."},
                {"symbol_id": "evidence-missing", "description": "Evidence supports the file remaining missing."},
            ],
            "observation_rules": [
                {
                    "rule_id": f"{agent_id}-inspect-found",
                    "symbol_id": "evidence-found",
                    "likelihood_action_id": f"{agent_id}-inspect-file",
                    "event_kind": "inspect",
                    "outcome": "inspected",
                    "detail_name": "status",
                    "detail_value": "found",
                }
            ],
            "likelihoods": _likelihoods(action_ids),
            "actions": [
                {
                    "action_id": f"{agent_id}-inspect-file",
                    "kind": "inspect",
                    "target_id": "case-file",
                    "message": None,
                    "required_place_ids": ["archive"],
                    "repeatable": False,
                    "source_event_kinds": [],
                },
                {
                    "action_id": f"{agent_id}-move-archive",
                    "kind": "move",
                    "target_id": "meeting-archive",
                    "message": None,
                    "required_place_ids": ["meeting"],
                    "repeatable": False,
                    "source_event_kinds": [],
                },
                {
                    "action_id": f"{agent_id}-take-file",
                    "kind": "take",
                    "target_id": "case-file",
                    "message": None,
                    "required_place_ids": ["archive"],
                    "repeatable": False,
                    "source_event_kinds": [],
                },
                {
                    "action_id": f"{agent_id}-tell-status",
                    "kind": "tell",
                    "target_id": None,
                    "message": "The case file has been found.",
                    "required_place_ids": ["meeting"],
                    "repeatable": False,
                    "source_event_kinds": ["inspect", "tell"],
                },
                {
                    "action_id": f"{agent_id}-wait",
                    "kind": "wait",
                    "target_id": None,
                    "message": None,
                    "required_place_ids": [],
                    "repeatable": True,
                    "source_event_kinds": [],
                },
            ],
            "action_schedule": [list(action_ids), [f"{agent_id}-tell-status", f"{agent_id}-wait"]],
            "transitions": [],
            "goals": [
                {
                    "goal_id": "resolve-case",
                    "description": "Resolve the missing-document dispute.",
                    "weight": 1.0,
                }
            ],
            "rewards": _rewards(action_ids),
            "discount": 0.8,
            "beta": 4.0,
        },
        "memory": {
            "percept_policy": {
                "policy_id": "law-firm-percept-memory",
                "version": "1",
                "fidelities": [
                    {"fidelity": "detected", "confidence": 0.25, "salience": 0.35},
                    {"fidelity": "identified", "confidence": 0.6, "salience": 0.55},
                    {"fidelity": "exact", "confidence": 1.0, "salience": 1.0},
                ],
            },
            "channel_policy": {
                "policy_id": "law-firm-channel-memory",
                "version": "1",
                "channels": [
                    {"channel": "self", "confidence": 1.0, "salience": 0.5},
                    {"channel": "visual", "confidence": 0.9, "salience": 0.65},
                    {"channel": "auditory", "confidence": 0.7, "salience": 0.75},
                    {"channel": "inspection", "confidence": 1.0, "salience": 1.0},
                ],
            },
            "recall": {
                "max_memories_per_round": 4,
                "cues": [
                    {
                        "cue_id": "case-file-recall",
                        "text": "case file",
                        "required_place_ids": ["meeting"],
                        "event_kinds": ["inspect", "tell"],
                        "channels": ["auditory", "inspection"],
                        "min_confidence": 0.5,
                        "limit": 3,
                    }
                ],
            },
        },
        "social": {
            "topics": [
                {
                    "topic_id": "file-location",
                    "symbol_ids": ["evidence-found", "evidence-missing"],
                }
            ]
        },
        "knowledge_grants": list(direct_grants),
    }


def write_law_firm_package(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    documents: list[tuple[str, str, str, dict[str, object]]] = [
        (
            "physical.world",
            "world",
            "physical/world.json",
            {
                "model_id": "law-firm-world",
                "version": "1",
                "places": [
                    {"place_id": "lobby", "label": "Public lobby"},
                    {"place_id": "meeting", "label": "Meeting room"},
                    {"place_id": "archive", "label": "Restricted archive"},
                ],
                "passages": [
                    {
                        "passage_id": "lobby-meeting",
                        "source_place_id": "lobby",
                        "target_place_id": "meeting",
                        "initially_open": True,
                    },
                    {
                        "passage_id": "meeting-archive",
                        "source_place_id": "meeting",
                        "target_place_id": "archive",
                        "initially_open": False,
                    },
                ],
                "objects": [
                    {
                        "object_id": "case-file",
                        "kind": "document",
                        "initial_place_id": "archive",
                        "portable": True,
                        "evidence": [{"name": "status", "value": "found"}],
                    }
                ],
            },
        ),
        (
            "physical.perception",
            "perception",
            "physical/perception.json",
            {
                "model_id": "law-firm-perception",
                "version": "1",
                "edges": [
                    {
                        "edge_id": "visual-lobby-meeting",
                        "layer": "visibility",
                        "source_place_id": "lobby",
                        "target_place_id": "meeting",
                        "cost": 1.0,
                        "activation": "passage_open",
                        "passage_id": "lobby-meeting",
                    },
                    {
                        "edge_id": "auditory-meeting-archive",
                        "layer": "auditory",
                        "source_place_id": "meeting",
                        "target_place_id": "archive",
                        "cost": 12.0,
                        "activation": "passage_closed",
                        "passage_id": "meeting-archive",
                    },
                    {
                        "edge_id": "interaction-meeting",
                        "layer": "interaction",
                        "source_place_id": "meeting",
                        "target_place_id": "meeting",
                        "cost": 0.0,
                        "activation": "always",
                        "passage_id": None,
                    },
                ],
                "signal_profiles": [
                    {"kind": "wait", "visually_observable": False, "auditory_intensity": None},
                    {"kind": "move", "visually_observable": True, "auditory_intensity": 15.0},
                    {"kind": "inspect", "visually_observable": True, "auditory_intensity": None},
                    {"kind": "take", "visually_observable": True, "auditory_intensity": 10.0},
                    {"kind": "tell", "visually_observable": True, "auditory_intensity": 60.0},
                ],
            },
        ),
        (
            "physical.initial_state",
            "initial",
            "physical/initial-state.json",
            {
                "agents": [
                    {"agent_id": "alice", "place_id": "meeting"},
                    {"agent_id": "bob", "place_id": "lobby"},
                    {"agent_id": "client", "place_id": "lobby"},
                ],
                "objects": [
                    {
                        "object_id": "case-file",
                        "place_id": "archive",
                        "holder_agent_id": None,
                    }
                ],
                "passages": [
                    {"passage_id": "lobby-meeting", "open": True},
                    {"passage_id": "meeting-archive", "open": False},
                ],
            },
        ),
        (
            "physical.map",
            "map",
            "physical/map.json",
            {
                "map_id": "law-firm-map",
                "version": "1",
                "places": [
                    {"place_id": "lobby", "center_x": 0, "center_y": 0, "width": 4, "depth": 4, "height": 2.8},
                    {"place_id": "meeting", "center_x": 5, "center_y": 0, "width": 5, "depth": 4, "height": 2.8},
                    {"place_id": "archive", "center_x": 11, "center_y": 0, "width": 4, "depth": 3, "height": 2.8},
                ],
                "passages": [
                    {
                        "passage_id": "lobby-meeting",
                        "source_place_id": "lobby",
                        "target_place_id": "meeting",
                        "center_x": 2.25,
                        "center_y": 0,
                        "width": 0.5,
                        "depth": 1.2,
                        "height": 2.1,
                    },
                    {
                        "passage_id": "meeting-archive",
                        "source_place_id": "meeting",
                        "target_place_id": "archive",
                        "center_x": 8,
                        "center_y": 0,
                        "width": 0.5,
                        "depth": 1.2,
                        "height": 2.1,
                    },
                ],
            },
        ),
        (
            "social.institutions",
            "institutions",
            "social/institutions.json",
            {
                "institutions": [
                    {"institution_id": "firm", "institution_kind": "law_firm", "parent_institution_id": None},
                    {"institution_id": "legal-team", "institution_kind": "department", "parent_institution_id": "firm"},
                ],
                "memberships": [
                    {"agent_id": "alice", "institution_id": "firm", "role_id": "partner"},
                    {"agent_id": "bob", "institution_id": "legal-team", "role_id": "lawyer"},
                    {"agent_id": "client", "institution_id": "firm", "role_id": "client"},
                ],
            },
        ),
        (
            "social.relationships",
            "relationships",
            "social/relationships.json",
            {
                "model_id": "law-firm-social-memory",
                "version": "1",
                "memory_cognitive_model_id": "law-firm-memory-cognition",
                "relationships": [
                    {"source_agent_id": "alice", "target_agent_id": "bob", "relationship_type": "supervises", "strength": 0.8},
                    {"source_agent_id": "alice", "target_agent_id": "client", "relationship_type": "represents", "strength": 0.9},
                    {"source_agent_id": "bob", "target_agent_id": "alice", "relationship_type": "reports_to", "strength": 0.9},
                    {"source_agent_id": "bob", "target_agent_id": "client", "relationship_type": "advises", "strength": 0.6},
                    {"source_agent_id": "client", "target_agent_id": "alice", "relationship_type": "trusts", "strength": 0.7},
                    {"source_agent_id": "client", "target_agent_id": "bob", "relationship_type": "depends_on", "strength": 0.5},
                ],
                "policy": {
                    "initial_source_trust": 0.5,
                    "confirmation_rate": 0.2,
                    "contradiction_rate": 0.4,
                    "confirmation_affinity_delta": 0.1,
                    "contradiction_affinity_delta": 0.2,
                    "max_unresolved_age_rounds": 20,
                    "max_active_claims": 100,
                },
            },
        ),
        (
            "social.norms",
            "norms",
            "social/norms.json",
            {
                "norms": [
                    {
                        "norm_id": "client-confidentiality",
                        "subject_role_id": "client",
                        "effect": "deny_action",
                        "priority": 10,
                        "action_id": "client-take-file",
                        "resource_id": None,
                        "relationship_type": None,
                        "target_scope_id": "case-file",
                        "descriptive_text": "Clients may not remove the private case file.",
                    },
                    {
                        "norm_id": "lawyer-file-access",
                        "subject_role_id": "lawyer",
                        "effect": "require_knowledge_grant",
                        "priority": 5,
                        "action_id": None,
                        "resource_id": "case-file-brief",
                        "relationship_type": None,
                        "target_scope_id": "firm",
                        "descriptive_text": "Lawyers require a direct file grant.",
                    },
                ]
            },
        ),
        ("agent", "alice", "agents/alice.json", _agent_value("alice", "partner", "meeting", ("case-file-brief",))),
        ("agent", "bob", "agents/bob.json", _agent_value("bob", "lawyer", "lobby", ("case-file-brief",))),
        ("agent", "client", "agents/client.json", _agent_value("client", "client", "lobby", ("client-guide",))),
        (
            "story.outline",
            "outline",
            "story/outline.json",
            {
                "plan_id": "law-firm-outline",
                "version": "1",
                "mode": "hybrid",
                "acts": [{"act_id": "case", "scene_ids": ["discover", "confront"]}],
                "scenes": [
                    {
                        "scene_id": "discover",
                        "place_ids": ["archive"],
                        "participant_agent_ids": ["alice", "bob"],
                        "preconditions": [
                            {"kind": "object_at", "subject_id": "case-file", "object_id": "archive", "value": None}
                        ],
                        "exit_predicates": [
                            {"kind": "agent_holds", "subject_id": "alice", "object_id": "case-file", "value": None}
                        ],
                        "allowed_intervention_kinds": ["open_passage"],
                        "desired_outcome_ids": ["document-discovered"],
                        "maximum_rounds": 4,
                    },
                    {
                        "scene_id": "confront",
                        "place_ids": ["meeting"],
                        "participant_agent_ids": ["alice", "bob", "client"],
                        "preconditions": [
                            {"kind": "agent_holds", "subject_id": "alice", "object_id": "case-file", "value": None}
                        ],
                        "exit_predicates": [
                            {"kind": "belief_at_least", "subject_id": "client", "object_id": "file-found", "value": 0.7}
                        ],
                        "allowed_intervention_kinds": ["move_object"],
                        "desired_outcome_ids": ["dispute-addressed"],
                        "maximum_rounds": 6,
                    },
                ],
                "dependencies": [
                    {"predecessor_scene_id": "discover", "successor_scene_id": "confront"}
                ],
                "continuity_predicates": [
                    {"kind": "passage_open", "subject_id": "lobby-meeting", "object_id": None, "value": True}
                ],
                "terminal_predicates": [
                    {"kind": "belief_at_least", "subject_id": "client", "object_id": "file-found", "value": 0.8}
                ],
            },
        ),
        (
            "story.interventions",
            "interventions",
            "story/interventions.json",
            {"intervention_kinds": ["move_object", "open_passage"]},
        ),
        (
            "knowledge.catalog",
            "catalog",
            "knowledge/catalog.json",
            {
                "resources": [
                    {
                        "resource_id": "statute",
                        "kind": "document",
                        "content_hash": RESOURCE_HASH,
                        "uri": "https://example.test/statute.pdf",
                        "media_type": "application/pdf",
                        "language": "en",
                        "version": "2026-09-01",
                        "authority": "official",
                        "license_tag": "CC-BY-4.0",
                        "concept_ids": ["law", "evidence"],
                        "index_id": "index-statute",
                    },
                    {
                        "resource_id": "case-file-brief",
                        "kind": "document",
                        "content_hash": RESOURCE_HASH,
                        "uri": "https://example.test/private/case-file.pdf",
                        "media_type": "application/pdf",
                        "language": "en",
                        "version": "1",
                        "authority": "firm",
                        "license_tag": "private",
                        "concept_ids": ["case-file"],
                        "index_id": "index-case-file",
                    },
                    {
                        "resource_id": "client-guide",
                        "kind": "document",
                        "content_hash": RESOURCE_HASH,
                        "uri": "https://example.test/client-guide.pdf",
                        "media_type": "application/pdf",
                        "language": "en",
                        "version": "1",
                        "authority": "firm",
                        "license_tag": "client",
                        "concept_ids": ["procedure"],
                        "index_id": None,
                    },
                ]
            },
        ),
        (
            "knowledge.access",
            "access",
            "knowledge/access.json",
            {
                "grants": [
                    {"subject_scope": "public", "subject_id": None, "resource_ids": ["statute"]},
                    {"subject_scope": "agent", "subject_id": "alice", "resource_ids": ["case-file-brief"]},
                    {"subject_scope": "agent", "subject_id": "bob", "resource_ids": ["case-file-brief"]},
                    {"subject_scope": "agent", "subject_id": "client", "resource_ids": ["client-guide"]},
                ]
            },
        ),
        (
            "asset.catalog",
            "catalog",
            "assets/catalog.json",
            {
                "resources": [
                    {
                        "resource_id": "archive-cabinet",
                        "kind": "model_3d",
                        "content_hash": RESOURCE_HASH,
                        "uri": "https://example.test/archive-cabinet.glb",
                        "media_type": "model/gltf-binary",
                        "authority": "official",
                        "license_tag": "CC-BY-4.0",
                        "dimensions": [2.0, 0.6, 2.2],
                        "unit": "m",
                        "format": "glb",
                        "collection_id": "archive-props",
                        "rig_metadata_ids": [],
                        "collision_metadata_ids": ["cabinet-collider"],
                        "preview_hash": PREVIEW_HASH,
                    }
                ],
                "grants": [
                    {"subject_scope": "public", "subject_id": None, "resource_ids": ["archive-cabinet"]}
                ],
            },
        ),
        (
            "run",
            "run",
            "run.json",
            {
                "mode": "hybrid",
                "maximum_rounds": 12,
                "checkpoint_interval": 3,
                "maximum_output_records": 100,
                "allowed_output_kinds": ["network.metrics", "story.progress"],
                "public_journal": True,
                "blender_mode": "final_blend",
                "deterministic_seed": 7,
                "maximum_resource_bytes": 4096,
                "runtime_model": {
                    "model_id": "law-firm-network",
                    "version": "1",
                    "percept_memory_model_id": "law-firm-percept-memory-cognition",
                    "tracked_hypothesis_id": "file-found",
                    "adoption_threshold": 0.7,
                    "relationship_trust_threshold": 0.5,
                },
                "fallbacks": {},
            },
        ),
    ]

    locators: list[dict[str, object]] = []
    for role, logical_id, relative_path, value in documents:
        document = _document(value)
        _write_json(root / relative_path, document)
        locators.append(
            {
                "role": role,
                "logical_id": logical_id,
                "relative_path": relative_path,
                "expected_hash": _canonical_content_hash(document),
            }
        )
    _write_json(
        root / "scenario-package.json",
        {
            "schema": SCENARIO_PACKAGE_SCHEMA,
            "scenario_id": "law-firm-case",
            "version": "1",
            "documents": locators,
        },
    )
    return root


def mutate_json(path: Path, json_pointer: str, replacement: object) -> None:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    current: object = document["value"]
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in json_pointer.split("/")[1:]]
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]  # type: ignore[index]
    final = tokens[-1]
    if isinstance(current, list):
        current[int(final)] = replacement
    else:
        current[final] = replacement  # type: ignore[index]
    _write_json(Path(path), document)


def refresh_manifest_hash(root: Path, role: str, logical_id: str) -> None:
    root = Path(root)
    manifest_path = root / "scenario-package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locator = next(
        item
        for item in manifest["documents"]
        if item["role"] == role and item["logical_id"] == logical_id
    )
    document = json.loads((root / locator["relative_path"]).read_text(encoding="utf-8"))
    locator["expected_hash"] = _canonical_content_hash(document)
    _write_json(manifest_path, manifest)
