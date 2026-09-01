"""Headless assertions for a generated situated replay `.blend`."""

from __future__ import annotations

import bpy


required_collections = {"ND_Map", "ND_Agents", "ND_Labels", "ND_Events"}
missing = required_collections - set(bpy.data.collections.keys())
if missing:
    raise AssertionError(f"missing replay collections: {sorted(missing)}")

scene = bpy.context.scene
for key in (
    "nd_schema",
    "nd_replay_hash",
    "nd_runtime_model_hash",
    "nd_trajectory_hash",
    "nd_world_model_hash",
    "nd_spatial_map_hash",
):
    if key not in scene:
        raise AssertionError(f"missing scene provenance: {key}")

agent = bpy.data.objects.get("Agent::alice")
if agent is None:
    raise AssertionError("missing Alice stick-figure root")
if agent.animation_data is None or agent.animation_data.action is None:
    raise AssertionError("Alice root has no keyframed action")
if not scene.timeline_markers:
    raise AssertionError("replay has no objective event markers")
if bpy.data.objects.get("Place::records") is None:
    raise AssertionError("spatial map did not create the records place")
if bpy.data.objects.get("Passage::records-open") is None:
    raise AssertionError("spatial map did not create the records-open passage")

print("ND_BLEND_VERIFY_OK")
