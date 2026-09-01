"""Headless assertions for a generated situated replay `.blend`."""

from __future__ import annotations

import bpy


def animation_curves(obj):
    animation = obj.animation_data
    if animation is None or animation.action is None:
        return ()
    action = animation.action
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return tuple(legacy)
    slot = animation.action_slot
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            channelbag_method = getattr(strip, "channelbag", None)
            if channelbag_method is None:
                continue
            channelbag = channelbag_method(slot)
            if channelbag is not None:
                curves.extend(channelbag.fcurves)
    return tuple(curves)


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
agent_curves = animation_curves(agent)
location_curves = [item for item in agent_curves if item.data_path == "location"]
if len(location_curves) != 3:
    raise AssertionError("Alice root must keyframe three location axes")
for curve in location_curves:
    if {int(item.co[0]) for item in curve.keyframe_points} != {1, 25, 49}:
        raise AssertionError("Alice location keyframes do not cover every state")
    if any(item.interpolation != "LINEAR" for item in curve.keyframe_points):
        raise AssertionError("Alice movement must interpolate linearly")
for data_path in ('["nd_belief_probability"]', '["nd_active_claim_count"]'):
    curve = next((item for item in agent_curves if item.data_path == data_path), None)
    if curve is None:
        raise AssertionError(f"Alice missing discrete state curve: {data_path}")
    if any(item.interpolation != "CONSTANT" for item in curve.keyframe_points):
        raise AssertionError(f"Alice discrete state curve must step: {data_path}")
if not scene.timeline_markers:
    raise AssertionError("replay has no objective event markers")
if bpy.data.objects.get("Place::records") is None:
    raise AssertionError("spatial map did not create the records place")
if bpy.data.objects.get("Passage::records-open") is None:
    raise AssertionError("spatial map did not create the records-open passage")
passage = bpy.data.objects["Passage::records-open"]
passage_curves = animation_curves(passage)
if not passage_curves:
    raise AssertionError("records-open passage has no state animation")
if any(
    keyframe.interpolation != "CONSTANT"
    for curve in passage_curves
    for keyframe in curve.keyframe_points
):
    raise AssertionError("passage state must use constant interpolation")

print("ND_BLEND_VERIFY_OK")
