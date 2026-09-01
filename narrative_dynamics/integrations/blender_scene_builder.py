"""Standalone Blender 5.x scene builder for situated replay packets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _arguments() -> argparse.Namespace:
    if "--" not in sys.argv:
        raise ValueError("Blender replay builder requires arguments after --")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _move_to_collection(obj, collection) -> None:
    for owner in tuple(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def _material(bpy, name: str, color: list[float]):
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    return material


def _cube(bpy, collection, name, location, dimensions, material=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    if material is not None:
        obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    return obj


def _cylinder(bpy, collection, name, location, radius, depth, rotation=(0.0, 0.0, 0.0), material=None):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    return obj


def _text(bpy, collection, name, body, location, size=0.35):
    data = bpy.data.curves.new(name=f"{name}:Data", type="FONT")
    data.body = body
    data.align_x = "CENTER"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    collection.objects.link(obj)
    return obj


def _animation_curves(obj):
    animation = obj.animation_data
    action = None if animation is None else animation.action
    if action is None:
        return ()
    legacy_curves = getattr(action, "fcurves", None)
    if legacy_curves is not None:
        return tuple(legacy_curves)
    slot = animation.action_slot
    if slot is None:
        return ()
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


def _stick_figure(bpy, collection, actor, material):
    root = bpy.data.objects.new(f"Agent::{actor['agent_id']}", None)
    root.empty_display_type = "PLAIN_AXES"
    collection.objects.link(root)
    parts = [
        _cylinder(bpy, collection, f"{root.name}:Torso", (0.0, 0.0, 1.15), 0.13, 1.0, material=material),
        _cylinder(bpy, collection, f"{root.name}:Arms", (0.0, 0.0, 1.35), 0.055, 1.15, rotation=(0.0, math.pi / 2.0, 0.0), material=material),
        _cylinder(bpy, collection, f"{root.name}:LegL", (-0.15, 0.0, 0.42), 0.06, 0.8, material=material),
        _cylinder(bpy, collection, f"{root.name}:LegR", (0.15, 0.0, 0.42), 0.06, 0.8, material=material),
    ]
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.24, location=(0.0, 0.0, 1.92))
    head = bpy.context.object
    head.name = f"{root.name}:Head"
    head.data.materials.append(material)
    _move_to_collection(head, collection)
    parts.append(head)
    for part in parts:
        part.parent = root
        part["nd_agent_id"] = actor["agent_id"]
    root["nd_agent_id"] = actor["agent_id"]
    root["nd_role"] = actor["role"]
    return root


def build_scene(packet: dict[str, object], output_path: Path) -> None:
    import bpy

    if packet.get("schema") != "narrative-dynamics.situated-blend-replay/v1":
        raise ValueError("unsupported Blender replay schema")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    map_collection = bpy.data.collections.new("ND_Map")
    agent_collection = bpy.data.collections.new("ND_Agents")
    label_collection = bpy.data.collections.new("ND_Labels")
    event_collection = bpy.data.collections.new("ND_Events")
    for collection in (map_collection, agent_collection, label_collection, event_collection):
        scene.collection.children.link(collection)

    floor_material = _material(bpy, "ND_Floor", [0.32, 0.36, 0.42, 1.0])
    door_material = _material(bpy, "ND_Door", [0.72, 0.42, 0.16, 1.0])
    for place in packet["places"]:
        floor = _cube(
            bpy,
            map_collection,
            f"Place::{place['place_id']}",
            (place["center_x"], place["center_y"], -0.06),
            (place["width"], place["depth"], 0.12),
            floor_material,
        )
        floor["nd_place_id"] = place["place_id"]
        floor["nd_spatial_map_hash"] = packet["spatial_map_hash"]
        label = _text(
            bpy,
            label_collection,
            f"PlaceLabel::{place['place_id']}",
            place["label"],
            (place["center_x"], place["center_y"], 0.08),
        )
        label["nd_place_id"] = place["place_id"]

    for passage in packet["passages"]:
        door = _cube(
            bpy,
            map_collection,
            f"Passage::{passage['passage_id']}",
            (passage["center_x"], passage["center_y"], passage["height"] / 2.0),
            (passage["width"], passage["depth"], passage["height"]),
            door_material,
        )
        door["nd_passage_id"] = passage["passage_id"]
        door["nd_source_place_id"] = passage["source_place_id"]
        door["nd_target_place_id"] = passage["target_place_id"]
        for state in passage["states"]:
            door.rotation_euler[2] = math.pi / 2.0 if state["open"] else 0.0
            door.keyframe_insert(data_path="rotation_euler", index=2, frame=state["frame"])

    for actor in packet["actors"]:
        material = _material(bpy, f"AgentMaterial::{actor['agent_id']}", actor["color"])
        root = _stick_figure(bpy, agent_collection, actor, material)
        label = _text(
            bpy,
            label_collection,
            f"AgentLabel::{actor['agent_id']}",
            actor["agent_id"],
            (0.0, 0.0, 2.35),
            size=0.28,
        )
        label.parent = root
        label["nd_agent_id"] = actor["agent_id"]
        for state in actor["states"]:
            root.location = tuple(state["position"])
            root["nd_place_id"] = state["place_id"]
            root["nd_belief_probability"] = state["tracked_belief_probability"]
            root["nd_active_claim_count"] = state["active_claim_count"]
            root.keyframe_insert(data_path="location", frame=state["frame"])
            root.keyframe_insert(data_path='["nd_belief_probability"]', frame=state["frame"])
            root.keyframe_insert(data_path='["nd_active_claim_count"]', frame=state["frame"])

    for event in packet["events"]:
        scene.timeline_markers.new(
            f"{event['kind']}::{event['actor_agent_id']}::{event['event_id']}",
            frame=event["frame"],
        )

    for obj in bpy.data.objects:
        for curve in _animation_curves(obj):
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    scene.frame_start = packet["first_frame"]
    scene.frame_end = packet["last_frame"]
    scene.render.fps = 24
    scene["nd_schema"] = packet["schema"]
    scene["nd_replay_hash"] = packet["content_hash"]
    scene["nd_runtime_model_hash"] = packet["runtime_model_hash"]
    scene["nd_trajectory_hash"] = packet["trajectory_hash"]
    scene["nd_world_model_hash"] = packet["world_model_hash"]
    scene["nd_spatial_map_hash"] = packet["spatial_map_hash"]

    xs = [item["center_x"] for item in packet["places"]]
    ys = [item["center_y"] for item in packet["places"]]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), 8.0)
    camera_data = bpy.data.cameras.new("ND_OverviewCamera")
    camera = bpy.data.objects.new("ND_OverviewCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (center_x, center_y, span * 1.7)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 1.6
    scene.camera = camera
    light_data = bpy.data.lights.new("ND_Sun", type="SUN")
    light = bpy.data.objects.new("ND_Sun", light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (0.4, -0.5, -0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output_path),
        check_existing=False,
        compress=False,
    )


def main() -> int:
    options = _arguments()
    packet = json.loads(Path(options.input).read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("Blender replay packet must be an object")
    build_scene(packet, Path(options.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
