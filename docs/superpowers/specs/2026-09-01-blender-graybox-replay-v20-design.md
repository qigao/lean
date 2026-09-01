# Blender Graybox Replay V20 Design

## Goal

Export one V19 situated-network trajectory, its physical map, and its interaction timeline from Python into one editable `.blend` file. Narrative Dynamics owns deterministic scene compilation and provenance; Blender owns viewing, manual editing, materials, cameras, rendering, and every downstream media export.

## Product boundary

The only required user-facing artifact is `.blend`.

```text
Situated world + spatial map + V19 trajectory
                    |
                    v
       Python replay compiler (sanitized JSON in a temporary file)
                    |
                    v
          Blender headless Python (`bpy`)
                    |
                    v
                 scene.blend
```

The temporary JSON packet is an internal process boundary, not a second user workflow. It is deleted after export. The simulator never imports `bpy`; the optional adapter launches an explicit Blender executable in background/factory-startup mode.

## Spatial map

`SituatedSpatialMap` is a content-addressed geometry companion to one exact `SituatedWorldModel`. It does not replace the discrete world graph or change physical transition semantics.

- Every `SpatialPlace` binds one exact `place_id` to an axis-aligned floor rectangle in metres plus a display height.
- Every `SpatialPassage` binds one exact `passage_id` to an axis-aligned portal rectangle in metres plus a door height.
- A spatial map covers every world place and passage exactly once and rejects extras, omissions, duplicate IDs, non-finite coordinates, and non-positive dimensions.
- Paths, source JSON byte order, Tiled object IDs, layer ordering, and pixel units do not enter the canonical map hash.
- When no authored geometry is supplied, `auto_layout_situated_spatial_map` assigns lexicographically sorted places to a deterministic square grid and places each passage at the midpoint of its endpoint centres.

V20 initially imports orthogonal Tiled JSON object layers. Rectangle objects whose `class` or legacy `type` is `place` or `passage` use their non-empty `name` as the corresponding simulation ID. Tiled's pixel coordinates plus inherited group/object-layer pixel offsets are converted to metres and its downward Y axis is inverted for Blender's Z-up world. Nonzero tile-coordinate layer offsets, tile layers, image layers, rotated objects, polygons, ellipses, points, and arbitrary embedded scripts/assets are ignored or rejected rather than guessed.

## Replay packet

`compile_situated_blend_replay` validates an exact runtime model/trajectory/map chain and produces a canonical JSON object with:

- schema ID and packet hash;
- exact runtime model, trajectory, world model, and spatial-map hashes;
- frames per round and first/last frame;
- map rectangles and passage endpoint metadata;
- actors with stable IDs, roles, deterministic colors, and per-round positions;
- passage open/closed state per round;
- objective event markers with IDs, hashes, kind, actor, target, success, outcome, and frame;
- sanitized V19 transmissions with observer, source, fidelity, and channels;
- declared V19 node belief probability and active-claim count per actor state.

The packet never includes TELL message text, event details, memory rows, prompts, LLM output, private evidence, or database paths. Every runtime state's network snapshot and emergence metrics are recomputed and compared with the stored values. Movement is reconstructed directly from those validated story world states; the Blender adapter may interpolate between state positions but cannot invent a new destination or event.

## Generated `.blend`

The headless Blender script creates:

- `ND_Map`: gray floor boxes for places and door/portal boxes for passages;
- `ND_Agents`: one procedural stick figure per agent, parented to an animated root object;
- `ND_Labels`: place and agent text labels;
- timeline markers for every objective event;
- keyframes for actor root locations and passage open/closed rotation;
- stable scene frame range, linear actor location interpolation, and constant interpolation for passage and discrete cognitive/social state curves;
- custom properties on the scene and generated objects containing source IDs and hashes;
- one neutral world, sun light, and overview camera so the file opens in a useful state.

The script saves an uncompressed file with `bpy.ops.wm.save_as_mainfile`, preserving the stable `BLENDER` header used at the publication boundary. The output path must have a `.blend` suffix. Existing output is replaced only after Blender exits successfully and the staged file has that header; a failed process leaves the prior output untouched.

## Python API

Core ABM exports:

- `SpatialPlace`
- `SpatialPassage`
- `SituatedSpatialMap`
- `load_tiled_situated_spatial_map(path, world_model, *, meters_per_pixel)`
- `auto_layout_situated_spatial_map(world_model, *, room_size, gap)`

Optional integration exports:

- `compile_situated_blend_replay(model, trajectory, spatial_map, *, frames_per_round) -> dict[str, object]`
- `export_situated_network_blend(blender_executable, output_path, model, trajectory, spatial_map=None, *, frames_per_round=24) -> BlenderExportReport`
- `BlenderExportReport`

If `spatial_map` is omitted, the export uses deterministic auto-layout. The Blender executable is always explicit; the adapter never scans environment variables or silently downloads Blender.

## Atomicity and errors

- Validate all Python inputs and compile the complete packet before launching Blender.
- Write the packet and staged `.blend` under one temporary directory.
- Run with Blender's `--python-exit-code` so scene-script exceptions become nonzero process exits. Consume combined Blender stdout/stderr with a one-megabyte hard limit and raise one redacted `BlenderExportError` on missing executables, timeouts, excessive output, nonzero exit, missing staged output, or invalid output header.
- Publish with an atomic same-filesystem `os.replace` only after successful validation.
- Never include TELL messages, API keys, temporary packet contents, or environment values in representations or error messages.

## Verification

Pure Python tests cover map validation, inherited Tiled offsets, path-independent import, deterministic auto-layout, exact replay frames, authority rejection, movement, door states, sanitized transmission export, and secret absence. Subprocess protocol tests verify staging, bounded output, publication, and failure behavior. The explicit Blender 5.1 path supplied for this workspace runs a real smoke test that opens the generated file through Blender and inspects expected collections, exact keyframe frames, interpolation modes, and source properties before completion is claimed.

## Non-goals

- No PNG, MP4, glTF, FBX, or render export.
- No custom Blender add-on, playback UI, game controls, or bidirectional live link.
- No counterfactual action selection inside Blender.
- No photorealism, collision simulation, motion capture, facial animation, or asset marketplace integration.
- No change to V10-V19 simulation, perception, cognition, memory, social, or narrative authority.
