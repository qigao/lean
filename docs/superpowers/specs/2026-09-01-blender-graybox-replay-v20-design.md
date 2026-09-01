# Blender Graybox Replay V20 Design

## Goal

Turn an audited simulation trajectory into a driveable Blender graybox replay: simple map geometry, stick-figure agents, event and relationship overlays, editable narrative captions, timeline scrubbing, cameras, and deterministic still/video/scene export.

## Architectural boundary

The simulator remains authoritative. It exports a versioned, content-addressed `SceneReplayBundle` JSON; an optional Blender adapter consumes that bundle and creates `.blend` scene data. Blender never decides actions, perception, beliefs, memory, causality, or canonical event order, and generated keyframes never feed evidence back into the simulation.

The pipeline is:

```text
World/map + agents + V19 trajectory + optional V18 narrative
                         |
                         v
              SceneReplayBundle JSON
                 /               \
        text/editor tools      Blender importer
                                  |
                       graybox viewport / .blend
                         /        |        \
                      MP4       PNG       glTF
```

## Replay bundle

The bundle contains only portable, deterministic data:

- `manifest`: schema version, source model/state/trajectory hashes, coordinate system, units, frames per round, and bundle hash;
- `world`: places as polygons or boxes, passages as portals, optional terrain/props, and stable material tags;
- `actors`: agent ID, role, color tag, initial pose, and optional display label;
- `timeline`: round/frame mapping plus typed movement, action, interaction, perception, claim, belief, and relationship-overlay cues;
- `narrative_tracks`: scene/beat IDs and editable presentation text bound to source event IDs;
- `camera_cues`: optional authored shots; absent cues use deterministic overview and follow cameras;
- `assets`: optional relative references with hashes, never arbitrary executable scripts.

Canonical simulation facts and editable presentation are separate layers. Editing a caption, shot, pacing multiplier, color, or actor display name creates a presentation override document and a new preview hash; it does not change the source trajectory hash or claim that the edited prose is new evidence.

## Map and coordinates

V19.1 map import compiles IndoorGML/IMDF, Tiled JSON, GeoJSON, or selected IFC geometry into the internal world graph and optional spatial layout. V20 consumes that canonical result.

Coordinates are optional for an initial preview. If a world has no geometry, the exporter applies one deterministic graph layout and emits simple room boxes and portal locations. Imported or authored coordinates always take precedence. Blender uses metres, Z-up, and an explicit source-to-Blender transform in the manifest.

## Graybox representation

- Places: low-poly floor slabs and translucent wall volumes.
- Passages: colored door/portal objects whose open/closed state is keyframed.
- Agents: procedural stick figures built from primitives, one collection per agent, with stable colors and nameplates.
- Movement: linear or eased path keyframes between portal waypoints; simultaneous actions share a frame interval.
- Actions: small icons/markers and short pose clips for tell, inspect, take, drop, open, close, and wait.
- Perception: optional transient cones/arcs/lines derived only from sanitized access/transmission projections.
- Cognition/social overlays: optional HUD panels and edges showing declared scalar belief/trust metrics, never hidden private payloads.

## Viewer and driving controls

The generated `.blend` scene provides timeline scrubbing and playback without an add-on. An optional Blender add-on supplies:

- load/reload bundle;
- play, pause, next/previous round and next event;
- select/follow an agent and switch overview/first-person/follow cameras;
- toggle map, labels, perception, relationships, beliefs, and captions;
- change pacing and presentation overrides;
- jump from an object or caption to its source event ID/hash;
- export `.blend`, PNG frame sequences, MP4 through Blender/FFmpeg, and glTF for browser viewing.

“Driveable” in V20 means navigating cameras and replay time. Interactive counterfactual control—choosing an agent action and resuming simulation—is a later bridge that must submit a typed intervention to the simulator and receive a new branch/trajectory rather than mutate Blender keyframes.

## Determinism and provenance

- Identical source bundle plus presentation overrides produces stable object names, collections, frame allocation, transforms, colors, and camera defaults.
- Every generated object stores source IDs/hashes as custom properties.
- Unknown cue kinds fail validation; missing optional geometry falls back deterministically.
- Export supports headless operation: `blender -b -P import_replay.py -- bundle.json --output preview.blend`.
- `bpy`, Blender, FFmpeg, and rendering dependencies stay outside `narrative_dynamics.abm`.

## Delivery slices

1. V19.1: canonical world-map JSON and import adapters, including deterministic no-coordinate layout.
2. V20.0: `SceneReplayBundle` contracts/exporter and JSON validation with no Blender dependency.
3. V20.1: Blender Python importer producing rooms, portals, stick figures, labels, keyframes, and `.blend` output.
4. V20.2: playback controls, overlays, cameras, presentation overrides, and MP4/PNG/glTF export.
5. Later: typed counterfactual intervention bridge that forks a new simulation trajectory.

## Acceptance scenario

An office case imports or auto-lays out a corridor and meeting room separated by a door. Alice tells a fact, Bob receives either an anonymous sound through a closed door or an exact transmission through an open door, both agents continue acting across rounds, and trust/belief overlays evolve. A user can scrub rounds, follow Alice or Bob, edit the caption without altering the canonical event, inspect every preview element's source hash, save `.blend`, and render a short MP4.

## Non-goals

- No photorealism, rigid-body gameplay, motion capture, facial animation, or procedural storytelling inside Blender.
- No direct reading of private messages or evidence beyond the selected projection entitlement.
- No Blender dependency in the deterministic simulation core.
- No promise that edited presentation text is canonical simulation truth.
