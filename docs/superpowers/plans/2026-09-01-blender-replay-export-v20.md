# Blender Replay Export V20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a situated physical map and exact V19 interaction trajectory from Python into one provenance-bearing, animated Blender `.blend` scene.

**Architecture:** Add dependency-free, content-addressed spatial-map contracts to the ABM core, compile an entitlement-safe replay packet from exact V19 states, and keep `bpy` plus subprocess execution in an optional integration. Blender runs headlessly against a temporary packet and staged output; Python atomically publishes only a valid `.blend` result.

**Tech Stack:** Python standard library, frozen dataclasses, Tiled JSON object layers, subprocess, Blender 5.1 Python `bpy`, `unittest`/pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-blender-graybox-replay-v20-design.md`

## Global Constraints

- The only required user-facing artifact is `.blend`; Blender owns later editing and rendering.
- Do not modify V10-V19 contracts, transition semantics, or hashes.
- Never export TELL message text, event details, private memories/evidence, prompts, provider values, environment values, or SQLite paths.
- `narrative_dynamics.abm` remains free of Blender and subprocess dependencies.
- Every map and replay packet binds exact source hashes and is deterministic across paths and JSON ordering.
- Existing output is published only after successful Blender exit and a valid `BLENDER` file header.

---

### Task 1: Spatial-map contracts, Tiled import, and auto-layout

**Files:**
- Create: `narrative_dynamics/abm/situated_spatial_map_contracts.py`
- Create: `narrative_dynamics/abm/situated_spatial_map.py`
- Create: `tests/test_network_abm_situated_spatial_map.py`

**Interfaces:**
- Consumes: `SituatedWorldModel`, `stable_content_hash`, standard-library JSON/path/math.
- Produces: `SpatialPlace`, `SpatialPassage`, `SituatedSpatialMap`, `load_tiled_situated_spatial_map(path, world_model, *, meters_per_pixel=0.05)`, and `auto_layout_situated_spatial_map(world_model, *, room_size=6.0, gap=3.0)`.

- [ ] **Step 1: Write failing contract and Tiled behavior tests**

```python
def test_tiled_rectangles_compile_to_exact_z_up_world_geometry(self):
    spatial = load_tiled_situated_spatial_map(self.map_path, self.world, meters_per_pixel=0.1)
    meeting = next(item for item in spatial.places if item.place_id == "meeting")
    self.assertEqual((meeting.center_x, meeting.center_y, meeting.width, meeting.depth), (14.0, -8.0, 8.0, 6.0))
    self.assertEqual({item.place_id for item in spatial.places}, {"corridor", "meeting"})
    self.assertEqual({item.passage_id for item in spatial.passages}, {"door"})

def test_spatial_map_rejects_missing_or_extra_world_ids(self):
    with self.assertRaisesRegex(ValueError, "cover every world place"):
        SituatedSpatialMap("office-map", "1", self.world, (), valid_passages)
```

The first catches wrong Tiled Y inversion, scale, rectangle centre, or class mapping. The second catches geometry silently diverging from the simulation graph.

- [ ] **Step 2: Run Task 1 and verify RED**

Run: `python -m pytest tests/test_network_abm_situated_spatial_map.py -q`

Expected: collection fails because the spatial-map modules do not exist.

- [ ] **Step 3: Implement strict immutable contracts**

Implement finite-number and positive-dimension validation, canonical sorting, unique IDs, exact `SituatedWorldModel` binding, complete place/passage coverage, `to_dict()`, and `content_hash`. Geometry values use metres and Z-up coordinates only.

- [ ] **Step 4: Implement Tiled import and deterministic fallback**

Parse UTF-8 JSON object layers recursively. Accept only non-rotated rectangle objects classified by `class` or `type` as `place`/`passage`, use `name` as the simulation ID, scale pixels to metres, and invert Y. Auto-layout sorted places on a square grid and passages at endpoint midpoints.

- [ ] **Step 5: Run Task 1 GREEN and commit**

Run: `python -m pytest tests/test_network_abm_situated_spatial_map.py -q`

Expected: all spatial-map tests pass.

```text
git add narrative_dynamics/abm/situated_spatial_map_contracts.py narrative_dynamics/abm/situated_spatial_map.py tests/test_network_abm_situated_spatial_map.py
git commit -m "feat(abm): add situated spatial maps"
```

### Task 2: Privacy-safe V19-to-Blender replay compilation

**Files:**
- Create: `narrative_dynamics/integrations/blender_replay.py`
- Create: `tests/test_blender_replay.py`

**Interfaces:**
- Consumes: Task 1 `SituatedSpatialMap`; `SituatedNetworkRuntimeModel`; `SituatedNetworkTrajectory`; `stable_content_hash`.
- Produces: `compile_situated_blend_replay(model, trajectory, spatial_map, *, frames_per_round=24) -> dict[str, object]`.

- [ ] **Step 1: Write failing exact-timeline and privacy tests**

```python
def test_replay_compiles_map_movement_door_state_and_events(self):
    packet = compile_situated_blend_replay(model, trajectory, spatial, frames_per_round=12)
    self.assertEqual((packet["first_frame"], packet["last_frame"]), (1, 13))
    alice = next(item for item in packet["actors"] if item["agent_id"] == "alice")
    self.assertEqual([state["place_id"] for state in alice["states"]], ["corridor", "meeting"])
    self.assertEqual(packet["events"][0]["frame"], 13)

def test_replay_packet_never_contains_tell_message_or_private_details(self):
    serialized = json.dumps(compile_situated_blend_replay(model, trajectory, spatial), sort_keys=True)
    self.assertNotIn(SECRET, serialized)
    self.assertNotIn('"details"', serialized)
    self.assertNotIn('"message"', serialized)
```

These tests catch off-by-one frames, movement from the wrong state, and objective/private payload leakage.

- [ ] **Step 2: Run Task 2 and verify RED**

Run: `python -m pytest tests/test_blender_replay.py -q`

Expected: import failure because `blender_replay` does not exist.

- [ ] **Step 3: Implement deterministic replay compilation**

Validate exact model, trajectory, world, and spatial-map hashes. Compile initial plus successor states, stable actor colors and co-location offsets, passage states, payload-free objective event markers, V19 sanitized transmissions, belief scalars, and claim counts. Hash the packet before adding its `content_hash` field.

- [ ] **Step 4: Run Task 1-2 GREEN and commit**

Run: `python -m pytest tests/test_network_abm_situated_spatial_map.py tests/test_blender_replay.py -q`

Expected: all map and replay tests pass.

```text
git add narrative_dynamics/integrations/blender_replay.py tests/test_blender_replay.py
git commit -m "feat(integrations): compile Blender replay packets"
```

### Task 3: Headless Blender scene builder and atomic `.blend` export

**Files:**
- Create: `narrative_dynamics/integrations/blender_scene_builder.py`
- Modify: `narrative_dynamics/integrations/blender_replay.py`
- Modify: `narrative_dynamics/integrations/__init__.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Create: `tests/fake_blender.py`
- Modify: `tests/test_blender_replay.py`
- Modify: `tests/test_network_abm_public_api.py`

**Interfaces:**
- Consumes: Task 2 packet compiler, one explicit Blender executable, output `.blend` path.
- Produces: `BlenderExportError`, `BlenderExportReport`, and `export_situated_network_blend(blender_executable, output_path, model, trajectory, spatial_map=None, *, frames_per_round=24, timeout_seconds=120.0)`.

- [ ] **Step 1: Write failing subprocess and atomic-publication tests**

```python
def test_export_stages_packet_and_atomically_publishes_blend(self):
    report = export_situated_network_blend(fake_blender, output, model, trajectory, spatial)
    self.assertEqual(output.read_bytes()[:7], b"BLENDER")
    self.assertEqual(report.output_path, str(output.resolve()))
    self.assertEqual(report.trajectory_hash, trajectory.content_hash)

def test_failed_blender_process_preserves_existing_output(self):
    output.write_bytes(b"BLENDER-PRIOR")
    with self.assertRaises(BlenderExportError):
        export_situated_network_blend(failing_blender, output, model, trajectory, spatial)
    self.assertEqual(output.read_bytes(), b"BLENDER-PRIOR")
```

These tests exercise a real subprocess boundary and catch publishing invalid partial output or overwriting a previous scene on failure.

- [ ] **Step 2: Run Task 3 and verify RED**

Run: `python -m pytest tests/test_blender_replay.py tests/test_network_abm_public_api.py -q`

Expected: import failure for the export report/function.

- [ ] **Step 3: Implement the standalone `bpy` scene builder**

Parse `--input` and `--output` after Blender's `--`. Reset scene data; create map, agent, label, and event collections; generate gray place/door primitives and procedural stick figures; add actor-location and door-state keyframes; add event timeline markers, source custom properties, overview camera/light, and linear movement interpolation; save with `bpy.ops.wm.save_as_mainfile`.

- [ ] **Step 4: Implement guarded subprocess execution and atomic publish**

Validate executable/output/timeout, compile before launch, write a canonical temporary packet, run Blender with `--background --factory-startup --python-exit-code 21 --python ... -- --input ... --output ...`, bound captured output, save uncompressed, require zero exit and `BLENDER` header, then `os.replace` the staged file. Return stable report metadata and redact subprocess details from public errors.

- [ ] **Step 5: Export APIs and document the office command**

Export spatial values from `narrative_dynamics.abm`; export Blender values from `narrative_dynamics.integrations`. README shows Tiled map or auto-layout input, the explicit Blender 5.1 command path, the resulting `.blend`, and privacy/authority boundaries.

- [ ] **Step 6: Run real Blender smoke verification**

Run the focused Python exporter test with `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`, then reopen the result headlessly with a verification script that asserts `ND_Map`, `ND_Agents`, agent keyframes, timeline markers, and source custom properties.

Expected: both Blender processes exit zero and the real output begins with `BLENDER`.

- [ ] **Step 7: Run authoritative regression gates**

Run: `python -m pytest tests/test_network_abm_situated_spatial_map.py tests/test_blender_replay.py tests/test_network_abm_public_api.py -q`

Run: `python -m unittest discover -s tests -p "test_network_abm*.py"`

Run: `python -m compileall -q narrative_dynamics`

Run: `git diff --check`

Expected: all focused and network ABM tests pass; compilation and diff checks exit zero. Do not run the previously stopped repository-wide suite.

- [ ] **Step 8: Commit Task 3 and documentation**

```text
git add docs/superpowers/specs/2026-09-01-blender-graybox-replay-v20-design.md docs/superpowers/plans/2026-09-01-blender-replay-export-v20.md narrative_dynamics/abm/__init__.py narrative_dynamics/integrations/__init__.py narrative_dynamics/integrations/blender_replay.py narrative_dynamics/integrations/blender_scene_builder.py tests/fake_blender.py tests/test_blender_replay.py tests/test_network_abm_public_api.py README.md
git commit -m "feat(integrations): export situated Blender replays"
```

## Self-review

- Spec coverage: exact map alignment, Tiled import, auto-layout, replay timing/privacy, `.blend` collections/keyframes/provenance, subprocess isolation, atomic publication, public APIs, real Blender verification, and documentation map to Tasks 1-3.
- Placeholder scan: every production/test/document action is explicit; no unresolved implementation marker remains.
- Type consistency: Task 1's `SituatedSpatialMap` feeds Task 2's compiler; Task 2's packet feeds Task 3's builder and exporter; public names match the spec.
