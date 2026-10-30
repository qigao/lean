# R2.M0 — Read-only, source-owned network change observations

**Tracker:** [Issue #110](https://github.com/qigao/lean/issues/110).
**Parent stack:** [Draft PR #109](https://github.com/qigao/lean/pull/109)
(authoritative V19 social-memory checkpoint transitions).

## Correctness boundary

Network evolution is **not** only edge insertions/removals. Three distinct
inputs are recorded:

1. **Boundary**: real V19 `PassageState.open` and V24 Society membership
   or legally declared directed Bridges. Physical boundaries are owned by
   world state; Society scope is a derived/configuration view, never a
   synthetic social-memory event.
2. **Environment**: real `SituatedWorldState` Agent locations and object
   revisions; visibility/audio/interaction opportunities are computed by
   the existing perception model. Model/config hashes remain explicit.
3. **Multi-Agent interaction content**: real V19 `SituatedStory` event
   IDs, observation **channel counts** (not private observation payload),
   belief contents, claims and source-owned social trust. An actual TELL
   can change observed information without changing trust; an explicit
   administrative trust checkpoint is not mislabelled as a TELL event.

The three V24 selected relations remain separate: source-backed relationship
trust, physical direct-interaction opportunity and actually recorded
transmission. For each, the report lists source-edge and projected-edge
additions/removals, reachable ordered-pair counts, directed diameter when
defined and strongly connected component counts. All graph calculations
reuse existing `source_relation_edges`, `mesh_edges` and `observe_topology`;
there is no second graph algorithm or canonical state owner.

## API / provenance

`observe_network_layers(model, story, cognition, social, societies, bridges=())`
constructs a read-only `NetworkLayerView`, reprojecting the **same**
authoritative V19 source to verify exact model/story/cognitive/social hashes
and V24 legitimacy. `compare_network_layers(before, after)` revalidates both
inputs, bounds Agent/Society/event counts, requires the same runtime model
and Agent roster, and returns an immutable `NetworkChangeReport` with
`to_dict()` and a stable content hash.

Its `story_lineage` states `SAME_STORY`, `STORY_EXTENSION`,
`DIVERGENT_STORY_BRANCH` or `DIFFERENT_INITIAL_CONDITIONS`. Only a true
story-prefix extension exposes new **actual** event IDs and observation
channel counts. It never accepts unrelated counterfactual events as an
ordered causal history. `causal_attribution` is always `UNATTRIBUTED`;
different source owners may change together and the report does not
infer which one caused another to change.

**Privacy:** no message text, private observer payload, memory body,
per-agent observation list or private claim content is serialized by the
report. It reports safe event metadata, counts, identity hashes, and
selected graph facts. This is source-derived consistency evidence, **not**
external process authentication, statistical causal identification or a
universal six-hop property.

## Real fixtures and CI

`tests/test_network_evolution_observation.py` and the same-case
`tools/render_network_evolution_vectors.py` cover five contrasts:

- physical passage open/closed: **different initial conditions**, not an
  invented OPEN/CLOSE story event;
- Agent location change: a **counterfactual environment comparison**, not
  a claimed successful MOVE without an actual event;
- Society membership scope change while V19 source edges remain identical;
- a real successful Agent `TELL` story extension, with message redacted and
  no fabricated social-trust change;
- a real V19 source-authoritative trust-threshold checkpoint, with no
  fabricated TELL event or physical boundary change.

The reviewed `conformance/network_evolution_observation_v1.json` must
match the exact generated UTF-8 bytes in focused Python, full Python
and the independent Lean acceptance lane. The first missing-pin run is
intentionally RED and emits the actual reviewed candidate, then that
one-time handoff is removed. All earlier V19/V24, R1/R1.1, 35+11 theorem
trust, BB/PathN and World Studio/Playwright checks remain mandatory.

## Not yet implemented

R2.M0 *observes* immutable states. It does **not** introduce an admitted
physical passage or institutional-boundary action into the current story
runtime, rewrite model geometry in-place, manage shared-document revisions,
execute concurrent content merges, or attribute graph-edge changes to a
unique cause. Those are separately scoped later phases under Issue #110.
