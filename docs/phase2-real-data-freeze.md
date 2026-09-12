# Phase 2 real-data freeze

## Scope

Phase 2 moves the V0 harness from synthetic plumbing evidence to a real-data confirmatory comparison of temporal behavior recognition. The confirmatory claim remains narrowly topology-specific: under one frozen YOLO/Pose observation stream, split, feature encoder, training budget, seeds, and evaluation protocol, does the selected FlyWire topology outperform a matched rewired FlyWire control?

Synthetic results, `FlyWire > YOLO`, or `FlyWire > GRU` alone are not sufficient evidence for a topology-specific advantage.

## Dataset and task

The real-data family is **NTU RGB+D 120**, using **RGB video only** as classifier input. The V0 confirmatory subset is frozen to 10 single-person actions chosen to emphasize whole-body motion and coarse upper-body gesture while minimizing dependence on object appearance:

- A8 sitting down
- A9 standing up
- A22 cheer up
- A23 hand waving
- A26 hopping
- A27 jump up
- A31 pointing
- A34 rub two hands together
- A35 nod head / bow
- A36 shake head

The outer benchmark boundary is the official X-Sub120 split. Validation is derived only from the official training side by a deterministic subject-hash rule. Final-test subjects remain sealed from model, graph, threshold, and hyperparameter selection.

The repository must not redistribute NTU RGB+D 120 media. A legally acquired local copy is required for execution, and a content-derived dataset hash must be frozen before the final test.

## YOLO/Pose observation stream

The extractor family is frozen to Ultralytics `yolo26n-pose.pt`, with its 17 COCO body keypoints. The exact installed Ultralytics package version and the model-weight SHA-256 must be recorded from the execution environment before extraction.

The classifier observation contains only:

- 17 keypoints × `(x, y, confidence)`;
- person confidence;
- timestamps / frame order needed to preserve sequence timing.

No ByteTrack feature, depth estimate, optical-flow network output, RGB embedding, or NTU-provided skeleton coordinates are classifier inputs. NTU skeleton annotations may be used only as an extraction sanity check and must never enter training or evaluation features.

The normalized observation schema and encoder implementation must be hashed before the final-test run. The 17-point body schema supports coarse body/wrist gestures; it does not provide finger-joint observations for fine hand gestures.

## FlyWire source and frozen topology rule

The FlyWire source is frozen to the static visual-system snapshot associated with **FAFB v783** and the visual-system parts-list work. The immutable repository snapshot is:

- repository: `murthylab/visual-system-parts-list`
- commit: `0d8574d46627ce7fadd968a3c5d602e837325373`
- connectivity path: `data/type_to_type_connection_and_synapse_counts.csv`
- Git blob SHA-1: `5183755ecbb41d5c8cee1a4a2d99b8eecba75c52`
- byte-derived SHA-256: `215cf7a65895f9f84768db34052964542f7ebbe986ce9864b7e9ed2976c55e38`

Both the Git blob identifier and the file SHA-256 are checked against downloaded bytes by provenance CI; neither is merely a mutable branch label.

The selected type-level subgraph rule is frozen before final-test inspection:

1. Seed cell types are `T4a`, `T4b`, `T4c`, `T4d`, `T5a`, `T5b`, `T5c`, `T5d`.
2. Add every visual type receiving at least **5 aggregate incoming synapses** from any seed type in the frozen type-to-type table.
3. Form the induced directed graph over the resulting selected type set.
4. Edge weights are aggregate synapse counts from the same frozen table. Retain diagonal type edges as within-cell-type population connectivity; a type-level diagonal is not an assertion of neuron-level autapses.
5. Node ordering is lexicographic by exact type name before serialization / fingerprinting.
6. Require the selected graph fingerprint and statistics below before generation or verification of controls.

The selected graph has **187 nodes, 14,542 directed edges, and 126 diagonal edges**. Its fingerprint is `a7088c8590aa10d6b204c2b11ad51d5f7889dc10ea25b68ffcf24794b06d692d`.

This selection targets the T4/T5 motion families and their directly connected downstream visual partners rather than selecting a graph using behavior-recognition outcomes. The model is type/population-level and is not a full neuron-level physiological simulation.

## Matched controls

The confirmatory arms are:

- GRU baseline
- random sparse graph baseline
- FlyWire graph recurrent model
- matched rewired FlyWire graph recurrent model

The rewired control uses **`directed-double-edge-swap-v2-diagonal-fixed`**. It preserves node count, edge count, directed in/out degree sequence, the edge-weight multiset, and the exact diagonal edge+weight set. Only off-diagonal edges are eligible for randomization. Weight-matching here means the global edge-weight multiset, not equality of every node's incoming weighted strength.

The frozen budget is **10 successful swaps per off-diagonal edge**, or **144,160 successful swaps per seed**. Exhausting the attempt budget is an error, not permission to reduce the requested swaps. FlyWire and rewired arms must share the same observation schema, split, parameter ceiling, optimizer/training budget, and seed list.

The following full-budget fingerprints were independently identical in original CI #71 / run `34687807859` and optimized CI #73 / run `34689088871` artifacts. They are now stored in both real protocols:

| Seed | Rewired graph SHA-256 |
| --- | --- |
| 7 | `85a0a73e58a3ae9b1394cb59f7970538cd301fee03e04d22c2cf4af1dc38deda` |
| 11 | `e88cdf1652e39d6a2ef75dcbff3d2b79c7ed9b703b5beae3cee92e13e7c10387` |
| 19 | `aaae8ec540b26c1228916a3a24e08b8fe61a909c802f031b73ce659447e82231` |
| 23 | `c6aad8f35dcf875730a08f6607e2e3c67bb4943ad2710f8677d329b6b5daaed4` |
| 31 | `e80b997850d2ff16e024a4009cd7e66c0a86b2648caa81ce282ed59061e26469` |

The shared `validate_rewiring_protocol` gate rejects missing/extra seed entries, malformed hashes, a control identical to the original graph, stale algorithms, and missing/reduced/noninteger swap multipliers. It checks declarations only. Provenance generation and verification additionally compare actual graph fingerprints with the frozen map before publishing outputs. Cache hits still validate graph bytes and invariants; there is no partial-key or corrupt-cache fallback.

`v0-real-template.json` mirrors the preflight's entire experiment contract except its protocol identifier, preflight reference, and evidence-scope description. Tests enforce that agreement so the two copies cannot silently describe different control algorithms.

## Frozen seeds and budget

Seeds: `7, 11, 19, 23, 31`.

Training ceiling:

- 20 epochs
- 40 maximum updates
- 50,000 learnable parameters

Checkpoint selection uses validation only.

## Metrics and decision rule

Primary metric: **macro F1**.

Secondary metrics:

- balanced accuracy
- keypoint-noise macro F1
- keypoint-mask macro F1

The predeclared practical-effect rule for a topology-specific V0 success is:

- mean paired `(FlyWire - rewired)` macro F1 across the five seeds is at least **+0.02**; and
- the paired difference is positive in at least **4 of 5** seeds.

A negative or null result is a valid outcome and must be retained as such. No topology selection, region selection, threshold change, or retry-based cherry-picking is permitted after inspecting the sealed final test.

## Remaining byte-level freeze and execution boundary

Graph provenance is not behavior-recognition evidence. The unresolved real-data fields remain null in both protocols:

- NTU dataset content/archive hash;
- exact Ultralytics package version;
- `yolo26n-pose.pt` SHA-256;
- frozen split hash;
- observation-schema / encoder hash.

The real-data extraction and training/evaluation runner must also be implemented and verified. The current real `compare` branch is **protocol-validation-only**: even a structurally complete configuration does not cause it to train the four real-data arms. Populating hashes alone is therefore not a completed confirmatory experiment.

Final-test execution requires verified real input bytes, a completely frozen protocol, a verified runner that enforces those inputs and budgets, and validation-only model selection. Neither graph-only CI success nor a `protocol_validated_only` record establishes a FlyWire recognition advantage.
