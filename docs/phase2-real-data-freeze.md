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

The extractor is frozen to Ultralytics `yolo26n-pose.pt`, with its 17 COCO body keypoints. The exact installed Ultralytics package version and the model-weight SHA-256 must be recorded from the execution environment before extraction.

The classifier observation contains only:

- 17 keypoints × `(x, y, confidence)`;
- person confidence;
- timestamps / frame order needed to preserve sequence timing.

No ByteTrack feature, depth estimate, optical-flow network output, RGB embedding, or NTU-provided skeleton coordinates are classifier inputs. NTU skeleton annotations may be used only as an extraction sanity check and must never enter training or evaluation features.

The normalized observation schema and encoder implementation must be hashed before the final-test run.

## FlyWire source and frozen topology rule

The FlyWire source is frozen to the static visual-system snapshot associated with **FAFB v783** and the Nature visual-system parts-list work. The immutable repository snapshot is:

- repository: `murthylab/visual-system-parts-list`
- commit: `0d8574d46627ce7fadd968a3c5d602e837325373`
- connectivity path: `data/type_to_type_connection_and_synapse_counts.csv`
- Git blob SHA-1: `5183755ecbb41d5c8cee1a4a2d99b8eecba75c52`

The file SHA-256 must still be computed from the exact downloaded bytes and entered into the protocol. The Git blob identifier is an additional immutable provenance identifier, not a substitute for the requested SHA-256.

The selected type-level subgraph rule is frozen before final-test inspection:

1. Seed cell types are `T4a`, `T4b`, `T4c`, `T4d`, `T5a`, `T5b`, `T5c`, `T5d`.
2. Add every visual type receiving at least **5 aggregate incoming synapses** from any seed type in the frozen type-to-type table.
3. Form the induced directed graph over the resulting selected type set.
4. Edge weights are aggregate synapse counts from the same frozen table.
5. Node ordering is lexicographic by exact type name before serialization / fingerprinting.
6. The resulting graph fingerprint must be frozen before confirmatory training.

This intentionally targets the canonical motion-detecting T4/T5 families and their directly supported downstream visual partners instead of choosing a subgraph after inspecting behavior-recognition results.

## Matched controls

The confirmatory arms are:

- GRU baseline
- random sparse graph baseline
- FlyWire graph recurrent model
- degree-/weight-matched rewired FlyWire graph recurrent model

The rewired control uses `directed-double-edge-swap-v1`. It must preserve node count, edge count, directed in/out degree sequence, and the edge-weight multiset. FlyWire and rewired arms must share the same observation schema, split, parameter ceiling, optimizer/training budget, and seed list.

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

## Remaining byte-level freeze

Before confirmatory final-test execution, all of the following must be non-null and validated:

- NTU dataset content/archive hash;
- exact Ultralytics package version;
- `yolo26n-pose.pt` SHA-256;
- frozen split hash;
- observation-schema / encoder hash;
- FlyWire connectivity CSV SHA-256;
- selected graph fingerprint;
- matched rewired graph fingerprint.

Only after these values are frozen may the real final-test run be considered confirmatory evidence.
