# V0 readiness report

## Status

The V0 harness is ready for a real-data confirmatory phase, but it does **not** establish that FlyWire topology improves behavior recognition.

The validated harness includes:

- Lean protocol/claim contracts;
- immutable point-sequence schema;
- deterministic synthetic pose generation;
- frozen feature extraction and missing-keypoint handling;
- GRU and graph recurrent baselines;
- random sparse and degree-preserving rewired graph controls;
- FlyWire graph ingestion with provenance fields;
- deterministic training and held-out evaluation;
- canonical run manifests;
- robustness hooks for keypoint noise and masking;
- end-to-end four-arm comparison with paired FlyWire-vs-rewired evidence aggregation.

Synthetic results remain plumbing evidence only.

## Phase 2 frozen choices

The real-data preflight is now tracked in `protocols/v0-real-ntu120-preflight.json` and `docs/phase2-real-data-freeze.md`.

Frozen high-level choices are:

- NTU RGB+D 120, RGB modality only;
- 10-class motion / coarse-gesture subset;
- official X-Sub120 outer boundary;
- Ultralytics `yolo26n-pose.pt`, COCO-17 keypoints;
- no ByteTrack, no depth, no NTU skeleton features;
- FlyWire FAFB v783 visual-system snapshot;
- T4/T5-seeded type-level motion subgraph rule;
- matched degree-/weight-preserving rewiring;
- five fixed seeds and equalized budget;
- macro F1 primary metric;
- predeclared +0.02 mean paired practical-effect threshold with 4/5 positive seeds.

## Immutable FlyWire provenance now pinned

The static source named by the v783 visual-system work is frozen to:

- `murthylab/visual-system-parts-list`
- commit `0d8574d46627ce7fadd968a3c5d602e837325373`
- connectivity file `data/type_to_type_connection_and_synapse_counts.csv`
- Git blob SHA-1 `5183755ecbb41d5c8cee1a4a2d99b8eecba75c52`

The Git blob identifier is recorded as immutable source provenance. A SHA-256 over the exact downloaded CSV bytes is still required before confirmatory execution.

## Remaining confirmatory blockers

The following byte-derived values must be frozen before the sealed real final test:

- NTU dataset content/archive hash;
- exact Ultralytics package version;
- `yolo26n-pose.pt` SHA-256;
- split hash;
- observation-schema / encoder hash;
- FlyWire connectivity CSV SHA-256;
- selected FlyWire graph fingerprint;
- matched rewired graph fingerprint.

The comparison CLI now treats these real-data provenance fields as execution gates. Missing values must cause rejection rather than a partially specified topology claim.

## Evidence boundary

A real topology-specific claim requires FlyWire and matched rewired arms under the same observations, split, seeds, training budget, parameter ceiling, and evaluation protocol. GRU and random-graph baselines provide context but do not replace the decisive FlyWire-vs-rewired comparison.

Null and negative results are valid. Final-test outcomes must not be used to choose the FlyWire subgraph, graph threshold, success threshold, model hyperparameters, or retry policy.
