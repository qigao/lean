# Phase 2 real-data freeze: NTU RGB+D 120 × YOLO26 Pose × FlyWire v783

## Status

This document freezes the **design choices** for the first real-data confirmatory experiment. It does not authorize a sealed final-test run until the byte-level dataset/extraction hashes in `protocols/v0-real-ntu120-preflight.json` have been filled and validated.

## Confirmatory question

Under one frozen RGB observation stream, split, point encoder, training budget, seed set, and evaluation protocol, does a FlyWire motion-subsystem topology improve temporal action recognition relative to a degree-/weight-matched rewired control?

The decisive topology comparison is:

`FlyWire motion graph` vs `matched rewired FlyWire motion graph`.

GRU and random sparse graph are contextual baselines only.

## Real dataset

Dataset: **NTU RGB+D 120**, official ROSE Lab release.

V0 uses the RGB modality only for model input. Official Kinect skeleton data may be used only as an extraction sanity-check and must never be supplied to the classifier.

The dataset is research-only / non-commercial and may not be redistributed or used to derive a redistributed dataset without permission. Repository code therefore consumes a locally supplied dataset and stores only identifiers, hashes, extraction metadata, and aggregate metrics.

### Frozen action subset

The first confirmatory task deliberately focuses on actions that should be explainable mainly from body/wrist motion and do not require fine finger topology or object appearance:

- A8 `sitting down`
- A9 `standing up (from sitting position)`
- A22 `cheer up`
- A23 `hand waving`
- A26 `hopping (one foot jumping)`
- A27 `jump up`
- A31 `pointing to something with finger`
- A34 `rub two hands together`
- A35 `nod head/bow`
- A36 `shake head`

This subset mixes pedestrian/body dynamics with coarse gestures while avoiding classes where RGB object identity is the principal discriminant.

## Split

Use the official NTU RGB+D 120 **cross-subject** benchmark boundary as the outer train/final-test split. The official final-test side is sealed.

Create the validation split only from the official training side using a deterministic subject-level hash partition. The validation assignment algorithm and salt must be frozen before pose extraction metrics are inspected.

No sample from a final-test subject may be used for checkpoint selection, feature-schema selection, graph selection, threshold selection, or failure-driven retries.

## YOLO observation extractor

Freeze the pose extractor to:

- family: Ultralytics YOLO26 Pose
- weights: `yolo26n-pose.pt`
- task: COCO human pose
- keypoints: 17 COCO body keypoints
- input: NTU RGB frames only
- output per frame: `(x, y, confidence)` for every keypoint plus person detection confidence
- no ByteTrack-derived identity feature
- no depth input
- no NTU skeleton input

The exact Ultralytics package version and downloaded weights SHA-256 must be written into the executable protocol after acquisition.

Missing/low-confidence points must remain explicit masks; they must not be silently interpolated before the frozen feature encoder unless an interpolation rule is separately frozen and hashed.

## Observation schema

Each single-person sequence is normalized by a body-centered transform derived only from detected 2-D keypoints. The feature encoder may use position, first temporal difference, second temporal difference, joint-relative distances/angles already defined by the V0 feature contract, and explicit confidence/missingness masks.

The encoder must not receive:

- RGB pixels after pose extraction;
- depth maps;
- Kinect 3-D skeleton coordinates;
- action labels as input features;
- tracker IDs;
- object-class features.

This keeps the claim narrow: temporal structure over YOLO/Pose point motion.

## FlyWire source

Freeze the connectome source to the published **FlyWire FAFB v783** visual-system snapshot associated with the visual-system parts-list publication.

Data identity must include:

- FlyWire release: `783`;
- static source repository/commit for `murthylab/visual-system-parts-list`;
- hash of the connectivity table actually ingested;
- edge weight definition: published synapse-count connectivity;
- type identity from the same static snapshot.

Do not silently switch to BANC, MAOL, a newer Codex materialization, or updated annotations during this experiment.

## Frozen motion-subsystem selection rule

The V0 graph is type-level rather than one node per biological neuron so that all comparison arms fit the fixed parameter ceiling.

1. Seed cell types are exactly `T4a`, `T4b`, `T4c`, `T4d`, `T5a`, `T5b`, `T5c`, `T5d`.
2. Add every visual-system cell type receiving at least 5 published synapses in aggregate from any seed type.
3. Retain only types present in the v783 static visual-system connectivity table.
4. Build the induced directed graph over the resulting type set.
5. Edge weight is aggregate synapse count from source type to target type.
6. Remove self-loops only if the same rule is applied to both FlyWire and rewired controls; record that choice in the graph fingerprint.
7. Sort node IDs lexicographically by published type name before serialization.

The rule is selected from prior biological knowledge about the motion subsystem and is frozen without inspecting final-test outcomes.

## Rewired control

Use `directed-double-edge-swap-v1` and preserve, to the extent guaranteed by the implementation and checked by tests:

- node count;
- edge count;
- in/out degree sequence;
- edge-weight multiset.

Generate the rewired topology from the frozen FlyWire graph using the same per-run seed policy. The model architecture, input encoder, optimizer, training budget, and parameter ceiling must be identical between FlyWire and rewired arms.

## Baselines

Run the same frozen observation tensors through:

- GRU;
- random sparse graph matched in node/edge scale;
- rewired FlyWire graph;
- FlyWire graph.

`FlyWire > GRU` or `FlyWire > random` alone is not topology evidence.

## Metrics and success criterion

Primary metric: macro F1 across the 10 frozen action classes.

Secondary metrics:

- balanced accuracy;
- macro F1 under frozen keypoint-coordinate noise;
- macro F1 under frozen keypoint masking.

Freeze the topology success criterion before final-test execution as:

- mean paired `(FlyWire macro_f1 - rewired macro_f1)` across the five declared seeds >= **0.02**;
- and the paired difference must be positive for at least **4 of 5** seeds.

This is a predeclared practical-effect threshold, not a claim of statistical significance. A result below the threshold is reported as `topology_advantage_not_established` and is a valid outcome.

## Remaining preflight fields

Before execution, the following byte-derived values still must be filled from the legally acquired local data and exact extractor installation:

- dataset archive/content hash;
- exact Ultralytics package version;
- `yolo26n-pose.pt` SHA-256;
- deterministic train/validation/final-test split hash;
- observation-schema hash;
- visual-system-parts-list source commit;
- ingested connectivity-table SHA-256;
- resulting selected graph fingerprint.

Until those values are present, Phase 2 remains a frozen **preflight design**, not a confirmatory run.
