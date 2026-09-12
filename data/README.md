# Data and provenance boundary

This repository does not redistribute restricted or large external datasets. Real confirmatory execution requires locally mounted, legally acquired source data and byte-derived provenance hashes.

## NTU RGB+D 120

The Phase 2 protocol uses RGB video from NTU RGB+D 120. Raw video and official skeleton files are not committed here. The model input is generated from RGB through the frozen YOLO/Pose extractor; official NTU skeleton data is allowed only as an extraction sanity check.

Before confirmatory final-test execution, record:

- dataset/archive content hash;
- frozen X-Sub120-derived split hash;
- exact Ultralytics package version;
- `yolo26n-pose.pt` SHA-256;
- observation-schema / encoder hash.

## FlyWire visual-system snapshot

The FlyWire visual-system source is publicly reproducible and pinned independently of mutable branch names:

- repository: `murthylab/visual-system-parts-list`
- release/materialization: FAFB v783
- commit: `0d8574d46627ce7fadd968a3c5d602e837325373`
- connectivity path: `data/type_to_type_connection_and_synapse_counts.csv`
- Git blob SHA-1: `5183755ecbb41d5c8cee1a4a2d99b8eecba75c52`

CI downloads that exact path from that exact commit and produces a SHA-256 provenance record. The SHA-256 must be copied into the Phase 2 protocol before a real topology claim can execute.

No external source bytes are vendored merely to satisfy the protocol. Derived, small, license-compatible graph fingerprints and manifests may be committed once produced.
