# YOLO + FlyWire behavior recognition V0 readiness report

## Status

The V0 research harness is **implementation-ready for a frozen real-data topology comparison**, but it does **not** yet support a claim that FlyWire topology improves pedestrian/gesture recognition.

This report distinguishes software/protocol evidence from empirical connectome-topology evidence.

## Verified implementation gates

GitHub Actions is the authoritative execution environment for this branch. The following RED→GREEN gates were verified on exact heads:

1. Experiment/evidence contract and package skeleton — GREEN run `34681741564` at `9920b8522f2c6a891bf267fd71b29f0c0470290d`.
2. Synthetic point-sequence schema/generator — RED `34682156195`; GREEN `34682273631` at `1658e6cd19d5699853ee10a3a1201f3448a04852`.
3. Frozen temporal pose features — RED `34682355335`; GREEN `34682449104` at `2c79976ba3b77259d0f8509c861f16bb42649794`.
4. Graph controls and degree-preserving rewiring — RED `34682533141`; GREEN `34682621197` at `372860498f9cebfc64c7ac49c72c73c65d79bbb4`.
5. GRU + fixed-topology graph recurrent models — RED `34682707957`; GREEN `34682814760` at `ff52769e86c6df7163e45fc998ec3ed39ce7d8ec`.
6. Deterministic training/evaluation and evidence manifests — RED `34682907471`; GREEN `34683676790` at `d71c3df73c4018c154edac4d88d56544a209b656`.
7. FlyWire local-CSV ingestion/provenance and matched rewired control — RED `34685420892`; GREEN `34685511872` at `11ba5a493cdec7fa630f0384520646608ca0f3ab`.
8. Protocol-bound CLI — RED `34685610826`; GREEN `34685706709` at `c8159077a9c0456b4103b6777ddd2ecbf8894dd0`.
9. Frozen evaluation boundary — RED `34685836083`; GREEN `34685968084` at `ec626eef49df6f86a2551f583693c4d9e4b119d4`.
10. Robustness / graph-lesion utilities — RED `34686068210`; GREEN `34686182988` at `62dda08563890835dc89acc2a7736cc063f62ed0`.
11. Frozen YOLO/Pose sequence ingestion — RED `34686280328`; GREEN `34686373327` at `7ce669c0bf49963035a83e4935896b1f5785d679`.
12. Frozen YOLO/Pose data-contract documentation — GREEN run `34686381903` at `c073ce0d8ac58b80cd3b500c204abe4dc946024c`.

## What is established

The branch now establishes that:

- YOLO/Pose point sequences can be ingested under a frozen schema and transformed into the same temporal feature tensor for all model families.
- A conventional GRU baseline and fixed-topology graph recurrent model share the same input/output contract.
- Random, FlyWire-derived, and degree-preserving rewired graph conditions can be represented without silently learning or overwriting topology.
- Rewired controls preserve directed in/out degree sequences and the edge-weight multiset while changing topology.
- FlyWire graph ingestion requires explicit release identity and selection rule; V0 does not silently query remote services or embed credentials.
- Training is seeded from one declared run seed, checkpoint selection uses validation only, and final-test evaluation is separate.
- Manifests and Lean contracts reject a topology-specific claim that lacks a matched rewired FlyWire control.
- Noise, missing-keypoint, and graph-lesion perturbations are deterministic and remain separate robustness evidence.

## What is not established

None of the verified gates above demonstrate that a FlyWire-derived topology is more accurate, more robust, or more sample-efficient than a matched rewired topology on real behavior data.

The synthetic protocol is a plumbing/smoke protocol only. Its success threshold must not be cited as connectome evidence.

No claim of replacing YOLO, ByteTrack, metric depth, re-identification, or human cognition is supported by V0.

## Remaining frozen external inputs for the confirmatory real run

Before the sealed real-data test can run, the following must be supplied and frozen without consulting final-test outcomes:

1. **Real labeled behavior/gesture dataset identity and immutable split**.
2. **Exact YOLO/Pose extractor identity/version** and the exported point-sequence dataset matching the frozen ingestion contract.
3. **Observation-schema hash** produced from the frozen preprocessing/features.
4. **Exact FlyWire release/export identity**.
5. **Explicit FlyWire visual/motion subgraph selection rule**, selected independently of final-test results.
6. **Frozen success threshold** for the topology-specific primary metric.
7. Any real-run training budget fields that differ from the template, frozen before final-test use.

`protocols/v0-real-template.json` intentionally remains non-executable until these fields are populated.

## Confirmatory comparison

The minimum topology experiment remains:

- conventional temporal baseline (GRU),
- random sparse graph,
- degree-/weight-matched rewired FlyWire,
- FlyWire-derived topology.

All arms must share the same observation tensor, split, seeds, training budget, preprocessing, and metric definitions.

The decisive comparison for a topology-specific claim is **FlyWire vs matched rewired FlyWire**, not FlyWire vs YOLO alone and not FlyWire vs GRU alone.

## Decision boundary

If FlyWire fails to beat the frozen matched rewired control under the predeclared threshold, V0 records a null/negative topology result. The topology, threshold, or final-test split must not be revised after seeing the outcome and then presented as confirmatory evidence.

If the frozen criterion is met reproducibly, the permitted conclusion is limited to evidence that the selected FlyWire-derived topology supplied a useful inductive bias for the specified temporal behavior-recognition task under the frozen protocol.
