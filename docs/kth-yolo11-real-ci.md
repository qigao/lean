# KTH YOLO11 public real-CI

Tracked by Issue #68 and PR #69. This workflow provides a public real-RGB development benchmark that can execute on standard GitHub-hosted runners without repository secrets or a self-hosted runner. It is separate from the NTU RGB+D 120 evidence family and does not replace or reinterpret NTU confirmatory evidence.

## Sources

The workflow uses the official KTH Human Action Database only:

- `https://www.csc.kth.se/cvap/actions/00sequences.txt`
- `https://www.csc.kth.se/cvap/actions/boxing.zip`
- `https://www.csc.kth.se/cvap/actions/handclapping.zip`
- `https://www.csc.kth.se/cvap/actions/handwaving.zip`
- `https://www.csc.kth.se/cvap/actions/jogging.zip`
- `https://www.csc.kth.se/cvap/actions/running.zip`
- `https://www.csc.kth.se/cvap/actions/walking.zip`

The YOLO pose checkpoint is downloaded directly from the Ultralytics official GitHub Release asset:

`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt`

There is no source or model mirror fallback.

KTH states that the database is publicly available for non-commercial use. Work using the database should cite Christian Schuldt, Ivan Laptev, and Barbara Caputo, *Recognizing Human Actions: A Local SVM Approach*, ICPR 2004.

## Frozen task

Class order is:

1. `boxing`
2. `handclapping`
3. `handwaving`
4. `jogging`
5. `running`
6. `walking`

The official subject split frozen by this protocol is:

- train: persons 11, 12, 13, 14, 15, 16, 17, 18;
- validation: persons 19, 20, 21, 23, 24, 25, 01, 04;
- final test: persons 22, 02, 03, 05, 06, 07, 08, 09, 10.

The classification unit is one official interval from `00sequences.txt`, not a whole AVI. The complete source has 600 parent AVI files and 2,391 official action subsequences.

## Final-test seal

Final-test parent AVI bytes are materialized from the official ZIP only to establish byte size and SHA-256. The extractor never calls the video decoder or YOLO predictor for a final-test subject, and no final-test pose sidecar is created.

A separate reviewed protocol would be required before any KTH final-test decode or classifier evaluation.

## Pose extraction

For train and validation parent videos, each AVI is decoded once. Frames outside all official action ranges are used only to advance the stream; they are never passed to the pose predictor. Frames inside an official interval are passed to the exact frozen YOLO11n pose checkpoint and emitted as COCO17 geometry plus source PTS/time-base. Each emitted subsequence starts its sidecar `frame_index` at zero while preserving the parent video's source timing.

The pinned runtime is:

- Ultralytics `8.4.146`
- PyTorch `2.14.0`
- NumPy `2.4.6`
- PyAV `15.1.0`
- OpenCV `5.0.0.93`

The first successful official-source development run records exact archive, member, sequence-file, model, schema, encoder, and extraction identities. Those byte identities are evidence for that run; a later freeze commit may promote them into static expected pins.

## Four-arm comparison

After the six action shards are aggregated, the KTH-specific source/index layer binds the resulting train/validation sidecars into the existing `IndexedPoseDevelopment` type. Training and evaluation then reuse the existing four-arm core without a KTH-specific trainer:

- GRU;
- random sparse graph;
- degree/weight-matched rewired FlyWire graph;
- original FlyWire graph.

Seeds remain `7, 11, 19, 23, 31`; the budget remains 20 epochs, exactly 40 optimizer updates, and a 50,000-parameter ceiling.

KTH does not inherit the NTU confirmatory threshold. The report records validation macro-F1 and paired FlyWire-minus-rewired differences as development evidence only.

## GitHub Actions

`.github/workflows/yolo-flywire-kth-real.yml` supports manual dispatch and a narrow push trigger on `research/yolo-flywire-behavior-v0` when KTH source/protocol/workflow files change. It uses standard `ubuntu-latest` runners and no secrets.

The workflow is:

`coordinator -> six action extraction jobs -> aggregate + FlyWire four-arm comparison`

No KTH ZIP, AVI, YOLO checkpoint, or final-test observation is uploaded. The final artifact contains only source/protocol hashes, the aggregate manifest, execution config, FlyWire provenance, and the development report.

A successful ordinary fixture CI is not real-data evidence. Only a successful `yolo-flywire-kth-real` run that downloads the official KTH bytes and reaches the final development artifact is treated as a real KTH development experiment.
