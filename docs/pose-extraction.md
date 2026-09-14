# Development-only YOLO/Pose video extraction

`python -m yolo_flywire.pose_extract` connects the verified local NTU RGB inventory to actual PyAV decoding and Ultralytics pose inference. It processes **train and validation only**. There is no final-test switch. Final-test files are integrity-hashed by inventory verification but are never decoded or inferred by this command.

## Prepare explicit, trusted inputs

Use a private, immutable local NTU tree and the inventory produced by `ntu_io freeze`. Supply a trusted, independently acquired `yolo26n-pose.pt`. The extractor will not download weights, substitute another model or install missing dependencies. The checkpoint SHA binds bytes; it does not prove that a renamed file is an official pretrained checkpoint. PyTorch checkpoint loading requires trusted files because model deserialization can execute pickle code.

Install the optional dependencies in a dedicated Python environment:

```bash
python -m pip install --only-binary=av -e './python[dev,pose]'
```

The integration build pins Ultralytics 8.4.146 and PyAV 15.1.0. A 14.4.0 install attempted a source build without the required FFmpeg development libraries, so it was replaced before any real extraction. Binary-only PyAV installation prevents an implicit system-library build. Other numerical dependencies must also be explicitly identified in the extraction config; the installer alone does not freeze a real experiment environment.

Create an extraction configuration from independently approved weights and the intended installed environment. All six fields are mandatory:

```json
{
  "weights_sha256": "<verified lowercase 64-character SHA-256>",
  "ultralytics_version": "8.4.146",
  "torch_version": "<exact installed distribution version>",
  "numpy_version": "<exact installed distribution version>",
  "av_version": "15.1.0",
  "opencv_version": "<exact installed opencv-python distribution version>"
}
```

Placeholders are intentionally invalid. Compute the weight hash from the actual trusted file, compare it with the intended source, and freeze the config before extracting observations. Inspect distribution versions with `importlib.metadata.version`; this command does not accept ranges, `latest`, missing values or unknown config fields.

From the repository root:

```bash
PYTHONPATH=python python -m yolo_flywire.pose_extract \
  --root /datasets/ntu120 \
  --inventory /private-experiment/ntu120-rgb-inventory.json \
  --weights /private-models/yolo26n-pose.pt \
  --config /private-experiment/pose-extraction.json \
  --output /private-experiment/development-poses
```

The output directory must be new and outside the input tree. Keep media, observations, checkpoints and local inventories private; this workflow does not authorize their redistribution.

## Observation and timing contract

Inference is CPU float32 with one image at a time, `imgsz=640`, `conf=0.25`, `iou=0.7`, `rect=false`, `augment=false`, `max_det=300`, person class 0, and no saving, visualization or tracking. `max_det` is deliberately not 1: limiting to one would hide ambiguous multi-person frames. The model runs in evaluation/inference mode; the dedicated process sets one PyTorch compute thread and deterministic algorithms.

PyAV decodes every frame of exactly one AVI video stream. Original integer PTS and rational time bases are retained. Missing/nonincreasing timing, corrupt/empty videos, declared frame-count mismatches, invalid images and dimension drift are errors. The code does not guess a frame rate or silently repair a video.

No person detection produces 17 `[0,0,0]` keypoints and person confidence zero, preserving the frame and its time. One detection produces 17 original-image pixel x/y/confidence triples. Multiple detected people abort extraction rather than choosing the biggest box or highest score. Missing detections are uncertainty, not evidence that the video truly contains no person. This policy can reject real footage and does not implement identity tracking.

The output directory contains:

- `observations.jsonl`: existing pose-JSONL geometry fields, target label and frame index.
- `timing.jsonl`: matching `(sample_id, frame_index)` rows with `pts` and `time_base: [numerator, denominator]`.
- `manifest.json`: published last, binding input content/split/inventory hashes, exact package versions, Python/platform, extraction spec and code hashes, per-sample frame/missing counts and the SHA-256 of each JSONL file.

Timestamp seconds are exactly `pts * numerator / denominator`. Geometry uses pixels, not letterboxed model-input coordinates. The existing `load_pose_jsonl` can read the geometry but **does not consume the timing sidecar**. A future real temporal runner must join timing or explicitly freeze a time-resampling policy. This extraction schema hash is not a substitute for the final feature-encoder/schema hash in the real experiment protocol.

Input bytes and weight bytes are reverified before manifest publication. No existing output is overwritten. Caught failures clean only files created by this extraction. A killed process may leave an incomplete directory; consumers must require the final manifest and verify its referenced file hashes. The private/read-only input requirement remains: stat checks and repeated hashing are not an atomic filesystem snapshot against malicious races.

## Verification and limits

`tests/test_pose_extract.py` exercises orchestration with test doubles; it is not a detector-accuracy test. `integration/test_pose_backend.py` uses the real decoder and a locally built **untrained** YOLO26n-pose checkpoint on generated lossless videos, with outbound socket connections blocked. This checks actual library APIs, timing preservation, deterministic fixture outputs and exclusion of final-test fixtures. It does not download pretrained assets or use NTU media, and it provides no recognition-accuracy evidence.

The manifest records package versions and platform, not a hermetically frozen OS, FFmpeg build or wheel-hash environment. Repeatability in one tested environment is not a guarantee of bitwise identity on every platform. A confirmatory environment still needs a separate execution freeze.

The real dataset/weight/schema protocol fields remain unresolved; real four-arm training/evaluation is not implemented by this module. No FlyWire advantage, fine-finger recognition or real NTU result follows from this CI gate. See issue #68 for exact-head results.
