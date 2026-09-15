# Pose Graph + SSM real NTU skeleton CI

This workflow runs the frozen Graph + SSM development experiment on legally acquired NTU RGB+D 120 skeleton bytes without committing or uploading those raw bytes.

## Workflow

The real-data workflow is:

```text
.github/workflows/pose-graph-ssm-real-skeleton.yml
```

It performs, in one GitHub-hosted runner:

```text
private transport manifest
  -> validate canonical NTU skeleton identities
  -> download each selected .skeleton file over HTTPS
  -> verify exact size + SHA-256
  -> pose-graph-ssm prepare
  -> pose-graph-ssm compare-validation
  -> remove private manifest + raw skeleton files
  -> upload URL-free scientific evidence only
```

The workflow does **not** expose a final-test scoring command. Final-test skeletons may be present in the private inventory so their bytes and split identity are bound, but model preparation keeps them inventory-only and does not normalize or feature-transform them.

## Required GitHub Actions secret

Configure exactly this repository Actions secret:

```text
NTU120_SKELETON_MANIFEST_URL
```

Its value must be an HTTPS URL for a private authorized transport manifest. The URL itself is not written to public evidence.

If the secret is absent, the workflow intentionally fails before downloading any dataset bytes with:

```text
NTU120_SKELETON_MANIFEST_URL secret is required
```

## Transport manifest schema

The private JSON manifest uses:

```json
{
  "format_version": 1,
  "kind": "ntu120_skeleton_remote_transport",
  "samples": [
    {
      "filename": "S001C001P056R001A008.skeleton",
      "sample_id": "S001C001P056R001A008",
      "setup": 1,
      "camera": 1,
      "subject": 56,
      "repetition": 1,
      "action": 8,
      "split": "train",
      "size_bytes": 123456,
      "sha256": "<64 lowercase hex characters>",
      "locator": "https://private-storage.example/.../S001C001P056R001A008.skeleton"
    }
  ]
}
```

The validator independently recomputes sample identity and split from the canonical NTU filename. It rejects unknown fields, non-HTTPS locators, duplicate sample/content identities, actions outside the frozen ten-class task, and split drift.

The real workflow additionally requires all ten frozen actions to be represented in both development `train` and `validation` partitions before any byte materialization or training claim proceeds.

## Build a canonical manifest from local source bytes

Once the selected skeleton objects are uploaded to a private HTTPS location whose object names match their canonical NTU filenames, generate the manifest from the legally acquired local directory:

```bash
PYTHONPATH=pose_graph_ssm/src \
python -m pose_graph_ssm.remote_skeleton build \
  --root /datasets/ntu120-skeleton \
  --protocol pose_graph_ssm/protocols/v1-development-preflight.json \
  --locator-prefix https://PRIVATE-HOST/PRIVATE-PREFIX \
  --output /private/ntu120-skeleton-transport.json
```

The builder obtains `sample_id`, split, byte size, and SHA-256 from the actual local skeleton files. Local absolute paths are not serialized. The `locator-prefix` only supplies the private transport URL prefix.

Upload the generated manifest to a private HTTPS location, then set `NTU120_SKELETON_MANIFEST_URL` to that manifest URL.

## Validate without downloading source bytes

```bash
PYTHONPATH=pose_graph_ssm/src \
python -m pose_graph_ssm.remote_skeleton validate \
  --manifest /private/ntu120-skeleton-transport.json \
  --protocol pose_graph_ssm/protocols/v1-development-preflight.json \
  --public-output /private/scientific-source.json
```

`scientific-source.json` contains no transport locators.

## Evidence uploaded by CI

Only URL-free evidence is retained:

```text
scientific-source.json
protocol.json
binding.json
feature_stats.json
development_inventory.json
final_test_inventory.json
20 per-model/seed result JSON files
summary.json
```

Before artifact upload, CI removes:

```text
transport-manifest.json
ntu-skeleton/
```

The raw NTU skeleton files are therefore not published as GitHub Actions artifacts.

## Current scientific boundary

A GREEN code/test workflow proves the experiment machinery is reproducible; it does not establish that GraphSSM is better than GRU/SSMOnly/GraphTCN.

A scientific result exists only after the real skeleton workflow completes the frozen 4-model x 5-seed development matrix. The reducer then applies the predeclared EarlyAUC, retention/recovery, graph-attribution, and SSM-attribution thresholds. Full-sequence F1 cannot override a temporal null result.
