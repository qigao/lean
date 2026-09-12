# Padding-aware recurrent model ingestion

Tracked by #68, after [pose tensor collation](pose-batches.md). This development interface consumes a complete `PoseBatch`; it does not load or authenticate a dataset, choose samples, train the real four-arm experiment, or authorize final-test access.

## Explicit API

Both existing classifiers expose `encode_padded(batch)` and `forward_padded(batch)`. Supply the `PoseBatch` returned by `collate_pose_features`, not its feature tensor alone. Configure the model for 121 input features and use CPU float32 parameters for this development path.

```python
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier

# batch comes from verified development observations, one frozen encoder and
# split-separated collation. graph comes from the independently verified graph.
gru = GRUClassifier(input_dim=121, hidden_dim=4, num_classes=10).float().cpu()
graph_model = GraphRecurrentClassifier(
    input_dim=121, graph=graph, node_dim=2, num_classes=10,
).float().cpu()
logits = gru.forward_padded(batch)
graph_logits = graph_model.forward_padded(batch)
```

The example dimensions illustrate the API; they are not a new experiment budget. `encode_padded` returns the final observed hidden representation (`[B,hidden_dim]` for GRU; `[B,node_dim]` after node averaging for the graph model). `forward_padded` applies the existing readout and returns `[B,num_classes]` logits in the original caller order. Labels, sample IDs, subjects, paths and splits never enter these methods. There are no omitted-length defaults, visibility-derived lengths, compatibility adapters or alternative encoders.

## What padding changes, and what it must not change

GRU packs the sequences using their explicit lengths with unsorted input accepted. Its recurrent module returns each final observed state in caller order. The graph model gathers only samples whose observed-step mask is true at each time, applies its existing `_step`, and copies the updated states back to their original rows. A sample's state stops changing once its original clip ends. The graph loop ends at the largest actual length, not at any additional padded extent.

Only trailing zero rows with false `time_mask` are padding. Every true time step is consumed, including an all-zero first frame with no detected person and later missing-person observations with positive `dt`. Neither raw confidence nor joint-validity masks determine whether a time step happened. The 121 channels, including joint masks and actual frame intervals, enter the original input projections unchanged. No clipping, masking rewrite, temporal pooling, gap bridging, interpolation or duration fabrication is added.

For fixed inputs and model parameters, extending a batch with additional canonical trailing padding must leave final hidden states, predictions, parameter gradients and gradients on the original observed features unchanged. The tests require exact equality for this extension in the tested CPU runtime. Comparing different batch compositions or separate unpadded executions uses numerical tolerances because floating-point operation grouping may differ. Padded feature entries have zero input gradients; nonzero observed-feature and parameter gradients are verified.

The learned equations, parameter names/counts, graph adjacency, edge orientation, diagonal population connections, and nonpersistent-adjacency checkpoint policy are unchanged. The time interval is still an input feature: this is not a continuous-time neural model or a biophysical neuron simulation.

## Validation boundary

`PoseBatch` has frozen fields but mutable tensors. Each padded model call therefore checks the object, dense CPU tensor layouts, feature float32 / length int64 / mask bool dtypes, positive batch/time dimensions, 121 input channels matching the model, length bounds, and exact prefix-mask agreement. It also requires finite feature values and zero feature rows outside the observed mask. Wrong shapes, sparse/meta inputs, stale masks, interior mask holes, zero-length examples and dirty padding are rejected with `ValueError` before recurrence, rather than silently repaired or shortened. Dense noncontiguous feature views are accepted without detaching gradients.

This shared check validates tensor structure and the padding boundary. It does **not** re-encode geometry, infer an encoder threshold from rounded confidences, verify file hashes, validate class targets, or establish split membership. Upstream bundle verification and feature/collation contracts remain required. Callers must not mutate batch tensors or model state concurrently with execution. Supporting other devices, mixed precision, compilation/export, distributed wrappers or model-level forward hooks through this explicit method is outside this verified slice.

## Verification and remaining work

```bash
python -m pytest tests/test_padded_models.py -q -W error
python -m pytest tests -q
python -m pytest integration/test_pose_backend.py -q -s
```

The focused suite checks both classifier implementations against individual unpadded runs, caller permutation and batch companions, analytic nonzero missing-frame recurrence, padding-invariant forward/backward behavior, early malformed-batch rejection, graph active-step accounting and unchanged model/input state. The real-library smoke extends generated-video extraction, timed bundle reading, COCO17 encoding and split-separated collation through both padded classifier paths. Its YOLO checkpoint and classifiers are untrained, and the graph is a tiny fixture; this is not real NTU recognition or real FlyWire-versus-rewired evidence.

The existing `forward(x)` and `encode(x)` remain the fixed-length V0/synthetic paths. They do not infer padding and are not a fallback from the padded API. The existing `train_model`/`evaluate`/real `compare` plumbing is **not** upgraded by these methods. A real development runner must explicitly carry complete batches, keep labels aligned and outside features, enforce common budgets/seeds, and use validation-only checkpoint selection. It must never pass only `batch.features` to the fixed-length training path and call that variable-length support.

Next: bind verified development samples and class targets to padding-aware training/evaluation, then execute the real four-arm development runner. Authorized NTU bytes, independently approved pretrained-weight provenance, the complete data/runtime/encoder/collation/model freeze, and sealed confirmatory execution remain separate gates. No real protocol hash, topology/control fingerprint, training/rewiring budget, seed or success threshold is changed here.
