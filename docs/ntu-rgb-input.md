# NTU RGB input inventory

This is the input-integrity stage of Phase 2, before YOLO/Pose extraction.
It reads an explicitly supplied, legally acquired local directory. It does not
acquire data, decode videos, invoke YOLO, train a model, or evaluate the final test.

## Commands

From the repository root, using the research branch:

```bash
PYTHONPATH=python python -m yolo_flywire.ntu_io freeze \
  --root /datasets/ntu120 \
  --output /private-experiment/ntu120-rgb-inventory.json

PYTHONPATH=python python -m yolo_flywire.ntu_io verify \
  --root /datasets/ntu120 \
  --manifest /private-experiment/ntu120-rgb-inventory.json
```

With the package installed (`python -m pip install -e ./python`), the `PYTHONPATH`
prefix is unnecessary. The input module itself uses only the standard library.
On PowerShell, set `$env:PYTHONPATH = "python"` before the module command instead.
The output file must not already exist; freezing never overwrites prior evidence.
Use an immutable/read-only input mount. These checks detect observed changes but
are not a filesystem snapshot or a defense against a concurrent malicious writer.

## File and label boundary

Eligible filenames are exactly `SsssCcccPpppRrrrAaaa_rgb.avi` with three decimal
digits per field: setup 1..32, camera 1..3, subject 1..106, repetition 1..2,
action 1..120. Filenames are parsed independently of their containing folders.
The filename convention is documented by the dataset authors in their
[README at a pinned revision](https://github.com/shahroudy/NTURGB-D/blob/ac2ebc87e6e9777ea6bac67e65e53b325b903f74/README.md).

The selected actions remain A8, A9, A22, A23, A26, A27, A31, A34, A35, A36, using
the labels already declared in `protocols/v0-real-ntu120-preflight.json`.
Other action classes and non-RGB modalities are not read. In particular, the
missing-skeleton exclusion list is not used to silently drop RGB samples.
Malformed RGB filenames, symlinks, nonregular/empty selected files, duplicate
sample IDs, duplicate content (including across partitions), and incomplete
per-partition label coverage are errors. There is no alternate-name or
alternative-modality fallback.

A sample's subject ID, action label and filename are administrative metadata,
not allowed classifier features. Later feature extraction must emit only the
frozen point/confidence observation stream, never these identifiers.

## Explicit split reference

The outer subject list is transcribed from Liu et al.,
[arXiv:1905.04757v2, section 3.2.1](https://arxiv.org/pdf/1905.04757v2).
This precise reference has P056 on the training side and P106 on the test side;
do not replace it by an unspecified library's interpretation of "X-Sub120".
The complete lists and reference ID are carried in every manifest's split policy.

The existing project requirement for deterministic training-side validation is
concretized as follows. Rank the reference's 53 training-side subjects by SHA-256
of the ASCII string `yolo-flywire:ntu120:v0:validation:P{subject:03d}`, breaking a
hash tie by subject ID; reserve the first 11 for validation. This yields:

```text
validation: 14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103
```

The remaining 42 training-side subjects train. The 53-subject complement in
1..106 is final test. This validation split is project-defined, not an official
validation split. Assignment never depends on file availability, action class,
camera, repetition, frame count or any of the five model seeds. All observations
and cameras of the same subject therefore stay together. Integrity hashing of
sealed final-test files is not model selection or evaluation of their contents.

## Hash layout and verification

JSON is serialized with sorted object keys, compact separators, UTF-8 encoding
and no NaN values. Samples are ordered by canonical sample ID. Each row records
identity, label, split, relative POSIX path, file byte length and streaming
SHA-256. Absolute roots, filesystem timestamps and machine-specific paths are
not included in the manifest. Moving the same directory tree to another root
therefore does not change it.

`dataset_content_hash` is SHA-256 of the ordered JSON list of
`{sample_id, size_bytes, sha256}` records. `split_hash` is SHA-256 of the JSON object
with `dataset_content_hash`, `policy` (the entire split policy), `task_labels`
(the ordered labels), and `assignments` (ordered `{sample_id, split}` records).
Changing selected media bytes or membership changes the bound identity.

Verification does not trust row paths or advertised checksums. It enumerates the
explicit local root again, rehashes the selected files, rebuilds the complete
manifest and compares its canonical JSON to the supplied manifest. Changed rows,
labels, subject assignments, schema/policy fields or byte-derived hashes fail.
The publishing step uses an exclusive atomic link from a complete temporary file
in the output directory. A filesystem that cannot support this operation fails;
there is no overwrite or partial-write fallback.

## What this does not establish

A manifest requires all ten labels in each partition, but that is not proof that
all official release files have been supplied. It deliberately records
`official_release_completeness_verified: false`. An independent official roster
or archive reconciliation remains necessary before a complete-release claim.
Likewise, valid names and nonempty bytes do not prove that AVI files decode:
`media_decoding_verified` remains false until the future extraction stage checks
actual frames. Both flags must not be upgraded by this inventory module.

CI's opaque byte fixtures verify input plumbing only. The module does not populate
the real preflight's dataset, split, schema or weight hashes, and the existing
real `compare` command remains protocol-validation-only. The next stage must
bind this inventory to pinned YOLO weights/version, explicit extraction settings,
and the resulting point-sequence bytes before any real classifier run.

Keep NTU media, extracted observations and locally generated manifests out of
public repository commits and CI uploads unless the applicable permissions
explicitly allow redistribution. Follow the dataset's
[access and use terms](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
