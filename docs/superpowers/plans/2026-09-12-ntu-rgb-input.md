# NTU RGB Input Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the approved Phase 2 local RGB inputs and subject-isolated split to reproducible byte-derived hashes before YOLO extraction.

**Architecture:** Add one standard-library-only input module. It inventories actual selected RGB files, derives the declared subject split, and publishes a canonical, non-overwriting manifest. Verification rebuilds the inventory from local bytes; it never follows paths supplied by an unverified manifest. Existing graph, detector, training, and real-comparison code stays unchanged.

**Tech Stack:** Python 3.11+, pathlib/os/hashlib/json/dataclasses, pytest, existing GitHub Actions.

**Spec:** `docs/phase2-real-data-freeze.md` (approved RGB-only, ten-class, X-Sub120 outer boundary and deterministic training-side validation rule).

## Global Constraints

- No BB, ByteTrack, depth, NTU skeleton features, detector execution, or classifier training.
- No final-test model selection or evaluation. Inventory hashing is an integrity operation, not semantic access to test observations.
- No dataset/weight downloads or media redistribution. Tests create opaque byte fixtures, not NTU data.
- Do not populate real protocol hashes using fixtures or declare release completeness/video decodability from a filename and a checksum.
- Preserve the frozen ten labels, graph/control fingerprints, model seeds, training budget and success criterion.
- Keep all modifications on `research/yolo-flywire-behavior-v0`, with exact-head RED/GREEN evidence recorded in #68.

## Reference and split concretization

Use the explicit 53-subject training list in Liu et al., arXiv:1905.04757v2, section 3.2.1: https://arxiv.org/pdf/1905.04757v2 . In this reference P056 is on the training side and P106 is on the final-test side. Record this precise reference rather than relying on an unspecified third-party loader. Names follow the authors' repository README: https://github.com/shahroudy/NTURGB-D/blob/ac2ebc87e6e9777ea6bac67e65e53b325b903f74/README.md .

Validation is a project-defined concretization, not an official validation partition: rank all 53 training-side subject IDs by `(sha256(ASCII('yolo-flywire:ntu120:v0:validation:P{subject:03d}')).digest(), subject)` and take the first 11. The resulting validation IDs are 14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103. All other training-side subjects train; the 53-subject complement in 1..106 is final test. The rule never depends on present samples, labels, cameras, repetitions, frame counts, or model seeds.

## Task 1: Verified local inventory and CLI

**Files:** Create `tests/test_ntu_io.py`, `python/yolo_flywire/ntu_io.py`, `docs/ntu-rgb-input.md`. Do not change the existing real protocol's null hashes or the graph generator.

**Interfaces:**

```python
@dataclass(frozen=True)
class NtuSample:
    sample_id: str
    setup: int
    camera: int
    subject: int
    repetition: int
    action: int

def parse_rgb_name(name: str) -> NtuSample: ...
def split_for_subject(subject: int) -> str: ...
def build_rgb_manifest(root: str | Path) -> dict[str, Any]: ...
def verify_rgb_manifest(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]: ...
def main(argv: list[str] | None = None) -> int: ...
```

- [ ] Write the complete acceptance tests in `tests/test_ntu_io.py`: canonical identity/ranges, explicit subject partition, strict types, portable hashes, excluded modalities never read, camera isolation, changed same-size bytes/roster/labels/split/policy rejection, malformed/duplicate/incomplete data, symlinks, and CLI no-overwrite/no-output-on-rejection.
- [ ] Commit tests and this plan only. Run the existing full GitHub CI. Require the prior 81 tests and Lean to remain green while new tests fail only for the missing input module; provenance must be skipped behind the failing tests.
- [ ] Implement `parse_rgb_name` with fullmatch of `S[0-9]{3}C[0-9]{3}P[0-9]{3}R[0-9]{3}A[0-9]{3}_rgb.avi`; require setup 1..32, camera 1..3, subject 1..106, repetition 1..2, action 1..120. No path, alternate suffix or coercion fallback.
- [ ] Enumerate under an explicit local directory, reject symlink aliases and malformed RGB names, then filter to the frozen ten actions. Stream selected regular, nonempty files in 1 MiB chunks; reject observed file mutation, duplicate IDs and duplicate content. Require every target action in each of train/validation/final_test. Never apply the missing-skeleton exclusion list to RGB files.
- [ ] Build sample rows sorted by sample ID with identity, label, split, relative POSIX path, size and SHA-256. Content hash is SHA-256 of canonical JSON of ordered `{sample_id,size_bytes,sha256}` rows. Split hash binds this content hash, the complete split-policy descriptor, ordered labels, and ordered `{sample_id,split}` assignments. Do not include machine-specific absolute roots or timestamps.
- [ ] Rebuild the manifest for verification and compare the complete canonical JSON, including type distinctions. Reject any mismatch. Do not read paths from the untrusted manifest.
- [ ] Implement `python -m yolo_flywire.ntu_io freeze --root ROOT --output MANIFEST` and `verify --root ROOT --manifest MANIFEST`. Validate before writing; publish through a same-directory temporary file and exclusive atomic link, always clean the temporary file, and never replace existing evidence. Exit nonzero on invalid/missing inputs. Output scope stays `input_inventory_only`, with media-decoding and official-roster-completeness flags false.
- [ ] Document commands, source of the subject list, exact hash layout, immutable-local-input requirement, and distinction between local inventory completeness and an independently verified official release roster. State explicitly that YOLO extraction and real four-arm execution are not implemented by this module.
- [ ] Run targeted tests locally and full exact-head GitHub CI, review the diff for unchanged real hashes/graph budget, then record actual evidence in #68. Keep #68 open.

## Verification commands

```bash
PYTHONPATH=python python -m pytest tests/test_ntu_io.py -q
python -m pytest tests -q
lake build
```

The authoritative integration gate remains the existing GitHub CI; local targeted byte-fixture tests are supplementary. No empirical performance result is produced by this task.
