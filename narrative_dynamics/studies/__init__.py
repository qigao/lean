from __future__ import annotations

from .two_stage_source import (
    TwoStageSourceFile,
    TwoStageSourceManifest,
    VerifiedTwoStageSnapshot,
    freeze_feher_hare_v1_source_manifest,
    load_two_stage_source_manifest,
    verify_two_stage_snapshot,
    write_two_stage_source_manifest,
)

__all__ = [
    "TwoStageSourceFile",
    "TwoStageSourceManifest",
    "VerifiedTwoStageSnapshot",
    "freeze_feher_hare_v1_source_manifest",
    "load_two_stage_source_manifest",
    "verify_two_stage_snapshot",
    "write_two_stage_source_manifest",
]

# Install Study V1 Task 10 symbols on the canonical study module while keeping
# study-specific APIs out of the narrative_dynamics package root.
from . import feher_hare_two_stage_v1 as _feher_hare_two_stage_v1
from . import two_stage_final as _two_stage_final

for _name in _two_stage_final.__all__:
    setattr(
        _feher_hare_two_stage_v1,
        _name,
        getattr(_two_stage_final, _name),
    )
    if _name not in _feher_hare_two_stage_v1.__all__:
        _feher_hare_two_stage_v1.__all__.append(_name)

# Install Study V1 Task 11 orchestration on the same canonical module.
from . import two_stage_locked_final as _two_stage_locked_final

for _name in _two_stage_locked_final.__all__:
    setattr(
        _feher_hare_two_stage_v1,
        _name,
        getattr(_two_stage_locked_final, _name),
    )
    if _name not in _feher_hare_two_stage_v1.__all__:
        _feher_hare_two_stage_v1.__all__.append(_name)

# The pinned Feher/Hare source uses -1 response/RT sentinels on slow rows.
# Install the correction on the canonical transform module and on the Study V1
# module, which imported the transform before this package initializer finished.
from . import two_stage_transform as _two_stage_transform
from .two_stage_slow_filter import transform_two_stage_snapshot as _slow_safe_transform

_two_stage_transform.transform_two_stage_snapshot = _slow_safe_transform
_feher_hare_two_stage_v1.transform_two_stage_snapshot = _slow_safe_transform

# Real Feher/Hare scale makes repeated full transform hashing quadratic.
# Install a payload-equivalent dataset builder that computes immutable lineage
# hashes once, then reuses them for record metadata and dataset provenance.
from .two_stage_dataset_builder import (
    build_two_stage_observation_dataset as _hash_cached_dataset_builder,
)

_two_stage_transform.build_two_stage_observation_dataset = _hash_cached_dataset_builder
_feher_hare_two_stage_v1.build_two_stage_observation_dataset = _hash_cached_dataset_builder

# Candidate-parallel V2 keeps FINAL outside the executor and exports only the
# explicit TRAIN/SELECTION freeze entry point from the studies package.
from .feher_hare_two_stage_parallel import fit_and_freeze_feher_hare_models_parallel

__all__.append("fit_and_freeze_feher_hare_models_parallel")

# Measurement Validity V1 is a study-specific pipeline.  Export its runner
# here while keeping it out of the narrative_dynamics package root.
from .feher_hare_measurement_validity_v1 import (
    run_feher_hare_measurement_validity_v1,
)

__all__.append("run_feher_hare_measurement_validity_v1")
