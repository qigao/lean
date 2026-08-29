from __future__ import annotations

from .two_stage_source import (
    TwoStageSourceFile,
    TwoStageSourceManifest,
    VerifiedTwoStageSnapshot,
    load_two_stage_source_manifest,
    verify_two_stage_snapshot,
    write_two_stage_source_manifest,
)

__all__ = [
    "TwoStageSourceFile",
    "TwoStageSourceManifest",
    "VerifiedTwoStageSnapshot",
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
