from __future__ import annotations

from collections.abc import Mapping, Sequence

from narrative_dynamics.attestation import measure_implementation

from . import narrative_two_stage as _base


_ORIGINAL_CONFIGURATION = _base._configuration
_ORIGINAL_MANIFEST_IDENTITY = _base.NarrativeTwoStageModelSource.manifest_identity


def _spaceship_configuration(value: object) -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        rows = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = tuple(value)
    else:
        raise TypeError(
            "two-stage first-stage configuration must be a mapping or pair sequence"
        )

    parsed: dict[str, object] = {}
    for row in rows:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 2
        ):
            raise ValueError("two-stage first-stage configuration rows must be pairs")
        raw_key, raw_value = row
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("two-stage first-stage configuration keys must be text")
        if raw_key in parsed:
            raise ValueError("two-stage first-stage configuration keys must be unique")
        parsed[raw_key] = raw_value

    expected = {"symbol0", "symbol1"}
    if set(parsed) != expected:
        raise ValueError(
            "spaceship first-stage configuration must bind both presentation indices"
        )
    symbols = tuple(parsed[key] for key in sorted(expected))
    if any(
        isinstance(symbol, bool)
        or not isinstance(symbol, int)
        or symbol not in (0, 1)
        for symbol in symbols
    ):
        raise ValueError("spaceship presentation indices must be binary integers")

    # Upstream writes independent spaceship and planet indices here.  Equal
    # values (00/11) are valid and do not change the canonical first-stage
    # action decoding, which is derived outside model input from transition and
    # final state.
    return tuple((key, parsed[key]) for key in sorted(parsed))


def _configuration(
    task_variant: str,
    value: object,
) -> tuple[tuple[str, object], ...]:
    if task_variant != "spaceship":
        return _ORIGINAL_CONFIGURATION(task_variant, value)
    return _spaceship_configuration(value)


def _manifest_identity(self) -> dict[str, object]:
    identity = dict(_ORIGINAL_MANIFEST_IDENTITY(self))
    identity["spaceship_configuration_implementation_identity"] = (
        measure_implementation(_configuration).manifest_identity()
    )
    return identity


def install() -> None:
    _base._configuration = _configuration
    _base.NarrativeTwoStageModelSource.manifest_identity = _manifest_identity


__all__ = ["install"]
