"""Exact rational schema primitives for BB conformance fixtures.

This module deliberately uses :class:`fractions.Fraction` for every rational
field. Production floating-point belief values are outside this contract.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

SCHEMA = "bb-rational-conformance/v1"


def rat(value: int | Fraction) -> dict[str, int]:
    """Return the canonical structural JSON encoding of an exact rational."""

    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("rational value must be an int or Fraction")
    normalized = Fraction(value)
    return {"num": normalized.numerator, "den": normalized.denominator}


def parse_rat(value: object) -> Fraction:
    """Parse one strict, already-reduced structural rational value.

    The conformance schema rejects permissive coercions. In particular JSON
    floating-point values, booleans, nonpositive denominators, extra fields,
    and unreduced numerator/denominator pairs are invalid.
    """

    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise TypeError("rational must be exactly an object with num and den")

    num = value["num"]
    den = value["den"]
    if isinstance(num, bool) or not isinstance(num, int):
        raise TypeError("rational numerator must be an integer")
    if isinstance(den, bool) or not isinstance(den, int):
        raise TypeError("rational denominator must be an integer")
    if den <= 0:
        raise ValueError("rational denominator must be positive")

    normalized = Fraction(num, den)
    if normalized.numerator != num or normalized.denominator != den:
        raise ValueError("rational numerator and denominator must be reduced")
    return normalized


def canonical_json_line(case: dict[str, Any]) -> str:
    """Serialize one canonical compact JSONL record with a final newline."""

    return json.dumps(
        case,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
