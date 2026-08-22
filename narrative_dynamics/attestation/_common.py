from __future__ import annotations

import re


_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def validated_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


def validated_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def validated_commit(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a git commit or None")
    normalized = value.lower()
    if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a full 40-character git commit")
    return normalized
