"""Ensure every maintained public mesh theorem has an axiom report."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

THEOREM = re.compile(r"(?m)^theorem[ \t]+([A-Za-z_][A-Za-z_0-9.]*)\b")
REPORT = re.compile(
    r"(?m)^#print[ \t]+axioms[ \t]+"
    r"(NarrativeDynamics\.[A-Za-z_][A-Za-z_0-9.]*)[ \t]*$"
)


def required_theorems(sources: list[str], report_fixture: str) -> tuple[str, ...]:
    expected = tuple("NarrativeDynamics." + m.group(1)
                     for src in sources for m in THEOREM.finditer(src))
    if not expected:
        raise ValueError("mesh theorem coverage found no declarations")
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate public mesh theorem declaration")
    reported = tuple(m.group(1) for m in REPORT.finditer(report_fixture))
    if len(set(reported)) != len(reported):
        raise ValueError("duplicate Lean mesh axiom report")
    missing = sorted(set(expected) - set(reported))
    stale = sorted(set(reported) - set(expected))
    if missing or stale:
        raise ValueError(
            f"mesh theorem coverage mismatch: missing={missing}; stale={stale}"
        )
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("modules", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        names = required_theorems(
            [path.read_text(encoding="utf-8") for path in args.modules],
            args.reports.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
