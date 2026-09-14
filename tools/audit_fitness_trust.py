"""Fail closed on forbidden fitness proof source and unexpected Lean axioms.

This source check is a policy lint, not a replacement for Lean's kernel. The
separate log check inspects actual transitive dependencies printed by Lean.
Only Python's standard library is needed, including in the Lean proof job.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
FORBIDDEN = re.compile(r"\b(?:sorry|admit|native_decide|axiom|unsafe|unlock_limits)\b")
RESOURCE_OPTION = re.compile(
    r"\bset_option\s+(?P<option>(?:\w+\.)*(?:maxHeartbeats|maxRecDepth|maxMemory))"
    r"\s+(?P<value>0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|[\d_]+)\b"
)
REPORT = re.compile(
    r"'(?P<name>[^\n]+?)' (?:depends on axioms:\s*\[(?P<axioms>[^\]]*)\]"
    r"|does not depend on any axioms)"
)
REPORT_START = re.compile(r"'[^\n]+?' (?:depends on axioms:|does not depend on any axioms)")


def code_only(source: str) -> str:
    """Retain possible interpolation terms without guessing the enclosing macro.

    Ordinary-string braces are checked conservatively as Lean terms: built-in
    and user macros can consume interpolated strings without a bang prefix.
    Raw strings remain literal data. This is deliberately a source-policy lint.
    """
    result = list(source)

    def mask(start: int, end: int) -> None:
        for position in range(start, end):
            if source[position] != "\n":
                result[position] = " "

    def string(index: int) -> int:
        mask(index, index + 1)
        index += 1
        while index < len(source):
            if source[index] == "\\":
                end = min(index + 2, len(source))
                mask(index, end)
                index = end
            elif source[index] == '"':
                mask(index, index + 1)
                return index + 1
            elif source[index] == "{":
                mask(index, index + 1)
                index = code(index + 1, 1)
            else:
                mask(index, index + 1)
                index += 1
        raise ValueError("unterminated Lean string")

    def code(index: int, braces: int = 0) -> int:
        while index < len(source):
            pair = source[index:index + 2]
            if pair == "/-":
                start, depth = index, 1
                index += 2
                while index < len(source) and depth:
                    pair = source[index:index + 2]
                    if pair in ("/-", "-/"):
                        depth += 1 if pair == "/-" else -1
                        index += 2
                    else:
                        index += 1
                if depth:
                    raise ValueError("unterminated Lean comment")
                mask(start, index)
            elif pair == "--":
                end = source.find("\n", index)
                end = len(source) if end < 0 else end
                mask(index, end)
                index = end
            elif source[index] == "«":
                end = source.find("»", index + 1)
                if end < 0:
                    raise ValueError("unterminated Lean escaped identifier")
                # Quotes/comment markers inside an identifier are ordinary data.
                index = end + 1
            elif (index == 0 or not re.match(r"[\w']", source[index - 1])) and (
                raw := re.match(r'r(#*)"', source[index:])
            ):
                end = source.find('"' + raw[1], index + len(raw[0]))
                if end < 0:
                    raise ValueError("unterminated Lean raw string")
                end += 1 + len(raw[1])
                mask(index, end)
                index = end
            elif char := re.match(r"'(?:\\.|[^\\'\n])'", source[index:]):
                mask(index, index + len(char[0]))
                index += len(char[0])
            elif source[index] == '"':
                index = string(index)
            elif braces and source[index] == "}":
                braces -= 1
                index += 1
                if braces == 0:
                    mask(index - 1, index)
                    return index
            else:
                if braces and source[index] == "{":
                    braces += 1
                index += 1
        if braces:
            raise ValueError("unterminated Lean interpolation")
        return index

    code(0)
    return "".join(result)


def audit_source(path: Path) -> list[str]:
    source = code_only(path.read_text(encoding="utf-8")).replace("«", "").replace("»", "")
    errors = []
    for match in FORBIDDEN.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        errors.append(f"{path}:{line}: forbidden proof token {match.group()}")
    for match in RESOURCE_OPTION.finditer(source):
        value = match["value"].replace("_", "")
        base = {"0x": 16, "0b": 2, "0o": 8}.get(value[:2].lower(), 10)
        if int(value, base) == 0:
            line = source.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: unlimited {match['option']} setting")
    return errors


def audit_log(path: Path, required: list[str]) -> list[str]:
    log = path.read_text(encoding="utf-8")
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    log = re.sub(r"(?m)^\d{4}-\d\d-\d\dT\S+\s+", "", log)
    reports = list(REPORT.finditer(log))
    errors = []
    if not reports:
        errors.append(f"{path}: no complete Lean axiom report found")
    if len(reports) != len(REPORT_START.findall(log)):
        errors.append(f"{path}: malformed or truncated Lean axiom report")
    for report in reports:
        dependencies = {item.strip() for item in (report["axioms"] or "").split(",") if item.strip()}
        for dependency in sorted(dependencies - ALLOWED_AXIOMS):
            errors.append(f"{path}: {report['name']} depends on forbidden axiom {dependency}")
    names = {report["name"] for report in reports}
    for name in required:
        if name not in names:
            errors.append(f"{path}: missing required axiom report for {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    source = modes.add_parser("source", help="audit maintained fitness Lean files")
    source.add_argument("files", nargs="+", type=Path)
    log = modes.add_parser("log", help="audit actual #print axioms output")
    log.add_argument("file", type=Path)
    log.add_argument("--require", action="append", default=[], metavar="THEOREM")
    args = parser.parse_args()
    errors = []
    try:
        if args.mode == "source":
            for path in args.files:
                errors.extend(audit_source(path))
        else:
            errors.extend(audit_log(args.file, args.require))
    except (OSError, UnicodeError, ValueError) as error:
        errors.append(str(error))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"Fitness trust {args.mode} audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
