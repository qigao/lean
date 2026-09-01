"""Controlled stand-in for the Blender executable protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    if "--python-exit-code" not in sys.argv:
        return 9
    fail = "--fake-fail" in sys.argv
    invalid = "--fake-invalid" in sys.argv
    noisy = "--fake-noisy" in sys.argv
    missing = "--fake-missing" in sys.argv
    if fail:
        return 7
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args(arguments)
    packet = json.loads(Path(options.input).read_text(encoding="utf-8"))
    if noisy:
        sys.stdout.buffer.write(b"x" * (2 * 1024 * 1024))
        sys.stdout.buffer.flush()
    if missing:
        return 0
    header = b"NOT-BLEND" if invalid else b"BLENDER-vFAKE"
    Path(options.output).write_bytes(
        header + packet["content_hash"].encode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
