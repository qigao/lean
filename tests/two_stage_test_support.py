from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess


MAGIC_HEADER = (
    "trial,common,reward.1.1,reward.1.2,reward.2.1,reward.2.2,"
    "isymbol_lft,isymbol_rgt,rt1,choice1,final_state,fsymbol_lft,"
    "fsymbol_rgt,rt2,choice2,reward,slow"
)
SPACESHIP_HEADER = (
    "trial,rwrd_prob0,rwrd_prob1,rwrd_prob2,rwrd_prob3,symbol0,symbol1,"
    "common,choice1,rt1,final_state,choice2,rt2,reward,slow"
)


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _write_csv(path: Path, header: str, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header.split(","))
        writer.writerows(rows)


def _magic_rows(seed: int) -> list[list[object]]:
    return [
        [0, 1, .60, .65, .55, .45, 2, 1, .4, 1, 1, 1, 2, .5, 2, 1, 0],
        [1, 0, .61, .64, .54, .46, 2, 1, .5, 1, 2, 2, 1, .6, 1, 0, 1],
        [2, 1, .62, .63, .53, .47, 1, 2, .6, 2, 2, 1, 2, .7, 2, seed % 2, 0],
    ]


def _spaceship_rows(seed: int) -> list[list[object]]:
    return [
        [0, .60, .65, .55, .45, 0, 1, 1, 0, .4, 0, 1, .5, 1, 0],
        [1, .61, .64, .54, .46, 1, 0, 0, 1, .5, 1, 0, .6, 0, 1],
        [2, .62, .63, .53, .47, 0, 1, 1, 0, .6, 0, 1, .7, seed % 2, 0],
    ]


def build_synthetic_two_stage_checkout(
    root: Path,
    *,
    magic_n: int = 6,
    spaceship_n: int = 6,
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "init", str(root)), check=True, capture_output=True)
    run_git(root, "config", "user.email", "tests@example.invalid")
    run_git(root, "config", "user.name", "Two Stage Tests")

    magic_dir = root / "results" / "magic_carpet" / "choices"
    spaceship_dir = root / "results" / "spaceship" / "choices"
    for index in range(magic_n):
        participant = f"m{index:03d}"
        _write_csv(
            magic_dir / f"{participant}_game.csv",
            MAGIC_HEADER,
            _magic_rows(index),
        )
        (magic_dir / f"{participant}_config.txt").write_text(
            "Common transitions: 1 -> blue -> (5, 6); 2 -> pink -> (3, 4);\n",
            encoding="utf-8",
        )
    for index in range(spaceship_n):
        participant = f"s{index:03d}"
        _write_csv(
            spaceship_dir / f"{participant}.csv",
            SPACESHIP_HEADER,
            _spaceship_rows(index),
        )
        (spaceship_dir / f"{participant}_info.txt").write_text(
            "{'planets': ('Black', 'Red'), 'spaceships': ('Y', 'X')}",
            encoding="utf-8",
        )
        _write_csv(
            spaceship_dir / f"{participant}_practice.csv",
            SPACESHIP_HEADER,
            _spaceship_rows(index)[:1],
        )

    run_git(root, "add", ".")
    run_git(root, "commit", "-m", "test fixture")
    return root, run_git(root, "rev-parse", "HEAD")


def expected_spaceship_relative_choice(common: int, final_state: int) -> int:
    return final_state + 1 if common else 2 - final_state


def canonical_history(
    *,
    first_stage_action: str = "action_0",
    transition_common: bool = True,
    final_state: str = "state_0",
    second_stage_action: str = "action_0",
    reward: int = 1,
) -> dict[str, object]:
    return {
        "first_stage_action": first_stage_action,
        "transition_common": transition_common,
        "final_state": final_state,
        "second_stage_action": second_stage_action,
        "reward": reward,
    }
