#!/usr/bin/env python3
"""
Run scripts/run_grid_search.py once per (tc, training_mode) pair with the same forwarded flags.

Pairs are aligned by position: the first --tc matches the first --training-mode, and so on.
Any argument not consumed by this wrapper is passed through to run_grid_search.py (e.g. --limit,
--random-search, --jobs, --dry-run, --output-root, --seed, …).

Examples:
  uv run scripts/run_grid_search_batch.py conf/grid_search.yaml \\
    --tc tc12 tc13 --training-mode p2 p1 --random-search --limit 20

  uv run scripts/run_grid_search_batch.py conf/grid_search.yaml \\
    --tc tc12 --training-mode no_pretrain --limit 50 --jobs 2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def uv_bin() -> str:
    u = shutil.which("uv")
    if not u:
        raise SystemExit(
            "Could not find 'uv' on PATH. Install uv or run from the same environment as the e2e scripts."
        )
    return u


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=(
            "Batch driver: run run_grid_search.py for each (tc, training_mode) pair with the same "
            "extra arguments (grid search, random search, limits, parallelism, etc.)."
        ),
    )
    ap.add_argument(
        "config",
        type=Path,
        help="Grid YAML (e.g. conf/grid_search.yaml), same as run_grid_search.py",
    )
    ap.add_argument(
        "--tc",
        nargs="+",
        required=True,
        metavar="TC",
        help="One or more test case ids; length must match --training-mode",
    )
    ap.add_argument(
        "--training-mode",
        nargs="+",
        required=True,
        metavar="MODE",
        dest="training_modes",
        choices=("no_pretrain", "p1", "p2"),
        help="Training recipe per pair (same choices as run_grid_search.py); length must match --tc",
    )
    args, forward = ap.parse_known_args(argv)

    if len(args.tc) != len(args.training_modes):
        raise SystemExit(
            f"--tc ({len(args.tc)} values) and --training-mode ({len(args.training_modes)} values) "
            "must have the same length (aligned pairs)."
        )

    root = repo_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    grid_script = root / "scripts" / "run_grid_search.py"
    if not grid_script.is_file():
        raise SystemExit(f"Missing driver script: {grid_script}")

    cfg_arg = str(args.config) if args.config.is_absolute() else str(cfg_path)

    def run_one(tc: str, tm: str) -> None:
        cmd = [
            uv_bin(),
            "run",
            "python",
            str(grid_script),
            cfg_arg,
            "--tc",
            tc,
            "--training-mode",
            tm,
            *forward,
        ]
        subprocess.run(cmd, cwd=root, check=True)

    n = len(args.tc)
    for i, (tc, tm) in enumerate(zip(args.tc, args.training_modes, strict=True), start=1):
        print(f"[batch {i}/{n}] --tc {tc} --training-mode {tm}", file=sys.stderr)
        run_one(tc, tm)


if __name__ == "__main__":
    main()
