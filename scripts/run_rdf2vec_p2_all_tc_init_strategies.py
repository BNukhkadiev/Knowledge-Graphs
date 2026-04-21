#!/usr/bin/env python3
"""
Run RDF2Vec with P2 protograph pretraining for every DLCC test case (tc01–tc12 by default),
using three MASCHInE instance-initialization strategies (design-doc names):

  * most_specific
  * average_parent_hierarchy  (maps to train_word2vec: ancestor_mean)
  * weighted_ancestors       (maps to train_word2vec: ancestor_weighted)

This delegates to scripts/run_rdf2vec_p2_maschine_e2e.py; extra CLI args are forwarded.

Walk generation always receives an explicit ``--seed`` (from YAML ``walks.random_seed``, else
``word2vec.random_seed``, else 42). Override with ``--walk-random-seed S`` on the e2e script.

Examples:
  uv run python scripts/run_rdf2vec_p2_all_tc_init_strategies.py
  uv run python scripts/run_rdf2vec_p2_all_tc_init_strategies.py --config conf/rdf2vec_fixed.yaml
  uv run python scripts/run_rdf2vec_p2_all_tc_init_strategies.py --tc tc07 --walk-random-seed 42 --dry-run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STRATEGIES = (
    "most_specific",
    "average_parent_hierarchy",
    "weighted_ancestors",
)


def main() -> None:
    e2e = REPO_ROOT / "scripts" / "run_rdf2vec_p2_maschine_e2e.py"
    cmd = [
        sys.executable,
        str(e2e),
        "--strategies",
        *DEFAULT_STRATEGIES,
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
