#!/usr/bin/env python3
"""
Vanilla RDF2Vec on the DBpedia subgraph you built:

  * ``processed/G.ttl`` - 2-hop instance graph (+ types), N-Triples-style lines
  * ``schema/schema.ttl`` - filtered ontology (class / property axioms)

The script **concatenates** them into one file (streaming, no full-graph load),
runs ``src/walk/random_walks.py``, then ``src/train/train_word2vec.py`` in
``--mode none`` (single corpus), matching the usual jRDF2Vec duplicate-free
pipeline from ``scripts/e2e_no_pretrain.sh``.

Requires ``uv`` on PATH (same as other repo scripts).

Example::

  uv run scripts/train_rdf2vec_dbpedia_subgraph.py

  uv run scripts/train_rdf2vec_dbpedia_subgraph.py \\
    --instance processed/G.ttl \\
    --schema schema/schema.ttl \\
    --out-dir output/my_dbpedia_rdf2vec/run01
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def uv_bin() -> str:
    u = shutil.which("uv")
    if not u:
        raise SystemExit(
            "Could not find 'uv' on PATH. Install uv or run from the same environment as the e2e scripts."
        )
    return u


def uv_run_python(script: Path) -> list[str]:
    return [uv_bin(), "run", "python", str(script)]


def combine_graphs_binary(sources: list[Path], dest: Path) -> None:
    """Append each source to dest in order (binary copy; preserves line endings)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for src in sources:
            if not src.is_file():
                raise SystemExit(f"Missing graph file: {src}")
            with src.open("rb") as f:
                shutil.copyfileobj(f, out)


def main() -> None:
    root = repo_root()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--instance",
        type=Path,
        default=root / "processed" / "G.ttl",
        help="Instance (+ types) graph (default: processed/G.ttl)",
    )
    ap.add_argument(
        "--schema",
        type=Path,
        default=root / "schema" / "schema.ttl",
        help="Filtered ontology graph (default: schema/schema.ttl)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Run directory (default: output/dbpedia_vanilla_rdf2vec/<UTC timestamp>)",
    )
    ap.add_argument(
        "--combined-name",
        type=str,
        default="combined_instance_schema.nt",
        help="Filename under out-dir for the merged graph (default: combined_instance_schema.nt)",
    )
    ap.add_argument(
        "--schema-first",
        action="store_true",
        help="Concatenate schema before instance (default: instance then schema)",
    )
    ap.add_argument(
        "--keep-combined",
        action="store_true",
        help="Keep the merged .nt file after walks (default: delete to save disk)",
    )
    ap.add_argument(
        "--walk-mode",
        choices=("jrdf2vec-duplicate-free", "classic"),
        default="jrdf2vec-duplicate-free",
        help="Walk generator (default: jrdf2vec-duplicate-free; classic needs strict N-Triples)",
    )
    ap.add_argument("--walks-per-entity", type=int, default=20)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--walk-threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--num-walks", type=int, default=10_000, help="(classic only) total walks")
    ap.add_argument("--walk-length", type=int, default=40, help="(classic only) nodes per walk")
    ap.add_argument(
        "--directed",
        action="store_true",
        help="(classic only) directed entity graph",
    )
    ap.add_argument("--walk-seed", type=int, default=None, help="RNG seed for walks")
    ap.add_argument(
        "--architecture",
        choices=("skipgram", "cbow"),
        default="skipgram",
    )
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0, help="Gensim workers (0 = all cores)")
    ap.add_argument("--train-seed", type=int, default=None, help="RNG seed for Word2Vec")
    ap.add_argument("--loss-every-steps", type=int, default=100)
    ap.add_argument(
        "--eval-test",
        type=Path,
        default=None,
        help="Optional labeled test.txt for evaluate_embeddings.py after training",
    )
    ap.add_argument(
        "--train-for-eval",
        type=Path,
        default=None,
        help="Train split for eval (default: train.txt next to --eval-test)",
    )
    args = ap.parse_args()

    instance = args.instance if args.instance.is_absolute() else root / args.instance
    schema = args.schema if args.schema.is_absolute() else root / args.schema

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = root / "output" / "dbpedia_vanilla_rdf2vec" / stamp
    else:
        out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = out_dir / args.combined_name
    walks_path = out_dir / "walks.txt"
    checkpoint = out_dir / "rdf2vec_word2vec.pt"

    order = [schema, instance] if args.schema_first else [instance, schema]
    print(f"Merging graphs → {combined}", file=sys.stderr)
    combine_graphs_binary(order, combined)

    walk_argv = uv_run_python(root / "src" / "walk" / "random_walks.py") + [
        str(combined),
        str(walks_path),
        "--mode",
        args.walk_mode,
        "--token-format",
        "angled",
    ]
    if args.walk_seed is not None:
        walk_argv.extend(["--seed", str(args.walk_seed)])

    if args.walk_mode == "jrdf2vec-duplicate-free":
        walk_argv += [
            "--walks-per-entity",
            str(args.walks_per_entity),
            "--depth",
            str(args.depth),
            "--threads",
            str(args.walk_threads),
        ]
    else:
        walk_argv += [
            "--num-walks",
            str(args.num_walks),
            "--walk-length",
            str(args.walk_length),
        ]
        if args.directed:
            walk_argv.append("--directed")

    print("Running random walks…", file=sys.stderr)
    subprocess.run(walk_argv, cwd=str(root), check=True)

    if not args.keep_combined:
        try:
            combined.unlink()
            print(f"Removed merged graph {combined}", file=sys.stderr)
        except OSError as e:
            print(f"Warning: could not remove {combined}: {e}", file=sys.stderr)

    train_argv = uv_run_python(root / "src" / "train" / "train_word2vec.py") + [
        str(walks_path),
        "--mode",
        "none",
        "-o",
        str(checkpoint),
        "--architecture",
        args.architecture,
        "--dim",
        str(args.dim),
        "--window",
        str(args.window),
        "--epochs",
        str(args.epochs),
        "--negative",
        str(args.negative),
        "--workers",
        str(args.workers),
        "--loss-every-steps",
        str(args.loss_every_steps),
    ]
    if args.train_seed is not None:
        train_argv.extend(["--seed", str(args.train_seed)])

    print("Training Word2Vec…", file=sys.stderr)
    subprocess.run(train_argv, cwd=str(root), check=True)

    if args.eval_test is not None:
        test_p = args.eval_test if args.eval_test.is_absolute() else root / args.eval_test
        eval_argv = uv_run_python(root / "src" / "evaluate_embeddings.py") + [
            str(test_p),
            "-c",
            str(checkpoint),
        ]
        if args.train_for_eval is not None:
            tr = args.train_for_eval if args.train_for_eval.is_absolute() else root / args.train_for_eval
            eval_argv.extend(["--train", str(tr)])
        log_path = out_dir / "eval_metrics.txt"
        print(f"Evaluating → {log_path}", file=sys.stderr)
        with log_path.open("w", encoding="utf-8") as lf:
            subprocess.run(eval_argv, cwd=str(root), check=True, stdout=lf, stderr=subprocess.STDOUT)

    print(f"Done.\n  walks:      {walks_path}\n  checkpoint: {checkpoint}", file=sys.stderr)


if __name__ == "__main__":
    main()
