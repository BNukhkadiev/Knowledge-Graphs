#!/usr/bin/env python3
"""
E2E RDF2Vec for DLCC TCs: P2 protograph only, comparing MASCHInE class-init strategies.

For each test case:
  * shared instance walks (cached by config fingerprint, same as run_rdf2vec_fixed_e2e.py)
  * one protograph build + P2 walks
  * for each MASCHInE init strategy: two-stage train + eval

Strategy names (aliases map to train_word2vec --maschine-strategy):
  * most_specific
  * ancestor_mean | average_parent_hierarchy
  * ancestor_weighted | weighted_ancestors

Usage:
  uv run python scripts/run_rdf2vec_p2_maschine_e2e.py --config conf/rdf2vec_fixed.yaml
  uv run python scripts/run_rdf2vec_p2_maschine_e2e.py --tc tc07 tc08 --strategies most_specific ancestor_weighted
  uv run python scripts/run_rdf2vec_p2_all_tc_init_strategies.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Internal names passed to train_word2vec --maschine-strategy
MASCHINE_STRATEGIES_INTERNAL = ("most_specific", "ancestor_mean", "ancestor_weighted")

# CLI may use design-doc / paper-style aliases
MASCHINE_STRATEGY_ALIASES: dict[str, str] = {
    "average_parent_hierarchy": "ancestor_mean",
    "weighted_ancestors": "ancestor_weighted",
}

MASCHINE_STRATEGY_CHOICES = tuple(
    dict.fromkeys((*MASCHINE_STRATEGIES_INTERNAL, *MASCHINE_STRATEGY_ALIASES.keys()))
)


def _normalize_maschine_strategy(name: str) -> str:
    n = name.strip()
    return MASCHINE_STRATEGY_ALIASES.get(n, n)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ensure_walk_random_seed(cfg: dict[str, Any]) -> int:
    """
    Guarantee ``walks.random_seed`` is set so ``random_walks.py`` always gets ``--seed``.

    If the YAML omits it, reuse ``word2vec.random_seed`` when present, else 42.
    """
    walks = cfg.setdefault("walks", {})
    if walks.get("random_seed") is None:
        w2v = cfg.get("word2vec") or {}
        fallback = w2v.get("random_seed")
        walks["random_seed"] = 42 if fallback is None else int(fallback)
    seed = int(walks["random_seed"])
    walks["random_seed"] = seed
    return seed


def _walks_argv(cfg: dict[str, Any], graph_nt: Path, out_walks: Path) -> list[str]:
    w = cfg["walks"]
    cmd: list[str] = [
        "uv",
        "run",
        "src/walk/random_walks.py",
        str(graph_nt),
        str(out_walks),
        "--mode",
        str(w["mode"]),
        "--depth",
        str(int(w["depth"])),
        "--walks-per-entity",
        str(int(w["walks_per_entity"])),
        "--token-format",
        str(w["token_format"]),
    ]
    seed = w.get("random_seed")
    if seed is None:
        raise SystemExit("Internal error: walks.random_seed must be set (call _ensure_walk_random_seed first).")
    cmd.extend(["--seed", str(int(seed))])
    return cmd


def _w2v_p2_maschine_argv(
    cfg: dict[str, Any],
    pretrain_walks: Path,
    instance_walks: Path,
    ontology_nt: Path,
    checkpoint: Path,
    out_dir: Path,
    *,
    maschine_strategy: str,
    maschine_ancestor_decay: float,
) -> list[str]:
    w = cfg["word2vec"]
    pre_e = w.get("pretrain_epochs")
    ft_e = w.get("finetune_epochs")
    ep = int(w["epochs"])
    pretrain_epochs = int(pre_e if pre_e is not None else ep)
    finetune_epochs = int(ft_e if ft_e is not None else ep)
    l_pre = int(w.get("loss_every_steps_pretrain") or 1)
    l_ft = int(w.get("loss_every_steps_finetune") or 0)
    l_both = int(w.get("loss_every_steps") or 0)

    cmd: list[str] = [
        "uv",
        "run",
        "src/train/train_word2vec.py",
        "--mode",
        "p2",
        "--pretrain-walks",
        str(pretrain_walks),
        "--instance-walks",
        str(instance_walks),
        "--ontology",
        str(ontology_nt),
        "--out-dir",
        str(out_dir),
        "--architecture",
        str(w["architecture"]),
        "--dim",
        str(int(w["dim"])),
        "--window",
        str(int(w["window"])),
        "--lr",
        str(float(w["lr"])),
        "--min-alpha",
        str(float(w["min_alpha"])),
        "--min-count",
        str(int(w["min_count"])),
        "--negative",
        str(int(w["negative"])),
        "--pretrain-epochs",
        str(pretrain_epochs),
        "--finetune-epochs",
        str(finetune_epochs),
        "--maschine-strategy",
        maschine_strategy,
        "--maschine-ancestor-decay",
        str(maschine_ancestor_decay),
        "-o",
        str(checkpoint),
    ]
    rs = w.get("random_seed")
    if rs is not None:
        cmd.extend(["--seed", str(int(rs))])
    if l_both > 0:
        cmd.extend(["--loss-every-steps", str(l_both)])
    else:
        cmd.extend(
            [
                "--loss-every-steps-pretrain",
                str(l_pre),
                "--loss-every-steps-finetune",
                str(l_ft),
            ]
        )
    return cmd


def _eval_argv(test_txt: Path, checkpoint: Path, *, label: str | None = None) -> list[str]:
    cmd: list[str] = [
        "uv",
        "run",
        "src/evaluate_embeddings.py",
        str(test_txt),
        "-c",
        str(checkpoint),
    ]
    if label:
        cmd.extend(["--label", label])
    return cmd


def _protograph_argv(ontology_nt: Path, graph_nt: Path, out_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "protograph",
        "--schema",
        str(ontology_nt),
        "--kg",
        str(graph_nt),
        "--out-dir",
        str(out_dir),
    ]


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _run_capture_log(cmd: list[str], log: TextIO) -> None:
    """
    Stream child stdout to the log file and the terminal.

    ``subprocess.run(..., stdout=PIPE)`` buffers the full stream until the process
    exits; verbose trainers can fill the pipe and block forever (parent never reads).
    """
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            print(chunk, end="", flush=True)
    finally:
        proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    p = argparse.ArgumentParser(
        description="E2E P2 RDF2Vec with shared instance walks; one run per MASCHInE strategy."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "conf" / "rdf2vec_fixed.yaml",
        help="YAML with walks.* and word2vec.*",
    )
    p.add_argument(
        "--tc",
        nargs="+",
        default=[f"tc{i:02d}" for i in range(1, 13)],
        metavar="TC",
        help="Test case ids (default: tc01 … tc12)",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=list(MASCHINE_STRATEGIES_INTERNAL),
        choices=MASCHINE_STRATEGY_CHOICES,
        help="MASCHInE class-vector strategies to run (default: all three internal names). "
        "Aliases: average_parent_hierarchy → ancestor_mean, weighted_ancestors → ancestor_weighted.",
    )
    p.add_argument(
        "--maschine-ancestor-decay",
        type=float,
        default=None,
        metavar="R",
        help="Passed to train_word2vec for ancestor_weighted (default: from YAML word2vec."
        "maschine_ancestor_decay, else 0.5).",
    )
    p.add_argument(
        "--walk-random-seed",
        type=int,
        default=None,
        metavar="S",
        help="RNG seed for instance and P2 protograph walk generation (default: YAML walks.random_seed, "
        "else word2vec.random_seed, else 42). Always passed to random_walks.py as --seed.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "rdf2vec_p2_maschine",
        help="Root under repo for runs and cached walks",
    )
    p.add_argument(
        "--force-walks",
        action="store_true",
        help="Regenerate shared instance walks even if the file exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    args = p.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)
    walk_seed = _ensure_walk_random_seed(cfg)
    if args.walk_random_seed is not None:
        walk_seed = int(args.walk_random_seed)
        cfg.setdefault("walks", {})["random_seed"] = walk_seed
    fp = _config_fingerprint(cfg)

    wcfg = cfg.get("word2vec") or {}
    decay_default = float(wcfg.get("maschine_ancestor_decay", 0.5))
    maschine_decay = float(args.maschine_ancestor_decay) if args.maschine_ancestor_decay is not None else decay_default
    needs_decay = any(_normalize_maschine_strategy(s) == "ancestor_weighted" for s in args.strategies)
    if needs_decay and not (0.0 < maschine_decay <= 1.0):
        raise SystemExit("--maschine-ancestor-decay (or YAML maschine_ancestor_decay) must satisfy 0 < R <= 1.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.output_root.resolve() / f"run_{run_id}_cfg{fp}"
    if not args.dry_run:
        batch_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "config": str(cfg_path),
        "config_fingerprint": fp,
        "run_id": run_id,
        "strategies": args.strategies,
        "strategies_resolved": {
            s: _normalize_maschine_strategy(s) for s in args.strategies
        },
        "maschine_ancestor_decay": maschine_decay,
        "walk_random_seed": walk_seed,
        "repo_root": str(REPO_ROOT),
        "test_cases": [],
    }

    for tc in args.tc:
        tc = tc.strip().lower()
        base = REPO_ROOT / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
        graph_nt = base / "graph.nt"
        ontology_nt = base / "ontology.nt"
        test_txt = base / "1000" / "train_test" / "test.txt"
        for path, label in (
            (graph_nt, "graph.nt"),
            (ontology_nt, "ontology.nt"),
            (test_txt, "test.txt"),
        ):
            if not path.is_file():
                raise SystemExit(f"[{tc}] missing {label}: {path}")

        walks_dir = args.output_root.resolve() / "instance_walks" / fp / tc
        walks_dir.mkdir(parents=True, exist_ok=True)
        shared_walks = walks_dir / "instance_walks.txt"
        shared_p2 = batch_root / tc / "p2_shared"
        prot_p2 = shared_p2 / "protograph_p2.nt"
        walks_p2 = shared_p2 / "walks_p2.txt"

        tc_entry: dict[str, Any] = {
            "tc": tc,
            "shared_instance_walks": str(shared_walks),
            "p2_shared": str(shared_p2),
            "modes": {},
        }

        if args.force_walks and shared_walks.is_file():
            shared_walks.unlink()

        if not shared_walks.is_file():
            print(f"\n=== [{tc}] generate shared instance walks → {shared_walks}", flush=True)
            cmd = _walks_argv(cfg, graph_nt, shared_walks)
            if args.dry_run:
                print("+", " ".join(cmd))
            else:
                _run(cmd)
        else:
            print(f"\n=== [{tc}] reuse shared instance walks: {shared_walks}", flush=True)

        if args.dry_run:
            print("+", " ".join(_protograph_argv(ontology_nt, graph_nt, shared_p2)))
            print("+", " ".join(_walks_argv(cfg, prot_p2, walks_p2)))
        else:
            shared_p2.mkdir(parents=True, exist_ok=True)
            log_shared = shared_p2 / "build_p2.log"
            with open(log_shared, "w", encoding="utf-8") as log:
                _run_capture_log(_protograph_argv(ontology_nt, graph_nt, shared_p2), log)
            if not prot_p2.is_file():
                raise SystemExit(f"[{tc}] protograph did not write {prot_p2}")
            with open(log_shared, "a", encoding="utf-8") as log:
                _run_capture_log(_walks_argv(cfg, prot_p2, walks_p2), log)

        for strategy_cli in args.strategies:
            strategy_internal = _normalize_maschine_strategy(strategy_cli)
            out_tag = f"p2_{strategy_cli}"
            out_dir = batch_root / tc / out_tag
            ck = out_dir / "rdf2vec_final.pt"
            eval_label = (
                f"{tc} | {out_tag} | --maschine-strategy {strategy_internal} | checkpoint {ck.name}"
            )
            tc_entry["modes"][out_tag] = {
                "dir": str(out_dir),
                "checkpoint": str(ck),
                "strategy": strategy_cli,
                "strategy_train_word2vec": strategy_internal,
            }

            if args.dry_run:
                print("+", " ".join(
                    _w2v_p2_maschine_argv(
                        cfg,
                        walks_p2,
                        shared_walks,
                        ontology_nt,
                        ck,
                        out_dir,
                        maschine_strategy=strategy_internal,
                        maschine_ancestor_decay=maschine_decay,
                    )
                ))
                print("+", " ".join(_eval_argv(test_txt, ck, label=eval_label)))
            else:
                print(
                    f"\n>>> [{tc}] {out_tag}  (train --maschine-strategy {strategy_internal})",
                    flush=True,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                run_log = out_dir / "run.log"
                with open(run_log, "w", encoding="utf-8") as log:
                    _run_capture_log(
                        _w2v_p2_maschine_argv(
                            cfg,
                            walks_p2,
                            shared_walks,
                            ontology_nt,
                            ck,
                            out_dir,
                            maschine_strategy=strategy_internal,
                            maschine_ancestor_decay=maschine_decay,
                        ),
                        log,
                    )
                    _run_capture_log(_eval_argv(test_txt, ck, label=eval_label), log)

        manifest["test_cases"].append(tc_entry)

    manifest_path = batch_root / "manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote manifest: {manifest_path}", flush=True)
    print(f"Batch root: {batch_root}", flush=True)


if __name__ == "__main__":
    main()
