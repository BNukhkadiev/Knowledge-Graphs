#!/usr/bin/env python3
"""
Run RDF2Vec e2e for DLCC TCs with conf/rdf2vec_fixed.yaml (or another YAML).

For each test case, instance random walks are generated exactly once and reused for
no_pretrain, P1, and P2 so those three runs share the same instance walks file.

Usage:
  uv run python scripts/run_rdf2vec_fixed_e2e.py --config conf/rdf2vec_fixed.yaml
  uv run python scripts/run_rdf2vec_fixed_e2e.py --config conf/rdf2vec_fixed.yaml --tc tc01 tc07
  uv run python scripts/run_rdf2vec_fixed_e2e.py --config conf/rdf2vec_fixed.yaml --force-walks
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


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def _load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw)


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
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    return cmd


def _w2v_none_argv(cfg: dict[str, Any], walks: Path, checkpoint: Path) -> list[str]:
    w = cfg["word2vec"]
    epochs = int(w["epochs"])
    loss_steps = int(w.get("loss_every_steps") or 0)
    cmd: list[str] = [
        "uv",
        "run",
        "src/train/train_word2vec.py",
        str(walks),
        "--mode",
        "none",
        "--architecture",
        str(w["architecture"]),
        "--dim",
        str(int(w["dim"])),
        "--window",
        str(int(w["window"])),
        "--epochs",
        str(epochs),
        "--lr",
        str(float(w["lr"])),
        "--min-alpha",
        str(float(w["min_alpha"])),
        "--min-count",
        str(int(w["min_count"])),
        "--negative",
        str(int(w["negative"])),
        "-o",
        str(checkpoint),
    ]
    rs = w.get("random_seed")
    if rs is not None:
        cmd.extend(["--seed", str(int(rs))])
    if loss_steps > 0:
        cmd.extend(["--loss-every-steps", str(loss_steps)])
    return cmd


def _w2v_p1_p2_argv(
    cfg: dict[str, Any],
    mode: str,
    pretrain_walks: Path,
    instance_walks: Path,
    ontology_nt: Path,
    checkpoint: Path,
    out_dir: Path,
) -> list[str]:
    assert mode in ("p1", "p2")
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
        mode,
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


def _eval_argv(test_txt: Path, checkpoint: Path) -> list[str]:
    return [
        "uv",
        "run",
        "src/evaluate_embeddings.py",
        str(test_txt),
        "-c",
        str(checkpoint),
    ]


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

    ``subprocess.run(..., stdout=PIPE)`` only reads when the process exits; enough
    trainer output can fill the pipe and deadlock the child.
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
        description="E2E RDF2Vec (no_pretrain / P1 / P2) with shared instance walks per TC."
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
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "rdf2vec_fixed",
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
    fp = _config_fingerprint(cfg)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.output_root.resolve() / f"run_{run_id}_cfg{fp}"
    if not args.dry_run:
        batch_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "config": str(cfg_path),
        "config_fingerprint": fp,
        "run_id": run_id,
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
        tc_entry: dict[str, Any] = {"tc": tc, "shared_instance_walks": str(shared_walks), "modes": {}}

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

        # --- no_pretrain ---
        out_v = batch_root / tc / "no_pretrain"
        out_v.mkdir(parents=True, exist_ok=True)
        run_log = out_v / "run.log"
        ck_v = out_v / "rdf2vec_word2vec.pt"
        if args.dry_run:
            print("+", " ".join(_w2v_none_argv(cfg, shared_walks, ck_v)))
            print("+", " ".join(_eval_argv(test_txt, ck_v)))
        else:
            with open(run_log, "w", encoding="utf-8") as log:
                _run_capture_log(_w2v_none_argv(cfg, shared_walks, ck_v), log)
                _run_capture_log(_eval_argv(test_txt, ck_v), log)
        tc_entry["modes"]["no_pretrain"] = {"dir": str(out_v), "checkpoint": str(ck_v)}

        # --- P1 ---
        out_p1 = batch_root / tc / "p1"
        out_p1.mkdir(parents=True, exist_ok=True)
        prot_p1 = out_p1 / "protograph_p1.nt"
        walks_p1 = out_p1 / "walks_p1.txt"
        ck_p1 = out_p1 / "rdf2vec_final.pt"
        if args.dry_run:
            print("+", " ".join(_protograph_argv(ontology_nt, graph_nt, out_p1)))
            print("+", " ".join(_walks_argv(cfg, prot_p1, walks_p1)))
            print("+", " ".join(_w2v_p1_p2_argv(cfg, "p1", walks_p1, shared_walks, ontology_nt, ck_p1, out_p1)))
            print("+", " ".join(_eval_argv(test_txt, ck_p1)))
        else:
            log_p1 = out_p1 / "run.log"
            with open(log_p1, "w", encoding="utf-8") as log:
                _run_capture_log(_protograph_argv(ontology_nt, graph_nt, out_p1), log)
                if not prot_p1.is_file():
                    raise SystemExit(f"[{tc}] protograph did not write {prot_p1}")
                _run_capture_log(_walks_argv(cfg, prot_p1, walks_p1), log)
                _run_capture_log(
                    _w2v_p1_p2_argv(cfg, "p1", walks_p1, shared_walks, ontology_nt, ck_p1, out_p1),
                    log,
                )
                _run_capture_log(_eval_argv(test_txt, ck_p1), log)
        tc_entry["modes"]["p1"] = {"dir": str(out_p1), "checkpoint": str(ck_p1)}

        # --- P2 ---
        out_p2 = batch_root / tc / "p2"
        out_p2.mkdir(parents=True, exist_ok=True)
        prot_p2 = out_p2 / "protograph_p2.nt"
        walks_p2 = out_p2 / "walks_p2.txt"
        ck_p2 = out_p2 / "rdf2vec_final.pt"
        if args.dry_run:
            print("+", " ".join(_protograph_argv(ontology_nt, graph_nt, out_p2)))
            print("+", " ".join(_walks_argv(cfg, prot_p2, walks_p2)))
            print("+", " ".join(_w2v_p1_p2_argv(cfg, "p2", walks_p2, shared_walks, ontology_nt, ck_p2, out_p2)))
            print("+", " ".join(_eval_argv(test_txt, ck_p2)))
        else:
            log_p2 = out_p2 / "run.log"
            with open(log_p2, "w", encoding="utf-8") as log:
                _run_capture_log(_protograph_argv(ontology_nt, graph_nt, out_p2), log)
                if not prot_p2.is_file():
                    raise SystemExit(f"[{tc}] protograph did not write {prot_p2}")
                _run_capture_log(_walks_argv(cfg, prot_p2, walks_p2), log)
                _run_capture_log(
                    _w2v_p1_p2_argv(cfg, "p2", walks_p2, shared_walks, ontology_nt, ck_p2, out_p2),
                    log,
                )
                _run_capture_log(_eval_argv(test_txt, ck_p2), log)
        tc_entry["modes"]["p2"] = {"dir": str(out_p2), "checkpoint": str(ck_p2)}

        manifest["test_cases"].append(tc_entry)

    manifest_path = batch_root / "manifest.json"
    if not args.dry_run:
        batch_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote manifest: {manifest_path}", flush=True)
    print(f"Batch root: {batch_root}", flush=True)


if __name__ == "__main__":
    main()
