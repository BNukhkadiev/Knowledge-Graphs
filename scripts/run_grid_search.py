#!/usr/bin/env python3
"""
Run a hyperparameter grid from conf/grid_search.yaml for a synthetic-ontology test case (TC).

Walk corpora are written to a temporary directory and deleted after each run (not kept under
output/). Protographs, checkpoints, loss plots/CSVs, eval_metrics.txt (stdout from
evaluate_embeddings.py), and params.json remain in each run_* folder. With evaluation enabled,
a tqdm bar shows the last run's test accuracy and the best accuracy so far (and the run index
of the best).

Example:
  uv run scripts/run_grid_search.py conf/grid_search.yaml --tc tc12
  uv run scripts/run_grid_search.py conf/grid_search.yaml --tc tc12 --dry-run
  uv run scripts/run_grid_search.py conf/grid_search.yaml --tc tc12 --limit 3
  uv run scripts/run_grid_search.py conf/grid_search.yaml --tc tc12 --random-search --limit 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

try:
    import yaml
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyYAML is required. Install project deps: uv sync"
    ) from e


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _sample_from_lists(lists: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    """One independent random draw per key (uniform over each list)."""
    out: dict[str, Any] = {}
    for k, vals in lists.items():
        choices = as_list(vals)
        if not choices:
            raise SystemExit(f"Random search: empty list for key {k!r}")
        out[k] = rng.choice(choices)
    return out


def sample_random_run(cfg: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    """Sample a single configuration: one training_mode, then one value per hyperparameter list."""
    modes = as_list(cfg.get("training_mode"))
    if not modes:
        raise SystemExit("training_mode is empty in config")
    tm = str(rng.choice(modes)).strip()

    pre = cfg.get("pretrain") or {}
    fin = cfg.get("finetune") or {}

    pre_w = normalize_walk_block(pre.get("walks"))
    pre_w2v = normalize_w2v_block(pre.get("word2vec"))
    fin_w = normalize_walk_block(fin.get("walks"))
    fin_w2v = normalize_w2v_block(fin.get("word2vec"))

    if tm == "no_pretrain":
        w2v_lists = {k: v for k, v in fin_w2v.items() if k != "finetune_epochs"}
        if not w2v_lists.get("epochs"):
            raise SystemExit(
                "finetune.word2vec.epochs is required when training_mode may be no_pretrain"
            )
        flat = {
            "training_mode": tm,
            "finetune_walks": _sample_from_lists(fin_w, rng),
            "finetune_word2vec": _sample_from_lists(w2v_lists, rng),
        }
        return tm, flat

    if tm in ("P1", "P2"):
        w2v_lists = {k: v for k, v in fin_w2v.items() if k != "epochs"}
        if not w2v_lists.get("finetune_epochs"):
            raise SystemExit(
                "finetune.word2vec.finetune_epochs is required when training_mode may be P1 or P2"
            )
        flat = {
            "training_mode": tm,
            "pretrain_walks": _sample_from_lists(pre_w, rng),
            "pretrain_word2vec": _sample_from_lists(pre_w2v, rng),
            "finetune_walks": _sample_from_lists(fin_w, rng),
            "finetune_word2vec": _sample_from_lists(w2v_lists, rng),
        }
        return tm, flat

    raise SystemExit(f"Unknown training_mode: {tm!r} (expected no_pretrain, P1, P2)")


def dict_product(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Cartesian product over keys; each value must be a list (use as_list)."""
    keys = list(d.keys())
    if not keys:
        yield {}
        return
    lists = [as_list(d[k]) for k in keys]
    for combo in itertools.product(*lists):
        yield dict(zip(keys, combo))


def normalize_walk_block(block: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not block:
        return {}
    out: dict[str, list[Any]] = {}
    for k, v in block.items():
        if k.startswith("#") or v is None:
            continue
        out[k] = as_list(v)
    return out


def normalize_w2v_block(block: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not block:
        return {}
    out: dict[str, list[Any]] = {}
    for k, v in block.items():
        if k.startswith("#") or v is None:
            continue
        out[k] = as_list(v)
    return out


def walk_cli_args(w: dict[str, Any]) -> list[str]:
    m = w.get("mode")
    if m is None:
        raise ValueError("walk params missing 'mode'")
    args = [
        "--mode",
        str(m),
        "--depth",
        str(int(w["depth"])),
        "--walks-per-entity",
        str(int(w["walks_per_entity"])),
    ]
    if "token_format" in w and w["token_format"] is not None:
        args.extend(["--token-format", str(w["token_format"])])
    return args


def train_w2v_common_args(w: dict[str, Any]) -> list[str]:
    return [
        "--architecture",
        str(w["architecture"]),
        "--dim",
        str(int(w["dim"])),
        "--window",
        str(int(w["window"])),
        "--negative",
        str(int(w["negative"])),
        "--lr",
        str(float(w["lr"])),
        "--min-alpha",
        str(float(w["min_alpha"])),
        "--min-count",
        str(int(w["min_count"])),
    ]


def uv_bin() -> str:
    u = shutil.which("uv")
    if not u:
        raise SystemExit(
            "Could not find 'uv' on PATH. Install uv or run from the same environment as the e2e scripts."
        )
    return u


def uv_run_python(script: Path) -> list[str]:
    return [uv_bin(), "run", "python", str(script)]


def run_cmd(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    if dry_run:
        print(" ".join(shlex_quote(a) for a in argv))
        if log_path is not None:
            tqdm.write(f"  # log: {log_path}", file=sys.stderr)
        return
    merged = os.environ.copy()
    if env:
        merged.update(env)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            subprocess.run(
                argv,
                cwd=cwd,
                check=True,
                env=merged,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        subprocess.run(argv, cwd=cwd, check=True, env=merged)


def shlex_quote(s: str) -> str:
    if not s or any(c in s for c in " \t\n\"'"):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def parse_accuracy_from_eval_log(path: Path) -> float | None:
    """Read test accuracy from evaluate_embeddings.py stdout (eval_metrics.txt)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "accuracy":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def tc_paths(tc: str, root: Path) -> dict[str, Path]:
    base = root / "v1" / "synthetic_ontology" / tc / "synthetic_ontology"
    return {
        "base": base,
        "graph_nt": base / "graph.nt",
        "ontology_nt": base / "ontology.nt",
        "test_txt": base / "1000" / "train_test" / "test.txt",
    }


def iter_runs(cfg: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (training_mode, flat_param_dict) for each grid point."""
    modes = as_list(cfg.get("training_mode"))
    pre = cfg.get("pretrain") or {}
    fin = cfg.get("finetune") or {}

    pre_w = normalize_walk_block(pre.get("walks"))
    pre_w2v = normalize_w2v_block(pre.get("word2vec"))
    fin_w = normalize_walk_block(fin.get("walks"))
    fin_w2v = normalize_w2v_block(fin.get("word2vec"))

    for tm in modes:
        tm = str(tm).strip()
        if tm == "no_pretrain":
            w2v = {k: v for k, v in fin_w2v.items() if k != "finetune_epochs"}
            if not w2v.get("epochs"):
                raise SystemExit(
                    "finetune.word2vec.epochs is required when training_mode includes no_pretrain"
                )
            for fw in dict_product(fin_w):
                for fv in dict_product(w2v):
                    flat = {
                        "training_mode": tm,
                        "finetune_walks": fw,
                        "finetune_word2vec": fv,
                    }
                    yield tm, flat
        elif tm in ("P1", "P2"):
            w2v = {k: v for k, v in fin_w2v.items() if k != "epochs"}
            if not w2v.get("finetune_epochs"):
                raise SystemExit(
                    "finetune.word2vec.finetune_epochs is required when training_mode includes P1 or P2"
                )
            for pw in dict_product(pre_w):
                for p2v in dict_product(pre_w2v):
                    for fw in dict_product(fin_w):
                        for fv in dict_product(w2v):
                            flat = {
                                "training_mode": tm,
                                "pretrain_walks": pw,
                                "pretrain_word2vec": p2v,
                                "finetune_walks": fw,
                                "finetune_word2vec": fv,
                            }
                            yield tm, flat
        else:
            raise SystemExit(f"Unknown training_mode: {tm!r} (expected no_pretrain, P1, P2)")


def run_one(
    *,
    run_index: int,
    tm: str,
    flat: dict[str, Any],
    paths: dict[str, Path],
    out_dir: Path,
    dry_run: bool,
    no_eval: bool,
    workers: int | None,
    seed: int | None,
) -> None:
    root = repo_root()
    graph_nt = paths["graph_nt"]
    ontology_nt = paths["ontology_nt"]
    test_txt = paths["test_txt"]

    run_dir = out_dir / f"run_{run_index:04d}"
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "params.json").write_text(
            json.dumps(flat, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    checkpoint = (
        run_dir / "rdf2vec_word2vec.pt"
        if tm == "no_pretrain"
        else run_dir / "rdf2vec_final.pt"
    )

    def _walk_cmd(input_nt: Path, output_walks: Path, wdict: dict[str, Any]) -> None:
        argv = [
            *uv_run_python(root / "src/walk/random_walks.py"),
            str(input_nt),
            str(output_walks),
            *walk_cli_args(wdict),
        ]
        run_cmd(argv, cwd=root, dry_run=dry_run)

    def _train_cmd(extra: list[str], walks_positional: Path | None = None) -> None:
        argv = [*uv_run_python(root / "src/train/train_word2vec.py")]
        if walks_positional is not None:
            argv.append(str(walks_positional))
        argv.extend(extra)
        argv.extend(["-o", str(checkpoint)])
        if workers is not None:
            argv.extend(["--workers", str(workers)])
        if seed is not None:
            argv.extend(["--seed", str(seed)])
        run_cmd(argv, cwd=root, dry_run=dry_run)

    def _run_ephemeral_walks(walk_dir: Path) -> None:
        walks_instance = walk_dir / "walks_instance.txt"
        if tm == "no_pretrain":
            fw = flat["finetune_walks"]
            fv = flat["finetune_word2vec"]
            _walk_cmd(graph_nt, walks_instance, fw)
            _train_cmd(
                [
                    "--mode",
                    "none",
                    "--architecture",
                    str(fv["architecture"]),
                    "--dim",
                    str(int(fv["dim"])),
                    "--window",
                    str(int(fv["window"])),
                    "--negative",
                    str(int(fv["negative"])),
                    "--lr",
                    str(float(fv["lr"])),
                    "--min-alpha",
                    str(float(fv["min_alpha"])),
                    "--min-count",
                    str(int(fv["min_count"])),
                    "--epochs",
                    str(int(fv["epochs"])),
                ],
                walks_positional=walks_instance,
            )
            return

        prot_p1 = run_dir / "protograph_p1.nt"
        prot_p2 = run_dir / "protograph_p2.nt"
        walks_proto = walk_dir / ("walks_p1.txt" if tm == "P1" else "walks_p2.txt")
        pw = flat["pretrain_walks"]
        p2v = flat["pretrain_word2vec"]
        fw = flat["finetune_walks"]
        fv = flat["finetune_word2vec"]

        prot_argv = [
            uv_bin(),
            "run",
            "protograph",
            "--schema",
            str(ontology_nt),
            "--kg",
            str(graph_nt),
            "--out-dir",
            str(run_dir),
        ]
        run_cmd(prot_argv, cwd=root, dry_run=dry_run)

        proto_nt = prot_p1 if tm == "P1" else prot_p2
        _walk_cmd(proto_nt, walks_proto, pw)
        _walk_cmd(graph_nt, walks_instance, fw)

        mode_flag = "p1" if tm == "P1" else "p2"
        train_extra = [
            "--mode",
            mode_flag,
            "--pretrain-walks",
            str(walks_proto),
            "--instance-walks",
            str(walks_instance),
            "--ontology",
            str(ontology_nt),
            "--out-dir",
            str(run_dir),
            *train_w2v_common_args(fv),
            "--pretrain-epochs",
            str(int(p2v["pretrain_epochs"])),
        ]
        if "pretrain_lr" in p2v:
            train_extra.extend(["--pretrain-lr", str(float(p2v["pretrain_lr"]))])
        if "pretrain_min_alpha" in p2v:
            train_extra.extend(["--pretrain-min-alpha", str(float(p2v["pretrain_min_alpha"]))])
        train_extra.extend(
            [
                "--finetune-epochs",
                str(int(fv["finetune_epochs"])),
            ]
        )
        _train_cmd(train_extra, walks_positional=None)

    if dry_run:
        _run_ephemeral_walks(Path(tempfile.gettempdir()) / f"kg_grid_dryrun_run{run_index}")
    else:
        with tempfile.TemporaryDirectory(prefix="kg_grid_walks_") as td:
            _run_ephemeral_walks(Path(td))

    if not no_eval:
        eval_log = run_dir / "eval_metrics.txt"
        eval_argv = [
            *uv_run_python(root / "src/evaluate_embeddings.py"),
            str(test_txt),
            "-c",
            str(checkpoint),
        ]
        run_cmd(eval_argv, cwd=root, dry_run=dry_run, log_path=eval_log)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run grid or random search from YAML for a TC under v1/synthetic_ontology/<TC>/.",
    )
    ap.add_argument(
        "config",
        type=Path,
        help="Grid YAML (e.g. conf/grid_search.yaml)",
    )
    ap.add_argument(
        "--tc",
        required=True,
        help="Test case id (folder v1/synthetic_ontology/<TC>/synthetic_ontology/)",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory under repo root for this sweep (default: output/grid_search/<tc>/<timestamp>/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of running",
    )
    ap.add_argument(
        "--random-search",
        action="store_true",
        help="Sample N independent configs (--limit) instead of full Cartesian grid",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="S",
        help="RNG seed for --random-search sampling only (not train_word2vec --seed)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="With --random-search: number of trials. Without: cap first N grid points after expansion",
    )
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluate_embeddings.py after each train",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Forwarded to train_word2vec.py --workers",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Forwarded to train_word2vec.py --seed",
    )
    args = ap.parse_args()
    root = repo_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    paths = tc_paths(args.tc, root)
    if not paths["graph_nt"].is_file():
        raise SystemExit(f"Missing instance graph: {paths['graph_nt']}")
    if not paths["ontology_nt"].is_file():
        raise SystemExit(f"Missing ontology: {paths['ontology_nt']}")
    if not paths["test_txt"].is_file():
        raise SystemExit(f"Missing test split: {paths['test_txt']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root
    if out_root is None:
        sub = "random_search" if args.random_search else "grid_search"
        out_root = root / "output" / sub / args.tc / stamp
    else:
        out_root = out_root if out_root.is_absolute() else root / out_root

    search_label = "random" if args.random_search else "grid"
    if args.random_search:
        if args.limit is None or args.limit < 1:
            raise SystemExit("--random-search requires --limit N (number of sampled trials, >= 1)")
        rng = random.Random(args.random_seed)
        runs = [sample_random_run(cfg, rng) for _ in range(args.limit)]
    else:
        runs = list(iter_runs(cfg))
        if args.limit is not None:
            runs = runs[: max(0, args.limit)]

    manifest = out_root / "manifest.csv"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "config_copy.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {
            "search": search_label,
            "random_seed": args.random_seed if args.random_search else None,
            "n_runs": len(runs),
        }
        (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "run_index",
        "training_mode",
        "params_json",
        "run_dir",
        "status",
        "search",
        "test_accuracy",
    ]

    def write_manifest_row(
        idx: int,
        tm: str,
        flat: dict[str, Any],
        status: str,
        *,
        test_accuracy: float | None = None,
    ) -> None:
        if args.dry_run:
            return
        row = {
            "run_index": idx,
            "training_mode": tm,
            "params_json": json.dumps(flat, sort_keys=True),
            "run_dir": str(out_root / f"run_{idx:04d}"),
            "status": status,
            "search": search_label,
            "test_accuracy": "" if test_accuracy is None else f"{test_accuracy:.6f}",
        }
        write_header = not manifest.is_file()
        with manifest.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

    label = "Random trials" if args.random_search else "Grid points"
    print(f"{label}: {len(runs)}  →  {out_root}", file=sys.stderr)

    best_acc: float | None = None
    best_run_idx: int | None = None

    pbar = tqdm(
        enumerate(runs, start=1),
        total=len(runs),
        desc="grid search",
        unit="run",
        dynamic_ncols=True,
        file=sys.stderr,
        leave=True,
    )
    for i, (tm, flat) in pbar:
        pbar.set_description_str(f"[{i}/{len(runs)}] {tm}", refresh=False)
        acc: float | None = None
        try:
            run_one(
                run_index=i,
                tm=tm,
                flat=flat,
                paths=paths,
                out_dir=out_root,
                dry_run=args.dry_run,
                no_eval=args.no_eval,
                workers=args.workers,
                seed=args.seed,
            )
            if not args.no_eval and not args.dry_run:
                acc = parse_accuracy_from_eval_log(out_root / f"run_{i:04d}" / "eval_metrics.txt")
                if acc is not None and (best_acc is None or acc > best_acc):
                    best_acc = acc
                    best_run_idx = i
            postfix: dict[str, str] = {}
            if not args.no_eval:
                postfix["last"] = f"{acc:.4f}" if acc is not None else "—"
                postfix["best"] = f"{best_acc:.4f}" if best_acc is not None else "—"
                if best_run_idx is not None:
                    postfix["best@"] = f"run_{best_run_idx:04d}"
            else:
                postfix["eval"] = "off"
            pbar.set_postfix(postfix, refresh=True)
            write_manifest_row(i, tm, flat, "ok", test_accuracy=acc)
        except subprocess.CalledProcessError as e:
            write_manifest_row(i, tm, flat, f"error: exit {e.returncode}")
            raise SystemExit(e.returncode) from e
        except Exception as e:
            write_manifest_row(i, tm, flat, f"error: {e!s}")
            raise


if __name__ == "__main__":
    main()
