#!/usr/bin/env python3
"""
Evaluate embeddings on a labeled entity file (node classification).

Checkpoint: PyTorch ``.pt`` from ``train_word2vec.py`` (``embeddings`` + ``word2idx``);
gensim ``KeyedVectors`` ``.kv``; or gensim ``Word2Vec`` ``.model`` (e.g. two-stage
``rdf2vec_pretrained.model``). RDF2Vec stage-2 output is ``rdf2vec_final.pt``.

Each line: <entity_id>\\t<label> (tab-separated). Trains a logistic regression on
train.txt embeddings and reports test metrics with bootstrap standard errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm


def load_labeled_txt(path: Path) -> tuple[list[str], np.ndarray]:
    tokens: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            tokens.append(parts[0].strip())
            labels.append(int(parts[1].strip()))
    if not tokens:
        raise ValueError(f"No labeled rows in {path}")
    return tokens, np.asarray(labels, dtype=np.int64)


def tokens_to_embeddings(
    tokens: list[str],
    emb: np.ndarray,
    word2idx: dict[str, int],
    desc: str,
    chunk_size: int,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, int]:
    """Map tokens to embedding rows; OOV tokens get a zero vector. Returns (matrix, n_oov)."""
    def _candidates(t: str) -> tuple[str, ...]:
        t = t.strip()
        if not t:
            return ("",)
        if (t.startswith("<") and t.endswith(">")) and len(t) > 2:
            inner = t[1:-1]
            return (t, inner)
        # Common for RDF2Vec walks: URIs are stored as "<...>" tokens.
        return (t, f"<{t}>")

    n = len(tokens)
    d = emb.shape[1]
    out = np.zeros((n, d), dtype=np.float32)
    oov = 0
    starts = range(0, n, chunk_size)
    for start in tqdm(
        starts,
        desc=desc,
        dynamic_ncols=True,
        colour="green",
        leave=False,
        unit="chunk",
        disable=not progress,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        end = min(start + chunk_size, n)
        for i, t in enumerate(tokens[start:end]):
            j = None
            for cand in _candidates(t):
                j = word2idx.get(cand)
                if j is not None:
                    break
            if j is None:
                oov += 1
                continue
            out[start + i] = emb[j]
    return out, oov


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    seed: int,
    *,
    progress: bool = True,
) -> dict[str, tuple[float, float]]:
    """Return metric -> (point_estimate, bootstrap_std). Point estimate on full test set."""
    rng = np.random.default_rng(seed)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    point = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    if n_boot <= 0:
        return {k: (v, 0.0) for k, v in point.items()}

    n = len(y_true)
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []

    for _ in tqdm(
        range(n_boot),
        desc="Bootstrap (test resamples)",
        dynamic_ncols=True,
        colour="cyan",
        unit="draw",
        disable=not progress,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(float(accuracy_score(yt, yp)))
        precs.append(float(precision_score(yt, yp, average="binary", zero_division=0)))
        recs.append(float(recall_score(yt, yp, average="binary", zero_division=0)))
        f1s.append(float(f1_score(yt, yp, average="binary", zero_division=0)))

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return float(np.std(xs, ddof=1))

    return {
        "accuracy": (acc, _std(accs)),
        "precision": (prec, _std(precs)),
        "recall": (rec, _std(recs)),
        "f1": (f1, _std(f1s)),
    }


def default_train_path(test_path: Path) -> Path:
    return test_path.parent / "train.txt"


def load_embeddings_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Return (embedding matrix [vocab, dim], token -> row index)."""
    suffix = path.suffix.lower()
    if suffix == ".kv":
        wv = KeyedVectors.load(str(path), mmap="r")
        emb = np.asarray(wv.vectors, dtype=np.float32)
        word2idx: dict[str, int] = dict(wv.key_to_index)
        return emb, word2idx

    if suffix == ".model":
        model = Word2Vec.load(str(path))
        wv = model.wv
        emb = np.asarray(wv.vectors, dtype=np.float32)
        word2idx = dict(wv.key_to_index)
        return emb, word2idx

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "embeddings" not in ckpt or "word2idx" not in ckpt:
        raise ValueError(
            "Checkpoint must contain 'embeddings' and 'word2idx', or use a .kv / .model file."
        )
    emb_t = ckpt["embeddings"]
    if emb_t.dim() != 2:
        raise ValueError("'embeddings' must be 2D [vocab, dim].")
    emb = emb_t.detach().cpu().numpy().astype(np.float32, copy=False)
    return emb, ckpt["word2idx"]


def run_evaluation(
    test_path: Path,
    checkpoint_path: Path,
    *,
    train_path: Path | None = None,
    bootstrap: int = 1000,
    seed: int = 42,
    max_iter: int = 1000,
    chunk_size: int = 2048,
    progress: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Train logistic regression on train embeddings and return test metrics (+ bootstrap std).

    Raises FileNotFoundError if paths are missing, ValueError for invalid labeled data or
    checkpoint format.
    """
    test_path = test_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    tr_path = (
        train_path.resolve()
        if train_path is not None
        else default_train_path(test_path).resolve()
    )
    if not tr_path.is_file():
        raise FileNotFoundError(f"Training file not found: {tr_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    emb, word2idx = load_embeddings_checkpoint(checkpoint_path)
    train_tokens, y_train = load_labeled_txt(tr_path)
    test_tokens, y_test = load_labeled_txt(test_path)

    if verbose:
        tqdm.write(
            f"Train rows: {len(train_tokens)}  |  Test rows: {len(test_tokens)}"
        )
        tqdm.write(f"Embedding matrix: {emb.shape[0]} x {emb.shape[1]}")

    X_train, oov_tr = tokens_to_embeddings(
        train_tokens,
        emb,
        word2idx,
        "Embed train entities",
        chunk_size,
        progress=progress,
    )
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    if verbose:
        tqdm.write("Fitting logistic regression on train embeddings…")
    clf.fit(X_train, y_train)

    X_test, oov_te = tokens_to_embeddings(
        test_tokens,
        emb,
        word2idx,
        "Embed test entities",
        chunk_size,
        progress=progress,
    )
    if verbose and (oov_tr or oov_te):
        tqdm.write(
            f"Note: OOV tokens (zero vector): train={oov_tr}, test={oov_te}"
        )
    if verbose:
        tqdm.write("Predicting on test set…")
    y_pred = clf.predict(X_test)
    metrics = bootstrap_metrics(y_test, y_pred, bootstrap, seed, progress=progress)

    out: dict[str, Any] = {
        "test_path": str(test_path),
        "train_path": str(tr_path),
        "checkpoint_path": str(checkpoint_path),
        "n_train": len(train_tokens),
        "n_test": len(test_tokens),
        "emb_vocab": int(emb.shape[0]),
        "emb_dim": int(emb.shape[1]),
        "oov_train": oov_tr,
        "oov_test": oov_te,
        "bootstrap": bootstrap,
        "seed": seed,
    }
    for name in ("accuracy", "precision", "recall", "f1"):
        val, std = metrics[name]
        out[name] = val
        out[f"{name}_std"] = std
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate embedding checkpoint on labeled test.txt (node classification)."
    )
    p.add_argument(
        "test",
        type=Path,
        help="Labeled test file: <entity>\\t<label> per line",
    )
    p.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Embeddings: .pt (train_word2vec / rdf2vec_final.pt), gensim .kv, or Word2Vec .model",
    )
    p.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Labeled train file (default: train.txt next to test file)",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for metric std (0 to skip)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    p.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="LogisticRegression max_iter",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Rows per progress chunk when building embedding matrices",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Printed before the metrics table (e.g. TC + strategy) so batch logs stay readable.",
    )
    args = p.parse_args()

    train_path = args.train if args.train is not None else default_train_path(args.test)
    if not train_path.is_file():
        raise SystemExit(
            f"Training file not found: {train_path}\n"
            "Pass --train /path/to/train.txt or place train.txt beside test.txt."
        )
    if not args.test.is_file():
        raise SystemExit(f"Test file not found: {args.test}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    try:
        res = run_evaluation(
            args.test,
            args.checkpoint,
            train_path=train_path,
            bootstrap=args.bootstrap,
            seed=args.seed,
            max_iter=args.max_iter,
            chunk_size=args.chunk_size,
            progress=True,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    tqdm.write(
        f"Train rows: {res['n_train']}  |  Test rows: {res['n_test']}"
    )
    tqdm.write(f"Embedding matrix: {res['emb_vocab']} x {res['emb_dim']}")
    tqdm.write("Fitting logistic regression on train embeddings…")
    if res["oov_train"] or res["oov_test"]:
        tqdm.write(
            f"Note: OOV tokens (zero vector): train={res['oov_train']}, test={res['oov_test']}"
        )
    tqdm.write("Predicting on test set…")

    print()
    if args.label:
        print("─" * 52)
        print(args.label)
        print("─" * 52)
    print("─" * 52)
    print("Test metrics (binary classification, positive class = 1)")
    print("─" * 52)
    for name in ("accuracy", "precision", "recall", "f1"):
        val, std = res[name], res[f"{name}_std"]
        if args.bootstrap > 0:
            print(f"  {name:12s}  {val:.4f}  ±  {std:.4f}  (bootstrap std)")
        else:
            print(f"  {name:12s}  {val:.4f}")
    print("─" * 52)
    if args.bootstrap > 0:
        print(f"Bootstrap draws: {args.bootstrap}  |  seed: {args.seed}")


if __name__ == "__main__":
    main()
