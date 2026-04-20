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
from pathlib import Path

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
        raise SystemExit(
            "Checkpoint must contain 'embeddings' and 'word2idx', or use a .kv / .model file."
        )
    emb_t = ckpt["embeddings"]
    if emb_t.dim() != 2:
        raise SystemExit("'embeddings' must be 2D [vocab, dim].")
    emb = emb_t.detach().cpu().numpy().astype(np.float32, copy=False)
    return emb, ckpt["word2idx"]


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

    emb, word2idx = load_embeddings_checkpoint(args.checkpoint)

    train_tokens, y_train = load_labeled_txt(train_path)
    test_tokens, y_test = load_labeled_txt(args.test)

    tqdm.write(f"Train rows: {len(train_tokens)}  |  Test rows: {len(test_tokens)}")
    tqdm.write(f"Embedding matrix: {emb.shape[0]} x {emb.shape[1]}")

    X_train, oov_tr = tokens_to_embeddings(
        train_tokens,
        emb,
        word2idx,
        desc="Embed train entities",
        chunk_size=args.chunk_size,
    )
    clf = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    tqdm.write("Fitting logistic regression on train embeddings…")
    clf.fit(X_train, y_train)

    X_test, oov_te = tokens_to_embeddings(
        test_tokens,
        emb,
        word2idx,
        desc="Embed test entities",
        chunk_size=args.chunk_size,
    )
    if oov_tr or oov_te:
        tqdm.write(f"Note: OOV tokens (zero vector): train={oov_tr}, test={oov_te}")

    tqdm.write("Predicting on test set…")
    y_pred = clf.predict(X_test)

    results = bootstrap_metrics(y_test, y_pred, args.bootstrap, args.seed)

    print()
    print("─" * 52)
    print("Test metrics (binary classification, positive class = 1)")
    print("─" * 52)
    for name in ("accuracy", "precision", "recall", "f1"):
        val, std = results[name]
        if args.bootstrap > 0:
            print(f"  {name:12s}  {val:.4f}  ±  {std:.4f}  (bootstrap std)")
        else:
            print(f"  {name:12s}  {val:.4f}")
    print("─" * 52)
    if args.bootstrap > 0:
        print(f"Bootstrap draws: {args.bootstrap}  |  seed: {args.seed}")


if __name__ == "__main__":
    main()
