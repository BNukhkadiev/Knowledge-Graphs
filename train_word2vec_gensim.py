#!/usr/bin/env python3
"""
Train Word2Vec from a walks file using Gensim (fast CPU implementation).

Input format matches train_word2vec.py: one space-separated walk per line.

Logs per-epoch training loss, writes a CSV of losses, saves a loss plot, and shows
a tqdm progress bar over epochs. Optional per-step loss CSV/plot (--loss-every-steps) logs
cumulative loss every N training batches (Gensim jobs); requires a single worker.
Optional Gensim logger progress (within-epoch) can be enabled with --gensim-log-progress.

Saves a .pt checkpoint compatible with evaluate_embeddings.py (embeddings + word2idx).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from tqdm.auto import tqdm


class Word2VecWithStepLoss(Word2Vec):
    """Hooks Gensim's per-job training to log loss every ``step_loss_interval`` batches.

    A "step" is one :meth:`~gensim.models.word2vec.Word2Vec._do_train_job` call (a chunk
    of sentences up to ``batch_words``). This must run with ``workers=1`` so jobs are
    sequential and ``running_training_loss`` matches each batch.
    """

    def __init__(self, *args, step_loss_interval: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_loss_interval = step_loss_interval
        self.train_epoch_idx: int = 0
        self.step_loss_rows: list[tuple[int, int, float, float]] = []
        self._job_step: int = 0
        self._cum_at_last_interval: float = 0.0

    def _do_train_job(self, sentences, alpha, inits):  # type: ignore[no-untyped-def]
        tally, raw_tally = super()._do_train_job(sentences, alpha, inits)
        if self.step_loss_interval <= 0:
            return tally, raw_tally
        self._job_step += 1
        cum = float(self.running_training_loss)
        if self._job_step % self.step_loss_interval == 0:
            self.step_loss_rows.append(
                (
                    self._job_step,
                    self.train_epoch_idx,
                    cum,
                    cum - self._cum_at_last_interval,
                )
            )
            self._cum_at_last_interval = cum
        return tally, raw_tally


class EpochTagCallback(CallbackAny2Vec):
    """Sets ``model.train_epoch_idx`` at each epoch start (for step-loss CSV)."""

    def __init__(self, model: Word2VecWithStepLoss) -> None:
        self._model = model
        self._epoch = 0

    def on_epoch_begin(self, model: Word2Vec) -> None:
        self._epoch += 1
        self._model.train_epoch_idx = self._epoch


class EpochLossProgress(CallbackAny2Vec):
    """Per-epoch loss (from cumulative training loss) and tqdm over epochs."""

    def __init__(self, total_epochs: int, desc: str) -> None:
        self.total_epochs = total_epochs
        self.desc = desc
        self.prev_cumulative = 0.0
        self.epoch_losses: list[float] = []
        self._pbar: tqdm | None = None

    def on_train_begin(self, model: Word2Vec) -> None:
        self._pbar = tqdm(
            total=self.total_epochs,
            desc=self.desc,
            unit="epoch",
            dynamic_ncols=True,
            colour="green",
            file=sys.stdout,
        )

    def on_epoch_end(self, model: Word2Vec) -> None:
        cumulative = float(model.get_latest_training_loss())
        epoch_loss = cumulative - self.prev_cumulative
        self.prev_cumulative = cumulative
        self.epoch_losses.append(epoch_loss)
        if self._pbar is not None:
            self._pbar.set_postfix(loss=f"{epoch_loss:.4f}", refresh=False)
            self._pbar.update(1)

    def on_train_end(self, model: Word2Vec) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def save_step_loss_csv(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "cumulative_loss", "interval_loss"])
        for step, epoch, cum, interval in rows:
            w.writerow([step, epoch, f"{cum:.8f}", f"{interval:.8f}"])


def save_loss_csv(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        for i, loss in enumerate(losses, start=1):
            w.writerow([i, f"{loss:.8f}"])


def plot_losses(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(epochs), losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Gensim Word2Vec — loss per epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_step_losses(
    path: Path,
    rows: list[tuple[int, int, float, float]],
) -> None:
    """Plot cumulative and interval loss vs training batch step (two panels)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r[0] for r in rows]
    cumulative = [r[2] for r in rows]
    interval = [r[3] for r in rows]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax0.plot(steps, cumulative, marker="o", markersize=2)
    ax0.set_ylabel("Cumulative loss")
    ax0.set_title("Gensim Word2Vec — loss vs training batch step")
    ax0.grid(True, alpha=0.3)
    ax1.plot(steps, interval, marker="o", markersize=2, color="C1")
    ax1.set_xlabel("Step (batch index)")
    ax1.set_ylabel("Interval loss")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_gensim_to_pt(model: Word2Vec) -> dict:
    """Checkpoint dict compatible with evaluate_embeddings.py / train_word2vec.py."""
    wv = model.wv
    emb = torch.from_numpy(wv.vectors.copy())
    idx2word = list(wv.index_to_key)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return {
        "embeddings": emb,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "dim": int(model.vector_size),
        "architecture": "skipgram" if model.sg else "cbow",
        "window": int(model.window),
        "epochs": int(model.epochs),
        "trainer": "gensim",
        "gensim_version": __import__("gensim").__version__,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train Word2Vec with Gensim (walks file → .pt + loss log/plot).",
    )
    p.add_argument("walks", type=Path, help="Walks file: one space-separated walk per line")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("word2vec_gensim.pt"),
        help="Output .pt checkpoint (embeddings + word2idx)",
    )
    p.add_argument(
        "--architecture",
        "--arch",
        dest="architecture",
        choices=("skipgram", "cbow"),
        default="skipgram",
        help="sg=1 skip-gram, sg=0 CBOW",
    )
    p.add_argument("--dim", type=int, default=128, help="Vector size (vector_size)")
    p.add_argument("--window", type=int, default=5, help="Context window")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--negative", type=int, default=5, help="Negative samples")
    p.add_argument("--min-count", type=int, default=1, help="Ignore tokens below this count")
    p.add_argument("--lr", type=float, default=0.025, help="Initial learning rate (alpha)")
    p.add_argument(
        "--min-alpha",
        type=float,
        default=0.0001,
        help="Minimum learning rate after linear decay (min_alpha)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads (0 = use all available cores)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--loss-log",
        type=Path,
        default=None,
        help="CSV path for per-epoch losses (default: <output stem>_loss.csv)",
    )
    p.add_argument(
        "--loss-plot",
        type=Path,
        default=None,
        help="PNG path for loss curve (default: <output stem>_loss.png)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write loss PNGs (epoch curve or step curve)",
    )
    p.add_argument(
        "--gensim-log-progress",
        action="store_true",
        help="Enable Gensim INFO logging (within-epoch progress lines; can be noisy)",
    )
    p.add_argument(
        "--report-delay",
        type=float,
        default=1.0,
        help="Seconds between Gensim progress log lines (only with --gensim-log-progress)",
    )
    p.add_argument(
        "--loss-every-steps",
        type=int,
        default=0,
        metavar="N",
        help="Log cumulative loss every N training batches (Gensim jobs); implies --workers 1",
    )
    p.add_argument(
        "--loss-steps-log",
        type=Path,
        default=None,
        help="CSV for step losses (default: <output stem>_loss_steps.csv when --loss-every-steps > 0)",
    )
    p.add_argument(
        "--loss-steps-plot",
        type=Path,
        default=None,
        help="PNG for step loss curves (default: <output stem>_loss_steps.png when --loss-every-steps > 0)",
    )
    args = p.parse_args()

    if not args.walks.is_file():
        raise SystemExit(f"Walks file not found: {args.walks}")

    if args.gensim_log_progress:
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s",
            level=logging.INFO,
        )

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    if args.loss_every_steps > 0:
        if args.workers > 1:
            tqdm.write(
                "Note: --loss-every-steps requires a single worker; using --workers 1.",
                file=sys.stderr,
            )
        workers = 1

    sentences = LineSentence(str(args.walks))

    w2v_kw = dict(
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        workers=workers,
        sg=1 if args.architecture == "skipgram" else 0,
        hs=0,
        negative=args.negative,
        alpha=args.lr,
        min_alpha=args.min_alpha,
        seed=args.seed,
        epochs=args.epochs,
    )
    if args.loss_every_steps > 0:
        model = Word2VecWithStepLoss(step_loss_interval=args.loss_every_steps, **w2v_kw)
    else:
        model = Word2Vec(**w2v_kw)

    model.build_vocab(sentences, progress_per=10000)

    if len(model.wv) == 0:
        raise SystemExit("Empty vocabulary after min_count filtering.")

    cb = EpochLossProgress(
        total_epochs=args.epochs,
        desc=f"Gensim {args.architecture.upper()} [loss]",
    )
    callbacks: list[CallbackAny2Vec] = [cb]
    if args.loss_every_steps > 0:
        assert isinstance(model, Word2VecWithStepLoss)
        callbacks.insert(0, EpochTagCallback(model))

    train_kw: dict = {
        "corpus_iterable": sentences,
        "total_examples": model.corpus_count,
        "epochs": args.epochs,
        "compute_loss": True,
        "callbacks": callbacks,
    }
    if args.gensim_log_progress:
        train_kw["report_delay"] = args.report_delay

    model.train(**train_kw)

    losses = cb.epoch_losses
    if len(losses) != args.epochs:
        tqdm.write(
            f"Warning: expected {args.epochs} epoch losses, got {len(losses)}.",
            file=sys.stderr,
        )

    stem = args.output.with_suffix("")
    loss_log = args.loss_log if args.loss_log is not None else Path(f"{stem}_loss.csv")
    loss_plot = args.loss_plot if args.loss_plot is not None else Path(f"{stem}_loss.png")

    save_loss_csv(loss_log, losses)
    tqdm.write(f"Wrote loss log: {loss_log}")

    if args.loss_every_steps > 0 and isinstance(model, Word2VecWithStepLoss):
        steps_path = (
            args.loss_steps_log
            if args.loss_steps_log is not None
            else Path(f"{stem}_loss_steps.csv")
        )
        save_step_loss_csv(steps_path, model.step_loss_rows)
        tqdm.write(
            f"Wrote step loss log ({args.loss_every_steps} batch interval): {steps_path}",
        )
        if not args.no_plot and model.step_loss_rows:
            steps_plot = (
                args.loss_steps_plot
                if args.loss_steps_plot is not None
                else Path(f"{stem}_loss_steps.png")
            )
            plot_step_losses(steps_plot, model.step_loss_rows)
            tqdm.write(f"Wrote step loss plot: {steps_plot}")
        elif not args.no_plot and not model.step_loss_rows:
            tqdm.write(
                "No step loss rows to plot (try a smaller --loss-every-steps or more data).",
                file=sys.stderr,
            )

    if not args.no_plot:
        plot_losses(loss_plot, losses)
        tqdm.write(f"Wrote loss plot: {loss_plot}")

    ckpt = build_gensim_to_pt(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    tqdm.write(
        f"Saved checkpoint to {args.output}  (vocab={len(ckpt['word2idx'])}, dim={args.dim})",
    )


if __name__ == "__main__":
    main()
