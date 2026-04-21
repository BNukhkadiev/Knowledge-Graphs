#!/usr/bin/env python3
"""
Train Word2Vec with Gensim (fast CPU implementation).

Modes (--mode):
  none — Train on a single walks file (default).
  p1, p2 — Two-stage RDF2Vec: pretrain on protograph P1 or P2 walks, then finetune on
           instance walks from a file you supply (MASCHInE-style init when an ontology is available).
           Class-vector aggregation is configurable (--maschine-strategy).

Input: one space-separated walk per line.

Single-corpus mode logs per-epoch loss, CSV, PNG, optional per-step loss (--loss-every-steps).
p1/p2 mode writes pretrain_per_epoch.png and protograph_per_epoch.png; per-step PNGs use
--loss-every-steps-pretrain / --loss-every-steps-finetune (or --loss-every-steps N for both).
Without MASCHInE, optional predicate embedding transfer from stage 1 uses --transfer-predicate-embeddings (default on).
Writes instance_to_class.json (default): ``{ instance_iri_inner: [ class_iri_inner, ... ] }`` from ontology.nt.
Saves a .pt checkpoint compatible with evaluate_embeddings.py (embeddings + word2idx).

Reproducibility: pass ``--seed``; training then uses ``workers=1`` (Gensim's multi-worker
path is not bit-identical). ``PYTHONHASHSEED`` is fixed only if set in the environment
before starting the interpreter.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from tqdm.auto import tqdm

from src.protograph.maschine_init import (
    apply_maschine_initialization,
    instance_class_mapping_from_ontology,
    transfer_predicate_embeddings_from_pretrained,
)


def _workers_effective(workers: int) -> int:
    return workers if workers > 0 else (os.cpu_count() or 1)


def _set_training_rng(seed: int) -> None:
    """Seed stdlib, NumPy, and PyTorch RNGs (Gensim Word2Vec uses NumPy for sampling)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _effective_workers_for_reproducibility(seed: int | None, workers: int, *, label: str) -> int:
    """Return worker count. When ``seed`` is set, force ``workers=1`` (Gensim multi-worker SGD is nondeterministic)."""
    w = workers
    if seed is not None and w > 1:
        print(
            f"Note: --seed set: using workers=1 for {label} (Gensim Word2Vec is nondeterministic with workers>1).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return w


class Word2VecWithStepLoss(Word2Vec):
    """Hooks Gensim's per-job training to log loss on a step grid.

    A "step" is one :meth:`~gensim.models.word2vec.Word2Vec._do_train_job` call (a chunk
    of sentences up to ``batch_words``). This must run with ``workers=1`` so jobs are
    sequential and ``running_training_loss`` matches each batch.

    Samples are recorded at the **first** job, every ``step_loss_interval`` jobs thereafter,
    plus one :class:`StepLossEndFlush` callback appends the **final** job if it was not
    already sampled (so short runs still get a usable curve).
    """

    def __init__(self, *args, step_loss_interval: int = 0, step_loss_quiet: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_loss_interval = step_loss_interval
        self.step_loss_quiet = step_loss_quiet
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
        n = self.step_loss_interval
        log_this = self._job_step == 1 or (self._job_step % n == 0)
        if log_this:
            interval = cum - self._cum_at_last_interval
            self.step_loss_rows.append(
                (
                    self._job_step,
                    self.train_epoch_idx,
                    cum,
                    interval,
                )
            )
            if not self.step_loss_quiet:
                tqdm.write(
                    f"[step loss] step={self._job_step} epoch={self.train_epoch_idx} "
                    f"cumulative={cum:.6f} interval={interval:.6f}",
                    file=sys.stdout,
                )
            self._cum_at_last_interval = cum
        return tally, raw_tally


class StepLossEndFlush(CallbackAny2Vec):
    """Append one step-loss sample at the last training job if it was not already logged."""

    def on_train_end(self, model: Word2Vec) -> None:
        if not isinstance(model, Word2VecWithStepLoss):
            return
        m = model
        if m.step_loss_interval <= 0 or m._job_step == 0:
            return
        last_logged = m.step_loss_rows[-1][0] if m.step_loss_rows else 0
        if last_logged == m._job_step:
            return
        cum = float(model.get_latest_training_loss())
        interval = cum - m._cum_at_last_interval
        m.step_loss_rows.append((m._job_step, m.train_epoch_idx, cum, interval))
        if not m.step_loss_quiet:
            tqdm.write(
                f"[step loss] step={m._job_step} epoch={m.train_epoch_idx} "
                f"cumulative={cum:.6f} interval={interval:.6f} (end flush)",
                file=sys.stdout,
            )
        m._cum_at_last_interval = cum


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

    def __init__(self, total_epochs: int, desc: str, *, echo: bool = False) -> None:
        self.total_epochs = total_epochs
        self.desc = desc
        self.echo = echo
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
        if self.echo:
            tqdm.write(
                f"[{self.desc}] epoch {len(self.epoch_losses)}/{self.total_epochs} "
                f"loss={epoch_loss:.6f}",
                file=sys.stdout,
            )

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


def _plot_loss_per_epoch(
    path: Path,
    losses: list[float],
    *,
    title: str,
    suptitle: str | None = None,
) -> None:
    if not losses:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(epochs), losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_per_step(
    path: Path,
    rows: list[tuple[int, int, float, float]],
    *,
    title: str,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r[0] for r in rows]
    cumulative = [r[2] for r in rows]
    interval = [r[3] for r in rows]
    smax = max(steps)
    x_right = max(smax * 1.02, smax + 1.0)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax0.plot(steps, cumulative, marker="o", markersize=2)
    ax0.set_ylabel("Cumulative loss")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(0.0, x_right)
    ax1.plot(steps, interval, marker="o", markersize=2, color="C1")
    ax1.set_xlabel("Step (batch index)")
    ax1.set_ylabel("Interval loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0, x_right)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_instance_to_class_json(path: Path, ontology_path: Path | None) -> None:
    """Write ``{ instance_iri_inner: [ class_iri_inner, ... ] }`` from ontology (MASCHInE most-specific types)."""
    if ontology_path is None or not ontology_path.is_file():
        payload: dict[str, list[str]] = {}
    else:
        full = instance_class_mapping_from_ontology(ontology_path)
        payload = {ent: d["most_specific_class_iris"] for ent, d in full.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_gensim_to_pt(model: Word2Vec, *, training_seed: int | None = None) -> dict:
    """Checkpoint dict compatible with evaluate_embeddings.py / train_word2vec.py."""
    wv = model.wv
    emb = torch.from_numpy(wv.vectors.copy())
    idx2word = list(wv.index_to_key)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    out: dict = {
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
    if training_seed is not None:
        out["training_seed"] = int(training_seed)
    return out


def train_word2vec_stage(
    walks_path: Path,
    *,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    negative: int,
    epochs: int,
    seed: int | None,
    alpha: float,
    min_alpha: float,
    desc: str,
    step_loss_interval: int = 0,
    step_loss_quiet: bool = False,
) -> tuple[Word2Vec, list[float]]:
    """Train Word2Vec with per-epoch loss; optional per-batch step loss (``workers`` must be 1)."""
    sentences = LineSentence(str(walks_path))
    kw = dict(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        hs=0,
        negative=negative,
        alpha=alpha,
        min_alpha=min_alpha,
        seed=seed,
        epochs=epochs,
    )
    model = Word2VecWithStepLoss(
        step_loss_interval=step_loss_interval,
        step_loss_quiet=step_loss_quiet,
        **kw,
    )
    model.build_vocab(sentences, progress_per=10000)
    cb = EpochLossProgress(total_epochs=epochs, desc=desc)
    callbacks: list[CallbackAny2Vec] = [cb]
    if step_loss_interval > 0:
        assert isinstance(model, Word2VecWithStepLoss)
        callbacks.insert(0, EpochTagCallback(model))
        callbacks.append(StepLossEndFlush())
    model.train(
        corpus_iterable=sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=callbacks,
    )
    return model, cb.epoch_losses


def run_rdf2vec_two_stage(args: argparse.Namespace) -> None:
    """Pretrain on P1 or P2 protograph walks, then finetune on instance walks (``--mode p1`` / ``p2``)."""
    mode = args.mode
    assert mode in ("p1", "p2")

    if args.seed is not None:
        _set_training_rng(int(args.seed))

    if args.maschine_strategy == "ancestor_weighted" and not (0.0 < args.maschine_ancestor_decay <= 1.0):
        raise SystemExit(
            "--maschine-ancestor-decay must satisfy 0 < R <= 1 when using --maschine-strategy ancestor_weighted."
        )

    if args.pretrain_walks is None:
        args.pretrain_walks = Path("walks_p1.txt") if mode == "p1" else Path("walks_p2.txt")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.loss_every_steps > 0:
        pretrain_loss_every = args.loss_every_steps
        finetune_loss_every = args.loss_every_steps
    else:
        pretrain_loss_every = args.loss_every_steps_pretrain
        finetune_loss_every = args.loss_every_steps_finetune

    any_step_loss = pretrain_loss_every > 0 or finetune_loss_every > 0
    if not any_step_loss and not args.no_loss_plots:
        print(
            "Note: per-step loss PNGs are disabled (use --loss-every-steps-pretrain / "
            "--loss-every-steps-finetune, or --loss-every-steps N for both; implies workers=1).",
            file=sys.stderr,
            flush=True,
        )

    pretrained_path = out_dir / "rdf2vec_pretrained.model"
    final_pt_path = args.output
    if not final_pt_path.is_absolute():
        final_pt_path = out_dir / final_pt_path

    sg = 1 if args.architecture == "skipgram" else 0
    pretrain_losses: list[float] | None = None
    pretrain_step_rows: list[tuple[int, int, float, float]] = []

    if not args.skip_pretrain:
        if not args.pretrain_walks.is_file():
            raise SystemExit(f"Pretrain walks not found: {args.pretrain_walks}")
        print(f"Stage 1: training Word2Vec on {args.pretrain_walks} ...")
        st_interval = 0
        st_quiet = False
        if pretrain_loss_every > 0:
            st_interval = pretrain_loss_every
            st_quiet = True
            pre_workers = 1
            if args.workers > 1:
                print(
                    "Note: step loss logging requires a single worker during stage 1; using workers=1.",
                    flush=True,
                )
        else:
            pre_workers = _workers_effective(args.workers)
        pre_workers = _effective_workers_for_reproducibility(
            args.seed, pre_workers, label="stage 1 (pretrain)"
        )
        pre_lr = args.pretrain_lr if args.pretrain_lr is not None else args.lr
        pre_min_alpha = args.pretrain_min_alpha if args.pretrain_min_alpha is not None else args.min_alpha
        pre_model, pretrain_losses = train_word2vec_stage(
            args.pretrain_walks,
            vector_size=args.dim,
            window=args.window,
            min_count=args.min_count,
            workers=pre_workers,
            sg=sg,
            negative=args.negative,
            epochs=args.pretrain_epochs,
            seed=args.seed,
            alpha=pre_lr,
            min_alpha=pre_min_alpha,
            desc="Stage 1 pretrain",
            step_loss_interval=st_interval,
            step_loss_quiet=st_quiet,
        )
        if isinstance(pre_model, Word2VecWithStepLoss) and pretrain_loss_every > 0:
            pretrain_step_rows = list(pre_model.step_loss_rows)
        pre_model.save(str(pretrained_path))
        print(f"Wrote {pretrained_path} (vocab size {len(pre_model.wv)})")
    else:
        if not pretrained_path.is_file():
            raise SystemExit(f"--skip-pretrain but missing {pretrained_path}")
        pre_model = Word2Vec.load(str(pretrained_path))
        print(f"Loaded {pretrained_path} (vocab size {len(pre_model.wv)})")

    instance_path = args.instance_walks
    if instance_path is None:
        raise SystemExit(
            "Instance walks required for --mode p1/p2: pass positional <walks> or --instance-walks."
        )
    if not instance_path.is_file():
        raise SystemExit(f"Instance walks not found: {instance_path}")

    print(f"Stage 2: building vocabulary from instance walks {instance_path} ...", flush=True)

    ontology_path = args.ontology
    if ontology_path is None:
        cand = instance_path.parent / "ontology.nt"
        if cand.is_file():
            ontology_path = cand

    stage1_vectors = {
        w: np.asarray(pre_model.wv[w], dtype=np.float32).copy() for w in pre_model.wv.index_to_key
    }
    pre_model.build_vocab(LineSentence(str(instance_path)), update=True, progress_per=10000)
    n_inst = int(pre_model.corpus_count)
    print(f"Stage 2: continuing training on {instance_path} ({n_inst} sentences) ...", flush=True)

    new_token_init: dict[str, str] = {}
    maschine_applied = False
    if args.maschine_init and ontology_path is not None and ontology_path.is_file():
        maschine_applied = True
        n_ent, n_rel, n_rand = apply_maschine_initialization(
            pre_model,
            stage1_vectors,
            ontology_path,
            strategy=args.maschine_strategy,
            ancestor_decay=args.maschine_ancestor_decay,
            new_token_init=new_token_init,
        )
        print(
            f"MASCHInE init ({args.maschine_strategy}): entity rows from class vectors={n_ent}, "
            f"relation rows copied from pretrain={n_rel}, left random={n_rand}"
        )
    elif args.maschine_init and (ontology_path is None or not ontology_path.is_file()):
        print(
            "MASCHInE init skipped (no ontology.nt); pass --ontology or place ontology.nt beside instance walks"
        )

    n_pred_transfer = 0
    if not maschine_applied and args.transfer_predicate_embeddings:
        n_pred_transfer = transfer_predicate_embeddings_from_pretrained(
            pre_model,
            stage1_vectors,
            new_token_init=new_token_init,
        )
        if n_pred_transfer:
            print(
                f"Transferred predicate embeddings from pretrain (rows updated={n_pred_transfer}); "
                "use --no-transfer-predicate-embeddings to disable.",
                flush=True,
            )

    if not maschine_applied:
        for w in pre_model.wv.index_to_key:
            if w not in stage1_vectors and w not in new_token_init:
                new_token_init[w] = "gensim_default"

    if not args.no_pretrain_init_mapping:
        map_path = args.pretrain_init_mapping
        if map_path is None:
            map_path = out_dir / "instance_to_class.json"
        elif not map_path.is_absolute():
            map_path = out_dir / map_path
        save_instance_to_class_json(
            map_path,
            ontology_path if ontology_path is not None and ontology_path.is_file() else None,
        )
        print(f"Wrote instance→class mapping: {map_path}")

    if finetune_loss_every > 0 and args.workers > 1:
        print(
            "Note: step loss logging requires a single worker during stage 2; using workers=1.",
            flush=True,
        )
    ft_workers = 1 if finetune_loss_every > 0 else _workers_effective(args.workers)
    ft_workers = _effective_workers_for_reproducibility(
        args.seed, ft_workers, label="stage 2 (finetune)"
    )
    pre_model.workers = ft_workers

    # Finetune uses --lr / --min-alpha (independent of stage-1 pretrain-lr / pretrain-min-alpha).
    pre_model.alpha = args.lr
    pre_model.min_alpha = args.min_alpha

    ft_loss_cb = EpochLossProgress(
        total_epochs=args.finetune_epochs,
        desc="Stage 2 finetune",
    )
    ft_callbacks: list[CallbackAny2Vec] = [ft_loss_cb]
    step_model_ok = isinstance(pre_model, Word2VecWithStepLoss)
    if finetune_loss_every > 0 and not step_model_ok:
        print(
            "Warning: step-loss plots need a model saved from this script's pretrain "
            "(Word2VecWithStepLoss). Skipping per-step plots; epoch plot still saved.",
            file=sys.stderr,
        )
    if finetune_loss_every > 0 and step_model_ok:
        assert isinstance(pre_model, Word2VecWithStepLoss)
        pre_model.step_loss_interval = finetune_loss_every
        pre_model.step_loss_quiet = False
        pre_model.step_loss_rows.clear()
        pre_model._job_step = 0
        pre_model._cum_at_last_interval = 0.0
        pre_model.train_epoch_idx = 0
        ft_callbacks.insert(0, EpochTagCallback(pre_model))
        ft_callbacks.append(StepLossEndFlush())
    elif isinstance(pre_model, Word2VecWithStepLoss):
        pre_model.step_loss_interval = 0

    pre_model.train(
        corpus_iterable=LineSentence(str(instance_path)),
        total_examples=n_inst,
        epochs=args.finetune_epochs,
        compute_loss=True,
        callbacks=ft_callbacks,
    )
    finetune_losses = ft_loss_cb.epoch_losses

    if not args.skip_pretrain and pretrain_losses is not None and len(pretrain_losses) > 0:
        pre_csv = out_dir / "pretrain_loss.csv"
        save_loss_csv(pre_csv, pretrain_losses)
        print(f"Wrote pretrain loss log: {pre_csv}")
    if finetune_losses:
        ft_csv = out_dir / "finetune_loss.csv"
        save_loss_csv(ft_csv, finetune_losses)
        print(f"Wrote finetune loss log: {ft_csv}")
    if not args.skip_pretrain and pretrain_loss_every > 0 and pretrain_step_rows:
        pre_steps = out_dir / "pretrain_loss_steps.csv"
        save_step_loss_csv(pre_steps, pretrain_step_rows)
        print(f"Wrote pretrain step loss log: {pre_steps}")
    if finetune_loss_every > 0 and step_model_ok and isinstance(pre_model, Word2VecWithStepLoss):
        ft_rows = pre_model.step_loss_rows
        if ft_rows:
            ft_steps = out_dir / "finetune_loss_steps.csv"
            save_step_loss_csv(ft_steps, ft_rows)
            print(f"Wrote finetune step loss log: {ft_steps}")

    if not args.no_loss_plots:
        pre_ep = (
            args.loss_pretrain_epoch_plot
            if args.loss_pretrain_epoch_plot is not None
            else out_dir / "pretrain_per_epoch.png"
        )
        proto_ep = (
            args.loss_protograph_epoch_plot
            if args.loss_protograph_epoch_plot is not None
            else out_dir / "finetune_per_epoch.png"
        )
        pre_st = (
            args.loss_pretrain_steps_plot
            if args.loss_pretrain_steps_plot is not None
            else out_dir / "pretrain_per_step.png"
        )
        proto_st = (
            args.loss_steps_plot if args.loss_steps_plot is not None else out_dir / "finetune_per_step.png"
        )

        if not args.skip_pretrain and pretrain_losses and len(pretrain_losses) > 0:
            _plot_loss_per_epoch(
                pre_ep,
                pretrain_losses,
                title="Stage 1 — pretrain (protograph walks)",
                suptitle="RDF2Vec — pretrain loss per epoch",
            )
            print(f"Wrote pretrain per-epoch loss plot: {pre_ep}")
        elif not args.skip_pretrain:
            print("No pretrain epoch losses to plot.", file=sys.stderr)

        if finetune_losses:
            _plot_loss_per_epoch(
                proto_ep,
                finetune_losses,
                title="Stage 2 — finetune (instance walks)",
                suptitle="RDF2Vec — protograph loss per epoch",
            )
            print(f"Wrote protograph per-epoch loss plot: {proto_ep}")
        else:
            print("No finetune epoch losses to plot.", file=sys.stderr)

        if not args.skip_pretrain and pretrain_loss_every > 0:
            if pretrain_step_rows:
                _plot_loss_per_step(
                    pre_st,
                    pretrain_step_rows,
                    title="Stage 1 — pretrain (protograph walks) — loss vs batch step",
                )
                print(
                    f"Wrote pretrain per-step loss plot (every {pretrain_loss_every} batches): {pre_st}",
                )
            else:
                print(
                    "No stage-1 step-loss points to plot "
                    "(try a smaller --loss-every-steps-pretrain).",
                    file=sys.stderr,
                )

        if finetune_loss_every > 0 and step_model_ok:
            assert isinstance(pre_model, Word2VecWithStepLoss)
            if pre_model.step_loss_rows:
                _plot_loss_per_step(
                    proto_st,
                    pre_model.step_loss_rows,
                    title="Stage 2 — finetune (instance walks) — loss vs batch step",
                )
                print(
                    f"Wrote protograph per-step loss plot (every {finetune_loss_every} batches): {proto_st}",
                )
            else:
                print(
                    "No stage-2 step-loss points to plot (try a smaller --loss-every-steps-finetune).",
                    file=sys.stderr,
                )

    ckpt = build_gensim_to_pt(pre_model, training_seed=args.seed)
    ckpt["trainer"] = "rdf2vec_train"
    final_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, final_pt_path)
    print(f"Wrote {final_pt_path} (vocab size {len(pre_model.wv)})")


def run_single_corpus_mode(args: argparse.Namespace) -> None:
    """Train on ``args.walks`` (``--mode none``)."""
    if args.walks is None:
        raise SystemExit("Walks file required when --mode is none.")
    if not args.walks.is_file():
        raise SystemExit(f"Walks file not found: {args.walks}")

    if args.seed is not None:
        _set_training_rng(int(args.seed))

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

    workers = _effective_workers_for_reproducibility(args.seed, workers, label="single-corpus training")

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
        callbacks.append(StepLossEndFlush())

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
            _plot_loss_per_step(
                steps_plot,
                model.step_loss_rows,
                title="Gensim Word2Vec — loss vs training batch step",
            )
            tqdm.write(f"Wrote step loss plot: {steps_plot}")
        elif not args.no_plot and not model.step_loss_rows:
            tqdm.write(
                "No step loss rows to plot (try a smaller --loss-every-steps or more data).",
                file=sys.stderr,
            )

    if not args.no_plot:
        _plot_loss_per_epoch(
            loss_plot,
            losses,
            title="Gensim Word2Vec — loss per epoch",
        )
        tqdm.write(f"Wrote loss plot: {loss_plot}")

    ckpt = build_gensim_to_pt(model, training_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    tqdm.write(
        f"Saved checkpoint to {args.output}  (vocab={len(ckpt['word2idx'])}, dim={args.dim})",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train Word2Vec with Gensim: single corpus (--mode none) or RDF2Vec P1/P2 two-stage.",
    )
    p.add_argument(
        "--mode",
        choices=("none", "p1", "p2"),
        default="none",
        help="none: train on one walks file. p1/p2: pretrain on protograph walks then finetune on instance walks.",
    )
    p.add_argument(
        "walks",
        type=Path,
        nargs="?",
        default=None,
        help="Walks file: required for --mode none; for p1/p2, instance walks unless --instance-walks is set.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pt checkpoint (default: word2vec_gensim.pt or rdf2vec_final.pt in --out-dir)",
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
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (only --mode none; for p1/p2 use --pretrain-epochs / --finetune-epochs)",
    )
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
        "--pretrain-lr",
        type=float,
        default=None,
        help="Stage-1 initial alpha (--mode p1/p2 only; default: same as --lr)",
    )
    p.add_argument(
        "--pretrain-min-alpha",
        type=float,
        default=None,
        help="Stage-1 min_alpha (--mode p1/p2 only; default: same as --min-alpha)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Gensim worker threads (0 = all cores; single-corpus mode). For --mode p1/p2, stage 1/2 training.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (sets Python/NumPy/torch RNGs; with Gensim, use this for reproducible runs — "
        "when set, training uses workers=1 because multi-threaded Word2Vec updates are nondeterministic).",
    )
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
        help="Per-batch step loss: CSV/PNG in --mode none. For p1/p2, if N>0, sets both "
        "--loss-every-steps-pretrain and --loss-every-steps-finetune to N; if 0, use those flags separately.",
    )
    p.add_argument(
        "--loss-every-steps-pretrain",
        type=int,
        default=0,
        metavar="N",
        help="p1/p2 stage 1: log step loss every N batch jobs (default 0 = epoch bar only; N>0 forces "
        "workers=1 and slows pretrain). Ignored when --loss-every-steps > 0.",
    )
    p.add_argument(
        "--loss-every-steps-finetune",
        type=int,
        default=0,
        metavar="N",
        help="p1/p2 stage 2: log step loss every N batch jobs (default 0 = epoch plots only). "
        "Ignored when --loss-every-steps > 0.",
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

    # RDF2Vec (--mode p1 / p2)
    p.add_argument(
        "--pretrain-walks",
        type=Path,
        default=None,
        help="Stage-1 walks (default: walks_p1.txt for p1, walks_p2.txt for p2).",
    )
    p.add_argument(
        "--instance-walks",
        type=Path,
        default=None,
        help="Stage-2 instance walks file (required for p1/p2 unless set via positional walks).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for models and walk files for --mode p1/p2 (default: current directory).",
    )
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Load existing pretrained model from --out-dir instead of training stage 1.",
    )
    p.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="N-Triples ontology with rdf:type and rdfs:subClassOf. "
        "If omitted, uses <instance-walks-dir>/ontology.nt when present.",
    )
    p.add_argument(
        "--maschine-init",
        dest="maschine_init",
        action="store_true",
        default=True,
        help="MASCHInE initialization for new instance tokens (default: on when ontology is available).",
    )
    p.add_argument(
        "--no-maschine-init",
        dest="maschine_init",
        action="store_false",
        help="Skip MASCHInE init; keep gensim random init for new vocabulary after build_vocab(update=True).",
    )
    p.add_argument(
        "--transfer-predicate-embeddings",
        dest="transfer_predicate_embeddings",
        action="store_true",
        default=True,
        help="(--mode p1/p2) When MASCHInE is off, copy stage-1 vectors for new relation/P_ walk tokens. "
        "When MASCHInE is on, relation rows are already copied there; this flag is ignored.",
    )
    p.add_argument(
        "--no-transfer-predicate-embeddings",
        dest="transfer_predicate_embeddings",
        action="store_false",
        help="(--mode p1/p2) Do not copy pretrained predicate rows when MASCHInE init is disabled.",
    )
    p.add_argument(
        "--maschine-strategy",
        type=str,
        default="most_specific",
        choices=("most_specific", "ancestor_mean", "ancestor_weighted"),
        help="How to combine protograph class embeddings for new instance rows: "
        "most_specific (default), unweighted mean over asserted type(s) plus all superclasses (ancestor_mean), "
        "or distance-decayed mean toward roots (ancestor_weighted; see --maschine-ancestor-decay).",
    )
    p.add_argument(
        "--maschine-ancestor-decay",
        type=float,
        default=0.5,
        metavar="R",
        help="For --maschine-strategy ancestor_weighted: per-hop multiplier so weight ∝ R**distance "
        "(smaller R emphasizes types closer to the asserted most-specific class). Ignored otherwise. "
        "Required: 0 < R ≤ 1.",
    )
    p.add_argument(
        "--loss-pretrain-epoch-plot",
        type=Path,
        default=None,
        help="PNG for stage-1 pretrain per-epoch loss (default: <out-dir>/pretrain_per_epoch.png).",
    )
    p.add_argument(
        "--loss-protograph-epoch-plot",
        type=Path,
        default=None,
        help="PNG for stage-2 instance finetune per-epoch loss (default: <out-dir>/finetune_per_epoch.png).",
    )
    p.add_argument(
        "--loss-pretrain-steps-plot",
        type=Path,
        default=None,
        help="PNG for stage-1 pretrain per-step loss (default: <out-dir>/pretrain_per_step.png).",
    )
    p.add_argument(
        "--no-loss-plots",
        action="store_true",
        help="Do not write RDF2Vec loss PNGs.",
    )
    p.add_argument(
        "--pretrain-init-mapping",
        type=Path,
        default=None,
        help="JSON path for instance→class mapping (default: <out-dir>/instance_to_class.json).",
    )
    p.add_argument(
        "--no-pretrain-init-mapping",
        action="store_true",
        help="Do not write instance→class JSON.",
    )

    args = p.parse_args()

    if args.mode == "none":
        args.output = args.output or Path("word2vec_gensim.pt")
        run_single_corpus_mode(args)
        return

    # p1 / p2
    if args.walks is not None:
        args.instance_walks = args.instance_walks or args.walks
    if args.instance_walks is None:
        raise SystemExit(
            "Instance walks required for --mode p1/p2: pass positional <walks> or --instance-walks."
        )
    args.output = args.output or Path("rdf2vec_final.pt")
    run_rdf2vec_two_stage(args)


if __name__ == "__main__":
    main()
