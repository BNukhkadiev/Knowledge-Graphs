#!/usr/bin/env python3
"""
Train Word2Vec-style embeddings from a walks file (one space-separated walk per line).

Supports skip-gram or CBOW with negative sampling in PyTorch so training can run on CUDA
or Apple MPS when available. Optional initialization from a prior checkpoint produced by this
script (--pretrained). Progress is shown with tqdm.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def load_walks(path: Path) -> list[list[str]]:
    walks: list[list[str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            walks.append(line.split())
    return walks


def build_vocab(
    walks: list[list[str]],
    min_count: int,
) -> tuple[dict[str, int], list[str], torch.Tensor]:
    counts = Counter()
    for walk in walks:
        for w in walk:
            counts[w] += 1

    word2idx: dict[str, int] = {}
    idx2word: list[str] = []
    for w, c in counts.items():
        if c < min_count:
            continue
        word2idx[w] = len(idx2word)
        idx2word.append(w)

    if not word2idx:
        raise ValueError("Empty vocabulary after min_count filtering.")

    # float32: required for MPS (no float64 multinomial weights)
    freqs = torch.zeros(len(idx2word), dtype=torch.float32)
    for w, i in word2idx.items():
        freqs[i] = counts[w]

    return word2idx, idx2word, freqs


def noise_distribution(freqs: torch.Tensor, power: float = 0.75) -> torch.Tensor:
    p = torch.pow(freqs.float(), power)
    p = p / p.sum()
    return p


def walk_to_indices(sent: list[str], word2idx: dict[str, int]) -> list[int]:
    return [word2idx[w] for w in sent if w in word2idx]


def iter_skipgram_pairs(
    idx_sentences: list[list[int]],
    window: int,
) -> tuple[int, int]:
    """Yield (center_idx, context_idx) pairs."""
    for sent in idx_sentences:
        if len(sent) < 2:
            continue
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(len(sent), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                yield center, sent[j]


def batch_pairs(
    idx_sentences: list[list[int]],
    window: int,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize all pairs, shuffle, split into batches (for moderate corpora)."""
    pairs: list[tuple[int, int]] = list(iter_skipgram_pairs(idx_sentences, window))
    if not pairs:
        raise ValueError("No skip-gram pairs produced (walks too short or empty).")
    rng.shuffle(pairs)
    centers = [c for c, _ in pairs]
    contexts = [x for _, x in pairs]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(centers), batch_size):
        c_batch = torch.tensor(centers[i : i + batch_size], dtype=torch.long)
        x_batch = torch.tensor(contexts[i : i + batch_size], dtype=torch.long)
        batches.append((c_batch, x_batch))
    return batches


def iter_cbow_samples(
    idx_sentences: list[list[int]],
    window: int,
) -> tuple[int, list[int]]:
    """Yield (center_idx, context_indices) for CBOW (context predicts center)."""
    for sent in idx_sentences:
        if len(sent) < 2:
            continue
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(len(sent), i + window + 1)
            ctx = [sent[j] for j in range(lo, hi) if j != i]
            if ctx:
                yield center, ctx


def batch_cbow_samples(
    idx_sentences: list[list[int]],
    window: int,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """CBOW batches: (context_ids [B, L], mask [B, L], center [B])."""
    samples: list[tuple[int, list[int]]] = list(iter_cbow_samples(idx_sentences, window))
    if not samples:
        raise ValueError("No CBOW samples produced (walks too short or empty).")
    rng.shuffle(samples)
    max_ctx = 2 * window
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for s in range(0, len(samples), batch_size):
        chunk = samples[s : s + batch_size]
        bsz = len(chunk)
        ctx_ids = torch.zeros(bsz, max_ctx, dtype=torch.long)
        mask = torch.zeros(bsz, max_ctx, dtype=torch.float32)
        centers = torch.zeros(bsz, dtype=torch.long)
        for bi, (cen, ctx) in enumerate(chunk):
            centers[bi] = cen
            L = min(len(ctx), max_ctx)
            if L:
                ctx_ids[bi, :L] = torch.tensor(ctx[:L], dtype=torch.long)
                mask[bi, :L] = 1.0
        batches.append((ctx_ids, mask, centers))
    return batches


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        scale = 1.0 / math.sqrt(emb_dim)
        nn.init.uniform_(self.in_embed.weight, -scale, scale)
        nn.init.uniform_(self.out_embed.weight, -scale, scale)

    def forward(
        self,
        center: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        # center, pos: (B,) — neg: (B, k)
        v = self.in_embed(center)
        u_pos = self.out_embed(pos)
        u_neg = self.out_embed(neg)
        pos_loss = -F.logsigmoid((v * u_pos).sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()
        return pos_loss + neg_loss


class CBOWNeg(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        scale = 1.0 / math.sqrt(emb_dim)
        nn.init.uniform_(self.in_embed.weight, -scale, scale)
        nn.init.uniform_(self.out_embed.weight, -scale, scale)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        # context: (B, L), mask: (B, L), pos: (B,), neg: (B, k)
        emb = self.in_embed(context)
        denom = context_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        v = (emb * context_mask.unsqueeze(-1)).sum(dim=1) / denom
        u_pos = self.out_embed(pos)
        u_neg = self.out_embed(neg)
        pos_loss = -F.logsigmoid((v * u_pos).sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()
        return pos_loss + neg_loss


def pick_device(name: str) -> torch.device:
    n = name.lower()
    if n == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if n in ("cuda", "cpu"):
        return torch.device(n)
    if n == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("MPS requested but not available.")
        return torch.device("mps")
    raise SystemExit(f"Unknown device: {name}")


def load_pretrained_embeddings(
    model: nn.Module,
    word2idx: dict[str, int],
    pretrained_path: Path | None = None,
    ckpt: dict | None = None,
) -> int:
    """
    Copy rows from a checkpoint's `embeddings` into both in_embed and out_embed for
    overlapping vocabulary tokens. New tokens keep the existing random init.

    Pass either `ckpt` (already loaded) or `pretrained_path`.

    Returns the number of tokens initialized from the checkpoint.
    """
    if ckpt is None:
        if pretrained_path is None:
            raise ValueError("Either ckpt or pretrained_path is required.")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "embeddings" not in ckpt or "word2idx" not in ckpt:
        path_hint = f" ({pretrained_path})" if pretrained_path else ""
        raise SystemExit(
            f"Pretrained file must contain 'embeddings' and 'word2idx'{path_hint}."
        )
    pre_emb: torch.Tensor = ckpt["embeddings"].float()
    old_w2i: dict[str, int] = ckpt["word2idx"]
    if pre_emb.dim() != 2:
        raise SystemExit("Pretrained 'embeddings' must be a 2D tensor [vocab, dim].")
    d = pre_emb.shape[1]
    if d != model.in_embed.embedding_dim:
        raise SystemExit(
            f"Pretrained dim ({d}) does not match model dim ({model.in_embed.embedding_dim}). "
            "Use --dim to match the checkpoint."
        )
    if pre_emb.shape[0] != len(old_w2i):
        raise SystemExit(
            "Pretrained embeddings row count does not match len(word2idx); checkpoint may be corrupt."
        )

    n_copied = 0
    with torch.no_grad():
        for w, idx in word2idx.items():
            if w not in old_w2i:
                continue
            row = pre_emb[old_w2i[w]]
            model.in_embed.weight[idx].copy_(row)
            model.out_embed.weight[idx].copy_(row)
            n_copied += 1
    return n_copied


def main() -> None:
    p = argparse.ArgumentParser(description="Train Word2Vec from a walks file (PyTorch, GPU-capable).")
    p.add_argument("walks", type=Path, help="Walks file: one space-separated walk per line")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("word2vec.pt"),
        help="Output checkpoint (.pt) with embeddings and vocabulary",
    )
    p.add_argument(
        "--architecture",
        "--arch",
        dest="architecture",
        choices=("skipgram", "cbow"),
        default="skipgram",
        help="Word2Vec variant: skip-gram (center predicts context) or CBOW (context predicts center)",
    )
    p.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Initialize in/out embeddings from a .pt checkpoint from this script (overlapping vocab only)",
    )
    p.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    p.add_argument("--window", type=int, default=5, help="Context window size (each side)")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size (pairs per step)")
    p.add_argument("--negative", type=int, default=5, help="Number of negative samples per pair")
    p.add_argument("--min-count", type=int, default=1, help="Ignore tokens with count below this")
    p.add_argument("--lr", type=float, default=0.025, help="Learning rate")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device: auto picks CUDA, then MPS, then CPU",
    )
    args = p.parse_args()

    pre_ckpt: dict | None = None
    if args.pretrained is not None:
        if not args.pretrained.is_file():
            raise SystemExit(f"--pretrained not found: {args.pretrained}")
        pre_ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        if "dim" in pre_ckpt and int(pre_ckpt["dim"]) != args.dim:
            raise SystemExit(
                f"Checkpoint dim ({pre_ckpt['dim']}) does not match --dim ({args.dim}). "
                f"Use --dim {pre_ckpt['dim']} with this pretrained file."
            )

    device = pick_device(args.device)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    walks = load_walks(args.walks)
    if not walks:
        raise SystemExit("No walks found in input file.")

    word2idx, idx2word, freqs = build_vocab(walks, args.min_count)
    idx_sentences = [walk_to_indices(s, word2idx) for s in walks]
    idx_sentences = [s for s in idx_sentences if len(s) >= 2]
    if not idx_sentences:
        raise SystemExit("No sentences left after vocabulary filtering (need at least 2 known tokens).")

    vocab_size = len(idx2word)
    noise_p = noise_distribution(freqs).to(device)

    seed_base = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    probe_rng = random.Random(seed_base)
    if args.architecture == "skipgram":
        batches_probe = batch_pairs(idx_sentences, args.window, args.batch_size, probe_rng)
    else:
        batches_probe = batch_cbow_samples(idx_sentences, args.window, args.batch_size, probe_rng)
    num_batches = len(batches_probe)

    if args.architecture == "skipgram":
        model = SkipGramNeg(vocab_size, args.dim).to(device)
    else:
        model = CBOWNeg(vocab_size, args.dim).to(device)

    if pre_ckpt is not None:
        n_init = load_pretrained_embeddings(model, word2idx, ckpt=pre_ckpt, pretrained_path=args.pretrained)
        if n_init == 0:
            print(
                "Warning: no vocabulary overlap with --pretrained; training uses random init only.",
                flush=True,
            )
        else:
            print(f"Initialized {n_init} token rows from {args.pretrained}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    total_steps = args.epochs * num_batches
    arch_label = args.architecture.upper()
    desc = f"Word2Vec {arch_label} [{device.type.upper()}]  dim={args.dim}"

    pbar = tqdm(
        total=total_steps,
        desc=desc,
        unit="step",
        dynamic_ncols=True,
        colour="cyan",
        smoothing=0.08,
        mininterval=0.25,
        miniters=max(1, total_steps // 200),
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        postfix="",
    )

    for epoch in range(args.epochs):
        ep_rng = random.Random(seed_base + epoch * 1_000_003)
        if args.architecture == "skipgram":
            batches = batch_pairs(idx_sentences, args.window, args.batch_size, ep_rng)
        else:
            batches = batch_cbow_samples(idx_sentences, args.window, args.batch_size, ep_rng)
        epoch_loss = 0.0
        for batch in batches:
            if args.architecture == "skipgram":
                center_b, pos_b = batch
                center_b = center_b.to(device)
                pos_b = pos_b.to(device)
            else:
                ctx_b, mask_b, center_b = batch
                ctx_b = ctx_b.to(device)
                mask_b = mask_b.to(device)
                center_b = center_b.to(device)
                pos_b = center_b

            bsz = center_b.shape[0]
            neg_idx = torch.multinomial(noise_p, bsz * args.negative, replacement=True).view(
                bsz, args.negative
            )

            opt.zero_grad(set_to_none=True)
            if args.architecture == "skipgram":
                loss = model(center_b, pos_b, neg_idx)
            else:
                loss = model(ctx_b, mask_b, pos_b, neg_idx)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                epoch=f"{epoch + 1}/{args.epochs}",
                refresh=False,
            )
            pbar.update(1)

        avg = epoch_loss / max(num_batches, 1)
        tqdm.write(f"  ● epoch {epoch + 1}/{args.epochs}  mean loss: {avg:.6f}")

    pbar.close()

    # Final vectors: average input + output (common practice; similar to gensim syn0+syn1neg)
    with torch.no_grad():
        emb = (model.in_embed.weight + model.out_embed.weight) * 0.5
        emb_cpu = emb.cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": emb_cpu,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "dim": args.dim,
            "architecture": args.architecture,
            "window": args.window,
            "epochs": args.epochs,
            "device_trained": str(device),
            "pretrained_from": str(args.pretrained) if args.pretrained else None,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}  (vocab={vocab_size}, dim={args.dim})")


if __name__ == "__main__":
    main()
