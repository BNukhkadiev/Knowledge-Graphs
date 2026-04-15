#!/usr/bin/env python3
"""
Random walks guided by the 1-Weisfeiler–Lehman (1-WL) color refinement.

Builds the same entity graph as ``random_walks.py`` (N-Triples → adjacency),
runs iterative multiset hashing on neighbor colors (standard 1-WL), then
samples walks. Neighbor choice can be uniform (like plain random walks) or
biased toward rarer WL colors (inverse color frequency) so the walk explores
more diverse structural contexts.

Output: one walk per line, space-separated tokens (see ``--tokens``).
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from random_walks import build_adjacency, nt_term, random_walk


def _stable_hash(parts: tuple[str, ...]) -> str:
    """Short stable digest for WL labels (cross-run deterministic)."""
    h = hashlib.sha256("||".join(parts).encode()).hexdigest()
    return h[:16]


def initial_colors(
    nodes: list[str],
    adj: dict[str, list[str]],
    *,
    scheme: str,
) -> dict[str, str]:
    if scheme == "iri":
        return {v: _stable_hash((v,)) for v in nodes}
    if scheme == "degree":
        return {v: _stable_hash((str(len(adj.get(v, []))),)) for v in nodes}
    raise ValueError(f"unknown initial scheme: {scheme}")


def wl_refinement_step(
    nodes: list[str],
    adj: dict[str, list[str]],
    colors: dict[str, str],
) -> dict[str, str]:
    new: dict[str, str] = {}
    for v in nodes:
        c_v = colors[v]
        nbr_colors = tuple(sorted(colors[u] for u in adj.get(v, [])))
        new[v] = _stable_hash((c_v, str(nbr_colors)))
    return new


def compute_wl_colors(
    nodes: list[str],
    adj: dict[str, list[str]],
    *,
    iterations: int,
    initial_scheme: str,
) -> dict[str, str]:
    colors = initial_colors(nodes, adj, scheme=initial_scheme)
    for _ in range(iterations):
        colors = wl_refinement_step(nodes, adj, colors)
    return colors


def compute_wl_colors_until_converged(
    nodes: list[str],
    adj: dict[str, list[str]],
    *,
    initial_scheme: str,
    max_iter: int,
) -> tuple[dict[str, str], int]:
    colors = initial_colors(nodes, adj, scheme=initial_scheme)
    for it in range(max_iter):
        nxt = wl_refinement_step(nodes, adj, colors)
        if nxt == colors:
            return colors, it
        colors = nxt
    return colors, max_iter


def inverse_color_weights(colors: dict[str, str]) -> dict[str, float]:
    counts = Counter(colors.values())
    return {c: 1.0 / cnt for c, cnt in counts.items()}


def wl_biased_random_walk(
    adj: dict[str, list[str]],
    *,
    start: str,
    num_nodes: int,
    rng: random.Random,
    colors: dict[str, str],
    color_weight: dict[str, float],
) -> list[str]:
    """Random walk: next hop weighted by ``color_weight[colors[neighbor]]``."""
    if num_nodes < 1:
        return []
    walk = [start]
    cur = start
    for _ in range(num_nodes - 1):
        nbrs = adj.get(cur)
        if not nbrs:
            break
        weights = [color_weight[colors[u]] for u in nbrs]
        total = sum(weights)
        if total <= 0:
            cur = rng.choice(nbrs)
        else:
            r = rng.random() * total
            acc = 0.0
            for u, w in zip(nbrs, weights, strict=True):
                acc += w
                if r <= acc:
                    cur = u
                    break
            else:
                cur = nbrs[-1]
        walk.append(cur)
    return walk


def format_walk_tokens(
    walk: list[str],
    colors: dict[str, str],
    *,
    mode: str,
) -> str:
    if mode == "nodes":
        return " ".join(nt_term(n) for n in walk)
    if mode == "wl":
        return " ".join(colors[v] for v in walk)
    if mode == "node_wl":
        parts: list[str] = []
        for v in walk:
            parts.append(nt_term(v))
            parts.append(colors[v])
        return " ".join(parts)
    raise ValueError(f"unknown token mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Weisfeiler–Lehman–aware random walks on an N-Triples knowledge graph.",
    )
    p.add_argument("input_nt", type=Path, help="Path to input .nt file")
    p.add_argument("output_walks", type=Path, help="Path to output walks file (one walk per line)")
    p.add_argument(
        "--num-walks",
        type=int,
        default=10_000,
        metavar="N",
        help="Total walks with random starting nodes (no upper limit; ignored if --walks-per-entity is set)",
    )
    p.add_argument(
        "--walks-per-entity",
        type=int,
        default=None,
        metavar="K",
        help="If set, emit exactly K walks starting from each graph node; mutually exclusive with counting via --num-walks",
    )
    p.add_argument(
        "--walk-length",
        type=int,
        default=40,
        metavar="L",
        help="Number of nodes per walk (default: 40; max 20 when --enforce-limits)",
    )
    p.add_argument(
        "--enforce-limits",
        action="store_true",
        help="Require walk-length<=20 (for capped walk corpora)",
    )
    p.add_argument(
        "--wl-iterations",
        type=int,
        default=5,
        metavar="K",
        help="Number of 1-WL refinement rounds (default: 5); ignored if --until-converged",
    )
    p.add_argument(
        "--until-converged",
        action="store_true",
        help="Run 1-WL until colors stabilize (or --max-wl-iterations)",
    )
    p.add_argument(
        "--max-wl-iterations",
        type=int,
        default=100,
        metavar="M",
        help="Cap for --until-converged (default: 100)",
    )
    p.add_argument(
        "--initial-labels",
        choices=("iri", "degree"),
        default="iri",
        help="Initial WL colors: iri=hash of node IRI; degree=degree multiset (default: iri)",
    )
    p.add_argument(
        "--neighbor-policy",
        choices=("uniform", "inverse-wl-freq"),
        default="uniform",
        help="uniform: like random walks; inverse-wl-freq: prefer neighbors with rarer WL colors",
    )
    p.add_argument(
        "--tokens",
        choices=("nodes", "wl", "node_wl"),
        default="nodes",
        help="nodes: <iri> ...; wl: WL hash only; node_wl: <iri> then WL hash per step (default: nodes)",
    )
    p.add_argument(
        "--directed",
        action="store_true",
        help="Directed edges s→o only (default: undirected entity graph)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = p.parse_args()

    if args.walks_per_entity is not None and args.walks_per_entity < 1:
        raise SystemExit("--walks-per-entity must be at least 1")
    if args.walks_per_entity is None and args.num_walks < 1:
        raise SystemExit("--num-walks must be at least 1")
    if args.walk_length < 1:
        raise SystemExit("--walk-length must be at least 1")
    if args.enforce_limits:
        if args.walk_length > 20:
            raise SystemExit("--enforce-limits: --walk-length must be at most 20")
    if not args.until_converged and args.wl_iterations < 1:
        raise SystemExit("--wl-iterations must be at least 1")
    if args.max_wl_iterations < 1:
        raise SystemExit("--max-wl-iterations must be at least 1")

    rng = random.Random(args.seed)
    adj, nodes = build_adjacency(args.input_nt, directed=args.directed)
    if not nodes:
        raise SystemExit("No triples found; nothing to walk on.")

    if args.until_converged:
        colors, conv_it = compute_wl_colors_until_converged(
            nodes,
            adj,
            initial_scheme=args.initial_labels,
            max_iter=args.max_wl_iterations,
        )
    else:
        colors = compute_wl_colors(
            nodes,
            adj,
            iterations=args.wl_iterations,
            initial_scheme=args.initial_labels,
        )
        conv_it = None

    color_w = inverse_color_weights(colors) if args.neighbor_policy == "inverse-wl-freq" else None

    def sample_one_walk(start: str) -> list[str]:
        if args.neighbor_policy == "uniform":
            return random_walk(adj, start=start, num_nodes=args.walk_length, rng=rng)
        assert color_w is not None
        return wl_biased_random_walk(
            adj,
            start=start,
            num_nodes=args.walk_length,
            rng=rng,
            colors=colors,
            color_weight=color_w,
        )

    args.output_walks.parent.mkdir(parents=True, exist_ok=True)
    mode = "directed" if args.directed else "undirected"
    wl_note = f"conv@{conv_it}" if conv_it is not None else f"it={args.wl_iterations}"
    desc = f"WL walks [{mode}] {wl_note} policy={args.neighbor_policy}"

    with args.output_walks.open("w", encoding="utf-8") as out:
        if args.walks_per_entity is not None:
            total = len(nodes) * args.walks_per_entity
            pbar = tqdm(
                range(total),
                desc=f"{desc} [per-entity x{args.walks_per_entity}]",
                unit="walk",
                dynamic_ncols=True,
                colour="green",
                smoothing=0.05,
                mininterval=0.2,
                miniters=max(1, total // 300),
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            for start in nodes:
                for _ in range(args.walks_per_entity):
                    walk = sample_one_walk(start)
                    line = format_walk_tokens(walk, colors, mode=args.tokens)
                    out.write(line + "\n")
                    pbar.update(1)
                    pbar.set_postfix(last_len=len(walk), refresh=False)
            pbar.close()
        else:
            pbar = tqdm(
                range(args.num_walks),
                desc=desc,
                unit="walk",
                dynamic_ncols=True,
                colour="green",
                smoothing=0.05,
                mininterval=0.2,
                miniters=max(1, args.num_walks // 300),
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            for _ in pbar:
                start = rng.choice(nodes)
                walk = sample_one_walk(start)
                line = format_walk_tokens(walk, colors, mode=args.tokens)
                out.write(line + "\n")
                pbar.set_postfix(last_len=len(walk), refresh=False)


if __name__ == "__main__":
    main()
