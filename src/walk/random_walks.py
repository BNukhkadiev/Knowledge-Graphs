#!/usr/bin/env python3
"""
Random walks over an RDF Knowledge Graph in N-Triples (.nt) form.

Each triple <s> <p> <o> . contributes edges between s and o. By default the
entity graph is treated as undirected (common for embedding-style walks); use
--directed to follow only subject → object edges.

Modes
-----
* ``classic`` (default): sample many random walks over an undirected (or
  directed) entity graph; each output token is one node in N-Triples form.
* ``jrdf2vec-duplicate-free``: matches jRDF2Vec
  ``MemoryWalkGenerator.generateDuplicateFreeRandomWalksForEntity`` with
  ``WalkGenerationMode.RANDOM_WALKS_DUPLICATE_FREE`` — forward-only hops
  (subject → object), breadth-first expansion of triple chains, random trimming
  when the walk set exceeds ``--walks-per-entity``, one line per walk as
  ``entity p1 o1 p2 o2 ...`` (see ``--token-format``).

Output: one walk per line, space-separated tokens (see ``--token-format``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

# Same pattern as convert.py: bare identifiers in angle brackets are accepted.
TRIPLE_LINE = re.compile(r"^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*(?:#.*)?$")


@dataclass(frozen=True, slots=True)
class Triple:
    """One RDF object triple (subject, predicate, object) with bare inner strings."""

    subject: str
    predicate: str
    object: str


def iter_nt_triples(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                raise ValueError(f"Unrecognized N-Triples line: {line[:200]}")
            yield m.group(1), m.group(2), m.group(3)


def iter_nt_object_triples_lenient(path: Path):
    """IRI–IRI–IRI triples only; skips literals, comments, malformed lines (jRDF2Vec-style)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                continue
            yield m.group(1), m.group(2), m.group(3)


def nt_term(inner: str) -> str:
    return f"<{inner}>"


def build_adjacency(
    path: Path,
    *,
    directed: bool,
) -> tuple[dict[str, list[str]], list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in tqdm(
            f,
            desc="Scanning N-Triples (adjacency)",
            unit="line",
            dynamic_ncols=True,
            colour="cyan",
            mininterval=0.25,
        ):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                raise ValueError(f"Unrecognized N-Triples line: {line[:200]}")
            s, _p, o = m.group(1), m.group(2), m.group(3)
            all_nodes.add(s)
            all_nodes.add(o)
            if directed:
                neighbors[s].add(o)
            else:
                neighbors[s].add(o)
                neighbors[o].add(s)

    adj = {n: list(neighbors[n]) for n in tqdm(all_nodes, desc="Materializing neighbor lists", unit="node", dynamic_ncols=True, colour="cyan", mininterval=0.5)}
    return adj, list(all_nodes)


def random_walk(
    adj: dict[str, list[str]],
    *,
    start: str,
    num_nodes: int,
    rng: random.Random,
) -> list[str]:
    """Return a walk of `num_nodes` nodes starting at `start`. Stops early if stuck."""
    if num_nodes < 1:
        return []
    walk = [start]
    cur = start
    for _ in range(num_nodes - 1):
        nbrs = adj.get(cur)
        if not nbrs:
            break
        cur = rng.choice(nbrs)
        walk.append(cur)
    return walk


def build_forward_adjacency(path: Path) -> dict[str, list[Triple]]:
    """subject -> outgoing object triples (same as jRDF2Vec ``getObjectTriplesInvolvingSubject``)."""
    adj: dict[str, list[Triple]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in tqdm(
            f,
            desc="Scanning N-Triples (forward adjacency)",
            unit="line",
            dynamic_ncols=True,
            colour="cyan",
            mininterval=0.25,
        ):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                continue
            s, p, o = m.group(1), m.group(2), m.group(3)
            adj[s].append(Triple(s, p, o))
    return {k: v for k, v in adj.items()}


def collect_jrdf2vec_entities(adj: dict[str, list[Triple]]) -> list[str]:
    """All unique subjects ∪ objects of object triples (``MemoryEntitySelector``)."""
    entities: set[str] = set(adj.keys())
    for triples in adj.values():
        for t in triples:
            entities.add(t.object)
    return sorted(entities)


def _rng_for_entity(base_seed: int | None, entity: str) -> random.Random:
    """Stable per-entity RNG (for threaded runs) without relying on ``hash()`` salt."""
    s = base_seed if base_seed is not None else 0
    h = int(hashlib.md5(f"{s}:{entity}".encode()).hexdigest()[:16], 16)
    return random.Random(h)


def generate_duplicate_free_random_walks_for_entity(
    entity: str,
    number_of_walks: int,
    depth: int,
    adj_by_subject: dict[str, list[Triple]],
    rng: random.Random,
) -> list[list[Triple]]:
    """
    Port of jRDF2Vec ``MemoryWalkGenerator.generateDuplicateFreeRandomWalksForEntity``.

    Forward-only: each hop follows ``lastTriple.object`` as the next subject.
    After each depth step, if more than ``number_of_walks`` chains exist, trim
    with ``random.Random.sample`` — same uniform distribution over
    ``number_of_walks``-sized subsets as repeated ``pop(randrange(len))``, but
    not the same RNG stream (per-seed output can differ from a sequential-pop
    implementation).
    """
    walks: list[list[Triple]] = []
    for current_depth in range(depth):
        if current_depth == 0:
            neighbours = adj_by_subject.get(entity)
            if not neighbours:
                return []
            for t in neighbours:
                walks.append([t])
        else:
            new_walks: list[list[Triple]] = []
            for walk in walks:
                last_triple = walk[-1]
                next_iteration = adj_by_subject.get(last_triple.object)
                if next_iteration is not None:
                    for next_step in next_iteration:
                        new_walks.append(walk + [next_step])
                else:
                    new_walks.append(walk)
            walks = new_walks
        if len(walks) > number_of_walks:
            walks = rng.sample(walks, number_of_walks)
    return walks


def walk_triples_to_string(entity: str, triples: list[Triple], *, angled: bool) -> str:
    """Same string layout as jRDF2Vec ``Util.convertToStringWalks(walks, entity, ...)``."""
    parts: list[str] = [entity]
    for t in triples:
        parts.append(t.predicate)
        parts.append(t.object)
    if angled:
        return " ".join(nt_term(p) for p in parts)
    return " ".join(parts)


def run_jrdf2vec_duplicate_free(
    input_nt: Path,
    output_walks: Path,
    *,
    walks_per_entity: int,
    depth: int,
    threads: int,
    seed: int | None,
    token_format: str,
) -> None:
    adj = build_forward_adjacency(input_nt)
    if not adj:
        raise SystemExit("No object triples found; nothing to walk on.")
    entities = collect_jrdf2vec_entities(adj)
    angled = token_format == "angled"

    def lines_for_entity(entity: str) -> list[str]:
        rng = _rng_for_entity(seed, entity)
        triple_chains = generate_duplicate_free_random_walks_for_entity(
            entity,
            walks_per_entity,
            depth,
            adj,
            rng,
        )
        return [walk_triples_to_string(entity, chain, angled=angled) for chain in triple_chains]

    output_walks.parent.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, threads)
    if n_workers == 1:
        with output_walks.open("w", encoding="utf-8") as out:
            for entity in tqdm(
                entities,
                desc="jRDF2Vec duplicate-free [1 thread]",
                unit="entity",
                dynamic_ncols=True,
                colour="green",
            ):
                for line in lines_for_entity(entity):
                    out.write(line + "\n")
        return

    with output_walks.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(lines_for_entity, e): e for e in entities}
        for fut in tqdm(
            as_completed(futures),
            total=len(entities),
            desc=f"jRDF2Vec duplicate-free [{n_workers} threads]",
            unit="entity",
            dynamic_ncols=True,
            colour="green",
        ):
            for line in fut.result():
                out.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Random walks on an N-Triples knowledge graph.")
    p.add_argument("input_nt", type=Path, help="Path to input .nt file")
    p.add_argument("output_walks", type=Path, help="Path to output walks file (one walk per line)")
    p.add_argument(
        "--mode",
        choices=("classic", "jrdf2vec-duplicate-free"),
        default="classic",
        help="classic: random node sequences; jrdf2vec-duplicate-free: jRDF2Vec RANDOM_WALKS_DUPLICATE_FREE",
    )
    p.add_argument(
        "--num-walks",
        type=int,
        default=10_000,
        metavar="N",
        help="(classic) Total number of walks to sample (default: 10000)",
    )
    p.add_argument(
        "--walk-length",
        type=int,
        default=40,
        metavar="L",
        help="(classic) Number of nodes per walk (default: 40)",
    )
    p.add_argument(
        "--walks-per-entity",
        type=int,
        default=100,
        metavar="K",
        help="(jrdf2vec-duplicate-free) Max walks kept per entity after each depth trim (default: 100, jRDF2Vec default)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=4,
        metavar="D",
        help="(jrdf2vec-duplicate-free) Number of forward hops (default: 4, jRDF2Vec RDF2Vec default)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="(jrdf2vec-duplicate-free) Worker threads (default: max(1, cpu_count//2))",
    )
    p.add_argument(
        "--token-format",
        choices=("angled", "bare"),
        default="angled",
        help="(jrdf2vec-duplicate-free) angled: <s> <p> <o> ... (fits this repo's train/test); bare: jRDF2Vec file tokens without brackets",
    )
    p.add_argument(
        "--directed",
        action="store_true",
        help="(classic) Follow directed edges s→o only (default: undirected entity graph)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed (classic); also seeds per-entity trim in jrdf2vec mode")
    args = p.parse_args()

    if args.mode == "jrdf2vec-duplicate-free":
        if args.walks_per_entity < 1:
            raise SystemExit("--walks-per-entity must be at least 1")
        if args.depth < 1:
            raise SystemExit("--depth must be at least 1")
        run_jrdf2vec_duplicate_free(
            args.input_nt,
            args.output_walks,
            walks_per_entity=args.walks_per_entity,
            depth=args.depth,
            threads=args.threads,
            seed=args.seed,
            token_format=args.token_format,
        )
        return

    if args.num_walks < 1:
        raise SystemExit("--num-walks must be at least 1")
    if args.walk_length < 1:
        raise SystemExit("--walk-length must be at least 1")

    rng = random.Random(args.seed)
    adj, nodes = build_adjacency(args.input_nt, directed=args.directed)
    if not nodes:
        raise SystemExit("No triples found; nothing to walk on.")

    args.output_walks.parent.mkdir(parents=True, exist_ok=True)
    mode = "directed" if args.directed else "undirected"
    desc = f"Random walks [{mode}]  target_len={args.walk_length}"
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
    with args.output_walks.open("w", encoding="utf-8") as out:
        for _ in pbar:
            start = rng.choice(nodes)
            walk = random_walk(adj, start=start, num_nodes=args.walk_length, rng=rng)
            line = " ".join(nt_term(n) for n in walk)
            out.write(line + "\n")
            pbar.set_postfix(last_len=len(walk), refresh=False)


if __name__ == "__main__":
    main()
