#!/usr/bin/env python3
"""
2D PCA scatter of RDF2Vec embeddings colored by class (rdf:type).

Reads instance types from ``graph.nt`` (``rdf:type`` triples) and optionally refines
labels using ``rdfs:subClassOf`` in ``ontology.nt`` (most-specific class among
asserted types, same rule as MASCHInE init).

Embeddings: ``.pt`` (``train_word2vec`` / ``rdf2vec_final.pt``), gensim ``.kv``,
or full gensim ``Word2Vec`` ``.model``.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.evaluate_embeddings import load_embeddings_checkpoint
from src.protograph.maschine_init import (
    OWL_CLASS,
    RDF_TYPE,
    _most_specific_types,
)
from src.protograph.protograph_gen import RDFS_SUBCLASS, iter_nt_iris
from src.walk.random_walks import nt_term


def load_subclass_parents(ontology_path: Path) -> dict[str, set[str]]:
    """child class IRI -> direct parent class IRIs (``rdfs:subClassOf``)."""
    parents: dict[str, set[str]] = defaultdict(set)
    for s, p, o in iter_nt_iris(ontology_path):
        if p == RDFS_SUBCLASS:
            parents[s].add(o)
    return dict(parents)


def load_rdf_types_from_graph(graph_path: Path) -> dict[str, set[str]]:
    """Subject IRI -> asserted class IRIs (``rdf:type``, excluding ``owl:Class`` declarations)."""
    raw: dict[str, set[str]] = defaultdict(set)
    for s, p, o in iter_nt_iris(graph_path):
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        raw[s].add(o)
    return dict(raw)


def entity_class_label(
    types: set[str],
    parents: dict[str, set[str]],
) -> str | None:
    """Single class IRI string for plotting; uses most-specific types when ontology is available."""
    if not types:
        return None
    mst = _most_specific_types(types, parents)
    if not mst:
        return sorted(types)[0]
    mst_sorted = sorted(mst)
    return mst_sorted[0]


def shorten_iri(iri: str, max_len: int = 48) -> str:
    """Compact legend labels."""
    if len(iri) <= max_len:
        return iri
    return iri[: max_len - 3] + "..."


def load_embeddings_any(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Same as ``evaluate_embeddings.load_embeddings_checkpoint`` (``.pt``, ``.kv``, ``.model``)."""
    return load_embeddings_checkpoint(path)


def resolve_vocab_key(entity_inner: str, word2idx: dict[str, int]) -> str | None:
    """Map graph subject IRI to Word2Vec vocabulary key (angled ``<...>``)."""
    angled = nt_term(entity_inner)
    if angled in word2idx:
        return angled
    if entity_inner in word2idx:
        return entity_inner
    return None


EXTRA_CLASS_RE = re.compile(r"^EXTRA_I_FOR_CLASS_(C_tc07_\d+)_")


def infer_class_from_token(inner: str) -> str | None:
    """Optional fallback: synthetic ``EXTRA_I_FOR_CLASS_*`` instance names -> class id."""
    m = EXTRA_CLASS_RE.match(inner)
    if m:
        return m.group(1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PCA plot of RDF2Vec embeddings colored by rdf:type (from graph.nt).",
    )
    ap.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Instance N-Triples graph with rdf:type for entities (e.g. graph.nt).",
    )
    ap.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Optional: N-Triples with rdfs:subClassOf to pick most-specific class per entity.",
    )
    ap.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Embeddings: .pt, .kv, or gensim Word2Vec .model",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rdf2vec_pca.png"),
        help="Output PNG path",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Random subsample size for plotting (0 = all typed entities in vocab).",
    )
    ap.add_argument(
        "--max-classes",
        type=int,
        default=40,
        help="Keep only this many most frequent classes (legend readability).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling")
    ap.add_argument(
        "--extra-i-fallback",
        action="store_true",
        help="If no rdf:type for a vocab token, try EXTRA_I_FOR_CLASS_* -> class (synthetic dumps).",
    )
    args = ap.parse_args()

    if not args.graph.is_file():
        raise SystemExit(f"Graph not found: {args.graph}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.ontology is not None and not args.ontology.is_file():
        raise SystemExit(f"Ontology not found: {args.ontology}")

    parents = load_subclass_parents(args.ontology) if args.ontology is not None else {}
    raw_types = load_rdf_types_from_graph(args.graph)

    emb, word2idx = load_embeddings_any(args.checkpoint)

    entity_to_label: dict[str, str] = {}
    for ent, types in raw_types.items():
        lab = entity_class_label(types, parents)
        if lab is not None:
            entity_to_label[ent] = lab

    rows: list[int] = []
    labels: list[str] = []

    for inner, lab in entity_to_label.items():
        key = resolve_vocab_key(inner, word2idx)
        if key is None:
            continue
        rows.append(word2idx[key])
        labels.append(lab)

    if args.extra_i_fallback:
        for token in word2idx:
            if len(token) < 2 or token[0] != "<" or token[-1] != ">":
                continue
            inner = token[1:-1]
            if inner in entity_to_label:
                continue
            c = infer_class_from_token(inner)
            if c is None:
                continue
            types = {c}
            lab = entity_class_label(types, parents)
            if lab is None:
                continue
            rows.append(word2idx[token])
            labels.append(lab)

    if not rows:
        raise SystemExit(
            "No embeddings matched entities with rdf:type in --graph (and optional fallback). "
            "Check that graph.nt contains rdf:type and that instance IRIs match walk vocabulary."
        )

    rng = np.random.default_rng(args.seed)
    n = len(rows)
    if args.max_points > 0 and n > args.max_points:
        pick = rng.choice(n, size=args.max_points, replace=False)
        rows = [rows[i] for i in pick]
        labels = [labels[i] for i in pick]

    label_counts = Counter(labels)
    if args.max_classes > 0 and len(label_counts) > args.max_classes:
        keep = {lab for lab, _ in label_counts.most_common(args.max_classes)}
        filt_r: list[int] = []
        filt_l: list[str] = []
        other = 0
        for r, lab in zip(rows, labels, strict=True):
            if lab in keep:
                filt_r.append(r)
                filt_l.append(lab)
            else:
                filt_r.append(r)
                filt_l.append("__other__")
                other += 1
        rows, labels = filt_r, filt_l
        if other > 0:
            label_counts = Counter(labels)

    X = emb[np.asarray(rows, dtype=np.int64)]
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(X)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    cmap = plt.colormaps["tab20"].resampled(max(n_classes, 1))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    norm = max(n_classes - 1, 1)
    point_colors = [cmap(label_to_idx[lab] / norm) for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Z[:, 0], Z[:, 1], c=point_colors, s=8, alpha=0.65, linewidths=0, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.set_title("RDF2Vec embeddings — PCA by class (rdf:type)")
    ax.grid(True, alpha=0.25)

    handles = []
    for lab in unique_labels:
        color = cmap(label_to_idx[lab] / norm)
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=shorten_iri(lab)))
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"Points: {len(rows)}  |  Classes (after filter): {n_classes}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f}")


if __name__ == "__main__":
    main()
