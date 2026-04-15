"""
MASCHInE-style transfer of protograph (P2) embeddings to instance RDF2Vec vocabulary
(Hubert et al., arXiv:2306.03659, Section 3.2).

Builds entity → most-specific class mapping from ontology, then initializes new
Word2Vec rows from pretrained class vectors (mean if multiple).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from src.protograph.protograph_gen import iter_nt_iris
from src.walk.random_walks import nt_term

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

# Synthetic DLCC-style extra individuals: EXTRA_I_FOR_CLASS_C_tc07_152_640 -> C_tc07_152
EXTRA_CLASS_RE = re.compile(r"^EXTRA_I_FOR_CLASS_(C_tc07_\d+)_")


def _ancestors_generalizations(node: str, parents: dict[str, set[str]]) -> set[str]:
    """All classes reachable from ``node`` by walking up ``subClassOf`` (including ``node``)."""
    out: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(parents.get(cur, ()))
    return out


def _most_specific_types(asserted: set[str], parents: dict[str, set[str]]) -> list[str]:
    """
    Among asserted class IRIs (inner strings), keep those not strictly generalized
    by another asserted type (MASCHInE: most specific class(es)).
    """
    if not asserted:
        return []
    anc = {t: _ancestors_generalizations(t, parents) for t in asserted}
    minimal: list[str] = []
    for t in asserted:
        more_specific_exists = False
        for t_other in asserted:
            if t_other == t:
                continue
            if t in anc[t_other]:
                more_specific_exists = True
                break
        if not more_specific_exists:
            minimal.append(t)
    return minimal


def load_ontology_mapping(ontology_nt: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    Returns:
        parents: class IRI inner -> direct parent class IRI inners (subClassOf edges child -> parents)
        entity_types: entity IRI inner -> asserted rdf:type class inners (owl:Class declarations skipped)
    """
    parents: dict[str, set[str]] = defaultdict(set)
    raw_types: dict[str, set[str]] = defaultdict(set)

    for s, p, o in iter_nt_iris(ontology_nt):
        if p == RDFS_SUBCLASS:
            parents[s].add(o)
            continue
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        raw_types[s].add(o)

    entity_types: dict[str, list[str]] = {}
    for ent, types in raw_types.items():
        mst = _most_specific_types(types, parents)
        entity_types[ent] = mst

    return dict(parents), entity_types


def _class_from_extra_name(entity_inner: str) -> str | None:
    m = EXTRA_CLASS_RE.match(entity_inner)
    if m:
        return m.group(1)
    return None


def build_entity_to_class_tokens(
    parents: dict[str, set[str]],
    entity_types: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Map entity IRI inner string -> list of class tokens ``<C_...>`` for averaging.
    Uses rdf:type when present; otherwise EXTRA_I_* name pattern.
    """
    out: dict[str, list[str]] = {}
    for ent, cls_list in entity_types.items():
        if cls_list:
            out[ent] = [nt_term(c) for c in cls_list]
    for ent in entity_types:
        if ent in out:
            continue
        c = _class_from_extra_name(ent)
        if c is not None:
            mst = _most_specific_types({c}, parents)
            out[ent] = [nt_term(x) for x in mst]
    return out


def token_is_instance_entity(token: str) -> bool:
    if len(token) < 2 or token[0] != "<" or token[-1] != ">":
        return False
    inner = token[1:-1]
    if inner.startswith("I_tc07_"):
        return True
    if inner.startswith("EXTRA_I_FOR_CLASS_"):
        return True
    return False


def apply_maschine_initialization(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    ontology_nt: Path,
) -> tuple[int, int, int]:
    """
    After ``build_vocab(..., update=True)``, overwrite new entity rows using pretrained
    class vectors. Copies ``syn1neg`` rows to match ``wv.vectors`` for new words.

    Returns:
        (n_entity_initialized, n_relation_copied, n_left_random)
    """
    parents, entity_types = load_ontology_mapping(ontology_nt)
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)

    wv = model.wv
    n_entity = 0
    n_rel = 0
    n_random = 0

    for token in wv.index_to_key:
        if token in stage1_vectors:
            continue

        inner = token[1:-1] if len(token) >= 2 and token[0] == "<" else token
        class_tokens: list[str] | None = ent_to_classes.get(inner)

        if class_tokens is None and token_is_instance_entity(token):
            c = _class_from_extra_name(inner)
            if c is not None:
                mst = _most_specific_types({c}, parents)
                class_tokens = [nt_term(x) for x in mst]

        if class_tokens:
            parts = [stage1_vectors[t] for t in class_tokens if t in stage1_vectors]
            if parts:
                mean = np.mean(parts, axis=0).astype(np.float32, copy=False)
                idx = wv.key_to_index[token]
                wv.vectors[idx] = mean
                if model.syn1neg is not None:
                    model.syn1neg[idx] = mean.copy()
                n_entity += 1
                continue

        if inner.startswith("P_"):
            tok = nt_term(inner)
            if tok in stage1_vectors:
                vec = stage1_vectors[tok]
                idx = wv.key_to_index[token]
                wv.vectors[idx] = vec.copy()
                if model.syn1neg is not None:
                    model.syn1neg[idx] = vec.copy()
                n_rel += 1
                continue

        n_random += 1

    return n_entity, n_rel, n_random
