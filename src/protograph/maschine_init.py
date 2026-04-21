"""
MASCHInE-style transfer of protograph (P2) embeddings to instance RDF2Vec vocabulary
(Hubert et al., arXiv:2306.03659, Section 3.2).

Builds entity → most-specific class mapping from ontology, then initializes new
Word2Vec rows from pretrained class vectors. Strategies: most-specific only (default),
unweighted mean over the full superclass closure, or distance-decayed mean toward roots.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from src.protograph.protograph_gen import iter_rdf_iris
from src.walk.random_walks import nt_term

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
OWL_NS = "http://www.w3.org/2002/07/owl#"
OWL_CLASS = f"{OWL_NS}Class"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

# Subjects with any of these as asserted rdf:type are RDF properties, not individuals.
RDF_PROPERTY_TYPE_IRIS: frozenset[str] = frozenset(
    {
        RDF_PROPERTY,
        f"{OWL_NS}ObjectProperty",
        f"{OWL_NS}DatatypeProperty",
        f"{OWL_NS}AnnotationProperty",
        f"{OWL_NS}FunctionalProperty",
        f"{OWL_NS}InverseFunctionalProperty",
        f"{OWL_NS}SymmetricProperty",
        f"{OWL_NS}AsymmetricProperty",
        f"{OWL_NS}ReflexiveProperty",
        f"{OWL_NS}IrreflexiveProperty",
        f"{OWL_NS}TransitiveProperty",
    }
)

# Synthetic DLCC-style extra individuals: EXTRA_I_FOR_CLASS_C_tc07_152_640 -> C_tc07_152
EXTRA_CLASS_RE = re.compile(r"^EXTRA_I_FOR_CLASS_(C_tc07_\d+)_")

MASCHINE_CLASS_INIT_STRATEGIES = frozenset(
    {"most_specific", "ancestor_mean", "ancestor_weighted"},
)


def min_hops_upward_from_roots(
    roots: list[str],
    parents: dict[str, set[str]],
) -> dict[str, int]:
    """For each class reachable via ``rdfs:subClassOf`` upward from ``roots``, shortest hop count from the nearest root."""
    if not roots:
        return {}
    dist: dict[str, int] = {}
    q: deque[str] = deque()
    for r in roots:
        if r not in dist:
            dist[r] = 0
            q.append(r)
    while q:
        u = q.popleft()
        du = dist[u]
        for par in parents.get(u, ()):
            nd = du + 1
            if par not in dist or nd < dist[par]:
                dist[par] = nd
                q.append(par)
    return dist


def init_vector_from_class_roots(
    roots_inner: list[str],
    parents: dict[str, set[str]],
    stage1_vectors: dict[str, np.ndarray],
    *,
    strategy: str,
    ancestor_decay: float = 0.5,
) -> np.ndarray | None:
    """
    Combine pretrained class vectors for initializing an instance row.

    * ``most_specific`` — mean of vectors for ``roots_inner`` (after MASCHInE filtering).
    * ``ancestor_mean`` — mean over ``roots_inner`` and all transitive superclasses.
    * ``ancestor_weighted`` — weighted mean over that same closure with weight ``ancestor_decay**d``,
      where ``d`` is hop distance upward from the nearest root.
    """
    if strategy not in MASCHINE_CLASS_INIT_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(MASCHINE_CLASS_INIT_STRATEGIES)}")
    if not roots_inner:
        return None
    if strategy == "most_specific":
        tokens = [nt_term(r) for r in roots_inner]
        parts = [stage1_vectors[t] for t in tokens if t in stage1_vectors]
        if not parts:
            return None
        return np.mean(parts, axis=0).astype(np.float32, copy=False)

    closure = min_hops_upward_from_roots(roots_inner, parents)
    if not closure:
        return None
    if strategy == "ancestor_mean":
        parts = [stage1_vectors[nt_term(c_inner)] for c_inner in closure if nt_term(c_inner) in stage1_vectors]
        if not parts:
            return None
        return np.mean(parts, axis=0).astype(np.float32, copy=False)

    if strategy == "ancestor_weighted":
        if not (0.0 < ancestor_decay <= 1.0):
            raise ValueError("ancestor_weighted requires 0 < --maschine-ancestor-decay <= 1")
        w_accum = 0.0
        vec_accum: np.ndarray | None = None
        for c_inner, d in closure.items():
            tok = nt_term(c_inner)
            if tok not in stage1_vectors:
                continue
            w = ancestor_decay**d
            v = stage1_vectors[tok]
            if vec_accum is None:
                vec_accum = (w * v).astype(np.float32, copy=False)
            else:
                vec_accum = vec_accum + w * v
            w_accum += w
        if vec_accum is None or w_accum <= 0.0:
            return None
        return (vec_accum / w_accum).astype(np.float32, copy=False)
    return None


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


def _subject_is_rdf_property(raw_types: dict[str, set[str]], ent: str) -> bool:
    """True if ``ent`` is declared in the ontology as an RDF/OWL property (not an individual)."""
    ts = raw_types.get(ent)
    if not ts:
        return False
    return bool(ts & RDF_PROPERTY_TYPE_IRIS)


def load_ontology_mapping(ontology_nt: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    Returns:
        parents: class IRI inner -> direct parent class IRI inners (subClassOf edges child -> parents)
        entity_types: entity IRI inner -> asserted rdf:type class inners (owl:Class declarations skipped)
    """
    parents: dict[str, set[str]] = defaultdict(set)
    raw_types: dict[str, set[str]] = defaultdict(set)

    for s, p, o in iter_rdf_iris(ontology_nt):
        if p == RDFS_SUBCLASS:
            parents[s].add(o)
            continue
        if p != RDF_TYPE:
            continue
        if o == OWL_CLASS:
            continue
        raw_types[s].add(o)

    raw_dict = dict(raw_types)
    entity_types: dict[str, list[str]] = {}
    for ent, types in raw_dict.items():
        mst = _most_specific_types(types, parents)
        entity_types[ent] = mst

    for ent in list(entity_types.keys()):
        if _subject_is_rdf_property(raw_dict, ent):
            del entity_types[ent]

    return dict(parents), entity_types


def _class_from_extra_name(entity_inner: str) -> str | None:
    m = EXTRA_CLASS_RE.match(entity_inner)
    if m:
        return m.group(1)
    return None


def instance_class_mapping_from_ontology(ontology_nt: Path) -> dict[str, dict[str, list[str]]]:
    """
    For each individual in ``ontology_nt`` with ``rdf:type`` (or derivable EXTRA_I_* typing),
    return most-specific class IRIs and the corresponding walk tokens used for MASCHInE init.

    Keys are **entity IRI strings** (same as N-Triples subject inner, without angle brackets).
    """
    parents, entity_types = load_ontology_mapping(ontology_nt)
    ent_to_tokens = build_entity_to_class_tokens(parents, entity_types)
    return {
        ent: {
            "most_specific_class_iris": list(entity_types[ent]),
            "class_tokens": list(ent_to_tokens.get(ent, [])),
        }
        for ent in entity_types
    }


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


def transfer_predicate_embeddings_from_pretrained(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    *,
    new_token_init: dict[str, str] | None = None,
) -> int:
    """
    For each vocabulary token not present in ``stage1_vectors`` whose inner IRI starts with
    ``P_`` (DLCC relation/predicate walks), copy the pretrained row if that token existed in stage 1.

    Use when MASCHInE class init is disabled but RDF2Vec should still reuse relation vectors
    from protograph pretraining. Idempotent for rows already matching ``stage1_vectors``.

    Returns the number of predicate rows overwritten.
    """
    wv = model.wv
    n = 0
    for token in wv.index_to_key:
        if token in stage1_vectors:
            continue
        inner = token[1:-1] if len(token) >= 2 and token[0] == "<" else token
        if not inner.startswith("P_"):
            continue
        tok = nt_term(inner)
        if tok not in stage1_vectors:
            continue
        vec = stage1_vectors[tok]
        idx = wv.key_to_index[token]
        wv.vectors[idx] = vec.copy()
        if model.syn1neg is not None:
            model.syn1neg[idx] = vec.copy()
        n += 1
        if new_token_init is not None:
            new_token_init[token] = "predicate_transfer"
    return n


def apply_maschine_initialization(
    model: Word2Vec,
    stage1_vectors: dict[str, np.ndarray],
    ontology_nt: Path,
    *,
    strategy: str = "most_specific",
    ancestor_decay: float = 0.5,
    new_token_init: dict[str, str] | None = None,
) -> tuple[int, int, int]:
    """
    After ``build_vocab(..., update=True)``, overwrite new entity rows using pretrained
    class vectors. Copies ``syn1neg`` rows to match ``wv.vectors`` for new words.

    ``strategy`` selects how class vectors are combined (see ``init_vector_from_class_roots``).

    If ``new_token_init`` is provided, it is filled for each vocabulary token *not* in
    ``stage1_vectors`` with one of: ``maschine_class_mean``, ``maschine_ancestor_mean``,
    ``maschine_ancestor_weighted``, ``maschine_relation_copy``, ``random``.

    Returns:
        (n_entity_initialized, n_relation_copied, n_left_random)
    """
    if strategy not in MASCHINE_CLASS_INIT_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(MASCHINE_CLASS_INIT_STRATEGIES)}")

    parents, entity_types = load_ontology_mapping(ontology_nt)
    ent_to_classes = build_entity_to_class_tokens(parents, entity_types)

    init_labels = {
        "most_specific": "maschine_class_mean",
        "ancestor_mean": "maschine_ancestor_mean",
        "ancestor_weighted": "maschine_ancestor_weighted",
    }
    init_label = init_labels[strategy]

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

        roots_inner: list[str] = []
        if class_tokens:
            roots_inner = [t[1:-1] for t in class_tokens if len(t) >= 2 and t[0] == "<" and t[-1] == ">"]

        init_vec: np.ndarray | None = None
        if roots_inner:
            init_vec = init_vector_from_class_roots(
                roots_inner,
                parents,
                stage1_vectors,
                strategy=strategy,
                ancestor_decay=ancestor_decay,
            )

        if init_vec is not None:
            idx = wv.key_to_index[token]
            wv.vectors[idx] = init_vec
            if model.syn1neg is not None:
                model.syn1neg[idx] = init_vec.copy()
            n_entity += 1
            if new_token_init is not None:
                new_token_init[token] = init_label
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
                if new_token_init is not None:
                    new_token_init[token] = "maschine_relation_copy"
                continue

        n_random += 1
        if new_token_init is not None:
            new_token_init[token] = "random"

    return n_entity, n_rel, n_random
