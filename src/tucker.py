"""TuckER: Tucker factorization for knowledge graph completion.

Balažević, Allen, Hospedales (2019). "TuckER: Tensor Factorization for Knowledge Graph
Completion." https://arxiv.org/abs/1901.09590

The core tensor ``W`` has shape ``(d_e, d_r, d_e)`` over modes
(subject entity, relation, object entity). The logit for a triple ``(h, r, t)`` is
``sum_{i,j,k} W[i,j,k] * e_h[i] * r_r[j] * e_t[k]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TuckER(nn.Module):
    """Tucker decomposition of the binary KG tensor with learned entity/relation factors."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int,
        relation_dim: int,
        *,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.entity_emb = nn.Embedding(num_entities, entity_dim)
        self.relation_emb = nn.Embedding(num_relations, relation_dim)
        # Modes: (subject, relation, object) -> (d_e, d_r, d_e)
        self.core = nn.Parameter(torch.empty(entity_dim, relation_dim, entity_dim))

        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound_e = 1.0 / math.sqrt(self.entity_dim)
        bound_r = 1.0 / math.sqrt(self.relation_dim)
        nn.init.uniform_(self.entity_emb.weight, -bound_e, bound_e)
        nn.init.uniform_(self.relation_emb.weight, -bound_r, bound_r)
        nn.init.xavier_uniform_(self.core)

    def _embed_triple(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_h = self.input_dropout(self.entity_emb(h))
        e_t = self.input_dropout(self.entity_emb(t))
        r_vec = self.input_dropout(self.relation_emb(r))
        return e_h, r_vec, e_t

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Logits for triples ``(h, r, t)`` (integer indices). Shape ``(batch,)``."""
        e_h, r_vec, e_t = self._embed_triple(h, r, t)
        # M[b,i,k] = sum_j W[i,j,k] * r_vec[b,j]
        m = torch.einsum("ijk,bj->bik", self.core, r_vec)
        m = self.hidden_dropout(m)
        return torch.einsum("bik,bi,bk->b", m, e_h, e_t)

    def score_all_tails(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Scores for all tails: shape ``(batch, num_entities)`` for each ``(h, r)``."""
        e_h = self.input_dropout(self.entity_emb(h))
        r_vec = self.input_dropout(self.relation_emb(r))
        m = torch.einsum("ijk,bj->bik", self.core, r_vec)
        m = self.hidden_dropout(m)
        e_all = self.entity_emb.weight
        return torch.einsum("bik,bi,nk->bn", m, e_h, e_all)

    def score_all_heads(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Scores for all heads: shape ``(batch, num_entities)`` for each ``(t, r)``."""
        e_t = self.input_dropout(self.entity_emb(t))
        r_vec = self.input_dropout(self.relation_emb(r))
        m = torch.einsum("ijk,bj->bik", self.core, r_vec)
        m = self.hidden_dropout(m)
        e_all = self.entity_emb.weight
        return torch.einsum("bik,ni,bk->bn", m, e_all, e_t)
