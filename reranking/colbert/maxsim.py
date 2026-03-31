from __future__ import annotations

import torch


def maxsim_score(
    query_embeddings: torch.Tensor,
    query_mask: torch.Tensor,
    doc_embeddings: torch.Tensor,
    doc_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ColBERT-style MaxSim scores.

    query_embeddings: [Lq, D]
    query_mask:       [Lq]
    doc_embeddings:   [B, Ld, D]
    doc_mask:         [B, Ld]
    """
    scores = torch.einsum("qd,bpd->bqp", query_embeddings, doc_embeddings)
    scores = scores.masked_fill(~doc_mask[:, None, :], -1e9)
    max_scores = scores.max(dim=2).values
    max_scores = max_scores * query_mask[None, :].float()
    return max_scores.sum(dim=1)
