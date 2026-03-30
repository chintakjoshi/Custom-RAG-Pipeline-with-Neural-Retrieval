from __future__ import annotations

import math
from statistics import mean


def sort_run(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def dcg_at_k(relevance: list[int], k: int) -> float:
    total = 0.0
    for index, gain in enumerate(relevance[:k], start=1):
        total += (2**gain - 1) / math.log2(index + 1)
    return total


def ndcg_at_k(relevant_docs: dict[str, int], ranked_docs: list[tuple[str, float]], k: int) -> float:
    actual = [relevant_docs.get(doc_id, 0) for doc_id, _ in ranked_docs[:k]]
    ideal = sorted(relevant_docs.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(actual, k) / ideal_dcg


def reciprocal_rank_at_k(
    relevant_docs: dict[str, int],
    ranked_docs: list[tuple[str, float]],
    k: int,
) -> float:
    for rank, (doc_id, _) in enumerate(ranked_docs[:k], start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision(relevant_docs: dict[str, int], ranked_docs: list[tuple[str, float]]) -> float:
    positive_docs = {doc_id for doc_id, score in relevant_docs.items() if score > 0}
    if not positive_docs:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
        if doc_id in positive_docs:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(positive_docs)


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, float | int]:
    ndcg_scores: list[float] = []
    rr_scores: list[float] = []
    ap_scores: list[float] = []

    for query_id, relevant_docs in qrels.items():
        ranked_docs = sort_run(run.get(query_id, {}))
        ndcg_scores.append(ndcg_at_k(relevant_docs, ranked_docs, k))
        rr_scores.append(reciprocal_rank_at_k(relevant_docs, ranked_docs, k))
        ap_scores.append(average_precision(relevant_docs, ranked_docs))

    return {
        "num_queries": len(qrels),
        f"ndcg@{k}": mean(ndcg_scores) if ndcg_scores else 0.0,
        f"mrr@{k}": mean(rr_scores) if rr_scores else 0.0,
        "map": mean(ap_scores) if ap_scores else 0.0,
    }
