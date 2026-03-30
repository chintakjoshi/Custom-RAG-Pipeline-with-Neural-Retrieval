from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus, load_qrels, load_queries, load_triples, write_json, write_jsonl
from neural_rag.retrieval.bm25 import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare text-rich bi-encoder triplets from normalized MS MARCO assets."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def build_triplet_record(
    query_id: str,
    positive_doc_id: str,
    negative_doc_id: str,
    queries: dict[str, str],
    corpus: dict[str, str],
) -> dict[str, str] | None:
    query_text = queries.get(query_id)
    positive_text = corpus.get(positive_doc_id)
    negative_text = corpus.get(negative_doc_id)

    if not query_text or not positive_text or not negative_text:
        return None

    return {
        "query_id": query_id,
        "positive_doc_id": positive_doc_id,
        "negative_doc_id": negative_doc_id,
        "query_text": query_text,
        "positive_text": positive_text,
        "negative_text": negative_text,
    }


def prepare_from_official_triples(
    corpus: dict[str, str],
    queries: dict[str, str],
    triples_path: str,
) -> tuple[list[dict[str, str]], int]:
    records: list[dict[str, str]] = []
    skipped = 0

    for triple in load_triples(triples_path):
        record = build_triplet_record(
            query_id=str(triple["query_id"]),
            positive_doc_id=str(triple["positive_doc_id"]),
            negative_doc_id=str(triple["negative_doc_id"]),
            queries=queries,
            corpus=corpus,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)

    return records, skipped


def prepare_from_bm25(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels_path: str,
    mining_config: dict[str, Any],
) -> tuple[list[dict[str, str]], int]:
    qrels = load_qrels(qrels_path)
    retriever = BM25Retriever(corpus)
    top_k = int(mining_config.get("top_k", 100))
    min_rank = int(mining_config.get("min_rank", 20))
    max_rank = int(mining_config.get("max_rank", top_k))
    negatives_per_query = int(mining_config.get("negatives_per_query", 1))

    records: list[dict[str, str]] = []
    skipped = 0

    for query_id, relevant_docs in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            skipped += 1
            continue

        ranked_results = retriever.search(query_text, top_k=top_k)
        candidate_negatives = [
            result.doc_id
            for rank, result in enumerate(ranked_results, start=1)
            if min_rank <= rank <= max_rank and result.doc_id not in relevant_docs
        ]

        if not candidate_negatives:
            skipped += 1
            continue

        selected_negatives = candidate_negatives[:negatives_per_query]
        for positive_doc_id, relevance in relevant_docs.items():
            if relevance <= 0:
                continue
            for negative_doc_id in selected_negatives:
                record = build_triplet_record(
                    query_id=query_id,
                    positive_doc_id=positive_doc_id,
                    negative_doc_id=negative_doc_id,
                    queries=queries,
                    corpus=corpus,
                )
                if record is None:
                    skipped += 1
                    continue
                records.append(record)

    return records, skipped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    mining_config = config.get("mining", {})
    strategy = str(mining_config.get("strategy", "official_triples"))

    corpus = load_corpus(data_config["corpus_path"])
    queries = load_queries(data_config["queries_path"])
    output_path = data_config["output_path"]

    if strategy == "official_triples":
        triplets, skipped = prepare_from_official_triples(
            corpus=corpus,
            queries=queries,
            triples_path=data_config["triples_path"],
        )
    elif strategy == "bm25":
        triplets, skipped = prepare_from_bm25(
            corpus=corpus,
            queries=queries,
            qrels_path=data_config["qrels_path"],
            mining_config=mining_config,
        )
    else:
        raise ValueError(f"Unsupported triplet strategy: {strategy}")

    write_jsonl(output_path, triplets)
    summary = {
        "strategy": strategy,
        "output_path": output_path,
        "num_triplets": len(triplets),
        "skipped_records": skipped,
    }
    write_json(Path(output_path).with_suffix(".summary.json"), summary)

    print(f"Saved {len(triplets)} triplets to {output_path}")
    print(f"Skipped records: {skipped}")


if __name__ == "__main__":
    main()
