from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus, load_json, load_queries, write_json
from neural_rag.evaluation import sort_run
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run
from neural_rag.text import apply_text_prefix
from reranking.colbert.maxsim import maxsim_score
from reranking.colbert.model import ColBERTReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank candidates with ColBERT-style late interaction."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="colbert-rerank",
        default_tags={"stage": "colbert", "script": "reranking/colbert/rerank.py"},
    ) as tracker:
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        inference_config = config.get("inference", {})

        tracker.log_params(
            flatten_mapping(
                {
                    "data": data_config,
                    "model": model_config,
                    "inference": inference_config,
                }
            )
        )

        corpus = load_corpus(data_config["corpus_path"])
        queries = load_queries(data_config["queries_path"])
        candidate_run = load_json(data_config["candidate_run_path"])
        top_k = int(inference_config.get("top_k", 20))
        batch_size = int(inference_config.get("batch_size", 8))
        query_prefix = str(model_config.get("query_prefix", ""))
        passage_prefix = str(model_config.get("passage_prefix", ""))

        reranker = ColBERTReranker(
            str(model_config["model_name_or_path"]),
            max_length=int(model_config.get("max_length", 128)),
            use_cpu=bool(inference_config.get("use_cpu", False)),
        )

        reranked_run: dict[str, dict[str, float]] = {}
        for query_id, candidate_scores in candidate_run.items():
            query_text = queries.get(query_id)
            if not query_text:
                continue

            ranked_candidates = sort_run(candidate_scores)[:top_k]
            doc_ids: list[str] = []
            passage_texts: list[str] = []
            for doc_id, _ in ranked_candidates:
                passage_text = corpus.get(doc_id)
                if not passage_text:
                    continue
                doc_ids.append(doc_id)
                passage_texts.append(apply_text_prefix(passage_text, passage_prefix))

            if not passage_texts:
                reranked_run[query_id] = {}
                continue

            query_embeddings, query_mask = reranker.encode(
                [apply_text_prefix(query_text, query_prefix)]
            )
            query_embedding = query_embeddings[0]
            query_token_mask = query_mask[0]

            scored_docs: list[tuple[str, float]] = []
            for start in range(0, len(passage_texts), batch_size):
                batch_texts = passage_texts[start : start + batch_size]
                batch_doc_ids = doc_ids[start : start + batch_size]
                doc_embeddings, doc_mask = reranker.encode(batch_texts)
                batch_scores = maxsim_score(
                    query_embedding,
                    query_token_mask,
                    doc_embeddings,
                    doc_mask,
                )
                scored_docs.extend(
                    (doc_id, float(score))
                    for doc_id, score in zip(batch_doc_ids, batch_scores.tolist())
                )

            reranked_run[query_id] = {
                doc_id: score
                for doc_id, score in sorted(
                    scored_docs,
                    key=lambda item: (-item[1], item[0]),
                )
            }

        write_json(inference_config["run_output_path"], reranked_run)
        tracker.log_metrics(
            {
                "num_queries": len(reranked_run),
                "top_k": top_k,
                "batch_size": batch_size,
            }
        )
        tracker.log_artifact(inference_config["run_output_path"], artifact_path="outputs")
        print(f"Saved ColBERT-style reranked run to {inference_config['run_output_path']}")


if __name__ == "__main__":
    main()
