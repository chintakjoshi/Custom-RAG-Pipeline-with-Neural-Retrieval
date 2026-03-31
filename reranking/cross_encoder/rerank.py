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
from reranking.cross_encoder.model import load_cross_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank dense-retrieval candidates with a cross-encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})

    corpus = load_corpus(data_config["corpus_path"])
    queries = load_queries(data_config["queries_path"])
    candidate_run = load_json(data_config["candidate_run_path"])
    top_k = int(inference_config.get("top_k", 50))

    model = load_cross_encoder(
        str(model_config["model_name_or_path"]),
        use_cpu=bool(inference_config.get("use_cpu", False)),
        max_length=int(model_config.get("max_length", 256)),
        num_labels=int(model_config.get("num_labels", 1)),
    )

    reranked_run: dict[str, dict[str, float]] = {}
    for query_id, candidate_scores in candidate_run.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue

        ranked_candidates = sort_run(candidate_scores)[:top_k]
        pair_inputs: list[tuple[str, str]] = []
        doc_ids: list[str] = []

        for doc_id, _ in ranked_candidates:
            passage_text = corpus.get(doc_id)
            if not passage_text:
                continue
            pair_inputs.append((query_text, passage_text))
            doc_ids.append(doc_id)

        if not pair_inputs:
            reranked_run[query_id] = {}
            continue

        scores = model.predict(
            pair_inputs,
            batch_size=int(inference_config.get("batch_size", 16)),
            show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
            convert_to_numpy=True,
        )
        reranked_run[query_id] = {
            doc_id: float(score)
            for doc_id, score in sorted(
                zip(doc_ids, scores.tolist()),
                key=lambda item: (-item[1], item[0]),
            )
        }

    write_json(inference_config["run_output_path"], reranked_run)
    print(f"Saved reranked run to {inference_config['run_output_path']}")


if __name__ == "__main__":
    main()
