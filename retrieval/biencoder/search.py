from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus, load_queries, write_json
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run
from retrieval.biencoder.model import load_biencoder, prefixed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a retrieval run with the current bi-encoder using brute-force similarity."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_or_compute_corpus_embeddings(
    model,
    corpus: dict[str, str],
    passage_prefix: str,
    inference_config: dict[str, object],
) -> tuple[list[str], np.ndarray]:
    embeddings_path = inference_config.get("corpus_embeddings_path")
    ids_path = inference_config.get("corpus_ids_path")

    if embeddings_path and ids_path and Path(str(embeddings_path)).exists() and Path(str(ids_path)).exists():
        doc_ids = json.loads(Path(str(ids_path)).read_text(encoding="utf-8"))
        embeddings = np.load(str(embeddings_path))
        return list(doc_ids), embeddings

    doc_ids = list(corpus.keys())
    passage_texts = prefixed_texts(corpus.values(), passage_prefix)
    embeddings = model.encode(
        passage_texts,
        batch_size=int(inference_config.get("batch_size", 32)),
        normalize_embeddings=bool(inference_config.get("normalize_embeddings", True)),
        convert_to_numpy=True,
        show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
    )
    return doc_ids, embeddings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="biencoder-search",
        default_tags={"stage": "biencoder", "script": "retrieval/biencoder/search.py"},
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
        query_prefix = str(model_config.get("query_prefix", ""))
        passage_prefix = str(model_config.get("passage_prefix", ""))
        model_name_or_path = str(model_config["model_name_or_path"])
        model = load_biencoder(
            model_name_or_path,
            use_cpu=bool(inference_config.get("use_cpu", False)),
        )

        doc_ids, corpus_embeddings = load_or_compute_corpus_embeddings(
            model=model,
            corpus=corpus,
            passage_prefix=passage_prefix,
            inference_config=inference_config,
        )
        query_embeddings = model.encode(
            prefixed_texts(queries.values(), query_prefix),
            batch_size=int(inference_config.get("batch_size", 32)),
            normalize_embeddings=bool(inference_config.get("normalize_embeddings", True)),
            convert_to_numpy=True,
            show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
        )

        top_k = int(inference_config.get("top_k", 10))
        run: dict[str, dict[str, float]] = {}
        query_ids = list(queries.keys())

        for query_id, query_embedding in zip(query_ids, query_embeddings):
            scores = corpus_embeddings @ query_embedding
            top_indices = np.argsort(-scores)[:top_k]
            run[query_id] = {
                doc_ids[index]: float(scores[index])
                for index in top_indices
            }

        write_json(inference_config["run_output_path"], run)
        tracker.log_metrics(
            {
                "num_queries": len(queries),
                "num_documents": len(doc_ids),
                "top_k": top_k,
            }
        )
        tracker.log_artifact(inference_config["run_output_path"], artifact_path="outputs")
        print(f"Saved run to {inference_config['run_output_path']}")


if __name__ == "__main__":
    main()
