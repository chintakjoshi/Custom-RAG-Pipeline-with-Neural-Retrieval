from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_queries, write_json
from retrieval.biencoder.model import load_biencoder, prefixed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a FAISS index with query embeddings.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    index_config = config.get("index", {})
    inference_config = config.get("inference", {})

    index_path = Path(index_config["index_path"])
    ids_path = Path(index_config["ids_path"])
    top_k = int(inference_config.get("top_k", 10))

    index = faiss.read_index(str(index_path))
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    queries = load_queries(data_config["queries_path"])

    model = load_biencoder(
        str(model_config["model_name_or_path"]),
        use_cpu=bool(inference_config.get("use_cpu", False)),
    )
    query_embeddings = model.encode(
        prefixed_texts(queries.values(), str(model_config.get("query_prefix", ""))),
        batch_size=int(inference_config.get("batch_size", 32)),
        normalize_embeddings=bool(inference_config.get("normalize_embeddings", True)),
        convert_to_numpy=True,
        show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
    )
    query_embeddings = np.ascontiguousarray(query_embeddings.astype("float32", copy=False))

    scores, indices = index.search(query_embeddings, top_k)

    run: dict[str, dict[str, float]] = {}
    for query_id, query_scores, query_indices in zip(queries.keys(), scores, indices):
        ranked_docs: dict[str, float] = {}
        for score, row_index in zip(query_scores.tolist(), query_indices.tolist()):
            if row_index < 0:
                continue
            ranked_docs[doc_ids[row_index]] = float(score)
        run[query_id] = ranked_docs

    write_json(inference_config["run_output_path"], run)
    print(f"Saved FAISS run to {inference_config['run_output_path']}")


if __name__ == "__main__":
    main()
