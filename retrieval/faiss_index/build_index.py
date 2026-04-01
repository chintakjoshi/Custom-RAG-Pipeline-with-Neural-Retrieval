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
from neural_rag.datasets import write_json
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from passage embeddings.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def create_index(dimension: int, metric: str):
    metric_name = metric.lower()
    if metric_name == "ip":
        return faiss.IndexFlatIP(dimension), faiss.METRIC_INNER_PRODUCT
    if metric_name == "l2":
        return faiss.IndexFlatL2(dimension), faiss.METRIC_L2
    raise ValueError(f"Unsupported FAISS metric: {metric}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="faiss-build-index",
        default_tags={"stage": "faiss", "script": "retrieval/faiss_index/build_index.py"},
    ) as tracker:
        embeddings_config = config.get("embeddings", {})
        index_config = config.get("index", {})

        tracker.log_params(
            flatten_mapping(
                {
                    "embeddings": embeddings_config,
                    "index": index_config,
                }
            )
        )

        embeddings_path = Path(embeddings_config["embeddings_path"])
        ids_path = Path(embeddings_config["ids_path"])
        index_path = Path(index_config["index_path"])
        metadata_path = Path(index_config["metadata_path"])
        metric = str(index_config.get("metric", "ip"))

        embeddings = np.load(embeddings_path).astype("float32", copy=False)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape {embeddings.shape}"
            )
        embeddings = np.ascontiguousarray(embeddings)
        doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))

        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                "The number of embedding rows does not match the number of passage ids."
            )

        index, metric_type = create_index(embeddings.shape[1], metric)
        index.add(embeddings)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        metadata = {
            "index_path": str(index_path),
            "ids_path": str(ids_path),
            "num_vectors": int(index.ntotal),
            "dimension": int(embeddings.shape[1]),
            "metric": metric.lower(),
            "faiss_metric_type": int(metric_type),
        }
        write_json(metadata_path, metadata)

        tracker.log_metrics(
            {
                "num_vectors": int(index.ntotal),
                "dimension": int(embeddings.shape[1]),
            }
        )
        tracker.log_artifact(index_path, artifact_path="outputs")
        tracker.log_artifact(metadata_path, artifact_path="outputs")

        print(f"Saved FAISS index to {index_path}")
        print(f"Indexed vectors: {index.ntotal}")
        print(f"Dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
