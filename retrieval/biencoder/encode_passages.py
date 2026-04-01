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
from neural_rag.datasets import load_corpus
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run
from retrieval.biencoder.model import load_biencoder, prefixed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode passages with a trained bi-encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="biencoder-encode-passages",
        default_tags={"stage": "biencoder", "script": "retrieval/biencoder/encode_passages.py"},
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
        doc_ids = list(corpus.keys())
        texts = prefixed_texts(corpus.values(), str(model_config.get("passage_prefix", "")))

        model = load_biencoder(
            str(model_config["model_name_or_path"]),
            use_cpu=bool(inference_config.get("use_cpu", False)),
        )
        embeddings = model.encode(
            texts,
            batch_size=int(inference_config.get("batch_size", 32)),
            normalize_embeddings=bool(inference_config.get("normalize_embeddings", True)),
            convert_to_numpy=True,
            show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
        )

        embeddings_path = Path(inference_config["embeddings_path"])
        ids_path = Path(inference_config["ids_path"])
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(embeddings_path, embeddings)
        ids_path.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")

        tracker.log_metrics(
            {
                "num_passages": len(doc_ids),
                "embedding_dimension": int(embeddings.shape[1]),
            }
        )
        tracker.log_artifact(embeddings_path, artifact_path="outputs")
        tracker.log_artifact(ids_path, artifact_path="outputs")

        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved passage ids to {ids_path}")


if __name__ == "__main__":
    main()
