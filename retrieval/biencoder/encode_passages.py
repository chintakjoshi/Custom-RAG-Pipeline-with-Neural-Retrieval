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
from retrieval.biencoder.model import load_biencoder, prefixed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode passages with a trained bi-encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})

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

    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved passage ids to {ids_path}")


if __name__ == "__main__":
    main()
