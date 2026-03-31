from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_triples, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare labeled cross-encoder training pairs from bi-encoder triplets."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    triplets = load_triples(data_config["triplets_path"])
    output_path = data_config["output_path"]

    pairs: list[dict[str, object]] = []
    for triplet in triplets:
        pairs.append(
            {
                "query_id": triplet["query_id"],
                "doc_id": triplet["positive_doc_id"],
                "query_text": triplet["query_text"],
                "passage_text": triplet["positive_text"],
                "label": 1.0,
            }
        )
        pairs.append(
            {
                "query_id": triplet["query_id"],
                "doc_id": triplet["negative_doc_id"],
                "query_text": triplet["query_text"],
                "passage_text": triplet["negative_text"],
                "label": 0.0,
            }
        )

    write_jsonl(output_path, pairs)
    write_json(
        Path(output_path).with_suffix(".summary.json"),
        {
            "input_triplets": len(triplets),
            "output_pairs": len(pairs),
            "output_path": output_path,
        },
    )

    print(f"Saved {len(pairs)} reranker pairs to {output_path}")


if __name__ == "__main__":
    main()
