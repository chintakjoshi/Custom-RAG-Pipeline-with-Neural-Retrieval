from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.datasets import stable_text_id, write_json, write_jsonl

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for local export
    load_dataset = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a lightweight MS MARCO-derived dataset into the repo's "
            "normalized JSONL format."
        )
    )
    parser.add_argument("--output-dir", default="artifacts/msmarco_validation")
    parser.add_argument("--config-name", default="v2.1")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of examples to export. Use 0 to export the full split.",
    )
    parser.add_argument(
        "--include-empty-qrels",
        action="store_true",
        help="Keep queries even when the row has no selected passages.",
    )
    return parser.parse_args()


def load_ms_marco_split(config_name: str, split: str):
    if load_dataset is None:
        raise RuntimeError(
            "The datasets package is required for download_msmarco.py. "
            "Install requirements.txt first."
        )

    dataset_names = ("microsoft/ms_marco", "ms_marco")
    for dataset_name in dataset_names:
        try:
            return load_dataset(dataset_name, config_name, split=split)
        except Exception:
            continue

    raise RuntimeError(
        "Unable to load MS MARCO from Hugging Face. "
        "Tried both 'microsoft/ms_marco' and 'ms_marco'."
    )


def main() -> None:
    args = parse_args()
    dataset = load_ms_marco_split(args.config_name, args.split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    corpus_records: dict[str, dict[str, object]] = {}
    query_records: list[dict[str, object]] = []
    qrel_records: list[dict[str, object]] = []
    skipped_queries = 0

    for row in dataset:
        passages = row["passages"]
        selected_indexes = {
            index
            for index, flag in enumerate(passages["is_selected"])
            if int(flag) > 0
        }

        if not selected_indexes and not args.include_empty_qrels:
            skipped_queries += 1
            continue

        query_id = str(row["query_id"])
        query_text = str(row["query"]).strip()
        query_records.append({"id": query_id, "text": query_text})

        urls = passages.get("url", [])
        for index, passage_text in enumerate(passages["passage_text"]):
            cleaned_text = str(passage_text).strip()
            if not cleaned_text:
                continue

            doc_id = stable_text_id(cleaned_text)
            if doc_id not in corpus_records:
                metadata = {"source": "ms_marco", "split": args.split}
                if index < len(urls) and urls[index]:
                    metadata["url"] = urls[index]
                corpus_records[doc_id] = {
                    "id": doc_id,
                    "text": cleaned_text,
                    "metadata": metadata,
                }

            if index in selected_indexes:
                qrel_records.append(
                    {"query_id": query_id, "doc_id": doc_id, "relevance": 1}
                )

    write_jsonl(output_dir / "corpus.jsonl", corpus_records.values())
    write_jsonl(output_dir / "queries.jsonl", query_records)
    write_jsonl(output_dir / "qrels.jsonl", qrel_records)
    write_json(
        output_dir / "metadata.json",
        {
            "dataset": "ms_marco",
            "config_name": args.config_name,
            "split": args.split,
            "input_rows": len(dataset),
            "exported_queries": len(query_records),
            "exported_corpus_size": len(corpus_records),
            "exported_qrels": len(qrel_records),
            "skipped_queries_without_selected_passages": skipped_queries,
            "note": (
                "This export is a lightweight candidate-set view for local experimentation. "
                "It is not a drop-in replacement for the official MS MARCO passage ranking corpus."
            ),
        },
    )

    print(f"Exported corpus to {output_dir}")
    print(f"Queries: {len(query_records)}")
    print(f"Corpus passages: {len(corpus_records)}")
    print(f"Qrels: {len(qrel_records)}")


if __name__ == "__main__":
    main()
