from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import write_json, write_jsonl
from neural_rag.msmarco import (
    iter_collection,
    iter_qid_pid_triples,
    iter_qrels,
    iter_queries,
)

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize official MS MARCO passage-ranking files into repo-local JSONL assets."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--collection")
    return parser.parse_args()


def take_limit(items: Iterable[T], limit: int) -> list[T]:
    if limit <= 0:
        return list(items)

    taken: list[T] = []
    for index, item in enumerate(items):
        if index >= limit:
            break
        taken.append(item)
    return taken


def normalize_split_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    return {str(split): str(path) for split, path in mapping.items()}


def export_records(
    output_path: Path,
    iterator_factory: Callable[[], Iterable[dict[str, Any]]],
    limit: int,
) -> int:
    records = take_limit(iterator_factory(), limit)
    write_jsonl(output_path, records)
    return len(records)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    input_config = config.get("inputs", {})
    limits = config.get("limits", {})

    output_dir = Path(args.output_dir or config.get("output_dir", "artifacts/msmarco_passage"))
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_path = Path(args.collection or input_config["collection"])
    query_paths = normalize_split_mapping(input_config.get("queries", {}))
    qrels_paths = normalize_split_mapping(input_config.get("qrels", {}))
    triple_paths = normalize_split_mapping(input_config.get("triples", {}))

    manifest: dict[str, Any] = {
        "dataset": "ms_marco_passage_ranking",
        "source_format": "official_tsv",
        "inputs": {
            "collection": str(collection_path),
            "queries": query_paths,
            "qrels": qrels_paths,
            "triples": triple_paths,
        },
        "outputs": {},
        "stats": {},
    }

    collection_limit = int(limits.get("collection", 0))
    collection_count = export_records(
        output_dir / "corpus.jsonl",
        lambda: iter_collection(collection_path),
        collection_limit,
    )
    manifest["outputs"]["corpus"] = str(output_dir / "corpus.jsonl")
    manifest["stats"]["corpus"] = collection_count

    query_limit = int(limits.get("queries", 0))
    manifest["outputs"]["queries"] = {}
    manifest["stats"]["queries"] = {}
    for split, path in query_paths.items():
        output_path = output_dir / f"queries.{split}.jsonl"
        count = export_records(output_path, lambda path=path: iter_queries(path), query_limit)
        manifest["outputs"]["queries"][split] = str(output_path)
        manifest["stats"]["queries"][split] = count

    qrels_limit = int(limits.get("qrels", 0))
    manifest["outputs"]["qrels"] = {}
    manifest["stats"]["qrels"] = {}
    for split, path in qrels_paths.items():
        output_path = output_dir / f"qrels.{split}.jsonl"
        count = export_records(output_path, lambda path=path: iter_qrels(path), qrels_limit)
        manifest["outputs"]["qrels"][split] = str(output_path)
        manifest["stats"]["qrels"][split] = count

    triples_limit = int(limits.get("triples", 0))
    manifest["outputs"]["triples"] = {}
    manifest["stats"]["triples"] = {}
    for split, path in triple_paths.items():
        output_path = output_dir / f"triples.{split}.jsonl"
        count = export_records(
            output_path,
            lambda path=path: iter_qid_pid_triples(path),
            triples_limit,
        )
        manifest["outputs"]["triples"][split] = str(output_path)
        manifest["stats"]["triples"][split] = count

    write_json(output_dir / "manifest.json", manifest)

    print(f"Normalized dataset written to {output_dir}")
    print(f"Corpus rows: {collection_count}")
    for split, count in manifest["stats"]["queries"].items():
        print(f"Queries[{split}]: {count}")
    for split, count in manifest["stats"]["qrels"].items():
        print(f"Qrels[{split}]: {count}")
    for split, count in manifest["stats"]["triples"].items():
        print(f"Triples[{split}]: {count}")


if __name__ == "__main__":
    main()
