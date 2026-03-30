from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_jsonl(path: str | Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[JsonDict]) -> None:
    output_path = _ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def load_corpus(path: str | Path) -> dict[str, str]:
    return {
        str(record["id"]): str(record["text"])
        for record in read_jsonl(path)
    }


def load_queries(path: str | Path) -> dict[str, str]:
    return {
        str(record["id"]): str(record["text"])
        for record in read_jsonl(path)
    }


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for record in read_jsonl(path):
        query_id = str(record["query_id"])
        doc_id = str(record["doc_id"])
        relevance = int(record["relevance"])
        qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def write_json(path: str | Path, payload: JsonDict) -> None:
    output_path = _ensure_parent(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> JsonDict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_text_id(text: str) -> str:
    """Generate a reproducible synthetic id for corpora that lack passage ids."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
