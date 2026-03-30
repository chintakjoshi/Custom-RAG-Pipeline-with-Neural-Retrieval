from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Iterator, TextIO


def open_maybe_gzip(path: str | Path) -> TextIO:
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8", newline="")
    return file_path.open("r", encoding="utf-8", newline="")


def _iter_tsv(path: str | Path) -> Iterator[list[str]]:
    with open_maybe_gzip(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            yield row


def iter_collection(path: str | Path) -> Iterator[dict[str, str]]:
    for row_number, row in enumerate(_iter_tsv(path), start=1):
        if len(row) < 2:
            raise ValueError(
                f"Invalid collection row at line {row_number} in {path}: expected pid<TAB>passage"
            )
        yield {"id": row[0], "text": row[1]}


def iter_queries(path: str | Path) -> Iterator[dict[str, str]]:
    for row_number, row in enumerate(_iter_tsv(path), start=1):
        if len(row) < 2:
            raise ValueError(
                f"Invalid query row at line {row_number} in {path}: expected qid<TAB>query"
            )
        yield {"id": row[0], "text": row[1]}


def iter_qrels(path: str | Path) -> Iterator[dict[str, int | str]]:
    for row_number, row in enumerate(_iter_tsv(path), start=1):
        if len(row) < 4:
            raise ValueError(
                f"Invalid qrels row at line {row_number} in {path}: expected qid 0 pid rel"
            )
        yield {
            "query_id": row[0],
            "doc_id": row[2],
            "relevance": int(row[3]),
        }


def iter_qid_pid_triples(path: str | Path) -> Iterator[dict[str, str]]:
    for row_number, row in enumerate(_iter_tsv(path), start=1):
        if len(row) < 3:
            raise ValueError(
                f"Invalid triple row at line {row_number} in {path}: expected qid<TAB>pos_pid<TAB>neg_pid"
            )
        yield {
            "query_id": row[0],
            "positive_doc_id": row[1],
            "negative_doc_id": row[2],
        }
