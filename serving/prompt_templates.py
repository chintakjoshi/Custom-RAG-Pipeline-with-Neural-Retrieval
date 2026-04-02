from __future__ import annotations

from dataclasses import dataclass
import re


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class RetrievedPassage:
    rank: int
    doc_id: str
    score: float
    text: str


def build_cited_rag_prompt(
    query: str,
    passages: list[RetrievedPassage],
    *,
    unknown_response: str = "I don't know.",
) -> str:
    if not passages:
        context = "[1] No passages were retrieved."
    else:
        context = "\n\n".join(
            f"[{passage.rank}] (doc_id={passage.doc_id}, score={passage.score:.4f}) {passage.text}"
            for passage in passages
        )

    return (
        "Answer the question using only the provided passages.\n"
        "Cite every factual sentence with bracketed passage ids like [1] or [2].\n"
        f"If the answer is not supported by the passages, reply exactly: {unknown_response}\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer with citations:"
    )


def extract_citation_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(text):
        if match not in seen:
            seen.append(match)
    return seen
