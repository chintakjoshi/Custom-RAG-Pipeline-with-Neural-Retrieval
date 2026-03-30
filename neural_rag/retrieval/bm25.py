from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from neural_rag.text import simple_tokenize


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


class BM25Retriever:
    """Small pure-Python BM25 implementation for the Phase 1 lexical baseline."""

    def __init__(
        self,
        corpus: dict[str, str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus is empty.")

        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus.keys())
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []

        for doc_id in self.doc_ids:
            tokens = simple_tokenize(corpus[doc_id])
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            self.document_frequencies.update(frequencies.keys())

        self.num_docs = len(self.doc_ids)
        self.avg_doc_length = sum(self.document_lengths) / self.num_docs

    def _idf(self, term: str) -> float:
        doc_frequency = self.document_frequencies.get(term, 0)
        return math.log(1.0 + (self.num_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))

    def score(self, query: str) -> dict[str, float]:
        query_terms = Counter(simple_tokenize(query))
        scores: dict[str, float] = {}

        for doc_index, doc_id in enumerate(self.doc_ids):
            frequencies = self.term_frequencies[doc_index]
            doc_length = self.document_lengths[doc_index]
            score = 0.0

            for term, query_term_count in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score += self._idf(term) * (numerator / denominator) * query_term_count

            scores[doc_id] = score

        return scores

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]
