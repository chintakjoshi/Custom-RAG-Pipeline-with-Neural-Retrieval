from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus, load_json, load_queries, write_json
from neural_rag.evaluation import sort_run
from neural_rag.retrieval.bm25 import BM25Retriever
from neural_rag.text import apply_text_prefix
from reranking.colbert.maxsim import maxsim_score
from reranking.colbert.model import ColBERTReranker
from reranking.cross_encoder.model import load_cross_encoder
from reranking.distillation.model import TensorFlowStudentReranker
from retrieval.biencoder.model import load_biencoder, prefixed_texts


StageFn = Callable[[str, str], dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval and reranking stage latency."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_samples(samples_ms: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples_ms)
    return {
        "num_samples": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms) if samples_ms else 0.0,
        "median_ms": statistics.median(samples_ms) if samples_ms else 0.0,
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0] if ordered else 0.0,
        "max_ms": ordered[-1] if ordered else 0.0,
    }


def build_candidate_pairs(
    query_id: str,
    query_text: str,
    candidate_run: dict[str, dict[str, float]],
    corpus: dict[str, str],
    *,
    top_k: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    ranked_candidates = sort_run(candidate_run.get(query_id, {}))[:top_k]
    doc_ids: list[str] = []
    pair_inputs: list[tuple[str, str]] = []

    for doc_id, _ in ranked_candidates:
        passage_text = corpus.get(doc_id)
        if not passage_text:
            continue
        doc_ids.append(doc_id)
        pair_inputs.append((query_text, passage_text))

    return doc_ids, pair_inputs


def benchmark_stage(
    stage_id: str,
    display_name: str,
    query_items: list[tuple[str, str]],
    stage_fn: StageFn,
    *,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup_iterations):
        for query_id, query_text in query_items:
            stage_fn(query_id, query_text)

    samples_ms: list[float] = []
    for _ in range(iterations):
        for query_id, query_text in query_items:
            start = time.perf_counter()
            stage_fn(query_id, query_text)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "stage_id": stage_id,
        "display_name": display_name,
        **summarize_samples(samples_ms),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    benchmark_config = config.get("benchmark", {})

    corpus = load_corpus(data_config["corpus_path"])
    queries = load_queries(data_config["queries_path"])
    query_items = list(queries.items())
    if not query_items:
        raise ValueError("No queries found for benchmarking.")

    warmup_iterations = int(benchmark_config.get("warmup_iterations", 1))
    iterations = int(benchmark_config.get("iterations", 3))
    stages: dict[str, dict[str, object]] = {}

    bm25_config = config.get("bm25", {})
    bm25 = BM25Retriever(
        corpus,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    bm25_top_k = int(bm25_config.get("top_k", 10))
    stages["bm25"] = benchmark_stage(
        "bm25",
        "BM25",
        query_items,
        lambda _query_id, query_text: {
            result.doc_id: result.score
            for result in bm25.search(query_text, top_k=bm25_top_k)
        },
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    faiss_config = config.get("faiss", {})
    index = faiss.read_index(str(data_config["faiss_index_path"]))
    doc_ids = json.loads(Path(data_config["faiss_ids_path"]).read_text(encoding="utf-8"))
    biencoder = load_biencoder(
        str(faiss_config["model_name_or_path"]),
        use_cpu=bool(faiss_config.get("use_cpu", False)),
    )
    faiss_top_k = int(faiss_config.get("top_k", 10))
    faiss_batch_size = int(faiss_config.get("batch_size", 1))
    normalize_embeddings = bool(faiss_config.get("normalize_embeddings", True))
    query_prefix = str(faiss_config.get("query_prefix", ""))

    def faiss_stage(_query_id: str, query_text: str) -> dict[str, float]:
        query_embedding = biencoder.encode(
            prefixed_texts([query_text], query_prefix),
            batch_size=faiss_batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = np.ascontiguousarray(
            query_embedding.astype("float32", copy=False)
        )
        scores, indices = index.search(query_embedding, faiss_top_k)
        return {
            doc_ids[row_index]: float(score)
            for score, row_index in zip(scores[0].tolist(), indices[0].tolist())
            if row_index >= 0
        }

    stages["faiss_biencoder"] = benchmark_stage(
        "faiss_biencoder",
        "Bi-Encoder + FAISS",
        query_items,
        faiss_stage,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    candidate_run = load_json(data_config["candidate_run_path"])

    cross_encoder_config = config.get("cross_encoder", {})
    cross_encoder = load_cross_encoder(
        str(cross_encoder_config["model_name_or_path"]),
        use_cpu=bool(cross_encoder_config.get("use_cpu", False)),
        max_length=int(cross_encoder_config.get("max_length", 256)),
        num_labels=int(cross_encoder_config.get("num_labels", 1)),
    )
    cross_encoder_top_k = int(cross_encoder_config.get("top_k", 10))
    cross_encoder_batch_size = int(cross_encoder_config.get("batch_size", 16))

    def cross_encoder_stage(query_id: str, query_text: str) -> dict[str, float]:
        pair_doc_ids, pair_inputs = build_candidate_pairs(
            query_id,
            query_text,
            candidate_run,
            corpus,
            top_k=cross_encoder_top_k,
        )
        if not pair_inputs:
            return {}

        scores = cross_encoder.predict(
            pair_inputs,
            batch_size=cross_encoder_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return {
            doc_id: float(score)
            for doc_id, score in zip(pair_doc_ids, scores.tolist())
        }

    stages["cross_encoder"] = benchmark_stage(
        "cross_encoder",
        "Cross-Encoder",
        query_items,
        cross_encoder_stage,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    colbert_config = config.get("colbert", {})
    colbert = ColBERTReranker(
        str(colbert_config["model_name_or_path"]),
        max_length=int(colbert_config.get("max_length", 128)),
        use_cpu=bool(colbert_config.get("use_cpu", False)),
    )
    colbert_top_k = int(colbert_config.get("top_k", 10))
    colbert_batch_size = int(colbert_config.get("batch_size", 8))
    colbert_query_prefix = str(colbert_config.get("query_prefix", ""))
    colbert_passage_prefix = str(colbert_config.get("passage_prefix", ""))

    def colbert_stage(query_id: str, query_text: str) -> dict[str, float]:
        ranked_candidates = sort_run(candidate_run.get(query_id, {}))[:colbert_top_k]
        doc_batch_ids: list[str] = []
        passage_texts: list[str] = []
        for doc_id, _ in ranked_candidates:
            passage_text = corpus.get(doc_id)
            if not passage_text:
                continue
            doc_batch_ids.append(doc_id)
            passage_texts.append(apply_text_prefix(passage_text, colbert_passage_prefix))

        if not passage_texts:
            return {}

        query_embeddings, query_mask = colbert.encode(
            [apply_text_prefix(query_text, colbert_query_prefix)]
        )
        query_embedding = query_embeddings[0]
        query_token_mask = query_mask[0]

        scores_by_doc: dict[str, float] = {}
        for start in range(0, len(passage_texts), colbert_batch_size):
            batch_texts = passage_texts[start : start + colbert_batch_size]
            batch_doc_ids = doc_batch_ids[start : start + colbert_batch_size]
            doc_embeddings, doc_mask = colbert.encode(batch_texts)
            batch_scores = maxsim_score(
                query_embedding,
                query_token_mask,
                doc_embeddings,
                doc_mask,
            )
            for doc_id, score in zip(batch_doc_ids, batch_scores.tolist()):
                scores_by_doc[doc_id] = float(score)
        return scores_by_doc

    stages["colbert"] = benchmark_stage(
        "colbert",
        "ColBERT MaxSim",
        query_items,
        colbert_stage,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    tf_student_config = config.get("tf_student", {})
    tf_student = TensorFlowStudentReranker(
        str(tf_student_config["model_name_or_path"]),
        max_length=int(tf_student_config.get("max_length", 256)),
        num_labels=int(tf_student_config.get("num_labels", 1)),
        from_pt=bool(tf_student_config.get("from_pt", False)),
        use_safetensors=bool(tf_student_config.get("use_safetensors", False)),
    )
    tf_student_top_k = int(tf_student_config.get("top_k", 10))
    tf_student_batch_size = int(tf_student_config.get("batch_size", 16))

    def tf_student_stage(query_id: str, query_text: str) -> dict[str, float]:
        pair_doc_ids, pair_inputs = build_candidate_pairs(
            query_id,
            query_text,
            candidate_run,
            corpus,
            top_k=tf_student_top_k,
        )
        if not pair_inputs:
            return {}

        scores = tf_student.predict(pair_inputs, batch_size=tf_student_batch_size)
        return {
            doc_id: float(score)
            for doc_id, score in zip(pair_doc_ids, scores.tolist())
        }

    stages["tf_student"] = benchmark_stage(
        "tf_student",
        "TensorFlow Student",
        query_items,
        tf_student_stage,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    output = {
        "queries_path": str(data_config["queries_path"]),
        "num_queries": len(query_items),
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "stages": stages,
    }
    write_json(benchmark_config["output_path"], output)

    print(f"Saved latency benchmark to {benchmark_config['output_path']}")
    for stage_id, summary in stages.items():
        print(
            f"{stage_id}: mean={summary['mean_ms']:.2f} ms "
            f"median={summary['median_ms']:.2f} ms p95={summary['p95_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
