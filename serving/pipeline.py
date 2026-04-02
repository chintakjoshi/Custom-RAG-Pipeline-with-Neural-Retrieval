from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

import faiss
import numpy as np

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus
from neural_rag.evaluation import sort_run
from retrieval.biencoder.model import load_biencoder, prefixed_texts
from reranking.cross_encoder.model import load_cross_encoder
from serving.ollama_generator import OllamaGenerator
from serving.prompt_templates import RetrievedPassage, extract_citation_ids


class GeneratorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelinePassage:
    rank: int
    doc_id: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralRAGPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self._lock = threading.RLock()

        data_config = self.config.get("data", {})
        retrieval_config = self.config.get("retrieval", {})
        reranking_config = self.config.get("reranking", {})
        generator_config = self.config.get("generator", {})
        prompt_config = self.config.get("prompt", {})

        self.corpus = load_corpus(data_config["corpus_path"])
        self.doc_ids = json.loads(Path(data_config["faiss_ids_path"]).read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(data_config["faiss_index_path"]))

        self.biencoder = load_biencoder(
            str(retrieval_config["model_name_or_path"]),
            use_cpu=bool(retrieval_config.get("use_cpu", False)),
        )
        self.query_prefix = str(retrieval_config.get("query_prefix", ""))
        self.retrieval_batch_size = int(retrieval_config.get("batch_size", 8))
        self.retrieval_normalize_embeddings = bool(
            retrieval_config.get("normalize_embeddings", True)
        )
        self.default_retrieve_k = int(retrieval_config.get("top_k", 10))

        self.cross_encoder = load_cross_encoder(
            str(reranking_config["model_name_or_path"]),
            use_cpu=bool(reranking_config.get("use_cpu", False)),
            max_length=int(reranking_config.get("max_length", 256)),
            num_labels=int(reranking_config.get("num_labels", 1)),
        )
        self.rerank_batch_size = int(reranking_config.get("batch_size", 16))
        self.default_rerank_k = int(reranking_config.get("top_k", 5))

        self.generator_enabled = bool(generator_config.get("enabled", True))
        self.generator_preview_only = bool(generator_config.get("preview_only", False))
        self.generator_validate_on_startup = bool(
            generator_config.get("validate_on_startup", True)
        )
        self.unknown_response = str(prompt_config.get("unknown_response", "I don't know."))
        self.generator: OllamaGenerator | None = None
        self.generator_status: dict[str, Any] = {
            "enabled": self.generator_enabled,
            "preview_only": self.generator_preview_only,
            "ready": False,
            "model": generator_config.get("model"),
            "base_url": generator_config.get("base_url", "http://localhost:11434"),
            "available_models": [],
            "error": None,
        }

        if self.generator_enabled:
            self.generator = OllamaGenerator(
                base_url=str(generator_config.get("base_url", "http://localhost:11434")),
                model=str(generator_config["model"]),
                timeout_seconds=int(generator_config.get("timeout_seconds", 180)),
                system_prompt=str(generator_config.get("system_prompt", "")) or None,
                options=dict(generator_config.get("options", {})),
                keep_alive=str(generator_config.get("keep_alive", "")) or None,
                strict_model_check=bool(generator_config.get("strict_model_check", True)),
            )
            if self.generator_preview_only:
                self.generator_status["ready"] = True
            elif self.generator_validate_on_startup:
                self._probe_generator()

    def _probe_generator(self) -> None:
        if not self.generator_enabled or self.generator is None:
            self.generator_status.update(
                {
                    "ready": False,
                    "available_models": [],
                    "error": "Generator is disabled.",
                }
            )
            return

        try:
            available_models = self.generator.validate_model()
        except RuntimeError as exc:
            self.generator_status.update(
                {
                    "ready": False,
                    "available_models": [],
                    "error": str(exc),
                }
            )
            return

        self.generator_status.update(
            {
                "ready": True,
                "available_models": available_models,
                "error": None,
            }
        )

    def health(self) -> dict[str, Any]:
        status = "ok"
        if self.generator_enabled and not self.generator_status["ready"]:
            status = "degraded"

        return {
            "status": status,
            "config_path": str(self.config_path),
            "num_corpus_docs": len(self.corpus),
            "num_indexed_vectors": int(self.index.ntotal),
            "retrieval": {
                "top_k": self.default_retrieve_k,
                "query_prefix": self.query_prefix,
            },
            "reranking": {
                "top_k": self.default_rerank_k,
            },
            "generator": dict(self.generator_status),
        }

    def _retrieve_no_lock(self, query: str, retrieve_k: int) -> tuple[list[PipelinePassage], float]:
        retrieval_start = time.perf_counter()
        query_embedding = self.biencoder.encode(
            prefixed_texts([query], self.query_prefix),
            batch_size=self.retrieval_batch_size,
            normalize_embeddings=self.retrieval_normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = np.ascontiguousarray(
            query_embedding.astype("float32", copy=False)
        )
        scores, indices = self.index.search(query_embedding, retrieve_k)

        passages: list[PipelinePassage] = []
        for rank, (score, row_index) in enumerate(
            zip(scores[0].tolist(), indices[0].tolist()),
            start=1,
        ):
            if row_index < 0:
                continue
            doc_id = self.doc_ids[row_index]
            text = self.corpus.get(doc_id)
            if not text:
                continue
            passages.append(
                PipelinePassage(
                    rank=rank,
                    doc_id=doc_id,
                    text=text,
                    retrieval_score=float(score),
                )
            )

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        return passages, retrieval_latency_ms

    def _rerank_no_lock(
        self,
        query: str,
        passages: list[PipelinePassage],
        rerank_k: int,
    ) -> tuple[list[PipelinePassage], float]:
        rerank_start = time.perf_counter()
        if not passages:
            return [], 0.0

        pair_inputs = [(query, passage.text) for passage in passages]
        scores = self.cross_encoder.predict(
            pair_inputs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored_passages = [
            PipelinePassage(
                rank=0,
                doc_id=passage.doc_id,
                text=passage.text,
                retrieval_score=passage.retrieval_score,
                rerank_score=float(score),
            )
            for passage, score in zip(passages, scores.tolist())
        ]
        scored_passages.sort(
            key=lambda passage: (
                -(passage.rerank_score if passage.rerank_score is not None else float("-inf")),
                passage.doc_id,
            )
        )
        reranked = [
            PipelinePassage(
                rank=rank,
                doc_id=passage.doc_id,
                text=passage.text,
                retrieval_score=passage.retrieval_score,
                rerank_score=passage.rerank_score,
            )
            for rank, passage in enumerate(scored_passages[:rerank_k], start=1)
        ]
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000.0
        return reranked, rerank_latency_ms

    def retrieve(
        self,
        *,
        query: str,
        retrieve_k: int | None = None,
        rerank_k: int | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        final_rerank_k = rerank_k or self.default_rerank_k
        final_retrieve_k = max(retrieve_k or self.default_retrieve_k, final_rerank_k)

        with self._lock:
            passages, retrieval_latency_ms = self._retrieve_no_lock(query, final_retrieve_k)
            reranked, rerank_latency_ms = self._rerank_no_lock(query, passages, final_rerank_k)

        total_latency_ms = retrieval_latency_ms + rerank_latency_ms
        return {
            "query": query,
            "retrieve_k": final_retrieve_k,
            "rerank_k": final_rerank_k,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "total_latency_ms": total_latency_ms,
            "passages": [passage.to_dict() for passage in reranked],
        }

    def _ensure_generator_ready(self) -> None:
        if not self.generator_enabled or self.generator is None:
            raise GeneratorUnavailableError("Generator is disabled in the serving config.")
        if self.generator_preview_only:
            return

        self._probe_generator()
        if not self.generator_status["ready"]:
            raise GeneratorUnavailableError(str(self.generator_status["error"]))

    def answer(
        self,
        *,
        query: str,
        retrieve_k: int | None = None,
        rerank_k: int | None = None,
        include_prompt: bool = False,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        final_rerank_k = rerank_k or self.default_rerank_k
        final_retrieve_k = max(retrieve_k or self.default_retrieve_k, final_rerank_k)

        with self._lock:
            passages, retrieval_latency_ms = self._retrieve_no_lock(query, final_retrieve_k)
            reranked, rerank_latency_ms = self._rerank_no_lock(query, passages, final_rerank_k)
            self._ensure_generator_ready()

            prompt_passages = [
                RetrievedPassage(
                    rank=passage.rank,
                    doc_id=passage.doc_id,
                    score=(
                        passage.rerank_score
                        if passage.rerank_score is not None
                        else passage.retrieval_score
                    ),
                    text=passage.text,
                )
                for passage in reranked
            ]

            generation_start = time.perf_counter()
            if self.generator_preview_only:
                from serving.prompt_templates import build_cited_rag_prompt

                response_payload = {
                    "prompt": build_cited_rag_prompt(
                        query,
                        prompt_passages,
                        unknown_response=self.unknown_response,
                    ),
                    "answer": "[preview_only] No model call was made.",
                    "raw_response": None,
                }
            else:
                response_payload = self.generator.generate_grounded_answer(
                    query=query,
                    passages=prompt_passages,
                    unknown_response=self.unknown_response,
                )
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        answer_text = str(response_payload["answer"])
        total_latency_ms = (
            retrieval_latency_ms + rerank_latency_ms + generation_latency_ms
        )
        return {
            "query": query,
            "retrieve_k": final_retrieve_k,
            "rerank_k": final_rerank_k,
            "answer": answer_text,
            "citation_ids": extract_citation_ids(answer_text),
            "prompt": response_payload["prompt"] if include_prompt else None,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "passages": [passage.to_dict() for passage in reranked],
            "generator": {
                "model": self.generator_status["model"],
                "preview_only": self.generator_preview_only,
            },
        }
