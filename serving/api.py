from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from serving.pipeline import GeneratorUnavailableError, NeuralRAGPipeline


DEFAULT_CONFIG_PATH = "configs/serving_api.yaml"
CONFIG_ENV_VAR = "NEURAL_RAG_API_CONFIG"


class PassageResponse(BaseModel):
    rank: int
    doc_id: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    retrieve_k: int | None = Field(default=None, ge=1)


class RetrieveResponse(BaseModel):
    query: str
    retrieve_k: int
    rerank_k: int
    retrieval_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float
    passages: list[PassageResponse]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    retrieve_k: int | None = Field(default=None, ge=1)
    include_prompt: bool = False


class QueryResponse(RetrieveResponse):
    answer: str
    citation_ids: list[str]
    prompt: str | None = None
    generation_latency_ms: float
    generator: dict[str, str | bool | None]


class HealthResponse(BaseModel):
    status: str
    config_path: str
    num_corpus_docs: int
    num_indexed_vectors: int
    retrieval: dict[str, object]
    reranking: dict[str, object]
    generator: dict[str, object]


def resolve_config_path(config_path: str | None = None) -> Path:
    candidate = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    resolved = Path(candidate)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Serving config was not found at {resolved}. "
            f"Set {CONFIG_ENV_VAR} or create {DEFAULT_CONFIG_PATH}."
        )
    return resolved


def _normalize_retrieve_k(top_k: int | None, retrieve_k: int | None) -> tuple[int | None, int | None]:
    final_rerank_k = top_k
    final_retrieve_k = retrieve_k
    if final_rerank_k is not None and final_retrieve_k is None:
        final_retrieve_k = final_rerank_k
    return final_retrieve_k, final_rerank_k


def create_app(config_path: str | None = None) -> FastAPI:
    resolved_config_path = resolve_config_path(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = NeuralRAGPipeline(resolved_config_path)
        yield

    app = FastAPI(
        title="Custom Neural RAG API",
        version="0.1.0",
        description=(
            "Local FastAPI wrapper for the FAISS retrieval, cross-encoder reranking, "
            "and Ollama answer-generation pipeline."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_pipeline(request: Request) -> NeuralRAGPipeline:
        return request.app.state.pipeline

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse.model_validate(get_pipeline(request).health())

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
        pipeline = get_pipeline(request)
        final_retrieve_k, final_rerank_k = _normalize_retrieve_k(
            payload.top_k,
            payload.retrieve_k,
        )
        try:
            result = pipeline.retrieve(
                query=payload.query,
                retrieve_k=final_retrieve_k,
                rerank_k=final_rerank_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RetrieveResponse.model_validate(result)

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        pipeline = get_pipeline(request)
        final_retrieve_k, final_rerank_k = _normalize_retrieve_k(
            payload.top_k,
            payload.retrieve_k,
        )
        try:
            result = pipeline.answer(
                query=payload.query,
                retrieve_k=final_retrieve_k,
                rerank_k=final_rerank_k,
                include_prompt=payload.include_prompt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GeneratorUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryResponse.model_validate(result)

    return app


app = create_app()
