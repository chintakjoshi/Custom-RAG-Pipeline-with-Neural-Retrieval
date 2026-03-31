# Custom-RAG-Pipeline-with-Neural-Retrieval
Design specification and roadmap for a neural RAG pipeline that replaces a BM25-first retriever with learned retrieval: a fine-tuned bi-encoder over FAISS, a cross-encoder reranker, ColBERT-style late interaction scoring, and a locally hosted LLM generator.

The current repository contains the project documentation and implementation plan. See `project.md` for the full architecture, evaluation strategy, planned project structure, and roadmap.

## Build Status

Phase 1 is now in progress in the codebase:

- Project scaffold and reusable core modules
- Pure-Python BM25 baseline retriever
- Reusable evaluation metrics for NDCG, MRR, and MAP
- Sample dataset for smoke testing
- MS MARCO export helper for local experimentation
- Official MS MARCO passage-ranking ingest pipeline for local normalized assets
- Bi-encoder triplet prep, training, passage encoding, and brute-force retrieval for small-scale evaluation
- FAISS index build and indexed dense retrieval for the trained bi-encoder
- Cross-encoder pair prep, training, and reranking over FAISS candidates
- ColBERT-style late interaction reranking over the cross-encoder shortlist

## Phase 1 Quick Start

1. Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Run the sample BM25 baseline:

```bash
.\.venv\Scripts\python evaluation/run_baseline_bm25.py --config configs/bm25_baseline.yaml
```

3. Evaluate an existing run file:

```bash
.\.venv\Scripts\python evaluation/evaluate.py \
  --qrels data/sample/qrels.jsonl \
  --run results/sample_bm25_run.json
```

4. Export a lightweight MS MARCO-derived dataset snapshot:

```bash
.\.venv\Scripts\python data/download_msmarco.py --limit 1000
```

The export step creates normalized `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl` files under `artifacts/msmarco_validation/`. This is useful for local iteration, but it is not yet the final official MS MARCO passage-ranking corpus pipeline.

## Official MS MARCO Ingest

To normalize extracted official MS MARCO passage-ranking files into the repo format:

```bash
.\.venv\Scripts\python data/ingest_msmarco_passage.py --config configs/msmarco_passage_ingest_sample.yaml
```

The sample config points at tiny official-format fixture files under `data/sample_raw/msmarco_passage/`. For the real dataset, copy that config, replace the file paths with your local `collection.tsv`, `queries.*.tsv`, `qrels.*.tsv`, and optional `qidpidtriples.*.tsv(.gz)` paths, then rerun the same command.

Normalized outputs are written as:

- `corpus.jsonl`
- `queries.<split>.jsonl`
- `qrels.<split>.jsonl`
- `triples.<split>.jsonl`
- `manifest.json`

## Phase 2 Bi-Encoder

With the normalized sample ingest in place, the next phase is:

```bash
.\.venv\Scripts\python data/prepare_triples.py --config configs/biencoder_prepare_sample.yaml
.\.venv\Scripts\python retrieval/biencoder/train.py --config configs/biencoder_train_sample.yaml
.\.venv\Scripts\python retrieval/biencoder/search.py --config configs/biencoder_search_sample.yaml
.\.venv\Scripts\python evaluation/evaluate.py --qrels artifacts/sample_msmarco_passage/qrels.dev.jsonl --run results/sample_biencoder_run.json --k 5
```

This uses a brute-force embedding search for small datasets so we can validate the bi-encoder before introducing FAISS in the next retrieval phase.

## Phase 3 FAISS Retrieval

To move from brute-force similarity search to indexed dense retrieval:

```bash
.\.venv\Scripts\python retrieval/biencoder/encode_passages.py --config configs/biencoder_encode_sample.yaml
.\.venv\Scripts\python retrieval/faiss_index/build_index.py --config configs/faiss_build_sample.yaml
.\.venv\Scripts\python retrieval/faiss_index/search.py --config configs/faiss_search_sample.yaml
.\.venv\Scripts\python evaluation/evaluate.py --qrels artifacts/sample_msmarco_passage/qrels.dev.jsonl --run results/sample_faiss_run.json --k 5
```

The sample configuration uses `IndexFlatIP` over normalized embeddings, which corresponds to cosine-similarity ranking while keeping the setup simple for the current phase.

## Phase 4 Cross-Encoder Reranking

To add the second-stage reranker on top of the FAISS candidates:

```bash
.\.venv\Scripts\python data/prepare_reranker_pairs.py --config configs/cross_encoder_prepare_sample.yaml
.\.venv\Scripts\python reranking/cross_encoder/train.py --config configs/cross_encoder_train_sample.yaml
.\.venv\Scripts\python reranking/cross_encoder/rerank.py --config configs/cross_encoder_rerank_sample.yaml
.\.venv\Scripts\python evaluation/evaluate.py --qrels artifacts/sample_msmarco_passage/qrels.dev.jsonl --run results/sample_cross_encoder_reranked_run.json --k 5
```

This stage trains a cross-encoder on labeled query-passage pairs derived from the triplets, then reranks the top FAISS candidates with full pairwise attention.

## Phase 5 ColBERT-Style Late Interaction

To add a MaxSim late-interaction reranker on top of the cross-encoder shortlist:

```bash
.\.venv\Scripts\python reranking/colbert/rerank.py --config configs/colbert_rerank_sample.yaml
.\.venv\Scripts\python evaluation/evaluate.py --qrels artifacts/sample_msmarco_passage/qrels.dev.jsonl --run results/sample_colbert_reranked_run.json --k 5
```

This phase uses token-level embeddings plus MaxSim scoring as a final reranking layer over a small candidate set, which keeps the implementation practical before introducing a fully trained ColBERT-style retriever.
