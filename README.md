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

## Phase 1 Quick Start

1. Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Run the sample BM25 baseline:

```bash
python evaluation/run_baseline_bm25.py --config configs/bm25_baseline.yaml
```

3. Evaluate an existing run file:

```bash
python evaluation/evaluate.py \
  --qrels data/sample/qrels.jsonl \
  --run results/sample_bm25_run.json
```

4. Export a lightweight MS MARCO-derived dataset snapshot:

```bash
python data/download_msmarco.py --limit 1000
```

The export step creates normalized `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl` files under `artifacts/msmarco_validation/`. This is useful for local iteration, but it is not yet the final official MS MARCO passage-ranking corpus pipeline.

## Official MS MARCO Ingest

To normalize extracted official MS MARCO passage-ranking files into the repo format:

```bash
python data/ingest_msmarco_passage.py --config configs/msmarco_passage_ingest_sample.yaml
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
