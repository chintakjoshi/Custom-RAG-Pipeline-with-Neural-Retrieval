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

## Phase 1 Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
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
