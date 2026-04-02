# Custom-RAG-Pipeline-with-Neural-Retrieval
Design specification and roadmap for a neural RAG pipeline that replaces a BM25-first retriever with learned retrieval: a fine-tuned bi-encoder over FAISS, a cross-encoder reranker, ColBERT-style late interaction scoring, and a locally hosted LLM generator.

The current repository contains the project documentation and implementation plan. See `project.md` for the full architecture, evaluation strategy, planned project structure, and roadmap.

## Build Status

Implemented so far in the codebase:

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
- TensorFlow student distillation with soft-label export, student training, and student reranking
- Latency benchmarking and combined results-table generation across retrieval stages
- CUDA-enabled PyTorch in `.venv` for GPU-backed retrieval and reranking on this machine
- BEIR zero-shot evaluation with GPU-backed dense retrieval, multi-dataset sweeps, and saved benchmark summaries
- MLflow experiment logging across the training, retrieval, reranking, benchmarking, and BEIR entrypoints
- Ollama-based grounded answer generation from reranked passages with prompt and answer artifact export
- FastAPI serving layer with `/health`, `/retrieve`, `/query`, and Swagger docs over the local pipeline

## Phase 1 Quick Start

1. Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Optional but recommended on this machine for CUDA-backed PyTorch phases:

```bash
.\.venv\Scripts\python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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

## Phase 6 TensorFlow Distillation

To distill the cross-encoder teacher into a smaller TensorFlow student reranker:

```bash
.\.venv\Scripts\python reranking/distillation/generate_soft_labels.py --config configs/distillation_generate_soft_labels_sample.yaml
.\.venv\Scripts\python reranking/distillation/train_student_tf.py --config configs/distillation_train_student_sample.yaml
.\.venv\Scripts\python reranking/distillation/rerank_student_tf.py --config configs/distillation_rerank_student_sample.yaml
.\.venv\Scripts\python evaluation/evaluate.py --qrels artifacts/sample_msmarco_passage/qrels.dev.jsonl --run results/sample_tf_student_reranked_run.json --k 5
```

This phase exports soft teacher scores from the trained cross-encoder, fine-tunes a TensorFlow DistilBERT student on those scores, then uses the student as a lower-latency reranker over the FAISS candidate set. On native Windows, TensorFlow runs CPU-only in this setup.

## Phase 7 Benchmarking and Comparison

To benchmark stage latency and generate a comparison table across the sample runs:

```bash
.\.venv\Scripts\python evaluation/latency_benchmark.py --config configs/latency_benchmark_sample.yaml
.\.venv\Scripts\python evaluation/build_results_table.py --config configs/results_table_sample.yaml
```

This phase measures BM25, bi-encoder + FAISS, cross-encoder reranking, ColBERT MaxSim reranking, and the TensorFlow student reranker on the same query set. It also produces:

- `results/sample_latency_benchmark.json`
- `results/sample_results_table.json`
- `results/sample_results_table.md`

On the current sample workflow, the TensorFlow student is slower than the PyTorch cross-encoder on native Windows CPU, so the benchmark now captures the real tradeoff instead of assuming distillation is always faster.

## Phase 8 BEIR Zero-Shot Evaluation

To run a GPU-backed BEIR zero-shot evaluation:

```bash
.\.venv\Scripts\python evaluation/beir_zero_shot.py --config configs/beir_zero_shot_sample.yaml
```

The sample config runs SciFact with `sentence-transformers/all-MiniLM-L6-v2` and stores:

- `results/beir_zero_shot/sample_minilm/scifact.run.json`
- `results/beir_zero_shot/sample_minilm/scifact.metrics.json`
- `results/beir_zero_shot/sample_minilm_summary.json`
- `results/beir_zero_shot/sample_minilm_summary.md`

On the current GTX 1050 Ti setup, the SciFact smoke test completed on `cuda` with `NDCG@10=0.6451`, `MRR@10=0.6047`, and `MAP@10=0.5959`.

To run a broader zero-shot sweep across SciFact, FiQA, and TREC-COVID:

```bash
.\.venv\Scripts\python evaluation/beir_zero_shot.py --config configs/beir_zero_shot_broad.yaml
```

This configuration stores per-dataset runs and metrics under `results/beir_zero_shot/broad_minilm/`, plus consolidated summaries in:

- `results/beir_zero_shot/broad_minilm_summary.json`
- `results/beir_zero_shot/broad_minilm_summary.md`

On the current GTX 1050 Ti setup, that broader sweep completed on `cuda` with:

- `scifact`: `NDCG@10=0.6451`, `MRR@10=0.6047`, `MAP@10=0.5959`
- `fiqa`: `NDCG@10=0.3687`, `MRR@10=0.4451`, `MAP@10=0.2914`
- `trec-covid`: `NDCG@10=0.4725`, `MRR@10=0.7244`, `MAP@10=0.0105`
- `macro average`: `NDCG@10=0.4954`, `MRR@10=0.5914`, `MAP@10=0.2993`

## Phase 9 MLflow Experiment Tracking

The sample configs for BM25, bi-encoder training/search, FAISS, cross-encoder training/reranking, ColBERT reranking, TensorFlow distillation, latency benchmarking, and BEIR evaluation now include an `mlflow` section by default. Using `tracking_uri: artifacts/mlruns` creates a local SQLite-backed MLflow store at `artifacts/mlruns/mlflow.db` plus colocated artifacts under `artifacts/mlruns/artifacts/`.

To install the tracking dependency inside `.venv`:

```bash
.\.venv\Scripts\python -m pip install -r requirements.txt
```

To launch the local MLflow UI against the repo store:

```bash
.\.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000` in your browser. Each logged run includes the config file as an artifact, plus the generated JSON and Markdown outputs for that script. A few useful examples are:

```bash
.\.venv\Scripts\python evaluation/run_baseline_bm25.py --config configs/bm25_baseline.yaml
.\.venv\Scripts\python evaluation/latency_benchmark.py --config configs/latency_benchmark_sample.yaml
.\.venv\Scripts\python evaluation/build_results_table.py --config configs/results_table_sample.yaml
.\.venv\Scripts\python evaluation/beir_zero_shot.py --config configs/beir_zero_shot_broad.yaml
```

## Phase 10 Ollama Generator Integration

Install Ollama locally, pull a small grounded-answer model, then run the generation step against an existing reranked run:

```bash
ollama pull phi3:mini
.\.venv\Scripts\python serving/generate_answers.py --config configs/ollama_generate_sample.yaml
```

The sample config reads the cross-encoder reranked sample run, keeps the top-2 passages per query, and writes:

- `results/sample_ollama_answers.json`
- `results/sample_ollama_prompts.json`
- `results/sample_ollama_summary.json`
- `results/sample_ollama_answers.md`

The generator uses the local Ollama chat API, requires sentence-level citations like `[1]`, and falls back to `I don't know.` when the retrieved passages do not support the answer.

## Phase 11 FastAPI Serving Layer

The serving layer loads the FAISS index, bi-encoder, cross-encoder, and Ollama generator into one local process. The sample API config is `configs/serving_api_sample.yaml`.

To launch it in PowerShell:

```bash
$env:NEURAL_RAG_API_CONFIG="configs/serving_api_sample.yaml"
.\.venv\Scripts\python -m uvicorn serving.api:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

Example query:

```bash
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/query" `
  -ContentType "application/json" `
  -Body '{"query":"What causes inflation?","top_k":2,"include_prompt":true}'
```

Available endpoints:

- `GET /health` for model and generator readiness
- `POST /retrieve` for reranked passages without generation
- `POST /query` for end-to-end grounded answer generation
