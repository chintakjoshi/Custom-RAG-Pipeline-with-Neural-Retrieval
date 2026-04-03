# Custom-RAG-Pipeline-with-Neural-Retrieval
A neural retrieval and RAG stack built around a BM25 baseline, a fine-tuned bi-encoder over FAISS, a cross-encoder reranker, ColBERT-style late interaction scoring, a TensorFlow student reranker, local Ollama generation, FastAPI serving, MLflow tracking, and BEIR evaluation.

## Included

- BM25 baseline retrieval and evaluation
- MS MARCO passage-format ingest and local normalized dataset assets
- Bi-encoder training, passage encoding, brute-force search, and FAISS search
- Cross-encoder training and reranking
- ColBERT-style MaxSim reranking
- TensorFlow distillation from the cross-encoder teacher
- Latency benchmarking and results-table generation
- BEIR zero-shot evaluation with GPU support for PyTorch-based retrieval
- MLflow experiment logging
- Ollama-backed grounded answer generation
- FastAPI endpoints for health, retrieval, and end-to-end querying
- An end-to-end notebook demo with a mock Ollama-compatible backend

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Optional on this machine for CUDA-backed PyTorch workflows:

```bash
.\.venv\Scripts\python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Data Ingest

To normalize extracted MS MARCO passage-ranking files into the local JSONL format:

```bash
.\.venv\Scripts\python data/ingest_msmarco_passage.py --config configs/msmarco_passage_ingest.yaml
```

The bundled ingest config uses tiny fixture files under `data/fixtures/msmarco_passage/`. Normalized outputs are written as:

- `corpus.jsonl`
- `queries.<split>.jsonl`
- `qrels.<split>.jsonl`
- `triples.<split>.jsonl`
- `manifest.json`

## Baseline Retrieval

Run the BM25 baseline:

```bash
.\.venv\Scripts\python evaluation/run_baseline_bm25.py --config configs/bm25_baseline.yaml
```

Evaluate a run file:

```bash
.\.venv\Scripts\python evaluation/evaluate.py --qrels data/fixtures/bm25_baseline/qrels.jsonl --run results/bm25_run.json
```

## Dense Retrieval And Reranking

Prepare triplets and train the bi-encoder:

```bash
.\.venv\Scripts\python data/prepare_triples.py --config configs/biencoder_prepare.yaml
.\.venv\Scripts\python retrieval/biencoder/train.py --config configs/biencoder_train.yaml
.\.venv\Scripts\python retrieval/biencoder/search.py --config configs/biencoder_search.yaml
```

Encode passages and search with FAISS:

```bash
.\.venv\Scripts\python retrieval/biencoder/encode_passages.py --config configs/biencoder_encode.yaml
.\.venv\Scripts\python retrieval/faiss_index/build_index.py --config configs/faiss_build.yaml
.\.venv\Scripts\python retrieval/faiss_index/search.py --config configs/faiss_search.yaml
```

Train and run the cross-encoder reranker:

```bash
.\.venv\Scripts\python data/prepare_reranker_pairs.py --config configs/cross_encoder_prepare.yaml
.\.venv\Scripts\python reranking/cross_encoder/train.py --config configs/cross_encoder_train.yaml
.\.venv\Scripts\python reranking/cross_encoder/rerank.py --config configs/cross_encoder_rerank.yaml
```

Run ColBERT-style late interaction reranking:

```bash
.\.venv\Scripts\python reranking/colbert/rerank.py --config configs/colbert_rerank.yaml
```

Run TensorFlow distillation and student reranking:

```bash
.\.venv\Scripts\python reranking/distillation/generate_soft_labels.py --config configs/distillation_generate_soft_labels.yaml
.\.venv\Scripts\python reranking/distillation/train_student_tf.py --config configs/distillation_train_student.yaml
.\.venv\Scripts\python reranking/distillation/rerank_student_tf.py --config configs/distillation_rerank_student.yaml
```

## Benchmarking And Evaluation

Generate latency summaries and a consolidated comparison table:

```bash
.\.venv\Scripts\python evaluation/latency_benchmark.py --config configs/latency_benchmark.yaml
.\.venv\Scripts\python evaluation/build_results_table.py --config configs/results_table.yaml
```

Run BEIR zero-shot evaluation on SciFact:

```bash
.\.venv\Scripts\python evaluation/beir_zero_shot.py --config configs/beir_zero_shot_scifact.yaml
```

Run the broader BEIR sweep across SciFact, FiQA, and TREC-COVID:

```bash
.\.venv\Scripts\python evaluation/beir_zero_shot.py --config configs/beir_zero_shot_broad.yaml
```

Current zero-shot summary on this machine:

- `scifact`: `NDCG@10=0.6451`, `MRR@10=0.6047`, `MAP@10=0.5959`
- `fiqa`: `NDCG@10=0.3687`, `MRR@10=0.4451`, `MAP@10=0.2914`
- `trec-covid`: `NDCG@10=0.4725`, `MRR@10=0.7244`, `MAP@10=0.0105`
- `macro average`: `NDCG@10=0.4954`, `MRR@10=0.5914`, `MAP@10=0.2993`

## Experiment Tracking

MLflow runs are stored locally under `artifacts/mlruns`, with a SQLite backend at `artifacts/mlruns/mlflow.db`.

Start the local MLflow UI:

```bash
.\.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000`.

## Local Generation

Pull a local Ollama model and generate grounded answers from a reranked run:

```bash
ollama pull phi3:mini
.\.venv\Scripts\python serving/generate_answers.py --config configs/ollama_generate.yaml
```

This writes:

- `results/ollama_answers.json`
- `results/ollama_prompts.json`
- `results/ollama_summary.json`
- `results/ollama_answers.md`

## API

The FastAPI app loads the FAISS index, bi-encoder, cross-encoder, and Ollama generator into one local process.

Start the API in PowerShell:

```bash
$env:NEURAL_RAG_API_CONFIG="configs/serving_api.yaml"
.\.venv\Scripts\python -m uvicorn serving.api:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /retrieve`
- `POST /query`

Swagger UI is available at `http://127.0.0.1:8000/docs`.

Example PowerShell request:

```bash
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/query" `
  -ContentType "application/json" `
  -Body '{"query":"What causes inflation?","top_k":2,"include_prompt":true}'
```

## Demo Notebook

The notebook demo is [06_end_to_end_demo.ipynb](notebooks/06_end_to_end_demo.ipynb). It uses `configs/serving_api_demo_mock.yaml`, which points at a lightweight mock Ollama-compatible server on `127.0.0.1:11435`, so the notebook can run without a live Ollama daemon.

The notebook demonstrates:

- API health checks
- reranked retrieval
- end-to-end grounded answer generation
- prompt inspection
- the current benchmark table
