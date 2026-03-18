# Neural RAG Pipeline with Learned Retrieval

> A production-grade Retrieval-Augmented Generation system that replaces standard BM25/FAISS retrieval with a fully learned neural retrieval stack -- including a fine-tuned bi-encoder, a cross-encoder reranker, ColBERT-style late interaction scoring, and a locally hosted LLM generator. Built entirely with free and open-source tools.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Architecture Overview](#architecture-overview)
3. [Component Deep Dive](#component-deep-dive)
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)
6. [Training Strategy](#training-strategy)
7. [Evaluation](#evaluation)
8. [Free Infrastructure Plan](#free-infrastructure-plan)
9. [Project Structure](#project-structure)
10. [Setup and Running](#setup-and-running)
11. [Results Baseline](#results-baseline)
12. [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
13. [Scaling Considerations](#scaling-considerations)
14. [Roadmap](#roadmap)

---

## Project Goals

This project demonstrates senior-level ML engineering by going beyond off-the-shelf RAG tooling. The primary goals are:

- Replace BM25 sparse retrieval with a **fine-tuned dense bi-encoder** trained with hard negative mining
- Add a **cross-encoder reranker** as a precision stage after initial retrieval
- Implement **ColBERT-style MaxSim late interaction** scoring as a senior differentiator
- Distill the PyTorch cross-encoder teacher into a lightweight **TensorFlow student model** for faster inference
- Run a **local LLM generator** via Ollama with zero API cost
- Produce rigorous **NDCG@10 and MRR@10** evaluations at every stage to prove improvement
- Keep the **entire stack free** -- no paid APIs, no cloud bills, no vendor lock-in

---

## Architecture Overview

```
                          USER QUERY
                              |
                    +---------v----------+
                    |  Query Bi-Encoder  |  (fine-tuned MiniLM-L6, PyTorch)
                    +--------------------+
                              |
                         Query Vector
                              |
                    +---------v----------+
                    |    FAISS Index     |  (local, IVF or Flat)
                    |  (1M+ passages)    |
                    +--------------------+
                              |
                        Top-50 Candidates
                              |
              +---------------v----------------+
              |   Cross-Encoder Reranker        |  (PyTorch, pairwise loss)
              |   [ CLS ] query [SEP] passage   |
              +---------------------------------+
                              |
                         Top-50 scored
                              |
              +---------------v----------------+
              |   ColBERT MaxSim Layer          |  (token-level late interaction)
              +---------------------------------+
                              |
                         Top-5 Reranked
                              |
              +---------------v----------------+
              |   Ollama LLM Generator          |  (Mistral 7B or LLaMA 3.1 8B)
              |   local, no API cost            |
              +---------------------------------+
                              |
                         FINAL ANSWER
                    + source passages cited


  -- Offline Path --

  Raw Passages
      |
  Passage Bi-Encoder  -->  Passage Embeddings  -->  FAISS Index (persisted to disk)

  -- Distillation Path (TensorFlow) --

  PyTorch Cross-Encoder (Teacher)
      |
  Soft Score Labels
      |
  TF Keras Student Model (smaller, faster inference)
```

---

## Component Deep Dive

### 1. Bi-Encoder (Dense Retrieval)

The bi-encoder encodes queries and passages independently into a shared dense vector space. At retrieval time only the query needs encoding -- all passage vectors are precomputed and stored in FAISS.

**Base model:** `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dim embeddings)

**Why this model:**
- Small enough to fine-tune on Kaggle free GPU in under 2 hours
- Strong zero-shot baseline to compare against after fine-tuning
- Well-documented, widely benchmarked

**Training objective:** InfoNCE loss with in-batch negatives and hard negatives

```python
# In-batch contrastive loss (InfoNCE)
query_emb    = query_encoder(query_input)       # [B, 384]
passage_emb  = passage_encoder(passage_input)   # [B, 384]
scores       = torch.matmul(query_emb, passage_emb.T)  # [B, B]
labels       = torch.arange(B).to(device)
loss         = F.cross_entropy(scores * temperature, labels)
```

**Hard Negative Mining:**
- First pass: retrieve top-100 with BM25
- Passages ranked 20-100 (not relevant but lexically similar) are used as hard negatives
- Hard negatives force the encoder to learn semantic distinction, not just keyword overlap

---

### 2. Cross-Encoder Reranker

The cross-encoder attends jointly over the query and passage, producing a single relevance score. It is far more accurate than the bi-encoder but too slow to run over millions of passages -- so it only reranks the top-50 returned by FAISS.

**Base model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Input format:** `[CLS] query [SEP] passage [SEP]`

```python
class CrossEncoderReranker(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.classifier(cls).squeeze(-1)
```

**Training loss:** Pairwise margin ranking loss

```python
loss = F.margin_ranking_loss(
    pos_score, neg_score,
    target=torch.ones(B).to(device),
    margin=1.0
)
```

---

### 3. ColBERT MaxSim Late Interaction

Instead of compressing a passage to a single vector, ColBERT retains token-level embeddings and scores using MaxSim -- the sum of maximum similarities between each query token and any passage token.

This is the architectural differentiator of this project.

```python
def maxsim_score(Q, P):
    """
    Q: [B, Lq, D]  -- query token embeddings
    P: [B, Lp, D]  -- passage token embeddings
    """
    # Cosine similarity across all query-passage token pairs
    scores   = torch.einsum('bqd,bpd->bqp', Q, P)   # [B, Lq, Lp]
    max_sim  = scores.max(dim=2).values               # [B, Lq]
    return max_sim.sum(dim=1)                          # [B]  -- final score
```

This is used as a final scoring layer on top of the cross-encoder shortlist, not as a standalone retriever (which keeps compute manageable on free hardware).

---

### 4. TensorFlow Student Model (Knowledge Distillation)

The cross-encoder is accurate but slow. A smaller distilled student model in TensorFlow serves the same reranking task at lower latency.

**Process:**
1. Run the PyTorch cross-encoder (teacher) over all query-passage pairs and save soft scores
2. Train a TF Keras student on those soft scores using MSE loss
3. Benchmark student vs teacher on NDCG@10 and latency

```python
import tensorflow as tf

student = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(768,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64,  activation='relu'),
    tf.keras.layers.Dense(1)
])

student.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mse'
)

# teacher_scores are soft labels from PyTorch cross-encoder
student.fit(passage_representations, teacher_scores, epochs=10, batch_size=64)
```

---

### 5. LLM Generator (Ollama -- Local)

Ollama runs open-source LLMs locally with no API key or cost. The generator receives the top-5 reranked passages and produces a grounded answer.

**Recommended models by hardware:**

| RAM Available | Model | Size |
|---|---|---|
| 16GB+ | Mistral 7B or LLaMA 3.1 8B | ~4-5GB |
| 8-12GB | Phi-3 Mini 3.8B | ~2.3GB |
| Under 8GB | Phi-3 Mini (quantized) | ~1.8GB |

```python
import ollama

def generate_answer(query: str, passages: list[str]) -> str:
    context = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    prompt  = f"""Answer the following question using only the provided passages.
If the answer is not in the passages, say "I don't know."

Passages:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
```

---

## Tech Stack

| Layer | Tool | Version | Cost |
|---|---|---|---|
| Bi-encoder training | PyTorch + sentence-transformers | torch>=2.0 | Free |
| Cross-encoder reranker | PyTorch + HuggingFace Transformers | transformers>=4.38 | Free |
| Late interaction scoring | PyTorch (custom MaxSim) | -- | Free |
| Distilled student model | TensorFlow / Keras | tf>=2.14 | Free |
| Vector index | FAISS | faiss-cpu or faiss-gpu | Free |
| LLM generator | Ollama | latest | Free |
| Evaluation | pytrec_eval | 0.5.x | Free |
| Experiment tracking | MLflow | 2.x (local) | Free |
| Dataset loading | HuggingFace Datasets | datasets>=2.x | Free |
| API serving | FastAPI + uvicorn | -- | Free |
| Training compute | Kaggle Notebooks | T4 GPU, 30hr/week | Free |
| Dev compute | Google Colab | T4 GPU, session-based | Free |

---

## Dataset

**Primary: MS MARCO Passage Ranking (v2.1)**

- 8.8 million passages
- 500k+ training queries with relevance labels
- Industry standard -- your NDCG/MRR numbers are directly comparable to published papers

```python
from datasets import load_dataset

# Downloads and caches locally -- free
dataset = load_dataset("ms_marco", "v2.1")
train   = dataset["train"]
val     = dataset["validation"]
```

**Development subset** (to iterate fast before full training):

```python
# Use 50k examples for fast iteration on Kaggle
small_train = dataset["train"].select(range(50_000))
```

**Secondary: BEIR Benchmark**

After training on MS MARCO, evaluate zero-shot generalization on BEIR subsets (TREC-COVID, FiQA, SciFact). This shows the model generalizes, not just memorizes -- a key signal of a well-engineered retriever.

```python
beir_dataset = load_dataset("BeIR/beir", "scifact")
```

---

## Training Strategy

### Phase 1 -- Bi-Encoder Fine-tuning

1. Load MS MARCO training triples (query, positive passage, negative passage)
2. Mine hard negatives using BM25 on the training queries
3. Fine-tune MiniLM-L6 with InfoNCE + hard negatives
4. Save checkpoint to `/kaggle/working/biencoder_checkpoint/`

**Kaggle training time estimate:** ~90 minutes on T4 for 50k examples

### Phase 2 -- Cross-Encoder Fine-tuning

1. Use bi-encoder top-50 as candidate pool
2. Fine-tune cross-encoder on pairwise (positive, hard negative) pairs
3. Train with margin ranking loss for 3 epochs

**Kaggle training time estimate:** ~60 minutes on T4

### Phase 3 -- Distillation

1. Run cross-encoder over 200k query-passage pairs, save soft scores
2. Train TF student on those scores (CPU-friendly, no GPU needed)

### Phase 4 -- Evaluation and Benchmarking

Run full evaluation pipeline, generate the results table, log everything to MLflow.

---

## Evaluation

**Library:** `pytrec_eval` -- the standard IR evaluation library used in academic research

```python
import pytrec_eval

# qrels: {query_id: {passage_id: relevance_score}}
# run:   {query_id: {passage_id: model_score}}

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {'ndcg_cut.10', 'recip_rank', 'map'}
)

results = evaluator.evaluate(run)

ndcg_10 = np.mean([v['ndcg_cut_10']  for v in results.values()])
mrr_10  = np.mean([v['recip_rank']   for v in results.values()])
```

**Target Results Table:**

| Method | NDCG@10 | MRR@10 | Latency per Query |
|---|---|---|---|
| BM25 (baseline) | ~0.27 | ~0.18 | ~10ms |
| Bi-Encoder zero-shot | ~0.31 | ~0.21 | ~45ms |
| Bi-Encoder fine-tuned | ~0.38 | ~0.27 | ~45ms |
| + Cross-Encoder Rerank | ~0.43 | ~0.33 | ~180ms |
| + ColBERT MaxSim | ~0.45 | ~0.35 | ~220ms |
| TF Student Reranker | ~0.41 | ~0.31 | ~90ms |

Actual numbers will vary by dataset split and model checkpoint. The progression matters more than absolute values.

---

## Free Infrastructure Plan

### Training

| Task | Platform | GPU | Time |
|---|---|---|---|
| Bi-encoder fine-tuning | Kaggle Notebooks | T4 16GB | ~90 min |
| Cross-encoder training | Kaggle Notebooks | T4 16GB | ~60 min |
| BEIR evaluation | Kaggle Notebooks | T4 16GB | ~30 min |
| TF student distillation | Local CPU or Colab | CPU | ~40 min |

**Kaggle tips:**
- Use `accelerator: GPU T4 x1` in notebook settings
- Save all model checkpoints to `/kaggle/working/` and download before session ends
- MS MARCO is available as a Kaggle Dataset -- add it directly to avoid re-downloading
- Use `--resume_from_checkpoint` to continue across sessions

### Inference and Serving

Runs entirely on your local machine:

- FAISS index loaded into RAM (~1.5GB for 1M passages at 384-dim float32)
- Ollama runs the LLM in the background as a local daemon
- FastAPI serves the pipeline on `localhost:8000`
- MLflow UI runs on `localhost:5000`

**Minimum local machine spec:**
- 16GB RAM (8GB workable with quantized LLM)
- No GPU required for inference

---

## Project Structure

```
neural-rag-pipeline/
|
+-- data/
|   +-- download_msmarco.py          # Dataset download and preprocessing
|   +-- hard_negative_mining.py      # BM25-based hard negative mining
|   +-- prepare_triples.py           # Build (query, pos, neg) training triples
|
+-- retrieval/
|   +-- biencoder/
|   |   +-- model.py                 # Bi-encoder model definition
|   |   +-- train.py                 # Fine-tuning with InfoNCE loss
|   |   +-- encode_passages.py       # Batch encode all passages
|   +-- faiss_index/
|   |   +-- build_index.py           # Build and save FAISS index
|   |   +-- search.py                # Query the index
|
+-- reranking/
|   +-- cross_encoder/
|   |   +-- model.py                 # CrossEncoderReranker class
|   |   +-- train.py                 # Pairwise ranking loss training
|   |   +-- rerank.py                # Rerank top-K candidates
|   +-- colbert/
|   |   +-- maxsim.py                # MaxSim late interaction scoring
|   +-- distillation/
|       +-- generate_soft_labels.py  # Run teacher, save scores
|       +-- train_student_tf.py      # TF Keras student training
|
+-- generation/
|   +-- ollama_generator.py          # Ollama local LLM wrapper
|   +-- prompt_templates.py          # RAG prompt templates
|
+-- evaluation/
|   +-- evaluate.py                  # NDCG@10, MRR@10 via pytrec_eval
|   +-- run_baseline_bm25.py         # BM25 baseline for comparison
|   +-- latency_benchmark.py         # Per-stage latency measurement
|
+-- serving/
|   +-- api.py                       # FastAPI application
|   +-- pipeline.py                  # End-to-end query pipeline
|
+-- tracking/
|   +-- log_experiment.py            # MLflow logging helpers
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_biencoder_training.ipynb  # Run on Kaggle
|   +-- 03_reranker_training.ipynb   # Run on Kaggle
|   +-- 04_distillation_tf.ipynb
|   +-- 05_evaluation.ipynb
|   +-- 06_end_to_end_demo.ipynb
|
+-- configs/
|   +-- biencoder_config.yaml
|   +-- reranker_config.yaml
|   +-- pipeline_config.yaml
|
+-- requirements.txt
+-- README.md
+-- project.md                       # This file
```

---

## Setup and Running

### 1. Install Dependencies

```bash
pip install torch transformers sentence-transformers
pip install faiss-cpu datasets pytrec_eval
pip install tensorflow fastapi uvicorn mlflow
pip install rank_bm25 tqdm numpy pandas
```

### 2. Install Ollama and Pull a Model

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Pull your preferred model
ollama pull mistral        # recommended (7B, ~4GB)
ollama pull phi3           # lightweight alternative (3.8B, ~2.3GB)
```

### 3. Prepare Data

```bash
python data/download_msmarco.py
python data/hard_negative_mining.py --top_k 100 --negatives_per_query 5
python data/prepare_triples.py
```

### 4. Build BM25 Baseline

```bash
python evaluation/run_baseline_bm25.py --split validation --output results/bm25_run.json
```

### 5. Train Bi-Encoder (run on Kaggle)

```bash
# Upload notebooks/02_biencoder_training.ipynb to Kaggle
# Download checkpoint after run completes
python retrieval/faiss_index/build_index.py --checkpoint checkpoints/biencoder/
```

### 6. Train Cross-Encoder (run on Kaggle)

```bash
# Upload notebooks/03_reranker_training.ipynb to Kaggle
# Download checkpoint after run completes
```

### 7. Run Evaluation

```bash
python evaluation/evaluate.py \
  --biencoder_checkpoint checkpoints/biencoder/ \
  --reranker_checkpoint  checkpoints/reranker/ \
  --split validation \
  --output results/
```

### 8. Start MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

### 9. Start the API Server

```bash
uvicorn serving.api:app --reload --port 8000
# Open http://localhost:8000/docs for Swagger UI
```

**Example query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes inflation?", "top_k": 5}'
```

---

## Results Baseline

Every experiment is logged to MLflow. Key metrics tracked per run:

- `ndcg_cut_10` -- NDCG at rank 10
- `recip_rank` -- Mean Reciprocal Rank (MRR)
- `map` -- Mean Average Precision
- `latency_biencoder_ms` -- bi-encoder query time
- `latency_reranker_ms` -- reranker time for top-50
- `latency_llm_ms` -- Ollama generation time
- `model_name`, `hard_negatives`, `temperature`, `batch_size` -- hyperparameters

---

## Design Decisions and Tradeoffs

**Why bi-encoder + cross-encoder and not just cross-encoder?**
A cross-encoder over millions of passages is computationally infeasible at query time. The bi-encoder does a fast approximate sweep to get top-50; the cross-encoder spends more compute on a small candidate set. This is the standard two-stage retrieval pattern used in production at scale.

**Why FAISS and not a vector database like ChromaDB?**
FAISS is a pure library with no server overhead and no persistence layer to manage. For a research project it is faster to iterate with. ChromaDB would be appropriate if persistence, filtering, or metadata queries were a priority.

**Why hard negatives and not just random negatives?**
Random negatives are too easy -- the model quickly learns to separate them. Hard negatives (passages that are lexically similar but not relevant) force the encoder to learn genuine semantic distinctions and produce a much stronger retriever.

**Why distill to TensorFlow instead of just using a smaller PyTorch model?**
The TF distillation component exists to demonstrate cross-framework proficiency in a justified way. The teacher-student pattern also shows understanding of production ML constraints (accuracy vs latency tradeoff) beyond just training models.

**Why Ollama instead of a quantized HuggingFace model?**
Ollama handles model management, quantization, and serving with a single command. Running a quantized model through HuggingFace directly requires more boilerplate and more RAM. For a local free setup, Ollama is the most practical choice.

**Why MS MARCO and not a domain-specific dataset?**
MS MARCO allows direct comparison against published results (DPR, ColBERT, SPLADE) which gives your evaluation table credibility. A domain-specific dataset would require building your own qrels from scratch which adds significant overhead for no measurable benefit at the demonstration stage.

---

## Scaling Considerations

These are not implemented but should be documented in the README as production awareness:

**Scaling the index beyond 1M passages:**
- Switch from `IndexFlatIP` (exact) to `IndexIVFPQ` (approximate, 8x compression)
- Cluster passages into Voronoi cells, search only top-N cells at query time
- Expected: 10-50x speedup with ~2-3% NDCG degradation

**Scaling reranking throughput:**
- Batch queries together and run the cross-encoder with dynamic batching
- Use the TF student model in production -- same accuracy, half the latency
- Async FastAPI endpoints to handle concurrent requests

**Scaling the LLM generator:**
- Ollama supports concurrent requests via its REST API
- For higher throughput, replace Ollama with vLLM (PagedAttention, continuous batching)
- Both are free and local

---

## Roadmap

- [x] Architecture design
- [x] Free infrastructure plan
- [ ] BM25 baseline + evaluation harness
- [ ] Bi-encoder fine-tuning with hard negatives
- [ ] FAISS index build and search
- [ ] Cross-encoder reranker training
- [ ] ColBERT MaxSim integration
- [ ] TF student distillation
- [ ] Ollama generator integration
- [ ] FastAPI serving layer
- [ ] BEIR zero-shot generalization evaluation
- [ ] MLflow experiment dashboard screenshots in README
- [ ] Latency benchmark table
- [ ] End-to-end demo notebook

---

## References

- Karpukhin et al. (2020) -- Dense Passage Retrieval for Open-Domain Question Answering (DPR)
- Nogueira and Cho (2019) -- Passage Re-ranking with BERT
- Khattab and Zaharia (2020) -- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction
- Thakur et al. (2021) -- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models
- MS MARCO: https://microsoft.github.io/msmarco/
- sentence-transformers: https://www.sbert.net/
- Ollama: https://ollama.com/
