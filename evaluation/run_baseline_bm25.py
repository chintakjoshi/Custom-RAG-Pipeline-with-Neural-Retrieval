from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import (
    load_corpus,
    load_qrels,
    load_queries,
    write_json,
)
from neural_rag.evaluation import evaluate_run
from neural_rag.retrieval.bm25 import BM25Retriever


DEFAULTS: dict[str, Any] = {
    "corpus": "data/sample/corpus.jsonl",
    "queries": "data/sample/queries.jsonl",
    "qrels": "data/sample/qrels.jsonl",
    "output": "results/sample_bm25_run.json",
    "metrics_output": "results/sample_bm25_metrics.json",
    "top_k": 10,
    "k1": 1.5,
    "b": 0.75,
    "eval_k": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 BM25 baseline.")
    parser.add_argument("--config")
    parser.add_argument("--corpus")
    parser.add_argument("--queries")
    parser.add_argument("--qrels")
    parser.add_argument("--output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--k1", type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--eval-k", type=int)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(DEFAULTS)

    if args.config:
        config = load_config(args.config)
        dataset = config.get("dataset", {})
        retrieval = config.get("retrieval", {})
        outputs = config.get("outputs", {})

        settings.update(
            {
                "corpus": dataset.get("corpus_path", settings["corpus"]),
                "queries": dataset.get("queries_path", settings["queries"]),
                "qrels": dataset.get("qrels_path", settings["qrels"]),
                "output": outputs.get("run_path", settings["output"]),
                "metrics_output": outputs.get(
                    "metrics_path",
                    settings["metrics_output"],
                ),
                "top_k": retrieval.get("top_k", settings["top_k"]),
                "k1": retrieval.get("k1", settings["k1"]),
                "b": retrieval.get("b", settings["b"]),
                "eval_k": retrieval.get("eval_k", settings["eval_k"]),
            }
        )

    cli_overrides = {
        "corpus": args.corpus,
        "queries": args.queries,
        "qrels": args.qrels,
        "output": args.output,
        "metrics_output": args.metrics_output,
        "top_k": args.top_k,
        "k1": args.k1,
        "b": args.b,
        "eval_k": args.eval_k,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            settings[key] = value

    return settings


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)

    corpus_path = Path(settings["corpus"])
    queries_path = Path(settings["queries"])
    qrels_path = Path(settings["qrels"])

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    retriever = BM25Retriever(
        corpus,
        k1=float(settings["k1"]),
        b=float(settings["b"]),
    )

    run: dict[str, dict[str, float]] = {}
    for query_id, query_text in queries.items():
        results = retriever.search(query_text, top_k=int(settings["top_k"]))
        run[query_id] = {result.doc_id: result.score for result in results}

    write_json(settings["output"], run)
    metrics = evaluate_run(qrels, run, k=int(settings["eval_k"]))
    write_json(settings["metrics_output"], metrics)

    print(f"Saved run to {settings['output']}")
    print(f"Saved metrics to {settings['metrics_output']}")
    for key, value in metrics.items():
        if key == "num_queries":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
