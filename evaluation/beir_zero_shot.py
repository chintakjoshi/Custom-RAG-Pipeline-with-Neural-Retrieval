from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BEIR zero-shot dense retrieval evaluation."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve_device(use_cpu: bool = False) -> str:
    if use_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def extract_metric_at_k(metrics: dict[str, float], prefix: str, k: int) -> float | None:
    preferred_key = f"{prefix}@{k}"
    if preferred_key in metrics:
        return float(metrics[preferred_key])

    for key, value in metrics.items():
        if key.lower() == preferred_key.lower():
            return float(value)

    return None


def load_dataset_path(
    dataset_name: str,
    datasets_root: Path,
    *,
    download: bool,
) -> Path:
    candidate_path = datasets_root / dataset_name
    if candidate_path.exists():
        return candidate_path

    if not download:
        raise FileNotFoundError(
            f"BEIR dataset '{dataset_name}' was not found at {candidate_path} and download=false."
        )

    datasets_root.mkdir(parents=True, exist_ok=True)
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        f"{dataset_name}.zip"
    )
    downloaded_path = util.download_and_unzip(url, str(datasets_root))
    return Path(downloaded_path)


def format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    retrieval_config = config.get("retrieval", {})
    output_config = config.get("outputs", {})

    dataset_names = [str(name) for name in dataset_config.get("names", [])]
    if not dataset_names:
        raise ValueError("dataset.names must contain at least one BEIR dataset name.")

    datasets_root = Path(dataset_config.get("root_dir", "artifacts/beir/datasets"))
    split = str(dataset_config.get("split", "test"))
    download = bool(dataset_config.get("download", True))

    device = resolve_device(use_cpu=bool(model_config.get("use_cpu", False)))
    prompts = None
    if model_config.get("query_prefix") or model_config.get("passage_prefix"):
        prompts = {
            "query": str(model_config.get("query_prefix", "")),
            "passage": str(model_config.get("passage_prefix", "")),
        }

    batch_size = int(retrieval_config.get("batch_size", 16))
    corpus_chunk_size = int(retrieval_config.get("corpus_chunk_size", 2048))
    score_function = str(retrieval_config.get("score_function", "cos_sim"))
    k_values = [int(k) for k in retrieval_config.get("k_values", [1, 3, 5, 10, 100])]
    report_k = int(retrieval_config.get("report_k", 10))

    beir_model = SentenceBERT(
        str(model_config["model_name_or_path"]),
        max_length=int(model_config.get("max_length", 256)),
        prompts=prompts,
        device=device,
    )
    retriever = EvaluateRetrieval(
        DRES(
            beir_model,
            batch_size=batch_size,
            corpus_chunk_size=corpus_chunk_size,
            show_progress_bar=bool(retrieval_config.get("show_progress_bar", True)),
            convert_to_tensor=True,
        ),
        k_values=k_values,
        score_function=score_function,
    )

    output_dir = Path(output_config.get("output_dir", "results/beir_zero_shot"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict[str, object]] = []
    markdown_lines = [
        "| Dataset | Split | NDCG@{} | MRR@{} | MAP@{} | Queries | Corpus | Device | Time (s) |".format(
            report_k,
            report_k,
            report_k,
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for dataset_name in dataset_names:
        data_path = load_dataset_path(dataset_name, datasets_root, download=download)
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)

        start = time.perf_counter()
        results = retriever.retrieve(corpus, queries)
        elapsed_seconds = time.perf_counter() - start

        ndcg, map_scores, recall, precision = retriever.evaluate(
            qrels,
            results,
            retriever.k_values,
        )
        mrr = retriever.evaluate_custom(
            qrels,
            results,
            retriever.k_values,
            metric="mrr",
        )

        dataset_run_path = output_dir / f"{dataset_name}.run.json"
        dataset_metrics_path = output_dir / f"{dataset_name}.metrics.json"
        write_json(dataset_run_path, results)

        metrics_payload = {
            "dataset": dataset_name,
            "split": split,
            "model_name_or_path": str(model_config["model_name_or_path"]),
            "device": device,
            "num_queries": len(queries),
            "num_corpus_docs": len(corpus),
            "elapsed_seconds": elapsed_seconds,
            "score_function": score_function,
            "k_values": k_values,
            "metrics": {
                "ndcg": ndcg,
                "map": map_scores,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
            },
        }
        write_json(dataset_metrics_path, metrics_payload)

        ndcg_report = extract_metric_at_k(ndcg, "NDCG", report_k)
        mrr_report = extract_metric_at_k(mrr, "MRR", report_k)
        map_report = extract_metric_at_k(map_scores, "MAP", report_k)

        dataset_summary = {
            "dataset": dataset_name,
            "split": split,
            "data_path": str(data_path),
            "run_path": str(dataset_run_path),
            "metrics_path": str(dataset_metrics_path),
            "num_queries": len(queries),
            "num_corpus_docs": len(corpus),
            "elapsed_seconds": elapsed_seconds,
            "ndcg_at_report_k": ndcg_report,
            "mrr_at_report_k": mrr_report,
            "map_at_report_k": map_report,
        }
        dataset_summaries.append(dataset_summary)

        markdown_lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
                dataset_name,
                split,
                format_optional_metric(ndcg_report),
                format_optional_metric(mrr_report),
                format_optional_metric(map_report),
                len(queries),
                len(corpus),
                device,
                elapsed_seconds,
            )
        )

        print(
            f"{dataset_name}: "
            f"NDCG@{report_k}={format_optional_metric(ndcg_report)} "
            f"MRR@{report_k}={format_optional_metric(mrr_report)} "
            f"MAP@{report_k}={format_optional_metric(map_report)} "
            f"time={elapsed_seconds:.2f}s"
        )

    total_elapsed_seconds = sum(
        float(dataset["elapsed_seconds"]) for dataset in dataset_summaries
    )
    aggregate_summary = {
        "num_datasets": len(dataset_summaries),
        "num_queries": sum(int(dataset["num_queries"]) for dataset in dataset_summaries),
        "num_corpus_docs": sum(int(dataset["num_corpus_docs"]) for dataset in dataset_summaries),
        "total_elapsed_seconds": total_elapsed_seconds,
        "mean_elapsed_seconds": (
            total_elapsed_seconds / len(dataset_summaries) if dataset_summaries else 0.0
        ),
        "ndcg_at_report_k": mean_optional(
            [dataset["ndcg_at_report_k"] for dataset in dataset_summaries]
        ),
        "mrr_at_report_k": mean_optional(
            [dataset["mrr_at_report_k"] for dataset in dataset_summaries]
        ),
        "map_at_report_k": mean_optional(
            [dataset["map_at_report_k"] for dataset in dataset_summaries]
        ),
    }

    markdown_lines.append(
        "| Macro Average | - | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
            format_optional_metric(aggregate_summary["ndcg_at_report_k"]),
            format_optional_metric(aggregate_summary["mrr_at_report_k"]),
            format_optional_metric(aggregate_summary["map_at_report_k"]),
            aggregate_summary["num_queries"],
            aggregate_summary["num_corpus_docs"],
            device,
            aggregate_summary["total_elapsed_seconds"],
        )
    )

    summary_payload = {
        "model_name_or_path": str(model_config["model_name_or_path"]),
        "device": device,
        "score_function": score_function,
        "batch_size": batch_size,
        "corpus_chunk_size": corpus_chunk_size,
        "k_values": k_values,
        "report_k": report_k,
        "datasets": dataset_summaries,
        "aggregate": aggregate_summary,
    }

    summary_json_path = Path(output_config["summary_json_path"])
    summary_markdown_path = Path(output_config["summary_markdown_path"])
    write_json(summary_json_path, summary_payload)
    summary_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary_markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"Saved BEIR summary JSON to {summary_json_path}")
    print(f"Saved BEIR summary Markdown to {summary_markdown_path}")


if __name__ == "__main__":
    main()
