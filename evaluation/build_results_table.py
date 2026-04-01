from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_json, load_qrels, write_json
from neural_rag.evaluation import evaluate_run
from neural_rag.mlflow_utils import flatten_mapping, sanitize_metric_name, start_mlflow_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison table from run metrics and latency benchmarks."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="results-table",
        default_tags={"stage": "evaluation", "script": "evaluation/build_results_table.py"},
    ) as tracker:
        evaluation_config = config.get("evaluation", {})
        outputs_config = config.get("outputs", {})
        benchmark_config = config.get("benchmark", {})

        tracker.log_params(flatten_mapping(config))

        qrels = load_qrels(evaluation_config["qrels_path"])
        k = int(evaluation_config.get("k", 10))

        benchmark_payload = {}
        if benchmark_config.get("benchmark_path"):
            benchmark_payload = load_json(benchmark_config["benchmark_path"])
        stage_summaries = benchmark_payload.get("stages", {})

        methods = []
        for method in config.get("methods", []):
            run_payload = load_json(method["run_path"])
            metrics = evaluate_run(qrels, run_payload, k=k)
            latency_stage = str(method.get("latency_stage", ""))
            latency_summary = stage_summaries.get(latency_stage, {})
            methods.append(
                {
                    "id": str(method["id"]),
                    "name": str(method["name"]),
                    "run_path": str(method["run_path"]),
                    "latency_stage": latency_stage or None,
                    "metrics": metrics,
                    "latency": latency_summary or None,
                }
            )

        write_json(outputs_config["json_path"], {"k": k, "methods": methods})

        header = f"| Method | NDCG@{k} | MRR@{k} | MAP | Mean Latency (ms) | P95 Latency (ms) |"
        divider = f"|---|---|---|---|---|---|"
        rows = [header, divider]
        for method in methods:
            metrics = method["metrics"]
            latency = method["latency"] or {}
            rows.append(
                "| "
                + " | ".join(
                    [
                        method["name"],
                        format_float(float(metrics[f"ndcg@{k}"])),
                        format_float(float(metrics[f"mrr@{k}"])),
                        format_float(float(metrics["map"])),
                        format_latency(
                            float(latency["mean_ms"]) if latency.get("mean_ms") is not None else None
                        ),
                        format_latency(
                            float(latency["p95_ms"]) if latency.get("p95_ms") is not None else None
                        ),
                    ]
                )
                + " |"
            )

        markdown_path = Path(outputs_config["markdown_path"])
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        tracker.log_metrics(
            {
                sanitize_metric_name(f"{method['id']}.{metric_name}"): metric_value
                for method in methods
                for metric_name, metric_value in method["metrics"].items()
            }
        )
        tracker.log_artifact(outputs_config["json_path"], artifact_path="outputs")
        tracker.log_artifact(outputs_config["markdown_path"], artifact_path="outputs")

        print(f"Saved comparison JSON to {outputs_config['json_path']}")
        print(f"Saved comparison Markdown to {outputs_config['markdown_path']}")


if __name__ == "__main__":
    main()
