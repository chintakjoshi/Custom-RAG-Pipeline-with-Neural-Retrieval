from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import read_jsonl, write_json
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run
from reranking.cross_encoder.model import load_cross_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cross-encoder reranker.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="cross-encoder-train",
        default_tags={"stage": "cross_encoder", "script": "reranking/cross_encoder/train.py"},
    ) as tracker:
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        training_config = config.get("training", {})

        tracker.log_params(
            flatten_mapping(
                {
                    "data": data_config,
                    "model": model_config,
                    "training": training_config,
                }
            )
        )

        pair_records = read_jsonl(data_config["train_pairs_path"])
        train_examples = [
            InputExample(
                texts=[str(record["query_text"]), str(record["passage_text"])],
                label=float(record["label"]),
            )
            for record in pair_records
        ]
        if not train_examples:
            raise ValueError(f"No reranker pairs found in {data_config['train_pairs_path']}")

        model = load_cross_encoder(
            str(model_config["model_name"]),
            use_cpu=bool(training_config.get("use_cpu", False)),
            max_length=int(model_config.get("max_length", 256)),
            num_labels=int(model_config.get("num_labels", 1)),
        )

        dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=int(training_config.get("train_batch_size", 8)),
            pin_memory=False,
        )

        warmup_ratio = float(training_config.get("warmup_ratio", 0.1))
        total_steps = max(1, len(dataloader) * int(training_config.get("epochs", 1)))
        warmup_steps = int(total_steps * warmup_ratio)

        model.fit(
            train_dataloader=dataloader,
            epochs=int(training_config.get("epochs", 1)),
            warmup_steps=warmup_steps,
            optimizer_params={"lr": float(training_config.get("learning_rate", 2e-5))},
            weight_decay=float(training_config.get("weight_decay", 0.01)),
            output_path=str(training_config["output_dir"]),
            show_progress_bar=bool(training_config.get("show_progress_bar", True)),
            use_amp=False,
            loss_fct=torch.nn.BCEWithLogitsLoss(),
        )
        model.save(str(training_config["output_dir"]))

        summary_path = Path(training_config["output_dir"]) / "training_summary.json"
        write_json(
            summary_path,
            {
                "model_name": str(model_config["model_name"]),
                "train_pairs_path": str(data_config["train_pairs_path"]),
                "num_examples": len(train_examples),
                "epochs": int(training_config.get("epochs", 1)),
                "train_batch_size": int(training_config.get("train_batch_size", 8)),
                "learning_rate": float(training_config.get("learning_rate", 2e-5)),
                "use_cpu": bool(training_config.get("use_cpu", False)),
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
        )

        tracker.log_metrics(
            {
                "num_examples": len(train_examples),
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            }
        )
        tracker.log_artifact(summary_path, artifact_path="outputs")

        print(f"Training finished. Model saved to {training_config['output_dir']}")
        print(f"Examples: {len(train_examples)}")


if __name__ == "__main__":
    main()
