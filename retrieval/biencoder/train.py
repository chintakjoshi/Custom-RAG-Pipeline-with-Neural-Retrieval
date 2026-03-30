from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from neural_rag.config import load_config
from neural_rag.datasets import load_triples, write_json
from retrieval.biencoder.model import load_biencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 2 bi-encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def build_train_dataset(triples_path: str, query_prefix: str, passage_prefix: str) -> Dataset:
    rows = []
    for record in load_triples(triples_path):
        rows.append(
            {
                "anchor": f"{query_prefix}{record['query_text']}",
                "positive": f"{passage_prefix}{record['positive_text']}",
                "negative": f"{passage_prefix}{record['negative_text']}",
            }
        )

    if not rows:
        raise ValueError(f"No triplets found in {triples_path}")

    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})

    query_prefix = str(model_config.get("query_prefix", ""))
    passage_prefix = str(model_config.get("passage_prefix", ""))
    output_dir = str(training_config["output_dir"])
    use_cpu = bool(training_config.get("use_cpu", False))

    train_dataset = build_train_dataset(
        triples_path=str(data_config["train_triplets_path"]),
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )

    model = load_biencoder(str(model_config["model_name"]), use_cpu=use_cpu)
    max_seq_length = model_config.get("max_seq_length")
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)

    loss = MultipleNegativesRankingLoss(
        model,
        scale=float(loss_config.get("scale", 20.0)),
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 8)),
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_steps=float(training_config.get("warmup_steps", training_config.get("warmup_ratio", 0.1))),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_strategy=str(training_config.get("save_strategy", "epoch")),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        seed=int(training_config.get("seed", 42)),
        report_to="none",
        run_name=str(training_config.get("run_name", "biencoder-training")),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        use_cpu=use_cpu,
        dataloader_num_workers=0,
        dataloader_pin_memory=not use_cpu,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    trainer.save_model()

    summary = {
        "model_name": str(model_config["model_name"]),
        "train_triplets_path": str(data_config["train_triplets_path"]),
        "output_dir": output_dir,
        "num_examples": len(train_dataset),
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "use_cpu": use_cpu,
    }
    write_json(Path(output_dir) / "training_summary.json", summary)

    print(f"Training finished. Model saved to {output_dir}")
    print(f"Examples: {len(train_dataset)}")


if __name__ == "__main__":
    main()
