from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import read_jsonl, write_json
from reranking.distillation.model import TensorFlowStudentReranker, require_tensorflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow student reranker from soft labels."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    require_tensorflow()
    import tensorflow as tf
    import tf_keras as keras

    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    records = read_jsonl(data_config["train_soft_labels_path"])
    if not records:
        raise ValueError(
            f"No distillation training records found in {data_config['train_soft_labels_path']}"
        )

    seed = int(training_config.get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    train_pairs = [
        (str(record["query_text"]), str(record["passage_text"]))
        for record in records
    ]
    teacher_scores = [float(record["teacher_score"]) for record in records]
    target_transform = str(training_config.get("target_transform", "zscore")).lower()
    target_mean = statistics.fmean(teacher_scores)
    target_std = 1.0
    train_targets = teacher_scores

    if target_transform == "zscore":
        variance = statistics.fmean(
            (score - target_mean) ** 2 for score in teacher_scores
        )
        target_std = variance ** 0.5 or 1.0
        train_targets = [
            (score - target_mean) / target_std
            for score in teacher_scores
        ]
    elif target_transform != "none":
        raise ValueError(
            "training.target_transform must be either 'zscore' or 'none'."
        )

    student = TensorFlowStudentReranker(
        str(model_config["model_name_or_path"]),
        max_length=int(model_config.get("max_length", 256)),
        num_labels=int(model_config.get("num_labels", 1)),
        from_pt=bool(model_config.get("from_pt", False)),
        use_safetensors=bool(model_config.get("use_safetensors", False)),
    )

    train_dataset = student.build_training_dataset(
        train_pairs,
        train_targets,
        batch_size=int(training_config.get("train_batch_size", 8)),
        shuffle=True,
        seed=seed,
    )

    student.model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(training_config.get("learning_rate", 3e-5))
        )
    )
    history = student.model.fit(
        train_dataset,
        epochs=int(training_config.get("epochs", 1)),
        verbose=1 if bool(training_config.get("verbose", True)) else 0,
    )

    output_dir = Path(training_config["output_dir"])
    student.save_pretrained(output_dir)

    write_json(
        output_dir / "training_summary.json",
        {
            "base_model_name_or_path": str(model_config["model_name_or_path"]),
            "train_soft_labels_path": str(data_config["train_soft_labels_path"]),
            "num_examples": len(records),
            "epochs": int(training_config.get("epochs", 1)),
            "train_batch_size": int(training_config.get("train_batch_size", 8)),
            "learning_rate": float(training_config.get("learning_rate", 3e-5)),
            "seed": seed,
            "target_transform": target_transform,
            "target_mean": target_mean,
            "target_std": target_std,
            "final_loss": float(history.history["loss"][-1]),
        },
    )

    print(f"Training finished. TensorFlow student saved to {output_dir}")
    print(f"Examples: {len(records)}")


if __name__ == "__main__":
    main()
