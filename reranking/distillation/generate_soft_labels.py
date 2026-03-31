from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import read_jsonl, write_json, write_jsonl
from reranking.cross_encoder.model import load_cross_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-encoder soft labels for TensorFlow distillation."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("teacher_model", {})
    inference_config = config.get("inference", {})

    pair_records = read_jsonl(data_config["pairs_path"])
    if not pair_records:
        raise ValueError(f"No query-passage pairs found in {data_config['pairs_path']}")

    teacher = load_cross_encoder(
        str(model_config["model_name_or_path"]),
        use_cpu=bool(inference_config.get("use_cpu", False)),
        max_length=int(model_config.get("max_length", 256)),
        num_labels=int(model_config.get("num_labels", 1)),
    )

    pair_inputs = [
        (str(record["query_text"]), str(record["passage_text"]))
        for record in pair_records
    ]
    scores = teacher.predict(
        pair_inputs,
        batch_size=int(inference_config.get("batch_size", 16)),
        show_progress_bar=bool(inference_config.get("show_progress_bar", True)),
        convert_to_numpy=True,
    )

    output_records = []
    for record, score in zip(pair_records, scores.tolist()):
        output_record = dict(record)
        output_record["teacher_score"] = float(score)
        output_records.append(output_record)

    output_path = data_config["output_path"]
    write_jsonl(output_path, output_records)

    summary_path = Path(output_path).with_suffix(".summary.json")
    score_values = [float(score) for score in scores.tolist()]
    write_json(
        summary_path,
        {
            "teacher_model_name_or_path": str(model_config["model_name_or_path"]),
            "pairs_path": str(data_config["pairs_path"]),
            "output_path": str(output_path),
            "num_pairs": len(output_records),
            "min_teacher_score": min(score_values),
            "max_teacher_score": max(score_values),
            "mean_teacher_score": statistics.fmean(score_values),
        },
    )

    print(f"Saved distillation soft labels to {output_path}")
    print(f"Pairs: {len(output_records)}")


if __name__ == "__main__":
    main()
