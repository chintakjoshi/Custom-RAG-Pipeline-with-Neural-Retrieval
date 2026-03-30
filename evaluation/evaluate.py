from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.datasets import load_json, load_qrels, write_json
from neural_rag.evaluation import evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TREC-style run file.")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = load_qrels(args.qrels)
    run = load_json(args.run)
    metrics = evaluate_run(qrels, run, k=args.k)

    if args.output:
        write_json(args.output, metrics)

    for key, value in metrics.items():
        if key == "num_queries":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
