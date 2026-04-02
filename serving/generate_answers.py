from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_rag.config import load_config
from neural_rag.datasets import load_corpus, load_json, load_queries, write_json
from neural_rag.evaluation import sort_run
from neural_rag.mlflow_utils import flatten_mapping, start_mlflow_run
from serving.ollama_generator import OllamaGenerator
from serving.prompt_templates import (
    RetrievedPassage,
    build_cited_rag_prompt,
    extract_citation_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded answers from a reranked retrieval run using Ollama."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_query_filter(config: dict[str, object]) -> set[str] | None:
    query_ids = config.get("query_ids", [])
    if not query_ids:
        return None
    return {str(query_id) for query_id in query_ids}


def build_passages(
    corpus: dict[str, str],
    scores: dict[str, float],
    *,
    top_k: int,
) -> list[RetrievedPassage]:
    passages: list[RetrievedPassage] = []
    for rank, (doc_id, score) in enumerate(sort_run(scores)[:top_k], start=1):
        text = corpus.get(doc_id)
        if not text:
            continue
        passages.append(
            RetrievedPassage(
                rank=rank,
                doc_id=doc_id,
                score=float(score),
                text=text,
            )
        )
    return passages


def write_markdown(path: Path, answers: list[dict[str, object]]) -> None:
    lines = ["# Ollama Grounded Answers", ""]
    for answer in answers:
        lines.append(f"## Query {answer['query_id']}")
        lines.append("")
        lines.append(f"Question: {answer['query_text']}")
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append(str(answer["answer"]))
        lines.append("")
        lines.append("Passages:")
        lines.append("")
        for passage in answer["passages"]:
            lines.append(
                f"- [{passage['rank']}] doc_id={passage['doc_id']} "
                f"score={passage['score']:.4f} {passage['text']}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    with start_mlflow_run(
        config.get("mlflow", {}),
        config_path=args.config,
        default_run_name="ollama-generate-answers",
        default_tags={"stage": "generation", "script": "serving/generate_answers.py"},
    ) as tracker:
        data_config = config.get("data", {})
        retrieval_config = config.get("retrieval", {})
        generator_config = config.get("generator", {})
        prompt_config = config.get("prompt", {})
        output_config = config.get("outputs", {})

        tracker.log_params(flatten_mapping(config))

        corpus = load_corpus(data_config["corpus_path"])
        queries = load_queries(data_config["queries_path"])
        run = load_json(data_config["run_path"])
        top_k = int(retrieval_config.get("top_k_passages", 5))
        query_filter = load_query_filter(data_config)
        preview_only = bool(generator_config.get("preview_only", False))

        generator = OllamaGenerator(
            base_url=str(generator_config.get("base_url", "http://localhost:11434")),
            model=str(generator_config["model"]),
            timeout_seconds=int(generator_config.get("timeout_seconds", 180)),
            system_prompt=str(generator_config.get("system_prompt", "")) or None,
            options=dict(generator_config.get("options", {})),
            keep_alive=str(generator_config.get("keep_alive", "")) or None,
            strict_model_check=bool(generator_config.get("strict_model_check", True)),
        )

        available_models: list[str] = []
        if not preview_only:
            available_models = generator.validate_model()

        answers: list[dict[str, object]] = []
        generation_latencies_ms: list[float] = []
        answers_with_citations = 0
        unknown_answers = 0
        unknown_response = str(prompt_config.get("unknown_response", "I don't know."))

        for query_id, query_text in queries.items():
            if query_filter is not None and query_id not in query_filter:
                continue

            passages = build_passages(
                corpus,
                run.get(query_id, {}),
                top_k=top_k,
            )
            if not passages:
                continue

            start = time.perf_counter()
            if preview_only:
                response_payload = {
                    "prompt": build_cited_rag_prompt(
                        query_text,
                        passages,
                        unknown_response=unknown_response,
                    ),
                    "answer": "[preview_only] No model call was made.",
                    "raw_response": None,
                }
            else:
                response_payload = generator.generate_grounded_answer(
                    query=query_text,
                    passages=passages,
                    unknown_response=unknown_response,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            generation_latencies_ms.append(elapsed_ms)

            answer_text = str(response_payload["answer"])
            citation_ids = extract_citation_ids(answer_text)
            if citation_ids:
                answers_with_citations += 1
            if answer_text.strip() == unknown_response:
                unknown_answers += 1

            answers.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "answer": answer_text,
                    "citation_ids": citation_ids,
                    "latency_ms": elapsed_ms,
                    "prompt": response_payload["prompt"],
                    "passages": [
                        {
                            "rank": passage.rank,
                            "doc_id": passage.doc_id,
                            "score": passage.score,
                            "text": passage.text,
                        }
                        for passage in passages
                    ],
                    "raw_response": response_payload["raw_response"],
                }
            )

            print(
                f"{query_id}: generated answer with {len(passages)} passages "
                f"in {elapsed_ms:.2f} ms"
            )

        summary = {
            "base_url": str(generator_config.get("base_url", "http://localhost:11434")),
            "model": str(generator_config["model"]),
            "preview_only": preview_only,
            "available_models": available_models,
            "num_answers": len(answers),
            "top_k_passages": top_k,
            "answers_with_citations": answers_with_citations,
            "unknown_answers": unknown_answers,
            "mean_latency_ms": statistics.fmean(generation_latencies_ms)
            if generation_latencies_ms
            else 0.0,
            "median_latency_ms": statistics.median(generation_latencies_ms)
            if generation_latencies_ms
            else 0.0,
        }

        answers_payload = {
            "summary": summary,
            "answers": answers,
        }

        answers_json_path = Path(output_config["answers_json_path"])
        prompts_json_path = Path(output_config["prompts_json_path"])
        summary_json_path = Path(output_config["summary_json_path"])
        answers_markdown_path = Path(output_config["answers_markdown_path"])

        write_json(answers_json_path, answers_payload)
        write_json(
            prompts_json_path,
            {
                "prompts": [
                    {
                        "query_id": answer["query_id"],
                        "prompt": answer["prompt"],
                    }
                    for answer in answers
                ]
            },
        )
        write_json(summary_json_path, summary)
        write_markdown(answers_markdown_path, answers)

        tracker.log_metrics(
            {
                "num_answers": len(answers),
                "answers_with_citations": answers_with_citations,
                "unknown_answers": unknown_answers,
                "mean_latency_ms": summary["mean_latency_ms"],
                "median_latency_ms": summary["median_latency_ms"],
            }
        )
        tracker.log_artifact(answers_json_path, artifact_path="outputs")
        tracker.log_artifact(prompts_json_path, artifact_path="outputs")
        tracker.log_artifact(summary_json_path, artifact_path="outputs")
        tracker.log_artifact(answers_markdown_path, artifact_path="outputs")

        print(f"Saved answers to {answers_json_path}")
        print(f"Saved prompt payloads to {prompts_json_path}")
        print(f"Saved summary to {summary_json_path}")
        print(f"Saved Markdown to {answers_markdown_path}")


if __name__ == "__main__":
    main()
