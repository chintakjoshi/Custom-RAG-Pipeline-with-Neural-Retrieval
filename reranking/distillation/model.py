from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency for TensorFlow reranking
    tf = None

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency for TensorFlow reranking
    AutoTokenizer = None
    TFAutoModelForSequenceClassification = None


def require_tensorflow() -> None:
    if tf is None or AutoTokenizer is None or TFAutoModelForSequenceClassification is None:
        raise RuntimeError(
            "TensorFlow-compatible Transformers support is required for the distillation workflow. "
            "Install dependencies from requirements.txt inside .venv first."
        )


class TensorFlowStudentReranker:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int = 256,
        num_labels: int = 1,
        from_pt: bool = False,
        use_safetensors: bool = False,
    ) -> None:
        require_tensorflow()
        self.max_length = max_length
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            from_pt=from_pt,
            use_safetensors=use_safetensors,
        )
        self.model.config.problem_type = "regression"

    def tokenize_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> dict[str, tf.Tensor]:
        require_tensorflow()
        return self.tokenizer(
            list(queries),
            list(passages),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )

    def build_training_dataset(
        self,
        pairs: Sequence[tuple[str, str]],
        labels: Sequence[float],
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tf.data.Dataset:
        require_tensorflow()
        if len(pairs) != len(labels):
            raise ValueError("Pairs and labels must have the same length.")

        queries = [query for query, _ in pairs]
        passages = [passage for _, passage in pairs]
        features = dict(self.tokenize_pairs(queries, passages))
        features["labels"] = tf.convert_to_tensor(labels, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices(features)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=max(len(pairs), 1),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        require_tensorflow()
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            if not batch_pairs:
                continue
            queries = [query for query, _ in batch_pairs]
            passages = [passage for _, passage in batch_pairs]
            features = self.tokenize_pairs(queries, passages)
            logits = self.model(features, training=False).logits
            batch_scores = tf.reshape(logits, [-1]).numpy().tolist()
            scores.extend(float(score) for score in batch_scores)
        return np.asarray(scores, dtype=np.float32)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
