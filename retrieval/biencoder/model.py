from __future__ import annotations

from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer

from neural_rag.text import apply_text_prefix


def resolve_device(use_cpu: bool = False) -> str:
    if use_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def load_biencoder(model_name_or_path: str, *, use_cpu: bool = False) -> SentenceTransformer:
    device = resolve_device(use_cpu=use_cpu)
    return SentenceTransformer(model_name_or_path, device=device)


def prefixed_texts(texts: Iterable[str], prefix: str | None) -> list[str]:
    return [apply_text_prefix(text, prefix) for text in texts]
