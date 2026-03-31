from __future__ import annotations

import torch
from sentence_transformers.cross_encoder import CrossEncoder


def resolve_device(use_cpu: bool = False) -> str:
    if use_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def load_cross_encoder(
    model_name_or_path: str,
    *,
    use_cpu: bool = False,
    max_length: int | None = None,
    num_labels: int = 1,
) -> CrossEncoder:
    return CrossEncoder(
        model_name_or_path,
        num_labels=num_labels,
        max_length=max_length,
        device=resolve_device(use_cpu=use_cpu),
    )
