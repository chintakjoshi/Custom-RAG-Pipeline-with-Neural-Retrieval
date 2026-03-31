from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def resolve_device(use_cpu: bool = False) -> str:
    if use_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


class ColBERTReranker:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int = 128,
        use_cpu: bool = False,
    ) -> None:
        self.device = resolve_device(use_cpu=use_cpu)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        token_embeddings = F.normalize(outputs.last_hidden_state, p=2, dim=-1)
        token_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
        return token_embeddings, token_mask
