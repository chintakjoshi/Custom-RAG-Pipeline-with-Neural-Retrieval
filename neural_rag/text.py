from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase tokenization that is good enough for a BM25 baseline."""
    return TOKEN_PATTERN.findall(text.lower())
