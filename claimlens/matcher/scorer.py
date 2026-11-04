"""Pretrained cross-encoder as clause matcher (SPEC.md 7.3). No fine-tuning in Phase 1.

`Matcher` takes a Hugging Face model name or a local checkpoint path, so a fine-tuned
model can drop in later (SPEC.md 12.1).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.schemas import Chunk, MatchResult
from claimlens.utils.config import load_config

log = logging.getLogger(__name__)
MAX_LEN = 512


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class Matcher:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        name = model_name or load_config()["matcher"]["model"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(self.device).eval()
        self.name = name

    def encode(self, reason: str, clause: str) -> Any:
        """Tokenise (reason, clause); only the clause is truncated."""
        return self.tokenizer(
            reason, clause, truncation="only_second", max_length=MAX_LEN, return_tensors="pt"
        ).to(self.device)

    @torch.no_grad()
    def logits(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = self.tokenizer(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=True,
                truncation="only_second",
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            lg = self.model(**enc).logits
            out.append(lg[:, 0].float().cpu().numpy())  # single-logit relevance head
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, reason: str, clauses: Sequence[Chunk]) -> list[MatchResult]:
        lg = self.logits([(reason, c.text) for c in clauses])
        return [
            MatchResult(clause_id=c.chunk_id, score=float(s), prob=float(sigmoid(s)))
            for c, s in zip(clauses, lg, strict=True)
        ]
