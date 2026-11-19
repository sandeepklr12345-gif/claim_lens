"""Cross-encoder clause matcher (SPEC.md 7.3, 13.3).

`Matcher` takes a Hugging Face model name or a local checkpoint path. configs/app.yaml picks
the model. models/ is git-ignored, so a fresh clone has no checkpoint: FALLBACK to
`matcher.fallback_model` (the pretrained Phase 1 model) with a logged warning.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.schemas import Chunk, MatchResult
from claimlens.utils.config import ROOT, load_config

log = logging.getLogger(__name__)
MAX_LEN = 512
TRAIN_META = "train_meta.json"  # written next to a fine-tuned checkpoint


def resolve_model(name: str | None = None) -> str:
    """Return a loadable model id: a local checkpoint if present, else the configured model."""
    cfg = load_config()["matcher"]
    name = name or cfg["model"]
    local = ROOT / name  # an absolute `name` replaces ROOT here
    if local.is_dir():
        return str(local)
    if name.replace("\\", "/").startswith("models/"):
        log.warning("checkpoint %s not found, using pretrained %s", name, cfg["fallback_model"])
        return str(cfg["fallback_model"])
    return name


def checkpoint_max_len(name: str) -> int:
    """Score at the length a checkpoint was trained with; pretrained models keep MAX_LEN."""
    meta = Path(name) / TRAIN_META
    if meta.is_file():
        return int(json.loads(meta.read_text(encoding="utf-8"))["max_length"])
    return MAX_LEN


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class Matcher:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        name = resolve_model(model_name)
        self.max_len = checkpoint_max_len(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(self.device).eval()
        self.name = name

    def encode(self, reason: str, clause: str) -> Any:
        """Tokenise (reason, clause); only the clause is truncated."""
        return self.tokenizer(
            reason, clause, truncation="only_second", max_length=self.max_len,
            return_tensors="pt",
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
                max_length=self.max_len,
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
