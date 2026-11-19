"""SHAP text explainer over the clause, with the reason held fixed (SPEC.md 13.4).

The masker uses the matcher's own tokenizer so values line up one to one with the same
word-pieces as IG; `attribute` then merges them into words.
"""

from __future__ import annotations

import numpy as np
import shap
import torch

from claimlens.matcher.scorer import Matcher

MAX_EVALS = 2000  # the default 500 leaves about 2 samples per token on a 200-word clause


def visible_clause(matcher: Matcher, reason: str, clause_text: str, ids: torch.Tensor) -> str:
    """The part of the clause that survives truncation next to this reason."""
    tok = matcher.tokenizer
    enc = tok(reason, clause_text, truncation="only_second", max_length=matcher.max_len,
              return_offsets_mapping=True)
    if enc["input_ids"] != ids[0].tolist():
        raise ValueError("tokenised pair differs from the one being explained")
    ends = [end for (_, end), kind in zip(enc["offset_mapping"], enc.sequence_ids(), strict=True)
            if kind == 1]
    return clause_text[:ends[-1]] if ends else ""


def shap_token_values(matcher: Matcher, reason: str, clause_text: str, ids: torch.Tensor,
                      in_clause: np.ndarray, max_evals: int = MAX_EVALS) -> np.ndarray:
    """SHAP value of each token of `ids`; zero outside the clause."""
    clause = visible_clause(matcher, reason, clause_text, ids)

    def predict(texts) -> np.ndarray:
        return matcher.logits([(reason, str(t)) for t in texts])

    masker = shap.maskers.Text(matcher.tokenizer, mask_token=matcher.tokenizer.mask_token,
                               collapse_mask_token=False)
    explanation = shap.Explainer(predict, masker, algorithm="partition")(
        [clause], max_evals=max_evals, silent=True)
    values = np.asarray(explanation.values[0], dtype=np.float64).reshape(-1)
    values = values[1:-1]  # the masker adds [CLS] and [SEP] around the clause
    if len(values) != int(in_clause.sum()):
        raise ValueError(f"SHAP gave {len(values)} tokens, expected {int(in_clause.sum())}")
    out = np.zeros(len(in_clause))
    out[in_clause] = values
    return out
