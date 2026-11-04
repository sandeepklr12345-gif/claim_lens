"""Integrated Gradients over the cross-encoder's embedding layer (SPEC.md 7.4).

`attribute(method=...)` is the single entry point so SHAP / attention rollout can plug in
later (SPEC.md 12.1). Attributions are for the *clause* tokens, with respect to the
matcher's relevance logit.
"""

from __future__ import annotations

import torch
from captum.attr import LayerIntegratedGradients

from claimlens.explain.wordpieces import l2_normalise, merge_wordpieces
from claimlens.matcher.scorer import Matcher
from claimlens.schemas import Attribution


def attribute(
    matcher: Matcher,
    reason: str,
    clause_text: str,
    clause_id: str = "",
    method: str = "integrated_gradients",
    n_steps: int = 50,
) -> Attribution:
    if method != "integrated_gradients":
        raise NotImplementedError(f"attribution method {method!r} is Phase 2")
    tok, model = matcher.tokenizer, matcher.model
    enc = matcher.encode(reason, clause_text)
    ids = enc["input_ids"]
    token_type = enc["token_type_ids"] if "token_type_ids" in enc else torch.zeros_like(ids)
    mask = enc["attention_mask"]

    # baseline: pad token everywhere except special tokens ([CLS], [SEP]), which are kept
    special = torch.tensor(
        tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
        device=ids.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    baseline = torch.where(special, ids, torch.full_like(ids, tok.pad_token_id))

    def forward(input_ids: torch.Tensor, tt: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, token_type_ids=tt, attention_mask=am).logits[:, 0]

    lig = LayerIntegratedGradients(forward, model.base_model.embeddings.word_embeddings)
    attr = lig.attribute(
        ids,
        baselines=baseline,
        additional_forward_args=(token_type, mask),
        n_steps=n_steps,
        internal_batch_size=10,
    )
    per_token = attr.sum(dim=-1)[0].detach().float().cpu().numpy()

    all_tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    in_clause = (token_type[0] == 1).cpu().numpy() & ~special[0].cpu().numpy()
    toks = [t for t, k in zip(all_tokens, in_clause, strict=True) if k]
    sc = l2_normalise(per_token[in_clause])
    words, word_scores = merge_wordpieces(toks, sc.tolist())
    return Attribution(clause_id=clause_id, method=method, tokens=words, scores=word_scores)
