"""Word attributions for the matcher (SPEC.md 7.4, 13.4).

`attribute(method=...)` is the single entry point. "ig", "shap" and "rollout" all return the
same `Attribution`: the clause's merged words with one score each, L2-normalised. IG takes a
`baseline`: "pad" (pad token, the default) or "zero" (zero embedding), for sensitivity checks.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from claimlens.explain.rollout import rollout_scores
from claimlens.explain.wordpieces import l2_normalise, merge_wordpieces
from claimlens.matcher.scorer import Matcher
from claimlens.schemas import Attribution

IG_NAMES = ("ig", "integrated_gradients")


def _prepare(matcher: Matcher, reason: str, clause_text: str):
    tok = matcher.tokenizer
    enc = matcher.encode(reason, clause_text)
    ids = enc["input_ids"]
    token_type = enc["token_type_ids"] if "token_type_ids" in enc else torch.zeros_like(ids)
    special = torch.tensor(
        tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
        device=ids.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    in_clause = (token_type[0] == 1).cpu().numpy() & ~special[0].cpu().numpy()
    return ids, token_type, enc["attention_mask"], special, in_clause


def _to_words(matcher: Matcher, ids: torch.Tensor, per_token: np.ndarray, in_clause: np.ndarray,
              clause_id: str, method: str) -> Attribution:
    all_tokens = matcher.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    toks = [t for t, k in zip(all_tokens, in_clause, strict=True) if k]
    words, scores = merge_wordpieces(toks, l2_normalise(per_token[in_clause]).tolist())
    return Attribution(clause_id=clause_id, method=method, tokens=words, scores=scores)


def _ig_tokens(matcher: Matcher, ids, token_type, mask, special, n_steps: int,
               baseline: str) -> np.ndarray:
    tok, model = matcher.tokenizer, matcher.model
    layer = model.base_model.embeddings.word_embeddings
    if baseline == "pad":  # pad token everywhere except special tokens, which are kept
        def forward(input_ids, tt, am):
            return model(input_ids=input_ids, token_type_ids=tt, attention_mask=am).logits[:, 0]

        base = torch.where(special, ids, torch.full_like(ids, tok.pad_token_id))
        attr = LayerIntegratedGradients(forward, layer).attribute(
            ids, baselines=base, additional_forward_args=(token_type, mask),
            n_steps=n_steps, internal_batch_size=10)
    elif baseline == "zero":  # zero embedding, special tokens kept
        def forward_embeds(embeds, tt, am):
            return model(inputs_embeds=embeds, token_type_ids=tt, attention_mask=am).logits[:, 0]

        embeds = layer(ids).detach()
        base = embeds.masked_fill(~special.unsqueeze(-1), 0.0)
        attr = IntegratedGradients(forward_embeds).attribute(
            embeds, baselines=base, additional_forward_args=(token_type, mask),
            n_steps=n_steps, internal_batch_size=10)
    else:
        raise ValueError(f"unknown IG baseline {baseline!r}")
    return attr.sum(dim=-1)[0].detach().float().cpu().numpy()


@torch.no_grad()
def _rollout_tokens(matcher: Matcher, ids, token_type, mask) -> np.ndarray:
    out = matcher.model(input_ids=ids, token_type_ids=token_type, attention_mask=mask,
                        output_attentions=True)
    layers = [a[0].float().cpu().numpy() for a in out.attentions]
    return rollout_scores(layers)


def attribute(
    matcher: Matcher,
    reason: str,
    clause_text: str,
    clause_id: str = "",
    method: str = "ig",
    n_steps: int = 50,
    baseline: str = "pad",
) -> Attribution:
    ids, token_type, mask, special, in_clause = _prepare(matcher, reason, clause_text)
    if method in IG_NAMES:
        per_token = _ig_tokens(matcher, ids, token_type, mask, special, n_steps, baseline)
        return _to_words(matcher, ids, per_token, in_clause, clause_id, "integrated_gradients")
    if method == "rollout":
        per_token = _rollout_tokens(matcher, ids, token_type, mask)
        return _to_words(matcher, ids, per_token, in_clause, clause_id, "rollout")
    if method == "shap":
        from claimlens.explain.shap_text import shap_token_values

        per_token = shap_token_values(matcher, reason, clause_text, ids, in_clause)
        return _to_words(matcher, ids, per_token, in_clause, clause_id, "shap")
    raise ValueError(f"unknown attribution method {method!r}")
