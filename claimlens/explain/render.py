"""Render attributions as HTML: background shading proportional to positive attribution."""

from __future__ import annotations

import html
from pathlib import Path

from claimlens.schemas import Attribution

STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}"
    "h2{font-size:1rem;margin:1.5rem 0 .3rem}.meta{color:#555;font-size:.85rem}"
    ".clause{line-height:1.9;border:1px solid #ddd;padding:.8rem;border-radius:6px}"
    "span.w{padding:1px 0;border-radius:3px}"
)


def attribution_html(attr: Attribution) -> str:
    pos = [max(0.0, s) for s in attr.scores]
    top = max(pos) if pos else 0.0
    parts = []
    for tok, p in zip(attr.tokens, pos, strict=True):
        alpha = 0.0 if top == 0 else p / top
        style = f' style="background:rgba(255,140,0,{alpha:.2f})"' if alpha > 0.02 else ""
        parts.append(f'<span class="w"{style} title="{p:.3f}">{html.escape(tok)}</span>')
    return " ".join(parts)


def write_html(path: Path, items: list[tuple[str, str, float | None, Attribution]]) -> None:
    """items: (title, reason, match_prob, attribution)."""
    body = []
    for title, reason, prob, attr in items:
        p = f" &middot; match probability {prob:.3f}" if prob is not None else ""
        body.append(
            f"<h2>{html.escape(title)}</h2><div class='meta'>Reason: {html.escape(reason)}{p}"
            f" &middot; method: {attr.method}</div>"
            f"<div class='clause'>{attribution_html(attr)}</div>"
        )
    doc = (
        "<!doctype html><meta charset='utf-8'><title>IG attributions</title>"
        f"<style>{STYLE}</style><h1>Integrated Gradients (synthetic sample pairs)</h1>"
        + "".join(body)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
