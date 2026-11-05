"""Attention rollout (Abnar and Zuidema, 2020), adapted to text. Torch-free."""

from __future__ import annotations

import numpy as np


def rollout_scores(layers: list[np.ndarray]) -> np.ndarray:
    """Attention flow from [CLS] to each token.

    `layers` holds one (heads, n, n) array per layer, input side first. Heads are averaged,
    half of each token's weight stays on itself (the residual connection), rows are
    renormalised, and the matrices are multiplied through the layers.
    """
    n = layers[0].shape[-1]
    flow = np.eye(n)
    for attention in layers:
        step = (attention.mean(axis=0) + np.eye(n)) / 2
        step = step / step.sum(axis=-1, keepdims=True)
        flow = step @ flow
    return flow[0]
