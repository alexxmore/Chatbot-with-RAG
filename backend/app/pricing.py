"""Token → USD cost estimation.

Prices are USD per 1M tokens, from OpenAI's public pricing (may change over
time — update the table when it does). Only models we actually call are listed;
for any other model (e.g. an OpenRouter model) `chat_cost` returns None so the
UI can omit the cost rather than show a wrong number.
"""
from __future__ import annotations

# Fixed embedding model (see indexing.py).
EMBED_MODEL = "text-embedding-3-small"

# USD per 1M tokens: (input, output). Embedding models have no output price.
_PRICES: dict[str, tuple[float, float]] = {
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
}


def embedding_cost(tokens: int) -> float:
    in_price, _ = _PRICES.get(EMBED_MODEL, (0.0, 0.0))
    return round(tokens / 1_000_000 * in_price, 6)


def chat_cost(
    embedding_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float | None:
    """Estimated USD cost of one /chat request, or None for an unpriced model."""
    price = _PRICES.get(model)
    if price is None:
        return None  # unknown model (e.g. OpenRouter) → can't price reliably
    in_price, out_price = price
    cost = embedding_cost(embedding_tokens)
    cost += prompt_tokens / 1_000_000 * in_price
    cost += completion_tokens / 1_000_000 * out_price
    return round(cost, 6)
