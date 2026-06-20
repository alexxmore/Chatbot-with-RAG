"""Offline tests for the cost-estimation table (app/pricing.py)."""
from app import pricing


def test_embedding_cost_one_million_tokens():
    # text-embedding-3-small = $0.02 / 1M input tokens
    assert pricing.embedding_cost(1_000_000) == 0.02


def test_embedding_cost_zero():
    assert pricing.embedding_cost(0) == 0.0


def test_embedding_cost_scales_linearly():
    assert pricing.embedding_cost(500_000) == 0.01


def test_chat_cost_known_model_sums_all_three_components():
    # gpt-4o-mini: input $0.15/1M, output $0.60/1M; embedding adds $0.02/1M.
    cost = pricing.chat_cost(
        embedding_tokens=1_000_000,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        model="gpt-4o-mini",
    )
    assert cost == round(0.02 + 0.15 + 0.60, 6)


def test_chat_cost_unknown_model_returns_none():
    # OpenRouter / unlisted models can't be priced reliably → None (not a wrong number).
    assert pricing.chat_cost(100, 100, 100, "some/openrouter-model") is None


def test_chat_cost_zero_tokens_known_model():
    assert pricing.chat_cost(0, 0, 0, "gpt-4o") == 0.0
