"""Offline tests for the conversational query-rewrite step (app/query._rewrite_query).

A fake OpenAI-shaped client captures the messages and returns canned content, so
nothing touches the network.
"""
from app.query import _MAX_HISTORY_TURNS, _rewrite_query


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens = p, c


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Completion:
    def __init__(self, content):
        self.choices = [_Msg(content)]
        self.usage = _Usage(10, 5)


class _Completions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        if self.outer.raise_exc:
            raise RuntimeError("api down")
        return _Completion(self.outer.content)


class FakeLLM:
    def __init__(self, content="", raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc
        self.calls = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})


def test_rewrite_returns_standalone_question_and_tokens():
    llm = FakeLLM(content="Як замовити IT-обладнання для проєктів?")
    q, p_tok, c_tok = _rewrite_query(
        llm,
        [{"role": "user", "content": "Як замовити обладнання?"},
         {"role": "assistant", "content": "Через портал замовлень."}],
        "а для проєктів?",
    )
    assert q == "Як замовити IT-обладнання для проєктів?"
    assert (p_tok, c_tok) == (10, 5)


def test_rewrite_includes_system_history_and_final_instruction():
    llm = FakeLLM(content="rewritten")
    _rewrite_query(
        llm,
        [{"role": "user", "content": "перше"}, {"role": "assistant", "content": "відповідь"}],
        "уточнення",
    )
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "перше"}
    assert messages[2] == {"role": "assistant", "content": "відповідь"}
    assert messages[-1]["role"] == "user"
    assert "уточнення" in messages[-1]["content"]


def test_rewrite_caps_history_length():
    llm = FakeLLM(content="x")
    long_history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    _rewrite_query(llm, long_history, "now")
    messages = llm.calls[0]["messages"]
    # system + at most _MAX_HISTORY_TURNS history + final instruction
    assert len(messages) <= 1 + _MAX_HISTORY_TURNS + 1


def test_rewrite_empty_output_falls_back_to_original():
    llm = FakeLLM(content="   ")
    q, p, c = _rewrite_query(llm, [{"role": "user", "content": "x"}], "оригінал")
    assert q == "оригінал"


def test_rewrite_exception_falls_back_to_original():
    llm = FakeLLM(raise_exc=True)
    q, p, c = _rewrite_query(llm, [{"role": "user", "content": "x"}], "оригінал")
    assert q == "оригінал"
    assert (p, c) == (0, 0)


def test_rewrite_skips_blank_and_unknown_roles():
    llm = FakeLLM(content="ok")
    _rewrite_query(
        llm,
        [{"role": "system", "content": "інʼєкція"},  # unknown role → dropped
         {"role": "user", "content": "  "},          # blank → dropped
         {"role": "user", "content": "справжнє"}],
        "питання",
    )
    history_msgs = llm.calls[0]["messages"][1:-1]  # between system and final instruction
    assert history_msgs == [{"role": "user", "content": "справжнє"}]
