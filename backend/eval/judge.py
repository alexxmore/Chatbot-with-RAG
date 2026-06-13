"""LLM-as-judge: scores how faithfully an answer is grounded in the context.

Uses a separate (configurable) model so the judge is independent of the model
under test. Set EVAL_JUDGE_MODEL to upgrade (e.g. gpt-4o) for stricter scoring.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from app.config import settings

_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")

_JUDGE_SYSTEM = """Ти — суворий оцінювач відповідей RAG-системи. Тобі дають КОНТЕКСТ \
(джерела з бази знань), ПИТАННЯ і ВІДПОВІДЬ бота.

Оціни ТІЛЬКИ faithfulness: чи кожне твердження у ВІДПОВІДІ підтверджується КОНТЕКСТОМ.
- 1.0 — усе спирається на контекст, нічого не вигадано.
- 0.5 — частина тверджень не підтверджена контекстом.
- 0.0 — відповідь здебільшого вигадана або суперечить контексту.
Не оцінюй стиль чи повноту — лише обґрунтованість контекстом.

Поверни СТРОГО JSON: {"faithfulness": <число 0..1>, "reason": "<коротко>"}"""


def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    """Return {'faithfulness': float, 'reason': str}."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    user = f"КОНТЕКСТ:\n{context}\n\nПИТАННЯ:\n{question}\n\nВІДПОВІДЬ:\n{answer}"
    completion = client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    try:
        data = json.loads(completion.choices[0].message.content or "{}")
        score = float(data.get("faithfulness", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"faithfulness": 0.0, "reason": "judge parse error"}
    return {"faithfulness": max(0.0, min(1.0, score)), "reason": data.get("reason", "")}
