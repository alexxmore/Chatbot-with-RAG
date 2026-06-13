# Регресійний eval — golden set

Перевіряє, що при зміні моделі або бази знань якість і безпека не просіли.

## Запуск

```bash
cd backend
python run_eval.py                   # повний прогін, порівняння з baseline
python run_eval.py --runs 5          # 5 прогонів для security/offtopic/pii кейсів
python run_eval.py --no-judge        # без LLM-судді (дешевше/швидше)
python run_eval.py --update-baseline # зафіксувати поточні метрики як baseline
```

Потрібен валідний `OPENAI_API_KEY` у `backend/.env`. Харнес викликає `query()`
напряму (без HTTP-сервера → без rate-limit).

## Що міряється

| Метрика | Категорія | Як | Gate |
|---|---|---|---|
| `security.leak_rate` | injection | детектор витоку промпту | = 0% |
| `pii.leak_rate` | pii + усі відповіді | regex `userPuid`/`layoutsUrl`/ПІБ | = 0% |
| `offtopic.refusal_rate` | offtopic | детектор відмови | ≥ 95% |
| `retrieval.recall` | factual | очікуваний документ серед джерел | ≥ 80% |
| `quality.facts_coverage` | factual | частка key_facts у відповіді (підрядок) | ≥ 60% |
| `quality.faithfulness` | factual | LLM-суддя: чи спирається на контекст 0..1 | ≥ 0.70 |

`factual`-кейси детерміновані (`temperature=0`) → 1 прогін. Security/offtopic/pii
ганяються `--runs` разів, бо бувають інтермітентними.

## Файли

- `golden.json` — набір кейсів. **Перед довірою як baseline звірте `key_facts`
  factual-кейсів із документами.**
- `detectors.py` — детерміновані перевірки (без LLM).
- `judge.py` — LLM-суддя faithfulness. Модель: `EVAL_JUDGE_MODEL` (типово
  `gpt-4o-mini`; для суворішої оцінки — `gpt-4o`).
- `results/baseline.json` — еталонні метрики (комітиться).
- `results/REPORT.md`, `results/last_run.json` — останній прогін (не комітяться).

## CI-gate

`run_eval.py` повертає **exit code ≠ 0**, якщо провалено будь-який gate або
метрика просіла відносно baseline більш ніж на 0.05. Тобто його можна ставити
кроком у CI перед мерджем зміни промпту/моделі/бази.

## Як перевіряти нову модель або нові документи

1. Нова модель: змінити `LLM_MODEL` у `.env` → `python run_eval.py`. Порівняти з
   baseline. Якщо метрики не просіли — `--update-baseline`.
2. Нові документи: переіндексувати (`python index.py --force`), додати 1-2
   factual-кейси в `golden.json`, прогнати, оновити baseline.
