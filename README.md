# RAG-чатбот технічної підтримки

Чатбот-консультант на основі Retrieval-Augmented Generation (RAG).  
База знань — набір HTML/ASPX-інструкцій із локальної папки.  
Бот відповідає тільки на основі цих інструкцій і показує джерела.

## Скріншоти

| Чат | Адмін-панель |
|:---:|:---:|
| ![Чат](docs/screenshots/chat.PNG) | ![Адмін](docs/screenshots/admin.PNG) |

## Архітектура

```
HTML-папка → [Indexing pipeline] → ChromaDB
                                       ↑
React UI ⇄ FastAPI ⇄ retrieval ⇄ ─────┘
                   ⇄ LLM (OpenAI / OpenRouter)
```

| Шар | Технологія |
|---|---|
| Очищення HTML | trafilatura + BeautifulSoup |
| Чанкінг | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Векторна БД | ChromaDB (embedded, persist у файл) |
| Пошук | Гібридний: dense (ChromaDB) + BM25, злиття через RRF |
| LLM | OpenAI ↔ OpenRouter (перемикач через `.env`) |
| Бекенд | FastAPI |
| Фронтенд | React + Vite |

**Пошук — гібридний.** Dense-embeddings добре узагальнюють, але недооцінюють
точні токени (назви систем: SAP, F5 BigIP, Fortinet, FZClient). BM25 ловить саме
їх. Кандидати від обох ретриверів зливаються через Reciprocal Rank Fusion (RRF),
після чого фільтруються за порогом косинусної відстані (`RELEVANCE_THRESHOLD`,
відкаліброваним на golden set). **Контекст розмови:** бекенд приймає історію
діалогу і переписує уточнювальні запитання («а для проєктів?») у самодостатні
перед пошуком.

## Структура репозиторію

```
/
├── backend/
│   ├── app/
│   │   ├── config.py      # Pydantic Settings (читає .env)
│   │   ├── cleaner.py     # HTML/ASPX → чистий текст
│   │   ├── indexing.py    # chunking + embeddings + ChromaDB (атомарний reindex)
│   │   ├── retrieval.py   # гібридний пошук: dense + BM25, RRF-злиття
│   │   ├── query.py       # retrieval + query-rewrite + LLM
│   │   └── main.py        # FastAPI: /chat, /reindex, /status, /logs
│   ├── index.py           # CLI: індексування
│   ├── diagnose.py        # CLI: перевірка якості (Чекпоінти А та Б)
│   ├── run_eval.py        # CLI: регресійний eval (golden set)
│   ├── tools/
│   │   └── calibrate_threshold.py  # калібрування RELEVANCE_THRESHOLD
│   ├── eval/              # golden.json, detectors, LLM-суддя, baseline
│   ├── tests/             # офлайн pytest-набір (без мережі/LLM)
│   ├── data/
│   │   └── instructions/  # ← кладіть HTML-файли сюди
│   ├── requirements.txt
│   └── requirements-dev.txt  # pytest (для tests/)
├── frontend/              # React + Vite
├── .env.example
├── .gitignore
└── README.md
```

## Швидкий старт

### 1. Налаштування бекенду

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Скопіюйте `.env.example` → `backend/.env` і заповніть ключі:

```bash
cp ../.env.example .env
# відредагуйте .env: вставте OPENAI_API_KEY, за потреби інші змінні
```

### 2. Додайте HTML-файли

Помістіть файли інструкцій у `backend/data/instructions/`  
(підтримуються `.html`, `.htm`, `.aspx`).

### 3. Індексування

```bash
# з директорії backend/
python index.py           # інкрементальне оновлення
python index.py --force   # повна переіндексація
```

### 4. Діагностика якості (необов'язково)

```bash
python diagnose.py --all         # всі чекпоінти
python diagnose.py --clean       # Чекпоінт А: якість очищення
python diagnose.py --chunks      # Чекпоінт А: статистика чанків
python diagnose.py --retrieval   # Чекпоінт Б: якість пошуку
```

### 5. Запуск бекенду

```bash
# з директорії backend/
uvicorn app.main:app --reload
# API доступне на http://localhost:8000
```

### 6. Запуск фронтенду

```bash
cd frontend
npm install
npm run dev
# UI доступне на http://localhost:5173
```

## Тести

Офлайн-набір (без мережі та LLM) покриває найкрихкіше: регекс-екстракцію
HTML/ASPX (`cleaner`), розрахунок вартості (`pricing`), детектори гейтів,
атомарність реіндексації + гард сумісності embedding-моделі, та гібридний пошук.

```bash
# з директорії backend/
pip install -r requirements-dev.txt
python -m pytest
```

Запускаються автоматично в CI (`.github/workflows/tests.yml`) на кожен push/PR.

## Калібрування порогу релевантності

`RELEVANCE_THRESHOLD` (косинусна відстань) визначає, що вважати релевантним.
Замість магічного числа його калібрують на golden set — лише через ретривал
(embeddings, без генерації відповідей, тож майже безкоштовно):

```bash
# з директорії backend/
python tools/calibrate_threshold.py
```

Інструмент проганяє кожен golden-кейс і для діапазону порогів показує factual
recall, частку відповіданих питань і частку відмов на offtopic, після чого
рекомендує найбільший поріг, що ще відсікає 100% offtopic. Звіт:
`backend/eval/results/threshold_calibration.md`.

## Регресійний eval (golden set)

Перевіряє, що при зміні моделі, промпту чи бази знань **якість і безпека не
просіли**. Запускається з CLI (не з адмінки — це інженерний quality-gate, до
того ж кожен прогін платний). Потрібен `OPENAI_API_KEY` у `backend/.env`.

```bash
# з директорії backend/
python run_eval.py                   # повний прогін, порівняння з baseline
python run_eval.py --runs 5          # більше прогонів для security/offtopic/pii
python run_eval.py --no-judge        # без LLM-судді (дешевше)
python run_eval.py --update-baseline # зафіксувати поточні метрики як еталон
```

Міряються leak rate (injection), PII, refusal rate (offtopic), retrieval
recall@k, покриття ключових фактів і faithfulness (LLM-суддя). `run_eval.py`
повертає **exit code ≠ 0**, якщо провалено gate або метрика просіла відносно
baseline — тож його можна вставити кроком у CI.

Звіт: `backend/eval/results/REPORT.md`. Деталі та опис кейсів — у
[`backend/eval/README.md`](backend/eval/README.md).

> Перевірка нової моделі: змінити `LLM_MODEL` у `.env` → `python run_eval.py`.
> Нові документи: `python index.py --force`, додати кейси в `golden.json`, прогнати.

## API

| Метод | Ендпоінт | Опис |
|---|---|---|
| `POST` | `/chat` | `{message, top_k?, history?}` → `{answer, sources, usage}` |
| `POST` | `/reindex` | Запуск індексування у фоні (лише localhost) |
| `GET` | `/status` | Статус індексування |
| `GET` | `/logs` | Останні події логу, `?limit=&level=` (лише localhost) |

Кожна відповідь містить заголовок `X-Request-ID` для кореляції з логами.

## Логування

Структуроване логування у форматі **JSON-lines** (один JSON-обʼєкт на рядок):
консоль + ротований файл `backend/logs/app.log` (5 МБ × 3). Кожен HTTP-запит
отримує `request_id` (заголовок `X-Request-ID`), яким повʼязані всі його логи.

Логуються: підсумок `/chat` (latency, токени, вартість, к-сть джерел), події
безпеки (`prompt_leak_blocked`, `pii_scrubbed`), відмови (`refusal`),
`rate_limited`, реіндексація та помилки.

Останні події видно прямо в **адмін-панелі** (розділ «Останні події», з фільтром
за рівнем) — через ендпоінт `GET /logs`, доступний **лише з localhost** (як і
`/reindex`; окремої авторизації немає, бо проєкт не виставляється назовні).

**Приватність:** тексти питань/відповідей **не** логуються; вмикається лише
прев'ю (`LOG_PROMPTS=true`) для дебагу. Налаштування в `.env`:

```env
LOG_LEVEL=INFO       # DEBUG | INFO | WARNING | ERROR
LOG_DIR=./logs
LOG_PROMPTS=false    # true → у лог /chat додається обрізане прев'ю повідомлення
```

## Конфігурація

Всі параметри задаються через `backend/.env`:

```env
EMBEDDING_PROVIDER=openai       # openai (не змінювати після індексації!)
OPENAI_API_KEY=sk-...

LLM_PROVIDER=openai             # openai | openrouter
OPENROUTER_API_KEY=             # тільки для openrouter
LLM_MODEL=gpt-4o-mini

HTML_DIR=./data/instructions
CHROMA_DIR=./data/chroma

RELEVANCE_THRESHOLD=0.70         # косинусна відстань; калібрується tools/calibrate_threshold.py
DENSE_POOL=20                    # к-сть dense-кандидатів до злиття
BM25_POOL=20                     # к-сть BM25-кандидатів до злиття
```

> **Увага:** embedding-модель прив'язана до бази (вектори різних моделей несумісні).
> Її імʼя зберігається в metadata колекції; при спробі реіндексації з іншою моделлю
> поверх наявних даних `run_indexing` **кине явну помилку** замість тихого псування
> індексу. Щоб змінити модель — видаліть `CHROMA_DIR` і переіндексуйте з нуля.
