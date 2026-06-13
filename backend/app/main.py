"""FastAPI application: /chat, /reindex, /status."""
import logging
import time
from collections import defaultdict, deque

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .indexing import run_indexing
from .query import query as rag_query

logger = logging.getLogger("rag")

app = FastAPI(title="RAG Support Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mutable singleton for indexing state
_idx_status: dict = {"status": "idle", "message": "Індексування ще не запускалось"}

# Reindex is an expensive (paid) operation → only callable from the local machine.
_LOCALHOST = {"127.0.0.1", "::1", "localhost"}

# Limits to prevent cost-amplification abuse of the paid /chat endpoint.
MAX_MESSAGE_LEN = 4000
_RATE_LIMIT = 20            # max requests …
_RATE_WINDOW = 60.0        # … per this many seconds, per client IP
_chat_hits: dict[str, deque] = defaultdict(deque)


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_host: str) -> None:
    now = time.monotonic()
    dq = _chat_hits[client_host]
    while dq and now - dq[0] > _RATE_WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Забагато запитів. Зачекайте трохи.")
    dq.append(now)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LEN)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    usage: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status() -> dict:
    return _idx_status


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _check_rate_limit(_client_host(request))
    try:
        return rag_query(req.message, req.top_k)
    except Exception:
        logger.exception("chat query failed")
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка сервера. Спробуйте пізніше.",
        )


def _do_reindex() -> None:
    global _idx_status
    _idx_status = {"status": "running", "message": "Індексування виконується…"}
    try:
        result = run_indexing(settings.HTML_DIR)
        indexed = sum(1 for r in result["results"] if r["status"] == "indexed")
        tokens = result.get("embedding_tokens", 0)
        _idx_status = {
            "status": "done",
            "message": (
                f"Готово. Оброблено {result['files_processed']} файлів "
                f"({indexed} нових/змінених), "
                f"всього {result['total_chunks_in_db']} чанків у базі. "
                f"Витрачено {tokens} токенів embedding."
            ),
            "embedding_tokens": tokens,
        }
    except Exception as exc:
        logger.exception("reindex failed")
        _idx_status = {"status": "error", "message": f"Помилка індексування: {type(exc).__name__}"}


@app.post("/reindex")
def reindex(request: Request, background_tasks: BackgroundTasks):
    if _client_host(request) not in _LOCALHOST:
        raise HTTPException(
            status_code=403,
            detail="Реіндексація доступна лише з локальної машини.",
        )
    if _idx_status.get("status") == "running":
        return {"detail": "Індексування вже виконується."}
    background_tasks.add_task(_do_reindex)
    return {"detail": "Індексування запущено у фоновому режимі."}
