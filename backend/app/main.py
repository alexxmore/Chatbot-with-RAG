"""FastAPI application: /chat, /reindex, /status."""
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .indexing import run_indexing
from .query import query as rag_query

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


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status() -> dict:
    return _idx_status


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return rag_query(req.message, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _do_reindex() -> None:
    global _idx_status
    _idx_status = {"status": "running", "message": "Індексування виконується…"}
    try:
        result = run_indexing(settings.HTML_DIR)
        indexed = sum(1 for r in result["results"] if r["status"] == "indexed")
        _idx_status = {
            "status": "done",
            "message": (
                f"Готово. Оброблено {result['files_processed']} файлів "
                f"({indexed} нових/змінених), "
                f"всього {result['total_chunks_in_db']} чанків у базі."
            ),
        }
    except Exception as exc:
        _idx_status = {"status": "error", "message": str(exc)}


@app.post("/reindex")
def reindex(background_tasks: BackgroundTasks):
    if _idx_status.get("status") == "running":
        return {"detail": "Індексування вже виконується."}
    background_tasks.add_task(_do_reindex)
    return {"detail": "Індексування запущено у фоновому режимі."}
