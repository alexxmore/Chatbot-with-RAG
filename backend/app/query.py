"""RAG query pipeline: embed → retrieve → generate."""
from __future__ import annotations

from openai import OpenAI

from .config import settings
from .indexing import _make_embed_fn, get_collection, _chroma_client

_SYSTEM_PROMPT = """Ти — консультант технічної підтримки. Відповідай на основі наданого контексту з бази знань.

Правила:
- Відповідай тією ж мовою, якою написане запитання.
- Якщо в контексті є часткова або непряма інформація — використай її, але чітко зазнач що інформація неповна.
- Якщо контекст взагалі не стосується запитання — кажи: «Я не маю інформації з цього питання в базі знань.»
- Не вигадуй факти, назви систем, конкретні кроки яких немає в контексті.
- Якщо є кілька кроків — нумеруй їх.
- Будь чітким і конкретним."""

_RELEVANCE_THRESHOLD = 0.75  # cosine distance; lower = more similar


def _llm_client() -> OpenAI:
    if settings.LLM_PROVIDER == "openai":
        return OpenAI(api_key=settings.OPENAI_API_KEY)
    if settings.LLM_PROVIDER == "openrouter":
        return OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def query(user_message: str, top_k: int = 6) -> dict:
    """Return {answer: str, sources: list[dict]}."""
    embed_fn = _make_embed_fn()
    collection = get_collection(_chroma_client())

    if collection.count() == 0:
        return {
            "answer": "База знань порожня. Спочатку виконайте індексування.",
            "sources": [],
        }

    q_emb = embed_fn([user_message])[0]
    n = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # filter irrelevant chunks
    relevant = [
        (doc, meta, dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if dist < _RELEVANCE_THRESHOLD
    ]

    if not relevant:
        return {
            "answer": "Я не маю інформації з цього питання в базі знань.",
            "sources": [],
        }

    # build context block
    context_parts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()

    for doc, meta, dist in relevant:
        context_parts.append(f"[{meta['title']}]\n{doc}")

        src_key = meta["source_file"]
        if src_key not in seen:
            seen.add(src_key)
            sources.append(
                {
                    "file": meta["source_file"],
                    "title": meta["title"],
                    "section": meta.get("section", ""),
                    "relevance": round(1.0 - dist, 3),
                }
            )

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"Контекст:\n{context}\n\nЗапитання: {user_message}"

    llm = _llm_client()
    completion = llm.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return {
        "answer": completion.choices[0].message.content,
        "sources": sources,
    }
