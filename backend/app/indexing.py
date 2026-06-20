"""Indexing pipeline: clean → chunk → embed → store in ChromaDB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from .cleaner import extract_text
from .config import settings
from .pricing import EMBED_MODEL

_GLOB_PATTERNS = ("*.html", "*.htm", "*.aspx")

# ~1500 chars ≈ 350–500 tokens for Ukrainian text; overlap ~13%
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n## ", "\n", ". ", " ", ""],
)


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _chroma_client() -> Any:
    Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.CHROMA_DIR)


def get_collection(client: Any = None):
    c = client or _chroma_client()
    return c.get_or_create_collection(
        name="instructions",
        metadata={"hnsw:space": "cosine"},
    )


# ── Embedding function ────────────────────────────────────────────────────────

def _make_embed_fn():
    if settings.EMBEDDING_PROVIDER == "openai":
        oai = OpenAI(api_key=settings.OPENAI_API_KEY)

        def embed(texts: list[str]) -> tuple[list[list[float]], int]:
            """Return (embeddings, tokens_used)."""
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            tokens = resp.usage.total_tokens if resp.usage else 0
            return [item.embedding for item in resp.data], tokens

        return embed
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


# ── Hash store (incremental updates) ─────────────────────────────────────────

def _hash_store_path() -> Path:
    return Path(settings.CHROMA_DIR) / "file_hashes.json"


def _load_hashes() -> dict[str, str]:
    p = _hash_store_path()
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _save_hashes(hashes: dict[str, str]) -> None:
    _hash_store_path().write_text(json.dumps(hashes, ensure_ascii=False, indent=2), "utf-8")


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Single-file indexing ──────────────────────────────────────────────────────

def _index_one(filepath: Path, collection, embed_fn) -> dict:
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    title, text = extract_text(raw, filepath.name)
    if not title:
        title = filepath.stem

    if len(text.strip()) < 50:
        return {"file": filepath.name, "status": "skipped", "reason": "no_content"}

    chunks = _SPLITTER.split_text(text)
    if not chunks:
        return {"file": filepath.name, "status": "skipped", "reason": "no_chunks"}

    # Delete previous chunks for this file
    try:
        existing = collection.get(where={"source_file": filepath.name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    embeddings, embed_tokens = embed_fn(chunks)

    ids, docs, metas = [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        ids.append(f"{filepath.stem}_{i}_{chunk_hash}")
        docs.append(chunk)

        # extract first heading as section label
        section = ""
        for line in chunk.split("\n"):
            if line.startswith("## "):
                section = line[3:].strip()
                break

        metas.append({
            "source_file": filepath.name,
            "title": title,
            "section": section,
            "content_hash": chunk_hash,
            "chunk_index": i,
        })

    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    return {
        "file": filepath.name,
        "status": "indexed",
        "title": title,
        "chunks": len(chunks),
        "text_length": len(text),
        "embedding_tokens": embed_tokens,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_indexing(html_dir: str | None = None, force: bool = False) -> dict:
    """Index (or incrementally update) all HTML/ASPX files in html_dir."""
    html_path = Path(html_dir or settings.HTML_DIR)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_path}")

    files: list[Path] = []
    for pat in _GLOB_PATTERNS:
        files.extend(html_path.glob(pat))
        files.extend(html_path.glob(f"**/{pat}"))
    files = list({f.resolve() for f in files})  # deduplicate

    client = _chroma_client()
    collection = get_collection(client)
    embed_fn = _make_embed_fn()

    stored_hashes = _load_hashes()
    new_hashes: dict[str, str] = {}
    results: list[dict] = []

    for fp in files:
        key = str(fp)
        fhash = _file_md5(fp)

        if not force and stored_hashes.get(key) == fhash:
            results.append({"file": fp.name, "status": "unchanged"})
            new_hashes[key] = fhash
            continue

        result = _index_one(fp, collection, embed_fn)
        results.append(result)
        if result["status"] == "indexed":
            new_hashes[key] = fhash
        else:
            # keep old hash so skipped files aren't retried needlessly
            if key in stored_hashes:
                new_hashes[key] = stored_hashes[key]

    # Prune hashes for files no longer on disk
    current_keys = {str(f) for f in files}
    new_hashes = {k: v for k, v in new_hashes.items() if k in current_keys}
    _save_hashes(new_hashes)

    return {
        "files_processed": len(results),
        "results": results,
        "total_chunks_in_db": collection.count(),
        "embedding_tokens": sum(r.get("embedding_tokens", 0) for r in results),
    }
