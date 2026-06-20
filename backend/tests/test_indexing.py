"""Offline tests for indexing atomicity and the embedding-model guard.

Uses an ephemeral in-memory Chroma collection and a deterministic stub embedder,
so nothing here touches OpenAI or the network.
"""
import hashlib

import chromadb
import pytest

from app import indexing
from app.indexing import (
    _index_one,
    collection_embedding_model,
    ensure_embedding_model_compatible,
)
from app.pricing import EMBED_MODEL

_DIM = 8


def _stub_embed(texts):
    """Deterministic 8-dim vectors; returns (embeddings, token_count)."""
    vecs = []
    for t in texts:
        digest = hashlib.md5(t.encode("utf-8")).digest()
        vecs.append([b / 255.0 for b in digest[:_DIM]])
    return vecs, len(texts)


def _boom(_texts):
    raise RuntimeError("embedding API is down")


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name="test_coll", metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL}
    )


def _write_html(tmp_path, name, paragraph):
    p = tmp_path / name
    p.write_text(
        f"<html><head><title>Док</title></head><body><article><p>{paragraph}</p>"
        f"</article></body></html>",
        encoding="utf-8",
    )
    return p


# ── Atomicity ────────────────────────────────────────────────────────────────

def test_index_one_adds_chunks(tmp_path, collection):
    fp = _write_html(tmp_path, "a.html", "Перший абзац документа з достатньою довжиною тексту для індексації.")
    res = _index_one(fp, collection, _stub_embed)
    assert res["status"] == "indexed"
    assert collection.count() >= 1


def test_reindex_prunes_stale_chunks(tmp_path, collection):
    # A long document → several chunks.
    long_para = ("Розділ один. " + "Детальний опис кроку. " * 200)
    fp = _write_html(tmp_path, "a.html", long_para)
    _index_one(fp, collection, _stub_embed)
    first_count = collection.count()
    assert first_count > 1

    # Shrink the document → fewer chunks; stale chunks must be pruned, not orphaned.
    fp.write_text(
        "<html><head><title>Док</title></head><body><article><p>"
        "Короткий замінений зміст документа достатньої довжини для індексації.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )
    _index_one(fp, collection, _stub_embed)
    assert collection.count() < first_count
    # No orphans from the old version remain.
    remaining = collection.get(where={"source_file": "a.html"})
    assert len(remaining["ids"]) == collection.count()


def test_failed_embed_does_not_lose_existing_chunks(tmp_path, collection):
    fp = _write_html(tmp_path, "a.html", "Стабільний вміст документа достатньої довжини для індексації тексту.")
    _index_one(fp, collection, _stub_embed)
    before = collection.count()
    assert before >= 1

    # A reindex whose embedding call fails must leave the old chunks intact.
    fp.write_text(
        "<html><head><title>Док</title></head><body><article><p>"
        "Новий вміст який ніколи не запишеться бо embedding впаде під час обробки.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _index_one(fp, collection, _boom)
    assert collection.count() == before


# ── Embedding-model guard ────────────────────────────────────────────────────

def test_matching_model_is_noop(collection):
    ensure_embedding_model_compatible(collection, strict=True)  # must not raise
    assert collection_embedding_model(collection) == EMBED_MODEL


def test_legacy_collection_gets_stamped():
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(name="legacy", metadata={"hnsw:space": "cosine"})
    assert collection_embedding_model(coll) is None
    ensure_embedding_model_compatible(coll, strict=True)
    assert collection_embedding_model(coll) == EMBED_MODEL


def test_mismatched_model_raises_when_strict():
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="mismatch", metadata={"hnsw:space": "cosine", "embedding_model": "other-model"}
    )
    with pytest.raises(RuntimeError, match="model mismatch"):
        ensure_embedding_model_compatible(coll, strict=True)


def test_mismatched_model_warns_when_not_strict(caplog):
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        name="mismatch2", metadata={"hnsw:space": "cosine", "embedding_model": "other-model"}
    )
    import logging

    with caplog.at_level(logging.WARNING):
        ensure_embedding_model_compatible(coll, strict=False)  # must not raise
    assert any("model mismatch" in r.message.lower() for r in caplog.records)
