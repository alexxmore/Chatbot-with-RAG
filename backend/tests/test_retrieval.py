"""Offline tests for hybrid retrieval (app/retrieval.py).

Uses an ephemeral Chroma collection with hand-placed embeddings, so dense ranking
is fully controlled and no OpenAI call happens.
"""
import uuid

import chromadb
import pytest

from app.retrieval import _cosine_distance, _tokenize, hybrid_retrieve, invalidate_bm25_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    invalidate_bm25_cache()
    yield
    invalidate_bm25_cache()


def _coll(docs_embs):
    """docs_embs: list of (id, text, embedding)."""
    client = chromadb.EphemeralClient()
    c = client.get_or_create_collection(
        name=f"ret_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    c.add(
        ids=[d[0] for d in docs_embs],
        documents=[d[1] for d in docs_embs],
        embeddings=[d[2] for d in docs_embs],
        metadatas=[{"source_file": d[0], "title": d[0], "section": ""} for d in docs_embs],
    )
    return c


def test_tokenize_handles_cyrillic_and_acronyms():
    assert _tokenize("Створення SQL Login у SAP!") == ["створення", "sql", "login", "sap"]


def test_cosine_distance_identical_is_zero():
    assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_is_one():
    assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-9)


def test_dense_only_ranks_by_distance():
    # Query vector closest to "b".
    coll = _coll([
        ("a", "альфа документ", [1.0, 0.0]),
        ("b", "бета документ", [0.9, 0.1]),
        ("c", "гама документ", [0.0, 1.0]),
    ])
    out = hybrid_retrieve(coll, "немає збігів слів", [0.9, 0.1])
    assert out[0]["id"] == "b"
    # Every candidate carries a usable cosine distance.
    assert all("distance" in c for c in out)


def test_bm25_recovers_keyword_chunk_missed_by_dense():
    # "target" is dense-far from the query but is the only exact keyword match.
    coll = _coll([
        ("near1", "загальний текст один", [1.0, 0.0, 0.0]),
        ("near2", "загальний текст два", [0.99, 0.01, 0.0]),
        ("target", "інструкція Fortinet налаштування", [0.0, 0.0, 1.0]),
    ])
    out = hybrid_retrieve(coll, "Fortinet", [1.0, 0.0, 0.0], dense_pool=2, bm25_pool=5)
    ids = [c["id"] for c in out]
    # Even though dense_pool=2 excluded it, BM25 surfaced "target"...
    assert "target" in ids
    # ...and its real cosine distance was recovered (orthogonal → ~1.0).
    target = next(c for c in out if c["id"] == "target")
    assert target["distance"] == pytest.approx(1.0, abs=1e-9)
    assert target["bm25_score"] > 0


def test_rrf_promotes_chunk_strong_in_both_signals():
    # "both" is 2nd by dense but the unique keyword match → fusion lifts it to #1.
    coll = _coll([
        ("top_dense", "загальний опис процесу роботи", [1.0, 0.0]),
        ("both", "налаштування FZClient система", [0.95, 0.05]),
        ("other", "ще один абзац тексту", [0.5, 0.5]),
    ])
    out = hybrid_retrieve(coll, "FZClient", [1.0, 0.0], dense_pool=3, bm25_pool=3)
    assert out[0]["id"] == "both"


def test_empty_collection_returns_empty():
    client = chromadb.EphemeralClient()
    c = client.get_or_create_collection(
        name=f"empty_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    assert hybrid_retrieve(c, "будь-що", [1.0, 0.0]) == []
