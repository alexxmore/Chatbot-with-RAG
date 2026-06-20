"""Hybrid retrieval: dense (ChromaDB) + lexical (BM25), fused with RRF.

Why hybrid: dense embeddings generalise well but under-weight exact tokens —
system names, acronyms and codes (SAP, F5 BigIP, Fortinet, FZClient, MME). BM25
catches those. Each retriever proposes candidates; Reciprocal Rank Fusion (RRF)
merges them by rank, so no score normalisation between cosine distance and BM25
is needed.

Relevance is still gated by the *dense cosine distance* in query.py: BM25 can
surface a keyword-strong chunk that dense ranked low (or missed), but we recover
that chunk's true cosine distance and let the same threshold decide. So off-topic
queries — where every distance is large — still refuse, while on-topic keyword
queries get better recall and ordering.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

# Tokens for BM25: Cyrillic/Latin/digit runs of length >= 2, lowercased.
_TOKEN_RE = re.compile(r"[A-Za-zА-ЯҐЄІЇа-яґєії0-9]{2,}")

# RRF dampening constant; 60 is the value from the original RRF paper.
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ── BM25 index cache ─────────────────────────────────────────────────────────
# Rebuilt lazily when the collection size changes, or on explicit invalidation
# after a reindex (edits that keep the chunk count identical).

_cache: dict[str, Any] = {"count": None, "bm25": None, "ids": None}


def invalidate_bm25_cache() -> None:
    _cache.update(count=None, bm25=None, ids=None)


def _get_bm25(collection) -> tuple[BM25Okapi | None, list[str]]:
    count = collection.count()
    if _cache["bm25"] is None or _cache["count"] != count:
        data = collection.get(include=["documents"])
        ids = data["ids"]
        corpus = [_tokenize(d) for d in data["documents"]]
        _cache.update(count=count, ids=ids, bm25=BM25Okapi(corpus) if corpus else None)
    return _cache["bm25"], _cache["ids"]


def _cosine_distance(a, b) -> float:
    """Cosine distance (1 - cosine similarity), matching ChromaDB's 'cosine' space."""
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv)) or 1.0
    return 1.0 - float(np.dot(av, bv) / denom)


def hybrid_retrieve(
    collection,
    query_text: str,
    query_embedding: list[float],
    *,
    dense_pool: int = 20,
    bm25_pool: int = 20,
) -> list[dict]:
    """Return fused candidates, best first.

    Each candidate: {id, doc, meta, distance, bm25_score, rrf}. `distance` is the
    cosine distance to the query for every candidate (recovered for BM25-only ones),
    so the caller can apply a single relevance threshold uniformly.
    """
    total = collection.count()
    if total == 0:
        return []

    cand: dict[str, dict] = {}

    # Dense candidates (already carry distances).
    dres = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(dense_pool, total),
        include=["documents", "metadatas", "distances"],
    )
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(dres["ids"][0], dres["documents"][0], dres["metadatas"][0], dres["distances"][0])
    ):
        cand[cid] = {"id": cid, "doc": doc, "meta": meta, "distance": dist,
                     "dense_rank": rank, "bm25_rank": None, "bm25_score": 0.0}

    # Lexical candidates.
    bm25, all_ids = _get_bm25(collection)
    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(query_text))
        # Only positive scores mean real token overlap.
        ranked_idx = [j for j in np.argsort(scores)[::-1] if scores[j] > 0][:bm25_pool]
        for rank, j in enumerate(ranked_idx):
            cid = all_ids[j]
            if cid in cand:
                cand[cid]["bm25_rank"] = rank
                cand[cid]["bm25_score"] = float(scores[j])
            else:
                cand[cid] = {"id": cid, "dense_rank": None, "bm25_rank": rank,
                             "bm25_score": float(scores[j])}

        # Recover BM25-only candidates' docs/metas and true cosine distance.
        missing = [cid for cid, c in cand.items() if "doc" not in c]
        if missing:
            got = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
            for cid, doc, meta, emb in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            ):
                c = cand[cid]
                c["doc"], c["meta"] = doc, meta
                c["distance"] = _cosine_distance(query_embedding, emb)

    # Reciprocal Rank Fusion.
    for c in cand.values():
        score = 0.0
        if c["dense_rank"] is not None:
            score += 1.0 / (_RRF_K + c["dense_rank"])
        if c["bm25_rank"] is not None:
            score += 1.0 / (_RRF_K + c["bm25_rank"])
        c["rrf"] = score

    return sorted(cand.values(), key=lambda c: c["rrf"], reverse=True)
