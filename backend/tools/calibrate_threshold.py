#!/usr/bin/env python
"""Calibrate RELEVANCE_THRESHOLD from the golden set — retrieval only, no LLM generation.

The threshold (a cosine distance) only affects the *relevance gate* in query.py:
a chunk is used when its distance is below it. So we can calibrate it without
generating any answers — we just embed each golden question once, run hybrid
retrieval once, and then sweep candidate thresholds over the cached distances.
Cost is ~30 short embeddings (< $0.001); no chat/judge calls.

For each threshold τ we report:
  - factual.recall        — expected document retrieved within top_k (↑ better)
  - factual.answered      — factual question got ≥1 relevant chunk, i.e. NOT
                            wrongly refused (↑ better)
  - offtopic.refusal      — off-topic question correctly refused (↑ better)

The recommendation is the LARGEST τ that still keeps offtopic.refusal == 1.0
(maximising recall/answered without letting off-topic queries through).

Usage:
    python tools/calibrate_threshold.py
    python tools/calibrate_threshold.py --top-k 5 --min 0.3 --max 1.0 --step 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from app.config import settings  # noqa: E402
from app.indexing import _chroma_client, _make_embed_fn, get_collection  # noqa: E402
from app.retrieval import hybrid_retrieve  # noqa: E402

_GOLDEN = _BACKEND / "eval" / "golden.json"
_REPORT = _BACKEND / "eval" / "results" / "threshold_calibration.md"


def _frange(lo: float, hi: float, step: float) -> list[float]:
    out, x = [], lo
    while x <= hi + 1e-9:
        out.append(round(x, 3))
        x += step
    return out


def _relevant_sources(candidates: list[dict], tau: float, top_k: int) -> list[str]:
    """Source files of the chunks that pass the gate at τ (query.py semantics)."""
    rel = [c for c in candidates if c["distance"] < tau][:top_k]
    seen, out = set(), []
    for c in rel:
        sf = c["meta"]["source_file"]
        if sf not in seen:
            seen.add(sf)
            out.append(sf)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate RELEVANCE_THRESHOLD")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min", type=float, default=0.30)
    ap.add_argument("--max", type=float, default=1.00)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    golden = json.loads(_GOLDEN.read_text("utf-8"))
    cases = golden["cases"]

    collection = get_collection(_chroma_client())
    if collection.count() == 0:
        sys.exit("Index is empty — run `python index.py` first.")
    embed = _make_embed_fn()

    # Retrieve once per case; cache the candidate distances for the whole sweep.
    print(f"Embedding + retrieving {len(cases)} golden cases once…")
    cached: list[tuple[dict, list[dict]]] = []
    for c in cases:
        emb, _ = embed([c["question"]])
        cands = hybrid_retrieve(
            collection, c["question"], emb[0],
            dense_pool=settings.DENSE_POOL, bm25_pool=settings.BM25_POOL,
        )
        cached.append((c, cands))

    factual = [(c, cands) for c, cands in cached if c["type"] == "factual"]
    offtopic = [(c, cands) for c, cands in cached if c["type"] == "offtopic"]

    rows = []
    for tau in _frange(args.min, args.max, args.step):
        recalls, answered, refusals = [], [], []
        for c, cands in factual:
            srcs = _relevant_sources(cands, tau, args.top_k)
            recalls.append(1.0 if c["expected_source"] in srcs else 0.0)
            answered.append(1.0 if srcs else 0.0)
        for c, cands in offtopic:
            srcs = _relevant_sources(cands, tau, args.top_k)
            refusals.append(1.0 if not srcs else 0.0)
        rows.append({
            "tau": tau,
            "recall": _mean(recalls),
            "answered": _mean(answered),
            "offtopic_refusal": _mean(refusals),
        })

    # Recommend the largest τ that still refuses every off-topic question.
    safe = [r for r in rows if r["offtopic_refusal"] >= 1.0]
    best = max(safe, key=lambda r: (r["recall"], r["answered"], r["tau"])) if safe else None

    header = f"{'τ':>6} | {'recall':>7} | {'answered':>9} | {'offtopic_refuse':>16}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        mark = "  <-- current" if abs(r["tau"] - settings.RELEVANCE_THRESHOLD) < 1e-9 else ""
        if best and abs(r["tau"] - best["tau"]) < 1e-9:
            mark += "  <== recommended"
        lines.append(
            f"{r['tau']:>6.2f} | {r['recall']:>7.2f} | {r['answered']:>9.2f} | "
            f"{r['offtopic_refusal']:>16.2f}{mark}"
        )
    table = "\n".join(lines)
    print("\n" + table)

    rec_text = (
        f"\nRecommended RELEVANCE_THRESHOLD = {best['tau']:.2f} "
        f"(recall {best['recall']:.2f}, answered {best['answered']:.2f}, "
        f"offtopic refusal {best['offtopic_refusal']:.2f}).\n"
        f"Current value = {settings.RELEVANCE_THRESHOLD:.2f}. "
        f"Set it in .env as RELEVANCE_THRESHOLD if different."
        if best else
        "\nNo threshold keeps all off-topic queries refused — inspect the data."
    )
    print(rec_text)

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(
        f"# Relevance-threshold calibration\n\n"
        f"- Cases: {len(cases)} (factual {len(factual)}, offtopic {len(offtopic)})\n"
        f"- top_k={args.top_k}, dense_pool={settings.DENSE_POOL}, bm25_pool={settings.BM25_POOL}\n\n"
        f"```\n{table}\n```\n{rec_text}\n",
        "utf-8",
    )
    print(f"\nReport written to {_REPORT}")


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


if __name__ == "__main__":
    main()
