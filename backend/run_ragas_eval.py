#!/usr/bin/env python
"""Supplementary RAG-quality eval using the RAGAS framework.

A SEPARATE, optional layer on top of run_eval.py — it does NOT replace the
regression harness or its gates. It reports two reference-free metrics the
custom harness does not measure:

  * answer_relevancy   — does the answer actually address the question?
  * context_precision  — are the relevant retrieved chunks ranked near the top?

Only the `factual` golden cases are scored (injection/offtopic/pii cases have no
meaningful "relevant context"). The golden set has no full reference answers, so
we use RAGAS's reference-free metric variants.

Requires the extra deps in requirements-eval.txt:
    pip install -r requirements-eval.txt
Uses the same OPENAI_API_KEY as the app (from .env). Costs a few cents per run.

Usage:
    python run_ragas_eval.py
    python run_ragas_eval.py --top-k 5 --judge-model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1251 → UTF-8

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.query import query as rag_query

_EVAL_DIR = Path(__file__).parent / "eval"
_RESULTS_DIR = _EVAL_DIR / "results"
_CONTEXT_SEP = "\n\n---\n\n"  # must match the join separator in query.py


def _split_contexts(context: str) -> list[str]:
    """Recover the per-chunk list that query() joined into one context string."""
    if not context:
        return []
    return [c.strip() for c in context.split(_CONTEXT_SEP) if c.strip()]


def _build_samples(golden: dict, top_k: int) -> list[dict]:
    """Run every factual case through query() and shape it for RAGAS."""
    samples: list[dict] = []
    for c in golden["cases"]:
        if c["type"] != "factual":
            continue
        r = rag_query(c["question"], top_k)
        contexts = _split_contexts(r.get("context", ""))
        if not contexts:
            continue  # nothing retrieved → nothing for RAGAS to score
        samples.append(
            {
                "id": c["id"],
                "user_input": c["question"],
                "response": r["answer"],
                "retrieved_contexts": contexts,
            }
        )
    return samples


def _mean_metrics(df) -> dict:
    """Mean of whichever metric columns RAGAS produced (names vary by version)."""
    known = {
        "answer_relevancy",
        "response_relevancy",
        "context_precision",
        "llm_context_precision_without_reference",
    }
    return {c: round(float(df[c].mean()), 3) for c in df.columns if c in known}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RAGAS supplementary eval (answer_relevancy, context_precision)"
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM RAGAS uses to compute the metrics (gpt-4o for stricter scoring)",
    )
    ap.add_argument("--golden", default=str(_EVAL_DIR / "golden.json"))
    args = ap.parse_args()

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )
    except ImportError as exc:
        sys.exit(
            f"RAGAS deps missing ({exc}).\n"
            f"Install them with:  pip install -r requirements-eval.txt"
        )

    if not settings.OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set (.env) — RAGAS needs it for the judge LLM + embeddings.")

    golden = json.loads(Path(args.golden).read_text("utf-8"))
    print(f"Collecting answers from query() (top_k={args.top_k}) …")
    samples = _build_samples(golden, args.top_k)
    if not samples:
        sys.exit("No factual samples with retrieved context — is the index populated?")

    dataset = EvaluationDataset.from_list(
        [{k: v for k, v in s.items() if k != "id"} for s in samples]
    )

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=args.judge_model, temperature=0, api_key=settings.OPENAI_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
    )
    metrics = [
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]

    print(f"Scoring {len(samples)} factual cases with RAGAS (judge={args.judge_model}) …")
    result = evaluate(dataset=dataset, metrics=metrics)

    df = result.to_pandas()
    df.insert(0, "id", [s["id"] for s in samples])
    means = _mean_metrics(df)

    print("\n=== RAGAS metrics (mean over factual cases) ===")
    for k, v in means.items():
        print(f"  {k:42s} {v:.3f}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "judge_model": args.judge_model,
        "model_under_test": settings.LLM_MODEL,
        "n_cases": len(samples),
        "metrics_mean": means,
        "per_case": json.loads(df.to_json(orient="records", force_ascii=False)),
    }
    out = _RESULTS_DIR / "ragas_last_run.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
