#!/usr/bin/env python
"""Diagnostic script — run after indexing to verify quality.

Checkpoint A: cleaning + chunking quality
Checkpoint B: retrieval quality

Usage:
    python diagnose.py --all
    python diagnose.py --clean
    python diagnose.py --chunks
    python diagnose.py --retrieval
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.cleaner import extract_text
from app.config import settings
from app.indexing import (
    _GLOB_PATTERNS,
    _SPLITTER,
    _chroma_client,
    _make_embed_fn,
    get_collection,
)

import sys as _sys
_USE_COLOR = _sys.platform != "win32" or _sys.stdout.encoding.lower() in ("utf-8", "utf-16")
RED    = "\033[31m" if _USE_COLOR else ""
GREEN  = "\033[32m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""


def _collect_files() -> list[Path]:
    html_path = Path(settings.HTML_DIR)
    files: list[Path] = []
    for pat in _GLOB_PATTERNS:
        files.extend(html_path.glob(pat))
        files.extend(html_path.glob(f"**/{pat}"))
    return list({f.resolve() for f in files})


# ── Checkpoint A: cleaning ────────────────────────────────────────────────────

def check_cleaning() -> None:
    print("\n" + "=" * 64)
    print("CHECKPOINT A — HTML CLEANING")
    print("=" * 64)

    files = _collect_files()
    if not files:
        print(f"{RED}No files found in {settings.HTML_DIR}{RESET}")
        return

    print(f"\nFound {len(files)} file(s)\n")

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="replace")
        title, text = extract_text(raw, fp.name)

        orig_len = len(raw)
        clean_len = len(text)
        ratio = clean_len / orig_len if orig_len else 0

        flag = ""
        if ratio < 0.01:
            flag = f"  {RED}⚠ TOO SHORT (<1%){RESET}"
        elif ratio > 0.60:
            flag = f"  {YELLOW}⚠ SUSPICIOUSLY LONG (>60%){RESET}"
        else:
            flag = f"  {GREEN}✓{RESET}"

        print(f"{'─'*64}")
        print(f"File   : {fp.name}")
        print(f"Title  : {title}")
        print(f"Lengths: {orig_len} → {clean_len} chars  ({ratio:.1%}){flag}")

        headings = [l.strip() for l in text.split("\n") if l.strip().startswith("## ")]
        if headings:
            print(f"Headings ({len(headings)}):")
            for h in headings[:8]:
                print(f"  {h}")

        preview = text[:400].replace("\n", " ")
        print(f"Preview: {preview}")
        print()


# ── Checkpoint A: chunks ──────────────────────────────────────────────────────

def check_chunks(n_samples: int = 10) -> None:
    print("\n" + "=" * 64)
    print("CHECKPOINT A — CHUNK QUALITY")
    print("=" * 64)

    files = _collect_files()
    if not files:
        print(f"{RED}No files found{RESET}")
        return

    all_chunks: list[str] = []
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="replace")
        _, text = extract_text(raw, fp.name)
        if text:
            all_chunks.extend(_SPLITTER.split_text(text))

    if not all_chunks:
        print(f"{RED}No chunks produced!{RESET}")
        return

    lengths = [len(c) for c in all_chunks]
    print(f"\nTotal chunks : {len(all_chunks)}")
    print(f"Min / Max    : {min(lengths)} / {max(lengths)} chars")
    print(f"Average      : {sum(lengths)//len(lengths)} chars")

    buckets = {"< 200": 0, "200–600": 0, "600–1500": 0, "1500–3000": 0, "> 3000": 0}
    for l in lengths:
        if l < 200:
            buckets["< 200"] += 1
        elif l < 600:
            buckets["200–600"] += 1
        elif l < 1500:
            buckets["600–1500"] += 1
        elif l < 3000:
            buckets["1500–3000"] += 1
        else:
            buckets["> 3000"] += 1

    print("\nDistribution:")
    total = len(all_chunks)
    for label, count in buckets.items():
        bar = "█" * int(count / total * 40)
        warn = f" {YELLOW}⚠{RESET}" if label in ("< 200", "> 3000") and count > 0 else ""
        print(f"  {label:12s}: {count:4d}  {bar}{warn}")

    sample = random.sample(all_chunks, min(n_samples, len(all_chunks)))
    print(f"\n{n_samples} RANDOM CHUNKS:")
    for i, chunk in enumerate(sample, 1):
        print(f"\n── Chunk {i} ({len(chunk)} chars) ──")
        print(chunk[:500])
        if len(chunk) > 500:
            print("…")


# ── Checkpoint B: retrieval ───────────────────────────────────────────────────

_DEFAULT_QUERIES = [
    "Як зробити звірку по АЗС?",
    "Що робити якщо документ не проводиться?",
    "Як сторнувати документ?",
    "Де знайти шаблон звернення?",
    "Що робити якщо документ відсутній в СО?",
]


def check_retrieval(queries: list[str] | None = None, top_k: int = 5) -> None:
    print("\n" + "=" * 64)
    print("CHECKPOINT B — RETRIEVAL QUALITY")
    print("=" * 64)

    collection = get_collection(_chroma_client())
    total = collection.count()
    print(f"\nChunks in DB: {total}")

    if total == 0:
        print(f"{RED}Database is empty — run indexing first!{RESET}")
        return

    embed_fn = _make_embed_fn()
    test_queries = queries or _DEFAULT_QUERIES

    for q in test_queries:
        print(f"\n{'─'*64}")
        print(f"Query: {q}")
        emb = embed_fn([q])[0]
        res = collection.query(
            query_embeddings=[emb],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
        ):
            relevance = 1.0 - dist
            color = GREEN if relevance > 0.35 else YELLOW if relevance > 0.25 else RED
            snippet = doc[:120].replace("\n", " ")
            print(
                f"  [{i}] {color}{meta['title']}{RESET} "
                f"| section: {meta['section'][:40]} "
                f"| relevance: {relevance:.3f}"
            )
            print(f"       {snippet}…")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic checks for the RAG pipeline")
    parser.add_argument("--clean", action="store_true", help="Check HTML cleaning quality")
    parser.add_argument("--chunks", action="store_true", help="Check chunk statistics")
    parser.add_argument("--retrieval", action="store_true", help="Check retrieval quality")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()

    if not any([args.clean, args.chunks, args.retrieval, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.all or args.clean:
        check_cleaning()
    if args.all or args.chunks:
        check_chunks()
    if args.all or args.retrieval:
        check_retrieval()


if __name__ == "__main__":
    main()
