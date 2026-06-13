#!/usr/bin/env python
"""CLI: index HTML/ASPX files into ChromaDB.

Usage:
    python index.py                  # incremental update
    python index.py --force          # reindex everything
    python index.py --dir path/to/html
"""
import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1251 → Ukrainian apostrophe (ʼ) etc. crash on print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow running from the backend/ directory
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.indexing import run_indexing


def main() -> None:
    parser = argparse.ArgumentParser(description="Index HTML/ASPX files into ChromaDB")
    parser.add_argument("--dir", default=settings.HTML_DIR, help="Directory with HTML files")
    parser.add_argument("--force", action="store_true", help="Force full reindex")
    args = parser.parse_args()

    print(f"HTML directory : {args.dir}")
    print(f"Chroma directory: {settings.CHROMA_DIR}")
    print(f"Embedding model : {settings.EMBEDDING_PROVIDER}")
    print()

    result = run_indexing(args.dir, force=args.force)

    print(f"Files processed    : {result['files_processed']}")
    print(f"Total chunks in DB : {result['total_chunks_in_db']}")
    print(f"Embedding tokens   : {result.get('embedding_tokens', 0)}")
    print()

    for r in result["results"]:
        status = r["status"]
        fname = r["file"]
        if status == "indexed":
            print(f"  [+] {fname}  ({r['chunks']} chunks | {r['text_length']} chars | \"{r['title'][:60]}\")")
        elif status == "unchanged":
            print(f"  [=] {fname}  (unchanged)")
        else:
            print(f"  [!] {fname}  {status}: {r.get('reason', '')}")


if __name__ == "__main__":
    main()
