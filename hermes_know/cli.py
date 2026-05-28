"""
CLI entry point for hermes_know.

Usage:
    python -m hermes_know build [-k knowledge.json] [-o embeddings.pkl]
    python -m hermes_know search [-e embeddings.pkl] [-t 0.45] [-n 1] <query>
"""

import argparse
import sys
from pathlib import Path

from .core import (
    embed_texts,
    load_knowledge,
    load_embeddings,
    save_embeddings,
    search,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hermes_know",
        description="Local semantic search for project knowledge bases",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ────────────────────────────────────────────────────────
    build_parser = sub.add_parser("build", help="Generate embeddings")
    build_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    build_parser.add_argument(
        "-o", "--output", default="embeddings.pkl",
        help="Output path for embeddings (default: embeddings.pkl)",
    )

    # ── search ──────────────────────────────────────────────────────
    search_parser = sub.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("query", help="The search query")
    search_parser.add_argument(
        "-e", "--embeddings", default="embeddings.pkl",
        help="Path to embeddings file (default: embeddings.pkl)",
    )
    search_parser.add_argument(
        "-t", "--threshold", type=float, default=0.45,
        help="Minimum similarity score 0.0-1.0 (default: 0.45)",
    )
    search_parser.add_argument(
        "-n", "--top", type=int, default=1,
        help="Number of top results (default: 1)",
    )
    search_parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (for machine parsing)",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        _do_build(args.knowledge, args.output)
    elif args.command == "search":
        _do_search(args.query, args.embeddings, args.threshold, args.top, args.json)


def _do_build(knowledge_path: str, output_path: str) -> None:
    path = Path(knowledge_path)
    if not path.exists():
        print(f"Error: {knowledge_path} not found.", file=sys.stderr)
        sys.exit(1)

    entries = load_knowledge(path)
    if not entries:
        print("knowledge.json is empty. Nothing to embed.")
        return

    print(f"Found {len(entries)} entries. Generating embeddings...")
    embedded = embed_texts(entries)
    save_embeddings(embedded, output_path)


def _do_search(
    query: str,
    embeddings_path: str,
    threshold: float,
    top_n: int,
    json_output: bool,
) -> None:
    path = Path(embeddings_path)
    if not path.exists():
        print(f"Error: {embeddings_path} not found. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    entries = load_embeddings(path)
    if not entries:
        print("Embeddings file is empty. Run 'build' first.")
        return

    results = search(query, entries, threshold=threshold, top_n=top_n)

    if json_output:
        import json as _json
        data = [
            {
                "id": entry.id,
                "text": entry.text,
                "score": round(float(score), 4),
            }
            for entry, score in results
        ]
        print(_json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not results:
        print(f"[KNOW]: No relevant information found (below {threshold:.2f} threshold).")
    else:
        for entry, score in results:
            tag = f" [{entry.id}]" if entry.id else ""
            print(f"[KNOW{tag}]: {entry.text} (score: {score:.3f})")


if __name__ == "__main__":
    main()
