"""
CLI entry point for hermes_know.

Usage:
    python -m hermes_know build [-k knowledge.json] [-o embeddings.pkl]
    python -m hermes_know search [-e embeddings.pkl] [-t 0.45] [-n 1] [--json] <query>
    python -m hermes_know add <id> <text>
    python -m hermes_know list
    python -m hermes_know remove <id>
"""

import argparse
import sys
from pathlib import Path

from .core import (
    KnowledgeEntry,
    embed_texts,
    load_knowledge,
    load_embeddings,
    save_embeddings,
    search,
    write_knowledge,
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

    # ── add ─────────────────────────────────────────────────────────
    add_parser = sub.add_parser("add", help="Add a new entry")
    add_parser.add_argument("id", help="Entry identifier")
    add_parser.add_argument("text", help="The factual content")
    add_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    add_parser.add_argument(
        "-o", "--output", default="embeddings.pkl",
        help="Output path for embeddings (default: embeddings.pkl)",
    )

    # ── list ────────────────────────────────────────────────────────
    list_parser = sub.add_parser("list", help="List all entries")
    list_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )

    # ── remove ──────────────────────────────────────────────────────
    remove_parser = sub.add_parser("remove", help="Remove an entry by id")
    remove_parser.add_argument("id", help="Id of entry to remove")
    remove_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    remove_parser.add_argument(
        "-o", "--output", default="embeddings.pkl",
        help="Output path for embeddings (default: embeddings.pkl)",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        _do_build(args.knowledge, args.output)
    elif args.command == "search":
        _do_search(args.query, args.embeddings, args.threshold, args.top, args.json)
    elif args.command == "add":
        _do_add(args.id, args.text, args.knowledge, args.output)
    elif args.command == "list":
        _do_list(args.knowledge)
    elif args.command == "remove":
        _do_remove(args.id, args.knowledge, args.output)


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


def _do_add(id: str, text: str, knowledge_path: str, output_path: str) -> None:
    kp = Path(knowledge_path)
    entries = load_knowledge(kp) if kp.exists() else []

    if any(e.id == id for e in entries):
        print(f"Entry with id '{id}' already exists. Use 'remove {id}' first.")
        sys.exit(1)

    entries.append(KnowledgeEntry(id=id, text=text))
    write_knowledge(entries, kp)
    print(f"Added: [{id}]")

    # Rebuild
    print("Rebuilding embeddings...")
    embedded = embed_texts(entries)
    save_embeddings(embedded, output_path)


def _do_list(knowledge_path: str) -> None:
    kp = Path(knowledge_path)
    if not kp.exists():
        print(f"No entries — {knowledge_path} not found.")
        return

    entries = load_knowledge(kp)
    if not entries:
        print(f"No entries in {knowledge_path}.")
        return

    print(f"{len(entries)} entries in {knowledge_path}:\n")
    for i, entry in enumerate(entries, 1):
        preview = entry.text[:77] + "..." if len(entry.text) > 80 else entry.text
        print(f"  {i}. [{entry.id}] {preview}")


def _do_remove(id: str, knowledge_path: str, output_path: str) -> None:
    kp = Path(knowledge_path)
    if not kp.exists():
        print(f"Error: {knowledge_path} not found.", file=sys.stderr)
        sys.exit(1)

    entries = load_knowledge(kp)
    before = len(entries)
    entries = [e for e in entries if e.id != id]

    if len(entries) == before:
        print(f"Entry with id '{id}' not found.")
        sys.exit(1)

    write_knowledge(entries, kp)
    print(f"Removed: [{id}]")

    if not entries:
        print("No entries left — embeddings deleted.")
        Path(output_path).unlink(missing_ok=True)
    else:
        print("Rebuilding embeddings...")
        embedded = embed_texts(entries)
        save_embeddings(embedded, output_path)


if __name__ == "__main__":
    main()
