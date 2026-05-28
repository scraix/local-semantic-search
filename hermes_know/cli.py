"""
CLI entry point for hermes_know.

Usage:
    python -m hermes_know build [-k knowledge.json] [-o embeddings.pkl]
    python -m hermes_know search [--tags arch,cli] [-e embeddings.pkl] [-t 0.45] [-n 1] [--json] <query>
    python -m hermes_know get <id> [--json]
    python -m hermes_know add <id> <text> [--tags arch,cli]
    python -m hermes_know edit <id> <text> [--tags new,tags]
    python -m hermes_know list [--tags arch]
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


def _comma_tags(val: str) -> list[str]:
    """Parse comma-separated tags from CLI arg."""
    return [t.strip() for t in val.split(",") if t.strip()]


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
    search_parser.add_argument(
        "--tags", type=_comma_tags, default=None,
        help="Filter by tags, e.g. --tags arch,cli",
    )

    # ── get ─────────────────────────────────────────────────────────
    get_parser = sub.add_parser("get", help="Get a single entry by id")
    get_parser.add_argument("id", help="Entry identifier")
    get_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    get_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
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
    add_parser.add_argument(
        "--tags", type=_comma_tags, default=None,
        help="Comma-separated tags, e.g. --tags arch,cli",
    )

    # ── edit ────────────────────────────────────────────────────────
    edit_parser = sub.add_parser("edit", help="Update an existing entry")
    edit_parser.add_argument("id", help="Entry identifier to update")
    edit_parser.add_argument("text", help="New text content")
    edit_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    edit_parser.add_argument(
        "-o", "--output", default="embeddings.pkl",
        help="Output path for embeddings (default: embeddings.pkl)",
    )
    edit_parser.add_argument(
        "--tags", type=_comma_tags, default=None,
        help="Replace tags with new comma-separated set, e.g. --tags arch,config",
    )

    # ── list ────────────────────────────────────────────────────────
    list_parser = sub.add_parser("list", help="List all entries")
    list_parser.add_argument(
        "-k", "--knowledge", default="knowledge.json",
        help="Path to knowledge.json (default: knowledge.json)",
    )
    list_parser.add_argument(
        "--tags", type=_comma_tags, default=None,
        help="Filter by tags, e.g. --tags arch",
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
        _do_search(args.query, args.embeddings, args.threshold, args.top, args.json, args.tags)
    elif args.command == "get":
        _do_get(args.id, args.knowledge, args.json)
    elif args.command == "add":
        _do_add(args.id, args.text, args.tags or [], args.knowledge, args.output)
    elif args.command == "edit":
        _do_edit(args.id, args.text, args.tags, args.knowledge, args.output)
    elif args.command == "list":
        _do_list(args.knowledge, args.tags)
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
    tags: list[str] | None,
) -> None:
    path = Path(embeddings_path)
    if not path.exists():
        print(f"Error: {embeddings_path} not found. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    entries = load_embeddings(path)
    if not entries:
        print("Embeddings file is empty. Run 'build' first.")
        return

    results = search(query, entries, threshold=threshold, top_n=top_n, tags=tags)

    if json_output:
        import json as _json
        data = [
            {
                "id": entry.id,
                "text": entry.text,
                "tags": entry.tags,
                "score": round(float(score), 4),
            }
            for entry, score in results
        ]
        print(_json.dumps(
            {"success": True, "query": query, "results": data, "count": len(data)},
            ensure_ascii=False, indent=2,
        ))
        return

    if not results:
        print(f"[KNOW]: No relevant information found (below {threshold:.2f} threshold).")
    else:
        for entry, score in results:
            tag = f" [{entry.id}]" if entry.id else ""
            tags_str = f" tags:({','.join(entry.tags)})" if entry.tags else ""
            print(f"[KNOW{tag}]: {entry.text}{tags_str} (score: {score:.3f})")


def _do_get(id: str, knowledge_path: str, json_output: bool) -> None:
    kp = Path(knowledge_path)
    if not kp.exists():
        if json_output:
            import json as _json
            print(_json.dumps({"success": False, "error": f"{knowledge_path} not found."}))
        else:
            print(f"Error: {knowledge_path} not found.", file=sys.stderr)
        return

    entries = load_knowledge(kp)
    for e in entries:
        if e.id == id:
            if json_output:
                import json as _json
                print(_json.dumps(
                    {"success": True, "id": e.id, "text": e.text, "tags": e.tags},
                    ensure_ascii=False, indent=2,
                ))
            else:
                tags_line = f"\ntags: {', '.join(e.tags)}" if e.tags else ""
                print(f"[{e.id}]{tags_line}\n{e.text}")
            return

    if json_output:
        import json as _json
        print(_json.dumps({"success": False, "error": f"Entry '{id}' not found."}))
    else:
        print(f"Entry '{id}' not found.")


def _do_add(
    id: str,
    text: str,
    tags: list[str],
    knowledge_path: str,
    output_path: str,
) -> None:
    kp = Path(knowledge_path)
    entries = load_knowledge(kp) if kp.exists() else []

    if any(e.id == id for e in entries):
        print(f"Entry with id '{id}' already exists. Use 'edit' to update.")
        sys.exit(1)

    entries.append(KnowledgeEntry(id=id, text=text, tags=tags))
    write_knowledge(entries, kp)
    tag_str = f" tags: {', '.join(tags)}" if tags else ""
    print(f"Added: [{id}]{tag_str}")

    print("Rebuilding embeddings...")
    embedded = embed_texts(entries)
    save_embeddings(embedded, output_path)


def _do_edit(
    id: str,
    text: str,
    tags: list[str] | None,
    knowledge_path: str,
    output_path: str,
) -> None:
    kp = Path(knowledge_path)
    if not kp.exists():
        print(f"Error: {knowledge_path} not found.", file=sys.stderr)
        sys.exit(1)

    entries = load_knowledge(kp)
    found = False
    for e in entries:
        if e.id == id:
            e.text = text
            if tags is not None:
                e.tags = tags
            found = True
            break

    if not found:
        print(f"Entry '{id}' not found.")
        sys.exit(1)

    write_knowledge(entries, kp)
    print(f"Edited: [{id}]")

    print("Rebuilding embeddings...")
    embedded = embed_texts(entries)
    save_embeddings(embedded, output_path)


def _do_list(knowledge_path: str, tags: list[str] | None) -> None:
    kp = Path(knowledge_path)
    if not kp.exists():
        print(f"No entries — {knowledge_path} not found.")
        return

    entries = load_knowledge(kp)

    if tags:
        from .core import matches_tags
        entries = [e for e in entries if matches_tags(e.tags, tags)]

    if not entries:
        if tags:
            print(f"No entries matching tags [{', '.join(tags)}] in {knowledge_path}.")
        else:
            print(f"No entries in {knowledge_path}.")
        return

    label = "y" if len(entries) == 1 else "ies"
    print(f"{len(entries)} entr{label} in {knowledge_path}:\n")
    for i, entry in enumerate(entries, 1):
        preview = entry.text[:70] + "..." if len(entry.text) > 73 else entry.text
        tags_str = f" [{','.join(entry.tags)}]" if entry.tags else ""
        print(f"  {i}. [{entry.id}]{tags_str} {preview}")


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
