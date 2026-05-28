"""
Hermes Agent tool registration for project knowledge base operations.

Registers tools:
  1. ``knowledge_search(query, top_n, threshold)`` — semantic search
  2. ``knowledge_get(id)`` — get a single entry by id (fast)
  3. ``knowledge_add(id, text)`` — add a new entry (auto-builds)
  4. ``knowledge_edit(id, text)`` — update an existing entry (auto-builds)
  5. ``knowledge_list()`` — list all entries
  6. ``knowledge_remove(id)`` — remove an entry by id (auto-builds)
  7. ``knowledge_build()`` — rebuild embeddings

All tools read/write the same ``knowledge.json`` as the Rust CLI.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .core import (
    KnowledgeEntry,
    embed_texts,
    load_knowledge,
    load_embeddings,
    save_embeddings,
    search as _search,
    write_knowledge,
)

logger = logging.getLogger(__name__)

# Default paths — resolve relative to the calling project's working directory.
# Override with env vars KNOWLEDGE_JSON and EMBEDDINGS_FILE.
DEFAULT_KNOWLEDGE_JSON = os.environ.get("KNOWLEDGE_JSON", "knowledge.json")
DEFAULT_EMBEDDINGS = os.environ.get("EMBEDDINGS_FILE", "embeddings.pkl")


def _resolve_path(p: Optional[str], default: str) -> Path:
    """Resolve path, defaulting to cwd-relative."""
    return Path(p).expanduser().resolve() if p else Path(default).resolve()


def _auto_build(knowledge_path: Path, output_path: Path) -> bool:
    """Build embeddings if knowledge.json exists."""
    if not knowledge_path.exists():
        return False
    entries = load_knowledge(str(knowledge_path))
    if not entries:
        return False
    logger.info("Auto-building embeddings from %s", knowledge_path)
    embedded = embed_texts(entries)
    save_embeddings(embedded, str(output_path))
    return True


# ═══════════════════════════════════════════════════════════════════════
# Tool functions
# ═══════════════════════════════════════════════════════════════════════

def knowledge_search(
    query: str,
    top_n: int = 3,
    threshold: float = 0.45,
    embeddings_path: Optional[str] = None,
) -> str:
    """Semantic search over the project knowledge base.

    Returns matching entries with similarity scores (JSON string).
    Auto-builds embeddings if missing but knowledge.json exists.

    Args:
        query: The search query (natural language).
        top_n: Maximum number of results (default: 3).
        threshold: Minimum similarity score 0.0–1.0 (default: 0.45).
    """
    kp = _resolve_path(None, DEFAULT_KNOWLEDGE_JSON)
    ep = _resolve_path(embeddings_path, DEFAULT_EMBEDDINGS)

    if not ep.exists():
        if kp.exists():
            _auto_build(kp, ep)
        else:
            return _json_error(f"No knowledge.json found at {kp} — create one first.")

    try:
        entries = load_embeddings(str(ep))
    except Exception as e:
        return _json_error(f"Failed to load embeddings: {e}")

    if not entries:
        return _json_error("Embeddings file is empty. Run knowledge_build first.")

    try:
        results = _search(query, entries, threshold=threshold, top_n=top_n)
    except Exception as e:
        return _json_error(f"Search failed: {e}")

    if not results:
        return _json_ok({
            "query": query,
            "results": [],
            "count": 0,
            "message": f"No relevant information found (below {threshold:.2f} threshold).",
        })

    data = [
        {"id": entry.id, "text": entry.text, "score": round(float(score), 4)}
        for entry, score in results
    ]
    return _json_ok({
        "query": query,
        "results": data,
        "count": len(data),
        "message": f"Found {len(data)} matching entr{'y' if len(data) == 1 else 'ies'}.",
    })


def knowledge_build(
    knowledge_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """(Re)build embeddings from knowledge.json.

    Use when knowledge.json has been updated and embeddings need regeneration.

    Returns JSON with build status.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)
    op = _resolve_path(output_path, DEFAULT_EMBEDDINGS)

    try:
        entries = load_knowledge(str(kp))
    except FileNotFoundError:
        return _json_error(f"knowledge.json not found at {kp}")
    except Exception as e:
        return _json_error(f"Failed to read knowledge.json: {e}")

    if not entries:
        return _json_ok({"warning": "knowledge.json is empty.", "entries_built": 0})

    try:
        embedded = embed_texts(entries)
        save_embeddings(embedded, str(op))
    except Exception as e:
        return _json_error(f"Embedding failed: {e}")

    return _json_ok({
        "entries_built": len(embedded),
        "output_path": str(op),
        "dimensions": 384,
    })


def knowledge_add(
    id: str,
    text: str,
    knowledge_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Add a new entry to the knowledge base and rebuild embeddings.

    Call this PROACTIVELY when you discover an important project-specific
    fact that should be persisted for future sessions — architecture
    decisions, configuration quirks, rationale behind choices, etc.

    Args:
        id: Short unique identifier (e.g. 'arch-decision', 'config-tip').
        text: The factual content to store.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)
    op = _resolve_path(output_path, DEFAULT_EMBEDDINGS)

    entries = load_knowledge(str(kp)) if kp.exists() else []

    if any(e.id == id for e in entries):
        return _json_error(f"Entry with id '{id}' already exists. Use knowledge_remove first.")

    entries.append(KnowledgeEntry(id=id, text=text))

    try:
        write_knowledge(entries, str(kp))
    except Exception as e:
        return _json_error(f"Failed to write knowledge.json: {e}")

    # Rebuild embeddings
    try:
        embedded = embed_texts(entries)
        save_embeddings(embedded, str(op))
    except Exception as e:
        # knowledge.json was updated, but embeddings failed
        return _json_ok({
            "id": id,
            "added": True,
            "embeddings_rebuilt": False,
            "error": str(e),
            "message": "Entry added to knowledge.json but embedding rebuild failed. Run knowledge_build later.",
        })

    return _json_ok({
        "id": id,
        "added": True,
        "embeddings_rebuilt": True,
        "entries_count": len(embedded),
        "message": f"Added [{id}] ({len(embedded)} entries total).",
    })


def knowledge_list(
    knowledge_path: Optional[str] = None,
) -> str:
    """List all entries in the knowledge base.

    Returns JSON with id, text preview, and count.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)

    if not kp.exists():
        return _json_ok({"entries": [], "count": 0, "message": "No knowledge.json found."})

    try:
        entries = load_knowledge(str(kp))
    except Exception as e:
        return _json_error(f"Failed to read knowledge.json: {e}")

    data = [
        {"id": e.id, "text_preview": e.text[:100] + ("..." if len(e.text) > 100 else "")}
        for e in entries
    ]
    return _json_ok({
        "entries": data,
        "count": len(data),
        "message": f"{len(data)} entr{'y' if len(data) == 1 else 'ies'} in knowledge base.",
    })


def knowledge_get(
    id: str,
    knowledge_path: Optional[str] = None,
) -> str:
    """Get a single entry by id (fast — no model load, no embedding).

    Args:
        id: The identifier of the entry to retrieve.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)

    if not kp.exists():
        return _json_error(f"knowledge.json not found at {kp}")

    try:
        entries = load_knowledge(str(kp))
    except Exception as e:
        return _json_error(f"Failed to read knowledge.json: {e}")

    for e in entries:
        if e.id == id:
            return _json_ok({"id": e.id, "text": e.text})

    return _json_error(f"Entry '{id}' not found.")


def knowledge_edit(
    id: str,
    text: str,
    knowledge_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Update an existing entry and rebuild embeddings.

    Use instead of remove + add when you want to update a fact in-place.

    Args:
        id: The identifier of the entry to update.
        text: The new text content.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)
    op = _resolve_path(output_path, DEFAULT_EMBEDDINGS)

    if not kp.exists():
        return _json_error(f"knowledge.json not found at {kp}")

    try:
        entries = load_knowledge(str(kp))
    except Exception as e:
        return _json_error(f"Failed to read knowledge.json: {e}")

    found = False
    for e in entries:
        if e.id == id:
            e.text = text
            found = True
            break

    if not found:
        return _json_error(f"Entry '{id}' not found.")

    try:
        write_knowledge(entries, str(kp))
    except Exception as e:
        return _json_error(f"Failed to write knowledge.json: {e}")

    # Rebuild embeddings
    try:
        embedded = embed_texts(entries)
        save_embeddings(embedded, str(op))
    except Exception as e:
        return _json_ok({
            "id": id,
            "edited": True,
            "embeddings_rebuilt": False,
            "error": str(e),
            "message": "Entry updated in knowledge.json but embedding rebuild failed. Run knowledge_build later.",
        })

    return _json_ok({
        "id": id,
        "edited": True,
        "embeddings_rebuilt": True,
        "entries_count": len(embedded),
        "message": f"Edited [{id}] ({len(embedded)} entries total).",
    })


def knowledge_remove(
    id: str,
    knowledge_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Remove an entry from the knowledge base by id and rebuild embeddings.

    Args:
        id: The identifier of the entry to remove.
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)
    op = _resolve_path(output_path, DEFAULT_EMBEDDINGS)

    if not kp.exists():
        return _json_error(f"knowledge.json not found at {kp}")

    try:
        entries = load_knowledge(str(kp))
    except Exception as e:
        return _json_error(f"Failed to read knowledge.json: {e}")

    before = len(entries)
    entries = [e for e in entries if e.id != id]

    if len(entries) == before:
        return _json_error(f"Entry with id '{id}' not found.")

    try:
        write_knowledge(entries, str(kp))
    except Exception as e:
        return _json_error(f"Failed to write knowledge.json: {e}")

    # Rebuild or delete embeddings
    result = {"id": id, "removed": True}
    if not entries:
        # No entries left — clean up
        op_path = Path(str(op))
        if op_path.exists():
            op_path.unlink()
        result["embeddings_deleted"] = True
        result["message"] = f"Removed [{id}]. No entries left, embeddings deleted."
    else:
        try:
            embedded = embed_texts(entries)
            save_embeddings(embedded, str(op))
            result["embeddings_rebuilt"] = True
            result["entries_count"] = len(embedded)
            result["message"] = f"Removed [{id}] ({len(embedded)} entries remain)."
        except Exception as e:
            result["embeddings_rebuilt"] = False
            result["error"] = str(e)
            result["message"] = f"Removed [{id}] from knowledge.json but embeddings rebuild failed."

    return _json_ok(result)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _json_ok(data: dict) -> str:
    """Return a success JSON response."""
    data["success"] = True
    return json.dumps(data, ensure_ascii=False)


def _json_error(msg: str) -> str:
    """Return an error JSON response."""
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Hermes tool registry integration
# ═══════════════════════════════════════════════════════════════════════

KNOWLEDGE_SEARCH_SCHEMA = {
    "name": "knowledge_search",
    "description": (
        "Semantic search over the project's local knowledge base (knowledge.json). "
        "Returns matching entries with similarity scores. "
        "CALL THIS FIRST when the user asks a project-specific question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query in natural language",
            },
            "top_n": {
                "type": "integer",
                "default": 3,
                "description": "Maximum number of results (default: 3)",
            },
            "threshold": {
                "type": "number",
                "default": 0.45,
                "description": "Minimum similarity score 0.0–1.0 (default: 0.45)",
            },
        },
        "required": ["query"],
    },
}

KNOWLEDGE_BUILD_SCHEMA = {
    "name": "knowledge_build",
    "description": "(Re)build embeddings from knowledge.json.",
    "parameters": {"type": "object", "properties": {}},
}

KNOWLEDGE_ADD_SCHEMA = {
    "name": "knowledge_add",
    "description": (
        "Add a new entry to the project knowledge base. "
        "Call this PROACTIVELY when you learn a project-specific fact that "
        "should persist across sessions — architecture, config, rationale, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Short unique identifier (e.g. 'arch-decision', 'config-tip')",
            },
            "text": {
                "type": "string",
                "description": "The factual content to store",
            },
        },
        "required": ["id", "text"],
    },
}

KNOWLEDGE_LIST_SCHEMA = {
    "name": "knowledge_list",
    "description": "List all entries in the project knowledge base.",
    "parameters": {"type": "object", "properties": {}},
}

KNOWLEDGE_REMOVE_SCHEMA = {
    "name": "knowledge_remove",
    "description": "Remove an entry from the knowledge base by id.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id of the entry to remove",
            },
        },
        "required": ["id"],
    },
}

KNOWLEDGE_GET_SCHEMA = {
    "name": "knowledge_get",
    "description": (
        "Get a single entry by its id. "
        "Fast — no model load. Use this when you know the id and just "
        "need the full text, rather than searching semantically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id of the entry to retrieve",
            },
        },
        "required": ["id"],
    },
}

KNOWLEDGE_EDIT_SCHEMA = {
    "name": "knowledge_edit",
    "description": (
        "Update an existing entry's text. "
        "Use instead of remove+add when a fact changes — one call, auto-rebuilds embeddings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id of the entry to update",
            },
            "text": {
                "type": "string",
                "description": "The new text content",
            },
        },
        "required": ["id", "text"],
    },
}


def _check_knowledge_available() -> bool:
    """Check if knowledge files exist in the current directory."""
    return (
        Path(DEFAULT_KNOWLEDGE_JSON).exists()
        or Path(os.environ.get("EMBEDDINGS_FILE", DEFAULT_EMBEDDINGS)).exists()
    )


# Attempt to register with Hermes tool registry
try:
    from tools.registry import registry

    _TOOLSET = "knowledge"

    registry.register(
        name="knowledge_search",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_SEARCH_SCHEMA,
        handler=lambda args, **kw: knowledge_search(
            query=args["query"],
            top_n=args.get("top_n", 3),
            threshold=args.get("threshold", 0.45),
        ),
        check_fn=_check_knowledge_available,
        emoji="📚",
        description=KNOWLEDGE_SEARCH_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_build",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_BUILD_SCHEMA,
        handler=lambda args, **kw: knowledge_build(),
        check_fn=_check_knowledge_available,
        emoji="🔨",
        description=KNOWLEDGE_BUILD_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_add",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_ADD_SCHEMA,
        handler=lambda args, **kw: knowledge_add(
            id=args["id"],
            text=args["text"],
        ),
        check_fn=_check_knowledge_available,
        emoji="➕",
        description=KNOWLEDGE_ADD_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_list",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_LIST_SCHEMA,
        handler=lambda args, **kw: knowledge_list(),
        check_fn=_check_knowledge_available,
        emoji="📋",
        description=KNOWLEDGE_LIST_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_remove",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_REMOVE_SCHEMA,
        handler=lambda args, **kw: knowledge_remove(
            id=args["id"],
        ),
        check_fn=_check_knowledge_available,
        emoji="🗑️",
        description=KNOWLEDGE_REMOVE_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_get",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_GET_SCHEMA,
        handler=lambda args, **kw: knowledge_get(id=args["id"]),
        check_fn=_check_knowledge_available,
        emoji="🔍",
        description=KNOWLEDGE_GET_SCHEMA["description"],
    )
    registry.register(
        name="knowledge_edit",
        toolset=_TOOLSET,
        schema=KNOWLEDGE_EDIT_SCHEMA,
        handler=lambda args, **kw: knowledge_edit(id=args["id"], text=args["text"]),
        check_fn=_check_knowledge_available,
        emoji="✏️",
        description=KNOWLEDGE_EDIT_SCHEMA["description"],
    )
    logger.info("knowledge tools (7) registered with Hermes Agent")
except ImportError:
    pass
except Exception as e:
    logger.debug("Failed to register knowledge tools: %s", e)
