"""
Hermes Agent tool registration for ``knowledge_search`` and ``knowledge_build``.

Can be loaded by Hermes as a custom tool module. Registers two tools:
  1. ``knowledge_search(query, top_n, threshold)`` — semantic search
  2. ``knowledge_build(knowledge_path=None, output_path=None)`` — rebuild

Usage in Hermes agent context:
    >>> from hermes_know.tool import knowledge_search
    >>> result = knowledge_search("how does search work")
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .core import (
    embed_texts,
    load_knowledge,
    load_embeddings,
    save_embeddings,
    search as _search,
)

logger = logging.getLogger(__name__)

# Default paths — resolve relative to the calling project's working directory.
# Override with env vars KNOWLEDGE_JSON and EMBEDDINGS_FILE.
DEFAULT_KNOWLEDGE_JSON = os.environ.get("KNOWLEDGE_JSON", "knowledge.json")
DEFAULT_EMBEDDINGS = os.environ.get("EMBEDDINGS_FILE", "embeddings.pkl")


def _resolve_path(p: Optional[str], default: str) -> Path:
    """Resolve path, defaulting to cwd-relative."""
    return Path(p).expanduser().resolve() if p else Path(default).resolve()


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

    Returns matching entries with similarity scores.
    Returns JSON string with results.

    Args:
        query: The search query (natural language).
        top_n: Maximum number of results to return (default: 3).
        threshold: Minimum similarity score 0.0–1.0 (default: 0.45).
        embeddings_path: Path to embeddings file, or auto-detect.
    """
    ep = _resolve_path(embeddings_path, DEFAULT_EMBEDDINGS)

    if not ep.exists():
        # Try building on-the-fly
        kp = _resolve_path(None, DEFAULT_KNOWLEDGE_JSON)
        if kp.exists():
            logger.info("Embeddings not found at %s — building from %s", ep, kp)
            _do_build(str(kp), str(ep))
        else:
            return json.dumps({
                "success": False,
                "error": f"Embeddings file '{ep}' not found, and no knowledge.json found to build from.",
            })

    try:
        entries = load_embeddings(str(ep))
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to load embeddings: {e}",
        })

    if not entries:
        return json.dumps({
            "success": False,
            "error": "Embeddings file is empty. Run knowledge_build first.",
        })

    try:
        results = _search(query, entries, threshold=threshold, top_n=top_n)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {e}",
        })

    if not results:
        return json.dumps({
            "success": True,
            "query": query,
            "results": [],
            "count": 0,
            "message": f"No relevant information found (below {threshold:.2f} threshold).",
        })

    data = [
        {
            "id": entry.id,
            "text": entry.text,
            "score": round(float(score), 4),
        }
        for entry, score in results
    ]

    return json.dumps({
        "success": True,
        "query": query,
        "results": data,
        "count": len(data),
        "message": f"Found {len(data)} matching entr{'y' if len(data) == 1 else 'ies'}.",
    }, ensure_ascii=False)


def knowledge_build(
    knowledge_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """(Re)build embeddings from knowledge.json.

    Use this when knowledge.json has been updated and embeddings need
    regeneration.

    Args:
        knowledge_path: Path to knowledge.json (default: auto-detect).
        output_path: Output path for embeddings (default: auto-detect).
    """
    kp = _resolve_path(knowledge_path, DEFAULT_KNOWLEDGE_JSON)
    op = _resolve_path(output_path, DEFAULT_EMBEDDINGS)

    try:
        entries = load_knowledge(str(kp))
    except FileNotFoundError:
        return json.dumps({
            "success": False,
            "error": f"knowledge.json not found at {kp}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to read knowledge.json: {e}",
        })

    if not entries:
        return json.dumps({
            "success": True,
            "warning": "knowledge.json is empty.",
            "entries_built": 0,
        })

    try:
        embedded = embed_texts(entries)
        save_embeddings(embedded, str(op))
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Embedding failed: {e}",
        })

    return json.dumps({
        "success": True,
        "entries_built": len(embedded),
        "output_path": str(op),
        "dimensions": 384,
    })


def _do_build(knowledge_path: str, output_path: str) -> None:
    """Non-JSON build (internal use)."""
    entries = load_knowledge(knowledge_path)
    if not entries:
        print("knowledge.json is empty. Nothing to embed.")
        return
    print(f"Found {len(entries)} entries. Generating embeddings...")
    embedded = embed_texts(entries)
    save_embeddings(embedded, output_path)


# ═══════════════════════════════════════════════════════════════════════
# Hermes tool registry integration (optional)
# ═══════════════════════════════════════════════════════════════════════

# Tool schemas for Hermes Agent registry
KNOWLEDGE_SEARCH_SCHEMA = {
    "name": "knowledge_search",
    "description": (
        "Semantic search over the project's local knowledge base (knowledge.json). "
        "Use this to answer project-specific questions about architecture, decisions, "
        "configuration, rationale, and conventions. "
        "Returns matching entries with similarity scores. "
        "If embeddings are missing, automatically builds them from knowledge.json."
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
    "description": (
        "(Re)build embeddings from knowledge.json. "
        "Use when knowledge.json has been updated and embeddings need regeneration."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


def _check_knowledge_available() -> bool:
    """Check if knowledge.json exists in current directory."""
    return Path(DEFAULT_KNOWLEDGE_JSON).exists() or Path(
        os.environ.get("EMBEDDINGS_FILE", "embeddings.pkl")
    ).exists()


# Attempt to register with Hermes tool registry when loaded in agent context
try:
    from tools.registry import registry

    registry.register(
        name="knowledge_search",
        toolset="knowledge",
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
        toolset="knowledge",
        schema=KNOWLEDGE_BUILD_SCHEMA,
        handler=lambda args, **kw: knowledge_build(),
        check_fn=_check_knowledge_available,
        emoji="🔨",
        description=KNOWLEDGE_BUILD_SCHEMA["description"],
    )
    logger.info("knowledge tools registered with Hermes Agent")
except ImportError:
    # Not running inside Hermes — tools are available as direct imports
    pass
except Exception as e:
    logger.debug("Failed to register knowledge tools: %s", e)
