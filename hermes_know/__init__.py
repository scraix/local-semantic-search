"""
hermes_know — Local semantic search for project knowledge bases.

Provides two main entry points:
  - ``knowledge_search(query, top_n, threshold)`` — Hermes Agent tool
  - ``knowledge_build(knowledge_path, output_path)`` — rebuild embeddings
  - CLI: ``python -m hermes_know search "..."``
"""

from .core import (
    cosine_similarity,
    embed_query,
    embed_texts,
    load_embeddings,
    load_knowledge,
    save_embeddings,
    search,
    KnowledgeEntry,
    EmbeddingEntry,
)

from .cli import main as cli_main
from .tool import knowledge_search, knowledge_build

__all__ = [
    "cosine_similarity",
    "embed_query",
    "embed_texts",
    "load_embeddings",
    "load_knowledge",
    "save_embeddings",
    "search",
    "knowledge_search",
    "knowledge_build",
    "cli_main",
    "KnowledgeEntry",
    "EmbeddingEntry",
]
