"""
Core logic: embedding, storage, search.

Compatible with the same ``knowledge.json`` format as the Rust CLI.
Uses fastembed (Python) with the same model:
  ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dimensions).

Embeddings are stored as a simple pickle format (``.pkl``) for Python.
The Rust CLI uses ``.bin`` (bincode) — incompatible, but both read the
same ``knowledge.json``.
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from fastembed import TextEmbedding

# The model now uses mean pooling instead of CLS — harmless, suppress warning
warnings.filterwarnings("ignore", message=".*now uses mean pooling.*")
logging.getLogger("fastembed").setLevel(logging.WARNING)

# ── Model ──────────────────────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

_model_instance: Optional[TextEmbedding] = None


def _get_model() -> TextEmbedding:
    """Lazy-loaded singleton embedding model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(Path.home() / ".cache" / "know" / "models"),
        )
    return _model_instance


# ── Data types ─────────────────────────────────────────────────────────

@dataclass
class KnowledgeEntry:
    id: str
    text: str
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        return cls(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            tags=list(d.get("tags", [])),
        )


@dataclass
class EmbeddingEntry:
    id: str
    text: str
    vector: np.ndarray  # shape: (384,)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "tags": self.tags,
            "vector": self.vector.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingEntry":
        return cls(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            tags=list(d.get("tags", [])),
            vector=np.array(d["vector"], dtype=np.float32),
        )


# ── Helpers ────────────────────────────────────────────────────────────

def matches_tags(entry_tags: List[str], required: List[str]) -> bool:
    """Check if entry has ALL required tags."""
    if not required:
        return True
    return all(r in entry_tags for r in required)


# ── I/O ────────────────────────────────────────────────────────────────

def load_knowledge(path: str | Path = "knowledge.json") -> List[KnowledgeEntry]:
    """Read knowledge.json (same format as Rust CLI)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [KnowledgeEntry.from_dict(item) for item in raw]


def write_knowledge(
    entries: List[KnowledgeEntry],
    path: str | Path = "knowledge.json",
) -> None:
    """Write entries to knowledge.json."""
    path = Path(path)
    data = []
    for e in entries:
        entry: dict = {"id": e.id, "text": e.text}
        if e.tags:
            entry["tags"] = e.tags
        data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_embeddings(entries: List[EmbeddingEntry], path: str | Path = "embeddings.pkl") -> None:
    """Save embeddings to pickle (Python-native format)."""
    path = Path(path)
    data = [e.to_dict() for e in entries]
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"{path} written ({len(entries)} entries, {EMBEDDING_DIM} dimensions)")


def load_embeddings(path: str | Path = "embeddings.pkl") -> List[EmbeddingEntry]:
    """Load embeddings from pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [EmbeddingEntry.from_dict(item) for item in data]


# ── Embedding ──────────────────────────────────────────────────────────

def embed_texts(
    entries: List[KnowledgeEntry],
) -> List[EmbeddingEntry]:
    """Generate embeddings for all knowledge entries."""
    model = _get_model()
    texts = [e.text for e in entries]

    # fastembed returns Iterable[np.ndarray]
    vectors = list(model.embed(texts))

    result: List[EmbeddingEntry] = []
    for entry, vec in zip(entries, vectors):
        # vec comes as (dim,) already from fastembed
        result.append(
            EmbeddingEntry(id=entry.id, text=entry.text, tags=entry.tags, vector=vec)
        )
    return result


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    model = _get_model()
    vectors = list(model.embed([query]))
    return vectors[0]


# ── Search ─────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(
    query: str,
    entries: List[EmbeddingEntry],
    threshold: float = 0.45,
    top_n: int = 1,
    tags: Optional[List[str]] = None,
) -> List[Tuple[EmbeddingEntry, float]]:
    """Search embeddings by cosine similarity.

    If ``tags`` is provided, only entries containing ALL of the specified
    tags are considered. This filtering happens before similarity computation
    (no need to embed entries that are filtered out).

    Returns list of (entry, score) sorted descending, filtered by threshold.
    """
    # Pre-filter by tags
    filtered = entries if not tags else [e for e in entries if matches_tags(e.tags, tags)]
    if not filtered:
        return []

    query_vec = embed_query(query)

    scored: List[Tuple[EmbeddingEntry, float]] = []
    for entry in filtered:
        score = cosine_similarity(query_vec, entry.vector)
        if score >= threshold:
            scored.append((entry, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
