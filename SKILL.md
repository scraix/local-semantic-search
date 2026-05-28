---
name: local-knowledge-search
description: >
  Automatically search a project-level local knowledge base
  (knowledge.json + precomputed embeddings) to answer project-specific
  questions without overloading the agent's persistent memory.
  Supports both a fast Rust CLI and a native Python Hermes tool.
version: 2.0.0
---

# Local Project Knowledge Base (know)

When the user asks about a **specific project** (architecture, decisions,
configuration, rationale):

1. Check if `knowledge.json` exists in the current project directory
2. If yes — search it using the best available method:
   - **Python tool preferred** (if running inside Hermes): call `knowledge_search(query)`
   - **Rust CLI fallback** (terminal mode): run `know search "<question>"`
3. Use the matching context in your answer
4. If nothing matches — answer from general knowledge or memory()

## Python tool (Hermes Agent native)

```python
# Direct import
from hermes_know.tool import knowledge_search

result = knowledge_search("question about project", top_n=3)
# Returns JSON: {"success": true, "results": [...], "count": ...}
```

Auto-registers as `knowledge_search` + `knowledge_build` in Hermes tool
registry when `hermes_know` is in the Python path.

## Rust CLI (fallback)

```bash
# Build knowledge base (one-time per project)
know build

# Search (returns the best match)
know search "question about project"

# Search with custom parameters
know search -n 3 -t 0.5 "question"

# JSON output for structured parsing
know search --json "question"
```

## knowledge.json format

```json
[
  { "id": "unique_id", "text": "Fact about the project..." },
  { "id": "another_id", "text": "Another fact..." }
]
```

Embeddings model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions).
Default threshold: 0.45 | Default top-k: 1 (Rust) / 3 (Python tool).
