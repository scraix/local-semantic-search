# SKILL: know — local semantic search

**Version:** 3.0.0
**Purpose:** Persist and retrieve project knowledge across sessions
**Files:** `knowledge.json` (source), `embeddings.bin` (Rust), `embeddings.pkl` (Python)

## Overview

The `know` system gives the agent a **read/write knowledge base**. Use it both to
look up project facts AND to proactively save important information so it
survives across sessions.

Two equivalent entry points:

| Method | Source | When to use |
|---|---|---|
| **Rust CLI** (`know build/search/add/list/remove`) | `~/Projects/tmp/local-semantic-search/target/release/know` | When the terminal tool is available. Fast. |
| **Python tools** (`knowledge_search`, `knowledge_add`, etc.) | `hermes_know.tool` | When direct tool registration is available in Hermes. |

## Workflow — the agent should do this automatically

### 1. BEFORE answering a project question → ALWAYS search first

```rust
knowledge_search(query="user's question", top_n=3)
// or via CLI:
// know search "user's question" -n 3
```

If results match, incorporate them into your answer. If nothing relevant
is found (empty results), answer from your own knowledge.

### 2. AFTER learning something important → PROACTIVELY save it

```python
knowledge_add(
    id="short-unique-id",
    text="The factual content that should persist across sessions.",
)
```

**What to save:** Anything you discover about the project that the user
would expect you to remember next time:
- Architecture decisions and rationale
- Configuration quirks (`config.yml` has this specific setting)
- Project conventions (branches, naming, testing style)
- Dependencies and their versions
- Decisions made during a conversation
- Workaround for tool limitations

**What NOT to save:** Transient task state, in-progress work, PR numbers,
commit SHAs, or things that expire within a week.

### 3. To REVIEW what's in the knowledge base

```python
knowledge_list()
```
Or: `know list`

### 4. To REMOVE stale information

```python
knowledge_remove(id="outdated-entry")
```
Or: `know remove outdated-entry`

## Command Reference (Rust CLI)

```bash
# Build / rebuild embeddings
know build                     # from knowledge.json → embeddings.bin
know build -k path/to/knowledge.json -o custom.bin

# Search
know search "what model do we use"     # top 1 result
know search "query" -n 3               # top 3 results
know search "query" -t 0.3             # lower threshold (more results)
know search "query" --json             # machine-parseable JSON output
know search "query" -n 3 -t 0.5 --json  # combine flags

# Write (add/list/remove)
know add "unique-id" "The fact to persist"
know list
know remove "unique-id"
```

## Command Reference (Python CLI)

```bash
python -m hermes_know build
python -m hermes_know search "query" -n 3 --json
python -m hermes_know add "unique-id" "text"
python -m hermes_know list
python -m hermes_know remove "unique-id"
```

## Tool Reference (Python Hermes Agent tools)

```python
knowledge_search(query="...", top_n=3, threshold=0.45)
    # → JSON with results, scores, count

knowledge_add(id="...", text="...")
    # → Adds to knowledge.json + rebuilds embeddings

knowledge_list()
    # → Returns all entries with id and text preview

knowledge_remove(id="...")
    # → Removes from knowledge.json + rebuilds embeddings

knowledge_build()
    # → Explicit rebuild (rarely needed — add/remove auto-build)
```

## JSON response format

All tools return JSON with `success: true/false`. For `knowledge_search`:

```json
{
  "success": true,
  "query": "...",
  "results": [
    { "id": "arch-model", "text": "...", "score": 0.89 }
  ],
  "count": 1,
  "message": "Found 1 matching entry."
}
```

## Important notes

- Both Rust and Python read/write the same `knowledge.json`
- Rust uses `embeddings.bin` (bincode), Python uses `embeddings.pkl` (pickle)
- After `add`/`remove`, embeddings are rebuilt automatically
- The model is `paraphrase-multilingual-MiniLM-L12-v2` (384 dims)
- Threshold default: 0.45 (lower = more results, higher = stricter)
