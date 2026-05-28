# SKILL: know — local semantic search

**Version:** 4.0.0
**Purpose:** Persist and retrieve project knowledge across sessions
**Files:** `knowledge.json` (source), `embeddings.bin` (Rust), `embeddings.pkl` (Python)

## Overview

The `know` system gives the agent a **full CRUD knowledge base** (Create, Read,
Update, Delete). Use it to look up project facts, save important information,
and keep the knowledge base current.

Two equivalent entry points:

| Method | Source | When to use |
|---|---|---|
| **Rust CLI** (`know`) | `~/Projects/tmp/local-semantic-search/target/release/know` | When the terminal tool is available. Fast. |
| **Python tools** | `hermes_know.tool` | When direct tool registration is available in Hermes. |

## 7 Hermes Agent tools

| Tool | Action | When |
|---|---|---|
| `knowledge_search(query, top_n, threshold)` | Semantic search | **Before** answering project questions |
| `knowledge_get(id)` | Get entry by id | When you know the id — fast, no model load |
| `knowledge_add(id, text)` | Save a new fact | **After** learning something important |
| `knowledge_edit(id, text)` | Update existing fact | When a known fact becomes outdated |
| `knowledge_list()` | List all entries | Overview / maintenance |
| `knowledge_remove(id)` | Delete entry | When info is stale or wrong |
| `knowledge_build()` | Rebuild embeddings | Rarely needed (add/edit/remove auto-build) |

## Workflow — the agent should do this automatically

### 1. BEFORE answering a project question → ALWAYS search first

```python
knowledge_search(query="user's question", top_n=3)
```

If results match, incorporate them into your answer. If nothing relevant
is found, answer from your own knowledge.

### 2. AFTER learning something important → PROACTIVELY save it

```python
knowledge_add(
    id="short-unique-id",
    text="The factual content that should persist across sessions.",
)
```

**What to save:** Architecture decisions, config quirks, project conventions,
dependency details, rationale behind choices — anything the user would expect
you to remember next session.

**What NOT to save:** Transient task state, PR numbers, commit SHAs, temp work.

### 3. BEFORE adding → check if entry already exists

```python
knowledge_get(id="existing-id")      # does it exist?
knowledge_edit(id="existing-id", text="Updated content")  # update in place
# Instead of: remove + add (two calls)
```

### 4. To REVIEW → list what's in the base

```python
knowledge_list()
```

### 5. To REMOVE stale facts

```python
knowledge_remove(id="outdated-entry")
```

## Command Reference (Rust CLI)

```bash
# Build / rebuild embeddings
know build                     # from knowledge.json → embeddings.bin
know build -k path/to/knowledge.json -o custom.bin

# Search (semantic)
know search "what model do we use"     # top 1 result
know search "query" -n 3 -t 0.3       # top 3, lower threshold
know search "query" --json            # machine-parseable JSON

# Get (fast — by id, no model)
know get "arch-model"                 # show full text
know get "arch-model" --json          # JSON format

# Write
know add "unique-id" "The fact to persist"
know edit "existing-id" "Updated text"
know list
know remove "unique-id"
```

## Command Reference (Python CLI)

```bash
python -m hermes_know build
python -m hermes_know search "query" -n 3 --json
python -m hermes_know get "arch-model"
python -m hermes_know add "id" "text"
python -m hermes_know edit "id" "new text"
python -m hermes_know list
python -m hermes_know remove "id"
```

## JSON response format (all tools)

```json
{
  "success": true,
  "id": "entry-id",
  "text": "Full entry text",
  "embeddings_rebuilt": true,
  "entries_count": 12,
  "message": "Edited [entry-id] (12 entries total)."
}
```

Error response:

```json
{
  "success": false,
  "error": "Entry 'nonexistent' not found."
}
```

## Important notes

- Both Rust and Python read/write the same `knowledge.json`
- Rust uses `embeddings.bin` (bincode), Python uses `embeddings.pkl` (pickle)
- After `add`/`edit`/`remove`, embeddings are rebuilt **automatically**
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims)
- Search threshold default: 0.45 (lower = more results, higher = stricter)
- `knowledge_get` is instant — no model load. Use it instead of search when you know the id.
