# SKILL: know — local semantic search

**Version:** 5.0.0
**Purpose:** Persist and retrieve project knowledge across sessions
**Files:** `knowledge.json` (source), `embeddings.bin` (Rust), `embeddings.pkl` (Python)

## Available commands

| Command | Rust CLI | Python / Hermes tool |
|---|---|---|
| Search | `know search <query>` | `knowledge_search(query, tags=[])` |
| Get by id | `know get <id>` | `knowledge_get(id)` |
| Add | `know add <id> <text> [--tags a,b]` | `knowledge_add(id, text, tags=[])` |
| Edit | `know edit <id> <text> [--tags a,b]` | `knowledge_edit(id, text, tags=[])` |
| Remove | `know remove <id>` | `knowledge_remove(id)` |
| List | `know list [--tags a]` | `knowledge_list(tags=[])` |
| Build | `know build` | `knowledge_build()` |

## Tags (v5.0.0)

Each entry can have optional categorisation tags (e.g. `arch`, `config`,
`workflow`, `design`). Tags are set on add, can be changed on edit, and can
be used as a filter on search and list.

Using tags in search improves relevance and reduces noise — especially as
the knowledge base grows.

Tag filter is `AND` — entry must have ALL specified tags to match.

## Agent workflow — PROACTIVE knowledge base usage

The agent MUST follow this workflow automatically, without being asked by
the user:

### Before answering a project question
1. Call `knowledge_search(query)` (or `knowledge_get(id)` if id is known)
2. If results found → use them to inform the answer
3. If no results → state honestly that no info is in the knowledge base

### When learning something new
1. Call `knowledge_add(id, text, tags=[...])` proactively when discovering:
   - Architecture decisions and rationale
   - Configuration quirks or environment setup details
   - Design tradeoffs and why choices were made
   - Important conventions or patterns
   - Any fact the user mentions that seems worth persisting
2. Use meaningful, short IDs (e.g. `arch-cli-design`, `config-env-var`)
3. Categorise with tags (e.g. `arch`, `config`, `design`, `troubleshoot`)

### Before adding
1. Call `knowledge_get(id)` to check if entry already exists
2. If exists → call `knowledge_edit(id, new_text)` instead of add+remove

### Maintenance
- Periodically call `knowledge_list()` to survey what's stored
- `knowledge_list(tags=["arch"])` to see entries in a specific category

### End-of-session: update AGENTS.md

At the end of each session (or when learning something important that isn't
already in AGENTS.md), update `AGENTS.md` in the project root.

AGENTS.md is auto-read by Hermes on session start — no need to tell the user
to read it.

Write in Caveman Compression format — stripped grammar, minimal tokens:
```
## fact-name
- key: value
- another: thing
```
No prose, no water. Just facts that will matter next session.

Commit AGENTS.md changes to git.

## Why use this instead of Hermes memory/history

- Persists across sessions — not lost on reset
- Semantic search finds conceptually related facts, not just keyword matches
- Can be shared with the Rust CLI — no Python dependency required
- Tags allow categorisation and filtered search as the KB grows
