# know — Local Semantic Search

Fast, local semantic search for project knowledge bases. Designed for AI
agents (Hermes) and developers who want to keep project-specific facts in a
simple JSON file and query them via cosine similarity embeddings.

## Quick start

```bash
# Build embeddings
know build

# Search
know search "how does the search algorithm work"

# List all entries
know list

# Add a new fact
know add arch-my-decision "We chose X because..." --tags arch,design

# Get a specific entry by id
know get arch-search

# Update an entry
know edit arch-search "Updated content" --tags arch,search

# Remove an entry
know remove outdated-entry

# Search only within a tag category
know search "error handling" --tags config

# Structured JSON output (for AI agents)
know search --json "what model" | jq .

# Top-3 with lower threshold
know search -n 3 -t 0.3 "error handling"
```

### Python equivalent

```bash
python -m hermes_know build
python -m hermes_know add my-id "content" --tags arch,cli
python -m hermes_know search "question" -n 3 --tags arch
python -m hermes_know list --tags cli
python -m hermes_know get my-id [--json]
python -m hermes_know edit my-id "new text" [--tags new,tags]
python -m hermes_know remove my-id
```

```python
# Use as a Hermes Agent tool
from hermes_know.tool import knowledge_search, knowledge_add, knowledge_get

# Search with tag filter
result = knowledge_search("architecture decisions", top_n=3, tags=["arch"])

# Get by id (fast, no model load)
result = knowledge_get("arch-search")
```

## Installation

### Rust (requires Rust toolchain)

```bash
cargo build --release
# Binary at target/release/know
```

### Python

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Or install directly from GitHub:

```bash
uv pip install git+https://github.com/scraix/local-semantic-search
```

## knowledge.json format

```json
[
  { "id": "arch-search", "text": "Search uses cosine similarity...", "tags": ["arch", "search"] },
  { "id": "config-threshold", "text": "Default threshold is 0.45...", "tags": ["config"] }
]
```

- `id` — unique identifier (for get, edit, remove)
- `text` — the factual content (English recommended)
- `tags` — optional array of categorisation tags (added in v5.0.0)

## CLI commands

| Command | Description |
|---------|-------------|
| `build` | Generate embeddings from knowledge.json |
| `search <query>` | Semantic search with optional `--tags` filter |
| `get <id>` | Get a single entry by id (fast, no embedding) |
| `add <id> <text>` | Add a new entry with optional `--tags` |
| `edit <id> [text]` | Update text and/or `--tags` |
| `remove <id>` | Delete an entry |
| `list` | List all entries, optional `--tags` filter |

### Build flags
| Flag | Default | Description |
|------|---------|-------------|
| `-k, --knowledge` | `knowledge.json` | Input JSON path |
| `-o, --output` | `embeddings.bin` / `embeddings.pkl` | Output path |

### Search flags
| Flag | Default | Description |
|------|---------|-------------|
| `query` | — | Natural language query |
| `-e, --embeddings` | `embeddings.bin` / `embeddings.pkl` | Embeddings file |
| `-t, --threshold` | `0.45` | Min similarity (0.0–1.0) |
| `-n, --top` | `1` | Number of results |
| `--tags` | — | Filter by tag (e.g. `--tags arch,cli`) |
| `--json` | — | Structured JSON output |

### Add / Edit flags
| Flag | Description |
|------|-------------|
| `--tags arch,cli` | Comma-separated tags on add; replace tags on edit |

### List flags
| Flag | Description |
|------|-------------|
| `--tags arch` | Filter by tag(s) |

## Tags (v5.0.0)

Each entry can have optional tags for categorisation. Tags are `AND` —
when filtering, an entry must have **all** specified tags.

Common tag categories used in practice:
- `arch` — architecture decisions and rationale
- `config` — configuration quirks and environment setup
- `design` — design patterns and tradeoffs
- `workflow` — development workflows and tooling
- `cli` — CLI commands and usage
- `search` — search algorithm and threshold tuning
- `troubleshoot` — known issues and fixes

## Hermes Agent integration

This project includes a **Hermes skill** (SKILL.md) that tells the agent how
to use `know` automatically. The Python package also registers 7 native tools:

| Tool | Description |
|------|-------------|
| `knowledge_search(query, top_n, threshold, tags)` | Semantic search |
| `knowledge_get(id)` | Get by id (fast, no model) |
| `knowledge_add(id, text, tags)` | Add entry + auto-build |
| `knowledge_edit(id, text, tags)` | Update entry + auto-build |
| `knowledge_list(tags)` | List entries |
| `knowledge_remove(id)` | Remove entry + auto-build |
| `knowledge_build()` | Rebuild embeddings |

Set environment variables to customise paths:

```bash
export KNOWLEDGE_JSON=/path/to/knowledge.json
export EMBEDDINGS_FILE=/path/to/embeddings.pkl
```

## Technical stack

| Component | Rust | Python |
|-----------|------|--------|
| Embedding | `fastembed 4.x` | `fastembed >=0.4` |
| Model | `paraphrase-multilingual-MiniLM-L12-v2` | same |
| Dimensions | 384 | 384 |
| Serialization | bincode (`.bin`) | pickle (`.pkl`) |
| Similarity | cosine | cosine (numpy) |

Both read the same `knowledge.json` format. Regenerate embeddings per
language using the appropriate `build` command.

## Project structure

```
├── Cargo.toml              # Rust package
├── src/                    # Rust source (main, embed, search, storage)
├── pyproject.toml          # Python package
├── hermes_know/            # Python package
│   ├── __init__.py
│   ├── core.py             # Embedding + search + storage
│   ├── cli.py              # CLI interface
│   └── tool.py             # Hermes Agent tool registration
├── knowledge.json          # Your knowledge base (edit this)
├── SKILL.md                # Hermes agent skill
├── README.md               # This file
└── .gitignore
```

## License

MIT
