# know — Local Semantic Search

Fast, local semantic search for project knowledge bases. Designed for AI
agents (Hermes) and developers who want to keep project-specific facts in a
simple JSON file and query them via cosine similarity embeddings.

## How it works

1. Create a `knowledge.json` in the project root (array of `{id, text}`)
2. Run `know build` once — generates embeddings
3. Search with natural language queries

## Quick start

### Rust CLI (fast)

```bash
# Build embeddings
know build

# Search
know search "how does the search algorithm work"

# Structured JSON output (for AI agents)
know search --json "what model" | jq .

# Top-3 with lower threshold
know search -n 3 -t 0.3 "error handling"
```

### Python tool (Hermes Agent native)

```bash
# Install
uv pip install .

# Or run directly
python -m hermes_know build
python -m hermes_know search "your question" -n 3
python -m hermes_know search --json "question"
```

```python
# Use as a Hermes Agent tool
from hermes_know.tool import knowledge_search

result = knowledge_search("architecture decisions", top_n=3)
# Returns structured JSON
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
  { "id": "arch-search", "text": "Search uses cosine similarity..." },
  { "id": "arch-storage", "text": "Embeddings use bincode..." }
]
```

- `id` — short identifier (optional, can be empty string)
- `text` — the factual content (English recommended)

## CLI options

### Build
| Flag | Default | Description |
|------|---------|-------------|
| `-k, --knowledge` | `knowledge.json` | Input JSON path |
| `-o, --output` | `embeddings.bin` (Rust) / `embeddings.pkl` (Python) | Output path |

### Search
| Flag | Default | Description |
|------|---------|-------------|
| `query` | — | Natural language query (positional) |
| `-e, --embeddings` | `embeddings.bin` / `embeddings.pkl` | Embeddings file |
| `-t, --threshold` | `0.45` | Min similarity score (0.0–1.0) |
| `-n, --top` | `1` (Rust) / `3` (Python tool) | Number of results |
| `--json` | — | Structured JSON output |

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

## Hermes Agent integration

This project includes a **Hermes skill** (SKILL.md) that tells the agent how
to use `know` automatically. The Python package also registers native tools:

- `knowledge_search(query, top_n, threshold)` — semantic search
- `knowledge_build()` — rebuild embeddings from knowledge.json

Set environment variables to customise paths:

```bash
export KNOWLEDGE_JSON=/path/to/knowledge.json
export EMBEDDINGS_FILE=/path/to/embeddings.pkl
```

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
├── READ.md                 # This file
└── .gitignore
```

## License

MIT
