# AGENTS.md — project context for Hermes Agent

This file is auto-read on session start. Written in Caveman Compression™ —
stripped grammar, minimal tokens, only facts that matter. If you understand
this format, you understand the project.

## project: local-semantic-search (know)

- Rust + Python dual CLI semantic search tool
- knowledge.json holds entries {id, text, tags}
- Rust: fastembed 4.x, bincode (.bin), 384d
- Python: fastembed >=0.4, pickle (.pkl), 384d
- Model: paraphrase-multilingual-MiniLM-L12-v2
- Search: cosine similarity, threshold 0.45 default

## CRUD commands (7 tools)

- know build / knowledge_build() — rebuild embeddings
- know search / knowledge_search(query, tags=[]) — sem search with tag filter
- know get / knowledge_get(id) — by id, fast no model
- know add / knowledge_add(id, text, tags=[]) — add + auto-build
- know edit / knowledge_edit(id, text, tags=[]) — update + auto-build
- know remove / knowledge_remove(id) — delete + auto-build
- know list / knowledge_list(tags=[]) — list with optional tag filter

## tags (v5.0.0)

- Optional per-entry categorisation
- AND filter: entry must have ALL specified tags
- Common: arch, config, design, workflow, cli, search, troubleshoot
- --tags arch,cli on add/edit/search/list

## usage

- Agent workflow: search first → if found use it → if not found say so
- Learn something new → knowledge_add(id, text, tags=[])
- Before add → knowledge_get(id) to check exists → if yes, knowledge_edit
- Periodically knowledge_list() to survey KB
- knowledge_remove() for stale facts

## file layout

- Cargo.toml — Rust
- pyproject.toml — Python
- src/main.rs — Rust CLI dispatch + CRUD commands
- src/embed.rs — fastembed wrapping
- src/search.rs — cosine similarity
- src/storage.rs — JSON/bincode I/O, KnowledgeEntry/EmbeddingEntry structs
- hermes_know/core.py — Python embedding/search/storage
- hermes_know/cli.py — Python CLI
- hermes_know/tool.py — Hermes tool registration
- knowledge.json — source of truth (12 entries tagged)
- SKILL.md — Hermes skill v5.0.0
- embeddings.bin — Rust binary embeddings
- embeddings.pkl — Python pickle embeddings
- README.md — docs
- .venv/ — Python venv

## quirks

- Rust and Python share knowledge.json format
- embeddings are independent — rebuild each after knowledge.json changes
- edit command: text optional, update only tags with --tags
- user runs this patched Hermes v0.14.0 (not stock) from ~/Projects/hermes-dev/hermes-patches
- user prefers uv/pipx over pip
- github: scraix, email: nixso959@gmail.com
- user Lithuanian/English, prefers English for tech
- user says "just do it if you think it's good" — proactive action expected
- host: Linux 7.0.10-1-cachyos, cwd: /home/nix/Projects/tmp/local-semantic-search
