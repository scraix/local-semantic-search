use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod embed;
mod search;
mod storage;

#[derive(Parser)]
#[command(
    name = "know",
    version,
    about = "Local semantic search for project knowledge bases",
    long_about = concat!(
        "know build   — generate embeddings from knowledge.json\n",
        "know search  — semantic search (--tags to filter)\n",
        "know get     — get entry by id\n",
        "know add     — add a new entry (--tags arch,cli)\n",
        "know edit    — update an entry (--tags new,tags)\n",
        "know list    — list entries (--tags to filter)\n",
        "know remove  — remove an entry by id",
    )
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate embeddings from knowledge.json
    Build {
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
    },
    /// Search project knowledge for best match(es)
    Search {
        query: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        embeddings: String,
        #[arg(short, long, default_value_t = 0.45)]
        threshold: f32,
        #[arg(short = 'n', long, default_value_t = 1)]
        top: usize,
        #[arg(long)]
        json: bool,
        /// Filter by tags (comma-separated, e.g. --tags arch,config)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
    },
    /// Get a single entry by id (fast — no model load)
    Get {
        id: String,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(long)]
        json: bool,
    },
    /// Add a new entry and rebuild embeddings
    Add {
        id: String,
        text: String,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
        /// Comma-separated tags, e.g. --tags arch,cli
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
    },
    /// Update an existing entry by id and rebuild embeddings.
    ///
    /// If TEXT is omitted, only tags are updated (if --tags provided).
    /// If --tags is omitted, tags stay unchanged.
    Edit {
        id: String,
        #[arg(required = false)]
        text: Option<String>,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
        /// Comma-separated tags, e.g. --tags arch,config. Omit to keep current.
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,
    },
    /// List all entries in knowledge.json
    List {
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        /// Filter by tags (comma-separated, e.g. --tags arch)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
    },
    /// Remove an entry by id and rebuild embeddings
    Remove {
        id: String,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { knowledge, output } => cmd_build(&knowledge, &output)?,
        Commands::Search {
            query,
            embeddings,
            threshold,
            top,
            json,
            tags,
        } => cmd_search(&query, &embeddings, threshold, top, json, &tags)?,
        Commands::Get { id, knowledge, json } => cmd_get(&id, &knowledge, json)?,
        Commands::Add {
            id,
            text,
            knowledge,
            output,
            tags,
        } => cmd_add(&id, &text, &tags, &knowledge, &output)?,
        Commands::Edit {
            id,
            text,
            tags,
            knowledge,
            output,
        } => cmd_edit(
            &id,
            text.as_deref(),
            tags.as_deref(),
            &knowledge,
            &output,
        )?,
        Commands::List { knowledge, tags } => cmd_list(&knowledge, &tags)?,
        Commands::Remove {
            id,
            knowledge,
            output,
        } => cmd_remove(&id, &knowledge, &output)?,
    }

    Ok(())
}

// ── helpers ────────────────────────────────────────────────────────────────

fn rebuild_embeddings(
    entries: &[storage::KnowledgeEntry],
    output_path: &str,
) -> Result<Vec<(String, String, Vec<String>, Vec<f32>)>> {
    let text_entries: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.id.as_str(), e.text.as_str()))
        .collect();
    let vectors = embed::embed_texts(&text_entries)
        .context("Embeddings generation failed")?;

    // Zip vectors back with full KnowledgeEntry data (including tags)
    let result: Vec<(String, String, Vec<String>, Vec<f32>)> = entries
        .iter()
        .zip(vectors.iter())
        .map(|(e, (_, _, vec))| (e.id.clone(), e.text.clone(), e.tags.clone(), vec.clone()))
        .collect();

    storage::write_embeddings(output_path, &result)
        .with_context(|| format!("Failed to write {}", output_path))?;

    Ok(result)
}

// ── Build ────────────────────────────────────────────────────────────────────

fn cmd_build(knowledge_path: &str, output_path: &str) -> Result<()> {
    let entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    if entries.is_empty() {
        eprintln!("knowledge.json is empty. Nothing to embed.");
        return Ok(());
    }

    eprintln!("Found {} entries. Generating embeddings...", entries.len());
    let vectors = rebuild_embeddings(&entries, output_path)?;

    println!(
        "{} written ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].3.len()
    );
    Ok(())
}

// ── Search ───────────────────────────────────────────────────────────────────

fn cmd_search(
    query: &str,
    embeddings_path: &str,
    threshold: f32,
    top: usize,
    json: bool,
    tags: &[String],
) -> Result<()> {
    let embeddings_data = storage::read_embeddings(embeddings_path)
        .with_context(|| format!("Failed to read {}", embeddings_path))?;

    if embeddings_data.is_empty() {
        eprintln!("Embeddings file is empty. Run 'know build' first.");
        return Ok(());
    }

    // Filter by tags if specified
    let filtered: Vec<_> = if tags.is_empty() {
        embeddings_data.iter().collect()
    } else {
        embeddings_data
            .iter()
            .filter(|e| storage::matches_tags(&e.tags, tags))
            .collect()
    };

    if filtered.is_empty() {
        let msg = if tags.is_empty() {
            "Embeddings file is empty."
        } else {
            "No entries match the specified tags."
        };
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "success": false,
                    "error": msg,
                }))
                .unwrap()
            );
        } else {
            eprintln!("{}", msg);
        }
        return Ok(());
    }

    let query_vec = embed::embed_query(query).context("Query embedding failed")?;
    let results = search::find_top(&query_vec, &filtered, threshold, top);

    if json {
        let output: Vec<serde_json::Value> = results
            .iter()
            .map(|(entry, score)| {
                serde_json::json!({
                    "id": entry.id,
                    "text": entry.text,
                    "tags": entry.tags,
                    "score": format!("{:.4}", score),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "success": !results.is_empty(),
                "query": query,
                "results": output,
                "count": results.len(),
            }))
            .unwrap()
        );
    } else if results.is_empty() {
        let tag_info = if !tags.is_empty() {
            format!(" with tags [{}]", tags.join(","))
        } else {
            String::new()
        };
        println!(
            "[KNOW]: No relevant information found. Searched {} entries{}, threshold {:.2}, top {}. Try lowering threshold or changing query.",
            filtered.len(),
            tag_info,
            threshold,
            top
        );
    } else {
        for (entry, score) in &results {
            let tag = if !entry.id.is_empty() {
                format!(" [{}]", entry.id)
            } else {
                String::new()
            };
            let tags_str = if entry.tags.is_empty() {
                String::new()
            } else {
                format!(" tags:({})", entry.tags.join(","))
            };
            println!(
                "[KNOW{}]: {}{} (score: {:.3})",
                tag, entry.text, tags_str, score
            );
        }
    }
    Ok(())
}

// ── Get ─────────────────────────────────────────────────────────────────────

fn cmd_get(id: &str, knowledge_path: &str, json: bool) -> Result<()> {
    let entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    match entries.iter().find(|e| e.id == id) {
        None => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "success": false,
                        "error": format!("Entry '{}' not found.", id),
                    }))
                    .unwrap()
                );
            } else {
                eprintln!("Entry '{}' not found.", id);
            }
        }
        Some(entry) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "success": true,
                        "id": entry.id,
                        "text": entry.text,
                        "tags": entry.tags,
                    }))
                    .unwrap()
                );
            } else {
                let tags_line = if entry.tags.is_empty() {
                    String::new()
                } else {
                    format!("\ntags: {}", entry.tags.join(", "))
                };
                println!("[{}]{}\n{}", entry.id, tags_line, entry.text);
            }
        }
    }
    Ok(())
}

// ── Add ─────────────────────────────────────────────────────────────────────

fn cmd_add(
    id: &str,
    text: &str,
    tags: &[String],
    knowledge_path: &str,
    output_path: &str,
) -> Result<()> {
    let mut entries = if std::fs::metadata(knowledge_path).is_ok() {
        storage::read_knowledge_json(knowledge_path)
            .with_context(|| format!("Failed to read {}", knowledge_path))?
    } else {
        Vec::new()
    };

    if entries.iter().any(|e| e.id == id) {
        eprintln!(
            "Entry with id '{}' already exists. Use 'know edit' to update.",
            id
        );
        return Ok(());
    }

    entries.push(storage::KnowledgeEntry {
        id: id.to_string(),
        text: text.to_string(),
        tags: tags.to_vec(),
    });

    storage::write_knowledge_json(knowledge_path, &entries)?;
    println!("Added: [{}]", id);
    if !tags.is_empty() {
        println!("tags: {}", tags.join(", "));
    }

    eprintln!("Rebuilding embeddings...");
    let vectors = rebuild_embeddings(&entries, output_path)?;
    println!(
        "{} updated ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].3.len()
    );
    Ok(())
}

// ── Edit ────────────────────────────────────────────────────────────────────

fn cmd_edit(
    id: &str,
    text: Option<&str>,
    tags: Option<&[String]>,
    knowledge_path: &str,
    output_path: &str,
) -> Result<()> {
    let mut entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    let mut found = false;
    for entry in entries.iter_mut() {
        if entry.id == id {
            if let Some(t) = text {
                entry.text = t.to_string();
            }
            if let Some(tags_list) = tags {
                entry.tags = tags_list.to_vec();
            }
            found = true;
            break;
        }
    }

    if !found {
        eprintln!("Entry '{}' not found.", id);
        return Ok(());
    }

    storage::write_knowledge_json(knowledge_path, &entries)?;
    println!("Edited: [{}]", id);

    eprintln!("Rebuilding embeddings...");
    let vectors = rebuild_embeddings(&entries, output_path)?;
    println!(
        "{} updated ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].3.len()
    );
    Ok(())
}

// ── List ─────────────────────────────────────────────────────────────────────

fn cmd_list(knowledge_path: &str, tags: &[String]) -> Result<()> {
    let entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    let filtered: Vec<_> = entries
        .iter()
        .filter(|e| storage::matches_tags(&e.tags, tags))
        .collect();

    if filtered.is_empty() {
        if tags.is_empty() {
            println!("No entries in {}.", knowledge_path);
        } else {
            println!(
                "No entries matching tags [{}] in {}.",
                tags.join(", "),
                knowledge_path
            );
        }
        return Ok(());
    }

    println!(
        "{} entr{} in {}:\n",
        filtered.len(),
        if filtered.len() == 1 { "y" } else { "ies" },
        knowledge_path,
    );
    for (i, entry) in filtered.iter().enumerate() {
        let preview: String = if entry.text.len() > 70 {
            format!("{}...", &entry.text[..67])
        } else {
            entry.text.clone()
        };
        let tags_str = if entry.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", entry.tags.join(","))
        };
        println!("  {}. [{}]{} {}", i + 1, entry.id, tags_str, preview);
    }
    Ok(())
}

// ── Remove ───────────────────────────────────────────────────────────────────

fn cmd_remove(id: &str, knowledge_path: &str, output_path: &str) -> Result<()> {
    let mut entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    let before = entries.len();
    entries.retain(|e| e.id != id);

    if entries.len() == before {
        eprintln!("Entry with id '{}' not found.", id);
        return Ok(());
    }

    storage::write_knowledge_json(knowledge_path, &entries)?;
    println!("Removed: [{}]", id);

    if entries.is_empty() {
        println!("No entries left — embeddings deleted.");
        let _ = std::fs::remove_file(output_path);
    } else {
        eprintln!("Rebuilding embeddings...");
        let vectors = rebuild_embeddings(&entries, output_path)?;
        println!(
            "{} updated ({} entries, {} dimensions)",
            output_path,
            vectors.len(),
            vectors[0].3.len()
        );
    }
    Ok(())
}
