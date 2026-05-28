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
    long_about = "know build   — generate embeddings from knowledge.json\nknow search  — find best match for a query\nknow get     — get a single entry by id\nknow add     — add a new entry\nknow edit    — update an existing entry\nknow list    — list all entries\nknow remove  — remove an entry by id"
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
    },
    /// Get a single entry by id (fast — no model load)
    Get {
        /// Id of the entry to retrieve
        id: String,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        /// Output as JSON
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
    },
    /// Update an existing entry by id and rebuild embeddings
    Edit {
        /// Id of the entry to update
        id: String,
        /// New text content
        text: String,
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
    },
    /// List all entries in knowledge.json
    List {
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
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
        } => cmd_search(&query, &embeddings, threshold, top, json)?,
        Commands::Get { id, knowledge, json } => cmd_get(&id, &knowledge, json)?,
        Commands::Add {
            id,
            text,
            knowledge,
            output,
        } => cmd_add(&id, &text, &knowledge, &output)?,
        Commands::Edit {
            id,
            text,
            knowledge,
            output,
        } => cmd_edit(&id, &text, &knowledge, &output)?,
        Commands::List { knowledge } => cmd_list(&knowledge)?,
        Commands::Remove {
            id,
            knowledge,
            output,
        } => cmd_remove(&id, &knowledge, &output)?,
    }

    Ok(())
}

// ── Build ──────────────────────────────────────────────────────────────

fn cmd_build(knowledge_path: &str, output_path: &str) -> Result<()> {
    let entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    if entries.is_empty() {
        eprintln!("knowledge.json is empty. Nothing to embed.");
        return Ok(());
    }

    eprintln!("Found {} entries. Generating embeddings...", entries.len());
    let text_entries: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.id.as_str(), e.text.as_str()))
        .collect();

    let vectors = embed::embed_texts(&text_entries)
        .context("Embeddings generation failed")?;

    storage::write_embeddings(output_path, &vectors)
        .with_context(|| format!("Failed to write {}", output_path))?;

    println!(
        "{} written ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].2.len()
    );
    Ok(())
}

// ── Search ─────────────────────────────────────────────────────────────

fn cmd_search(
    query: &str,
    embeddings_path: &str,
    threshold: f32,
    top: usize,
    json: bool,
) -> Result<()> {
    let embeddings_data = storage::read_embeddings(embeddings_path)
        .with_context(|| format!("Failed to read {}", embeddings_path))?;

    if embeddings_data.is_empty() {
        eprintln!("Embeddings file is empty. Run 'know build' first.");
        return Ok(());
    }

    let query_vec = embed::embed_query(query).context("Query embedding failed")?;
    let results = search::find_top(&query_vec, &embeddings_data, threshold, top);

    if json {
        let output: Vec<serde_json::Value> = results
            .iter()
            .map(|(entry, score)| {
                serde_json::json!({
                    "id": entry.id,
                    "text": entry.text,
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
        println!(
            "[KNOW]: No relevant information found (below {:.2} threshold).",
            threshold
        );
    } else {
        for (entry, score) in &results {
            let tag = if !entry.id.is_empty() {
                format!(" [{}]", entry.id)
            } else {
                String::new()
            };
            println!("[KNOW{}]: {} (score: {:.3})", tag, entry.text, score);
        }
    }
    Ok(())
}

// ── Get ────────────────────────────────────────────────────────────────

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
                    }))
                    .unwrap()
                );
            } else {
                println!("[{}]\n{}", entry.id, entry.text);
            }
        }
    }
    Ok(())
}

// ── Add ────────────────────────────────────────────────────────────────

fn cmd_add(id: &str, text: &str, knowledge_path: &str, output_path: &str) -> Result<()> {
    let mut entries = if std::fs::metadata(knowledge_path).is_ok() {
        storage::read_knowledge_json(knowledge_path)
            .with_context(|| format!("Failed to read {}", knowledge_path))?
    } else {
        Vec::new()
    };

    if entries.iter().any(|e| e.id == id) {
        eprintln!(
            "Entry with id '{}' already exists. Use 'know edit' to update, or 'know remove' first.",
            id
        );
        return Ok(());
    }

    entries.push(storage::KnowledgeEntry {
        id: id.to_string(),
        text: text.to_string(),
    });

    storage::write_knowledge_json(knowledge_path, &entries)?;
    println!("Added: [{}]", id);
    println!("Knowledge: {}", knowledge_path);

    eprintln!("Rebuilding embeddings...");
    let text_entries: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.id.as_str(), e.text.as_str()))
        .collect();
    let vectors = embed::embed_texts(&text_entries)
        .context("Embeddings generation failed")?;
    storage::write_embeddings(output_path, &vectors)
        .with_context(|| format!("Failed to write {}", output_path))?;
    println!(
        "{} updated ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].2.len()
    );
    Ok(())
}

// ── Edit ───────────────────────────────────────────────────────────────

fn cmd_edit(id: &str, text: &str, knowledge_path: &str, output_path: &str) -> Result<()> {
    let mut entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    let before = entries.len();
    // Update in-place
    for entry in entries.iter_mut() {
        if entry.id == id {
            entry.text = text.to_string();
            break;
        }
    }

    if entries.len() != before {
        // We didn't find it — length wouldn't change
    }

    // Check if actually found by looking for it
    let found = entries.iter().any(|e| e.id == id);
    if !found {
        eprintln!("Entry '{}' not found.", id);
        return Ok(());
    }

    storage::write_knowledge_json(knowledge_path, &entries)?;
    println!("Edited: [{}]", id);

    eprintln!("Rebuilding embeddings...");
    let text_entries: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.id.as_str(), e.text.as_str()))
        .collect();
    let vectors = embed::embed_texts(&text_entries)
        .context("Embeddings generation failed")?;
    storage::write_embeddings(output_path, &vectors)
        .with_context(|| format!("Failed to write {}", output_path))?;
    println!(
        "{} updated ({} entries, {} dimensions)",
        output_path,
        vectors.len(),
        vectors[0].2.len()
    );
    Ok(())
}

// ── List ───────────────────────────────────────────────────────────────

fn cmd_list(knowledge_path: &str) -> Result<()> {
    let entries = storage::read_knowledge_json(knowledge_path)
        .with_context(|| format!("Failed to read {}", knowledge_path))?;

    if entries.is_empty() {
        println!("No entries in {}.", knowledge_path);
        return Ok(());
    }

    println!("{} entries in {}:\n", entries.len(), knowledge_path);
    for (i, entry) in entries.iter().enumerate() {
        let preview: String = if entry.text.len() > 80 {
            format!("{}...", &entry.text[..77])
        } else {
            entry.text.clone()
        };
        println!("  {}. [{}] {}", i + 1, entry.id, preview);
    }
    Ok(())
}

// ── Remove ─────────────────────────────────────────────────────────────

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
        let text_entries: Vec<(&str, &str)> = entries
            .iter()
            .map(|e| (e.id.as_str(), e.text.as_str()))
            .collect();
        let vectors = embed::embed_texts(&text_entries)
            .context("Embeddings generation failed")?;
        storage::write_embeddings(output_path, &vectors)
            .with_context(|| format!("Failed to write {}", output_path))?;
        println!(
            "{} updated ({} entries, {} dimensions)",
            output_path,
            vectors.len(),
            vectors[0].2.len()
        );
    }
    Ok(())
}
