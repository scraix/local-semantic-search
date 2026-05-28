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
    long_about = "know build  — generate embeddings from knowledge.json\nknow search — find best match for a query"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate embeddings from knowledge.json
    Build {
        /// Path to knowledge.json (default: ./knowledge.json)
        #[arg(short, long, default_value = "knowledge.json")]
        knowledge: String,
        /// Output path for embeddings (default: ./embeddings.bin)
        #[arg(short, long, default_value = "embeddings.bin")]
        output: String,
    },
    /// Search project knowledge for best match(es)
    Search {
        /// The search query
        query: String,
        /// Path to embeddings.bin (default: ./embeddings.bin)
        #[arg(short, long, default_value = "embeddings.bin")]
        embeddings: String,
        /// Minimum similarity score (0.0–1.0, default: 0.45)
        #[arg(short, long, default_value_t = 0.45)]
        threshold: f32,
        /// Number of top results to return (default: 1)
        #[arg(short = 'n', long, default_value_t = 1)]
        top: usize,
        /// Output as JSON (for machine parsing)
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { knowledge, output } => {
            let entries = storage::read_knowledge_json(&knowledge)
                .with_context(|| format!("Failed to read {}", knowledge))?;

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

            storage::write_embeddings(&output, &vectors)
                .with_context(|| format!("Failed to write {}", output))?;

            println!(
                "{} written ({} entries, {} dimensions)",
                output,
                vectors.len(),
                vectors[0].2.len()
            );
        }
        Commands::Search {
            query,
            embeddings,
            threshold,
            top,
            json,
        } => {
            let embeddings_data = storage::read_embeddings(&embeddings)
                .with_context(|| format!("Failed to read {}", embeddings))?;

            if embeddings_data.is_empty() {
                eprintln!("Embeddings file is empty. Run 'know build' first.");
                return Ok(());
            }

            let query_vec = embed::embed_query(&query).context("Query embedding failed")?;

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
        }
    }

    Ok(())
}
