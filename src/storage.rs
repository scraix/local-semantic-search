use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single knowledge entry.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Stored embedding data (precomputed).
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub vector: Vec<f32>,
}

/// Check whether an entry's tags contain ALL the required tags.
pub fn matches_tags(entry_tags: &[String], required: &[String]) -> bool {
    if required.is_empty() {
        return true;
    }
    required.iter().all(|r| entry_tags.iter().any(|t| t == r))
}

/// Read knowledge.json from disk.
pub fn read_knowledge_json<P: AsRef<Path>>(path: P) -> Result<Vec<KnowledgeEntry>> {
    let file = std::fs::File::open(path.as_ref())
        .with_context(|| format!("File not found: {}", path.as_ref().display()))?;
    let reader = std::io::BufReader::new(file);
    let entries: Vec<KnowledgeEntry> = serde_json::from_reader(reader)
        .context("knowledge.json format error — expected array of {id, text}")?;
    Ok(entries)
}

/// Write entries to knowledge.json.
pub fn write_knowledge_json<P: AsRef<Path>>(path: P, entries: &[KnowledgeEntry]) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .with_context(|| format!("Failed to create {}", path.as_ref().display()))?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, entries)
        .context("Failed to write knowledge.json")?;
    Ok(())
}

/// Write embeddings to disk using bincode.
pub fn write_embeddings<P: AsRef<Path>>(
    path: P,
    entries: &[(String, String, Vec<String>, Vec<f32>)],
) -> Result<()> {
    let data: Vec<EmbeddingEntry> = entries
        .iter()
        .map(|(id, text, tags, vector)| EmbeddingEntry {
            id: id.clone(),
            text: text.clone(),
            tags: tags.clone(),
            vector: vector.clone(),
        })
        .collect();

    let encoded = bincode::serialize(&data).context("Failed to serialize embeddings")?;
    std::fs::write(path.as_ref(), encoded)
        .with_context(|| format!("Failed to write {}", path.as_ref().display()))?;
    Ok(())
}

/// Read embeddings from bincode file.
pub fn read_embeddings<P: AsRef<Path>>(path: P) -> Result<Vec<EmbeddingEntry>> {
    let encoded = std::fs::read(path.as_ref())
        .with_context(|| format!("File not found: {}", path.as_ref().display()))?;
    let data: Vec<EmbeddingEntry> =
        bincode::deserialize(&encoded).context("embeddings.bin format error")?;
    Ok(data)
}
