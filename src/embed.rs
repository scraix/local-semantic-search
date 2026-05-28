use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Embed multiple texts at once (for build).
/// Each entry: (id, text). Returns vec of (id, text, vector).
pub fn embed_texts<'a>(entries: &[(&'a str, &'a str)]) -> Result<Vec<(String, String, Vec<f32>)>> {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::ParaphraseMLMiniLML12V2)
            .with_show_download_progress(true)
            .with_cache_dir(get_cache_dir()),
    )
    .context("Failed to load embedding model")?;

    let texts: Vec<&str> = entries.iter().map(|(_, text)| *text).collect();
    let embeddings = model
        .embed(texts, None)
        .context("Embedding inference failed")?;

    let result: Vec<(String, String, Vec<f32>)> = entries
        .iter()
        .zip(embeddings.into_iter())
        .map(|((id, text), vec)| (id.to_string(), text.to_string(), vec))
        .collect();

    Ok(result)
}

/// Embed a single query string (for search).
/// Returns the vector.
pub fn embed_query(query: &str) -> Result<Vec<f32>> {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::ParaphraseMLMiniLML12V2)
            .with_show_download_progress(false)
            .with_cache_dir(get_cache_dir()),
    )
    .context("Failed to load embedding model")?;

    let mut embeddings = model
        .embed(vec![query], None)
        .context("Query embedding failed")?;

    embeddings
        .pop()
        .map(Ok)
        .unwrap_or_else(|| Err(anyhow::anyhow!("Embedding returned empty result")))
}

fn get_cache_dir() -> std::path::PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    base.join("know").join("models")
}
