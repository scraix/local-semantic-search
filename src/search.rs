use crate::storage::EmbeddingEntry;

/// Find top-k matching entries by cosine similarity.
/// Results below threshold are excluded.
/// Takes a slice of references (to support pre-filtered lists by tags).
pub fn find_top<'a>(
    query_vec: &[f32],
    entries: &[&'a EmbeddingEntry],
    threshold: f32,
    top_k: usize,
) -> Vec<(&'a EmbeddingEntry, f32)> {
    let mut scored: Vec<(&EmbeddingEntry, f32)> = entries
        .iter()
        .map(|e| {
            let score = cosine_similarity(query_vec, &e.vector);
            (*e, score)
        })
        .filter(|(_, s)| *s >= threshold)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
