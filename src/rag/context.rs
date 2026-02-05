//! Context building for RAG responses.

use super::ContextChunk;
use crate::embedding::Embedder;
use crate::error::Result;
use crate::vector_store::{SearchResult, VectorStore};
use std::sync::Arc;

/// Builds context from search results for RAG.
pub struct ContextBuilder {
    vector_store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    max_chunks: usize,
    min_score: f32,
}

impl ContextBuilder {
    /// Create a new context builder.
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            vector_store,
            embedder,
            max_chunks: 10,
            min_score: 0.3,
        }
    }

    /// Set the maximum number of context chunks.
    pub fn with_max_chunks(mut self, max_chunks: usize) -> Self {
        self.max_chunks = max_chunks;
        self
    }

    /// Set the minimum similarity score threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Build context for a query.
    pub async fn build(&self, query: &str) -> Result<Vec<ContextChunk>> {
        // Generate query embedding
        let query_embedding = self.embedder.embed(query).await?;

        // Search for relevant documents
        let results = self
            .vector_store
            .search_with_threshold(&query_embedding, self.max_chunks, self.min_score)
            .await?;

        // Convert to context chunks
        let chunks: Vec<ContextChunk> = results
            .into_iter()
            .map(|r| {
                let mut chunk = ContextChunk::from(r.clone());
                // Build YouTube URL with timestamp if it's a YouTube video
                if !r.document.video_id.starts_with("local_") {
                    chunk.url = Some(format!(
                        "https://youtube.com/watch?v={}&t={}s",
                        r.document.video_id,
                        r.document.start_seconds as u32
                    ));
                }
                chunk
            })
            .collect();

        Ok(chunks)
    }

    /// Build context from raw search results.
    pub fn from_results(results: Vec<SearchResult>) -> Vec<ContextChunk> {
        results
            .into_iter()
            .map(|r| {
                let mut chunk = ContextChunk::from(r.clone());
                if !r.document.video_id.starts_with("local_") {
                    chunk.url = Some(format!(
                        "https://youtube.com/watch?v={}&t={}s",
                        r.document.video_id,
                        r.document.start_seconds as u32
                    ));
                }
                chunk
            })
            .collect()
    }
}

/// Format context chunks for display in a prompt.
pub fn format_context_for_prompt(chunks: &[ContextChunk]) -> String {
    chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            format!(
                "---\n[{}] {} @ {}\n{}\n---",
                i + 1,
                chunk.video_title,
                chunk.timestamp,
                chunk.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Format context chunks for display to the user.
pub fn format_context_for_display(chunks: &[ContextChunk]) -> String {
    chunks
        .iter()
        .map(|chunk| {
            let url_part = chunk
                .url
                .as_ref()
                .map(|u| format!("\n  Link: {}", u))
                .unwrap_or_default();

            format!(
                "{} @ {} (score: {:.2}){}",
                chunk.video_title,
                chunk.timestamp,
                chunk.score,
                url_part
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
