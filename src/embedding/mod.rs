//! Embedding generation for semantic search and retrieval.

mod openai;

pub use openai::OpenAIEmbedder;

use crate::error::Result;
use async_trait::async_trait;

/// Trait for embedding generation.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generate an embedding for a single text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimensions.
    fn dimensions(&self) -> usize;
}
