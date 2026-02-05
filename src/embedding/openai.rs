//! OpenAI embeddings implementation.

use super::Embedder;
use crate::error::{Result, LyttError};
use async_openai::types::{CreateEmbeddingRequestArgs, EmbeddingInput};
use crate::openai::create_client;
use async_trait::async_trait;
use tracing::{debug, instrument};

/// OpenAI-based embedder.
pub struct OpenAIEmbedder {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    dimensions: usize,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder with default settings.
    pub fn new() -> Self {
        Self::with_config("text-embedding-3-small", 1536)
    }

    /// Create a new OpenAI embedder with custom model and dimensions.
    pub fn with_config(model: &str, dimensions: usize) -> Self {
        Self {
            client: create_client(),
            model: model.to_string(),
            dimensions,
        }
    }
}

impl Default for OpenAIEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    #[instrument(skip(self, text))]
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| LyttError::Embedding("Empty embedding response".to_string()))
    }

    #[instrument(skip(self, texts), fields(count = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Generating embeddings for {} texts", texts.len());

        // OpenAI has a limit on batch size, process in chunks
        const BATCH_SIZE: usize = 100;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let input: Vec<String> = chunk.to_vec();

            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model)
                .input(EmbeddingInput::StringArray(input))
                .dimensions(self.dimensions as u32)
                .build()
                .map_err(|e| LyttError::Embedding(format!("Failed to build request: {}", e)))?;

            let response = self.client.embeddings().create(request).await.map_err(|e| {
                LyttError::OpenAI(format!("Embedding API error: {}", e))
            })?;

            // Sort by index to ensure correct order
            let mut embeddings: Vec<_> = response.data.into_iter().collect();
            embeddings.sort_by_key(|e| e.index);

            for embedding_data in embeddings {
                all_embeddings.push(embedding_data.embedding);
            }
        }

        debug!("Generated {} embeddings", all_embeddings.len());
        Ok(all_embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = OpenAIEmbedder::new();
        assert_eq!(embedder.dimensions(), 1536);

        let embedder = OpenAIEmbedder::with_config("text-embedding-3-large", 3072);
        assert_eq!(embedder.dimensions(), 3072);
    }
}
