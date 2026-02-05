//! Search command implementation.

use crate::cli::Output;
use crate::config::Settings;
use crate::embedding::OpenAIEmbedder;
use crate::orchestrator::Orchestrator;
use crate::rag::context::ContextBuilder;
use anyhow::Result;
use std::sync::Arc;

/// Run the search command.
pub async fn run_search(
    query: &str,
    limit: usize,
    min_score: f32,
    settings: Settings,
) -> Result<()> {
    let orchestrator = Orchestrator::new(settings.clone())?;

    let embedder = Arc::new(OpenAIEmbedder::with_config(
        &settings.embedding.model,
        settings.embedding.dimensions as usize,
    ));

    let context_builder = ContextBuilder::new(orchestrator.vector_store(), embedder)
        .with_max_chunks(limit)
        .with_min_score(min_score);

    let spinner = Output::spinner("Searching...");

    let results = context_builder.build(query).await;
    spinner.finish_and_clear();

    match results {
        Ok(chunks) => {
            if chunks.is_empty() {
                Output::warning("No results found matching your query.");
            } else {
                Output::success(&format!("Found {} results", chunks.len()));

                for chunk in &chunks {
                    Output::search_result(
                        &chunk.video_title,
                        &chunk.timestamp,
                        chunk.score,
                        &chunk.content,
                        chunk.url.as_deref(),
                    );
                }
            }
        }
        Err(e) => {
            Output::error(&format!("Search failed: {}", e));
            return Err(anyhow::anyhow!("{}", e));
        }
    }

    Ok(())
}
