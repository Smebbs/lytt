//! Ask command implementation.

use crate::cli::preflight::{self, Operation};
use crate::cli::Output;
use crate::config::Settings;
use crate::embedding::OpenAIEmbedder;
use crate::orchestrator::Orchestrator;
use crate::rag::RagEngine;
use anyhow::Result;
use std::sync::Arc;

/// Run the ask command.
pub async fn run_ask(
    question: &str,
    model: Option<String>,
    max_chunks: usize,
    settings: Settings,
) -> Result<()> {
    // Pre-flight checks
    if let Err(e) = preflight::check(Operation::Ask) {
        Output::error(&format!("{}", e));
        Output::info("Run 'lytt doctor' for detailed diagnostics.");
        return Err(e.into());
    }

    let orchestrator = Orchestrator::new(settings.clone())?;

    let model = model.unwrap_or_else(|| settings.rag.model.clone());

    let embedder = Arc::new(OpenAIEmbedder::with_config(
        &settings.embedding.model,
        settings.embedding.dimensions as usize,
    ));

    let engine = RagEngine::new(
        orchestrator.vector_store(),
        embedder,
        &model,
        max_chunks,
    );

    let spinner = Output::spinner("Searching knowledge base...");

    match engine.ask(question).await {
        Ok(response) => {
            spinner.finish_and_clear();

            println!("\n{}\n", response.answer);

            if !response.sources.is_empty() {
                Output::header("Sources");
                for source in &response.sources {
                    Output::search_result(
                        &source.video_title,
                        &source.timestamp,
                        source.score,
                        &source.content[..source.content.len().min(100)],
                        source.url.as_deref(),
                    );
                }
            }
        }
        Err(e) => {
            spinner.finish_and_clear();
            Output::error(&format!("Failed to generate answer: {}", e));
            return Err(e.into());
        }
    }

    Ok(())
}
