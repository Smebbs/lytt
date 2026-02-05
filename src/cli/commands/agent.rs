//! Agent command implementation.

use crate::cli::preflight::{self, Operation};
use crate::cli::Output;
use crate::config::Settings;
use crate::embedding::OpenAIEmbedder;
use crate::orchestrator::Orchestrator;
use crate::agent::{Agent, ToolContext};
use anyhow::Result;
use std::sync::Arc;

/// Run the agent command.
pub async fn run_agent(
    task: &str,
    video_id: Option<String>,
    model: Option<String>,
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

    // Build context if video_id is provided
    let context = video_id.as_ref().map(|id| format!("Focus on video ID: {}", id));

    // Create tool context
    let tool_context = ToolContext::new(orchestrator.vector_store(), embedder);

    // Create and run agent
    let agent = Agent::new(tool_context, &model);

    let spinner = Output::spinner("Agent working...");

    match agent.run(task, context.as_deref()).await {
        Ok(response) => {
            spinner.finish_and_clear();

            // Show the agent's response
            println!("\n{}\n", response.content);

            // Show tool calls summary if verbose
            if !response.tool_calls.is_empty() {
                Output::header(&format!("Tool calls ({})", response.tool_calls.len()));
                for call in &response.tool_calls {
                    Output::info(&format!("  {} {}", call.name, truncate(&call.arguments, 60)));
                }
                println!();
            }

            Output::info(&format!(
                "Completed in {} iteration(s)",
                response.iterations
            ));
        }
        Err(e) => {
            spinner.finish_and_clear();
            Output::error(&format!("Agent failed: {}", e));
            return Err(e.into());
        }
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
