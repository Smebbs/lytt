//! List command implementation.

use crate::cli::Output;
use crate::config::Settings;
use crate::orchestrator::Orchestrator;
use anyhow::Result;

/// Run the list command.
pub async fn run_list(settings: Settings) -> Result<()> {
    let orchestrator = Orchestrator::new(settings)?;

    match orchestrator.vector_store().list_videos().await {
        Ok(media) => {
            if media.is_empty() {
                Output::info("No media indexed yet. Use 'lytt transcribe <input>' to add content.");
            } else {
                Output::header(&format!("Indexed Media ({})", media.len()));
                println!();

                for item in &media {
                    Output::media_info(
                        &item.video_title,
                        &item.video_id,
                        item.chunk_count,
                        item.total_duration_seconds,
                    );
                }

                let total_chunks: u32 = media.iter().map(|m| m.chunk_count).sum();
                println!();
                Output::kv("Total items", &media.len().to_string());
                Output::kv("Total chunks", &total_chunks.to_string());
            }
        }
        Err(e) => {
            Output::error(&format!("Failed to list media: {}", e));
            return Err(e.into());
        }
    }

    Ok(())
}
