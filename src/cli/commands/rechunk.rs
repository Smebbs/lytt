//! Rechunk command implementation.

use crate::cli::Output;
use crate::config::Settings;
use crate::orchestrator::Orchestrator;
use anyhow::Result;

/// Run the rechunk command.
pub async fn run_rechunk(video_id: &str, settings: Settings) -> Result<()> {
    let orchestrator = Orchestrator::new(settings)?;

    if video_id == "all" {
        // Rechunk all videos with stored transcripts
        let videos = orchestrator.list_rechunkable()?;

        if videos.is_empty() {
            Output::warning("No videos with stored transcripts found.");
            Output::info("Transcripts are stored for videos transcribed after this feature was added.");
            Output::info("Use 'lytt transcribe VIDEO_URL --force' to re-transcribe and store the transcript.");
            return Ok(());
        }

        Output::info(&format!("Found {} videos to rechunk", videos.len()));
        println!();

        let mut success_count = 0;
        let mut error_count = 0;

        for (i, (vid_id, title, _duration)) in videos.iter().enumerate() {
            Output::info(&format!("[{}/{}] Rechunking: {}", i + 1, videos.len(), title));

            match orchestrator.rechunk_media(vid_id).await {
                Ok(result) => {
                    Output::success(&format!("  Rechunked ({} chunks)", result.chunks_indexed));
                    success_count += 1;
                }
                Err(e) => {
                    Output::error(&format!("  Failed: {}", e));
                    error_count += 1;
                }
            }
        }

        println!();
        Output::info(&format!(
            "Rechunking complete: {} succeeded, {} failed",
            success_count, error_count
        ));
    } else {
        // Rechunk single video
        Output::info(&format!("Rechunking video: {}", video_id));

        let spinner = Output::spinner("Rechunking...");

        match orchestrator.rechunk_media(video_id).await {
            Ok(result) => {
                spinner.finish_and_clear();
                Output::success(&format!(
                    "Successfully rechunked '{}' ({} chunks)",
                    result.title, result.chunks_indexed
                ));
            }
            Err(e) => {
                spinner.finish_and_clear();
                Output::error(&format!("Failed to rechunk: {}", e));
                return Err(e.into());
            }
        }
    }

    Ok(())
}
