//! Export command implementation.

use crate::cli::Output;
use crate::config::Settings;
use crate::transcription::{OutputFormat, Transcript, TranscriptSegment};
use crate::vector_store::{SqliteVectorStore, VectorStore};
use anyhow::Result;
use serde::Serialize;

/// Exportable transcript with metadata.
#[derive(Debug, Serialize)]
pub struct ExportedTranscript {
    pub video_id: String,
    pub video_title: String,
    pub total_duration_seconds: f64,
    pub chunk_count: usize,
    pub segments: Vec<ExportedSegment>,
}

#[derive(Debug, Serialize)]
pub struct ExportedSegment {
    pub title: String,
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// Run the export command.
pub async fn run_export(
    video_id: &str,
    output: Option<String>,
    format: &str,
    settings: Settings,
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    // Open vector store
    let store = SqliteVectorStore::new(&settings.sqlite_path())?;

    // Get all chunks for this video
    let chunks = store.get_by_video_id(video_id).await?;

    if chunks.is_empty() {
        Output::error(&format!("No indexed content found for video ID: {}", video_id));
        Output::info("Use 'lytt list' to see indexed media.");
        return Ok(());
    }

    // Get video title from first chunk
    let video_title = chunks.first().map(|c| c.video_title.clone()).unwrap_or_default();

    // Sort chunks by start time
    let mut chunks = chunks;
    chunks.sort_by(|a, b| a.start_seconds.partial_cmp(&b.start_seconds).unwrap());

    // Calculate total duration
    let total_duration = chunks
        .iter()
        .map(|c| c.end_seconds)
        .fold(0.0f64, |a, b| a.max(b));

    // Format based on requested format
    let output_str = match output_format {
        OutputFormat::Json => {
            let export = ExportedTranscript {
                video_id: video_id.to_string(),
                video_title: video_title.clone(),
                total_duration_seconds: total_duration,
                chunk_count: chunks.len(),
                segments: chunks
                    .iter()
                    .map(|c| ExportedSegment {
                        title: c.section_title.clone().unwrap_or_default(),
                        text: c.content.clone(),
                        start_seconds: c.start_seconds,
                        end_seconds: c.end_seconds,
                    })
                    .collect(),
            };
            serde_json::to_string_pretty(&export)?
        }
        OutputFormat::Srt => {
            let transcript = Transcript::new(
                video_id.to_string(),
                chunks
                    .iter()
                    .map(|c| TranscriptSegment {
                        text: c.content.clone(),
                        start_seconds: c.start_seconds,
                        end_seconds: c.end_seconds,
                    })
                    .collect(),
            );
            crate::transcription::format_transcript(&transcript, OutputFormat::Srt)
        }
        OutputFormat::Vtt => {
            let transcript = Transcript::new(
                video_id.to_string(),
                chunks
                    .iter()
                    .map(|c| TranscriptSegment {
                        text: c.content.clone(),
                        start_seconds: c.start_seconds,
                        end_seconds: c.end_seconds,
                    })
                    .collect(),
            );
            crate::transcription::format_transcript(&transcript, OutputFormat::Vtt)
        }
    };

    // Write output
    match output {
        Some(path) if path != "-" => {
            std::fs::write(&path, &output_str)?;
            Output::success(&format!(
                "Exported '{}' to {} ({} segments)",
                video_title,
                path,
                chunks.len()
            ));
        }
        _ => {
            // Output to stdout
            println!("{}", output_str);
        }
    }

    Ok(())
}
