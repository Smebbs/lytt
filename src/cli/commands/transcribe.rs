//! Transcribe command implementation.

use crate::audio::download_audio;
use crate::audio_source::{parse_input, YoutubeSource, AudioSource};
use crate::chunking::{ChunkingConfig, ChunkingStrategy, create_chunker};
use crate::cli::preflight::{self, Operation};
use crate::cli::Output;
use crate::config::{TranscriptionProcessingSettings, Prompts, Settings, TranscriptionProvider};
use crate::embedding::{Embedder, OpenAIEmbedder};
use crate::orchestrator::Orchestrator;
use crate::transcription::{format_transcript, TranscriptionProcessor, OutputFormat, Transcriber};
use anyhow::Result;
use serde::Serialize;

/// Run the transcribe command.
pub async fn run_transcribe(
    input: &str,
    force: bool,
    output: Option<String>,
    format: &str,
    chunk: bool,
    embed: bool,
    playlist: bool,
    limit: Option<usize>,
    settings: Settings,
) -> Result<()> {
    // Pre-flight checks
    if let Err(e) = preflight::check(Operation::Transcribe) {
        Output::error(&format!("{}", e));
        Output::info("Run 'lytt doctor' for detailed diagnostics.");
        return Err(e.into());
    }

    // Validate flags
    if embed && !chunk {
        Output::error("--embed requires --chunk flag");
        return Err(anyhow::anyhow!("--embed requires --chunk"));
    }

    if (chunk || embed) && output.is_none() {
        Output::error("--chunk and --embed require --output flag");
        return Err(anyhow::anyhow!("--chunk/--embed require --output"));
    }

    if playlist && output.is_some() {
        Output::error("--playlist cannot be combined with --output");
        return Err(anyhow::anyhow!("--playlist cannot be combined with --output"));
    }

    // Handle playlist mode
    if playlist {
        return run_transcribe_playlist(input, force, limit, settings).await;
    }

    // If --output is specified, just transcribe and export (no indexing)
    if let Some(output_path) = output {
        return run_transcribe_only(input, &output_path, format, chunk, embed, &settings).await;
    }

    // Standard flow: transcribe and index
    run_transcribe_single(input, force, settings).await
}

/// Transcribe a single video and index it.
async fn run_transcribe_single(input: &str, force: bool, settings: Settings) -> Result<()> {
    Output::info(&format!("Processing: {}", input));

    let orchestrator = Orchestrator::new(settings)?;

    match orchestrator.process_media(input, force).await {
        Ok(result) => {
            if result.skipped {
                Output::warning(&format!(
                    "'{}' is already indexed. Use --force to reprocess.",
                    result.title
                ));
            } else {
                Output::success(&format!(
                    "Successfully indexed '{}' ({} chunks)",
                    result.title, result.chunks_indexed
                ));
            }
        }
        Err(e) => {
            Output::error(&format!("Failed to process: {}", e));
            return Err(e.into());
        }
    }

    Ok(())
}

/// Transcribe all videos from a playlist/channel.
async fn run_transcribe_playlist(
    input: &str,
    force: bool,
    limit: Option<usize>,
    settings: Settings,
) -> Result<()> {
    Output::info(&format!("Fetching playlist: {}", input));

    // Use YoutubeSource to list videos
    let source = YoutubeSource::new();

    if !source.can_handle(input) {
        Output::error("Input doesn't appear to be a valid YouTube playlist or channel URL");
        return Err(anyhow::anyhow!("Invalid playlist URL"));
    }

    let spinner = Output::spinner("Fetching video list...");
    let videos = source.list_media(input, limit).await?;
    spinner.finish_and_clear();

    if videos.is_empty() {
        Output::warning("No videos found in playlist");
        return Ok(());
    }

    let total = videos.len();
    Output::info(&format!("Found {} videos to transcribe", total));
    println!();

    let orchestrator = Orchestrator::new(settings)?;

    let mut success_count = 0;
    let mut skip_count = 0;
    let mut error_count = 0;

    for (i, video) in videos.iter().enumerate() {
        let progress = format!("[{}/{}]", i + 1, total);
        Output::info(&format!("{} Processing: {}", progress, video.title));

        match orchestrator.process_media(&video.id, force).await {
            Ok(result) => {
                if result.skipped {
                    Output::warning("  Skipped (already indexed)");
                    skip_count += 1;
                } else {
                    Output::success(&format!("  Indexed ({} chunks)", result.chunks_indexed));
                    success_count += 1;
                }
            }
            Err(e) => {
                Output::error(&format!("  Failed: {}", e));
                error_count += 1;
            }
        }
    }

    println!();
    Output::info(&format!(
        "Playlist complete: {} indexed, {} skipped, {} failed",
        success_count, skip_count, error_count
    ));

    Ok(())
}

/// Output format for chunked transcripts (for RAG integration).
#[derive(Debug, Serialize)]
struct ChunkedOutput {
    video_id: String,
    title: String,
    duration_seconds: f64,
    chunk_count: usize,
    chunks: Vec<ChunkOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_dimensions: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ChunkOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    content: String,
    start_seconds: f64,
    end_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
}

/// Transcribe only (no indexing) and output to file.
async fn run_transcribe_only(
    input: &str,
    output_path: &str,
    format: &str,
    chunk: bool,
    embed: bool,
    settings: &Settings,
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    // SRT/VTT don't support chunking or embeddings
    if chunk && output_format != OutputFormat::Json {
        Output::error("--chunk only supports JSON format");
        return Err(anyhow::anyhow!("--chunk requires --format json"));
    }

    Output::info(&format!("Transcribing: {}", input));

    // Parse input to get source and media ID
    let (source, media_id) = parse_input(input)
        .ok_or_else(|| anyhow::anyhow!("Could not parse input: {}", input))?;

    // Fetch metadata
    let metadata = source.fetch_media(&media_id).await?;
    Output::info(&format!("Title: {}", metadata.title));

    // Create temp directory
    let temp_dir = settings.temp_dir();
    std::fs::create_dir_all(&temp_dir)?;

    // Download audio
    let spinner = Output::spinner("Downloading audio...");
    let audio_path = download_audio(&metadata.source_url, &media_id, &temp_dir).await?;
    spinner.finish_and_clear();

    // Create transcriber - always use FusionTranscriber for LLM cleanup
    let prompts = Prompts::load(
        settings.prompts.custom_dir.as_deref(),
        Some(&settings.prompts.variables),
    )?;
    let processing_settings = match settings.transcription.provider {
        TranscriptionProvider::Whisper => TranscriptionProcessingSettings {
            timestamp_model: settings.transcription.model.clone(),
            text_model: None,
            cleanup_model: settings.transcription.processing.cleanup_model.clone(),
            max_concurrent: settings.transcription.max_concurrent_chunks,
        },
        TranscriptionProvider::Fusion => settings.transcription.processing.clone(),
    };
    let transcriber: Box<dyn Transcriber> = Box::new(TranscriptionProcessor::with_config(
        &processing_settings,
        &prompts.cleanup.system,
    )?);

    // Transcribe
    let spinner = Output::spinner("Transcribing...");
    let transcript = transcriber.transcribe(&audio_path).await?;
    spinner.finish_and_clear();

    // Format output based on flags
    let output_str = if chunk {
        // Apply semantic chunking
        let spinner = Output::spinner("Chunking...");

        let strategy: ChunkingStrategy = settings
            .chunking
            .strategy
            .parse()
            .unwrap_or(ChunkingStrategy::Semantic);

        let chunker = create_chunker(strategy);

        let config = ChunkingConfig {
            target_duration: settings.chunking.target_chunk_seconds,
            min_duration: settings.chunking.min_chunk_seconds,
            max_duration: settings.chunking.max_chunk_seconds,
        };

        let chunks = chunker.chunk(&transcript, &config).await?;
        spinner.finish_and_clear();

        // Generate embeddings if requested
        let (chunks_with_embeddings, embedding_model, embedding_dims) = if embed {
            let spinner = Output::spinner("Generating embeddings...");

            let embedder = OpenAIEmbedder::with_config(
                &settings.embedding.model,
                settings.embedding.dimensions as usize,
            );

            let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
            let embeddings = embedder.embed_batch(&texts).await?;

            spinner.finish_and_clear();

            let chunks_with_emb: Vec<ChunkOutput> = chunks
                .into_iter()
                .zip(embeddings)
                .map(|(c, emb)| ChunkOutput {
                    title: c.title,
                    content: c.content,
                    start_seconds: c.start_seconds,
                    end_seconds: c.end_seconds,
                    embedding: Some(emb),
                })
                .collect();

            (
                chunks_with_emb,
                Some(settings.embedding.model.clone()),
                Some(settings.embedding.dimensions as usize),
            )
        } else {
            let chunks_without_emb: Vec<ChunkOutput> = chunks
                .into_iter()
                .map(|c| ChunkOutput {
                    title: c.title,
                    content: c.content,
                    start_seconds: c.start_seconds,
                    end_seconds: c.end_seconds,
                    embedding: None,
                })
                .collect();

            (chunks_without_emb, None, None)
        };

        let output = ChunkedOutput {
            video_id: media_id.clone(),
            title: metadata.title.clone(),
            duration_seconds: transcript.duration_seconds,
            chunk_count: chunks_with_embeddings.len(),
            chunks: chunks_with_embeddings,
            embedding_model,
            embedding_dimensions: embedding_dims,
        };

        serde_json::to_string_pretty(&output)?
    } else {
        // Raw transcript output
        format_transcript(&transcript, output_format)
    };

    // Write to file or stdout
    if output_path == "-" {
        println!("{}", output_str);
    } else {
        std::fs::write(output_path, &output_str)?;

        let msg = if embed {
            format!("Saved {} chunks with embeddings to {}",
                output_str.matches("\"content\"").count(), output_path)
        } else if chunk {
            format!("Saved {} chunks to {}",
                output_str.matches("\"content\"").count(), output_path)
        } else {
            format!("Transcript saved to {} ({} segments)",
                output_path, transcript.segments.len())
        };
        Output::success(&msg);
    }

    // Cleanup
    if let Err(e) = std::fs::remove_file(&audio_path) {
        tracing::warn!("Failed to cleanup audio file: {}", e);
    }

    Ok(())
}
