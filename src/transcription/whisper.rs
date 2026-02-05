//! OpenAI Whisper transcription implementation.

use super::{Transcriber, Transcript, TranscriptSegment, WhisperWord, WordLevelTranscript};
use crate::audio::split_audio;
use crate::error::{Result, LyttError};
use async_openai::types::{AudioResponseFormat, CreateTranscriptionRequestArgs, TimestampGranularity};
use crate::openai::create_client;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// OpenAI Whisper-based transcriber.
pub struct WhisperTranscriber {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    chunk_duration_seconds: u32,
    max_concurrent_chunks: usize,
}

impl WhisperTranscriber {
    /// Create a new Whisper transcriber with default settings.
    pub fn new() -> Result<Self> {
        Self::with_config("whisper-1", 120, 3)
    }

    /// Create a new Whisper transcriber with custom configuration.
    pub fn with_config(
        model: &str,
        chunk_duration_seconds: u32,
        max_concurrent_chunks: usize,
    ) -> Result<Self> {
        let client = create_client();

        Ok(Self {
            client,
            model: model.to_string(),
            chunk_duration_seconds,
            max_concurrent_chunks,
        })
    }

    /// Transcribe a single audio file (no splitting).
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    async fn transcribe_single(&self, audio_path: &Path, language: Option<&str>) -> Result<Vec<TranscriptSegment>> {
        debug!("Transcribing audio file");

        let file_bytes = tokio::fs::read(audio_path).await?;

        let mut request_builder = CreateTranscriptionRequestArgs::default();
        request_builder
            .file(async_openai::types::AudioInput::from_vec_u8(
                audio_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio.mp3")
                    .to_string(),
                file_bytes,
            ))
            .model(&self.model)
            .response_format(AudioResponseFormat::VerboseJson);

        if let Some(lang) = language {
            request_builder.language(lang);
        }

        let request = request_builder.build()
            .map_err(|e| LyttError::Transcription(format!("Failed to build request: {}", e)))?;

        let response = self.client.audio().transcribe_verbose_json(request).await
            .map_err(|e| LyttError::OpenAI(format!("Whisper API error: {}", e)))?;

        // Parse segments from verbose JSON response
        let segments: Vec<TranscriptSegment> = response.segments
            .map(|segs| {
                segs.iter()
                    .map(|s| TranscriptSegment::new(
                        s.start as f64,
                        s.end as f64,
                        s.text.trim().to_string(),
                    ))
                    .collect()
            })
            .unwrap_or_else(|| {
                // Fallback: create single segment from full text
                vec![TranscriptSegment::new(
                    0.0,
                    response.duration as f64,
                    response.text.trim().to_string(),
                )]
            });

        debug!("Transcribed {} segments", segments.len());
        Ok(segments)
    }

    /// Transcribe an audio file, splitting if necessary.
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    async fn transcribe_with_splitting(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        video_id: &str,
    ) -> Result<Transcript> {
        let temp_dir = tempfile::tempdir()?;
        let chunks = split_audio(audio_path, temp_dir.path(), self.chunk_duration_seconds).await?;

        if chunks.len() == 1 {
            // No splitting needed
            let segments = self.transcribe_single(audio_path, language).await?;
            return Ok(Transcript::new(video_id.to_string(), segments));
        }

        let chunk_count = chunks.len();
        info!("Processing {} audio chunks with {}", chunk_count, self.model);

        // Create progress bar
        let pb = Arc::new(ProgressBar::new(chunk_count as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.green} Whisper   [{bar:30.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("█▓░"),
        );

        // Process chunks in parallel with concurrency limit, fail fast on error
        let mut results: Vec<(usize, f64, Vec<TranscriptSegment>)> = Vec::with_capacity(chunk_count);

        let mut stream = stream::iter(chunks.into_iter().enumerate())
            .map(|(idx, (chunk_path, time_offset))| {
                let language = language.map(|s| s.to_string());
                async move {
                    let result = self.transcribe_single(&chunk_path, language.as_deref()).await;
                    (idx, time_offset, result)
                }
            })
            .buffer_unordered(self.max_concurrent_chunks);

        while let Some((idx, time_offset, result)) = stream.next().await {
            pb.inc(1);
            match result {
                Ok(segments) => results.push((idx, time_offset, segments)),
                Err(e) => {
                    pb.finish_and_clear();
                    drop(temp_dir);
                    let err_msg = format!("Chunk {} at {:.0}s failed: {}", idx, time_offset, e);
                    eprintln!("  Error: {}", err_msg);
                    return Err(LyttError::Transcription(err_msg));
                }
            }
        }

        pb.finish_and_clear();

        // Sort by chunk index and merge segments
        results.sort_by_key(|(idx, _, _)| *idx);

        let mut all_segments = Vec::new();
        for (_, time_offset, mut segments) in results {
            // Adjust timestamps by the chunk's time offset
            for segment in &mut segments {
                segment.start_seconds += time_offset;
                segment.end_seconds += time_offset;
            }
            all_segments.extend(segments);
        }

        // Clean up temp files
        drop(temp_dir);

        Ok(Transcript::new(video_id.to_string(), all_segments))
    }

    // ========================================================================
    // Word-level timestamp methods (for fusion transcription)
    // ========================================================================

    /// Transcribe a single audio file with word-level timestamps.
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    pub async fn transcribe_single_with_words(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<Vec<WhisperWord>> {
        debug!("Transcribing audio file with word-level timestamps");

        let file_bytes = tokio::fs::read(audio_path).await?;

        let mut request_builder = CreateTranscriptionRequestArgs::default();
        request_builder
            .file(async_openai::types::AudioInput::from_vec_u8(
                audio_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio.mp3")
                    .to_string(),
                file_bytes,
            ))
            .model(&self.model)
            .response_format(AudioResponseFormat::VerboseJson)
            .timestamp_granularities(vec![TimestampGranularity::Word]);

        if let Some(lang) = language {
            request_builder.language(lang);
        }

        let request = request_builder.build().map_err(|e| {
            LyttError::Transcription(format!("Failed to build request: {}", e))
        })?;

        let response = self
            .client
            .audio()
            .transcribe_verbose_json(request)
            .await
            .map_err(|e| LyttError::OpenAI(format!("Whisper API error: {}", e)))?;

        // Parse words from verbose JSON response
        let words: Vec<WhisperWord> = response
            .words
            .map(|ws| {
                ws.iter()
                    .map(|w| WhisperWord {
                        word: w.word.clone(),
                        start: w.start as f64,
                        end: w.end as f64,
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                warn!("No word-level timestamps returned, falling back to segment-level");
                // Fallback: approximate word timestamps from segments
                response
                    .segments
                    .map(|segs| {
                        segs.iter()
                            .flat_map(|s| {
                                let words: Vec<&str> = s.text.split_whitespace().collect();
                                if words.is_empty() {
                                    return vec![];
                                }
                                let duration = (s.end - s.start) as f64;
                                let word_duration = duration / words.len() as f64;
                                words
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, word)| WhisperWord {
                                        word: word.to_string(),
                                        start: s.start as f64 + i as f64 * word_duration,
                                        end: s.start as f64 + (i + 1) as f64 * word_duration,
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            });

        debug!("Transcribed {} words", words.len());
        Ok(words)
    }

    /// Transcribe with word-level timestamps, handling chunking.
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    pub async fn transcribe_with_words(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        media_id: &str,
    ) -> Result<WordLevelTranscript> {
        self.transcribe_with_words_shared(audio_path, language, media_id, None).await
    }

    /// Transcribe with word-level timestamps, with optional shared progress counter.
    /// Returns the number of chunks processed (for progress bar setup).
    pub async fn transcribe_with_words_shared(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        media_id: &str,
        progress: Option<Arc<AtomicU64>>,
    ) -> Result<WordLevelTranscript> {
        let temp_dir = tempfile::tempdir()?;
        let chunks = split_audio(audio_path, temp_dir.path(), self.chunk_duration_seconds).await?;

        if chunks.len() == 1 {
            let words = self
                .transcribe_single_with_words(audio_path, language)
                .await?;
            if let Some(p) = &progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(WordLevelTranscript::new(media_id.to_string(), words));
        }

        let chunk_count = chunks.len();
        info!(
            "Processing {} audio chunks for word-level timestamps",
            chunk_count
        );

        // Process chunks in parallel
        let results: Vec<(usize, f64, Result<Vec<WhisperWord>>)> =
            stream::iter(chunks.into_iter().enumerate())
                .map(|(idx, (chunk_path, time_offset))| {
                    let language = language.map(|s| s.to_string());
                    let progress = progress.clone();
                    async move {
                        let result = self
                            .transcribe_single_with_words(&chunk_path, language.as_deref())
                            .await;
                        if let Some(p) = progress {
                            p.fetch_add(1, Ordering::Relaxed);
                        }
                        (idx, time_offset, result)
                    }
                })
                .buffer_unordered(self.max_concurrent_chunks)
                .collect()
                .await;

        // Sort by chunk index and merge
        let mut sorted_results: Vec<_> = results.into_iter().collect();
        sorted_results.sort_by_key(|(idx, _, _)| *idx);

        let mut all_words = Vec::new();
        let mut errors = Vec::new();

        for (idx, time_offset, result) in sorted_results {
            match result {
                Ok(mut words) => {
                    // Adjust timestamps by chunk offset
                    for word in &mut words {
                        word.start += time_offset;
                        word.end += time_offset;
                    }
                    all_words.extend(words);
                }
                Err(e) => {
                    errors.push(format!("Chunk {}: {}", idx, e));
                }
            }
        }

        drop(temp_dir);

        if !errors.is_empty() {
            return Err(LyttError::Transcription(format!(
                "Word-level transcription failed for {} chunk(s):\n{}",
                errors.len(),
                errors.join("\n")
            )));
        }

        Ok(WordLevelTranscript::new(media_id.to_string(), all_words))
    }
}

impl Default for WhisperTranscriber {
    fn default() -> Self {
        Self::new().expect("Failed to create WhisperTranscriber")
    }
}

#[async_trait]
impl Transcriber for WhisperTranscriber {
    async fn transcribe(&self, audio_path: &Path) -> Result<Transcript> {
        let video_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        self.transcribe_with_splitting(audio_path, None, &video_id).await
    }

    async fn transcribe_with_language(&self, audio_path: &Path, language: &str) -> Result<Transcript> {
        let video_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        self.transcribe_with_splitting(audio_path, Some(language), &video_id).await
    }
}

/// Check if the OpenAI API key is configured.
pub fn is_api_key_configured() -> bool {
    std::env::var("OPENAI_API_KEY").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_check() {
        // This just tests that the function works
        let _ = is_api_key_configured();
    }
}
