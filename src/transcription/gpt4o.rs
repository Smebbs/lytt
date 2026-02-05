//! GPT-4o transcribe implementation.
//!
//! Provides high-accuracy text transcription without word-level timestamps.
//! Used in fusion mode alongside Whisper (which provides timestamps).

use super::PlainTranscript;
use crate::audio::split_audio;
use crate::error::{Result, LyttError};
use async_openai::types::{AudioResponseFormat, CreateTranscriptionRequestArgs};
use crate::openai::create_client;
use futures::StreamExt;
use futures::stream;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// GPT-4o-based transcriber for high-accuracy text (no timestamps).
pub struct Gpt4oTranscriber {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    chunk_duration_seconds: u32,
    max_concurrent_chunks: usize,
}

impl Gpt4oTranscriber {
    /// Create a new GPT-4o transcriber with default settings.
    pub fn new() -> Result<Self> {
        Self::with_config("gpt-4o-transcribe", 120, 3)
    }

    /// Create a new GPT-4o transcriber with custom configuration.
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

    /// Transcribe a single audio file to plain text.
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    pub async fn transcribe_single(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<String> {
        debug!("Transcribing audio with {}", self.model);

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
            .response_format(AudioResponseFormat::Json);

        if let Some(lang) = language {
            request_builder.language(lang);
        }

        let request = request_builder.build().map_err(|e| {
            LyttError::Transcription(format!("Failed to build request: {}", e))
        })?;

        let response = self
            .client
            .audio()
            .transcribe(request)
            .await
            .map_err(|e| LyttError::OpenAI(format!("{} API error: {}", self.model, e)))?;

        Ok(response.text.trim().to_string())
    }

    /// Transcribe audio to plain text, handling chunking.
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    pub async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        media_id: &str,
    ) -> Result<PlainTranscript> {
        self.transcribe_shared(audio_path, language, media_id, None).await
    }

    /// Transcribe audio to plain text, with optional shared progress counter.
    pub async fn transcribe_shared(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        media_id: &str,
        progress: Option<Arc<AtomicU64>>,
    ) -> Result<PlainTranscript> {
        let temp_dir = tempfile::tempdir()?;
        let chunks = split_audio(audio_path, temp_dir.path(), self.chunk_duration_seconds).await?;

        if chunks.len() == 1 {
            let text = self.transcribe_single(audio_path, language).await?;
            if let Some(p) = &progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(PlainTranscript {
                media_id: media_id.to_string(),
                text,
                language: language.map(|s| s.to_string()),
            });
        }

        let chunk_count = chunks.len();
        info!("Processing {} audio chunks with {}", chunk_count, self.model);

        // Process chunks in parallel, fail fast on error
        let mut results: Vec<(usize, String)> = Vec::with_capacity(chunk_count);

        let mut stream = stream::iter(chunks.into_iter().enumerate())
            .map(|(idx, (chunk_path, time_offset))| {
                let language = language.map(|s| s.to_string());
                let progress = progress.clone();
                async move {
                    let result = self
                        .transcribe_single(&chunk_path, language.as_deref())
                        .await;
                    if let Some(p) = &progress {
                        p.fetch_add(1, Ordering::Relaxed);
                    }
                    (idx, time_offset, result)
                }
            })
            .buffer_unordered(self.max_concurrent_chunks);

        while let Some((idx, time_offset, result)) = stream.next().await {
            match result {
                Ok(text) => results.push((idx, text)),
                Err(e) => {
                    drop(temp_dir);
                    let err_msg = format!("{} chunk {} at {:.0}s failed: {}", self.model, idx, time_offset, e);
                    eprintln!("  Error: {}", err_msg);
                    return Err(LyttError::Transcription(err_msg));
                }
            }
        }

        // Sort by chunk index and concatenate
        results.sort_by_key(|(idx, _)| *idx);

        let mut full_text = String::new();
        for (_, text) in results {
            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&text);
        }

        drop(temp_dir);

        Ok(PlainTranscript {
            media_id: media_id.to_string(),
            text: full_text,
            language: language.map(|s| s.to_string()),
        })
    }
}

impl Default for Gpt4oTranscriber {
    fn default() -> Self {
        Self::new().expect("Failed to create Gpt4oTranscriber")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model() {
        // Just verify it creates without panicking (no API call)
        let transcriber = Gpt4oTranscriber::with_config("test-model", 120, 3).unwrap();
        assert_eq!(transcriber.model, "test-model");
    }
}
