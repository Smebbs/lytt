//! Fusion transcription: combines Whisper timestamps with accurate text from another model.
//!
//! Architecture:
//! 1. Split audio into segments (e.g., 2 minutes each)
//! 2. For each segment, run Whisper AND GPT-4o in parallel
//! 3. Fuse each segment pairwise (Whisper words + GPT-4o text → fused segment)
//! 4. Merge all fused segments into complete transcript with timestamps

use super::{
    gpt4o::Gpt4oTranscriber, whisper::WhisperTranscriber, FusedSegment,
    Transcriber, Transcript, TranscriptSegment, WhisperWord,
};
use crate::audio::split_audio;
use crate::config::TranscriptionProcessingSettings;
use crate::error::{LyttError, Result};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ResponseFormat,
};
use crate::openai::create_client;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::{debug, info, instrument, warn};

/// Transcription processor combining Whisper timestamps with optional secondary text model.
///
/// Two modes:
/// - Full fusion: Whisper (timestamps) + GPT-4o (text) → LLM fuses both
/// - Whisper + cleanup: Whisper only → LLM cleans up and creates natural segments
pub struct TranscriptionProcessor {
    whisper: WhisperTranscriber,
    gpt4o: Option<Gpt4oTranscriber>,
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    cleanup_model: String,
    system_prompt: String,
    segment_duration_seconds: u32,
    max_concurrent_segments: usize,
}

impl TranscriptionProcessor {
    /// Create with custom settings and system prompt.
    pub fn with_config(settings: &TranscriptionProcessingSettings, system_prompt: &str) -> Result<Self> {
        let gpt4o = if settings.has_text_model() {
            Some(Gpt4oTranscriber::with_config(
                settings.text_model.as_ref().unwrap(),
                // Don't chunk internally - we handle segmentation
                u32::MAX,
                1,
            )?)
        } else {
            None
        };

        Ok(Self {
            whisper: WhisperTranscriber::with_config(
                &settings.timestamp_model,
                // Don't chunk internally - we handle segmentation
                u32::MAX,
                1,
            )?,
            gpt4o,
            client: create_client(),
            cleanup_model: settings.cleanup_model.clone(),
            system_prompt: system_prompt.to_string(),
            // 5 minutes per segment for parallel processing
            segment_duration_seconds: 300,
            max_concurrent_segments: settings.max_concurrent,
        })
    }

    /// Check if running in full fusion mode (with secondary text model).
    pub fn is_full_fusion(&self) -> bool {
        self.gpt4o.is_some()
    }

    /// Process a single segment: transcribe and clean up/fuse.
    async fn process_segment(
        &self,
        segment_path: &Path,
        time_offset: f64,
        language: Option<&str>,
    ) -> Result<Vec<TranscriptSegment>> {
        debug!("Starting segment at {:.0}s", time_offset);

        let (whisper_words, text) = if let Some(ref gpt4o) = self.gpt4o {
            // Full fusion: run Whisper and GPT-4o in parallel
            debug!("Running Whisper + GPT-4o for segment at {:.0}s", time_offset);
            let (words, gpt4o_text) = tokio::try_join!(
                self.whisper.transcribe_single_with_words(segment_path, language),
                gpt4o.transcribe_single(segment_path, language),
            )?;
            (words, gpt4o_text)
        } else {
            // Whisper-only: get words and construct text from them
            debug!("Running Whisper-only for segment at {:.0}s", time_offset);
            let words = self.whisper.transcribe_single_with_words(segment_path, language).await?;
            let text = words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
            (words, text)
        };

        info!(
            "Segment at {:.0}s: {} words, {} chars (fusion: {})",
            time_offset,
            whisper_words.len(),
            text.len(),
            self.is_full_fusion()
        );

        // Clean up / fuse this segment with LLM
        debug!("Running LLM fusion for segment at {:.0}s", time_offset);
        let fused = self
            .fuse_segment(&whisper_words, &text, time_offset)
            .await?;
        debug!("Segment at {:.0}s complete: {} fused segments", time_offset, fused.len());

        // Convert to TranscriptSegments
        Ok(fused.into_iter().map(Into::into).collect())
    }

    /// Fuse a single segment's Whisper words with GPT-4o text using LLM.
    async fn fuse_segment(
        &self,
        words: &[WhisperWord],
        text: &str,
        time_offset: f64,
    ) -> Result<Vec<FusedSegment>> {
        // For very short segments or empty content, use fallback
        if words.is_empty() || text.trim().is_empty() {
            if !text.trim().is_empty() {
                let duration = words.last().map(|w| w.end).unwrap_or(10.0);
                return Ok(vec![FusedSegment {
                    text: text.trim().to_string(),
                    start_seconds: time_offset,
                    end_seconds: time_offset + duration,
                }]);
            }
            return Ok(vec![]);
        }

        // Try LLM fusion
        match self.fuse_with_llm(words, text).await {
            Ok(segments) if !segments.is_empty() => {
                // Adjust timestamps by offset
                Ok(segments
                    .into_iter()
                    .map(|mut s| {
                        s.start_seconds += time_offset;
                        s.end_seconds += time_offset;
                        s
                    })
                    .collect())
            }
            Ok(_) => {
                warn!("LLM returned empty for segment at {:.0}s, using fallback", time_offset);
                Ok(self.align_by_position(words, text, time_offset))
            }
            Err(e) => {
                warn!("LLM fusion failed for segment at {:.0}s: {}, using fallback", time_offset, e);
                Ok(self.align_by_position(words, text, time_offset))
            }
        }
    }

    /// Call LLM to fuse words with text.
    async fn fuse_with_llm(&self, words: &[WhisperWord], text: &str) -> Result<Vec<FusedSegment>> {
        let words_json = serde_json::to_string_pretty(words).unwrap_or_default();

        let user_prompt = format!(
            "Word timestamps:\n{}\n\nAccurate text:\n{}\n\nReturn JSON: {{\"segments\": [{{\"text\": \"...\", \"start_seconds\": 0.0, \"end_seconds\": 5.0}}]}}",
            words_json,
            text
        );

        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt.clone())
                .build()
                .map_err(|e| LyttError::Transcription(e.to_string()))?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()
                .map_err(|e| LyttError::Transcription(e.to_string()))?
                .into(),
        ];

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.cleanup_model)
            .messages(messages)
            .temperature(0.0)
            .response_format(ResponseFormat::JsonObject)
            .build()
            .map_err(|e| LyttError::Transcription(e.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| LyttError::OpenAI(format!("Fusion error: {}", e)))?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| LyttError::Transcription("Empty response".to_string()))?;

        // Parse segments from JSON response
        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| LyttError::Transcription(format!("Invalid JSON: {}", e)))?;

        let segments = parsed
            .get("segments")
            .ok_or_else(|| LyttError::Transcription("Missing segments".to_string()))?;

        serde_json::from_value(segments.clone())
            .map_err(|e| LyttError::Transcription(format!("Invalid segments: {}", e)))
    }

    /// Fallback: align text to word timestamps by position.
    fn align_by_position(&self, words: &[WhisperWord], text: &str, time_offset: f64) -> Vec<FusedSegment> {
        if words.is_empty() || text.is_empty() {
            let duration = words.last().map(|w| w.end).unwrap_or(10.0);
            return vec![FusedSegment {
                text: text.to_string(),
                start_seconds: time_offset,
                end_seconds: time_offset + duration,
            }];
        }

        // Split accurate text into sentences
        let sentences: Vec<&str> = text
            .split(['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            let duration = words.last().map(|w| w.end).unwrap_or(10.0);
            return vec![FusedSegment {
                text: text.to_string(),
                start_seconds: time_offset,
                end_seconds: time_offset + duration,
            }];
        }

        // Count words in each sentence to map to timestamp positions
        let total_words = words.len();
        let mut segments = Vec::new();
        let mut word_idx = 0;

        for sentence in sentences {
            let sentence_word_count = sentence.split_whitespace().count();
            if sentence_word_count == 0 {
                continue;
            }

            let start_idx = word_idx.min(total_words.saturating_sub(1));
            let end_idx = (word_idx + sentence_word_count).min(total_words).saturating_sub(1);

            let start = words.get(start_idx).map(|w| w.start).unwrap_or(0.0);
            let end = words.get(end_idx).map(|w| w.end).unwrap_or(words.last().map(|w| w.end).unwrap_or(10.0));

            segments.push(FusedSegment {
                text: format!("{}.", sentence),
                start_seconds: time_offset + start,
                end_seconds: time_offset + end.max(start + 0.1),
            });

            word_idx += sentence_word_count;
        }

        // Ensure last segment extends to end
        if let Some(last) = segments.last_mut() {
            let segment_end = time_offset + words.last().map(|w| w.end).unwrap_or(10.0);
            if last.end_seconds < segment_end - 1.0 {
                last.end_seconds = segment_end;
            }
        }

        segments
    }
}

impl Default for TranscriptionProcessor {
    fn default() -> Self {
        use crate::config::CleanupPrompts;
        let prompts = CleanupPrompts::default();
        Self::with_config(&TranscriptionProcessingSettings::default(), &prompts.system)
            .expect("Failed to create TranscriptionProcessor")
    }
}

#[async_trait]
impl Transcriber for TranscriptionProcessor {
    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    async fn transcribe(&self, audio_path: &Path) -> Result<Transcript> {
        self.transcribe_with_language(audio_path, "").await
    }

    #[instrument(skip(self), fields(audio_path = %audio_path.display()))]
    async fn transcribe_with_language(
        &self,
        audio_path: &Path,
        language: &str,
    ) -> Result<Transcript> {
        let media_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let lang = if language.is_empty() {
            None
        } else {
            Some(language)
        };

        info!("Fusion transcription: {}", media_id);

        // 1. Split audio into segments
        let temp_dir = tempfile::tempdir()?;
        let segments = split_audio(audio_path, temp_dir.path(), self.segment_duration_seconds).await?;
        let segment_count = segments.len();

        info!("Split into {} parts for processing", segment_count);
        eprintln!("  Processing {} audio parts", segment_count);

        // Create progress bar
        let pb = ProgressBar::new(segment_count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.green} Transcribing [{bar:30.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        // 2. Process segments with concurrency control
        // Each segment: Whisper + GPT-4o in parallel, then fuse
        // Fail fast on first error
        let lang_owned = lang.map(|s| s.to_string());
        let mut all_segments: Vec<TranscriptSegment> = Vec::new();

        let mut stream = stream::iter(segments)
            .map(|(segment_path, time_offset)| {
                let lang_ref = lang_owned.as_deref();
                async move {
                    (time_offset, self.process_segment(&segment_path, time_offset, lang_ref).await)
                }
            })
            .buffer_unordered(self.max_concurrent_segments);

        while let Some((time_offset, result)) = stream.next().await {
            pb.inc(1);
            match result {
                Ok(segments) => all_segments.extend(segments),
                Err(e) => {
                    pb.finish_and_clear();
                    drop(temp_dir);
                    let err_msg = format!("Segment at {:.0}s failed: {}", time_offset, e);
                    eprintln!("  Error: {}", err_msg);
                    return Err(LyttError::Transcription(err_msg));
                }
            }
        }

        pb.finish_and_clear();

        // Clean up temp files
        drop(temp_dir);

        // Sort by start time (in case of out-of-order processing)
        all_segments.sort_by(|a, b| a.start_seconds.partial_cmp(&b.start_seconds).unwrap());

        info!("Fusion complete: {} timestamped sections", all_segments.len());
        eprintln!("  Transcription complete");

        Ok(Transcript::new(media_id, all_segments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_processor() -> TranscriptionProcessor {
        // Create with test settings (won't actually call APIs in tests)
        TranscriptionProcessor {
            whisper: WhisperTranscriber::with_config("whisper-1", 120, 1).unwrap(),
            gpt4o: Some(Gpt4oTranscriber::with_config("gpt-4o-transcribe", 120, 1).unwrap()),
            client: create_client(),
            cleanup_model: "gpt-4.1".to_string(),
            system_prompt: "Test".to_string(),
            segment_duration_seconds: 120,
            max_concurrent_segments: 2,
        }
    }

    #[test]
    fn test_align_by_position() {
        let transcriber = create_test_processor();
        let words = vec![
            WhisperWord { word: "Hello".into(), start: 0.0, end: 0.5 },
            WhisperWord { word: "world".into(), start: 0.5, end: 1.0 },
            WhisperWord { word: "this".into(), start: 1.0, end: 1.3 },
            WhisperWord { word: "is".into(), start: 1.3, end: 1.5 },
            WhisperWord { word: "a".into(), start: 1.5, end: 1.6 },
            WhisperWord { word: "test".into(), start: 1.6, end: 2.0 },
        ];

        let text = "Hello world. This is a test.";
        let segments = transcriber.align_by_position(&words, text, 0.0);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Hello world.");
        assert!((segments[0].start_seconds - 0.0).abs() < 0.01);
        assert_eq!(segments[1].text, "This is a test.");
    }

    #[test]
    fn test_align_with_offset() {
        let transcriber = create_test_processor();
        let words = vec![
            WhisperWord { word: "Hello".into(), start: 0.0, end: 0.5 },
            WhisperWord { word: "world".into(), start: 0.5, end: 1.0 },
        ];

        let text = "Hello world.";
        let segments = transcriber.align_by_position(&words, text, 120.0);

        assert_eq!(segments.len(), 1);
        assert!((segments[0].start_seconds - 120.0).abs() < 0.01);
        assert!((segments[0].end_seconds - 121.0).abs() < 0.01);
    }

    #[test]
    fn test_align_empty_input() {
        let transcriber = create_test_processor();
        let segments = transcriber.align_by_position(&[], "Hello", 5.0);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello");
        assert_eq!(segments[0].start_seconds, 5.0);
    }
}
