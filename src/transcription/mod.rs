//! Transcription module for Lytt.
//!
//! Handles audio transcription using OpenAI Whisper, GPT-4o, and the processing pipeline.
//!
//! # Transcription Modes
//!
//! - **Whisper** (default): Uses OpenAI Whisper with LLM cleanup for proper segmentation.
//! - **Fusion**: Combines Whisper's word-level timestamps with GPT-4o's accurate text,
//!   using an LLM to intelligently fuse both transcriptions.

mod format;
mod fusion;
mod gpt4o;
mod models;
mod whisper;

pub use format::{format_transcript, OutputFormat, SegmentExport, TranscriptExport};
pub use fusion::TranscriptionProcessor;
pub use gpt4o::Gpt4oTranscriber;
pub use models::{
    FusedSegment, PlainTranscript, Transcript, TranscriptSegment, WhisperWord, WordLevelTranscript,
};
pub use whisper::{is_api_key_configured, WhisperTranscriber};

use crate::error::Result;
use async_trait::async_trait;
use std::path::Path;

/// Trait for transcription services.
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribe an audio file and return segments with timestamps.
    async fn transcribe(&self, audio_path: &Path) -> Result<Transcript>;

    /// Transcribe an audio file with a specific language hint.
    async fn transcribe_with_language(
        &self,
        audio_path: &Path,
        language: &str,
    ) -> Result<Transcript>;
}
