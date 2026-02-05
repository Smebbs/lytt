//! RAG (Retrieval-Augmented Generation) for question answering with sources.
//!
//! Provides the ability to ask questions and get answers from the video knowledge base.

pub mod context;
mod response;

pub use context::ContextBuilder;
pub use response::{RagEngine, RagResponse};

use crate::vector_store::SearchResult;

/// A search result with formatted context for display.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// Video ID.
    pub video_id: String,
    /// Video title.
    pub video_title: String,
    /// Formatted timestamp (e.g., "02:34").
    pub timestamp: String,
    /// Start time in seconds.
    pub start_seconds: f64,
    /// Text content.
    pub content: String,
    /// Similarity score.
    pub score: f32,
    /// URL with timestamp (if available).
    pub url: Option<String>,
}

impl From<SearchResult> for ContextChunk {
    fn from(result: SearchResult) -> Self {
        Self {
            video_id: result.document.video_id.clone(),
            video_title: result.document.video_title.clone(),
            timestamp: result.document.format_timestamp(),
            start_seconds: result.document.start_seconds,
            content: result.document.content.clone(),
            score: result.score,
            url: None, // Will be populated by the engine
        }
    }
}
