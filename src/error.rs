//! Error types for Lytt.

use thiserror::Error;

/// Library-level error type for Lytt operations.
#[derive(Error, Debug)]
pub enum LyttError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Media source error: {0}")]
    VideoSource(String),

    #[error("Audio download failed: {0}")]
    AudioDownload(String),

    #[error("Transcription failed: {0}")]
    Transcription(String),

    #[error("Embedding generation failed: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("RAG error: {0}")]
    Rag(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("OpenAI API error: {0}")]
    OpenAI(String),

    #[error("Media not found: {0}")]
    VideoNotFound(String),

    #[error("External tool not found: {0}. Please install it and ensure it's in your PATH.")]
    ToolNotFound(String),

    #[error("External tool failed: {0}")]
    ToolFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Agent error: {0}")]
    Agent(String),
}

/// Result type alias for Lytt operations.
pub type Result<T> = std::result::Result<T, LyttError>;
