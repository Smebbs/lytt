//! Configuration settings for Lytt.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Settings {
    pub general: GeneralSettings,
    pub transcription: TranscriptionSettings,
    pub embedding: EmbeddingSettings,
    pub chunking: ChunkingSettings,
    pub vector_store: VectorStoreSettings,
    pub youtube: YoutubeSettings,
    pub rag: RagSettings,
    pub prompts: PromptSettings,
}


/// General application settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralSettings {
    /// Directory for storing application data.
    pub data_dir: String,
    /// Directory for temporary files.
    pub temp_dir: String,
    /// Log level (trace, debug, info, warn, error).
    pub log_level: String,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            data_dir: "~/.lytt".to_string(),
            temp_dir: "/tmp/lytt".to_string(),
            log_level: "info".to_string(),
        }
    }
}

/// Transcription provider type.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptionProvider {
    /// OpenAI Whisper (default, single model).
    #[default]
    Whisper,
    /// Fusion mode: Whisper timestamps + GPT-4o text + LLM fusion.
    Fusion,
}

impl std::str::FromStr for TranscriptionProvider {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "whisper" | "openai" => Ok(TranscriptionProvider::Whisper),
            "fusion" => Ok(TranscriptionProvider::Fusion),
            _ => Err(format!("Unknown transcription provider: {}", s)),
        }
    }
}

impl std::fmt::Display for TranscriptionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranscriptionProvider::Whisper => write!(f, "whisper"),
            TranscriptionProvider::Fusion => write!(f, "fusion"),
        }
    }
}

/// Settings for the transcription processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscriptionProcessingSettings {
    /// Model for word-level timestamps (default: whisper-1).
    pub timestamp_model: String,
    /// Model for secondary text transcription. None = Whisper-only with cleanup.
    pub text_model: Option<String>,
    /// Model for LLM cleanup and segment structuring (default: gpt-4.1).
    pub cleanup_model: String,
    /// Maximum concurrent API calls.
    pub max_concurrent: usize,
}

impl Default for TranscriptionProcessingSettings {
    fn default() -> Self {
        Self {
            timestamp_model: "whisper-1".to_string(),
            text_model: Some("gpt-4o-transcribe".to_string()),
            cleanup_model: "gpt-4.1".to_string(),
            max_concurrent: 2,
        }
    }
}

impl TranscriptionProcessingSettings {
    /// Check if secondary text model is enabled (full fusion mode).
    pub fn has_text_model(&self) -> bool {
        self.text_model.as_ref().is_some_and(|m| !m.is_empty())
    }
}

/// Transcription service settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscriptionSettings {
    /// Transcription provider (whisper, fusion).
    pub provider: TranscriptionProvider,
    /// Whisper model to use (for whisper provider or fusion timestamps).
    pub model: String,
    /// Duration in seconds for splitting long audio files.
    pub chunk_duration_seconds: u32,
    /// Maximum media duration to process (in seconds).
    pub max_duration_seconds: u32,
    /// Maximum concurrent chunk processing.
    pub max_concurrent_chunks: usize,
    /// Processing pipeline settings (cleanup model, text model, etc.).
    pub processing: TranscriptionProcessingSettings,
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        Self {
            provider: TranscriptionProvider::Whisper,
            model: "whisper-1".to_string(),
            chunk_duration_seconds: 120,
            max_duration_seconds: 7200, // 2 hours
            max_concurrent_chunks: 3,
            processing: TranscriptionProcessingSettings::default(),
        }
    }
}

/// Embedding generation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingSettings {
    /// Embedding provider (openai).
    pub provider: String,
    /// Embedding model to use.
    pub model: String,
    /// Embedding dimensions.
    pub dimensions: u32,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            dimensions: 1536,
        }
    }
}

/// Content chunking settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingSettings {
    /// Chunking strategy (semantic, temporal, hybrid).
    pub strategy: String,
    /// Target chunk duration in seconds.
    pub target_chunk_seconds: u32,
    /// Minimum chunk duration in seconds.
    pub min_chunk_seconds: u32,
    /// Maximum chunk duration in seconds.
    pub max_chunk_seconds: u32,
    /// Model to use for semantic chunking.
    pub model: String,
}

impl Default for ChunkingSettings {
    fn default() -> Self {
        Self {
            strategy: "semantic".to_string(),
            target_chunk_seconds: 180,
            min_chunk_seconds: 60,
            max_chunk_seconds: 600,
            model: "gpt-4o-mini".to_string(),
        }
    }
}

/// Vector store settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorStoreSettings {
    /// Vector store provider (sqlite, memory).
    pub provider: String,
    /// Path to SQLite database (for sqlite provider).
    pub sqlite_path: String,
}

impl Default for VectorStoreSettings {
    fn default() -> Self {
        Self {
            provider: "sqlite".to_string(),
            sqlite_path: "~/.lytt/vectors.db".to_string(),
        }
    }
}

/// YouTube-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct YoutubeSettings {
    /// YouTube Data API key (optional, for playlist/channel fetching).
    pub api_key: Option<String>,
}


/// RAG (Retrieval-Augmented Generation) settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RagSettings {
    /// Enable RAG responses.
    pub enabled: bool,
    /// LLM model for response generation.
    pub model: String,
    /// Maximum number of context chunks to include.
    pub max_context_chunks: u32,
    /// Include video timestamps in citations.
    pub include_timestamps: bool,
}

impl Default for RagSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "gpt-4o-mini".to_string(),
            max_context_chunks: 10,
            include_timestamps: true,
        }
    }
}

/// Prompt customization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct PromptSettings {
    /// Directory for custom prompts (overrides defaults).
    pub custom_dir: Option<String>,
    /// Custom variables available in all prompts as {{variable_name}}.
    pub variables: std::collections::HashMap<String, String>,
}


impl Settings {
    /// Load settings from the default configuration file.
    pub fn load() -> crate::error::Result<Self> {
        Self::load_from(None)
    }

    /// Load settings from a specific path, or default location if None.
    pub fn load_from(path: Option<&PathBuf>) -> crate::error::Result<Self> {
        let config_path = match path {
            Some(p) => p.clone(),
            None => Self::default_config_path(),
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let settings: Settings = toml::from_str(&content)?;
            Ok(settings)
        } else {
            Ok(Settings::default())
        }
    }

    /// Save settings to the default configuration file.
    pub fn save(&self) -> crate::error::Result<()> {
        self.save_to(&Self::default_config_path())
    }

    /// Save settings to a specific path.
    pub fn save_to(&self, path: &PathBuf) -> crate::error::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)
            .map_err(|e| crate::error::LyttError::Config(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the default configuration file path.
    pub fn default_config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("lytt")
            .join("config.toml")
    }

    /// Expand shell variables in paths (e.g., ~).
    pub fn expand_path(path: &str) -> PathBuf {
        PathBuf::from(shellexpand::tilde(path).to_string())
    }

    /// Get the expanded data directory path.
    pub fn data_dir(&self) -> PathBuf {
        Self::expand_path(&self.general.data_dir)
    }

    /// Get the expanded temp directory path.
    pub fn temp_dir(&self) -> PathBuf {
        Self::expand_path(&self.general.temp_dir)
    }

    /// Get the expanded SQLite database path.
    pub fn sqlite_path(&self) -> PathBuf {
        Self::expand_path(&self.vector_store.sqlite_path)
    }
}
