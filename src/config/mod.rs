//! Configuration module for Lytt.
//!
//! Handles loading and managing application settings and prompt templates.

mod prompts;
mod settings;

pub use prompts::{ChunkingPrompts, CleanupPrompts, Prompts, RagPrompts};
pub use settings::{
    ChunkingSettings, EmbeddingSettings, GeneralSettings, PromptSettings,
    RagSettings, Settings, TranscriptionProcessingSettings, TranscriptionProvider,
    TranscriptionSettings, VectorStoreSettings, YoutubeSettings,
};
