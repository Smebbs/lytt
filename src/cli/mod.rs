//! CLI module for Lytt.

pub mod commands;
mod output;
pub mod preflight;

pub use output::Output;

use clap::{Parser, Subcommand};

/// Lytt - Audio Transcription and RAG
///
/// A local-first CLI tool for transcribing audio and building a searchable knowledge base.
/// The name "Lytt" comes from the Norwegian/Scandinavian word for "listen."
#[derive(Parser, Debug)]
#[command(name = "lytt")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Increase verbosity (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Path to configuration file
    #[arg(short, long, global = true)]
    pub config: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize Lytt and verify system requirements
    Init,

    /// Check system requirements and configuration
    Doctor,

    /// Transcribe and index audio/video content
    Transcribe {
        /// YouTube URL/ID, or local audio/video file path
        input: String,

        /// Force re-processing even if already indexed
        #[arg(short, long)]
        force: bool,

        /// Output transcript to file instead of indexing (for integration with other systems)
        #[arg(short, long)]
        output: Option<String>,

        /// Output format when using --output (json, srt, vtt)
        #[arg(long, default_value = "json")]
        format: String,

        /// Apply semantic chunking to output (use with --output for RAG integration)
        #[arg(long)]
        chunk: bool,

        /// Include embeddings in output (requires --chunk, for vector DB import)
        #[arg(long)]
        embed: bool,

        /// Treat input as a playlist/channel URL and transcribe all videos
        #[arg(long)]
        playlist: bool,

        /// Maximum number of videos to transcribe from playlist (default: all)
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Ask a question and get an answer from your audio library
    Ask {
        /// The question to ask
        question: String,

        /// LLM model to use for response generation
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum number of context chunks to include
        #[arg(short = 'c', long, default_value = "10")]
        max_chunks: usize,
    },

    /// Search for relevant audio segments
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Minimum similarity score (0.0-1.0)
        #[arg(short, long, default_value = "0.3")]
        min_score: f32,
    },

    /// Start an interactive chat session
    Chat {
        /// LLM model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Run an AI agent to perform tasks (summarize, create quiz, research, etc.)
    Agent {
        /// The task for the agent to perform (e.g., "Summarize the main points")
        task: String,

        /// Focus on a specific video (optional)
        #[arg(short, long)]
        video: Option<String>,

        /// LLM model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// List indexed media
    List,

    /// Rechunk indexed media without re-transcribing
    Rechunk {
        /// Video ID to rechunk (use 'all' to rechunk everything)
        video_id: String,
    },

    /// Export transcript from indexed media
    Export {
        /// Video ID to export
        video_id: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,

        /// Output format (json, srt, vtt)
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Start HTTP API server for integration with other systems
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },

    /// Start MCP server for AI assistant integration (Claude, etc.)
    Mcp,

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key (e.g., "rag.model")
        key: String,
        /// Configuration value
        value: String,
    },

    /// Open configuration file in editor
    Edit,

    /// Show configuration file path
    Path,
}
