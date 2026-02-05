//! Pre-flight checks before expensive operations.
//!
//! Validates that required tools and configuration are available
//! before starting operations that would otherwise fail midway.

use crate::error::{Result, LyttError};
use std::process::Command;

/// Requirements for different operations.
#[derive(Debug, Clone, Copy)]
pub enum Operation {
    /// Transcription requires tools and API key.
    Transcribe,
    /// Asking questions requires API key and database.
    Ask,
    /// Search requires database.
    Search,
}

/// Run pre-flight checks for the given operation.
///
/// Returns Ok(()) if all checks pass, or an error describing what's missing.
pub fn check(operation: Operation) -> Result<()> {
    match operation {
        Operation::Transcribe => {
            check_api_key()?;
            check_tool("yt-dlp")?;
            check_tool("ffmpeg")?;
            check_tool("ffprobe")?;
        }
        Operation::Ask => {
            check_api_key()?;
        }
        Operation::Search => {
            // No external requirements for search
        }
    }
    Ok(())
}

/// Check if OpenAI API key is configured.
fn check_api_key() -> Result<()> {
    match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => Ok(()),
        Ok(_) => Err(LyttError::Config(
            "OPENAI_API_KEY is empty. Set it with: export OPENAI_API_KEY='sk-...'".to_string(),
        )),
        Err(_) => Err(LyttError::Config(
            "OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY='sk-...'".to_string(),
        )),
    }
}

/// Check if an external tool is available.
fn check_tool(name: &str) -> Result<()> {
    // ffmpeg/ffprobe use -version (single dash), others use --version
    let version_arg = match name {
        "ffmpeg" | "ffprobe" => "-version",
        _ => "--version",
    };
    match Command::new(name).arg(version_arg).output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(_) => Err(LyttError::ToolNotFound(format!(
            "{} is installed but not working correctly",
            name
        ))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(LyttError::ToolNotFound(name.to_string()))
        }
        Err(e) => Err(LyttError::ToolNotFound(format!(
            "{}: {}",
            name, e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_search_no_requirements() {
        // Search should always pass pre-flight (no external requirements)
        assert!(check(Operation::Search).is_ok());
    }
}
