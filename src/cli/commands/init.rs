//! Init command - interactive first-run setup.

use crate::cli::Output;
use crate::config::Settings;
use console::style;
use std::io::{self, Write};

/// Simple check result for init command.
struct CheckIssue {
    name: String,
    hint: String,
}

/// Run the init command for first-time setup.
pub fn run_init(settings: &Settings) -> anyhow::Result<()> {
    Output::header("Lytt Setup");
    println!();
    println!("Welcome to Lytt! Let's make sure everything is configured correctly.\n");

    // Step 1: Check prerequisites
    println!("{}", style("Step 1: Checking prerequisites").bold().cyan());
    println!();

    let tool_issues = check_prerequisites();

    if !tool_issues.is_empty() {
        Output::warning("Some tools are missing. Please install them:");
        println!();
        for issue in &tool_issues {
            println!("  {} {} - not found", style("✗").red(), style(&issue.name).bold());
            println!("    {} {}", style("→").dim(), style(&issue.hint).dim());
        }
        println!();

        if !prompt_continue("Continue anyway?")? {
            println!();
            Output::info("Setup cancelled. Install the missing tools and run 'lytt init' again.");
            return Ok(());
        }
    } else {
        Output::success("All required tools are installed!");
    }

    println!();

    // Step 2: Check API key
    println!("{}", style("Step 2: Checking API configuration").bold().cyan());
    println!();

    if std::env::var("OPENAI_API_KEY").is_err() {
        Output::warning("OPENAI_API_KEY environment variable is not set.");
        println!();
        println!("  Lytt requires an OpenAI API key for transcription and embeddings.");
        println!("  Get your API key from: {}", style("https://platform.openai.com/api-keys").underlined());
        println!();
        println!("  Set it in your shell configuration (~/.bashrc, ~/.zshrc, etc.):");
        println!("  {}", style("export OPENAI_API_KEY='sk-...'").green());
        println!();

        if !prompt_continue("Continue without API key?")? {
            println!();
            Output::info("Setup cancelled. Set your API key and run 'lytt init' again.");
            return Ok(());
        }
    } else {
        Output::success("OpenAI API key is configured!");
    }

    println!();

    // Step 3: Create directories
    println!("{}", style("Step 3: Setting up directories").bold().cyan());
    println!();

    let data_dir = settings.data_dir();
    let temp_dir = settings.temp_dir();

    if !data_dir.exists() {
        std::fs::create_dir_all(&data_dir)?;
        Output::success(&format!("Created data directory: {}", data_dir.display()));
    } else {
        Output::info(&format!("Data directory exists: {}", data_dir.display()));
    }

    if !temp_dir.exists() {
        std::fs::create_dir_all(&temp_dir)?;
        Output::success(&format!("Created temp directory: {}", temp_dir.display()));
    } else {
        Output::info(&format!("Temp directory exists: {}", temp_dir.display()));
    }

    println!();

    // Step 4: Create config file
    println!("{}", style("Step 4: Configuration file").bold().cyan());
    println!();

    let config_path = Settings::default_config_path();
    if config_path.exists() {
        Output::info(&format!("Config file exists: {}", config_path.display()));
    } else if prompt_continue("Create default configuration file?")? {
        // Create parent directory if needed
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        settings.save_to(&config_path)?;
        Output::success(&format!("Created config file: {}", config_path.display()));
        println!();
        println!("  Edit your config with: {}", style("lytt config edit").green());
    } else {
        Output::info("Skipped config file creation. Using defaults.");
    }

    println!();

    // Summary
    println!("{}", style("Setup Complete!").bold().green());
    println!();
    println!("Next steps:");
    println!("  {} Check system status", style("lytt doctor").cyan());
    println!("  {} Transcribe your first video", style("lytt transcribe <url>").cyan());
    println!("  {} Ask questions about your content", style("lytt ask \"<question>\"").cyan());
    println!();
    println!("For more help: {}", style("lytt --help").cyan());

    Ok(())
}

/// Check prerequisites and return any issues.
fn check_prerequisites() -> Vec<CheckIssue> {
    use std::process::Command;

    let mut issues = Vec::new();

    // Check yt-dlp
    if Command::new("yt-dlp").arg("--version").output().is_err() {
        issues.push(CheckIssue {
            name: "yt-dlp".to_string(),
            hint: install_hint("yt-dlp").to_string(),
        });
    }

    // Check ffmpeg
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        issues.push(CheckIssue {
            name: "ffmpeg".to_string(),
            hint: install_hint("ffmpeg").to_string(),
        });
    }

    // Check ffprobe
    if Command::new("ffprobe").arg("-version").output().is_err() {
        issues.push(CheckIssue {
            name: "ffprobe".to_string(),
            hint: install_hint("ffprobe").to_string(),
        });
    }

    issues
}

/// Get platform-specific install hint.
fn install_hint(tool: &str) -> &'static str {
    match tool {
        "yt-dlp" => {
            if cfg!(target_os = "macos") {
                "Install with: brew install yt-dlp"
            } else if cfg!(target_os = "linux") {
                "Install with: pip install yt-dlp"
            } else {
                "Install from: https://github.com/yt-dlp/yt-dlp"
            }
        }
        "ffmpeg" | "ffprobe" => {
            if cfg!(target_os = "macos") {
                "Install with: brew install ffmpeg"
            } else if cfg!(target_os = "linux") {
                "Install with: sudo apt install ffmpeg"
            } else {
                "Install from: https://ffmpeg.org/download.html"
            }
        }
        _ => "Check the documentation for installation instructions",
    }
}

/// Prompt user for yes/no confirmation.
fn prompt_continue(message: &str) -> io::Result<bool> {
    print!("{} {} ", style("?").cyan(), message);
    print!("{} ", style("[y/N]").dim());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_hint_ytdlp() {
        let hint = install_hint("yt-dlp");
        assert!(hint.contains("yt-dlp"));
    }

    #[test]
    fn test_install_hint_ffmpeg() {
        let hint = install_hint("ffmpeg");
        assert!(hint.contains("ffmpeg"));
    }
}
