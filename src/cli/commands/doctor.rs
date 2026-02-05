//! Doctor command - verify system requirements and configuration.

use crate::cli::Output;
use crate::config::Settings;
use console::style;
use std::process::Command;

/// Check result for a single item.
#[derive(Debug)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
    pub hint: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum CheckStatus {
    Ok,
    Warning,
    Error,
}

impl CheckResult {
    fn ok(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Ok,
            message: message.to_string(),
            hint: None,
        }
    }

    fn warning(name: &str, message: &str, hint: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Warning,
            message: message.to_string(),
            hint: Some(hint.to_string()),
        }
    }

    fn error(name: &str, message: &str, hint: &str) -> Self {
        Self {
            name: name.to_string(),
            status: CheckStatus::Error,
            message: message.to_string(),
            hint: Some(hint.to_string()),
        }
    }

    fn print(&self) {
        let icon = match self.status {
            CheckStatus::Ok => style("✓").green(),
            CheckStatus::Warning => style("!").yellow(),
            CheckStatus::Error => style("✗").red(),
        };

        println!("  {} {} - {}", icon, style(&self.name).bold(), self.message);

        if let Some(hint) = &self.hint {
            println!("    {} {}", style("→").dim(), style(hint).dim());
        }
    }
}

/// Run all diagnostic checks.
pub fn run_doctor(settings: &Settings) -> anyhow::Result<()> {
    Output::header("Lytt Doctor");
    println!();
    println!("Checking system requirements and configuration...\n");

    let mut checks = Vec::new();

    // Check external tools
    println!("{}", style("External Tools").bold());
    checks.push(check_tool("yt-dlp", "yt-dlp --version", install_hint_ytdlp()));
    checks.push(check_tool("ffmpeg", "ffmpeg -version", install_hint_ffmpeg()));
    checks.push(check_tool("ffprobe", "ffprobe -version", install_hint_ffmpeg()));
    for check in &checks[checks.len() - 3..] {
        check.print();
    }

    println!();

    // Check API keys
    println!("{}", style("API Configuration").bold());
    let api_check = check_openai_api_key();
    api_check.print();
    checks.push(api_check);

    println!();

    // Check directories
    println!("{}", style("Directories").bold());
    let dir_checks = check_directories(settings);
    for check in &dir_checks {
        check.print();
    }
    checks.extend(dir_checks);

    println!();

    // Check configuration
    println!("{}", style("Configuration").bold());
    let config_check = check_config_file();
    config_check.print();
    checks.push(config_check);

    println!();

    // Summary
    let errors = checks.iter().filter(|c| c.status == CheckStatus::Error).count();
    let warnings = checks.iter().filter(|c| c.status == CheckStatus::Warning).count();

    if errors > 0 {
        Output::error(&format!(
            "{} error(s) found. Please fix them before using Lytt.",
            errors
        ));
        std::process::exit(1);
    } else if warnings > 0 {
        Output::warning(&format!(
            "All checks passed with {} warning(s).",
            warnings
        ));
    } else {
        Output::success("All checks passed! Lytt is ready to use.");
    }

    Ok(())
}

/// Check if an external tool is available.
fn check_tool(name: &str, version_cmd: &str, hint: &str) -> CheckResult {
    let parts: Vec<&str> = version_cmd.split_whitespace().collect();
    let cmd = parts[0];
    let args = &parts[1..];

    match Command::new(cmd).args(args).output() {
        Ok(output) if output.status.success() => {
            // Try to extract version from first line
            let version = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("installed")
                .trim()
                .to_string();

            // Truncate long version strings
            let version_display = if version.len() > 50 {
                format!("{}...", &version[..50])
            } else {
                version
            };

            CheckResult::ok(name, &version_display)
        }
        Ok(_) => CheckResult::error(name, "installed but not working", hint),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            CheckResult::error(name, "not found", hint)
        }
        Err(e) => CheckResult::error(name, &format!("error: {}", e), hint),
    }
}

/// Check if OpenAI API key is configured.
fn check_openai_api_key() -> CheckResult {
    match std::env::var("OPENAI_API_KEY") {
        Ok(key) if key.starts_with("sk-") && key.len() > 20 => {
            let masked = format!("{}...{}", &key[..7], &key[key.len() - 4..]);
            CheckResult::ok("OPENAI_API_KEY", &format!("configured ({})", masked))
        }
        Ok(key) if key.is_empty() => CheckResult::error(
            "OPENAI_API_KEY",
            "empty",
            "Set with: export OPENAI_API_KEY='sk-...'",
        ),
        Ok(_) => CheckResult::warning(
            "OPENAI_API_KEY",
            "set but format looks unusual",
            "Expected format: sk-... (OpenAI API key)",
        ),
        Err(_) => CheckResult::error(
            "OPENAI_API_KEY",
            "not set",
            "Set with: export OPENAI_API_KEY='sk-...'",
        ),
    }
}

/// Check data directories.
fn check_directories(settings: &Settings) -> Vec<CheckResult> {
    let mut results = Vec::new();

    let data_dir = settings.data_dir();
    if data_dir.exists() {
        results.push(CheckResult::ok(
            "Data directory",
            &format!("{}", data_dir.display()),
        ));
    } else {
        results.push(CheckResult::warning(
            "Data directory",
            &format!("{} (will be created)", data_dir.display()),
            "Directory will be created on first use",
        ));
    }

    let db_path = settings.sqlite_path();
    if db_path.exists() {
        let size = std::fs::metadata(&db_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "unknown size".to_string());
        results.push(CheckResult::ok(
            "Database",
            &format!("{} ({})", db_path.display(), size),
        ));
    } else {
        results.push(CheckResult::warning(
            "Database",
            &format!("{} (not created yet)", db_path.display()),
            "Database will be created on first transcription",
        ));
    }

    results
}

/// Check if config file exists.
fn check_config_file() -> CheckResult {
    let config_path = Settings::default_config_path();
    if config_path.exists() {
        CheckResult::ok("Config file", &format!("{}", config_path.display()))
    } else {
        CheckResult::warning(
            "Config file",
            "using defaults",
            "Create with: lytt init (or lytt config edit)",
        )
    }
}

/// Format file size in human-readable format.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Platform-specific install hint for yt-dlp.
fn install_hint_ytdlp() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install with: brew install yt-dlp"
    } else if cfg!(target_os = "linux") {
        "Install with: pip install yt-dlp (or your package manager)"
    } else {
        "Install from: https://github.com/yt-dlp/yt-dlp"
    }
}

/// Platform-specific install hint for ffmpeg.
fn install_hint_ffmpeg() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install with: brew install ffmpeg"
    } else if cfg!(target_os = "linux") {
        "Install with: sudo apt install ffmpeg (or your package manager)"
    } else {
        "Install from: https://ffmpeg.org/download.html"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_ok() {
        let result = CheckResult::ok("test", "passed");
        assert_eq!(result.status, CheckStatus::Ok);
        assert!(result.hint.is_none());
    }

    #[test]
    fn test_check_result_error() {
        let result = CheckResult::error("test", "failed", "fix it");
        assert_eq!(result.status, CheckStatus::Error);
        assert_eq!(result.hint, Some("fix it".to_string()));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }
}
