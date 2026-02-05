//! CLI output formatting utilities.

use console::{style, Style};
use indicatif::{ProgressBar, ProgressStyle};

/// Output helper for CLI formatting.
pub struct Output;

impl Output {
    /// Print an info message.
    pub fn info(msg: &str) {
        println!("{} {}", style(">>").cyan().bold(), msg);
    }

    /// Print a success message.
    pub fn success(msg: &str) {
        println!("{} {}", style(">>").green().bold(), msg);
    }

    /// Print a warning message.
    pub fn warning(msg: &str) {
        eprintln!("{} {}", style(">>").yellow().bold(), msg);
    }

    /// Print an error message.
    pub fn error(msg: &str) {
        eprintln!("{} {}", style(">>").red().bold(), msg);
    }

    /// Print a header.
    pub fn header(msg: &str) {
        println!("\n{}", style(msg).bold().underlined());
    }

    /// Print a key-value pair.
    pub fn kv(key: &str, value: &str) {
        println!("  {}: {}", style(key).dim(), value);
    }

    /// Print a list item.
    pub fn list_item(msg: &str) {
        println!("  {} {}", style("*").cyan(), msg);
    }

    /// Print media info.
    pub fn media_info(title: &str, id: &str, chunks: u32, duration: f64) {
        let duration_str = format_duration(duration);
        println!(
            "  {} {} ({}, {} chunks, {})",
            style("*").cyan(),
            style(title).bold(),
            style(id).dim(),
            chunks,
            duration_str
        );
    }

    /// Print search result.
    pub fn search_result(title: &str, timestamp: &str, score: f32, content: &str, url: Option<&str>) {
        println!(
            "\n{} {} @ {} (score: {:.2})",
            style(">>").green(),
            style(title).bold(),
            style(timestamp).cyan(),
            score
        );
        println!("   {}", content_preview(content, 200));
        if let Some(u) = url {
            println!("   {}", style(u).dim());
        }
    }

    /// Create a progress bar.
    pub fn progress_bar(len: u64, msg: &str) -> ProgressBar {
        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(msg.to_string());
        pb
    }

    /// Create a spinner.
    pub fn spinner(msg: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message(msg.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    }

    /// Style for titles.
    pub fn title_style() -> Style {
        Style::new().bold()
    }

    /// Style for dim text.
    pub fn dim_style() -> Style {
        Style::new().dim()
    }
}

/// Format duration in seconds to a human-readable string.
fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u32;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Truncate content with ellipsis.
fn content_preview(content: &str, max_len: usize) -> String {
    let content = content.replace('\n', " ");
    if content.len() <= max_len {
        content
    } else {
        format!("{}...", &content[..max_len])
    }
}
