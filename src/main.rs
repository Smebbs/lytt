//! Lytt CLI entry point.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use lytt::cli::{commands, Cli, Commands};
use lytt::config::Settings;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    tracing_subscriber::registry()
        .with(EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| format!("lytt={}", log_level)),
        ))
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    // Load configuration
    let settings = match &cli.config {
        Some(path) => Settings::load_from(Some(&std::path::PathBuf::from(path)))?,
        None => Settings::load()?,
    };

    // Ensure data directories exist
    std::fs::create_dir_all(settings.data_dir())?;
    std::fs::create_dir_all(settings.temp_dir())?;

    // Execute command
    match &cli.command {
        Commands::Init => {
            commands::run_init(&settings)?;
        }

        Commands::Doctor => {
            commands::run_doctor(&settings)?;
        }

        Commands::Transcribe { input, force, output, format, chunk, embed, playlist, limit } => {
            commands::run_transcribe(input, *force, output.clone(), format, *chunk, *embed, *playlist, *limit, settings).await?;
        }

        Commands::Ask {
            question,
            model,
            max_chunks,
        } => {
            commands::run_ask(question, model.clone(), *max_chunks, settings).await?;
        }

        Commands::Search {
            query,
            limit,
            min_score,
        } => {
            commands::run_search(query, *limit, *min_score, settings).await?;
        }

        Commands::Chat { model } => {
            commands::run_chat(model.clone(), settings).await?;
        }

        Commands::Agent { task, video, model } => {
            commands::run_agent(task, video.clone(), model.clone(), settings).await?;
        }

        Commands::List => {
            commands::run_list(settings).await?;
        }

        Commands::Rechunk { video_id } => {
            commands::run_rechunk(video_id, settings).await?;
        }

        Commands::Export { video_id, output, format } => {
            commands::run_export(video_id, output.clone(), format, settings).await?;
        }

        Commands::Serve { host, port } => {
            commands::run_serve(host, *port, settings).await?;
        }

        Commands::Mcp => {
            commands::run_mcp(settings).await?;
        }

        Commands::Config { action } => {
            commands::run_config(action, settings)?;
        }
    }

    Ok(())
}
