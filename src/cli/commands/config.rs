//! Config command implementation.

use crate::cli::{ConfigAction, Output};
use crate::config::Settings;
use anyhow::Result;

/// Run the config command.
pub fn run_config(action: &ConfigAction, settings: Settings) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let toml_str = toml::to_string_pretty(&settings)
                .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;
            println!("{}", toml_str);
        }

        ConfigAction::Set { key, value } => {
            Output::warning("Config set is not yet implemented.");
            Output::info(&format!("Would set {} = {}", key, value));
            Output::info("Please edit the config file directly for now.");
        }

        ConfigAction::Edit => {
            let config_path = Settings::default_config_path();

            // Create default config if it doesn't exist
            if !config_path.exists() {
                settings.save()?;
                Output::info(&format!("Created default config at {:?}", config_path));
            }

            // Try to open in editor
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vim".to_string());

            Output::info(&format!("Opening config in {}...", editor));

            let status = std::process::Command::new(&editor)
                .arg(&config_path)
                .status();

            match status {
                Ok(s) if s.success() => {
                    Output::success("Config saved.");
                }
                Ok(_) => {
                    Output::warning("Editor exited with non-zero status.");
                }
                Err(e) => {
                    Output::error(&format!("Failed to open editor: {}", e));
                    Output::info(&format!("Config file is at: {:?}", config_path));
                }
            }
        }

        ConfigAction::Path => {
            let config_path = Settings::default_config_path();
            println!("{}", config_path.display());
        }
    }

    Ok(())
}
