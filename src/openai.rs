//! OpenAI client configuration with sensible defaults.

use async_openai::{config::OpenAIConfig, Client};
use std::time::Duration;

/// Default timeout for OpenAI API requests (5 minutes).
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Create an OpenAI client with configured timeout.
///
/// Uses a 5-minute timeout by default to prevent hung API calls.
pub fn create_client() -> Client<OpenAIConfig> {
    create_client_with_timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
}

/// Create an OpenAI client with a custom timeout.
pub fn create_client_with_timeout(timeout: Duration) -> Client<OpenAIConfig> {
    let http_client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("Failed to create HTTP client");

    Client::with_config(OpenAIConfig::default()).with_http_client(http_client)
}
