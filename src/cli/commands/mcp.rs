//! MCP command implementation.

use crate::config::Settings;
use crate::mcp::McpServer;
use anyhow::Result;

/// Run the MCP server.
pub async fn run_mcp(settings: Settings) -> Result<()> {
    let mut server = McpServer::new(settings);
    server.run().await
}
