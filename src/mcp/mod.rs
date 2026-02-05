//! MCP (Model Context Protocol) server for Lytt.
//!
//! Allows AI assistants like Claude to use Lytt as a tool.
//! Implements JSON-RPC 2.0 over stdio.

mod protocol;
mod server;
mod tools;

pub use server::McpServer;
