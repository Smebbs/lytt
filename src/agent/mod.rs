//! Agent system for intelligent task execution with tool calling.
//!
//! Provides an LLM agent that can use tools to interact with the video
//! knowledge base, enabling tasks like summarization, quiz generation,
//! and research across indexed content.

mod runner;
mod tools;

pub use runner::{Agent, AgentResponse, ToolCallRecord};
pub use tools::{parse_tool_call, tool_definitions, ToolCall, ToolContext};
