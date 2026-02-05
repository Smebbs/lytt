//! MCP server implementation.

use super::protocol::*;
use super::tools::get_tools;
use crate::config::Settings;
use crate::embedding::{Embedder, OpenAIEmbedder};
use crate::orchestrator::Orchestrator;
use crate::rag::RagEngine;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "lytt";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP Server for Lytt.
pub struct McpServer {
    settings: Settings,
    orchestrator: Option<Orchestrator>,
}

impl McpServer {
    /// Create a new MCP server.
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            orchestrator: None,
        }
    }

    /// Run the MCP server (reads from stdin, writes to stdout).
    pub async fn run(&mut self) -> anyhow::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        // Log to stderr so it doesn't interfere with JSON-RPC
        eprintln!("Lytt MCP server starting...");

        for line in stdin.lock().lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("Failed to parse request: {}", e);
                    let response = JsonRpcResponse::error(None, -32700, "Parse error");
                    writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(request).await;
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }

        Ok(())
    }

    /// Handle a single JSON-RPC request.
    async fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id, request.params),
            "initialized" => {
                // Notification, no response needed but we'll send empty success
                JsonRpcResponse::success(request.id, json!({}))
            }
            "tools/list" => self.handle_tools_list(request.id),
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            _ => JsonRpcResponse::error(
                request.id,
                -32601,
                &format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle initialize request.
    fn handle_initialize(&mut self, id: Option<Value>, _params: Option<Value>) -> JsonRpcResponse {
        // Initialize the orchestrator lazily
        match Orchestrator::new(self.settings.clone()) {
            Ok(orch) => {
                self.orchestrator = Some(orch);
                eprintln!("Orchestrator initialized");
            }
            Err(e) => {
                eprintln!("Failed to initialize orchestrator: {}", e);
                return JsonRpcResponse::error(id, -32000, &format!("Init failed: {}", e));
            }
        }

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability { list_changed: false },
            },
            server_info: ServerInfo {
                name: SERVER_NAME.to_string(),
                version: SERVER_VERSION.to_string(),
            },
        };

        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
    }

    /// Handle tools/list request.
    fn handle_tools_list(&self, id: Option<Value>) -> JsonRpcResponse {
        let result = ToolsListResult { tools: get_tools() };
        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
    }

    /// Handle tools/call request.
    async fn handle_tools_call(
        &self,
        id: Option<Value>,
        params: Option<Value>,
    ) -> JsonRpcResponse {
        let params: ToolCallParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(params) => params,
                Err(e) => {
                    return JsonRpcResponse::error(id, -32602, &format!("Invalid params: {}", e))
                }
            },
            None => return JsonRpcResponse::error(id, -32602, "Missing params"),
        };

        let result = match params.name.as_str() {
            "transcribe" => self.tool_transcribe(params.arguments).await,
            "search" => self.tool_search(params.arguments).await,
            "ask" => self.tool_ask(params.arguments).await,
            "list_media" => self.tool_list_media().await,
            "get_transcript" => self.tool_get_transcript(params.arguments).await,
            _ => ToolCallResult::error(format!("Unknown tool: {}", params.name)),
        };

        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
    }

    /// Transcribe tool.
    async fn tool_transcribe(&self, args: Option<Value>) -> ToolCallResult {
        let args = match args {
            Some(a) => a,
            None => return ToolCallResult::error("Missing arguments".to_string()),
        };

        let input = match args.get("input").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => return ToolCallResult::error("Missing 'input' argument".to_string()),
        };

        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        let orchestrator = match &self.orchestrator {
            Some(o) => o,
            None => return ToolCallResult::error("Server not initialized".to_string()),
        };

        match orchestrator.process_media(input, force).await {
            Ok(result) => {
                if result.skipped {
                    ToolCallResult::text(format!(
                        "'{}' is already indexed ({} chunks). Use force=true to reprocess.",
                        result.title, result.chunks_indexed
                    ))
                } else {
                    ToolCallResult::text(format!(
                        "Successfully transcribed and indexed '{}' ({} chunks created).",
                        result.title, result.chunks_indexed
                    ))
                }
            }
            Err(e) => ToolCallResult::error(format!("Transcription failed: {}", e)),
        }
    }

    /// Search tool.
    async fn tool_search(&self, args: Option<Value>) -> ToolCallResult {
        let args = match args {
            Some(a) => a,
            None => return ToolCallResult::error("Missing arguments".to_string()),
        };

        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolCallResult::error("Missing 'query' argument".to_string()),
        };

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let min_score = args.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;

        let orchestrator = match &self.orchestrator {
            Some(o) => o,
            None => return ToolCallResult::error("Server not initialized".to_string()),
        };

        // Generate query embedding
        let embedder = OpenAIEmbedder::with_config(
            &self.settings.embedding.model,
            self.settings.embedding.dimensions as usize,
        );

        let query_embedding = match embedder.embed(query).await {
            Ok(emb) => emb,
            Err(e) => return ToolCallResult::error(format!("Embedding failed: {}", e)),
        };

        // Search
        match orchestrator
            .vector_store()
            .search_with_threshold(&query_embedding, limit, min_score)
            .await
        {
            Ok(results) => {
                if results.is_empty() {
                    return ToolCallResult::text("No matching results found.".to_string());
                }

                let mut output = format!("Found {} results:\n\n", results.len());
                for (i, result) in results.iter().enumerate() {
                    output.push_str(&format!(
                        "{}. **{}** @ {}\n   Score: {:.2}\n   {}\n\n",
                        i + 1,
                        result.document.video_title,
                        result.document.format_timestamp(),
                        result.score,
                        truncate(&result.document.content, 200)
                    ));
                }

                ToolCallResult::text(output)
            }
            Err(e) => ToolCallResult::error(format!("Search failed: {}", e)),
        }
    }

    /// Ask tool (RAG).
    async fn tool_ask(&self, args: Option<Value>) -> ToolCallResult {
        let args = match args {
            Some(a) => a,
            None => return ToolCallResult::error("Missing arguments".to_string()),
        };

        let question = match args.get("question").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolCallResult::error("Missing 'question' argument".to_string()),
        };

        let max_chunks = args
            .get("max_chunks")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let orchestrator = match &self.orchestrator {
            Some(o) => o,
            None => return ToolCallResult::error("Server not initialized".to_string()),
        };

        let embedder = Arc::new(OpenAIEmbedder::with_config(
            &self.settings.embedding.model,
            self.settings.embedding.dimensions as usize,
        ));

        let engine = RagEngine::new(
            orchestrator.vector_store(),
            embedder,
            &self.settings.rag.model,
            max_chunks,
        );

        match engine.ask(question).await {
            Ok(response) => {
                let mut output = format!("{}\n\n", response.answer);

                if !response.sources.is_empty() {
                    output.push_str("**Sources:**\n");
                    for source in &response.sources {
                        output.push_str(&format!(
                            "- {} @ {} (score: {:.2})\n",
                            source.video_title, source.timestamp, source.score
                        ));
                    }
                }

                ToolCallResult::text(output)
            }
            Err(e) => ToolCallResult::error(format!("RAG query failed: {}", e)),
        }
    }

    /// List media tool.
    async fn tool_list_media(&self) -> ToolCallResult {
        let orchestrator = match &self.orchestrator {
            Some(o) => o,
            None => return ToolCallResult::error("Server not initialized".to_string()),
        };

        match orchestrator.vector_store().list_videos().await {
            Ok(media) => {
                if media.is_empty() {
                    return ToolCallResult::text(
                        "No media indexed yet. Use the transcribe tool to add content."
                            .to_string(),
                    );
                }

                let mut output = format!("Indexed media ({} items):\n\n", media.len());
                for item in &media {
                    let duration = format_duration(item.total_duration_seconds);
                    output.push_str(&format!(
                        "- **{}** (ID: {})\n  {} chunks, {}\n\n",
                        item.video_title, item.video_id, item.chunk_count, duration
                    ));
                }

                ToolCallResult::text(output)
            }
            Err(e) => ToolCallResult::error(format!("Failed to list media: {}", e)),
        }
    }

    /// Get transcript tool.
    async fn tool_get_transcript(&self, args: Option<Value>) -> ToolCallResult {
        let args = match args {
            Some(a) => a,
            None => return ToolCallResult::error("Missing arguments".to_string()),
        };

        let video_id = match args.get("video_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolCallResult::error("Missing 'video_id' argument".to_string()),
        };

        let orchestrator = match &self.orchestrator {
            Some(o) => o,
            None => return ToolCallResult::error("Server not initialized".to_string()),
        };

        match orchestrator.vector_store().get_by_video_id(video_id).await {
            Ok(chunks) if chunks.is_empty() => {
                ToolCallResult::error(format!("No transcript found for video ID: {}", video_id))
            }
            Ok(mut chunks) => {
                // Sort by start time
                chunks.sort_by(|a, b| a.start_seconds.partial_cmp(&b.start_seconds).unwrap());

                let title = chunks.first().map(|c| c.video_title.clone()).unwrap_or_default();
                let mut output = format!("**{}**\n\n", title);

                for chunk in &chunks {
                    let timestamp = chunk.format_timestamp();
                    let section = chunk.section_title.as_deref().unwrap_or("Segment");
                    output.push_str(&format!(
                        "[{}] **{}**\n{}\n\n",
                        timestamp, section, chunk.content
                    ));
                }

                ToolCallResult::text(output)
            }
            Err(e) => ToolCallResult::error(format!("Failed to get transcript: {}", e)),
        }
    }
}

/// Truncate text with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Format duration as human-readable string.
fn format_duration(seconds: f64) -> String {
    let total = seconds as u32;
    let hours = total / 3600;
    let mins = (total % 3600) / 60;
    let secs = total % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, mins, secs)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs)
    } else {
        format!("{}s", secs)
    }
}
