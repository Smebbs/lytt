//! Tool definitions and implementations for the agent system.

use crate::embedding::Embedder;
use crate::error::{LyttError, Result};
use crate::vector_store::VectorStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Available tools for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum ToolCall {
    /// Search the video knowledge base.
    Search {
        query: String,
        #[serde(default = "default_limit")]
        limit: u32,
    },

    /// Get full transcript for a video.
    GetTranscript { video_id: String },

    /// Get a specific time range from a video.
    GetSegment {
        video_id: String,
        start_seconds: f64,
        end_seconds: f64,
    },

    /// List all indexed videos.
    ListVideos,

    /// Get video metadata.
    GetVideoInfo { video_id: String },
}

fn default_limit() -> u32 {
    5
}

/// Tool execution context with access to vector store and embedder.
pub struct ToolContext {
    pub vector_store: Arc<dyn VectorStore>,
    pub embedder: Arc<dyn Embedder>,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(vector_store: Arc<dyn VectorStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            vector_store,
            embedder,
        }
    }

    /// Execute a tool call and return the result as a string.
    pub async fn execute(&self, tool: &ToolCall) -> Result<String> {
        match tool {
            ToolCall::Search { query, limit } => self.execute_search(query, *limit).await,
            ToolCall::GetTranscript { video_id } => self.execute_get_transcript(video_id).await,
            ToolCall::GetSegment {
                video_id,
                start_seconds,
                end_seconds,
            } => {
                self.execute_get_segment(video_id, *start_seconds, *end_seconds)
                    .await
            }
            ToolCall::ListVideos => self.execute_list_videos().await,
            ToolCall::GetVideoInfo { video_id } => self.execute_get_video_info(video_id).await,
        }
    }

    async fn execute_search(&self, query: &str, limit: u32) -> Result<String> {
        let embedding = self.embedder.embed(query).await?;
        let results = self
            .vector_store
            .search_with_threshold(&embedding, limit as usize, 0.3)
            .await?;

        if results.is_empty() {
            return Ok("No relevant results found.".to_string());
        }

        let formatted = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "{}. [{}] {} @ {}\n   {}",
                    i + 1,
                    r.document.video_id,
                    r.document.video_title,
                    r.document.format_timestamp(),
                    r.document.content.chars().take(500).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(format!("Found {} results:\n\n{}", results.len(), formatted))
    }

    async fn execute_get_transcript(&self, video_id: &str) -> Result<String> {
        let documents = self.vector_store.get_by_video_id(video_id).await?;

        if documents.is_empty() {
            return Err(LyttError::VideoNotFound(video_id.to_string()));
        }

        // Sort by chunk order and concatenate
        let mut sorted_docs = documents.clone();
        sorted_docs.sort_by_key(|d| d.chunk_order);

        let title = sorted_docs.first().map(|d| d.video_title.clone()).unwrap_or_default();
        let duration = sorted_docs.iter().map(|d| d.end_seconds).fold(0.0_f64, f64::max);

        let full_text = sorted_docs
            .iter()
            .map(|d| format!("[{}] {}", d.format_timestamp(), d.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(format!(
            "# {}\n\nDuration: {:.0} seconds\n\n{}",
            title, duration, full_text
        ))
    }

    async fn execute_get_segment(
        &self,
        video_id: &str,
        start_seconds: f64,
        end_seconds: f64,
    ) -> Result<String> {
        let documents = self.vector_store.get_by_video_id(video_id).await?;

        if documents.is_empty() {
            return Err(LyttError::VideoNotFound(video_id.to_string()));
        }

        // Filter documents that overlap with the requested time range
        let matching: Vec<_> = documents
            .iter()
            .filter(|d| d.start_seconds < end_seconds && d.end_seconds > start_seconds)
            .collect();

        if matching.is_empty() {
            return Ok(format!(
                "No content found between {} and {} seconds.",
                start_seconds, end_seconds
            ));
        }

        let content = matching
            .iter()
            .map(|d| {
                format!(
                    "[{} - {}] {}",
                    d.format_timestamp(),
                    format_seconds(d.end_seconds),
                    d.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(content)
    }

    async fn execute_list_videos(&self) -> Result<String> {
        let videos = self.vector_store.list_videos().await?;

        if videos.is_empty() {
            return Ok("No videos indexed yet.".to_string());
        }

        let formatted = videos
            .iter()
            .map(|v| {
                format!(
                    "- {} (ID: {}, {} chunks, {:.0}s)",
                    v.video_title, v.video_id, v.chunk_count, v.total_duration_seconds
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!("Indexed videos ({}):\n\n{}", videos.len(), formatted))
    }

    async fn execute_get_video_info(&self, video_id: &str) -> Result<String> {
        let video = self
            .vector_store
            .get_video(video_id)
            .await?
            .ok_or_else(|| LyttError::VideoNotFound(video_id.to_string()))?;

        Ok(format!(
            "Video: {}\nID: {}\nChunks: {}\nDuration: {:.0} seconds\nIndexed: {}",
            video.video_title,
            video.video_id,
            video.chunk_count,
            video.total_duration_seconds,
            video.indexed_at.format("%Y-%m-%d %H:%M:%S")
        ))
    }
}

/// Format seconds as timestamp string.
fn format_seconds(seconds: f64) -> String {
    let total = seconds as u32;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{:02}:{:02}", minutes, secs)
    }
}

/// Get OpenAI function/tool definitions for the agent.
pub fn tool_definitions() -> Vec<async_openai::types::ChatCompletionTool> {
    use async_openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};

    vec![
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "search".to_string(),
                description: Some(
                    "Search the video knowledge base for relevant content. \
                    Use this when you need to find specific information across all videos."
                        .to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                })),
                strict: None,
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "get_transcript".to_string(),
                description: Some(
                    "Get the full transcript of a video. Use this when you need complete context, \
                    like for summaries, quizzes, or comprehensive analysis."
                        .to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "The video ID"
                        }
                    },
                    "required": ["video_id"]
                })),
                strict: None,
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "get_segment".to_string(),
                description: Some(
                    "Get a specific time range from a video transcript. \
                    Use this when you need content from a particular part of a video."
                        .to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "The video ID"
                        },
                        "start_seconds": {
                            "type": "number",
                            "description": "Start time in seconds"
                        },
                        "end_seconds": {
                            "type": "number",
                            "description": "End time in seconds"
                        }
                    },
                    "required": ["video_id", "start_seconds", "end_seconds"]
                })),
                strict: None,
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "list_videos".to_string(),
                description: Some(
                    "List all indexed videos in the knowledge base. \
                    Use this to see what content is available."
                        .to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
                strict: None,
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "get_video_info".to_string(),
                description: Some(
                    "Get metadata about a specific video (title, duration, chunk count). \
                    Use this to get details about a particular video."
                        .to_string(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "The video ID"
                        }
                    },
                    "required": ["video_id"]
                })),
                strict: None,
            },
        },
    ]
}

/// Parse a tool call from the OpenAI response format.
pub fn parse_tool_call(name: &str, arguments: &str) -> Result<ToolCall> {
    // Parse the arguments JSON and construct the appropriate ToolCall variant
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|e| LyttError::Agent(format!("Invalid tool arguments: {}", e)))?;

    match name {
        "search" => {
            let query = args["query"]
                .as_str()
                .ok_or_else(|| LyttError::Agent("Missing 'query' argument".to_string()))?
                .to_string();
            let limit = args["limit"].as_u64().unwrap_or(5) as u32;
            Ok(ToolCall::Search { query, limit })
        }
        "get_transcript" => {
            let video_id = args["video_id"]
                .as_str()
                .ok_or_else(|| LyttError::Agent("Missing 'video_id' argument".to_string()))?
                .to_string();
            Ok(ToolCall::GetTranscript { video_id })
        }
        "get_segment" => {
            let video_id = args["video_id"]
                .as_str()
                .ok_or_else(|| LyttError::Agent("Missing 'video_id' argument".to_string()))?
                .to_string();
            let start_seconds = args["start_seconds"]
                .as_f64()
                .ok_or_else(|| LyttError::Agent("Missing 'start_seconds' argument".to_string()))?;
            let end_seconds = args["end_seconds"]
                .as_f64()
                .ok_or_else(|| LyttError::Agent("Missing 'end_seconds' argument".to_string()))?;
            Ok(ToolCall::GetSegment {
                video_id,
                start_seconds,
                end_seconds,
            })
        }
        "list_videos" => Ok(ToolCall::ListVideos),
        "get_video_info" => {
            let video_id = args["video_id"]
                .as_str()
                .ok_or_else(|| LyttError::Agent("Missing 'video_id' argument".to_string()))?
                .to_string();
            Ok(ToolCall::GetVideoInfo { video_id })
        }
        _ => Err(LyttError::Agent(format!("Unknown tool: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_search_tool() {
        let tool = parse_tool_call("search", r#"{"query": "authentication", "limit": 10}"#).unwrap();
        match tool {
            ToolCall::Search { query, limit } => {
                assert_eq!(query, "authentication");
                assert_eq!(limit, 10);
            }
            _ => panic!("Expected Search tool"),
        }
    }

    #[test]
    fn test_parse_get_transcript_tool() {
        let tool = parse_tool_call("get_transcript", r#"{"video_id": "abc123"}"#).unwrap();
        match tool {
            ToolCall::GetTranscript { video_id } => {
                assert_eq!(video_id, "abc123");
            }
            _ => panic!("Expected GetTranscript tool"),
        }
    }

    #[test]
    fn test_format_seconds() {
        assert_eq!(format_seconds(65.0), "01:05");
        assert_eq!(format_seconds(3665.0), "01:01:05");
        assert_eq!(format_seconds(0.0), "00:00");
    }
}
