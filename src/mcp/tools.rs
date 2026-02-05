//! MCP tool definitions for Lytt.

use super::protocol::Tool;
use serde_json::json;

/// Get all available tools.
pub fn get_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "transcribe".to_string(),
            description: "Transcribe audio or video content from a YouTube URL or local file. \
                Returns the transcript with timestamps. Use this to add new content to the knowledge base."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "YouTube URL, video ID, or local file path"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-processing even if already indexed",
                        "default": false
                    }
                },
                "required": ["input"]
            }),
        },
        Tool {
            name: "search".to_string(),
            description: "Search the audio knowledge base for relevant content. \
                Returns matching segments with timestamps and relevance scores."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.3
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "ask".to_string(),
            description: "Ask a question and get an AI-generated answer based on the audio knowledge base. \
                The answer includes citations with video titles and timestamps."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask"
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum context chunks to include",
                        "default": 10
                    }
                },
                "required": ["question"]
            }),
        },
        Tool {
            name: "list_media".to_string(),
            description: "List all indexed audio/video content in the knowledge base."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        Tool {
            name: "get_transcript".to_string(),
            description: "Get the full transcript of a specific indexed video by its ID."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The video ID to get transcript for"
                    }
                },
                "required": ["video_id"]
            }),
        },
    ]
}
