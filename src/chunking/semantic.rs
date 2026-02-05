//! LLM-based semantic chunking implementation.
//!
//! Uses an LLM to identify logical content sections in transcripts.

use super::{Chunker, ChunkingConfig, ContentChunk};
use crate::config::Prompts;
use crate::error::{Result, LyttError};
use crate::transcription::Transcript;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use crate::openai::create_client;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// LLM-based semantic chunker.
pub struct SemanticChunker {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    prompts: Prompts,
}

impl SemanticChunker {
    pub fn new() -> Self {
        Self::with_model("gpt-4o-mini")
    }

    pub fn with_model(model: &str) -> Self {
        Self {
            client: create_client(),
            model: model.to_string(),
            prompts: Prompts::default(),
        }
    }

    /// Set custom prompts (with user-defined variables).
    pub fn with_prompts(mut self, prompts: Prompts) -> Self {
        self.prompts = prompts;
        self
    }

    /// Parse the LLM response into sections.
    fn parse_sections(response: &str) -> Result<Vec<LLMSection>> {
        // Try to extract JSON from the response
        let json_start = response.find('[');
        let json_end = response.rfind(']');

        let json_str = match (json_start, json_end) {
            (Some(start), Some(end)) if end > start => &response[start..=end],
            _ => response,
        };

        serde_json::from_str(json_str).map_err(|e| {
            LyttError::Transcription(format!(
                "Failed to parse chunking response: {}. Response was: {}",
                e,
                &response[..response.len().min(500)]
            ))
        })
    }

    /// Build chunks from LLM-identified sections and the original transcript.
    fn build_chunks(
        sections: Vec<LLMSection>,
        transcript: &Transcript,
        config: &ChunkingConfig,
    ) -> Vec<ContentChunk> {
        let mut chunks: Vec<ContentChunk> = Vec::new();

        for (order, section) in sections.into_iter().enumerate() {
            // Get content for this time range
            let content = transcript.text_between(section.start_seconds, section.end_seconds);

            if content.trim().is_empty() {
                continue;
            }

            // Check if chunk duration is within bounds
            let duration = section.end_seconds - section.start_seconds;
            if duration < config.min_duration as f64 && !chunks.is_empty() {
                // Merge with previous chunk if too short
                if let Some(last) = chunks.last_mut() {
                    last.content.push(' ');
                    last.content.push_str(&content);
                    last.end_seconds = section.end_seconds;
                    continue;
                }
            }

            let mut chunk = ContentChunk::new(
                Some(section.title),
                content,
                section.start_seconds,
                section.end_seconds,
                order as i32,
            );
            chunk.summary = section.summary;

            chunks.push(chunk);
        }

        chunks
    }
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct LLMSection {
    title: String,
    start_seconds: f64,
    end_seconds: f64,
    #[serde(default)]
    summary: Option<String>,
}

#[async_trait]
impl Chunker for SemanticChunker {
    async fn chunk(&self, transcript: &Transcript, config: &ChunkingConfig) -> Result<Vec<ContentChunk>> {
        if transcript.segments.is_empty() {
            return Ok(Vec::new());
        }

        // For very short transcripts, just return a single chunk
        if transcript.duration_seconds < config.min_duration as f64 {
            return Ok(vec![ContentChunk::new(
                None,
                transcript.full_text.clone(),
                0.0,
                transcript.duration_seconds,
                0,
            )]);
        }

        info!("Performing semantic chunking on transcript");

        // Build prompt with custom variables
        let mut vars = HashMap::new();
        vars.insert("title".to_string(), transcript.video_id.clone());
        vars.insert("transcript".to_string(), transcript.format_with_timestamps());
        vars.insert("target_duration".to_string(), config.target_duration.to_string());
        vars.insert("min_duration".to_string(), config.min_duration.to_string());
        vars.insert("max_duration".to_string(), config.max_duration.to_string());

        let system_message = self.prompts.render_with_custom(&self.prompts.chunking.system, &vars);
        let user_message = self.prompts.render_with_custom(&self.prompts.chunking.user, &vars);

        // Call LLM
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_message)
                .build()
                .map_err(|e| LyttError::Transcription(e.to_string()))?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()
                .map_err(|e| LyttError::Transcription(e.to_string()))?
                .into(),
        ];

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(0.3)
            .build()
            .map_err(|e| LyttError::Transcription(e.to_string()))?;

        let response = self.client.chat().create(request).await.map_err(|e| {
            LyttError::OpenAI(format!("Failed to get chunking response: {}", e))
        })?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| LyttError::Transcription("Empty response from LLM".to_string()))?;

        debug!("LLM chunking response: {}", &content[..content.len().min(500)]);

        // Parse and build chunks
        match Self::parse_sections(content) {
            Ok(sections) => {
                let chunks = Self::build_chunks(sections, transcript, config);
                info!("Created {} semantic chunks", chunks.len());
                Ok(chunks)
            }
            Err(e) => {
                // Fallback to temporal chunking if semantic fails
                warn!("Semantic chunking failed, falling back to temporal: {}", e);
                let temporal = super::TemporalChunker::new();
                temporal.chunk(transcript, config).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sections() {
        let json = r#"[
            {"title": "Introduction", "start_seconds": 0, "end_seconds": 60},
            {"title": "Main Content", "start_seconds": 60, "end_seconds": 180, "summary": "The main ideas"}
        ]"#;

        let sections = SemanticChunker::parse_sections(json).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].title, "Introduction");
        assert_eq!(sections[1].summary, Some("The main ideas".to_string()));
    }

    #[test]
    fn test_parse_sections_with_markdown() {
        let response = r#"Here are the sections:

```json
[
    {"title": "Part 1", "start_seconds": 0, "end_seconds": 100}
]
```

These represent the main topics."#;

        let sections = SemanticChunker::parse_sections(response).unwrap();
        assert_eq!(sections.len(), 1);
    }
}
