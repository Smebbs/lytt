//! Prompt templates for Lytt.
//!
//! Prompts can be customized by placing TOML files in the custom prompts directory.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Collection of all prompt templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Prompts {
    pub chunking: ChunkingPrompts,
    pub rag: RagPrompts,
    /// Prompts for transcription cleanup and segment structuring.
    pub cleanup: CleanupPrompts,
    /// Custom variables from config, available in all prompts.
    #[serde(skip)]
    pub variables: std::collections::HashMap<String, String>,
}


/// Prompts for semantic chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingPrompts {
    pub system: String,
    pub user: String,
}

impl Default for ChunkingPrompts {
    fn default() -> Self {
        Self {
            system: r#"You are a video content analyst. Your task is to analyze transcripts and identify logical content sections while filtering out filler content.

When analyzing a transcript:
1. Look for natural topic transitions and subject changes
2. Group related discussions together
3. Identify distinct segments that cover specific topics
4. Consider speaker context if apparent from the content

IMPORTANT - Filter out these types of content from chunk boundaries:
- Subscription/like/notification requests ("please subscribe", "hit the bell", "smash that like button")
- Generic introductions ("Hi everyone", "Welcome back", "Hey guys")
- Channel/creator self-promotion ("Check out my other videos", "Link in description")
- Outros and sign-offs ("Thanks for watching", "See you next time")
- Sponsor segments and ad reads
- Patreon/membership plugs

The goal is to have chunks that contain only the substantive educational or informational content.

Output your analysis as a JSON array of sections."#.to_string(),

            user: r#"Analyze this video transcript and identify logical content sections.

Video Title: {{title}}

Transcript:
{{transcript}}

For each section, provide:
- "title": A brief descriptive title (3-8 words) summarizing the actual content
- "start_seconds": Start timestamp in seconds (skip past any intro fluff)
- "end_seconds": End timestamp in seconds (stop before any outro fluff)
- "summary": One sentence describing the substantive content

Target section length: {{target_duration}} seconds (minimum {{min_duration}}, maximum {{max_duration}})

Rules:
1. Do NOT include subscription requests, intros like "welcome back", or outros in any chunk
2. If a chunk would start with filler, move start_seconds forward to skip it
3. If a chunk would end with filler, move end_seconds backward to exclude it
4. Section titles should reflect the actual topic, not "Introduction" unless it's truly introducing the topic

Respond with a JSON array of section objects. Example:
[
  {"title": "Binary Number Representation", "start_seconds": 45, "end_seconds": 220, "summary": "Explains how binary numbers work and their relationship to decimal."},
  {"title": "Bitwise Operations", "start_seconds": 220, "end_seconds": 450, "summary": "Covers AND, OR, XOR and shift operations with examples."}
]"#.to_string(),
        }
    }
}

/// Prompts for RAG response generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RagPrompts {
    pub system: String,
    pub user: String,
    pub chat_system: String,
}

impl Default for RagPrompts {
    fn default() -> Self {
        Self {
            system: r#"You are a helpful assistant that answers questions based on video content from the user's knowledge base.

Guidelines:
- Answer questions using only the provided context from video transcripts
- Always cite your sources with video titles and timestamps
- Use the format [Video Title @ MM:SS] for citations
- If the context doesn't contain relevant information, say so clearly
- Be concise but thorough in your responses
- When multiple sources are relevant, synthesize information across them"#.to_string(),

            user: r#"Question: {{question}}

{{#if user_context}}
Additional context provided by user:
{{user_context}}

{{/if}}
Relevant excerpts from your video knowledge base:

{{#each chunks}}
---
Source: {{this.video_title}} @ {{this.timestamp}}
{{this.content}}
---

{{/each}}
Please answer the question based on the above context."#.to_string(),

            chat_system: r#"You are a helpful assistant for exploring video content. You have access to transcripts from the user's video library.

In this conversation:
- Answer questions using the video context provided
- Remember previous questions in the conversation for follow-ups
- Cite sources with [Video Title @ MM:SS] format
- Ask clarifying questions if the user's intent is unclear
- If asked about something not in the videos, say so honestly"#.to_string(),
        }
    }
}

/// Prompts for transcription cleanup and segment structuring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanupPrompts {
    pub system: String,
}

impl Default for CleanupPrompts {
    fn default() -> Self {
        Self {
            system: r#"You are a transcription cleanup and fusion expert. You process word-level timestamps from Whisper, optionally combined with text from a secondary transcription model.

## Input
- Word timestamps from Whisper (JSON array with word, start, end)
- Text transcription (from secondary model, or reconstructed from Whisper words)

## Output
JSON object with "segments" array. Each segment has:
- "text": The cleaned transcribed text for this segment
- "start_seconds": Start time (from Whisper word timestamps)
- "end_seconds": End time (from Whisper word timestamps)

## Rules
- Use Whisper word timestamps to determine when each segment starts and ends
- Group words into natural segments (sentences or phrases, typically 5-15 seconds each)
- If two transcripts are provided, compare them - neither is always correct
- For names, places, and technical terms: pick what makes sense in context
- Never hallucinate content not present in the transcript(s)
- Output segments should cover the full duration of the audio
- Ensure proper punctuation and capitalization in sentences
- Segments must not overlap
- Fix obvious transcription errors (e.g., "there" vs "their" based on context)"#
                .to_string(),
        }
    }
}

impl Prompts {
    /// Load prompts from the default location, with optional custom directory and variables.
    pub fn load(
        custom_dir: Option<&str>,
        custom_variables: Option<&std::collections::HashMap<String, String>>,
    ) -> crate::error::Result<Self> {
        let mut prompts = Prompts::default();

        // Store custom variables
        if let Some(vars) = custom_variables {
            prompts.variables = vars.clone();
        }

        if let Some(dir) = custom_dir {
            let custom_path = PathBuf::from(shellexpand::tilde(dir).to_string());

            // Load chunking prompts if file exists
            let chunking_path = custom_path.join("chunking.toml");
            if chunking_path.exists() {
                let content = std::fs::read_to_string(&chunking_path)?;
                prompts.chunking = toml::from_str(&content)?;
            }

            // Load RAG prompts if file exists
            let rag_path = custom_path.join("rag.toml");
            if rag_path.exists() {
                let content = std::fs::read_to_string(&rag_path)?;
                prompts.rag = toml::from_str(&content)?;
            }

            // Load cleanup prompts if file exists
            let cleanup_path = custom_path.join("cleanup.toml");
            if cleanup_path.exists() {
                let content = std::fs::read_to_string(&cleanup_path)?;
                prompts.cleanup = toml::from_str(&content)?;
            }
        }

        Ok(prompts)
    }

    /// Render a prompt template with the given variables.
    pub fn render(template: &str, vars: &std::collections::HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }
        result
    }

    /// Render a prompt template with both provided variables and custom config variables.
    /// Provided variables take precedence over custom config variables.
    pub fn render_with_custom(
        &self,
        template: &str,
        vars: &std::collections::HashMap<String, String>,
    ) -> String {
        // Start with custom variables, then override with provided vars
        let mut merged = self.variables.clone();
        for (key, value) in vars {
            merged.insert(key.clone(), value.clone());
        }
        Self::render(template, &merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_prompts() {
        let prompts = Prompts::default();
        assert!(!prompts.chunking.system.is_empty());
        assert!(!prompts.rag.system.is_empty());
    }

    #[test]
    fn test_render_template() {
        let template = "Hello {{name}}, you have {{count}} messages.";
        let mut vars = std::collections::HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("count".to_string(), "5".to_string());

        let result = Prompts::render(template, &vars);
        assert_eq!(result, "Hello Alice, you have 5 messages.");
    }
}
