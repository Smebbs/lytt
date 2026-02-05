//! RAG response generation.

use super::{context::format_context_for_prompt, ContextBuilder, ContextChunk};
use crate::config::Prompts;
use crate::embedding::Embedder;
use crate::error::{Result, LyttError};
use crate::vector_store::VectorStore;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use crate::openai::create_client;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// RAG engine for question answering.
pub struct RagEngine {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    context_builder: ContextBuilder,
    prompts: Prompts,
    conversation_history: Vec<ChatCompletionRequestMessage>,
}

impl RagEngine {
    /// Create a new RAG engine.
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        model: &str,
        max_context_chunks: usize,
    ) -> Self {
        let context_builder = ContextBuilder::new(vector_store, embedder)
            .with_max_chunks(max_context_chunks)
            .with_min_score(0.3);

        Self {
            client: create_client(),
            model: model.to_string(),
            context_builder,
            prompts: Prompts::default(),
            conversation_history: Vec::new(),
        }
    }

    /// Set custom prompts (with user-defined variables).
    pub fn with_prompts(mut self, prompts: Prompts) -> Self {
        self.prompts = prompts;
        self
    }

    /// Ask a single question and get a response.
    #[instrument(skip(self), fields(question = %question))]
    pub async fn ask(&self, question: &str) -> Result<RagResponse> {
        info!("Processing question: {}", question);

        // Build context from the knowledge base
        let context_chunks = self.context_builder.build(question).await?;

        if context_chunks.is_empty() {
            return Ok(RagResponse {
                answer: "I couldn't find any relevant information in your video library for this question.".to_string(),
                sources: Vec::new(),
            });
        }

        // Build prompt
        let context_text = format_context_for_prompt(&context_chunks);

        let mut vars = HashMap::new();
        vars.insert("question".to_string(), question.to_string());
        vars.insert("context".to_string(), context_text);

        let user_prompt = self.prompts.render_with_custom(&self.prompts.rag.user, &vars);

        // Call LLM
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.prompts.rag.system.clone())
                .build()
                .map_err(|e| LyttError::Rag(e.to_string()))?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()
                .map_err(|e| LyttError::Rag(e.to_string()))?
                .into(),
        ];

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(0.7)
            .build()
            .map_err(|e| LyttError::Rag(e.to_string()))?;

        let response = self.client.chat().create(request).await.map_err(|e| {
            LyttError::OpenAI(format!("Failed to generate response: {}", e))
        })?;

        let answer = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| LyttError::Rag("Empty response from LLM".to_string()))?
            .clone();

        debug!("Generated response with {} sources", context_chunks.len());

        Ok(RagResponse {
            answer,
            sources: context_chunks,
        })
    }

    /// Start or continue a chat session.
    #[instrument(skip(self), fields(message = %message))]
    pub async fn chat(&mut self, message: &str) -> Result<RagResponse> {
        info!("Chat message: {}", message);

        // Build context from the knowledge base
        let context_chunks = self.context_builder.build(message).await?;
        let context_text = format_context_for_prompt(&context_chunks);

        // Build the user message with context
        let mut vars = HashMap::new();
        vars.insert("question".to_string(), message.to_string());
        vars.insert("context".to_string(), context_text);

        let user_content = if context_chunks.is_empty() {
            format!("Question: {}\n\n(No relevant context found in video library)", message)
        } else {
            format!(
                "Question: {}\n\nRelevant context from videos:\n{}",
                message,
                format_context_for_prompt(&context_chunks)
            )
        };

        // Add to conversation history
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(user_content)
            .build()
            .map_err(|e| LyttError::Rag(e.to_string()))?;

        self.conversation_history.push(user_message.into());

        // Build full message list
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.prompts.rag.chat_system.clone())
                .build()
                .map_err(|e| LyttError::Rag(e.to_string()))?
                .into(),
        ];
        messages.extend(self.conversation_history.clone());

        // Call LLM
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(0.7)
            .build()
            .map_err(|e| LyttError::Rag(e.to_string()))?;

        let response = self.client.chat().create(request).await.map_err(|e| {
            LyttError::OpenAI(format!("Failed to generate response: {}", e))
        })?;

        let answer = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| LyttError::Rag("Empty response from LLM".to_string()))?
            .clone();

        // Add assistant response to history
        let assistant_message = ChatCompletionRequestAssistantMessageArgs::default()
            .content(answer.clone())
            .build()
            .map_err(|e| LyttError::Rag(e.to_string()))?;
        self.conversation_history.push(assistant_message.into());

        // Trim history if too long
        if self.conversation_history.len() > 20 {
            self.conversation_history = self.conversation_history[self.conversation_history.len() - 20..].to_vec();
        }

        Ok(RagResponse {
            answer,
            sources: context_chunks,
        })
    }

    /// Clear conversation history.
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }
}

/// A RAG response with answer and sources.
#[derive(Debug, Clone)]
pub struct RagResponse {
    /// The generated answer.
    pub answer: String,
    /// Source chunks used for the answer.
    pub sources: Vec<ContextChunk>,
}

impl RagResponse {
    /// Format the response for display.
    pub fn format_for_display(&self) -> String {
        let mut output = self.answer.clone();

        if !self.sources.is_empty() {
            output.push_str("\n\n--- Sources ---\n");
            for source in &self.sources {
                output.push_str(&format!(
                    "\n{} @ {} (score: {:.2})",
                    source.video_title, source.timestamp, source.score
                ));
                if let Some(url) = &source.url {
                    output.push_str(&format!("\n  {}", url));
                }
            }
        }

        output
    }
}
