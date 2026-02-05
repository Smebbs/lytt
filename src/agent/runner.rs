//! Agent runner with tool calling loop.

use super::tools::{parse_tool_call, tool_definitions, ToolContext};
use crate::error::{LyttError, Result};
use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use crate::openai::create_client;
use tracing::{debug, info};

/// Default system prompt for the agent.
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an intelligent assistant with access to a video knowledge base.

You have tools to search videos, get full transcripts, and retrieve specific segments.
Think step-by-step about what information you need, then use the appropriate tools.

Guidelines:
- Use 'list_videos' first if you need to know what content is available
- Use 'search' to find specific topics across all videos
- Use 'get_transcript' to get a full video transcript for summaries, quizzes, or deep analysis
- Use 'get_segment' to get content from a specific time range
- Use 'get_video_info' to get metadata about a video

When you have gathered enough information, provide your final response.
Always cite your sources with video titles and timestamps when relevant.
Format your responses clearly with appropriate structure (headers, lists, etc.)."#;

/// Agent that can use tools to interact with the video knowledge base.
pub struct Agent {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    tools: ToolContext,
    max_iterations: usize,
    system_prompt: String,
}

impl Agent {
    /// Create a new agent with the given tool context and model.
    pub fn new(tools: ToolContext, model: &str) -> Self {
        Self {
            client: create_client(),
            model: model.to_string(),
            tools,
            max_iterations: 15,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Set a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    /// Set maximum iterations for the agent loop.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Run the agent with a user task.
    pub async fn run(&self, task: &str, context: Option<&str>) -> Result<AgentResponse> {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt.clone())
                .build()
                .map_err(|e| LyttError::Agent(e.to_string()))?
                .into(),
        ];

        // Build user message with optional context
        let user_message = match context {
            Some(ctx) => format!("Context: {}\n\nTask: {}", ctx, task),
            None => task.to_string(),
        };

        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()
                .map_err(|e| LyttError::Agent(e.to_string()))?
                .into(),
        );

        let mut iterations = 0;
        let mut tool_calls_made = Vec::new();

        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(LyttError::Agent(format!(
                    "Agent exceeded maximum iterations ({})",
                    self.max_iterations
                )));
            }

            debug!("Agent iteration {}", iterations);

            // Call LLM with tools
            let request = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .messages(messages.clone())
                .tools(tool_definitions())
                .build()
                .map_err(|e| LyttError::Agent(e.to_string()))?;

            let response = self
                .client
                .chat()
                .create(request)
                .await
                .map_err(|e| LyttError::OpenAI(format!("Agent API error: {}", e)))?;

            let choice = response
                .choices
                .first()
                .ok_or_else(|| LyttError::Agent("No response from model".to_string()))?;

            // Check if LLM wants to call tools
            if let Some(ref tool_calls) = choice.message.tool_calls {
                if tool_calls.is_empty() {
                    // No tool calls, treat as final response
                    return self.build_response(&choice.message.content, tool_calls_made, iterations);
                }

                // Add assistant message with tool calls to history
                let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()
                    .map_err(|e| LyttError::Agent(e.to_string()))?;
                messages.push(assistant_msg.into());

                // Execute each tool call
                for tool_call in tool_calls {
                    let record = self.execute_tool_call(tool_call).await;

                    // Add tool result to messages
                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(&tool_call.id)
                        .content(record.result.clone())
                        .build()
                        .map_err(|e| LyttError::Agent(e.to_string()))?;
                    messages.push(tool_msg.into());

                    tool_calls_made.push(record);
                }
            } else {
                // No tool calls - LLM is done, return final response
                return self.build_response(&choice.message.content, tool_calls_made, iterations);
            }
        }
    }

    /// Execute a single tool call and return a record of it.
    async fn execute_tool_call(&self, tool_call: &ChatCompletionMessageToolCall) -> ToolCallRecord {
        let name = &tool_call.function.name;
        let arguments = &tool_call.function.arguments;

        info!("Agent calling tool: {} with args: {}", name, arguments);

        // Parse and execute the tool
        let result = match parse_tool_call(name, arguments) {
            Ok(tool) => match self.tools.execute(&tool).await {
                Ok(output) => output,
                Err(e) => format!("Tool error: {}", e),
            },
            Err(e) => format!("Failed to parse tool call: {}", e),
        };

        ToolCallRecord {
            name: name.clone(),
            arguments: arguments.clone(),
            result,
        }
    }

    /// Build the final agent response.
    fn build_response(
        &self,
        content: &Option<String>,
        tool_calls: Vec<ToolCallRecord>,
        iterations: usize,
    ) -> Result<AgentResponse> {
        let content = content.clone().unwrap_or_default();

        Ok(AgentResponse {
            content,
            tool_calls,
            iterations,
        })
    }
}

/// Response from an agent run.
#[derive(Debug)]
pub struct AgentResponse {
    /// The final response content from the agent.
    pub content: String,
    /// Record of all tool calls made during execution.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Number of iterations (LLM calls) used.
    pub iterations: usize,
}

/// Record of a tool call made by the agent.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Name of the tool called.
    pub name: String,
    /// JSON arguments passed to the tool.
    pub arguments: String,
    /// Result returned by the tool.
    pub result: String,
}

impl std::fmt::Display for ToolCallRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name, self.arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_record_display() {
        let record = ToolCallRecord {
            name: "search".to_string(),
            arguments: r#"{"query": "test"}"#.to_string(),
            result: "Found results".to_string(),
        };
        assert_eq!(format!("{}", record), r#"search({"query": "test"})"#);
    }
}
