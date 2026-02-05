//! Interactive chat command with tool calling support.

use crate::agent::{parse_tool_call, tool_definitions, ToolContext};
use crate::cli::preflight::{self, Operation};
use crate::cli::Output;
use crate::config::Settings;
use crate::embedding::OpenAIEmbedder;
use crate::error::{LyttError, Result};
use crate::orchestrator::Orchestrator;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use crate::openai::create_client;
use console::style;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tracing::{debug, info};

/// System prompt for the chat agent.
const CHAT_SYSTEM_PROMPT: &str = r#"You are a helpful assistant for exploring video and audio content. You have access to the user's media knowledge base.

You have tools to search, get transcripts, and retrieve information. Use them to answer questions accurately.

Guidelines:
- Use 'list_videos' to see what content is available
- Use 'search' to find specific topics across all media
- Use 'get_transcript' to get a full transcript for summaries or deep analysis
- Use 'get_segment' to get content from a specific time range
- Use 'get_video_info' to get metadata about a video

Always cite your sources with titles and timestamps when relevant.
Be conversational and helpful. Remember context from earlier in the conversation."#;

/// Run the interactive chat command.
pub async fn run_chat(model: Option<String>, settings: Settings) -> Result<()> {
    // Pre-flight checks
    if let Err(e) = preflight::check(Operation::Ask) {
        Output::error(&format!("{}", e));
        Output::info("Run 'lytt doctor' for detailed diagnostics.");
        return Err(e.into());
    }

    let orchestrator = Orchestrator::new(settings.clone())?;
    let model = model.unwrap_or_else(|| settings.rag.model.clone());

    let embedder = Arc::new(OpenAIEmbedder::with_config(
        &settings.embedding.model,
        settings.embedding.dimensions as usize,
    ));

    let tool_context = ToolContext::new(orchestrator.vector_store(), embedder);

    let mut chat = ChatSession::new(tool_context, &model);

    println!("\n{}", style("Lytt Chat").bold().cyan());
    println!(
        "{}\n",
        style("Type your questions, or 'exit' to quit. Use 'clear' to reset conversation.").dim()
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("{} ", style("You:").green().bold());
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            Output::info("Goodbye!");
            break;
        }

        if input.eq_ignore_ascii_case("clear") {
            chat.clear_history();
            Output::info("Conversation history cleared.");
            continue;
        }

        match chat.send_message(input).await {
            Ok(response) => {
                println!("\n{} {}\n", style("Lytt:").cyan().bold(), response);
            }
            Err(e) => {
                Output::error(&format!("Error: {}", e));
            }
        }
    }

    Ok(())
}

/// Interactive chat session with tool calling support.
struct ChatSession {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    tools: ToolContext,
    messages: Vec<ChatCompletionRequestMessage>,
    max_tool_iterations: usize,
}

impl ChatSession {
    /// Create a new chat session.
    fn new(tools: ToolContext, model: &str) -> Self {
        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(CHAT_SYSTEM_PROMPT)
            .build()
            .expect("Failed to build system message");

        Self {
            client: create_client(),
            model: model.to_string(),
            tools,
            messages: vec![system_message.into()],
            max_tool_iterations: 10,
        }
    }

    /// Clear conversation history (keeps system prompt).
    fn clear_history(&mut self) {
        self.messages.truncate(1); // Keep system message
    }

    /// Send a message and get a response, handling tool calls.
    async fn send_message(&mut self, user_input: &str) -> Result<String> {
        // Add user message to history
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(user_input)
            .build()
            .map_err(|e| LyttError::Agent(e.to_string()))?;
        self.messages.push(user_message.into());

        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > self.max_tool_iterations {
                return Err(LyttError::Agent("Too many tool iterations".to_string()).into());
            }

            debug!("Chat iteration {}, {} messages", iterations, self.messages.len());

            // Call LLM with tools
            let request = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .messages(self.messages.clone())
                .tools(tool_definitions())
                .build()
                .map_err(|e| LyttError::Agent(e.to_string()))?;

            let response = self
                .client
                .chat()
                .create(request)
                .await
                .map_err(|e| LyttError::OpenAI(format!("Chat API error: {}", e)))?;

            let choice = response
                .choices
                .first()
                .ok_or_else(|| LyttError::Agent("No response from model".to_string()))?;

            // Check if LLM wants to call tools
            if let Some(ref tool_calls) = choice.message.tool_calls {
                if tool_calls.is_empty() {
                    // No tool calls, this is the final response
                    let content = choice.message.content.clone().unwrap_or_default();
                    self.add_assistant_message(&content)?;
                    return Ok(content);
                }

                // Add assistant message with tool calls
                let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()
                    .map_err(|e| LyttError::Agent(e.to_string()))?;
                self.messages.push(assistant_msg.into());

                // Execute each tool call
                for tool_call in tool_calls {
                    let name = &tool_call.function.name;
                    let arguments = &tool_call.function.arguments;

                    info!("Chat calling tool: {} with args: {}", name, arguments);
                    print!("{}", style(format!("  [{}] ", name)).dim());
                    io::stdout().flush().ok();

                    let result = match parse_tool_call(name, arguments) {
                        Ok(tool) => match self.tools.execute(&tool).await {
                            Ok(output) => {
                                println!("{}", style("✓").green());
                                output
                            }
                            Err(e) => {
                                println!("{}", style("✗").red());
                                format!("Tool error: {}", e)
                            }
                        },
                        Err(e) => {
                            println!("{}", style("✗").red());
                            format!("Failed to parse tool call: {}", e)
                        }
                    };

                    // Add tool result to messages
                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(&tool_call.id)
                        .content(result)
                        .build()
                        .map_err(|e| LyttError::Agent(e.to_string()))?;
                    self.messages.push(tool_msg.into());
                }
            } else {
                // No tool calls - final response
                let content = choice.message.content.clone().unwrap_or_default();
                self.add_assistant_message(&content)?;

                // Trim history if too long (keep system + last N exchanges)
                self.trim_history(30);

                return Ok(content);
            }
        }
    }

    /// Add an assistant text message to history.
    fn add_assistant_message(&mut self, content: &str) -> Result<()> {
        let msg = ChatCompletionRequestAssistantMessageArgs::default()
            .content(content)
            .build()
            .map_err(|e| LyttError::Agent(e.to_string()))?;
        self.messages.push(msg.into());
        Ok(())
    }

    /// Trim conversation history to keep it manageable.
    fn trim_history(&mut self, max_messages: usize) {
        if self.messages.len() > max_messages {
            // Keep system message (index 0) and last N-1 messages
            let start = self.messages.len() - (max_messages - 1);
            let mut trimmed = vec![self.messages[0].clone()];
            trimmed.extend(self.messages[start..].iter().cloned());
            self.messages = trimmed;
        }
    }
}
