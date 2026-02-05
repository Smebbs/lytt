# Lytt - Agent Integration Guide

This guide is for AI agents and developers building agent systems that integrate with Lytt.

## What is Lytt?

Lytt is a local-first audio transcription and RAG (Retrieval-Augmented Generation) tool. It:

- Transcribes YouTube videos and local audio/video files
- Chunks transcripts semantically with timestamps
- Generates embeddings and stores them in a local SQLite vector database
- Provides semantic search and RAG-based question answering

## Integration Methods

| Method | Best For | Requires Server |
|--------|----------|-----------------|
| **MCP** | Claude Desktop/Code | No |
| **HTTP API** | Any language/platform | Yes (`lytt serve`) |
| **CLI** | Shell scripts, simple automation | No |
| **Rust Library** | Rust applications | No |

---

## MCP Integration (Claude)

### Configuration

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "lytt": {
      "command": "lytt",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code** (`.claude/settings.json`):
```json
{
  "mcpServers": {
    "lytt": {
      "command": "lytt",
      "args": ["mcp"]
    }
  }
}
```

### Available Tools

#### `transcribe`
Transcribe and index audio/video content.

```json
{
  "input": "https://youtube.com/watch?v=dQw4w9WgXcQ",
  "force": false
}
```

- `input` (required): YouTube URL, video ID, or local file path
- `force` (optional): Re-process even if already indexed

#### `search`
Semantic search across indexed content.

```json
{
  "query": "machine learning basics",
  "limit": 5,
  "min_score": 0.3
}
```

- `query` (required): Search query
- `limit` (optional, default 5): Maximum results
- `min_score` (optional, default 0.3): Minimum similarity threshold

#### `ask`
RAG query - get an AI-generated answer with sources.

```json
{
  "question": "What are the main points about neural networks?",
  "max_chunks": 10
}
```

- `question` (required): The question to answer
- `max_chunks` (optional, default 10): Context chunks to include

#### `list_media`
List all indexed content.

```json
{}
```

#### `get_transcript`
Get full transcript of a specific video.

```json
{
  "video_id": "dQw4w9WgXcQ"
}
```

---

## HTTP API Integration

Start the server:
```bash
lytt serve --port 3000
```

### Endpoints

#### `POST /transcribe`
```json
{
  "input": "https://youtube.com/watch?v=...",
  "force": false
}
```

Response:
```json
{
  "success": true,
  "media_id": "dQw4w9WgXcQ",
  "title": "Video Title",
  "chunks_indexed": 12
}
```

#### `POST /search`
```json
{
  "query": "search terms",
  "limit": 5,
  "min_score": 0.3
}
```

Response:
```json
{
  "results": [
    {
      "video_id": "abc123",
      "video_title": "Video Title",
      "chunk_title": "Section Name",
      "content": "Transcript text...",
      "start_seconds": 120.0,
      "end_seconds": 180.0,
      "timestamp": "02:00",
      "score": 0.85
    }
  ]
}
```

#### `POST /ask`
```json
{
  "question": "What is discussed about X?",
  "max_chunks": 10,
  "model": "gpt-4o-mini"
}
```

Response:
```json
{
  "answer": "Based on the videos, ...",
  "sources": [
    {
      "video_id": "abc123",
      "video_title": "Video Title",
      "timestamp": "02:00",
      "score": 0.85,
      "content": "Relevant excerpt..."
    }
  ]
}
```

#### `GET /media`
List all indexed media.

#### `GET /media/:video_id`
Get details and chunks for a specific video.

#### `GET /health`
Health check endpoint.

---

## CLI Integration

### Transcribe and Index
```bash
lytt transcribe "https://youtube.com/watch?v=..."
lytt transcribe ./local-video.mp4
lytt transcribe ./audio.mp3 --force
```

### Transcribe Playlists and Channels
```bash
# Transcribe entire playlist
lytt transcribe "https://youtube.com/playlist?list=PLxxxxxxx" --playlist

# Limit to first N videos
lytt transcribe "https://youtube.com/playlist?list=PLxxxxxxx" --playlist --limit 10

# Transcribe from channel
lytt transcribe "https://youtube.com/@channelname" --playlist --limit 20
```

### Rechunk Without Re-transcribing
```bash
# Rechunk a single video (uses stored transcript)
lytt rechunk VIDEO_ID

# Rechunk all videos with stored transcripts
lytt rechunk all
```

Note: Rechunking only works for videos transcribed after the rechunk feature was added. Older videos need `--force` to re-transcribe first.

### Export Transcript (No Indexing)
```bash
# Raw transcript
lytt transcribe video.mp4 --output transcript.json

# With semantic chunking
lytt transcribe video.mp4 --output chunks.json --chunk

# With embeddings (for vector DB import)
lytt transcribe video.mp4 --output chunks.json --chunk --embed
```

### Search
```bash
lytt search "query terms" --limit 10
```

### Ask (RAG)
```bash
lytt ask "What are the key points about X?"
```

### List Indexed Content
```bash
lytt list
```

### Export Existing Transcript
```bash
lytt export <video_id> --output transcript.json
lytt export <video_id> --format srt --output subtitles.srt
```

---

## Output Formats

### Raw Transcript (`--output`)
```json
{
  "media_id": "abc123",
  "duration_seconds": 1847.5,
  "segments": [
    {
      "text": "Hello and welcome...",
      "start_seconds": 0.0,
      "end_seconds": 5.2
    }
  ]
}
```

### Chunked (`--output --chunk`)
```json
{
  "video_id": "abc123",
  "title": "Video Title",
  "duration_seconds": 1847.5,
  "chunk_count": 12,
  "chunks": [
    {
      "title": "Introduction",
      "content": "Hello and welcome to this video...",
      "start_seconds": 0.0,
      "end_seconds": 180.0
    }
  ]
}
```

### With Embeddings (`--output --chunk --embed`)
```json
{
  "video_id": "abc123",
  "title": "Video Title",
  "duration_seconds": 1847.5,
  "chunk_count": 12,
  "chunks": [
    {
      "title": "Introduction",
      "content": "Hello and welcome...",
      "start_seconds": 0.0,
      "end_seconds": 180.0,
      "embedding": [0.023, -0.041, ...]
    }
  ],
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536
}
```

---

## Common Agent Workflows

### 1. Build Knowledge Base
```
1. transcribe("https://youtube.com/watch?v=video1")
2. transcribe("https://youtube.com/watch?v=video2")
3. transcribe("./local-recording.mp4")
4. list_media() -> verify all indexed
```

### 2. Build Knowledge Base from Playlist/Channel
```bash
# Index entire playlist
lytt transcribe "https://youtube.com/playlist?list=PLxxxxxxx" --playlist

# Index latest videos from a channel
lytt transcribe "https://youtube.com/@channelname" --playlist --limit 50
```

### 3. Answer Questions
```
1. ask("What are the main topics discussed?")
2. If answer references specific video, get_transcript(video_id) for details
```

### 4. Research Workflow
```
1. search("specific topic") -> find relevant segments
2. For each result, note video_id and timestamp
3. ask("Summarize what was said about X") -> synthesized answer
```

### 5. Update Chunking Strategy
```bash
# 1. Modify chunking prompt or settings
nano ~/.lytt/prompts/chunking.toml
nano ~/.config/lytt/config.toml

# 2. Rechunk all content with new settings (no re-transcription needed)
lytt rechunk all
```

### 6. Export for External System
```bash
# Transcribe without indexing, export with embeddings
lytt transcribe video.mp4 --output chunks.json --chunk --embed

# Import chunks.json into your own vector DB (Pinecone, Weaviate, etc.)
```

---

## Environment Requirements

- `OPENAI_API_KEY` - Required for transcription and embeddings
- `yt-dlp` - Required for YouTube downloads
- `ffmpeg` / `ffprobe` - Required for audio processing

### Setup Commands
```bash
# Interactive first-run setup
lytt init

# Verify all requirements
lytt doctor

# Show current configuration
lytt config show

# Edit configuration
lytt config edit
```

---

## Error Handling

All tools return errors in a consistent format:

**MCP**: `is_error: true` with error message in content
**HTTP**: Appropriate status code with `{ "error": "message" }`
**CLI**: Non-zero exit code with error printed to stderr

Common errors:
- `OPENAI_API_KEY not set` - Set the environment variable
- `Tool not found: yt-dlp` - Install yt-dlp
- `Media not found` - Video ID doesn't exist in index
- `No matching results` - Search returned no results above threshold

---

## Development

### Rust Library Usage

```rust
use lytt::config::Settings;
use lytt::orchestrator::Orchestrator;
use lytt::embedding::{Embedder, OpenAIEmbedder};
use lytt::rag::RagEngine;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let settings = Settings::load()?;
    let orchestrator = Orchestrator::new(settings.clone())?;

    // Transcribe
    let result = orchestrator.process_media("https://youtube.com/...", false).await?;
    println!("Indexed {} chunks", result.chunks_indexed);

    // Search
    let embedder = Arc::new(OpenAIEmbedder::with_config(
        &settings.embedding.model,
        settings.embedding.dimensions as usize,
    ));

    let query_embedding = embedder.embed("search query").await?;
    let results = orchestrator.vector_store()
        .search_with_threshold(&query_embedding, 5, 0.3)
        .await?;

    // RAG
    let engine = RagEngine::new(
        orchestrator.vector_store(),
        embedder,
        "gpt-4o-mini",
        10,
    );

    let response = engine.ask("What is discussed?").await?;
    println!("{}", response.answer);

    Ok(())
}
```

### Configuration

Default config location: `~/.config/lytt/config.toml`

```toml
[general]
data_dir = "~/.lytt"
log_level = "info"

[transcription]
provider = "whisper"  # or "fusion"
model = "whisper-1"

[embedding]
model = "text-embedding-3-small"
dimensions = 1536

[chunking]
strategy = "semantic"
target_chunk_seconds = 180
min_chunk_seconds = 60
max_chunk_seconds = 600

[rag]
model = "gpt-4o-mini"
max_context_chunks = 10
```

### Custom Prompts

Create custom prompts in `~/.lytt/prompts/`:

**chunking.toml**:
```toml
system = "Your custom chunking system prompt..."
user = "Your custom user prompt with {{variables}}..."
```

**rag.toml**:
```toml
system = "Your custom RAG system prompt..."
user = "Your custom user prompt..."
chat_system = "Your custom chat system prompt..."
```

**cleanup.toml** (for transcription cleanup and segment structuring):
```toml
system = "Your custom cleanup prompt..."
```

Set in config:
```toml
[prompts]
custom_dir = "~/.lytt/prompts"
```

### Custom Prompt Variables

Define custom variables in config that can be used in all prompts:

```toml
[prompts]
custom_dir = "~/.lytt/prompts"

[prompts.variables]
language = "Norwegian"
style = "concise and technical"
audience = "software developers"
```

Use in any prompt file:
```toml
system = """Answer in {{language}}.
Keep your tone {{style}}.
The audience is {{audience}}."""
```

### Built-in Template Variables

| Prompt | Available Variables |
|--------|---------------------|
| chunking.user | `{{title}}`, `{{transcript}}`, `{{target_duration}}`, `{{min_duration}}`, `{{max_duration}}` |
| rag.user | `{{question}}`, `{{chunks}}` (array with `video_title`, `timestamp`, `content`) |
| cleanup | No variables (receives JSON input directly) |

Custom variables from `[prompts.variables]` are available in all prompts and merged with built-in variables.
