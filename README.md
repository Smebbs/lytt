# Lytt

> **Note:** This project is a work in progress. Features and APIs may change.

A high-quality, local-first CLI tool for audio transcription and RAG (Retrieval-Augmented Generation).

The name "Lytt" comes from the Norwegian/Scandinavian word for "listen."

Lytt lets you transcribe audio content (YouTube videos or local audio/video files), build a searchable knowledge base, and ask questions to get AI-powered answers with citations.

## Features

- **Transcribe Audio**: Download and transcribe YouTube videos or local audio/video files using OpenAI Whisper
- **Playlist/Channel Support**: Batch transcribe entire YouTube playlists or channels
- **Smart Chunking**: Semantically or temporally chunk transcripts for better retrieval
- **Custom Prompts**: Customize chunking and RAG prompts with your own templates and variables
- **Vector Search**: Store embeddings locally in SQLite for fast similarity search
- **Ask Questions**: Get AI-generated answers from your audio library with source citations
- **Interactive Chat**: Have a conversation about your audio content
- **Rechunk Without Re-transcribing**: Update chunking strategy without re-downloading or re-transcribing
- **MCP Integration**: Use as a tool in Claude Desktop/Code via Model Context Protocol
- **HTTP API**: REST API for integration with any language/platform
- **Extensible**: Pluggable architecture for audio sources, vector stores, and more

## Installation

### Prerequisites

- Rust 1.75+ (for building)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - for downloading audio from YouTube
- [ffmpeg](https://ffmpeg.org/) - for audio processing
- OpenAI API key - for transcription and embeddings

### Build from source

```bash
git clone https://github.com/smebbs/lytt.git
cd lytt
cargo build --release

# Optional: install to PATH
cargo install --path .
```

### Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

```bash
# First-time setup (verifies requirements)
lytt init

# Transcribe a YouTube video
lytt transcribe https://youtube.com/watch?v=VIDEO_ID
lytt transcribe VIDEO_ID  # Just the ID works too

# Transcribe an entire playlist or channel
lytt transcribe "https://youtube.com/playlist?list=PLxxxxxxx" --playlist
lytt transcribe "https://youtube.com/@channelname" --playlist --limit 20

# Transcribe local files
lytt transcribe /path/to/audio.mp3
lytt transcribe /path/to/video.mp4

# Ask a question about your audio library
lytt ask "How does the authentication system work?"

# Search for relevant segments
lytt search "error handling patterns"

# Start an interactive chat
lytt chat

# List indexed media
lytt list

# Rechunk with new settings (no re-transcription)
lytt rechunk VIDEO_ID
lytt rechunk all

# Run an AI agent task
lytt agent "Summarize all videos about machine learning"

# View/edit configuration
lytt config show
lytt config edit
```

## CLI Reference

### `lytt transcribe <input>`

Transcribe and index audio content.

```bash
lytt transcribe INPUT [OPTIONS]

Options:
  -f, --force       Force re-processing even if already indexed
  --playlist        Treat input as playlist/channel URL, transcribe all videos
  --limit N         Max videos to transcribe from playlist (default: 50)
  -o, --output FILE Export transcript to file instead of indexing
  --format FORMAT   Output format: json, srt, vtt (default: json)
  --chunk           Apply semantic chunking to output (use with --output)
  --embed           Include embeddings in output (requires --chunk)
  -v, --verbose     Increase verbosity (-v for debug, -vv for trace)
```

Supported inputs:
- YouTube URLs (`https://youtube.com/watch?v=...`)
- YouTube video IDs (`dQw4w9WgXcQ`)
- YouTube playlists (`https://youtube.com/playlist?list=...`) with `--playlist`
- YouTube channels (`https://youtube.com/@channel`) with `--playlist`
- Local audio files (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.opus`, `.m4a`, `.wma`, `.aiff`, `.alac`)
- Local video files (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpeg`, `.mpg`, `.3gp`)

### `lytt ask <question>`

Ask a question and get an answer from your audio library.

```bash
lytt ask "your question here" [--model MODEL] [--max-chunks N]

Options:
  -m, --model MODEL        LLM model for response generation (default: gpt-4o-mini)
  -c, --max-chunks N       Maximum context chunks to include (default: 10)
```

### `lytt search <query>`

Search for relevant audio segments.

```bash
lytt search "search query" [--limit N] [--min-score SCORE]

Options:
  -l, --limit N          Maximum number of results (default: 5)
  -m, --min-score SCORE  Minimum similarity score 0.0-1.0 (default: 0.3)
```

### `lytt chat`

Start an interactive chat session with your audio knowledge base.

```bash
lytt chat [--model MODEL]

Commands during chat:
  exit, quit  - Exit the chat
  clear       - Clear conversation history
```

### `lytt list`

List all indexed media.

### `lytt rechunk <video_id>`

Re-chunk indexed media without re-transcribing.

```bash
lytt rechunk VIDEO_ID  # Rechunk single video
lytt rechunk all       # Rechunk all videos with stored transcripts
```

Useful when you've updated chunking settings or prompts and want to apply them to existing content.

Note: Only works for videos transcribed after the rechunk feature was added. Older videos need `--force` to re-transcribe first.

### `lytt serve`

Start HTTP API server for integration with other systems.

```bash
lytt serve [--host HOST] [--port PORT]

Options:
  --host HOST   Host to bind to (default: 127.0.0.1)
  -p, --port N  Port to bind to (default: 3000)
```

### `lytt mcp`

Start MCP (Model Context Protocol) server for Claude Desktop/Code integration.

```bash
lytt mcp
```

See [AGENTS.md](AGENTS.md) for MCP configuration details.

### `lytt doctor`

Verify system requirements and configuration.

```bash
lytt doctor
```

### `lytt init`

Interactive first-run setup.

```bash
lytt init
```

### `lytt config`

Manage configuration.

```bash
lytt config show   # Display current configuration
lytt config edit   # Open config file in editor
lytt config path   # Show config file path
```

## Configuration

Configuration is stored at `~/.config/lytt/config.toml`. Example:

```toml
[general]
data_dir = "~/.lytt"
temp_dir = "/tmp/lytt"
log_level = "info"

[transcription]
provider = "whisper"  # or "fusion"
model = "whisper-1"
chunk_duration_seconds = 120
max_duration_seconds = 7200  # 2 hours

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536

[chunking]
strategy = "semantic"  # or "temporal"
target_chunk_seconds = 180
min_chunk_seconds = 60
max_chunk_seconds = 600

[vector_store]
provider = "sqlite"
sqlite_path = "~/.lytt/vectors.db"

[rag]
enabled = true
model = "gpt-4o-mini"
max_context_chunks = 10
include_timestamps = true

[prompts]
custom_dir = "~/.lytt/prompts"

[prompts.variables]
# Custom variables available in all prompts as {{variable_name}}
language = "English"
style = "concise"
```

### Transcription Modes

Lytt supports two transcription modes. Both use LLM cleanup for better punctuation, sentence structure, and error correction.

#### Whisper (Default)

Uses OpenAI's Whisper API with LLM cleanup. Good balance of speed, cost, and quality.

```toml
[transcription]
provider = "whisper"
model = "whisper-1"

[transcription.processing]
cleanup_model = "gpt-4.1"  # Model used for cleanup
```

**How it works:**
1. Whisper transcribes with word-level timestamps
2. LLM cleans up the text (fixes punctuation, sentence structure, obvious errors)

**Advantages:**
- Fast transcription
- Lower API costs than full fusion
- LLM cleanup improves quality over raw Whisper output

#### Fusion Mode

Combines multiple models for maximum accuracy: Whisper provides word-level timestamps, a secondary model (GPT-4o) provides a separate transcription, and an LLM intelligently fuses both together.

```toml
[transcription]
provider = "fusion"

[transcription.processing]
timestamp_model = "whisper-1"      # For word timestamps
text_model = "gpt-4o-transcribe"   # Secondary transcription
cleanup_model = "gpt-4.1"          # For cleanup and fusion
max_concurrent = 2
```

**How it works:**
1. Whisper transcribes with word-level timestamps
2. GPT-4o independently transcribes the same audio
3. LLM fuses both transcripts, picking the best interpretation for each segment

**Advantages:**
- Highest accuracy, especially for technical terms, names, and jargon
- Cross-references two independent transcriptions
- Better handling of unclear audio or accents

**Trade-offs:**
- Higher API costs (multiple model calls per segment)
- Slower processing time

### Custom Prompts

Create custom prompt files in `~/.lytt/prompts/`:

- `chunking.toml` - Controls how transcripts are split into chunks
- `rag.toml` - Controls question answering responses
- `cleanup.toml` - Controls transcription cleanup and segment structuring

Example `chunking.toml`:
```toml
system = "You are a video content analyst..."
user = "Analyze this transcript: {{transcript}}\nTarget duration: {{target_duration}} seconds..."
```

Custom variables from `[prompts.variables]` are available in all prompts as `{{variable_name}}`.

## Architecture

Lytt is built with a modular, extensible architecture:

```
Audio Input → Download/Extract (yt-dlp/ffmpeg) → Transcribe (Whisper)
                                                      ↓
                                            Chunk (Semantic/Temporal)
                                                      ↓
                                            Embed (OpenAI) → Store (SQLite)
                                                      ↓
                            Query → Search → Context → LLM → Response
```

### Extending Lytt

The codebase uses Rust traits for extensibility:

- `AudioSource` - Add new audio sources (Spotify podcasts, RSS feeds, etc.)
- `VectorStore` - Use different vector databases (Qdrant, Pinecone, etc.)
- `Transcriber` - Implement local transcription models
- `Embedder` - Use different embedding providers
- `Chunker` - Implement custom chunking strategies

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://openai.com/research/whisper) for transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for audio downloading
- [ffmpeg](https://ffmpeg.org/) for audio processing
- The Rust community for excellent libraries
