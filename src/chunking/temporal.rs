//! Time-based chunking implementation.
//!
//! Splits transcripts into chunks based on time intervals.

use super::{Chunker, ChunkingConfig, ContentChunk};
use crate::error::Result;
use crate::transcription::Transcript;
use async_trait::async_trait;

/// Time-based chunker.
///
/// Splits transcripts into fixed-duration chunks.
pub struct TemporalChunker;

impl TemporalChunker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TemporalChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Chunker for TemporalChunker {
    async fn chunk(&self, transcript: &Transcript, config: &ChunkingConfig) -> Result<Vec<ContentChunk>> {
        let mut chunks = Vec::new();
        let target_duration = config.target_duration as f64;

        if transcript.segments.is_empty() {
            return Ok(chunks);
        }

        let total_duration = transcript.duration_seconds;
        let mut chunk_start = 0.0;
        let mut chunk_order = 0;

        while chunk_start < total_duration {
            let chunk_end = (chunk_start + target_duration).min(total_duration);

            // Collect all segments that fall within this time range
            let chunk_content: String = transcript
                .segments
                .iter()
                .filter(|seg| {
                    // Include segment if it overlaps with the chunk time range
                    seg.start_seconds < chunk_end && seg.end_seconds > chunk_start
                })
                .map(|seg| seg.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            if !chunk_content.trim().is_empty() {
                chunks.push(ContentChunk::new(
                    None, // No semantic title for temporal chunks
                    chunk_content.trim().to_string(),
                    chunk_start,
                    chunk_end,
                    chunk_order,
                ));
                chunk_order += 1;
            }

            chunk_start = chunk_end;
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::TranscriptSegment;

    #[tokio::test]
    async fn test_temporal_chunking() {
        let chunker = TemporalChunker::new();

        let segments = vec![
            TranscriptSegment::new(0.0, 30.0, "First segment".to_string()),
            TranscriptSegment::new(30.0, 60.0, "Second segment".to_string()),
            TranscriptSegment::new(60.0, 90.0, "Third segment".to_string()),
            TranscriptSegment::new(90.0, 120.0, "Fourth segment".to_string()),
        ];

        let transcript = Transcript::new("test".to_string(), segments);

        let config = ChunkingConfig {
            target_duration: 60,
            min_duration: 30,
            max_duration: 120,
        };

        let chunks = chunker.chunk(&transcript, &config).await.unwrap();

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("First"));
        assert!(chunks[0].content.contains("Second"));
        assert!(chunks[1].content.contains("Third"));
        assert!(chunks[1].content.contains("Fourth"));
    }
}
