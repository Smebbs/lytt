//! Transcript output formatting (JSON, SRT, VTT).
//!
//! Provides utilities for exporting transcripts in standard formats
//! for integration with other systems.

use super::Transcript;
use serde::Serialize;

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Json,
    Srt,
    Vtt,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "srt" => Ok(OutputFormat::Srt),
            "vtt" | "webvtt" => Ok(OutputFormat::Vtt),
            _ => Err(format!("Unknown format: {}. Use json, srt, or vtt.", s)),
        }
    }
}

/// JSON-serializable transcript for export.
#[derive(Debug, Serialize)]
pub struct TranscriptExport {
    pub media_id: String,
    pub duration_seconds: f64,
    pub segments: Vec<SegmentExport>,
}

#[derive(Debug, Serialize)]
pub struct SegmentExport {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl From<&Transcript> for TranscriptExport {
    fn from(transcript: &Transcript) -> Self {
        Self {
            media_id: transcript.video_id.clone(),
            duration_seconds: transcript.duration_seconds,
            segments: transcript
                .segments
                .iter()
                .map(|s| SegmentExport {
                    text: s.text.clone(),
                    start_seconds: s.start_seconds,
                    end_seconds: s.end_seconds,
                    title: None,
                })
                .collect(),
        }
    }
}

/// Format a transcript for output.
pub fn format_transcript(transcript: &Transcript, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => format_json(transcript),
        OutputFormat::Srt => format_srt(transcript),
        OutputFormat::Vtt => format_vtt(transcript),
    }
}

/// Format as JSON.
fn format_json(transcript: &Transcript) -> String {
    let export = TranscriptExport::from(transcript);
    serde_json::to_string_pretty(&export).unwrap_or_else(|_| "{}".to_string())
}

/// Format as SRT (SubRip).
fn format_srt(transcript: &Transcript) -> String {
    let mut output = String::new();

    for (i, segment) in transcript.segments.iter().enumerate() {
        // Sequence number (1-indexed)
        output.push_str(&format!("{}\n", i + 1));

        // Timestamps: 00:00:00,000 --> 00:00:00,000
        output.push_str(&format!(
            "{} --> {}\n",
            format_srt_timestamp(segment.start_seconds),
            format_srt_timestamp(segment.end_seconds)
        ));

        // Text
        output.push_str(&segment.text);
        output.push_str("\n\n");
    }

    output
}

/// Format as WebVTT.
fn format_vtt(transcript: &Transcript) -> String {
    let mut output = String::from("WEBVTT\n\n");

    for (i, segment) in transcript.segments.iter().enumerate() {
        // Optional cue identifier
        output.push_str(&format!("{}\n", i + 1));

        // Timestamps: 00:00:00.000 --> 00:00:00.000
        output.push_str(&format!(
            "{} --> {}\n",
            format_vtt_timestamp(segment.start_seconds),
            format_vtt_timestamp(segment.end_seconds)
        ));

        // Text
        output.push_str(&segment.text);
        output.push_str("\n\n");
    }

    output
}

/// Format timestamp for SRT (00:00:00,000).
fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, ms)
}

/// Format timestamp for VTT (00:00:00.000).
fn format_vtt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::TranscriptSegment;

    fn sample_transcript() -> Transcript {
        Transcript::new(
            "test123".to_string(),
            vec![
                TranscriptSegment {
                    text: "Hello world.".to_string(),
                    start_seconds: 0.0,
                    end_seconds: 2.5,
                },
                TranscriptSegment {
                    text: "This is a test.".to_string(),
                    start_seconds: 2.5,
                    end_seconds: 5.0,
                },
            ],
        )
    }

    #[test]
    fn test_format_json() {
        let transcript = sample_transcript();
        let json = format_transcript(&transcript, OutputFormat::Json);
        assert!(json.contains("\"media_id\": \"test123\""));
        assert!(json.contains("Hello world."));
    }

    #[test]
    fn test_format_srt() {
        let transcript = sample_transcript();
        let srt = format_transcript(&transcript, OutputFormat::Srt);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world."));
    }

    #[test]
    fn test_format_vtt() {
        let transcript = sample_transcript();
        let vtt = format_transcript(&transcript, OutputFormat::Vtt);
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.500"));
    }

    #[test]
    fn test_parse_format() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("srt".parse::<OutputFormat>().unwrap(), OutputFormat::Srt);
        assert_eq!("vtt".parse::<OutputFormat>().unwrap(), OutputFormat::Vtt);
        assert_eq!("webvtt".parse::<OutputFormat>().unwrap(), OutputFormat::Vtt);
    }

    #[test]
    fn test_srt_timestamp() {
        assert_eq!(format_srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(format_srt_timestamp(61.5), "00:01:01,500");
        assert_eq!(format_srt_timestamp(3661.123), "01:01:01,123");
    }
}
