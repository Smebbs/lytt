//! Local file source implementation.
//!
//! Supports both audio and video files.

use super::{AudioSource, MediaMetadata, SourceType};
use crate::error::{Result, LyttError};
use async_trait::async_trait;
use std::path::Path;

/// Supported audio file extensions.
const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "wav", "flac", "aac", "ogg", "opus", "m4a", "wma", "aiff", "alac",
];

/// Supported video file extensions (audio will be extracted).
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "mpeg", "mpg", "3gp",
];

/// Local file source for audio and video files.
pub struct LocalSource;

impl LocalSource {
    pub fn new() -> Self {
        Self
    }

    /// Check if path is a supported audio file.
    fn is_audio_file(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Check if path is a supported video file.
    fn is_video_file(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| VIDEO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// Check if path is a supported media file (audio or video).
    fn is_media_file(path: &Path) -> bool {
        Self::is_audio_file(path) || Self::is_video_file(path)
    }

    /// Get media metadata using ffprobe.
    async fn get_metadata_ffprobe(path: &Path) -> Result<(Option<u32>, Option<String>)> {
        let output = tokio::process::Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path.to_str().unwrap_or(""),
            ])
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    LyttError::ToolNotFound("ffprobe".to_string())
                } else {
                    LyttError::AudioDownload(format!("Failed to run ffprobe: {}", e))
                }
            })?;

        if !output.status.success() {
            // ffprobe failed, but we can still proceed without metadata
            return Ok((None, None));
        }

        let json_str = String::from_utf8_lossy(&output.stdout);
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();

        let duration = json["format"]["duration"]
            .as_str()
            .and_then(|d| d.parse::<f64>().ok())
            .map(|d| d as u32);

        let title = json["format"]["tags"]["title"]
            .as_str()
            .map(|s| s.to_string());

        Ok((duration, title))
    }
}

impl Default for LocalSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioSource for LocalSource {
    fn source_type(&self) -> SourceType {
        SourceType::Local
    }

    async fn fetch_media(&self, id: &str) -> Result<MediaMetadata> {
        let path = Path::new(id);

        if !path.exists() {
            return Err(LyttError::VideoNotFound(format!(
                "File not found: {}",
                id
            )));
        }

        if !Self::is_media_file(path) {
            return Err(LyttError::InvalidInput(format!(
                "Not a recognized audio or video file: {}",
                id
            )));
        }

        let (duration, metadata_title) = Self::get_metadata_ffprobe(path).await?;

        let title = metadata_title.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string()
        });

        // Generate a stable ID from the file path
        let media_id = format!(
            "local_{}",
            path.canonicalize()
                .unwrap_or_else(|_| path.to_path_buf())
                .to_string_lossy()
                .replace(['/', '\\', ' '], "_")
        );

        Ok(MediaMetadata {
            id: media_id,
            title,
            description: None,
            duration_seconds: duration,
            source_type: SourceType::Local,
            source_url: path
                .canonicalize()
                .unwrap_or_else(|_| path.to_path_buf())
                .to_string_lossy()
                .to_string(),
            published_at: None,
            channel: None,
            thumbnail_url: None,
        })
    }

    async fn list_media(&self, source: &str, limit: Option<usize>) -> Result<Vec<MediaMetadata>> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LyttError::VideoNotFound(format!(
                "Directory not found: {}",
                source
            )));
        }

        if !path.is_dir() {
            return Err(LyttError::InvalidInput(format!(
                "Not a directory: {}",
                source
            )));
        }

        let mut media_files = Vec::new();
        let limit = limit.unwrap_or(usize::MAX);

        let entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .collect();

        for entry in entries {
            if media_files.len() >= limit {
                break;
            }

            let entry_path = entry.path();
            if Self::is_media_file(&entry_path) {
                match self.fetch_media(entry_path.to_str().unwrap_or("")).await {
                    Ok(metadata) => media_files.push(metadata),
                    Err(e) => {
                        tracing::warn!("Failed to get metadata for {:?}: {}", entry_path, e);
                    }
                }
            }
        }

        Ok(media_files)
    }

    fn can_handle(&self, input: &str) -> bool {
        let path = Path::new(input);
        path.exists() || Self::is_media_file(path)
    }

    fn extract_id(&self, input: &str) -> Option<String> {
        let path = Path::new(input);
        if path.exists() || Self::is_media_file(path) {
            Some(input.to_string())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_audio_file() {
        assert!(LocalSource::is_audio_file(Path::new("audio.mp3")));
        assert!(LocalSource::is_audio_file(Path::new("audio.WAV")));
        assert!(LocalSource::is_audio_file(Path::new("/path/to/audio.flac")));
        assert!(!LocalSource::is_audio_file(Path::new("video.mp4")));
        assert!(!LocalSource::is_audio_file(Path::new("document.pdf")));
    }

    #[test]
    fn test_is_video_file() {
        assert!(LocalSource::is_video_file(Path::new("video.mp4")));
        assert!(LocalSource::is_video_file(Path::new("video.MKV")));
        assert!(!LocalSource::is_video_file(Path::new("audio.mp3")));
    }

    #[test]
    fn test_is_media_file() {
        assert!(LocalSource::is_media_file(Path::new("video.mp4")));
        assert!(LocalSource::is_media_file(Path::new("audio.mp3")));
        assert!(!LocalSource::is_media_file(Path::new("document.pdf")));
    }
}
