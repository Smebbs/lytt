//! Audio download and processing module.

mod downloader;

pub use downloader::{download_audio, split_audio};
