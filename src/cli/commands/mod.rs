//! CLI command implementations.

mod agent;
mod ask;
mod chat;
mod config;
mod doctor;
mod export;
mod init;
mod list;
mod mcp;
mod rechunk;
mod search;
mod serve;
mod transcribe;

pub use agent::run_agent;
pub use ask::run_ask;
pub use chat::run_chat;
pub use config::run_config;
pub use doctor::run_doctor;
pub use export::run_export;
pub use init::run_init;
pub use list::run_list;
pub use mcp::run_mcp;
pub use rechunk::run_rechunk;
pub use search::run_search;
pub use serve::run_serve;
pub use transcribe::run_transcribe;
