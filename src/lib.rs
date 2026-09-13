//! ferox-ai provides a shared interface for interacting with LLM providers.
#![warn(missing_docs)]
mod adapters;
mod gateway;

pub mod error;
pub mod models;
mod ports;

pub use adapters::providers::openai_compatible;
pub use gateway::Gateway;
pub use ports::llm::LlmProvider;
