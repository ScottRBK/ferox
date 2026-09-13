//! ferox
//!
//! ferrox is a Large Language Model Provider gateway so that you can interact with multiple LLM
//! providers agnostically.
#![warn(missing_docs)]
mod adapters;
mod gateway;

pub mod error;
pub mod models;
mod ports;

pub use adapters::providers::openai_compatible;
pub use gateway::Gateway;
pub use ports::llm::LlmProvider;
