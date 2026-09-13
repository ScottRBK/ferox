mod client;
mod errors;
mod mapping;
mod models;

pub use client::{OpenAiCompatibleClient, OpenAiCompatibleClientBuilder};
pub use errors::ClientBuildError;
