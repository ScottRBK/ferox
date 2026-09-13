//! Client configuration and provider support for OpenAI-compatible APIs.
//!
//! Set the API root with [`OpenAiCompatibleClient::builder`] and pass the built client
//! to [`crate::Gateway`]. An API key is optional for endpoints that allow unauthenticated requests.

mod client;
mod errors;
mod mapping;
mod models;

pub use client::{OpenAiCompatibleClient, OpenAiCompatibleClientBuilder};
pub use errors::ClientBuildError;
