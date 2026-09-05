mod client;
mod models;
mod mapping;
mod errors;

pub use client::{ OpenAiCompatibleClient, OpenAiCompatibleClientBuilder };
pub use errors:: { ClientBuildError };


