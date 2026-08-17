use futures_core::Stream;
use std::future::Future;

use crate::error::LlmError;
use crate::models::{CompletionChunk, CompletionRequest, CompletionResponse, Model};

pub trait LlmProvider {
    type CompletionStream: Stream<Item = Result<CompletionChunk, LlmError>> + Send + 'static;

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, LlmError>> + Send;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<Self::CompletionStream, LlmError>>;

    fn list_models(&self) -> impl Future<Output = Result<Vec<Model>, LlmError>>;
}
