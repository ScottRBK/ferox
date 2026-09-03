use futures_core::Stream;
use std::future::Future;

use crate::error::GatewayError;
use crate::models::{CompletionChunk, CompletionRequest, CompletionResponse, Model};

pub trait LlmProvider {
    type CompletionStream: Stream<Item = Result<CompletionChunk, GatewayError>> + Send + 'static;

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, GatewayError>> + Send;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<Self::CompletionStream, GatewayError>>;

    fn list_models(&self) -> impl Future<Output = Result<Vec<Model>, GatewayError>>;
}
