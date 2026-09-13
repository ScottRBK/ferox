use futures_core::Stream;
use std::future::Future;

use crate::error::LlmError;
use crate::models::{CompletionChunk, CompletionRequest, CompletionResponse, Model};

/// The operations a provider adapter must implement for use with [`crate::Gateway`].
///
/// Adapters translate provider requests, responses, and errors into the shared ferox types.
pub trait LlmProvider {
    /// An owned response stream that can outlive the provider borrow and request.
    ///
    /// Each item contains a response update or an error encountered while reading it.
    type CompletionStream: Stream<Item = Result<CompletionChunk, LlmError>> + Send + 'static;

    /// Requests a full response for the supplied conversation.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] if the request fails or the response cannot be decoded.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, LlmError>> + Send;

    /// Starts a completion and returns a stream of response updates.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] if the stream cannot be started. Failures after startup
    /// are returned as error items in [`Self::CompletionStream`].
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<Self::CompletionStream, LlmError>> + Send;

    /// Fetches the models advertised by this provider.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] if the model list cannot be fetched or decoded.
    fn list_models(&self) -> impl Future<Output = Result<Vec<Model>, LlmError>> + Send;
}
