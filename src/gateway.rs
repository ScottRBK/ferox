use crate::error::GatewayError;
use crate::models::{CompletionChunk, CompletionRequest, CompletionResponse, Model};
use crate::ports::llm::LlmProvider;
use futures_core::Stream;
use futures_util::StreamExt;

/// Sends completion and model-list requests to one configured provider.
///
/// Provider failures are returned as [`GatewayError::Llm`].
pub struct Gateway<P> {
    provider: P,
}

impl<P> Gateway<P>
where
    P: LlmProvider,
{
    /// Creates a gateway that owns the supplied provider.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Requests a complete response, waiting until the provider finishes.
    ///
    /// # Errors
    ///
    /// Returns [`GatewayError::Llm`] if the provider cannot complete the request.
    pub async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<CompletionResponse, GatewayError> {
        self.provider
            .complete(request)
            .await
            .map_err(GatewayError::from)
    }

    /// Starts a completion and returns a stream of response updates.
    ///
    /// Append each chunk's text and reasoning to assemble the response.
    ///
    /// # Errors
    ///
    /// Returns [`GatewayError::Llm`] if the provider cannot start the stream.
    /// Failures while reading the response are returned as error items in the stream.
    pub async fn stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<
        impl Stream<Item = Result<CompletionChunk, GatewayError>> + Send + use<P>,
        GatewayError,
    > {
        let stream = self
            .provider
            .stream(request)
            .await
            .map_err(GatewayError::from)?;

        Ok(stream.map(|chunk| chunk.map_err(GatewayError::from)))
    }

    /// Fetches the models advertised by the configured provider.
    ///
    /// # Errors
    ///
    /// Returns [`GatewayError::Llm`] if the provider cannot fetch or decode its model list.
    pub async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        self.provider
            .list_models()
            .await
            .map_err(GatewayError::from)
    }
}
