use crate::error::GatewayError;
use crate::models::{CompletionChunk, CompletionRequest, CompletionResponse, Model};
use crate::ports::llm::LlmProvider;
use futures_core::Stream;
use futures_util::StreamExt;

pub struct Gateway<P> {
    provider: P,
}

impl<P> Gateway<P>
where
    P: LlmProvider,
{
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    pub async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<CompletionResponse, GatewayError> {
        self.provider
            .complete(request)
            .await
            .map_err(GatewayError::from)
    }

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

    pub async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        self.provider
            .list_models()
            .await
            .map_err(GatewayError::from)
    }
}
