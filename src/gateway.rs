use crate::error::GatewayError;
use crate::models::{CompletionRequest, CompletionResponse, Model};
use crate::ports::llm::LlmProvider;

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
    }

    pub async fn stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<P::CompletionStream, GatewayError> {
        self.provider
            .stream(request)
            .await
    }

    pub async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        self.provider.list_models()
            .await
    }
}
