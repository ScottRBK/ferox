use thiserror::Error; 
use std::time::Duration;

#[derive(Debug,Error)]
#[non_exhaustive]
pub enum LlmError {
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("authentication failed")]
    AuthenticationFailed,
    #[error("permission denied")]
    PermissionDenied,
    #[error("rate limited (retry after: {retry_after:?})")]
    RateLimited { retry_after: Option<Duration> },
    #[error("timeout")]
    Timeout,
    #[error("provider unavailable")]
    ProviderUnavailable,
    #[error("provider failure: {message}")]
    ProviderFailure { message: String },
    #[error("transport error: {message}")]
    Transport { message: String },
    #[error("invalid response: {message}")]
    InvalidResponse { message: String },
    #[error("invalid model modality {modality}")]
    InvalidModelModality{ modality: String },
}


#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GatewayError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("provider not configured: {0}")]
    ProviderNotConfigured(String),
    #[error("policy denied")]
    PolicyDenied,
    #[error("{0}")]
    Llm(#[from]LlmError),
}
