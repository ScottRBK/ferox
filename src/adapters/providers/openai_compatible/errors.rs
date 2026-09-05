use thiserror::Error;
use crate::error::LlmError;

#[derive(Debug, Error)]
pub enum OpenAiClientError {
    #[error("transport error: {0}")]
    Transport(reqwest::Error),
    #[error("provider status {code}: {body}")]
    Status { code: u16, body: String },
    #[error("invalid response {0}")]
    Decode(serde_json::Error),
    #[error("invalid utf8: {0}")]
    Utf8(std::str::Utf8Error),
}

impl From<OpenAiClientError> for LlmError {
    fn from(error: OpenAiClientError) -> Self {
        match error {
            OpenAiClientError::Transport(err) if err.is_timeout() => LlmError::Timeout,
            OpenAiClientError::Transport(err) => LlmError::Transport {
                message: err.to_string(),
            },
            OpenAiClientError::Status { code: 401, .. } => LlmError::AuthenticationFailed,
            OpenAiClientError::Status { code: 403, .. } => LlmError::PermissionDenied,
            OpenAiClientError::Status { code: 429, .. } => {
                LlmError::RateLimited { retry_after: None }
            }
            OpenAiClientError::Status {
                code: 400 | 422,
                body,
            } => LlmError::InvalidRequest { message: body },
            OpenAiClientError::Status { code: 408, .. } => LlmError::Timeout,
            OpenAiClientError::Status {
                code: 500..=599, ..
            } => LlmError::ProviderUnavailable,
            OpenAiClientError::Status { body, .. } => LlmError::ProviderFailure { message: body },
            OpenAiClientError::Decode(err) => LlmError::InvalidResponse {
                message: err.to_string(),
            },
            OpenAiClientError::Utf8(err) => LlmError::InvalidResponse {
                message: err.to_string(),
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum ClientBuildError {
    #[error("missing base_url parameter")]
    MissingBaseUrl,
    #[error("failed to build HTTP client {0}")]
    HttpClient(#[from]reqwest::Error),
}

