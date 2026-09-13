//! Shared errors for provider operations and gateway requests.

use std::time::Duration;
use thiserror::Error;

/// A provider failure expressed independently of the provider's API.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlmError {
    /// The provider rejected the request as invalid.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Details explaining why the request was rejected.
        message: String,
    },
    /// The provider did not accept the supplied credentials.
    #[error("authentication failed")]
    AuthenticationFailed,
    /// The caller does not have permission to perform this operation.
    #[error("permission denied")]
    PermissionDenied,
    /// The provider rejected the request because a rate limit was reached.
    #[error("rate limited (retry after: {retry_after:?})")]
    RateLimited {
        /// Suggested delay before retrying, when available. Ferox does not retry automatically.
        retry_after: Option<Duration>,
    },
    /// The request timed out locally or at the provider.
    #[error("timeout")]
    Timeout,
    /// The provider returned a server error.
    #[error("provider unavailable ({code}): {message}")]
    ProviderUnavailable {
        /// HTTP status code returned by the provider.
        code: u16,
        /// Error details returned by the provider.
        message: String,
    },
    /// The provider reported a failure without a more specific error classification.
    #[error("provider failure: {message}")]
    ProviderFailure {
        /// Details of the provider failure.
        message: String,
    },
    /// Sending the request or reading the response failed.
    #[error("transport error: {message}")]
    Transport {
        /// Details of the connection or data transfer failure.
        message: String,
    },
    /// The response could not be decoded or was missing required data.
    #[error("invalid response: {message}")]
    InvalidResponse {
        /// Details of the unexpected response data.
        message: String,
    },
    /// The provider advertised an input or output format that ferox does not recognise.
    #[error("invalid model modality {modality}")]
    InvalidModelModality {
        /// Unrecognised format name from the provider's model metadata.
        modality: String,
    },
}

/// A gateway request failure.
///
/// The current gateway forwards provider failures through [`Self::Llm`].
/// The other variants describe routing and policy failures and are not currently emitted by it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GatewayError {
    /// The requested model identifier is unsupported.
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    /// The named provider has not been configured.
    #[error("provider not configured: {0}")]
    ProviderNotConfigured(String),
    /// Gateway policy denied the request.
    #[error("policy denied")]
    PolicyDenied,
    /// The underlying provider operation failed.
    #[error("{0}")]
    Llm(#[from] LlmError),
}
