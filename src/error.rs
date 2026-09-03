use std::error::Error;
use std::fmt;
use std::time::Duration;

#[derive(Debug)]
#[non_exhaustive]
pub enum LlmError {
    InvalidRequest { message: String },
    AuthenticationFailed,
    PermissionDenied,
    RateLimited { retry_after: Option<Duration> },
    Timeout,
    ProviderUnavailable,
    ProviderFailure { message: String },
    Transport { message: String },
    InvalidResponse { message: String },
    InvalidModelModality{ modality: String },
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LlmError::*;
        match self {
            InvalidRequest { message } => write!(f, "invalid request: {message}"),
            AuthenticationFailed => write!(f, "authentication failed"),
            PermissionDenied => write!(f, "permission denied"),
            RateLimited { retry_after } => {
                write!(f, "rate limited (retry after: {retry_after:?})")
            }
            Timeout => write!(f, "timeout"),
            ProviderUnavailable => write!(f, "provider unavailable"),
            ProviderFailure { message } => write!(f, "provider failure: {message}"),
            Transport { message } => write!(f, "transport error: {message}"),
            InvalidResponse { message } => write!(f, "invalid response: {message}"),
            InvalidModelModality{ modality }=> write!(f, "invalid model modality {modality}")
        }
    }
}

impl Error for LlmError {}

#[derive(Debug)]
#[non_exhaustive]
pub enum GatewayError {
    UnsupportedModel(String),
    ProviderNotConfigured(String),
    PolicyDenied,
    Llm(LlmError),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GatewayError::*;
        match self {
            UnsupportedModel(m) => write!(f, "unsupported model: {m}"),
            ProviderNotConfigured(p) => write!(f, "provider not configured: {p}"),
            PolicyDenied => write!(f, "policy denied"),
            Llm(e) => write!(f, "{e}"),
        }
    }
}

impl Error for GatewayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            GatewayError::Llm(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LlmError> for GatewayError {
    fn from(e: LlmError) -> Self {
        GatewayError::Llm(e)
    }
}
