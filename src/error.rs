use std::fmt;
use std::error::Error;
use std::time::Duration;

#[derive(Debug)]
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
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::InvalidRequest { message } => write!(f, "invalid request: {message}"),
            LlmError::AuthenticationFailed => write!(f, "authentication failed"),
            LlmError::PermissionDenied => write!(f, "permission denied"),
            LlmError::RateLimited{ retry_after } => write!(f, "rate limited (retry after: {retry_after:?})"),
            LlmError::Timeout => write!(f, "timeout"),
            LlmError::ProviderUnavailable => write!(f, "provider unavailable"),
            LlmError::ProviderFailure { message } => write!(f, "provider failure: {message}"),
            LlmError::Transport { message } => write!(f, "transport error: {message}"),
            LlmError::InvalidResponse { message } => write!(f, "invalid response: {message}"),
        }
    }
}

impl Error for LlmError {}

#[derive(Debug)]
pub enum GatewayError {
    UnsupportedModel(String),
    ProviderNotConfigured(String),
    PolicyDenied,
    Llm(LlmError),
}


impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GatewayError::UnsupportedModel(m) => write!(f, "unsupported model: {m}"),
            GatewayError::ProviderNotConfigured(p) => write!(f, "provider not configured: {p}"),
            GatewayError::PolicyDenied => write!(f, "policy denied"),
            GatewayError::Llm(e) => write!(f, "{e}"),
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

impl From<LlmError> for GatewayError {fn from(e: LlmError) -> Self {GatewayError::Llm(e)}}
