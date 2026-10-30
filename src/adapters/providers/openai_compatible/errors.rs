use crate::error::LlmError;
use thiserror::Error;

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
fn error_details(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();

    while let Some(error) = cause {
        message.push_str(&format!("\ncaused by: {error}"));
        cause = error.source();
    }

    message
}

impl From<OpenAiClientError> for LlmError {
    fn from(error: OpenAiClientError) -> Self {
        match error {
            OpenAiClientError::Transport(err) if err.is_timeout() => LlmError::Timeout,
            OpenAiClientError::Transport(err) => LlmError::Transport {
                message: error_details(&err),
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
                code: code @ 500..=599,
                body,
            } => LlmError::ProviderUnavailable {
                code,
                message: body,
            },
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

/// A failure to configure an OpenAI-compatible client before any request is sent.
#[derive(Debug, Error)]
pub enum ClientBuildError {
    /// No base URL was supplied to the builder.
    #[error("missing base_url parameter")]
    MissingBaseUrl,
    /// The underlying HTTP client could not be created.
    #[error("failed to build HTTP client {0}")]
    HttpClient(#[from] reqwest::Error),
}

#[cfg(test)]
mod tests {
    use super::error_details;

    #[derive(Debug, thiserror::Error)]
    #[error("{message}")]
    struct TestError {
        message: &'static str,
        source: Option<Box<TestError>>,
    }

    #[test]
    fn error_details_without_a_cause_returns_only_the_message() {
        // Arrange
        let error = TestError {
            message: "request failed",
            source: None,
        };

        // Act
        let details = error_details(&error);

        // Assert
        assert_eq!(details, "request failed");
    }

    #[test]
    fn error_details_includes_all_nested_causes_in_order() {
        // Arrange
        let error = TestError {
            message: "request failed",
            source: Some(Box::new(TestError {
                message: "connection failed",
                source: Some(Box::new(TestError {
                    message: "connection refused",
                    source: None,
                })),
            })),
        };

        // Act
        let details = error_details(&error);

        // Assert
        assert_eq!(
            details,
            "request failed\ncaused by: connection failed\ncaused by: connection refused"
        );
    }
}
