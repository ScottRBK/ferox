use ferox::Gateway;
use ferox::error::{GatewayError, LlmError};
use ferox::models::{CompletionRequest, Message};
use ferox::openai_compatible::OpenAiCompatibleClient;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn provider_unavailable_preserves_status_and_body() {
    // Arrange
    let error = completion_error(503, Some("model backend failed")).await;

    // Act and Assert
    match error {
        GatewayError::Llm(LlmError::ProviderUnavailable { code, message }) => {
            assert_eq!(code, 503);
            assert_eq!(message, "model backend failed");
        }
        other => panic!("expected provider unavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_request_preserves_provider_body() {
    // Arrange
    let error = completion_error(400, Some("unsupported reasoning effort: max")).await;

    // Act and Assert
    match error {
        GatewayError::Llm(LlmError::InvalidRequest { message }) => {
            assert_eq!(message, "unsupported reasoning effort: max");
        }
        other => panic!("expected invalid request, got {other:?}"),
    }
}

#[tokio::test]
async fn unprocessable_request_preserves_provider_body() {
    // Arrange
    let error = completion_error(422, Some("reasoning effort is not supported")).await;

    // Act and Assert
    match error {
        GatewayError::Llm(LlmError::InvalidRequest { message }) => {
            assert_eq!(message, "reasoning effort is not supported");
        }
        other => panic!("expected invalid request, got {other:?}"),
    }
}

#[tokio::test]
async fn provider_unavailable_without_body_still_preserves_status() {
    // Arrange
    let error = completion_error(500, None).await;

    // Act and Assert
    match error {
        GatewayError::Llm(LlmError::ProviderUnavailable { code, message }) => {
            assert_eq!(code, 500);
            assert!(message.is_empty());
        }
        other => panic!("expected provider unavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn authentication_failure_is_mapped_from_401() {
    let error = completion_error(401, Some("invalid key")).await;

    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::AuthenticationFailed)
    ));
}

#[tokio::test]
async fn permission_denied_is_mapped_from_403() {
    let error = completion_error(403, Some("forbidden model")).await;

    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::PermissionDenied)
    ));
}

#[tokio::test]
async fn rate_limit_is_mapped_from_429() {
    let error = completion_error(429, Some("slow down")).await;

    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::RateLimited { retry_after: None })
    ));
}

#[tokio::test]
async fn timeout_is_mapped_from_408() {
    let error = completion_error(408, Some("request timed out")).await;

    assert!(matches!(error, GatewayError::Llm(LlmError::Timeout)));
}

#[tokio::test]
async fn other_client_errors_are_mapped_to_provider_failure() {
    let error = completion_error(404, Some("model not found")).await;

    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::ProviderFailure { message })
            if message == "model not found"
    ));
}

#[tokio::test]
async fn streaming_provider_error_is_returned_before_stream_creation() {
    // Arrange
    let error = streaming_error(503, Some("stream backend failed")).await;

    // Act and Assert
    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::ProviderUnavailable { code: 503, message })
            if message == "stream backend failed"
    ));
}

async fn completion_error(status: u16, body: Option<&str>) -> GatewayError {
    let server = MockServer::start().await;
    let response = match body {
        Some(body) => ResponseTemplate::new(status).set_body_string(body),
        None => ResponseTemplate::new(status),
    };

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(response)
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiCompatibleClient::builder()
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .expect("test client should build");
    let gateway = Gateway::new(client);
    let messages = [Message::User {
        content: "Hello".into(),
    }];
    let request = CompletionRequest::new("test-model".into(), &messages);

    match gateway.complete(request).await {
        Ok(_) => panic!("the mocked provider should return an error"),
        Err(error) => error,
    }
}

async fn streaming_error(status: u16, body: Option<&str>) -> GatewayError {
    let server = MockServer::start().await;
    let response = match body {
        Some(body) => ResponseTemplate::new(status).set_body_string(body),
        None => ResponseTemplate::new(status),
    };

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(response)
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiCompatibleClient::builder()
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .expect("test client should build");
    let gateway = Gateway::new(client);
    let messages = [Message::User {
        content: "Hello".into(),
    }];
    let request = CompletionRequest::new("test-model".into(), &messages);

    match gateway.stream(request).await {
        Ok(_) => panic!("the mocked provider should return an error"),
        Err(error) => error,
    }
}
