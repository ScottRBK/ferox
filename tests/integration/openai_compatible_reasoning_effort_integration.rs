use ferox_ai::Gateway;
use ferox_ai::models::{CompletionRequest, FinishReason, Message, ReasoningEffort};
use ferox_ai::openai_compatible::OpenAiCompatibleClient;
use futures_util::{StreamExt, pin_mut};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STREAMING_RESPONSE_FIXTURE: &str =
    include_str!("../../src/adapters/fixtures/chat_completions_response_stream.json");

#[tokio::test]
async fn reasoning_effort_is_omitted_when_not_supplied() {
    // Arrange
    let server = MockServer::start().await;
    let expected_request = json!({
        "model": "qwen3.6-35b",
        "messages": [{
            "role": "user",
            "content": "Hello"
        }],
        "stream": false,
        "tools": null
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(expected_request))
        .respond_with(completion_response())
        .expect(1)
        .mount(&server)
        .await;

    let gateway = gateway(&server);
    let messages = [Message::User {
        content: "Hello".into(),
    }];

    // Act
    let request = CompletionRequest::new("qwen3.6-35b".into(), &messages);
    let response = gateway.complete(request).await.unwrap();

    // Assert
    assert_eq!(response.text.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn reasoning_effort_is_sent_when_supplied() {
    // Arrange
    let server = MockServer::start().await;
    let expected_request = json!({
        "model": "qwen3.6-35b",
        "messages": [{
            "role": "user",
            "content": "Hello"
        }],
        "stream": false,
        "tools": null,
        "reasoning_effort": "medium"
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(expected_request))
        .respond_with(completion_response())
        .expect(1)
        .mount(&server)
        .await;

    let gateway = gateway(&server);
    let messages = [Message::User {
        content: "Hello".into(),
    }];

    // Act
    let mut request = CompletionRequest::new("qwen3.6-35b".into(), &messages);
    request.reasoning_effort = Some(ReasoningEffort::Medium);
    let response = gateway.complete(request).await.unwrap();

    // Assert
    assert_eq!(response.text.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn reasoning_effort_is_sent_for_streaming_requests() {
    // Arrange
    let server = MockServer::start().await;
    let expected_request = json!({
        "model": "qwen3.6-35b",
        "messages": [{
            "role": "user",
            "content": "Hello"
        }],
        "stream": true,
        "tools": null,
        "reasoning_effort": "high"
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(expected_request))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(STREAMING_RESPONSE_FIXTURE, "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let gateway = gateway(&server);
    let messages = [Message::User {
        content: "Hello".into(),
    }];

    // Act
    let mut request = CompletionRequest::new("qwen3.6-35b".into(), &messages);
    request.reasoning_effort = Some(ReasoningEffort::High);
    let stream = gateway.stream(request).await.unwrap();
    pin_mut!(stream);
    let mut reasoning = String::new();
    let mut text = String::new();
    let mut finished_reason = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("expected a valid completion chunk");
        if let Some(value) = chunk.reasoning {
            reasoning.push_str(&value);
        }
        if let Some(value) = chunk.text {
            text.push_str(&value);
        }
        if chunk.finished_reason.is_some() {
            finished_reason = chunk.finished_reason;
        }
    }

    // Assert
    assert!(reasoning.starts_with("Here's"));
    assert!(reasoning.contains("thinking process"));
    assert_eq!(text, "Hey! How can I help you today? 😊");
    assert!(matches!(finished_reason, Some(FinishReason::Stop)));
}

fn gateway(server: &MockServer) -> Gateway<OpenAiCompatibleClient> {
    let client = OpenAiCompatibleClient::builder()
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .unwrap();

    Gateway::new(client)
}

fn completion_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "completion-1",
        "created": 1,
        "model": "qwen3.6-35b",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello"
            }
        }]
    }))
}
