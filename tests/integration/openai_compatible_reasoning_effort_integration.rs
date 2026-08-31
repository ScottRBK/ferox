use ferox::adapters::providers::openai_compatible::OpenAiCompatibleClient;
use ferox::gateway::Gateway;
use ferox::models::{CompletionRequest, Message, ReasoningEffort};
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
    let response = gateway
        .complete(CompletionRequest {
            model: "qwen3.6-35b".into(),
            messages: &messages,
            tools: None,
            reasoning_effort: None,
        })
        .await
        .unwrap();

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
    let response = gateway
        .complete(CompletionRequest {
            model: "qwen3.6-35b".into(),
            messages: &messages,
            tools: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
        })
        .await
        .unwrap();

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
    let stream = gateway
        .stream(CompletionRequest {
            model: "qwen3.6-35b".into(),
            messages: &messages,
            tools: None,
            reasoning_effort: Some(ReasoningEffort::High),
        })
        .await
        .unwrap();
    pin_mut!(stream);
    stream
        .next()
        .await
        .expect("expected the initial completion chunk")
        .expect("expected a valid initial completion chunk");
    let reasoning_chunk = stream
        .next()
        .await
        .expect("expected a reasoning chunk")
        .expect("expected a valid reasoning chunk");

    // Assert
    assert_eq!(reasoning_chunk.reasoning.as_deref(), Some("Here"));
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
