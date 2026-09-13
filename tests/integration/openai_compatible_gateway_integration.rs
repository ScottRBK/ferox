use ferox_ai::Gateway;
use ferox_ai::error::{GatewayError, LlmError};
use ferox_ai::models::{CompletionRequest, Message, ModelModality};
use ferox_ai::openai_compatible::OpenAiCompatibleClient;
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODELS_RESPONSE: &str =
    include_str!("../../src/adapters/fixtures/models_response_with_modalities.json");

#[tokio::test]
async fn gateway_lists_models_and_maps_modalities() {
    // Arrange
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(MODELS_RESPONSE, "application/json"))
        .expect(1)
        .mount(&server)
        .await;
    let gateway = gateway(&server);

    // Act
    let models = gateway.list_models().await.unwrap();

    // Assert
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "text-model");
    assert!(matches!(
        models[0].input_modalities.as_slice(),
        [ModelModality::Text]
    ));
    assert!(matches!(
        models[0].output_modalities.as_slice(),
        [ModelModality::Text]
    ));
}

#[tokio::test]
async fn gateway_complete_maps_text_reasoning_and_tool_calls() {
    // Arrange
    let server = MockServer::start().await;
    let expected_request = json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": "Use the calculator."
        }],
        "stream": false,
        "tools": null
    });
    let response = json!({
        "id": "completion-1",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "I will calculate that.",
                "reasoning_content": "The user asked for a calculation.",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": "{\"a\":2,\"b\":3}"
                    }
                }]
            }
        }]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(expected_request))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .expect(1)
        .mount(&server)
        .await;
    let gateway = gateway(&server);
    let messages = [Message::User {
        content: "Use the calculator.".into(),
    }];

    // Act
    let request = CompletionRequest::new("test-model".into(), &messages);
    let completion = gateway.complete(request).await.unwrap();

    // Assert
    assert_eq!(completion.model, "test-model");
    assert_eq!(completion.text.as_deref(), Some("I will calculate that."));
    assert_eq!(
        completion.reasoning.as_deref(),
        Some("The user asked for a calculation.")
    );
    assert_eq!(completion.tool_calls.len(), 1);
    assert_eq!(completion.tool_calls[0].id, "call-1");
    assert_eq!(completion.tool_calls[0].name, "calculator");
    assert_eq!(completion.tool_calls[0].arguments, r#"{"a":2,"b":3}"#);
}

#[tokio::test]
async fn gateway_list_models_maps_provider_errors() {
    // Arrange
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).set_body_string("models unavailable"))
        .expect(1)
        .mount(&server)
        .await;
    let gateway = gateway(&server);

    // Act
    let error = gateway.list_models().await.unwrap_err();

    // Assert
    assert!(matches!(
        error,
        GatewayError::Llm(LlmError::ProviderUnavailable { code: 503, message })
            if message == "models unavailable"
    ));
}

fn gateway(server: &MockServer) -> Gateway<OpenAiCompatibleClient> {
    let client = OpenAiCompatibleClient::builder()
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .unwrap();
    Gateway::new(client)
}
