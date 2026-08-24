use ferox::adapters::providers::openai_compatible::OpenAiCompatibleClient;
use ferox::gateway::Gateway;
use ferox::models::{CompletionRequest, Message};
use ferox::ports::llm::LlmProvider;

const BASE_URL: &str = "http://192.168.1.202:8080/v1";
const MODEL: &str = "qwen3.6-35b";

#[tokio::test]
async fn test_list_models_returns_ok() {
    // Arrange
    let client = OpenAiCompatibleClient::builder()
        .base_url(BASE_URL)
        .build()
        .unwrap();

    // Act
    let body = client.list_models().await.unwrap();

    // Assert
    assert!(!body.is_empty());
}

#[tokio::test]
async fn model_returns_completion() {
    // Arrange
    let client = OpenAiCompatibleClient::builder()
        .base_url(BASE_URL)
        .build()
        .unwrap();
    let gateway = Gateway::new(client);
    let messages = [Message::User {
        content: "Reply with exactly FEROX_E2E_OK and nothing else.".into(),
    }];

    // Act
    let response = gateway
        .complete(CompletionRequest {
            model: MODEL.into(),
            messages: &messages,
            tools: None,
        })
        .await
        .unwrap();

    // Assert
    assert!(response.text.as_deref().is_some_and(|text| text.contains("FEROX_E2E_OK")));
}
