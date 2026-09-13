use ferox::Gateway;
use ferox::LlmProvider;
use ferox::models::{CompletionRequest, Message};
use ferox::openai_compatible::OpenAiCompatibleClient;
use std::sync::OnceLock;

const BASE_URL_ENV: &str = "FEROX_E2E_BASE_URL";
const MODEL_ENV: &str = "FEROX_E2E_MODEL";
const API_KEY_ENV: &str = "FEROX_E2E_API_KEY";

struct E2eConfig {
    base_url: String,
    model: String,
    api_key: Option<String>,
}

static CONFIG: OnceLock<E2eConfig> = OnceLock::new();

fn e2e_config() -> &'static E2eConfig {
    CONFIG.get_or_init(|| {
        let _ = dotenvy::dotenv();

        E2eConfig {
            base_url: required_env(BASE_URL_ENV),
            model: required_env(MODEL_ENV),
            api_key: std::env::var(API_KEY_ENV).ok(),
        }
    })
}

fn required_env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("Set {name} in .env or the shell before running e2e tests"))
}

fn build_client(config: &E2eConfig) -> OpenAiCompatibleClient {
    let mut builder = OpenAiCompatibleClient::builder().base_url(&config.base_url);

    if let Some(api_key) = &config.api_key {
        builder = builder.api_key(api_key);
    }

    builder.build().unwrap()
}

#[tokio::test]
async fn test_list_models_returns_ok() {
    // Arrange
    let client = build_client(e2e_config());

    // Act
    let body = client.list_models().await.unwrap();

    // Assert
    assert!(!body.is_empty());
}

#[tokio::test]
async fn model_returns_completion() {
    // Arrange
    let client = build_client(e2e_config());
    let gateway = Gateway::new(client);
    let messages = [Message::User {
        content: "Reply with exactly FEROX_E2E_OK and nothing else.".into(),
    }];

    // Act
    let request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    let response = gateway.complete(request).await.unwrap();

    // Assert
    assert!(
        response
            .text
            .as_deref()
            .is_some_and(|text| text.contains("FEROX_E2E_OK"))
    );
}
