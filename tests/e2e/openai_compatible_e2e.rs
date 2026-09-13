use ferox::Gateway;
use ferox::models::{
    CompletionRequest, Message, ReasoningEffort, Tool, ToolParameterProperty,
    ToolParameterPropertyType,
};
use ferox::openai_compatible::OpenAiCompatibleClient;

use futures_util::{StreamExt, pin_mut};
use futures::lock::Mutex;
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

// Mutex added here to ensure only 1 test involving a model at a time
// poor potato for local inference
static E2E_REQUEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

// creates a mutex or if already exits retrieves it
fn e2e_request_lock() -> &'static Mutex<()> {
    E2E_REQUEST_LOCK.get_or_init(|| Mutex::new(()))
}

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

fn build_gateway(config: &E2eConfig) -> Gateway<OpenAiCompatibleClient> {
    Gateway::new(build_client(config))
}

fn calculator_tool() -> Tool {
    Tool::new("add_two_numbers", "Add two numbers together")
        .required_parameter(ToolParameterProperty::new(
            "first_number",
            ToolParameterPropertyType::Integer,
            "The first number",
        ))
        .required_parameter(ToolParameterProperty::new(
            "second_number",
            ToolParameterPropertyType::Integer,
            "The second number",
        ))
}

#[tokio::test]
async fn test_list_models_returns_ok() {
    // Arrange
    let gateway = build_gateway(e2e_config());

    // Act
    let models = gateway.list_models().await.unwrap();

    // Assert
    assert!(
        models
            .iter()
            .any(|model| model.id == e2e_config().model.as_str())
    );
}

#[tokio::test]
async fn model_returns_completion() {
    let _e2e_lock = e2e_request_lock().lock().await;

    // Arrange
    let gateway = build_gateway(e2e_config());
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

#[tokio::test]
async fn gateway_streams_completion_to_completion() {
    let _e2e_lock = e2e_request_lock().lock().await;

    // Arrange
    let gateway = build_gateway(e2e_config());
    let messages = [Message::User {
        content: "Reply with exactly FEROX_STREAM_OK and nothing else.".into(),
    }];

    // Act
    let request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    let stream = gateway.stream(request).await.unwrap();
    pin_mut!(stream);
    let mut text = String::new();
    let mut saw_finish = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(chunk_text) = chunk.text {
            text.push_str(&chunk_text);
        }
        saw_finish |= chunk.finished_reason.is_some();
    }

    // Assert
    assert!(text.contains("FEROX_STREAM_OK"));
    assert!(saw_finish, "stream did not include a finish reason");
}

#[tokio::test]
async fn reasoning_model_returns_reasoning_and_completion() {
    let _e2e_lock = e2e_request_lock().lock().await;

    // Arrange
    let gateway = build_gateway(e2e_config());
    let messages = [Message::User {
        content: "Work out 17 + 25. Return the answer after reasoning about it.".into(),
    }];
    let mut request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    request.reasoning_effort = Some(ReasoningEffort::Medium);

    // Act
    let response = gateway.complete(request).await.unwrap();

    // Assert
    assert!(
        response
            .reasoning
            .as_deref()
            .is_some_and(|reasoning| !reasoning.trim().is_empty()),
        "configured model did not return reasoning"
    );
    assert!(
        response
            .text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty()),
        "configured model did not return completion text"
    );
}

#[tokio::test]
async fn model_returns_a_tool_call_through_gateway() {
    let _e2e_lock = e2e_request_lock().lock().await;

    // Arrange
    let gateway = build_gateway(e2e_config());
    let messages = [Message::User {
        content: "You must call add_two_numbers with 12 and 6. Do not answer in text.".into(),
    }];
    let mut request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    request.tools = Some(vec![calculator_tool()]);

    // Act
    let response = gateway.complete(request).await.unwrap();

    // Assert
    let tool_call = response
        .tool_calls
        .first()
        .expect("configured model did not return a tool call");
    assert_eq!(tool_call.name, "add_two_numbers");
    let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
        .expect("tool call arguments were not valid JSON");
    assert_eq!(arguments["first_number"], 12);
    assert_eq!(arguments["second_number"], 6);
}

#[tokio::test]
async fn gateway_supports_a_multi_turn_tool_conversation() {
    let _e2e_lock = e2e_request_lock().lock().await;

    // Arrange
    let gateway = build_gateway(e2e_config());
    let mut messages = vec![Message::User {
        content: "You must call add_two_numbers with 12 and 6. Do not answer in text.".into(),
    }];
    let mut first_request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    first_request.tools = Some(vec![calculator_tool()]);

    // Act
    let first_response = gateway.complete(first_request).await.unwrap();
    let tool_call = first_response
        .tool_calls
        .first()
        .expect("configured model did not return a tool call");
    let tool_call_id = tool_call.id.clone();
    messages.push(Message::Assistant {
        content: first_response.text,
        tool_calls: first_response.tool_calls,
        reasoning: first_response.reasoning,
    });
    messages.push(Message::Tool {
        tool_call_id,
        content: "18".into(),
    });
    messages.push(Message::User {
        content: "Use the tool result and reply with exactly FEROX_MULTI_TURN_OK.".into(),
    });

    let second_request = CompletionRequest::new(e2e_config().model.clone(), &messages);
    let second_response = gateway.complete(second_request).await.unwrap();

    // Assert
    assert!(
        second_response
            .text
            .as_deref()
            .is_some_and(|text| text.contains("FEROX_MULTI_TURN_OK"))
    );
}
