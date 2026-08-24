use ferox::adapters::providers::openai_compatible::OpenAiCompatibleClient;
use ferox::gateway::Gateway;
use ferox::models::{
    CompletionRequest, Message, Tool, ToolParameterProperty, ToolParameterPropertyType,
};
use serde_json::json;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn registered_tools_are_sent_to_openai_compatible_provider() {
    // Arrange
    let server = MockServer::start().await;
    let expected_request = json!({
        "model": "qwen3.6-35b",
        "messages": [
            {
                "role": "system",
                "content": "Use get_weather for weather questions."
            },
            {
                "role": "user",
                "content": "What is the weather in Leeds, United Kingdom?"
            }
        ],
        "stream": false,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "IANA timezone"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            }
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(expected_request))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "completion-1",
            "created": 1,
            "model": "qwen3.6-35b",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The tools were registered."
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiCompatibleClient::builder()
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .unwrap();
    let gateway = Gateway::new(client);
    let messages = [
        Message::System {
            content: "Use get_weather for weather questions.".into(),
        },
        Message::User {
            content: "What is the weather in Leeds, United Kingdom?".into(),
        },
    ];
    let tools = vec![
        Tool::new(
            "get_weather",
            "Get the current weather for a location",
        )
        .required_parameter(ToolParameterProperty {
            name: "location".into(),
            property_type: ToolParameterPropertyType::String,
            description: "City and country".into(),
            property_enum: None,
        })
        .optional_parameter(ToolParameterProperty {
            name: "unit".into(),
            property_type: ToolParameterPropertyType::String,
            description: "Temperature unit".into(),
            property_enum: Some(vec!["celsius".into(), "fahrenheit".into()]),
        }),
        Tool::new("get_current_time", "Get the current time for a timezone")
            .required_parameter(ToolParameterProperty {
                name: "timezone".into(),
                property_type: ToolParameterPropertyType::String,
                description: "IANA timezone".into(),
                property_enum: None,
            }),
    ];

    // Act
    let response = gateway
        .complete(CompletionRequest {
            model: "qwen3.6-35b".into(),
            messages: &messages,
            tools: Some(tools),
        })
        .await
        .unwrap();

    // Assert
    assert_eq!(response.text.as_deref(), Some("The tools were registered."));
}
