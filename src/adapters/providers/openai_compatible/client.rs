use crate::{
    error::{GatewayError, LlmError},
    models::{
        CompletionChunk, CompletionRequest, CompletionResponse, Model
    },
    ports::llm::LlmProvider,
    adapters::providers::openai_compatible::models::*,
};
use super::mapping::{
    to_domain_model, to_domain_toolcall, to_provider_message,
    to_provider_reasoning_effort, to_provider_tools,
};
use super::errors::{ ClientBuildError, OpenAiClientError };
use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json;
use std::collections::{BTreeMap};
use std::pin::Pin;
use std::time::Duration;

pub struct OpenAiCompatibleClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Duration,
}

impl Default for OpenAiCompatibleClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl OpenAiCompatibleClientBuilder {
    pub fn new() -> Self {
        Self {
            base_url: None,
            api_key: None,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<OpenAiCompatibleClient, ClientBuildError> {
        let base_url = self.base_url.ok_or(ClientBuildError::MissingBaseUrl)?;
        let http = Client::builder()
            .read_timeout(self.timeout)
            .build()
            .map_err(ClientBuildError::HttpClient)?;

        Ok(OpenAiCompatibleClient {
            http,
            base_url,
            api_key: self.api_key,
        })
    }
}

pub struct OpenAiCompatibleClient {
    http: Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleClient {
    pub fn builder() -> OpenAiCompatibleClientBuilder {
        OpenAiCompatibleClientBuilder::new()
    }

    async fn fetch_models(&self) -> Result<Vec<ProviderModel>, OpenAiClientError> {
        let body = self.fetch_models_body().await?;
        Self::parse_models(&body)
    }

    async fn fetch_models_body(&self) -> Result<String, OpenAiClientError> {
        let mut req = self.http.get(format!("{}/models", self.base_url));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let response = req.send().await.map_err(OpenAiClientError::Transport)?;

        let code = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(OpenAiClientError::Transport)?;

        if !(200..300).contains(&code) {
            return Err(OpenAiClientError::Status { code, body });
        }
        Ok(body)
    }

    fn parse_models(body: &str) -> Result<Vec<ProviderModel>, OpenAiClientError> {
        let response: ModelsResponse =
            serde_json::from_str(body).map_err(OpenAiClientError::Decode)?;
        Ok(response.data)
    }

    async fn create_chat_completion(
        &self,
        chat_request: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, OpenAiClientError> {
        let request_body =
            serde_json::to_string(chat_request).map_err(OpenAiClientError::Decode)?;
        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_body);

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req.send().await.map_err(OpenAiClientError::Transport)?;

        let code = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(OpenAiClientError::Transport)?;

        if !(200..300).contains(&code) {
            return Err(OpenAiClientError::Status { code, body });
        }

        let resp = Self::parse_chat_completions_response(&body)?;
        Ok(resp)
    }

    fn parse_chat_completions_response(
        body: &str,
    ) -> Result<ChatCompletionsResponse, OpenAiClientError> {
        let resp: ChatCompletionsResponse =
            serde_json::from_str(body).map_err(OpenAiClientError::Decode)?;
        Ok(resp)
    }

    async fn generate_chat_response(
        &self,
        chat_request: &ChatCompletionsRequest,
    ) -> Result<reqwest::Response, OpenAiClientError> {
        let request_body =
            serde_json::to_string(chat_request).map_err(OpenAiClientError::Decode)?;

        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_body);

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req.send().await.map_err(OpenAiClientError::Transport)?;

        Ok(response)
    }

    fn handle_sse_line(
        line: &str,
    ) -> Option<Result<ChatCompletionsStreamResponse, OpenAiClientError>> {
        let line = line.trim();

        if line.is_empty() {
            return None;
        }

        let data = line.strip_prefix("data:")?.trim_start();

        if data == "[DONE]" {
            return None;
        };

        Some(Self::parse_chat_completions_stream_response(data))
    }

    fn stream_chat_response(
        mut response: reqwest::Response,
    ) -> impl Stream<Item = Result<ChatCompletionsStreamResponse, OpenAiClientError>> {
        try_stream! {
                let mut buffer: Vec<u8> = Vec::new();

                while let Some(chunk) = response.chunk().await.map_err(OpenAiClientError::Transport)? {
                    buffer.extend_from_slice(&chunk);

                    while let Some(nl) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = buffer.drain(..=nl).collect();
                        let line = std::str::from_utf8(&line_bytes)
                            .map_err(OpenAiClientError::Utf8)?;
                        if let Some(result) = Self::handle_sse_line(line) {
                            yield result?;
                        }
                    }
                }

                if !buffer.is_empty() {
                    let line = std::str::from_utf8(&buffer)
                        .map_err(OpenAiClientError::Utf8)?;
                    if let Some(result) = Self::handle_sse_line(line)  {
                        yield result?;
                    }
                }
        }
    }

    fn parse_chat_completions_stream_response(
        chunk: &str,
    ) -> Result<ChatCompletionsStreamResponse, OpenAiClientError> {
        let resp: ChatCompletionsStreamResponse =
            serde_json::from_str(chunk).map_err(OpenAiClientError::Decode)?;
        Ok(resp)
    }
}





impl LlmProvider for OpenAiCompatibleClient {
    type CompletionStream = Pin<Box<dyn Stream<Item = Result<CompletionChunk, GatewayError>> + Send>>;

    async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<CompletionResponse, GatewayError> {
        let provider_request = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages.iter().map(to_provider_message).collect(),
            stream: false,
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(to_provider_tools).collect()),
            reasoning_effort: request.reasoning_effort.map(to_provider_reasoning_effort),
        };

        let response = self
            .create_chat_completion(&provider_request)
            .await
            .map_err(LlmError::from)?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::InvalidResponse {
                message: "Provider returned no choices".to_string(),
            })?;

        Ok(CompletionResponse {
            model: response.model,
            text: choice.message.content().map(str::to_string),
            reasoning: choice.message.reasoning_content().map(str::to_string),
            tool_calls: choice
                .message
                .tool_calls()
                .iter()
                .map(to_domain_toolcall)
                .collect(),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<Self::CompletionStream, GatewayError> {
        let provider_request = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages.iter().map(to_provider_message).collect(),
            stream: true,
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(to_provider_tools).collect()),
            reasoning_effort: request.reasoning_effort.map(to_provider_reasoning_effort),
        };

        let response = self
            .generate_chat_response(&provider_request)
            .await
            .map_err(LlmError::from)?;

        let code = response.status().as_u16();
        if !(200..300).contains(&code) {
            let body = response
                .text()
                .await
                .map_err(OpenAiClientError::Transport)
                .map_err(LlmError::from)?;

            return Err(GatewayError::Llm(OpenAiClientError::Status { code, body }.into()));
        }

        let mut pending_tool_calls = BTreeMap::<usize, PendingToolCall>::new();

        let stream = Self::stream_chat_response(response).map(move |item| {
            let chunk = item.map_err(LlmError::from)?;
            let choice = chunk.choices.first();

            if let Some(choice) = choice {
                for delta in &choice.delta.tool_calls {
                    pending_tool_calls
                        .entry(delta.index)
                        .or_default()
                        .apply(delta);
                }
            }

            let finished = choice
                .and_then(|choice| choice.finish_reason.as_ref())
                .is_some();

            let mut tool_calls = Vec::new();

            if finished {
                let completed_calls = std::mem::take(&mut pending_tool_calls);

                for (_, pending_call) in completed_calls {
                    let tool_call = pending_call.finish()?;
                    tool_calls.push(tool_call);
                }
            }

            Ok(CompletionChunk {
                text: choice.and_then(|choice| choice.delta.content.clone()),
                reasoning: choice.and_then(|choice| choice.delta.reasoning_content.clone()),
                tool_calls,
                finished,
            })
        });

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<Model>, GatewayError> {
        let provider_models = self.fetch_models().await.map_err(LlmError::from)?;
        provider_models
            .into_iter()
            .map(|m| to_domain_model(m).map_err(GatewayError::from))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::providers::openai_compatible::mapping::to_provider_property_type;
    use crate::models::{
        ModelModality, Tool, ToolParameterProperty, ToolParameterPropertyType,
    };
    use futures_util::StreamExt;
    use futures_util::pin_mut;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const MODELS_FIXTURE: &str = include_str!("../../fixtures/models_response.json");
    const MODELS_WITH_MODALITIES_FIXTURE: &str =
        include_str!("../../fixtures/models_response_with_modalities.json");
    const RESPONSE_FIXUTRE: &str = include_str!("../../fixtures/chat_completions_response.json");
    const RESPONSE_FIXUTRE_STREAM: &str =
        include_str!("../../fixtures/chat_completions_response_stream.json");
    const TOOL_CALLS_FIXTURE: &str =
        include_str!("../../fixtures/chat_completions_response_tool_calls.json");

    #[test]
    fn test_deserialise_moels() {
        let models = OpenAiCompatibleClient::parse_models(MODELS_FIXTURE).unwrap();
        assert!(!models.is_empty());
    }

    #[tokio::test]
    async fn list_models_returns_text_input_and_output_modalities() {
        // Arrange
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(MODELS_WITH_MODALITIES_FIXTURE, "application/json"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OpenAiCompatibleClient::builder()
            .base_url(server.uri())
            .build()
            .unwrap();

        // Act
        let models = client.list_models().await.unwrap();

        // Assert
        let model = models.first().expect("expected at least one model");
        assert!(matches!(
            model.input_modalities.as_slice(),
            [ModelModality::Text]
        ));
        assert!(matches!(
            model.output_modalities.as_slice(),
            [ModelModality::Text]
        ));
    }

    #[test]
    fn test_deserialise_chat_response() {
        let resp =
            OpenAiCompatibleClient::parse_chat_completions_response(RESPONSE_FIXUTRE).unwrap();
        assert!(!resp.model.is_empty());
        assert!(!resp.choices.is_empty());
    }

    #[test]
    fn deserialises_tool_calls_from_response() {
        // Arrange
        let body = TOOL_CALLS_FIXTURE;

        // Act
        let resp = OpenAiCompatibleClient::parse_chat_completions_response(body).unwrap();
        let choice = &resp.choices[0];

        // Assert

        let tool_calls = choice.message.tool_calls();

        let tool_call = tool_calls.first().expect("expected one tool call");
        assert_eq!(tool_call.id, "call_abc123");
        assert_eq!(tool_call.function.name, "add_two_numbers");
        assert_eq!(
            tool_call.function.arguments,
            "{\"first_number\": 2, \"second_number\": 3}"
        );
    }

    #[tokio::test]
    async fn test_stream_chat_response() {
        // Arrange
        let response = reqwest::Response::from(http::Response::new(RESPONSE_FIXUTRE_STREAM));

        // Act
        let stream = OpenAiCompatibleClient::stream_chat_response(response);
        pin_mut!(stream);
        let mut content = String::new();

        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(c) = &chunk.choices[0].delta.content {
                content.push_str(c);
            }
        }

        //Assert
        assert_eq!(content, "Hey! How can I help you today? 😊");
    }

    #[tokio::test]
    async fn test_stream_chat_response_accepts_data_without_space() {
        // Arrange
        let body = RESPONSE_FIXUTRE_STREAM.replace("data: ", "data:");
        let response = reqwest::Response::from(http::Response::new(body));

        // Act
        let stream = OpenAiCompatibleClient::stream_chat_response(response);
        pin_mut!(stream);
        let mut content = String::new();

        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(chunk_content) = &chunk.choices[0].delta.content {
                content.push_str(chunk_content);
            }
        }

        // Assert
        assert_eq!(content, "Hey! How can I help you today? 😊");
    }

    #[tokio::test]
    async fn test_utf8_byte_split_stream_chat_response() {
        let split = RESPONSE_FIXUTRE_STREAM.find('😊').unwrap() + 2;

        let (request_one, request_two) = RESPONSE_FIXUTRE_STREAM.as_bytes().split_at(split);

        let chunks: Vec<Result<_, ::std::io::Error>> = vec![Ok(request_one), Ok(request_two)];

        let stream = futures_util::stream::iter(chunks);

        let body = reqwest::Body::wrap_stream(stream);

        // Arrange
        let response = reqwest::Response::from(http::Response::new(body));

        // Act
        let stream = OpenAiCompatibleClient::stream_chat_response(response);
        pin_mut!(stream);
        let mut content = String::new();

        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();
            if let Some(c) = &chunk.choices[0].delta.content {
                content.push_str(c);
            }
        }

        //Assert
        assert_eq!(content, "Hey! How can I help you today? 😊");
    }

    #[test]
    fn tool_serializes_to_openai_function_schema() {
        // Arrange
        let tool = Tool::new("get_weather", "Get the current weather")
            .required_parameter(ToolParameterProperty {
                name: "location".into(),
                property_type: ToolParameterPropertyType::String,
                description: "City and state".into(),
                property_enum: None,
            })
            .optional_parameter(ToolParameterProperty {
                name: "unit".into(),
                property_type: ToolParameterPropertyType::String,
                description: "Temp unit".into(),
                property_enum: Some(vec!["celsius".into(), "fahrenheit".into()]),
            });

        // Act
        let json = serde_json::to_value(to_provider_tools(tool)).unwrap();

        // Assert
        let expected = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temp unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        });
        assert_eq!(json, expected);
    }

    #[test]
    fn property_types_serialize_to_openai_json_schema_values() {
        // Arrange
        let cases = [
            (ToolParameterPropertyType::String, "string"),
            (ToolParameterPropertyType::Number, "number"),
            (ToolParameterPropertyType::Integer, "integer"),
            (ToolParameterPropertyType::Boolean, "boolean"),
        ];

        for (property_type, expected) in cases {
            // Act
            let provider_type = to_provider_property_type(property_type);
            let json = serde_json::to_value(provider_type).unwrap();

            // Assert
            assert_eq!(json, serde_json::json!(expected));
        }
    }

    #[test]
    fn parameterless_tool_omits_parameters_from_openai_function_schema() {
        // Arrange
        let tool = Tool::new("get_system_status", "Get the current system status");

        // Act
        let json = serde_json::to_value(to_provider_tools(tool)).unwrap();

        // Assert — OpenAI defines omitted parameters as an empty parameter list
        let expected = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_system_status",
                "description": "Get the current system status"
            }
        });
        assert_eq!(json, expected);
    }
}
