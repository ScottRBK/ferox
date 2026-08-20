use crate::{
    error::LlmError,
    models::{
        CompletionChunk, 
        CompletionRequest, 
        CompletionResponse, 
        Message, 
        Model, 
        Tool, 
        ToolParameters,
        ToolParameterProperty,
    },
    ports::llm::LlmProvider
};
use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use std::error::Error;
use std::pin::Pin;
use std::time::Duration;
use std::fmt;
use std::collections::HashMap;

#[derive(Debug)]
pub enum OpenAiClientError {
    Transport(reqwest::Error),
    Status { code: u16, body: String },
    Decode(serde_json::Error),
    Utf8(std::str::Utf8Error),
}


#[derive(Debug)]
pub enum ClientBuildError {
    MissingBaseUrl,
    HttpClient(reqwest::Error),
}

impl fmt::Display for ClientBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientBuildError::MissingBaseUrl => write!(f, "missing base_url parameter"),
            ClientBuildError::HttpClient(e) => write!(f, "failed to build HTTP client {e}"),
        }
    }
}

impl Error for ClientBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ClientBuildError::HttpClient(e) => Some(e),
            _ => None,
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ProviderModel {
    pub id: String,
}

#[derive(Deserialize, Debug)]
struct ModelsResponse {
    data: Vec<ProviderModel>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionsMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
}

impl ChatCompletionsMessage {

    pub fn content(&self) -> &str {
        match self {
            ChatCompletionsMessage::System { content } => content,
            ChatCompletionsMessage::User { content } => content,
            ChatCompletionsMessage::Assistant { content, .. } => content,
        }
    }

    pub fn reasoning_content(&self) -> Option<&str> {
        match self {
            ChatCompletionsMessage::Assistant {
                reasoning_content, ..
            } => reasoning_content.as_deref(),
            _ => None,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionToolParameterProperty {
    #[serde(rename= "type")]
    pub property_type: String,
    pub description: String,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub property_enum: Option<Vec<String>>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionToolParameters {
    #[serde(rename= "type")]
    pub parameter_type: String,
    pub properties: HashMap<String, ChatCompletionToolParameterProperty>, 
    pub required: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionTool {
    #[serde(rename= "type")]
    pub tool_type: String, 
    pub function: ChatCompletionFunction,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionFunction {
    pub name: String,
    pub description: String,
    pub parameters: ChatCompletionToolParameters,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionsMessage>,
    pub stream: bool,
    pub tools: Option<Vec<ChatCompletionTool>>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ChoicesFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionChoices {
    pub finish_reason: ChoicesFinishReason,
    pub index: i32,
    pub message: ChatCompletionsMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionChoices>,
    pub created: i64,
    pub model: String,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionsStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(alias = "reasoning")]
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionsStreamChoice {
    pub index: i32,
    pub delta: ChatCompletionsStreamDelta,
    pub finish_reason: Option<ChoicesFinishReason>,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionsStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionsStreamChoice>,
}

pub struct OpenAiCompatibleClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Duration,
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

    pub async fn fetch_models(&self) -> Result<Vec<ProviderModel>, OpenAiClientError> {
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
        let body = response.text().await.map_err(OpenAiClientError::Transport)?;

        if !(200..300).contains(&code) {
            return Err(OpenAiClientError::Status { code, body });
        }
        Ok(body)
    }
    fn parse_models(body: &str) -> Result<Vec<ProviderModel>, OpenAiClientError> {
        let response: ModelsResponse = serde_json::from_str(body).map_err(OpenAiClientError::Decode)?;
        Ok(response.data)
    }
    pub async fn create_chat_completion(
        &self,
        chat_request: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, OpenAiClientError> {
        let request_body = serde_json::to_string(chat_request).map_err(OpenAiClientError::Decode)?;
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
        let body = response.text().await.map_err(OpenAiClientError::Transport)?;

        if !(200..300).contains(&code) {
            return Err(OpenAiClientError::Status { code, body });
        }

        let resp = Self::parse_chat_completions_response(&body)?;
        Ok(resp)
    }

    fn parse_chat_completions_response(
        body: &str,
    ) -> Result<ChatCompletionsResponse, OpenAiClientError> {
        let resp: ChatCompletionsResponse = serde_json::from_str(body)
            .map_err(OpenAiClientError::Decode)?;
        Ok(resp)
    }

    pub async fn generate_chat_response(
        &self,
        chat_request: &ChatCompletionsRequest,
    ) -> Result<reqwest::Response, OpenAiClientError> {
        let request_body = serde_json::to_string(chat_request)
            .map_err(OpenAiClientError::Decode)?;

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

    fn handle_sse_line(line: &str) -> Option<Result<ChatCompletionsStreamResponse, OpenAiClientError>> {
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

    pub fn stream_chat_response(
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
        let resp: ChatCompletionsStreamResponse = serde_json::from_str(chunk)
            .map_err(OpenAiClientError::Decode)?;
        Ok(resp)
    }
}

fn to_provider_message(message: &Message) -> ChatCompletionsMessage {
    match message {
        Message::System{ content } => ChatCompletionsMessage::System {
            content: content.clone(),
        },
        Message::User{ content } => ChatCompletionsMessage::User {
            content: content.clone(),
        },
        Message::Assistant{ content } => ChatCompletionsMessage::Assistant {
            content: content.clone(),
            reasoning_content: None,
        },
    }
}

fn to_provider_tools(tool: Tool) -> ChatCompletionTool{
    ChatCompletionTool {
        tool_type: String::from("function"),
        function: ChatCompletionFunction {
            name: tool.name,
            description: tool.description,
            parameters: to_provider_parameters(tool.parameters),
        }
    }
}

fn to_provider_parameters(parameter: ToolParameters) -> ChatCompletionToolParameters {
    ChatCompletionToolParameters {
        parameter_type: String::from("object"),
        properties: parameter.properties
            .into_iter()
            .map(to_provider_parameter_property)
            .collect(),
        required: parameter.required,
    }
}

fn to_provider_parameter_property(property: ToolParameterProperty) -> 
    (String, ChatCompletionToolParameterProperty) {
    (
        property.name, 
        ChatCompletionToolParameterProperty { 
            property_type: property.property_type.to_string(), 
            description: property.description.to_string(), 
            property_enum: property.property_enum, 
        }
    )
}


fn to_llm_error(error: OpenAiClientError) -> LlmError {
    match error {
        OpenAiClientError::Transport(err) if err.is_timeout() => LlmError::Timeout,
        OpenAiClientError::Transport(err) => LlmError::Transport {
            message: err.to_string(),
        },

        OpenAiClientError::Status { code: 401, .. } => LlmError::AuthenticationFailed,
        OpenAiClientError::Status { code: 403, .. } => LlmError::PermissionDenied,
        OpenAiClientError::Status { code: 429, .. } => LlmError::RateLimited {
            retry_after: None,
        },
        OpenAiClientError::Status {
            code: 400 | 422,
            body,
        } => LlmError::InvalidRequest { message: body },
        OpenAiClientError::Status { code: 408, .. } => LlmError::Timeout,
        OpenAiClientError::Status {
            code: 500..=599, ..
        } => LlmError::ProviderUnavailable,
        OpenAiClientError::Status { body, .. } => LlmError::ProviderFailure { message: body },

        OpenAiClientError::Decode(err) => LlmError::InvalidResponse {
            message: err.to_string(),
        },
        OpenAiClientError::Utf8(err) => LlmError::InvalidResponse {
            message: err.to_string(),
        },
    }
}

impl LlmProvider for OpenAiCompatibleClient {
    type CompletionStream = Pin<Box<dyn Stream<Item = Result<CompletionChunk, LlmError>> + Send>>;

    async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<CompletionResponse, LlmError> {
        let provider_request = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages.iter().map(to_provider_message).collect(),
            stream: false,
            tools: request.tools.map(|tools| {
                tools.into_iter().map(to_provider_tools).collect()
            }),
        };

        let response = self
            .create_chat_completion(&provider_request)
            .await
            .map_err(to_llm_error)?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::InvalidResponse {
                message: "Provider returned no choices".to_string(),
            })?;

        Ok(CompletionResponse {
            model: response.model,
            text: choice.message.content().to_string(),
            reasoning: choice.message.reasoning_content().map(str::to_string),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<Self::CompletionStream, LlmError> {
        let provider_request = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages.iter().map(to_provider_message).collect(),
            stream: true,
            tools: request.tools.map(|tools| {
                tools.into_iter().map(to_provider_tools).collect()
            }),
        };

        let response = self
            .generate_chat_response(&provider_request)
            .await
            .map_err(to_llm_error)?;

        let code = response.status().as_u16();
        if !(200..300).contains(&code) {
            let body = response
                .text()
                .await
                .map_err(OpenAiClientError::Transport)
                .map_err(to_llm_error)?;

            return Err(to_llm_error(OpenAiClientError::Status{ code, body }));
        }

        let stream = Self::stream_chat_response(response).map(|item| {
            let chunk = item.map_err(to_llm_error)?;
            let choice = chunk.choices.first();
            Ok(CompletionChunk {
                text: choice.and_then(|choice| choice.delta.content.clone()),
                reasoning: choice.and_then(|choice| choice.delta.reasoning_content.clone()),
                finished: choice
                    .and_then(|choice| choice.finish_reason.as_ref())
                    .is_some(),
            })
        });

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<Model>, LlmError> {
        let provider_models = self.fetch_models().await.map_err(to_llm_error)?;
        let models = provider_models
            .into_iter()
            .map(|provider_model| Model {
                id: provider_model.id,
            })
            .collect();
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use futures_util::pin_mut;

    const MODELS_FIXTURE: &str = include_str!("../fixtures/models_response.json");
    const RESPONSE_FIXUTRE: &str = include_str!("../fixtures/chat_completions_response.json");
    const RESPONSE_FIXUTRE_STREAM: &str =
        include_str!("../fixtures/chat_completions_response_stream.json");

    #[test]
    fn test_deserialise_models() {
        let models = OpenAiCompatibleClient::parse_models(MODELS_FIXTURE).unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_deserialise_chat_response() {
        let resp =
            OpenAiCompatibleClient::parse_chat_completions_response(RESPONSE_FIXUTRE).unwrap();
        assert!(!resp.model.is_empty());
        assert!(!resp.choices.is_empty());
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
        let tool = Tool {
            name: "get_weather".into(),
            description: "Get the current weather".into(),
            parameters: ToolParameters {
                properties: vec![
                    ToolParameterProperty {
                        name: "location".into(),
                        property_type: "string".into(),
                        description: "City and state".into(),
                        property_enum: None,
                    },
                    ToolParameterProperty {
                        name: "unit".into(),
                        property_type: "string".into(),
                        description: "Temp unit".into(),
                        property_enum: Some(vec!["celsius".into(), "fahrenheit".into()]),
                    },
                ],
                required: vec!["location".into()],
            },
        };

        // Act
        let json = serde_json::to_value(&to_provider_tools(tool)).unwrap();

        // Assert — copied from the OpenAI docs, not from our structs
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
}
