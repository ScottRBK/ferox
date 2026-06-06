use reqwest::Client; 
use serde:: {Deserialize, Serialize};
use serde_json;
use std::error::Error;
use std::time::Duration;
use async_stream::try_stream;
use futures_core::stream::Stream; 


#[derive(Deserialize, Serialize, Debug)]
pub struct Model{
    pub id: String 
}

#[derive(Deserialize, Debug)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")] 
pub enum ChatCompletionsMessage {
    System { content: String },
    User { content: String },
    Assistant { 
        content: String, 
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    }
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
            ChatCompletionsMessage::Assistant { reasoning_content, .. } => reasoning_content.as_deref(),
            _ => None,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionsMessage>,
    pub stream: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ChoicesFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall, // Deprecated 
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionChoices {
    pub finish_reason : ChoicesFinishReason,
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
    pub choices: Vec<ChatCompletionsStreamChoice>
}

pub struct OpenAiCompatibleClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Duration,
}

impl OpenAiCompatibleClientBuilder {
    pub fn new() ->  Self {
        Self {
            base_url: None,
            api_key: None,
            timeout: Duration::from_secs(240),
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

    pub fn build(self) -> Result<OpenAiCompatibleClient, Box<dyn Error>> {
        let base_url = self.base_url.ok_or("base_url is required")?;
        let http = Client::builder().timeout(self.timeout).build()?;
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

    pub async fn list_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        let body = self.fetch_models_body().await?;
        Self::parse_models(&body)
    }

    async fn fetch_models_body(&self) -> Result<String, reqwest::Error> {
        let mut req = self.http.get(format!("{}/models", self.base_url)); 
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let body = req.send().await?.text().await?;
        Ok(body)
    }
    fn parse_models(body: &str) -> Result<Vec<Model>, Box<dyn Error>> {
        let response: ModelsResponse = serde_json::from_str(body)?;
        Ok(response.data)
    }
    pub async fn create_chat_completion(&self, chat_request: &ChatCompletionsRequest)
        -> Result<ChatCompletionsResponse, Box<dyn Error>> {
        let request_body = serde_json::to_string(chat_request)?;
        let body = self.http.post(format!("{}/chat/completions", self.base_url))
                .body(request_body)
                .send()
                .await?.text().await?;
        let resp = Self::parse_chat_completions_response(&body)?; 
        Ok(resp)
    }

    fn parse_chat_completions_response(body: &str) -> Result<ChatCompletionsResponse,Box<dyn Error>> {
        let resp: ChatCompletionsResponse = serde_json::from_str(body)?;
        Ok(resp)  
    }

    pub async fn create_chat_completion_stream(
        &self, 
        chat_request: &ChatCompletionsRequest,
    ) -> impl Stream<Item = Result<ChatCompletionsStreamResponse, Box<dyn Error>>> {
        try_stream! {
            let request_body = serde_json::to_string(chat_request)?;
            let mut response = self.http.post(format!("{}/chat/completions", self.base_url))
                    .body(request_body)
                    .send()
                    .await?;

            while let Some(chunk) = response.chunk().await? {
                let chunk_str = std::str::from_utf8(&chunk)?;
                for line in chunk_str.lines() {
                    let line = line.trim(); 
                    if line.is_empty() { continue; }
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    if data == "[DONE]" {
                        continue;
                    }

                    let stream_response = Self::parse_chat_completions_stream_response(data)?;
                    yield stream_response;
                }
            }
        }
    }

    fn parse_chat_completions_stream_response(chunk: &str) -> 
        Result<ChatCompletionsStreamResponse, Box<dyn Error>> {
        let resp: ChatCompletionsStreamResponse = serde_json::from_str(chunk)?;
        Ok(resp)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS_FIXTURE: &str = include_str!("fixtures/models_response.json");
    const RESPONSE_FIXUTRE: &str = include_str!("fixtures/chat_completions_response.json");
    const RESPONSE_FIXUTRE_STREAM: &str = include_str!("fixtures/chat_completions_response_stream.json");

    //TODO: revisit these tests and split into proper integration versus e2e tests. 

   #[tokio::test]
    async fn test_list_models_returns_ok() {
        let client = OpenAiCompatibleClient::builder()
            .base_url("http://192.168.1.202:8080/v1")
            .build().unwrap();

        let body = client.list_models().await.unwrap();
        assert!(!body.is_empty());
    }

    #[test]
    fn test_deserialise_models() {
        let models = OpenAiCompatibleClient::parse_models(MODELS_FIXTURE).unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_deserialise_chat_response() {
        let resp = OpenAiCompatibleClient::parse_chat_completions_response(RESPONSE_FIXUTRE).unwrap();
        assert!(!resp.model.is_empty());
        assert!(!resp.choices.is_empty());
    }

    #[test]
    fn test_deserialise_chat_response_stream() {
        let mut full_message = String::new();
        for line in RESPONSE_FIXUTRE_STREAM.lines() {
            let line = line.trim(); 

            if line.is_empty() { continue; }

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };


            if data == "[DONE]" {
                continue;
            }

            let stream_response = OpenAiCompatibleClient::parse_chat_completions_stream_response(data).unwrap();
            if let Some(content) = &stream_response.choices[0].delta.content {
                full_message.push_str(content); 
            }
        }
        assert!(!full_message.is_empty())
    }
}

