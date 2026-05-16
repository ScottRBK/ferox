use reqwest::Client; 
use serde:: {Deserialize, Serialize};
use serde_json;
use std::error::Error;
use std::time::{Duration, SystemTime};


pub struct OpenAiCompatibleClient {
    http: Client,
    base_url: String,
    api_key: Option<String>,
}

pub struct OpenAiCompatibleClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Duration,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Model{
    pub id: String 
}

#[derive(Deserialize, Debug)]
struct ModelsResponse {
    data: Vec<Model>,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionsMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionsMessage>,
    pub stream: bool,
}

#[derive(Deserialize, Debug)]
pub enum ChoicesFinishReason {
    stop,
    length,
    tool_calls,
    content_filter,
    function_call,
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
    pub created: i32,
    pub model: String,
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
        let resp: ChatCompletionsResponse = serde_json::from_str(&body)?;
        Ok(resp) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS_FIXTURE: &str = include_str!("fixtures/models_response.json");

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
}
