use reqwest; 
use serde::Deserialize;
use serde_json;
use std::error::Error;


pub struct OpenAiCompatibleClient {
    models: Vec<Model>,
    base_url: String,
}

impl OpenAiCompatibleClient {
        pub async fn get_available_models() -> Result<Vec<Model>, Box<dyn Error>>{
        let model_body = get_models_body().await?;
        let models = deserialise_models(&model_body)?;
        Ok(models)
    }
}

#[derive(Deserialize, Debug)]
struct Model{
    id: String 
}

#[derive(Deserialize, Debug)]
struct ModelsResponse {
    data: Vec<Model>,
}

async fn  get_models_body () -> Result<String, reqwest::Error> {
    let body = reqwest::get("http://192.168.1.202:8080/models")
        .await?
        .text()
        .await?;
    Ok(body)
}

fn deserialise_models(body: &str) -> Result<Vec<Model>, serde_json::Error> {
    let response: ModelsResponse = serde_json::from_str(body)?;
    Ok(response.data)

}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS_FIXTURE: &str = include_str!("fixtures/models_response.json");

    #[tokio::test]
    async fn test_list_models_returns_ok() {
        let body = get_models_body().await.unwrap();
        assert!(!body.is_empty());
    }

    #[test]
    fn test_deserialise_models() {
        let models = deserialise_models(MODELS_FIXTURE).unwrap();
        assert!(!models.is_empty());
    }
}
