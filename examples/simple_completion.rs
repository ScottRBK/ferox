use std::error::Error;
use ferox::{ Gateway };
use ferox::models::{
    CompletionRequest, Message, Model
};
use ferox::openai_compatible::OpenAiCompatibleClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let base_url = String::from("http://localhost:8080/v1");

    let builder = OpenAiCompatibleClient::builder().base_url(base_url);

    let client = builder.build()?;
    let gateway = Gateway::new(client);
    let model = Model::new("qwen3.6-35b", None, None);
    let messages = [Message::User { content: "What is the capital of France?".into() }];
    let request = CompletionRequest::new(model.id.clone(), &messages);
    let completion = gateway.complete(request).await?;
    let response = &completion.text.unwrap_or("No LLM Response".into());

    println!("{response}");

    Ok(())
}
