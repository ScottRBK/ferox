// use settings::Settings;
use cli::repl; 
use llm_providers::openai_compatible::OpenAiCompatibleClient;
mod settings;
mod cli; 
mod llm_providers;

#[tokio::main]
async fn main() {

    // let settings = Settings::new();

    let client = OpenAiCompatibleClient::builder()
        .base_url("http:192.168.1.201:8080/v1")
        .build()
        .expect("error building llm provider client");

    let repl = repl(client).await.expect("error starting repl");

    
}
