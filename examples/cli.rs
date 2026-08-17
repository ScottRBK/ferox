use std::error::Error;
use std::io::{self, Write};

use futures_util::pin_mut;
use futures_util::StreamExt;

use ferox::adapters::providers::openai_compatible::{OpenAiCompatibleClient};
use ferox::ports::llm::LlmProvider;
use ferox::gateway::Gateway;
use ferox::models::{Model, CompletionRequest, CompletionResponse, CompletionChunk, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = OpenAiCompatibleClient::builder()
        .base_url("http://192.168.1.202:8080/v1")
        .api_key("")
        .build()?;

    let gateway = Gateway::new(client);
    repl(gateway, true).await
}

async fn repl<P>(gateway: Gateway<P>, stream: bool) -> Result<(), Box<dyn Error + Send + Sync>> 
where 
    P: LlmProvider 
{

    println!["Welcome to an example of the ferox libary (press q to quit)"];

    let models = gateway.list_models().await?;
    let selected_model = select_models(&models).await?;  
     
    println!["selected model: {}", selected_model.id];

    chat_session(selected_model, gateway).await?;

    Ok(())
}

async fn chat_session<P>(model: &Model, gateway: Gateway<P>) -> Result<(), Box<dyn Error + Send + Sync>> 
where 
    P: LlmProvider 
{
    let mut messages = Vec::<Message>::new();

    loop {
        let mut user_input = String::new();
        print!("> ");
        io::stdout().flush()?;
        io::stdin()
            .read_line(&mut user_input)
            .expect("Error reading input");


        match user_input.trim() {
            "q" => break,
            _ => {
                messages.push(Message::User{content: user_input.trim().into()});
            }
        }
        let stream = gateway.stream(CompletionRequest{
            model: model.id.clone(),
            messages: &messages,
            stream: true, 
        }).await?;

        pin_mut!(stream);

        while let Some(completion) = stream.next().await {
            match completion {
                Ok(completion) => {
                    if let Some(response) = &completion.text {
                        print!("{}", response);
                        std::io::Write::flush(&mut  std::io::stdout())?; 
                    }
                }
                Err(e) => println!("Error fetching response from provier {}", e)
            }
        }
        println!();
    }
    Ok(())
}

fn print_models(models: &[Model]) {
    for (i, model) in models.iter().enumerate() {
            println!["{}. {}", i+1, model.id];
    }
}
 
async fn select_models(models: &[Model]) -> Result<&Model, Box<dyn Error + Send + Sync>> {
    println!("Please enter a number for a model to talk to");
    
    print_models(models);

    loop {    
        let mut user_input = String::new();
        io::stdout().flush()?;
        io::stdin().read_line(&mut user_input).expect("error reading input");
        match user_input.trim().parse::<usize>() {
            Ok(n) if (1..=models.len()).contains(&n) => {
                return Ok(&models[n - 1]);
            }
            _ => println!("Invalid selection, try again"),
        }
    }
}
