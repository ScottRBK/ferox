use std::error::Error;
use std::io::{self, Write};
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, 
    },
    Client,
};

pub async fn repl() -> io::Result<()> {
   
    println!("Welcome to ferox! (type q to exit)");
    
    loop {
        let mut user_input = String::new();
        print!("> "); 
        io::stdout().flush()?; 
        io::stdin()
            .read_line(&mut user_input)
            .expect("Error reading input");
        
        match user_input.trim() {
           "q" => break Ok(()),
           _ =>  {
                    let response = llm_request(user_input.trim()).await.map_err(|e| {
                        eprintln!("Error: {e}");
                        io::Error::new(io::ErrorKind::Other, e.to_string())
                    })?;
                    println!("Assistant: {response}");
                 }
        }
    }
}

async fn llm_request(user_input: &str) -> Result<String, Box<dyn Error>> {
    let api_base = "http://192.168.1.201:8080/v1";
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(api_base),
    );
    let model = "qwen3.6-27b";

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_input)
                .build()?
                .into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;
    let mut output_str = String::new();
    
    for choice in response.choices {
        output_str += &choice.message.content.unwrap_or(String::new());
    }

    Ok(output_str)
}
