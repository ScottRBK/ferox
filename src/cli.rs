use std::error::Error;
use std::io::{self, Write};
use async_openai::types::chat::{ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage};
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
        ChatCompletionRequestMessage,ChatCompletionRequestAssistantMessageContent,
    },
    Client,
};

pub async fn repl() -> Result<(), Box<dyn Error>> {
   
    println!("Welcome to ferox! (type q to exit)");

    let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
    let system_message: ChatCompletionRequestSystemMessage = ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()?;
    messages.push(system_message.into());

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
                    let user_message: ChatCompletionRequestUserMessage = ChatCompletionRequestUserMessageArgs::default()
                    .content(user_input)
                    .build()?;
                    messages.push(user_message.into());
                    let response = llm_request(&mut messages).await.map_err(|e| {
                        eprintln!("Error: {e}");
                        io::Error::new(io::ErrorKind::Other, e.to_string())
                    })?;
                    match response.content {
                        None =>  println!("Assistant: Empty Response"), 
                        Some(ChatCompletionRequestAssistantMessageContent::Text(ref text)) => 
                    println!("Assistant: {}", text),
                        Some(_) => println!("Assistant: (non-text content)"),
                    }
                    
                messages.push(response.into());
                 }
        }
    }
}

async fn llm_request(messages: &mut Vec<ChatCompletionRequestMessage>) -> 
Result<ChatCompletionRequestAssistantMessage, Box<dyn Error>> {

    let api_base = "http://192.168.1.201:8080/v1";
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(api_base),
    );

    let model = "qwen3.6-27b";
    

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages.to_vec())
        .build()?;

    let response = client.chat().create(request).await?;
    
    let response_msg = &response.choices[0].message; 

    let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
        .content(response_msg.content.clone().unwrap_or_default())
        .build()?;

    Ok(assistant_msg)
    

}
