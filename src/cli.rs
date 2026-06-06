use std::error::Error;
use std::io::{self, Write};

use futures_util::pin_mut;
use futures_util::stream::StreamExt;

use crate::llm_providers::openai_compatible:: { 
    Model, 
    OpenAiCompatibleClient,
    ChatCompletionsMessage,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatCompletionChoices,
    ChoicesFinishReason,
};

pub async fn repl(client: OpenAiCompatibleClient, stream: bool) -> Result<(), Box<dyn Error>> {
   
    println!("Welcome to ferox! (type q to exit)");

    let models = client.list_models().await?;
    let model  = select_model(&models)?;
    let messages = Vec::<ChatCompletionsMessage>::new();

    println!("Model Selected: {}", model);
     
    let mut chat_request = ChatCompletionsRequest {
        model,
        messages,
        stream,
    };

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
                   let message = ChatCompletionsMessage::User {
                        content: user_input.trim().into(), 
                   };
                   chat_request.messages.push(message);
                 }
        }
        
        match stream {
            false => {
                let response = client.create_chat_completion(&chat_request).await;
                match response {
                   Ok(response) => { 
                        let agent_message = response.choices[0].message.clone();
                        println!("{}", agent_message.content());
                        chat_request.messages.push(agent_message);
                   }
                   Err(e) => println!("Error fetching response from provider: {}", e)
                }
            },
            true => {
                let mut message_content = String::new();
                {
                    let response = client.create_chat_completion_stream(&chat_request).await;
                    pin_mut!(response);

                    while let Some(completion) = response.next().await {
                        match completion {
                            Ok(completion) => {
                                if let Some(content) = &completion.choices[0].delta.content {
                                    print!("{}", content);
                                    std::io::Write::flush(&mut std::io::stdout())?;
                                    message_content.push_str(content);
                                }
                            }
                            Err(e) => println!("Error fetching response from pprovider: {}", e)
                        }
                    }
                }
                chat_request.messages.push(ChatCompletionsMessage::Assistant{
                    content: message_content, reasoning_content:None
                });
               println!();
            }
        }
    }
}

fn select_model(models: &[Model])-> Result<String, Box<dyn Error>> {

    loop {
        println!("\nAvailable models:");
        for (i, m) in models.iter().enumerate() {
            println!(" {}) {}", i+1, m.id);
        }

        print!("Select a model [1-{}]:", models.len());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match input.trim().parse::<usize>() {
            Ok(n) if (1..=models.len()).contains(&n) => {
                return Ok(models[n - 1].id.clone());
            },
            _ => println!("Invalid selection, try again"),
        }
    }
}

   
