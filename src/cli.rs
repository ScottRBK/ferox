use std::error::Error;
use std::io::{self, Write};

use crate::llm_providers::openai_compatible:: { 
    Model, 
    OpenAiCompatibleClient,
};
use crate::llm_providers::openai_compatible:: {
    ChatCompletionsRequestMessage,
    ChatCompletionsRequest,
};



pub async fn repl(client: OpenAiCompatibleClient) -> Result<(), Box<dyn Error>> {
   
    println!("Welcome to ferox! (type q to exit)");

    let models = client.list_models().await?;
    let model  = select_model(&models)?;
    let mut messages = Vec::<ChatCompletionsRequestMessage>::new();

    println!("Model Selected: {}", model);
     
    let mut chat_request = ChatCompletionsRequest {
        model: model,
        messages: messages,
        stream: false,
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
                   let message = ChatCompletionsRequestMessage {
                        role: String::from("user"),
                        content: user_input.trim().into()
                   };
                   chat_request.messages.push(message);
                 }
        }
        
        let response = client.create_chat_completion(&chat_request).await?;
        println!("{}", response)

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

   
