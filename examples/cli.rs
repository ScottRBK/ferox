use std::error::Error;
use std::io::{self, Write, IsTerminal};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::pin_mut;
use futures_util::StreamExt;

use ferox::adapters::providers::openai_compatible::{OpenAiCompatibleClient};
use ferox::ports::llm::LlmProvider;
use ferox::gateway::Gateway;
use ferox::models::{
    Model, 
    CompletionRequest, 
    Message,
    Tool,
    ToolCall,
    ToolParameterProperty,
    ToolParameterPropertyType,
};

const BASE_URL: &str = "http://192.168.1.202:8080/v1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = OpenAiCompatibleClient::builder()
        .base_url(BASE_URL)
        .api_key("")
        .build()?;

    let gateway = Gateway::new(client);
    repl(gateway).await
}

async fn repl<P>(gateway: Gateway<P>) -> Result<(), Box<dyn Error + Send + Sync>> 
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


fn build_tools() -> Vec<Tool> {
 vec![
     Tool::new("get_current_datetime", "gets the current date and time in seconds since unix epoch"),
     Tool::new("add_two_numbers", "adds two integers together")
         .required_parameter(
             ToolParameterProperty {
                 name: String::from("first_number"),
                 property_type: ToolParameterPropertyType::Integer,
                 description: String::from("first number to be added"),
                 property_enum: None,
             }
         )
         .required_parameter(
             ToolParameterProperty {
                 name: String::from("second_number"),
                 property_type: ToolParameterPropertyType::Integer,
                 description: String::from("second number to be added"),
                 property_enum: None,
             }
         )
]
}

async fn get_user_input() -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut user_input = String::new();
    print!("> ");
    io::stdout().flush()?;
    io::stdin()
        .read_line(&mut user_input)
        .expect("Error reading input");

    Ok(user_input)
}

async fn chat_session<P>(model: &Model, gateway: Gateway<P>) -> Result<(), Box<dyn Error + Send + Sync>> 
where 
    P: LlmProvider 
{
    let mut messages = Vec::<Message>::new();
    let tty = std::io::stdout().is_terminal();
    let dim = if tty {"\x1b[90m"} else { "" };
    let reset = if tty {"\x1b[0m"} else { "" };


    loop {
         
        let user_input = get_user_input().await?;

        match user_input.trim() {
            "q" => break,
            _ => {
                messages.push(Message::User{content: user_input.trim().into()});
            }
        }

        loop {

            let completion = gateway.complete(CompletionRequest{
                model: model.id.clone(),
                messages: &messages,
                tools: Some(build_tools()),
            }).await?;

            let mut seen_reasoning = false;
            let mut agent_response = String::new();

            if let Some(response) = &completion.reasoning {
                if !seen_reasoning {
                    println!["REASONING"];
                    println!();
                    seen_reasoning = true;
                }

                print!("{dim}{}", response);
                std::io::Write::flush(&mut  std::io::stdout())?; 
            }

            if let Some(response) = &completion.text && seen_reasoning && !response.is_empty()  {
                    println!();
                    println!["{reset}AGENT RESPONSE:"];
                    println!();
                    print!("{}", response);
                    std::io::Write::flush(&mut  std::io::stdout())?; 
                    agent_response.push_str(response);
            }

            messages.push(Message::Assistant { 
                content: Some(agent_response),
                tool_calls: completion.tool_calls.clone(),
            });

            let tool_calls = completion.tool_calls.clone(); 
            messages.extend(handle_tool_calls(&tool_calls).unwrap());
            if tool_calls.is_empty() {
                break;
            }
       }

        println!();
    }
    Ok(())
}

fn handle_tool_calls(tool_calls: &Vec<ToolCall>) -> Result<Vec<Message>, Box<dyn Error>> {
    let mut tool_messages: Vec<Message> = Vec::new();
                    
    for tool in tool_calls{
        println!("executing tool {}", tool.name);
        std::io::Write::flush(&mut  std::io::stdout())?; 
        let result = execute_tool_call(tool)?;
        tool_messages.push(result)
    }

    Ok(tool_messages)
}

fn execute_tool_call(tool_call: &ToolCall) -> Result<Message, Box<dyn Error>>{

    let content = match tool_call.name.as_str() {
        "get_current_datetime" => get_current_datetime().to_string(),
        "add_two_numbers"  => {
            let arguments: AddTwoNumbersArguments = parse_tool_arguments(&tool_call.arguments)?;
            add_two_numbers ( arguments.first_number, arguments.second_number, ).to_string()
        },
        other => format!("unsupported/not-implemented {other}"),
    };

    Ok(Message::Tool {
        tool_call_id: tool_call.id.clone(),
        content,
    })
}

fn parse_tool_arguments<T>(arguments: &str) -> Result<T, serde_json::Error>
where 
    T: DeserializeOwned,
{
    serde_json::from_str(arguments)
}

use serde::de::DeserializeOwned;
use serde::Deserialize; 

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddTwoNumbersArguments {
    first_number: u32, 
    second_number: u32,
}

fn add_two_numbers(first_number: u32, second_number: u32) -> u32 {
    first_number + second_number 
}

fn get_current_datetime() -> u64 {

     SystemTime::now()
         .duration_since(UNIX_EPOCH)
         .unwrap()
         .as_secs()
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
