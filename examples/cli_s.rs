use std::error::Error;
use std::io::{self, IsTerminal, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use futures_util::pin_mut;

use serde::Deserialize;
use serde::de::DeserializeOwned;

use ferox::adapters::providers::openai_compatible::OpenAiCompatibleClient;
use ferox::gateway::Gateway;
use ferox::models::{
    CompletionRequest, Message, Model, Tool, ToolCall, ToolParameterProperty,
    ToolParameterPropertyType,
};
use ferox::ports::llm::LlmProvider;

const BASE_URL: &str = "http://192.168.1.201:8080/v1";

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
    P: LlmProvider,
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
        Tool::new("get_current_datetime", "gets the current date and time"),
        Tool::new("add_two_numbers", "adds two numbers together")
            .required_parameter(ToolParameterProperty {
                name: String::from("first_number"),
                property_type: ToolParameterPropertyType::Integer,
                description: String::from("first number to be added"),
                property_enum: None,
            })
            .required_parameter(ToolParameterProperty {
                name: String::from("second_number"),
                property_type: ToolParameterPropertyType::Integer,
                description: String::from("second number to be added"),
                property_enum: None,
            }),
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

async fn chat_session<P>(
    model: &Model,
    gateway: Gateway<P>,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    P: LlmProvider,
{
    let mut messages = Vec::<Message>::new();
    let tty = std::io::stdout().is_terminal();
    let dim = if tty { "\x1b[90m" } else { "" };
    let reset = if tty { "\x1b[0m" } else { "" };

    loop {
        let user_input = get_user_input().await?;

        match user_input.trim() {
            "q" => break,
            _ => {
                messages.push(Message::User {
                    content: user_input.trim().into(),
                });
            }
        }

        loop {
            let stream = gateway
                .stream(CompletionRequest {
                    model: model.id.clone(),
                    messages: &messages,
                    tools: Some(build_tools()),
                    reasoning_effort: None,
                })
                .await?;

            pin_mut!(stream);

            let mut seen_reasoning = false;
            let mut seen_agent_response = false;
            let mut agent_response = String::new();
            let mut tool_calls = Vec::new();

            while let Some(completion) = stream.next().await {
                let completion = completion?;

                if let Some(response) = &completion.reasoning {
                    if !seen_reasoning {
                        println!("REASONING");
                        println!();
                        seen_reasoning = true;
                    }

                    print!("{dim}{response}");
                    io::stdout().flush()?;
                }

                if let Some(response) = &completion.text
                    && !response.is_empty()
                {
                    if !seen_agent_response {
                        if seen_reasoning {
                            println!();
                        }

                        println!("{reset}AGENT RESPONSE:");
                        println!();
                        seen_agent_response = true;
                    }

                    print!("{response}");
                    io::stdout().flush()?;
                    agent_response.push_str(response);
                }

                tool_calls.extend(completion.tool_calls);
            }

            if seen_reasoning && !seen_agent_response {
                println!("{reset}");
            }

            let assistant_content = if agent_response.is_empty() {
                None
            } else {
                Some(agent_response)
            };

            messages.push(Message::Assistant {
                content: assistant_content,
                tool_calls: tool_calls.clone(),
            });

            if tool_calls.is_empty() {
                break;
            }

            messages.extend(handle_tool_calls(&tool_calls)?);
        }

        println!();
    }
    Ok(())
}

fn handle_tool_calls(
    tool_calls: &[ToolCall],
) -> Result<Vec<Message>, Box<dyn Error + Send + Sync>> {
    let mut tool_messages: Vec<Message> = Vec::new();

    for tool in tool_calls {
        println!("executing tool {}", tool.name);
        io::stdout().flush()?;

        let message = match execute_tool_call(tool) {
            Ok(message) => message,
            Err(error) => Message::Tool {
                tool_call_id: tool.id.clone(),
                content: format!("error executing tool: {error}"),
            },
        };

        tool_messages.push(message);
    }

    Ok(tool_messages)
}

fn execute_tool_call(tool_call: &ToolCall) -> Result<Message, Box<dyn Error + Send + Sync>> {
    let content = match tool_call.name.as_str() {
        "get_current_datetime" => get_current_unix_epoch_datetime()?.to_string(),
        "add_two_numbers" => {
            let arguments: AddTwoNumbersArguments = parse_tool_arguments(&tool_call.arguments)?;
            add_two_numbers(arguments.first_number, arguments.second_number)?.to_string()
        }
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddTwoNumbersArguments {
    first_number: i32,
    second_number: i32,
}

fn add_two_numbers(
    first_number: i32,
    second_number: i32,
) -> Result<i32, Box<dyn Error + Send + Sync>> {
    first_number
        .checked_add(second_number)
        .ok_or_else(|| "addition result is outside of supported integer range".into())
}

fn get_current_unix_epoch_datetime() -> Result<u64, Box<dyn Error + Send + Sync>> {
    let duration = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(duration.as_secs())
}

fn print_models(models: &[Model]) {
    for (i, model) in models.iter().enumerate() {
        println!["{}. {}", i + 1, model.id];
    }
}

async fn select_models(models: &[Model]) -> Result<&Model, Box<dyn Error + Send + Sync>> {
    println!("Please enter a number for a model to talk to");

    print_models(models);

    loop {
        let mut user_input = String::new();
        io::stdout().flush()?;
        io::stdin()
            .read_line(&mut user_input)
            .expect("error reading input");
        match user_input.trim().parse::<usize>() {
            Ok(n) if (1..=models.len()).contains(&n) => {
                return Ok(&models[n - 1]);
            }
            _ => println!("Invalid selection, try again"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle_add_tool_call(arguments: &str) -> Message {
        let tool_call = ToolCall {
            id: "call-1".into(),
            name: "add_two_numbers".into(),
            arguments: arguments.into(),
        };

        handle_tool_calls(&[tool_call])
            .expect("tool errors should be returned as tool messages")
            .into_iter()
            .next()
            .expect("expected one tool message")
    }

    fn assert_tool_message(message: Message, expected_content: &str) {
        let Message::Tool {
            tool_call_id,
            content,
        } = message
        else {
            panic!("expected a tool message");
        };

        assert_eq!(tool_call_id, "call-1");
        assert!(
            content.contains(expected_content),
            "expected `{content}` to contain `{expected_content}`"
        );
    }

    #[test]
    fn valid_arguments_return_tool_result() {
        let message = handle_add_tool_call(r#"{"first_number":2,"second_number":3}"#);

        assert_tool_message(message, "5");
    }

    #[test]
    fn invalid_argument_type_returns_tool_error() {
        let message = handle_add_tool_call(r#"{"first_number":"two","second_number":3}"#);

        assert_tool_message(message, "error executing tool:");
        assert_tool_message(
            handle_add_tool_call(r#"{"first_number":"two","second_number":3}"#),
            "expected i32",
        );
    }

    #[test]
    fn addition_overflow_returns_tool_error_without_panicking() {
        let message = handle_add_tool_call(r#"{"first_number":2147483647,"second_number":1}"#);

        assert_tool_message(message, "outside of supported integer range");
    }
}
