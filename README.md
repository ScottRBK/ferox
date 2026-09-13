# ferox-ai

ferox-ai is my first open source Rust library. It provides a shared interface for interacting
with LLM providers, with an OpenAI-compatible adapter available today.

Add `ferox-ai` as a dependency in `Cargo.toml`; use `ferox_ai` in Rust imports.

## Example usage

This example uses an OpenAI-compatible chat completions provider at `http://localhost:8080/v1`.

See [examples](./examples/) for more information

```rust 
use std::error::Error;
use ferox_ai::Gateway;
use ferox_ai::models::{CompletionRequest, Message, Model};
use ferox_ai::openai_compatible::OpenAiCompatibleClient;

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
```


`Model::new(id, None, None)` defaults both input and output modalities to text.
Pass `Some(vec![...])` to specify either list. An explicitly empty list stays empty.
These defaults apply to the constructor; the provider's model listing does not use them.

## Development roadmap

- [x] OpenAI Compatible Chat Completions endpoint 
    - [x] handle reasoning injection into context array
    - [x] handle reasoning effort being passed to inference provider 
- [ ] Basic Logging 
- [x] Provider Configuration
- [ ] Open Telemtry and Advanced Logging
- [ ] Publish as cargo package
