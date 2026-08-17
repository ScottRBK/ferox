#[derive(Debug)]
pub struct Model {
    pub id: String,
}

pub struct CompletionRequest<'a> {
    pub model: String,
    pub messages: &'a [Message],
    pub stream: bool,
}

pub struct CompletionResponse {
    pub model: String,
    pub text: String,
    pub reasoning: Option<String>,
}

pub struct CompletionChunk {
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub finished: bool,
}

pub enum Message {
    System {content: String},
    User {content: String},
    Assistant {content: String},
     
}
