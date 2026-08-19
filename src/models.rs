#[derive(Debug)]
pub struct Model {
    pub id: String,
}

pub struct ToolParameterProperty {
    pub name: String,
    pub property_type: String,
    pub description: String,
    pub property_enum: Option<Vec<String>>,
}

pub struct ToolParameters {
    pub properties: Vec<ToolParameterProperty>, 
    pub required: Vec<String>,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

pub struct CompletionRequest<'a> {
    pub model: String,
    pub messages: &'a [Message],
    pub tools: Option<Vec<Tool>>,
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
