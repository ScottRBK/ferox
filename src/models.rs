#[derive(Debug)]
pub struct Model {
    pub id: String,
    pub input_modalities: Vec<ModelModality>,
    pub output_modalities: Vec<ModelModality>,
}

#[derive(Debug)]
pub enum ModelModality {
    Text,
    Image,
    Video,
    Audio,
}

pub enum ToolParameterPropertyType {
    String,
    Number,
    Integer,
    Boolean,
    // TODO: Need to add support for Object and Array types
}

pub struct ToolParameterProperty {
    pub name: String,
    pub property_type: ToolParameterPropertyType,
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
    pub parameters: Option<ToolParameters>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: None,
        }
    }

    pub fn required_parameter(self, parameter_property: ToolParameterProperty) -> Self {
        self.parameter(parameter_property, true)
    }

    pub fn optional_parameter(self, parameter_property: ToolParameterProperty) -> Self {
        self.parameter(parameter_property, false)
    }

    fn parameter(mut self, parameter_property: ToolParameterProperty, required: bool) -> Self {
        let parameters = self.parameters.get_or_insert_with(|| ToolParameters {
            properties: Vec::new(),
            required: Vec::new(),
        });

        if required {
            parameters.required.push(parameter_property.name.clone());
        }

        parameters.properties.push(parameter_property);

        self
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub const ALL: [ReasoningEffort; 7] = [
        ReasoningEffort::None,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ];
}


pub struct CompletionRequest<'a> {
    pub model: String,
    pub messages: &'a [Message],
    pub tools: Option<Vec<Tool>>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

pub struct CompletionResponse {
    pub model: String,
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

pub struct CompletionChunk {
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finished: bool,
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}
