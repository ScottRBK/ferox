#[derive(Debug)]
pub struct Model {
    pub id: String,
    pub input_modalities: Vec<ModelModality>,
    pub output_modalities: Vec<ModelModality>,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ModelModality {
    Text,
    Image,
    Video,
    Audio,
}

#[non_exhaustive]
pub enum ToolParameterPropertyType {
    String,
    Number,
    Integer,
    Boolean,
    // TODO: Need to add support for Object and Array types
}

#[non_exhaustive]
pub struct ToolParameterProperty {
    pub name: String,
    pub property_type: ToolParameterPropertyType,
    pub description: String,
    pub property_enum: Option<Vec<String>>,
}

impl ToolParameterProperty {
    pub fn new(
        name: impl Into<String>, 
        property_type: ToolParameterPropertyType, 
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            property_type,
            description: description.into(),
            property_enum: None,
        }
    }
}

#[non_exhaustive]
pub struct ToolParameters {
    pub properties: Vec<ToolParameterProperty>,
    pub required: Vec<String>,
}

impl ToolParameters {
    pub fn new(properties: Vec<ToolParameterProperty>, required: Vec<String>) -> Self {
        Self { properties, required }
    }
}

#[non_exhaustive]
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

#[non_exhaustive]
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

#[non_exhaustive]
pub struct CompletionRequest<'a> {
    pub model: String,
    pub messages: &'a [Message],
    pub tools: Option<Vec<Tool>>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl<'a> CompletionRequest<'a> {
    pub fn new(
        model: String, 
        messages: &'a [Message],
    ) -> Self {
        Self {
            model,
            messages,
            tools: None, 
            reasoning_effort: None,
        }
    }
}

#[non_exhaustive]
pub struct CompletionResponse {
    pub model: String,
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl CompletionResponse{
    pub fn new (
        model: String,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            model,
            text: None,
            reasoning: None,
            tool_calls,
        }
    }
}

#[non_exhaustive]
pub struct CompletionChunk {
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finished: bool,
}

impl CompletionChunk {
    pub fn new(tool_calls: Vec<ToolCall>, finished: bool,) -> Self {
        Self {
            text: None,
            reasoning: None,
            tool_calls,
            finished,
        }
    }
} 

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn new(id: String, name: String, arguments: String) -> Self {
        Self { id, name, arguments }
    }
}

#[non_exhaustive]
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
