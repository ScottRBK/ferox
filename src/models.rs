//! Shared request, response, and tool types used by the gateway and provider adapters.

/// A model advertised by a provider, with its supported input and output formats.
#[derive(Debug)]
#[non_exhaustive]
pub struct Model {
    /// Identifier to use in a completion request.
    pub id: String,
    /// Formats the model accepts as input.
    pub input_modalities: Vec<ModelModality>,
    /// Formats the model can produce.
    pub output_modalities: Vec<ModelModality>,
}

impl Model {
    /// Creates model metadata from a provider identifier and its supported formats.
    pub fn new(
        id: impl Into<String>,
        input_modalities: Option<Vec<ModelModality>>,
        output_modalities: Option<Vec<ModelModality>>,
    ) -> Self {
        Self {
            id: id.into(),
            input_modalities: input_modalities.unwrap_or_else(|| vec![ModelModality::Text]),
            output_modalities: output_modalities.unwrap_or_else(|| vec![ModelModality::Text]),
        }
    }
}

/// A format a model can accept or produce.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelModality {
    /// Written text.
    Text,
    /// Still images.
    Image,
    /// Video.
    Video,
    /// Audio.
    Audio,
}

/// A scalar JSON type accepted by a tool parameter.
#[non_exhaustive]
pub enum ToolParameterPropertyType {
    /// A JSON string.
    String,
    /// A JSON number, including fractional values.
    Number,
    /// A whole number.
    Integer,
    /// A true or false value.
    Boolean,
    // TODO: Need to add support for Object and Array types
}

/// The name, type, and description of one tool argument.
#[non_exhaustive]
pub struct ToolParameterProperty {
    /// Argument name in the tool call JSON object.
    pub name: String,
    /// JSON type expected for this argument.
    pub property_type: ToolParameterPropertyType,
    /// Description sent to the model to explain the argument.
    pub description: String,
    /// Allowed string values, or `None` when no enum constraint is specified.
    pub property_enum: Option<Vec<String>>,
}

impl ToolParameterProperty {
    /// Creates an argument definition without an enum constraint.
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

/// The argument definitions for a tool.
#[non_exhaustive]
pub struct ToolParameters {
    /// Arguments the tool accepts.
    pub properties: Vec<ToolParameterProperty>,
    /// Names of arguments that must be supplied.
    pub required: Vec<String>,
}

impl ToolParameters {
    /// Creates a parameter schema without validating the required names.
    pub fn new(properties: Vec<ToolParameterProperty>, required: Vec<String>) -> Self {
        Self {
            properties,
            required,
        }
    }
}

/// A function the model can ask the caller to execute.
///
/// Ferox passes tool definitions and calls between the caller and provider.
/// The caller is responsible for executing the requested function.
///
/// ```
/// use ferox::models::{Tool, ToolParameterProperty, ToolParameterPropertyType};
///
/// let tool = Tool::new("get_weather", "Get the current weather")
///     .required_parameter(ToolParameterProperty::new(
///         "city",
///         ToolParameterPropertyType::String,
///         "City to get the weather for",
///     ));
///
/// let parameters = tool.parameters.as_ref().unwrap();
/// assert_eq!(parameters.required, ["city"]);
/// assert_eq!(parameters.properties.len(), 1);
/// ```
#[non_exhaustive]
pub struct Tool {
    /// Function name used to identify the tool in requests and responses.
    pub name: String,
    /// Description sent to the model to explain when to use the tool.
    pub description: String,
    /// Argument schema, or `None` for a tool with no parameters.
    pub parameters: Option<ToolParameters>,
}

impl Tool {
    /// Creates a tool with no parameters.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: None,
        }
    }

    /// Adds an argument and records its name in the required list.
    pub fn required_parameter(self, parameter_property: ToolParameterProperty) -> Self {
        self.parameter(parameter_property, true)
    }

    /// Adds an argument without marking it as required.
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

/// The requested reasoning level. Support and meaning depend on the provider and model.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ReasoningEffort {
    /// Requests no reasoning.
    None,
    /// Requests minimal reasoning.
    Minimal,
    /// Requests low reasoning effort.
    Low,
    /// Requests medium reasoning effort.
    Medium,
    /// Requests high reasoning effort.
    High,
    /// Requests extra-high reasoning effort.
    XHigh,
    /// Requests the maximum reasoning effort.
    Max,
}

impl ReasoningEffort {
    /// All reasoning levels, ordered from none to maximum.
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

/// A completion request that borrows its conversation history.
///
/// ```
/// use ferox::models::{CompletionRequest, Message};
///
/// let messages = [Message::User {
///     content: "What is the capital of France?".into(),
/// }];
/// let request = CompletionRequest::new("my-model".into(), &messages);
///
/// assert_eq!(request.messages.len(), 1);
/// assert!(request.tools.is_none());
/// assert!(request.reasoning_effort.is_none());
/// ```
#[non_exhaustive]
pub struct CompletionRequest<'a> {
    /// Provider model identifier to send the request to.
    pub model: String,
    /// Conversation history in the order the model should receive it.
    pub messages: &'a [Message],
    /// Tools available to the model, or `None` to omit tool definitions.
    pub tools: Option<Vec<Tool>>,
    /// Requested reasoning level, or `None` to leave it to the provider.
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl<'a> CompletionRequest<'a> {
    /// Creates a request with no tools or explicit reasoning level.
    pub fn new(model: String, messages: &'a [Message]) -> Self {
        Self {
            model,
            messages,
            tools: None,
            reasoning_effort: None,
        }
    }
}

/// A completed model response, which may contain text, reasoning, or tool calls.
#[non_exhaustive]
pub struct CompletionResponse {
    /// Model identifier reported by the provider.
    pub model: String,
    /// Response text, when the provider supplies it.
    pub text: Option<String>,
    /// Reasoning text, when the provider exposes it.
    pub reasoning: Option<String>,
    /// Function calls requested by the model for the caller to execute.
    pub tool_calls: Vec<ToolCall>,
}

impl CompletionResponse {
    /// Creates a response with the supplied tool calls and no text or reasoning.
    pub fn new(model: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            model,
            text: None,
            reasoning: None,
            tool_calls,
        }
    }
}

/// The reason a provider stopped generating a response.
#[non_exhaustive]
pub enum FinishReason {
    /// The model finished normally or reached a stop sequence.
    Stop,
    /// Generation reached the output length limit.
    Length,
    /// The model requested one or more tool calls.
    ToolCalls,
    /// A content filter stopped generation.
    ContentFilter,
}

/// One update from a streaming completion.
///
/// Text and reasoning contain new fragments, not the full response so far.
#[non_exhaustive]
pub struct CompletionChunk {
    /// New response text to append, if present.
    pub text: Option<String>,
    /// New reasoning text to append, if present.
    pub reasoning: Option<String>,
    /// Tool calls delivered in this update.
    /// The OpenAI-compatible adapter assembles them before emitting them on a finish event.
    pub tool_calls: Vec<ToolCall>,
    /// Reason generation stopped, if reported in this update.
    pub finished_reason: Option<FinishReason>,
}

impl CompletionChunk {
    /// Creates a chunk with the supplied tool calls and finish reason, without text or reasoning.
    pub fn new(tool_calls: Vec<ToolCall>, finished_reason: Option<FinishReason>) -> Self {
        Self {
            text: None,
            reasoning: None,
            tool_calls,
            finished_reason,
        }
    }
}

/// A function call requested by the model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolCall {
    /// Provider call identifier used to match a tool result to this call.
    pub id: String,
    /// Name of the function to execute.
    pub name: String,
    /// Arguments as a JSON string. The caller must parse and validate them before use.
    pub arguments: String,
}

impl ToolCall {
    /// Creates a tool call without parsing or validating its arguments.
    pub fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }
}

/// One entry in a conversation sent to a model.
#[non_exhaustive]
pub enum Message {
    /// Instructions that guide the model throughout the conversation.
    System {
        /// Instructions for the model.
        content: String,
    },
    /// Input from the user.
    User {
        /// Text supplied by the user.
        content: String,
    },
    /// A previous model response, including any requested tool calls.
    Assistant {
        /// Text from the model, if present.
        content: Option<String>,
        /// Function calls requested in this response.
        tool_calls: Vec<ToolCall>,
        /// Reasoning text to include in the conversation history, if present.
        reasoning: Option<String>,
    },
    /// The result of a tool call executed by the caller.
    Tool {
        /// Identifier of the tool call this result answers.
        tool_call_id: String,
        /// Tool output to return to the model.
        content: String,
    },
}
