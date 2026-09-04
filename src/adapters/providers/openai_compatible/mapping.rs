use crate::{
    adapters::providers::openai_compatible::models::{
        ChatCompletionFunction, ChatCompletionTool, ChatCompletionToolCall,
        ChatCompletionToolCallFunction, ChatCompletionToolParameterProperty,
        ChatCompletionToolParameterPropertyType, ChatCompletionToolParameters,
        ChatCompletionsMessageRequest, ProviderModel,
    },
    error::LlmError,
    models::{
        Message, Model, ModelModality, ReasoningEffort, Tool, ToolCall,
        ToolParameterProperty, ToolParameterPropertyType, ToolParameters,
    },
};

pub(super) fn to_provider_message(message: &Message) -> ChatCompletionsMessageRequest {
    match message {
        Message::System { content } => ChatCompletionsMessageRequest::System {
            content: content.clone(),
        },
        Message::User { content } => ChatCompletionsMessageRequest::User {
            content: content.clone(),
        },
        Message::Assistant {
            content,
            tool_calls,
        } => ChatCompletionsMessageRequest::Assistant {
            content: content.clone(),
            tool_calls: tool_calls.iter().map(to_provider_toolcall).collect(),
            reasoning_content: None,
        },
        Message::Tool {
            tool_call_id,
            content,
        } => ChatCompletionsMessageRequest::Tool {
            tool_call_id: tool_call_id.clone(),
            content: content.clone(),
        },
    }
}

pub(super) fn to_provider_reasoning_effort(reasoning_effort: ReasoningEffort) -> String {
    match reasoning_effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
    .into()
}

fn to_domain_model_modality(modality: &str) -> Result<ModelModality, LlmError> {
    match modality {
        "text" => Ok(ModelModality::Text),
        "image" => Ok(ModelModality::Image),
        "video" => Ok(ModelModality::Video),
        "audio" => Ok(ModelModality::Audio),
        _ => Err(LlmError::InvalidModelModality { modality: (modality.into()) }), 
    }
}

fn to_domain_model_modalities(modalities: Vec<String>) -> Result<Vec<ModelModality>, LlmError> {
   modalities
       .into_iter()
       .map(|modality| to_domain_model_modality(&modality))
       .collect()
}

pub(super) fn to_domain_model(provider_model: ProviderModel) -> Result<Model, LlmError> {
    Ok( Model {
        id: provider_model.id,
        input_modalities: to_domain_model_modalities(provider_model.input_modalities)?,
        output_modalities: to_domain_model_modalities(provider_model.output_modalities)?,
    })
}

pub(super) fn to_domain_toolcall(tool_call: &ChatCompletionToolCall) -> ToolCall {
    ToolCall {
        id: tool_call.id.clone(),
        name: tool_call.function.name.clone(),
        arguments: tool_call.function.arguments.clone(),
    }
}

fn to_provider_toolcall(tool_call: &ToolCall) -> ChatCompletionToolCall {
    ChatCompletionToolCall {
        id: tool_call.id.to_string(),
        tool_type: String::from("function"),
        function: ChatCompletionToolCallFunction {
            name: tool_call.name.to_string(),
            arguments: tool_call.arguments.to_string(),
        },
    }
}

pub(super) fn to_provider_tools(tool: Tool) -> ChatCompletionTool {
    ChatCompletionTool {
        tool_type: String::from("function"),
        function: ChatCompletionFunction {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters.map(to_provider_parameters),
        },
    }
}

fn to_provider_parameters(parameter: ToolParameters) -> ChatCompletionToolParameters {
    ChatCompletionToolParameters {
        parameter_type: String::from("object"),
        properties: parameter
            .properties
            .into_iter()
            .map(to_provider_parameter_property)
            .collect(),
        required: parameter.required,
    }
}

fn to_provider_parameter_property(
    property: ToolParameterProperty,
) -> (String, ChatCompletionToolParameterProperty) {
    (
        property.name,
        ChatCompletionToolParameterProperty {
            property_type: to_provider_property_type(property.property_type),
            description: property.description.to_string(),
            property_enum: property.property_enum,
        },
    )
}

pub(super) fn to_provider_property_type(
    property_type: ToolParameterPropertyType,
) -> ChatCompletionToolParameterPropertyType {
    match property_type {
        ToolParameterPropertyType::String => ChatCompletionToolParameterPropertyType::String,
        ToolParameterPropertyType::Number => ChatCompletionToolParameterPropertyType::Number,
        ToolParameterPropertyType::Integer => ChatCompletionToolParameterPropertyType::Integer,
        ToolParameterPropertyType::Boolean => ChatCompletionToolParameterPropertyType::Boolean,
    }
}
