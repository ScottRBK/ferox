use crate::{
    error::{LlmError},
    models::{ToolCall}
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug)]
pub(super) struct ProviderModel {
    pub(super) id: String,
    #[serde(default)]
    pub(super) input_modalities: Vec<String>,
    #[serde(default)]
    pub(super) output_modalities: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub(super) struct ModelsResponse {
    pub(super) data: Vec<ProviderModel>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(super) struct ChatCompletionToolCallFunction {
    pub(super) name: String,
    pub(super) arguments: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(super) struct ChatCompletionToolCall {
    pub(super) id: String,
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: ChatCompletionToolCallFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum ChatCompletionsMessageRequest {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ChatCompletionToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum ChatCompletionsMessageResponse {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ChatCompletionToolCall>,
        reasoning_content: Option<String>,
    },
}

impl ChatCompletionsMessageResponse {
    pub(super) fn content(&self) -> Option<&str> {
        match self {
            ChatCompletionsMessageResponse::System { content } => Some(content),
            ChatCompletionsMessageResponse::User { content } => Some(content),
            ChatCompletionsMessageResponse::Assistant { content, .. } => content.as_deref(),
        }
    }

    pub(super) fn reasoning_content(&self) -> Option<&str> {
        match self {
            ChatCompletionsMessageResponse::Assistant {
                reasoning_content, ..
            } => reasoning_content.as_deref(),
            _ => None,
        }
    }

    pub(super) fn tool_calls(&self) -> &[ChatCompletionToolCall] {
        match self {
            ChatCompletionsMessageResponse::Assistant { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum ChatCompletionToolParameterPropertyType {
    String,
    Number,
    Integer,
    Boolean,
    // TODO: Need to add support for Object and Array types
}

#[derive(Serialize, Debug)]
pub(super) struct ChatCompletionToolParameterProperty {
    #[serde(rename = "type")]
    pub(super) property_type: ChatCompletionToolParameterPropertyType,
    pub(super) description: String,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub(super) property_enum: Option<Vec<String>>,
}

#[derive(Serialize, Debug)]
pub(super) struct ChatCompletionToolParameters {
    #[serde(rename = "type")]
    pub(super) parameter_type: String,
    pub(super) properties: HashMap<String, ChatCompletionToolParameterProperty>,
    pub(super) required: Vec<String>,
}

#[derive(Serialize, Debug)]
pub(super) struct ChatCompletionTool {
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: ChatCompletionFunction,
}

#[derive(Serialize, Debug)]
pub(super) struct ChatCompletionFunction {
    pub(super) name: String,
    pub(super) description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) parameters: Option<ChatCompletionToolParameters>,
}

#[derive(Serialize, Debug)]
pub(super) struct ChatCompletionsRequest {
    pub(super) model: String,
    pub(super) messages: Vec<ChatCompletionsMessageRequest>,
    pub(super) stream: bool,
    pub(super) tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning_effort: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub(super) enum ChoicesFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionChoices {
    pub(super) message: ChatCompletionsMessageResponse,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsResponse {
    pub(super) choices: Vec<ChatCompletionChoices>,
    pub(super) model: String,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsToolCallDelta {
    pub(super) index: usize,
    pub(super) id: Option<String>,
    pub(super) function: ChatCompletionsToolCallFunctionDelta,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsToolCallFunctionDelta {
    pub(super) name: Option<String>,
    pub(super) arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsStreamDelta {
    pub(super) content: Option<String>,
    #[serde(alias = "reasoning")]
    pub(super) reasoning_content: Option<String>,
    #[serde(default)]
    pub(super) tool_calls: Vec<ChatCompletionsToolCallDelta>,
}

#[derive(Default)]
pub(super) struct PendingToolCall {
    pub(super) id: Option<String>,
    pub(super) name: Option<String>,
    pub(super) arguments: String,
}

impl PendingToolCall {
    pub(super) fn apply(&mut self, delta: &ChatCompletionsToolCallDelta) {
        if let Some(id) = &delta.id {
            self.id = Some(id.clone());
        }

        if let Some(name) = &delta.function.name {
            self.name = Some(name.clone());
        }

        if let Some(arguments) = &delta.function.arguments {
            self.arguments.push_str(arguments);
        }
    }

    pub(super) fn finish(self) -> Result<ToolCall, LlmError> {
        let id = self.id.ok_or_else(|| LlmError::InvalidResponse {
            message: "streamed tool call was missing an id".into(),
        })?;

        let name = self.name.ok_or_else(|| LlmError::InvalidResponse {
            message: "streamed tool call was missing a function name".into(),
        })?;

        Ok(ToolCall {
            id,
            name,
            arguments: self.arguments,
        })
    }
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsStreamChoice {
    pub(super) delta: ChatCompletionsStreamDelta,
    pub(super) finish_reason: Option<ChoicesFinishReason>,
}

#[derive(Deserialize, Debug)]
pub(super) struct ChatCompletionsStreamResponse {
    pub(super) choices: Vec<ChatCompletionsStreamChoice>,
}
