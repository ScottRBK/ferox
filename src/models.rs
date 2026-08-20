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
    pub parameters: Option<ToolParameters>,
}

impl Tool {
    pub fn new (
        name: impl Into<String>,
        description: impl Into<String>, 
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: None,
     }        
    }

    pub fn required_parameter (self, parameter_property: ToolParameterProperty) -> Self {
        self.parameter(parameter_property, true)
    }
    
    pub fn optional_parameter (self, parameter_property: ToolParameterProperty) -> Self {
        self.parameter(parameter_property, false)
    }
    fn parameter (mut self, parameter_property: ToolParameterProperty, required: bool) -> Self {

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
