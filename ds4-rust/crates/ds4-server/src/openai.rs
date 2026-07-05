// DS4 (DwarfStar) — OpenAI-compatible request/response types.
//
// Targets the wire format of the OpenAI Python SDK (v1):
//   POST /v1/chat/completions   ChatCompletionRequest / Response
//   POST /v1/completions        CompletionRequest / Response
//   GET  /v1/models             ModelList
//   POST /v1/responses          ResponsesRequest / Response
//
// All fields mirror the upstream `ds4_server.c` JSON shapes; the
// Rust port is serde-friendly and uses `#[serde(default)]` for
// forward-compat with new fields.

// Many of these struct fields / enum variants are part of the wire
// schema and are read by external clients (the `ds4-server`
// integration tests in the lib target). They are intentionally not
// consumed inside `main.rs`, so silence dead-code for the module.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ToolSpecFunction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolSpecFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<usize>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOut>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallOut {
    pub id: String,
    pub r#type: &'static str,
    pub function: ToolCallFunctionOut,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionOut {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub output: Vec<ResponsesOutput>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesOutput {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        content: Vec<ResponsesContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning { id: String, summary: Vec<String> },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

/// SSE chunk payloads.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOut>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorBody {
    pub fn new(message: impl Into<String>, kind: &'static str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: kind,
                param: None,
                code: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_deserialises_minimal() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(r.messages.len(), 1);
        assert_eq!(r.messages[0].role, "user");
    }

    #[test]
    fn chat_request_deserialises_with_tools() {
        let body = r#"{
            "model":"m",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}]
        }"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.tools[0].function.as_ref().unwrap().name, "f");
    }

    #[test]
    fn error_body_serialises() {
        let body = ErrorBody::new("bad", "invalid_request_error");
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("invalid_request_error"));
    }
}
