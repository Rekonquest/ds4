// DS4 (DwarfStar) — Anthropic-compatible request/response types.
//
// Targets the Messages API wire format (`POST /v1/messages`).
// Streamed SSE events use the Anthropic `event: <type>\ndata: {...}`
// shape rather than OpenAI's flat `data: {...}`.

// Many of these struct fields / enum variants are part of the wire
// schema and are read by external clients (the `ds4-server`
// integration tests in the lib target). They are intentionally not
// consumed inside `main.rs`, so silence dead-code for the module.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicErrorBody {
    pub r#type: &'static str,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicErrorDetail {
    pub r#type: &'static str,
    pub message: String,
}

// SSE event payloads.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessagesResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaBody,
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicErrorDetail },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MessageDeltaBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_request_deserialises() {
        let body = r#"{
            "model":"claude-x",
            "max_tokens":256,
            "messages":[{"role":"user","content":"hi"}]
        }"#;
        let r: MessagesRequest = serde_json::from_str(body).unwrap();
        assert_eq!(r.model, "claude-x");
        assert_eq!(r.max_tokens, 256);
        assert_eq!(r.messages.len(), 1);
    }

    #[test]
    fn content_block_serialises_text() {
        let block = ContentBlock::Text {
            text: "hi".to_string(),
        };
        let s = serde_json::to_string(&block).unwrap();
        assert!(s.contains("\"type\":\"text\""));
    }

    #[test]
    fn content_block_serialises_tool_use() {
        let block = ContentBlock::ToolUse {
            id: "id_1".to_string(),
            name: "f".to_string(),
            input: serde_json::json!({"x": 1}),
        };
        let s = serde_json::to_string(&block).unwrap();
        assert!(s.contains("\"type\":\"tool_use\""));
        assert!(s.contains("\"name\":\"f\""));
    }
}
