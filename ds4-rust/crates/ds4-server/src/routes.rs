// DS4 (DwarfStar) -- HTTP route table + hyper service implementation.
//
// Replaces the giant `ds4_server.c` request dispatcher. Endpoints:
//   - GET  /v1/models
//   - POST /v1/chat/completions
//   - POST /v1/completions
//   - POST /v1/responses
//   - POST /v1/messages           (Anthropic-compatible)
//   - GET  /healthz
//   - GET  /v1/ds4/info           (dwarfstar-specific diagnostics)
//
// Inference routes use EnginePool when a runtime model is loaded.
// Requests without a loaded model fail closed with JSON or SSE errors.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::CONTENT_LENGTH;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use serde::Serialize;
use serde_json::json;

use crate::anthropic::{
    AnthropicErrorBody, ContentBlock, MessagesRequest, MessagesResponse, StreamEvent,
};
use crate::engine_pool::{EnginePool, JobResult, SamplingParams, MAX_GENERATION_TOKENS};
use crate::openai::{
    AssistantMessage, ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, CompletionChoice, CompletionRequest, CompletionResponse,
    ErrorBody, ModelCard, ModelList, ResponsesContent, ResponsesOutput, ResponsesRequest,
    ResponsesResponse, ToolCallFunctionOut, ToolCallOut, Usage,
};
use crate::streaming::SseWriter;

const NOT_IMPLEMENTED_MSG: &str = "the selected model/backend has no loaded runtime model";
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Server-side context shared across requests.
#[derive(Clone)]
pub struct ServerState {
    pub pool: EnginePool,
    pub model_id: String,
}

impl ServerState {
    pub fn new(pool: EnginePool, model_id: String) -> Self {
        Self { pool, model_id }
    }
}

/// Top-level request dispatcher. Returns a `Response<Full<Bytes>>`
/// for hyper 1.x compatibility.
pub async fn handle(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let res = dispatch(method, path, req, state).await;
    Ok(res)
}

async fn dispatch(
    method: Method,
    path: String,
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Response<Full<Bytes>> {
    let route = format!("{method} {path}");
    log::info!("{route}");
    if method == Method::OPTIONS {
        return options_response();
    }
    match (method, path.as_str()) {
        (Method::GET, "/healthz") => healthz(&state),
        (Method::GET, "/v1/models") => list_models(&state),
        (Method::POST, "/v1/chat/completions") => chat_completions(req, state).await,
        (Method::POST, "/v1/completions") => completions(req, state).await,
        (Method::POST, "/v1/responses") => responses(req, state).await,
        (Method::POST, "/v1/messages") => anthropic_messages(req, state).await,
        (Method::GET, "/v1/ds4/info") => ds4_info(&state),
        _ => not_found(&route),
    }
}

fn healthz(state: &Arc<ServerState>) -> Response<Full<Bytes>> {
    if state.pool.is_engine_ready() {
        plain_text(StatusCode::OK, "ok")
    } else {
        plain_text(StatusCode::SERVICE_UNAVAILABLE, "engine_not_ready")
    }
}
fn list_models(state: &Arc<ServerState>) -> Response<Full<Bytes>> {
    let now = now_secs();
    let body = ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.clone(),
            object: "model",
            created: now,
            owned_by: "dwarfstar".to_string(),
        }],
    };
    json_response(StatusCode::OK, &body)
}

fn ds4_info(state: &Arc<ServerState>) -> Response<Full<Bytes>> {
    let body = json!({
        "model_id": state.model_id,
        "engine_ready": state.pool.is_engine_ready(),
        "engine_pending": state.pool.is_engine_pending(),
        "queued_jobs": state.pool.queue_len(),
        "enqueued_jobs": state.pool.enqueued_count(),
        "ctx": state.pool.config().ctx,
    });
    json_response(StatusCode::OK, &body)
}
enum BodyReadError {
    InvalidLength(String),
    TooLarge,
    Read(String),
}

async fn read_limited_body(req: Request<Incoming>) -> Result<Bytes, BodyReadError> {
    if let Some(value) = req.headers().get(CONTENT_LENGTH) {
        let raw = value.to_str().map_err(|e| {
            BodyReadError::InvalidLength(format!("invalid content-length header: {e}"))
        })?;
        let len = raw.parse::<usize>().map_err(|e| {
            BodyReadError::InvalidLength(format!("invalid content-length header: {e}"))
        })?;
        if len > MAX_BODY_BYTES {
            return Err(BodyReadError::TooLarge);
        }
    }

    let mut body = req.into_body();
    let mut out = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|e| BodyReadError::Read(e.to_string()))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        if out.len().saturating_add(data.len()) > MAX_BODY_BYTES {
            return Err(BodyReadError::TooLarge);
        }
        out.extend_from_slice(&data);
    }
    Ok(Bytes::from(out))
}

fn openai_body_read_error(err: BodyReadError) -> Response<Full<Bytes>> {
    match err {
        BodyReadError::InvalidLength(msg) | BodyReadError::Read(msg) => bad_request(msg),
        BodyReadError::TooLarge => {
            payload_too_large(format!("request body exceeds {} bytes", MAX_BODY_BYTES))
        }
    }
}

fn anthropic_body_read_error(err: BodyReadError) -> Response<Full<Bytes>> {
    match err {
        BodyReadError::InvalidLength(msg) | BodyReadError::Read(msg) => anthropic_bad_request(msg),
        BodyReadError::TooLarge => {
            anthropic_payload_too_large(format!("request body exceeds {} bytes", MAX_BODY_BYTES))
        }
    }
}

async fn chat_completions(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Response<Full<Bytes>> {
    let body_bytes = match read_limited_body(req).await {
        Ok(bytes) => bytes,
        Err(e) => return openai_body_read_error(e),
    };
    let parsed: Result<ChatCompletionRequest, _> = serde_json::from_slice(&body_bytes);
    let req = match parsed {
        Ok(r) => r,
        Err(e) => return bad_request(format!("invalid chat-completion request: {e}")),
    };

    let message_texts: Vec<(String, String)> = match req
        .messages
        .iter()
        .map(|m| {
            Ok((
                m.role.clone(),
                match m.content.as_ref() {
                    Some(content) => content_value_to_text(content)?,
                    None => String::new(),
                },
            ))
        })
        .collect::<Result<_, String>>()
    {
        Ok(messages) => messages,
        Err(e) => return bad_request(e),
    };
    let message_refs: Vec<(&str, &str)> = message_texts
        .iter()
        .map(|(role, content)| (role.as_str(), content.as_str()))
        .collect();
    let prompt_tokens = match state.pool.encode_chat_messages(&message_refs) {
        Ok(tokens) => tokens,
        Err(e) => return engine_error_response(e),
    };

    let stream = req.stream.unwrap_or(false);
    let sampling = sampling_from_openai(&req);
    let max_tokens = req.max_tokens.unwrap_or(256);
    if let Err(e) = validate_generation_tokens("max_tokens", max_tokens) {
        return bad_request(e);
    }
    if let Err(e) = validate_prompt_budget(&state, prompt_tokens.len()) {
        return bad_request(e);
    }

    if stream {
        return chat_completions_stream(&state, &req, prompt_tokens, max_tokens, sampling).await;
    }
    chat_completions_unary(&state, &req, prompt_tokens, max_tokens, sampling).await
}

async fn chat_completions_unary(
    state: &Arc<ServerState>,
    req: &ChatCompletionRequest,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Response<Full<Bytes>> {
    let (_id, rx) = state.pool.submit(prompt_tokens, max_tokens, sampling);
    match rx.await {
        Ok(Ok(result)) => build_chat_response(req, state, result),
        Ok(Err(e)) => engine_error_response(e),
        Err(_) => engine_error_response_text("engine worker dropped the response"),
    }
}

async fn chat_completions_stream(
    state: &Arc<ServerState>,
    req: &ChatCompletionRequest,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Response<Full<Bytes>> {
    let (_id, rx) = state.pool.submit(prompt_tokens, max_tokens, sampling);
    let mut w = SseWriter::new();

    match rx.await {
        Ok(Ok(result)) => {
            let id = format!("chatcmpl-{}", result.job_id);
            let created = now_secs();
            let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
            let text = match decode_tokens(state, &result.completion_tokens) {
                Ok(text) => text,
                Err(e) => {
                    let body = ErrorBody::new(e.message, "engine_error");
                    if let Ok(s) = serde_json::to_string(&body) {
                        w.push_data_json(&s);
                    }
                    w.push_done();
                    return sse_response(w.take());
                }
            };
            let usage = Usage {
                prompt_tokens: result.usage.prompt_tokens,
                completion_tokens: result.usage.completion_tokens,
                total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
            };
            let (content, tool_calls, finish_reason) = openai_message_content_and_tools(&text);
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant"),
                        content,
                        tool_calls,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            if let Ok(s) = serde_json::to_string(&chunk) {
                w.push_data_json(&s);
            }

            let final_chunk = ChatCompletionChunk {
                id,
                object: "chat.completion.chunk",
                created,
                model,
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta::default(),
                    finish_reason: Some(finish_reason),
                }],
                usage: Some(usage),
            };
            if let Ok(s) = serde_json::to_string(&final_chunk) {
                w.push_data_json(&s);
            }
        }
        Ok(Err(e)) => {
            let kind = if e.kind == ds4_types::Ds4ErrorKind::NotImplemented {
                "engine_not_implemented"
            } else {
                "engine_error"
            };
            let msg = if e.kind == ds4_types::Ds4ErrorKind::NotImplemented {
                NOT_IMPLEMENTED_MSG.to_string()
            } else {
                e.message
            };
            let body = ErrorBody::new(msg, kind);
            if let Ok(s) = serde_json::to_string(&body) {
                w.push_data_json(&s);
            }
        }
        Err(_) => {
            let body = ErrorBody::new("engine worker dropped the response", "engine_error");
            if let Ok(s) = serde_json::to_string(&body) {
                w.push_data_json(&s);
            }
        }
    }

    w.push_done();
    sse_response(w.take())
}
fn build_chat_response(
    req: &ChatCompletionRequest,
    state: &Arc<ServerState>,
    result: JobResult,
) -> Response<Full<Bytes>> {
    let id = format!("chatcmpl-{}", result.job_id);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let usage = Usage {
        prompt_tokens: result.usage.prompt_tokens,
        completion_tokens: result.usage.completion_tokens,
        total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
    };
    let text = match decode_tokens(state, &result.completion_tokens) {
        Ok(text) => text,
        Err(e) => return engine_error_response(e),
    };
    let (content, tool_calls, finish_reason) = openai_message_content_and_tools(&text);
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now_secs(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
                refusal: None,
                tool_calls,
            },
            logprobs: None,
            finish_reason: Some(finish_reason),
        }],
        usage,
        system_fingerprint: None,
    };
    json_response(StatusCode::OK, &resp)
}

async fn completions(req: Request<Incoming>, state: Arc<ServerState>) -> Response<Full<Bytes>> {
    let body_bytes = match read_limited_body(req).await {
        Ok(bytes) => bytes,
        Err(e) => return openai_body_read_error(e),
    };
    let parsed: Result<CompletionRequest, _> = serde_json::from_slice(&body_bytes);
    let req = match parsed {
        Ok(r) => r,
        Err(e) => return bad_request(format!("invalid completion request: {e}")),
    };
    let stream = req.stream.unwrap_or(false);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let max_tokens = req.max_tokens.unwrap_or(256);
    if let Err(e) = validate_generation_tokens("max_tokens", max_tokens) {
        return bad_request(e);
    }
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        min_p: 0.0,
        seed: req.seed,
    };
    let prompt_text = match req.prompt {
        serde_json::Value::String(s) => s,
        serde_json::Value::Array(arr) => {
            let mut parts = Vec::with_capacity(arr.len());
            for value in arr {
                let Some(text) = value.as_str() else {
                    return bad_request("`prompt` array items must be strings".to_string());
                };
                parts.push(text.to_string());
            }
            parts.join("\n")
        }
        _ => return bad_request("`prompt` must be string or string[]".to_string()),
    };
    let prompt_tokens = match state.pool.tokenize_text(&prompt_text) {
        Ok(tokens) => tokens,
        Err(e) => return engine_error_response(e),
    };
    if let Err(e) = validate_prompt_budget(&state, prompt_tokens.len()) {
        return bad_request(e);
    }
    let (_id, rx) = state.pool.submit(prompt_tokens, max_tokens, sampling);
    match rx.await {
        Ok(Ok(result)) => {
            if stream {
                completion_stream_response(&state, model, result)
            } else {
                match completion_response(&state, model, result) {
                    Ok(resp) => json_response(StatusCode::OK, &resp),
                    Err(e) => engine_error_response(e),
                }
            }
        }
        Ok(Err(e)) if stream => openai_stream_error_response(e),
        Ok(Err(e)) => engine_error_response(e),
        Err(_) if stream => openai_stream_error_response_text("engine worker dropped the response"),
        Err(_) => engine_error_response_text("engine worker dropped the response"),
    }
}

async fn responses(req: Request<Incoming>, state: Arc<ServerState>) -> Response<Full<Bytes>> {
    let body_bytes = match read_limited_body(req).await {
        Ok(bytes) => bytes,
        Err(e) => return openai_body_read_error(e),
    };
    let parsed: Result<ResponsesRequest, _> = serde_json::from_slice(&body_bytes);
    let req = match parsed {
        Ok(r) => r,
        Err(e) => return bad_request(format!("invalid responses request: {e}")),
    };
    let stream = req.stream.unwrap_or(false);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let max_tokens = req.max_output_tokens.unwrap_or(256);
    if let Err(e) = validate_generation_tokens("max_output_tokens", max_tokens) {
        return bad_request(e);
    }
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        min_p: 0.0,
        seed: None,
    };
    let prompt_text = match responses_prompt_text(&req) {
        Ok(text) => text,
        Err(e) => return bad_request(e),
    };
    let prompt_tokens = match state.pool.tokenize_text(&prompt_text) {
        Ok(tokens) => tokens,
        Err(e) => return engine_error_response(e),
    };
    if let Err(e) = validate_prompt_budget(&state, prompt_tokens.len()) {
        return bad_request(e);
    }
    let (_id, rx) = state.pool.submit(prompt_tokens, max_tokens, sampling);
    match rx.await {
        Ok(Ok(result)) => {
            if stream {
                responses_stream_response(&state, model, result)
            } else {
                match responses_response(&state, model, result) {
                    Ok(resp) => json_response(StatusCode::OK, &resp),
                    Err(e) => engine_error_response(e),
                }
            }
        }
        Ok(Err(e)) if stream => openai_stream_error_response(e),
        Ok(Err(e)) => engine_error_response(e),
        Err(_) if stream => openai_stream_error_response_text("engine worker dropped the response"),
        Err(_) => engine_error_response_text("engine worker dropped the response"),
    }
}

fn completion_response(
    state: &Arc<ServerState>,
    model: String,
    result: JobResult,
) -> ds4_types::Ds4Result<CompletionResponse> {
    let text = decode_tokens(state, &result.completion_tokens)?;
    Ok(CompletionResponse {
        id: format!("cmpl-{}", result.job_id),
        object: "text_completion",
        created: now_secs(),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: Some("stop".to_string()),
        }],
        usage: usage_from_result(&result),
    })
}

fn completion_stream_response(
    state: &Arc<ServerState>,
    model: String,
    result: JobResult,
) -> Response<Full<Bytes>> {
    let mut w = SseWriter::new();
    let text = match decode_tokens(state, &result.completion_tokens) {
        Ok(text) => text,
        Err(e) => return openai_stream_error_response(e),
    };
    let id = format!("cmpl-{}", result.job_id);
    let created = now_secs();
    let text_chunk = CompletionResponse {
        id: id.clone(),
        object: "text_completion",
        created,
        model: model.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: None,
        }],
        usage: usage_from_result(&result),
    };
    if let Ok(s) = serde_json::to_string(&text_chunk) {
        w.push_data_json(&s);
    }
    let stop_chunk = CompletionResponse {
        id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            text: String::new(),
            index: 0,
            logprobs: None,
            finish_reason: Some("stop".to_string()),
        }],
        usage: usage_from_result(&result),
    };
    if let Ok(s) = serde_json::to_string(&stop_chunk) {
        w.push_data_json(&s);
    }
    w.push_done();
    sse_response(w.take())
}

fn responses_response(
    state: &Arc<ServerState>,
    model: String,
    result: JobResult,
) -> ds4_types::Ds4Result<ResponsesResponse> {
    let text = decode_tokens(state, &result.completion_tokens)?;
    Ok(ResponsesResponse {
        id: format!("resp-{}", crate::id::new_id()),
        object: "response",
        created: now_secs(),
        model,
        output: responses_output_from_text(&text),
        usage: usage_from_result(&result),
    })
}

fn responses_stream_response(
    state: &Arc<ServerState>,
    model: String,
    result: JobResult,
) -> Response<Full<Bytes>> {
    let mut w = SseWriter::new();
    let resp = match responses_response(state, model, result) {
        Ok(resp) => resp,
        Err(e) => return openai_stream_error_response(e),
    };
    if let Ok(s) = serde_json::to_string(&serde_json::json!({
        "type": "response.created",
        "response": &resp,
    })) {
        w.push_event("response.created", &s);
    }
    let text_event = match resp.output.first() {
        Some(ResponsesOutput::Message { id, content, .. }) => content
            .first()
            .map(|ResponsesContent::OutputText { text }| (id.as_str(), text.as_str())),
        _ => None,
    };
    if let Some((item_id, text)) = text_event {
        if let Ok(s) = serde_json::to_string(&serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        })) {
            w.push_event("response.output_text.delta", &s);
        }
        if let Ok(s) = serde_json::to_string(&serde_json::json!({
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        })) {
            w.push_event("response.output_text.done", &s);
        }
    }
    if let Ok(s) = serde_json::to_string(&serde_json::json!({
        "type": "response.completed",
        "response": &resp,
    })) {
        w.push_event("response.completed", &s);
    }
    w.push_done();
    sse_response(w.take())
}

async fn anthropic_messages(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Response<Full<Bytes>> {
    let body_bytes = match read_limited_body(req).await {
        Ok(bytes) => bytes,
        Err(e) => return anthropic_body_read_error(e),
    };
    let parsed: Result<MessagesRequest, _> = serde_json::from_slice(&body_bytes);
    let req = match parsed {
        Ok(r) => r,
        Err(e) => return anthropic_bad_request(format!("invalid messages request: {e}")),
    };
    if let Err(e) = validate_generation_tokens("max_tokens", req.max_tokens) {
        return anthropic_bad_request(e);
    }
    let mut message_texts: Vec<(String, String)> = Vec::new();
    if let Some(system) = &req.system {
        let text = match content_value_to_text(system) {
            Ok(text) => text,
            Err(e) => return anthropic_bad_request(e),
        };
        if !text.is_empty() {
            message_texts.push(("system".to_string(), text));
        }
    }
    for message in &req.messages {
        let text = match content_value_to_text(&message.content) {
            Ok(text) => text,
            Err(e) => return anthropic_bad_request(e),
        };
        message_texts.push((message.role.clone(), text));
    }
    let message_refs: Vec<(&str, &str)> = message_texts
        .iter()
        .map(|(role, content)| (role.as_str(), content.as_str()))
        .collect();
    let prompt_tokens = match state.pool.encode_chat_messages(&message_refs) {
        Ok(tokens) => tokens,
        Err(e) => return anthropic_error_response(e),
    };
    if let Err(e) = validate_prompt_budget(&state, prompt_tokens.len()) {
        return anthropic_bad_request(e);
    }
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.9),
        min_p: 0.0,
        seed: None,
    };
    let stream = req.stream.unwrap_or(false);
    if stream {
        return anthropic_stream(&req, &state, prompt_tokens, req.max_tokens, sampling).await;
    }
    let (_id, rx) = state.pool.submit(prompt_tokens, req.max_tokens, sampling);
    match rx.await {
        Ok(Ok(result)) => {
            let text = match decode_tokens(&state, &result.completion_tokens) {
                Ok(text) => text,
                Err(e) => return anthropic_error_response(e),
            };
            let content = anthropic_content_from_text(&text);
            let resp = MessagesResponse {
                id: format!("msg_{}", crate::id::new_id()),
                r#type: "message",
                role: "assistant",
                model: req.model.clone(),
                content,
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
                usage: crate::anthropic::AnthropicUsage {
                    input_tokens: result.usage.prompt_tokens,
                    output_tokens: result.usage.completion_tokens,
                },
            };
            json_response(StatusCode::OK, &resp)
        }
        Ok(Err(e)) => anthropic_error_response(e),
        Err(_) => anthropic_error_response_text("engine worker dropped the response"),
    }
}

async fn anthropic_stream(
    req: &MessagesRequest,
    state: &Arc<ServerState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Response<Full<Bytes>> {
    let (_id, rx) = state.pool.submit(prompt_tokens, max_tokens, sampling);
    let mut w = SseWriter::new();

    match rx.await {
        Ok(Ok(result)) => {
            let text = match decode_tokens(state, &result.completion_tokens) {
                Ok(text) => text,
                Err(e) => {
                    let detail = crate::anthropic::AnthropicErrorDetail {
                        r#type: "api_error",
                        message: e.message,
                    };
                    if let Ok(payload) =
                        serde_json::to_string(&StreamEvent::Error { error: detail })
                    {
                        w.push_event("error", &payload);
                    }
                    return sse_response(w.take());
                }
            };
            let message = MessagesResponse {
                id: format!("msg_{}", crate::id::new_id()),
                r#type: "message",
                role: "assistant",
                model: req.model.clone(),
                content: Vec::new(),
                stop_reason: None,
                stop_sequence: None,
                usage: crate::anthropic::AnthropicUsage {
                    input_tokens: result.usage.prompt_tokens,
                    output_tokens: 0,
                },
            };
            if let Ok(payload) = serde_json::to_string(&StreamEvent::MessageStart { message }) {
                w.push_event("message_start", &payload);
            }
            if let Ok(payload) = serde_json::to_string(&StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Text {
                    text: String::new(),
                },
            }) {
                w.push_event("content_block_start", &payload);
            }
            if !text.is_empty() {
                let delta = crate::anthropic::ContentDelta::TextDelta { text };
                if let Ok(payload) =
                    serde_json::to_string(&StreamEvent::ContentBlockDelta { index: 0, delta })
                {
                    w.push_event("content_block_delta", &payload);
                }
            }
            if let Ok(payload) = serde_json::to_string(&StreamEvent::ContentBlockStop { index: 0 })
            {
                w.push_event("content_block_stop", &payload);
            }
            let stop_delta = crate::anthropic::StreamEvent::MessageDelta {
                delta: crate::anthropic::MessageDeltaBody {
                    stop_reason: Some("end_turn".to_string()),
                    stop_sequence: None,
                },
                usage: Some(crate::anthropic::AnthropicUsage {
                    input_tokens: result.usage.prompt_tokens,
                    output_tokens: result.usage.completion_tokens,
                }),
            };
            if let Ok(payload) = serde_json::to_string(&stop_delta) {
                w.push_event("message_delta", &payload);
            }
            if let Ok(payload) = serde_json::to_string(&StreamEvent::MessageStop) {
                w.push_event("message_stop", &payload);
            }
        }
        Ok(Err(e)) => {
            let detail = crate::anthropic::AnthropicErrorDetail {
                r#type: match e.kind {
                    ds4_types::Ds4ErrorKind::NotImplemented => "not_implemented_error",
                    ds4_types::Ds4ErrorKind::InvalidArgument => "invalid_request_error",
                    _ => "api_error",
                },
                message: e.message,
            };
            if let Ok(payload) = serde_json::to_string(&StreamEvent::Error { error: detail }) {
                w.push_event("error", &payload);
            }
        }
        Err(_) => {
            let detail = crate::anthropic::AnthropicErrorDetail {
                r#type: "api_error",
                message: "engine worker dropped the response".to_string(),
            };
            if let Ok(payload) = serde_json::to_string(&StreamEvent::Error { error: detail }) {
                w.push_event("error", &payload);
            }
        }
    }

    sse_response(w.take())
}
fn content_value_to_text(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(items) => items
            .iter()
            .map(content_part_to_text)
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.join("\n")),
        serde_json::Value::Object(map) => content_object_to_text(map),
        _ => Err("message content must be text or a supported text content part".to_string()),
    }
}

fn content_part_to_text(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Object(map) => content_object_to_text(map),
        _ => Err("content array items must be text objects".to_string()),
    }
}

fn content_object_to_text(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    if let Some(kind) = map.get("type").and_then(serde_json::Value::as_str) {
        match kind {
            "text" | "input_text" => {
                let Some(text) = map.get("text").and_then(serde_json::Value::as_str) else {
                    return Err(format!("content part `{kind}` requires string `text`"));
                };
                return Ok(text.to_string());
            }
            other => {
                return Err(format!(
                    "unsupported content part type `{other}`; only text parts are accepted"
                ));
            }
        }
    }
    if let Some(text) = map.get("text").and_then(serde_json::Value::as_str) {
        return Ok(text.to_string());
    }
    Err("content object must contain string `text`".to_string())
}

fn responses_prompt_text(req: &ResponsesRequest) -> Result<String, String> {
    let mut parts = Vec::new();
    if let Some(instructions) = &req.instructions {
        if !instructions.is_empty() {
            parts.push(instructions.clone());
        }
    }
    if let Some(input) = &req.input {
        let input_text = responses_input_value_to_text(input)?;
        if !input_text.is_empty() {
            parts.push(input_text);
        }
    }
    Ok(parts.join("\n"))
}

fn responses_input_value_to_text(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(items) => items
            .iter()
            .map(responses_input_item_to_text)
            .collect::<Result<Vec<_>, _>>()
            .map(|items| {
                items
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
        serde_json::Value::Object(map) => responses_input_object_to_text(map),
        _ => Err("`input` must be a string, object, array, or null".to_string()),
    }
}

fn responses_input_item_to_text(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Object(map) => responses_input_object_to_text(map),
        _ => Err("`input` array items must be strings or objects".to_string()),
    }
}

fn responses_input_object_to_text(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    if let Some(content) = map.get("content") {
        return content_value_to_text(content);
    }
    if let Some(text) = map.get("text").and_then(serde_json::Value::as_str) {
        return Ok(text.to_string());
    }
    Err("`input` object must contain `content` or `text`".to_string())
}

fn sampling_from_openai(req: &ChatCompletionRequest) -> SamplingParams {
    SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        min_p: 0.0,
        seed: req.seed,
    }
}

fn validate_generation_tokens(field: &str, value: usize) -> Result<(), String> {
    if value == 0 {
        return Err(format!("`{field}` must be greater than zero"));
    }
    if value > MAX_GENERATION_TOKENS {
        return Err(format!(
            "`{field}` {} exceeds server limit {}",
            value, MAX_GENERATION_TOKENS
        ));
    }
    Ok(())
}

fn validate_prompt_budget(state: &Arc<ServerState>, prompt_tokens: usize) -> Result<(), String> {
    let ctx = state.pool.config().ctx;
    if prompt_tokens >= ctx {
        return Err(format!(
            "prompt token count {} leaves no room in context window {}",
            prompt_tokens, ctx
        ));
    }
    Ok(())
}

fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let bytes = serde_json::to_vec(body)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialise failed: {e}\"}}").into_bytes());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap()
}

fn plain_text(status: StatusCode, body: &'static str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .header("access-control-allow-origin", "*")
        .body(Full::new(Bytes::from(body.as_bytes().to_vec())))
        .unwrap()
}

fn options_response() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header("access-control-allow-origin", "*")
        .header("access-control-allow-methods", "GET,POST,OPTIONS")
        .header(
            "access-control-allow-headers",
            "authorization,content-type,x-api-key,anthropic-version,anthropic-beta",
        )
        .header("access-control-max-age", "86400")
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn sse_response(body: Vec<u8>) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .header("access-control-allow-origin", "*")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

fn usage_from_result(result: &JobResult) -> Usage {
    Usage {
        prompt_tokens: result.usage.prompt_tokens,
        completion_tokens: result.usage.completion_tokens,
        total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
    }
}

fn decode_tokens(state: &Arc<ServerState>, tokens: &[u32]) -> ds4_types::Ds4Result<String> {
    state.pool.detokenize_tokens(tokens)
}

fn openai_message_content_and_tools(text: &str) -> (Option<String>, Vec<ToolCallOut>, String) {
    match parse_dsml_tool_calls(text) {
        Some(calls) => (
            None,
            calls
                .iter()
                .enumerate()
                .map(|(idx, call)| ToolCallOut {
                    id: format!("call_{}", idx + 1),
                    r#type: "function",
                    function: ToolCallFunctionOut {
                        name: call.name.clone(),
                        arguments: dsml_arguments_json(call),
                    },
                })
                .collect(),
            "tool_calls".to_string(),
        ),
        None => (Some(text.to_string()), Vec::new(), "stop".to_string()),
    }
}

fn responses_output_from_text(text: &str) -> Vec<ResponsesOutput> {
    if let Some(calls) = parse_dsml_tool_calls(text) {
        return calls
            .iter()
            .enumerate()
            .map(|(idx, call)| ResponsesOutput::FunctionCall {
                id: format!("fc_{}", crate::id::new_id()),
                call_id: format!("call_{}", idx + 1),
                name: call.name.clone(),
                arguments: dsml_arguments_json(call),
            })
            .collect();
    }
    vec![ResponsesOutput::Message {
        id: format!("msg-{}", crate::id::new_id()),
        role: "assistant",
        content: vec![ResponsesContent::OutputText {
            text: text.to_string(),
        }],
    }]
}

fn anthropic_content_from_text(text: &str) -> Vec<ContentBlock> {
    if let Some(calls) = parse_dsml_tool_calls(text) {
        return calls
            .iter()
            .enumerate()
            .map(|(idx, call)| ContentBlock::ToolUse {
                id: format!("toolu_{}", idx + 1),
                name: call.name.clone(),
                input: dsml_arguments_value(call),
            })
            .collect();
    }
    vec![ContentBlock::Text {
        text: text.to_string(),
    }]
}

fn parse_dsml_tool_calls(text: &str) -> Option<Vec<crate::dsml::DsmlToolCall>> {
    if !crate::dsml::contains_tool_calls(text) {
        return None;
    }
    match crate::dsml::parse(text) {
        Ok(calls) if !calls.is_empty() => Some(calls),
        _ => None,
    }
}

fn dsml_arguments_json(call: &crate::dsml::DsmlToolCall) -> String {
    dsml_arguments_value(call).to_string()
}

fn dsml_arguments_value(call: &crate::dsml::DsmlToolCall) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for param in &call.parameters {
        let value = if param.is_string {
            serde_json::Value::String(param.value.clone())
        } else {
            serde_json::from_str(&param.value)
                .unwrap_or_else(|_| serde_json::Value::String(param.value.clone()))
        };
        map.insert(param.name.clone(), value);
    }
    serde_json::Value::Object(map)
}

fn not_found(route: &str) -> Response<Full<Bytes>> {
    let body = ErrorBody::new(format!("route not found: {route}"), "not_found");
    json_response(StatusCode::NOT_FOUND, &body)
}

fn bad_request(msg: String) -> Response<Full<Bytes>> {
    let body = ErrorBody::new(msg, "invalid_request_error");
    json_response(StatusCode::BAD_REQUEST, &body)
}

fn payload_too_large(msg: String) -> Response<Full<Bytes>> {
    let body = ErrorBody::new(msg, "invalid_request_error");
    json_response(StatusCode::PAYLOAD_TOO_LARGE, &body)
}

fn anthropic_bad_request(msg: String) -> Response<Full<Bytes>> {
    let body = AnthropicErrorBody {
        r#type: "error",
        error: crate::anthropic::AnthropicErrorDetail {
            r#type: "invalid_request_error",
            message: msg,
        },
    };
    json_response(StatusCode::BAD_REQUEST, &body)
}

fn anthropic_payload_too_large(msg: String) -> Response<Full<Bytes>> {
    let body = AnthropicErrorBody {
        r#type: "error",
        error: crate::anthropic::AnthropicErrorDetail {
            r#type: "invalid_request_error",
            message: msg,
        },
    };
    json_response(StatusCode::PAYLOAD_TOO_LARGE, &body)
}

fn engine_error_response(err: ds4_types::Ds4Error) -> Response<Full<Bytes>> {
    match err.kind {
        ds4_types::Ds4ErrorKind::InvalidArgument => {
            let body = ErrorBody::new(err.message, "invalid_request_error");
            json_response(StatusCode::BAD_REQUEST, &body)
        }
        ds4_types::Ds4ErrorKind::NotImplemented => {
            let body = ErrorBody::new(NOT_IMPLEMENTED_MSG, "engine_not_implemented");
            json_response(StatusCode::NOT_IMPLEMENTED, &body)
        }
        _ => engine_error_response_text(&err.message),
    }
}

fn engine_error_response_text(msg: &str) -> Response<Full<Bytes>> {
    let body = ErrorBody::new(msg.to_string(), "engine_error");
    json_response(StatusCode::INTERNAL_SERVER_ERROR, &body)
}

fn openai_stream_error_response(err: ds4_types::Ds4Error) -> Response<Full<Bytes>> {
    match err.kind {
        ds4_types::Ds4ErrorKind::InvalidArgument => {
            openai_stream_error_response_body(ErrorBody::new(err.message, "invalid_request_error"))
        }
        ds4_types::Ds4ErrorKind::NotImplemented => openai_stream_error_response_body(
            ErrorBody::new(NOT_IMPLEMENTED_MSG, "engine_not_implemented"),
        ),
        _ => openai_stream_error_response_text(&err.message),
    }
}

fn openai_stream_error_response_text(msg: &str) -> Response<Full<Bytes>> {
    openai_stream_error_response_body(ErrorBody::new(msg.to_string(), "engine_error"))
}

fn openai_stream_error_response_body(body: ErrorBody) -> Response<Full<Bytes>> {
    let mut w = SseWriter::new();
    if let Ok(s) = serde_json::to_string(&body) {
        w.push_data_json(&s);
    }
    w.push_done();
    sse_response(w.take())
}

fn anthropic_error_response(err: ds4_types::Ds4Error) -> Response<Full<Bytes>> {
    let status = match err.kind {
        ds4_types::Ds4ErrorKind::InvalidArgument => StatusCode::BAD_REQUEST,
        ds4_types::Ds4ErrorKind::NotImplemented => StatusCode::NOT_IMPLEMENTED,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let detail = crate::anthropic::AnthropicErrorDetail {
        r#type: match err.kind {
            ds4_types::Ds4ErrorKind::NotImplemented => "not_implemented_error",
            ds4_types::Ds4ErrorKind::InvalidArgument => "invalid_request_error",
            _ => "api_error",
        },
        message: err.message,
    };
    let body = AnthropicErrorBody {
        r#type: "error",
        error: detail,
    };
    json_response(status, &body)
}

fn anthropic_error_response_text(msg: &str) -> Response<Full<Bytes>> {
    let body = AnthropicErrorBody {
        r#type: "error",
        error: crate::anthropic::AnthropicErrorDetail {
            r#type: "api_error",
            message: msg.to_string(),
        },
    };
    json_response(StatusCode::INTERNAL_SERVER_ERROR, &body)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Public entry point used by `main.rs` to drive the listener.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<ServerState>,
) -> anyhow::Result<()> {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                log::error!("accept error: {e}");
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            let io = hyper_util::rt::TokioIo::new(stream);
            let svc = service_fn(move |req| {
                let state = state.clone();
                async move { handle(req, state).await }
            });
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                log::warn!("connection error: {e}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatMessage;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SYNTH_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn synthetic_state() -> Arc<ServerState> {
        let id = SYNTH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("ds4-server-route-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("synth.gguf");
        ds4_core::engine::Ds4Engine::write_synthetic_gguf(&model).unwrap();
        let pool = EnginePool::open(crate::engine_pool::PoolConfig {
            model,
            mtp: None,
            ctx: 64,
            prefill_chunk: 8,
            n_threads: 1,
        })
        .unwrap();
        Arc::new(ServerState::new(pool, "ds4-test".to_string()))
    }

    fn pending_state() -> Arc<ServerState> {
        let id = SYNTH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let missing = std::env::temp_dir()
            .join(format!(
                "ds4-server-route-test-missing-{}-{id}",
                std::process::id()
            ))
            .join("missing.gguf");
        let pool = EnginePool::open(crate::engine_pool::PoolConfig {
            model: missing,
            mtp: None,
            ctx: 64,
            prefill_chunk: 8,
            n_threads: 1,
        })
        .unwrap();
        Arc::new(ServerState::new(pool, "ds4-test".to_string()))
    }

    fn minimal_chat_request(stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some("ds4-test".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("hi".to_string())),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            stream: Some(stream),
            max_tokens: Some(3),
            temperature: Some(0.0),
            ..ChatCompletionRequest::default()
        }
    }

    #[test]
    fn json_response_has_cors_header() {
        let r = json_response(StatusCode::OK, &json!({"ok": true}));
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(r.headers().get("access-control-allow-origin").unwrap(), "*");
    }

    #[test]
    fn plain_text_response_has_cors_header() {
        let r = plain_text(StatusCode::OK, "ok");
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(r.headers().get("access-control-allow-origin").unwrap(), "*");
    }

    #[test]
    fn options_response_has_preflight_headers() {
        let r = options_response();
        assert_eq!(r.status(), StatusCode::NO_CONTENT);
        assert_eq!(r.headers().get("access-control-allow-origin").unwrap(), "*");
        assert_eq!(
            r.headers().get("access-control-allow-methods").unwrap(),
            "GET,POST,OPTIONS"
        );
        assert!(r
            .headers()
            .get("access-control-allow-headers")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("content-type"));
    }

    #[test]
    fn healthz_reflects_engine_readiness() {
        let ready = healthz(&synthetic_state());
        assert_eq!(ready.status(), StatusCode::OK);
        let pending = healthz(&pending_state());
        assert_eq!(pending.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn ds4_info_reports_unavailable_engine_as_not_pending() {
        let state = pending_state();
        let r = ds4_info(&state);
        assert_eq!(r.status(), StatusCode::OK);
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["engine_ready"], false);
        assert_eq!(body["engine_pending"], false);
        assert_eq!(body["queued_jobs"], 0);
    }

    #[test]
    fn generation_token_cap_rejects_oversized_requests() {
        let err = validate_generation_tokens("max_tokens", MAX_GENERATION_TOKENS + 1)
            .err()
            .unwrap();
        assert!(err.contains("server limit"));
    }

    #[test]
    fn content_parser_rejects_unsupported_parts() {
        let err = content_value_to_text(&json!([
            {"type": "text", "text": "ok"},
            {"type": "image_url", "image_url": {"url": "x"}}
        ]))
        .err()
        .unwrap();
        assert!(err.contains("unsupported content part type"));
    }

    #[test]
    fn dsml_text_transcodes_to_openai_tool_call() {
        let text = crate::dsml::render(&[crate::dsml::DsmlToolCall {
            name: "lookup".to_string(),
            parameters: vec![crate::dsml::DsmlParameter {
                name: "q".to_string(),
                is_string: true,
                value: "rust".to_string(),
            }],
        }]);
        let (content, calls, finish) = openai_message_content_and_tools(&text);
        assert!(content.is_none());
        assert_eq!(finish, "tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert!(calls[0].function.arguments.contains("rust"));
    }

    #[test]
    fn responses_prompt_text_accepts_structured_input() {
        let req = ResponsesRequest {
            instructions: Some("system".to_string()),
            input: Some(json!([
                {"role": "user", "content": "hello"},
                {"role": "user", "content": [{"type": "input_text", "text": "world"}]}
            ])),
            ..ResponsesRequest::default()
        };
        assert_eq!(responses_prompt_text(&req).unwrap(), "system\nhello\nworld");
    }

    #[test]
    fn responses_prompt_text_rejects_unsupported_object() {
        let req = ResponsesRequest {
            input: Some(json!({"bad": "shape"})),
            ..ResponsesRequest::default()
        };
        assert!(responses_prompt_text(&req).is_err());
    }

    #[test]
    fn sse_response_uses_event_stream_content_type() {
        let r = sse_response(b"data: [DONE]\n\n".to_vec());
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(
            r.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    #[test]
    fn not_found_returns_404() {
        let r = not_found("GET /missing");
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_stream_with_synthetic_model_returns_completion_chunks() {
        let state = synthetic_state();
        let req = minimal_chat_request(true);
        let r = chat_completions_stream(
            &state,
            &req,
            state.pool.tokenize_text("hi").unwrap(),
            3,
            sampling_from_openai(&req),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
        assert!(!body.contains("engine_not_implemented"));
    }

    #[tokio::test]
    async fn chat_unary_with_synthetic_model_returns_message_text() {
        let state = synthetic_state();
        let req = minimal_chat_request(false);
        let r = chat_completions_unary(
            &state,
            &req,
            state.pool.tokenize_text("hi").unwrap(),
            3,
            sampling_from_openai(&req),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert!(body["choices"][0]["message"]["content"].as_str().is_some());
        assert_eq!(body["usage"]["prompt_tokens"], 1);
    }

    #[tokio::test]
    async fn completions_stream_returns_sse_chunks() {
        let state = synthetic_state();
        let r = completion_stream_response(
            &state,
            "ds4-test".to_string(),
            JobResult {
                job_id: "unit".to_string(),
                completion_tokens: vec![8, 9],
                usage: crate::engine_pool::TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                },
            },
        );
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(
            r.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("text_completion"));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn responses_stream_returns_sse_response_payload() {
        let state = synthetic_state();
        let r = responses_stream_response(
            &state,
            "ds4-test".to_string(),
            JobResult {
                job_id: "unit".to_string(),
                completion_tokens: vec![8, 9],
                usage: crate::engine_pool::TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                },
            },
        );
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(
            r.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.output_text.done"));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains("\"object\":\"response\""));
        assert!(body.contains("\"output_text\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn anthropic_stream_with_synthetic_model_uses_engine() {
        let state = synthetic_state();
        let req = MessagesRequest {
            model: "ds4-test".to_string(),
            messages: vec![crate::anthropic::AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::Value::String("hi".to_string()),
            }],
            max_tokens: 2,
            system: None,
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            stream: Some(true),
            stop_sequences: None,
            tools: Vec::new(),
            tool_choice: None,
            metadata: None,
        };
        let prompt = state.pool.encode_chat_messages(&[("user", "hi")]).unwrap();
        let r = anthropic_stream(&req, &state, prompt, 2, SamplingParams::default()).await;
        assert_eq!(r.status(), StatusCode::OK);
        let bytes = r.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: message_stop"));
        assert!(!body.contains(NOT_IMPLEMENTED_MSG));
    }
}
