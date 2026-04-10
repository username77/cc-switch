
//! Codex compatibility bridge:
//! OpenAI Chat Completions <-> OpenAI Responses (request/response + SSE)

use crate::proxy::error::ProxyError;
use crate::proxy::sse::strip_sse_field;
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
struct StreamToolCallState {
    call_id: String,
    name: String,
    arguments: String,
    item_id: String,
    output_index: u64,
    added: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiStreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[inline]
fn map_tool_choice_to_openai(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(_) => tool_choice.clone(),
        Value::Object(obj) => {
            if obj.get("type").and_then(|t| t.as_str()) == Some("function") {
                let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("");
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            } else {
                tool_choice.clone()
            }
        }
        _ => tool_choice.clone(),
    }
}

#[inline]
fn normalize_responses_role_for_chat(role: &str) -> &str {
    match role {
        // DeepSeek chat-completions rejects `developer`, while Responses clients
        // legitimately emit it. Preserve intent by downgrading to `system`.
        "developer" => "system",
        other => other,
    }
}

fn convert_response_input_item_to_chat_messages(item: &Value) -> Vec<Value> {
    let mut out = Vec::new();

    if let Some(role) = item.get("role").and_then(|v| v.as_str()) {
        let normalized_role = normalize_responses_role_for_chat(role);
        let content = item.get("content");
        if let Some(text) = content.and_then(|v| v.as_str()) {
            out.push(json!({
                "role": normalized_role,
                "content": text
            }));
            return out;
        }

        if let Some(parts) = content.and_then(|v| v.as_array()) {
            let mut converted_parts = Vec::new();
            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "input_text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            converted_parts.push(json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                    }
                    "input_image" => {
                        if let Some(url) = part.get("image_url").and_then(|u| u.as_str()) {
                            converted_parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": url }
                            }));
                        }
                    }
                    "refusal" => {
                        if let Some(refusal) = part.get("refusal").and_then(|r| r.as_str()) {
                            converted_parts.push(json!({
                                "type": "text",
                                "text": refusal
                            }));
                        }
                    }
                    _ => {}
                }
            }

            let content_value = if converted_parts.is_empty() {
                Value::Null
            } else if converted_parts.len() == 1
                && converted_parts[0]
                    .get("type")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t == "text")
            {
                converted_parts[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::String(String::new()))
            } else {
                Value::Array(converted_parts)
            };

            out.push(json!({
                "role": normalized_role,
                "content": content_value
            }));
            return out;
        }

        out.push(json!({
            "role": normalized_role,
            "content": Value::Null
        }));
        return out;
    }

    let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match item_type {
        "function_call" => {
            let call_id = item.get("call_id").and_then(|i| i.as_str()).unwrap_or("");
            let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let arguments = item
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            out.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }]
            }));
        }
        "function_call_output" => {
            let call_id = item.get("call_id").and_then(|i| i.as_str()).unwrap_or("");
            let output_text = match item.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(v) => serde_json::to_string(v).unwrap_or_default(),
                None => String::new(),
            };
            out.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output_text
            }));
        }
        _ => {}
    }

    out
}

/// Convert OpenAI Responses request body to OpenAI Chat Completions request body.
pub fn responses_request_to_openai_chat(body: Value) -> Result<Value, ProxyError> {
    // Already in chat format.
    if body.get("messages").is_some() && body.get("input").is_none() {
        return Ok(body);
    }

    let mut result = json!({});
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": instructions
            }));
        }
    }

    if let Some(input_text) = body.get("input").and_then(|v| v.as_str()) {
        messages.push(json!({
            "role": "user",
            "content": input_text
        }));
    } else if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            messages.extend(convert_response_input_item_to_chat_messages(item));
        }
    }

    result["messages"] = Value::Array(messages);

    if let Some(v) = body.get("max_output_tokens") {
        result["max_tokens"] = v.clone();
    }
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }
    if let Some(v) = body.get("stream") {
        result["stream"] = v.clone();
    }
    if let Some(v) = body.get("parallel_tool_calls") {
        result["parallel_tool_calls"] = v.clone();
    }
    if let Some(effort) = body.pointer("/reasoning/effort").and_then(|v| v.as_str()) {
        result["reasoning_effort"] = json!(effort);
    }

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mapped_tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if name.is_empty() {
                    return None;
                }
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": tool.get("parameters").cloned().unwrap_or_else(|| json!({}))
                    }
                }))
            })
            .collect();
        result["tools"] = Value::Array(mapped_tools);
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        result["tool_choice"] = map_tool_choice_to_openai(tool_choice);
    }

    Ok(result)
}

fn chat_message_to_responses_content(message: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(content) = message.get("content") {
        if let Some(text) = content.as_str() {
            if !text.is_empty() {
                out.push(json!({
                    "type": "output_text",
                    "text": text
                }));
            }
        } else if let Some(parts) = content.as_array() {
            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                out.push(json!({
                                    "type": "output_text",
                                    "text": text
                                }));
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(refusal) = part.get("refusal").and_then(|r| r.as_str()) {
                            if !refusal.is_empty() {
                                out.push(json!({
                                    "type": "refusal",
                                    "refusal": refusal
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(refusal) = message.get("refusal").and_then(|r| r.as_str()) {
        if !refusal.is_empty() {
            out.push(json!({
                "type": "refusal",
                "refusal": refusal
            }));
        }
    }

    out
}

fn map_openai_usage_to_responses(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return json!({
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        });
    };

    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(input_tokens + output_tokens);

    let mut mapped = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    });

    if let Some(cached) = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        mapped["input_tokens_details"] = json!({
            "cached_tokens": cached
        });
    }

    mapped
}

/// Convert OpenAI Chat Completions response body to OpenAI Responses response body.
pub fn openai_chat_response_to_responses(body: Value) -> Result<Value, ProxyError> {
    let choice = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| ProxyError::TransformError("No choices in chat response".to_string()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| ProxyError::TransformError("No message in first choice".to_string()))?;
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");

    let response_id = body
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("resp_ccswitch");
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let created = body.get("created").cloned().unwrap_or_else(|| json!(0));

    let mut output: Vec<Value> = Vec::new();
    let content_parts = chat_message_to_responses_content(message);
    if !content_parts.is_empty() {
        output.push(json!({
            "id": format!("{response_id}_msg"),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": content_parts
        }));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for (idx, tool_call) in tool_calls.iter().enumerate() {
            let call_id = tool_call
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let function = tool_call.get("function").cloned().unwrap_or_else(|| json!({}));
            let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            output.push(json!({
                "id": format!("{response_id}_fc_{idx}"),
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments
            }));
        }
    }

    let mut response = json!({
        "id": response_id,
        "object": "response",
        "type": "response",
        "created_at": created,
        "model": model,
        "output": output,
        "usage": map_openai_usage_to_responses(body.get("usage"))
    });

    if finish_reason == "length" {
        response["status"] = json!("incomplete");
        response["incomplete_details"] = json!({
            "reason": "max_output_tokens"
        });
    } else {
        response["status"] = json!("completed");
    }

    Ok(response)
}

fn sse_event(event: &str, data: Value) -> Bytes {
    Bytes::from(format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
    ))
}

fn done_sse() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

/// Convert OpenAI Chat Completions SSE stream to OpenAI Responses SSE stream.
pub fn create_responses_sse_stream_from_openai_chat<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();

        let mut response_id = String::new();
        let mut model = String::new();
        let mut sent_response_created = false;

        let mut message_item_id = String::new();
        let mut sent_message_item_added = false;
        let mut sent_message_part_added = false;
        let mut message_text = String::new();

        let mut tool_calls: BTreeMap<usize, StreamToolCallState> = BTreeMap::new();
        let mut next_output_index: u64 = 0;
        let mut latest_usage: Option<Value> = None;
        let mut completed = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::proxy::sse::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for part in line.lines() {
                            if let Some(data) = strip_sse_field(part, "data") {
                                if data.trim() == "[DONE]" {
                                    if !completed && sent_response_created {
                                        let mut output_items: Vec<Value> = Vec::new();
                                        if sent_message_item_added {
                                            output_items.push(json!({
                                                "id": message_item_id,
                                                "type": "message",
                                                "role": "assistant",
                                                "status": "completed",
                                                "content": [{
                                                    "type": "output_text",
                                                    "text": message_text
                                                }]
                                            }));
                                        }
                                        for state in tool_calls.values() {
                                            if state.added {
                                                output_items.push(json!({
                                                    "id": state.item_id,
                                                    "type": "function_call",
                                                    "call_id": state.call_id,
                                                    "name": state.name,
                                                    "arguments": state.arguments
                                                }));
                                            }
                                        }

                                        let response_obj = json!({
                                            "id": response_id,
                                            "object": "response",
                                            "type": "response",
                                            "status": "completed",
                                            "model": model,
                                            "output": output_items,
                                            "usage": map_openai_usage_to_responses(latest_usage.as_ref())
                                        });
                                        yield Ok(sse_event("response.completed", json!({
                                            "type": "response.completed",
                                            "response": response_obj
                                        })));
                                        yield Ok(done_sse());
                                        completed = true;
                                    }
                                    continue;
                                }

                                let parsed = serde_json::from_str::<OpenAiStreamChunk>(data);
                                let Ok(chunk_json) = parsed else {
                                    continue;
                                };

                                if let Some(id) = &chunk_json.id {
                                    if response_id.is_empty() {
                                        response_id = id.clone();
                                    }
                                }
                                if let Some(m) = &chunk_json.model {
                                    if model.is_empty() {
                                        model = m.clone();
                                    }
                                }
                                if response_id.is_empty() {
                                    response_id = "resp_ccswitch".to_string();
                                }
                                if model.is_empty() {
                                    model = "unknown".to_string();
                                }
                                if let Some(usage) = chunk_json.usage.as_ref() {
                                    if !usage.is_null() {
                                        latest_usage = Some(usage.clone());
                                    }
                                }

                                if !sent_response_created {
                                    message_item_id = format!("{response_id}_msg");
                                    yield Ok(sse_event("response.created", json!({
                                        "type": "response.created",
                                        "response": {
                                            "id": response_id,
                                            "object": "response",
                                            "type": "response",
                                            "status": "in_progress",
                                            "model": model,
                                            "output": [],
                                            "usage": Value::Null
                                        }
                                    })));
                                    sent_response_created = true;
                                }

                                if let Some(choice) = chunk_json.choices.first() {
                                    if let Some(delta_text) = choice.delta.content.as_deref() {
                                        if !sent_message_item_added {
                                            yield Ok(sse_event("response.output_item.added", json!({
                                                "type": "response.output_item.added",
                                                "response_id": response_id,
                                                "output_index": next_output_index,
                                                "item": {
                                                    "id": message_item_id,
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "status": "in_progress",
                                                    "content": []
                                                }
                                            })));
                                            sent_message_item_added = true;
                                            next_output_index = 1;
                                        }

                                        if !sent_message_part_added {
                                            yield Ok(sse_event("response.content_part.added", json!({
                                                "type": "response.content_part.added",
                                                "response_id": response_id,
                                                "output_index": 0,
                                                "item_id": message_item_id,
                                                "content_index": 0,
                                                "part": {
                                                    "type": "output_text",
                                                    "text": ""
                                                }
                                            })));
                                            sent_message_part_added = true;
                                        }

                                        if !delta_text.is_empty() {
                                            message_text.push_str(delta_text);
                                            yield Ok(sse_event("response.output_text.delta", json!({
                                                "type": "response.output_text.delta",
                                                "response_id": response_id,
                                                "output_index": 0,
                                                "item_id": message_item_id,
                                                "content_index": 0,
                                                "delta": delta_text
                                            })));
                                        }
                                    }

                                    if let Some(tool_deltas) = &choice.delta.tool_calls {
                                        for delta in tool_deltas {
                                            let state = tool_calls.entry(delta.index).or_insert_with(|| {
                                                let call_id = delta.id.clone().unwrap_or_else(|| format!("call_{}", delta.index));
                                                StreamToolCallState {
                                                    call_id: call_id.clone(),
                                                    name: String::new(),
                                                    arguments: String::new(),
                                                    item_id: format!("{response_id}_fc_{}", delta.index),
                                                    output_index: next_output_index,
                                                    added: false,
                                                }
                                            });

                                            if !state.added {
                                                state.output_index = next_output_index;
                                                next_output_index += 1;
                                            }

                                            if let Some(id) = delta.id.as_deref() {
                                                if !id.is_empty() {
                                                    state.call_id = id.to_string();
                                                }
                                            }
                                            if let Some(function) = &delta.function {
                                                if let Some(name) = function.name.as_deref() {
                                                    if !name.is_empty() {
                                                        state.name = name.to_string();
                                                    }
                                                }
                                                if let Some(arguments) = function.arguments.as_deref() {
                                                    state.arguments.push_str(arguments);
                                                }
                                            }

                                            if !state.added && !state.name.is_empty() {
                                                yield Ok(sse_event("response.output_item.added", json!({
                                                    "type": "response.output_item.added",
                                                    "response_id": response_id,
                                                    "output_index": state.output_index,
                                                    "item": {
                                                        "id": state.item_id,
                                                        "type": "function_call",
                                                        "status": "in_progress",
                                                        "call_id": state.call_id,
                                                        "name": state.name,
                                                        "arguments": ""
                                                    }
                                                })));
                                                state.added = true;
                                            }

                                            if let Some(function) = &delta.function {
                                                if let Some(arguments) = function.arguments.as_deref() {
                                                    if state.added && !arguments.is_empty() {
                                                        yield Ok(sse_event("response.function_call_arguments.delta", json!({
                                                            "type": "response.function_call_arguments.delta",
                                                            "response_id": response_id,
                                                            "output_index": state.output_index,
                                                            "item_id": state.item_id,
                                                            "delta": arguments
                                                        })));
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if let Some(finish_reason) = choice.finish_reason.as_deref() {
                                        if sent_message_part_added {
                                            yield Ok(sse_event("response.content_part.done", json!({
                                                "type": "response.content_part.done",
                                                "response_id": response_id,
                                                "output_index": 0,
                                                "item_id": message_item_id,
                                                "content_index": 0,
                                                "part": {
                                                    "type": "output_text",
                                                    "text": message_text
                                                }
                                            })));
                                        }

                                        if sent_message_item_added {
                                            yield Ok(sse_event("response.output_item.done", json!({
                                                "type": "response.output_item.done",
                                                "response_id": response_id,
                                                "output_index": 0,
                                                "item": {
                                                    "id": message_item_id,
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "status": "completed",
                                                    "content": [{
                                                        "type": "output_text",
                                                        "text": message_text
                                                    }]
                                                }
                                            })));
                                        }

                                        for state in tool_calls.values() {
                                            if state.added {
                                                yield Ok(sse_event("response.output_item.done", json!({
                                                    "type": "response.output_item.done",
                                                    "response_id": response_id,
                                                    "output_index": state.output_index,
                                                    "item": {
                                                        "id": state.item_id,
                                                        "type": "function_call",
                                                        "status": "completed",
                                                        "call_id": state.call_id,
                                                        "name": state.name,
                                                        "arguments": state.arguments
                                                    }
                                                })));
                                            }
                                        }

                                        let mut output_items: Vec<Value> = Vec::new();
                                        if sent_message_item_added {
                                            output_items.push(json!({
                                                "id": message_item_id,
                                                "type": "message",
                                                "role": "assistant",
                                                "status": "completed",
                                                "content": [{
                                                    "type": "output_text",
                                                    "text": message_text
                                                }]
                                            }));
                                        }
                                        for state in tool_calls.values() {
                                            if state.added {
                                                output_items.push(json!({
                                                    "id": state.item_id,
                                                    "type": "function_call",
                                                    "call_id": state.call_id,
                                                    "name": state.name,
                                                    "arguments": state.arguments
                                                }));
                                            }
                                        }

                                        let mut response_obj = json!({
                                            "id": response_id,
                                            "object": "response",
                                            "type": "response",
                                            "status": "completed",
                                            "model": model,
                                            "output": output_items,
                                            "usage": map_openai_usage_to_responses(latest_usage.as_ref())
                                        });
                                        if finish_reason == "length" {
                                            response_obj["status"] = json!("incomplete");
                                            response_obj["incomplete_details"] = json!({
                                                "reason": "max_output_tokens"
                                            });
                                        }

                                        yield Ok(sse_event("response.completed", json!({
                                            "type": "response.completed",
                                            "response": response_obj
                                        })));
                                        yield Ok(done_sse());
                                        completed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Err(std::io::Error::other(e.to_string()));
                    break;
                }
            }
        }
    }
}

