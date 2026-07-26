//! Loop-side message model.
//!
//! Rust mirror of the harness message types in `packages/ai/src/types.ts`
//! (`AssistantMessage` :712, content blocks :587-652, `StopReason` :654,
//! `Message` :781) and the catalog `Usage` shape
//! (`packages/catalog/src/types.ts:95`). JSON field names and discriminator
//! values are kept byte-identical to the TS serialization (camelCase, `type` /
//! `role` tags) so session files and golden transcripts interoperate across the
//! two implementations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::wire::StopDetails;

// ─── Content blocks ─────────────────────────────────────────────────────────

/// `TextContent` (types.ts:587).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "text", rename_all = "camelCase")]
pub struct TextContent {
	pub text:           String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub text_signature: Option<String>,
}

/// `ThinkingContent` (types.ts:593).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "thinking", rename_all = "camelCase")]
pub struct ThinkingContent {
	pub thinking:           String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking_signature: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub item_id:            Option<String>,
}

/// `RedactedThinkingContent` (types.ts:600).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "redactedThinking")]
pub struct RedactedThinkingContent {
	pub data: String,
}

/// `AnthropicFallbackContent` (types.ts:613).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "fallback")]
pub struct FallbackContent {
	pub from: crate::wire::ModelRef,
	pub to:   crate::wire::ModelRef,
}

/// `ImageContent` (types.ts:619).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "image", rename_all = "camelCase")]
pub struct ImageContent {
	/// Base64-encoded image data.
	pub data:      String,
	pub mime_type: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub detail:    Option<String>,
}

/// `ToolCall` (types.ts:631). The TS `[kStreamingPartialJson]` symbol key is
/// streaming-builder internal state and never serializes, so it has no field
/// here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "toolCall", rename_all = "camelCase")]
pub struct ToolCall {
	pub id:                String,
	pub name:              String,
	/// Parsed argument object (`Record<string, unknown>`).
	pub arguments:         Value,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thought_signature: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub intent:            Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub raw_block:         Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub custom_wire_name:  Option<String>,
}

/// Assistant content union (types.ts:714-721). Each arm is an internally
/// tagged struct, so the untagged union dispatches on the embedded `type`
/// discriminator exactly like the TS union.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
	Text(TextContent),
	Thinking(ThinkingContent),
	RedactedThinking(RedactedThinkingContent),
	Fallback(FallbackContent),
	Image(ImageContent),
	ToolCall(ToolCall),
}

// ─── Usage ──────────────────────────────────────────────────────────────────

/// `Usage.orchestration` (catalog/types.ts:107).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestrationUsage {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input:      Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_read: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output:     Option<u64>,
}

/// `Usage.cttl` (catalog/types.ts:131) — Anthropic cache-write TTL breakdown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CttlUsage {
	#[serde(rename = "ephemeral5m", default, skip_serializing_if = "Option::is_none")]
	pub ephemeral_5m: Option<u64>,
	#[serde(rename = "ephemeral1h", default, skip_serializing_if = "Option::is_none")]
	pub ephemeral_1h: Option<u64>,
}

/// `Usage.server` (catalog/types.ts:140) — server-side tool invocation counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerUsage {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub web_search: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub web_fetch:  Option<u64>,
}

/// `Usage.cost` (catalog/types.ts:145). WP-1.1a leaves all components zero —
/// pricing lives in the catalog layer, which lands with a later WP.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
	pub input:       f64,
	pub output:      f64,
	pub cache_read:  f64,
	pub cache_write: f64,
	pub total:       f64,
}

/// Harness `Usage` (catalog/types.ts:95).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
	pub input:            u64,
	pub output:           u64,
	pub cache_read:       u64,
	pub cache_write:      u64,
	// Legacy/aborted sessions persist `usage` without `totalTokens` (the TS type
	// requires it but the runtime never validates loaded JSONL); default to 0 so
	// those messages load instead of dropping to an opaque unknown entry.
	#[serde(default)]
	pub total_tokens:     u64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub orchestration:    Option<OrchestrationUsage>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub premium_requests: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cttl:             Option<CttlUsage>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub server:           Option<ServerUsage>,
	pub cost:             UsageCost,
}

// ─── Assistant message ──────────────────────────────────────────────────────

/// Harness `StopReason` (types.ts:654) — distinct from the wire enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
	Stop,
	Length,
	ToolUse,
	Error,
	Aborted,
}

/// `ContextSnapshot` (types.ts:706).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSnapshot {
	pub prompt_tokens:          u64,
	pub non_message_tokens:     u64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub last_message_timestamp: Option<i64>,
}

/// `AssistantMessage` (types.ts:712). `retryRecovery` and `providerPayload`
/// are harness-managed opaque payloads here (`Value`), typed on the TS side
/// only; the provider layer never inspects them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "assistant", rename_all = "camelCase")]
pub struct AssistantMessage {
	pub content: Vec<AssistantContent>,
	pub api: String,
	pub provider: String,
	pub model: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub context_snapshot: Option<ContextSnapshot>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub retry_recovery: Option<Value>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub response_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub upstream_provider: Option<String>,
	pub usage: Usage,
	pub stop_reason: StopReason,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_details: Option<StopDetails>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_call_abort_messages: Option<BTreeMap<String, String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error_status: Option<u16>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error_id: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub disabled_features: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub provider_payload: Option<Value>,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub duration: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub ttft: Option<u64>,
}

// ─── Other roles (request-side context, minimal surface) ────────────────────

/// `MessageAttribution` (types.ts:94).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MessageAttribution {
	User,
	Agent,
}

/// `UserMessage.content` / `DeveloperMessage.content`: string or text/image
/// blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
	Text(String),
	Blocks(Vec<UserContentBlock>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContentBlock {
	Text(TextContent),
	Image(ImageContent),
}

/// `UserMessage` (types.ts:665).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "user", rename_all = "camelCase")]
pub struct UserMessage {
	pub content:          UserContent,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub synthetic:        Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub steering:         Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub attribution:      Option<MessageAttribution>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub provider_payload: Option<Value>,
	pub timestamp:        i64,
}

/// `DeveloperMessage` (types.ts:679).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "developer", rename_all = "camelCase")]
pub struct DeveloperMessage {
	pub content:          UserContent,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub attribution:      Option<MessageAttribution>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub provider_payload: Option<Value>,
	pub timestamp:        i64,
}

/// `ToolResultMessage` (types.ts:761) with `details` carried opaquely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename = "toolResult", rename_all = "camelCase")]
pub struct ToolResultMessage {
	pub tool_call_id: String,
	pub tool_name:    String,
	pub content:      Vec<UserContentBlock>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details:      Option<Value>,
	pub is_error:     bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub attribution:  Option<MessageAttribution>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub pruned_at:    Option<i64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub useless:      Option<bool>,
	pub timestamp:    i64,
}

/// `Message` union (types.ts:781), dispatched on the embedded `role` tag.
///
/// Serialization stays untagged (each variant struct writes its own `role`
/// via `#[serde(tag = "role")]`). Deserialization must NOT be untagged:
/// struct-level `tag` is not validated when probing untagged variants, so a
/// `toolResult` JSON (whose extra fields serde ignores) would match the more
/// permissive `UserMessage` first. Dispatch on `role` explicitly instead.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Message {
	User(UserMessage),
	Developer(DeveloperMessage),
	Assistant(Box<AssistantMessage>),
	ToolResult(ToolResultMessage),
}

impl<'de> Deserialize<'de> for Message {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = Value::deserialize(deserializer)?;
		let role = value
			.get("role")
			.and_then(Value::as_str)
			.ok_or_else(|| serde::de::Error::missing_field("role"))?;
		match role {
			"user" => UserMessage::deserialize(&value)
				.map(Self::User)
				.map_err(serde::de::Error::custom),
			"developer" => DeveloperMessage::deserialize(&value)
				.map(Self::Developer)
				.map_err(serde::de::Error::custom),
			"assistant" => AssistantMessage::deserialize(&value)
				.map(|message| Self::Assistant(Box::new(message)))
				.map_err(serde::de::Error::custom),
			"toolResult" => ToolResultMessage::deserialize(&value)
				.map(Self::ToolResult)
				.map_err(serde::de::Error::custom),
			other => Err(serde::de::Error::unknown_variant(other, &[
				"user",
				"developer",
				"assistant",
				"toolResult",
			])),
		}
	}
}
