//! Anthropic Messages API wire types.
//!
//! Rust mirror of `packages/ai/src/providers/anthropic-wire.ts`
//! (hand-maintained against <https://docs.anthropic.com/en/api/messages>). Field and tag names are
//! kept byte-identical to the TS wire layer so serialized requests can be
//! snapshot-diffed against the TS implementation.
//!
//! Serialization discipline: every optional field carries
//! `skip_serializing_if = "Option::is_none"` so the JSON output matches TS
//! `JSON.stringify` (which omits `undefined` members). Deserialization is
//! tolerant: unknown object fields are ignored, unknown content-block tags fall
//! back to [`ResponseBlock::Unknown`], and unknown stop reasons fall back to
//! [`WireStopReason::Other`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Cache control ──────────────────────────────────────────────────────────

/// `CacheControlEphemeral` (anthropic-wire.ts:19).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
	Ephemeral {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		ttl:   Option<CacheTtl>,
		/// Claude Code prompt-caching-scope beta.
		#[serde(default, skip_serializing_if = "Option::is_none")]
		scope: Option<CacheScope>,
	},
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTtl {
	#[serde(rename = "1h")]
	OneHour,
	#[serde(rename = "5m")]
	FiveMinutes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
	Global,
}

// ─── Content blocks (request) ───────────────────────────────────────────────

/// `ImageSource` (anthropic-wire.ts:28-38).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
	Base64 { media_type: String, data: String },
	Url { url: String },
	File { file_id: String },
}

/// `{ model: string }` reference used by fallback boundary markers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
	pub model: String,
}

/// `tool_result.content`: either a plain string or content blocks
/// (anthropic-wire.ts:63 restricts blocks to text/image; kept open here).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
	Text(String),
	Blocks(Vec<ContentBlockParam>),
}

/// `ContentBlockParam` union (anthropic-wire.ts:91-98).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
	Text {
		text:          String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Image {
		source:        ImageSource,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ToolUse {
		id:            String,
		name:          String,
		input:         Value,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ToolResult {
		tool_use_id:   String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		content:       Option<ToolResultContent>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		is_error:      Option<bool>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Thinking {
		thinking:  String,
		signature: String,
	},
	RedactedThinking {
		data: String,
	},
	/// Server-side fallback beta boundary marker (anthropic-wire.ts:85).
	Fallback {
		from: ModelRef,
		to:   ModelRef,
	},
}

/// `MessageParam.role`; `system` is the mid-conversation system role beta
/// (anthropic-wire.ts:107).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
	User,
	Assistant,
	System,
}

/// `MessageParam.content`: plain string or block array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
	Text(String),
	Blocks(Vec<ContentBlockParam>),
}

/// A single conversation turn (anthropic-wire.ts:107).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageParam {
	pub role:    Role,
	pub content: MessageContent,
}

// ─── Tools ──────────────────────────────────────────────────────────────────

/// `Tool` (anthropic-wire.ts:121). `input_schema` is carried as opaque JSON;
/// Anthropic-specific schema normalization lives in
/// [`crate::schema::normalize_anthropic_tool_schema`] (WP-1.2 C1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
	pub name:                  String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub description:           Option<String>,
	pub input_schema:          Value,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_control:         Option<CacheControl>,
	/// Structured-outputs beta.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub strict:                Option<bool>,
	/// Fine-grained tool streaming beta.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub eager_input_streaming: Option<bool>,
}

/// `ToolChoice` (anthropic-wire.ts:132-137).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
	Auto {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
	Any {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
	Tool {
		name: String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
	None,
}

// ─── Request ────────────────────────────────────────────────────────────────

/// `Metadata` (anthropic-wire.ts:141).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub user_id: Option<String>,
}

/// Opus 4.7+ reasoning display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
	Summarized,
	Omitted,
}

/// `ThinkingConfigParam` (anthropic-wire.ts:143-158).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
	Enabled {
		budget_tokens: u64,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		display:       Option<ThinkingDisplay>,
	},
	Disabled,
	Adaptive {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		display: Option<ThinkingDisplay>,
	},
}

/// `TokenTaskBudget` (types.ts:88).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskBudget {
	Tokens {
		total:     u64,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		remaining: Option<u64>,
	},
}

/// `OutputConfig` (anthropic-wire.ts:160).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputConfig {
	/// Adaptive-thinking effort level (effort beta).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub effort:      Option<String>,
	/// Task-budgets beta.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub task_budget: Option<TaskBudget>,
}

/// `FallbackParam` (anthropic-wire.ts:172).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FallbackParam {
	pub model:         String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub max_tokens:    Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking:      Option<ThinkingConfig>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_config: Option<OutputConfig>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub speed:         Option<String>,
}

/// `ContextManagement` (anthropic-wire.ts:181).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextManagement {
	pub edits: Vec<ContextManagementEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextManagementEdit {
	#[serde(rename = "type")]
	pub edit_type: String,
	pub keep:      String,
}

/// `system`: plain string or text blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
	Text(String),
	Blocks(Vec<ContentBlockParam>),
}

/// `MessageCreateParams` (anthropic-wire.ts:185).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageCreateParams {
	pub model:              String,
	pub messages:           Vec<MessageParam>,
	pub max_tokens:         u64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub system:             Option<SystemPrompt>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub temperature:        Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub top_p:              Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub top_k:              Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_sequences:     Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stream:             Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tools:              Option<Vec<Tool>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_choice:        Option<ToolChoice>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub metadata:           Option<Metadata>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking:           Option<ThinkingConfig>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_config:      Option<OutputConfig>,
	/// Fast-mode beta.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub speed:              Option<String>,
	/// Claude Code context-management beta.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub context_management: Option<ContextManagement>,
	/// Server-side fallback beta chain.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub fallbacks:          Option<Vec<FallbackParam>>,
}

// ─── Response / usage ───────────────────────────────────────────────────────

/// `StopReason` (anthropic-wire.ts:216). Unknown values ship server-side first;
/// they deserialize into [`WireStopReason::Other`] instead of failing the turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireStopReason {
	Known(KnownStopReason),
	Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownStopReason {
	EndTurn,
	MaxTokens,
	StopSequence,
	ToolUse,
	PauseTurn,
	Refusal,
	Sensitive,
	ModelContextWindowExceeded,
}

/// `CacheCreation` (anthropic-wire.ts:226).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheCreation {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub ephemeral_5m_input_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub ephemeral_1h_input_tokens: Option<u64>,
}

/// `ServerToolUsage` (anthropic-wire.ts:231).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerToolUsage {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub web_search_requests: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub web_fetch_requests:  Option<u64>,
}

/// `UsageIteration` (anthropic-wire.ts:242).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageIteration {
	#[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
	pub iteration_type: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub model: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_read_input_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_creation_input_tokens: Option<u64>,
}

/// `Usage` (anthropic-wire.ts:251). Every field is optional: compatible
/// endpoints (e.g. `DeepSeek`) omit cache fields and add private extras,
/// which serde ignores.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireUsage {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input_tokens:                Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_tokens:               Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_read_input_tokens:     Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_creation_input_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_creation:              Option<CacheCreation>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub server_tool_use:             Option<ServerToolUsage>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub iterations:                  Option<Vec<UsageIteration>>,
}

/// Known `content_block` payload shapes (anthropic-wire.ts:276).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
	Text {
		text: String,
	},
	Thinking {
		thinking:  String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		signature: Option<String>,
	},
	RedactedThinking {
		data: String,
	},
	ToolUse {
		id:    String,
		name:  String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		input: Option<Value>,
	},
	Fallback {
		from: ModelRef,
		to:   ModelRef,
	},
}

/// A response content block with unknown-tag tolerance.
///
/// Known shapes parse into [`ResponseContentBlock`]; anything else is
/// preserved verbatim as [`ResponseBlock::Unknown`] (the TS layer types
/// response content as `unknown[]` and never fails on new block kinds).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseBlock {
	Known(ResponseContentBlock),
	Unknown(Value),
}

/// `ResponseMessage` (anthropic-wire.ts:262) — the non-streaming 200 body and
/// the `message_start` envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseMessage {
	pub id:            String,
	#[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
	pub message_type:  Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub role:          Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub model:         Option<String>,
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub content:       Vec<ResponseBlock>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_reason:   Option<WireStopReason>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_sequence: Option<String>,
	pub usage:         WireUsage,
}

/// `StopDetails` (anthropic-wire.ts:289). Shared verbatim with the model layer
/// (`AssistantMessage.stopDetails`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopDetails {
	#[serde(rename = "type")]
	pub detail_type: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub category:    Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub explanation: Option<String>,
}

// ─── Stream events (types only in WP-1.1a; SSE transport lands in WP-1.1b) ──

/// `ContentBlockDelta` (anthropic-wire.ts:283).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
	TextDelta { text: String },
	InputJsonDelta { partial_json: String },
	ThinkingDelta { thinking: String },
	SignatureDelta { signature: String },
}

/// `MessageDelta` (anthropic-wire.ts:295).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDelta {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_reason:   Option<WireStopReason>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_sequence: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stop_details:  Option<StopDetails>,
}

/// `RawMessageStreamEvent` (anthropic-wire.ts:312).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawMessageStreamEvent {
	MessageStart { message: ResponseMessage },
	ContentBlockStart { index: u64, content_block: ResponseBlock },
	ContentBlockDelta { index: u64, delta: ContentBlockDelta },
	ContentBlockStop { index: u64 },
	MessageDelta { delta: MessageDelta, usage: WireUsage },
	MessageStop,
}

// ─── Error envelope ─────────────────────────────────────────────────────────

/// `{"type":"error","error":{"type":…,"message":…}}` body on non-2xx responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
	pub error: ApiErrorBody,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorBody {
	#[serde(rename = "type", default)]
	pub error_type: String,
	#[serde(default)]
	pub message:    String,
}
