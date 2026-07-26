//! Session entry model — the logical `FileEntry` tree after the physical title
//! slot is stripped. Rust mirror of
//! `packages/coding-agent/src/session/session-entries.ts`.
//!
//! Two design invariants carried over verbatim from the TS source:
//!
//! * **Dual timestamp shapes coexist** — the entry *envelope* `timestamp` is an
//!   ISO-8601 string (`nowIso()`), while a nested `message.timestamp` is a
//!   unix-ms number. They are NOT normalized to one representation.
//! * **Strong types for the entries the loop reasons about; raw passthrough for
//!   everything else.** [`SessionEntry`] models the entries
//!   `build_session_context` inspects ([`message`](KnownEntry::Message),
//!   [`compaction`](KnownEntry::Compaction), `model_change`,
//!   `thinking_level_change`, `service_tier_change`, `mode_change`,
//!   `custom_message`). Every other type — `custom`, `label`, `title_change`,
//!   `ttsr_injection`, `session_init`, `branch_summary`, and any future kind —
//!   lands in [`SessionEntry::Unknown`] as an opaque [`serde_json::Value`] and
//!   is never rewritten (the writer is strictly append-only; see `writer.rs`).

use pi_ai::message::AssistantMessage;
pub use pi_ai::message::{ImageContent, Message, MessageAttribution, TextContent, UserContent};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

/// Deserialize a `message` payload by dispatching on its `role`.
///
/// pi-ai's `Message` is an untagged enum whose arms are `#[serde(tag =
/// "role")]` structs. serde does not verify that internal tag under untagged
/// buffering, so a `toolResult` payload would silently deserialize as the
/// more-permissive `user` arm. We dispatch on `role` explicitly to pick the
/// correct arm without modifying pi-ai. A payload whose `role` is not one of
/// the four standard roles errors here, which makes the enclosing `message`
/// entry fall through to [`SessionEntry::Unknown`] (whole-entry passthrough,
/// per U-omp-28).
fn deserialize_message<'de, D>(deserializer: D) -> Result<Message, D::Error>
where
	D: Deserializer<'de>,
{
	let value = Value::deserialize(deserializer)?;
	message_from_json(value).map_err(D::Error::custom)
}

/// Build a [`Message`] from a JSON payload, dispatching on `role`.
///
/// See [`deserialize_message`] for why the direct untagged path is unsafe.
/// Errors for a non-standard role so callers can route the entry to unknown
/// passthrough or a hard failure as appropriate. Reused by the writer's
/// fixtures.
pub fn message_from_json(value: Value) -> Result<Message, serde_json::Error> {
	let role = value
		.get("role")
		.and_then(Value::as_str)
		.map(str::to_string);
	match role.as_deref() {
		Some("user") => Ok(Message::User(serde_json::from_value(value)?)),
		Some("developer") => Ok(Message::Developer(serde_json::from_value(value)?)),
		Some("assistant") => {
			let assistant: AssistantMessage = serde_json::from_value(value)?;
			Ok(Message::Assistant(Box::new(assistant)))
		},
		Some("toolResult") => Ok(Message::ToolResult(serde_json::from_value(value)?)),
		other => {
			Err(serde::de::Error::custom(format!("non-standard or missing message role: {other:?}")))
		},
	}
}

/// Current session version. The loader is v3-only (U-omp-29); `< 3` is a hard
/// error (migration deferred, see `loader.rs`).
pub const CURRENT_SESSION_VERSION: u32 = 3;

/// Fixed-width first-line title slot byte budget (`SESSION_TITLE_SLOT_BYTES`).
pub const SESSION_TITLE_SLOT_BYTES: usize = 256;

/// `SessionHeader` (session-entries.ts:27). Unknown fields pass through via
/// [`extra`](SessionHeader::extra) so a header written by a newer TS build
/// round trips without loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
	#[serde(rename = "type")]
	pub kind: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub version: Option<u32>,
	pub id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub title: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub title_source: Option<String>,
	pub timestamp: String,
	pub cwd: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_session: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub provider_prompt_cache_key: Option<String>,
	/// Any header field a newer writer added that this build does not model.
	#[serde(flatten)]
	pub extra: serde_json::Map<String, Value>,
}

/// `SessionMessageEntry` (session-entries.ts:55).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
	pub id:        String,
	pub parent_id: Option<String>,
	pub timestamp: String,
	#[serde(deserialize_with = "deserialize_message")]
	pub message:   Message,
}

/// `ThinkingLevelChangeEntry` (session-entries.ts:60).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelChangeEntry {
	pub id:             String,
	pub parent_id:      Option<String>,
	pub timestamp:      String,
	#[serde(default)]
	pub thinking_level: Option<String>,
	#[serde(default)]
	pub configured:     Option<String>,
}

/// `ModelChangeEntry` (session-entries.ts:71).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
	pub id:        String,
	pub parent_id: Option<String>,
	pub timestamp: String,
	pub model:     String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub role:      Option<String>,
}

/// `ServiceTierChangeEntry` (session-entries.ts:79). The tier value itself is
/// carried opaquely (`ServiceTierByFamily | null`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceTierChangeEntry {
	pub id:           String,
	pub parent_id:    Option<String>,
	pub timestamp:    String,
	pub service_tier: Value,
}

/// `ModeChangeEntry` (session-entries.ts:181).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeChangeEntry {
	pub id:        String,
	pub parent_id: Option<String>,
	pub timestamp: String,
	pub mode:      String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data:      Option<Value>,
}

/// `CustomMessageEntry` (session-entries.ts:201) — participates in LLM context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
	pub id:          String,
	pub parent_id:   Option<String>,
	pub timestamp:   String,
	pub custom_type: String,
	pub content:     UserContent,
	pub display:     bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details:     Option<Value>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub attribution: Option<MessageAttribution>,
}

/// `CompactionEntry` (session-entries.ts:84). `summary` / `short_summary` /
/// `preserve_data` are mutated in place by superseded-compaction elision (see
/// `loader::elide_superseded_compaction_entries`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
	pub id:                  String,
	pub parent_id:           Option<String>,
	pub timestamp:           String,
	pub summary:             String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub short_summary:       Option<String>,
	pub first_kept_entry_id: String,
	pub tokens_before:       i64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details:             Option<Value>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub preserve_data:       Option<Value>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub from_extension:      Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub warning:             Option<String>,
}

/// The strongly-typed entries the loop reasons about, internally tagged on the
/// `type` discriminator exactly like the TS `SessionEntry` union.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnownEntry {
	Message(MessageEntry),
	ThinkingLevelChange(ThinkingLevelChangeEntry),
	ModelChange(ModelChangeEntry),
	ServiceTierChange(ServiceTierChangeEntry),
	ModeChange(ModeChangeEntry),
	CustomMessage(CustomMessageEntry),
	Compaction(CompactionEntry),
}

/// A logical session entry: a [`KnownEntry`] the loop reasons about, or an
/// opaque [`SessionEntry::Unknown`] value preserved verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(
	clippy::large_enum_variant,
	reason = "the common Known arm carries the payload; boxing it would tax every entry to shrink \
	          the rare Unknown arm"
)]
pub enum SessionEntry {
	Known(KnownEntry),
	Unknown(Value),
}

impl SessionEntry {
	/// The entry's `id`, or `None` for a malformed unknown entry without one.
	pub fn id(&self) -> Option<&str> {
		match self {
			Self::Known(k) => Some(k.base().id.as_str()),
			Self::Unknown(v) => v.get("id").and_then(Value::as_str),
		}
	}

	/// The entry's `parentId`, or `None` at the tree root / for a malformed
	/// entry.
	pub fn parent_id(&self) -> Option<&str> {
		match self {
			Self::Known(k) => k.base().parent_id.as_deref(),
			Self::Unknown(v) => v.get("parentId").and_then(Value::as_str),
		}
	}

	/// The `type` discriminator.
	pub fn entry_type(&self) -> &str {
		match self {
			Self::Known(k) => k.entry_type(),
			Self::Unknown(v) => v.get("type").and_then(Value::as_str).unwrap_or(""),
		}
	}
}

/// Shared envelope fields borrowed from any [`KnownEntry`] variant.
pub struct EntryBase<'a> {
	pub id:        &'a String,
	pub parent_id: &'a Option<String>,
}

impl KnownEntry {
	pub(crate) const fn base(&self) -> EntryBase<'_> {
		match self {
			Self::Message(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::ThinkingLevelChange(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::ModelChange(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::ServiceTierChange(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::ModeChange(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::CustomMessage(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
			Self::Compaction(e) => EntryBase { id: &e.id, parent_id: &e.parent_id },
		}
	}

	pub(crate) const fn entry_type(&self) -> &'static str {
		match self {
			Self::Message(_) => "message",
			Self::ThinkingLevelChange(_) => "thinking_level_change",
			Self::ModelChange(_) => "model_change",
			Self::ServiceTierChange(_) => "service_tier_change",
			Self::ModeChange(_) => "mode_change",
			Self::CustomMessage(_) => "custom_message",
			Self::Compaction(_) => "compaction",
		}
	}
}
