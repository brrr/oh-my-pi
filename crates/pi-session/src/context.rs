//! Read-only LLM message view.
//!
//! Rust mirror of the **non-transcript** path of `buildSessionContext` in
//! `packages/coding-agent/src/session/session-context.ts` (the shape returned
//! by `loadSessionMessagesReadOnly`).
//!
//! The builder walks the leaf→root path, folds settings entries, replays the
//! active branch — emitting the compaction summary first, then the kept range
//! `firstKeptEntryId..compaction`, then everything after the compaction — and
//! finally strips dangling `toolCall` blocks (an assistant `toolCall` with no
//! paired `toolResult` on the resolved path).
//!
//! Deferred (out of WP-1.3 scope, faithful to the pinned surface): transcript
//! mode, provider remote-compaction replacement history, `retryRecovery` skip,
//! and synthesizing `branchSummary` messages (a `branch_summary` entry stays an
//! opaque unknown and contributes no message). Fixtures avoid these so the
//! `loadSessionMessagesReadOnly` parity holds; they are logged here so a later
//! WP that needs them knows exactly what is missing.

use std::collections::{BTreeMap, BTreeSet};

use pi_ai::message::{AssistantContent, Message};
use serde::Serialize;
use serde_json::Value;

use crate::{
	entries::{KnownEntry, MessageAttribution, SessionEntry, UserContent},
	time::iso_to_unix_ms,
};

/// `CompactionSummaryMessage` (agent/compaction/messages.ts:47). Only the
/// fields a WP-1.3 compaction produces are populated; snapcompact
/// `blocks`/`images` and `providerPayload` are always absent here.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename = "compactionSummary", rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
	pub summary:          String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub short_summary:    Option<String>,
	pub tokens_before:    i64,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub provider_payload: Option<Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub blocks:           Option<Vec<Value>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub images:           Option<Vec<Value>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub warning:          Option<String>,
	pub timestamp:        i64,
}

/// `BranchSummaryMessage` (agent/compaction/messages.ts:40). Modeled for the
/// output union; not emitted by WP-1.3 (branch summaries stay opaque).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename = "branchSummary", rename_all = "camelCase")]
pub struct BranchSummaryMessage {
	pub summary:   String,
	pub from_id:   String,
	pub timestamp: i64,
}

/// `CustomMessage` (agent/compaction/messages.ts:17) — an extension-injected
/// message that participates in LLM context.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename = "custom", rename_all = "camelCase")]
pub struct CustomMessage {
	pub custom_type: String,
	pub content:     UserContent,
	pub display:     bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub details:     Option<Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub attribution: Option<MessageAttribution>,
	pub timestamp:   i64,
}

/// One message in the read-only context view.
///
/// Either a core [`Message`] or one of the compaction-domain synthetic
/// messages. Serializes untagged, so each arm emits its own `role`-tagged
/// object exactly like the TS `AgentMessage[]`.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ContextMessage {
	Standard(Message),
	CompactionSummary(CompactionSummaryMessage),
	Custom(CustomMessage),
	#[allow(dead_code, reason = "modeled for the union; branch summaries deferred (see module doc)")]
	BranchSummary(BranchSummaryMessage),
}

/// The rebuilt session context (message array + folded settings).
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
	pub messages:       Vec<ContextMessage>,
	/// Model roles: `{ default: "provider/modelId", ... }`.
	pub models:         BTreeMap<String, String>,
	pub thinking_level: Option<String>,
	pub service_tier:   Option<Value>,
	pub mode:           String,
}

/// Build the read-only context from `entries` along the leaf→root path. When
/// `leaf_id` is `None` the persisted leaf (last entry) is used, matching
/// `loadSessionMessagesReadOnly`.
pub fn build_session_context(entries: &[SessionEntry], leaf_id: Option<&str>) -> SessionContext {
	let by_id: BTreeMap<&str, &SessionEntry> = entries
		.iter()
		.filter_map(|e| e.id().map(|id| (id, e)))
		.collect();

	let leaf: Option<&SessionEntry> = match leaf_id {
		Some(id) => by_id.get(id).copied(),
		None => entries.last(),
	};
	let Some(leaf) = leaf else {
		return SessionContext { mode: "none".into(), ..SessionContext::default() };
	};

	// Walk leaf→root, then reverse to chronological order.
	let mut path: Vec<&SessionEntry> = Vec::new();
	let mut seen: BTreeSet<&str> = BTreeSet::new();
	let mut current = Some(leaf);
	while let Some(entry) = current {
		match entry.id() {
			Some(id) if seen.insert(id) => {},
			_ => break, // missing id or a cycle: stop the walk
		}
		path.push(entry);
		current = entry.parent_id().and_then(|p| by_id.get(p).copied());
	}
	path.reverse();

	// ── First pass: fold settings and find the active compaction. ──
	let mut models: BTreeMap<String, String> = BTreeMap::new();
	let mut thinking_level: Option<String> = Some("off".into());
	let mut service_tier: Option<Value> = None;
	let mut mode = "none".to_string();
	let mut compaction: Option<&crate::entries::CompactionEntry> = None;
	let mut has_explicit_default_model = false;

	for entry in &path {
		let SessionEntry::Known(known) = entry else {
			continue;
		};
		match known {
			KnownEntry::ThinkingLevelChange(e) => {
				thinking_level = Some(e.thinking_level.clone().unwrap_or_else(|| "off".into()));
			},
			KnownEntry::ModelChange(e) => {
				let role = e.role.clone().unwrap_or_else(|| "default".into());
				if role == "default" {
					has_explicit_default_model = true;
				}
				models.insert(role, e.model.clone());
			},
			KnownEntry::ServiceTierChange(e) => service_tier = Some(e.service_tier.clone()),
			KnownEntry::Message(e) => {
				if let Message::Assistant(a) = &e.message
					&& !has_explicit_default_model
				{
					models.insert("default".into(), format!("{}/{}", a.provider, a.model));
				}
			},
			KnownEntry::Compaction(e) => compaction = Some(e),
			KnownEntry::ModeChange(e) => mode.clone_from(&e.mode),
			KnownEntry::CustomMessage(_) => {},
		}
	}

	// ── Second pass: emit messages. ──
	let mut messages: Vec<ContextMessage> = Vec::new();
	if let Some(comp) = compaction {
		messages.push(ContextMessage::CompactionSummary(compaction_summary_message(comp)));
		let compaction_idx = path
			.iter()
			.position(|e| e.entry_type() == "compaction" && e.id() == Some(comp.id.as_str()))
			.unwrap_or(path.len());
		// Kept messages: from firstKeptEntryId up to the compaction.
		let mut found_first_kept = false;
		for entry in &path[..compaction_idx] {
			if entry.id() == Some(comp.first_kept_entry_id.as_str()) {
				found_first_kept = true;
			}
			if found_first_kept {
				append_message(entry, &mut messages);
			}
		}
		// Everything after the compaction.
		for entry in &path[compaction_idx + 1..] {
			append_message(entry, &mut messages);
		}
	} else {
		for entry in &path {
			append_message(entry, &mut messages);
		}
	}

	strip_dangling_tool_calls(&mut messages);

	SessionContext { messages, models, thinking_level, service_tier, mode }
}

fn compaction_summary_message(comp: &crate::entries::CompactionEntry) -> CompactionSummaryMessage {
	CompactionSummaryMessage {
		summary:          comp.summary.clone(),
		short_summary:    comp.short_summary.clone(),
		tokens_before:    comp.tokens_before,
		provider_payload: None,
		blocks:           None,
		images:           None,
		warning:          comp.warning.clone(),
		timestamp:        iso_to_unix_ms(&comp.timestamp).unwrap_or_default(),
	}
}

/// Emit the message(s) for one path entry. Only `message` and `custom_message`
/// entries contribute; settings entries and opaque unknowns are silent.
fn append_message(entry: &SessionEntry, out: &mut Vec<ContextMessage>) {
	let SessionEntry::Known(known) = entry else {
		return;
	};
	match known {
		KnownEntry::Message(e) => out.push(ContextMessage::Standard(e.message.clone())),
		KnownEntry::CustomMessage(e) => out.push(ContextMessage::Custom(CustomMessage {
			custom_type: e.custom_type.clone(),
			content:     e.content.clone(),
			display:     e.display,
			details:     e.details.clone(),
			attribution: e.attribution,
			timestamp:   iso_to_unix_ms(&e.timestamp).unwrap_or_default(),
		})),
		_ => {},
	}
}

/// Strip dangling `toolCall` blocks — a `toolCall` with no paired `toolResult`
/// on the resolved path — from any assistant turn, dropping `redactedThinking`
/// and clearing `thinking` signatures on a rewritten turn
/// (session-context.ts:438+).
fn strip_dangling_tool_calls(messages: &mut Vec<ContextMessage>) {
	let mut paired: BTreeSet<String> = BTreeSet::new();
	for msg in messages.iter() {
		if let ContextMessage::Standard(Message::ToolResult(tr)) = msg {
			paired.insert(tr.tool_call_id.clone());
		}
	}

	let mut i = messages.len();
	while i > 0 {
		i -= 1;
		let ContextMessage::Standard(Message::Assistant(assistant)) = &messages[i] else {
			continue;
		};
		let stripped = assistant
			.content
			.iter()
			.filter(|b| matches!(b, AssistantContent::ToolCall(tc) if !paired.contains(&tc.id)))
			.count();
		if stripped == 0 {
			continue;
		}
		let normalized: Vec<AssistantContent> = assistant
			.content
			.iter()
			.filter(|b| !matches!(b, AssistantContent::ToolCall(tc) if !paired.contains(&tc.id)))
			.filter(|b| !matches!(b, AssistantContent::RedactedThinking(_)))
			.map(|b| match b {
				AssistantContent::Thinking(t) if t.thinking_signature.is_some() => {
					let mut t = t.clone();
					t.thinking_signature = None;
					AssistantContent::Thinking(t)
				},
				other => other.clone(),
			})
			.collect();
		if normalized.is_empty() {
			messages.remove(i);
		} else {
			let mut rewritten = assistant.clone();
			rewritten.content = normalized;
			messages[i] = ContextMessage::Standard(Message::Assistant(rewritten));
		}
	}
}
