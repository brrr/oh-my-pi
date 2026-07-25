//! The pinned provider↔loop event interface.
//!
//! 1:1 semantic copy of `AssistantMessageEvent`
//! (`packages/ai/src/types.ts:900-923`, 13 variants). Contract:
//! `docs/omp-headless/provider-event-contract.md` — any change to this enum
//! (variants, fields, serde names) must go through the contract's change
//! process first.
//!
//! Every incremental event carries `partial`: an immutable snapshot of the
//! accumulated [`AssistantMessage`] so far. Snapshots are shared via `Arc`
//! (serde-transparent — the JSON is identical to inlining the message), so
//! cloning an event is O(1) and consumers must not assume object identity
//! across events.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::message::{AssistantMessage, ImageContent, ToolCall};

/// `Extract<StopReason, "stop" | "length" | "toolUse">` — terminal reasons a
/// successful turn can end with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DoneReason {
	Stop,
	Length,
	ToolUse,
}

/// `Extract<StopReason, "aborted" | "error">`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorReason {
	Aborted,
	Error,
}

/// `AssistantMessageEvent` (types.ts:900-923).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum AssistantMessageEvent {
	Start {
		partial: Arc<AssistantMessage>,
	},
	TextStart {
		content_index: usize,
		partial:       Arc<AssistantMessage>,
	},
	TextDelta {
		content_index: usize,
		delta:         String,
		partial:       Arc<AssistantMessage>,
	},
	TextEnd {
		content_index: usize,
		content:       String,
		partial:       Arc<AssistantMessage>,
	},
	ThinkingStart {
		content_index: usize,
		partial:       Arc<AssistantMessage>,
	},
	ThinkingDelta {
		content_index: usize,
		delta:         String,
		partial:       Arc<AssistantMessage>,
	},
	ThinkingEnd {
		content_index: usize,
		content:       String,
		partial:       Arc<AssistantMessage>,
	},
	ImageEnd {
		content_index: usize,
		content:       ImageContent,
		partial:       Arc<AssistantMessage>,
	},
	ToolcallStart {
		content_index: usize,
		partial:       Arc<AssistantMessage>,
	},
	ToolcallDelta {
		content_index: usize,
		delta:         String,
		partial:       Arc<AssistantMessage>,
	},
	ToolcallEnd {
		content_index: usize,
		tool_call:     ToolCall,
		partial:       Arc<AssistantMessage>,
	},
	Done {
		reason:  DoneReason,
		message: Arc<AssistantMessage>,
	},
	Error {
		reason: ErrorReason,
		error:  Arc<AssistantMessage>,
	},
}

impl AssistantMessageEvent {
	/// True for the two terminal variants (`done` / `error`).
	#[must_use]
	pub const fn is_terminal(&self) -> bool {
		matches!(self, Self::Done { .. } | Self::Error { .. })
	}

	/// The terminal message carried by `done`/`error`, if this is one.
	#[must_use]
	pub const fn terminal_message(&self) -> Option<&Arc<AssistantMessage>> {
		match self {
			Self::Done { message, .. } => Some(message),
			Self::Error { error, .. } => Some(error),
			_ => None,
		}
	}
}
