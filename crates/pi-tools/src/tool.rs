//! The pinned tool-execution contract: [`Tool`], [`ToolResult`], [`ToolError`].
//!
//! Rust mirror of the coding-agent tool surface
//! (`packages/coding-agent/src/tools/*`): a tool declares a name, a
//! human-facing description (the raw prompt `.md`, no template rendering at
//! this layer — see the `prompts/` directory and WP-1.4), and a wire
//! `input_schema` (the *un-normalized* shape; Anthropic cleanup happens in
//! [`pi_ai::normalize_anthropic_tool_schema`]). Execution returns a
//! [`ToolResult`] or raises a [`ToolError`] whose `Display` is the exact text
//! surfaced to the model (`ToolError.render()`,
//! `packages/coding-agent/src/tools/tool-errors.ts:22`).

use std::fmt;

use pi_ai::{TextContent, UserContentBlock};
use pi_shell::cancel::CancelToken;
use serde_json::Value;

/// Outcome of a successful tool run.
///
/// Mirrors the `ToolResult` fields threaded through the TS tool loop:
/// `content` is what the model sees (a list of user content blocks), `details`
/// is opaque UI/log-only metadata, `is_error` marks a soft failure returned as
/// a normal turn (as opposed to a thrown [`ToolError`]), and `useless` marks a
/// result the compactor may drop wholesale (e.g. a zero-match search).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolResult {
	/// Blocks handed to the model as the tool's reply.
	pub content:  Vec<UserContentBlock>,
	/// UI/log-only payload; never sent to the model.
	pub details:  Option<Value>,
	/// Soft error: a normal turn the model should treat as a failure.
	pub is_error: bool,
	/// Droppable by the compactor without information loss.
	pub useless:  bool,
}

impl ToolResult {
	/// Build a result from a single [`TextContent`] block.
	#[must_use]
	pub fn text(text: impl Into<String>) -> Self {
		Self {
			content:  vec![UserContentBlock::Text(TextContent {
				text:           text.into(),
				text_signature: None,
			})],
			details:  None,
			is_error: false,
			useless:  false,
		}
	}

	/// Attach opaque UI/log-only details.
	#[must_use]
	pub fn with_details(mut self, details: impl Into<Value>) -> Self {
		self.details = Some(details.into());
		self
	}

	/// Mark this result as a soft error.
	#[must_use]
	pub const fn error(mut self) -> Self {
		self.is_error = true;
		self
	}

	/// Mark this result as compactor-droppable.
	#[must_use]
	pub const fn useless(mut self) -> Self {
		self.useless = true;
		self
	}
}

/// A tool execution failure.
///
/// `Display` yields the model-facing error text, matching `ToolError.render()`
/// in the TS loop. `context` is optional structured metadata for
/// logs/telemetry, never rendered to the model.
#[derive(Debug, Clone)]
pub struct ToolError {
	message: String,
	context: Option<Value>,
}

impl ToolError {
	/// Construct an error carrying the given model-facing message.
	#[must_use]
	pub fn new(message: impl Into<String>) -> Self {
		Self { message: message.into(), context: None }
	}

	/// Attach structured, log-only context.
	#[must_use]
	pub fn with_context(mut self, context: impl Into<Value>) -> Self {
		self.context = Some(context.into());
		self
	}

	/// The raw message (equivalent to `Error.message` before `render()`).
	#[must_use]
	pub fn message(&self) -> &str {
		&self.message
	}

	/// The structured context, if any.
	#[must_use]
	pub const fn context(&self) -> Option<&Value> {
		self.context.as_ref()
	}
}

impl fmt::Display for ToolError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.message)
	}
}

impl std::error::Error for ToolError {}

impl From<&str> for ToolError {
	fn from(message: &str) -> Self {
		Self::new(message)
	}
}

impl From<String> for ToolError {
	fn from(message: String) -> Self {
		Self::new(message)
	}
}

/// A callable tool.
///
/// `input_schema` returns the *wire* schema in its pre-normalization shape;
/// Anthropic-specific cleanup is applied downstream by
/// [`pi_ai::normalize_anthropic_tool_schema`]. `description` returns the raw
/// prompt text (no template rendering at this layer).
///
/// Streaming `on_update` callbacks are intentionally absent in this WP; they
/// are a WP-1.4 prerequisite and will extend `execute` then.
///
/// `execute` returns `impl Future + Send` (rather than a bare `async fn`) so
/// the trait is dyn-compatible through the [`crate::erased::DynTool`]
/// boxed-future wrapper the WP-1.4 heterogeneous registry needs. Impls may
/// still write `async fn execute` — the desugaring satisfies the `+ Send` bound
/// as long as the body holds nothing non-`Send` across an await.
pub trait Tool: Send + Sync {
	/// Stable wire name of the tool.
	fn name(&self) -> &'static str;

	/// Raw description text (the copied prompt `.md`, no rendering).
	fn description(&self) -> &str;

	/// Wire `input_schema` in its pre-normalization shape.
	fn input_schema(&self) -> Value;

	/// Execute the tool with parsed `args`, honoring `ct` for cancellation.
	fn execute(
		&self,
		tool_call_id: &str,
		args: Value,
		ct: &CancelToken,
	) -> impl std::future::Future<Output = Result<ToolResult, ToolError>> + Send;
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	#[test]
	fn tool_result_builder_chains() {
		let ok = ToolResult::text("hi").with_details(json!({ "k": 1 }));
		assert_eq!(ok.content, vec![UserContentBlock::Text(TextContent {
			text:           "hi".to_owned(),
			text_signature: None,
		})]);
		assert_eq!(ok.details, Some(json!({ "k": 1 })));
		assert!(!ok.is_error);
		assert!(!ok.useless);

		let bad = ToolResult::text("no matches").error().useless();
		assert!(bad.is_error);
		assert!(bad.useless);
	}

	#[test]
	fn tool_error_renders_message() {
		let e = ToolError::new("boom").with_context(json!({ "path": "/x" }));
		assert_eq!(e.to_string(), "boom");
		assert_eq!(e.message(), "boom");
		assert_eq!(e.context(), Some(&json!({ "path": "/x" })));
		let from_str: ToolError = "oops".into();
		assert_eq!(from_str.to_string(), "oops");
	}

	struct Echo;

	impl Tool for Echo {
		fn name(&self) -> &'static str {
			"echo"
		}

		#[allow(
			clippy::unnecessary_literal_bound,
			reason = "trait signature returns &str, not &'static str"
		)]
		fn description(&self) -> &str {
			"echoes its args"
		}

		fn input_schema(&self) -> Value {
			json!({ "type": "object", "properties": { "msg": { "type": "string" } } })
		}

		async fn execute(
			&self,
			_id: &str,
			args: Value,
			_ct: &CancelToken,
		) -> Result<ToolResult, ToolError> {
			let msg = args
				.get("msg")
				.and_then(Value::as_str)
				.ok_or_else(|| ToolError::new("missing msg"))?;
			Ok(ToolResult::text(msg))
		}
	}

	#[tokio::test]
	async fn tool_execute_round_trips() {
		let tool = Echo;
		assert_eq!(tool.name(), "echo");
		let ct = CancelToken::default();
		let out = tool
			.execute("call-1", json!({ "msg": "pong" }), &ct)
			.await
			.expect("ok");
		assert_eq!(out.content, ToolResult::text("pong").content);

		let err = tool
			.execute("call-2", json!({}), &ct)
			.await
			.expect_err("missing msg");
		assert_eq!(err.to_string(), "missing msg");
	}
}
