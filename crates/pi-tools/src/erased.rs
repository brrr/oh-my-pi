//! Object-safe [`DynTool`] wrapper over the native-async [`Tool`] trait.
//!
//! [`Tool::execute`] is an `async fn` in a trait (`async_fn_in_trait`), which
//! is not object-safe — a `Box<dyn Tool>` cannot be formed. The agent loop
//! needs a heterogeneous registry keyed by tool name (`Vec<Box<dyn DynTool>>`),
//! so this module provides the boxed-future erasure the `tool.rs` doc comment
//! defers to "WP-1.4". `DynTool` mirrors `Tool`'s four methods but returns a
//! `Pin<Box<dyn Future>>` from `execute`; a blanket `impl<T: Tool> DynTool for
//! T` makes every concrete tool usable through the trait object with no
//! per-tool boilerplate.

use std::{future::Future, pin::Pin};

use pi_shell::cancel::CancelToken;
use serde_json::Value;

use crate::tool::{Concurrency, Tool, ToolError, ToolResult};

/// Boxed future returned by [`DynTool::execute`]. Borrows `self`/`ct`/`id` for
/// the duration of the call, matching the native [`Tool::execute`] lifetime.
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>>;

/// Object-safe counterpart of [`Tool`]. Same surface, but `execute` yields a
/// boxed future so `dyn DynTool` is a valid trait object.
pub trait DynTool: Send + Sync {
	/// Stable wire name of the tool.
	fn name(&self) -> &'static str;

	/// Raw description text (the copied prompt `.md`, no rendering).
	fn description(&self) -> &str;

	/// Wire `input_schema` in its pre-normalization shape.
	fn input_schema(&self) -> Value;

	/// Batch-scheduling class (TS `Tool.concurrency`, default `"shared"`).
	fn concurrency(&self) -> Concurrency;

	/// Execute the tool with parsed `args`, honoring `ct` for cancellation.
	fn execute<'a>(
		&'a self,
		tool_call_id: &'a str,
		args: Value,
		ct: &'a CancelToken,
	) -> ToolFuture<'a>;
}

impl<T: Tool> DynTool for T {
	fn name(&self) -> &'static str {
		Tool::name(self)
	}

	fn description(&self) -> &str {
		Tool::description(self)
	}

	fn input_schema(&self) -> Value {
		Tool::input_schema(self)
	}

	fn concurrency(&self) -> Concurrency {
		Tool::concurrency(self)
	}

	fn execute<'a>(
		&'a self,
		tool_call_id: &'a str,
		args: Value,
		ct: &'a CancelToken,
	) -> ToolFuture<'a> {
		Box::pin(Tool::execute(self, tool_call_id, args, ct))
	}
}
