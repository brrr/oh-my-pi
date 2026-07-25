//! Error taxonomy for the provider layer.
//!
//! Mirrors the split used by the TS client (`packages/ai/src/error/classes.ts`:
//! `AnthropicApiError` / `AnthropicConnectionError` /
//! `AnthropicConnectionTimeoutError`) plus auth-resolution and decode failures
//! that TS surfaces earlier in its stack.

use crate::wire::ApiErrorBody;

#[derive(Debug, thiserror::Error)]
pub enum AiError {
	/// Network-level failure after retries were exhausted.
	#[error("Connection error.")]
	Connection(#[source] reqwest::Error),
	/// No response before the request deadline.
	#[error("Request timed out.")]
	ConnectionTimeout,
	/// Non-2xx response. `message` carries `"<status> <body>"` like the TS
	/// `AnthropicApiError`; `body` is the parsed error envelope when the body
	/// was well-formed JSON.
	#[error("{message}")]
	Api {
		status:     u16,
		message:    String,
		body:       Option<ApiErrorBody>,
		request_id: Option<String>,
	},
	/// Request was cancelled by the caller (reserved for WP-1.1b wiring).
	#[error("Request was aborted.")]
	Aborted,
	/// API key resolution failed (no explicit key, env, or auth store entry).
	#[error("auth: {0}")]
	Auth(String),
	/// Response body did not parse as the expected wire shape.
	#[error("invalid response: {0}")]
	Decode(#[from] serde_json::Error),
	/// The event producer went away without pushing a terminal event
	/// (TS: "Stream ended without a final result").
	#[error("stream ended without a final result")]
	StreamEnded,
}
