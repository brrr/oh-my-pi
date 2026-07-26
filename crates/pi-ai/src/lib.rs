//! `pi-ai` — pure-Rust provider layer for the headless omp control plane.
//!
//! WP-1.1a scope: Anthropic Messages wire types, the non-streaming client, the
//! minimal auth/config surface, and the pinned provider↔loop event interface
//! (see `docs/omp-headless/provider-event-contract.md`). WP-1.1b adds the real
//! SSE streaming path (`sse` + `builder`) behind the same interface, with
//! mid-stream cancellation via `Client::stream_with_cancel`.

pub mod auth;
pub mod builder;
pub mod client;
pub mod convert;
pub mod error;
pub mod event;
pub mod json_repair;
pub mod message;
pub mod schema;
pub mod sse;
pub mod stream;
pub mod stream_runner;
pub mod wire;

pub use auth::AnthropicAuthConfig;
pub use client::Client;
pub use error::AiError;
pub use event::AssistantMessageEvent;
pub use message::{AssistantMessage, ImageContent, TextContent, UserContentBlock};
pub use schema::normalize_anthropic_tool_schema;
pub use stream::AssistantMessageEventStream;
