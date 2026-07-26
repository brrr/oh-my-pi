//! `pi-agent` — pure-Rust agent-loop for the headless omp control plane
//! (WP-1.4a).
//!
//! A streaming state machine that drives one assistant turn at a time against a
//! provider ([`pi_ai`]) and runs the turn's tool calls **serially** through a
//! heterogeneous registry ([`pi_tools::DynTool`]). It is a folded port of
//! `packages/agent/src/agent-loop.ts` (2434 lines) — only the control-flow core
//! is carried over; the TS file has grown far past this WP's surface.
//!
//! Entry points: [`agent_loop`] (new prompt) / [`agent_loop_continue`] (resume
//! from existing context). Both return an [`AgentEventStream`] whose terminal
//! event is [`AgentEvent::AgentEnd`]. The provider call is injected via
//! [`AgentConfig::stream_fn`]; [`provider::client_stream_fn`] wraps a real
//! pi-ai [`pi_ai::Client`], and tests inject scripted event sequences.
//!
//! ## Emitted events
//! 8 of the 11 [`AgentEvent`] variants are produced: `agent_start`,
//! `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_update`,
//! `message_end`, `tool_execution_start`, `tool_execution_end`.
//! ([`AgentEvent::ToolExecutionUpdate`] exists for wire-shape parity but is
//! never emitted here.)
//!
//! ## Schema normalization layer
//! `pi-tools` tools return the **un-normalized** `input_schema`, and the pi-ai
//! client serializes `input_schema` opaquely (no cleanup on the client side —
//! verified against `client.rs`/`wire.rs`). So the loop applies
//! [`pi_ai::normalize_anthropic_tool_schema`] exactly **once**, when it builds
//! `LlmContext.tools` in `stream_assistant_response` — no double cleanup.
//!
//! ## Deferred (registered here; not implemented in WP-1.4a)
//! - shared/exclusive tool concurrency scheduling (WP-1.4b) — tools run
//!   serially;
//! - steering / aside / follow-up / IRC / interruptible / pause gate;
//! - `SoftToolRequirement` remind-then-escalate;
//! - GPT-5 Harmony-leak detection + retry;
//! - in-band tool-calling dialects;
//! - `appendOnlyContext` caching;
//! - intent tracing;
//! - telemetry + OTEL spans + run collector/coverage;
//! - deadline enforcement;
//! - `pause_turn` continuation (folded out; the `max_steps` cap guards spins);
//! - `transformContext` / `transformProviderContext` / `convertToLlm` host
//!   layer (Rust `convertToLlm` is identity — messages are already
//!   `pi_ai::Message`);
//! - `beforeToolCall` / `afterToolCall` / `onTurnEnd` hooks;
//! - session persistence (host-assembled, same layering as TS);
//! - streaming `tool_execution_update` emission.
//!
//! No N-API: pure-core crate guarded by `scripts/check-pure-core-no-napi.sh`.

pub mod event;
pub mod execute;
pub mod loop_;
pub mod provider;

pub use event::{AgentEvent, AgentEventSink, AgentEventStream};
pub use execute::{
	SyntheticReason, coerce_tool_result, create_aborted_tool_result,
	create_synthetic_tool_result_message, execute_tool_calls,
};
pub use loop_::{
	AgentConfig, AgentContext, DEFAULT_MAX_STEPS, LlmContext, STREAM_INTERRUPTED_AFTER_CONTENT,
	StreamFn, WireTool, agent_loop, agent_loop_continue, recover_transient_error_tool_turn,
	retain_completed_tool_calls,
};
pub use provider::{client_stream_fn, encode_params};
