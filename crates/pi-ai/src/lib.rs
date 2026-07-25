//! `pi-ai` — pure-Rust provider layer for the headless omp control plane.
//!
//! WP-1.1a scope: Anthropic Messages wire types, the non-streaming client, the
//! minimal auth/config surface, and the pinned provider↔loop event interface
//! (see `docs/omp-headless/provider-event-contract.md`). SSE streaming lands in
//! WP-1.1b behind the same interface.

pub mod error;
pub mod wire;

pub use error::AiError;
