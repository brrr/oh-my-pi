//! `pi-tools` — pure-Rust tool framework for the headless omp control plane
//! (WP-1.2 C1).
//!
//! Holds the pinned tool-execution contract ([`Tool`] / [`ToolResult`] /
//! [`ToolError`]), output truncation helpers ([`truncate`]), hashline format
//! primitives ([`hashline`]), and the copied tool description prompts
//! ([`prompts`]). No N-API dependency: this is a pure-core crate guarded by
//! `scripts/check-pure-core-no-napi.sh`.
//!
//! Concrete tool implementations (read/write/replace/bash/grep/glob) and the
//! streaming `on_update` callback land in later work packages; this crate
//! provides the framework they slot into.

pub mod hashline;
pub mod prompts;
pub mod tool;
pub mod truncate;

pub use tool::{Tool, ToolError, ToolResult};
pub use truncate::{
	ByteTruncationResult, TruncateOptions, TruncationResult, truncate_head, truncate_head_bytes,
};
