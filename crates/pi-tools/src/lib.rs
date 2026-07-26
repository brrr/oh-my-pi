//! `pi-tools` — pure-Rust tool framework for the headless omp control plane
//! (WP-1.2 C1).
//!
//! Holds the pinned tool-execution contract ([`Tool`] / [`ToolResult`] /
//! [`ToolError`]), output truncation helpers ([`truncate`]), hashline format
//! primitives ([`hashline`]), and the copied tool description prompts
//! ([`prompts`]). No N-API dependency: this is a pure-core crate guarded by
//! `scripts/check-pure-core-no-napi.sh`.
//!
//! Concrete tool implementations (WP-1.2 C2) live in [`tools`]: the six-tool
//! minimal face (read/write/edit/bash/grep/glob) pinned against the TS
//! reference by golden parity tests (`tests/golden_parity.rs`). The streaming
//! `on_update` callback lands in a later work package.

pub mod hashline;
pub mod prompts;
pub mod tool;
pub mod tools;
pub mod truncate;

pub use tool::{Tool, ToolError, ToolResult};
pub use tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};
pub use truncate::{
	ByteTruncationResult, TruncateOptions, TruncationResult, truncate_head, truncate_head_bytes,
};
