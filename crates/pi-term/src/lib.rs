//! Terminal-domain pure logic for the Oh My Pi toolchain.
//!
//! Extracted from `pi-natives` as a pure Rust core with no N-API dependency;
//! `pi-natives` keeps only a thin napi shell over these engines.
//!
//! # Modules
//! - [`keys`] — Kitty keyboard protocol sequence parsing and key matching.
//! - [`text`] — ANSI-aware UTF-16 text measurement and slicing. The data plane
//!   is `&[u16]`/`Vec<u16>`, the exact value space of JS strings.
//! - [`snapcompact`] — snapcompact frame rasterization (bundled pixel/TrueType
//!   fonts) and PNG encoding.

pub mod keys;
pub mod snapcompact;
pub mod text;
