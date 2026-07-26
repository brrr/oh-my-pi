//! `omp-headless` — 把 pi-agent loop 包成 **stdio ACP standard
//! agent**（WP-1.5）。
//!
//! 无头纯 Rust omp 的控制面出口：进程经 stdin/stdout 讲 ACP（Agent Client
//! Protocol）JSON-RPC，可被任何 ACP client（Stage 0 barm-driver / 编辑器宿主）
//! 驱动。处理点 `initialize` / `session/new` / `session/load` / `session/list`
//! / `session/prompt` / `session/cancel` 织在 [`server`] 里；[`mapping`] 把
//! pi-agent `AgentEvent` 流翻成 ACP `session/update`；[`config`] 解析 system
//! prompt + `DeepSeek` LLM。
//!
//! ## MCP client（WP-1.7 接通）
//! `session/new` 的 `mcpServers`（Http 变体）真连：`initialize` + `tools/list`
//! → 生成 [`mcp::McpTool`] 存 session，prompt 时与六内置工具合并注册进 loop
//! （重名内置优先并 warn）。Sse/Stdio 变体 warn 忽略（defer）；连接失败 warn +
//! 该 server 工具缺席（不崩 session）。见 [`mcp`]。
//!
//! ## Session 生命周期（WP-1.6 段 3 接通）
//! `session/load`（按 id 定位 journal + resume + 历史续跑）与 `session/list`
//! （列目录 id/title/timestamp/cwd）已实现并声明能力；`session/fork`·`resume`·
//! `close`·`delete` 仍 defer（不声明、不处理）。
//!
//! ## Deferred（登记，不实现）
//! - **`request_permission` 发起**：Stage 2 与 `acp-permission` 汇合后才发起
//!   agent→client 的权限请求；本 WP 不发。
//! - **image·embeddedContext** prompt content block：`promptCapabilities`
//!   不声明， 只取 text block。
//! - **plan update**：无来源（pi-agent 无 todo/plan 事件），不产。
//!
//! No N-API：纯核心 crate，由 `scripts/check-pure-core-no-napi.sh` 守门。

mod config;
mod mapping;
mod mcp;
mod server;

use anyhow::{Result, bail};
use clap::Parser;

/// headless omp ACP agent 命令行。
#[derive(Parser)]
#[command(name = "omp-headless", about = "headless omp as a stdio ACP standard agent (WP-1.5)")]
struct Cli {
	/// 运行模式；当前仅 `acp`（stdio ACP agent），其他值报错。
	#[arg(long, default_value = "acp")]
	mode:          String,
	/// system prompt：文件路径或字面文本（优先于
	/// `OMP_HEADLESS_SYSTEM_PROMPT`）。
	#[arg(long)]
	system_prompt: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
	let cli = Cli::parse();
	if cli.mode != "acp" {
		bail!("unsupported --mode {:?}: 当前仅支持 acp", cli.mode);
	}
	let system_prompt = config::resolve_system_prompt(cli.system_prompt.as_deref())?;
	server::run(system_prompt).await
}
