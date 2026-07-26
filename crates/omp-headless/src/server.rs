//! ACP server 装配：把 pi-agent loop 包成一个 stdio ACP standard agent。
//!
//! 用 `agent-client-protocol` 的 builder（per-request handler 链，无 `trait
//! Agent`）注册处理点：`initialize` / `session/new` / `session/load` /
//! `session/list` / `session/prompt` / `session/cancel`。prompt 处理点把重活
//! `Handle::spawn` 到 tokio 运行时——ACP 事件循环在 handler 返回后立刻空出来收
//! 下一帧（含 `session/cancel`），且 pi-agent `agent_loop` 内部的
//! `tokio::spawn` 在该 spawned future 里有环境 runtime 可用。
//!
//! ## Session 生命周期（WP-1.6 段 3）
//! - `session/new`：建 journal，ACP sessionId ≡ journal header
//!   id（文件名后缀）。
//! - `session/load`（B1）：按 id 在磁盘定位 journal，resume writer 续接 leaf；
//!   后续 prompt 从 journal 载回历史续跑（B3）。
//! - `session/list`（B2）：列目录里每个 journal 的 id/title/timestamp/cwd。
//! - **defer**：`session/fork`（ACP unstable + barm
//!   暂无需求）、`session/resume`、 `session/close`、`session/delete`
//!   均不声明、不实现（capabilities 亦不声明）。
//!
//! ## 并发模型
//! handler 回调跑在 ACP 事件循环上，阻塞期间不收新帧（crate 文档明示）。故
//! prompt 不 inline await 整个 loop，而是 `Handle::spawn` 后即返回 `Ok(())`
//! （responder 随 spawned future 一起搬走、跑完再 `respond`）。`session/cancel`
//! 于是能在 prompt 进行中被送达，触发该 session 的 `CancelToken`。

use std::{
	collections::HashMap,
	path::{Path, PathBuf},
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, Ordering},
	},
	time::{SystemTime, UNIX_EPOCH},
};

use acp::{
	Agent, Client, ConnectionTo, Responder, Stdio,
	schema::v1::{
		AgentCapabilities, CancelNotification, ContentBlock, InitializeRequest, InitializeResponse,
		ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse,
		McpCapabilities, McpServer, NewSessionRequest, NewSessionResponse, PromptRequest,
		PromptResponse, SessionCapabilities, SessionInfo, SessionListCapabilities,
		SessionNotification,
	},
};
use agent_client_protocol as acp;
use anyhow::Result;
use pi_agent::{AgentConfig, AgentContext, AgentEvent, agent_loop, client_stream_fn};
use pi_ai::message::{Message, MessageAttribution, UserContent, UserMessage};
use pi_session::{ContextMessage, SessionWriter, load_session_context};
use pi_shell::cancel::{AbortReason, AbortToken, CancelToken};
use pi_tools::{BashTool, DynTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};

use crate::{
	config::LlmConfig,
	mapping::{map_event, resolve_stop_reason},
	mcp::{self, McpTool},
};

/// 每个 ACP session 的宿主状态（对应一份 pi-session journal + 取消句柄）。
struct SessionState {
	/// newSession 的 cwd（工具作用域 + prompt 消息落盘上下文）。
	cwd:              PathBuf,
	/// 追加式 session journal。
	writer:           Mutex<SessionWriter>,
	/// 当前在跑的 prompt 的取消句柄（空 = 无在途 turn）。
	cancel:           Mutex<Option<AbortToken>>,
	/// 是否收到过 `session/cancel`（决定 `stop_reason` 是否落 `Cancelled`）。
	cancel_requested: AtomicBool,
	/// session/new 时接通的 MCP 工具（每轮 prompt clone 进 loop 注册表）。
	mcp_tools:        Vec<McpTool>,
}

impl SessionState {
	fn arm_cancel(&self, token: AbortToken) {
		*self.cancel.lock().expect("cancel mutex poisoned") = Some(token);
		self.cancel_requested.store(false, Ordering::SeqCst);
	}

	fn disarm_cancel(&self) {
		*self.cancel.lock().expect("cancel mutex poisoned") = None;
	}

	fn request_cancel(&self) {
		self.cancel_requested.store(true, Ordering::SeqCst);
		if let Some(token) = self.cancel.lock().expect("cancel mutex poisoned").as_ref() {
			token.abort(AbortReason::User);
		}
	}

	fn was_cancel_requested(&self) -> bool {
		self.cancel_requested.load(Ordering::SeqCst)
	}
}

/// 进程级共享状态：LLM 配置 + system prompt + session 表 + tokio handle。
struct AppState {
	llm:           LlmConfig,
	system_prompt: String,
	sessions:      Mutex<HashMap<String, Arc<SessionState>>>,
	runtime:       tokio::runtime::Handle,
}

impl AppState {
	fn get_session(&self, session_id: &str) -> Option<Arc<SessionState>> {
		self
			.sessions
			.lock()
			.expect("sessions mutex poisoned")
			.get(session_id)
			.map(Arc::clone)
	}
}

/// session journal 目录：`$OMP_HEADLESS_SESSION_DIR` 或
/// `<cwd>/.omp-headless/sessions`。
fn session_dir(cwd: &Path) -> PathBuf {
	std::env::var_os("OMP_HEADLESS_SESSION_DIR")
		.map_or_else(|| cwd.join(".omp-headless").join("sessions"), PathBuf::from)
}

fn now_ms() -> i64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |d| d.as_millis() as i64)
}

/// list/load 用的 session 目录：`$OMP_HEADLESS_SESSION_DIR` 优先，否则由请求
/// cwd 推导 `<cwd>/.omp-headless/sessions`；两者皆无则 `None`（无从列举）。
fn list_dir(cwd: Option<&Path>) -> Option<PathBuf> {
	if let Some(dir) = std::env::var_os("OMP_HEADLESS_SESSION_DIR") {
		return Some(PathBuf::from(dir));
	}
	cwd.map(|c| c.join(".omp-headless").join("sessions"))
}

/// 一条 session 的轻量元数据（只读 header + title slot，不解析全文）。
struct SessionMeta {
	id:         String,
	cwd:        String,
	title:      Option<String>,
	path:       PathBuf,
	updated_ms: i64,
}

/// 读 journal 的头部：peel title slot（首物理行）+ 解析 header
/// （id/cwd/title）。`updated_ms` 取文件 mtime。整文件不读入内存。
fn read_session_meta(path: &Path) -> Option<SessionMeta> {
	use std::io::{BufRead, BufReader};

	let file = std::fs::File::open(path).ok()?;
	let mut reader = BufReader::new(file);

	let mut first = String::new();
	reader.read_line(&mut first).ok()?;
	let (slot_title, header_line) = match pi_session::parse_title_slot_line(first.trim()) {
		Some(slot) => {
			let mut second = String::new();
			reader.read_line(&mut second).ok()?;
			(Some(slot.title), second)
		},
		None => (None, first),
	};

	let header: serde_json::Value = serde_json::from_str(header_line.trim()).ok()?;
	if header.get("type").and_then(serde_json::Value::as_str) != Some("session") {
		return None;
	}
	let id = header
		.get("id")
		.and_then(serde_json::Value::as_str)?
		.to_string();
	let cwd = header
		.get("cwd")
		.and_then(serde_json::Value::as_str)
		.unwrap_or("")
		.to_string();
	let title = slot_title.filter(|t| !t.is_empty()).or_else(|| {
		header
			.get("title")
			.and_then(serde_json::Value::as_str)
			.map(str::to_string)
	});

	let updated_ms = std::fs::metadata(path)
		.and_then(|m| m.modified())
		.ok()
		.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
		.map_or(0, |d| d.as_millis() as i64);

	Some(SessionMeta { id, cwd, title, path: path.to_path_buf(), updated_ms })
}

/// 目录里所有 `*.jsonl` journal 的元数据（跳过读不动的文件）。
fn read_all_metas(dir: &Path) -> Vec<SessionMeta> {
	let Ok(entries) = std::fs::read_dir(dir) else {
		return Vec::new();
	};
	let mut metas = Vec::new();
	for entry in entries.flatten() {
		let path = entry.path();
		if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
			continue;
		}
		if let Some(meta) = read_session_meta(&path) {
			metas.push(meta);
		}
	}
	metas
}

/// 按 sessionId（≡ header id）在目录里定位 journal 文件。
fn find_session_file(dir: &Path, session_id: &str) -> Option<PathBuf> {
	read_all_metas(dir)
		.into_iter()
		.find(|m| m.id == session_id)
		.map(|m| m.path)
}

/// 目录里所有 session 的 [`SessionInfo`]，按最近修改倒序。
fn list_session_infos(dir: &Path) -> Vec<SessionInfo> {
	let mut metas = read_all_metas(dir);
	metas.sort_by_key(|m| std::cmp::Reverse(m.updated_ms));
	metas
		.into_iter()
		.map(|m| {
			SessionInfo::new(m.id, m.cwd)
				.title(m.title)
				.updated_at(pi_session::time::unix_ms_to_iso(m.updated_ms))
		})
		.collect()
}

/// 从 journal 载回该 session 之前的消息作为续跑上下文（B1/B3）。只取标准
/// [`Message`]；合成消息（compactionSummary / branchSummary / custom）在无头续
/// 跑里跳过。
fn load_prior_messages(path: &Path) -> Vec<Message> {
	let Ok(ctx) = load_session_context(path) else {
		return Vec::new();
	};
	ctx.messages
		.into_iter()
		.filter_map(|m| match m {
			ContextMessage::Standard(msg) => Some(msg),
			_ => None,
		})
		.collect()
}

/// 六工具（read/write/edit/bash/grep/glob），全部作用域到 session cwd。
fn build_tools(cwd: &Path) -> Vec<Box<dyn DynTool>> {
	vec![
		Box::new(ReadTool::new(cwd)),
		Box::new(WriteTool::new(cwd)),
		Box::new(EditTool::new(cwd)),
		Box::new(BashTool::new(cwd)),
		Box::new(GrepTool::new(cwd)),
		Box::new(GlobTool::new(cwd)),
	]
}

/// 合并六内置工具 + session 的 MCP 工具（重名以内置优先并 warn）。
fn merge_tools(session: &SessionState) -> Vec<Box<dyn DynTool>> {
	let mut tools = build_tools(&session.cwd);
	let builtin: std::collections::HashSet<&'static str> = tools.iter().map(|t| t.name()).collect();
	for mcp_tool in &session.mcp_tools {
		if builtin.contains(mcp_tool.name()) {
			eprintln!("[omp-headless] MCP 工具 {:?} 与内置同名，内置优先，跳过", mcp_tool.name());
			continue;
		}
		tools.push(Box::new(mcp_tool.clone()));
	}
	tools
}

/// 取 prompt 里所有 text block 拼成一段（image/embeddedContext defer）。
fn extract_prompt_text(blocks: &[ContentBlock]) -> String {
	let mut text = String::new();
	for block in blocks {
		if let ContentBlock::Text(content) = block {
			text.push_str(&content.text);
		}
	}
	text
}

/// 启动 ACP server（阻塞至 stdio EOF）。
///
/// # Errors
///
/// LLM 配置解析失败、或 ACP 传输层错误。
pub async fn run(system_prompt: String) -> Result<()> {
	let llm = LlmConfig::resolve()?;
	let state = Arc::new(AppState {
		llm,
		system_prompt,
		sessions: Mutex::new(HashMap::new()),
		runtime: tokio::runtime::Handle::current(),
	});

	let new_state = Arc::clone(&state);
	let load_state = Arc::clone(&state);
	let list_state = Arc::clone(&state);
	let prompt_state = Arc::clone(&state);
	let cancel_state = Arc::clone(&state);

	Agent
		.builder()
		.name("omp-headless")
		.on_receive_request(
			async move |req: InitializeRequest, responder: Responder<InitializeResponse>, _cx| {
				// 诚实 capabilities：声明 loadSession（B1）+ sessionCapabilities.list
				// （B2）；fork/resume/close/delete 均**不**声明（defer，见模块文档）。
				// promptCapabilities 不声明 image/embeddedContext；mcpCapabilities 声明
				// http（WP-1.7 接通 HTTP MCP），不声明 sse/acp（defer）。
				responder.respond(
					InitializeResponse::new(req.protocol_version).agent_capabilities(
						AgentCapabilities::new()
							.load_session(true)
							.mcp_capabilities(McpCapabilities::new().http(true))
							.session_capabilities(
								SessionCapabilities::new().list(SessionListCapabilities::new()),
							),
					),
				)
			},
			acp::on_receive_request!(),
		)
		.on_receive_request(
			async move |req: NewSessionRequest, responder: Responder<NewSessionResponse>, _cx| {
				let state = Arc::clone(&new_state);
				handle_new_session(&state, req, responder).await
			},
			acp::on_receive_request!(),
		)
		.on_receive_request(
			async move |req: LoadSessionRequest, responder: Responder<LoadSessionResponse>, _cx| {
				let state = Arc::clone(&load_state);
				handle_load_session(&state, req, responder).await
			},
			acp::on_receive_request!(),
		)
		.on_receive_request(
			async move |req: ListSessionsRequest, responder: Responder<ListSessionsResponse>, _cx| {
				let state = Arc::clone(&list_state);
				handle_list_sessions(&state, req, responder)
			},
			acp::on_receive_request!(),
		)
		.on_receive_request(
			async move |req: PromptRequest, responder: Responder<PromptResponse>, cx| {
				let state = Arc::clone(&prompt_state);
				// 重活搬到 tokio runtime：ACP 事件循环立刻空出来收 cancel；
				// agent_loop 内部 tokio::spawn 在此 spawned future 里有 runtime。
				state.runtime.clone().spawn(async move {
					run_prompt_turn(&state, req, responder, cx).await;
				});
				Ok(())
			},
			acp::on_receive_request!(),
		)
		.on_receive_notification(
			async move |notif: CancelNotification, _cx| {
				let session_id = notif.session_id.0.to_string();
				if let Some(session) = cancel_state.get_session(&session_id) {
					session.request_cancel();
				}
				Ok(())
			},
			acp::on_receive_notification!(),
		)
		.connect_to(Stdio::new())
		.await
		.map_err(|error| anyhow::anyhow!("ACP 传输层错误: {error:?}"))
}

/// `session/new`：建 pi-session journal + 接通 MCP servers + 生成 sessionId +
/// 登记。
///
/// MCP 接线（WP-1.7）：对每个 `Http` server 跑 `initialize` +
/// `tools/list`，成功 则把 [`McpTool`] 收进 session；连接失败仅 stderr warn +
/// 该 server 工具缺席（不崩 session）。`Sse`/`Stdio` 变体 warn 忽略（defer）。
async fn handle_new_session(
	state: &Arc<AppState>,
	req: NewSessionRequest,
	responder: Responder<NewSessionResponse>,
) -> Result<(), acp::Error> {
	let cwd = req.cwd;
	let dir = session_dir(&cwd);
	let writer = SessionWriter::create(&cwd.to_string_lossy(), &dir).map_err(|error| {
		acp::Error::internal_error().data(format!("建 session journal 失败: {error}"))
	})?;

	let mcp_tools = connect_mcp_servers(&req.mcp_servers).await;

	// ACP sessionId ≡ pi-session 文件 header id（也是文件名后缀），使 loadSession /
	// list 可按同一 id 定位磁盘上的 journal。
	let session_id = writer.session_id().to_string();
	let session = Arc::new(SessionState {
		cwd,
		writer: Mutex::new(writer),
		cancel: Mutex::new(None),
		cancel_requested: AtomicBool::new(false),
		mcp_tools,
	});
	state
		.sessions
		.lock()
		.expect("sessions mutex poisoned")
		.insert(session_id.clone(), session);

	responder.respond(NewSessionResponse::new(session_id))
}

/// `session/load`（B1）：按 sessionId 在磁盘定位 journal → resume writer（leaf
/// 续接）→ 登记 session，后续 prompt 在其历史上下文续跑（B3）。找不到文件即
/// 报错。
async fn handle_load_session(
	state: &Arc<AppState>,
	req: LoadSessionRequest,
	responder: Responder<LoadSessionResponse>,
) -> Result<(), acp::Error> {
	let session_id = req.session_id.0.to_string();
	let cwd = req.cwd;
	let dir = session_dir(&cwd);

	let Some(path) = find_session_file(&dir, &session_id) else {
		return responder.respond_with_internal_error(format!(
			"未找到 session {session_id}（目录 {}）",
			dir.display()
		));
	};

	let writer = SessionWriter::resume(&path).map_err(|error| {
		acp::Error::internal_error().data(format!("resume session journal 失败: {error}"))
	})?;
	let mcp_tools = connect_mcp_servers(&req.mcp_servers).await;

	let session = Arc::new(SessionState {
		cwd,
		writer: Mutex::new(writer),
		cancel: Mutex::new(None),
		cancel_requested: AtomicBool::new(false),
		mcp_tools,
	});
	state
		.sessions
		.lock()
		.expect("sessions mutex poisoned")
		.insert(session_id, session);

	responder.respond(LoadSessionResponse::new())
}

/// `session/list`（B2）：列 session 目录里每个 journal 的
/// id/title/timestamp/cwd（从 header + title slot 读）。按 mtime 倒序，不分页
/// （cursor defer）。
fn handle_list_sessions(
	state: &Arc<AppState>,
	req: ListSessionsRequest,
	responder: Responder<ListSessionsResponse>,
) -> Result<(), acp::Error> {
	// 收尾 in-flight 写，保证列表反映最新落盘状态。
	for session in state
		.sessions
		.lock()
		.expect("sessions mutex poisoned")
		.values()
	{
		let _ = session
			.writer
			.lock()
			.expect("session writer mutex poisoned")
			.flush();
	}

	let Some(dir) = list_dir(req.cwd.as_deref()) else {
		return responder.respond(ListSessionsResponse::new(Vec::new()));
	};
	let infos = list_session_infos(&dir);
	responder.respond(ListSessionsResponse::new(infos))
}

/// 接通 `mcpServers` 里的每个 HTTP server，汇总其工具（容错：单 server
/// 失败不影响 其余，也不崩 session）。
async fn connect_mcp_servers(servers: &[McpServer]) -> Vec<McpTool> {
	let mut tools = Vec::new();
	for server in servers {
		match server {
			McpServer::Http(http) => {
				let headers = http
					.headers
					.iter()
					.map(|h| (h.name.clone(), h.value.clone()))
					.collect();
				match mcp::connect_http(&http.url, headers).await {
					Ok(server_tools) => {
						eprintln!(
							"[omp-headless] MCP server {:?} 接通，注入 {} 个工具",
							http.name,
							server_tools.len()
						);
						tools.extend(server_tools);
					},
					Err(error) => eprintln!(
						"[omp-headless] MCP server {:?}（{}）连接失败，忽略其工具: {error:#}",
						http.name, http.url
					),
				}
			},
			McpServer::Sse(sse) => {
				eprintln!(
					"[omp-headless] MCP server {:?} 为 SSE 变体，暂不支持（defer），忽略",
					sse.name
				);
			},
			other => {
				eprintln!(
					"[omp-headless] MCP server {other:?} 非 HTTP 变体（Stdio \
					 等），暂不支持（defer），忽略"
				);
			},
		}
	}
	tools
}

/// `session/prompt` 的实体：跑一轮 `agent_loop`，流式回吐 update，收尾
/// respond。
async fn run_prompt_turn(
	state: &Arc<AppState>,
	req: PromptRequest,
	responder: Responder<PromptResponse>,
	cx: ConnectionTo<Client>,
) {
	let session_id = req.session_id.0.to_string();
	let Some(session) = state.get_session(&session_id) else {
		let _ = responder.respond_with_internal_error(format!("未知 session: {session_id}"));
		return;
	};

	let prompt_text = extract_prompt_text(&req.prompt);
	let prompt_message = Message::User(UserMessage {
		content:          UserContent::Text(prompt_text),
		synthetic:        None,
		steering:         None,
		attribution:      Some(MessageAttribution::User),
		provider_payload: None,
		timestamp:        now_ms(),
	});

	// 可被 cancel 的 token：emplace_abort_token 装入 flag 后 clone 进 loop。
	let mut cancel_token = CancelToken::new(None);
	let abort = cancel_token.emplace_abort_token();
	session.arm_cancel(abort);

	// 续跑上下文（B1/B3）：从 journal 载回本 session 之前的全部消息。fresh
	// session 为空；loadSession/多轮下含历史，agent 于是在完整上下文里续跑。
	let prior_messages = {
		let path = session
			.writer
			.lock()
			.expect("session writer mutex poisoned")
			.path()
			.to_path_buf();
		load_prior_messages(&path)
	};

	let context = AgentContext {
		system_prompt: vec![state.system_prompt.clone()],
		messages:      prior_messages,
		tools:         merge_tools(&session),
	};
	let client = state.llm.client();
	let config = AgentConfig::new(state.llm.model.clone(), state.llm.max_tokens).with_stream_fn(
		client_stream_fn(client, state.llm.model.clone(), state.llm.max_tokens, None),
	);

	let mut stream = agent_loop(vec![prompt_message], context, config, cancel_token);

	let mut final_messages: Vec<Message> = Vec::new();
	while let Some(event) = stream.next().await {
		if let AgentEvent::AgentEnd { messages } = &event {
			final_messages.clone_from(messages);
		}
		for update in map_event(&event) {
			let _ = cx.send_notification(SessionNotification::new(session_id.clone(), update));
		}
	}

	// 宿主拼装层：把本轮新消息（user prompt + assistant + toolResult）落 journal。
	{
		let mut writer = session
			.writer
			.lock()
			.expect("session writer mutex poisoned");
		for message in &final_messages {
			let _ = writer.append_message(message.clone());
		}
		let _ = writer.flush();
	}

	session.disarm_cancel();
	let stop = resolve_stop_reason(&final_messages, session.was_cancel_requested());
	let _ = responder.respond(PromptResponse::new(stop));
}

#[cfg(test)]
mod tests {
	use pi_ai::message::{Message, UserContent, UserMessage};
	use pi_session::{SessionWriter, load_entries_from_file, message_from_json};

	use super::{
		find_session_file, list_dir, list_session_infos, load_prior_messages, read_session_meta,
	};

	fn tmp(tag: &str) -> std::path::PathBuf {
		let dir = std::env::temp_dir().join(format!("omp-headless-{tag}-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		std::fs::create_dir_all(&dir).unwrap();
		dir
	}

	fn user(text: &str) -> Message {
		Message::User(UserMessage {
			content:          UserContent::Text(text.to_string()),
			synthetic:        None,
			steering:         None,
			attribution:      None,
			provider_payload: None,
			timestamp:        1,
		})
	}

	fn assistant(text: &str) -> Message {
		message_from_json(serde_json::json!({
			"role":"assistant",
			"content":[{"type":"text","text":text}],
			"api":"x","provider":"p","model":"m",
			"usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,
				"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
			"stopReason":"stop","timestamp":1
		}))
		.unwrap()
	}

	// B1 (context load) + B2 (list/find) + B3 (resume append to same file, leaf
	// continuous) — the full session-continuity machinery, no LLM.
	#[test]
	fn load_resume_list_find_roundtrip() {
		let dir = tmp("roundtrip");
		let cwd = "/tmp/proj";

		// ── Turn 1: fresh session, one user + one assistant. ──
		let (session_id, path) = {
			let mut w = SessionWriter::create(cwd, &dir).unwrap();
			w.append_message(user("remember codeword BANANA")).unwrap();
			w.append_message(assistant("noted: BANANA")).unwrap();
			w.flush().unwrap();
			(w.session_id().to_string(), w.path().to_path_buf())
		};

		// B1: prior messages rebuild from the journal.
		let prior = load_prior_messages(&path);
		assert_eq!(prior.len(), 2, "turn 1 messages load back");

		// B2: list + find by id.
		let infos = list_session_infos(&dir);
		assert_eq!(infos.len(), 1);
		assert_eq!(infos[0].session_id.0.as_ref(), session_id.as_str());
		assert_eq!(infos[0].cwd.to_string_lossy(), cwd);
		assert_eq!(find_session_file(&dir, &session_id).as_deref(), Some(path.as_path()));
		assert!(find_session_file(&dir, "no-such-id").is_none());

		// B3: resume the SAME file (leaf continues) and append turn 2.
		let leaf_before = {
			let w = SessionWriter::resume(&path).unwrap();
			w.leaf_id().map(str::to_string)
		};
		{
			let mut w = SessionWriter::resume(&path).unwrap();
			assert_eq!(w.leaf_id().map(str::to_string), leaf_before, "resume picks up prior leaf");
			w.append_message(user("what is the codeword?")).unwrap();
			w.append_message(assistant("BANANA")).unwrap();
			w.flush().unwrap();
		}

		// All four messages now present, in order, in one continuous file.
		let after = load_prior_messages(&path);
		assert_eq!(after.len(), 4, "turn 1 + turn 2 both in the same journal");

		// Leaf→root parentId chain is continuous across the resume boundary.
		let loaded = load_entries_from_file(&path).unwrap();
		let ids: std::collections::BTreeMap<&str, Option<&str>> = loaded
			.entries
			.iter()
			.filter_map(|e| e.id().map(|id| (id, e.parent_id())))
			.collect();
		let mut cursor = loaded.entries.last().and_then(|e| e.id());
		let mut walked = 0;
		while let Some(id) = cursor {
			walked += 1;
			cursor = ids.get(id).and_then(|p| *p);
		}
		assert_eq!(walked, 4, "single continuous leaf→root chain over all 4 entries");
	}

	#[test]
	fn read_session_meta_fresh_session_has_no_title() {
		let dir = tmp("meta");
		let mut w = SessionWriter::create("/tmp/x", &dir).unwrap();
		w.append_message(user("hi")).unwrap();
		w.flush().unwrap();
		let meta = read_session_meta(w.path()).unwrap();
		assert_eq!(meta.id, w.session_id());
		assert_eq!(meta.cwd, "/tmp/x");
		assert!(meta.title.is_none(), "fresh session writes an empty title slot → None");
	}

	#[test]
	fn list_dir_prefers_env_then_cwd() {
		// Without env, derived from cwd.
		// SAFETY: single-threaded test; no other thread reads this env var here.
		unsafe { std::env::remove_var("OMP_HEADLESS_SESSION_DIR") };
		let cwd = std::path::Path::new("/tmp/proj");
		assert_eq!(list_dir(Some(cwd)), Some(cwd.join(".omp-headless").join("sessions")));
		assert_eq!(list_dir(None), None, "no env + no cwd → nothing to list");
	}
}
