//! ACP server 装配：把 pi-agent loop 包成一个 stdio ACP standard agent。
//!
//! 用 `agent-client-protocol` 的 builder（per-request handler 链，无 `trait
//! Agent`）注册四个处理点：`initialize` / `session/new` / `session/prompt` /
//! `session/cancel`。prompt 处理点把重活 `Handle::spawn` 到 tokio 运行时——
//! ACP 事件循环在 handler 返回后立刻空出来收下一帧（含 `session/cancel`），
//! 且 pi-agent `agent_loop` 内部的 `tokio::spawn` 在该 spawned future 里有环境
//! runtime 可用。
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
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	time::{SystemTime, UNIX_EPOCH},
};

use acp::{
	Agent, Client, ConnectionTo, Responder, Stdio,
	schema::v1::{
		AgentCapabilities, CancelNotification, ContentBlock, InitializeRequest, InitializeResponse,
		McpCapabilities, McpServer, NewSessionRequest, NewSessionResponse, PromptRequest,
		PromptResponse, SessionNotification,
	},
};
use agent_client_protocol as acp;
use anyhow::Result;
use pi_agent::{AgentConfig, AgentContext, AgentEvent, agent_loop, client_stream_fn};
use pi_ai::message::{Message, MessageAttribution, UserContent, UserMessage};
use pi_session::SessionWriter;
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
	llm:             LlmConfig,
	system_prompt:   String,
	sessions:        Mutex<HashMap<String, Arc<SessionState>>>,
	runtime:         tokio::runtime::Handle,
	session_counter: AtomicU64,
}

impl AppState {
	/// 新 sessionId（进程内单调 + 毫秒时间戳，够唯一且可读）。
	fn mint_session_id(&self) -> String {
		let n = self.session_counter.fetch_add(1, Ordering::SeqCst);
		format!("omp-{:x}-{n}", now_ms())
	}

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
		session_counter: AtomicU64::new(0),
	});

	let new_state = Arc::clone(&state);
	let prompt_state = Arc::clone(&state);
	let cancel_state = Arc::clone(&state);

	Agent
		.builder()
		.name("omp-headless")
		.on_receive_request(
			async move |req: InitializeRequest, responder: Responder<InitializeResponse>, _cx| {
				// 最小诚实 capabilities：不声明 loadSession/list/fork，promptCapabilities
				// 不声明 image/embeddedContext；mcpCapabilities 声明 http（WP-1.7 接通
				// HTTP MCP），不声明 sse/acp（defer）。
				responder.respond(InitializeResponse::new(req.protocol_version).agent_capabilities(
					AgentCapabilities::new().mcp_capabilities(McpCapabilities::new().http(true)),
				))
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

	let session_id = state.mint_session_id();
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

	let context = AgentContext {
		system_prompt: vec![state.system_prompt.clone()],
		messages:      Vec::new(),
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
