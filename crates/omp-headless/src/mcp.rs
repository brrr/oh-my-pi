//! 最小 MCP client（WP-1.7）——手写 JSON-RPC over HTTP，与 Stage 0 driver 的 MCP
//! HTTP sim（`driver-services/src/mcp.rs`）互通。
//!
//! 只实现三方法 `initialize` / `tools/list` / `tools/call`（外加
//! `notifications/initialized` 通知），**不引** rmcp/官方 SDK。传输取 driver
//! sim 的实际形状：streamable HTTP 的最简子集——单条 `POST` 到
//! `mcp_url`、JSON-RPC 2.0 envelope、响应体即 JSON（无 SSE、无 `Mcp-Session-Id`
//! 头要求；sim 完全忽略请求 头，故本 client 只带 `Accept: application/json` +
//! 用户 headers）。真实 streamable-HTTP server（含 session 头 / SSE 流）留后续
//! WP，接口不变。
//!
//! [`McpTool`] 把一个远端工具适配成 [`pi_tools::Tool`]：`name` / `description`
//! / `input_schema` 取自 `tools/list`，`execute` 转发 `tools/call` 并把结果
//! `content` 的 text 块映射回 [`ToolResult`]。concurrency 恒 `Shared`（MCP
//! 工具无 本地副作用序，走并发批）。

use std::{
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use pi_shell::cancel::CancelToken;
use pi_tools::{Concurrency, Tool, ToolError, ToolResult};
use serde_json::{Value, json};

/// 单次 HTTP 请求超时（连接 + 传输）。
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// MCP 端点 client（一个 server 一个实例，多个 [`McpTool`] 共享）。
///
/// 内含一个 `reqwest::Client`（连接池廉价，`Arc` 共享），单调 JSON-RPC id。
pub struct McpClient {
	http:    reqwest::Client,
	url:     String,
	/// 用户 headers（ACP `McpServerHttp.headers`；sim 不校验，真 server 需要）。
	headers: Vec<(String, String)>,
	next_id: AtomicU64,
}

/// `tools/list` 里一条工具的最小面（`name` / `description` / `input_schema`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolDef {
	pub name:         String,
	pub description:  String,
	pub input_schema: Value,
}

impl McpClient {
	/// 建一个指向 `url` 的 client（不发请求）。
	///
	/// # Errors
	///
	/// `reqwest::Client` 构建失败（罕见——TLS 后端初始化等）。
	pub fn new(url: impl Into<String>, headers: Vec<(String, String)>) -> Result<Self> {
		let http = reqwest::Client::builder()
			.timeout(HTTP_TIMEOUT)
			.build()
			.context("构建 MCP HTTP client 失败")?;
		Ok(Self { http, url: url.into(), headers, next_id: AtomicU64::new(0) })
	}

	/// 组装带用户 headers + `Accept` 的 POST builder。
	fn post(&self) -> reqwest::RequestBuilder {
		let mut rb = self
			.http
			.post(&self.url)
			.header("Accept", "application/json");
		for (name, value) in &self.headers {
			rb = rb.header(name.as_str(), value.as_str());
		}
		rb
	}

	/// 发一条有 id 的 JSON-RPC 请求，返回 `result`（`error` 字段 → `Err`）。
	async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
		let id = self.next_id.fetch_add(1, Ordering::SeqCst);
		let envelope = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
		let resp = self
			.post()
			.json(&envelope)
			.send()
			.await
			.with_context(|| format!("MCP {method} 请求发送失败"))?;
		let status = resp.status();
		let text = resp
			.text()
			.await
			.with_context(|| format!("MCP {method} 响应读取失败"))?;
		if !status.is_success() {
			bail!("MCP {method} HTTP {status}: {text}");
		}
		let body: Value = serde_json::from_str(&text)
			.with_context(|| format!("MCP {method} 响应非 JSON: {text}"))?;
		if let Some(err) = body.get("error").filter(|e| !e.is_null()) {
			bail!("MCP {method} JSON-RPC error: {err}");
		}
		Ok(body.get("result").cloned().unwrap_or(Value::Null))
	}

	/// 发一条无 id 的通知（不解析响应体——sim 仍回一帧，忽略即可）。
	async fn notify(&self, method: &str) -> Result<()> {
		let envelope = json!({ "jsonrpc": "2.0", "method": method });
		let _ = self
			.post()
			.json(&envelope)
			.send()
			.await
			.with_context(|| format!("MCP 通知 {method} 发送失败"))?;
		Ok(())
	}

	/// `initialize`——形状照 driver sim（sim 忽略 params，仅回固定 serverInfo）。
	///
	/// # Errors
	///
	/// 传输失败 / server 回 JSON-RPC error。
	pub async fn initialize(&self) -> Result<Value> {
		self
			.rpc(
				"initialize",
				json!({
					"protocolVersion": "2025-11-25",
					"capabilities": {},
					"clientInfo": { "name": "omp-headless", "version": env!("CARGO_PKG_VERSION") }
				}),
			)
			.await
	}

	/// `notifications/initialized`——握手收尾通知。
	///
	/// # Errors
	///
	/// 传输失败。
	pub async fn notify_initialized(&self) -> Result<()> {
		self.notify("notifications/initialized").await
	}

	/// `tools/list`——解析 `result.tools[]` 为 [`McpToolDef`]（缺 name
	/// 的条目跳过）。
	///
	/// # Errors
	///
	/// 传输失败 / server error / 响应结构非法。
	pub async fn list_tools(&self) -> Result<Vec<McpToolDef>> {
		let result = self.rpc("tools/list", Value::Null).await?;
		let tools = result
			.get("tools")
			.and_then(Value::as_array)
			.ok_or_else(|| anyhow!("tools/list 响应缺 tools 数组: {result}"))?;
		let mut defs = Vec::with_capacity(tools.len());
		for tool in tools {
			let Some(name) = tool.get("name").and_then(Value::as_str) else {
				continue;
			};
			defs.push(McpToolDef {
				name:         name.to_owned(),
				description:  tool
					.get("description")
					.and_then(Value::as_str)
					.unwrap_or_default()
					.to_owned(),
				input_schema: tool
					.get("inputSchema")
					.cloned()
					.unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
			});
		}
		Ok(defs)
	}

	/// `tools/call`——转发 name + arguments，返回原始 `result`（含
	/// content/isError）。
	///
	/// # Errors
	///
	/// 传输失败 / server 回 JSON-RPC error（协议级失败，区别于 `isError`
	/// 软失败）。
	pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
		self
			.rpc("tools/call", json!({ "name": name, "arguments": arguments }))
			.await
	}
}

/// 把 MCP `tools/call` 的 `result` 映射为 [`ToolResult`]：拼所有 text 块，
/// `isError:true` 落软失败。
fn map_call_result(result: &Value) -> ToolResult {
	let mut text = String::new();
	if let Some(blocks) = result.get("content").and_then(Value::as_array) {
		for block in blocks {
			if block.get("type").and_then(Value::as_str) == Some("text")
				&& let Some(t) = block.get("text").and_then(Value::as_str)
			{
				text.push_str(t);
			}
		}
	}
	let out = ToolResult::text(text);
	if result.get("isError").and_then(Value::as_bool) == Some(true) {
		out.error()
	} else {
		out
	}
}

/// 单个远端工具的 [`Tool`] 适配器（可 clone——每轮 prompt 复制进 loop 注册表）。
///
/// `name` 是 `&'static str`：由 [`connect_http`] 在 session 建立时对每个工具名
/// **一次性** `Box::leak`（[`Tool::name`] 契约要求 `'static`，而 MCP
/// 工具名运行时 才知）。泄漏量有界（= 一个 session 的 MCP 工具数），session
/// 生命周期内复用，不随 prompt 轮次增长。
#[derive(Clone)]
pub struct McpTool {
	name:         &'static str,
	description:  String,
	input_schema: Value,
	client:       Arc<McpClient>,
}

impl Tool for McpTool {
	fn name(&self) -> &'static str {
		self.name
	}

	fn description(&self) -> &str {
		&self.description
	}

	fn input_schema(&self) -> Value {
		self.input_schema.clone()
	}

	fn concurrency(&self) -> Concurrency {
		Concurrency::Shared
	}

	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		_ct: &CancelToken,
	) -> Result<ToolResult, ToolError> {
		match self.client.call_tool(self.name, args).await {
			Ok(result) => Ok(map_call_result(&result)),
			// 协议级失败（传输错误 / JSON-RPC error）→ 抛 ToolError（模型看到文本）。
			Err(error) => Err(ToolError::new(format!("MCP 工具 {} 调用失败: {error}", self.name))),
		}
	}
}

/// 泄漏一份工具名为 `&'static str`（见 [`McpTool`] 文档的有界性说明）。
fn leak_name(name: &str) -> &'static str {
	Box::leak(name.to_owned().into_boxed_str())
}

/// 连一个 HTTP MCP server：`initialize` → `notifications/initialized` →
/// `tools/list`，返回该 server 的 [`McpTool`] 集合。
///
/// # Errors
///
/// 任一握手步骤（连接 / initialize / tools/list）失败——调用方决定是否容错。
pub async fn connect_http(url: &str, headers: Vec<(String, String)>) -> Result<Vec<McpTool>> {
	let client = Arc::new(McpClient::new(url, headers)?);
	client.initialize().await.context("MCP initialize 失败")?;
	client
		.notify_initialized()
		.await
		.context("MCP notifications/initialized 失败")?;
	let defs = client.list_tools().await.context("MCP tools/list 失败")?;
	Ok(defs
		.into_iter()
		.map(|def| McpTool {
			name:         leak_name(&def.name),
			description:  def.description,
			input_schema: def.input_schema,
			client:       Arc::clone(&client),
		})
		.collect())
}

#[cfg(test)]
mod tests {
	use std::{
		io::{Read as _, Write as _},
		net::{TcpListener, TcpStream},
		thread::JoinHandle,
	};

	use super::*;

	/// 极简进程内 HTTP/1.1 server，仿 driver sim 的 JSON-RPC
	/// 面（不引外部依赖）。
	struct FakeMcp {
		addr:   String,
		handle: Option<JoinHandle<()>>,
	}

	impl FakeMcp {
		fn url(&self) -> String {
			format!("http://{}/mcp", self.addr)
		}
	}

	impl Drop for FakeMcp {
		fn drop(&mut self) {
			// 叫醒 accept 循环（连一下让它跑完当前 incoming 后 join）。
			if let Ok(s) = TcpStream::connect(&self.addr) {
				drop(s);
			}
			if let Some(h) = self.handle.take() {
				let _ = h.join();
			}
		}
	}

	/// 读一条 HTTP 请求的 body（按 Content-Length）。
	fn read_body(stream: &mut TcpStream) -> Option<String> {
		let mut buf: Vec<u8> = Vec::new();
		let mut tmp = [0u8; 1024];
		loop {
			let sep = buf.windows(4).position(|w| w == b"\r\n\r\n");
			if let Some(pos) = sep {
				let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
				let len = head
					.lines()
					.find_map(|l| l.strip_prefix("content-length:"))
					.and_then(|v| v.trim().parse::<usize>().ok())
					.unwrap_or(0);
				let body_start = pos + 4;
				while buf.len() < body_start + len {
					let n = stream.read(&mut tmp).ok()?;
					if n == 0 {
						break;
					}
					buf.extend_from_slice(&tmp[..n]);
				}
				return Some(
					String::from_utf8_lossy(&buf[body_start..(body_start + len).min(buf.len())])
						.into_owned(),
				);
			}
			let n = stream.read(&mut tmp).ok()?;
			if n == 0 {
				return None;
			}
			buf.extend_from_slice(&tmp[..n]);
		}
	}

	fn write_json(stream: &mut TcpStream, body: &str) {
		let resp = format!(
			"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
			 close\r\n\r\n{}",
			body.len(),
			body
		);
		let _ = stream.write_all(resp.as_bytes());
		let _ = stream.flush();
	}

	/// server 侧 JSON-RPC dispatch（kanban-lite：两工具 + `create_comment`
	/// 写回）。
	fn handle_rpc(body: &str) -> String {
		let req: Value = serde_json::from_str(body).unwrap_or(Value::Null);
		let id = req.get("id").cloned().unwrap_or(Value::Null);
		let method = req.get("method").and_then(Value::as_str).unwrap_or("");
		let params = req.get("params").cloned().unwrap_or(Value::Null);
		let result = match method {
			"initialize" => json!({
				"protocolVersion": "2024-11-05",
				"capabilities": { "tools": {} },
				"serverInfo": { "name": "FakeMcp", "version": "1.0.0" }
			}),
			"notifications/initialized" => Value::Null,
			"tools/list" => json!({ "tools": [
				{"name":"kanban_get_card","description":"读卡片","inputSchema":{"type":"object","properties":{"card_id":{"type":"string"}},"required":["card_id"]}},
				{"name":"kanban_create_comment","description":"写回评论","inputSchema":{"type":"object","properties":{"card_id":{"type":"string"},"content":{"type":"string"}},"required":["card_id","content"]}}
			]}),
			"tools/call" => {
				let name = params.get("name").and_then(Value::as_str).unwrap_or("");
				let args = params.get("arguments").cloned().unwrap_or(Value::Null);
				match name {
					"kanban_get_card" => json!({ "content": [{ "type": "text", "text": "{\"card\":{\"id\":\"c1\"}}" }] }),
					"kanban_create_comment" => json!({ "content": [{ "type": "text", "text": format!("{{\"comment\":{{\"content\":{}}}}}", args.get("content").cloned().unwrap_or(Value::Null)) }] }),
					"boom" => {
						// isError 软失败
						return json!({ "jsonrpc": "2.0", "id": id, "result": { "content": [{ "type": "text", "text": "错误: 炸了" }], "isError": true } }).to_string();
					}
					other => {
						// 协议级 error（未知工具）
						return json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": format!("未知工具: {other}") } }).to_string();
					}
				}
			}
			_ => return json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "method not found" } }).to_string(),
		};
		json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
	}

	fn spawn_fake() -> FakeMcp {
		let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
		let addr = listener.local_addr().expect("addr").to_string();
		let handle = std::thread::spawn(move || {
			for stream in listener.incoming() {
				let Ok(mut stream) = stream else { break };
				// Drop 的唤醒连接不带 body → read_body None → 退出 accept 循环。
				let Some(body) = read_body(&mut stream) else {
					break;
				};
				if body.is_empty() {
					break;
				}
				let resp = handle_rpc(&body);
				write_json(&mut stream, &resp);
			}
		});
		FakeMcp { addr, handle: Some(handle) }
	}

	/// 取一个不再监听的地址（bind 后立刻 drop）→ 连它必 refused。
	fn dead_addr() -> String {
		let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
		let addr = listener.local_addr().expect("addr").to_string();
		drop(listener);
		format!("http://{addr}/mcp")
	}

	#[tokio::test]
	async fn initialize_returns_server_info() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		let info = client.initialize().await.expect("initialize");
		assert_eq!(info["serverInfo"]["name"], "FakeMcp");
	}

	#[tokio::test]
	async fn notify_initialized_ok() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		client.notify_initialized().await.expect("notify");
	}

	#[tokio::test]
	async fn list_tools_parses_defs() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		let defs = client.list_tools().await.expect("list");
		let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
		assert_eq!(names, vec!["kanban_get_card", "kanban_create_comment"]);
		assert_eq!(defs[1].description, "写回评论");
		assert_eq!(defs[1].input_schema["required"], json!(["card_id", "content"]));
	}

	#[tokio::test]
	async fn call_tool_success_maps_text() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		let result = client
			.call_tool("kanban_get_card", json!({ "card_id": "c1" }))
			.await
			.expect("call");
		let mapped = map_call_result(&result);
		assert!(!mapped.is_error);
		assert!(!mapped.content.is_empty());
	}

	#[tokio::test]
	async fn call_tool_iserror_maps_soft_failure() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		let result = client.call_tool("boom", json!({})).await.expect("call");
		let mapped = map_call_result(&result);
		assert!(mapped.is_error, "isError:true 应落软失败");
	}

	#[tokio::test]
	async fn call_tool_rpc_error_is_err() {
		let srv = spawn_fake();
		let client = McpClient::new(srv.url(), Vec::new()).expect("client");
		let out = client.call_tool("nope", json!({})).await;
		assert!(out.is_err(), "JSON-RPC error 应为 Err（协议级失败）");
	}

	#[tokio::test]
	async fn connect_http_yields_adapted_tools() {
		let srv = spawn_fake();
		let tools = connect_http(&srv.url(), Vec::new()).await.expect("connect");
		assert_eq!(tools.len(), 2);
		assert_eq!(tools[0].name(), "kanban_get_card");
		assert_eq!(tools[1].name(), "kanban_create_comment");
		assert_eq!(tools[1].description(), "写回评论");
		assert_eq!(tools[1].input_schema()["required"], json!(["card_id", "content"]));
	}

	#[tokio::test]
	async fn mcp_tool_execute_forwards_call() {
		let srv = spawn_fake();
		let tools = connect_http(&srv.url(), Vec::new()).await.expect("connect");
		let create = tools
			.iter()
			.find(|t| t.name() == "kanban_create_comment")
			.expect("tool");
		let ct = CancelToken::default();
		let out = create
			.execute("call-1", json!({ "card_id": "c1", "content": "结论：ZUIHOU" }), &ct)
			.await
			.expect("execute ok");
		assert!(!out.is_error);
		// server 把 content 回显进 text
		let joined = format!("{:?}", out.content);
		assert!(joined.contains("ZUIHOU"), "写回内容应回显: {joined}");
	}

	#[tokio::test]
	async fn mcp_tool_execute_rpc_error_is_tool_error() {
		let srv = spawn_fake();
		// 手工造一个只含未知工具的适配器（复用 client）。
		let client = Arc::new(McpClient::new(srv.url(), Vec::new()).expect("client"));
		let tool = McpTool {
			name: leak_name("nope"),
			description: String::new(),
			input_schema: json!({}),
			client,
		};
		let ct = CancelToken::default();
		let err = tool
			.execute("c", json!({}), &ct)
			.await
			.expect_err("应 ToolError");
		assert!(err.to_string().contains("调用失败"));
	}

	#[tokio::test]
	async fn connect_http_connection_refused_is_err() {
		let out = connect_http(&dead_addr(), Vec::new()).await;
		assert!(out.is_err(), "连不上的 server 应返 Err（由上层容错忽略）");
	}
}
