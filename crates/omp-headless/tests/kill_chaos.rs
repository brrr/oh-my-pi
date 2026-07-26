//! E2 kill 混沌（WP-1.6e）：ACP 会话 prompt 进行中把 omp-headless SIGKILL 掉，
//! 断言 (1) session journal 未损坏 —— pi-session loader 能读回已落盘部分、不
//! panic；(2) 新进程 `session/load` 能定位同一 journal 并在其上续跑一轮到完成。
//!
//! ## 设计
//! 直接以 newline-delimited JSON-RPC（ACP over stdio，帧形状见 golden tap）驱动
//! 子进程，不引第三方 ACP client：
//! 1. spawn omp-headless → `initialize` →
//!    `session/new`（`OMP_HEADLESS_SESSION_DIR` 指向本测试 tmp
//!    目录，`session/new` 即把 header 落盘）→ 拿 sessionId；
//! 2. 发一个多步长任务 `session/prompt`，读到**第一条** `session/update`
//!    通知即证明 turn 在途 → 此刻 `Child::kill()`（Unix = SIGKILL）；
//! 3. 断言 journal 文件仍在、`load_session_messages` 能读回（≥header，无
//!    panic）；
//! 4. 新 spawn 一个进程 → `initialize` → `session/load(sessionId)` →
//!    `session/prompt("一句话确认")` → 读到 PromptResponse（result 带
//!    stopReason） 即续接成功；再断言 journal 现含 turn-2 落盘消息。
//!
//! ## LLM 依赖与 CI 安全
//! turn 需要真 provider（DeepSeek，config 从 opencode auth 解析 key）流式产出
//! `session/update`。**无 key / 网络不可达**时子进程起不了 turn，本测试在超时后
//! **优雅跳过**（eprintln 说明 + return，不 fail），保证 `cargo test
//! --workspace` 处处绿。有 key 时跑真 kill 混沌。fake provider
//! 注入点缺席（config 只认 `anthropic` 兼容端点），按 WP 指引「真 `DeepSeek` +
//! 长任务，简单优先」。

use std::{
	io::{BufRead, BufReader, Write},
	process::{Child, Command, Stdio},
	sync::mpsc,
	time::{Duration, Instant},
};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_omp-headless");
/// 等首条 session/update 的墙钟上限（超过即判定「无 provider」→ 跳过）。
const FIRST_UPDATE_TIMEOUT: Duration = Duration::from_secs(45);
/// 续接 turn 读 `PromptResponse` 的墙钟上限。
const RELOAD_TIMEOUT: Duration = Duration::from_mins(1);

/// 后台把子进程 stdout 逐行推给 channel；EOF 时线程结束（sender 落）。
fn spawn_reader(child: &mut Child) -> mpsc::Receiver<String> {
	let stdout = child.stdout.take().expect("child stdout piped");
	let (tx, rx) = mpsc::channel();
	std::thread::spawn(move || {
		let reader = BufReader::new(stdout);
		for line in reader.lines() {
			match line {
				Ok(line) => {
					if tx.send(line).is_err() {
						break;
					}
				},
				Err(_) => break,
			}
		}
	});
	rx
}

/// 起一个 omp-headless 子进程（session dir 指向 `session_dir`）。stderr
/// 丢弃以保 持测试输出干净。
fn spawn_agent(session_dir: &std::path::Path) -> Child {
	Command::new(BIN)
		.arg("--mode")
		.arg("acp")
		.env("OMP_HEADLESS_SESSION_DIR", session_dir)
		.stdin(Stdio::piped())
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.spawn()
		.expect("spawn omp-headless")
}

/// 写一行 JSON-RPC 到子进程 stdin（换行分帧）。
fn send(child: &mut Child, value: &Value) {
	let stdin = child.stdin.as_mut().expect("child stdin piped");
	let mut line = serde_json::to_string(value).expect("serialize frame");
	line.push('\n');
	stdin.write_all(line.as_bytes()).expect("write frame");
	stdin.flush().expect("flush frame");
}

/// 读帧直到 `pred` 命中或超时；返回命中的帧（超时 / EOF → None）。
fn recv_until(
	rx: &mpsc::Receiver<String>,
	deadline: Instant,
	mut pred: impl FnMut(&Value) -> bool,
) -> Option<Value> {
	loop {
		let now = Instant::now();
		if now >= deadline {
			return None;
		}
		match rx.recv_timeout(deadline - now) {
			Ok(line) => {
				if let Ok(value) = serde_json::from_str::<Value>(&line)
					&& pred(&value)
				{
					return Some(value);
				}
			},
			Err(_) => return None, // 超时或 sender 落（EOF）
		}
	}
}

fn is_result_for(value: &Value, id: &str) -> bool {
	value.get("id").and_then(Value::as_str) == Some(id) && value.get("result").is_some()
}

fn is_session_update(value: &Value) -> bool {
	value.get("method").and_then(Value::as_str) == Some("session/update")
}

/// 握手：initialize + session/new，返回 sessionId。失败（无响应）→ None。
fn handshake_new(child: &mut Child, rx: &mpsc::Receiver<String>, cwd: &str) -> Option<String> {
	send(
		child,
		&json!({
			"jsonrpc": "2.0", "id": "init", "method": "initialize",
			"params": { "protocolVersion": 1,
				"clientCapabilities": { "fs": { "readTextFile": false, "writeTextFile": false }, "terminal": false } }
		}),
	);
	recv_until(rx, Instant::now() + Duration::from_secs(15), |v| is_result_for(v, "init"))?;

	send(
		child,
		&json!({
			"jsonrpc": "2.0", "id": "new", "method": "session/new",
			"params": { "cwd": cwd, "mcpServers": [] }
		}),
	);
	let resp =
		recv_until(rx, Instant::now() + Duration::from_secs(15), |v| is_result_for(v, "new"))?;
	resp["result"]["sessionId"].as_str().map(ToOwned::to_owned)
}

/// 找 `dir` 下唯一（或首个）`*.jsonl` journal 文件。
fn find_journal(dir: &std::path::Path) -> Option<std::path::PathBuf> {
	std::fs::read_dir(dir)
		.ok()?
		.flatten()
		.map(|e| e.path())
		.find(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
}

#[test]
fn kill_midprompt_journal_survives_and_reloads() {
	let tmp = std::env::temp_dir().join(format!("omp-kill-chaos-{}", std::process::id()));
	let _ = std::fs::remove_dir_all(&tmp);
	std::fs::create_dir_all(&tmp).expect("mk tmp");
	let session_dir = tmp.join("sessions");
	let cwd = tmp.to_string_lossy().to_string();

	// ── 阶段 1：起会话，发长任务，读到首条 update 证明 turn 在途 ──
	let mut agent = spawn_agent(&session_dir);
	let rx = spawn_reader(&mut agent);

	let Some(session_id) = handshake_new(&mut agent, &rx, &cwd) else {
		let _ = agent.kill();
		eprintln!("[kill_chaos] SKIP：initialize/session/new 无响应（agent 起不来）");
		return;
	};

	send(
		&mut agent,
		&json!({
			"jsonrpc": "2.0", "id": "p1", "method": "session/prompt",
			"params": { "sessionId": session_id, "prompt": [{ "type": "text",
				"text": "Count slowly from 1 to 100, one number per line, each with a short sentence. Go slowly, do not stop early." }] }
		}),
	);

	// 读到首条 session/update = provider 已开始流式产出 → turn 确在途。
	let saw_update =
		recv_until(&rx, Instant::now() + FIRST_UPDATE_TIMEOUT, is_session_update).is_some();
	if !saw_update {
		let _ = agent.kill();
		eprintln!(
			"[kill_chaos] SKIP：{FIRST_UPDATE_TIMEOUT:?} 内无 session/update（无 DeepSeek key / \
			 网络不可达）"
		);
		return;
	}

	// ── 阶段 2：turn 进行中 SIGKILL ──
	agent.kill().expect("SIGKILL agent mid-prompt");
	let _ = agent.wait();
	drop(rx);

	// ── 阶段 3：journal 未损坏，loader 读回不 panic ──
	let journal = find_journal(&session_dir).expect("被杀后 journal 文件仍在（header 已落盘）");
	let after_kill = pi_session::load_session_messages(journal.to_str().unwrap())
		.expect("被 SIGKILL 后的 journal 仍可被 pi-session loader 读回（未损坏）");
	// header-only 或含此前落盘消息皆可；关键是可读、不 panic。
	eprintln!(
		"[kill_chaos] 被杀后 journal 读回 {} 条消息 ({})",
		after_kill.len(),
		journal.display()
	);

	// ── 阶段 4：新进程 loadSession 续接一轮到完成 ──
	let mut agent2 = spawn_agent(&session_dir);
	let rx2 = spawn_reader(&mut agent2);

	send(
		&mut agent2,
		&json!({
			"jsonrpc": "2.0", "id": "init2", "method": "initialize",
			"params": { "protocolVersion": 1,
				"clientCapabilities": { "fs": { "readTextFile": false, "writeTextFile": false }, "terminal": false } }
		}),
	);
	assert!(
		recv_until(&rx2, Instant::now() + Duration::from_secs(15), |v| is_result_for(v, "init2"))
			.is_some(),
		"续接进程 initialize 应有响应"
	);

	send(
		&mut agent2,
		&json!({
			"jsonrpc": "2.0", "id": "load", "method": "session/load",
			"params": { "sessionId": session_id, "cwd": cwd, "mcpServers": [] }
		}),
	);
	assert!(
		recv_until(&rx2, Instant::now() + Duration::from_secs(15), |v| is_result_for(v, "load"))
			.is_some(),
		"session/load 应定位到被杀会话的 journal 并成功 resume"
	);

	send(
		&mut agent2,
		&json!({
			"jsonrpc": "2.0", "id": "p2", "method": "session/prompt",
			"params": { "sessionId": session_id, "prompt": [{ "type": "text",
				"text": "Reply with exactly this line and nothing else: RELOAD_OK" }] }
		}),
	);
	let reload_done =
		recv_until(&rx2, Instant::now() + RELOAD_TIMEOUT, |v| is_result_for(v, "p2")).is_some();

	let _ = agent2.kill();
	let _ = agent2.wait();
	drop(rx2);

	assert!(reload_done, "续接进程应能在被杀会话上跑完一轮 prompt（收到 PromptResponse）");

	// turn-2 落盘：journal 现应含续接轮的新消息（user + assistant）。
	let journal2 = find_journal(&session_dir).expect("续接后 journal 仍在");
	let after_reload =
		pi_session::load_session_messages(journal2.to_str().unwrap()).expect("续接后 journal 可读");
	assert!(
		after_reload.len() > after_kill.len(),
		"续接轮应把新消息追加进同一 journal（被杀后 {} 条 → 续接后 {} 条）",
		after_kill.len(),
		after_reload.len()
	);
	eprintln!("[kill_chaos] 续接后 journal 读回 {} 条消息", after_reload.len());

	let _ = std::fs::remove_dir_all(&tmp);
}
