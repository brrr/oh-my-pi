//! PTY-backed interactive command execution (pure Rust core).
//!
//! # Overview
//! Stateful PTY session supporting streaming output and stdin passthrough
//! while a command runs. Extracted from `pi-natives/src/pty.rs` so the logic
//! is callable in-process without N-API; the natives crate keeps only the
//! JS-facing adapter (threadsafe-function callbacks, promise plumbing).
//!
//! Callbacks are plain boxed closures invoked on the blocking runner thread;
//! adapters decide how to marshal chunks across whatever boundary they serve.

use std::{
	collections::HashMap,
	io::{Read, Write},
	str,
	sync::Arc,
	time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use portable_pty::{Child, CommandBuilder, PtySize, native_pty_system};

use crate::{cancel::CancelToken, process};

/// What to execute inside the PTY.
#[derive(Clone)]
pub enum PtyCommand {
	/// A shell command string run through `shell -lc`/`-Command`/`/c`.
	Shell { command: String, shell: Option<String> },
	/// An executable + argv, passed through without shell quoting.
	Argv { application: String, args: Vec<String> },
}

/// Fully-resolved run configuration.
#[derive(Clone)]
pub struct PtyRunConfig {
	pub command: PtyCommand,
	pub cwd:     Option<String>,
	pub env:     Option<HashMap<String, String>>,
	pub cols:    u16,
	pub rows:    u16,
}

/// Result of a PTY command run.
#[derive(Debug, Clone, Copy)]
pub struct PtyRunResult {
	pub exit_code: Option<i32>,
	pub cancelled: bool,
	pub timed_out: bool,
}

/// Output chunk callback, invoked on the blocking runner thread.
pub type ChunkFn = Box<dyn Fn(&str) + Send>;
/// Spawn notification callback carrying the child PID.
pub type StartFn = Box<dyn FnOnce(u32) + Send>;

/// Default/clamped PTY dimensions (mirrors the historical napi-layer policy).
#[must_use]
pub fn clamp_cols(cols: Option<u16>) -> u16 {
	cols.unwrap_or(120).clamp(20, 400)
}

#[must_use]
pub fn clamp_rows(rows: Option<u16>) -> u16 {
	rows.unwrap_or(40).clamp(5, 200)
}

enum ReaderEvent {
	Chunk(String),
	Done,
}

enum ControlMessage {
	Input(String),
	Resize { cols: u16, rows: u16 },
	Kill,
}

const CONTROL_MESSAGES_PER_TICK: usize = 64;
const READER_EVENTS_PER_TICK: usize = 256;
const POST_CANCEL_DRAIN_TIMEOUT: Duration = Duration::from_millis(300);
const POST_EXIT_DRAIN_TIMEOUT: Duration = Duration::from_millis(300);
#[cfg(not(windows))]
const FINAL_READER_DRAIN_TIMEOUT: Duration = Duration::from_millis(50);

struct PtySessionCore {
	control_tx: flume::Sender<ControlMessage>,
}

type SessionSlot = Arc<Mutex<Option<PtySessionCore>>>;

/// Stateful PTY session for interactive stdin/stdout passthrough. One run at
/// a time; `write`/`resize`/`kill` address the currently-running command.
#[derive(Default)]
pub struct PtySession {
	core: SessionSlot,
}

impl PtySession {
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Register the control channel for a new run. Fails when a run is
	/// already active. Registration is synchronous so `write()`/`kill()`
	/// work the moment this returns, before the blocking runner spins up.
	///
	/// # Errors
	///
	/// When a PTY run is already active on this session.
	pub fn prepare(&self) -> Result<PtyRunHandle> {
		let (control_tx, control_rx) = flume::unbounded::<ControlMessage>();
		let mut guard = self.core.lock();
		if guard.is_some() {
			bail!("PTY session already running");
		}
		*guard = Some(PtySessionCore { control_tx });
		drop(guard);
		Ok(PtyRunHandle { slot: Arc::clone(&self.core), control_rx })
	}

	/// Write raw input bytes to PTY stdin.
	///
	/// # Errors
	///
	/// When no run is active or the runner is gone.
	pub fn write(&self, data: String) -> Result<()> {
		self.send_control(ControlMessage::Input(data))
	}

	/// Resize the active PTY (dimensions clamped to sane bounds).
	///
	/// # Errors
	///
	/// When no run is active or the runner is gone.
	pub fn resize(&self, cols: u16, rows: u16) -> Result<()> {
		self.send_control(ControlMessage::Resize {
			cols: cols.clamp(20, 400),
			rows: rows.clamp(5, 200),
		})
	}

	/// Force-kill the active PTY command.
	///
	/// # Errors
	///
	/// When no run is active or the runner is gone.
	pub fn kill(&self) -> Result<()> {
		self.send_control(ControlMessage::Kill)
	}

	fn send_control(&self, message: ControlMessage) -> Result<()> {
		let guard = self.core.lock();
		let core = guard.as_ref().context("PTY session is not running")?;
		core
			.control_tx
			.send(message)
			.map_err(|_| anyhow!("PTY session is no longer available"))
	}
}

/// A prepared (registered) run. Dropping it — normally at the end of
/// [`PtyRunHandle::run`], including on panic unwind — frees the session slot.
pub struct PtyRunHandle {
	slot:       SessionSlot,
	control_rx: flume::Receiver<ControlMessage>,
}

impl Drop for PtyRunHandle {
	fn drop(&mut self) {
		*self.slot.lock() = None;
	}
}

impl PtyRunHandle {
	/// Run the command to completion on the current thread (blocking). Call
	/// from a dedicated/blocking thread.
	///
	/// # Errors
	///
	/// PTY setup failures (openpty/spawn/reader) and pre-spawn cancellation.
	/// Post-spawn cancellation/timeout is not an error — it is reported via
	/// [`PtyRunResult::cancelled`] / [`PtyRunResult::timed_out`].
	pub fn run(
		self,
		config: PtyRunConfig,
		ct: &CancelToken,
		on_chunk: Option<ChunkFn>,
		on_start: Option<StartFn>,
	) -> Result<PtyRunResult> {
		run_pty_sync(config, on_chunk, on_start, &self.control_rx, ct)
	}
}

fn terminate_pty_processes(
	child: &mut Box<dyn Child + Send + Sync>,
	child_pid: Option<i32>,
	process_group_id: Option<i32>,
) {
	let mut targets = process::TerminationTargets::new();
	if let Some(pgid) = process_group_id {
		targets.add_pgid(pgid);
	}
	if let Some(pid) = child_pid {
		targets.add_pid(pid);
	}

	targets.signal(process::TERM_SIGNAL);
	let _ = child.kill();
	targets.signal(process::KILL_SIGNAL);
}

#[expect(clippy::too_many_lines, reason = "linear teardown-sensitive state machine, kept whole")]
fn run_pty_sync(
	config: PtyRunConfig,
	on_chunk: Option<ChunkFn>,
	on_start: Option<StartFn>,
	control_rx: &flume::Receiver<ControlMessage>,
	ct: &CancelToken,
) -> Result<PtyRunResult> {
	let pty_system = native_pty_system();
	ct.heartbeat()
		.context("PTY setup cancelled before openpty")?;

	const PTY_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
	let pair = if cfg!(windows) {
		// Windows ConPTY openpty() can hang indefinitely when the console
		// subsystem isn't properly initialized. Use a short startup timeout
		// so the caller gets an error instead of hanging forever.
		let (tx, rx) = flume::unbounded();
		std::thread::spawn(move || {
			let result = pty_system.openpty(PtySize {
				rows:         config.rows,
				cols:         config.cols,
				pixel_width:  0,
				pixel_height: 0,
			});
			let _ = tx.send(result);
		});
		match rx.recv_timeout(PTY_STARTUP_TIMEOUT) {
			Ok(Ok(pair)) => pair,
			Ok(Err(e)) => bail!("Failed to open PTY: {e}"),
			Err(_) => {
				bail!("PTY creation timed out (5s). ConPTY may be unavailable on this system.")
			},
		}
	} else {
		pty_system
			.openpty(PtySize {
				rows:         config.rows,
				cols:         config.cols,
				pixel_width:  0,
				pixel_height: 0,
			})
			.map_err(|err| anyhow!("Failed to open PTY: {err}"))?
	};

	let mut cmd = match config.command {
		PtyCommand::Shell { command, shell } => {
			let shell = shell.as_deref().unwrap_or("sh");
			let mut cmd = CommandBuilder::new(shell);
			let lower = shell.to_lowercase();
			if lower.ends_with("cmd.exe") || lower.ends_with("cmd") {
				cmd.arg("/c");
			} else if lower.contains("powershell") || lower.contains("pwsh") {
				cmd.arg("-Command");
			} else {
				cmd.arg("-lc");
			}
			cmd.arg(command);
			cmd
		},
		PtyCommand::Argv { application, args } => {
			let mut cmd = CommandBuilder::new(application);
			for arg in args {
				cmd.arg(arg);
			}
			cmd
		},
	};
	if let Some(cwd) = config.cwd.as_ref() {
		cmd.cwd(cwd);
	}
	if let Some(env) = config.env.as_ref() {
		for (key, value) in env {
			cmd.env(key, value);
		}
	}
	ct.heartbeat().context("PTY setup cancelled before spawn")?;

	let mut child = pair
		.slave
		.spawn_command(cmd)
		.map_err(|err| anyhow!("Failed to spawn PTY command: {err}"))?;
	drop(pair.slave);
	let child_process_id = child.process_id();
	let child_pid = child_process_id.and_then(|value| i32::try_from(value).ok());
	if let Some(callback) = on_start {
		callback(child_process_id.unwrap_or(0));
	}
	ct.heartbeat()
		.context("PTY setup cancelled before reader")?;

	let master = pair.master;
	let mut writer = master
		.take_writer()
		.map_err(|err| anyhow!("Failed to create PTY writer: {err}"))?;
	// ConPTY sends ESC[6n (cursor position query) and blocks until we reply.
	// Reply with cursor at 1,1 so it unblocks the child spawn.
	// Only needed on Windows; on Unix/macOS this would corrupt stdin.
	#[cfg(windows)]
	{
		let _ = writer.write_all(b"\x1b[1;1R");
		let _ = writer.flush();
	}
	let mut reader = master
		.try_clone_reader()
		.map_err(|err| anyhow!("Failed to create PTY reader: {err}"))?;

	let (reader_tx, reader_rx) = flume::unbounded::<ReaderEvent>();
	let reader_thread = std::thread::spawn(move || {
		const REPLACEMENT: &str = "\u{FFFD}";
		const BUF: usize = 65536;
		let mut buf = vec![0u8; BUF + 4];
		let mut it = 0;
		loop {
			match reader.read(&mut buf[it..BUF]) {
				Ok(0) => {
					break;
				},
				Ok(n) => {
					it += n;
					while it > 0 {
						let pending = &buf[..it];
						match str::from_utf8(pending) {
							Ok(text) => {
								let _ = reader_tx.send(ReaderEvent::Chunk(text.to_string()));
								it = 0;
								break;
							},
							Err(err) => {
								let valid_up_to = err.valid_up_to();
								if valid_up_to > 0 {
									// SAFETY: [..valid_up_to] is guaranteed valid UTF-8 by valid_up_to().
									let text = unsafe { str::from_utf8_unchecked(&pending[..valid_up_to]) };
									let _ = reader_tx.send(ReaderEvent::Chunk(text.to_string()));
									buf.copy_within(valid_up_to..it, 0);
									it -= valid_up_to;
								}
								match err.error_len() {
									Some(invalid_len) => {
										let _ = reader_tx.send(ReaderEvent::Chunk(REPLACEMENT.to_string()));
										buf.copy_within(invalid_len..it, 0);
										it -= invalid_len;
									},
									None => {
										break;
									},
								}
							},
						}
					}
				},
				Err(_) => {
					break;
				},
			}
		}
		for chunk in buf[..it].utf8_chunks() {
			let valid = chunk.valid();
			if !valid.is_empty() {
				let _ = reader_tx.send(ReaderEvent::Chunk(valid.to_string()));
			}
			if !chunk.invalid().is_empty() {
				let _ = reader_tx.send(ReaderEvent::Chunk(REPLACEMENT.to_string()));
			}
		}
		let _ = reader_tx.send(ReaderEvent::Done);
	});

	#[cfg(unix)]
	let process_group_id = master.process_group_leader().filter(|pgid| *pgid > 0);
	#[cfg(not(unix))]
	let process_group_id: Option<i32> = None;
	let mut timed_out = false;
	let mut cancelled = false;
	let mut reader_done = false;
	let mut exit_code: Option<i32> = None;
	let mut terminate_requested = false;
	let mut reader_drain_deadline: Option<Instant> = None;
	let emit = |text: &str| {
		if let Some(callback) = on_chunk.as_ref() {
			callback(text);
		}
	};
	while exit_code.is_none() || !reader_done {
		if !terminate_requested && let Err(err) = ct.heartbeat() {
			let message = err.to_string();
			timed_out = message.contains("Timeout");
			cancelled = !timed_out;
			terminate_pty_processes(&mut child, child_pid, process_group_id);
			terminate_requested = true;
			reader_drain_deadline = Some(Instant::now() + POST_CANCEL_DRAIN_TIMEOUT);
		}

		for _ in 0..CONTROL_MESSAGES_PER_TICK {
			match control_rx.try_recv() {
				Ok(ControlMessage::Input(data)) => {
					let _ = writer.write_all(data.as_bytes());
					let _ = writer.flush();
				},
				Ok(ControlMessage::Resize { cols, rows }) => {
					let _ = master.resize(PtySize { rows, cols, pixel_width: 0, pixel_height: 0 });
				},
				Ok(ControlMessage::Kill) => {
					cancelled = true;
					if !terminate_requested {
						terminate_pty_processes(&mut child, child_pid, process_group_id);
						terminate_requested = true;
						reader_drain_deadline = Some(Instant::now() + POST_CANCEL_DRAIN_TIMEOUT);
					}
				},
				Err(flume::TryRecvError::Empty | flume::TryRecvError::Disconnected) => break,
			}
		}

		for _ in 0..READER_EVENTS_PER_TICK {
			match reader_rx.try_recv() {
				Ok(ReaderEvent::Chunk(chunk)) => emit(&chunk),
				Ok(ReaderEvent::Done) => {
					reader_done = true;
					break;
				},
				Err(flume::TryRecvError::Empty) => break,
				Err(flume::TryRecvError::Disconnected) => {
					reader_done = true;
					break;
				},
			}
		}
		if exit_code.is_none()
			&& let Some(status) = child
				.try_wait()
				.map_err(|err| anyhow!("Failed checking PTY status: {err}"))?
		{
			exit_code = Some(i32::try_from(status.exit_code()).unwrap_or(i32::MAX));
			if !reader_done && reader_drain_deadline.is_none() {
				reader_drain_deadline = Some(Instant::now() + POST_EXIT_DRAIN_TIMEOUT);
			}
		}

		if let Some(deadline) = reader_drain_deadline
			&& Instant::now() >= deadline
		{
			break;
		}
		if exit_code.is_none() || !reader_done {
			let wait_duration = reader_drain_deadline.map_or(Duration::from_millis(16), |deadline| {
				deadline
					.saturating_duration_since(Instant::now())
					.min(Duration::from_millis(16))
			});
			match reader_rx.recv_timeout(wait_duration) {
				Ok(ReaderEvent::Chunk(chunk)) => emit(&chunk),
				Ok(ReaderEvent::Done) => reader_done = true,
				Err(flume::RecvTimeoutError::Timeout) => {},
				Err(flume::RecvTimeoutError::Disconnected) => {
					reader_done = true;
					if exit_code.is_none() {
						std::thread::sleep(wait_duration);
					}
				},
			}
		}
	}
	if exit_code.is_none() {
		if terminate_requested {
			if let Some(status) = child
				.try_wait()
				.map_err(|err| anyhow!("Failed checking PTY status: {err}"))?
			{
				exit_code = Some(i32::try_from(status.exit_code()).unwrap_or(i32::MAX));
			}
		} else {
			// On Windows, child.wait() can hang indefinitely in ConPTY.
			// Poll try_wait() with a short timeout instead.
			#[cfg(windows)]
			{
				let wait_start = Instant::now();
				while exit_code.is_none() && wait_start.elapsed() < Duration::from_secs(5) {
					if let Some(status) = child
						.try_wait()
						.map_err(|err| anyhow!("Failed checking PTY status: {err}"))?
					{
						exit_code = Some(i32::try_from(status.exit_code()).unwrap_or(i32::MAX));
						break;
					}
					std::thread::sleep(Duration::from_millis(50));
				}
			}
			#[cfg(not(windows))]
			{
				let status = child
					.wait()
					.map_err(|err| anyhow!("Failed waiting PTY process: {err}"))?;
				exit_code = Some(i32::try_from(status.exit_code()).unwrap_or(i32::MAX));
			}
		}
	}
	// --- Teardown ---

	// Step 1: Close the ConPTY input pipe first.
	// Per Microsoft docs, close the input handle before calling ClosePseudoConsole.
	// This signals to ConPTY that no more input will arrive, allowing its internal
	// I/O threads to finish processing and eventually close the output pipe.
	drop(writer);

	// Step 2: Drain the reader thread.
	// After the child exits and input is closed, ConPTY should flush remaining
	// output and signal EOF on the output pipe, causing the reader thread to exit.
	// On Windows, use a generous timeout to accommodate ConPTY's async teardown.
	if !reader_done {
		#[cfg(windows)]
		let drain_timeout = Duration::from_millis(500);
		#[cfg(not(windows))]
		let drain_timeout = FINAL_READER_DRAIN_TIMEOUT;
		let finalize_deadline = Instant::now() + drain_timeout;
		while Instant::now() < finalize_deadline {
			let remaining = finalize_deadline.saturating_duration_since(Instant::now());
			let wait_duration = remaining.min(Duration::from_millis(5));
			match reader_rx.recv_timeout(wait_duration) {
				Ok(ReaderEvent::Chunk(chunk)) => emit(&chunk),
				Ok(ReaderEvent::Done) => {
					reader_done = true;
					break;
				},
				Err(flume::RecvTimeoutError::Timeout) => {},
				Err(flume::RecvTimeoutError::Disconnected) => {
					reader_done = true;
					break;
				},
			}
		}
	}

	// Step 3: Drop master (calls ClosePseudoConsole on Windows).
	// ClosePseudoConsole can deadlock if ConPTY tries to flush output
	// while nobody is reading the pipe (microsoft/terminal#1810).
	// Always offload to a background thread on Windows, then wait with
	// a timeout so the thread is reclaimed when ClosePseudoConsole
	// completes cleanly. If it hangs, we walk away — the thread leaks,
	// but the main thread never blocks.
	#[cfg(windows)]
	{
		let (drop_tx, drop_rx) = flume::unbounded::<()>();
		std::thread::spawn(move || {
			drop(master);
			let _ = drop_tx.send(());
		});
		let _ = drop_rx.recv_timeout(Duration::from_secs(2));
	}
	#[cfg(not(windows))]
	{
		drop(master);
	}

	// Step 4: Join reader thread if it finished.
	// A detached descendant can keep the PTY slave open forever; do not block
	// completion waiting on join when the reader thread did not reach EOF.
	if reader_done {
		let _ = reader_thread.join();
	}
	Ok(PtyRunResult { exit_code, cancelled, timed_out })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn clamps_apply_defaults_and_bounds() {
		assert_eq!(clamp_cols(None), 120);
		assert_eq!(clamp_cols(Some(10)), 20);
		assert_eq!(clamp_cols(Some(1000)), 400);
		assert_eq!(clamp_rows(None), 40);
		assert_eq!(clamp_rows(Some(1)), 5);
		assert_eq!(clamp_rows(Some(999)), 200);
	}

	#[test]
	fn prepare_rejects_concurrent_runs_and_slot_frees_on_drop() {
		let session = PtySession::new();
		let handle = session.prepare().expect("first prepare");
		assert!(session.prepare().is_err(), "second prepare must fail while running");
		assert!(session.write("x".into()).is_ok(), "control channel live after prepare");
		drop(handle);
		assert!(session.write("x".into()).is_err(), "slot must free on handle drop");
		let _second = session.prepare().expect("slot reusable after drop");
	}

	#[cfg(unix)]
	#[test]
	fn runs_argv_command_and_streams_output() {
		let session = PtySession::new();
		let handle = session.prepare().expect("prepare");
		let (tx, rx) = flume::unbounded::<String>();
		let on_chunk: ChunkFn = Box::new(move |text| {
			let _ = tx.send(text.to_string());
		});
		let (pid_tx, pid_rx) = flume::unbounded::<u32>();
		let on_start: StartFn = Box::new(move |pid| {
			let _ = pid_tx.send(pid);
		});
		let result = handle
			.run(
				PtyRunConfig {
					command: PtyCommand::Argv {
						application: "/bin/echo".into(),
						args:        vec!["pty-core-ok".into()],
					},
					cwd:     None,
					env:     None,
					cols:    clamp_cols(None),
					rows:    clamp_rows(None),
				},
				&CancelToken::default(),
				Some(on_chunk),
				Some(on_start),
			)
			.expect("pty run");
		assert_eq!(result.exit_code, Some(0));
		assert!(!result.cancelled && !result.timed_out);
		assert!(pid_rx.try_recv().expect("pid reported") > 0);
		let output: String = rx.drain().collect();
		assert!(output.contains("pty-core-ok"), "missing output: {output:?}");
	}

	#[cfg(unix)]
	#[test]
	fn timeout_reports_timed_out() {
		let session = PtySession::new();
		let handle = session.prepare().expect("prepare");
		let result = handle
			.run(
				PtyRunConfig {
					command: PtyCommand::Shell { command: "sleep 30".into(), shell: None },
					cwd:     None,
					env:     None,
					cols:    80,
					rows:    24,
				},
				&CancelToken::new(Some(200)),
				None,
				None,
			)
			.expect("pty run");
		assert!(result.timed_out, "expected timeout, got {result:?}");
	}

	#[cfg(unix)]
	#[test]
	fn kill_reports_cancelled() {
		let session = PtySession::new();
		let handle = session.prepare().expect("prepare");
		let killer = {
			let session_kill = PtySession { core: Arc::clone(&session.core) };
			std::thread::spawn(move || {
				std::thread::sleep(Duration::from_millis(150));
				let _ = session_kill.kill();
			})
		};
		let result = handle
			.run(
				PtyRunConfig {
					command: PtyCommand::Shell { command: "sleep 30".into(), shell: None },
					cwd:     None,
					env:     None,
					cols:    80,
					rows:    24,
				},
				&CancelToken::default(),
				None,
				None,
			)
			.expect("pty run");
		killer.join().expect("killer thread");
		assert!(result.cancelled, "expected cancelled, got {result:?}");
	}
}
