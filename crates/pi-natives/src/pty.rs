//! PTY-backed interactive command execution exported via N-API.
//!
//! # Overview
//! Thin adapter over `pi_shell::pty` (where the PTY session logic lives since
//! the pi-core extraction): napi options → core config, threadsafe-function
//! callbacks → boxed closures, `anyhow::Error` → napi `Error`.

use std::{collections::HashMap, sync::Arc};

use napi::{
	bindgen_prelude::*,
	threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use napi_derive::napi;
use pi_shell::pty::{
	ChunkFn, PtyCommand, PtyRunConfig, PtySession as CorePtySession, StartFn, clamp_cols, clamp_rows,
};

use crate::task;

/// Options for running a command in a PTY session.
#[napi(object)]
pub struct PtyStartOptions<'env> {
	/// Command string to execute.
	pub command:    String,
	/// Working directory for command execution.
	pub cwd:        Option<String>,
	/// Environment variables for this command.
	pub env:        Option<HashMap<String, String>>,
	/// Timeout in milliseconds before cancelling.
	pub timeout_ms: Option<u32>,
	/// Abort signal for cancelling the operation.
	pub signal:     Option<Unknown<'env>>,
	/// PTY column count.
	pub cols:       Option<u16>,
	/// PTY row count.
	pub rows:       Option<u16>,
	/// Shell binary to use (e.g. "sh", "bash", or an absolute path).
	/// Defaults to "sh" if not provided.
	pub shell:      Option<String>,
}

/// Options for running an executable and argument vector in a PTY session.
#[napi(object)]
pub struct PtyArgvStartOptions<'env> {
	/// Executable name or path.
	pub application: String,
	/// Arguments passed directly to the executable.
	pub args:        Vec<String>,
	/// Working directory for command execution.
	pub cwd:         Option<String>,
	/// Environment variables for this command.
	pub env:         Option<HashMap<String, String>>,
	/// Timeout in milliseconds before cancelling.
	pub timeout_ms:  Option<u32>,
	/// Abort signal for cancelling the operation.
	pub signal:      Option<Unknown<'env>>,
	/// PTY column count.
	pub cols:        Option<u16>,
	/// PTY row count.
	pub rows:        Option<u16>,
}

/// Result of a PTY command run.
#[napi(object)]
pub struct PtyRunResult {
	/// Exit code when the command completes.
	pub exit_code: Option<i32>,
	/// Whether command was cancelled by signal/user kill.
	pub cancelled: bool,
	/// Whether command timed out.
	pub timed_out: bool,
}

impl From<pi_shell::pty::PtyRunResult> for PtyRunResult {
	fn from(result: pi_shell::pty::PtyRunResult) -> Self {
		Self { exit_code: result.exit_code, cancelled: result.cancelled, timed_out: result.timed_out }
	}
}

/// Stateful PTY session for interactive stdin/stdout passthrough.
#[napi]
pub struct PtySession {
	inner: Arc<CorePtySession>,
}

impl Default for PtySession {
	fn default() -> Self {
		Self::new()
	}
}

#[napi]
impl PtySession {
	#[napi(constructor)]
	pub fn new() -> Self {
		Self { inner: Arc::new(CorePtySession::new()) }
	}

	/// Start a shell command, stream output chunks, and report the spawned child
	/// PID.
	#[napi]
	pub fn start<'env>(
		&self,
		env: &'env Env,
		options: PtyStartOptions<'env>,
		#[napi(ts_arg_type = "((error: Error | null, chunk: string) => void) | undefined | null")]
		on_chunk: Option<ThreadsafeFunction<String>>,
		#[napi(ts_arg_type = "((error: Error | null, pid: number) => void) | undefined | null")]
		on_start: Option<ThreadsafeFunction<u32>>,
	) -> Result<PromiseRaw<'env, PtyRunResult>> {
		let run_config = PtyRunConfig {
			command: PtyCommand::Shell { command: options.command, shell: options.shell },
			cwd:     options.cwd,
			env:     options.env,
			cols:    clamp_cols(options.cols),
			rows:    clamp_rows(options.rows),
		};
		self.start_config(env, run_config, options.timeout_ms, options.signal, on_chunk, on_start)
	}

	/// Start an executable with separate arguments, stream output chunks, and
	/// report the spawned child PID.
	#[napi]
	pub fn start_argv<'env>(
		&self,
		env: &'env Env,
		options: PtyArgvStartOptions<'env>,
		#[napi(ts_arg_type = "((error: Error | null, chunk: string) => void) | undefined | null")]
		on_chunk: Option<ThreadsafeFunction<String>>,
		#[napi(ts_arg_type = "((error: Error | null, pid: number) => void) | undefined | null")]
		on_start: Option<ThreadsafeFunction<u32>>,
	) -> Result<PromiseRaw<'env, PtyRunResult>> {
		let run_config = PtyRunConfig {
			command: PtyCommand::Argv { application: options.application, args: options.args },
			cwd:     options.cwd,
			env:     options.env,
			cols:    clamp_cols(options.cols),
			rows:    clamp_rows(options.rows),
		};
		self.start_config(env, run_config, options.timeout_ms, options.signal, on_chunk, on_start)
	}

	/// Write raw input bytes to PTY stdin.
	#[napi]
	pub fn write(&self, data: String) -> Result<()> {
		self.inner.write(data).map_err(to_napi_error)
	}

	/// Resize the active PTY.
	#[napi]
	pub fn resize(&self, cols: u16, rows: u16) -> Result<()> {
		self.inner.resize(cols, rows).map_err(to_napi_error)
	}

	/// Force-kill the active PTY command.
	#[napi]
	pub fn kill(&self) -> Result<()> {
		self.inner.kill().map_err(to_napi_error)
	}
}

impl PtySession {
	fn start_config<'env>(
		&self,
		env: &'env Env,
		run_config: PtyRunConfig,
		timeout_ms: Option<u32>,
		signal: Option<Unknown<'env>>,
		on_chunk: Option<ThreadsafeFunction<String>>,
		on_start: Option<ThreadsafeFunction<u32>>,
	) -> Result<PromiseRaw<'env, PtyRunResult>> {
		let ct = task::CancelToken::new(timeout_ms, signal);
		// Register the control channel synchronously so write()/kill() work
		// immediately; the handle frees the session slot on drop.
		let handle = self.inner.prepare().map_err(to_napi_error)?;
		let on_chunk: Option<ChunkFn> = on_chunk.map(|callback| {
			Box::new(move |text: &str| {
				callback.call(Ok(text.to_string()), ThreadsafeFunctionCallMode::NonBlocking);
			}) as ChunkFn
		});
		let on_start: Option<StartFn> = on_start.map(|callback| {
			Box::new(move |pid: u32| {
				callback.call(Ok(pid), ThreadsafeFunctionCallMode::NonBlocking);
			}) as StartFn
		});
		task::future(env, "pty.start", async move {
			let run_result = tokio::task::spawn_blocking(move || {
				handle.run(run_config, &ct.into_core(), on_chunk, on_start)
			})
			.await;
			match run_result {
				Ok(Ok(result)) => Ok(result.into()),
				Ok(Err(err)) => Err(to_napi_error(err)),
				Err(err) => Err(Error::from_reason(format!("PTY execution task failed: {err}"))),
			}
		})
	}
}

fn to_napi_error(err: anyhow::Error) -> Error {
	Error::from_reason(err.to_string())
}
