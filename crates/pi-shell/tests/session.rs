//! Shell session behavior tests, relocated from `pi-natives/src/shell.rs` —
//! they exercise `pi_shell` public API only and never touched the napi layer.

use std::time::Duration;

use pi_shell::{
	ChildSessionAction, Shell, ShellRunOptions,
	cancel::{AbortReason, CancelToken},
	child_session_action,
};
use tokio::time;

mod child_session_action_tests {
	use super::*;

	#[test]
	fn interactive_with_terminal_stdin_takes_foreground() {
		assert_eq!(child_session_action(true, true, false), ChildSessionAction::TakeForeground);
		assert_eq!(child_session_action(true, true, true), ChildSessionAction::TakeForeground);
	}

	#[test]
	fn non_terminal_stdin_detaches_regardless_of_pipeline() {
		assert_eq!(child_session_action(true, false, false), ChildSessionAction::DetachSession);
		// A leading-new-pgroup stage of a pipeline still detaches: setsid keeps
		// it off the host's controlling tty.
		assert_eq!(child_session_action(true, false, true), ChildSessionAction::DetachSession);
	}

	#[test]
	fn non_interactive_with_terminal_stdin_does_nothing() {
		assert_eq!(child_session_action(false, true, false), ChildSessionAction::None);
	}

	#[test]
	fn non_interactive_terminal_stdin_in_pipeline_does_nothing() {
		assert_eq!(child_session_action(false, true, true), ChildSessionAction::None);
	}

	#[test]
	fn embedded_host_with_non_terminal_stdin_detaches() {
		assert_eq!(child_session_action(false, false, false), ChildSessionAction::DetachSession);
	}

	#[test]
	fn pipeline_stage_with_non_terminal_stdin_detaches() {
		// Regression: an interactive child inside a pipeline (`zsh -i | awk`)
		// must not stay in the host session and seize its tty. Pre-fix this
		// returned `None`, leaving the stage attached and able to SIGTTIN the host.
		assert_eq!(child_session_action(false, false, true), ChildSessionAction::DetachSession);
	}
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn embedded_external_command_runs_in_its_own_session() {
	let shell = Shell::new(None);
	let (tx, rx) = flume::unbounded::<String>();
	let handle = tokio::spawn(async move {
		shell
			.run(
				ShellRunOptions {
					command:    "/bin/sh -c 'printf \"%d\\n\" \"$$\"; sleep 0.5'".to_string(),
					cwd:        None,
					env:        None,
					timeout_ms: None,
				},
				Some(tx),
				CancelToken::default(),
			)
			.await
	});
	let child_pid = time::timeout(Duration::from_secs(5), rx.recv_async())
		.await
		.expect("timed out waiting for child pid")
		.expect("missing child pid chunk")
		.trim()
		.parse::<i32>()
		.expect("child pid parses");
	// SAFETY: `getsid(0)` only queries the current process session; the
	// return value is checked below. Inside a PID namespace (e.g. the
	// containerized CI runner) the host's session leader can live outside
	// the namespace, so `getsid(0)` legitimately reports 0 — only -1 is a
	// real failure. The meaningful invariant is that the child detached
	// into its own session (`child_sid == child_pid`, distinct from host).
	let host_sid = unsafe { libc::getsid(0) };
	assert!(host_sid >= 0, "getsid(0) failed: {}", std::io::Error::last_os_error());
	// SAFETY: `child_pid` is a live positive PID reported by the child; the
	// return value is checked below.
	let child_sid = unsafe { libc::getsid(child_pid) };
	assert!(child_sid > 0, "getsid({child_pid}) failed: {}", std::io::Error::last_os_error());
	let result = handle
		.await
		.expect("shell task panicked")
		.expect("shell run");
	assert_eq!(result.exit_code, Some(0));
	assert_ne!(child_sid, host_sid);
	assert_eq!(child_sid, child_pid);
}

#[tokio::test]
async fn read_output_stops_when_cancelled_before_pipe_eof() {
	let shell = Shell::new(None);
	let mut cancel = CancelToken::default();
	let abort = cancel.emplace_abort_token();
	let handle = tokio::spawn(async move {
		shell
			.run(
				ShellRunOptions {
					command:    "sh -c 'sleep 30 & wait'".to_string(),
					cwd:        None,
					env:        None,
					timeout_ms: None,
				},
				None,
				cancel,
			)
			.await
	});

	time::sleep(Duration::from_millis(10)).await;
	abort.abort(AbortReason::Signal);
	let result = time::timeout(Duration::from_secs(3), handle)
		.await
		.expect("shell run should stop after cancellation")
		.expect("shell task should not panic")
		.expect("shell run should return");
	assert!(result.cancelled);
}

#[tokio::test(flavor = "multi_thread")]
async fn timeout_drains_pipeline_output_before_stopping_reader() {
	let shell = Shell::new(None);
	let (tx, rx) = flume::unbounded::<String>();
	// `tail` runs as an in-process builtin, so cancellation kills only the
	// external `yes`; tail then sees EOF and flushes its final 5 lines into
	// the post-cancel reader grace window. The deadline must be generous
	// enough that `yes` has demonstrably spawned and produced before the
	// timeout fires — a 50ms budget lost that race on cold CI runners and
	// tail flushed an empty ring buffer.
	const TIMEOUT_MS: u32 = 750;
	let result = shell
		.run(
			ShellRunOptions {
				command:    "yes x | tail -5".to_string(),
				cwd:        None,
				env:        None,
				timeout_ms: Some(TIMEOUT_MS),
			},
			Some(tx),
			CancelToken::new(Some(TIMEOUT_MS)),
		)
		.await
		.expect("shell run");

	let mut output = String::new();
	while let Ok(chunk) = rx.recv_async().await {
		output.push_str(&chunk);
	}

	assert!(result.timed_out);
	assert_eq!(output.lines().filter(|line| *line == "x").count(), 5);
}
