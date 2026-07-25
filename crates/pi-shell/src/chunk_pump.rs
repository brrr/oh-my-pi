//! Bounded chunk bridge: pipe readers → a slow downstream consumer.
//!
//! Extracted from `pi-natives/src/shell.rs` (#4078). The natives crate keeps
//! only the threadsafe-function adapter; the queue-bounding and coalescing
//! policy lives here so any in-process consumer (headless agent loop, tests)
//! gets identical backpressure behavior.

/// Capacity (in chunks) of the queue between the pipe readers and the
/// forwarding pump.
///
/// One queued chunk is at most one pipe read (≤64 KiB), so
/// the bridge holds ~4 MiB worst case before the readers' `send_async` parks —
/// which in turn parks the child on its stdout/stderr pipe (ordinary pipe
/// backpressure) instead of buffering the surplus in process memory (#4078).
pub const BRIDGE_QUEUE_CHUNKS: usize = 64;

/// Drain `rx`, greedily coalescing queued chunks into ≤64 KiB batches, and
/// feed each batch to `forward`, awaiting its completion before pulling more.
///
/// Returns when `rx` disconnects (all senders dropped) or `forward` reports
/// the consumer is gone; dropping `rx` then disconnects the channel so
/// parked/future senders fail fast and the pipe readers keep draining the
/// child instead of wedging it.
pub async fn pump_chunks(
	rx: flume::Receiver<String>,
	mut forward: impl AsyncFnMut(String) -> bool,
) {
	// Hard cap on one coalesced batch so the consumer never sees a multi-MB
	// callback (a giant single string would stall downstream processing for
	// the whole copy).
	const MAX_BATCH_BYTES: usize = 64 * 1024;
	// Initial capacity sized for typical bursty pipe output. Re-allocated
	// each batch because `String` ownership is moved into the forward call.
	const INITIAL_BATCH_CAP: usize = 8 * 1024;
	let mut batch = String::with_capacity(INITIAL_BATCH_CAP);
	while let Ok(first) = rx.recv_async().await {
		batch.push_str(&first);
		// Greedily drain everything already queued. Child processes that
		// write byte-at-a-time (printf-style progress, llama-cli token
		// streams) otherwise produce one callback per `write(2)`, saturating
		// the consumer and leaving the queue draining long after the child
		// exits.
		while batch.len() < MAX_BATCH_BYTES {
			match rx.try_recv() {
				Ok(more) => batch.push_str(&more),
				Err(_) => break,
			}
		}
		let payload = std::mem::replace(&mut batch, String::with_capacity(INITIAL_BATCH_CAP));
		if !forward(payload).await {
			return;
		}
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use tokio::time;

	use super::*;

	/// Regression for #4078: the reader→consumer bridge queue must stay
	/// bounded when the consumer (here: a deliberately slow `forward`) cannot
	/// keep up with a fast producer, and backpressure must never drop or
	/// reorder chunks. On the pre-fix bridge (`flume::unbounded` +
	/// fire-and-forget forwarding) the same harness accumulates the producer's
	/// entire surplus in the queue (measured: a 32 MiB stream queued all
	/// `33_554_432` bytes while the consumer stalled).
	#[tokio::test(flavor = "multi_thread")]
	async fn bridge_pump_bounds_queue_and_delivers_all_bytes() {
		const CHUNKS: usize = 512;
		const CHUNK_BYTES: usize = 4096;
		let (tx, rx) = flume::bounded::<String>(BRIDGE_QUEUE_CHUNKS);
		let producer = tokio::spawn(async move {
			let mut expected = String::with_capacity(CHUNKS * CHUNK_BYTES);
			let mut max_queued = 0usize;
			for i in 0..CHUNKS {
				let chunk = format!("[{i:06}]{}", "x".repeat(CHUNK_BYTES - 8));
				expected.push_str(&chunk);
				tx.send_async(chunk)
					.await
					.expect("pump should outlive the producer");
				max_queued = max_queued.max(tx.len());
			}
			(expected, max_queued)
		});

		let mut received = String::with_capacity(CHUNKS * CHUNK_BYTES);
		time::timeout(
			Duration::from_secs(30),
			pump_chunks(rx, async |payload: String| {
				received.push_str(&payload);
				// Emulate a busy consumer: each callback takes a while.
				time::sleep(Duration::from_micros(500)).await;
				true
			}),
		)
		.await
		.expect("pump should finish once the producer hangs up");

		let (expected, max_queued) = producer.await.expect("producer task");
		assert!(
			max_queued <= BRIDGE_QUEUE_CHUNKS,
			"bridge queue grew past its bound: {max_queued} chunks",
		);
		assert_eq!(received.len(), expected.len(), "bytes were dropped or duplicated");
		assert_eq!(received, expected, "chunks must arrive losslessly and in order");
	}

	/// When the consumer dies (`forward` fails), the pump must drop its
	/// receiver so parked and future sends fail fast — the pipe readers keep
	/// draining the child instead of wedging it on a full bridge queue.
	#[tokio::test(flavor = "multi_thread")]
	async fn bridge_pump_death_disconnects_channel_without_blocking_senders() {
		let (tx, rx) = flume::bounded::<String>(4);
		let pump = tokio::spawn(pump_chunks(rx, async |_payload: String| false));
		let producer = tokio::spawn(async move {
			let mut disconnected = 0usize;
			for _ in 0..64 {
				if tx.send_async("x".repeat(1024)).await.is_err() {
					disconnected += 1;
				}
			}
			disconnected
		});
		let disconnected = time::timeout(Duration::from_secs(5), producer)
			.await
			.expect("sends must not park once the consumer died")
			.expect("producer task");
		assert!(disconnected > 0, "channel should disconnect after the pump stops");
		time::timeout(Duration::from_secs(5), pump)
			.await
			.expect("pump should exit after forward fails")
			.expect("pump task");
	}
}
