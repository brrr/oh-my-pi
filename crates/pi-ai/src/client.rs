//! Messages client: non-streaming completion + SSE streaming.
//!
//! `POST {base}/v1/messages`. The retry contract is a minimal-surface port of
//! `AnthropicMessagesClient` (anthropic-client.ts:88-121 / :223-294):
//! maxRetries=2, retry on connection errors / timeout / 408 / 409 / 429 / 5xx,
//! `x-should-retry` overrides both ways, `retry-after-ms` then `retry-after`
//! (seconds form; HTTP-date form is ignored) then exponential backoff
//! `min(0.5·2^n, 8s)` with 25% jitter. The 600s deadline guards **until the
//! response head arrives** (TS parity) — established SSE streams are not
//! killed by it. Retries stop once a stream is established; mid-stream
//! failures surface as a terminal `error` event (stream resume is WP-1.6
//! hardening).
//!
//! Not ported (yet): custom fetch/TLS injection (Bun-specific), lazy request
//! handles (TS test seam — Rust tests use serialization fixtures instead).

use std::{
	sync::Arc,
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tokio_util::sync::CancellationToken;

use crate::{
	AiError,
	auth::AnthropicAuthConfig,
	builder::StreamingBuilder,
	convert::{RequestMeta, convert_response, emit_nonstream_events, error_to_message},
	event::AssistantMessageEvent,
	message::{AssistantMessage, StopReason},
	sse::{SseParser, parse_message_event, stream_error_message},
	stream::{AssistantMessageEventStream, EventSink},
	wire::{ErrorEnvelope, MessageCreateParams, RawMessageStreamEvent, ResponseMessage},
};

/// Wire-family id stamped on every produced [`AssistantMessage`].
pub const API_ID: &str = "anthropic-messages";

const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(10);
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Client {
	http:        reqwest::Client,
	auth:        AnthropicAuthConfig,
	provider:    String,
	max_retries: u32,
	beta_query:  bool,
}

impl Client {
	/// Build a client for one provider endpoint. `provider` is the configured
	/// provider id recorded on produced messages (e.g. `"deepseek"`).
	#[must_use]
	pub fn new(auth: AnthropicAuthConfig, provider: impl Into<String>) -> Self {
		let http = reqwest::Client::builder()
			.build()
			.expect("reqwest client construction cannot fail with static config");
		Self {
			http,
			auth,
			provider: provider.into(),
			max_retries: DEFAULT_MAX_RETRIES,
			beta_query: false,
		}
	}

	/// Override the retry budget (default 2, matching TS).
	#[must_use]
	pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
		self.max_retries = max_retries;
		self
	}

	/// Append `?beta=true` like the TS client does against the official API.
	#[must_use]
	pub const fn with_beta_query(mut self, beta_query: bool) -> Self {
		self.beta_query = beta_query;
		self
	}

	/// One non-streaming completion, returning the raw wire body.
	///
	/// # Errors
	///
	/// [`AiError::Api`] on a non-2xx after retries, [`AiError::Connection`] /
	/// [`AiError::ConnectionTimeout`] on transport failure,
	/// [`AiError::Decode`] when a 2xx body is not a `ResponseMessage`.
	pub async fn complete(&self, params: &MessageCreateParams) -> Result<ResponseMessage, AiError> {
		let mut request = params.clone();
		request.stream = Some(false);
		let response = self.open_with_retry(&request).await?;
		let body = response.text().await.map_err(AiError::Connection)?;
		serde_json::from_str(&body).map_err(AiError::Decode)
	}

	/// One completion converted into the harness [`AssistantMessage`].
	///
	/// # Errors
	///
	/// Same as [`Client::complete`].
	pub async fn complete_message(
		&self,
		params: &MessageCreateParams,
	) -> Result<AssistantMessage, AiError> {
		let meta = self.request_meta(params);
		let started = Instant::now();
		let result = self.complete(params).await;
		let meta = RequestMeta { duration: Some(elapsed_ms(started)), ..meta };
		result.map(|response| convert_response(&response, &meta))
	}

	/// SSE streaming entry point producing the contract event sequence.
	#[must_use]
	pub fn stream(&self, params: &MessageCreateParams) -> AssistantMessageEventStream {
		self.stream_with_cancel(params, CancellationToken::new())
	}

	/// Like [`Client::stream`], but the caller can cancel mid-stream:
	/// cancellation emits a terminal `error` event with reason `aborted`,
	/// keeping any content accumulated so far on the message.
	#[must_use]
	pub fn stream_with_cancel(
		&self,
		params: &MessageCreateParams,
		cancel: CancellationToken,
	) -> AssistantMessageEventStream {
		let (sink, stream) = AssistantMessageEventStream::channel();
		let client = self.clone();
		let mut request = params.clone();
		tokio::spawn(async move {
			request.stream = Some(true);
			client.run_stream(&request, &sink, &cancel).await;
		});
		stream
	}

	async fn run_stream(
		&self,
		request: &MessageCreateParams,
		sink: &EventSink,
		cancel: &CancellationToken,
	) {
		let meta = self.request_meta(request);
		let started = Instant::now();
		let mut builder = StreamingBuilder::new(&meta);
		// TS pushes `start` before the first byte arrives (anthropic.ts:2027).
		sink.push(AssistantMessageEvent::Start { partial: builder.snapshot() });

		let mut response = match self.open_with_retry(request).await {
			Ok(response) => response,
			Err(error) => {
				// The request never became a stream: standard error turn.
				let message = Arc::new(error_to_message(&error, &meta));
				for event in emit_nonstream_events(&message) {
					sink.push(event);
				}
				return;
			},
		};

		let mut parser = SseParser::new();
		let mut saw_first_content = false;
		loop {
			let chunk = tokio::select! {
				biased;
				() = cancel.cancelled() => {
					builder.set_duration(elapsed_ms(started));
					let (_, events) =
						builder.fail(StopReason::Aborted, AiError::Aborted.to_string());
					push_all(sink, events);
					return;
				},
				chunk = response.chunk() => chunk,
			};
			match chunk {
				Ok(Some(bytes)) => {
					for frame in parser.push(&bytes) {
						if frame.event.as_deref() == Some("error") {
							builder.set_duration(elapsed_ms(started));
							let (_, events) =
								builder.fail(StopReason::Error, stream_error_message(&frame.data));
							push_all(sink, events);
							return;
						}
						let Some(raw) = parse_message_event(&frame) else {
							continue;
						};
						if !saw_first_content
							&& matches!(raw, RawMessageStreamEvent::ContentBlockStart { .. })
						{
							saw_first_content = true;
							builder.set_ttft_once(elapsed_ms(started));
						}
						push_all(sink, builder.on_event(raw));
					}
				},
				Ok(None) => break,
				Err(error) => {
					builder.set_duration(elapsed_ms(started));
					let (_, events) = builder
						.fail(StopReason::Error, format!("Connection error while streaming: {error}"));
					push_all(sink, events);
					return;
				},
			}
		}
		builder.set_duration(elapsed_ms(started));
		let (_, events) = builder.finish();
		push_all(sink, events);
	}

	fn request_meta(&self, params: &MessageCreateParams) -> RequestMeta {
		RequestMeta {
			api:       API_ID.to_string(),
			provider:  self.provider.clone(),
			model:     params.model.clone(),
			timestamp: unix_millis(),
			duration:  None,
		}
	}

	/// Send until a 2xx response head arrives, applying the retry policy.
	/// Non-2xx bodies are consumed for the error envelope.
	async fn open_with_retry(
		&self,
		request: &MessageCreateParams,
	) -> Result<reqwest::Response, AiError> {
		let url = self.auth.messages_url(self.beta_query);
		let mut attempt: u32 = 0;
		loop {
			let failure = match self.send_head(&url, request).await {
				Ok(response) => return Ok(response),
				Err(failure) => failure,
			};
			let retriable = match &failure {
				RequestFailure::Transport(_) | RequestFailure::Timeout => attempt < self.max_retries,
				RequestFailure::Http { retry_hint, status, .. } => {
					let default_retry = matches!(*status, 408 | 409 | 429) || *status >= 500;
					attempt < self.max_retries && retry_hint.unwrap_or(default_retry)
				},
			};
			if !retriable {
				return Err(failure.into_error());
			}
			tokio::time::sleep(failure.retry_delay(attempt)).await;
			attempt += 1;
		}
	}

	async fn send_head(
		&self,
		url: &str,
		request: &MessageCreateParams,
	) -> Result<reqwest::Response, RequestFailure> {
		let mut builder = self
			.http
			.post(url)
			.header("accept", "application/json")
			.header("anthropic-version", ANTHROPIC_VERSION)
			.json(request);
		builder = if self.auth.is_oauth {
			builder.header("authorization", format!("Bearer {}", self.auth.api_key))
		} else {
			builder.header("x-api-key", self.auth.api_key.clone())
		};

		let response = tokio::time::timeout(DEFAULT_TIMEOUT, builder.send())
			.await
			.map_err(|_elapsed| RequestFailure::Timeout)?
			.map_err(|error| {
				if error.is_timeout() {
					RequestFailure::Timeout
				} else {
					RequestFailure::Transport(error)
				}
			})?;
		let status = response.status().as_u16();
		if (200..300).contains(&status) {
			return Ok(response);
		}

		let headers = response.headers().clone();
		let body = response.text().await.unwrap_or_default();
		let retry_hint = headers
			.get("x-should-retry")
			.and_then(|value| value.to_str().ok())
			.and_then(|value| match value {
				"true" => Some(true),
				"false" => Some(false),
				_ => None,
			});
		let retry_after = headers
			.get("retry-after-ms")
			.and_then(|value| value.to_str().ok())
			.and_then(|value| value.parse::<f64>().ok())
			.map(|millis| Duration::from_secs_f64(millis / 1000.0))
			.or_else(|| {
				headers
					.get("retry-after")
					.and_then(|value| value.to_str().ok())
					.and_then(|value| value.parse::<u64>().ok())
					.map(Duration::from_secs)
			});
		let request_id = headers
			.get("request-id")
			.and_then(|value| value.to_str().ok())
			.map(ToOwned::to_owned);
		let parsed = serde_json::from_str::<ErrorEnvelope>(&body)
			.ok()
			.map(|envelope| envelope.error);
		Err(RequestFailure::Http { status, body, parsed, retry_hint, retry_after, request_id })
	}
}

fn push_all(sink: &EventSink, events: Vec<AssistantMessageEvent>) {
	for event in events {
		sink.push(event);
	}
}

enum RequestFailure {
	Transport(reqwest::Error),
	Timeout,
	Http {
		status:      u16,
		body:        String,
		parsed:      Option<crate::wire::ApiErrorBody>,
		retry_hint:  Option<bool>,
		retry_after: Option<Duration>,
		request_id:  Option<String>,
	},
}

impl RequestFailure {
	fn into_error(self) -> AiError {
		match self {
			Self::Transport(error) => AiError::Connection(error),
			Self::Timeout => AiError::ConnectionTimeout,
			Self::Http { status, body, parsed, request_id, .. } => AiError::Api {
				status,
				message: format!("{status} {}", body.trim()),
				body: parsed,
				request_id,
			},
		}
	}

	/// `retry-after-ms` → `retry-after` → capped exponential backoff with 25%
	/// jitter (anthropic-client.ts:263-291).
	fn retry_delay(&self, attempt: u32) -> Duration {
		if let Self::Http { retry_after: Some(delay), .. } = self
			&& *delay <= Duration::from_mins(1)
		{
			return *delay;
		}
		let base = f64::from(1u32 << attempt.min(8)).mul_add(0.5, 0.0).min(8.0);
		Duration::from_secs_f64(base * fastrand::f64().mul_add(-0.25, 1.0))
	}
}

fn elapsed_ms(started: Instant) -> u64 {
	u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn unix_millis() -> i64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}
