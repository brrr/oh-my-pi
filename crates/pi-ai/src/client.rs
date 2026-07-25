//! Non-streaming Messages client.
//!
//! `POST {base}/v1/messages` with `stream: false`. The TS client only ships an
//! SSE path (anthropic-client.ts forces `stream: true`), so the non-streaming
//! parse is new code; the retry contract is a minimal-surface port of
//! `AnthropicMessagesClient` (anthropic-client.ts:88-121 / :223-294):
//! maxRetries=2, retry on connection errors / 408 / 409 / 429 / 5xx,
//! `x-should-retry` overrides both ways, `retry-after-ms` then `retry-after`
//! (seconds form; HTTP-date form is ignored) then exponential backoff
//! `min(0.5·2^n, 8s)` with 25% jitter, 600s pre-response timeout.
//!
//! Not ported (yet): caller abort wiring (`AiError::Aborted` reserved for
//! WP-1.1b), custom fetch/TLS injection (Bun-specific), lazy request handles
//! (TS test seam — Rust tests use serialization fixtures instead).

use std::{
	sync::Arc,
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
	AiError,
	auth::AnthropicAuthConfig,
	convert::{RequestMeta, convert_response, emit_nonstream_events, error_to_message},
	message::AssistantMessage,
	stream::AssistantMessageEventStream,
	wire::{ErrorEnvelope, MessageCreateParams, ResponseMessage},
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
			.timeout(DEFAULT_TIMEOUT)
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
		let url = self.auth.messages_url(self.beta_query);

		let mut attempt: u32 = 0;
		loop {
			let failure = match self.send_once(&url, &request).await {
				Ok(response) => return Ok(response),
				Err(failure) => failure,
			};
			let retriable = match &failure {
				RequestFailure::Transport(_) => attempt < self.max_retries,
				RequestFailure::Http { retry_hint, status, .. } => {
					let default_retry = matches!(*status, 408 | 409 | 429) || *status >= 500;
					attempt < self.max_retries && retry_hint.unwrap_or(default_retry)
				},
				RequestFailure::Decode(_) => false,
			};
			if !retriable {
				return Err(failure.into_error());
			}
			tokio::time::sleep(failure.retry_delay(attempt)).await;
			attempt += 1;
		}
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
		let meta = RequestMeta {
			duration: Some(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)),
			..meta
		};
		result.map(|response| convert_response(&response, &meta))
	}

	/// Event-stream entry point. WP-1.1a: performs the non-streaming call and
	/// synthesizes the contract event sequence (a request failure becomes a
	/// single `error` event). WP-1.1b replaces the internals with true SSE
	/// behind this same signature.
	#[must_use]
	pub fn stream(&self, params: &MessageCreateParams) -> AssistantMessageEventStream {
		let (sink, stream) = AssistantMessageEventStream::channel();
		let client = self.clone();
		let params = params.clone();
		tokio::spawn(async move {
			let meta = client.request_meta(&params);
			let message = match client.complete_message(&params).await {
				Ok(message) => message,
				Err(error) => error_to_message(&error, &meta),
			};
			let message = Arc::new(message);
			for event in emit_nonstream_events(&message) {
				sink.push(event);
			}
		});
		stream
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

	async fn send_once(
		&self,
		url: &str,
		request: &MessageCreateParams,
	) -> Result<ResponseMessage, RequestFailure> {
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

		let response = builder.send().await.map_err(RequestFailure::Transport)?;
		let status = response.status().as_u16();
		let headers = response.headers().clone();
		let body = response.text().await.map_err(RequestFailure::Transport)?;

		if (200..300).contains(&status) {
			return serde_json::from_str(&body).map_err(RequestFailure::Decode);
		}

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

enum RequestFailure {
	Transport(reqwest::Error),
	Http {
		status:      u16,
		body:        String,
		parsed:      Option<crate::wire::ApiErrorBody>,
		retry_hint:  Option<bool>,
		retry_after: Option<Duration>,
		request_id:  Option<String>,
	},
	Decode(serde_json::Error),
}

impl RequestFailure {
	fn into_error(self) -> AiError {
		match self {
			Self::Transport(error) if error.is_timeout() => AiError::ConnectionTimeout,
			Self::Transport(error) => AiError::Connection(error),
			Self::Http { status, body, parsed, request_id, .. } => AiError::Api {
				status,
				message: format!("{status} {}", body.trim()),
				body: parsed,
				request_id,
			},
			Self::Decode(error) => AiError::Decode(error),
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

fn unix_millis() -> i64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}
