//! Messages client: non-streaming completion + SSE streaming.
//!
//! `POST {base}/v1/messages`. The retry contract is a minimal-surface port of
//! `AnthropicMessagesClient` (anthropic-client.ts:88-121 / :223-294):
//! maxRetries=2, retry on connection errors / timeout / 408 / 409 / 429 / 5xx,
//! `x-should-retry` overrides both ways, `retry-after-ms` then `retry-after`
//! (integer-seconds form first, then RFC 7231 §7.1.1.1 IMF-fixdate HTTP-date
//! form resolved against the local clock — TS `Date.parse(retryAfter) -
//! Date.now()` parity) then exponential backoff `min(0.5·2^n, 8s)` with 25%
//! jitter. The 600s deadline guards **until the
//! response head arrives** (TS parity) — established SSE streams are not
//! killed by it.
//!
//! WP-1.6 hardening: once the head arrives, the SSE body is consumed by
//! [`crate::stream_runner::drive_stream`], which layers a retry-before-first-
//! content loop (A1) and a dual first-event/idle watchdog (A2) over the parser
//! and builder. This client supplies the reqwest-backed [`StreamOpener`] and
//! the resolved policy (retry budget plus watchdog deadlines).
//!
//! Not ported (yet): custom fetch/TLS injection (Bun-specific), lazy request
//! handles (TS test seam — Rust tests use serialization fixtures instead).

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use crate::{
	AiError,
	auth::AnthropicAuthConfig,
	convert::{RequestMeta, convert_response},
	message::AssistantMessage,
	stream::AssistantMessageEventStream,
	stream_runner::{
		ChunkStream, DEFAULT_MAX_STREAM_RETRIES, StreamOpener, WatchdogConfig, drive_stream,
	},
	wire::{ErrorEnvelope, MessageCreateParams, ResponseMessage},
};

/// Wire-family id stamped on every produced [`AssistantMessage`].
pub const API_ID: &str = "anthropic-messages";

const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(10);
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Client {
	http:               reqwest::Client,
	auth:               AnthropicAuthConfig,
	provider:           String,
	max_retries:        u32,
	max_stream_retries: u32,
	watchdog:           WatchdogConfig,
	beta_query:         bool,
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
			max_stream_retries: DEFAULT_MAX_STREAM_RETRIES,
			watchdog: WatchdogConfig::from_env(),
			beta_query: false,
		}
	}

	/// Override the head-request retry budget (default 2, matching TS).
	#[must_use]
	pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
		self.max_retries = max_retries;
		self
	}

	/// Override the streaming retry-before-first-content budget (default 10 =
	/// TS `PROVIDER_MAX_RETRIES`).
	#[must_use]
	pub const fn with_max_stream_retries(mut self, max_stream_retries: u32) -> Self {
		self.max_stream_retries = max_stream_retries;
		self
	}

	/// Override the streaming watchdog deadlines (default: resolved from env).
	#[must_use]
	pub const fn with_watchdog(mut self, watchdog: WatchdogConfig) -> Self {
		self.watchdog = watchdog;
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
		request.stream = Some(true);
		let meta = self.request_meta(&request);
		let watchdog = self.watchdog;
		let max_stream_retries = self.max_stream_retries;
		tokio::spawn(async move {
			let opener = ReqwestStreamOpener { client, request };
			drive_stream(&opener, &sink, &cancel, &meta, watchdog, max_stream_retries).await;
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
					.and_then(parse_retry_after)
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

/// Reqwest-backed [`StreamOpener`]: each `open()` re-issues the POST through
/// the head-level retry policy, so the driver's retry-before-first-content loop
/// gets a genuinely fresh connection per attempt.
struct ReqwestStreamOpener {
	client:  Client,
	request: MessageCreateParams,
}

impl StreamOpener for ReqwestStreamOpener {
	type Stream = ReqwestChunkStream;

	async fn open(&self) -> Result<Self::Stream, AiError> {
		let response = self.client.open_with_retry(&self.request).await?;
		Ok(ReqwestChunkStream { response })
	}
}

/// Adapts `reqwest::Response::chunk` to the driver's [`ChunkStream`] surface.
struct ReqwestChunkStream {
	response: reqwest::Response,
}

impl ChunkStream for ReqwestChunkStream {
	async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, String> {
		match self.response.chunk().await {
			Ok(chunk) => Ok(chunk.map(|bytes| bytes.to_vec())),
			Err(error) => Err(error.to_string()),
		}
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

/// Parse a `retry-after` header value into a delay. Integer seconds first
/// (`"120"`), then RFC 7231 §7.1.1.1 IMF-fixdate HTTP-date form
/// (`"Wed, 21 Oct 2015 07:28:00 GMT"`) resolved against the local clock. A date
/// already in the past yields `Duration::ZERO`; an unparseable value yields
/// `None`, letting the backoff fall through to exponential jitter. Mirrors the
/// TS SDK's `parseInt(v) || (Date.parse(v) - Date.now())` two-step.
fn parse_retry_after(value: &str) -> Option<Duration> {
	let trimmed = value.trim();
	if let Ok(secs) = trimmed.parse::<u64>() {
		return Some(Duration::from_secs(secs));
	}
	let target = parse_imf_fixdate(trimmed)?;
	let now = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.ok()?
		.as_secs()
		.try_into()
		.unwrap_or(i64::MAX);
	Some(Duration::from_secs(u64::try_from(target - now).unwrap_or(0)))
}

/// RFC 7231 §7.1.1.1 IMF-fixdate → Unix seconds (GMT only, the sole form a
/// conformant server sends). Example: `Sun, 06 Nov 1994 08:49:37 GMT`. Returns
/// `None` on any structural mismatch (obsolete RFC 850 / asctime forms are not
/// accepted — same as the strict path servers are required to emit).
fn parse_imf_fixdate(value: &str) -> Option<i64> {
	// `Sun, 06 Nov 1994 08:49:37 GMT` → ["Sun,", "06", "Nov", "1994",
	// "08:49:37", "GMT"].
	let parts: Vec<&str> = value.split_whitespace().collect();
	if parts.len() != 6 || parts[5] != "GMT" {
		return None;
	}
	let day: i64 = parts[1].parse().ok()?;
	let month = match parts[2] {
		"Jan" => 1,
		"Feb" => 2,
		"Mar" => 3,
		"Apr" => 4,
		"May" => 5,
		"Jun" => 6,
		"Jul" => 7,
		"Aug" => 8,
		"Sep" => 9,
		"Oct" => 10,
		"Nov" => 11,
		"Dec" => 12,
		_ => return None,
	};
	let year: i64 = parts[3].parse().ok()?;
	let mut hms = parts[4].split(':');
	let hour: i64 = hms.next()?.parse().ok()?;
	let minute: i64 = hms.next()?.parse().ok()?;
	let second: i64 = hms.next()?.parse().ok()?;
	if hms.next().is_some() || !(0..=23).contains(&hour) || minute > 59 || second > 60 {
		return None;
	}
	Some(days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second)
}

/// Days since the Unix epoch (1970-01-01) for a proleptic-Gregorian date.
/// Howard Hinnant's `days_from_civil` (public-domain chrono algorithm).
const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
	let y = if month <= 2 { year - 1 } else { year };
	let era = if y >= 0 { y } else { y - 399 } / 400;
	let yoe = y - era * 400;
	let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
	let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
	era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::{days_from_civil, parse_imf_fixdate, parse_retry_after};

	#[test]
	fn imf_fixdate_known_epochs() {
		// Unix epoch and a few reference points (verified against `date -u`).
		assert_eq!(days_from_civil(1970, 1, 1), 0);
		assert_eq!(parse_imf_fixdate("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
		// 2015-10-21 07:28:00 UTC = 1_445_412_480 (RFC 7231 canonical example).
		assert_eq!(parse_imf_fixdate("Wed, 21 Oct 2015 07:28:00 GMT"), Some(1_445_412_480));
		// Leap day.
		assert_eq!(parse_imf_fixdate("Mon, 29 Feb 2016 00:00:00 GMT"), Some(1_456_704_000));
	}

	#[test]
	fn imf_fixdate_rejects_malformed() {
		assert_eq!(parse_imf_fixdate("Wed, 21 Oct 2015 07:28:00 PST"), None); // non-GMT zone
		assert_eq!(parse_imf_fixdate("21 Oct 2015 07:28:00 GMT"), None); // missing weekday
		assert_eq!(parse_imf_fixdate("Wed, 21 Foo 2015 07:28:00 GMT"), None); // bad month
		assert_eq!(parse_imf_fixdate("Wed, 21 Oct 2015 07:28 GMT"), None); // truncated time
		assert_eq!(parse_imf_fixdate("Sunday, 06-Nov-94 08:49:37 GMT"), None); // RFC 850 form
	}

	#[test]
	fn retry_after_prefers_integer_seconds() {
		assert_eq!(parse_retry_after("45"), Some(Duration::from_secs(45)));
		assert_eq!(parse_retry_after("  7  "), Some(Duration::from_secs(7)));
	}

	#[test]
	fn retry_after_http_date_relative_to_now() {
		// A far-past date clamps to zero rather than going negative.
		let past = parse_retry_after("Thu, 01 Jan 1970 00:00:00 GMT").expect("parses");
		assert_eq!(past, Duration::ZERO);
		// A future date yields a positive, sane delay (well under a year).
		let year_3000 = parse_retry_after("Sat, 01 Jan 3000 00:00:00 GMT").expect("parses");
		assert!(year_3000 > Duration::ZERO);
	}

	#[test]
	fn retry_after_garbage_is_none() {
		assert_eq!(parse_retry_after("soon"), None);
		assert_eq!(parse_retry_after(""), None);
	}
}
