//! ISO-8601 ↔ unix-milliseconds conversion, dependency-free.
//!
//! The session journal stores two timestamp shapes side by side by design (see
//! `entries.rs`): envelope timestamps are ISO-8601 strings written by the TS
//! `nowIso()` = `new Date().toISOString()`, while nested `message.timestamp`
//! values are unix-ms numbers. The compaction / branch-summary / custom
//! synthetic messages derive their unix-ms `timestamp` from the ISO envelope
//! via the TS `new Date(iso).getTime()`, so the loader needs a faithful
//! ISO→unix-ms parse to reproduce those values byte-for-byte.
//!
//! Only the exact `YYYY-MM-DDTHH:MM:SS.sssZ` (UTC, millisecond) shape emitted
//! by `Date.prototype.toISOString()` is parsed; anything else is an error. The
//! civil-date math is Howard Hinnant's `days_from_civil` / `civil_from_days`
//! (public domain), valid across the full proleptic Gregorian range.

use anyhow::{Context, Result, bail};

const MS_PER_DAY: i64 = 86_400_000;

/// Days since the unix epoch (1970-01-01) for a proleptic Gregorian date.
const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
	let y = if m <= 2 { y - 1 } else { y };
	let era = if y >= 0 { y } else { y - 399 } / 400;
	let yoe = y - era * 400; // [0, 399]
	let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
	let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
	era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`]: unix-epoch day count → `(year, month, day)`.
const fn civil_from_days(z: i64) -> (i64, i64, i64) {
	let z = z + 719_468;
	let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
	let doe = z - era * 146_097; // [0, 146096]
	let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
	let y = yoe + era * 400;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
	let mp = (5 * doy + 2) / 153; // [0, 11]
	let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
	let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
	(if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse a `Date.prototype.toISOString()` string to unix milliseconds.
pub fn iso_to_unix_ms(iso: &str) -> Result<i64> {
	let bytes = iso.as_bytes();
	// Fixed layout: 2026-01-01T00:00:00.000Z (24 chars).
	if bytes.len() != 24 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
		bail!("unexpected ISO-8601 timestamp shape: {iso:?}");
	}
	if bytes[13] != b':' || bytes[16] != b':' || bytes[19] != b'.' || bytes[23] != b'Z' {
		bail!("unexpected ISO-8601 timestamp shape: {iso:?}");
	}
	let field = |lo: usize, hi: usize| -> Result<i64> {
		iso[lo..hi]
			.parse::<i64>()
			.with_context(|| format!("bad numeric field in {iso:?}"))
	};
	let year = field(0, 4)?;
	let month = field(5, 7)?;
	let day = field(8, 10)?;
	let hour = field(11, 13)?;
	let minute = field(14, 16)?;
	let second = field(17, 19)?;
	let millis = field(20, 23)?;
	let days = days_from_civil(year, month, day);
	Ok(days * MS_PER_DAY + hour * 3_600_000 + minute * 60_000 + second * 1000 + millis)
}

/// Format unix milliseconds as a `Date.prototype.toISOString()` string.
pub fn unix_ms_to_iso(ms: i64) -> String {
	let days = ms.div_euclid(MS_PER_DAY);
	let rem = ms.rem_euclid(MS_PER_DAY);
	let (year, month, day) = civil_from_days(days);
	let millis = rem % 1000;
	let secs = rem / 1000;
	let second = secs % 60;
	let minute = (secs / 60) % 60;
	let hour = secs / 3600;
	format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn roundtrip_known_instants() {
		// 2026-01-01T00:00:00.000Z — the fixture wall clock.
		assert_eq!(iso_to_unix_ms("2026-01-01T00:00:00.000Z").unwrap(), 1_767_225_600_000);
		// 2024-01-01T00:00:00.000Z — the fixture message clock.
		assert_eq!(iso_to_unix_ms("2024-01-01T00:00:00.000Z").unwrap(), 1_704_067_200_000);
		// unix epoch.
		assert_eq!(iso_to_unix_ms("1970-01-01T00:00:00.000Z").unwrap(), 0);
	}

	#[test]
	fn format_matches_iso() {
		assert_eq!(unix_ms_to_iso(1_767_225_600_000), "2026-01-01T00:00:00.000Z");
		assert_eq!(unix_ms_to_iso(0), "1970-01-01T00:00:00.000Z");
		assert_eq!(unix_ms_to_iso(1_704_067_200_123), "2024-01-01T00:00:00.123Z");
	}

	#[test]
	fn format_parse_roundtrip() {
		for ms in [0_i64, 1_704_067_200_000, 1_767_225_600_999, 1_600_000_000_000] {
			assert_eq!(iso_to_unix_ms(&unix_ms_to_iso(ms)).unwrap(), ms);
		}
	}

	#[test]
	fn rejects_bad_shape() {
		assert!(iso_to_unix_ms("2026-01-01").is_err());
		assert!(iso_to_unix_ms("2026-01-01T00:00:00Z").is_err());
	}
}
