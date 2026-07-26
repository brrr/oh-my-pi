#!/usr/bin/env bash
# Invariant guard for the pi-core extraction (headless roadmap P0): the pure
# Rust core crates must never grow a napi/node dependency — only pi-natives is
# allowed to touch N-API. Run from the repo root; exits non-zero on violation.
set -euo pipefail

CORE_CRATES=(pi-shell pi-ai pi-grep pi-diff pi-ast pi-term pi-tools)

for crate in "${CORE_CRATES[@]}"; do
	if cargo tree -p "$crate" -e normal | grep -qi '\bnapi\b'; then
		echo "FAIL: $crate dependency tree contains napi:" >&2
		cargo tree -p "$crate" -e normal -i napi >&2 || true
		exit 1
	fi
	echo "OK: $crate is napi-free"
done
