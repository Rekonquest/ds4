#!/usr/bin/env bash
# vendor-check: assert every third_party subtree carries its expected LICENSE.
# Add new vendored upstreams here as you add them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

declare -A EXPECTED=(
    ["third_party/ggml"]="LICENSE"
    ["third_party/candle-core"]="LICENSE"
    ["third_party/tract-linalg"]="LICENSE-MIT"
    ["third_party/mistralrs-paged-attn"]="LICENSE"
    ["third_party/tgi-proto"]="LICENSE"
)

fail=0
for dir in "${!EXPECTED[@]}"; do
    if [[ ! -d "$dir" ]]; then
        printf 'MISSING directory: %s\n' "$dir"
        fail=1
        continue
    fi
    if [[ ! -f "$dir/${EXPECTED[$dir]}" ]]; then
        printf 'MISSING %s in %s\n' "${EXPECTED[$dir]}" "$dir"
        fail=1
    fi
done

if (( fail )); then
    printf '\nvendor-check FAILED\n'
    exit 1
fi

printf 'vendor-check OK\n'