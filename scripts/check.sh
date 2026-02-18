#!/bin/bash
# check.sh — run all quality checks (format check + tests)
# Exit code is non-zero if any check fails.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0

run_check() {
    local name="$1"
    shift
    echo ""
    echo "==> $name"
    if "$@"; then
        echo "    PASS"
        PASS=$((PASS + 1))
    else
        echo "    FAIL"
        FAIL=$((FAIL + 1))
    fi
}

run_check "Black format check" uv run black --check backend/ main.py
run_check "Pytest"             uv run pytest backend/tests/ -q

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
