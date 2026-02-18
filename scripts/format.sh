#!/bin/bash
# format.sh — auto-format all Python source files using black

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Running black formatter..."
uv run black backend/ main.py
echo "Done."
