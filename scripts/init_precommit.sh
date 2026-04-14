#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
echo "Configured git to use .githooks"

uv sync --group dev
echo "Installed dev dependencies"
