#!/usr/bin/env bash
set -euo pipefail

# Example env vars — override as needed
export WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-/path/to/whisper-large-v3/snapshots/xxx}"
export OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
export OLLAMA_PORT="${OLLAMA_PORT:-11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma-4-E4b-it:latest}"
export ASR_DEVICE="${ASR_DEVICE:-cuda:0}"

cd "$(dirname "$0")/.."

python -m whisper_asr "$@"
