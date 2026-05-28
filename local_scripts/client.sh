#!/usr/bin/env bash
# Process a directory of PDFs/images by sharding the file list across N
# already-running vLLM servers (launched by server.sh), balanced by total
# page count per shard, with per-shard progress bars in the terminal.
#
# Usage:
#   ./client.sh <input_dir> <output_dir> [num_servers] [extra chandra args...]
#
# Example:
#   ./client.sh ~/datasets/ecclesia/raw ~/datasets/ecclesia/out 8 --no-images
#
# Env vars:
#   BASE_PORT              base port of the first vLLM server (default 8000)
#   MAX_WORKERS_PER_SHARD  chandra --max-workers per shard (default 16)
#   MAX_RETRIES            chandra --max-retries (default 6)
#   BATCH_SIZE             chandra --batch-size (default 28)
#   SKIP_EXISTING          0/1 -- skip files whose <stem>/<stem>.md exists (default 1)
#
# Assumes servers are at http://localhost:8000/v1 ... http://localhost:(8000+N-1)/v1.
# Run inside the chandra-vllm conda env (so `chandra` is on PATH and pypdfium2
# is importable).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/client.py" "$@"
