#!/usr/bin/env bash
# Install Traceome and store the Nebius Token Factory key where the CLI reads it.
#
#   ./scripts/setup-keys.sh            # prompts for the key
#   NEBIUS_API_KEY=... ./scripts/setup-keys.sh
#
# The key goes to .bio-neb-local/model.env (directory 0700, file 0600), which git ignores.
# Nothing here contacts Nebius.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --locked

key="${NEBIUS_API_KEY:-}"
if [ -z "$key" ]; then
    read -r -s -p "Nebius Token Factory API key: " key
    echo
fi
if ! [[ "$key" =~ ^[A-Za-z0-9_./:+,@%=-]+$ ]]; then
    echo "The key has characters a plain env file cannot hold unquoted; check it and retry." >&2
    exit 1
fi

umask 077
mkdir -p .bio-neb-local/live-runs
chmod 700 .bio-neb-local .bio-neb-local/live-runs
printf 'NEBIUS_API_KEY=%s\n' "$key" > .bio-neb-local/model.env
chmod 600 .bio-neb-local/model.env

# Offline checks: the resolved locations and the model configuration that needs the key.
uv run --locked python -m bio_neb.locations >/dev/null
uv run --locked --env-file .bio-neb-local/model.env python -m bio_neb.cli runtime-drift >/dev/null

cat <<NEXT
Ready. Start a Run with:

  uv run --env-file .bio-neb-local/model.env python -m bio_neb.cli begin \\
    --run-root $PWD/.bio-neb-local/live-runs/<unique-run-id> \\
    --prompt-file /abs/path/question.txt \\
    --fastq read_1=/abs/path/R1.fastq.gz --fastq read_2=/abs/path/R2.fastq.gz --interactive
NEXT
