#!/usr/bin/env bash
# Start a Traceome Run and follow it in the researcher workspace (the GUI).
#
#   ./scripts/start-run.sh "Does this sample carry rs4149056 and rs429358?" R1.fastq.gz R2.fastq.gz [--paper paper.pdf]
#
# Then open http://localhost:8770 and answer each Review there. Run ./scripts/setup-keys.sh first.
# --dry-run prints the command instead of starting the Run.
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$PWD"

usage() { sed -n '4,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 3 ] || usage
question="$1"; read_1="$2"; read_2="$3"; shift 3
paper=(); dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --paper) [ $# -ge 2 ] || usage; paper+=(--paper-file "$(realpath "$2")"); shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) usage ;;
    esac
done

env_file="$repo/.bio-neb-local/model.env"
[ -f "$env_file" ] || { echo "No key yet: run ./scripts/setup-keys.sh first." >&2; exit 1; }
for f in "$read_1" "$read_2"; do
    [ -f "$f" ] || { echo "Not a file: $f" >&2; exit 1; }
done

# A fresh, unique Run: its name also names the Run's storage bucket.
run_id="run-$(date -u +%Y%m%dt%H%M%Sz)"
run_root="$repo/.bio-neb-local/live-runs/$run_id"
prompt_file="$repo/.bio-neb-local/prompts/$run_id.txt"
mkdir -p "$(dirname "$prompt_file")"
printf '%s\n' "$question" > "$prompt_file"

command=(uv run --locked python start_run.py --run-root "$run_root" --env-file "$env_file"
         --prompt-file "$prompt_file" --fastq "read_1=$(realpath "$read_1")" --fastq "read_2=$(realpath "$read_2")"
         ${paper[@]+"${paper[@]}"})
if [ "$dry_run" = 1 ]; then
    printf '%q ' cd gui '&&' "${command[@]}"; echo
    exit 0
fi
cd gui
uv sync --locked
echo "Starting $run_id. Open http://localhost:8770 to follow it and answer its Reviews."
exec "${command[@]}"
