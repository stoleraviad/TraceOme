"""Start a BioNeb Run and follow it in the GUI: `cli begin` detached, then the app live on the Run's Journey.

    uv run python start_run.py --run-root <new folder outside the repository> --env-file <controller.env> \\
        --prompt-file <prompt.txt> --fastq read_1=<path or s3://…> --fastq read_2=<…> \\
        [--paper-file <file>] [--request-file <file>] [--runtime-config <file>] [--port 8770]

`begin` runs the first pass and stops at the first Review; from then on the GUI answers each Review (`answer`, with
the same credentials file). The Run outlives the GUI: restart the app on it with
`LIVE=1 BIONEB_ENV_FILE=<env file> RUN_DIR=<run root>/<journey id> uv run python app.py`.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cli_bridge

HERE = Path(__file__).resolve().parent


def env_file_problem(path: Path) -> str | None:
    """Why the credentials file cannot be used, read from its mode only (its values are never read here)."""
    if not path.is_file():
        return f"{path} is not a file"
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        return f"{path} must be readable by its owner only (chmod 600)"
    return None


def wait_for_journey(run_root: Path, process: subprocess.Popen, timeout: float = 300, poll: float = 0.5) -> Path | None:
    """The Run's Journey folder (`<run root>/<journey id>`) once `begin` has created its journey.jsonl; None when
    `begin` ended first or the wait timed out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = sorted(run_root.glob("*/journey.jsonl"))
        if found:
            return found[0].parent
        if process.poll() is not None:
            return None
        time.sleep(poll)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--fastq", action="append", required=True, metavar="ROLE=PATH|s3://BUCKET/KEY")
    parser.add_argument("--paper-file", type=Path, action="append", default=[])
    parser.add_argument("--request-file", type=Path, action="append", default=[])
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--port", default=os.environ.get("PORT", "8770"))
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if run_root.exists():
        parser.error(f"{run_root} exists: every Run takes a new folder")
    if problem := env_file_problem(args.env_file):
        parser.error(problem)
    os.environ["BIONEB_ENV_FILE"] = str(args.env_file.resolve())
    begin = cli_bridge.begin_argv(
        run_root, args.prompt_file.resolve(), args.fastq, paper_files=[f.resolve() for f in args.paper_file],
        request_files=[f.resolve() for f in args.request_file], runtime_config=args.runtime_config and args.runtime_config.resolve(),
    )  # fmt: skip
    log = Path(os.environ.get("COMMAND_LOGS") or Path(tempfile.gettempdir()) / "bioneb-gui-commands") / f"begin-{run_root.name}"
    process = cli_bridge.start_detached(begin, log)
    print(f"begin started (pid {process.pid}); its output: {log}.out, {log}.err", flush=True)
    journey = wait_for_journey(run_root, process)
    if journey is None:
        refusal = cli_bridge.refusal(process.wait(), log) if process.poll() is not None else {"code": "no_journey_yet"}
        print(f"No Journey to follow: {refusal}", file=sys.stderr)
        return 1
    print(f"Following {journey} on http://localhost:{args.port}", flush=True)
    env = {**os.environ, "LIVE": "1", "RUN_DIR": str(journey), "PORT": str(args.port)}
    os.chdir(HERE)
    os.execve(sys.executable, [sys.executable, str(HERE / "app.py")], env)


if __name__ == "__main__":
    sys.exit(main())
