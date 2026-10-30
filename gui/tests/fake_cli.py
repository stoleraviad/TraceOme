"""A stand-in for `uv run ... python -m bio_neb.cli` in the GUI tests: it answers `observe` and plays `answer` and
`resume` from a script, so no real Run is ever answered.

    python fake_cli.py <command> [arguments...]        # FAKE_CLI names its folder

The folder holds `observe.json` (what `observe` prints now), `steps.json` (the writing commands still to come, in
order), `calls.jsonl` (every argv it was given, one JSON list per line) and `pids` (each started wrapper's pid). A step is
`{"command", "append", "snapshot", "exit", "stdout", "stderr", "until"}`: it appends its Journey `append` events, makes
`snapshot` what `observe` prints next, prints `stdout`/`stderr` (one JSON line each) and exits with `exit`. With
`until: "release"` it first waits for a `release` file in the folder, or for SIGINT: then it does the same with the
CLI's interrupted line and exit 130, as `bio_neb.cli` does (#327).

Like `uv run`, the process the GUI starts is a wrapper that ignores SIGINT and waits for a child doing the work, so
only a signal sent to the whole process group reaches the command.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(os.environ["FAKE_CLI"])


def wrapper() -> int:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with HERE.joinpath("pids").open("a", encoding="utf-8") as pids:  # the tests reap what the app started
        pids.write(f"{os.getpid()}\n")
    return subprocess.run([sys.executable, __file__, *sys.argv[1:]], env={**os.environ, "FAKE_CLI_CHILD": "1"}).returncode


def journey_file(argv: list[str]) -> Path:
    return Path(argv[argv.index("--journey-file") + 1])


def play(argv: list[str]) -> int:
    steps = json.loads(HERE.joinpath("steps.json").read_text(encoding="utf-8"))
    if not steps or steps[0]["command"] != argv[0]:
        print(json.dumps({"status": "failed", "code": "fake_cli_unexpected_command", "fields": [argv[0]]}), file=sys.stderr)
        return 99
    step, rest = steps[0], steps[1:]
    HERE.joinpath("steps.json").write_text(json.dumps(rest), encoding="utf-8")
    interrupted = False
    if step.get("until") == "release":
        try:
            while not HERE.joinpath("release").exists():
                time.sleep(0.05)
        except KeyboardInterrupt:
            interrupted = True
    with journey_file(argv).open("a", encoding="utf-8") as journey:
        journey.write("".join(json.dumps(e) + "\n" for e in step.get("append", [])))
    if step.get("snapshot") is not None:
        HERE.joinpath("observe.json").write_text(json.dumps(step["snapshot"]), encoding="utf-8")
    if interrupted:
        print(json.dumps({"status": "failed", "code": "interrupted", "fields": [argv[0]]}), file=sys.stderr)
        return 130
    if step.get("stdout") is not None:
        print(json.dumps(step["stdout"]))
    if step.get("stderr") is not None:
        print(json.dumps(step["stderr"]), file=sys.stderr)
    return step.get("exit", 0)


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)  # the wrapper's SIG_IGN is inherited
    with HERE.joinpath("calls.jsonl").open("a", encoding="utf-8") as calls:
        calls.write(json.dumps(argv) + "\n")
    if argv[0] == "observe":
        print(HERE.joinpath("observe.json").read_text(encoding="utf-8"))
        return 0
    return play(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if os.environ.get("FAKE_CLI_CHILD") else wrapper())
