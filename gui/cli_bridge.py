"""The only way the GUI talks to BioNeb: the CLI, run in a BioNeb checkout.

The checkout defaults to the repository this folder sits in; BIONEB_CHECKOUT overrides it. The CLI
runs in the checkout's own uv environment. Nothing here imports BioNeb. The commands that contact a model or the
cloud (`begin`, `answer`, `resume`) take the credentials file BIONEB_ENV_FILE names through `uv run --env-file`, as
the live operator runs them; the GUI never reads its values. `observe` and the other reads run without it.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path

CLI = ["uv", "run", "--locked", "--offline", "python", "-m", "bio_neb.cli"]


def checkout() -> Path:
    value = os.environ.get("BIONEB_CHECKOUT")
    return Path(value) if value else Path(__file__).resolve().parent.parent


def credentials() -> Path | None:
    value = os.environ.get("BIONEB_ENV_FILE")
    return Path(value) if value else None


def credentialed(argv: list[str]) -> list[str]:
    """`argv` with the credentials file passed to `uv run` (the flag belongs to `run`, not to `uv`)."""
    env = credentials()
    return [*argv[:2], "--env-file", str(env), *argv[2:]] if env and argv[:2] == ["uv", "run"] else argv


def begin_argv(run_root: Path, prompt_file: Path, fastqs: list[str], *, paper_files: list[Path] = (),
               request_files: list[Path] = (), runtime_config: Path | None = None) -> list[str]:  # fmt: skip
    """Start a Run without `--interactive`: `begin` runs the first pass and stops at the first Review, which the GUI
    then answers. Each FASTQ is `ROLE=<absolute path or s3://bucket/key>`."""
    argv = [*CLI, "begin", "--run-root", str(run_root), "--prompt-file", str(prompt_file)]
    argv += [a for f in fastqs for a in ("--fastq", f)]
    argv += [a for f in paper_files for a in ("--paper-file", str(f))]
    argv += [a for f in request_files for a in ("--request-file", str(f))]
    argv += ["--runtime-config", str(runtime_config)] if runtime_config else []
    return credentialed(argv)


def observe_argv(journey_file: Path) -> list[str]:
    return [*CLI, "observe", "--journey-file", str(journey_file)]


def journey_feed_argv(journey_file: Path) -> list[str]:
    return [*CLI, "journey-feed", "--journey-file", str(journey_file), "--all"]


def run_artifacts_argv(journey_file: Path) -> list[str]:
    return [*CLI, "run-artifacts", "--journey-file", str(journey_file)]


def answer_argv(journey_file: Path, request_event_id: str, response: dict) -> list[str]:
    return credentialed([
        *CLI, "answer", "--journey-file", str(journey_file),
        "--request-event-id", request_event_id,
        "--response-json", json.dumps(response, separators=(",", ":")),
    ])  # fmt: skip


def resume_argv(journey_file: Path) -> list[str]:
    return credentialed([*CLI, "resume", "--journey-file", str(journey_file)])


def trim(snapshot: dict) -> dict:
    """Drop per-pass contact lists (megabytes of model contact detail the UI never shows)."""
    for item in snapshot.get("passes", []):
        contacts = item.pop("contacts", None)
        item["contact_count"] = len(contacts or [])
    return snapshot


def _document(returncode: int | None, out: bytes, err: bytes) -> tuple[dict | None, dict | None]:
    """A command's JSON document, or its refusal: the CLI's own (`code` and the `fields` it names, one JSON line on
    stderr after any tool warnings, or a failed document), or the bridge's in the same shape when the output names
    none."""
    try:
        data = json.loads(out)
    except ValueError:
        refusals = [line for line in err.decode(errors="replace").splitlines() if line.startswith("{")]
        try:
            return None, json.loads(refusals[-1])
        except (IndexError, ValueError):
            return None, {"code": "unreadable_output", "fields": []}
    if returncode != 0 or data.get("status") == "failed":
        return None, data if "code" in data else {"code": f"exit_{returncode}", "fields": []}
    return data, None


async def _run(argv: list[str]) -> tuple[dict | None, dict | None]:
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=checkout(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await process.communicate()
    return _document(process.returncode, out, err)


async def observe(journey_file: Path) -> tuple[dict | None, str | None]:
    """Returns (snapshot, None) or (None, refusal code). A refusal is normal mid-pass."""
    data, refusal = await _run(observe_argv(journey_file))
    return (trim(data), None) if data is not None else (None, refusal["code"])


async def run_artifacts(journey_file: Path) -> tuple[dict | None, dict | None]:
    """The finished Run's verified files (`bio-neb.run-artifacts/1`), or (None, the refusal: its `code` and `fields`).
    A copied Run is refused: its recorded Journey path no longer matches."""
    return await _run(run_artifacts_argv(journey_file))


def journey_feed(journey_file: Path) -> tuple[dict | None, str | None]:
    """The backend's `bio-neb.journey-feed/1` document with every event (`--all`), or (None, refusal code).

    Blocking: it re-reads the whole file each call (a half-written last line is skipped), so live mode runs it
    through `run.io_bound`.
    """
    process = subprocess.run(journey_feed_argv(journey_file), cwd=checkout(), capture_output=True, check=False)
    data, refusal = _document(process.returncode, process.stdout, process.stderr)
    return data, refusal and refusal["code"]


def start_detached(argv: list[str], log: Path) -> subprocess.Popen:
    """Start a backend command that outlives the UI, in its own session (so its process group is its own), its stdout
    and stderr in `<log>.out` and `<log>.err`. The next observe is the truth, not the exit code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.with_suffix(".out").open("ab") as out, log.with_suffix(".err").open("ab") as err:
        # No terminal input: a command that would prompt (begin without inputs) fails instead of waiting on the shell.
        return subprocess.Popen(argv, cwd=checkout(), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)


def interrupt(process: subprocess.Popen) -> None:
    """SIGINT to the command's whole process group, as Ctrl-C would: `uv run` and the CLI under it. The CLI then holds
    a Run it interrupted mid-work (#327)."""
    os.killpg(process.pid, signal.SIGINT)


def refusal(returncode: int, log: Path) -> dict | None:
    """What a finished command said when it did not succeed: the CLI's refusal (`code` and `fields`, `interrupted` on
    SIGINT), or the bridge's in the same shape. None on success."""
    if returncode == 0:
        return None
    return _document(returncode, log.with_suffix(".out").read_bytes(), log.with_suffix(".err").read_bytes())[1]
