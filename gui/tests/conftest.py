from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from nicegui.testing import User
from nicegui.testing.user_simulation import user_simulation
from runs import JOURNEY_ID, feed

import cli_bridge

APP = Path(__file__).resolve().parent.parent / "app.py"
FAKE_CLI = Path(__file__).resolve().with_name("fake_cli.py")


@pytest.fixture
def run_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Write a synthetic Run and its snapshots, start app.py on them, and yield a NiceGUI `User`.

    app.py reads RUN_DIR, SNAPSHOTS and LIVE when it starts, so each run executes the main file afresh. The backend's
    `journey-feed` answers through `journey_feed(journey_file)`: by default the synthetic feed of the file's events.
    An ERROR log (a raising page or handler) fails the test, as NiceGUI's own `user` fixture does.
    """

    @asynccontextmanager
    async def start(
        events: list[dict],
        snapshots: dict[int, dict],
        journey_feed=None,
        *,
        live: bool = False,
        artifacts: tuple[dict | None, dict | None] | None = None,
    ):
        run = tmp_path / "runs" / JOURNEY_ID
        run.mkdir(parents=True)
        run.joinpath("journey.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        snaps = tmp_path / "snapshots"
        snaps.mkdir()
        for line, snap in snapshots.items():
            snaps.joinpath(f"{line}.json").write_text(json.dumps(snap), encoding="utf-8")
        if artifacts:
            document, refusal = artifacts
            name = "artifacts.json" if document is not None else "artifacts.refused.json"
            snaps.joinpath(name).write_text(json.dumps(document if document is not None else refusal), encoding="utf-8")
        monkeypatch.setenv("RUN_DIR", str(run))
        monkeypatch.setenv("SNAPSHOTS", str(snaps))
        monkeypatch.setenv("ANSWER_FILES", str(tmp_path / "answers"))
        if live:
            monkeypatch.setenv("LIVE", "1")
        else:
            monkeypatch.delenv("LIVE", raising=False)
        monkeypatch.setattr(cli_bridge, "journey_feed", journey_feed or (lambda journey_file: (feed(read(journey_file)), None)))
        with caplog.at_level(logging.ERROR):
            async with user_simulation(main_file=APP) as user:
                yield user
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        if errors:
            pytest.fail("The page logged errors: " + "; ".join(r.getMessage() for r in errors), pytrace=False)

    return start


def read(journey_file: Path) -> list[dict]:
    return [json.loads(line) for line in journey_file.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def replay(run_app):
    """`run_app` in recorded replay; `journey_feed`, when given, is the feed's one answer (a document, or None and a
    refusal code), and `artifacts` what `make_snapshots.py` recorded from `run-artifacts` (a document, or None and its
    refusal)."""

    def start(events: list[dict], snapshots: dict[int, dict], journey_feed: tuple[dict | None, str | None] | None = None,
              artifacts: tuple[dict | None, dict | None] | None = None):  # fmt: skip
        return run_app(events, snapshots, journey_feed and (lambda journey_file: journey_feed), artifacts=artifacts)

    return start


@pytest.fixture
def follow(run_app, monkeypatch: pytest.MonkeyPatch):
    """`run_app` following the Run live (LIVE=1): the page tails `journey.jsonl` as a test appends to it (`append`).
    `observe` refuses here; these tests are about the Journey drawer."""

    async def observe(journey_file: Path) -> tuple[dict | None, str | None]:
        return None, "not_observed_in_this_test"

    monkeypatch.setattr(cli_bridge, "observe", observe)

    def start(events: list[dict], journey_feed=None):
        return run_app(events, {}, journey_feed, live=True)

    return start


class FakeCli:
    """The fake CLI's folder (`tests/fake_cli.py`): what `observe` prints, the writing commands to play, and the argv
    of every call the app made."""

    def __init__(self, folder: Path, snapshot: dict, steps: list[dict]) -> None:
        self.folder = folder
        folder.mkdir()
        folder.joinpath("observe.json").write_text(json.dumps(snapshot), encoding="utf-8")
        folder.joinpath("steps.json").write_text(json.dumps(steps), encoding="utf-8")

    def all_calls(self) -> list[list[str]]:
        """Every argv the app gave the CLI, in order."""
        path = self.folder / "calls.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def calls(self) -> list[list[str]]:
        """Every argv but `observe`'s (the app observes at each pass end), in order."""
        return [argv for argv in self.all_calls() if argv[0] != "observe"]

    def release(self) -> None:
        """Let a step that waits (`until: "release"`) finish."""
        self.folder.joinpath("release").touch()

    async def reap(self) -> None:
        """Release any waiting step and wait for every command the app started to end (and be reaped): SIGKILL to its
        process group after 5 s. A command the app's own wait already reaped is gone."""
        self.release()
        path = self.folder / "pids"
        for pid in map(int, path.read_text(encoding="utf-8").split() if path.exists() else []):
            for _ in range(50):
                try:
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        break
                except ChildProcessError:
                    break
                await asyncio.sleep(0.1)
            else:
                os.killpg(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

    async def started(self, count: int = 1) -> None:
        for _ in range(50):
            if len(self.calls()) >= count:
                return
            await asyncio.sleep(0.1)
        pytest.fail(f"the app started {len(self.calls())} commands, not {count}")


@pytest.fixture
def answering(run_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """`run_app` following the Run live with the fake CLI in place of the checkout's: `observe` prints `snapshot` until
    a step changes it, and `answer`/`resume` play `steps`. Yields the `User` and the `FakeCli`. With
    FAKE_CLI_RECORDS set, each test's recorded argv (`calls.jsonl`) is copied there as `<test name>.jsonl`."""

    @asynccontextmanager
    async def start(events: list[dict], snapshot: dict, steps: list[dict]):
        fake = FakeCli(tmp_path / "fake-cli", snapshot, steps)
        monkeypatch.setenv("FAKE_CLI", str(fake.folder))
        monkeypatch.setenv("COMMAND_LOGS", str(tmp_path / "commands"))
        monkeypatch.setattr(cli_bridge, "CLI", [sys.executable, str(FAKE_CLI)])
        async with run_app(events, {}, live=True) as user:
            try:
                yield user, fake
            finally:
                await fake.reap()  # also when the test fails: no command outlives it
        if records := os.environ.get("FAKE_CLI_RECORDS"):
            Path(records).mkdir(parents=True, exist_ok=True)
            Path(records, f"{request.node.name}.jsonl").write_text("".join(json.dumps(a) + "\n" for a in fake.all_calls()))

    return start


def append(events: list[dict]) -> None:
    """Append events to the Journey the running app follows."""
    with Path(os.environ["RUN_DIR"], "journey.jsonl").open("a", encoding="utf-8") as journey:
        journey.write("".join(json.dumps(e) + "\n" for e in events))


async def wait_to_see(user: User, text: str) -> None:
    """Live mode reads the feed in the background (a poll every 0.5 s, then the feed call), so wait up to 5 s for any
    element showing `text`."""
    for _ in range(50):
        try:
            user.find(content=text)
            return
        except AssertionError:
            await asyncio.sleep(0.1)
    await user.should_see(text)


__all__ = ["User", "answering", "append", "follow", "replay", "wait_to_see"]
