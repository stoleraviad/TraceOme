"""The BioNeb command the app runs for its Run (`answer` or `resume`), and the bar that shows it on a page.

A command starts detached (`cli_bridge.start_detached`), so closing the GUI never stops it. It belongs to the app,
not to one page: every open page disables its decision buttons while it runs, and each page, seeing it end, asks
`observe` for the Run's next state. Its exit only tells whether the CLI refused (its code and fields) or was
interrupted; the next `observe` is the truth.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from nicegui import background_tasks, ui

import cli_bridge


class Command:
    """The one command the app has running, if any (`idle` is what decision buttons bind to), and how the last one
    ended: `ended` counts the commands that have finished, `outcome` is the last one's refusal or None."""

    def __init__(self, logs: Path) -> None:
        self.logs = logs  # each command's stdout and stderr, outside the repository
        self.process: subprocess.Popen | None = None
        self.name = ""
        self.started_at: datetime | None = None
        self.ended = 0
        self.outcome: dict | None = None

    @property
    def idle(self) -> bool:
        return self.process is None

    def start(self, argv: list[str], name: str) -> None:
        if not self.idle:
            raise RuntimeError(f"{self.name} is still running")
        self.name, self.started_at = name, datetime.now(timezone.utc)
        log = self.logs / f"{self.started_at:%Y%m%dT%H%M%S%fZ}-{name.split()[0]}"
        self.process = cli_bridge.start_detached(argv, log)
        background_tasks.create(self.wait(log), name=f"wait for {name}")

    async def wait(self, log: Path) -> None:
        while (code := self.process.poll()) is None:
            await asyncio.sleep(0.1)
        self.outcome = cli_bridge.refusal(code, log)
        self.process = None
        self.ended += 1

    def interrupt(self) -> None:
        if self.process is not None:
            cli_bridge.interrupt(self.process)


class CommandBar:
    """A page's line above the centre: the running command with Interrupt, how the last one ended, and Resume while
    `observe` says the Run is held. Rebuilt only when one of those changes."""

    def __init__(self, command: Command, resume: Callable[[], None]) -> None:
        self.command = command
        self.resume = resume
        self.key: tuple | None = None
        self.box = ui.column().classes("w-full gap-1").mark("command-bar")

    def update(self, snapshot: dict | None) -> None:
        held = snapshot is not None and snapshot["frontier"] == "held"
        key = (self.command.idle, self.command.ended, held)
        if key == self.key:
            return
        self.key = key
        self.box.clear()
        with self.box:
            if not self.command.idle:
                with ui.row().classes("w-full items-center gap-3 bg-blue-1 rounded p-2"):
                    ui.spinner(size="sm")
                    ui.label(f"Running: {self.command.name} · started {self.command.started_at:%H:%M:%S} UTC").classes("text-sm")
                    ui.space()
                    ui.button("Interrupt", icon="stop_circle", on_click=self.command.interrupt).props(
                        "outline no-caps color=negative"
                    )
            elif self.command.ended and (outcome := self.command.outcome):
                fields = ", ".join(outcome["fields"])
                with ui.row().classes("w-full items-center gap-2 bg-red-1 rounded p-2").mark("command-refused"):
                    ui.icon("block", color="negative")
                    ui.label(f"{self.command.name}: {outcome['code']}" + (f" ({fields})" if fields else "")).classes("text-sm")
            if held and self.command.idle:
                failure = snapshot.get("failure_code")
                with ui.row().classes("w-full items-center gap-2 bg-amber-1 rounded p-2").mark("run-held"):
                    ui.icon("pause_circle", color="amber-9")
                    ui.label("The Run is held" + (f": {failure}" if failure else "") + ".").classes("text-sm")
                    ui.space()
                    ui.button("Resume", icon="play_arrow", on_click=self.start_resume).props("no-caps unelevated")

    def start_resume(self) -> None:
        if not self.command.idle:  # another page started a command since this bar was drawn
            ui.notify(f"Wait for {self.command.name} to end.", type="warning")
            return
        self.resume()
