"""The host test queue: one set of test slots shared by every checkout and lane (#391).

Before the testing runner starts any child unit it takes slots here, one per test process
it may run at once. The queue is `CAPACITY` lock files in one host directory
(`bio_neb.locations.test_queue`), so every worktree on the host counts against the same
slots. A slot is held by an exclusive `flock` on its file: the kernel releases it when
every process sharing the open file has ended, so a killed holder's slots return to the
queue with no cleanup, and a module child that inherits a slot keeps it while it runs.

A request waits for all of its slots rather than starting smaller (owner, 2026-10-19:
batch concurrency stays as it is). One request at a time collects slots, behind the
queue's gate, so two partial holders never wait on each other. A request may name a wider
`sole` count, which it takes instead when it finds no slot held, so a batch that is the
host's sole runner runs wider (#455). A runner re-executing itself (a snapshot's own
runner, the checkpoint's suite batch) inherits `HELD` and takes nothing a second time.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

# Sixteen slots, so several default batches of 4 and reviewer checks run side by side; a
# batch's own default width stays 4.
CAPACITY = 16
# A checkpoint takes eight slots, not the whole host, so the rest stay open for focused
# and reviewer checks while it runs.
CHECKPOINT_WIDTH = 8
# The slot count a runner's re-execution inherits from the controller that holds them.
HELD = "BIO_NEB_TEST_QUEUE_HELD"
_POLL_SECONDS = 0.2
_HEARTBEAT_SECONDS = 30.0


@dataclass(frozen=True)
class Held:
    """The slots one request holds and how long it waited for them.

    An `inherited` request holds its controller's slots: it took none and waited for
    none itself, so its wait is the controller's to record.
    """

    count: int
    fds: tuple[int, ...]
    waited_seconds: float
    inherited: bool = False


def _try_lock(stream: TextIO) -> bool:
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


class QueueRefused(RuntimeError):
    """A request the queue cannot ever satisfy."""


@contextmanager
def slots(
    count: int,
    *,
    queue: Path,
    holder: str,
    sole: int | None = None,
    stream: TextIO | None = None,
    heartbeat: float = _HEARTBEAT_SECONDS,
) -> Iterator[Held]:
    """Hold `count` of the host's slots for the duration of the block, waiting for them.

    With `sole`, a request that holds the gate while no slot is held takes `sole` slots
    instead; the choice is made once, and `Held.count` names it. A re-executed runner
    takes the width its controller holds, up to `sole`. While it waits, a `test_queue` heartbeat line goes to `stream` (stderr by default)
    every `heartbeat` seconds, naming the processes that hold slots. Inside the block
    `HELD` names the count to the runners this process re-executes.
    """
    for asked in (count, sole or count):
        if not 1 <= asked <= CAPACITY:
            raise QueueRefused(
                f"a request takes from 1 to {CAPACITY} slots, not {asked}"
            )
    inherited = os.environ.get(HELD)
    if inherited is not None:
        if count > int(inherited):
            raise QueueRefused(
                f"this runner inherits {inherited} slot(s) from its controller and "
                f"asks for {count}"
            )
        width = min(max(count, sole or count), int(inherited))
        yield Held(width, (), 0.0, inherited=True)
        return
    waiter = _Waiter(count, queue, stream or sys.stderr, heartbeat, sole)
    with _taken(waiter, holder) as held:
        os.environ[HELD] = str(held.count)
        try:
            yield held
        finally:
            del os.environ[HELD]


class _Waiter:
    """Polls for the gate and then the slots, printing a heartbeat while it waits."""

    def __init__(
        self,
        count: int,
        queue: Path,
        stream: TextIO,
        every: float,
        sole: int | None = None,
    ) -> None:
        self.count, self.queue, self.stream, self.every = count, queue, stream, every
        self.sole = sole
        self.started = time.monotonic()
        self.next_beat: float | None = None

    def waited(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def wait(self, held: dict[int, TextIO]) -> None:
        now = time.monotonic()
        if self.next_beat is None or now >= self.next_beat:
            self.next_beat = now + self.every
            beat = {
                "waiting_for": self.count,
                "held": len(held),
                "waited_seconds": self.waited(),
                "holders": _holders(self.queue, held),
            }
            print(json.dumps({"test_queue": beat}), file=self.stream, flush=True)
        time.sleep(_POLL_SECONDS)


def _holders(queue: Path, held: dict[int, TextIO]) -> list[dict[str, object]]:
    """The other processes holding slots now, from the records they wrote."""
    counts: dict[tuple[object, object], int] = {}
    for index in range(CAPACITY):
        if index in held:
            continue
        with (queue / f"slot-{index}.lock").open("a+") as slot:
            if _try_lock(slot):
                continue
            slot.seek(0)
            try:
                record = json.loads(slot.read())
            except json.JSONDecodeError:
                record = {}  # its holder is still collecting slots
        key = (record.get("pid"), record.get("holder"))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"pid": pid, "holder": holder, "slots": taken}
        for (pid, holder), taken in counts.items()
    ]


@contextmanager
def _taken(waiter: _Waiter, holder: str) -> Iterator[Held]:
    waiter.queue.mkdir(parents=True, exist_ok=True)
    held: dict[int, TextIO] = {}
    with ExitStack() as stack:
        gate = stack.enter_context((waiter.queue / "gate.lock").open("a"))
        while not _try_lock(gate):
            waiter.wait(held)
        # Holding the gate, no other request is collecting: a held slot is a runner's.
        if waiter.sole is not None and not _holders(waiter.queue, held):
            waiter.count = waiter.sole
        while len(held) < waiter.count:
            for index in range(CAPACITY):
                if len(held) == waiter.count:
                    break
                if index in held:
                    continue
                slot = (waiter.queue / f"slot-{index}.lock").open("a+")
                if _try_lock(slot):
                    held[index] = stack.enter_context(slot)
                else:
                    slot.close()
            if len(held) < waiter.count:
                waiter.wait(held)
        record = json.dumps(
            {"pid": os.getpid(), "holder": holder, "slots": waiter.count}
        )
        for slot in held.values():
            slot.truncate(0)
            slot.write(record)
            slot.flush()
        fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
        try:
            yield Held(
                waiter.count,
                tuple(slot.fileno() for slot in held.values()),
                waiter.waited(),
            )
        finally:
            for slot in held.values():
                slot.truncate(0)
