"""Shared polling mechanism; callers supply semantic closure and live display.

The window bounds recent samples in memory. A job source independently chooses
its provider log tail bound and retains marker knowledge in its closure. Healthy
samples never become Journey facts; only the caller's verified closure does.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from bio_neb.contracts import FrozenContract
from bio_neb.failure import ProviderError

Sample = TypeVar("Sample")
Result = TypeVar("Result")


class LiveEmitter(Generic[Sample]):
    """A disposable observer cannot stop execution; its errors stay diagnostic."""

    def __init__(self, sink: Callable[[Sample], None] | None):
        self.sink = sink
        self.errors: list[str] = []

    def __call__(self, value: Sample) -> None:
        if self.sink is not None:
            try:
                self.sink(value)
            except Exception as error:
                # This is the sole non-provider isolation boundary: display code
                # has no authority over provider execution or path evidence.
                self.errors.append(type(error).__name__)
                self.sink = None


@dataclass(frozen=True)
class ObservationPolicy(Generic[Sample, Result]):
    interval: float
    max_wait: float
    window: int
    closure: Callable[[tuple[Sample, ...]], Result | None]
    liveness: Callable[[Sample], None] | None = None
    clock: Callable[[], float] = time.monotonic
    pause: Callable[[float], None] = time.sleep

    def __post_init__(self):
        if (
            not math.isfinite(self.interval)
            or not math.isfinite(self.max_wait)
            or self.interval <= 0
            or self.max_wait < 0
            or self.window < 1
        ):
            raise ValueError(
                "observation requires positive interval/window and bounded wait"
            )


def observe(
    source: Callable[[], Sample], policy: ObservationPolicy[Sample, Result]
) -> Result:
    """Sample until closure, or raise exhausted without cancelling the source.

    Closure returns None while incomplete. ProviderError and programming faults
    propagate to the caller: the stage engine owns retry/escalation, not polling.
    There is one immediate sample even for a zero-wait policy.
    """
    recent: deque[Sample] = deque(maxlen=policy.window)
    emit = LiveEmitter(policy.liveness)
    deadline = policy.clock() + policy.max_wait
    while True:
        sample = source()
        recent.append(sample)
        emit(sample)
        result = policy.closure(tuple(recent))
        if result is not None:
            return result
        remaining = deadline - policy.clock()
        if remaining <= 0:
            raise ProviderError(
                "exhausted", "observation_timeout", {"max_wait": policy.max_wait}
            )
        policy.pause(min(policy.interval, remaining))


@contextmanager
def live_heartbeat(sink, interval, *, kind):
    """Observe synchronous pending work locally, without invoking its effects."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("heartbeat requires a finite positive interval")
    live = LiveEmitter(sink)
    stop = threading.Event()

    def pulse():
        while not stop.wait(interval):
            live({"kind": f"{kind}_heartbeat"})

    thread = threading.Thread(target=pulse, daemon=True, name="operation-liveness")
    live({"kind": f"{kind}_started"})
    thread.start()
    try:
        yield live
    finally:
        stop.set()
        thread.join()
        live({"kind": f"{kind}_finished"})


class AdapterHeartbeat(FrozenContract):
    kind: Literal["heartbeat"] = "heartbeat"
