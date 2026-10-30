"""Optional invocation observation; sinks cannot supply product authority."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from bio_neb.observation import LiveEmitter


class InvocationObserver(Protocol):
    def inspection_event(self, name: str, **attributes: Any) -> None: ...

    def inspect_agent(self, agent: Any, **identities: str): ...


_observer: ContextVar[InvocationObserver | None] = ContextVar(
    "invocation_observer", default=None
)


def inspection_event(name: str, **attributes: Any) -> None:
    observer = _observer.get()
    if observer is not None:
        observer.inspection_event(name, **attributes)


@contextmanager
def inspect_agent(agent: Any, **identities: str):
    observer = _observer.get()
    if observer is None:
        yield
    else:
        with observer.inspect_agent(agent, **identities):
            yield


_turn_progress: ContextVar[LiveEmitter | None] = ContextVar(
    "agent_turn_progress", default=None
)


@contextmanager
def agent_turn_progress(sink: Callable[[Any], None]):
    """Bind the live sink for Agent-turn progress, such as each repeated call.

    Progress is ephemeral: it is never retained, and a failing sink is disabled
    while the work continues (ADR 0036).
    """
    token = _turn_progress.set(LiveEmitter(sink))
    try:
        yield
    finally:
        _turn_progress.reset(token)


def turn_progress(value: Any) -> None:
    live = _turn_progress.get()
    if live is not None:
        live(value)
