"""Count and time named Run prerequisite builds when a build log is named.

Measurement only: with `BIO_NEB_PREREQUISITE_BUILD_LOG` unset every measured
builder runs unchanged. With it set to an absolute file path, each call appends
one JSON line naming the state, its variant, the seconds it took, the builder
it ran inside (nested builds such as the review-pending build inside an
approved one are kept, marked by `depth`), and the test module and test that
asked. Lines are single appends, so concurrent module processes may share the
file.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import threading
import time
import unittest
from collections.abc import Callable
from typing import Any, TypeVar

BUILD_LOG = "BIO_NEB_PREREQUISITE_BUILD_LOG"

_Builder = TypeVar("_Builder", bound=Callable[..., Any])
_active = threading.local()
_PLAIN = (str, int, float, bool, bytes, tuple, frozenset)


def _variant(arguments: dict[str, Any]) -> str:
    """The non-default keyword choices, plain values verbatim, others by type."""
    parts = []
    for name, value in sorted(arguments.items()):
        if value is None:
            continue
        shown = repr(value) if isinstance(value, _PLAIN) else type(value).__name__
        parts.append(f"{name}={shown[:80]}")
    return ",".join(parts)


def _asker() -> tuple[str, str | None]:
    """The test module this process runs and the test case on the stack."""
    module = (
        sys.argv[1] if sys.argv and sys.argv[0] == "-c" and len(sys.argv) > 1 else None
    )
    test_id = None
    for frame in inspect.stack(context=0):
        owner = frame.frame.f_locals.get("self")
        if isinstance(owner, unittest.TestCase):
            test_id = owner.id()
            break
        owner = frame.frame.f_locals.get("cls")
        if isinstance(owner, type) and issubclass(owner, unittest.TestCase):
            test_id = f"{owner.__module__}.{owner.__qualname__}"
            break
    if module is None and test_id is not None:
        module = test_id.split(".")[0] + "." + test_id.split(".")[1]
    return module or "unknown", test_id


def _append(path: str, record: dict[str, Any]) -> None:
    line = (json.dumps(record, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def measured_build(state: str) -> Callable[[_Builder], _Builder]:
    """Name a prerequisite builder's state for the build log."""

    def decorate(builder: _Builder) -> _Builder:
        @functools.wraps(builder)
        def measured(*args: Any, **kwargs: Any) -> Any:
            path = os.environ.get(BUILD_LOG)
            if not path:
                return builder(*args, **kwargs)
            stack = _active.__dict__.setdefault("stack", [])
            parent = stack[-1] if stack else None
            stack.append(state)
            started = time.perf_counter()
            succeeded = False
            try:
                result = builder(*args, **kwargs)
                succeeded = True
                return result
            finally:
                seconds = time.perf_counter() - started
                stack.pop()
                module, test_id = _asker()
                _append(
                    path,
                    {
                        "state": state,
                        "variant": _variant(kwargs),
                        "seconds": round(seconds, 4),
                        "depth": len(stack),
                        "inside": parent,
                        "succeeded": succeeded,
                        "module": module,
                        "test": test_id,
                        "pid": os.getpid(),
                    },
                )

        return measured  # type: ignore[return-value]

    return decorate


__all__ = ["BUILD_LOG", "measured_build"]
