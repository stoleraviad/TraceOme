"""The review flag: how many tickets may be in their review phase at once, host-wide (#391).

A ticket worker claims the flag before it asks for its reviewers and keeps it through the
whole review loop, the live Run and the rounds over its findings included, then releases it
when it declares the review final (owner, 2026-10-19: "we dont want to have paraales tickt
going tru the review steps"). How the reviews themselves run is unchanged.

A review phase outlives any one process, so a claim is a record, not a lock: the holders
file in the host test queue's directory names each holding ticket, its map, worker and
orchestrator, so a worker that finds the flag taken can ping them. Only the orchestrator or
the owner releases a stale claim, with `--stale <reason>`. Every claim and release is
appended to the flag history. `CAPACITY` tickets may hold it at once (owner, 2026-10-21:
eight; two since 2026-10-20, one before).
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAPACITY = 8
TAKEN_EXIT = 3
_HOLDERS = "review-flag.json"
_HISTORY = "review-flag-history.jsonl"


@contextmanager
def _holders(queue: Path) -> Iterator[list[dict[str, Any]]]:
    """The holders, read and written back under the flag's own lock."""
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "review-flag.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = queue / _HOLDERS
        holders = json.loads(path.read_text()) if path.exists() else []
        before = list(holders)
        yield holders
        if holders != before:
            written = tempfile.NamedTemporaryFile(
                "w", dir=queue, prefix=".review-flag-", delete=False
            )
            with written:
                json.dump(holders, written)
            os.replace(written.name, path)


def _history(queue: Path, event: str, holder: dict[str, Any], **extra: Any) -> None:
    entry = {"event": event, **holder, "at": datetime.now(UTC).isoformat(), **extra}
    with (queue / _HISTORY).open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def _result(result: str, holders: list[dict[str, Any]]) -> dict[str, Any]:
    return {"result": result, "holders": list(holders)}


def claim(
    queue: Path, *, ticket: int, map_id: int, worker: str, orchestrator: str
) -> tuple[int, dict[str, Any]]:
    """Claim a flag for `ticket`; a ticket that already holds one keeps it."""
    with _holders(queue) as holders:
        if any(holder["ticket"] == ticket for holder in holders):
            return 0, _result("held", holders)
        if len(holders) >= CAPACITY:
            return TAKEN_EXIT, _result("taken", holders)
        holder = {
            "ticket": ticket,
            "map": map_id,
            "worker": worker,
            "orchestrator": orchestrator,
            "claimed_at": datetime.now(UTC).isoformat(),
        }
        holders.append(holder)
        _history(queue, "claimed", holder)
        return 0, _result("claimed", holders)


def release(
    queue: Path, *, ticket: int, stale: str | None = None
) -> tuple[int, dict[str, Any]]:
    """Release `ticket`'s flag; a stale release keeps its reason in the history."""
    with _holders(queue) as holders:
        held = [holder for holder in holders if holder["ticket"] == ticket]
        if not held:
            return 2, _result("not_held", holders)
        holders.remove(held[0])
        _history(queue, "released", held[0], **({"stale": stale} if stale else {}))
        return 0, _result("released", holders)


def show(queue: Path) -> tuple[int, dict[str, Any]]:
    with _holders(queue) as holders:
        return 0, _result("taken" if len(holders) >= CAPACITY else "free", holders)
