"""Test reads and tampers of Execution contacts and Execution path facts.

An Execution Pass records its contacts as `domain` records of the coordinator
contact journal (`runs/<id>/contacts/<pass>/journal.jsonl`; an outcome's
`contact_journal` names the retained `execution-prefix-<sha256>.jsonl` snapshot
beside it). Each record's `domain.payload` is one Execution contact row
(`row_kind`, `stage`, `step`, `phase`, `receipt`, ...). The Execution path facts
are `ExecutionPathRecorded` Journey events whose payload carries the fact's own
`event_type` and `payload`.

These helpers only unwrap and rewrite those envelopes; every tamper is written
back as canonical JSON so the reader under test reaches its own check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from bio_neb.content import canonical_json

_PATH_ENVELOPE = "ExecutionPathRecorded"


def _lines(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write(path: str | Path, rows: list[dict]) -> bytes:
    content = "".join(canonical_json(row) + "\n" for row in rows).encode()
    Path(path).write_bytes(content)
    return content


def _is_contact(record: dict) -> bool:
    # The Pass's `started` record opens the operation and carries no contact row.
    domain = record["domain"]
    return domain is not None and domain["phase"] != "started"


def execution_contact_rows(journal: str | Path) -> list[dict]:
    """The Execution contact rows of a coordinator contact journal, in order."""
    return [
        record["domain"]["payload"] for record in _lines(journal) if _is_contact(record)
    ]


def execution_contact_row(journal: str | Path, proof_ref: str) -> dict:
    """The contact row a `cj:<pass>:<sequence>` proof reference names."""
    _, execution_pass, sequence = proof_ref.split(":")
    (row,) = [
        record["domain"]["payload"]
        for record in _lines(journal)
        if _is_contact(record)
        and record["domain"]["iteration"] == int(execution_pass)
        and record["domain"]["payload"]["sequence"] == int(sequence)
    ]
    return row


def rewrite_execution_contact_rows(
    journal: str | Path, change: Callable[[dict], None]
) -> bytes:
    """Apply change to every contact row in place and rewrite the journal."""
    records = _lines(journal)
    for record in records:
        if _is_contact(record):
            change(record["domain"]["payload"])
    return _write(journal, records)


def rewrite_journey(
    journey_file: str | Path, change: Callable[[list[dict]], None]
) -> bytes:
    """Apply change to the Journey's events in place and rewrite it canonically.

    Path facts are reached inside their `ExecutionPathRecorded` envelopes, for
    example through `execution_path_fact`.
    """
    events = _lines(journey_file)
    change(events)
    return _write(journey_file, events)


def execution_path_fact(events: list[dict], event_type: str) -> dict:
    """The last envelope of events whose inner fact has this event type."""
    return next(
        event["payload"]
        for event in reversed(events)
        if event["event_type"] == _PATH_ENVELOPE
        and event["payload"]["event_type"] == event_type
    )


__all__ = [
    "execution_contact_row",
    "execution_contact_rows",
    "execution_path_fact",
    "rewrite_execution_contact_rows",
    "rewrite_journey",
]
