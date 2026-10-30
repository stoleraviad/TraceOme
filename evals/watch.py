"""Watch one eval capsule's Journey and interrupt it when a run limit trips.

Eval-runner tooling, contact-free: it reads the Journey, the Run's contact journals and the
operator records, and calls only the operator's `interrupt`. Limits (owner, 2026-10-13, until
the product bounds no-progress turns): logical model requests and physical source GETs per run
(every `source.get` start in the contact journals, failed or not), consecutive identical Tool
calls with no new fact between them, and one model request without an answer for too long.
Unreadable contact accounting trips as well (fail closed). Run

    python -m evals.watch --run-root <capsule root> [--journey-file <journey.jsonl>]
        [--max-requests 50] [--max-source-reads 10] [--repeat 5] [--stall-seconds 180]

An `answer` capsule writes into the begin capsule's Journey: pass that file. Counts cover the
whole Journey, so a limit applies per Run, not per capsule.

Each line printed is one JSON event: `trip` (and the interrupt it caused), `review_prompt`
(the interactive pane waits for an answer), `truncated` (a reply was cut off), `turn_fault`
(one entry of the product's fault ledger, `evals.score.turn_faults`), `reply` (one reply's
size against the ceiling its request sent, `large` above half of it; printed again with
`updated` when a later cut-off or routing fact changes it) or `finished`, after one last read
of the Journey. Cut-off
replies, turn faults and reply sizes are reported, never interrupted: the product owns those
stops (ADR 0044).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from bio_neb.agentic_journey.repeated_calls import is_loop_record
from bio_neb.recording import read_contacts
from evals.score import replies, turn_faults

Event = Mapping[str, Any]
_PROMPT = "Enter one exact legal Review response JSON"
_ANSWERED = frozenset({"ModelResponseRecorded", "ModelRequestFailed"})
_CUT_FIELDS = ("request_number", "completion_tokens", "requested_max_tokens")


@dataclass(frozen=True)
class Limits:
    max_requests: int = 50
    max_source_reads: int = 10
    repeat: int = 5
    stall_seconds: float = 180


def _when(event: Event) -> datetime:
    return datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00"))


def _repeat_streak(events: list[Event]) -> int:
    """The latest run of identical Tool calls with no new fact between them."""
    streak, last = 0, None
    for event in events:
        kind = event["event_type"]
        if kind == "ToolCallValidated":
            call = (
                event["payload"]["tool_name"],
                json.dumps(event["payload"]["arguments"], sort_keys=True),
            )
            streak = streak + 1 if call == last else 1
            last = call
        elif not is_loop_record(event):
            # The product's rule: the loop's own records never change what a call finds.
            streak, last = 0, None
    return streak


def cut_replies(events: Iterable[Event]) -> list[dict]:
    """Each cut-off reply (`ModelCompletionTruncated`), reported but never interrupted.

    `requested_max_tokens` is the configured ceiling; `final_max_completion_tokens` the one
    its request actually sent, which the context budget lowers.
    """
    events = list(events)
    sent = {
        (e["payload"].get("attempt_id"), e["payload"].get("request_number")): e[
            "payload"
        ].get("final_max_completion_tokens")
        for e in events
        if e["event_type"] == "ModelTransportRequestRecorded"
    }
    return [
        {
            **{k: e["payload"].get(k) for k in _CUT_FIELDS},
            "final_max_completion_tokens": sent.get(
                (e["payload"].get("attempt_id"), e["payload"].get("request_number"))
            ),
        }
        for e in events
        if e["event_type"] == "ModelCompletionTruncated"
    ]


def reply_rows(events: Iterable[Event]) -> list[dict]:
    """Each reply's size row (`evals.score.replies`), `large` above half its allowance."""
    return [
        {
            **row,
            "large": bool(
                row["allowance"]
                and row["output_tokens"]
                and row["output_tokens"] > row["allowance"] / 2
            ),
        }
        for row in replies(list(events))
    ]


def source_gets(run_dir: Path, events: Iterable[Event]) -> int:
    """Physical source GETs: every `source.get` start in the Run's contact journals.

    Every Pass the Journey records must have its journal; a missing one, or one the product's
    reader refuses, raises `ValueError` (unknown accounting stays unknown, ADR 0033). A row
    still being written (no final newline) is left for the next poll.
    """
    recorded = {
        e["payload"]["pass_id"] for e in events if e["event_type"] == "PassStarted"
    }
    present = {
        path.parent.name for path in (run_dir / "contacts").glob("*/journal.jsonl")
    }
    missing = recorded - present
    if missing:
        raise ValueError(f"contact journal missing for Pass {sorted(missing)}")
    count = 0
    for name in sorted(recorded | present):
        content = (run_dir / "contacts" / name / "journal.jsonl").read_bytes()
        content = content[: content.rfind(b"\n") + 1]
        rows = read_contacts(content, run_id=run_dir.name, pass_id=name)
        count += sum(
            row.kind == "start" and row.operation == "source.get" for row in rows
        )
    return count


def trips(
    events: Iterable[Event], limits: Limits, now: datetime, gets: int | None
) -> list[dict]:
    """Every limit the Run has reached; `gets` is None when its accounting is unreadable."""
    events = list(events)
    found = []
    requests = sum(e["event_type"] == "ModelRequestRecorded" for e in events)
    if requests >= limits.max_requests:
        found.append({"limit": "model_requests", "count": requests})
    if gets is None:
        found.append({"limit": "source_accounting_unreadable"})
    elif gets >= limits.max_source_reads:
        found.append({"limit": "source_gets", "count": gets})
    streak = _repeat_streak(events)
    if streak >= limits.repeat:
        found.append({"limit": "repeated_call", "count": streak})
    sent = [e for e in events if e["event_type"] == "ModelTransportRequestRecorded"]
    if sent and not any(
        e["event_type"] in _ANSWERED for e in events[events.index(sent[-1]) :]
    ):
        waited = (now - _when(sent[-1])).total_seconds()
        if waited >= limits.stall_seconds:
            found.append({"limit": "stalled_request", "seconds": round(waited)})
    return found


def _events(run_root: Path, journey_file: Path | None) -> list[Event]:
    journeys = (
        [journey_file]
        if journey_file
        else sorted((run_root / "runs").glob("*/journey.jsonl"))
    )
    if not journeys or not journeys[0].exists():
        return []
    lines = journeys[0].read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            break
    return rows


def _emit(**fields: Any) -> None:
    fields["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(json.dumps(fields), flush=True)


class _Report:
    """What the watcher prints from each read, once each: cut-off replies and turn
    faults as they are appended, reply rows again whenever later facts change them."""

    def __init__(self) -> None:
        self.cut = 0
        self.faults = 0
        self.replies: dict[str | None, dict] = {}
        self.unreadable: str | None = None

    def emit(self, events: list[Event]) -> None:
        cut = cut_replies(events)
        for row in cut[self.cut :]:
            _emit(event="truncated", **row)
        self.cut = max(self.cut, len(cut))
        try:
            faults = turn_faults(events)
        except ValueError as error:  # an unreadable ledger is reported, never guessed
            faults = []
            if self.unreadable != str(error):
                self.unreadable = str(error)
                _emit(event="turn_faults_unreadable", error=self.unreadable)
        for row in faults[self.faults :]:
            _emit(event="turn_fault", **row)
        self.faults = max(self.faults, len(faults))
        for row in reply_rows(events):
            earlier = self.replies.get(row["response_event_id"])
            if earlier != row:
                updated = {"updated": True} if earlier is not None else {}
                _emit(event="reply", **row, **updated)
                self.replies[row["response_event_id"]] = row


def watch(
    run_root: Path,
    limits: Limits,
    journey_file: Path | None = None,
    interval: float = 2.0,
) -> int:
    operator = run_root / "operator"
    prompts = 0
    report = _Report()
    while not (operator / "finished.json").exists():
        events = _events(run_root, journey_file)
        report.emit(events)
        journey = journey_file or next(
            iter(sorted(run_root.glob("runs/*/journey.jsonl"))), None
        )
        try:
            gets = source_gets(journey.parent, events) if journey else 0
        except (OSError, ValueError):
            gets = None
        found = trips(events, limits, datetime.now(timezone.utc), gets)
        if found:
            interrupt = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bio_neb.devtools.native_ubuntu_operator",
                    "interrupt",
                    "--run-root",
                    str(run_root),
                ],
                capture_output=True,
                text=True,
            )
            _emit(event="trip", trips=found, interrupt_exit=interrupt.returncode)
            return 2
        stderr = operator / "stderr.log"
        seen = (
            stderr.read_text(errors="replace").count(_PROMPT) if stderr.exists() else 0
        )
        if seen > prompts:
            prompts = seen
            _emit(event="review_prompt", count=seen)
        time.sleep(interval)
    # What arrived with the finish record is still reported.
    report.emit(_events(run_root, journey_file))
    _emit(event="finished")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.watch")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--journey-file", type=Path)
    parser.add_argument("--max-requests", type=int, default=Limits.max_requests)
    parser.add_argument("--max-source-reads", type=int, default=Limits.max_source_reads)
    parser.add_argument("--repeat", type=int, default=Limits.repeat)
    parser.add_argument("--stall-seconds", type=float, default=Limits.stall_seconds)
    args = parser.parse_args(argv)
    limits = Limits(
        args.max_requests, args.max_source_reads, args.repeat, args.stall_seconds
    )
    return watch(args.run_root, limits, args.journey_file)


if __name__ == "__main__":
    raise SystemExit(main())
