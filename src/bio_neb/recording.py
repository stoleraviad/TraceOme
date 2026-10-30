"""Coordinator processing intervals and their correlated mechanical evidence.

The Journey owns meaning. This journal records contact evidence and never grants
authority, chooses domain work, or retries an operation.
"""

import hashlib
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial, wraps
from pathlib import Path
from typing import Literal, NamedTuple, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from bio_neb.content import ContentId, canonical_json
from bio_neb.failure import HeldFailure
from bio_neb.journey import FORK_ORIGIN_EVENT_TYPE, read_only_json
from bio_neb.persistence import append_durable

PassDisposition = Literal["waiting", "completed", "interrupted", "held"]


class PassStarted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    pass_id: str


class PassClosed(PassStarted):
    started_event_id: str
    disposition: PassDisposition
    journal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    journal_bytes: int = Field(ge=0)
    accounting_error: str | None = None
    # The failure module's record when it held the Run; absent otherwise.
    failure: HeldFailure | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def held_names_its_failure(self) -> "PassClosed":
        if (self.disposition == "held") != (self.failure is not None):
            raise ValueError("a held Pass, and only a held Pass, records its failure")
        return self


class ContactScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    attempt_id: str
    request_id: str
    item_id: str | None = None
    revision_id: str | None = None
    call_id: str | None = None
    evidence_event_id: str | None = None


class ContactEvidence(BaseModel):
    """One typed mechanical record, ordered against an existing semantic prefix.

    The anchor orders evidence; it grants no semantic ancestry or authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    event_id: str
    event_type: Literal[
        "ObservedToolExecutionStarted",
        "ObservedToolExecutionSucceeded",
        "ObservedToolExecutionFailed",
        "ToolOperationFailureRecorded",
        "ModelPhysicalContactStarted",
        "ModelPhysicalContactFailed",
        "ModelPhysicalContactCancelled",
        "ModelAvailabilityRetryScheduled",
        "ModelAvailabilityRetryDelayCompleted",
        "ModelAvailabilityRetryDelayCancelled",
        "SourceContactScopeStarted",
        "SourceContactScopeFinished",
        "SourcePhysicalContactStarted",
        "SourcePhysicalContactFinished",
        "SourceOperationFailureRecorded",
        "SourceContactLimitReached",
    ]
    after_event_id: str
    occurred_at: str
    payload: dict[str, JsonValue]


_CONTACT_EVIDENCE_TYPES = frozenset(
    get_args(ContactEvidence.model_fields["event_type"].annotation)
)


class ContactRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    sequence: int = Field(ge=1)
    run_id: str
    pass_id: str
    kind: Literal["start", "terminal", "record", "final"]
    contact_id: str | None = None
    scope: ContactScope | None = None
    operation: str | None = None
    disposition: Literal["succeeded", "failed", "cancelled"] | None = None
    cause: str | None = None
    prefix_sha256: str | None = None
    complete: bool | None = None
    evidence: ContactEvidence | None = None
    domain: "DomainContactRecord | None" = None


class DomainContactRecord(BaseModel):
    """Domain-typed mechanics anchored to one admitted operation and interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    owner_event_id: str
    after_event_id: str
    operation_id: str
    iteration: int = Field(ge=1)
    phase: Literal["started", "record", "closed"] = "record"
    complete: bool | None = None
    payload: dict[str, JsonValue]


class ContactJournal:
    """One serialized journal; starts precede effects and closure is immutable."""

    def __init__(self, path: Path, run_id: str, pass_id: str):
        self.path = path
        self.run_id = run_id
        self.pass_id = pass_id
        self._lock = threading.RLock()
        self._prefix = b""
        self._sequence = 0
        self._pending: dict[str, tuple[ContactScope, str]] = {}
        self._domains = set()
        self._domain_starts = set()
        self._domain_closed = set()
        self._closed = False
        self._failed = False
        self._reader = _ContactReader(run_id, pass_id)
        self.accounting_error: str | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)

    def _append(self, **fields):
        if self._closed or self._failed:
            raise ValueError("contact journal is not writable")
        row = ContactRow(
            sequence=self._sequence + 1,
            run_id=self.run_id,
            pass_id=self.pass_id,
            **fields,
        )
        content = (canonical_json(row.model_dump(mode="json")) + "\n").encode()
        try:
            append_durable(self.path, content)
        except OSError as error:
            self.mark_damaged(error)
            raise
        self._prefix += content
        self._sequence += 1

    def mark_damaged(self, error: OSError) -> None:
        self._failed = True
        self.accounting_error = type(error).__name__

    def snapshot(self) -> bytes:
        """Only the acknowledged durable prefix; an uncertain write stays unknown."""
        with self._lock:
            return self._prefix

    def reader(self) -> "_ContactReader":
        """The acknowledged prefix read once per row; later reads add only new rows.

        A rejected row discards the partial reader, so every read refuses again.
        """
        with self._lock:
            try:
                self._reader.feed(self._prefix)
            except ValueError:
                self._reader = _ContactReader(self.run_id, self.pass_id)
                raise
            return self._reader

    def start(
        self,
        scope: ContactScope,
        operation: str,
        *,
        evidence: ContactEvidence | None = None,
    ) -> str:
        with self._lock:
            contact_id = (
                evidence.event_id if evidence is not None else str(uuid.uuid4())
            )
            self._append(
                kind="start",
                contact_id=contact_id,
                scope=scope,
                operation=operation,
                evidence=evidence,
            )
            self._pending[contact_id] = (scope, operation)
            return contact_id

    def finish(
        self,
        contact_id: str,
        disposition: str,
        *,
        cause: str | None = None,
        evidence: ContactEvidence | None = None,
    ):
        with self._lock:
            if contact_id not in self._pending:
                raise ValueError("contact terminal lacks its exact open start")
            scope, operation = self._pending[contact_id]
            self._append(
                kind="terminal",
                contact_id=contact_id,
                scope=scope,
                operation=operation,
                disposition=disposition,
                cause=cause,
                evidence=evidence,
            )
            del self._pending[contact_id]

    def record(self, evidence: ContactEvidence):
        with self._lock:
            self._append(kind="record", evidence=evidence)

    def record_domain(self, record: DomainContactRecord):
        with self._lock:
            identity = (record.owner_event_id, record.operation_id, record.iteration)
            if record.phase == "started":
                if identity in self._domain_starts:
                    raise ValueError("domain contact scope already started")
                self._domain_starts.add(identity)
                self._domains.add(identity)
            elif identity not in self._domains or identity in self._domain_closed:
                raise ValueError("domain contact lacks its exact open operation")
            self._append(kind="record", domain=record)
            if record.phase == "closed":
                self._domain_closed.add(identity)
                if record.complete:
                    self._domains.remove(identity)

    def close(self) -> bytes:
        with self._lock:
            if not self._closed:
                self._append(
                    kind="final",
                    prefix_sha256=hashlib.sha256(self._prefix).hexdigest(),
                    complete=not self._pending and not self._domains,
                )
                self._closed = True
            return self._prefix


_current_pass: ContextVar["ProcessingPass | None"] = ContextVar(
    "coordinator_pass", default=None
)


class ProcessingPass:
    """Lazy interval: observing, replaying or refusing an answer adds no records."""

    def __init__(self, journey):
        self.journey = journey
        self.pass_id = str(uuid.uuid4())
        self.started = None
        self.journal = None
        self.closed = False
        # A resume's interval lifts a hold while it runs; see lift_hold.
        self.lifts_hold = False

    def start(self):
        if self.started is not None:
            return
        # Disable the lazy hook before appending the interval's own fact.
        self.journey.before_write = None
        head = self.journey.head_event_id
        self.started = self.journey.record(
            "PassStarted", PassStarted(pass_id=self.pass_id).model_dump(), [head]
        )
        self.journal = ContactJournal(
            self.journey.run_directory / "contacts" / self.pass_id / "journal.jsonl",
            self.journey.journey_id,
            self.pass_id,
        )

    def close(self, disposition, failure: HeldFailure | None = None):
        if self.started is None or self.closed or self.journal is None:
            # Initialization failed before any contact. The semantic writer may
            # be poisoned; retain the unfinished start and preserve the cause.
            return
        if self.journal.accounting_error is None:
            try:
                self.journal.close()
            except OSError as error:
                self.journal.mark_damaged(error)
        content = self.journal.snapshot()
        fact = PassClosed(
            pass_id=self.pass_id,
            started_event_id=self.started["event_id"],
            disposition=disposition,
            journal_sha256=hashlib.sha256(content).hexdigest(),
            journal_bytes=len(content),
            accounting_error=self.journal.accounting_error,
            failure=failure,
        )
        head = self.journey.head_event_id
        self.journey.record_with_artifact(
            "PassClosed",
            fact.model_dump(),
            content,
            "application/jsonl",
            [head, self.started["event_id"]]
            if head != self.started["event_id"]
            else [head],
        )
        self.closed = True

    def hold(self, failure: HeldFailure) -> None:
        """Record the failure module's hold: this interval closes held with it.

        A hold is retained even when the interval wrote nothing yet.
        """
        self.lifts_hold = False
        self.start()
        self.close("held", failure)

    def lift_hold(self) -> None:
        """Resume after a fix: this interval reads past a recorded hold.

        Its first write starts the fresh Pass as usual (lazily, so Attempt
        parentage is unchanged); ``settle_lift`` records it when nothing was
        written, so the Journey agrees with what the resume reported.
        """
        self.lifts_hold = True

    def settle_lift(self) -> None:
        if self.lifts_hold and self.started is None:
            self.lifts_hold = False
            if held_failure(self.journey) is not None:
                self.start()
        self.lifts_hold = False

    def close_at_child_point(self) -> None:
        """Close this interval at an ADR 0031 child point and arm a fresh one.

        The closed Pass is the retained point a Fork can admit; the coordinator's
        next write starts the new interval. An interval that wrote nothing has no
        point to close.
        """
        if self.started is None:
            return
        self.close("completed")
        self.pass_id = str(uuid.uuid4())
        self.started = None
        self.journal = None
        self.closed = False
        self.journey.before_write = self.start


class PassView(PassStarted):
    started_event_id: str
    closed_event_id: str | None = None
    disposition: Literal["active", "waiting", "completed", "interrupted", "held"] = (
        "active"
    )
    contacts: tuple[ContactRow, ...] = ()
    journal_complete: bool = False
    accounting_error: str | None = None
    failure: HeldFailure | None = None


def held_failure(journey) -> HeldFailure | None:
    """The hold the Run's latest Pass closed with; a fresh Pass lifts it.

    Within a resume's processing interval the hold is already lifted.
    """
    interval = _current_pass.get()
    if (
        interval is not None
        and interval.lifts_hold
        and interval.journey.journey_id == journey.journey_id
    ):
        return None
    latest = journey.latest_event(("PassStarted", "PassClosed"))
    if latest is None or latest["event_type"] != "PassClosed":
        return None
    return PassClosed.model_validate(latest["payload"]).failure


# Interval metadata and a Fork origin change no reviewed domain object or authority.
REVIEW_METADATA = frozenset({"PassStarted", "PassClosed", FORK_ORIGIN_EVENT_TYPE})


def review_head(events):
    """The latest event that is not ``REVIEW_METADATA``."""
    return next(
        event
        for event in reversed(events)
        if event["event_type"] not in REVIEW_METADATA
    )


def work_frontier(events):
    """The latest domain event that work continues from.

    Like ``review_head``, and past a retained configuration fault too: the hold
    left it as a side leaf at the causal head, and the fixed Run resumes from
    that head (#216, #107).
    """
    return review_head(
        [
            event
            for event in events
            if event["event_type"] != "RuntimeConfigurationFailed"
        ]
    )


def _validate_contact_detail(row, evidence_ids):
    if row.domain is not None:
        if (
            row.kind != "record"
            or row.evidence is not None
            or row.contact_id is not None
            or row.scope is not None
            or row.operation is not None
            or row.disposition is not None
        ):
            raise ValueError("domain contact changed its record shape")
        return
    if row.kind == "record":
        if (
            row.evidence is None
            or row.contact_id is not None
            or row.scope is not None
            or row.operation is not None
            or row.disposition is not None
            or row.evidence.event_type.startswith("ObservedToolExecution")
        ):
            raise ValueError("mechanical detail changed its record contract")
    if row.evidence is not None:
        evidence = row.evidence
        if evidence.event_id in evidence_ids:
            raise ValueError("mechanical evidence identity was reused")
        if str(uuid.UUID(evidence.event_id)) != evidence.event_id:
            raise ValueError("mechanical evidence identity is not canonical")
        if str(uuid.UUID(evidence.after_event_id)) != evidence.after_event_id:
            raise ValueError("mechanical evidence anchor is not canonical")
        evidence_ids.add(evidence.event_id)
    if row.operation == "tool.execute":
        evidence = row.evidence
        if evidence is None or row.scope is None:
            raise ValueError("Tool contact lost its unique typed evidence")
        if evidence.payload.get("attempt_id") != row.scope.attempt_id:
            raise ValueError("Tool contact changed its owning Attempt")
        if row.kind == "start":
            if (
                evidence.event_type != "ObservedToolExecutionStarted"
                or evidence.event_id != row.contact_id
                or evidence.payload.get("validation_event_id") != row.scope.request_id
                or evidence.payload.get("tool_call_id") != row.scope.call_id
            ):
                raise ValueError("Tool start changed its exact execution identity")
        elif row.kind == "terminal":
            expected = {
                "succeeded": "ObservedToolExecutionSucceeded",
                "failed": "ObservedToolExecutionFailed",
                "cancelled": "ObservedToolExecutionFailed",
            }.get(row.disposition)
            if (
                evidence.event_type != expected
                or evidence.payload.get("started_event_id") != row.contact_id
            ):
                raise ValueError("Tool terminal changed its exact execution identity")
    elif row.evidence is not None and row.kind != "record":
        raise ValueError("contact carries unsupported mechanical evidence")


class _ContactReader:
    """Validate journal rows once each, in order, as the acknowledged prefix grows.

    It also holds the indexes a new contact start is checked against: the
    journal's evidence records and the first contact citing each evidence id.
    """

    def __init__(self, run_id: str, pass_id: str):
        self.run_id = run_id
        self.pass_id = pass_id
        self.rows: tuple[ContactRow, ...] = ()
        self.consumed = 0
        self.evidence: dict[str, ContactRow] = {}
        self.evidence_owners: dict[str, str | None] = {}
        self._digest = hashlib.sha256()
        self._pending = {}
        self._seen = set()
        self._closed = False
        self._evidence_ids = set()
        self._domains = set()
        self._domain_starts = set()
        self._domain_closed = set()

    def feed(self, content: bytes) -> tuple[ContactRow, ...]:
        """Admit the rows after those already read; ``content`` only extends."""
        rows, consumed = list(self.rows), self.consumed
        for line in content[consumed:].splitlines(keepends=True):
            rows.append(self._admit(line, len(rows) + 1))
            consumed += len(line)
        self.rows, self.consumed = tuple(rows), consumed
        return self.rows

    def _admit(self, line: bytes, sequence: int) -> ContactRow:
        row = ContactRow.model_validate_json(line)
        if (
            self._closed
            or not line.endswith(b"\n")
            or (
                canonical_json(row.model_dump(mode="json", exclude_unset=True)) + "\n"
            ).encode()
            != line
            or row.sequence != sequence
            or row.run_id != self.run_id
            or row.pass_id != self.pass_id
        ):
            raise ValueError("contact journal prefix or identity changed")
        _validate_contact_detail(row, self._evidence_ids)
        if row.domain is not None:
            domain = row.domain
            identity = (domain.owner_event_id, domain.operation_id, domain.iteration)
            if domain.phase == "started":
                if (
                    identity in self._domain_starts
                    or domain.payload
                    or domain.complete is not None
                ):
                    raise ValueError("domain contact scope start changed")
                self._domain_starts.add(identity)
                self._domains.add(identity)
            elif identity not in self._domains or identity in self._domain_closed:
                raise ValueError("domain contact lacks its exact open operation")
            elif domain.phase == "closed":
                if domain.complete is None:
                    raise ValueError("domain contact closure lacks completeness")
                if domain.complete:
                    self._domains.remove(identity)
                self._domain_closed.add(identity)
        if row.kind == "start":
            if (
                not row.contact_id
                or row.contact_id in self._seen
                or row.scope is None
                or not row.operation
                or row.disposition is not None
            ):
                raise ValueError("contact start is not unique and scoped")
            self._pending[row.contact_id] = (row.scope, row.operation)
            self._seen.add(row.contact_id)
        elif row.kind == "terminal":
            if (
                self._pending.get(row.contact_id) != (row.scope, row.operation)
                or row.disposition is None
            ):
                raise ValueError("contact terminal changed its exact start")
            del self._pending[row.contact_id]
        elif row.kind == "final":
            if row.prefix_sha256 != self._digest.hexdigest() or row.complete != (
                not self._pending and not self._domains
            ):
                raise ValueError("contact final binding changed")
            if (
                row.contact_id is not None
                or row.scope is not None
                or row.operation is not None
                or row.disposition is not None
            ):
                raise ValueError("contact final carries an effect identity")
            self._closed = True
        if row.evidence is not None:
            self.evidence[row.evidence.event_id] = row
        if row.scope is not None and row.scope.evidence_event_id is not None:
            self.evidence_owners.setdefault(row.scope.evidence_event_id, row.contact_id)
        self._digest.update(line)
        return row


def read_contacts(
    content: bytes, *, run_id: str, pass_id: str
) -> tuple[ContactRow, ...]:
    """Validate actual start/terminal identity, including an unfinished prefix."""
    return _ContactReader(run_id, pass_id).feed(content)


def validate_contact_ownership(rows, events, started_event_id, closed_event_id=None):
    """Same exact correlation checks for active evidence and retained closures."""
    by_id = {event["event_id"]: event for event in events}
    positions = {event["event_id"]: index for index, event in enumerate(events)}
    lower = positions[started_event_id]
    upper = positions[closed_event_id] if closed_event_id else len(events)
    evidence_owners = {}
    domain_validators = set()

    def located(event_id):
        event = by_id.get(event_id)
        return None if event is None else (positions[event_id], event)

    def claim(evidence_id, contact_id):
        return evidence_owners.setdefault(evidence_id, contact_id)

    for row in rows:
        if row.domain is not None:
            domain = row.domain
            if (
                row.kind != "record"
                or domain.owner_event_id not in positions
                or domain.after_event_id not in positions
                or not lower <= positions[domain.after_event_id] < upper
                or positions[domain.owner_event_id] > positions[domain.after_event_id]
            ):
                raise ValueError(
                    "domain contact is outside its admitted operation/Pass"
                )
            from bio_neb.journey_contracts import semantic_families

            owner = by_id[domain.owner_event_id]
            identities = [
                family.contact_identity(owner)
                for family in semantic_families(frozenset({owner["event_type"]}))
                if owner["event_type"] in family.payloads
                and family.contact_identity is not None
            ]
            if identities != [domain.operation_id]:
                raise ValueError("domain contact changed its exact bound operation")
            domain_validators.update(
                family.contact_validation
                for family in semantic_families(frozenset({owner["event_type"]}))
                if owner["event_type"] in family.payloads
                and family.contact_validation is not None
            )
        if row.scope is not None:
            _check_contact_scope(row, located, lower, upper, claim)
    for validate in domain_validators:
        validate(rows)


class PassFacts(NamedTuple):
    """Checked interval state: the open Pass and every Pass id already used."""

    active: tuple[str, str] | None = None
    seen: frozenset[str] = frozenset()


def step_pass_facts(state: PassFacts, event, prefix=()) -> PassFacts:
    """Admit one fact into the interval state; other event types leave it as is."""
    active, seen = state
    if event["event_type"] == "PassStarted":
        fact = PassStarted.model_validate(event["payload"])
        if (
            str(uuid.UUID(fact.pass_id)) != fact.pass_id
            or fact.pass_id in seen
            or active is not None
        ):
            raise ValueError("Pass start is not an independent processing interval")
        if len(event["parent_ids"]) != 1:
            raise ValueError("Pass start lacks its admitted predecessor")
        return PassFacts((fact.pass_id, event["event_id"]), seen | {fact.pass_id})
    if event["event_type"] == "PassClosed":
        fact = PassClosed.model_validate(event["payload"])
        if (
            active != (fact.pass_id, fact.started_event_id)
            or fact.started_event_id not in event["parent_ids"]
        ):
            raise ValueError("Pass closure changed its exact interval")
        refs = event["artifact_references"]
        if (
            len(refs) != 1
            or refs[0]["artifact_id"] != str(ContentId(fact.journal_sha256))
            or refs[0]["byte_count"] != fact.journal_bytes
        ):
            raise ValueError("Pass closure changed its journal artifact")
        return PassFacts(None, seen)
    return state


_EXPECTED_CONTACT_TYPES = {
    "model.request": ("ModelRequestRecorded", "ModelPhysicalContactStarted"),
    "source.get": ("SourceContactScopeStarted", "SourcePhysicalContactStarted"),
    "source.head": ("SourceContactScopeStarted", "SourcePhysicalContactStarted"),
    "tool.execute": ("ToolCallValidated", "ObservedToolExecutionStarted"),
}


def _check_contact_scope(row, located, lower, upper, claim):
    """One scoped row's exact request/evidence ownership within its Pass.

    ``located(event_id)`` returns the merged-evidence order and event, or None;
    orders compare like positions in ``_merge_evidence``'s combined prefix.
    ``claim(evidence_id, contact_id)`` returns the first contact citing it.
    """
    scope = row.scope
    request_at, request = located(scope.request_id) or (None, None)
    if request is None or not lower < request_at < upper:
        raise ValueError("contact request is outside its Pass")
    payload = request["payload"]
    if payload.get("attempt_id") != scope.attempt_id:
        raise ValueError("contact changed its owning Attempt/request")
    for field in ("item_id", "revision_id"):
        if payload.get(field) != getattr(scope, field):
            raise ValueError("contact changed its exact item/revision")
    if payload.get("tool_call_id") != scope.call_id:
        raise ValueError("contact changed its exact Tool call")
    expected = _EXPECTED_CONTACT_TYPES.get(row.operation)
    if expected is not None and (
        scope.evidence_event_id is None or request["event_type"] != expected[0]
    ):
        raise ValueError("physical contact lacks its exact retained evidence")
    if scope.evidence_event_id is not None:
        if claim(scope.evidence_event_id, row.contact_id) != row.contact_id:
            raise ValueError("physical evidence was reused by another contact")
        evidence_at, evidence = located(scope.evidence_event_id) or (None, None)
        if evidence is None or not request_at < evidence_at < upper:
            raise ValueError("contact evidence is outside its exact request")
        if expected is not None and evidence["event_type"] != expected[1]:
            raise ValueError("physical evidence has the wrong contract")
        detail = evidence["payload"]
        if (
            detail.get(
                "logical_request_event_id",
                detail.get("scope_event_id", detail.get("validation_event_id")),
            )
            != scope.request_id
        ):
            raise ValueError("contact evidence changed its exact request")
        if detail.get("attempt_id", scope.attempt_id) != scope.attempt_id:
            raise ValueError("contact evidence changed its Attempt")


def _check_contact_candidate(interval, row: ContactRow) -> None:
    """Check a new start against held indexes, as the full check would place it.

    Semantic order comes from the owner's held positions and journal evidence
    sorts directly after its anchor, so ``(position, record, sequence)`` orders
    exactly like the merged evidence prefix. The journal reader parses earlier
    rows once; their ownership is validated in full at every read, close and open.
    """
    journey = interval.journey
    reader = interval.journal.reader()

    def located(event_id):
        position = journey.event_position(event_id)
        if position is not None:
            return (position, 0, 0), journey.events_view()[position]
        held = reader.evidence.get(event_id)
        anchor = (
            None
            if held is None
            else journey.event_position(held.evidence.after_event_id)
        )
        if anchor is None:
            return None
        return (anchor, 1, held.sequence), _evidence_event(held)

    lower = (journey.event_position(interval.started["event_id"]), 0, 0)
    _check_contact_scope(
        row, located, lower, (float("inf"),), reader.evidence_owners.get
    )


def validate_pass_facts(events) -> PassFacts:
    """Typed semantic admission, also used for preview without journal effects."""
    state = PassFacts()
    for event in events:
        state = step_pass_facts(state, event)
    return state


def read_passes(journey) -> tuple[PassView, ...]:
    """Reopen interval evidence without contacts or interpretation from diagnostics."""
    views = []
    active = None
    events = journey.events_view()
    validate_pass_facts(events)
    for event in events:
        if event["event_type"] == "PassStarted":
            fact = PassStarted.model_validate(event["payload"])
            active = PassView(pass_id=fact.pass_id, started_event_id=event["event_id"])
        elif event["event_type"] == "PassClosed":
            assert active is not None  # Admitted by validate_pass_facts.
            # A closed journal is an immutable artifact named by the admitted
            # prefix: one full validation per owner; an independent open repeats it.
            views.append(
                journey.derived_once(
                    ("closed_pass", event["event_id"]),
                    partial(_closed_pass_view, journey, events, active, event),
                )
            )
            active = None
    if active is not None:
        path = journey.run_directory / "contacts" / active.pass_id / "journal.jsonl"
        interval = _current_pass.get()
        accounting_error = "journal_unavailable" if not path.exists() else None
        if (
            interval is not None
            and interval.pass_id == active.pass_id
            and interval.journal is not None
        ):
            # The active owner's acknowledged rows, each parsed once as it grew.
            rows = interval.journal.reader().rows
        else:
            content = path.read_bytes() if path.exists() else b""
            if content and not content.endswith(b"\n"):
                # A torn mechanical tail grants no terminal or final success. Keep
                # the complete admitted rows and expose the missing accounting.
                content = content[: content.rfind(b"\n") + 1]
                accounting_error = "truncated_prefix"
            rows = read_contacts(
                content, run_id=journey.journey_id, pass_id=active.pass_id
            )
        validate_contact_ownership(
            rows,
            _merge_evidence(events, rows, active.started_event_id),
            active.started_event_id,
        )
        views.append(
            active.model_copy(
                update={
                    "contacts": rows,
                    "accounting_error": accounting_error,
                }
            )
        )
    return tuple(views)


def _closed_pass_view(journey, events, active: PassView, event) -> PassView:
    """Fully validate one closed Pass journal against its exact semantic interval."""
    fact = PassClosed.model_validate(event["payload"])
    content = journey.read_artifact(event["artifact_references"][0])
    # A Fork's inherited Pass keeps its parent Run identity.
    rows = read_contacts(content, run_id=event["journey_id"], pass_id=fact.pass_id)
    if fact.accounting_error is None and (not rows or rows[-1].kind != "final"):
        raise ValueError("Pass closure lacks a final contact binding")
    validate_contact_ownership(
        rows,
        _merge_evidence(events, rows, active.started_event_id, event["event_id"]),
        active.started_event_id,
        event["event_id"],
    )
    return active.model_copy(
        update={
            "closed_event_id": event["event_id"],
            "disposition": fact.disposition,
            "failure": fact.failure,
            "contacts": rows,
            "journal_complete": fact.accounting_error is None
            and bool(rows)
            and rows[-1].complete,
            "accounting_error": fact.accounting_error,
        }
    )


def _evidence_event(row: ContactRow) -> dict:
    assert row.evidence is not None
    return _merged_evidence_event(row.run_id, row.evidence)


def _merged_evidence_event(run_id: str, evidence: ContactEvidence) -> dict:
    return {
        "contract_version": "1.0",
        "event_id": evidence.event_id,
        "event_type": evidence.event_type,
        "journey_id": run_id,
        "occurred_at": evidence.occurred_at,
        # Ordering is not semantic parentage. Domain facts keep their real parents.
        "parent_ids": [],
        "payload": evidence.payload,
        "artifact_references": [],
    }


def _merge_evidence(events, rows, started_event_id, closed_event_id=None):
    """Project exact recorded ordering without manufacturing authority edges."""
    positions = {event["event_id"]: index for index, event in enumerate(events)}
    lower = positions[started_event_id]
    upper = positions[closed_event_id] if closed_event_id else len(events)
    anchored = {}
    previous = lower
    seen = set(positions)
    for row in rows:
        evidence = row.evidence
        if evidence is None:
            continue
        position = positions.get(evidence.after_event_id, -1)
        if (
            not lower <= position < upper
            or position < previous
            or evidence.event_id in seen
        ):
            raise ValueError("contact evidence changed its exact semantic prefix")
        previous = position
        seen.add(evidence.event_id)
        anchored.setdefault(evidence.after_event_id, []).append(_evidence_event(row))
    combined = []
    for event in events:
        combined.append(event)
        combined.extend(anchored.get(event["event_id"], ()))
    return combined


def is_contact_evidence(event) -> bool:
    """Whether a merged-evidence event is a journal record, not a semantic fact."""
    return not event["parent_ids"] and event["event_type"] in _CONTACT_EVIDENCE_TYPES


def contact_evidence_anchor(events, record) -> str:
    """Return the exact `after_event_id` of one record in merged evidence.

    Merging places each record directly after its anchor, so the nearest
    preceding semantic fact is that anchor. It orders; it is never a parent.
    """
    for event in reversed(events[: events.index(record)]):
        if not is_contact_evidence(event):
            return event["event_id"]
    raise ValueError("contact evidence lacks its exact semantic anchor")


def read_evidence(journey) -> list[dict]:
    """Read semantic facts and correlated mechanical evidence with one grammar.

    The Contact journal remains the only mechanical store. These ephemeral
    records are consumed by the existing typed domain and correlation readers;
    they are never written back to the Journey or interpreted as authority.
    Facts and journal rows are read as one snapshot under the writer lock, so a
    Tool thread never sees a row whose anchor its fact list lacks (#105).
    """
    from bio_neb.journey import _writers_lock

    with _writers_lock:
        return _with_pass_evidence(journey, journey.read())


def _with_pass_evidence(journey, events) -> list:
    for interval in read_passes(journey):
        events = _merge_evidence(
            events,
            interval.contacts,
            interval.started_event_id,
            interval.closed_event_id,
        )
    return list(events)


def evidence_view(journey) -> list:
    """``read_evidence`` for owner-internal readers, without copying history.

    Semantic facts are the owner's shared read-only events and the merged
    journal records are read-only too; mutating either raises ``TypeError``.
    """
    from bio_neb.journey import _writers_lock

    with _writers_lock:
        events = _with_pass_evidence(journey, journey.events_view())
    return [read_only_json(event) for event in events]


def evidence_revision(journey) -> tuple:
    """Key a contact-dependent reader by held prefix and active journal growth.

    The admitted prefix only extends, and closed Pass journals are immutable
    artifacts named by it, so the held revision and head identify both. Only
    the active Pass journal grows outside the semantic prefix.
    """
    latest = journey.latest_event(("PassStarted", "PassClosed"))
    active = None
    if latest is not None and latest["event_type"] == "PassStarted":
        pass_id = latest["payload"]["pass_id"]
        path = journey.run_directory / "contacts" / pass_id / "journal.jsonl"
        interval = _current_pass.get()
        exists = path.exists()
        if (
            interval is not None
            and interval.pass_id == pass_id
            and interval.journal is not None
        ):
            size = len(interval.journal.snapshot())
        else:
            size = path.stat().st_size if exists else 0
        active = (pass_id, exists, size)
    return (journey.revision, journey.head_event_id, active)


def record_contact_evidence(journey, fact: BaseModel) -> dict:
    """Retain typed mechanical evidence under the one shared contact owner."""
    interval = _current_pass.get()
    if interval is None or interval.journey is not journey:
        raise RuntimeError("mechanical evidence requires its owning processing Pass")
    interval.start()
    payload = fact.model_dump(mode="json")
    event_type = payload.pop("event_type")
    from bio_neb.journey import _writers_lock

    # A Tool thread records beside concurrent Attempts' writes: the anchor
    # is read and the row retained under the one writer lock, so rows follow
    # the Journey's order.
    with _writers_lock:
        return _retain_contact_evidence(journey, interval, event_type, payload)


def _retain_contact_evidence(journey, interval, event_type, payload) -> dict:
    evidence = ContactEvidence(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        after_event_id=journey.head_event_id,
        occurred_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        payload=payload,
    )
    journal = interval.journal
    if event_type == "ObservedToolExecutionStarted":
        scope = ContactScope(
            attempt_id=payload["attempt_id"],
            request_id=payload["validation_event_id"],
            call_id=payload["tool_call_id"],
            evidence_event_id=evidence.event_id,
        )
        journal.start(scope, "tool.execute", evidence=evidence)
    elif event_type in {
        "ObservedToolExecutionSucceeded",
        "ObservedToolExecutionFailed",
    }:
        journal.finish(
            payload["started_event_id"],
            "succeeded" if event_type == "ObservedToolExecutionSucceeded" else "failed",
            cause=payload.get("mechanical_error_type"),
            evidence=evidence,
        )
    else:
        journal.record(evidence)
    # The journal admitted this typed evidence; return it in its retained canonical
    # form without reparsing the whole journal.
    retained = ContactEvidence.model_validate_json(
        canonical_json(evidence.model_dump(mode="json"))
    )
    return _merged_evidence_event(journey.journey_id, retained)


@contextmanager
def processing_pass(journey):
    existing = _current_pass.get()
    if existing is not None:
        if existing.journey is not journey:
            raise RuntimeError("another Journey owns this processing Pass")
        yield existing
        return
    interval = ProcessingPass(journey)
    token = _current_pass.set(interval)
    journey.before_write = interval.start
    try:
        yield interval
    finally:
        journey.before_write = None
        _current_pass.reset(token)


def current_processing_pass():
    """The current interval; callers cannot create a second mechanical owner."""
    return _current_pass.get()


def invocation_pass(invoke=None, *, read_result=None):
    """Give standalone assignments a journal owner and an optional closed-Pass view."""
    if invoke is None:
        return lambda function: invocation_pass(function, read_result=read_result)

    @wraps(invoke)
    def run(owner, journey, *args, **kwargs):
        existing = _current_pass.get()
        with processing_pass(journey) as interval:
            try:
                result = invoke(owner, journey, *args, **kwargs)
            except BaseException as error:
                if existing is None:
                    try:
                        interval.close("interrupted")
                    except (OSError, ValueError) as closure_error:
                        error.add_note(
                            f"Pass closure could not be retained: {type(closure_error).__name__}"
                        )
                raise
            if existing is None:
                interval.close(
                    "waiting"
                    if "pending" in getattr(result, "status", "")
                    else "completed"
                )
                if read_result is not None:
                    result = read_result(journey)
            return result

    return run


@contextmanager
def physical_contact(scope: ContactScope, operation: str):
    """Record an already-authorized contact; never construct or invoke an adapter."""
    interval = _current_pass.get()
    if interval is None:
        yield
        return
    interval.start()
    journal = interval.journal
    _check_contact_candidate(
        interval,
        ContactRow(
            sequence=1,
            run_id=interval.journey.journey_id,
            pass_id=interval.pass_id,
            kind="start",
            contact_id=str(uuid.uuid4()),
            scope=scope,
            operation=operation,
        ),
    )
    contact_id = journal.start(scope, operation)
    try:
        yield
    except BaseException as error:
        disposition = (
            "cancelled"
            if type(error).__name__ in {"CancelledError", "KeyboardInterrupt"}
            else "failed"
        )
        try:
            journal.finish(contact_id, disposition, cause=type(error).__name__)
        except OSError as accounting_error:
            journal.mark_damaged(accounting_error)
        raise
    else:
        try:
            journal.finish(contact_id, "succeeded")
        except OSError as accounting_error:
            journal.mark_damaged(accounting_error)
