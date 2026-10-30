"""A Run paper in the Journey: retrieved or supplied once, then an evidence reference.

A Run paper enters a Run as a researcher-supplied file (``begin --paper-file``) or as a
citation the planning Agent types from the researcher's words and retrieves (#420). Each
file is retained once, content-addressed, by the same append as its
``PaperDocumentRetained``; each readable document's normalized text is retained by its own
``PaperDocumentRead``, so a reading span's ``start:end`` always points into the same text.
One ``PaperRecorded`` then names the paper: ``paper_id`` (derived from its files' digests),
its origin, the ids its source gave, its documents, and what it could not fetch or read.

A file that cannot be read, or a citation that cannot be retrieved, records
``PaperIntakeRefused`` and asks the failure module's Review (ADR 0031): ``retry``,
``amend`` (another ``paper_file`` path, or another ``paper_reference``), ``continue``
without the paper, or ``abort`` the Run (owner, #420 P1). The answer is the intake's one
next step; no paper work runs while its Review is pending.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from bio_neb.agentic_journey.source_answers import SourceAnswer
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, ReviewAnswer
from bio_neb.paper_intake.documents import FileRole

PAPER_TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"
# The named choice that a search page holds no row for the citation, as
# `record_reference_choice`'s "none".
NO_PAPER = "none"
PAPER_REVIEW_STAGE = "paper_intake"
Origin = Literal["file", "citation"]
RefusalCode = Literal[
    "paper_file_missing",
    "paper_file_unreadable",
    "paper_not_found",
    "paper_not_open_access",
    "paper_article_missing",
    "paper_download_stopped",
]
# The answers, in the order the Review shows them.
REVIEW_ACTIONS = ("retry", "amend", "continue", "abort")
AMENDABLE: Mapping[Origin, tuple[str, ...]] = {
    "file": ("paper_file",),
    "citation": ("paper_reference", "paper_file"),
}


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _absolute(value: str) -> str:
    if not value or not os.path.isabs(value):
        raise ValueError("a paper file path must be absolute")
    return value


class PaperFileRegistered(_Fact):
    """One researcher-supplied paper file, by path, in ``begin``'s order."""

    event_type: Literal["PaperFileRegistered"] = "PaperFileRegistered"
    intake_id: StrictStr
    paper_file: StrictStr

    _path = field_validator("paper_file")(_absolute)


class PaperDocumentRetained(_Fact):
    """One file of a Run paper, its exact bytes retained by this append."""

    event_type: Literal["PaperDocumentRetained"] = "PaperDocumentRetained"
    intake_id: StrictStr
    name: StrictStr = Field(min_length=1)
    role: FileRole
    media_type: StrictStr = Field(min_length=1)
    # Where it came from: the GET's URL for a download, the path for a supplied file.
    location: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)


class PaperDocumentRead(_Fact):
    """One readable document's normalized text, retained by this append."""

    event_type: Literal["PaperDocumentRead"] = "PaperDocumentRead"
    intake_id: StrictStr
    retained_event_id: StrictStr
    document_id: StrictStr = Field(pattern=r"^(main|supp:.+)$")
    name: StrictStr = Field(min_length=1)
    text_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    characters: int = Field(ge=0)


class PaperIds(FrozenContract):
    """What the paper's source says it is, copied; absent values are unknown."""

    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year: str | None = None
    open_access: bool | None = None


class PaperSearchRow(PaperIds):
    """One row of a Europe PMC search page, copied, with the record id the model
    cites it by (Europe PMC's ``source:id``)."""

    candidate_id: StrictStr = Field(min_length=1)

    def paper_ids(self) -> PaperIds:
        return PaperIds(**self.model_dump(exclude={"candidate_id"}))


class PaperSearchInput(FrozenContract):
    """A ``find_paper`` call as admitted: the cited paper it serves (its
    ``cited_paper`` requirement's subject key, #420 D1 b; one the plan declares, or,
    before the plan, will declare), Europe PMC's options,
    verbatim, and the researcher's words the model searches for."""

    subject_key: StrictStr = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    query: StrictStr = Field(min_length=1, max_length=1000)
    quote: StrictStr = Field(min_length=1, max_length=300)
    page_size: int = Field(ge=1, le=100)
    cursor_mark: StrictStr = Field(min_length=1)
    sort: StrictStr | None = None


class PaperSearchResult(FrozenContract):
    """What Europe PMC answered: one page of rows, copied in its order with its
    ``hitCount`` and ``nextCursorMark`` (``observed``), or its own answer about the
    query with another status (``answered``), whose body ``source_answers`` keeps."""

    status: Literal["observed", "answered"]
    summary: StrictStr
    hit_count: int | None = Field(default=None, ge=0)
    next_cursor_mark: StrictStr | None = None
    rows: tuple[PaperSearchRow, ...] = ()

    def row(self, candidate_id: str) -> PaperSearchRow | None:
        return next((r for r in self.rows if r.candidate_id == candidate_id), None)


class PaperSearchRecorded(_Fact):
    """One ``find_paper`` call: its admitted input, the copied page and what each
    of its GETs answered; each GET's bytes are retained on its own
    ``ReferenceSourceRetrieved`` (#458)."""

    event_type: Literal["PaperSearchRecorded"] = "PaperSearchRecorded"
    attempt_id: StrictStr
    tool_input: PaperSearchInput
    result: PaperSearchResult
    source_answers: tuple[SourceAnswer, ...]


class PaperRetrievalInput(FrozenContract):
    """A ``retrieve_paper`` call as admitted: the row it names on a recorded search
    page, the researcher's words it rests on and why, and the id downloaded (the
    row's PMCID, else its DOI, else its PMID, copied)."""

    search_event_id: StrictStr
    # A row's candidate id, or NO_PAPER: the page holds no row for the citation.
    candidate_id: StrictStr = Field(min_length=1)
    quote: StrictStr = Field(min_length=1, max_length=300)
    reason: StrictStr = Field(min_length=1, max_length=2000)
    reference: StrictStr | None = Field(default=None, min_length=1)


class PaperRetrievalRecorded(_Fact):
    """One ``retrieve_paper`` call's outcome: the paper recorded, or refused with its
    Review asked (owner, #420 P1)."""

    event_type: Literal["PaperRetrievalRecorded"] = "PaperRetrievalRecorded"
    attempt_id: StrictStr
    tool_input: PaperRetrievalInput
    intake_id: StrictStr
    status: Literal["recorded", "refused"]
    summary: StrictStr
    paper_event_id: StrictStr | None = None
    review_request_event_id: StrictStr | None = None
    code: RefusalCode | None = None


class RecordedDocument(FrozenContract):
    document_id: str
    name: str
    sha256: str  # the file it was read from
    text_sha256: str
    read_event_id: str


class RecordedFile(FrozenContract):
    """One file of a recorded paper: its ``PaperDocumentRetained`` and digest."""

    retained_event_id: str
    sha256: str


class UnreadFile(FrozenContract):
    """A file, or a zip member, the intake holds no text for, and why."""

    name: str
    reason: str


class PaperRecorded(_Fact):
    """One Run paper: the evidence reference every reading cites."""

    event_type: Literal["PaperRecorded"] = "PaperRecorded"
    intake_id: StrictStr
    paper_id: StrictStr = Field(pattern=r"^paper-[0-9a-f]{64}$")
    origin: Origin
    paper_file: StrictStr | None = None
    reference: StrictStr | None = None  # the citation as the model typed it
    quote: StrictStr | None = None  # the researcher's words it stands in
    ids: PaperIds | None = None
    # Every file this intake retained for this recording, in order.
    files: tuple[RecordedFile, ...] = Field(min_length=1)
    documents: tuple[RecordedDocument, ...] = Field(min_length=1)
    unread: tuple[UnreadFile, ...] = ()
    unfetched: tuple[UnreadFile, ...] = ()
    contact_event_ids: tuple[StrictStr, ...] = ()


class PaperIntakeRefused(_Fact):
    event_type: Literal["PaperIntakeRefused"] = "PaperIntakeRefused"
    intake_id: StrictStr
    origin: Origin
    code: RefusalCode
    detail: StrictStr = Field(min_length=1, max_length=2000)
    paper_file: StrictStr | None = None
    reference: StrictStr | None = None


class PaperReviewRequested(_Fact):
    event_type: Literal["PaperReviewRequested"] = "PaperReviewRequested"
    intake_id: StrictStr
    request: Ask


class PaperReviewAnswered(_Fact):
    event_type: Literal["PaperReviewAnswered"] = "PaperReviewAnswered"
    request_event_id: StrictStr
    response: ReviewAnswer


PAYLOADS = {
    c.model_fields["event_type"].default: c
    for c in (
        PaperFileRegistered,
        PaperDocumentRetained,
        PaperDocumentRead,
        PaperRecorded,
        PaperIntakeRefused,
        PaperReviewRequested,
        PaperReviewAnswered,
        PaperSearchRecorded,
        PaperRetrievalRecorded,
    )
}


def paper_id(file_sha256s: Sequence[str]) -> str:
    """``paper-`` and the sha256 of its files' sorted digests, one per line."""

    joined = "\n".join(sorted(file_sha256s)).encode()
    return f"paper-{hashlib.sha256(joined).hexdigest()}"


# ---------------------------------------------------------------------------
# The checked view
# ---------------------------------------------------------------------------

IntakeState = Literal[
    "due",  # its file is to be read (registered, retried or amended to a file)
    "reference_due",  # its citation is to be typed again by planning (amended)
    "recorded",
    "review_pending",
    "dropped",  # continue without it
    "aborted",
]


class PaperIntake(FrozenContract):
    intake_id: str
    origin: Origin
    state: IntakeState
    source_event_id: str  # the fact its next step rests on
    paper_file: str | None = None
    reference: str | None = None
    paper_event_id: str | None = None
    refusal: PaperIntakeRefused | None = None
    # The researcher's free text of an `amend paper_reference` answer, for planning.
    answer_event_id: str | None = None


class PaperReviewRequest(FrozenContract):
    """The shared Review view of one refused paper."""

    request_event_id: str
    intake_id: str
    origin: Origin
    refusal: PaperIntakeRefused
    request: Ask
    pending: bool


class PaperReviewReceipt(FrozenContract):
    request_event_id: str
    response_event_id: str
    response: ReviewAnswer


class PaperSearchView(PaperSearchRecorded):
    source_event_id: StrictStr


class PaperRetrievalView(PaperRetrievalRecorded):
    source_event_id: StrictStr


class RunPapers(FrozenContract):
    intakes: tuple[PaperIntake, ...] = ()
    papers: tuple[PaperRecorded, ...] = ()
    requests: tuple[PaperReviewRequest, ...] = ()
    receipts: tuple[PaperReviewReceipt, ...] = ()
    # The planning Agent's searches and retrievals, in order.
    searches: tuple[PaperSearchView, ...] = ()
    retrievals: tuple[PaperRetrievalView, ...] = ()

    def search(self, event_id: str) -> PaperSearchView | None:
        return next((s for s in self.searches if s.source_event_id == event_id), None)

    @property
    def files_due(self) -> tuple[PaperIntake, ...]:
        return tuple(i for i in self.intakes if i.state == "due")

    @property
    def references_due(self) -> tuple[PaperIntake, ...]:
        return tuple(i for i in self.intakes if i.state == "reference_due")

    @property
    def review_pending(self) -> bool:
        return any(r.pending for r in self.requests)

    @property
    def aborted(self) -> bool:
        return any(i.state == "aborted" for i in self.intakes)

    def intake(self, intake_id: str) -> PaperIntake | None:
        return next((i for i in self.intakes if i.intake_id == intake_id), None)


def _fact(event: Mapping[str, Any]):
    return PAYLOADS[event["event_type"]].model_validate(
        {"event_type": event["event_type"], **event["payload"]}
    )


def validate_paper_answer(request: PaperReviewRequest, response: ReviewAnswer) -> None:
    if (
        response.request_id != request.request.request_id
        or response.action not in request.request.legal_actions
    ):
        raise ValueError("paper review changed its exact request or legal action")
    if response.action == "amend":
        if response.field not in request.request.amendable or not isinstance(
            response.value, str
        ):
            raise ValueError("a paper amendment names one declared field and a text")
        if response.field == "paper_file":
            _absolute(response.value)
        elif not response.value.strip():
            raise ValueError("a paper reference amendment names the paper")


def _refused(code: str, identity: str) -> ValueError:
    return ValueError(f"paper review {code}: {identity}")


class _Fold:
    """The run_papers fold: one method per fact, each checking its own lineage."""

    def __init__(self) -> None:
        self.intakes: dict[str, PaperIntake] = {}
        self.retained: dict[str, PaperDocumentRetained] = {}
        # Each intake's files retained since its last refusal or answer: the files
        # its next PaperRecorded must name, exactly and in order.
        self.open_files: dict[str, list[str]] = {}
        self.read: dict[str, PaperDocumentRead] = {}
        self.papers: list[PaperRecorded] = []
        self.refusals: dict[str, PaperIntakeRefused] = {}
        self.requests: list[PaperReviewRequest] = []
        self.receipts: list[PaperReviewReceipt] = []
        self.searches: list[PaperSearchView] = []
        self.retrievals: list[PaperRetrievalView] = []
        self.events: dict[str, Mapping[str, Any]] = {}
        # The recorded plan's cited papers, by subject key; None before the plan,
        # when a search names the key its plan will declare.
        self.cited: frozenset[str] | None = None

    def settle(self, intake_id: str, **changes) -> None:
        self.intakes[intake_id] = self.intakes[intake_id].model_copy(update=changes)

    def open_intake(self, intake_id: str) -> None:
        current = self.intakes.get(intake_id)
        if current is not None and current.state not in {"due", "reference_due"}:
            raise ValueError("a paper intake was settled already")

    def registered(self, fact: PaperFileRegistered, event) -> None:
        if fact.intake_id in self.intakes:
            raise ValueError("paper intake registered twice")
        self.intakes[fact.intake_id] = PaperIntake(
            intake_id=fact.intake_id,
            origin="file",
            state="due",
            source_event_id=event["event_id"],
            paper_file=fact.paper_file,
        )

    def file_retained(self, fact: PaperDocumentRetained, event) -> None:
        if not _holds(event, fact.sha256):
            raise ValueError("a retained paper document names its own bytes")
        self.open_intake(fact.intake_id)
        self.retained[event["event_id"]] = fact
        self.open_files.setdefault(fact.intake_id, []).append(event["event_id"])

    def document_read(self, fact: PaperDocumentRead, event) -> None:
        source = self.retained.get(fact.retained_event_id)
        if (
            source is None
            or source.intake_id != fact.intake_id
            or not _holds(event, fact.text_sha256)
        ):
            raise ValueError("a read paper document names its file and its text")
        self.read[event["event_id"]] = fact

    def recorded(self, fact: PaperRecorded, event) -> None:
        self.open_intake(fact.intake_id)
        named = [file.retained_event_id for file in fact.files]
        if named != self.open_files.pop(fact.intake_id, []) or any(
            self.retained[file.retained_event_id].sha256 != file.sha256
            for file in fact.files
        ):
            raise ValueError("a recorded paper names other files than its intake kept")
        for document in fact.documents:
            text = self.read.get(document.read_event_id)
            # Each document is read from one of this recording's own files, never
            # from an earlier refused attempt's (r2).
            if (
                text is None
                or text.intake_id != fact.intake_id
                or text.retained_event_id not in named
                or (text.document_id, text.text_sha256)
                != (document.document_id, document.text_sha256)
                or self.retained[text.retained_event_id].sha256 != document.sha256
            ):
                raise ValueError("a recorded paper names a document it did not read")
        if fact.paper_id != paper_id([file.sha256 for file in fact.files]):
            raise ValueError("paper_id is not derived from its files' digests")
        self.papers.append(fact)
        self.intakes[fact.intake_id] = PaperIntake(
            intake_id=fact.intake_id,
            origin=fact.origin,
            state="recorded",
            source_event_id=event["event_id"],
            paper_file=fact.paper_file,
            reference=fact.reference,
            paper_event_id=event["event_id"],
        )

    def refused(self, fact: PaperIntakeRefused, event) -> None:
        self.open_intake(fact.intake_id)
        self.open_files.pop(fact.intake_id, None)
        self.refusals[event["event_id"]] = fact
        self.intakes[fact.intake_id] = PaperIntake(
            intake_id=fact.intake_id,
            origin=fact.origin,
            state="review_pending",
            source_event_id=event["event_id"],
            paper_file=fact.paper_file,
            reference=fact.reference,
            refusal=fact,
        )

    def asked(self, fact: PaperReviewRequested, event) -> None:
        refusal = self.refusals.get(fact.request.failure_event_id)
        if (
            refusal is None
            or refusal.intake_id != fact.intake_id
            or event["parent_ids"] != [fact.request.failure_event_id]
            or fact.request.stage != PAPER_REVIEW_STAGE
            or fact.request.legal_actions != REVIEW_ACTIONS
            or fact.request.amendable != AMENDABLE[refusal.origin]
        ):
            raise ValueError("paper review changed its refusal or scope")
        if any(r.pending and r.intake_id == fact.intake_id for r in self.requests):
            raise ValueError("paper review duplicated pending work")
        self.requests.append(
            PaperReviewRequest(
                request_event_id=event["event_id"],
                intake_id=fact.intake_id,
                origin=refusal.origin,
                refusal=refusal,
                request=fact.request,
                pending=True,
            )
        )

    def answered(self, fact: PaperReviewAnswered, event) -> None:
        from bio_neb.junction import evaluate_scoped_answer

        request, replay = evaluate_scoped_answer(
            self.requests,
            self.receipts,
            fact.request_event_id,
            fact.response,
            validate=validate_paper_answer,
            refusal=_refused,
        )
        if replay is not None or event["parent_ids"] != [fact.request_event_id]:
            raise ValueError("paper review repeated its answer or changed lineage")
        self.requests = [
            r.model_copy(update={"pending": False})
            if r.request_event_id == fact.request_event_id
            else r
            for r in self.requests
        ]
        self.receipts.append(
            PaperReviewReceipt(
                request_event_id=fact.request_event_id,
                response_event_id=event["event_id"],
                response=fact.response,
            )
        )
        self.settle(
            request.intake_id,
            **_answered_state(
                self.intakes[request.intake_id], fact.response, event["event_id"]
            ),
        )

    def searched(self, fact: PaperSearchRecorded, event) -> None:
        if fact.result.status == "answered" and fact.result.rows:
            raise ValueError("a paper search's own answer lists no rows")
        if self.cited is not None and fact.tool_input.subject_key not in self.cited:
            # After the plan, a search serves one of its cited papers (membership).
            raise ValueError("a paper search names no cited paper of the plan")
        self.searches.append(
            PaperSearchView(**fact.model_dump(), source_event_id=event["event_id"])
        )

    def retrieved(self, fact: PaperRetrievalRecorded, event) -> None:
        """A retrieval names a row of the search page it cites, and the paper or the
        Review its own intake recorded (ADR 0049 membership)."""
        search = next(
            (
                s
                for s in self.searches
                if s.source_event_id == fact.tool_input.search_event_id
            ),
            None,
        )
        none = fact.tool_input.candidate_id == NO_PAPER
        row = (
            None if search is None else search.result.row(fact.tool_input.candidate_id)
        )
        outcome = (
            fact.paper_event_id
            if fact.status == "recorded"
            else fact.review_request_event_id
        )
        cited = self.events.get(outcome) if outcome else None
        expected = (
            "PaperRecorded" if fact.status == "recorded" else "PaperReviewRequested"
        )
        if (
            search is None
            or (row is None) != none
            or (none and (fact.status, fact.code) != ("refused", "paper_not_found"))
            or (fact.tool_input.reference is None) != none
            or (fact.paper_event_id is None) == (fact.status == "recorded")
            or (fact.review_request_event_id is None) == (fact.status == "refused")
            or (fact.code is None) == (fact.status == "refused")
            or cited is None
            or cited["event_type"] != expected
            or cited["payload"]["intake_id"] != fact.intake_id
        ):
            raise ValueError("a paper retrieval changed its search row or outcome")
        self.retrievals.append(
            PaperRetrievalView(**fact.model_dump(), source_event_id=event["event_id"])
        )

    def view(self) -> RunPapers:
        return RunPapers(
            intakes=tuple(self.intakes.values()),
            papers=tuple(self.papers),
            requests=tuple(self.requests),
            receipts=tuple(self.receipts),
            searches=tuple(self.searches),
            retrievals=tuple(self.retrievals),
        )


def _holds(event: Mapping[str, Any], digest: str) -> bool:
    """The event retains exactly one artifact, these bytes."""
    artifacts = event["artifact_references"]
    return len(artifacts) == 1 and artifacts[0]["artifact_id"] == f"sha256:{digest}"


def _answered_state(
    intake: PaperIntake, response: ReviewAnswer, event_id: str
) -> dict[str, Any]:
    """What one answer makes of its intake (owner, #420 P1)."""
    if response.action == "continue":
        return {"state": "dropped", "source_event_id": event_id}
    if response.action == "abort":
        return {"state": "aborted", "source_event_id": event_id}
    if response.action == "amend" and response.field == "paper_file":
        return {
            "origin": "file",
            "state": "due",
            "paper_file": response.value,
            "source_event_id": event_id,
        }
    if response.action == "amend" or intake.origin == "citation":
        # A new reference, or a retried citation, is typed again by planning from
        # the researcher's words (ADR 0049 re-entry).
        return {
            "state": "reference_due",
            "answer_event_id": event_id,
            "source_event_id": event_id,
        }
    return {"state": "due", "source_event_id": event_id}


_STEPS = {
    "PaperFileRegistered": _Fold.registered,
    "PaperDocumentRetained": _Fold.file_retained,
    "PaperDocumentRead": _Fold.document_read,
    "PaperRecorded": _Fold.recorded,
    "PaperIntakeRefused": _Fold.refused,
    "PaperReviewRequested": _Fold.asked,
    "PaperReviewAnswered": _Fold.answered,
    "PaperSearchRecorded": _Fold.searched,
    "PaperRetrievalRecorded": _Fold.retrieved,
}


def project_run_papers(events: Sequence[Mapping[str, Any]]) -> RunPapers:
    """Each paper intake's current state, its documents, Reviews and answers."""
    fold = _Fold()
    for event in events:
        if event["event_type"] == "ProposalRequirementsIdentified":
            fold.cited = frozenset(
                item["subject_key"]
                for item in event["payload"]["plan"]["requirements"]
                if item["kind"] == "cited_paper"
            )
        if step := _STEPS.get(event["event_type"]):
            step(fold, _fact(event), event)
            fold.events[event["event_id"]] = event
    return fold.view()


def new_intake_id() -> str:
    return str(uuid.uuid4())


__all__ = [
    "AMENDABLE",
    "PAPER_REVIEW_STAGE",
    "NO_PAPER",
    "PAPER_TEXT_MEDIA_TYPE",
    "PAYLOADS",
    "REVIEW_ACTIONS",
    "PaperDocumentRead",
    "PaperDocumentRetained",
    "PaperFileRegistered",
    "PaperIds",
    "PaperIntake",
    "PaperIntakeRefused",
    "PaperRecorded",
    "PaperReviewAnswered",
    "PaperReviewReceipt",
    "PaperReviewRequest",
    "PaperReviewRequested",
    "PaperRetrievalInput",
    "PaperRetrievalRecorded",
    "PaperSearchInput",
    "PaperSearchRecorded",
    "PaperSearchResult",
    "PaperSearchRow",
    "RecordedDocument",
    "RecordedFile",
    "RunPapers",
    "UnreadFile",
    "new_intake_id",
    "paper_id",
    "project_run_papers",
    "validate_paper_answer",
]
