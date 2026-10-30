"""Paper intake: a Run paper's files retained once, read once and recorded (#420).

The coordinator takes each supplied paper file before planning, as it inspects the
FASTQs (`take_supplied_papers`). The planning Agent's `retrieve_paper` Tool records a
downloaded paper through the same `record_paper`. A file that cannot be read, or a
citation that cannot be retrieved, is refused and its Review asked (`refuse_paper`);
`answer_paper_review` records the researcher's one answer.
"""

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bio_neb.failure import Ask, ReviewAnswer
from bio_neb.paper_intake.documents import SourceFile, document_texts
from bio_neb.paper_intake.journey import (
    AMENDABLE,
    PAPER_REVIEW_STAGE,
    PAPER_TEXT_MEDIA_TYPE,
    REVIEW_ACTIONS,
    Origin,
    PaperDocumentRead,
    PaperDocumentRetained,
    PaperFileRegistered,
    PaperIds,
    PaperIntakeRefused,
    PaperRecorded,
    PaperReviewAnswered,
    PaperReviewRequested,
    RecordedDocument,
    RecordedFile,
    RefusalCode,
    UnreadFile,
    new_intake_id,
    paper_id,
    validate_paper_answer,
)

# A fact and, for a retained one, its bytes and media type.
Record = Callable[[BaseModel], dict[str, Any]]
Retain = Callable[[BaseModel, bytes, str], dict[str, Any]]


class _Chain:
    """Appends to the Journey, each fact a child of the one before it. The first
    rests on the Run's head and on the intake's own last fact, so later work keeps
    both the FASTQs' inspection and the paper in its lineage."""

    def __init__(self, journey, source_event_id: str) -> None:
        self._journey = journey
        head = journey.head_event_id
        self._parents = list(dict.fromkeys((head, source_event_id)))

    def _payload(self, fact: BaseModel) -> dict[str, Any]:
        return fact.model_dump(mode="json", exclude={"event_type"})

    def record(self, fact: BaseModel) -> dict[str, Any]:
        event = self._journey.record(
            fact.event_type, self._payload(fact), self._parents
        )
        self._parents = [event["event_id"]]
        return event

    def retain(self, fact: BaseModel, content: bytes, media_type: str) -> dict:
        event = self._journey.record_with_artifact(
            fact.event_type, self._payload(fact), content, media_type, self._parents
        )
        self._parents = [event["event_id"]]
        return event


def register_paper_files(
    journey, paper_files: Sequence[str], parent_event_id: str
) -> None:
    """One ``PaperFileRegistered`` per supplied file, in order, by path."""
    for path in paper_files:
        fact = PaperFileRegistered(intake_id=new_intake_id(), paper_file=path)
        journey.record(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            [parent_event_id],
        )


def record_paper(
    record: Record,
    retain: Retain,
    *,
    intake_id: str,
    origin: Origin,
    files: Sequence[SourceFile],
    locations: Sequence[str],
    paper_file: str | None = None,
    reference: str | None = None,
    quote: str | None = None,
    ids: PaperIds | None = None,
    unfetched: Sequence[UnreadFile] = (),
    contact_event_ids: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Retain each file, then each readable document's text, then ``PaperRecorded``.

    None when no file holds a readable document: the caller refuses the paper. The
    files are retained even then, as what the intake was given."""
    retained = [
        retain(
            PaperDocumentRetained(
                intake_id=intake_id,
                name=file.name,
                role=file.role,
                media_type=file.media_type,
                location=location,
                sha256=_sha256(file.content),
                byte_count=len(file.content),
            ),
            file.content,
            file.media_type,
        )
        for file, location in zip(files, locations, strict=True)
    ]
    texts = document_texts(files)
    if not texts.documents:
        return None
    documents = []
    for document in texts.documents:
        content = document.text.encode()
        event = retain(
            PaperDocumentRead(
                intake_id=intake_id,
                retained_event_id=retained[document.source]["event_id"],
                document_id=document.document_id,
                name=document.name,
                text_sha256=_sha256(content),
                characters=len(document.text),
            ),
            content,
            PAPER_TEXT_MEDIA_TYPE,
        )
        documents.append(
            RecordedDocument(
                document_id=document.document_id,
                name=document.name,
                sha256=_sha256(files[document.source].content),
                text_sha256=_sha256(content),
                read_event_id=event["event_id"],
            )
        )
    digests = tuple(_sha256(file.content) for file in files)
    return record(
        PaperRecorded(
            intake_id=intake_id,
            paper_id=paper_id(digests),
            origin=origin,
            paper_file=paper_file,
            reference=reference,
            quote=quote,
            ids=ids,
            files=tuple(
                RecordedFile(retained_event_id=event["event_id"], sha256=digest)
                for event, digest in zip(retained, digests, strict=True)
            ),
            documents=tuple(documents),
            unread=tuple(
                UnreadFile(name=r.name, reason=r.reason)
                for r in texts.readings
                if r.text is None
            ),
            unfetched=tuple(unfetched),
            contact_event_ids=tuple(contact_event_ids),
        )
    )


def refuse_paper(
    record: Record,
    *,
    intake_id: str,
    origin: Origin,
    code: RefusalCode,
    detail: str,
    paper_file: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """Record the refusal and ask its Review (ADR 0031; owner, #420 P1)."""
    refused = record(
        PaperIntakeRefused(
            intake_id=intake_id,
            origin=origin,
            code=code,
            detail=detail[:2000],
            paper_file=paper_file,
            reference=reference,
        )
    )
    return record(
        PaperReviewRequested(
            intake_id=intake_id,
            request=Ask(
                stage=PAPER_REVIEW_STAGE,
                failure_event_id=refused["event_id"],
                request_id=str(uuid.uuid4()),
                reason=f"{code}: {detail}"[:2000],
                legal_actions=REVIEW_ACTIONS,
                amendable=AMENDABLE[origin],
            ),
        )
    )


def _sha256(content: bytes) -> str:
    from bio_neb.content import sha256_bytes

    return sha256_bytes(content)


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def take_supplied_papers(journey) -> tuple[dict[str, Any], ...]:
    """Read each supplied paper file that is due: record it, or refuse it."""
    papers = journey.checked_view("run_papers")
    outcomes = []
    for intake in papers.files_due:
        chain = _Chain(journey, intake.source_event_id)
        path = Path(intake.paper_file)
        try:
            content = path.read_bytes()
        except OSError as error:
            outcomes.append(
                refuse_paper(
                    chain.record,
                    intake_id=intake.intake_id,
                    origin="file",
                    code="paper_file_missing",
                    detail=f"{path}: {error.strerror or type(error).__name__}",
                    paper_file=intake.paper_file,
                )
            )
            continue
        file = SourceFile(path.name, path.name, "supplied", _media_type(path), content)
        recorded = record_paper(
            chain.record,
            chain.retain,
            intake_id=intake.intake_id,
            origin="file",
            files=(file,),
            locations=(intake.paper_file,),
            paper_file=intake.paper_file,
        )
        outcomes.append(
            recorded
            if recorded is not None
            else refuse_paper(
                chain.record,
                intake_id=intake.intake_id,
                origin="file",
                code="paper_file_unreadable",
                detail=f"{path.name}: no readable text (formats: JATS or XML, HTML, "
                "text, .csv, .docx, .xlsx, PDF, or a zip of these)",
                paper_file=intake.paper_file,
            )
        )
    return tuple(outcomes)


def answer_paper_review(
    journey, request_event_id: str, response: ReviewAnswer
) -> dict[str, Any]:
    """Record the one answer to a pending paper Review; replay is inert."""
    from bio_neb.junction import evaluate_scoped_answer

    state = journey.checked_view("run_papers")
    _, replay = evaluate_scoped_answer(
        state.requests,
        state.receipts,
        request_event_id,
        response,
        validate=validate_paper_answer,
        refusal=lambda code, identity: ValueError(f"paper review {code}: {identity}"),
    )
    if replay is not None:
        return replay
    answered = PaperReviewAnswered(request_event_id=request_event_id, response=response)
    event = journey.record(
        answered.event_type,
        answered.model_dump(mode="json", exclude={"event_type"}),
        [request_event_id],
    )
    if response.action == "abort":
        # The researcher's abort ends the Run.
        from bio_neb.application.run_failure import RunFailed, record_run_fact

        record_run_fact(
            journey,
            RunFailed(
                run_id=journey.journey_id,
                code="rejected_by_user",
                source_event_id=event["event_id"],
            ),
            event["event_id"],
        )
    return event


__all__ = [
    "answer_paper_review",
    "record_paper",
    "refuse_paper",
    "register_paper_files",
    "take_supplied_papers",
]
