"""Shared setup for the paper reader's tests (ADR 0046): small papers as documents, the
values a model states, the answer code keeps from them, and a paper stored in a Journey
as a Run's intake stores it (#420), for the reading's product entry.

Executable tests may not import executable tests; both reading test modules and the
targets support read these.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bio_neb.journey import Journey
from bio_neb.paper_intake.documents import normalize_text
from bio_neb.paper_intake.journey import (
    PAPER_TEXT_MEDIA_TYPE,
    PaperDocumentRead,
    PaperDocumentRetained,
    PaperIds,
    PaperRecorded,
    RecordedDocument,
    RecordedFile,
    paper_id,
)
from bio_neb.paper_reading.answer import (
    finding_outcome,
    judge_all,
    kept_finding,
    paper_answer,
    stated_reference,
)
from bio_neb.paper_reading.contracts import (
    FindingPointer,
    PaperAnswer,
    PaperDocument,
    ReferencePointer,
)
from bio_neb.paper_reading.reader import Paper


def document(document_id: str, text: str, kind: str = "main_text") -> PaperDocument:
    return PaperDocument(
        document_id=document_id,
        kind=kind,
        url=f"https://example.org/{document_id}",
        text=normalize_text(text),
    )


def value(slot: str, written: str, span: str, **more) -> dict:
    """One stated value as the model sends it: its slot, its written text and its span."""

    return {"slot": slot, "written": written, "span": span, **more}


def finding(*values: dict, reason: str = "The paper reports it.", **more) -> dict:
    return {"values": list(values), "reason": reason, **more}


@dataclass(frozen=True)
class Stated:
    """What one model reading states: its findings by span and written value, and the
    reference it names, as the reading's output Tools take them."""

    findings: tuple[FindingPointer, ...]
    reference: ReferencePointer | None = None
    not_answered: str | None = None


def reading_of(
    *findings: dict,
    reference: dict | None = None,
    not_answered: str | None = "Testing findings only.",
) -> Stated:
    return Stated(
        findings=tuple(FindingPointer.model_validate(f) for f in findings),
        reference=None
        if reference is None
        else ReferencePointer.model_validate(reference),
        not_answered=not_answered,
    )


def kept_answer(question: str, reading: Stated, paper: Paper) -> PaperAnswer:
    """The answer code keeps from one reading, with no repair, as a reading item and its
    answer keep it: each finding decided on its own (``judge_all``), the accepted ones
    kept, the others recorded with their reasons, and the reference where its span
    writes it."""

    judged = judge_all(reading.findings, paper)
    return paper_answer(
        question,
        paper,
        stated_reference(reading.reference, paper),
        [kept_finding(j) for j in judged if j.outcome == "accepted"],
        reading.not_answered,
        [finding_outcome(j) for j in judged if j.outcome == "unresolved"],
        [finding_outcome(j) for j in judged if j.outcome == "refused"],
    )


def store_paper(
    journey: Journey, documents: Sequence[PaperDocument], ids: PaperIds | None = None
) -> str:
    """Record ``documents`` as one Run paper, as the intake records one (#420): each text
    retained as its file (``PaperDocumentRetained``) and as its read text
    (``PaperDocumentRead``), then ``PaperRecorded``. Synthetic: the files are the texts
    themselves, so the reader reads exactly these texts. The ``PaperRecorded`` id."""

    intake_id = str(uuid.uuid4())
    parents = [journey.head_event_id] if journey.head_event_id else []

    def retain(fact, content: bytes) -> dict[str, Any]:
        event = journey.record_with_artifact(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            content,
            PAPER_TEXT_MEDIA_TYPE,
            parents,
        )
        parents[:] = [event["event_id"]]
        return event

    files, recorded = [], []
    for document in documents:
        content = document.text.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        name = document.title or f"{document.document_id.removeprefix('supp:')}.txt"
        retained = retain(
            PaperDocumentRetained(
                intake_id=intake_id,
                name=name,
                role="article" if document.document_id == "main" else "supplement",
                media_type=PAPER_TEXT_MEDIA_TYPE,
                location=document.url,
                sha256=digest,
                byte_count=len(content),
            ),
            content,
        )
        read = retain(
            PaperDocumentRead(
                intake_id=intake_id,
                retained_event_id=retained["event_id"],
                document_id=document.document_id,
                name=name,
                text_sha256=digest,
                characters=len(document.text),
            ),
            content,
        )
        files.append(
            RecordedFile(retained_event_id=retained["event_id"], sha256=digest)
        )
        recorded.append(
            RecordedDocument(
                document_id=document.document_id,
                name=name,
                sha256=digest,
                text_sha256=digest,
                read_event_id=read["event_id"],
            )
        )
    fact = PaperRecorded(
        intake_id=intake_id,
        paper_id=paper_id([file.sha256 for file in files]),
        origin="citation" if ids is not None else "file",
        paper_file=None if ids is not None else "/papers/supplied.txt",
        ids=ids,
        files=tuple(files),
        documents=tuple(recorded),
    )
    return journey.record(
        fact.event_type, fact.model_dump(mode="json", exclude={"event_type"}), parents
    )["event_id"]


def paper_ids(paper: Paper) -> PaperIds | None:
    """The ids a cited paper's source gave, copied from the paper's identity."""

    identity = paper.identity
    if identity is None:
        return None
    return PaperIds(
        doi=identity.doi,
        pmid=identity.pmid,
        pmcid=identity.pmcid,
        title=identity.title,
        journal=identity.journal,
        year=identity.year,
        open_access=identity.open_access,
    )


def stored_journey(root: Path, paper: Paper) -> tuple[Journey, str]:
    """A fresh Journey holding ``paper`` as its one Run paper, and its ``PaperRecorded``."""

    journey = Journey.create(root.absolute())
    return journey, store_paper(
        journey, list(paper.documents.values()), paper_ids(paper)
    )
