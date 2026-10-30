"""Contact-free public Grounded-report publication and reopen, and the public
request and reopen of its optional Agent explanation (#109)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.queries import (
    CompletionRead,
    read_completion,
    read_grounded_report,
)
from bio_neb.content import sha256_bytes
from bio_neb.journey import Journey, JourneyFile, JourneyValidationError
from bio_neb.recording import read_evidence
from bio_neb.reporting.explanation import (
    READABLE_REPORT_FILE,
    PublishedExplanation,
    ReadableReportWritten,
    ReportExplanationRequested,
    ReportExplanationView,
    project_report_explanations,
    report_to_explain,
)
from bio_neb.reporting.finalization import (
    ReportBasis,
)
from bio_neb.reporting.grounded_report import (
    REPORT_MEDIA_TYPE,
    GroundedReport,
    GroundedReportProjection,
    GroundedReportPublished,
    derive_grounded_report,
)


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublishGroundedReportCommand(_Command):
    journey_file: JourneyFile
    report_basis_digest: StrictStr = Field(pattern=r"^report-basis:[0-9a-f]{64}$")
    head_event_id: StrictStr


class ReopenGroundedReportCommand(_Command):
    journey_file: JourneyFile
    report_digest: StrictStr = Field(pattern=r"^grounded-report:[0-9a-f]{64}$")


class GroundedReportApplicationRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


def _open_verified(path: str) -> tuple[Journey, CompletionRead]:
    try:
        journey = Journey.open(Path(path))
        return journey, read_completion(journey)
    except (JourneyValidationError, ValueError) as error:
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_unverifiable", "journey_file"
        ) from error


def _approved_basis(read: CompletionRead) -> ReportBasis:
    finalization = read.finalization
    if finalization.status == "not_started":
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_absent", "report_basis_digest"
        )
    if finalization.status == "stopped":
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_stopped", "report_basis_digest"
        )
    if finalization.status == "returned_to_investigation":
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_stale", "report_basis_digest"
        )
    if finalization.status == "review_pending":
        code = (
            "grounded_report_basis_corrected_unapproved"
            if len(finalization.bases) > 1
            else "grounded_report_basis_unapproved"
        )
        raise GroundedReportApplicationRefused(code, "report_basis_digest")
    if finalization.report_basis is None:
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_absent", "report_basis_digest"
        )
    return finalization.report_basis


def _project_verified(
    journey: Journey, *, report_digest: str | None = None
) -> GroundedReportProjection:
    try:
        return read_grounded_report(journey, report_digest=report_digest)
    except ValueError as error:
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_unverifiable", "journey_file"
        ) from error


def publish_grounded_report(
    command: PublishGroundedReportCommand,
) -> GroundedReportProjection:
    """Publish one exact approved Report basis as one immutable artifact."""

    trusted = PublishGroundedReportCommand.model_validate(command, strict=True)
    journey, read = _open_verified(trusted.journey_file)
    timeline = read.investigation.timeline
    retained = _project_verified(journey)
    if retained.status == "published":
        report = retained.report
        assert report is not None
        if (
            report.report_basis_digest == trusted.report_basis_digest
            and report.source_head_event_id == trusted.head_event_id
        ):
            return retained
    report_basis = _approved_basis(read)
    if report_basis.report_basis_digest != trusted.report_basis_digest:
        raise GroundedReportApplicationRefused(
            "grounded_report_basis_mismatched", "report_basis_digest"
        )
    if (
        read.approved_report is None
        or read.approved_report.approval_event_id != trusted.head_event_id
        or timeline[-1]["event_id"] != trusted.head_event_id
    ):
        raise GroundedReportApplicationRefused(
            "grounded_report_binding_stale", "head_event_id"
        )
    report = derive_grounded_report(
        report_basis,
        journey_id=journey.journey_id,
        source_head_event_id=trusted.head_event_id,
    )
    fact = GroundedReportPublished(report=report)
    current, current_read = _open_verified(trusted.journey_file)
    if current_read.investigation.timeline != timeline:
        raise GroundedReportApplicationRefused(
            "grounded_report_binding_stale", "journey_file"
        )
    current.record_with_artifact(
        fact.event_type,
        fact.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        report.canonical_bytes(),
        REPORT_MEDIA_TYPE,
        parent_ids=[trusted.head_event_id],
    )
    return _project_verified(current)


def reopen_grounded_report(
    command: ReopenGroundedReportCommand,
) -> GroundedReportProjection:
    """Strictly reopen one retained report without mutation or dependencies."""

    trusted = ReopenGroundedReportCommand.model_validate(command, strict=True)
    journey, _ = _open_verified(trusted.journey_file)
    return _reopened(journey, trusted.report_digest)


def _reopened(journey: Journey, report_digest: str) -> GroundedReportProjection:
    projection = _project_verified(journey, report_digest=report_digest)
    if projection.status != "published" or projection.report is None:
        raise GroundedReportApplicationRefused(
            "grounded_report_absent", "report_digest"
        )
    if projection.report.report_digest != report_digest:
        raise GroundedReportApplicationRefused(
            "grounded_report_digest_mismatched", "report_digest"
        )
    return projection


class ReadReadableReportCommand(_Command):
    journey_file: JourneyFile
    report_digest: StrictStr = Field(pattern=r"^grounded-report:[0-9a-f]{64}$")


# Why the explainer's words are not on the page yet, by the latest request's status.
_UNWRITTEN = {
    "requested": "the explanation is still being written",
    "answered": "the explanation is still being written",
    "review_pending": "the explanation waits for a researcher's Review",
    "stopped": "the explanation was stopped",
}


def read_readable_report(command: ReadReadableReportCommand) -> str:
    """The researcher's readable Markdown report of one published report, with
    its latest published explanation's words; contact-free, writing nothing."""
    trusted = ReadReadableReportCommand.model_validate(command, strict=True)
    return _readable_page(trusted.journey_file, trusted.report_digest)


def _readable_page(journey_file: str, report_digest: str) -> str:
    journey, _ = _open_verified(journey_file)
    report = _reopened(journey, report_digest).report
    requests = [
        item
        for item in read_report_explanations(journey_file, journey=journey).requests
        if item.request.report_digest == report_digest
    ]
    return _page(report, requests, _run_facts(journey, report))


def _run_facts(journey: Journey, report: GroundedReport):
    """What the page reads from the Run beside its report (#491): when the report
    was published, and each source's own answer, from the raw artifact the
    report's accounting names, verified against the source's sha256 (owner Q3, D1
    a); a source whose answer the report's accounting does not name reads None,
    and a named answer that does not read or verify refuses the page."""
    from bio_neb.reporting.readable_report import ReadableReportInconsistent, RunFacts

    published_at = next(
        (
            event["occurred_at"]
            for event in journey.events_view()
            if event["event_type"] == "GroundedReportPublished"
            and event["payload"]["report"]["report_digest"] == report.report_digest
        ),
        None,
    )
    references = {
        reference.artifact_id: reference
        for entry in report.accounting
        for reference in entry.value.artifact_references
    }

    def source_answer(source) -> bytes | None:
        reference = references.get(source.raw_artifact_id)
        if reference is None:
            return None
        fault = f"source {source.source_id}: its retained answer does not verify"
        try:
            body = journey.read_artifact(reference)
        except JourneyValidationError as error:
            raise ReadableReportInconsistent((fault,)) from error
        if sha256_bytes(body) != source.raw_response_sha256:
            raise ReadableReportInconsistent((fault,))
        return body

    return RunFacts(published_at=published_at, source_answer=source_answer)


def _page(report, requests, run) -> str:
    """The page of ``report`` with the state of its explanation ``requests``."""
    from bio_neb.reporting.readable_report import ExplanationState

    published = [item.explanation for item in requests if item.explanation is not None]
    if published:
        state = ExplanationState(published[-1].explanation)
    elif not requests:
        state = ExplanationState(None, "no explanation was requested")
    elif requests[-1].failure is not None:
        state = ExplanationState(
            None, f"the explanation failed ({requests[-1].failure.code})"
        )
    else:
        state = ExplanationState(None, _UNWRITTEN.get(requests[-1].status))
    return _rendered(report, state, run)


def _rendered(report, state, run) -> str:
    from bio_neb.reporting.readable_report import (
        ReadableReportInconsistent,
        render_readable_report,
    )

    try:
        return render_readable_report(report, state, run)
    except ReadableReportInconsistent as error:
        raise GroundedReportApplicationRefused(error.code, "report_digest") from error


def write_readable_report(journey: Journey, report: GroundedReport) -> bool:
    """Write the Run's own readable report once its explanation publishes.

    The page of ``report``, the published report the coordinator already holds,
    with its latest published explanation, is written into the Run root as
    ``report.md`` and its bytes' identity is kept in the Journey after the
    publication (owner, #476 G5/C13). False when no explanation waits for its page.
    """
    from bio_neb.reporting.readable_report import ExplanationState

    current = read_report_explanations(str(journey.path), journey=journey).current
    if (
        current is None
        or current.explanation is None
        or current.readable_report is not None
        or current.request.report_digest != report.report_digest
    ):
        return False
    explanation = current.explanation
    page = _rendered(
        report,
        ExplanationState(explanation.explanation),
        _run_facts(journey, report),
    ).encode("utf-8")
    (Path(journey.path).parent.parent / READABLE_REPORT_FILE).write_bytes(page)
    fact = ReadableReportWritten(
        report_digest=report.report_digest,
        explanation_id=explanation.explanation.explanation_id,
        sha256=sha256_bytes(page),
        byte_count=len(page),
    )
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    journey.record(fact.event_type, payload, [explanation.publication_event_id])
    return True


def written_readable_report(journey_file: str) -> str | None:
    """The Run's own report as its Run root holds it, or None before it is
    written; read without opening the Journey again, which keeps its sha256
    (``ReadableReportWritten``) for any reader to check."""
    page = Path(journey_file).parent.parent / READABLE_REPORT_FILE
    return page.read_text(encoding="utf-8") if page.is_file() else None


class ReportExplanationRefused(ValueError):
    """A request the published report cannot take, refused before any write."""

    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class RequestReportExplanationCommand(_Command):
    journey_file: JourneyFile
    report_digest: StrictStr = Field(pattern=r"^grounded-report:[0-9a-f]{64}$")
    requested_by: StrictStr = Field(min_length=1)
    supersedes_explanation_id: StrictStr | None = Field(
        default=None, pattern=r"^report-explanation:[0-9a-f]{64}$"
    )


class ReopenReportExplanationCommand(_Command):
    journey_file: JourneyFile
    explanation_id: StrictStr = Field(pattern=r"^report-explanation:[0-9a-f]{64}$")


def open_report_journey(path: str) -> Journey:
    """The Run's Journey for an explanation command; an unreadable one refuses."""
    try:
        return Journey.open(Path(path))
    except (JourneyValidationError, ValueError) as error:
        raise ReportExplanationRefused(
            "report_explanation_unverifiable", "journey_file"
        ) from error


def _open(
    path: str, journey: Journey | None = None
) -> tuple[Journey, ReportExplanationView]:
    if journey is None:
        journey = open_report_journey(path)
    try:
        if journey.path != Path(path):
            raise ValueError("Journey owner path disagrees with the command")
        return journey, project_report_explanations(
            read_evidence(journey), owner=journey
        )
    except (JourneyValidationError, ValueError) as error:
        raise ReportExplanationRefused(
            "report_explanation_unverifiable", "journey_file"
        ) from error


# Who requests the explanation every published report gets by default.
RUN_DEFAULT_REQUESTER = "run_default"


def read_report_explanations(
    journey_file: str, *, journey: Journey | None = None
) -> ReportExplanationView:
    """The Run's explanation requests and published explanations, contact-free.

    ``journey`` is the caller's owner of the same Run, when it already holds one.
    """
    return _open(journey_file, journey)[1]


def request_report_explanation(
    command: RequestReportExplanationCommand, *, journey: Journey | None = None
) -> ReportExplanationView:
    """Record one request to explain a published report; the coordinator runs it.

    A rewrite names the published explanation of the same report it supersedes;
    while a request is still open no other is taken. ``journey`` is the caller's
    owner of the same Run, when it already holds one.
    """
    trusted = RequestReportExplanationCommand.model_validate(command, strict=True)
    journey, view = _open(trusted.journey_file, journey)
    report = report_to_explain(journey, read_evidence(journey), trusted.report_digest)
    if report.report is None:
        raise ReportExplanationRefused(
            "report_explanation_report_absent", "report_digest"
        )
    current = view.current
    if current is not None and current.status in {
        "requested",
        "review_pending",
        "answered",
    }:
        raise ReportExplanationRefused("report_explanation_pending", "report_digest")
    explained = [
        item.explanation
        for item in view.explanations
        if item.explanation.report_digest == trusted.report_digest
    ]
    if trusted.supersedes_explanation_id is None and explained:
        raise ReportExplanationRefused(
            "report_explanation_exists", "supersedes_explanation_id"
        )
    if trusted.supersedes_explanation_id is not None and (
        trusted.supersedes_explanation_id
        not in {item.explanation_id for item in explained}
    ):
        raise ReportExplanationRefused(
            "report_explanation_prior_unknown", "supersedes_explanation_id"
        )
    fact = ReportExplanationRequested(
        report_digest=trusted.report_digest,
        report_basis_digest=report.report.report_basis_digest,
        publication_event_id=report.publication_event_id,
        supersedes_explanation_id=trusted.supersedes_explanation_id,
        requested_by=trusted.requested_by,
    )
    journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        [journey.head_event_id],
    )
    return read_report_explanations(trusted.journey_file, journey=journey)


def reopen_report_explanation(
    command: ReopenReportExplanationCommand,
) -> PublishedExplanation:
    """Strictly reopen one published explanation without mutation or contact."""
    trusted = ReopenReportExplanationCommand.model_validate(command, strict=True)
    _journey, view = _open(trusted.journey_file)
    for item in view.explanations:
        if item.explanation.explanation_id == trusted.explanation_id:
            return item
    raise ReportExplanationRefused("report_explanation_absent", "explanation_id")


__all__ = [
    "GroundedReportApplicationRefused",
    "PublishGroundedReportCommand",
    "ReopenGroundedReportCommand",
    "ReopenReportExplanationCommand",
    "ReportExplanationRefused",
    "RUN_DEFAULT_REQUESTER",
    "ReadReadableReportCommand",
    "RequestReportExplanationCommand",
    "publish_grounded_report",
    "open_report_journey",
    "read_readable_report",
    "read_report_explanations",
    "reopen_grounded_report",
    "reopen_report_explanation",
    "request_report_explanation",
]
