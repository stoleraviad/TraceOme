"""Contact-free public Finalization compilation, inspection, and review."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, field_validator

from bio_neb.agentic_journey.continuation_policy import FinalizationReviewPolicy
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.queries import (
    CompletionRead,
    read_completion,
)
from bio_neb.investigation.investigation import (
    InvestigationReviewResponded,
    compile_investigation_collection_revision,
)
from bio_neb.journey import Journey, JourneyFile, canonical_json
from bio_neb.recording import review_head
from bio_neb.reporting.finalization import (
    ApproveFinalizationResponse,
    CorrectFinalizationResponse,
    FinalizationCorrectionAssertion,
    FinalizationProjection,
    FinalizationReturnToInvestigation,
    FinalizationReviewReceipt,
    FinalizationReviewRecorded,
    FinalizationReviewRequested,
    FinalizationStop,
    FinalizationStopAssertion,
    ReportBasis,
    StopFinalizationResponse,
    compile_finalization_basis,
    derive_finalization_correction,
)


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def strict_strings(cls, value: Any) -> Any:
        return value


def _absolute(value: str, field: str) -> str:
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise FinalizationApplicationRefused("finalization_path_invalid", field)
    return value


class CompileFinalizationCommand(_Command):
    journey_file: JourneyFile


class InspectFinalizationCommand(_Command):
    journey_file: JourneyFile
    basis_digest: StrictStr


class ApproveFinalizationCommand(_Command):
    journey_file: JourneyFile
    request_event_id: StrictStr
    basis_digest: StrictStr
    head_event_id: StrictStr


class CorrectFinalizationCommand(_Command):
    journey_file: JourneyFile
    request_event_id: StrictStr
    basis_digest: StrictStr
    head_event_id: StrictStr
    correction_file: StrictStr


class StopFinalizationCommand(_Command):
    journey_file: JourneyFile
    request_event_id: StrictStr
    basis_digest: StrictStr
    head_event_id: StrictStr
    reason_file: StrictStr


class FinalizationApplicationRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


def _owned_journey(path: str, journey: Journey | None) -> Journey:
    if journey is None:
        return Journey.open(Path(path))
    if journey.path != Path(path) or not journey.is_current():
        raise FinalizationApplicationRefused(
            "finalization_journey_context_stale", "journey_file"
        )
    return journey


def _verified_read(journey: Journey) -> CompletionRead:
    try:
        return read_completion(journey)
    except ValueError as error:
        raise FinalizationApplicationRefused(
            "finalization_basis_invalid", "journey_file"
        ) from error


def compile_finalization(
    command: CompileFinalizationCommand, *, journey: Journey | None = None
) -> FinalizationProjection:
    trusted = CompileFinalizationCommand.model_validate(command, strict=True)
    journey = _owned_journey(trusted.journey_file, journey)
    completion = _verified_read(journey)
    read = completion.investigation
    investigation = read.investigation
    retained = completion.finalization
    candidate = completion.candidate
    if retained.status != "not_started" and (
        retained.status not in {"approved", "returned_to_investigation"}
        or candidate is None
    ):
        return retained
    if candidate is None:
        if (
            investigation.proposals
            and investigation.proposals[-1].unresolved_request_member_ids
        ):
            raise FinalizationApplicationRefused(
                "finalization_request_identity_unresolved", "request_member_ids"
            )
        raise FinalizationApplicationRefused(
            "finalization_policy_ineligible", *read.work_policy.completion_blockers
        )
    if (
        retained.bases
        and candidate.collection is not None
        and (
            candidate.collection.evidence.collection_state_id
            == retained.bases[-1].evidence_state_id
        )
    ):
        raise FinalizationApplicationRefused(
            "finalization_evidence_unchanged", "evidence_state_id"
        )
    source_head = candidate.source_head_event["event_id"]
    try:
        basis = compile_finalization_basis(candidate)
    except ValueError as error:
        raise FinalizationApplicationRefused(
            "finalization_basis_invalid", "journey_file"
        ) from error
    if retained.bases:
        basis = basis.model_copy(
            update={"supersedes_basis_digest": retained.bases[-1].basis_digest}
        )
    fact = FinalizationReviewRequested(
        review_policy=FinalizationReviewPolicy(),
        source_head_event_id=source_head,
        basis=basis,
    )
    current = journey
    if not current.is_current():
        raise FinalizationApplicationRefused(
            "finalization_evidence_head_changed", "journey_file"
        )
    current.record(
        fact.event_type,
        fact.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        [source_head],
    )
    return _verified_read(current).finalization


def inspect_finalization(
    command: InspectFinalizationCommand, *, journey: Journey | None = None
) -> FinalizationProjection:
    trusted = InspectFinalizationCommand.model_validate(command, strict=True)
    projection = _verified_read(
        _owned_journey(trusted.journey_file, journey)
    ).finalization
    if trusted.basis_digest not in {item.basis_digest for item in projection.bases}:
        raise FinalizationApplicationRefused(
            "finalization_basis_unknown", "basis_digest"
        )
    return projection


def approve_finalization(
    command: ApproveFinalizationCommand,
    *,
    journey: Journey | None = None,
) -> FinalizationReviewRecorded:
    trusted = ApproveFinalizationCommand.model_validate(command, strict=True)
    journey = _owned_journey(trusted.journey_file, journey)
    completion = _verified_read(journey)
    read = completion.investigation
    timeline = read.timeline
    projection = completion.finalization
    existing = _review_fact_for_request(timeline, trusted.request_event_id)
    if existing is not None:
        receipt = existing.receipt
        if (
            receipt.response.action == "approve"
            and receipt.basis_digest == trusted.basis_digest
            and receipt.response_head_event_id == trusted.head_event_id
        ):
            return existing
        raise FinalizationApplicationRefused(
            "finalization_review_response_conflict", "request_event_id"
        )
    pending = projection.pending_request
    if pending is None or pending.request_event_id != trusted.request_event_id:
        raise FinalizationApplicationRefused(
            "finalization_review_response_stale", "request_event_id"
        )
    if (
        not timeline
        or review_head(timeline)["event_id"] != trusted.head_event_id
        or trusted.head_event_id != pending.request_event_id
        or pending.basis.basis_digest != trusted.basis_digest
    ):
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "head_event_id", "basis_digest"
        )
    receipt = FinalizationReviewReceipt(
        request_event_id=trusted.request_event_id,
        response_head_event_id=trusted.head_event_id,
        basis_digest=trusted.basis_digest,
        source_head_event_id=pending.source_head_event_id,
        response=ApproveFinalizationResponse(),
    )
    report_basis = ReportBasis(finalization_basis=pending.basis, approval=receipt)
    fact = FinalizationReviewRecorded(
        review_policy=FinalizationReviewPolicy(),
        receipt=receipt,
        report_basis=report_basis,
    )
    current = journey
    if not current.is_current():
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "journey_file"
        )
    current.record(
        fact.event_type,
        fact.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        [trusted.head_event_id],
    )
    return fact


def _load_correction(path: str) -> FinalizationCorrectionAssertion:
    _absolute(path, "correction_file")
    try:
        return FinalizationCorrectionAssertion.model_validate_json(
            Path(path).read_text(encoding="utf-8"), strict=True
        )
    except (OSError, UnicodeError, ValidationError) as error:
        raise FinalizationApplicationRefused(
            "finalization_correction_invalid", "correction_file"
        ) from error


def _review_fact_for_request(
    timeline: list[dict[str, Any]], request_event_id: str
) -> FinalizationReviewRecorded | None:
    return next(
        (
            fact
            for event in timeline
            if (
                fact := (
                    FinalizationReviewRecorded.model_validate_json(
                        canonical_json(
                            {
                                "event_type": event["event_type"],
                                **event.get("payload", {}),
                            }
                        ),
                        strict=True,
                    )
                    if event.get("event_type") == "FinalizationReviewRecorded"
                    else None
                )
            )
            is not None
            and fact.receipt.request_event_id == request_event_id
        ),
        None,
    )


def correct_finalization(
    command: CorrectFinalizationCommand, *, journey: Journey | None = None
) -> FinalizationProjection:
    trusted = CorrectFinalizationCommand.model_validate(command, strict=True)
    correction = _load_correction(trusted.correction_file)
    journey = _owned_journey(trusted.journey_file, journey)
    completion = _verified_read(journey)
    read = completion.investigation
    timeline = read.timeline
    projection = completion.finalization
    existing = _review_fact_for_request(timeline, trusted.request_event_id)
    if existing is not None:
        response = existing.receipt.response
        if (
            response.action == "correct"
            and response.correction.correction_id == correction.correction_id
            and existing.receipt.basis_digest == trusted.basis_digest
            and existing.receipt.response_head_event_id == trusted.head_event_id
        ):
            return projection
        raise FinalizationApplicationRefused(
            "finalization_review_response_conflict", "request_event_id"
        )
    pending = projection.pending_request
    if pending is None or pending.request_event_id != trusted.request_event_id:
        raise FinalizationApplicationRefused(
            "finalization_review_response_stale", "request_event_id"
        )
    if (
        not timeline
        or review_head(timeline)["event_id"] != trusted.head_event_id
        or trusted.head_event_id != pending.request_event_id
        or pending.basis.basis_digest != trusted.basis_digest
    ):
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "head_event_id", "basis_digest"
        )
    response = CorrectFinalizationResponse(correction=correction)
    rework_command = None
    if correction.investigation_revision is not None:
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
        )

        investigation = read.investigation
        revision_response = correction.investigation_revision
        if not investigation.proposals or not investigation.review_requests:
            raise FinalizationApplicationRefused(
                "finalization_rework_stale", "investigation_revision"
            )
        proposal = investigation.proposals[-1]
        if (
            investigation.status not in {"approved", "review_pending"}
            or any(
                receipt.request_event_id
                == investigation.review_requests[-1].request_event_id
                and receipt.response.action != "approve"
                for receipt in investigation.review_receipts
            )
            or revision_response.proposal_id != proposal.proposal_id
            or pending.basis.work_items
            != tuple(sorted(proposal.work_items, key=lambda item: item.work_item_id))
        ):
            raise FinalizationApplicationRefused(
                "finalization_rework_stale", "investigation_revision"
            )
        rework_command = RespondInvestigationReviewCommand(
            journey_file=trusted.journey_file,
            request_event_id=investigation.review_requests[-1].request_event_id,
            response=revision_response,
        )
        try:
            responded = InvestigationReviewResponded.create(
                request_event_id=rework_command.request_event_id,
                response=revision_response,
            )
            compile_investigation_collection_revision(
                proposal=proposal,
                response_id=responded.response_id,
                response=revision_response,
            )
            collection = read.evidence
            if investigation.status in {"failed", "stopped"} or (
                collection is not None and collection.failed
            ):
                raise ValueError("failed Investigation cannot be reworked")
        except ValueError as error:
            raise FinalizationApplicationRefused(
                "finalization_rework_invalid", "investigation_revision"
            ) from error
    receipt = FinalizationReviewReceipt(
        request_event_id=trusted.request_event_id,
        response_head_event_id=trusted.head_event_id,
        basis_digest=trusted.basis_digest,
        source_head_event_id=pending.source_head_event_id,
        response=response,
    )
    try:
        disposition = derive_finalization_correction(pending.basis, correction)
    except ValueError as error:
        raise FinalizationApplicationRefused(
            "finalization_correction_noop", "correction_file"
        ) from error
    if isinstance(disposition, FinalizationReturnToInvestigation):
        fact = FinalizationReviewRecorded(
            review_policy=FinalizationReviewPolicy(),
            receipt=receipt,
            return_to_investigation=disposition,
        )
    else:
        fact = FinalizationReviewRecorded(
            review_policy=FinalizationReviewPolicy(),
            receipt=receipt,
            superseding_basis=disposition,
        )
    current = journey
    if not current.is_current():
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "journey_file"
        )
    current.record(
        fact.event_type,
        fact.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        [trusted.head_event_id],
    )
    if rework_command is not None:
        from bio_neb.investigation.investigation_commands import (
            respond_investigation_review,
        )

        respond_investigation_review(rework_command)
        current.refresh()
    return _verified_read(current).finalization


def _load_stop(path: str) -> FinalizationStopAssertion:
    _absolute(path, "reason_file")
    try:
        return FinalizationStopAssertion.model_validate_json(
            Path(path).read_text(encoding="utf-8"), strict=True
        )
    except (OSError, UnicodeError, ValidationError) as error:
        raise FinalizationApplicationRefused(
            "finalization_stop_invalid", "reason_file"
        ) from error


def stop_finalization(
    command: StopFinalizationCommand, *, journey: Journey | None = None
) -> FinalizationProjection:
    trusted = StopFinalizationCommand.model_validate(command, strict=True)
    stop = _load_stop(trusted.reason_file)
    journey = _owned_journey(trusted.journey_file, journey)
    completion = _verified_read(journey)
    read = completion.investigation
    timeline = read.timeline
    projection = completion.finalization
    existing = _review_fact_for_request(timeline, trusted.request_event_id)
    if existing is not None:
        response = existing.receipt.response
        if (
            response.action == "stop"
            and response.stop.stop_id == stop.stop_id
            and existing.receipt.basis_digest == trusted.basis_digest
            and existing.receipt.response_head_event_id == trusted.head_event_id
        ):
            return projection
        raise FinalizationApplicationRefused(
            "finalization_review_response_conflict", "request_event_id"
        )
    pending = projection.pending_request
    if pending is None or pending.request_event_id != trusted.request_event_id:
        raise FinalizationApplicationRefused(
            "finalization_review_response_stale", "request_event_id"
        )
    if (
        not timeline
        or review_head(timeline)["event_id"] != trusted.head_event_id
        or trusted.head_event_id != pending.request_event_id
        or pending.basis.basis_digest != trusted.basis_digest
    ):
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "head_event_id", "basis_digest"
        )
    receipt = FinalizationReviewReceipt(
        request_event_id=trusted.request_event_id,
        response_head_event_id=trusted.head_event_id,
        basis_digest=trusted.basis_digest,
        source_head_event_id=pending.source_head_event_id,
        response=StopFinalizationResponse(stop=stop),
    )
    fact = FinalizationReviewRecorded(
        review_policy=FinalizationReviewPolicy(),
        receipt=receipt,
        stop=FinalizationStop(basis_digest=trusted.basis_digest, stop=stop),
    )
    current = journey
    if not current.is_current():
        raise FinalizationApplicationRefused(
            "finalization_review_binding_stale", "journey_file"
        )
    current.record(
        fact.event_type,
        fact.model_dump(
            mode="json",
            exclude={"event_type"},
            exclude_computed_fields=True,
        ),
        [trusted.head_event_id],
    )
    return _verified_read(current).finalization


__all__ = [
    "ApproveFinalizationCommand",
    "CompileFinalizationCommand",
    "CorrectFinalizationCommand",
    "FinalizationApplicationRefused",
    "InspectFinalizationCommand",
    "StopFinalizationCommand",
    "approve_finalization",
    "compile_finalization",
    "correct_finalization",
    "inspect_finalization",
    "stop_finalization",
]
