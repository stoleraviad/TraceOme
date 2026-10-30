"""Parabricks planning application commands."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import (
    StrictStr,
    field_validator,
)

from bio_neb.agentic_journey.continuation_policy import ExecutionReviewPolicy
from bio_neb.application.planning_read import prefix_snapshot
from bio_neb.application.run_failure import RunFailed, record_run_fact
from bio_neb.contracts import FrozenContract
from bio_neb.journey import Journey, JourneyFile
from bio_neb.parabricks.execution_proposal import (
    ApproveExecutionResponse,
    ExecutionHitlReceipt,
    ExecutionHitlResponded,
    ExecutionHitlResponse,
    ExecutionProposalSubmitted,
    ProposalCompilationRefused,
    RejectExecutionResponse,
    evaluate_execution_hitl_response,
    project_execution_hitl,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlResponse,
    PlanningHitlResponseReceipt,
    evaluate_planning_hitl_response,
    make_planning_hitl_responded,
    planning_hitl_response_receipt,
)
from bio_neb.parabricks.planning_readiness import PlanningIncomplete
from bio_neb.parabricks.preparation import PreparationRefused, prepare_execution

if TYPE_CHECKING:
    from bio_neb.application.queries import JourneyReview


class RespondPlanningHitlCommand(FrozenContract):
    journey_file: JourneyFile
    request_event_id: StrictStr
    response: PlanningHitlResponse

    @field_validator("request_event_id")
    @classmethod
    def canonical_request_identity(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("request_event_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("request_event_id must be canonical")
        return value


class SubmitExecutionProposalCommand(FrozenContract):
    journey_file: JourneyFile


class RespondExecutionHitlCommand(FrozenContract):
    journey_file: JourneyFile
    request_event_id: StrictStr
    response: ExecutionHitlResponse

    @field_validator("request_event_id")
    @classmethod
    def canonical_request_identity(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("request_event_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("request_event_id must be canonical")
        return value


def respond_planning_hitl(
    command: RespondPlanningHitlCommand,
) -> PlanningHitlResponseReceipt:
    """Append one exact legal response or return its retained receipt."""

    journey = Journey.open(Path(command.journey_file))
    state = journey.project()
    projection = journey.checked_view("planning_review")
    disposition = evaluate_planning_hitl_response(
        projection, command.request_event_id, command.response
    )
    if disposition.status == "identical":
        assert disposition.retained_receipt is not None
        return disposition.retained_receipt
    pending = next(
        (
            r
            for r in projection.pending_requests
            if r.request_event_id == command.request_event_id
        ),
        None,
    )
    assert pending is not None and pending.request_event_id == command.request_event_id
    fact = make_planning_hitl_responded(command.request_event_id, command.response)
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    response_event_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{journey.journey_id}:{command.request_event_id}:planning-hitl-response",
        )
    )
    parents = list(dict.fromkeys((command.request_event_id, state.events[-1].event_id)))
    journey.preview(
        state,
        fact.event_type,
        payload,
        parents,
        event_id=response_event_id,
    )
    recorded, _ = journey.record_against(
        state,
        fact.event_type,
        payload,
        parents,
        event_id=response_event_id,
    )
    receipt = planning_hitl_response_receipt(pending, recorded["event_id"], fact)
    if command.response.action == "stop":
        record_run_fact(
            journey,
            RunFailed(
                run_id=journey.journey_id,
                code="required_evidence_unresolved",
                source_event_id=recorded["event_id"],
            ),
            recorded["event_id"],
        )
    return receipt


def submit_execution_proposal(
    command: SubmitExecutionProposalCommand,
) -> JourneyReview:
    """Compile consequences from Journey evidence and open their inseparable review."""

    from bio_neb.application.queries import (
        ReviewJourneyQuery,
        review_journey,
    )

    journey = Journey.open(Path(command.journey_file))
    retained = project_execution_hitl(prefix_snapshot(journey).timeline)
    if retained.pending_request is not None:
        return review_journey(ReviewJourneyQuery(journey_file=command.journey_file))
    prepared = prepare_execution(journey)
    if isinstance(prepared, PreparationRefused):
        raise ProposalCompilationRefused(prepared.code, prepared.details)
    if isinstance(prepared, PlanningIncomplete):
        # A planning gap is planning's stop, not a compilation fault.
        return review_journey(ReviewJourneyQuery(journey_file=command.journey_file))
    proposal = prepared.proposal
    fact = ExecutionProposalSubmitted(
        review_policy=ExecutionReviewPolicy(), proposal=proposal
    )
    event_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{journey.journey_id}:{proposal.proposal_id}:execution-hitl-request",
        )
    )
    existing = next(
        (event for event in journey.read() if event["event_id"] == event_id), None
    )
    if existing is None:
        payload = fact.model_dump(mode="json")
        payload.pop("event_type")
        events = journey.read()
        journey.record(
            fact.event_type,
            payload,
            [events[-1]["event_id"]],
            event_id=event_id,
        )
    return review_journey(ReviewJourneyQuery(journey_file=command.journey_file))


def respond_execution_hitl(
    command: RespondExecutionHitlCommand,
) -> ExecutionHitlReceipt:
    """Record one exact user decision; approval grants authority without contact."""

    journey = Journey.open(Path(command.journey_file))
    projection = project_execution_hitl(prefix_snapshot(journey).timeline)
    disposition = evaluate_execution_hitl_response(
        projection, command.request_event_id, command.response
    )
    if disposition == "identical":
        return next(
            item
            for item in projection.receipts
            if item.request_event_id == command.request_event_id
        )
    response_fact = ExecutionHitlResponded(
        request_event_id=command.request_event_id,
        response=command.response,
        authorizes_execution=isinstance(command.response, ApproveExecutionResponse),
    )
    response_event_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{journey.journey_id}:{command.request_event_id}:execution-hitl-response",
        )
    )
    payload = response_fact.model_dump(mode="json")
    payload.pop("event_type")
    response_event = journey.record(
        response_fact.event_type,
        payload,
        [command.request_event_id],
        event_id=response_event_id,
    )
    if isinstance(command.response, RejectExecutionResponse):
        from bio_neb.application.run_failure import RunFailed

        failure = RunFailed(
            run_id=journey.journey_id,
            code="rejected_by_user",
            source_event_id=response_event["event_id"],
        )
        failure_payload = failure.model_dump(mode="json")
        failure_payload.pop("event_type")
        journey.record(
            failure.event_type,
            failure_payload,
            [response_event["event_id"]],
            event_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{journey.journey_id}:{response_event['event_id']}:rejected-by-user",
                )
            ),
        )
    completed = project_execution_hitl(prefix_snapshot(journey).timeline)
    return next(
        item
        for item in completed.receipts
        if item.request_event_id == command.request_event_id
    )
