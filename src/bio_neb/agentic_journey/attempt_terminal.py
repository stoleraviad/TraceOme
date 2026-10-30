"""Application-owned compilation of one Attempt's administrative terminal."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from bio_neb.agentic_journey.attempt_journey import (
    AttemptSucceeded,
    SliceCheckpointReached,
)
from bio_neb.agentic_journey.continuation import pending_continuation
from bio_neb.journey import Journey, JourneyEvent
from bio_neb.recording import read_evidence

if TYPE_CHECKING:
    from bio_neb.application.queries import JourneyReview


@dataclass(frozen=True)
class AttemptDirection:
    before: JourneyReview
    causal_head: str | None = None
    researcher_instruction: str | None = None
    availability_retry_frontier: str | None = None


def prepare_attempt_direction(
    journey: Journey,
    *,
    agent_role: str,
    availability_review_response_event_id: str | None = None,
    effort_review_response_event_id: str | None = None,
    work_review_response_event_id: str | None = None,
    operational_review_response_event_id: str | None = None,
) -> AttemptDirection:
    """Bind one fresh researcher direction; each role still owns stage eligibility."""
    from bio_neb.agentic_journey.model_availability import (
        model_availability_direction,
    )
    from bio_neb.application.queries import (
        ReviewJourneyQuery,
        review_journey,
        review_model_availability_retry_prefix,
    )

    before = (
        review_model_availability_retry_prefix(str(journey.path))
        if availability_review_response_event_id is not None
        else review_journey(ReviewJourneyQuery(journey_file=str(journey.path)))
    )
    response_id = (
        operational_review_response_event_id
        or availability_review_response_event_id
        or effort_review_response_event_id
        or work_review_response_event_id
    )
    causal_head = (
        pending_continuation(
            read_evidence(journey), response_id, agent_role=agent_role
        ).causal_head
        if response_id is not None
        else None
    )
    if operational_review_response_event_id is not None:
        from bio_neb.agentic_journey.operational_review import pending_operational_work

        request, receipt = pending_operational_work(
            journey, operational_review_response_event_id, agent_role=agent_role
        )
        return AttemptDirection(
            before,
            causal_head,
            researcher_instruction=receipt.response.value
            if receipt.response.action == "amend"
            else None,
            availability_retry_frontier=request.retry_frontier_event_id
            if receipt.response.action == "retry"
            else causal_head,
        )
    if availability_review_response_event_id is not None:
        request, receipt = model_availability_direction(
            journey, availability_review_response_event_id, agent_role=agent_role
        )
        if receipt.response.action == "amend":
            return AttemptDirection(
                before,
                causal_head,
                researcher_instruction=receipt.response.instruction,
                availability_retry_frontier=causal_head,
            )
        return AttemptDirection(
            before,
            causal_head,
            availability_retry_frontier=request.retry_frontier_event_id,
        )
    if effort_review_response_event_id is not None:
        from bio_neb.agentic_journey.effort import pending_effort_work

        # Checks the answer is this role's exact pending effort work; raises if not.
        pending_effort_work(
            journey, effort_review_response_event_id, agent_role=agent_role
        )
        return AttemptDirection(
            before, causal_head, availability_retry_frontier=causal_head
        )
    if work_review_response_event_id is not None:
        from bio_neb.agentic_journey.agent_work_review import pending_agent_work

        if before.status != "agent_work_review_direction_recorded":
            raise ValueError("Agent work continuation is not eligible")
        _, instruction = pending_agent_work(
            journey, work_review_response_event_id, agent_role=agent_role
        )
        # The exact answer is authority; independent later facts remain ancestors.
        return AttemptDirection(
            before,
            causal_head,
            researcher_instruction=instruction,
        )
    return AttemptDirection(before)


AttemptTerminalRole = Literal["planning_agent"]
_TERMINAL_ROLES: frozenset[str] = frozenset({"planning_agent"})
AttemptTerminalCheckpoint = Literal[
    "registered_inputs_inspected",
    "planning_hitl_requested",
    "execution_planning_ready",
    "target_answer_read",
    "assembly_suggested",
    "paper_review_requested",
    "paper_reading_due",
]


class _Disposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContinueAttempt(_Disposition):
    """The retained frontier still offers substantive Agent work."""

    kind: Literal["continue"] = "continue"


class _TerminalDisposition(_Disposition):
    checkpoint: AttemptTerminalCheckpoint
    source_event_id: StrictStr

    @field_validator("source_event_id")
    @classmethod
    def canonical_source(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("source_event_id must be canonical")
        return value


class PlanningHitlTerminal(_TerminalDisposition):
    """One pending Planning HITL request is the Attempt frontier."""

    kind: Literal["planning_hitl"] = "planning_hitl"
    checkpoint: Literal["planning_hitl_requested"] = "planning_hitl_requested"


class SliceReachedTerminal(_TerminalDisposition):
    """One retained scientific/Tool fact completes the current stage slice."""

    kind: Literal["slice_reached"] = "slice_reached"
    checkpoint: Literal[
        "registered_inputs_inspected",
        "execution_planning_ready",
        "target_answer_read",
        "assembly_suggested",
        "paper_review_requested",
        "paper_reading_due",
    ]


AttemptTerminalDisposition = Annotated[
    ContinueAttempt | PlanningHitlTerminal | SliceReachedTerminal,
    Field(discriminator="kind"),
]


def _as_event(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def _canonical_uuid(value: Any, name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be canonical")
    return value


class _ActiveAttempt:
    def __init__(
        self,
        timeline: list[dict[str, Any]],
        role: AttemptTerminalRole,
        attempt_id: str,
    ) -> None:
        self.timeline = timeline
        self.by_id: dict[str, dict[str, Any]] = {}
        self.ancestry: dict[str, set[str]] = {}
        for event in timeline:
            event_id = _canonical_uuid(event.get("event_id"), "event_id")
            parents = event.get("parent_ids")
            if (
                event_id in self.by_id
                or not isinstance(parents, list)
                or any(parent not in self.by_id for parent in parents)
            ):
                raise ValueError(
                    "Journey terminal projection requires unique backward parentage"
                )
            ancestors = set(parents)
            for parent in parents:
                ancestors.update(self.ancestry[parent])
            self.by_id[event_id] = event
            self.ancestry[event_id] = ancestors

        starts = [
            event
            for event in timeline
            if event.get("event_type") == "AttemptStarted"
            and event.get("payload", {}).get("attempt_id") == attempt_id
        ]
        if len(starts) != 1:
            raise ValueError("terminal projection requires one exact active Attempt")
        start = starts[0]
        payload = start.get("payload", {})
        if payload.get("agent_role") != role:
            raise ValueError("active Attempt role does not match terminal projection")
        self.start_id = start["event_id"]

        descendants = [
            event
            for event in timeline
            if event["event_id"] == self.start_id
            or self.start_id in self.ancestry[event["event_id"]]
        ]
        if any(
            event.get("event_type") in {"AttemptSucceeded", "AttemptFailed"}
            and event.get("payload", {}).get("attempt_id") == attempt_id
            for event in descendants
        ):
            raise ValueError("terminal projection cannot close a completed Attempt")
        descendant_ids = {event["event_id"] for event in descendants}
        leaves = [
            event_id
            for event_id in descendant_ids
            if not any(
                event_id in self.ancestry[other_id]
                for other_id in descendant_ids
                if other_id != event_id
            )
        ]
        if len(leaves) != 1:
            raise ValueError("active Attempt does not have one current causal head")
        self.head_id = leaves[0]

    def require_source(
        self, source_event_id: str, expected_event_types: frozenset[str]
    ) -> None:
        source = self.by_id.get(source_event_id)
        if source is None:
            raise ValueError("terminal source is missing from retained Journey facts")
        if source.get("event_type") not in expected_event_types:
            raise ValueError("terminal source has the wrong retained fact type")
        if (
            source_event_id != self.start_id
            and self.start_id not in self.ancestry[source_event_id]
        ):
            raise ValueError("terminal source belongs to another Attempt")
        source_attempt_id = source.get("payload", {}).get("attempt_id")
        active_attempt_id = self.by_id[self.start_id]["payload"]["attempt_id"]
        if source_attempt_id is not None and source_attempt_id != active_attempt_id:
            raise ValueError("terminal source changed its owning Attempt")
        if (
            source_event_id != self.head_id
            and source_event_id not in self.ancestry[self.head_id]
        ):
            raise ValueError("terminal source is outside the current causal head")


def project_attempt_terminal(
    role: AttemptTerminalRole,
    active_attempt_id: str,
    journey_events: Journey | Iterable[JourneyEvent | Mapping[str, Any]],
) -> AttemptTerminalDisposition:
    """Compile the unique retained administrative frontier for one active Attempt.

    Projection status decides whether substantive work remains.  A separate causal audit
    then proves that any terminal source belongs to the exact active Attempt and its sole
    retained head.  No Agent output, Tool fact, HITL fact, or execution authority is made.
    """

    if role not in _TERMINAL_ROLES:
        raise ValueError("role is not an administrative terminal stage")
    _canonical_uuid(active_attempt_id, "active_attempt_id")
    owner = journey_events if isinstance(journey_events, Journey) else None
    if owner is not None:
        from bio_neb.recording import read_evidence

        journey_events = read_evidence(owner)
    timeline = [_as_event(event) for event in journey_events]
    active = _ActiveAttempt(timeline, role, active_attempt_id)

    from bio_neb.agentic_journey.planning_assignment import assignment_terminal

    disposition = assignment_terminal(
        owner if owner is not None else timeline, active_attempt_id
    )
    if not isinstance(disposition, ContinueAttempt):
        active.require_source(
            disposition.source_event_id,
            frozenset(
                {
                    "InputMetadataInspected",
                    "CompanionObservationRecorded",
                    "TaxonResolutionRecorded",
                    "ReferenceAssembliesRecorded",
                    "ReferenceDescriptionRecorded",
                    "PlanningHitlRequested",
                    "RecipeObservationRecorded",
                    "ResourceObservationRecorded",
                    "PlanningCandidateSelected",
                    "TargetAnswerRead",
                    "AssemblyWithIndexSuggested",
                    "PaperReviewRequested",
                    # A plan whose values cite a recorded paper to read.
                    "ProposalRequirementsIdentified",
                }
            ),
        )
    return disposition


def record_attempt_terminal(
    activity: Any,
    disposition: AttemptTerminalDisposition,
    *,
    attempt_id: str,
    input_set_id: str,
    proposal_plan_event_id: str | None = None,
) -> dict[str, Any]:
    """Record one compiled checkpoint followed immediately by Attempt success."""

    if isinstance(disposition, ContinueAttempt):
        raise TypeError("cannot record a continuing Attempt as successful")
    checkpoint_name = disposition.checkpoint
    checkpoint = activity.record(
        SliceCheckpointReached(
            attempt_id=attempt_id,
            checkpoint=checkpoint_name,
            input_set_id=input_set_id,
            inspection_event_id=(
                disposition.source_event_id
                if checkpoint_name == "registered_inputs_inspected"
                else None
            ),
            proposal_plan_event_id=proposal_plan_event_id,
            planning_hitl_request_event_id=(
                disposition.source_event_id
                if isinstance(disposition, PlanningHitlTerminal)
                else None
            ),
            execution_planning_evidence_event_id=(
                disposition.source_event_id
                if checkpoint_name == "execution_planning_ready"
                else None
            ),
            target_answer_read_event_id=(
                disposition.source_event_id
                if checkpoint_name == "target_answer_read"
                else None
            ),
            assembly_suggestion_event_id=(
                disposition.source_event_id
                if checkpoint_name == "assembly_suggested"
                else None
            ),
            paper_review_request_event_id=(
                disposition.source_event_id
                if checkpoint_name == "paper_review_requested"
                else None
            ),
        )
    )
    succeeded = activity.record(
        AttemptSucceeded(
            attempt_id=attempt_id, checkpoint_event_id=checkpoint["event_id"]
        )
    )
    activity.succeeded(succeeded)
    return succeeded


__all__ = [
    "AttemptTerminalCheckpoint",
    "AttemptTerminalDisposition",
    "AttemptTerminalRole",
    "ContinueAttempt",
    "PlanningHitlTerminal",
    "SliceReachedTerminal",
    "project_attempt_terminal",
    "record_attempt_terminal",
]
