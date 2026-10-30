"""Shared, contact-free Run interaction meaning from checked Journey observations.

Frontiers describe the current interaction checkpoint; they never grant an action.
Review response schemas and exact stage commands remain owned by their backend owners.
"""

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr

from bio_neb.agentic_journey.repeated_calls import RepeatProgress
from bio_neb.application.investigation_read import InvestigationMember
from bio_neb.application.queries import (
    ENDED_RUN_STATUSES,
    INVESTIGATION_WAIT_STATUSES,
    REPORT_APPROVED_STATUSES,
    REPORT_RETURNED_STATUSES,
    REPORT_STOPPED_STATUSES,
    AttemptEvidenceOutcome,
    JourneyObservation,
    ModelContactPostmortem,
    RunStatus,
)
from bio_neb.application.reviews import ReviewDetails
from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation import InvestigationProjection
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
)
from bio_neb.investigation.investigation_policy import InvestigationPolicyDecision
from bio_neb.investigation.investigation_request import CapturedInvestigationRequest
from bio_neb.parabricks.execution_proposal import (
    ExecutionAuthority,
    ExecutionProposal,
)
from bio_neb.parabricks.requirement_resolution import (
    ScientificInvestigationHandoff,
)
from bio_neb.recording import PassView
from bio_neb.reporting.finalization import FinalizationProjection
from bio_neb.reporting.grounded_report import GroundedReportProjection

RunFrontier = Literal[
    "progress",
    "review_required",
    "execution_authority",
    "checkpoint",
    "report_published",
    "terminal",
    "held",
]
# An ended Run or a stopped Finalization; a held Run is not terminal.
_TERMINAL_RUN_STATUSES: frozenset[RunStatus] = (
    ENDED_RUN_STATUSES | REPORT_STOPPED_STATUSES
)
_CHECKPOINT_RUN_STATUSES: frozenset[RunStatus] = (
    INVESTIGATION_WAIT_STATUSES | REPORT_APPROVED_STATUSES | REPORT_RETURNED_STATUSES
)


class AttemptActivity(FrozenContract):
    attempt_id: StrictStr
    agent_role: StrictStr
    outcome: AttemptEvidenceOutcome
    model_evidence_count: StrictInt = Field(ge=0)
    tool_evidence_count: StrictInt = Field(ge=0)
    lifecycle_evidence_count: StrictInt = Field(ge=0)


class RetainedEvidence(FrozenContract):
    source_event_id: StrictStr
    event_type: StrictStr


class ToolEvidence(RetainedEvidence):
    tool_name: StrictStr | None
    tool_call_id: StrictStr | None


class LifecycleEvidence(RetainedEvidence):
    stage: StrictStr | None
    activity_id: StrictStr | None


class RequirementProgress(FrozenContract):
    kind: StrictStr
    subject_key: StrictStr
    meaning: StrictStr
    execution_consequence: StrictBool
    state: Literal[
        "resolved",
        "needs_tool",
        "needs_review",
        "conflicted",
        "post_execution",
        "rejected",
    ]
    value_count: StrictInt = Field(ge=0)
    candidate_evidence_count: StrictInt = Field(ge=0)


class RunProgress(FrozenContract):
    activities: tuple[AttemptActivity, ...]
    model_evidence: tuple[RetainedEvidence, ...]
    requirements: tuple[RequirementProgress, ...]
    tool_evidence: tuple[ToolEvidence, ...]
    lifecycle_evidence: tuple[LifecycleEvidence, ...]
    repeats: tuple[RepeatProgress, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class ResearchIntent(FrozenContract):
    prompt: StrictStr
    prompt_file: StrictStr
    source_event_id: StrictStr


class ScientificHandoff(FrozenContract):
    research_intent: ResearchIntent
    handoff: ScientificInvestigationHandoff


class RunInteraction(FrozenContract):
    schema_version: Literal["bio-neb.run-interaction/1"] = "bio-neb.run-interaction/1"
    view: Literal["concise"] = "concise"
    frontier: RunFrontier
    journey_id: StrictStr
    journey_file: StrictStr
    status: RunStatus
    progress: RunProgress
    pending_reviews: tuple[ReviewDetails, ...] = ()
    passes: tuple[PassView, ...] = ()
    review: ReviewDetails | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    execution_proposal: ExecutionProposal | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_investigation_handoff: ScientificHandoff | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    execution_authority: tuple[ExecutionAuthority, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    investigation_request: CapturedInvestigationRequest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    investigation: InvestigationProjection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    investigation_evidence: InvestigationCollectionEvidenceRecorded | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    investigation_members: tuple[InvestigationMember, ...] = ()
    investigation_policy: InvestigationPolicyDecision | None = None
    grounded_report: GroundedReportProjection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    failure_code: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    failure_source_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_contact_postmortem: ModelContactPostmortem | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    finalization: FinalizationProjection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def _run_progress(review: JourneyObservation) -> RunProgress:
    activities: list[AttemptActivity] = []
    model_evidence: list[RetainedEvidence] = []
    tool_evidence: list[ToolEvidence] = []
    lifecycle_evidence: list[LifecycleEvidence] = []
    for attempt in review.attempts:
        activities.append(
            AttemptActivity(
                attempt_id=attempt.attempt_id,
                agent_role=attempt.agent_role,
                outcome=attempt.outcome,
                model_evidence_count=len(attempt.model_events),
                tool_evidence_count=len(attempt.tool_events),
                lifecycle_evidence_count=len(attempt.lifecycle_events),
            )
        )
        model_evidence.extend(
            RetainedEvidence(
                source_event_id=event.source_event_id,
                event_type=event.event_type,
            )
            for event in attempt.model_events
        )
        for event in attempt.tool_events:
            payload = event.payload
            tool_evidence.append(
                ToolEvidence(
                    source_event_id=event.source_event_id,
                    event_type=event.event_type,
                    tool_name=payload.get("tool_name"),
                    tool_call_id=payload.get("tool_call_id"),
                )
            )
        lifecycle_evidence.extend(
            LifecycleEvidence(
                source_event_id=event.source_event_id,
                event_type=event.event_type,
                stage=event.payload.get("stage"),
                activity_id=event.payload.get("activity_id"),
            )
            for event in attempt.lifecycle_events
        )
    requirements: list[RequirementProgress] = []
    if review.requirement_resolution is not None:
        requirements = [
            RequirementProgress(
                kind=item.kind,
                subject_key=item.subject_key,
                meaning=item.meaning,
                execution_consequence=item.execution_consequence,
                state=item.state,
                value_count=len(item.values),
                candidate_evidence_count=len(item.candidate_evidence),
            )
            for item in review.requirement_resolution.requirements
        ]
    return RunProgress(
        activities=tuple(activities),
        model_evidence=tuple(model_evidence),
        requirements=tuple(requirements),
        tool_evidence=tuple(tool_evidence),
        lifecycle_evidence=tuple(lifecycle_evidence),
        repeats=review.repeats,
    )


def run_interaction(review: JourneyObservation) -> RunInteraction:
    pending = review.current_review
    execution = review.execution_hitl
    investigation = getattr(review, "investigation", None)
    finalization = getattr(review, "finalization", None)
    authorities = () if execution is None else execution.authorities
    if review.status == "grounded_report_published":
        frontier: RunFrontier = "report_published"
    elif pending is not None:
        frontier = "review_required"
    elif review.status in _CHECKPOINT_RUN_STATUSES:
        frontier = "checkpoint"
    elif review.status in _TERMINAL_RUN_STATUSES:
        frontier = "terminal"
    elif review.status == "run_held":
        frontier = "held"
    elif authorities:
        frontier = "execution_authority"
    else:
        frontier = "progress"
    proposal = (
        execution.requests[-1].proposal
        if execution is not None and execution.requests
        else None
    )
    scientific_handoff = None
    if review.requirement_resolution is not None:
        scientific_handoff = ScientificHandoff(
            research_intent=ResearchIntent(
                prompt=review.request.prompt,
                prompt_file=review.request.prompt_file,
                source_event_id=review.request.source_event_id,
            ),
            handoff=review.requirement_resolution.scientific_investigation,
        )
    authority_documents = tuple(authorities) if authorities else None
    return RunInteraction(
        frontier=frontier,
        journey_id=review.journey_id,
        journey_file=review.journey_file,
        status=review.status,
        progress=_run_progress(review),
        pending_reviews=review.pending_reviews,
        passes=review.passes,
        review=pending,
        execution_proposal=proposal,
        scientific_investigation_handoff=scientific_handoff,
        execution_authority=authority_documents,
        investigation_request=getattr(review, "investigation_request", None),
        investigation=(
            investigation
            if investigation is not None and investigation.status != "not_started"
            else None
        ),
        investigation_evidence=getattr(review, "investigation_evidence", None),
        investigation_members=getattr(review, "investigation_members", ()),
        investigation_policy=getattr(review, "investigation_policy", None),
        grounded_report=getattr(review, "grounded_report", None),
        failure_code=getattr(review, "failure_code", None),
        failure_source_event_id=getattr(review, "failure_source_event_id", None),
        model_contact_postmortem=getattr(review, "model_contact_postmortem", None),
        finalization=(
            finalization
            if finalization is not None and finalization.status != "not_started"
            else None
        ),
    )
