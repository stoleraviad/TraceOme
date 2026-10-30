"""Contact-free typed review queries derived only from Journey evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    computed_field,
    model_validator,
)

from bio_neb.agentic_journey.agent_work_review import (
    AgentWorkReviewProjection,
    agent_work_status,
    validate_agent_work_history,
    validate_agent_work_terminal,
)
from bio_neb.agentic_journey.attempt_journey import (
    MODEL_REQUEST_TERMINALS,
    AttemptFailed,
    ModelAvailabilityRetryDelayCancelled,
    ModelAvailabilityRetryDelayCompleted,
    ModelAvailabilityRetryScheduled,
    ModelContactCompletion,
    ModelFailureReason,
    ModelPhysicalContactCancelled,
    ModelPhysicalContactFailed,
    ModelPhysicalContactStarted,
    ModelRequestFailed,
    ModelRequestRecorded,
    ModelRequestStalled,
    ModelResponseRecorded,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.attempt_terminal import (
    AttemptTerminalRole,
    ContinueAttempt,
    project_attempt_terminal,
)
from bio_neb.agentic_journey.continuation import (
    REVIEW_RESPONSE_KINDS,
    pending_continuation,
    validate_attempt_continuation,
    validate_review_answer_parentage,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationPolicy,
    ContinuationPolicy,
    validate_continuation,
)
from bio_neb.agentic_journey.effort import (
    AgentEffortReviewProjection,
    effort_status,
    validate_effort_history,
    validate_effort_terminal,
)
from bio_neb.agentic_journey.instruction_evidence import retained_instruction_reference
from bio_neb.agentic_journey.lifecycle import (
    ActivityCancelled,
    ActivityFailed,
    ActivityHeartbeat,
    ActivityProgress,
    ActivityStarted,
    ActivitySucceeded,
    validate_activity_fact,
)
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewProjection,
    RetryModelAvailabilityResponse,
    project_model_availability_review,
    validate_attempt_cancellation,
    validate_model_contact_window,
)
from bio_neb.agentic_journey.operational_review import (
    AgentFailureReviewProjection,
    operational_state,
    project_operational_review,
    validate_operational_terminal,
)
from bio_neb.agentic_journey.repeated_calls import RepeatProgress, journey_repeats
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.agentic_journey.runtime_config import (
    RuntimeSelections,
    retained_endpoint_policy,
    retained_invocation_policy,
    retained_runtime_selections,
    validate_retained_model_request,
)
from bio_neb.agentic_journey.tool_execution import (
    function_tool_contracts,
    recorded_within_call,
    validate_tool_failure_source,
)
from bio_neb.agentic_journey.transcript import check_stall_notes
from bio_neb.agentic_journey.turn_reopen import (
    compatible_model_request as _compatible_model_request,
)
from bio_neb.agentic_journey.turn_reopen import validate_action_turns
from bio_neb.application.input_journey import (
    InputSelection,
    RegisteredFastqInput,
    ResearchRequestRegistered,
    RunInputsRegistered,
    validate_fact,
)
from bio_neb.application.investigation_read import (
    CompletionRead,
    InvestigationMember,
    InvestigationRead,
    TableReadingStage,
    _InvestigationContext,
)
from bio_neb.application.planning_read import (
    PlanningRead,
    PrefixSnapshot,
    prefix_snapshot,
)
from bio_neb.application.reviews import ReviewDetails, current_review
from bio_neb.application.run_failure import (
    RunFailed,
    RunFailureCode,
    RuntimeConfigurationFailed,
    validate_run_fact,
)

# The tolerant Run reads every outside consumer shares; the CLI reaches them here.
from bio_neb.application.run_reader import (
    read_journey_feed as read_journey_feed,
)
from bio_neb.application.run_reader import (
    read_run_artifacts as read_run_artifacts,
)
from bio_neb.contracts import FrozenContract
from bio_neb.input_inspection.tool import (
    InputMetadataResult,
)
from bio_neb.investigation.download_review import (
    InvestigationDownloadReviews,
    project_download_reviews,
)
from bio_neb.investigation.investigation import InvestigationProjection
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
)
from bio_neb.investigation.investigation_policy import (
    InvestigationPolicyDecision,
)
from bio_neb.investigation.investigation_request import (
    CapturedInvestigationRequest,
    current_investigation_request,
    investigation_request_document_schema,
)
from bio_neb.investigation.run_health import RunHealthState, project_run_health
from bio_neb.journey import (
    FORK_ORIGIN_EVENT_TYPE,
    ArtifactReference,
    Journey,
    JourneyEvent,
    JourneyFile,
)
from bio_neb.journey import has_ancestor as _has_ancestor
from bio_neb.journey_path import has_execution_path
from bio_neb.paper_intake.journey import PAYLOADS as PAPER_PAYLOADS
from bio_neb.paper_intake.journey import RunPapers, project_run_papers
from bio_neb.parabricks.execution_package import (
    ExecutionPackageCompiled,
    validate_execution_package_fact,
)
from bio_neb.parabricks.execution_planning_journey import (
    ExecutionPlanningProjection,
    validate_execution_planning_fact,
)
from bio_neb.parabricks.execution_proposal import (
    ExecutionHitlProjection,
    ExecutionHitlResponded,
    ExecutionProposalSubmitted,
    RejectExecutionResponse,
    ReviseExecutionResponse,
    project_execution_hitl,
    validate_execution_fact,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlProjection,
    PlanningHitlResponded,
    StopPlanningHitlResponse,
    project_planning_hitl,
)
from bio_neb.parabricks.preparation_review import (
    PreparationReview,
    project_preparation_review,
)
from bio_neb.parabricks.proposal_information import (
    ProposalInformationPlan,
)
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
    validate_proposal_fact,
)
from bio_neb.parabricks.requirement_resolution import (
    RequirementResolutionProjection,
)
from bio_neb.parabricks.result import (
    ParabricksExecutionOutcome,
    ParabricksNoncompletion,
    ParabricksResult,
    read_parabricks_result,
)
from bio_neb.recording import PassView, held_failure, read_passes, review_head
from bio_neb.reference_discovery.reference_facts_journey import (
    ReferenceFactsProjection,
    validate_reference_facts_fact,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceIdentityProjection,
    validate_reference_fact,
)
from bio_neb.reference_discovery.reference_source import (
    ReferenceSourceAdmitted,
    validate_reference_source_fact,
)
from bio_neb.reporting.finalization import (
    FinalizationProjection,
)
from bio_neb.reporting.grounded_report import (
    GroundedReportProjection,
    project_grounded_report,
)


class ObserveRun(FrozenContract):
    """Locate the Journey that owns one Run's retained truth."""

    journey_file: JourneyFile


class ReviewJourneyQuery(ObserveRun):
    """Compatibility input for callers not yet moved to the Run observation seam."""


class RequestReview(FrozenContract):
    prompt: StrictStr
    prompt_file: StrictStr
    source_event_id: StrictStr


class RegisteredInputSetReview(FrozenContract):
    input_set_id: StrictStr
    run_root: StrictStr
    runtime_config_file: StrictStr | None
    runtime_selections: RuntimeSelections
    inputs: tuple[RegisteredFastqInput, ...] = Field(min_length=1)
    source_event_id: StrictStr


class InspectionReview(FrozenContract):
    result: InputMetadataResult
    source_event_id: StrictStr
    parent_event_ids: tuple[StrictStr, ...] = Field(min_length=1)


class JourneyLineageItem(FrozenContract):
    event_id: StrictStr
    event_type: StrictStr
    parent_event_ids: tuple[StrictStr, ...]


class JourneyEvidenceItem(FrozenContract):
    source_event_id: StrictStr
    event_type: StrictStr
    parent_event_ids: tuple[StrictStr, ...]
    payload: dict[str, JsonValue]


# A terminal reading's outcome, or ``incomplete`` while the Attempt is open.
AttemptEvidenceOutcome = Literal[
    "succeeded", "failed", "cancelled", "incomplete", "stopped"
]


class AttemptEvidenceReview(FrozenContract):
    attempt_id: StrictStr
    instruction_identity: StrictStr
    instruction_artifact: ArtifactReference
    effective_invocation_policy: AgentInvocationPolicy | None = None
    invocation_policy_id: StrictStr | None = None
    invocation_overrides: dict[str, JsonValue] | None = None
    agent_role: AgentRole
    outcome: AttemptEvidenceOutcome
    events: tuple[JourneyEvidenceItem, ...] = Field(min_length=1)
    model_events: tuple[JourneyEvidenceItem, ...]
    tool_events: tuple[JourneyEvidenceItem, ...]
    receipt_events: tuple[JourneyEvidenceItem, ...] = ()
    lifecycle_events: tuple[JourneyEvidenceItem, ...]
    validation_events: tuple[JourneyEvidenceItem, ...]
    checkpoint_events: tuple[JourneyEvidenceItem, ...]


class ProposalPlanReview(FrozenContract):
    plan: ProposalInformationPlan
    source_event_id: StrictStr
    parent_event_ids: tuple[StrictStr, ...] = Field(min_length=1)


class ModelContactFailureGroup(FrozenContract):
    error_type: StrictStr
    http_status_code: StrictInt | None
    failure_class: ModelFailureReason | None
    completion_certainty: ModelContactCompletion
    policy_disposition: Literal[
        "automatic_retry_scheduled",
        "stall_noted",
        "rejection_shown",
        "availability_review_requested",
        "terminal_failure",
    ]
    count: StrictInt = Field(gt=0)


class ModelContactRetryScheduleGroup(FrozenContract):
    delay_seconds: float = Field(ge=0, allow_inf_nan=False)
    delay_source: Literal[
        "retry_after",
        "exponential_jitter",
        "malformed_response",
    ]
    count: StrictInt = Field(gt=0)


class ModelContactPostmortem(FrozenContract):
    logical_request_count: StrictInt = Field(gt=0)
    # Contacts that reached the transport; a not_sent refusal is not one.
    physical_contact_count: StrictInt = Field(ge=0)
    admitted_response_count: StrictInt = Field(ge=0)
    # Requests that ended with no reply, each noted in the next.
    stalled_request_count: StrictInt = Field(ge=0)
    failure_groups: tuple[ModelContactFailureGroup, ...]
    retry_schedules: tuple[ModelContactRetryScheduleGroup, ...]
    availability_review_request_count: StrictInt = Field(ge=0)
    terminal_disposition: (
        Literal[
            "all_model_requests_resolved_and_admitted",
            "operational_review_pending",
            "operational_review_direction_recorded",
            "operational_stopped",
            "model_availability_review_pending",
            "model_availability_stopped",
            "model_contact_cancelled",
            "model_contact_terminal_failure",
            "model_availability_recovered_and_admitted",
        ]
        | None
    ) = None


class _ReviewObservation(FrozenContract):
    operational_review: AgentFailureReviewProjection = AgentFailureReviewProjection()
    investigation_members: tuple[InvestigationMember, ...] = ()
    investigation_policy: InvestigationPolicyDecision | None = None
    passes: tuple[PassView, ...] = ()
    # Each repeated call, derived from retained evidence by the cap's own reader.
    repeats: tuple[RepeatProgress, ...] = ()

    @computed_field
    @property
    def pending_reviews(self) -> tuple[ReviewDetails, ...]:
        from bio_neb.application.reviews import pending_reviews

        return pending_reviews(self)

    @computed_field
    @property
    def current_review(self) -> ReviewDetails | None:
        return current_review(self)

    @model_validator(mode="after")
    def exact_current_review(self):
        current_review(self)
        return self


RunStatus = Literal[
    "inputs_registered",
    "input_review_pending",
    "input_correction_pending",
    "inspection_succeeded",
    "inspection_failed",
    # A supplied Run paper to read, or a refused one asking its Review.
    "paper_intake_pending",
    "paper_review_pending",
    "operational_review_pending",
    "operational_review_direction_recorded",
    "operational_stopped",
    "model_availability_review_pending",
    "effort_review_pending",
    "effort_review_direction_recorded",
    "effort_stopped",
    "agent_work_review_pending",
    "agent_work_review_direction_recorded",
    "agent_work_rejected",
    "model_availability_review_direction_recorded",
    "model_availability_stopped",
    "incomplete",
    "proposal_requirements_identified",
    "reference_identity_progress",
    "planning_hitl_pending",
    "reference_identity",
    "reference_facts_progress",
    "reference_facts",
    "execution_planning_progress",
    "execution_planning_ready",
    "execution_hitl_pending",
    "execution_revision_pending",
    "execution_rejected",
    "execution_authorized",
    "cpu_preparation_review_pending",
    "cpu_preparation_authorized",
    "execution_failed",
    "execution_artifacts_verified",
    "investigation_work_ready",
    "investigation_review_pending",
    "investigation_download_review_pending",
    "run_health_review_pending",
    "investigation_approved",
    "investigation_evidence_ready",
    "finalization_review_pending",
    "finalization_approved",
    "grounded_report_published",
    "finalization_returned_to_investigation",
    "finalization_stopped",
    "investigation_stopped",
    "failed",
    "run_held",
]

# The status projection's named status sets. Each RunStatus is in exactly one set
# of RUN_STATUS_SETS; the coordinator builds its status lists from these. An
# Agent's stop: its Review pending, answered (the Agent work re-enters) or ended
# by the researcher.
AGENT_REVIEW_PENDING_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "operational_review_pending",
        "model_availability_review_pending",
        "effort_review_pending",
        "agent_work_review_pending",
    }
)
REENTRY_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "operational_review_direction_recorded",
        "model_availability_review_direction_recorded",
        "effort_review_direction_recorded",
        "agent_work_review_direction_recorded",
    }
)
AGENT_ENDED_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "operational_stopped",
        "model_availability_stopped",
        "effort_stopped",
        "agent_work_rejected",
    }
)
AGENT_STOP_STATUSES = (
    AGENT_REVIEW_PENDING_STATUSES | REENTRY_STATUSES | AGENT_ENDED_STATUSES
)
# A stage's researcher stop or a recorded failure.
STAGE_ENDED_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "inspection_failed",
        "incomplete",
        "execution_rejected",
        "execution_failed",
        "investigation_stopped",
        "failed",
    }
)
# Only a researcher stop/reject or a recorded failure ends a Run; a held Run
# is not ended.
ENDED_RUN_STATUSES = AGENT_ENDED_STATUSES | STAGE_ENDED_STATUSES
HELD_RUN_STATUSES: frozenset[RunStatus] = frozenset({"run_held"})
STAGE_REVIEW_PENDING_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "input_review_pending",
        "paper_review_pending",
        "planning_hitl_pending",
        "execution_hitl_pending",
        "finalization_review_pending",
    }
)
# Inputs through reference facts; then Execution planning, ready or revising.
PLANNING_PROGRESS_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "inputs_registered",
        "input_correction_pending",
        "inspection_succeeded",
        "paper_intake_pending",
        "proposal_requirements_identified",
        "reference_identity_progress",
        "reference_identity",
        "reference_facts_progress",
        "reference_facts",
    }
)
EXECUTION_PLANNING_PROGRESS_STATUSES: frozenset[RunStatus] = frozenset(
    {"execution_planning_progress", "execution_revision_pending"}
)
EXECUTION_PLANNING_READY_STATUSES: frozenset[RunStatus] = frozenset(
    {"execution_planning_ready"}
)
EXECUTION_PLANNING_STATUSES = (
    EXECUTION_PLANNING_PROGRESS_STATUSES | EXECUTION_PLANNING_READY_STATUSES
)
EXECUTION_AUTHORITY_STATUSES: frozenset[RunStatus] = frozenset({"execution_authorized"})
# Only a Parabricks result view carries these; it is read by its type.
EXECUTION_RESULT_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "cpu_preparation_review_pending",
        "cpu_preparation_authorized",
        "execution_artifacts_verified",
    }
)
# A grant's refused file download waits at its failure Review.
INVESTIGATION_REVIEW_PENDING_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "investigation_review_pending",
        "investigation_download_review_pending",
        # A failed health measurement holds the table stage at its Review.
        "run_health_review_pending",
    }
)
# Investigation's quiet waits: work or evidence ready, or approved.
INVESTIGATION_WAIT_STATUSES: frozenset[RunStatus] = frozenset(
    {
        "investigation_work_ready",
        "investigation_approved",
        "investigation_evidence_ready",
    }
)
INVESTIGATION_STATUSES = (
    INVESTIGATION_REVIEW_PENDING_STATUSES | INVESTIGATION_WAIT_STATUSES
)
REPORT_APPROVED_STATUSES: frozenset[RunStatus] = frozenset({"finalization_approved"})
REPORT_RETURNED_STATUSES: frozenset[RunStatus] = frozenset(
    {"finalization_returned_to_investigation"}
)
REPORT_STOPPED_STATUSES: frozenset[RunStatus] = frozenset({"finalization_stopped"})
REPORT_PUBLISHED_STATUSES: frozenset[RunStatus] = frozenset(
    {"grounded_report_published"}
)
# Where a Run paper's intake or Review holds the Run: before Execution approval.
_PAPER_WORK_STATUSES: frozenset[RunStatus] = (
    PLANNING_PROGRESS_STATUSES
    | EXECUTION_PLANNING_STATUSES
    | frozenset({"planning_hitl_pending"})
)
RUN_STATUS_SETS: Mapping[str, frozenset[RunStatus]] = MappingProxyType(
    {
        "agent_review_pending": AGENT_REVIEW_PENDING_STATUSES,
        "reentry": REENTRY_STATUSES,
        "agent_ended": AGENT_ENDED_STATUSES,
        "stage_ended": STAGE_ENDED_STATUSES,
        "held": HELD_RUN_STATUSES,
        "stage_review_pending": STAGE_REVIEW_PENDING_STATUSES,
        "planning_progress": PLANNING_PROGRESS_STATUSES,
        "execution_planning_progress": EXECUTION_PLANNING_PROGRESS_STATUSES,
        "execution_planning_ready": EXECUTION_PLANNING_READY_STATUSES,
        "execution_authority": EXECUTION_AUTHORITY_STATUSES,
        "execution_result": EXECUTION_RESULT_STATUSES,
        "investigation_review_pending": INVESTIGATION_REVIEW_PENDING_STATUSES,
        "investigation_wait": INVESTIGATION_WAIT_STATUSES,
        "report_approved": REPORT_APPROVED_STATUSES,
        "report_returned": REPORT_RETURNED_STATUSES,
        "report_stopped": REPORT_STOPPED_STATUSES,
        "report_published": REPORT_PUBLISHED_STATUSES,
    }
)


class JourneyReview(_ReviewObservation):
    # A Fork child's own origin classifies it; a fresh Run has none.
    run_provenance: Literal["causal_child_or_fork"] | None = None
    preparation_review: PreparationReview = PreparationReview()
    journey_id: StrictStr
    journey_file: StrictStr
    status: RunStatus
    request: RequestReview
    registered_input_set: RegisteredInputSetReview
    inspection: InspectionReview | None
    input_selection: InputSelection
    run_papers: RunPapers
    attempt: AttemptEvidenceReview | None
    attempts: tuple[AttemptEvidenceReview, ...]
    proposal_plan: ProposalPlanReview | None
    requirement_resolution: RequirementResolutionProjection | None
    agent_effort_review: AgentEffortReviewProjection
    agent_work_review: AgentWorkReviewProjection
    model_availability_review: ModelAvailabilityReviewProjection
    model_contact_postmortem: ModelContactPostmortem | None
    planning_hitl: PlanningHitlProjection
    reference_identity: ReferenceIdentityProjection | None
    reference_facts: ReferenceFactsProjection | None
    execution_planning: ExecutionPlanningProjection | None
    execution_hitl: ExecutionHitlProjection | None
    investigation: InvestigationProjection
    investigation_download_review: InvestigationDownloadReviews = (
        InvestigationDownloadReviews()
    )
    run_health: RunHealthState = RunHealthState()
    investigation_request: CapturedInvestigationRequest | None = None
    investigation_evidence: InvestigationCollectionEvidenceRecorded | None = None
    finalization: FinalizationProjection
    grounded_report: GroundedReportProjection
    failure_code: RunFailureCode | StrictStr | None
    failure_source_event_id: StrictStr | None
    lineage: tuple[JourneyLineageItem, ...] = Field(min_length=3)


class SupplyInvestigationRequestAction(FrozenContract):
    """The one legal action at request-needed: supply the researcher's SNP request."""

    action: Literal["supply_investigation_request"] = "supply_investigation_request"
    supplied_field: Literal["request_file"] = "request_file"
    effect: Literal["begin_investigation_with_supplied_request"] = (
        "begin_investigation_with_supplied_request"
    )
    request_schema: dict[str, JsonValue] = Field(
        default_factory=investigation_request_document_schema
    )


class InvestigationRequestNeeded(ParabricksExecutionOutcome):
    """A verified Execution whose Run has no researcher SNP request to investigate.

    It keeps the verified outcome's fields so the coordinator still stops here;
    its own schema and status say the Run's next step is supplying the request.
    """

    schema_version: Literal["bio-neb.investigation-request-needed/1"] = (
        "bio-neb.investigation-request-needed/1"
    )
    status: Literal["investigation_request_needed"] = "investigation_request_needed"
    legal_actions: tuple[SupplyInvestigationRequestAction, ...] = (
        SupplyInvestigationRequestAction(),
    )


JourneyObservation = JourneyReview | InvestigationRequestNeeded | ParabricksResult


def _as_dict(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def _is_execution_activity(event: Mapping[str, Any]) -> bool:
    fact = validate_activity_fact(event)
    return fact is not None and fact.stage in {
        "verify_execution_inputs",
        "prepare_reference_cpu",
        "execute_prepared_reference_h100",
    }


def _single(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    selected = [event for event in events if event.get("event_type") == event_type]
    if len(selected) != 1:
        raise ValueError(f"Journey must contain exactly one {event_type} event")
    return selected[0]


def _selected_rework_status(
    timeline: list[dict[str, Any]],
    investigation: InvestigationProjection,
    collection: InvestigationCollectionEvidenceRecorded | None,
    finalization: FinalizationProjection,
) -> str | None:
    """Later Investigation work takes precedence over earlier report history."""
    if (
        investigation.pending_request is not None
        and investigation.pending_request.proposal is not None
        and investigation.pending_request.proposal.unresolved_request_member_ids
    ):
        return "investigation_review_pending"
    if not investigation.revisions:
        return None
    stage_types = {
        "InvestigationCollectionRevisionRecorded",
        "InvestigationReviewRequested",
        "InvestigationAuthorityGranted",
        "InvestigationCollectionEvidenceRecorded",
        "InvestigationAttemptFailed",
        "FinalizationBasisCompiled",
        "FinalizationReviewRecorded",
        "GroundedReportPublished",
    }
    stage = next(
        (
            event["event_type"]
            for event in reversed(timeline)
            if event["event_type"] in stage_types
        ),
        None,
    )
    if stage in {
        "InvestigationCollectionRevisionRecorded",
        "InvestigationReviewRequested",
        "InvestigationAuthorityGranted",
        "InvestigationAttemptFailed",
    }:
        return {
            "review_pending": "investigation_review_pending",
            "approved": "investigation_approved",
            "investigation_work_ready": "investigation_work_ready",
            "stopped": "investigation_stopped",
        }.get(investigation.status)
    if stage == "InvestigationCollectionEvidenceRecorded":
        # A failed item is accounted, not an ended Run (R2).
        if investigation.status == "investigation_work_ready":
            return "investigation_work_ready"
        return "investigation_evidence_ready"
    if stage in {"FinalizationBasisCompiled", "FinalizationReviewRecorded"}:
        return {
            "review_pending": "finalization_review_pending",
            "approved": "finalization_approved",
            "returned_to_investigation": "finalization_returned_to_investigation",
            "stopped": "finalization_stopped",
        }.get(finalization.status)
    return None


def _project_model_contact_postmortem(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    typed_attempt_facts: Iterable[tuple[dict[str, Any], Any]],
    availability_review: ModelAvailabilityReviewProjection,
) -> ModelContactPostmortem | None:
    """Summarize only normalized model-contact facts already admitted by review."""

    retained = tuple(typed_attempt_facts)
    requests = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelRequestRecorded)
    ]
    contacts = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelPhysicalContactStarted)
    ]
    if not contacts:
        return None
    failures = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelPhysicalContactFailed)
    ]
    contact_cancellations = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelPhysicalContactCancelled)
    ]
    retry_schedules = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelAvailabilityRetryScheduled)
    ]
    retry_completions = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelAvailabilityRetryDelayCompleted)
    ]
    retry_cancellations = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelAvailabilityRetryDelayCancelled)
    ]
    responses = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelResponseRecorded)
    ]
    request_failures = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelRequestFailed)
    ]
    request_stalls = [
        (event, fact)
        for event, fact in retained
        if isinstance(fact, ModelRequestStalled)
    ]

    requests_by_key = {
        (fact.attempt_id, fact.request_number): (event, fact)
        for event, fact in requests
    }
    if len(requests_by_key) != len(requests):
        raise ValueError("model contact postmortem repeats a logical request")
    contacts_by_id = {event["event_id"]: (event, fact) for event, fact in contacts}
    failures_by_id = {event["event_id"]: (event, fact) for event, fact in failures}
    schedules_by_id = {
        event["event_id"]: (event, fact) for event, fact in retry_schedules
    }
    completions_by_id = {
        event["event_id"]: (event, fact) for event, fact in retry_completions
    }
    if (
        len(contacts_by_id) != len(contacts)
        or len(failures_by_id) != len(failures)
        or len(schedules_by_id) != len(retry_schedules)
        or len(completions_by_id) != len(retry_completions)
    ):
        raise ValueError("model contact postmortem repeats evidence identity")

    contact_ordinals: set[tuple[str, int, int]] = set()
    for _, contact in contacts:
        key = (contact.attempt_id, contact.request_number)
        logical_request = requests_by_key.get(key)
        ordinal_key = (*key, contact.contact_ordinal)
        if (
            logical_request is None
            or contact.logical_request_event_id != logical_request[0]["event_id"]
            or ordinal_key in contact_ordinals
        ):
            raise ValueError("model contact postmortem changed request ownership")
        contact_ordinals.add(ordinal_key)
        if contact.contact_ordinal > 1:
            completed = completions_by_id.get(
                contact.retry_delay_completed_event_id or ""
            )
            if (
                completed is None
                or completed[1].attempt_id != contact.attempt_id
                or completed[1].request_number != contact.request_number
                or completed[1].logical_request_event_id
                != contact.logical_request_event_id
                or completed[1].next_contact_ordinal != contact.contact_ordinal
            ):
                raise ValueError(
                    "model contact postmortem changed retry completion ownership"
                )

    failed_contact_ids: set[str] = set()
    for _, failure in failures:
        contact = contacts_by_id.get(failure.contact_started_event_id)
        if (
            contact is None
            or failure.contact_started_event_id in failed_contact_ids
            or failure.attempt_id != contact[1].attempt_id
            or failure.request_number != contact[1].request_number
            or failure.logical_request_event_id != contact[1].logical_request_event_id
            or failure.contact_ordinal != contact[1].contact_ordinal
        ):
            raise ValueError("model contact postmortem changed failure ownership")
        failed_contact_ids.add(failure.contact_started_event_id)

    cancelled_contact_ids: set[str] = set()
    for _, cancellation in contact_cancellations:
        contact = contacts_by_id.get(cancellation.contact_started_event_id)
        if (
            contact is None
            or cancellation.contact_started_event_id in cancelled_contact_ids
            or cancellation.attempt_id != contact[1].attempt_id
            or cancellation.request_number != contact[1].request_number
            or cancellation.logical_request_event_id
            != contact[1].logical_request_event_id
            or cancellation.contact_ordinal != contact[1].contact_ordinal
        ):
            raise ValueError("model contact postmortem changed cancellation ownership")
        cancelled_contact_ids.add(cancellation.contact_started_event_id)

    responded_contact_ids: set[str] = set()
    responses_per_request: dict[tuple[str, int], int] = {}
    for _, response in responses:
        contact = contacts_by_id.get(response.physical_contact_started_event_id)
        key = (response.attempt_id, response.request_number)
        if (
            contact is None
            or response.physical_contact_started_event_id in responded_contact_ids
            or key not in requests_by_key
            or response.attempt_id != contact[1].attempt_id
            or response.request_number != contact[1].request_number
            or response.physical_contact_ordinal != contact[1].contact_ordinal
        ):
            raise ValueError("model contact postmortem changed response ownership")
        responded_contact_ids.add(response.physical_contact_started_event_id)
        responses_per_request[key] = responses_per_request.get(key, 0) + 1

    completed_schedule_ids: set[str] = set()
    for _, completion in retry_completions:
        schedule = schedules_by_id.get(completion.retry_scheduled_event_id)
        if (
            schedule is None
            or completion.retry_scheduled_event_id in completed_schedule_ids
            or completion.attempt_id != schedule[1].attempt_id
            or completion.request_number != schedule[1].request_number
            or completion.logical_request_event_id
            != schedule[1].logical_request_event_id
            or completion.delay_seconds != schedule[1].delay_seconds
        ):
            raise ValueError("model contact postmortem changed retry completion")
        completed_schedule_ids.add(completion.retry_scheduled_event_id)

    cancelled_schedule_ids: set[str] = set()
    for _, cancellation in retry_cancellations:
        schedule = schedules_by_id.get(cancellation.retry_scheduled_event_id)
        if (
            schedule is None
            or cancellation.retry_scheduled_event_id in cancelled_schedule_ids
            or cancellation.attempt_id != schedule[1].attempt_id
            or cancellation.request_number != schedule[1].request_number
            or cancellation.logical_request_event_id
            != schedule[1].logical_request_event_id
            or cancellation.delay_seconds != schedule[1].delay_seconds
        ):
            raise ValueError("model contact postmortem changed retry cancellation")
        cancelled_schedule_ids.add(cancellation.retry_scheduled_event_id)

    for _, schedule in retry_schedules:
        failure = failures_by_id.get(schedule.failed_contact_event_id)
        if (
            failure is None
            or schedule.attempt_id != failure[1].attempt_id
            or schedule.request_number != failure[1].request_number
            or schedule.logical_request_event_id != failure[1].logical_request_event_id
            or schedule.failed_contact_ordinal != failure[1].contact_ordinal
            or failure[1].policy_disposition != "automatic_retry_scheduled"
        ):
            raise ValueError("model contact postmortem changed retry ownership")

    failed_requests_by_key: dict[
        tuple[str, int], tuple[dict[str, Any], ModelRequestFailed]
    ] = {}
    for event, failure in request_failures:
        key = (failure.attempt_id, failure.request_number)
        last_contact_failure = failures_by_id.get(
            failure.last_physical_contact_event_id
        )
        if (
            key not in requests_by_key
            or key in failed_requests_by_key
            or last_contact_failure is None
            or failure.attempt_id != last_contact_failure[1].attempt_id
            or failure.request_number != last_contact_failure[1].request_number
        ):
            raise ValueError("model contact postmortem changed request failure")
        failed_requests_by_key[key] = (event, failure)
    failed_request_keys = set(failed_requests_by_key)
    stalled_request_keys: set[tuple[str, int]] = set()
    for _, stall in request_stalls:
        key = (stall.attempt_id, stall.request_number)
        stalled_contact = failures_by_id.get(stall.last_physical_contact_event_id)
        if (
            key not in requests_by_key
            or key in stalled_request_keys
            or key in failed_request_keys
            or stalled_contact is None
            or stalled_contact[1].policy_disposition != "stall_noted"
            or (stalled_contact[1].attempt_id, stalled_contact[1].request_number) != key
        ):
            raise ValueError("model contact postmortem changed request stall")
        stalled_request_keys.add(key)

    if (
        failed_contact_ids & responded_contact_ids
        or failed_contact_ids & cancelled_contact_ids
        or responded_contact_ids & cancelled_contact_ids
        or completed_schedule_ids & cancelled_schedule_ids
    ):
        raise ValueError("model contact postmortem contradicts contact disposition")

    failure_counts: dict[tuple[str, int | None, str | None, str, str], int] = {}
    for _, failure in failures:
        key = (
            failure.error_type,
            failure.http_status_code,
            failure.failure_reason,
            failure.completion_certainty,
            failure.policy_disposition,
        )
        failure_counts[key] = failure_counts.get(key, 0) + 1
    failure_groups = tuple(
        ModelContactFailureGroup(
            error_type=key[0],
            http_status_code=key[1],
            failure_class=key[2],
            completion_certainty=key[3],
            policy_disposition=key[4],
            count=count,
        )
        for key, count in sorted(
            failure_counts.items(),
            key=lambda item: (
                item[0][0],
                item[0][1] if item[0][1] is not None else -1,
                item[0][2] or "",
                item[0][3],
                item[0][4],
            ),
        )
    )
    schedule_counts: dict[tuple[int, str], int] = {}
    for _, schedule in retry_schedules:
        key = (schedule.delay_seconds, schedule.delay_source)
        schedule_counts[key] = schedule_counts.get(key, 0) + 1
    schedule_groups = tuple(
        ModelContactRetryScheduleGroup(
            delay_seconds=key[0],
            delay_source=key[1],
            count=count,
        )
        for key, count in sorted(schedule_counts.items())
    )

    contact_ids = set(contacts_by_id)
    retained_positions = {
        event["event_id"]: index for index, (event, _) in enumerate(retained)
    }
    current_request = max(
        requests,
        key=lambda item: retained_positions[item[0]["event_id"]],
    )
    current_request_key = (
        current_request[1].attempt_id,
        current_request[1].request_number,
    )
    terminal_candidates = (
        *responses,
        *request_failures,
        *request_stalls,
        *contact_cancellations,
        *retry_cancellations,
    )
    current_terminal = (
        max(
            terminal_candidates,
            key=lambda item: retained_positions[item[0]["event_id"]],
        )
        if terminal_candidates
        else None
    )
    current_terminal_key = (
        (
            current_terminal[1].attempt_id,
            current_terminal[1].request_number,
        )
        if current_terminal is not None
        else None
    )
    current_cancelled = (
        current_terminal_key == current_request_key
        and current_terminal is not None
        and isinstance(
            current_terminal[1],
            (ModelPhysicalContactCancelled, ModelAvailabilityRetryDelayCancelled),
        )
    )
    current_terminal_failure = False
    if (
        current_terminal_key == current_request_key
        and current_terminal is not None
        and isinstance(current_terminal[1], ModelRequestFailed)
    ):
        final_failure = failures_by_id.get(
            current_terminal[1].last_physical_contact_event_id
        )
        current_terminal_failure = (
            final_failure is not None
            and final_failure[1].policy_disposition == "terminal_failure"
        )
    current_request_position = retained_positions[current_request[0]["event_id"]]
    retry_receipts = {
        receipt.request_event_id: receipt
        for receipt in availability_review.receipts
        if isinstance(receipt.response, RetryModelAvailabilityResponse)
    }
    historical_failures_retried = (
        bool(failed_requests_by_key)
        and len(availability_review.requests) == len(failed_requests_by_key)
        and len(retry_receipts) == len(failed_requests_by_key)
        and all(
            retained_positions[event["event_id"]] < current_request_position
            and failure.availability_exhaustion_reason is not None
            and (
                final_failure := failures_by_id.get(
                    failure.last_physical_contact_event_id
                )
            )
            is not None
            and final_failure[1].policy_disposition == "availability_review_requested"
            and len(
                matching_requests := [
                    request
                    for request in availability_review.requests
                    if request.attempt_id == failure.attempt_id
                    and request.request_number == failure.request_number
                    and request.logical_request_event_id
                    == requests_by_key[key][0]["event_id"]
                    and request.logical_failure_event_id == event["event_id"]
                ]
            )
            == 1
            and matching_requests[0].request_event_id in retry_receipts
            for key, (event, failure) in failed_requests_by_key.items()
        )
    )
    recovered_and_admitted = (
        historical_failures_retried
        and availability_review.pending_request is None
        and availability_review.stop_event_id is None
        and current_terminal_key == current_request_key
        and current_terminal is not None
        and isinstance(current_terminal[1], ModelResponseRecorded)
        and responses_per_request.get(current_request_key) == 1
        and not cancelled_contact_ids
        and not cancelled_schedule_ids
        and all(
            responses_per_request.get(key) == 1
            for key in requests_by_key
            if key not in failed_request_keys | stalled_request_keys
        )
        and set(schedules_by_id) == completed_schedule_ids
        and failed_contact_ids | responded_contact_ids == contact_ids
    )
    all_requests_resolved = (
        bool(requests_by_key)
        and not failed_request_keys
        and not cancelled_contact_ids
        and not cancelled_schedule_ids
        and all(
            responses_per_request.get(key) == 1 or key in stalled_request_keys
            for key in requests_by_key
        )
        and failed_contact_ids | responded_contact_ids == contact_ids
        and set(schedules_by_id) == completed_schedule_ids
    )
    terminal_disposition = None
    if availability_review.pending_request is not None:
        terminal_disposition = "model_availability_review_pending"
    elif availability_review.stop_event_id is not None:
        terminal_disposition = "model_availability_stopped"
    elif all_requests_resolved:
        terminal_disposition = "all_model_requests_resolved_and_admitted"
    elif current_cancelled:
        terminal_disposition = "model_contact_cancelled"
    elif current_terminal_failure:
        terminal_disposition = "model_contact_terminal_failure"
    elif recovered_and_admitted:
        terminal_disposition = "model_availability_recovered_and_admitted"
    return ModelContactPostmortem(
        logical_request_count=sum(
            not fact.continuation_request_event_id for _, fact in requests
        ),
        physical_contact_count=len(contacts)
        - sum(failure.completion_certainty == "not_sent" for _, failure in failures),
        admitted_response_count=len(responses),
        stalled_request_count=len(request_stalls),
        failure_groups=failure_groups,
        retry_schedules=schedule_groups,
        availability_review_request_count=len(availability_review.requests),
        terminal_disposition=terminal_disposition,
    )


def _evidence_item(event: dict[str, Any]) -> JourneyEvidenceItem:
    return JourneyEvidenceItem(
        source_event_id=event["event_id"],
        event_type=event["event_type"],
        parent_event_ids=tuple(event["parent_ids"]),
        payload=event["payload"],
    )


def _is_attempt_failure_evidence(event: Mapping[str, Any]) -> bool:
    reference_fact = validate_reference_fact(event)
    return (
        event["event_type"] in {"ModelRequestFailed", "ToolExecutionFailed"}
        # A retried truncation is a correction; only a stopping one is a fault.
        or (
            event["event_type"] == "ModelCompletionTruncated"
            and event["payload"].get("feedback") is None
        )
        or (
            event["event_type"] in {"ToolCallValidated", "AgentOutputValidated"}
            and event["payload"]["status"] == "invalid"
        )
        or (
            event["event_type"] == "InputMetadataInspected"
            and event["payload"]["result"]["status"] == "failed"
        )
        or (
            reference_fact is not None
            and event["payload"]["result"]["status"] == "failed"
        )
    )


def _recorded_assignment(timeline, attempt_id):
    """The coordinator's Agent work the Attempt recorded; its ``reopen`` drives the view."""
    from bio_neb.application.agent_reentry import (
        AgentReentryRefused,
        coordinator_assignments,
        reentry_assignment,
    )

    try:
        return reentry_assignment(timeline, attempt_id, coordinator_assignments())
    except AgentReentryRefused as error:
        raise ValueError("Attempt uses an unsupported Agent role") from error


def model_contact_units(
    events: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, ...], bool]:
    """Reuse the model owner's grammar without requiring an entire Attempt to close."""
    timeline = [dict(event) for event in events]
    requests = {}
    terminals = {}
    starts = []
    seen = set()
    for index, event in enumerate(timeline):
        if event["event_id"] in seen:
            raise ValueError("model accounting repeated a Journey identity")
        seen.add(event["event_id"])
        if not event.get("event_type", "").startswith(
            (
                "ModelRequest",
                "ModelResponse",
                "ModelPhysicalContact",
                "ModelAvailabilityRetry",
            )
        ):
            continue
        fact = validate_attempt_fact(event)
        if fact is None:
            raise ValueError("model accounting contains unsupported evidence")
        key = (fact.attempt_id, fact.request_number)
        if fact.event_type == "ModelRequestRecorded":
            if key in requests:
                raise ValueError("model accounting repeated a logical request")
            requests[key] = (index, event)
        elif key not in requests:
            raise ValueError("model accounting lost its logical request")
        elif fact.event_type in MODEL_REQUEST_TERMINALS:
            if key in terminals:
                raise ValueError("model accounting repeated a logical terminal")
            terminals[key] = (index, event)
        else:
            if (
                fact.logical_request_event_id != requests[key][1]["event_id"]
                or key in terminals
            ):
                raise ValueError("model accounting changed its open logical request")
            if fact.event_type == "ModelPhysicalContactStarted":
                starts.append(event["event_id"])
    for key, (end, terminal) in terminals.items():
        begin, request = requests[key]
        if end <= begin:
            raise ValueError("model accounting terminal precedes its request")
        _validate_model_contact_grammar(
            timeline[begin : end + 1], [request], [terminal]
        )
    return tuple(f"journey-model:{event_id}" for event_id in starts), bool(
        requests
    ) and set(requests) == set(terminals)


def _validate_model_contact_grammar(selected, model_requests, model_terminals):
    """Use the shared grammar and reject physical facts outside all owned windows."""

    def physical(event):
        return event["event_type"].startswith(
            ("ModelPhysicalContact", "ModelAvailabilityRetry")
        )

    owned = set()
    for request, terminal in zip(model_requests, model_terminals, strict=True):
        begin, end = selected.index(request), selected.index(terminal)
        window = selected[begin : end + 1]
        validate_model_contact_window(window, request, terminal)
        owned.update(event["event_id"] for event in window if physical(event))
    if owned != {event["event_id"] for event in selected if physical(event)}:
        raise ValueError("Attempt contains foreign physical-contact evidence")


def _validate_activities(  # noqa: C901 - one lifecycle reader; a domain stage skips only its typed reference
    selected: list[dict[str, Any]], *, domain_stage: str | None = None
) -> None:
    """Validate typed lifecycle shape, causal order, timing, and exact terminals.

    A domain-bound Attempt's own activity (``domain_stage``) closes on the
    terminal its domain checks; every other activity closes on its typed evidence.
    """

    by_id = {event["event_id"]: event for event in selected}
    indexed = {event["event_id"]: index for index, event in enumerate(selected)}
    activities: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    active_stack: list[str] = []
    for event in selected:
        fact = validate_activity_fact(event)
        if fact is None:
            if not active_stack:
                raise ValueError("Attempt evidence exists outside an active activity")
            continue
        activities.setdefault(fact.activity_id, []).append((event, fact))
        if isinstance(fact, ActivityStarted):
            active_stack.append(fact.activity_id)
        elif not active_stack or active_stack[-1] != fact.activity_id:
            raise ValueError("activity lifecycle violates causal stack ordering")
        elif isinstance(fact, (ActivitySucceeded, ActivityFailed, ActivityCancelled)):
            active_stack.pop()
    if active_stack:
        raise ValueError("activity lifecycle has an unclosed causal owner")

    expected_references = {
        "planning_attempt": {
            "activity_succeeded": ("AttemptSucceeded", "AttemptStopped"),
            "activity_failed": "AttemptFailed",
            "activity_cancelled": "AttemptCancelled",
        },
        "model_contact": {
            "activity_succeeded": (
                "ModelResponseRecorded",
                "ModelRequestStalled",
                "ModelRequestRejected",
            ),
            "activity_failed": "ModelRequestFailed",
            "activity_cancelled": (
                "ModelPhysicalContactCancelled",
                "ModelAvailabilityRetryDelayCancelled",
            ),
        },
        "inspect_input_metadata": {
            "activity_succeeded": "InputMetadataInspected",
            "activity_failed": "ToolExecutionFailed",
        },
        "verify_sample_attestation": {
            "activity_succeeded": "SampleAttestationRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "resolve_taxon": {
            "activity_succeeded": "TaxonResolutionRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "discover_reference_assemblies": {
            "activity_succeeded": "ReferenceAssembliesRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "describe_reference": {
            "activity_succeeded": "ReferenceDescriptionRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_companion_asset_selection": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_companion_refinement": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
        "describe_reference_companion": {
            "activity_succeeded": "CompanionObservationRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "search_prebuilt_indexes": {
            "activity_succeeded": "PrebuiltIndexesListed",
            "activity_failed": "ToolExecutionFailed",
        },
        "read_prebuilt_index": {
            "activity_succeeded": "PrebuiltIndexRead",
            "activity_failed": "ToolExecutionFailed",
        },
        "suggest_assembly_with_index": {
            "activity_succeeded": "AssemblyWithIndexSuggested",
            "activity_failed": "ToolExecutionFailed",
        },
        "find_paper": {
            "activity_succeeded": "PaperSearchRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "retrieve_paper": {
            "activity_succeeded": "PaperRetrievalRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "list_compatible_analysis_recipes": {
            "activity_succeeded": "RecipeObservationRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "lookup_compatible_resource_profiles": {
            "activity_succeeded": "ResourceObservationRecorded",
            "activity_failed": "ToolExecutionFailed",
        },
        "select_recipe_candidate": {
            "activity_succeeded": "PlanningCandidateSelected",
            "activity_failed": "ToolExecutionFailed",
        },
        "select_resource_candidate": {
            "activity_succeeded": "PlanningCandidateSelected",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_recipe_selection": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_recipe_eligibility_assertion": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_resource_selection": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
        "request_resource_recipe_dependency": {
            "activity_succeeded": "PlanningHitlRequested",
            "activity_failed": "ToolExecutionFailed",
        },
    }
    references_by_stage: dict[str, list[str]] = {
        stage: [] for stage in expected_references
    }
    for retained in activities.values():
        events = [event for event, _ in retained]
        facts = [fact for _, fact in retained]
        if (
            not isinstance(facts[0], ActivityStarted)
            or sum(isinstance(fact, ActivityStarted) for fact in facts) != 1
        ):
            raise ValueError("activity lifecycle does not begin with its unique start")
        terminals = [
            fact
            for fact in facts
            if isinstance(fact, (ActivitySucceeded, ActivityFailed, ActivityCancelled))
        ]
        if len(terminals) != 1 or facts[-1] is not terminals[0]:
            raise ValueError("activity lifecycle has no unique terminal event")
        identity = (
            facts[0].activity_id,
            facts[0].adapter,
            facts[0].stage,
            facts[0].attempt,
        )
        if any(
            (fact.activity_id, fact.adapter, fact.stage, fact.attempt) != identity
            for fact in facts
        ):
            raise ValueError("activity lifecycle identity changed")
        elapsed = [fact.elapsed_ms for fact in facts]
        if elapsed != sorted(elapsed):
            raise ValueError("activity elapsed time is not monotonic")

        progress_by_task: dict[str, tuple[int, int]] = {}
        for fact in facts:
            if not isinstance(fact, ActivityProgress) or fact.completed is None:
                continue
            previous = progress_by_task.get(fact.current_task)
            if previous is not None and (
                fact.total != previous[1] or fact.completed < previous[0]
            ):
                raise ValueError("activity progress counters are not monotonic")
            progress_by_task[fact.current_task] = (fact.completed, fact.total)
        if any(
            isinstance(fact, ActivityHeartbeat)
            and set(events[index]["payload"])
            != {"activity_id", "adapter", "stage", "attempt", "elapsed_ms"}
            for index, fact in enumerate(facts)
        ):
            raise ValueError("activity heartbeat must assert liveness only")

        terminal = terminals[0]
        if terminal.stage == domain_stage:
            continue
        reference_id = (
            terminal.result_event_id
            if isinstance(terminal, ActivitySucceeded)
            else terminal.failure_event_id
            if isinstance(terminal, ActivityFailed)
            else terminal.cancellation_event_id
        )
        reference = by_id.get(reference_id)
        terminal_event = events[-1]
        if (
            reference is None
            or indexed[events[0]["event_id"]] >= indexed[reference_id]
            or indexed[reference_id] >= indexed[terminal_event["event_id"]]
            or (
                reference["event_type"]
                not in expected_references[terminal.stage][terminal.event_type]
                if isinstance(
                    expected_references[terminal.stage][terminal.event_type], tuple
                )
                else reference["event_type"]
                != expected_references[terminal.stage][terminal.event_type]
            )
        ):
            raise ValueError(
                "activity terminal lacks its exact typed evidence reference"
            )
        references_by_stage[terminal.stage].append(reference_id)

    if not activities:
        raise ValueError("Attempt lacks activity lifecycle evidence")
    attempt_terminals = [
        event["event_id"]
        for event in selected
        if event["event_type"]
        in {"AttemptSucceeded", "AttemptFailed", "AttemptCancelled", "AttemptStopped"}
    ]
    exact_stage_evidence = {
        "planning_attempt": attempt_terminals
        if references_by_stage["planning_attempt"]
        else [],
        "model_contact": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {
                "ModelResponseRecorded",
                "ModelRequestFailed",
                "ModelRequestStalled",
                "ModelRequestRejected",
                "ModelPhysicalContactCancelled",
                "ModelAvailabilityRetryDelayCancelled",
            }
        ],
        "inspect_input_metadata": [
            event["event_id"]
            for event in selected
            if event["event_type"] in {"InputMetadataInspected", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "inspect_input_metadata"
            )
        ],
        "verify_sample_attestation": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"SampleAttestationRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "verify_sample_attestation"
            )
        ],
        "resolve_taxon": [
            event["event_id"]
            for event in selected
            if event["event_type"] in {"TaxonResolutionRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "resolve_taxon"
            )
        ],
        "discover_reference_assemblies": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"ReferenceAssembliesRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "discover_reference_assemblies"
            )
        ],
        "describe_reference": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"ReferenceDescriptionRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "describe_reference"
            )
        ],
        "request_companion_asset_selection": [
            event["event_id"]
            for event in selected
            if event["event_type"] in {"PlanningHitlRequested", "ToolExecutionFailed"}
            and event["event_id"]
            in references_by_stage["request_companion_asset_selection"]
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "request_companion_asset_selection"
            )
        ],
        "request_companion_refinement": [
            event["event_id"]
            for event in selected
            if event["event_type"] in {"PlanningHitlRequested", "ToolExecutionFailed"}
            and event["event_id"] in references_by_stage["request_companion_refinement"]
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "request_companion_refinement"
            )
        ],
        "describe_reference_companion": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"CompanionObservationRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"] == "describe_reference_companion"
            )
        ],
        **{
            stage: [
                event["event_id"]
                for event in selected
                if event["event_type"] in {fact, "ToolExecutionFailed"}
                and (
                    event["event_type"] != "ToolExecutionFailed"
                    or event["payload"]["tool_name"] == stage
                )
            ]
            for stage, fact in (
                ("search_prebuilt_indexes", "PrebuiltIndexesListed"),
                ("read_prebuilt_index", "PrebuiltIndexRead"),
                ("find_paper", "PaperSearchRecorded"),
                ("retrieve_paper", "PaperRetrievalRecorded"),
            )
        },
        "list_compatible_analysis_recipes": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"RecipeObservationRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"]
                in {"list_compatible_analysis_recipes", "reconsider_recipe_choice"}
            )
        ],
        "lookup_compatible_resource_profiles": [
            event["event_id"]
            for event in selected
            if event["event_type"]
            in {"ResourceObservationRecorded", "ToolExecutionFailed"}
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"]
                == "lookup_compatible_resource_profiles"
            )
        ],
        "select_recipe_candidate": [
            event["event_id"]
            for event in selected
            if event["event_type"] == "PlanningCandidateSelected"
            and event["payload"]["kind"] == "recipe"
        ],
        "select_resource_candidate": [
            event["event_id"]
            for event in selected
            if event["event_type"] == "PlanningCandidateSelected"
            and event["payload"]["kind"] == "resource"
        ],
        "request_recipe_selection": [
            event["event_id"]
            for event in selected
            if event["event_type"] == "PlanningHitlRequested"
            and any(
                candidate["candidate_id"].startswith("recipe-candidate-disposition:")
                for candidate in event["payload"]["candidates"]
            )
        ],
        "request_recipe_eligibility_assertion": [
            event["event_id"]
            for event in selected
            if event["event_type"] in {"PlanningHitlRequested", "ToolExecutionFailed"}
            and event["event_id"]
            in references_by_stage["request_recipe_eligibility_assertion"]
            and (
                event["event_type"] != "ToolExecutionFailed"
                or event["payload"]["tool_name"]
                == "request_recipe_eligibility_assertion"
            )
        ],
        "request_resource_selection": [
            event["event_id"]
            for event in selected
            if event["event_type"] == "PlanningHitlRequested"
            and any(
                candidate["candidate_id"].startswith("resource-candidate-disposition:")
                for candidate in event["payload"]["candidates"]
            )
        ],
        "request_resource_recipe_dependency": [
            event["event_id"]
            for event in selected
            if event["event_type"] == "PlanningHitlRequested"
            and any(
                candidate["candidate_id"].startswith("resource-recipe-dependency:")
                for candidate in event["payload"]["candidates"]
            )
        ],
    }
    if any(
        sorted(references_by_stage[stage]) != sorted(expected)
        for stage, expected in exact_stage_evidence.items()
    ):
        raise ValueError("activity lifecycle does not close exact observed evidence")


def _validate_compiled_checkpoint(
    timeline: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    *,
    role: AttemptTerminalRole,
    attempt_id: str,
) -> None:
    prefix = timeline[: timeline.index(checkpoint) + 1]
    disposition = project_attempt_terminal(role, attempt_id, prefix)
    if isinstance(disposition, ContinueAttempt):
        raise TypeError("Attempt checkpoint precedes its terminal Journey frontier")
    payload = checkpoint["payload"]
    expected_field = {
        "registered_inputs_inspected": "inspection_event_id",
        "planning_hitl_requested": "planning_hitl_request_event_id",
        "reference_identity": "reference_evidence_event_id",
        "reference_facts": "reference_evidence_event_id",
        "execution_planning_ready": "execution_planning_evidence_event_id",
        "target_answer_read": "target_answer_read_event_id",
        "assembly_suggested": "assembly_suggestion_event_id",
        "paper_review_requested": "paper_review_request_event_id",
        "paper_reading_due": "proposal_plan_event_id",
    }.get(disposition.checkpoint)
    if (
        expected_field is None
        or payload.get("checkpoint") != disposition.checkpoint
        or payload.get(expected_field) != disposition.source_event_id
    ):
        raise ValueError("Attempt checkpoint changed its compiled terminal source")


def _tool_fact_types(agent_role: AgentRole) -> set[str]:
    """The registry's Tool fact kinds for one role: its results and stage failures."""
    return {
        kind
        for contract in function_tool_contracts(agent_role)
        for kind in (
            contract.stage_result_event_type,
            contract.stage_failure_event_type,
        )
        if kind is not None
    }


def _validate_tool_fact_receipts(
    selected: list[dict[str, Any]],
    *,
    agent_role: AgentRole,
    by_id: Mapping[str, Mapping[str, Any]],
    exempt_ids: set[str],
) -> None:
    """Bind each registered Tool fact to exactly one terminal receipt, any outcome.

    A succeeded call cites its result; a failed call cites its stage failure or
    holds the facts recorded within it by exact lineage. The domain's own
    terminal record (``exempt_ids``) is not a Tool fact even when it shares a
    registered kind.
    """
    fact_types = _tool_fact_types(agent_role)
    cited = Counter(
        receipt["payload"]["stage_result_event_id"]
        for receipt in selected
        if receipt["event_type"] == "ObservedToolExecutionSucceeded"
    )
    failed = [e for e in selected if e["event_type"] == "ObservedToolExecutionFailed"]
    cited.update(
        receipt["payload"]["stage_failure_event_id"]
        for receipt in failed
        if receipt["payload"].get("stage_failure_event_id") is not None
    )
    for event in selected:
        if event["event_type"] not in fact_types or event["event_id"] in exempt_ids:
            continue
        # A failed call citing this fact as its stage failure binds it only once.
        held = sum(
            recorded_within_call(
                event, by_id[receipt["payload"]["started_event_id"]], by_id
            )
            for receipt in failed
            if receipt["payload"].get("stage_failure_event_id") != event["event_id"]
        )
        if cited[event["event_id"]] + held != 1:
            raise ValueError("Tool fact is not bound to exactly one terminal receipt")


def _project_one_attempt_evidence(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    timeline: list[dict[str, Any]],
    registration_event: dict[str, Any] | None,
    selected: list[dict[str, Any]],
    *,
    planning_context: PrefixSnapshot,
) -> AttemptEvidenceReview:
    starts = [event for event in selected if event["event_type"] == "AttemptStarted"]
    if len(starts) != 1:
        raise ValueError("Attempt window requires exactly one Attempt start")
    started = starts[0]
    if len(started["parent_ids"]) != 1:
        raise ValueError("Attempt start must have one lifecycle parent")
    by_id = {event["event_id"]: event for event in timeline}
    lifecycle_start = by_id.get(started["parent_ids"][0])
    if lifecycle_start is None or lifecycle_start["event_type"] != "activity_started":
        raise ValueError("Attempt start lacks lifecycle evidence")
    from bio_neb.recording import is_contact_evidence

    semantic_selected = [event for event in selected if not is_contact_evidence(event)]
    for index, event in enumerate(semantic_selected):
        if index == 0:
            if len(event["parent_ids"]) != 1:
                raise ValueError("Attempt activity start requires one causal parent")
            continue
        if event["parent_ids"] != [semantic_selected[index - 1]["event_id"]]:
            raise ValueError("Attempt evidence must form one causal Journey chain")
    attempt_id = started["payload"]["attempt_id"]
    agent_role = started["payload"]["agent_role"]
    assignment = _recorded_assignment(timeline, attempt_id)
    domain_bound = assignment.reopen.domain_bound
    if domain_bound is None:
        # The Run's registered slice binds the Attempt.
        if registration_event is None or not _has_ancestor(
            timeline,
            ancestor_id=registration_event["event_id"],
            descendant_id=started["parent_ids"][0],
        ):
            raise ValueError("Attempt is outside registration lineage")
        registration = validate_fact(registration_event)
        if not isinstance(registration, RunInputsRegistered):
            raise ValueError("Attempt registration evidence is invalid")
        registered_input_set_id = registration.input_set_id
        runtime_selections = retained_runtime_selections(
            registration_event["payload"]["runtime_selections"]
        )
    else:
        binding = domain_bound.binding(timeline, selected)
        registered_input_set_id = binding.input_set_id
        runtime_selections = binding.runtime_selections

    attempt_facts = []
    for event in selected:
        fact = validate_attempt_fact(event)
        if fact is not None:
            attempt_facts.append((event, fact))
    continuation_policy = ContinuationPolicy.model_validate(
        started["payload"].get("continuation_policy")
    )
    instruction_reference = retained_instruction_reference(started)
    effective_policy = retained_invocation_policy(
        started["payload"], runtime_selections, agent_role=agent_role
    )
    validate_continuation(timeline, attempt_id=attempt_id, policy=continuation_policy)
    if started["payload"]["input_set_id"] != registered_input_set_id:
        raise ValueError("Attempt references a different registered input set")
    if any(fact.attempt_id != attempt_id for _, fact in attempt_facts):
        raise ValueError("Attempt evidence identities disagree")
    model_requests = [
        event for event in selected if event["event_type"] == "ModelRequestRecorded"
    ]
    if [event["payload"]["request_number"] for event in model_requests] != list(
        range(1, len(model_requests) + 1)
    ):
        raise ValueError("model request numbers must be contiguous")
    model_terminals = [
        event for event in selected if event["event_type"] in MODEL_REQUEST_TERMINALS
    ]
    if len(model_terminals) != len(model_requests):
        cancelled_requests = [
            event for event in selected if event["event_type"] == "AttemptCancelled"
        ]
        if not (
            len(cancelled_requests) == 1
            and len(model_terminals) == len(model_requests) - 1
        ):
            raise ValueError("each model request requires one retained outcome")
    selected_endpoint = runtime_selections.model_for_policy(effective_policy)
    expected_endpoint = {
        "provider": selected_endpoint.provider,
        "model": selected_endpoint.model,
        "contacted_url": selected_endpoint.endpoint,
    }
    for event in (*model_requests, *model_terminals):
        if event["payload"].get("nebius_model_endpoint") != expected_endpoint:
            raise ValueError("model evidence changed its registered Nebius selection")
    if effective_policy is not None:
        for event in model_requests:
            validate_retained_model_request(event["payload"], effective_policy)
        endpoint = retained_endpoint_policy(effective_policy, runtime_selections)
        check_stall_notes(selected, endpoint.correction_wording)
    for index, (model_request, terminal) in enumerate(
        zip(model_requests, model_terminals)  # noqa: B905 - Preserve existing zip truncation; contract-base migration adds no options.
    ):
        request_number = model_request["payload"]["request_number"]
        if terminal["payload"]["request_number"] != request_number:
            raise ValueError("model request and response numbers disagree")
        if terminal["event_type"] == "ModelResponseRecorded":
            request_compatible = model_request["payload"]["compatible_adapter_evidence"]
            response_compatible = terminal["payload"]["compatible_adapter_evidence"]
            if (
                request_compatible["model_name"] != response_compatible["model_name"]
                or request_compatible["model_system"]
                != response_compatible["model_system"]
            ):
                raise ValueError(
                    "model request and response compatible identities disagree"
                )
        request_index = selected.index(model_request)
        terminal_index = selected.index(terminal)
        next_index = (
            selected.index(model_requests[index + 1])
            if index + 1 < len(model_requests)
            else len(selected)
        )
        if not request_index < terminal_index < next_index:
            raise ValueError("model outcomes are outside request order")
    if len(model_terminals) == len(model_requests):
        _validate_model_contact_grammar(selected, model_requests, model_terminals)
    else:
        if model_terminals:
            prefix = selected[: selected.index(model_terminals[-1]) + 1]
            _validate_model_contact_grammar(
                prefix, model_requests[:-1], model_terminals
            )
    terminals = [
        event
        for event, _ in attempt_facts
        if event["event_type"]
        in {"AttemptSucceeded", "AttemptFailed", "AttemptCancelled", "AttemptStopped"}
    ]
    if len(terminals) > 1:
        raise ValueError("Attempt has more than one terminal outcome")
    outcome = (
        "succeeded"
        if terminals and terminals[0]["event_type"] == "AttemptSucceeded"
        else "failed"
        if terminals and terminals[0]["event_type"] == "AttemptFailed"
        else "stopped"
        if terminals and terminals[0]["event_type"] == "AttemptStopped"
        else "cancelled"
        if terminals
        else "incomplete"
    )
    checkpoints = [
        event
        for event, _ in attempt_facts
        if event["event_type"] == "SliceCheckpointReached"
    ]
    output_validations = [
        event for event in selected if event["event_type"] == "AgentOutputValidated"
    ]
    role = agent_role
    # An administrative, Tool-driven role retains no model-authored typed output.
    if not assignment.reopen.structured_output:
        if output_validations:
            raise ValueError(
                "administrative Attempt cannot retain model-authored terminal output"
            )
        if any(
            _compatible_model_request(event).get("output_tools")
            for event in model_requests
        ):
            raise ValueError(
                "administrative Attempt exposed a terminal output Tool to the model"
            )
    # Fixed by the Attempt's own events and the prefix before them: read once per
    # owner for this exact window.
    window = (started["event_id"], selected[-1]["event_id"])
    ask_composer = planning_context.checked_once(
        ("attempt_turns", *window),
        lambda: validate_action_turns(
            selected,
            role=role,
            assignment=assignment,
            timeline=timeline,
        ),
    )
    domain_terminal_ids: set[str] = set()
    if domain_bound is not None:
        # The domain that bound the Attempt checked its terminal.
        if checkpoints or any(
            event["event_type"] == "AttemptSucceeded" for event in terminals
        ):
            raise ValueError("domain-bound Attempt retained a slice terminal")
        reading = domain_bound.terminal(planning_context, selected)
        outcome = reading.outcome
        domain_terminal_ids = {reading.terminal_event_id, *reading.domain_event_ids}
    _validate_tool_fact_receipts(
        selected, agent_role=agent_role, by_id=by_id, exempt_ids=domain_terminal_ids
    )
    if assignment.reopen.validate_facts is not None:
        assignment.reopen.validate_facts(planning_context, selected)
    if domain_bound is None and outcome == "succeeded":
        if len(checkpoints) != 1:
            raise ValueError("successful Attempt requires one slice checkpoint")
        if terminals[0]["payload"]["checkpoint_event_id"] != checkpoints[0]["event_id"]:
            raise ValueError("Attempt terminal references a different checkpoint")
        checkpoint = checkpoints[0]
        if checkpoint["payload"]["input_set_id"] != registered_input_set_id:
            raise ValueError("slice checkpoint references a different input set")
        if not assignment.reopen.compiled_checkpoint:
            raise ValueError("Attempt uses an unsupported Agent role")
        _validate_compiled_checkpoint(
            timeline, checkpoint, role=role, attempt_id=attempt_id
        )
    elif checkpoints:
        raise ValueError(
            "failed or incomplete Attempt cannot retain a slice checkpoint"
        )

    if domain_bound is None and outcome == "stopped":
        if terminals[0]["payload"]["code"] == "operational_failure":
            validate_operational_terminal(timeline, terminals[0])
        elif terminals[0]["payload"]["code"] == "effort_exhausted":
            validate_effort_terminal(timeline, terminals[0])
        else:
            validate_agent_work_terminal(timeline, terminals[0])
    validate_attempt_cancellation(
        selected,
        next((t for t in terminals if t["event_type"] == "AttemptCancelled"), None),
        model_requests,
        model_terminals,
    )
    if domain_bound is None and outcome == "cancelled":
        if any(event["event_type"] == "ModelRequestFailed" for event in selected):
            raise ValueError("cancelled Attempt manufactured a logical model failure")
    if domain_bound is None and outcome == "failed":
        terminal = terminals[0]
        terminal_fact = validate_attempt_fact(terminal)
        if not isinstance(terminal_fact, AttemptFailed):
            raise ValueError("Attempt failure terminal is invalid")
        source = by_id.get(terminal_fact.source_event_id)
        if source is None or source not in selected:
            raise ValueError("Attempt failure source is outside its causal lineage")
        if source["event_type"] == "ObservedToolExecutionFailed":
            validate_tool_failure_source(
                selected, source=source, terminal=terminal, attempt_id=attempt_id
            )
        elif selected.index(source) >= selected.index(terminal) or not _has_ancestor(
            selected,
            ancestor_id=source["event_id"],
            descendant_id=terminal["event_id"],
        ):
            raise ValueError("Attempt failure source is outside its causal lineage")
        validate_attempt_fact(source) or validate_fact(source)
        if terminal_fact.code == "agent_output_invalid":
            if source["event_type"] not in {
                "AgentOutputValidated",
                "ModelResponseRecorded",
            } or (
                source["event_type"] == "AgentOutputValidated"
                and source["payload"]["status"] != "invalid"
            ):
                raise ValueError(
                    "Attempt failure does not reference its exact terminal model evidence"
                )
        elif terminal_fact.code == "model_availability_exhausted":
            if (
                source["event_type"] != "ModelRequestFailed"
                or source["payload"].get("availability_exhaustion_reason") is None
            ):
                raise ValueError(
                    "availability exhaustion lacks its logical model terminal"
                )
        elif terminal_fact.code == "invariant":
            # Our bug names the Attempt's last retained response, or its start,
            # or the Tool failure that raised it.
            if source["event_type"] not in {
                "AttemptStarted",
                "ModelResponseRecorded",
                "ObservedToolExecutionFailed",
            }:
                raise ValueError(
                    "Attempt invariant does not reference its retained evidence"
                )
        # A refused action beyond its allowance and a cut-off completion stop for
        # the Agent work review; neither is an Attempt failure source.
        elif source["event_type"] not in {
            "ModelRequestFailed",
            "ModelResponseRecorded",
            "ToolExecutionFailed",
            "ObservedToolExecutionFailed",
            "AgentOutputValidated",
        } or (
            source["event_type"] == "AgentOutputValidated"
            and source["payload"]["status"] != "invalid"
        ):
            raise ValueError(
                "Attempt failure does not reference exact typed failure evidence"
            )
        # A failed call the model was shown, and its inner stage failure, is a
        # corrected fault like a refused call, not this terminal's.
        shown_calls = [
            by_id[failed["payload"]["started_event_id"]]
            for failed in selected
            if failed["event_type"] == "ObservedToolExecutionFailed"
            and any(
                refusal["event_type"] == "ToolExecutionRefused"
                and refusal["payload"]["failed_event_id"] == failed["event_id"]
                for refusal in selected
            )
        ]
        failure_evidence = [
            event
            for event in selected
            if (_is_attempt_failure_evidence(event) or event is source)
            and not (
                event is not source
                and event["event_type"] == "ToolExecutionFailed"
                and any(
                    recorded_within_call(event, started, by_id)
                    for started in shown_calls
                )
            )
        ]
        permitted = [source]
        if source["event_type"] == "ModelResponseRecorded":
            permitted = [
                event
                for event in failure_evidence
                if event is source
                or (
                    selected.index(event) < selected.index(source)
                    and _has_ancestor(
                        selected,
                        ancestor_id=event["event_id"],
                        descendant_id=source["event_id"],
                    )
                )
            ]
        if source["event_type"] == "ModelRequestFailed":
            permitted = [
                event
                for event in failure_evidence
                if event is source
                or (
                    event["event_type"] in {"ToolCallValidated", "AgentOutputValidated"}
                    and event["payload"]["status"] == "invalid"
                    and selected.index(event) < selected.index(source)
                    and _has_ancestor(
                        selected,
                        ancestor_id=event["event_id"],
                        descendant_id=source["event_id"],
                    )
                )
            ]
        if source["event_type"] == "ObservedToolExecutionFailed":
            # Refused actions the Attempt corrected before the failed call, and
            # that call's own inner stage failure of the same mechanical cause.
            tool_started = by_id.get(source["payload"]["started_event_id"])
            permitted = [
                event
                for event in failure_evidence
                if event is source
                or (
                    selected.index(event) < selected.index(source)
                    and (
                        (
                            event["event_type"]
                            in {"ToolCallValidated", "AgentOutputValidated"}
                            and event["payload"]["status"] == "invalid"
                            and _has_ancestor(
                                selected,
                                ancestor_id=event["event_id"],
                                descendant_id=terminal["event_id"],
                            )
                        )
                        or (
                            event["event_type"] == "ToolExecutionFailed"
                            and tool_started is not None
                            and event["payload"]["error_type"]
                            == source["payload"]["mechanical_error_type"]
                            and recorded_within_call(event, tool_started, by_id)
                        )
                    )
                )
            ]
        if failure_evidence != permitted:
            raise ValueError(
                "Attempt contains failure evidence outside its active causal chain"
            )

    _validate_activities(
        selected,
        domain_stage=None if domain_bound is None else assignment.activity_stage,
    )
    activities: dict[str, list[dict[str, Any]]] = {}
    for event in selected:
        fact = validate_activity_fact(event)
        if fact is not None:
            activities.setdefault(fact.activity_id, []).append(event)
    # An activity that failed inside a call shown to the model is a corrected
    # fault, like a refused call (Standards r1 handoff).
    shown_calls = [
        by_id[failed["payload"]["started_event_id"]]
        for failed in selected
        if failed["event_type"] == "ObservedToolExecutionFailed"
        and any(
            refusal["event_type"] == "ToolExecutionRefused"
            and refusal["payload"]["failed_event_id"] == failed["event_id"]
            for refusal in selected
        )
    ]
    if (
        domain_bound is None
        and outcome == "succeeded"
        and any(
            events[-1]["event_type"] != "activity_succeeded"
            and not (
                events[-1]["event_type"] == "activity_failed"
                and events[-1]["payload"]["failure_event_id"] in by_id
                and any(
                    recorded_within_call(
                        by_id[events[-1]["payload"]["failure_event_id"]],
                        started,
                        by_id,
                    )
                    for started in shown_calls
                )
            )
            for events in activities.values()
        )
    ):
        raise ValueError("successful Attempt contains a failed activity")

    # The composer is asked last, so a changed admission or domain fact keeps its
    # own refusal.
    planning_context.checked_once(("attempt_transcripts", *window), ask_composer)
    items = tuple(_evidence_item(event) for event in selected)

    def by_type(names: set[str]) -> tuple[JourneyEvidenceItem, ...]:
        return tuple(item for item in items if item.event_type in names)

    return AttemptEvidenceReview(
        attempt_id=attempt_id,
        instruction_identity=started["payload"]["instruction_identity"],
        instruction_artifact=instruction_reference,
        effective_invocation_policy=effective_policy,
        invocation_policy_id=started["payload"].get("invocation_policy_id"),
        invocation_overrides=started["payload"].get("invocation_overrides"),
        agent_role=started["payload"]["agent_role"],
        outcome=outcome,
        events=items,
        model_events=by_type(
            {
                "ModelRequestRecorded",
                "ModelTransportRequestRecorded",
                "ModelTransportResponseRecorded",
                "ModelPhysicalContactStarted",
                "ModelPhysicalContactFailed",
                "ModelPhysicalContactCancelled",
                "ModelAvailabilityRetryScheduled",
                "ModelAvailabilityRetryDelayCompleted",
                "ModelAvailabilityRetryDelayCancelled",
                "ModelTurnCancelled",
                "ModelResponseRecorded",
                "ModelRequestFailed",
                "ModelRequestStalled",
                "ModelRequestRejected",
            }
        ),
        tool_events=tuple(
            item
            for item in by_type({"ToolCallValidated", *_tool_fact_types(agent_role)})
            if item.source_event_id not in domain_terminal_ids
        ),
        receipt_events=by_type(
            {
                "ObservedToolExecutionStarted",
                "ObservedToolExecutionSucceeded",
                "ObservedToolExecutionFailed",
            }
        ),
        lifecycle_events=tuple(
            item for item in items if item.event_type.startswith("activity_")
        ),
        validation_events=by_type(
            {
                "ToolCallValidated",
                "AgentOutputValidated",
            }
        ),
        checkpoint_events=by_type({"SliceCheckpointReached"}),
    )


def _attempt_windows(
    timeline: list[dict[str, Any]], starts: list[dict[str, Any]]
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Each Attempt's exact lifecycle window and its own events, in Journey order.

    A slice Attempt composes through the coordinator's bridge, so its window
    overlaps no other. Domain-bound Attempts the coordinator runs together
    (#105) may interleave with each other; each then reads only its own events.
    """
    from bio_neb.agentic_journey.attempt_journey import (
        attempt_events,
        attempt_lifecycle_window,
    )

    windows = []
    for started in starts:
        begin, end = attempt_lifecycle_window(timeline, started)
        bound = (
            _recorded_assignment(
                timeline, started["payload"]["attempt_id"]
            ).reopen.domain_bound
            is not None
        )
        windows.append((begin, end, started, bound))
    if [w[:2] for w in windows] != sorted(w[:2] for w in windows):
        raise ValueError("Attempt order disagrees with Journey order")
    for index in range(len(windows) - 1):
        for later in windows[index + 1 :]:
            if later[0] >= windows[index][1]:
                break
            if not (windows[index][3] and later[3]):
                raise ValueError("Attempt order disagrees with Journey order")
    return [
        (
            begin,
            end,
            attempt_events(timeline, started) if bound else timeline[begin:end],
        )
        for begin, end, started, bound in windows
    ]


def _project_attempts_evidence(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    timeline: list[dict[str, Any]],
    registration_event: dict[str, Any],
    *,
    planning_context: PrefixSnapshot,
) -> tuple[tuple[AttemptEvidenceReview, ...], tuple[AttemptEvidenceReview, ...]]:
    """Project every Attempt through the one view; return all and the Run's slice.

    An Attempt in the Run's registered slice composes through the coordinator's
    exact re-entry bridge from the slice Attempt before it. A domain-bound
    Attempt's composition is its domain's, checked when its assignment reads its
    binding and terminal back.
    """
    requirement_review_ids = {
        event["event_id"]
        for event in timeline
        if event.get("event_type") == "PlanningHitlRequested"
        and event.get("payload", {}).get("owner")
        in {"requirement_resolution", "assembly_suggestion"}
    }
    starts = [event for event in timeline if event["event_type"] == "AttemptStarted"]
    if not starts:
        if any(
            validate_attempt_fact(event) is not None
            or validate_reference_fact(event) is not None
            or validate_reference_facts_fact(event) is not None
            or validate_execution_planning_fact(event) is not None
            or (
                event["event_type"] == "PlanningHitlRequested"
                and event["event_id"] not in requirement_review_ids
            )
            or event["event_type"]
            in {
                "ModelAvailabilityReviewRequested",
                "AgentEffortReviewRequested",
                "AgentWorkReviewRequested",
            }
            for event in timeline
        ):
            raise ValueError("Attempt evidence has no owning Attempt start")
        if any(
            isinstance(event.get("event_type"), str)
            and event["event_type"].startswith("activity_")
            and not _is_execution_activity(event)
            for event in timeline
        ):
            raise ValueError("activity lifecycle evidence has no owning Attempt")
        return (), ()
    windows = _attempt_windows(timeline, starts)
    attempts: list[AttemptEvidenceReview] = []
    projected: list[AttemptEvidenceReview] = []
    projected_owned_ids: set[str] = set()
    # A domain-bound Attempt the coordinator runs between slice Attempts (a cited
    # paper's reading items) is checked by its own projection; its events stand in
    # the slice's bridge.
    bound_owned_ids = {
        event["event_id"]
        for _begin, _end, selected in windows
        if _recorded_assignment(
            timeline,
            next(
                e["payload"]["attempt_id"]
                for e in selected
                if e["event_type"] == "AttemptStarted"
            ),
        ).reopen.domain_bound
        is not None
        for event in selected
    }
    previous_terminal = registration_event["event_id"]
    previous_terminal_index = timeline.index(registration_event)
    for begin, end, selected in windows:
        started = next(
            event for event in selected if event["event_type"] == "AttemptStarted"
        )
        if (
            _recorded_assignment(
                timeline, started["payload"]["attempt_id"]
            ).reopen.domain_bound
            is not None
        ):
            attempts.append(
                _project_one_attempt_evidence(
                    timeline,
                    registration_event,
                    selected,
                    planning_context=planning_context,
                )
            )
            continue
        bridge = timeline[previous_terminal_index + 1 : begin]
        causal_head = previous_terminal
        for event in bridge:
            if event["event_type"] == "PassStarted":
                # The coordinator interval is not an Attempt or its authority.
                continue
            elif event["event_id"] in bound_owned_ids:
                # Its own Attempt's projection checks it (see above).
                pass
            elif (
                event["event_type"] == "ModelAvailabilityStopped"
                and event["payload"]["request_event_id"] in bound_owned_ids
            ):
                # A member's availability stop ends only its member; the
                # availability projection checks it against its answer.
                pass
            elif event["event_type"] in {"ReadingItemNotRead", "PaperAnswerRecorded"}:
                # The coordinator settles a reading; its fold checks it.
                from bio_neb.paper_reading.journey import project_paper_readings

                project_paper_readings(timeline[: timeline.index(event) + 1])
            elif event["event_type"] == "PassClosed":
                pass
            elif event["event_type"] == "RuntimeConfigurationFailed":
                # A dispatch's configuration fault: a side leaf at the head it
                # admitted, checked with the Run's facts; the fixed Run resumes
                # from that head (I-65).
                if event["parent_ids"] != [causal_head]:
                    raise ValueError("configuration failure breaks exact re-entry")
            elif event["event_id"] in requirement_review_ids:
                if event["parent_ids"] != [causal_head]:
                    raise ValueError(
                        "requirement-resolution Review breaks exact re-entry"
                    )
            elif event["event_type"] == "ReferenceSourceAdmitted":
                admitted = validate_reference_source_fact(event)
                if not isinstance(admitted, ReferenceSourceAdmitted) or event[
                    "parent_ids"
                ] != [causal_head]:
                    raise ValueError(
                        "Reference source admission breaks exact revision re-entry"
                    )
            elif event["event_type"] == "ExecutionPackageCompiled":
                compiled = validate_execution_package_fact(event)
                if (
                    not isinstance(compiled, ExecutionPackageCompiled)
                    or causal_head not in event["parent_ids"]
                ):
                    raise ValueError("Execution package breaks exact revision re-entry")
            elif event["event_type"] == "PlanningHitlResponded":
                request_event_id = event["payload"].get("request_event_id")
                if (
                    event["parent_ids"]
                    not in (
                        [request_event_id],
                        list(dict.fromkeys((request_event_id, causal_head))),
                    )
                    or request_event_id
                    not in projected_owned_ids | requirement_review_ids
                ):
                    raise ValueError(
                        "Planning HITL response breaks exact Attempt re-entry"
                    )
            elif event["event_type"] == "AgentFailureRetryPreset":
                # A researcher's retry preset; the answers it gives check it.
                from bio_neb.agentic_journey.operational_review import (
                    AgentFailureRetryPreset,
                )

                AgentFailureRetryPreset.model_validate(event["payload"])
                if event["parent_ids"] != [causal_head]:
                    raise ValueError("retry preset breaks exact re-entry")
            elif event["event_type"] in REVIEW_RESPONSE_KINDS:
                validate_review_answer_parentage(timeline, event)
                if (
                    event["payload"]["request_event_id"]
                    not in projected_owned_ids | bound_owned_ids
                ):
                    raise ValueError("Agent review response lacks its owning Attempt")
            elif event["event_type"] in {
                "InputMetadataInspected",
                "InputInspectionReviewRequested",
                "InputInspectionReviewAnswered",
            }:
                from bio_neb.application.input_journey import project_input_selection

                project_input_selection(timeline[:begin])
            elif event["event_type"] in {
                # The coordinator's request checks its own lineage.
                "PreparationInformationRequested",
                "PreparationInformationAnswered",
            }:
                project_preparation_review(timeline[:begin])
            elif event["event_type"] == "ParabricksConfigurationRegistered":
                from bio_neb.parabricks.configuration import (
                    ParabricksConfigurationRegistered,
                )

                ParabricksConfigurationRegistered.model_validate(event["payload"])
            elif event["event_type"] == "InvestigationRequestRegistered":
                from bio_neb.investigation.investigation_request import (
                    project_captured_investigation_requests,
                )

                project_captured_investigation_requests(timeline[:begin])
            elif event["event_type"] in PAPER_PAYLOADS:
                # The coordinator's paper intake checks its own lineage.
                project_run_papers(timeline[:begin])
            elif event["event_type"] == "ExecutionProposalSubmitted":
                submitted = validate_execution_fact(event)
                if not isinstance(submitted, ExecutionProposalSubmitted) or event[
                    "parent_ids"
                ] != [causal_head]:
                    raise ValueError(
                        "Execution proposal breaks exact revision re-entry"
                    )
            elif event["event_type"] == "ExecutionHitlResponded":
                responded = validate_execution_fact(event)
                if (
                    not isinstance(responded, ExecutionHitlResponded)
                    or not isinstance(responded.response, ReviseExecutionResponse)
                    or event["parent_ids"] != [responded.request_event_id]
                    or responded.request_event_id
                    != review_head(timeline[: timeline.index(event)])["event_id"]
                ):
                    raise ValueError("Execution revision breaks exact Attempt re-entry")
            else:
                raise ValueError(
                    "Attempts contain non-HITL evidence outside an owning Attempt"
                )
            causal_head = event["event_id"]
        execution_bridge = [
            event["event_type"]
            for event in bridge
            if event["event_type"].startswith("Execution")
        ]
        if execution_bridge and execution_bridge not in (
            ["ExecutionProposalSubmitted", "ExecutionHitlResponded"],
            [
                "ExecutionPackageCompiled",
                "ExecutionProposalSubmitted",
                "ExecutionHitlResponded",
            ],
        ):
            raise ValueError("Execution revision re-entry is incomplete")
        binding = validate_attempt_continuation(
            timeline, started, checked_once=planning_context.checked_once
        )
        if not selected or selected[0]["parent_ids"] != [causal_head]:
            raise ValueError("Attempts do not compose through exact Journey lineage")
        review = _project_one_attempt_evidence(
            timeline,
            registration_event,
            selected,
            planning_context=planning_context,
        )
        projected.append(review)
        attempts.append(review)
        projected_owned_ids.update(item.source_event_id for item in review.events)
        previous_terminal = selected[-1]["event_id"]
        previous_terminal_index = end - 1
        if binding is not None:
            expected_role = binding.previous_attempt["payload"]["agent_role"]
        else:
            expected_role = review.agent_role
        if review.agent_role != expected_role:
            raise ValueError("Attempt roles do not follow the composed slice")
    owned_ids = {item.source_event_id for review in attempts for item in review.events}
    if any(
        (
            validate_attempt_fact(event) is not None
            or validate_reference_fact(event) is not None
            or validate_reference_facts_fact(event) is not None
            or validate_execution_planning_fact(event) is not None
            or (
                validate_activity_fact(event) is not None
                and not _is_execution_activity(event)
            )
            or event["event_type"] == "PlanningHitlRequested"
            or event["event_type"] == "ModelAvailabilityReviewRequested"
            or event["event_type"] == "AgentWorkReviewRequested"
        )
        and event["event_id"] not in owned_ids | requirement_review_ids
        for event in timeline
    ):
        raise ValueError("Attempt evidence exists outside its exact lifecycle window")
    return tuple(attempts), tuple(projected)


@dataclass(frozen=True)
class AttemptHistoryRead:
    """Correlated ordinary Run evidence, without stage scheduling authority."""

    journey_id: str
    request: RequestReview
    registered_input_set: RegisteredInputSetReview
    inspection: InspectionReview | None
    attempts: tuple[AttemptEvidenceReview, ...]
    proposal_plan: ProposalPlanReview | None
    agent_effort_review: AgentEffortReviewProjection
    model_availability_review: ModelAvailabilityReviewProjection
    model_contact_postmortem: ModelContactPostmortem | None
    run_failure: RunFailed | None
    # The Attempts the Run's registered slice composes; its stop is theirs.
    slice_attempts: tuple[AttemptEvidenceReview, ...] = ()

    @property
    def latest(self) -> AttemptEvidenceReview | None:
        return self.slice_attempts[-1] if self.slice_attempts else None


def _read_attempt_evidence(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    timeline: list[dict[str, Any]],
    *,
    planning_context: PrefixSnapshot,
    _allow_model_availability_retry_prefix: bool,
) -> AttemptHistoryRead:
    from bio_neb.agentic_journey.tool_execution import (
        validate_observed_function_tools,
    )

    validate_observed_function_tools(
        timeline, checked_once=planning_context.checked_once
    )
    started_event = _single(timeline, "JourneyStarted")
    request_event = _single(timeline, "ResearchRequestRegistered")
    registration_event = _single(timeline, "RunInputsRegistered")
    if started_event["parent_ids"]:
        raise ValueError("Journey start cannot have parents")
    if request_event["parent_ids"] != [started_event["event_id"]]:
        raise ValueError("research request lineage is invalid")
    if registration_event["parent_ids"] != [request_event["event_id"]]:
        raise ValueError("input registration lineage is invalid")

    for event in (started_event, request_event, registration_event):
        validate_fact(event)
    request = validate_fact(request_event)
    registration = validate_fact(registration_event)
    if not isinstance(request, ResearchRequestRegistered) or not isinstance(
        registration, RunInputsRegistered
    ):
        raise ValueError("registered input Journey facts are invalid")

    if any(event.get("event_type") == "JourneyFailed" for event in timeline):
        raise ValueError("untyped Journey failure evidence is not supported")
    typed_attempt_facts = [
        (event, fact)
        for event in timeline
        if (fact := validate_attempt_fact(event)) is not None
    ]
    typed_activity_facts = [
        (event, fact)
        for event in timeline
        if (fact := validate_activity_fact(event)) is not None
    ]
    typed_run_facts = [
        (event, fact)
        for event in timeline
        if (fact := validate_run_fact(event)) is not None
    ]
    configuration_failures = [
        (event, fact)
        for event, fact in typed_run_facts
        if isinstance(fact, RuntimeConfigurationFailed)
    ]
    run_failures = [
        (event, fact) for event, fact in typed_run_facts if isinstance(fact, RunFailed)
    ]
    # Each dispatch that met a configuration failure retained it at the causal
    # head that dispatch admitted; any number may precede the fix. It names the
    # Run that recorded it: a Fork child's own, not its origin's.
    for event, fact in configuration_failures:
        earlier = {e["event_id"] for e in timeline[: timeline.index(event)]}
        if (
            fact.run_id != event["journey_id"]
            or len(event["parent_ids"]) != 1
            or event["parent_ids"][0] not in earlier
        ):
            raise ValueError("runtime-configuration failure is outside its Run")
    if len(run_failures) > 1:
        raise ValueError("Run has more than one failure terminal")

    inspection_events = [
        event
        for event in timeline
        if event.get("event_type") == "InputMetadataInspected"
    ]
    from bio_neb.application.input_journey import project_input_selection

    project_input_selection(timeline)
    inspection_review = None
    inspection_event = inspection_events[-1] if inspection_events else None
    inspection = (
        validate_fact(inspection_event) if inspection_event is not None else None
    )
    if inspection is not None:
        inspection_review = InspectionReview(
            result=inspection.result,
            source_event_id=inspection_event["event_id"],
            parent_event_ids=tuple(inspection_event["parent_ids"]),
        )

    model_availability_review = project_model_availability_review(
        timeline, checked_once=planning_context.checked_once
    )
    stopped_receipts = {
        r.response_event_id: r
        for r in model_availability_review.receipts
        if r.response.action == "stop"
    }
    stops = [e for e in timeline if e["event_type"] == "ModelAvailabilityStopped"]
    if len(stops) != len(stopped_receipts) or {
        e["payload"]["response_event_id"] for e in stops
    } != set(stopped_receipts):
        raise ValueError("availability stop consequence is incomplete")
    attempt_reviews, slice_attempts = _project_attempts_evidence(
        timeline,
        registration_event,
        planning_context=planning_context,
    )
    if any(attempt.outcome == "incomplete" for attempt in attempt_reviews):
        raise ValueError("Attempt has no exact terminal outcome")
    attempt_review = slice_attempts[-1] if slice_attempts else None
    # A slice Attempt fails at its availability Review; a domain-bound one may hold
    # only the affected work, so its domain owns that relationship.
    slice_attempt_ids = {attempt.attempt_id for attempt in slice_attempts}
    availability_attempt_failures = [
        (event, fact)
        for event, fact in typed_attempt_facts
        if isinstance(fact, AttemptFailed)
        and fact.code == "model_availability_exhausted"
        and fact.attempt_id in slice_attempt_ids
    ]
    if len(availability_attempt_failures) != sum(
        r.attempt_id in slice_attempt_ids for r in model_availability_review.requests
    ):
        raise ValueError(
            "availability Attempt failures and review requests are not bijective"
        )
    for failed_event, failed_fact in availability_attempt_failures:
        matched = [
            request
            for request in model_availability_review.requests
            if request.attempt_id == failed_fact.attempt_id
            and request.logical_failure_event_id == failed_fact.source_event_id
        ]
        if len(matched) != 1 or not _has_ancestor(
            timeline,
            ancestor_id=matched[0].request_event_id,
            descendant_id=failed_event["event_id"],
        ):
            raise ValueError(
                "availability Attempt failure changed its exact review request"
            )
    for receipt in model_availability_review.receipts:
        if receipt.response.action not in {"retry", "amend"}:
            continue
        availability_request = next(
            item
            for item in model_availability_review.requests
            if item.request_event_id == receipt.request_event_id
        )
        response_index = next(
            index
            for index, event in enumerate(timeline)
            if event["event_id"] == receipt.response_event_id
        )
        direct_children = [
            event
            for event in timeline[response_index + 1 :]
            if event.get("parent_ids") == [receipt.response_event_id]
            and event["event_type"] not in {"PassStarted", "PassClosed"}
        ]
        retry_attempts = [
            event
            for event in timeline[response_index + 1 :]
            if event.get("event_type") == "AttemptStarted"
            and event["payload"].get("continuation_availability_review_event_id")
            == receipt.response_event_id
        ]
        if not retry_attempts:
            if (
                len(direct_children) == 1
                and direct_children[0].get("event_type") == "RuntimeConfigurationFailed"
            ):
                continue
            if _allow_model_availability_retry_prefix and pending_continuation(
                timeline,
                receipt.response_event_id,
                agent_role=availability_request.agent_role,
            ):
                continue
            raise ValueError("availability retry lacks its fresh Attempt")
        if (
            len(retry_attempts) != 1
            or retry_attempts[0]["payload"].get("agent_role")
            != availability_request.agent_role
        ):
            raise ValueError("availability retry changed its exact fresh Attempt")
    attempt_event_ids = {
        item.source_event_id for attempt in attempt_reviews for item in attempt.events
    }
    orphan_attempt_failures = [
        event
        for event, _ in typed_attempt_facts
        if (
            event["event_type"] == "AttemptFailed"
            or _is_attempt_failure_evidence(event)
        )
        and event["event_id"] not in attempt_event_ids
    ]
    if orphan_attempt_failures:
        raise ValueError("Attempt failure evidence is outside its owning Attempt")
    orphan_activity_failures = [
        event
        for event, fact in typed_activity_facts
        if isinstance(fact, ActivityFailed)
        and event["event_id"] not in attempt_event_ids
        and not _is_execution_activity(event)
    ]
    if orphan_activity_failures:
        raise ValueError("failed activity evidence is outside its owning Attempt")

    run_failure_event = run_failures[0][0] if run_failures else None
    run_failure_fact = run_failures[0][1] if run_failures else None
    has_attempt_failure = (
        attempt_review is not None and attempt_review.outcome == "failed"
    )
    if run_failure_fact is not None:
        from bio_neb.reporting.explanation import without_explanation

        after_failure = timeline[timeline.index(run_failure_event) + 1 :]
        # Only the ended Run's reporting, and its report's explanation, follow.
        if any(
            event["event_type"]
            not in {
                "PassStarted",
                "PassClosed",
                "FinalizationBasisCompiled",
                "FinalizationReviewRecorded",
                "GroundedReportPublished",
            }
            for event in without_explanation(after_failure)
        ):
            raise ValueError("Run failure is not causally final")
        # The failure names the Journey that recorded it: a Fork's own Run ends
        # under its own identity, never its parent's.
        if run_failure_fact.run_id != run_failure_event["journey_id"]:
            raise ValueError("Run failure identity disagrees with its Journey")
        source = next(
            (
                event
                for event in timeline
                if event["event_id"] == run_failure_fact.source_event_id
            ),
            None,
        )
        if source is None or not _has_ancestor(
            timeline,
            ancestor_id=source["event_id"],
            descendant_id=run_failure_event["event_id"],
        ):
            raise ValueError("Run failure source is outside its causal lineage")
        if source["event_type"] == "ExecutionHitlResponded":
            decision = validate_execution_fact(source)
            if (
                not isinstance(decision, ExecutionHitlResponded)
                or not isinstance(decision.response, RejectExecutionResponse)
                or run_failure_fact.code != "rejected_by_user"
                or run_failure_event["parent_ids"] != [source["event_id"]]
                or has_attempt_failure
            ):
                raise ValueError("Run failure does not reference exact user rejection")
        elif source["event_type"] == "PlanningHitlResponded":
            responded = PlanningHitlResponded.model_validate(
                {"event_type": source["event_type"], **source["payload"]}
            )
            planning_failure = project_planning_hitl(timeline)
            receipts = [
                receipt
                for receipt in planning_failure.receipts
                if receipt.response_event_id == source["event_id"]
            ]
            if (
                not isinstance(responded.response, StopPlanningHitlResponse)
                or len(receipts) != 1
                or receipts[0].request_event_id != responded.request_event_id
                or not isinstance(receipts[0].response, StopPlanningHitlResponse)
                or run_failure_fact.code != "required_evidence_unresolved"
                or run_failure_event["parent_ids"] != [source["event_id"]]
                or has_attempt_failure
            ):
                raise ValueError("Run failure does not reference exact Planning stop")
        elif source["event_type"] == "PaperReviewAnswered":
            # The researcher's abort of a refused Run paper ends the Run.
            answer = source["payload"]
            if (
                answer["response"]["action"] != "abort"
                or run_failure_fact.code != "rejected_by_user"
                or run_failure_event["parent_ids"] != [source["event_id"]]
            ):
                raise ValueError("Run failure does not reference exact paper abort")
        else:
            raise ValueError("Run failure references nonfailure evidence")

    proposal_events = [
        event
        for event in timeline
        if event.get("event_type") == "ProposalRequirementsIdentified"
    ]
    if len(proposal_events) > 1:
        raise ValueError("Journey contains more than one proposal-information plan")
    proposal_plan_review = None
    if proposal_events:
        proposal_fact = validate_proposal_fact(proposal_events[0])
        if not isinstance(proposal_fact, ProposalRequirementsIdentified):
            raise ValueError("proposal-information Journey fact is invalid")
        owning_attempts = [
            attempt
            for attempt in slice_attempts
            if proposal_events[0]["event_id"]
            in {item.source_event_id for item in attempt.events}
        ]
        if (
            len(owning_attempts) != 1
            or proposal_events[0]["event_id"] not in attempt_event_ids
        ):
            raise ValueError(
                "proposal plan exists outside a successful planning Attempt"
            )
        proposal_plan_review = ProposalPlanReview(
            plan=proposal_fact.plan,
            source_event_id=proposal_events[0]["event_id"],
            parent_event_ids=tuple(proposal_events[0]["parent_ids"]),
        )

    return AttemptHistoryRead(
        journey_id=started_event["journey_id"],
        request=RequestReview(
            prompt=request.prompt,
            prompt_file=request.prompt_file,
            source_event_id=request_event["event_id"],
        ),
        registered_input_set=RegisteredInputSetReview(
            input_set_id=registration.input_set_id,
            run_root=registration.run_root,
            runtime_config_file=registration.runtime_config_file,
            runtime_selections=registration.runtime_selections,
            inputs=registration.inputs,
            source_event_id=registration_event["event_id"],
        ),
        inspection=inspection_review,
        attempts=attempt_reviews,
        slice_attempts=slice_attempts,
        proposal_plan=proposal_plan_review,
        agent_effort_review=validate_effort_history(
            timeline, checked_once=planning_context.checked_once
        ),
        model_availability_review=model_availability_review,
        model_contact_postmortem=_project_model_contact_postmortem(
            typed_attempt_facts, model_availability_review
        ),
        run_failure=run_failure_fact,
    )


def project_journey_events(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    journey_file: str,
    _allow_model_availability_retry_prefix: bool = False,
) -> JourneyReview:
    context = _InterpretationContext.of(PrefixSnapshot(events))
    return deepcopy(
        _project_journey_events(
            context,
            journey_file=journey_file,
            _allow_model_availability_retry_prefix=_allow_model_availability_retry_prefix,
        )
    )


def _project_journey_events(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    context: _InterpretationContext,
    *,
    journey_file: str,
    _allow_model_availability_retry_prefix: bool = False,
    include_reporting: bool = True,
) -> JourneyReview:
    timeline = context.timeline
    pre_execution = context.checked_pre_execution(
        retry_prefix=_allow_model_availability_retry_prefix
    )
    checked = pre_execution.history
    planning = pre_execution.planning
    from bio_neb.application.input_journey import project_input_selection

    input_selection = project_input_selection(timeline)
    inspection_review = checked.inspection
    attempt_reviews = checked.attempts
    attempt_review = checked.latest
    proposal_plan_review = checked.proposal_plan
    model_availability_review = checked.model_availability_review
    model_contact_postmortem = checked.model_contact_postmortem
    run_failure_fact = checked.run_failure
    investigation_read = context.checked_investigation()
    investigation = investigation_read.investigation
    investigation_evidence = investigation_read.evidence

    planning_hitl = planning.hitl
    requirement_resolution = planning.requirements
    reference_identity = planning.reference
    reference_facts = planning.companions
    execution_planning = planning.execution
    finalization = (
        context.investigation_context.completion.finalization
        if include_reporting
        else FinalizationProjection(status="not_started")
    )
    grounded_report = (
        project_grounded_report(timeline, snapshot=context.planning_context)
        if include_reporting
        else GroundedReportProjection(status="not_published")
    )
    has_execution_review = any(
        event.get("event_type")
        in {"ExecutionProposalSubmitted", "ExecutionHitlResponded"}
        for event in timeline
    )
    execution_hitl = project_execution_hitl(timeline) if has_execution_review else None
    latest_availability_receipt = (
        model_availability_review.receipts[-1]
        if model_availability_review.receipts
        else None
    )
    availability_direction_waiting = (
        latest_availability_receipt is not None
        and latest_availability_receipt.response.action in {"retry", "amend"}
        and run_failure_fact is None
        and not any(
            event["event_type"] == "AttemptStarted"
            and timeline.index(event)
            > next(
                index
                for index, retained in enumerate(timeline)
                if retained["event_id"] == latest_availability_receipt.response_event_id
            )
            for event in timeline
        )
    )
    reporting_status = _current_reporting_status(
        timeline, finalization, grounded_report
    )
    if reporting_status is not None:
        status = reporting_status
    elif investigation.status in AGENT_STOP_STATUSES | {"incomplete"}:
        status = investigation.status
    elif rework_status := _selected_rework_status(
        timeline, investigation, investigation_evidence, finalization
    ):
        status = rework_status
    elif grounded_report.status == "published":
        status = "grounded_report_published"
    elif finalization.status == "approved":
        status = "finalization_approved"
    elif finalization.status == "returned_to_investigation":
        status = "finalization_returned_to_investigation"
    elif finalization.status == "stopped":
        status = "finalization_stopped"
    elif finalization.status == "review_pending":
        status = "finalization_review_pending"
    elif investigation.status == "stopped":
        # The researcher's stop ends the Investigation even after a collection,
        # so the Run reaches the one Finalization path.
        status = "investigation_stopped"
    elif investigation_evidence is not None:
        # A failed item is accounted, never an ended Run (R2).
        status = "investigation_evidence_ready"
    elif investigation.status == "review_pending":
        status = "investigation_review_pending"
    elif investigation.status == "approved":
        status = "investigation_approved"
    elif investigation.status == "investigation_work_ready":
        status = "investigation_work_ready"
    elif stopped_operation := operational_state(
        timeline, checked_once=context.planning_context.checked_once
    ):
        status = stopped_operation
    elif effort_state := effort_status(
        timeline, checked_once=context.planning_context.checked_once
    ):
        status = effort_state
    elif work_state := agent_work_status(
        timeline, checked_once=context.planning_context.checked_once
    ):
        status = work_state
    elif model_availability_review.pending_request is not None:
        status = "model_availability_review_pending"
    elif availability_direction_waiting:
        status = "model_availability_review_direction_recorded"
    elif model_availability_review.stop_event_id is not None:
        status = "model_availability_stopped"
    elif (
        run_failure_fact is not None
        and run_failure_fact.code == "rejected_by_user"
        and execution_hitl is not None
        and execution_hitl.status == "execution_rejected"
    ):
        status = "execution_rejected"
    elif run_failure_fact is not None:
        status = "failed"
    elif project_preparation_review(timeline).stopped:
        status = "incomplete"
    elif execution_hitl is not None and execution_hitl.status != "not_requested":
        status = execution_hitl.status
    elif planning_hitl.pending_request is not None:
        status = "planning_hitl_pending"
    elif execution_planning is not None and checked.slice_attempts:
        status = (
            "execution_planning_ready"
            if execution_planning.status == "execution_planning_ready"
            else "planning_hitl_pending"
            if execution_planning.status == "planning_hitl_pending"
            else "execution_planning_progress"
        )
    elif reference_facts is not None and checked.slice_attempts:
        if reference_facts.status == "planning_hitl_pending":
            status = "planning_hitl_pending"
        elif reference_facts.status == "reference_facts":
            status = "reference_facts"
        else:
            status = "reference_facts_progress"
    elif reference_identity is not None and checked.slice_attempts:
        if reference_identity.status == "planning_hitl_pending":
            status = "planning_hitl_pending"
        elif reference_identity.status == "reference_identity":
            status = "reference_identity"
        else:
            status = "reference_identity_progress"
    elif proposal_plan_review is not None and checked.slice_attempts:
        if proposal_plan_review is None:
            raise ValueError("successful planning Attempt lacks its plan projection")
        status = "proposal_requirements_identified"
    elif inspection_review is not None:
        status = (
            "inspection_succeeded"
            if inspection_review.result.status == "succeeded"
            else "inspection_failed"
        )
    elif run_failure_fact is not None:
        status = "failed"
    else:
        status = "inputs_registered"
    if any(r.pending for r in input_selection.requests) and status in {
        "inspection_failed",
        "inspection_succeeded",
        "inputs_registered",
    }:
        status = "input_review_pending"
    elif input_selection.correction_pending and status in {
        "inspection_failed",
        "inspection_succeeded",
        "inputs_registered",
    }:
        status = "input_correction_pending"
    elif input_selection.stopped and status in {
        "inspection_failed",
        "inspection_succeeded",
        "inputs_registered",
    }:
        status = "inspection_failed"
    run_papers = project_run_papers(timeline)
    if run_papers.review_pending and status in _PAPER_WORK_STATUSES:
        status = "paper_review_pending"
    elif run_papers.files_due and status in _PAPER_WORK_STATUSES - {
        "inputs_registered",
        "input_correction_pending",
    }:
        # After the FASTQs' inspection, before planning goes on.
        status = "paper_intake_pending"
    download_review = project_download_reviews(timeline)
    if download_review.pending and status in INVESTIGATION_STATUSES:
        # Last: the Investigation's own status ignores a download's Review.
        status = "investigation_download_review_pending"
    run_health = project_run_health(timeline)
    if run_health.pending and status in INVESTIGATION_STATUSES:
        status = "run_health_review_pending"
    return JourneyReview(
        journey_id=checked.journey_id,
        journey_file=journey_file,
        status=status,
        request=checked.request,
        registered_input_set=checked.registered_input_set,
        inspection=inspection_review,
        input_selection=project_input_selection(timeline),
        run_papers=run_papers,
        attempt=attempt_review,
        attempts=attempt_reviews,
        proposal_plan=proposal_plan_review,
        requirement_resolution=requirement_resolution,
        agent_effort_review=validate_effort_history(
            timeline, checked_once=context.planning_context.checked_once
        ),
        agent_work_review=validate_agent_work_history(
            timeline, checked_once=context.planning_context.checked_once
        ),
        operational_review=project_operational_review(
            timeline, checked_once=context.planning_context.checked_once
        ),
        repeats=journey_repeats(timeline),
        model_availability_review=model_availability_review,
        model_contact_postmortem=model_contact_postmortem,
        planning_hitl=planning_hitl,
        reference_identity=reference_identity,
        reference_facts=reference_facts,
        execution_planning=execution_planning,
        execution_hitl=execution_hitl,
        preparation_review=project_preparation_review(timeline),
        investigation=investigation,
        investigation_download_review=download_review,
        run_health=run_health,
        investigation_evidence=investigation_evidence,
        investigation_policy=investigation_read.work_policy,
        finalization=finalization,
        grounded_report=grounded_report,
        failure_code=(run_failure_fact.code if run_failure_fact is not None else None),
        failure_source_event_id=(
            run_failure_fact.source_event_id if run_failure_fact is not None else None
        ),
        lineage=tuple(
            JourneyLineageItem(
                event_id=event["event_id"],
                event_type=event["event_type"],
                parent_event_ids=tuple(event["parent_ids"]),
            )
            for event in timeline
        ),
    )


def _investigation_request_frontier(
    outcome: ParabricksExecutionOutcome,
    intent: ObserveRun,
    journey: Journey | None,
) -> ParabricksExecutionOutcome | InvestigationRequestNeeded:
    owner = _query_journey(intent.journey_file, journey)
    if current_investigation_request(owner) is not None:
        return outcome
    return InvestigationRequestNeeded.model_validate(
        outcome.model_dump(exclude={"schema_version", "status"})
    )


def observe(
    intent: ObserveRun, *, journey: Journey | None = None
) -> JourneyObservation:
    """Read Journey truth; an explicit owner retains its snapshot until refreshed."""

    observation = _project_run(intent, journey=journey)
    if isinstance(observation, ParabricksExecutionOutcome):
        return _investigation_request_frontier(observation, intent, journey)
    if isinstance(observation, ParabricksNoncompletion):
        return observation
    return _planning_stop(observation)


def _held(observation, journey):
    """A Run whose latest Pass closed held reads ``run_held`` until resumed (#214).

    An ended Run stays ended: only a Run that can continue is held.
    """
    if held_failure(journey) is None or observation.status in ENDED_RUN_STATUSES:
        return observation
    return observation.model_copy(update={"status": "run_held"})


def _planning_stop(observation):
    """Apply the same retained planning disposition to every checked Run view."""
    if observation.finalization.status != "not_started":
        return observation
    if observation.status in AGENT_STOP_STATUSES | HELD_RUN_STATUSES | {"failed"}:
        return observation
    if observation.planning_hitl.pending_request is not None:
        return observation.model_copy(update={"status": "planning_hitl_pending"})
    # A rejected output beside a resolved one is accounting, not a stop; any other
    # rejected consequence-bearing requirement leaves nothing to plan.
    if observation.requirement_resolution is not None and any(
        item.state == "rejected"
        and item.requirement_id
        in observation.requirement_resolution.unresolved_execution_requirement_ids
        for item in observation.requirement_resolution.requirements
    ):
        return observation.model_copy(update={"status": "incomplete"})
    return observation


def _current_reporting_status(timeline, finalization, grounded_report):
    from bio_neb.reporting.explanation import without_explanation

    # A report's explanation is presentation after it, never a reporting fact.
    head = review_head(without_explanation(timeline))["event_type"]
    if head not in {
        "FinalizationBasisCompiled",
        "FinalizationReviewRecorded",
        "GroundedReportPublished",
    }:
        return None
    # A newer basis after a published report reads as its own Review.
    if head == "GroundedReportPublished" and grounded_report.status == "published":
        return "grounded_report_published"
    return {
        "approved": "finalization_approved",
        "returned_to_investigation": "finalization_returned_to_investigation",
        "stopped": "finalization_stopped",
        "review_pending": "finalization_review_pending",
    }.get(finalization.status)


def ended_run_outcome(snapshot: PrefixSnapshot) -> str | None:
    """Interpret end state without recursively asking for its reporting projection.

    Pending questions and eligible work are continuation boundaries. The same
    checked status projection used by observe supplies terminal meaning here.
    """
    from bio_neb.journey_path import recorded_execution

    execution = recorded_execution(snapshot.timeline)
    if execution.outcome is not None and execution.disposition == "failed":
        return "execution_failed"
    # The enclosing read's interpretation of this snapshot, not a second one.
    context = _InterpretationContext.of(snapshot)
    observation = _planning_stop(
        _project_journey_events(
            context, journey_file="/checked-journey", include_reporting=False
        )
    )
    if observation.investigation.eligible_member_ids:
        return None
    investigation = observation.investigation
    proposal = investigation.proposals[-1] if investigation.proposals else None
    if (
        not investigation.attempts
        and investigation.not_checked
        and investigation.status == "incomplete"
    ):
        # Every request was recorded not checked before any model request.
        return "investigation_requests_not_checked"
    if (
        proposal is not None
        and proposal.request_members
        and not proposal.work_items
        and not proposal.unresolved_request_member_ids
        and not observation.pending_reviews
    ):
        # Every request ended without Work: rejected, an approved issue finding,
        # checked but not verified, or not checked by what it states.
        members = proposal.request_members
        if all(member.rejection is not None for member in members):
            return "investigation_requests_rejected"
        if any(member.issue_approval is not None for member in members):
            return "investigation_issues_approved"
        return "investigation_requests_not_checked"
    if observation.pending_reviews and observation.status != "failed":
        return None
    return observation.status if observation.status in ENDED_RUN_STATUSES else None


@dataclass(frozen=True)
class PreExecutionRead:
    """Ordinary completed Attempts and correlated planning for this exact prefix."""

    prefix_event_ids: tuple[str, ...]
    history: AttemptHistoryRead
    planning: PlanningRead


class _InterpretationContext:
    """One isolated evidence snapshot with lazy, policy-specific checked facets.

    Cached values stay private. Public readers return deep copies. Investigation
    reads do not require ordinary registration or pre-execution Attempt grammar.
    A snapshot has one interpretation, so nested readers reuse its facets.
    """

    @classmethod
    def of(cls, planning_context: PrefixSnapshot) -> _InterpretationContext:
        """The snapshot's one interpretation, derived through its one helper."""
        return planning_context.derived(cls)

    def __init__(self, planning_context: PrefixSnapshot):
        self.planning_context = planning_context
        self.timeline = self.planning_context.timeline
        self.prefix_event_ids = self.planning_context.prefix_event_ids
        self._attempts: dict[bool, AttemptHistoryRead] = {}
        self._investigation: InvestigationRead | None = None
        self.investigation_context = self.planning_context.derived(
            _InvestigationContext
        )

    def checked_pre_execution(self, *, retry_prefix: bool = False) -> PreExecutionRead:
        return PreExecutionRead(
            self.prefix_event_ids,
            self.checked_attempts(retry_prefix=retry_prefix),
            self.planning_context.read,
        )

    def checked_attempts(self, *, retry_prefix: bool = False) -> AttemptHistoryRead:
        if retry_prefix not in self._attempts:
            self._attempts[retry_prefix] = _read_attempt_evidence(
                self.timeline,
                planning_context=self.planning_context,
                _allow_model_availability_retry_prefix=retry_prefix,
            )
        return self._attempts[retry_prefix]

    def checked_investigation(self) -> InvestigationRead:
        if self._investigation is None:
            self._investigation = self.investigation_context.read
        return self._investigation


def _read_owned_interpretation(journey: Journey) -> _InterpretationContext:
    return _InterpretationContext.of(prefix_snapshot(journey))


def _interpretation_context(journey: Journey) -> _InterpretationContext:
    from bio_neb.recording import evidence_revision

    return journey._read_context(
        _read_owned_interpretation, revision=evidence_revision(journey)
    )


def read_planning(journey: Journey) -> PlanningRead:
    """Share this owner's planning read without requiring a completed Attempt."""
    return _interpretation_context(journey).planning_context.read


def read_pre_execution(journey: Journey) -> PreExecutionRead:
    """Read ordinary completed Attempts and planning, isolated from cached values.

    Documented reader: no product path calls it; tests and inspection read retained evidence through it.
    """
    return deepcopy(_interpretation_context(journey).checked_pre_execution())


def read_investigation(journey: Journey) -> InvestigationRead:
    """Read this owner's snapshot; commands separately require current eligibility.

    Only this assembler populates the cache. Each caller receives an isolated copy;
    appending changes prefix identity, while dropping the Journey ends ownership.
    """
    return deepcopy(_interpretation_context(journey).checked_investigation())


def read_completion(journey: Journey) -> CompletionRead:
    """Read checked Finalization/candidate and exact approved Report authority lazily."""
    context = _interpretation_context(journey)
    investigation = context.checked_investigation()
    return deepcopy(
        replace(context.investigation_context.completion, investigation=investigation)
    )


def read_table_reading_stage(journey: Journey) -> TableReadingStage | None:
    """Read the open post-approval VCF table stage from the checked view (ADR 0047)."""
    context = _interpretation_context(journey)
    context.checked_investigation()
    return deepcopy(context.investigation_context.table_reading_stage)


def read_grounded_report(
    journey: Journey, *, report_digest: str | None = None
) -> GroundedReportProjection:
    """Select retained publication using this owner's checked historical bases."""
    context = _interpretation_context(journey)
    context.checked_investigation()
    return deepcopy(
        project_grounded_report(
            context.timeline,
            report_digest=report_digest,
            snapshot=context.planning_context,
        )
    )


def _query_journey(journey_file: str, journey: Journey | None) -> Journey:
    if journey is None:
        return Journey.open(Path(journey_file))
    if journey.path != Path(journey_file):
        raise ValueError("Journey owner path disagrees with the query")
    return journey


def _with_investigation_request(
    observation: JourneyReview, journey: Journey
) -> JourneyReview:
    return observation.model_copy(
        update={"investigation_request": current_investigation_request(journey)}
    )


def _project_run(
    intent: ObserveRun, *, journey: Journey | None = None
) -> JourneyObservation:
    if journey is None:
        journey = Journey.processing_owner(Path(intent.journey_file))
    if has_execution_path(intent.journey_file):
        journey = _query_journey(intent.journey_file, journey)
        context = _interpretation_context(journey)
        # Once the verified Execution feeds Investigation, the Run's stop is its
        # Investigation's, read like any other Journey. A held Run reads as held.
        if (
            held_failure(journey) is None
            and context.checked_investigation().investigation.status == "not_started"
            and context.investigation_context.completion.finalization.status
            == "not_started"
        ):
            return read_parabricks_result(intent.journey_file)
    journey = _query_journey(intent.journey_file, journey)
    context = _interpretation_context(journey)
    read = context.checked_investigation()
    events = read.timeline
    if any(event["event_type"] == "ParabricksExecutionBound" for event in events):
        raise ValueError("legacy_execution_unsupported")
    observation = _with_investigation_request(
        deepcopy(_project_journey_events(context, journey_file=intent.journey_file)),
        journey,
    )
    observation = observation.model_copy(
        update={
            "investigation_members": deepcopy(read.active_members),
            "passes": read_passes(journey),
            # The reading Run; a Fork child's inherited facts keep the parent's id.
            "journey_id": journey.journey_id,
            "run_provenance": (
                "causal_child_or_fork"
                if journey.has_event_type(FORK_ORIGIN_EVENT_TYPE)
                else None
            ),
        }
    )
    return _held(observation, journey)


def review_journey(
    query: ReviewJourneyQuery, *, journey: Journey | None = None
) -> JourneyObservation:
    """Read an owned snapshot, or admit a fresh owner when none is supplied.

    External appends become visible to a supplied owner only after refresh. This
    observation does not grant permission to execute against stale history.
    """
    return _project_run(query, journey=journey)


def review_model_availability_retry_prefix(
    journey_file: str, *, journey: Journey | None = None
) -> JourneyReview:
    """Permit only the internal response-before-fresh-Attempt append prefix."""

    journey = _query_journey(journey_file, journey)
    return deepcopy(
        _project_journey_events(
            _interpretation_context(journey),
            journey_file=journey_file,
            _allow_model_availability_retry_prefix=True,
            include_reporting=False,
        )
    )
