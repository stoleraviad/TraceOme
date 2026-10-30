"""Public Run advancement governed by canonical Journey observation."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr, field_validator

from bio_neb import locations
from bio_neb.agentic_journey.agent_work_review import (
    AgentWorkReviewResponse,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.effort import (
    AgentEffortReviewResponse,
    respond_effort_review,
)
from bio_neb.agentic_journey.invocation_observation import (
    agent_turn_progress as live_agent_turns,
)
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewResponse,
)
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.agent_reentry import (
    AgentReentryRefused,
    answered_agent_work,
    coordinator_assignments,
    reentry_assignment,
)
from bio_neb.application.commands import (
    RegisterRunCommand,
    RespondModelAvailabilityReviewCommand,
    register_run,
    respond_model_availability_review,
)
from bio_neb.application.queries import (
    AGENT_ENDED_STATUSES,
    AGENT_REVIEW_PENDING_STATUSES,
    AGENT_STOP_STATUSES,
    ENDED_RUN_STATUSES,
    EXECUTION_AUTHORITY_STATUSES,
    EXECUTION_PLANNING_PROGRESS_STATUSES,
    EXECUTION_PLANNING_READY_STATUSES,
    EXECUTION_PLANNING_STATUSES,
    EXECUTION_RESULT_STATUSES,
    HELD_RUN_STATUSES,
    INVESTIGATION_REVIEW_PENDING_STATUSES,
    INVESTIGATION_STATUSES,
    INVESTIGATION_WAIT_STATUSES,
    PLANNING_PROGRESS_STATUSES,
    REENTRY_STATUSES,
    REPORT_APPROVED_STATUSES,
    REPORT_PUBLISHED_STATUSES,
    REPORT_RETURNED_STATUSES,
    REPORT_STOPPED_STATUSES,
    STAGE_ENDED_STATUSES,
    STAGE_REVIEW_PENDING_STATUSES,
    InvestigationRequestNeeded,
    JourneyObservation,
    JourneyReview,
    ObserveRun,
    RunStatus,
    observe,
    read_completion,
    read_table_reading_stage,
)
from bio_neb.application.reviews import (
    finalization_response_commands,
    review_for_request,
)
from bio_neb.failure import (
    HeldFailure,
    coordinator_fault,
    holds_run,
    validate_command_input,
)
from bio_neb.failure import ReviewAnswer as FailureReviewAnswer
from bio_neb.investigation.investigation import InvestigationReviewResponse
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    ContinueInvestigationCommand,
    RespondInvestigationReviewCommand,
    begin_investigation,
    continue_investigation,
    respond_investigation_review,
)
from bio_neb.journey import Journey, JourneyValidationError
from bio_neb.parabricks.accepted_pre_execution import (
    AcceptedPreExecutionBindingError,
)
from bio_neb.parabricks.execution_proposal import ExecutionHitlResponse
from bio_neb.parabricks.planning_commands import (
    RespondExecutionHitlCommand,
    RespondPlanningHitlCommand,
    SubmitExecutionProposalCommand,
    respond_execution_hitl,
    respond_planning_hitl,
    submit_execution_proposal,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlRequested,
    PlanningHitlResponse,
    assembly_question_ids,
)
from bio_neb.parabricks.preparation_review import (
    PreparationResponse,
    answer_preparation,
)
from bio_neb.parabricks.result import (
    ParabricksExecutionOutcome,
    ParabricksNoncompletion,
)
from bio_neb.reporting.explanation import ExplanationRequestView
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    CorrectFinalizationCommand,
    StopFinalizationCommand,
    approve_finalization,
    compile_finalization,
    correct_finalization,
    stop_finalization,
)
from bio_neb.reporting.report_commands import (
    PublishGroundedReportCommand,
    RequestReportExplanationCommand,
    open_report_journey,
    publish_grounded_report,
    read_report_explanations,
    request_report_explanation,
    write_readable_report,
)


class ResumeRun(ObserveRun):
    """Resume the Run whose complete retained truth is in one Journey."""


class BeginRun(RegisterRunCommand):
    """Register only the inputs needed to begin one Run."""


class BeginInvestigation(BeginInvestigationCommand):
    """Begin Investigation collection preparation with caller-supplied inputs.

    Omitted, the continuation is compiled from the Run's verified Execution and the
    request is the Run's captured one; a Fork child reads both in its inherited prefix.
    """


class _ReviewAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputReviewAnswer(_ReviewAnswer):
    kind: Literal["input"] = "input"
    response: FailureReviewAnswer


class PaperReviewAnswer(_ReviewAnswer):
    kind: Literal["paper"] = "paper"
    response: FailureReviewAnswer


class PlanningReviewAnswer(_ReviewAnswer):
    kind: Literal["planning"]
    response: PlanningHitlResponse


class ModelAvailabilityReviewAnswer(_ReviewAnswer):
    kind: Literal["model_availability"]
    response: ModelAvailabilityReviewResponse
    # The same answer for every unanswered item of this item's Review.
    every_item: bool = False


class AgentEffortReviewAnswer(_ReviewAnswer):
    kind: Literal["agent_effort"]
    response: AgentEffortReviewResponse


class AgentWorkReviewAnswer(_ReviewAnswer):
    kind: Literal["agent_work"]
    response: AgentWorkReviewResponse


class PreparationReviewAnswer(_ReviewAnswer):
    kind: Literal["preparation"]
    response: PreparationResponse


class ExecutionReviewAnswer(_ReviewAnswer):
    kind: Literal["execution"]
    response: ExecutionHitlResponse


class InvestigationReviewAnswer(_ReviewAnswer):
    kind: Literal["investigation"]
    response: InvestigationReviewResponse


class InvestigationDownloadReviewAnswer(_ReviewAnswer):
    kind: Literal["investigation_download"]
    response: FailureReviewAnswer


class RunHealthReviewAnswer(_ReviewAnswer):
    kind: Literal["run_health"]
    response: FailureReviewAnswer


class JunctionReviewAnswer(_ReviewAnswer):
    kind: Literal["junction"]
    response: dict[str, JsonValue]


class ExecutionPathReviewAnswer(_ReviewAnswer):
    kind: Literal["execution_path"]
    response: dict[str, JsonValue]


ReviewAnswer = Annotated[
    InputReviewAnswer
    | PaperReviewAnswer
    | PreparationReviewAnswer
    | PlanningReviewAnswer
    | ModelAvailabilityReviewAnswer
    | AgentEffortReviewAnswer
    | AgentWorkReviewAnswer
    | ExecutionReviewAnswer
    | InvestigationReviewAnswer
    | InvestigationDownloadReviewAnswer
    | RunHealthReviewAnswer
    | JunctionReviewAnswer
    | ExecutionPathReviewAnswer,
    Field(discriminator="kind"),
]


class AnswerReview(ObserveRun):
    """Answer one exact retained Review request."""

    request_event_id: StrictStr
    answer: ReviewAnswer

    @field_validator("request_event_id")
    @classmethod
    def canonical_request_identity(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("request_event_id must be canonical")
        return value


class _DefaultRunAdvancementComposition:
    @property
    def environ(self):
        bound = _ENVIRON.get()
        return os.environ if bound is None else bound

    def planning_assignment(self, _journey):
        from bio_neb.agentic_journey.planning_assignment import planning_assignment

        # A suggested assembly's targets are checked with the Investigation's own
        # sources: one binding serves both.
        sources = (_INVESTIGATION_DEPENDENCIES.get() or {}).get(
            "source_adapter_factory"
        )
        return planning_assignment(
            **(_PLANNING_DEPENDENCIES.get() or {}), variant_source_factory=sources
        )

    def continue_investigation(self, observation):
        return continue_investigation(
            ContinueInvestigationCommand(journey_file=observation.journey_file),
            environ=self.environ,
            **(_INVESTIGATION_DEPENDENCIES.get() or {}),
        )


_DEFAULT_COMPOSITION = _DefaultRunAdvancementComposition()
_COMPOSITION: ContextVar[object] = ContextVar(
    "run_advancement_composition", default=_DEFAULT_COMPOSITION
)


@contextmanager
def use_run_advancement_composition(composition):
    """Bind fixed application boundaries without changing the public Run intent."""

    token = _COMPOSITION.set(composition)
    try:
        yield
    finally:
        _COMPOSITION.reset(token)


_EXECUTION_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_execution_dependencies", default=None
)


_INVESTIGATION_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_investigation_dependencies", default=None
)


_OBSERVATION_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_observation_dependencies", default=None
)


_PLANNING_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_planning_dependencies", default=None
)


_REPORTING_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_reporting_dependencies", default=None
)


_PAPER_READING_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "run_paper_reading_dependencies", default=None
)


_ENVIRON: ContextVar[Mapping[str, str] | None] = ContextVar("run_environ", default=None)


def _bound(enclosing: dict | None, **selected) -> dict:
    return dict(
        enclosing or {},
        **{name: value for name, value in selected.items() if value is not None},
    )


@contextmanager
def use_run_advancement_dependencies(
    *,
    environ: Mapping[str, str] | None = None,
    planning_model_factory=None,
    reference_adapter=None,
    companion_adapter=None,
    galaxy_adapter=None,
    recipe_adapter=None,
    paper_adapter=None,
    investigation_model_factory=None,
    investigation_source_adapter_factory=None,
    investigation_observation_inputs=None,
    investigation_observation_adapter=None,
    investigation_observation_evidence_reader=None,
    investigation_vcf_source_factory=None,
    report_model_factory=None,
    paper_reading_model_factory=None,
    parabricks_infrastructure_factory=None,
    execution_progress=None,
    agent_turn_progress=None,
):
    """Bind the external effects the default Run composition uses.

    Factories run, and supplied adapters are used, only in their eligible backend
    stage; observing or answering an unrelated Review contacts none. ``environ``
    replaces ``os.environ`` as the credential source. ``agent_turn_progress``
    receives each Agent turn's live progress, such as a ``RepeatProgress``.
    An omitted value keeps the enclosing binding, then the product default.
    """
    bindings = {
        _PLANNING_DEPENDENCIES: _bound(
            _PLANNING_DEPENDENCIES.get(),
            model_factory=planning_model_factory,
            reference_adapter=reference_adapter,
            companion_adapter=companion_adapter,
            galaxy_adapter=galaxy_adapter,
            recipe_adapter=recipe_adapter,
            paper_adapter=paper_adapter,
        ),
        _INVESTIGATION_DEPENDENCIES: _bound(
            _INVESTIGATION_DEPENDENCIES.get(),
            model_factory=investigation_model_factory,
            source_adapter_factory=investigation_source_adapter_factory,
        ),
        _OBSERVATION_DEPENDENCIES: _bound(
            _OBSERVATION_DEPENDENCIES.get(),
            inputs=investigation_observation_inputs,
            adapter=investigation_observation_adapter,
            evidence_reader=investigation_observation_evidence_reader,
            object_read_factory=investigation_vcf_source_factory,
        ),
        _REPORTING_DEPENDENCIES: _bound(
            _REPORTING_DEPENDENCIES.get(), model_factory=report_model_factory
        ),
        _PAPER_READING_DEPENDENCIES: _bound(
            _PAPER_READING_DEPENDENCIES.get(), model_factory=paper_reading_model_factory
        ),
        _EXECUTION_DEPENDENCIES: _bound(
            _EXECUTION_DEPENDENCIES.get(),
            infrastructure_factory=parabricks_infrastructure_factory,
            progress=execution_progress,
        ),
    }
    if environ is not None:
        bindings[_ENVIRON] = environ
    tokens = [(variable, variable.set(value)) for variable, value in bindings.items()]
    try:
        if agent_turn_progress is None:
            yield
        else:
            with live_agent_turns(agent_turn_progress):
                yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


RunAdvancementRefusalCode = Literal[
    "target_question_basis_missing",
    "accepted_worker_binding_refused",
    "coordinator_dispatch_wrote_nothing",
    "planning_closed_preparation_incomplete",
    "proposal_compilation_refused",
    "coordinator_status_unhandled",
    "agent_reentry_unavailable",
    "coordinator_unhandled_error",
]
ReviewAnswerRefusalCode = Literal[
    "review_answer_kind_mismatch",
    "review_answer_not_current",
    "review_answer_request_unknown",
    "review_answer_value_unsupported",
]


class RunAdvancementRefused(ValueError):
    """A coordinator fault, typed by the failure module, that holds the Run.

    The code is the failure's invariant id. Raised from a coordinator site it
    names the status it met; ``advance`` records the hold and re-raises it with
    the held Run's status (``run_held``).
    """

    def __init__(
        self,
        failure: HeldFailure,
        *,
        observation: JourneyObservation | None = None,
    ) -> None:
        if failure.invariant_id is None:
            raise ValueError("a held coordinator fault is an invariant with its id")
        super().__init__(failure.invariant_id)
        self.code = failure.invariant_id
        self.failure = failure
        self.status = None if observation is None else observation.status
        self.journey_id = None if observation is None else observation.journey_id
        self.journey_file = None if observation is None else observation.journey_file
        self.disposition = RefusalDisposition(kind="hold_for_fix")


def _coordinator_fault(
    code: RunAdvancementRefusalCode,
    observation: JourneyObservation,
    cause: str | None = None,
    **evidence: JsonValue,
) -> RunAdvancementRefused:
    """One typed coordinator cause, with the status it met as evidence."""
    return RunAdvancementRefused(
        HeldFailure(
            failure_class="invariant",
            invariant_id=code,
            cause=cause or observation.status,
            evidence={"status": observation.status, **evidence},
        ),
        observation=observation,
    )


class ReviewAnswerRefused(ValueError):
    """Refuse an answer whose concrete owner disagrees with retained truth."""

    def __init__(self, code: ReviewAnswerRefusalCode, *, request_event_id: str) -> None:
        super().__init__(code)
        self.code = code
        self.request_event_id = request_event_id
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


def _stage_view(observation: JourneyObservation) -> JourneyObservation:
    """The checked stage view every Review family resolves against, on any Journey shape.

    A bound Execution's public result carries only the operation's stop; the Run's
    other Reviews stay retained in its Journey and remain answerable or replayable.
    """
    if not isinstance(
        observation, (ParabricksExecutionOutcome, ParabricksNoncompletion)
    ):
        return observation
    from bio_neb.application.queries import project_journey_events
    from bio_neb.recording import read_evidence

    return project_journey_events(
        read_evidence(Journey.open(Path(observation.journey_file))),
        journey_file=observation.journey_file,
    )


def _not_owned(observation: JourneyObservation, request_event_id: str):
    """Refuse an answer its family does not own: unknown everywhere, or another's."""
    owner = review_for_request(_stage_view(observation), request_event_id)
    return ReviewAnswerRefused(
        "review_answer_request_unknown"
        if owner is None
        else "review_answer_kind_mismatch",
        request_event_id=request_event_id,
    )


def answer_review_from_json(
    observation: JourneyObservation, request_event_id: str, response_json: str
) -> AnswerReview:
    """Bind JSON to its retained stage; applying it still requires fresh authority."""
    observation = _stage_view(observation)
    request = review_for_request(observation, request_event_id)
    if request is None or request.command_frontier != "answer":
        raise ReviewAnswerRefused(
            "review_answer_request_unknown"
            if request is None
            else "review_answer_kind_mismatch",
            request_event_id=request_event_id,
        )
    return AnswerReview.model_validate_json(
        json.dumps(
            {
                "journey_file": observation.journey_file,
                "request_event_id": request_event_id,
                "answer": {
                    "kind": request.kind,
                    "response": json.loads(response_json),
                },
            }
        )
    )


ReviewResponse = (
    AnswerReview
    | ApproveFinalizationCommand
    | CorrectFinalizationCommand
    | StopFinalizationCommand
)


def review_response_from_json(
    observation: JourneyObservation, request_event_id: str, response_json: str
) -> ReviewResponse:
    """The one answer entry: bind a response to the family owning ``request_event_id``.

    It resolves on every Journey shape, without deriving current authority from
    history. Finalization keeps its distinct command and requires the caller's exact
    reviewed basis/head plus correction/reason file where applicable. Stage commands
    still validate current eligibility, freshness and duplicate response on
    application.
    """
    observation = _stage_view(observation)
    review = review_for_request(observation, request_event_id)
    if review is None:
        raise ReviewAnswerRefused(
            "review_answer_request_unknown", request_event_id=request_event_id
        )
    if review.command_frontier == "answer":
        return answer_review_from_json(observation, request_event_id, response_json)
    response = json.loads(response_json)
    if not isinstance(response, dict):
        raise ValueError("Review response must be a JSON object")
    action = response.pop("action", None)
    commands = finalization_response_commands()
    if action not in commands or action not in review.request.legal_actions:
        raise ReviewAnswerRefused(
            "review_answer_value_unsupported", request_event_id=request_event_id
        )
    if {"journey_file", "request_event_id"}.intersection(response):
        raise ValueError("Review response cannot replace its correlation")
    return commands[action].model_validate(
        dict(
            response,
            journey_file=observation.journey_file,
            request_event_id=request_event_id,
        )
    )


CoordinatorRoute = Literal["stop", "review", "dispatch", "execution_result"]

# Every Run status has exactly one coordinator route:
# - stop: nothing for the coordinator to do; the Run ended, waits quietly or is held;
# - review: the Run stops at its pending Review;
# - dispatch: a branch of ``_advance_eligible_work`` advances it;
# - execution_result: only a Parabricks result view carries it, read by its type.
# A Parabricks result view is read by its type, not this table: its ``run_held``
# (a held Execution read inside ``resume``, the Run's hold lifted) re-enters.
# Each route is built from the projection's named status sets.
_ROUTE_STATUS_SETS: Mapping[CoordinatorRoute, tuple[frozenset[RunStatus], ...]] = {
    "stop": (
        AGENT_ENDED_STATUSES,
        STAGE_ENDED_STATUSES,
        HELD_RUN_STATUSES,
        INVESTIGATION_WAIT_STATUSES,
        REPORT_RETURNED_STATUSES,
        REPORT_STOPPED_STATUSES,
        REPORT_PUBLISHED_STATUSES,
    ),
    "review": (
        AGENT_REVIEW_PENDING_STATUSES,
        STAGE_REVIEW_PENDING_STATUSES,
        INVESTIGATION_REVIEW_PENDING_STATUSES,
    ),
    "dispatch": (
        PLANNING_PROGRESS_STATUSES,
        EXECUTION_PLANNING_PROGRESS_STATUSES,
        EXECUTION_PLANNING_READY_STATUSES,
        EXECUTION_AUTHORITY_STATUSES,
        REENTRY_STATUSES,
        REPORT_APPROVED_STATUSES,
    ),
    "execution_result": (EXECUTION_RESULT_STATUSES,),
}
COORDINATOR_ROUTES: Mapping[str, CoordinatorRoute] = MappingProxyType(
    {
        status: route
        for route, status_sets in _ROUTE_STATUS_SETS.items()
        for status_set in status_sets
        for status in status_set
    }
)
_TERMINAL_STATUSES = frozenset(
    status for status, route in COORDINATOR_ROUTES.items() if route == "stop"
)
# The dispatch branches' own status sets; each "dispatch" status is in one.
# A ready Execution plan goes to the compiler, never back to planning assignment.
_PLANNING_ASSIGNMENT_STATUSES = (
    PLANNING_PROGRESS_STATUSES | EXECUTION_PLANNING_PROGRESS_STATUSES
)


def _has_pending_review(observation: JourneyObservation) -> bool:
    return observation.current_review is not None


def _has_execution_authority(observation: JourneyObservation) -> bool:
    return observation.execution_hitl is not None and bool(
        observation.execution_hitl.authorities
    )


def independent_planning_work(observation: JourneyObservation) -> str | None:
    """Planning work the coordinator's target question does not hold (#342).

    The targets bear on no reference, companion, recipe or resource choice, so
    that work goes on beside the question; only the Execution proposal waits.
    """
    if (
        not isinstance(observation, JourneyReview)
        or observation.status in _TERMINAL_STATUSES | REENTRY_STATUSES
        or not observation.pending_reviews
    ):
        return None
    if any(
        r.kind != "planning" or r.request.owner != "requirement_resolution"
        for r in observation.pending_reviews
    ):
        return None
    rejected = {
        r.requirement_id
        for r in observation.requirement_resolution.requirements
        if r.state == "rejected"
    }
    identity = observation.reference_identity
    if (
        identity is not None
        and not any(r.requirement_id in rejected for r in identity.requirements)
        and identity.status not in {"reference_identity", "planning_hitl_pending"}
    ):
        return "reference_identity"
    facts = observation.reference_facts
    if (
        facts is not None
        and facts.requirement_id not in rejected
        and facts.status in {"companion_roles_pending", "reference_facts_progress"}
    ):
        return "reference_facts"
    planning = observation.execution_planning
    if (
        facts is not None
        and facts.status == "reference_facts"
        and (
            planning is None
            or planning.status
            in {
                "recipe_validation_pending",
                "recipe_selection_pending",
                "resource_validation_pending",
                "resource_selection_pending",
            }
        )
    ):
        return "execution_planning"
    return None


def _at_stop(observation: JourneyObservation) -> bool:
    # Bound execution advances through its public operation and review junction;
    # its noncompletion and a verified Execution without a request are retained stops.
    # A held Execution read here, with the Run's hold lifted by ``resume``, re-enters.
    if isinstance(observation, ParabricksNoncompletion):
        return observation.status != "run_held"
    if isinstance(observation, InvestigationRequestNeeded):
        return True
    if isinstance(observation, ParabricksExecutionOutcome):
        # A verified Execution with its request feeds Investigation preparation.
        return False
    # Approved partial Report membership proceeds even while other Investigation
    # members remain pending. Publication checks only the exact approved basis.
    if observation.status in REPORT_APPROVED_STATUSES | REENTRY_STATUSES:
        return False
    if observation.investigation.eligible_member_ids:
        return False
    if independent_planning_work(observation) is not None:
        return False
    return _has_pending_review(observation) or observation.status in _TERMINAL_STATUSES


def _open_target_questions(journey: Journey, observation: JourneyObservation) -> None:
    """Ask the researcher for each target the plan leaves unsettled (#342).

    No GPU job runs before the targets exist: the readiness gap
    ``investigation_targets_unsettled`` holds the Execution proposal until every
    item names its targets or the researcher leaves it out.
    """
    resolution = observation.requirement_resolution
    plan = observation.proposal_plan
    if resolution is None or plan is None:
        # Planning says a target holds it; the observation has no plan for it.
        raise _coordinator_fault("target_question_basis_missing", observation)
    for unsettled in resolution.unsettled_targets:
        # An open question waits for its answer, and an unread answer for planning;
        # an answer read with no target the Run can check is asked again.
        if unsettled.reason == "answer_unread" or any(
            r.requirement_id == unsettled.requirement_id
            for r in observation.planning_hitl.pending_requests
        ):
            continue
        item = next(
            item
            for item in resolution.scientific_investigation.items
            if item.requirement_id == unsettled.requirement_id
        )
        evidence_event_ids = (plan.source_event_id, observation.request.source_event_id)
        if unsettled.answer is not None:
            issue = (
                f'The answer "{unsettled.answer}" names no target the Investigation '
                "can check."
            )
            question = (
                "Which target do you mean? Give its rsID, or its position as "
                "sequence:position REF>ALT with the genome version it is on."
            )
        elif unsettled.reason == "no_target":
            issue = "The request names no target for this investigation item."
            question = (
                "Which targets should the Investigation check? Name each one by its "
                "rsID, or one position as sequence:position REF>ALT."
            )
        else:
            # Words the Run cannot check are a dead end the researcher answers
            # before Execution, never a target dropped in silence.
            words = "; ".join(target.words for target in item.targets)
            issue = (
                f'The request\'s words for this target, "{words}", state no rsID or '
                "position the Investigation can check."
            )
            question = (
                "Which target do these words name? Give its rsID, or its position as "
                "sequence:position REF>ALT with the genome version it is on."
            )
        head = journey.head_event_id
        request = PlanningHitlRequested(
            attempt_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{journey.journey_id}:{item.requirement_id}:{head}:resolution",
                )
            ),
            owner="requirement_resolution",
            plan_event_id=plan.source_event_id,
            plan_id=plan.plan.plan_id,
            requirement_id=item.requirement_id,
            subject_key=item.subject_key,
            issue=issue,
            question=question,
            evidence_event_ids=evidence_event_ids,
        )
        payload = request.model_dump(mode="json")
        payload.pop("event_type")
        journey.record(
            request.event_type,
            payload,
            [head],
            event_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{journey.journey_id}:{item.requirement_id}:{head}:review",
                )
            ),
        )


def _admit_target_questions(observation, journey_file):
    if (
        isinstance(observation, JourneyReview)
        and observation.proposal_plan is not None
        and observation.requirement_resolution is not None
        and observation.requirement_resolution.unsettled_targets
        and observation.status not in _TERMINAL_STATUSES
        and observation.finalization.status == "not_started"
        and not _has_execution_authority(observation)
    ):
        owner = Journey.open(Path(journey_file))
        head = owner.head_event_id
        _open_target_questions(owner, observation)
        if owner.head_event_id != head:
            observation = observe(ObserveRun(journey_file=journey_file))
    return observation


def _open_assembly_question(observation, journey_file) -> bool:
    """Ask the researcher about a suggestion of another assembly as soon as the
    planning Attempt records it, with no model turn (owner, #459 D4): the whole Run on
    the suggested assembly with its Galaxy index, or the CPU build of the verified one.
    True when the question was opened."""
    if (
        not isinstance(observation, JourneyReview)
        or observation.status in _TERMINAL_STATUSES
        or _has_execution_authority(observation)
    ):
        return False
    from bio_neb.application.planning_read import current_planning
    from bio_neb.reference_discovery.reference_identity_agent import (
        assembly_question,
    )
    from bio_neb.reference_discovery.reference_tools import unasked_suggestion

    journey = Journey.open(Path(journey_file))
    planning = current_planning(journey)
    if planning.reference is None or planning.companions is None:
        return False
    suggestion_event_id = unasked_suggestion(
        SimpleNamespace(journey=journey), planning.reference
    )
    if suggestion_event_id is None:
        return False
    request = assembly_question(journey, suggestion_event_id)
    payload = request.model_dump(mode="json")
    payload.pop("event_type")
    head = journey.head_event_id
    journey.record(
        request.event_type,
        payload,
        [head],
        event_id=assembly_question_ids(journey.journey_id, suggestion_event_id)[1],
    )
    return True


def _read_target_answers(observation, journey_file, composition) -> bool:
    """Send a researcher's unread target answer back to planning's target reading
    (#462, owner 2026-10-26) as soon as planning may run; True when planning recorded
    something. It waits while the planning model's own question is open (Spec r3 SP5,
    owner "a"), so a failed read never opens a second Review beside that question;
    other target questions may stay open beside it."""
    if not (
        isinstance(observation, JourneyReview)
        and observation.requirement_resolution is not None
        and any(
            unsettled.reason == "answer_unread"
            for unsettled in observation.requirement_resolution.unsettled_targets
        )
        and observation.status not in _TERMINAL_STATUSES
        # An Agent Review holds planning until its answer; the read waits.
        and observation.status not in AGENT_STOP_STATUSES | HELD_RUN_STATUSES
        and observation.finalization.status == "not_started"
        and not _has_execution_authority(observation)
        and all(
            request.owner == "requirement_resolution"
            for request in observation.planning_hitl.pending_requests
        )
    ):
        return False
    owner = Journey.open(Path(journey_file))
    head = owner.head_event_id
    composition.planning_assignment(owner).run(owner, environ=composition.environ)
    return Journey.open(Path(journey_file)).head_event_id != head


def _adaptive_assignment_ready(observation, independent):
    """Assign currently eligible planning work without reopening a complete proposal."""
    return (
        independent is not None or observation.status in _PLANNING_ASSIGNMENT_STATUSES
    ) and not (
        observation.execution_planning is not None
        and observation.execution_planning.status == "execution_planning_ready"
    )


def _read_cited_paper(observation, journey_file, composition) -> bool:
    """Take one step of the reading of a paper the plan's values cite (#383 A2),
    between planning Attempts and before the reference work; True when it took one.
    A stopped reading item holds the Run at its Review, and only that item: its
    siblings are read meanwhile (ADR 0039), and its answer, which the reading's own
    step reads (re-entry, or not read), releases the reading's answer."""
    if not (
        isinstance(observation, JourneyReview)
        and observation.status not in _TERMINAL_STATUSES
        # A continue answer is the reading's to re-enter; an ended stop holds it.
        and observation.status not in AGENT_ENDED_STATUSES | HELD_RUN_STATUSES
        and observation.finalization.status == "not_started"
        and not _has_execution_authority(observation)
    ):
        return False
    from bio_neb.paper_reading.planning import (
        advance_planned_reading,
        held_only_by_its_items,
    )

    owner = Journey.open(Path(journey_file))
    # A pending Review holds the reading unless the reading's own items asked it.
    if (
        observation.status in AGENT_REVIEW_PENDING_STATUSES
        and not held_only_by_its_items(owner)
    ):
        return False
    return advance_planned_reading(
        owner,
        environ=composition.environ,
        model_factory=(_PAPER_READING_DEPENDENCIES.get() or {}).get("model_factory"),
    )


def _continue_to_stop(journey_file: str) -> JourneyObservation:
    composition = _COMPOSITION.get()
    while True:
        observation = observe(ObserveRun(journey_file=journey_file))
        observation = _admit_target_questions(observation, journey_file)
        if _open_assembly_question(observation, journey_file):
            continue
        if _read_target_answers(observation, journey_file, composition):
            continue
        if _read_cited_paper(observation, journey_file, composition):
            continue
        owner = Journey.open(Path(journey_file))
        if (
            not isinstance(
                observation, (ParabricksExecutionOutcome, ParabricksNoncompletion)
            )
            and observation.status in INVESTIGATION_STATUSES
        ):
            from bio_neb.investigation.investigation_observation_commands import (
                advance_approved_observation,
            )

            if advance_approved_observation(
                owner, **(_OBSERVATION_DEPENDENCIES.get() or {})
            ):
                continue
            from bio_neb.investigation.vcf_table_reading import (
                advance_table_reading,
            )

            # After the lookups, before Finalization: every item's window in one
            # batch, the VCF table read in chunks, then the collection's item
            # fold (ADR 0047).
            if read_table_reading_stage(owner) is not None and advance_table_reading(
                owner,
                environ=composition.environ,
                model_factory=(_INVESTIGATION_DEPENDENCIES.get() or {}).get(
                    "model_factory"
                ),
                adapter=(_OBSERVATION_DEPENDENCIES.get() or {}).get("adapter"),
            ):
                continue
        completion = read_completion(owner)
        if completion.candidate is not None:
            if completion.finalization.status in {
                "not_started",
                "returned_to_investigation",
            }:
                # After completed collection, before Finalization: a child point.
                _close_at_child_point(owner)
            before = owner.head_event_id
            compile_finalization(
                CompileFinalizationCommand(journey_file=journey_file), journey=owner
            )
            if owner.head_event_id != before:
                continue
        if observation.status in REPORT_PUBLISHED_STATUSES:
            from bio_neb.reporting.explanation_agent import (
                advance_report_explanation,
                request_run_default_explanation,
            )

            # After the report: its explanation, by default, then a requested
            # explanation's step, or its answered Review.
            if request_run_default_explanation(
                owner, observation.grounded_report.report.report_digest
            ):
                continue
            if advance_report_explanation(
                owner,
                environ=composition.environ,
                model_factory=(_REPORTING_DEPENDENCIES.get() or {}).get(
                    "model_factory"
                ),
            ):
                continue
            # Its published explanation's page, as the Run's own report.
            if write_readable_report(owner, observation.grounded_report.report):
                continue
        if observation.status == "operational_review_pending":
            from bio_neb.agentic_journey.operational_review import (
                answer_from_retry_preset,
            )

            # A retry the researcher preset answers the Review as it opens.
            if answer_from_retry_preset(owner):
                continue
        if _at_stop(observation):
            return observation
        journey = Journey.open(Path(journey_file))
        before = journey.head_event_id
        independent = independent_planning_work(observation)
        try:
            with _finalizing_an_ended_run(journey_file):
                _advance_eligible_work(
                    composition, observation, journey, independent, before
                )
        except AcceptedPreExecutionBindingError as error:
            # Only planning admission raises it, before its Attempt writes anything.
            # Restore the accepted worker, then resume.
            raise _coordinator_fault(
                "accepted_worker_binding_refused",
                observation,
                error.code,
                fields=list(error.fields),
            ) from error
        after = Journey.open(Path(journey_file)).head_event_id
        if after == before:
            raise _coordinator_fault("coordinator_dispatch_wrote_nothing", observation)


def _advance_eligible_work(composition, observation, journey, independent, before):
    """Dispatch one lawful domain operation from the coordinator's checked state."""
    journey_file = observation.journey_file
    if observation.status == "finalization_approved":
        basis = observation.finalization.report_basis
        assert basis is not None
        publish_grounded_report(
            PublishGroundedReportCommand(
                journey_file=journey_file,
                report_basis_digest=basis.report_basis_digest,
                head_event_id=before,
            )
        )
    elif isinstance(observation, ParabricksExecutionOutcome):
        # After verified Execution, before Investigation: a child point.
        _close_at_child_point(journey)
        _begin_investigation(BeginInvestigationCommand(journey_file=journey_file))
    elif isinstance(observation, ParabricksNoncompletion):
        # A held Execution after its fix re-enters under its recorded authority.
        from bio_neb.parabricks.operation import execute_parabricks
        from bio_neb.parabricks.plan import approved_execution_input

        execute_parabricks(
            approved_execution_input(journey), **(_EXECUTION_DEPENDENCIES.get() or {})
        )
    elif (
        observation.investigation.eligible_member_ids
        or observation.status == "investigation_work_ready"
    ):
        # Members to plan, or a settled collection whose proposal closes.
        composition.continue_investigation(observation)
    elif (
        _has_execution_authority(observation)
        # Execution authority stays recorded after its verified result feeds
        # Investigation; from then on the Run's work is Investigation's.
        and observation.investigation.status == "not_started"
    ):
        from bio_neb.parabricks.operation import execute_parabricks
        from bio_neb.parabricks.plan import approved_execution_input

        execute_parabricks(
            approved_execution_input(journey), **(_EXECUTION_DEPENDENCIES.get() or {})
        )
    elif observation.status == "inputs_registered":
        from bio_neb.input_inspection.commands import (
            InspectRegisteredInputsCommand,
            inspect_registered_inputs,
        )

        inspect_registered_inputs(
            InspectRegisteredInputsCommand(journey_file=journey_file), journey=journey
        )
    elif observation.status == "paper_intake_pending":
        from bio_neb.paper_intake.intake import take_supplied_papers

        take_supplied_papers(journey)
    elif _adaptive_assignment_ready(observation, independent):
        composition.planning_assignment(journey).run(
            journey, environ=composition.environ
        )
    elif observation.status in REENTRY_STATUSES:
        from bio_neb.recording import read_evidence

        answered = answered_agent_work(observation, read_evidence(journey))
        _reentry(composition, observation, journey, answered.attempt_id).run(
            journey, environ=composition.environ, **answered.direction
        )
    elif (
        observation.status in EXECUTION_PLANNING_STATUSES
        and observation.execution_planning is not None
        and observation.execution_planning.status == "execution_planning_ready"
    ):
        from bio_neb.parabricks.planning_readiness import PlanningIncomplete
        from bio_neb.parabricks.preparation import (
            PreparationRefused,
            prepare_execution,
        )

        prepared = prepare_execution(journey)
        if (
            isinstance(prepared, PlanningIncomplete)
            and prepared.code == "investigation_targets_unsettled"
        ):
            # Only the researcher names targets, so the question needs no model
            # turn; an answer is read by planning first (`_read_target_answers`).
            _open_target_questions(journey, observation)
        elif isinstance(prepared, PlanningIncomplete) and (
            prepared.preparation_information
        ):
            # Only the researcher supplies it, so the question needs no model turn.
            from bio_neb.parabricks.preparation_review import open_preparation_review

            open_preparation_review(journey, prepared)
        elif isinstance(prepared, PlanningIncomplete):
            from bio_neb.agentic_journey.planning_assignment import (
                planning_closes_before_contact,
            )

            if planning_closes_before_contact(journey):
                # Planning has nothing left to ask or record, so a rerun would be
                # a contact-free Attempt with no new fact: planning closure and
                # the compiler disagree.
                raise _coordinator_fault(
                    "planning_closed_preparation_incomplete",
                    observation,
                    prepared.code,
                    details=list(prepared.details),
                )
            composition.planning_assignment(journey).run(
                journey, environ=composition.environ
            )
        elif isinstance(prepared, PreparationRefused):
            raise _coordinator_fault(
                "proposal_compilation_refused",
                observation,
                prepared.code,
                details=list(prepared.details),
            )
        else:
            submit_execution_proposal(
                SubmitExecutionProposalCommand(journey_file=journey_file)
            )
    else:
        raise _coordinator_fault("coordinator_status_unhandled", observation)


def _close_at_child_point(journey: Journey) -> None:
    """Close the processing Pass at an ADR 0031 child point; work continues in a new one."""
    from bio_neb.recording import current_processing_pass

    interval = current_processing_pass()
    if interval is None:
        return  # A historical execution file runs outside a Pass.
    if interval.journey is not journey:
        raise RuntimeError("another Journey owns this processing Pass")
    interval.close_at_child_point()


def _begin_investigation(command: BeginInvestigationCommand) -> None:
    begin_investigation(
        command,
        environ=_COMPOSITION.get().environ,
        **(_INVESTIGATION_DEPENDENCIES.get() or {}),
    )


def _reentry(
    composition,
    observation: JourneyObservation,
    journey: Journey,
    attempt_id: str,
):
    from bio_neb.recording import read_evidence

    events = read_evidence(journey)
    try:
        return reentry_assignment(
            events,
            attempt_id,
            coordinator_assignments(
                lambda: composition.planning_assignment(journey),
                _INVESTIGATION_DEPENDENCIES.get(),
                {
                    "model_factory": (_INVESTIGATION_DEPENDENCIES.get() or {}).get(
                        "model_factory"
                    )
                },
                {
                    "model_factory": (_REPORTING_DEPENDENCIES.get() or {}).get(
                        "model_factory"
                    )
                },
                _PAPER_READING_DEPENDENCIES.get(),
            ),
        )
    except AgentReentryRefused as error:
        # No current Agent work can re-enter the stopped Attempt.
        raise _coordinator_fault(
            "agent_reentry_unavailable", observation, str(error)
        ) from error


def _record_answer(intent: ReviewResponse, observation: JourneyObservation) -> None:
    request = review_for_request(observation, intent.request_event_id)
    if (
        request is not None
        and request.request.pending
        and request.request_event_id
        not in {item.request_event_id for item in observation.pending_reviews}
    ):
        raise ReviewAnswerRefused(
            "review_answer_not_current", request_event_id=intent.request_event_id
        )
    owner = None if request is None else request.kind
    kind = intent.answer.kind if isinstance(intent, AnswerReview) else "finalization"
    if owner is not None and owner != kind:
        raise ReviewAnswerRefused(
            "review_answer_kind_mismatch",
            request_event_id=intent.request_event_id,
        )
    if (
        isinstance(intent, AnswerReview)
        and isinstance(intent.answer, InvestigationReviewAnswer)
        and observation.current_review is not None
        and observation.current_review.kind == "finalization"
        and not any(
            receipt.request_event_id == intent.request_event_id
            and receipt.response == intent.answer.response
            for receipt in observation.investigation.review_receipts
        )
    ):
        # Pending partial Finalization owns its frozen collection/head. A new
        # member decision must follow its correction path; historical no-ops
        # remain stage-validated and do not invalidate that exact Review.
        raise ReviewAnswerRefused(
            "review_answer_not_current", request_event_id=intent.request_event_id
        )
    if not isinstance(intent, AnswerReview):
        handlers = {
            ApproveFinalizationCommand: approve_finalization,
            CorrectFinalizationCommand: correct_finalization,
            StopFinalizationCommand: stop_finalization,
        }
        handlers[type(intent)](intent)
        return
    composition = _COMPOSITION.get()
    if isinstance(intent.answer, PreparationReviewAnswer):
        answer_preparation(
            intent.journey_file, intent.request_event_id, intent.answer.response
        )
    elif isinstance(intent.answer, InputReviewAnswer):
        from bio_neb.application.input_journey import answer_input_review

        answer_input_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, PaperReviewAnswer):
        from bio_neb.paper_intake.intake import answer_paper_review

        answer_paper_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, InvestigationReviewAnswer):
        respond_investigation_review(
            RespondInvestigationReviewCommand(
                journey_file=intent.journey_file,
                request_event_id=intent.request_event_id,
                response=intent.answer.response,
            )
        )
    elif isinstance(intent.answer, InvestigationDownloadReviewAnswer):
        from bio_neb.investigation.download_review import answer_download_review

        answer_download_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, RunHealthReviewAnswer):
        from bio_neb.investigation.run_health import answer_health_review

        answer_health_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, PlanningReviewAnswer):
        respond_planning_hitl(
            RespondPlanningHitlCommand(
                journey_file=intent.journey_file,
                request_event_id=intent.request_event_id,
                response=intent.answer.response,
            )
        )
    elif isinstance(intent.answer, AgentEffortReviewAnswer):
        respond_effort_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, AgentWorkReviewAnswer):
        respond_agent_work_review(
            Journey.open(Path(intent.journey_file)),
            intent.request_event_id,
            intent.answer.response,
        )
    elif isinstance(intent.answer, ModelAvailabilityReviewAnswer):
        pending = next(
            (
                r
                for r in observation.model_availability_review.requests
                if r.request_event_id == intent.request_event_id and r.pending
            ),
            None,
        )
        attempt = (
            None
            if intent.answer.response.action == "stop"
            or pending is None
            or pending.request_event_id != intent.request_event_id
            else _reentry(
                composition,
                observation,
                Journey.open(Path(intent.journey_file)),
                pending.attempt_id,
            )
        )
        respond_model_availability_review(
            RespondModelAvailabilityReviewCommand(
                journey_file=intent.journey_file,
                request_event_id=intent.request_event_id,
                response=intent.answer.response,
                every_item=intent.answer.every_item,
            ),
            environ=composition.environ,
            attempt=attempt,
        )
    else:
        assert isinstance(intent.answer, ExecutionReviewAnswer)
        respond_execution_hitl(
            RespondExecutionHitlCommand(
                journey_file=intent.journey_file,
                request_event_id=intent.request_event_id,
                response=intent.answer.response,
            )
        )


def answer_from_json(
    journey_file: str, request_event_id: str, response_json: str
) -> JourneyObservation:
    """Answer one Review and advance, reading the Run once for both.

    The Review is read and the answer applied on one owner, so what the read
    derived serves the advance too; the owner must still be current when the
    writer's interval begins. A file that opens no Journey, such as a retained
    historical execution file, is read and answered without one, as before.
    """
    try:
        journey = Journey.open(Path(journey_file))
    except JourneyValidationError:
        journey = None
    review = observe(ObserveRun(journey_file=journey_file), journey=journey)
    return advance(
        review_response_from_json(review, request_event_id, response_json),
        journey=journey,
    )


def explain_report(
    command: RequestReportExplanationCommand,
) -> tuple[JourneyObservation, ExplanationRequestView]:
    """Request a report's explanation and run it, reading the Run once.

    The request, the coordinator's run and the explanation read share one owner,
    so each request's report and the Investigation under it are derived once.
    """
    journey = open_report_journey(command.journey_file)
    request_report_explanation(command, journey=journey)
    review = advance(ResumeRun(journey_file=command.journey_file), journey=journey)
    current = read_report_explanations(command.journey_file, journey=journey).current
    if current is None:
        raise ValueError("report explanation request is not retained")
    return review, current


def advance(
    intent: BeginRun | ResumeRun | BeginInvestigation | ReviewResponse,
    *,
    journey: Journey | None = None,
) -> JourneyObservation:
    """Advance through one admitted writer until a lawful public stop.

    ``journey`` is the caller's owner of the same Run, when it already read it.
    A fault escaping the coordinator goes to the failure module; a hold is
    recorded on the Pass and the typed refusal names the held Run.
    """
    from bio_neb.recording import processing_pass

    if isinstance(intent, BeginRun):
        # Only the product's own Run reads the operator home: its planning model's
        # credential and the account's Parabricks values. A bound factory or
        # composition supplies its own model.
        real_model = (
            _COMPOSITION.get() is _DEFAULT_COMPOSITION
            and (_PLANNING_DEPENDENCIES.get() or {}).get("model_factory") is None
        )
        registered = register_run(
            intent,
            environ=_DEFAULT_COMPOSITION.environ if real_model else None,
            parabricks_account=locations.parabricks_account() if real_model else None,
        )
        intent = ResumeRun(journey_file=registered.journey_file)
    if journey is not None and journey.path != Path(intent.journey_file):
        raise ValueError("Journey owner path disagrees with the command")
    try:
        journey = journey or Journey.open(Path(intent.journey_file))
    except JourneyValidationError:
        from bio_neb.journey_path import is_historical_execution_file

        if is_historical_execution_file(intent.journey_file):
            return _advance(intent)
        raise
    with journey.processing(), processing_pass(journey) as interval:
        if isinstance(intent, ResumeRun):
            # After the fix, resume continues the same Run.
            interval.lift_hold()
        try:
            result = _advance(intent)
        except BaseException as error:
            cancelled = _cancelled_attempts(interval, error)
            failure = coordinator_fault(
                error,
                cancelled_attempt=cancelled[-1] if cancelled else None,
                cancelled_attempts=cancelled,
            )
            held = (
                failure is not None
                and holds_run(failure.failure_class)
                and not _run_ended(intent.journey_file)
            )
            try:
                if held:
                    interval.hold(failure)
                else:
                    interval.close("interrupted")
            except (OSError, ValueError) as closure_error:
                error.add_note(
                    f"Pass closure could not be retained: {type(closure_error).__name__}"
                )
                held = False  # Without its record the fault is not held.
            if not held or failure.failure_class == "interrupted":
                raise  # The operator's interrupt goes on; the Run is held.
            raise _held_refusal(failure, intent.journey_file) from (
                error.__cause__ if isinstance(error, RunAdvancementRefused) else error
            )
        interval.settle_lift()
        from bio_neb.parabricks.result import ParabricksNoncompletion

        waiting = (
            result.pending_review is not None
            if isinstance(result, ParabricksNoncompletion)
            else getattr(result, "current_review", None) is not None
            and result.status not in _TERMINAL_STATUSES
        )
        interval.close("waiting" if waiting else "completed")
        return observe(ObserveRun(journey_file=intent.journey_file), journey=journey)


@contextmanager
def _finalizing_an_ended_run(journey_file: str):
    """Dispatch work; a fault whose own new record ended the Run goes to Finalization.

    Only the dispatched work's fault qualifies (#181, #273). A fault that left the
    Run as it was, one in an already-ended Run, or one from Finalization itself
    escapes to the failure module.
    """
    before = Journey.open(Path(journey_file)).head_event_id
    try:
        yield
    except Exception as error:
        if not _fault_ended_run(error, journey_file, before):
            raise


def _fault_ended_run(error: Exception, journey_file: str, before: str) -> bool:
    """Whether the Run ended at a fact recorded after ``before`` and now awaits a basis."""
    if coordinator_fault(error) is None:
        return False  # A command refusal is not a fault.
    try:
        journey = Journey.open(Path(journey_file))
        candidate = read_completion(journey).candidate
        if candidate is None or not _run_ended(journey_file):
            return False
        event_ids = [event["event_id"] for event in journey.read()]
        return event_ids.index(candidate.source_head_event["event_id"]) > (
            event_ids.index(before)
        )
    except Exception:  # The fault may be the projection's own; it is not routed.
        return False


def _run_ended(journey_file: str) -> bool:
    """Whether the fault already ended its Run; only a Run that can continue is held."""
    try:
        status = observe(ObserveRun(journey_file=journey_file)).status
    except Exception:  # The fault may be the projection's own; hold it.
        return False
    return status in ENDED_RUN_STATUSES


def _cancelled_attempts(interval, error: BaseException) -> tuple[dict[str, Any], ...]:
    """Every Agent Attempt an interrupt cancelled within this Pass, in order.

    Concurrent members or chunks each record their own cancellation (#105).
    """
    if isinstance(error, Exception) or interval.started is None:
        return ()
    events = Journey.open(interval.journey.path).read()
    started = next(
        index
        for index, event in enumerate(events)
        if event["event_id"] == interval.started["event_id"]
    )
    return tuple(
        event for event in events[started:] if event["event_type"] == "AttemptCancelled"
    )


def _held_refusal(failure: HeldFailure, journey_file: str) -> RunAdvancementRefused:
    """The refusal a caller receives once the hold is recorded."""
    try:
        observation = observe(ObserveRun(journey_file=journey_file))
    except Exception:  # The fault may be the projection's own; the hold stands.
        observation = None
    return RunAdvancementRefused(failure, observation=observation)


def _advance(
    intent: ResumeRun | BeginInvestigation | ReviewResponse,
) -> JourneyObservation:
    """Coordinate commands sharing the processing interval's Journey owner."""

    if isinstance(intent, BeginInvestigation):
        before = Journey.open(Path(intent.journey_file)).head_event_id
        with _finalizing_an_ended_run(intent.journey_file):
            _begin_investigation(intent)
        if Journey.open(Path(intent.journey_file)).head_event_id == before:
            return observe(ObserveRun(journey_file=intent.journey_file))
        return _continue_to_stop(intent.journey_file)

    if isinstance(intent, AnswerReview):
        from bio_neb.journey_path import has_execution_path
        from bio_neb.parabricks.review import answer_review, review_requests

        if isinstance(intent.answer, ExecutionReviewAnswer) and has_execution_path(
            intent.journey_file
        ):
            before = Journey.open(Path(intent.journey_file)).head_event_id
            respond_execution_hitl(
                RespondExecutionHitlCommand(
                    journey_file=intent.journey_file,
                    request_event_id=intent.request_event_id,
                    response=intent.answer.response,
                )
            )
            if Journey.open(Path(intent.journey_file)).head_event_id == before:
                return observe(ObserveRun(journey_file=intent.journey_file))
            return _continue_to_stop(intent.journey_file)

        if isinstance(intent.answer, JunctionReviewAnswer):
            from bio_neb.agentic_journey.operational_review import (
                answer_operational_review,
                evaluate_operational_review_response,
            )

            observation = _stage_view(
                observe(ObserveRun(journey_file=intent.journey_file))
            )
            request = next(
                (
                    r
                    for r in observation.operational_review.requests
                    if r.request_event_id == intent.request_event_id
                ),
                None,
            )
            if request is None:
                raise _not_owned(observation, intent.request_event_id)
            owner = Journey.open(Path(intent.journey_file))
            response = validate_command_input(
                FailureReviewAnswer, "answer.response", intent.answer.response
            )
            _, retained = evaluate_operational_review_response(
                observation.operational_review, intent.request_event_id, response
            )
            if retained is not None:
                return observation
            selected = None
            if response.action != "stop":
                selected = _reentry(
                    _COMPOSITION.get(), observation, owner, request.attempt_id
                )
            receipt, fresh = answer_operational_review(
                owner, intent.request_event_id, response
            )
            if fresh and selected is not None:
                with _finalizing_an_ended_run(intent.journey_file):
                    selected.run(
                        owner,
                        environ=_COMPOSITION.get().environ,
                        operational_review_response_event_id=receipt.response_event_id,
                    )
            return (
                _continue_to_stop(intent.journey_file)
                if fresh
                else observe(ObserveRun(journey_file=intent.journey_file))
            )

        if isinstance(intent.answer, ExecutionPathReviewAnswer):
            from bio_neb.journey_path import JourneyPath
            from bio_neb.parabricks.operation import execute_parabricks
            from bio_neb.parabricks.plan import ApprovedParabricksInput

            if not has_execution_path(intent.journey_file) or not any(
                request.request_event_id == intent.request_event_id
                for request in review_requests(intent.journey_file)
            ):
                raise _not_owned(
                    observe(ObserveRun(journey_file=intent.journey_file)),
                    intent.request_event_id,
                )
            before = JourneyPath.open(Path(intent.journey_file)).state.event_ids
            answered = answer_review(
                intent.journey_file, intent.request_event_id, intent.answer.response
            )
            if answered.event_ids == before:
                return observe(ObserveRun(journey_file=intent.journey_file))
            if answered.outcome is not None:
                return _continue_to_stop(intent.journey_file)
            # An answer that continues goes back to Parabricks: a CPU preparation
            # Review offers only abort or one CPU job.
            execute_parabricks(
                ApprovedParabricksInput.model_validate(answered.bound.approved_input),
                **(_EXECUTION_DEPENDENCIES.get() or {}),
            )
            return _continue_to_stop(intent.journey_file)

    if isinstance(
        intent,
        (
            AnswerReview,
            ApproveFinalizationCommand,
            CorrectFinalizationCommand,
            StopFinalizationCommand,
        ),
    ):
        journey_file = intent.journey_file
        # Retained Agent/input answers remain replayable after domain execution
        # resumes. The shared review owners still validate their exact receipts.
        observation = _stage_view(observe(ObserveRun(journey_file=journey_file)))
        before = Journey.open(Path(journey_file)).head_event_id
        _record_answer(intent, observation)
        after = Journey.open(Path(journey_file)).head_event_id
        if before == after:
            # A retained receipt is not permission to repeat downstream work,
            # including a prior continuation that failed before its first append.
            return observe(ObserveRun(journey_file=journey_file))
        current = observe(ObserveRun(journey_file=journey_file))
        # A member question's answer, or a member's settled Agent work stop, may
        # leave the collection ready to close.
        revised_investigation = isinstance(intent, CorrectFinalizationCommand) or (
            isinstance(intent, AnswerReview)
            and isinstance(
                intent.answer, (InvestigationReviewAnswer, AgentWorkReviewAnswer)
            )
        )
        if (
            revised_investigation
            and current.investigation.status == "investigation_work_ready"
            and read_completion(Journey.open(Path(journey_file))).candidate is None
        ):
            with _finalizing_an_ended_run(journey_file):
                _COMPOSITION.get().continue_investigation(current)
    else:
        journey_file = intent.journey_file
    return _continue_to_stop(journey_file)
