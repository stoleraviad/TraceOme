"""Shared Review meaning over checked stage projections, without command authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr, model_validator

from bio_neb.agentic_journey.agent_work_review import AgentWorkReviewRequestView
from bio_neb.agentic_journey.effort import (
    AgentEffortReviewRequestView,
    ExtendAgentEffortResponse,
    StopAgentEffortResponse,
)
from bio_neb.agentic_journey.model_availability import (
    AmendModelAvailabilityResponse,
    ModelAvailabilityReviewRequestView,
    RetryModelAvailabilityResponse,
    StopModelAvailabilityResponse,
)
from bio_neb.agentic_journey.operational_review import AgentFailureReviewRequestView
from bio_neb.application.input_journey import InputReviewRequest
from bio_neb.failure import ReviewAnswer as FailureReviewAnswer
from bio_neb.investigation.download_review import InvestigationDownloadReviewRequest
from bio_neb.investigation.investigation import (
    AmendInvestigationFindingResponse,
    ApproveInvestigationFindingResponse,
    ApproveInvestigationResponse,
    ClarifyInvestigationResponse,
    InvestigationReviewRequestView,
    RejectInvestigationResponse,
    ReviseInvestigationResponse,
    StopInvestigationResponse,
)
from bio_neb.investigation.run_health import RunHealthReviewRequest
from bio_neb.paper_intake.journey import PaperReviewRequest
from bio_neb.parabricks.execution_proposal import ExecutionHitlRequestView
from bio_neb.parabricks.planning_hitl import PlanningHitlRequestView
from bio_neb.parabricks.preparation_review import PreparationRequest
from bio_neb.parabricks.review import ExecutionPathReviewRequestView, review_requests
from bio_neb.reporting.finalization import FinalizationReviewRequestView

if TYPE_CHECKING:
    from bio_neb.application.queries import JourneyObservation


class ModelAvailabilityLegalAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["retry", "amend", "stop"]
    response_schema: dict[str, JsonValue]


def _model_availability_actions() -> tuple[ModelAvailabilityLegalAction, ...]:
    return (
        ModelAvailabilityLegalAction(
            action="retry",
            response_schema=RetryModelAvailabilityResponse.model_json_schema(),
        ),
        ModelAvailabilityLegalAction(
            action="amend",
            response_schema=AmendModelAvailabilityResponse.model_json_schema(),
        ),
        ModelAvailabilityLegalAction(
            action="stop",
            response_schema=StopModelAvailabilityResponse.model_json_schema(),
        ),
    )


class ModelAvailabilityRequestDetails(ModelAvailabilityReviewRequestView):
    legal_actions: tuple[ModelAvailabilityLegalAction, ...] = Field(
        default_factory=_model_availability_actions
    )


class ReviewLegalAction(BaseModel):
    """One stage-owned response shape accepted by the shared response binder."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: StrictStr
    response_schema: dict[str, JsonValue]


def finalization_response_commands():
    # Keep the explicit Finalization command frontier and its required bindings.
    from bio_neb.reporting.finalization_commands import (
        ApproveFinalizationCommand,
        CorrectFinalizationCommand,
        StopFinalizationCommand,
    )

    return {
        "approve": ApproveFinalizationCommand,
        "correct": CorrectFinalizationCommand,
        "stop": StopFinalizationCommand,
    }


def _legal_actions(request) -> tuple[ReviewLegalAction, ...]:
    if isinstance(request, ExecutionPathReviewRequestView):
        from bio_neb.journey_path import Answered, Pending

        answer = (
            Answered if isinstance(request.review, Pending) else FailureReviewAnswer
        )
        return tuple(
            ReviewLegalAction(action=a, response_schema=answer.model_json_schema())
            for a in request.legal_actions
        )
    if isinstance(request, AgentFailureReviewRequestView):
        return tuple(
            ReviewLegalAction(
                action=a, response_schema=FailureReviewAnswer.model_json_schema()
            )
            for a in request.request.legal_actions
        )
    if isinstance(
        request,
        InputReviewRequest
        | InvestigationDownloadReviewRequest
        | RunHealthReviewRequest
        | PaperReviewRequest,
    ):
        return tuple(
            ReviewLegalAction(
                action=a, response_schema=FailureReviewAnswer.model_json_schema()
            )
            for a in request.request.legal_actions
        )
    if isinstance(request, AgentEffortReviewRequestView):
        return tuple(
            ReviewLegalAction(
                action=action, response_schema=response.model_json_schema()
            )
            for action, response in (
                ("extend", ExtendAgentEffortResponse),
                ("stop", StopAgentEffortResponse),
            )
        )
    if isinstance(request, FinalizationReviewRequestView):
        actions = []
        for action, command in finalization_response_commands().items():
            if action not in request.legal_actions:
                continue
            schema = command.model_json_schema()
            for field in ("journey_file", "request_event_id"):
                schema["properties"].pop(field)
                schema["required"].remove(field)
            schema["properties"]["action"] = {"type": "string", "const": action}
            schema["required"].append("action")
            actions.append(ReviewLegalAction(action=action, response_schema=schema))
        return tuple(actions)
    if isinstance(request, InvestigationReviewRequestView):
        responses = {
            "clarify": ClarifyInvestigationResponse,
            "reject": RejectInvestigationResponse,
            "approve": ApproveInvestigationResponse,
            "revise": ReviseInvestigationResponse,
            "stop": StopInvestigationResponse,
        }
        if request.question is not None and request.question.issue is not None:
            # An issue finding's own Review answers that member alone.
            responses |= {
                "approve": ApproveInvestigationFindingResponse,
                "amend": AmendInvestigationFindingResponse,
            }
        return tuple(
            ReviewLegalAction(
                action=action, response_schema=responses[action].model_json_schema()
            )
            for action in request.legal_actions
        )
    return tuple(
        ReviewLegalAction(action=item.action, response_schema=item.response_schema)
        for item in request.legal_actions
    )


class _ReviewDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_event_id: StrictStr
    legal_actions: tuple[ReviewLegalAction, ...] = ()
    request: (
        InputReviewRequest
        | PaperReviewRequest
        | PreparationRequest
        | PlanningHitlRequestView
        | AgentFailureReviewRequestView
        | ExecutionPathReviewRequestView
        | ModelAvailabilityRequestDetails
        | AgentEffortReviewRequestView
        | ExecutionHitlRequestView
        | InvestigationReviewRequestView
        | InvestigationDownloadReviewRequest
        | RunHealthReviewRequest
        | FinalizationReviewRequestView
    )

    @model_validator(mode="after")
    def exact_request(self):
        if self.request_event_id != self.request.request_event_id:
            raise ValueError("Review changed its request identity")
        expected = _legal_actions(self.request)
        if self.legal_actions and self.legal_actions != expected:
            raise ValueError("Review changed its stage-owned legal responses")
        object.__setattr__(self, "legal_actions", expected)
        return self


class JunctionReviewDetails(_ReviewDetails):
    kind: Literal["junction"] = "junction"
    command_frontier: Literal["answer"] = "answer"
    request: AgentFailureReviewRequestView


class ExecutionPathReviewDetails(_ReviewDetails):
    kind: Literal["execution_path"] = "execution_path"
    command_frontier: Literal["answer"] = "answer"
    request: ExecutionPathReviewRequestView


class PreparationReviewDetails(_ReviewDetails):
    kind: Literal["preparation"] = "preparation"
    command_frontier: Literal["answer"] = "answer"
    request: PreparationRequest


class PlanningReviewDetails(_ReviewDetails):
    kind: Literal["planning"] = "planning"
    command_frontier: Literal["answer"] = "answer"
    request: PlanningHitlRequestView


class ModelAvailabilityReviewDetails(_ReviewDetails):
    kind: Literal["model_availability"] = "model_availability"
    command_frontier: Literal["answer"] = "answer"
    request: ModelAvailabilityRequestDetails


class AgentEffortReviewDetails(_ReviewDetails):
    kind: Literal["agent_effort"] = "agent_effort"
    command_frontier: Literal["answer"] = "answer"
    request: AgentEffortReviewRequestView


class AgentWorkReviewDetails(_ReviewDetails):
    kind: Literal["agent_work"] = "agent_work"
    command_frontier: Literal["answer"] = "answer"
    request: AgentWorkReviewRequestView


class ExecutionReviewDetails(_ReviewDetails):
    kind: Literal["execution"] = "execution"
    command_frontier: Literal["answer"] = "answer"
    request: ExecutionHitlRequestView


class InvestigationReviewDetails(_ReviewDetails):
    kind: Literal["investigation"] = "investigation"
    command_frontier: Literal["answer"] = "answer"
    request: InvestigationReviewRequestView


class InvestigationDownloadReviewDetails(_ReviewDetails):
    kind: Literal["investigation_download"] = "investigation_download"
    command_frontier: Literal["answer"] = "answer"
    request: InvestigationDownloadReviewRequest


class RunHealthReviewDetails(_ReviewDetails):
    """A failed health measurement's Review (#414): retry or abort."""

    kind: Literal["run_health"] = "run_health"
    command_frontier: Literal["answer"] = "answer"
    request: RunHealthReviewRequest


class FinalizationReviewDetails(_ReviewDetails):
    kind: Literal["finalization"] = "finalization"
    command_frontier: Literal["finalization"] = "finalization"
    request: FinalizationReviewRequestView


class InputReviewDetails(_ReviewDetails):
    kind: Literal["input"] = "input"
    command_frontier: Literal["answer"] = "answer"
    request: InputReviewRequest


class PaperReviewDetails(_ReviewDetails):
    """A refused Run paper's Review (#420): retry, amend, continue or abort."""

    kind: Literal["paper"] = "paper"
    command_frontier: Literal["answer"] = "answer"
    request: PaperReviewRequest


ReviewDetails = Annotated[
    InputReviewDetails
    | PaperReviewDetails
    | JunctionReviewDetails
    | ExecutionPathReviewDetails
    | PreparationReviewDetails
    | PlanningReviewDetails
    | ModelAvailabilityReviewDetails
    | AgentEffortReviewDetails
    | AgentWorkReviewDetails
    | ExecutionReviewDetails
    | InvestigationReviewDetails
    | InvestigationDownloadReviewDetails
    | RunHealthReviewDetails
    | FinalizationReviewDetails,
    Field(discriminator="kind"),
]


def _run_papers(observation: JourneyObservation):
    from bio_neb.application.queries import JourneyReview

    return observation.run_papers if isinstance(observation, JourneyReview) else None


def _run_health(observation: JourneyObservation):
    from bio_neb.application.queries import JourneyReview

    return observation.run_health if isinstance(observation, JourneyReview) else None


def _stage_reviews(observation: JourneyObservation):
    """The one correspondence between stage projections and shared Review types."""
    return (
        (
            PreparationReviewDetails,
            getattr(observation, "preparation_review", None),
            "requests",
        ),
        (InputReviewDetails, getattr(observation, "input_selection", None), "requests"),
        # A paper Review waits before Execution, so only a Run's Journey view
        # carries its papers.
        (PaperReviewDetails, _run_papers(observation), "requests"),
        (PlanningReviewDetails, observation.planning_hitl, "requests"),
        (
            ModelAvailabilityReviewDetails,
            observation.model_availability_review,
            "requests",
        ),
        (AgentEffortReviewDetails, observation.agent_effort_review, "requests"),
        (AgentWorkReviewDetails, observation.agent_work_review, "requests"),
        (JunctionReviewDetails, observation.operational_review, "requests"),
        (ExecutionReviewDetails, observation.execution_hitl, "requests"),
        (InvestigationReviewDetails, observation.investigation, "review_requests"),
        (
            InvestigationDownloadReviewDetails,
            observation.investigation_download_review,
            "requests",
        ),
        (RunHealthReviewDetails, _run_health(observation), "requests"),
        (FinalizationReviewDetails, observation.finalization, "requests"),
    )


def _details(details_type, request) -> ReviewDetails:
    if details_type is ModelAvailabilityReviewDetails:
        request = ModelAvailabilityRequestDetails.model_validate(
            request.model_dump(mode="python")
        )
    return details_type(request_event_id=request.request_event_id, request=request)


def pending_reviews(observation: JourneyObservation) -> tuple[ReviewDetails, ...]:
    """All independent requests, preserving the domain's declared current scope."""
    ended_requests = set()
    finalization = observation.finalization
    if finalization.bases and observation.status in {
        "finalization_review_pending",
        "finalization_approved",
        "finalization_stopped",
        "finalization_returned_to_investigation",
        "grounded_report_published",
    }:
        basis = finalization.bases[-1]
        # A checked Run abort ends its still-unanswered questions. Their exact
        # history remains in the frozen accounting, without a synthetic answer.
        if basis.outcome == "failed" and any(
            item.event_type == "RunFailed" for item in basis.accounting
        ):
            ended_requests = {item.source_event_id for item in basis.accounting}
    return tuple(
        _details(details_type, request)
        for details_type, projection, field in _stage_reviews(observation)
        if projection is not None
        for request in getattr(projection, field)
        if request.pending and request.request_event_id not in ended_requests
    )


def current_review(observation: JourneyObservation) -> ReviewDetails | None:
    """Select the presentation focus; other scoped requests remain addressable."""
    pending = pending_reviews(observation)
    downloads = [r for r in pending if r.kind == "investigation_download"]
    if downloads and all(
        r.kind in {"investigation_download", "investigation"} for r in pending
    ):
        # A grant's refused download stops its observation; the remaining
        # items' Investigation Review stays open beside it.
        return downloads[0]
    if len(pending) > 1:
        operational = [
            r
            for r in pending
            if r.kind
            in {
                "model_availability",
                "agent_effort",
                "agent_work",
                "junction",
            }
        ]
        # A table reading stop holds only that stage; the remaining items'
        # Investigation review stays open beside it (ADR 0047).
        item_queries = all(r.request.agent_role == "table_reader" for r in operational)
        if operational and all(
            r in operational
            or (r.kind == "planning" and r.request.owner == "requirement_resolution")
            or (
                r.kind == "investigation"
                and (r.request.question is not None or item_queries)
            )
            for r in pending
        ):
            return operational[0]
        if all(
            item.kind == "investigation" and item.request.question is not None
            for item in pending
        ):
            return pending[0]
        if all(item.kind == "planning" for item in pending):
            return next(
                item
                for item in pending
                if item.request_event_id
                == observation.planning_hitl.pending_request.request_event_id
            )
        if (
            len(pending) == 2
            and {item.kind for item in pending} == {"investigation", "finalization"}
            and _partial_finalization_is_current(observation)
        ):
            return next(item for item in pending if item.kind == "finalization")
        raise ValueError("Journey has more than one pending Review")
    return pending[0] if pending else None


def _partial_finalization_is_current(observation: JourneyObservation) -> bool:
    """Retain pending Investigation work inside its exact current Report review."""
    investigation = observation.investigation
    finalization = observation.finalization
    request = investigation.pending_request
    final_request = finalization.pending_request
    if request is None or request.proposal is None or final_request is None:
        return False
    basis = final_request.basis
    return (
        observation.status == "finalization_review_pending"
        and investigation.status == "review_pending"
        and finalization.status == "review_pending"
        and request.pending
        and final_request.pending
        and bool(investigation.pending_work_item_ids)
        and basis.collection_status == "partial"
        and basis.question is not None
        and basis.question.assertion_id == request.proposal.assertion_id
        and basis.work_items
        == tuple(
            sorted(request.proposal.work_items, key=lambda item: item.work_item_id)
        )
        and set(investigation.pending_work_item_ids)
        == {
            item.work_item_id
            for item in basis.collection_item_states
            if item.status == "approval_pending"
        }
    )


def review_for_request(
    observation: JourneyObservation, request_event_id: str
) -> ReviewDetails | None:
    """Resolve retained identity, including answered requests for stage-owned replay.

    The stage projections register every family the checked Run view carries; the
    execution operation's own reviews are read from its recorded Execution path.
    """
    from bio_neb.journey_path import has_execution_path

    matches = tuple(
        _details(details_type, request)
        for details_type, projection, requests_field in _stage_reviews(observation)
        if projection is not None
        for request in getattr(projection, requests_field)
        if request.request_event_id == request_event_id
    )
    if has_execution_path(observation.journey_file):
        matches += tuple(
            _details(ExecutionPathReviewDetails, request)
            for request in review_requests(observation.journey_file)
            if request.request_event_id == request_event_id
        )
    if len(matches) > 1:
        raise ValueError("Review request identity has more than one owner")
    return matches[0] if matches else None
