"""Retain scoped review requests and human answers without executing effects."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import StrictBool, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, ReviewAnswer, validate_command_input
from bio_neb.journey_path import (
    Answered,
    JourneyPath,
    Outcome,
    PathFact,
    Pending,
    exceeds_declared_authority,
)
from bio_neb.junction import evaluate_scoped_answer

_REQUEST_EVENT_TYPES = ("CpuPreparationReviewRequested", "FailureReviewRequested")


class ExecutionPathReviewRequestView(FrozenContract):
    """One retained CPU preparation or failure review, addressed by its request event."""

    request_event_id: StrictStr
    review: Pending | Ask
    pending: StrictBool

    @property
    def legal_actions(self):
        return self.review.legal_actions


def review_requests(journey_file) -> tuple[ExecutionPathReviewRequestView, ...]:
    """Every retained execution review; only the current pass's open one is pending."""
    journey = JourneyPath.open(Path(journey_file))
    state = journey.state
    open_requests = {
        state.latest_event_ids.get(event_type)
        for event_type, review in zip(
            _REQUEST_EVENT_TYPES, (state.review, state.failure_review), strict=True
        )
        if isinstance(review, (Pending, Ask))
    }
    return tuple(
        ExecutionPathReviewRequestView(
            request_event_id=event.event_id,
            review=event.payload,
            pending=event.event_id in open_requests,
        )
        for event in journey.read()
        if event.event_type in _REQUEST_EVENT_TYPES
    )


class ReviewAnswerExceedsAuthority(ValueError):
    """The answer chose what its request did not declare: refused before any write."""

    code = "review_answer_exceeds_authority"

    def __init__(self, request, answer):
        super().__init__("review answer exceeds declared authority")
        self.fields = (
            "answer.response.field"
            if answer.action in request.legal_actions
            else "answer.response.action",
        )


def request_review(state, plan, request, *, journey, contact_journal, journal=None):
    """Close this pass at a declared request. Requests grant no execution authority."""
    from bio_neb.parabricks.runner import record_outcome

    if isinstance(request, Ask) and not set(request.amendable) <= plan.amendable:
        raise ValueError("review amendment exceeds the plan")
    failure = isinstance(request, Ask)
    name = "FailureReviewRequested" if failure else "CpuPreparationReviewRequested"
    dependency = (
        request.failure_event_id
        if failure
        else state.latest_event_ids["ReferenceResolved"]
    )
    _, state = journey.path(
        state,
        PathFact(
            event_type=name,
            kind="review",
            grounding="agent_composed",
            **{"pass": state.pass_},
            rests_on=(dependency,),
            payload=request,
        ),
    )
    return record_outcome(
        state,
        journey,
        PathFact(
            event_type="ExecutionOutcome",
            kind="outcome",
            grounding="agent_composed",
            **{"pass": state.pass_},
            rests_on=(state.latest_event_ids[name],),
            payload=Outcome(disposition="incomplete", contact_journal=contact_journal),
        ),
        journal=journal,
    )


def _plan(journey, state):
    from bio_neb.parabricks.execution_proposal import current_execution_approval
    from bio_neb.parabricks.plan import (
        ApprovedParabricksInput,
        RetainedExecution,
        plan_execution,
    )

    if state.bound is None:
        raise ValueError("review requires a bound execution")
    approved = ApprovedParabricksInput.model_validate(state.bound.approved_input)
    return plan_execution(
        approved,
        RetainedExecution(
            state, *current_execution_approval(journey.planning_events())
        ),
    )


def answer_review(journey_file, request_event_id, response):
    """Answer one exact execution request, leaving provider execution to its runner."""
    from bio_neb.parabricks.runner import record_outcome

    journey = JourneyPath.open(Path(journey_file))
    state = journey.state
    _plan(journey, state)  # An answer requires the bound Execution's current plan.
    events = journey.read()
    request_event = next(
        (event for event in events if event.event_id == request_event_id), None
    )
    if request_event is None or not isinstance(request_event.payload, (Pending, Ask)):
        raise ValueError("review answer has no matching request")
    request = request_event.payload
    failure = isinstance(request, Ask)
    answer_type = ReviewAnswer if failure else Answered
    response = dict(response)
    response.setdefault("by", "owner")
    if not failure:
        response["action"] = {
            "approve": "approve_cpu_preparation",
            "decline": "abort",
        }.get(response.get("action"), response.get("action"))
    answer = validate_command_input(
        answer_type, "answer.response", {**response, "request_id": request.request_id}
    )
    name = "FailureReviewAnswered" if failure else "CpuPreparationReviewAnswered"

    @dataclass(frozen=True)
    class RetainedReview:
        request_event_id: str
        pending: bool
        response: ReviewAnswer | Answered | None = None

    receipts = tuple(
        RetainedReview(event.rests_on[0], False, event.payload)
        for event in events
        if event.event_type == name
    )
    requests = (
        RetainedReview(
            request_event_id,
            state.latest_event_ids.get(request_event.event_type) == request_event_id
            and isinstance(
                state.failure_review if failure else state.review, (Pending, Ask)
            ),
        ),
    )

    def refusal(code, identity):
        return ValueError(
            "review already answered differently"
            if code == "conflict"
            else "new pass must answer its current review"
        )

    fact = PathFact(
        event_type=name,
        kind="authority",
        grounding="human",
        **{"pass": state.pass_ + 1},
        rests_on=(request_event_id,),
        payload=answer,
    )

    def validate(pending, response):
        if exceeds_declared_authority(request, response):
            raise ReviewAnswerExceedsAuthority(request, response)
        if isinstance(request, Pending) and request.action_destinations:
            from bio_neb.failure import continuation_destination

            continuation_destination(request, response)
        candidate = journey.preview(state, fact)
        _plan(journey, candidate)

    _, replay = evaluate_scoped_answer(
        requests,
        receipts,
        request_event_id,
        answer,
        validate=validate,
        refusal=refusal,
    )
    if replay is not None:
        return state
    _, state = journey.path(state, fact)
    if answer.action == "abort":
        return record_outcome(
            state,
            journey,
            PathFact(
                event_type="ExecutionOutcome",
                kind="outcome",
                grounding="human",
                **{"pass": state.pass_},
                rests_on=(state.latest_event_ids[name],),
                payload=Outcome(
                    disposition="failed",
                    contact_journal=events[-1].payload.contact_journal,
                    journal_sha256=events[-1].payload.journal_sha256,
                    journal_bytes=events[-1].payload.journal_bytes,
                ),
            ),
        )
    return state
