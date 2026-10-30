"""Exact Agent-review consumption through the current causal Journey head.

The answer grants scoped continuation; intervening independent answers and Pass
closure remain ancestors of new work. Review owners validate policy and retained
transcripts. This owner binds their existing typed directions, without replaying
Tools or interpreting mechanical progress.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from bio_neb.recording import is_contact_evidence


@dataclass(frozen=True)
class ReviewContinuation:
    field: str
    request_kind: str
    response_kind: str
    actions: frozenset[str]


REVIEW_CONTINUATIONS = (
    ReviewContinuation(
        "continuation_availability_review_event_id",
        "ModelAvailabilityReviewRequested",
        "ModelAvailabilityReviewResponded",
        frozenset({"retry", "amend"}),
    ),
    ReviewContinuation(
        "continuation_effort_review_event_id",
        "AgentEffortReviewRequested",
        "AgentEffortReviewResponded",
        frozenset({"extend"}),
    ),
    ReviewContinuation(
        "continuation_work_review_event_id",
        "AgentWorkReviewRequested",
        "AgentWorkReviewResponded",
        frozenset({"continue"}),
    ),
    ReviewContinuation(
        "continuation_operational_review_event_id",
        "AgentFailureReviewRequested",
        "AgentFailureReviewResponded",
        frozenset({"retry", "amend"}),
    ),
)
REVIEW_RESPONSE_KINDS = frozenset(item.response_kind for item in REVIEW_CONTINUATIONS)


@dataclass(frozen=True)
class ContinuationBinding:
    kind: ReviewContinuation
    request: Mapping[str, Any]
    answer: Mapping[str, Any]
    previous_attempt: Mapping[str, Any]
    causal_head: str


def continuation_head(events):
    """Current causal head: a closed Pass counts, its active start never does."""
    return next(
        event["event_id"]
        for event in reversed(events)
        if event["event_type"] != "PassStarted" and not is_contact_evidence(event)
    )


def review_answer_parents(events, request_event_id):
    return list(dict.fromkeys((request_event_id, continuation_head(events))))


def validate_review_answer_parentage(events, answer):
    prefix = events[: events.index(answer)]
    if answer["parent_ids"] != review_answer_parents(
        prefix, answer["payload"]["request_event_id"]
    ):
        raise ValueError("Agent review answer changed its exact request or causal head")


def _ancestors(events, event_id):
    by_id = {event["event_id"]: event for event in events}
    pending = [event_id]
    ancestors = set()
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        if current not in by_id:
            raise ValueError("continuation contains an unknown causal parent")
        ancestors.add(current)
        pending.extend(by_id[current]["parent_ids"])
    return ancestors


def pending_continuation(events, response_event_id, *, agent_role):
    """Bind one unconsumed answer before an assignment creates its lifecycle."""
    events = list(events)
    by_id = {event["event_id"]: event for event in events}
    answer = by_id.get(response_event_id)
    kind = next(
        (
            item
            for item in REVIEW_CONTINUATIONS
            if answer is not None and item.response_kind == answer["event_type"]
        ),
        None,
    )
    if kind is None:
        raise ValueError("continuation lacks a typed Agent review answer")
    request = by_id.get(answer["payload"].get("request_event_id"))
    if request is None or request["event_type"] != kind.request_kind:
        raise ValueError("continuation answer changed its exact request kind")
    starts = [
        event
        for event in events
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == request["payload"]["attempt_id"]
    ]
    if len(starts) != 1 or any(
        value != agent_role
        for value in (
            request["payload"]["agent_role"],
            starts[0]["payload"]["agent_role"],
        )
    ):
        raise ValueError("continuation changed its owning Attempt or Agent role")
    if answer["payload"]["response"]["action"] not in kind.actions:
        raise ValueError("continuation answer does not authorize further work")
    if any(
        event["event_type"] == "AttemptStarted"
        and any(
            event["payload"].get(item.field) == response_event_id
            for item in REVIEW_CONTINUATIONS
        )
        for event in events
    ):
        raise ValueError("continuation requires exact unused authority")
    validate_review_answer_parentage(events, answer)
    head = continuation_head(events)
    if response_event_id not in _ancestors(events, head):
        raise ValueError("continuation head omitted its exact answer")
    return ContinuationBinding(kind, request, answer, starts[0], head)


def validate_attempt_continuation(events, started, *, checked_once=None):
    """Check the existing Attempt fields and current lifecycle parent together.

    The result is fixed by the prefix up to ``started``, so ``checked_once``, the
    reading owner's when ``events`` are its evidence, checks each start once.
    """
    if checked_once is None:
        return _check_attempt_continuation(events, started)
    return checked_once(
        ("attempt_continuation", started["event_id"]),
        lambda: _check_attempt_continuation(events, started),
    )


def _check_attempt_continuation(events, started):
    selected = [
        (item, started["payload"].get(item.field))
        for item in REVIEW_CONTINUATIONS
        if started["payload"].get(item.field) is not None
    ]
    if not selected:
        prefix = events[: events.index(started)]
        by_id = {event["event_id"]: event for event in prefix}
        consumed = {
            event["payload"].get(item.field)
            for event in prefix
            if event["event_type"] == "AttemptStarted"
            for item in REVIEW_CONTINUATIONS
        }
        for answer in prefix:
            kind = next(
                (
                    item
                    for item in REVIEW_CONTINUATIONS
                    if item.response_kind == answer["event_type"]
                ),
                None,
            )
            if (
                kind is None
                or answer["event_id"] in consumed
                or answer["payload"]["response"]["action"] not in kind.actions
            ):
                continue
            request = by_id.get(answer["payload"].get("request_event_id"))
            if (
                request is not None
                and request["payload"].get("agent_role")
                == started["payload"]["agent_role"]
            ):
                raise ValueError(
                    "Agent review continuation lost its exact direction binding"
                )
        return None
    if len(selected) != 1:
        raise ValueError("Attempt cannot consume two review directions")
    kind, answer_id = selected[0]
    by_id = {event["event_id"]: event for event in events}
    parents = started["parent_ids"]
    lifecycle = by_id.get(parents[0]) if len(parents) == 1 else None
    if lifecycle is None or lifecycle["event_type"] != "activity_started":
        raise ValueError("continuation lacks its exact lifecycle start")
    prefix = list(events[: events.index(lifecycle)])
    binding = pending_continuation(
        prefix, answer_id, agent_role=started["payload"]["agent_role"]
    )
    if binding.kind != kind or lifecycle["parent_ids"] != [binding.causal_head]:
        raise ValueError(
            "continuation changed its typed direction or current causal head"
        )
    if (
        started["payload"]["input_set_id"]
        != binding.previous_attempt["payload"]["input_set_id"]
    ):
        raise ValueError("continuation changed its input scope")
    return binding


@dataclass(frozen=True)
class ReentryDirection:
    """What one Review answer grants the Attempt that resumes a stopped one.

    ``previous_attempt_id`` is the stopped Attempt (``None`` for fresh work);
    ``instruction`` is an amend answer's researcher text; a retry answer retries
    from its Review's frontier, any other answer from the causal head.
    """

    causal_head: str
    availability_frontier: str
    previous_attempt_id: str | None = None
    instruction: str | None = None


def review_direction(journey, reentry, *, agent_role) -> ReentryDirection:
    """Read a role's re-entry from the one Review answer it consumes, if any."""
    from bio_neb.agentic_journey.agent_work_review import pending_agent_work
    from bio_neb.agentic_journey.effort import pending_effort_work
    from bio_neb.agentic_journey.model_availability import (
        model_availability_direction,
    )
    from bio_neb.agentic_journey.operational_review import pending_operational_work
    from bio_neb.recording import read_evidence

    response_id = (
        reentry.availability_review_response_event_id
        or reentry.operational_review_response_event_id
        or reentry.effort_review_response_event_id
        or reentry.work_review_response_event_id
    )
    if response_id is None:
        head = journey.prefix_event_ids[-1]
        return ReentryDirection(head, head)
    causal_head = pending_continuation(
        read_evidence(journey), response_id, agent_role=agent_role
    ).causal_head
    frontier = causal_head
    instruction = None
    if reentry.work_review_response_event_id is not None:
        request, instruction = pending_agent_work(
            journey, reentry.work_review_response_event_id, agent_role=agent_role
        )
    elif reentry.effort_review_response_event_id is not None:
        request, _messages, _overrides = pending_effort_work(
            journey, reentry.effort_review_response_event_id, agent_role=agent_role
        )
    elif reentry.availability_review_response_event_id is not None:
        request, receipt = model_availability_direction(
            journey,
            reentry.availability_review_response_event_id,
            agent_role=agent_role,
        )
        if receipt.response.action == "retry":
            frontier = request.retry_frontier_event_id
        elif receipt.response.action == "amend":
            instruction = receipt.response.instruction
    else:
        request, receipt = pending_operational_work(
            journey,
            reentry.operational_review_response_event_id,
            agent_role=agent_role,
        )
        if receipt.response.action == "retry":
            frontier = request.retry_frontier_event_id
        elif receipt.response.action == "amend":
            instruction = receipt.response.value
    return ReentryDirection(causal_head, frontier, request.attempt_id, instruction)
