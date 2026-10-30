"""The coordinator's one resolver from a stopped Attempt to the Agent work it re-enters.

Every Agent review names the Attempt it stopped, and that Attempt's ``AttemptStarted``
records its assignment. Re-entry selects the one candidate assignment whose own
identity is the recorded one, so coordinator code names no Agent role (ADR 0030).
Each assignment still owns its stage eligibility when it admits the re-entry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bio_neb.agentic_journey.agent_roles import AttemptAssignment
    from bio_neb.application.queries import JourneyObservation

# A checked re-entry status (the projection's ``REENTRY_STATUSES``) -> the review
# it answered and the ``AttemptAssignment.run`` argument that consumes that answer.
_ANSWERED_REVIEWS = {
    "operational_review_direction_recorded": (
        "operational_review",
        "operational_review_response_event_id",
    ),
    "effort_review_direction_recorded": (
        "agent_effort_review",
        "effort_review_response_event_id",
    ),
    "agent_work_review_direction_recorded": (
        "agent_work_review",
        "work_review_response_event_id",
    ),
    "model_availability_review_direction_recorded": (
        "model_availability_review",
        "availability_review_response_event_id",
    ),
}


class AgentReentryRefused(ValueError):
    """No current Agent work can re-enter the stopped Attempt; nothing was written."""


@dataclass(frozen=True)
class AnsweredAgentWork:
    """The stopped Attempt an answered review reopens and the answer it consumes."""

    attempt_id: str
    direction: Mapping[str, str]


# The ``AttemptStarted`` field that records which answer an Attempt consumed.
_CONSUMED_BY = {
    "operational_review_response_event_id": "continuation_operational_review_event_id",
    "effort_review_response_event_id": "continuation_effort_review_event_id",
    "work_review_response_event_id": "continuation_work_review_event_id",
    "availability_review_response_event_id": (
        "continuation_availability_review_event_id"
    ),
}


def answered_agent_work(
    observation: JourneyObservation, events: Sequence[Mapping[str, Any]]
) -> AnsweredAgentWork:
    """Read the latest answered Agent review no Attempt in ``events`` consumed yet.

    Collection members stop on their own Reviews (#105), so an earlier answer
    may still await its re-entry after a later one was consumed.
    """
    review_name, argument = _ANSWERED_REVIEWS[observation.status]
    review = getattr(observation, review_name)
    consumed = {
        event["payload"].get(_CONSUMED_BY[argument])
        for event in events
        if event["event_type"] == "AttemptStarted"
    }
    receipt = next(
        (
            item
            for item in reversed(review.receipts)
            if item.response_event_id not in consumed
            and item.response.action not in {"stop", "reject", "dismiss"}
        ),
        review.receipts[-1],
    )
    request = next(
        item
        for item in review.requests
        if item.request_event_id == receipt.request_event_id
    )
    return AnsweredAgentWork(request.attempt_id, {argument: receipt.response_event_id})


def recorded_assignment(events: Sequence[Mapping[str, Any]], attempt_id: str) -> str:
    """The assignment identity the stopped Attempt recorded when it started."""
    recorded = [
        event["payload"].get("agent_role")
        for event in events
        if event["event_type"] == "AttemptStarted"
        and event["payload"].get("attempt_id") == attempt_id
    ]
    if len(recorded) != 1 or not isinstance(recorded[0], str):
        raise AgentReentryRefused(
            "stopped Attempt does not record one exact assignment"
        )
    return recorded[0]


def reentry_assignment(
    events: Sequence[Mapping[str, Any]],
    attempt_id: str,
    candidates: Iterable[AttemptAssignment],
) -> AttemptAssignment:
    """Select the Agent work that re-enters the stopped Attempt, before any write.

    ``candidates`` are the coordinator's assignments, built lazily in order. An
    Attempt recording an assignment no candidate carries, such as a retired
    role, is refused.
    """
    recorded = recorded_assignment(events, attempt_id)
    for candidate in candidates:
        if candidate.agent_role != recorded:
            continue
        return candidate
    raise AgentReentryRefused(
        f"no Agent work re-enters the recorded assignment {recorded!r}"
    )


def coordinator_assignments(
    planning: Callable[[], AttemptAssignment] | None = None,
    investigation_dependencies: Mapping[str, Any] | None = None,
    table_reading_dependencies: Mapping[str, Any] | None = None,
    explanation_dependencies: Mapping[str, Any] | None = None,
    paper_reading_dependencies: Mapping[str, Any] | None = None,
) -> Iterable[AttemptAssignment]:
    """The coordinator's Agent assignments, each built only when the resolver reaches it.

    Investigation, the table reader, the report's explanation and the paper reader
    come first:
    building them has no effect, so the composition's planning factory runs only
    when planning is the recorded assignment.
    """
    from bio_neb.investigation.investigation_assignment import investigation_assignment
    from bio_neb.investigation.vcf_table_reading import table_reading_assignment
    from bio_neb.paper_reading.reading import paper_reading_assignment
    from bio_neb.reporting.explanation_agent import explanation_assignment

    yield investigation_assignment(**(investigation_dependencies or {}))
    yield table_reading_assignment(**(table_reading_dependencies or {}))
    yield explanation_assignment(**(explanation_dependencies or {}))
    yield paper_reading_assignment(**(paper_reading_dependencies or {}))
    if planning is None:
        from bio_neb.agentic_journey.planning_assignment import planning_assignment

        planning = planning_assignment
    yield planning()
