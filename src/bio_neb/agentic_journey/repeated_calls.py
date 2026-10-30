"""Repeated calls: the same call again with nothing changed (#282 decision 9; ADR 0044).

Within an Attempt, a lawful call whose Tool and canonical effective arguments
equal an earlier admitted call's is stopped before contact when nothing
meaningful has changed since that call completed. Meaning only counts: a Journey
search reads its result again over the Journey without the loop's own records,
and an equal result is no change; for any other Tool, no event of the Attempt
since the earlier call's receipt, other than the loop's own records, is no
change. A fetch's own retrieval time is part of its own result, so it is no
change either. Each stop is an ordinary refused call (``repeated_call``): one
more failed turn when nothing else of its reply works, so the failure rule's
limit opens the Agent work review (#447 B).
Each repeat is shown, never stored, as ``RepeatProgress``: live as it happens
and in the Run's progress.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import Field, StrictInt, StrictStr

from bio_neb.agentic_journey.action_policy import ProposedAgentAction, call_identity
from bio_neb.contracts import FrozenContract

# The refusal code of a call stopped as a repeat, and its fault key's prefix.
REPEATED_CALL = "repeated_call"
_REPEAT_KEY = REPEATED_CALL + ": "

# The Agent loop's own records: each turn adds its request, response, routing,
# validations, receipts, searches, output checks and lifecycle, which never
# change what a call would find. A completed collection member is one of them:
# the count runs across the whole Attempt.
_LOOP_RECORDS = frozenset(
    {
        "ModelRequestRecorded",
        "ModelResponseRecorded",
        "ModelTransportRequestRecorded",
        "ModelTransportResponseRecorded",
        "ModelCompletionTruncated",
        "ToolCallValidated",
        "AgentTurnRouted",
        "AgentActionTurnDispositionRecorded",
        "JourneySearchRead",
        "ObservedToolExecutionStarted",
        "ObservedToolExecutionSucceeded",
        "ObservedToolExecutionFailed",
        "ToolExecutionRefused",
        "StructuredOutputCallRecorded",
        "StructuredOutputValidationLinked",
        "AgentOutputValidated",
    }
)


def is_loop_record(event: Mapping[str, Any]) -> bool:
    """Whether an event is the Agent loop's own record, never a change of meaning."""
    from bio_neb.recording import is_contact_evidence

    return (
        event["event_type"] in _LOOP_RECORDS
        or event["event_type"].startswith("activity_")
        or is_contact_evidence(event)
    )


def repeat_key(tool_name: str, earlier_validation_event_id: str) -> str:
    """The fault key of a repeat: the earlier call it repeats.

    A changed result lets the call run again, so its later repeats repeat that
    newer call and count afresh.
    """
    return f"{_REPEAT_KEY}{tool_name}: {earlier_validation_event_id}"


def is_repeat_key(key: str) -> bool:
    return key.startswith(_REPEAT_KEY)


@dataclass(frozen=True)
class CallHistory:
    """Whose earlier calls a call repeats, and what counts as a change since one.

    ``attempt_ids`` own the earlier calls; an owned event other than the loop's own
    records is a change unless ``bookkeeping`` says it changes nothing (a later
    Attempt's start), and an event ``changes`` names is one whoever owns it (a
    researcher's answer to the work).
    """

    attempt_ids: frozenset[str]
    bookkeeping: Callable[[Mapping[str, Any]], bool] = lambda _event: False
    changes: Callable[[Mapping[str, Any]], bool] = lambda _event: False


CallHistoryReader = Callable[[list, str], CallHistory]


def attempt_call_history(events: list, attempt_id: str) -> CallHistory:
    """The default history: the Attempt and its exact provider continuations."""
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    return CallHistory(
        frozenset({*prior_model_attempt_ids(events, attempt_id), attempt_id})
    )


def repeated_call_of(
    events: Iterable[Mapping[str, Any]],
    *,
    attempt_id: str,
    action: ProposedAgentAction,
    history: CallHistoryReader = attempt_call_history,
) -> str | None:
    """The validation of the admitted call ``action`` repeats with nothing changed.

    The latest completed admitted call of the work's history with the same
    identity is the earlier call; without one, or after a meaningful change, the
    call is no repeat (None). The history is the Attempt's provider continuations
    by default; a role may read its work's own (#479 G2.3: an Investigation item's).
    """
    from bio_neb.agentic_journey.assignment_context import SEARCH_TOOL_NAME

    events = list(events)
    calls = history(events, attempt_id)
    owned = calls.attempt_ids
    position = {event["event_id"]: index for index, event in enumerate(events)}
    identity = call_identity(action)
    earlier = None
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        if (
            event["event_type"] != "ObservedToolExecutionSucceeded"
            or payload.get("attempt_id") not in owned
        ):
            continue
        start = events[position[payload["started_event_id"]]]["payload"]
        validation = events[position[start["validation_event_id"]]]["payload"]
        if (
            call_identity(
                ProposedAgentAction(
                    tool_name=validation["tool_name"],
                    tool_call_id=validation["tool_call_id"],
                    arguments=validation["arguments"],
                )
            )
            == identity
        ):
            earlier = (
                index,
                position[payload["stage_result_event_id"]],
                start["validation_event_id"],
            )
    if earlier is None:
        return None
    receipt, result, validation_id = earlier
    if action.tool_name == SEARCH_TOOL_NAME:
        unchanged = _search_result(events[:result], events[result]) == _search_result(
            events, events[result]
        )
    else:
        unchanged = not any(
            calls.changes(event)
            or (
                event["payload"].get("attempt_id") in owned
                and not is_loop_record(event)
                and not calls.bookkeeping(event)
            )
            for event in events[receipt + 1 :]
        )
    return validation_id if unchanged else None


def _search_result(events, read) -> Any:
    """A recorded search's query read again over ``events`` without loop records.

    An event the query names stays, and the page's own position is dropped.
    """
    from bio_neb.agentic_journey.assignment_context import search_journey_events

    query = read["payload"]["query"]
    named = {
        *query["event_ids"],
        query["before_event_id"],
        (query.get("payload_slice") or {}).get("event_id"),
    }
    result = search_journey_events(
        [e for e in events if not is_loop_record(e) or e["event_id"] in named],
        read["journey_id"],
        **query,
    )
    result.pop("through_event_id")
    return result


class RepeatProgress(FrozenContract):
    """One repeat of an Attempt's call: ``repeat`` of the Attempt's ``cap``.

    ``source_event_id`` is the repeat's refused ``ToolCallValidated``.
    """

    attempt_id: StrictStr
    tool_name: StrictStr
    repeat: StrictInt = Field(gt=0)
    cap: StrictInt = Field(gt=0)
    source_event_id: StrictStr


def journey_repeats(events: Iterable[Mapping[str, Any]]) -> tuple[RepeatProgress, ...]:
    """Every repeat in the Journey, in order, with its reply's place in the failed
    turns in a row and its Attempt's limit (#447 B).

    Each is a refused call whose turn recorded it under its repeat key.
    """
    events = list(events)
    by_id = {event["event_id"]: event for event in events}
    caps = {
        event["payload"]["attempt_id"]: event["payload"]["continuation_policy"][
            "failed_turns"
        ]
        for event in events
        if event["event_type"] == "AttemptStarted"
        and event["payload"].get("continuation_policy") is not None
    }
    repeats = []
    for event in events:
        payload = event["payload"]
        if (
            event["event_type"] != "AgentActionTurnDispositionRecorded"
            or not is_repeat_key(payload.get("correction_key") or "")
            or payload["attempt_id"] not in caps
        ):
            continue
        (validation_id,) = payload["validation_event_ids"]
        repeats.append(
            RepeatProgress(
                attempt_id=payload["attempt_id"],
                tool_name=by_id[validation_id]["payload"]["tool_name"],
                repeat=payload["correction_count"],
                cap=caps[payload["attempt_id"]],
                source_event_id=validation_id,
            )
        )
    return tuple(repeats)
