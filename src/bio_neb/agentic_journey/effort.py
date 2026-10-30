"""Exact effort stops and additive extensions of retained Agent assignments."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, StrictStr

from bio_neb.agentic_journey.continuation import (
    review_answer_parents,
    validate_attempt_continuation,
    validate_review_answer_parentage,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    AgentInvocationPolicy,
)
from bio_neb.agentic_journey.injection import injected_pattern
from bio_neb.agentic_journey.model_turn import messages_record, read_messages_record
from bio_neb.agentic_journey.role_declaration import AgentRole, role
from bio_neb.contracts import FrozenContract
from bio_neb.junction import evaluate_scoped_answer
from bio_neb.recording import read_evidence


class ExtendAgentEffortResponse(FrozenContract):
    action: Literal["extend"]
    additional_requests: StrictInt = Field(gt=0)


class StopAgentEffortResponse(FrozenContract):
    action: Literal["stop"]
    reason: StrictStr = Field(min_length=1, max_length=2000)


AgentEffortReviewResponse = Annotated[
    ExtendAgentEffortResponse | StopAgentEffortResponse, Field(discriminator="action")
]


class AgentEffortReviewRequested(FrozenContract):
    event_type: Literal["AgentEffortReviewRequested"] = "AgentEffortReviewRequested"
    attempt_id: StrictStr
    agent_role: AgentRole
    retry_frontier_event_id: StrictStr
    requests_used: StrictInt = Field(ge=0)
    request_limit: StrictInt = Field(gt=0)
    messages: list[dict[str, JsonValue]]


class AgentEffortReviewResponded(FrozenContract):
    event_type: Literal["AgentEffortReviewResponded"] = "AgentEffortReviewResponded"
    request_event_id: StrictStr
    response: AgentEffortReviewResponse
    authority: Literal["researcher_direction"] = "researcher_direction"
    authorizes_execution: Literal[False] = False


class AgentEffortReviewRequestView(AgentEffortReviewRequested):
    request_event_id: StrictStr
    pending: bool


class AgentEffortReviewReceipt(FrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: AgentEffortReviewResponse


class AgentEffortReviewProjection(FrozenContract):
    requests: tuple[AgentEffortReviewRequestView, ...] = ()
    receipts: tuple[AgentEffortReviewReceipt, ...] = ()
    pending_request: AgentEffortReviewRequestView | None = None


class AgentEffortReviewResponseRefused(ValueError):
    def __init__(self, reason, request_event_id):
        self.code = f"agent_effort_review_response_{reason}"
        self.request_event_id = request_event_id
        super().__init__(self.code)


class AgentEffortExhausted(Exception):
    def __init__(self, messages, requests_used):
        super().__init__("effort_exhausted")
        self.messages = messages_record(messages)
        self.requests_used = requests_used


def attempt_requests_used(events, attempt_id):
    """Completed logical requests, across every member and exact continuation."""
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    owned = {*prior_model_attempt_ids(events, attempt_id), attempt_id}
    return sum(
        e["event_type"] == "ModelResponseRecorded"
        and e["payload"].get("attempt_id") in owned
        for e in events
    )


def _started(events, attempt_id):
    starts = [
        e
        for e in events
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == attempt_id
    ]
    if len(starts) != 1:
        raise ValueError("effort requires one exact Attempt")
    return starts[0]


def _validate_answer(request, response):
    # The tagged response owns legality; no domain/source/execution authority is granted.
    if response.action not in {"extend", "stop"}:
        raise ValueError("unsupported effort response")


def extension_overrides(events, request, response):
    prior = _started(events, request.attempt_id)["payload"]
    values = dict(prior["invocation_overrides"])
    values["model_requests"] = request.request_limit + response.additional_requests
    return AgentInvocationOverrides.model_validate(values)


def _message_content(message):
    # Native run fills metadata and cost after the observed response is retained.
    # Those fields do not change the messages supplied to the resumed model.
    normalized = {
        key: value
        for key, value in message.items()
        if key not in {"timestamp", "run_id", "conversation_id", "cost"}
    }
    if isinstance(normalized.get("usage"), dict):
        normalized["usage"] = {
            key: value for key, value in normalized["usage"].items() if key != "cost"
        }
    return normalized


def _validate_effort_messages(events, request):
    """Resume the actual pending transcript, never substituted prompts or responses."""
    owned = [e for e in events if e["payload"].get("attempt_id") == request.attempt_id]
    if not request.messages:
        # Investigation checks effort before constructing the next member's prompt.
        # A completed member is the only permitted empty continuation transcript.
        outputs = [e for e in owned if e["event_type"] == "AgentOutputValidated"]
        responses = [e for e in owned if e["event_type"] == "ModelResponseRecorded"]
        if (
            request.agent_role != "investigation_planner"
            or not outputs
            or not responses
            or outputs[-1]["payload"]["status"] != "valid"
            or owned.index(outputs[-1]) < owned.index(responses[-1])
        ):
            raise ValueError("effort stop lost its pending transcript")
        return
    requests = [e for e in owned if e["event_type"] == "ModelRequestRecorded"]
    responses = [e for e in owned if e["event_type"] == "ModelResponseRecorded"]
    if (
        not requests
        or not responses
        or requests[-1]["payload"]["request_number"]
        != responses[-1]["payload"]["request_number"]
    ):
        raise ValueError("effort transcript lacks its completed logical request")
    prior = requests[-1]["payload"]["compatible_adapter_evidence"]["request"][
        "messages"
    ]
    from bio_neb.agentic_journey.transcript import kept_message

    # The reply as the loop kept it: a routed reply holds only its kept calls, and
    # a refused single call stays only where its correction answers it.
    response = kept_message(
        events, responses[-1], _answered(request.messages[-1].get("parts") or ())
    )
    expected = [*prior, response]
    if len(request.messages) != len(expected) + 1 or [
        _message_content(message) for message in request.messages[:-1]
    ] != [_message_content(message) for message in expected]:
        raise ValueError("effort transcript changed retained model messages")
    pending = request.messages[-1]
    # An injected part is lawful only for a pattern the Attempt's recorded policy
    # switches on, once per pattern in the request (ADR 0048).
    started = next(e for e in owned if e["event_type"] == "AttemptStarted")
    policy = AgentInvocationPolicy.model_validate(
        started["payload"]["effective_invocation_policy"]
    ).injection
    switched_on = frozenset() if policy is None else policy.switched_on()
    injected = [
        injected_pattern(part.get("content"))
        for part in pending.get("parts") or ()
        if part.get("part_kind") == "user-prompt"
    ]
    calls = {
        part["tool_call_id"]: part["tool_name"]
        for part in response["parts"]
        if part["part_kind"] == "tool-call"
    }
    if (
        pending.get("kind") != "request"
        or pending.get("instructions") != prior[-1].get("instructions")
        or not pending.get("parts")
        or not set(injected) <= switched_on
        or len(set(injected)) != len(injected)
        or any(
            part.get("part_kind") not in {"tool-return", "retry-prompt", "user-prompt"}
            or (
                part.get("tool_call_id") is not None
                and calls.get(part["tool_call_id"]) != part.get("tool_name")
            )
            for part in pending["parts"]
        )
    ):
        raise ValueError("effort transcript changed its pending Tool continuation")


def _answered(parts) -> frozenset[str]:
    """The calls a pending request answers with a result in their own slot."""
    return frozenset(
        part["tool_call_id"] for part in parts if part.get("part_kind") == "tool-return"
    )


def _as_sent(messages):
    """The transcript as the framework sends it: adjacent requests form one turn.

    Pydantic AI merges adjacent requests before sending and puts retry and
    Tool-return parts first.
    """
    sent = []
    for message in messages:
        last = sent[-1] if sent else None
        if (
            message.get("kind") == "request"
            and last is not None
            and last.get("kind") == "request"
            and (
                not last.get("instructions")
                or not message.get("instructions")
                or last.get("instructions") == message.get("instructions")
            )
        ):
            parts = [*last["parts"], *message["parts"]]
            parts.sort(
                key=lambda part: (
                    part.get("part_kind") not in {"tool-return", "retry-prompt"}
                )
            )
            sent[-1] = {
                **message,
                "parts": parts,
                "instructions": last.get("instructions") or message.get("instructions"),
            }
        else:
            sent.append(message)
    return sent


def project_effort_review(events, *, checked_once=None):
    """``checked_once`` is the reading owner's when ``events`` are its evidence."""
    from bio_neb.agentic_journey.agent_work_review import concurrent_stop_allowed
    from bio_neb.agentic_journey.attempt_journey import (
        attempt_members,
        attempts_since,
        shares_hold_scope,
    )

    timeline = list(events)
    by_id = {e["event_id"]: e for e in timeline}
    requests = []
    receipts = []
    consumed = set()
    if len(by_id) != len(timeline):
        raise ValueError("effort history repeats an event identity")
    for event in timeline:
        kind = event["event_type"]
        if kind == "AgentEffortReviewRequested":
            fact = AgentEffortReviewRequested.model_validate(event["payload"])
            prior = _started(timeline, fact.attempt_id)
            policy = prior["payload"]["effective_invocation_policy"]
            lifecycle = by_id.get(prior["parent_ids"][0])
            if lifecycle is None or lifecycle["parent_ids"] != [
                fact.retry_frontier_event_id
            ]:
                raise ValueError("effort review changed its exact assignment frontier")
            if (
                fact.request_limit != policy["model_requests"]
                or fact.agent_role != prior["payload"]["agent_role"]
                or fact.requests_used != fact.request_limit
                or fact.requests_used
                != attempt_requests_used(
                    timeline[: timeline.index(event)], fact.attempt_id
                )
                or fact.retry_frontier_event_id not in by_id
                or not concurrent_stop_allowed(
                    timeline[: timeline.index(event)],
                    [
                        r
                        for r in requests
                        if r.pending and r.agent_role == fact.agent_role
                    ],
                    fact.attempt_id,
                )
            ):
                raise ValueError("effort review changed its retained allowance")
            read_messages_record(fact.messages)
            _validate_effort_messages(timeline[: timeline.index(event)], fact)
            requests.append(
                AgentEffortReviewRequestView(
                    **fact.model_dump(),
                    request_event_id=event["event_id"],
                    pending=True,
                )
            )
        elif kind == "AgentEffortReviewResponded":
            fact = AgentEffortReviewResponded.model_validate(event["payload"])
            request, retained = evaluate_scoped_answer(
                requests,
                receipts,
                fact.request_event_id,
                fact.response,
                validate=_validate_answer,
                refusal=AgentEffortReviewResponseRefused,
            )
            validate_review_answer_parentage(timeline, event)
            if retained is not None:
                raise ValueError("effort answer changed its exact request")
            receipts.append(
                AgentEffortReviewReceipt(
                    request_event_id=fact.request_event_id,
                    response_event_id=event["event_id"],
                    response=fact.response,
                )
            )
            requests = [
                r.model_copy(update={"pending": False}) if r == request else r
                for r in requests
            ]
        elif kind == "ModelRequestRecorded":
            start = next(
                (
                    e
                    for e in timeline
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"]["attempt_id"] == event["payload"]["attempt_id"]
                ),
                None,
            )
            if start is None:
                continue
            answer_id = start["payload"].get("continuation_effort_review_event_id")
            earlier = [
                e
                for e in timeline[: timeline.index(event)]
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == event["payload"]["attempt_id"]
            ]
            if answer_id is not None and not earlier:
                receipt = next(r for r in receipts if r.response_event_id == answer_id)
                request = next(
                    r
                    for r in requests
                    if r.request_event_id == receipt.request_event_id
                )
                resumed = event["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ]
                if request.messages and [_message_content(m) for m in resumed] != [
                    _message_content(m) for m in _as_sent(request.messages)
                ]:
                    raise ValueError("effort continuation changed its pending messages")
        elif kind == "AttemptStarted":
            validate_attempt_continuation(timeline, event, checked_once=checked_once)
            answer_id = event["payload"].get("continuation_effort_review_event_id")
            if answer_id is None:
                # A stop holds fresh work in its own member scope.
                members = attempt_members(timeline[: timeline.index(event)])
                since = attempts_since(
                    timeline[: timeline.index(event)],
                    event["payload"].get("holds_from_event_id"),
                )
                in_scope = [
                    r
                    for r in requests
                    if r.agent_role == event["payload"]["agent_role"]
                    and (since is None or r.attempt_id in since)
                    and shares_hold_scope(
                        members.get(r.attempt_id), event["payload"].get("member_id")
                    )
                ]
                if any(r.pending for r in in_scope):
                    raise ValueError(
                        "effort exhaustion cannot silently start a fresh allowance"
                    )
                scoped_ids = {r.request_event_id for r in in_scope}
                settled = {
                    r.request_event_id
                    for r in in_scope
                    if event["payload"].get("member_id") is None
                    and _settles_member(r.agent_role, members.get(r.attempt_id))
                }
                blocked = [
                    r
                    for r in receipts
                    if r.request_event_id in scoped_ids
                    and r.request_event_id not in settled
                    and r.response.action == "stop"
                ]
                unconsumed = [
                    r
                    for r in receipts
                    if r.request_event_id in scoped_ids
                    and r.response.action == "extend"
                    and r.response_event_id not in consumed
                ]
                if blocked or unconsumed:
                    raise ValueError("effort continuation requires its exact extension")
                continue
            matches = [
                r
                for r in receipts
                if r.response_event_id == answer_id and r.response.action == "extend"
            ]
            if len(matches) != 1 or answer_id in consumed:
                raise ValueError("effort continuation lacks exact unused authority")
            receipt = matches[0]
            request = next(
                r for r in requests if r.request_event_id == receipt.request_event_id
            )
            previous = _started(timeline, request.attempt_id)["payload"]
            expected = dict(previous["effective_invocation_policy"])
            expected["model_requests"] = (
                request.request_limit + receipt.response.additional_requests
            )
            lifecycle = by_id.get(event["parent_ids"][0])
            if (
                event["payload"]["effective_invocation_policy"] != expected
                or event["payload"]["invocation_overrides"]
                != extension_overrides(
                    timeline, request, receipt.response
                ).explicit_settings()
                or any(
                    event["payload"].get(k) != previous.get(k)
                    for k in ("agent_role", "input_set_id", "instruction_identity")
                )
                or lifecycle is None
                or event["payload"].get("continuation_request_event_id") is not None
            ):
                raise ValueError(
                    "effort extension changed unrelated invocation authority"
                )
            consumed.add(answer_id)
    return AgentEffortReviewProjection(
        requests=tuple(requests),
        receipts=tuple(receipts),
        pending_request=next((r for r in requests if r.pending), None),
    )


def effort_status(
    events,
    *,
    agent_role=None,
    member_id=None,
    holds_from_event_id=None,
    checked_once=None,
):
    """The latest effort stop's state in a role, or in one member's scope (#105);
    with ``holds_from_event_id``, only stops since that request (#408)."""
    from bio_neb.agentic_journey.attempt_journey import (
        attempt_members,
        attempts_since,
        shares_hold_scope,
    )

    projection = project_effort_review(events, checked_once=checked_once)
    members = attempt_members(events)
    since = attempts_since(events, holds_from_event_id)
    requests = [
        r
        for r in projection.requests
        if (agent_role is None or r.agent_role == agent_role)
        and (since is None or r.attempt_id in since)
        and shares_hold_scope(members.get(r.attempt_id), member_id)
    ]
    if not requests:
        return None
    if member_id is None:
        # Each role and member scope has its own latest stop: a stop answer
        # ends the work, an answered extension re-enters, then a pending
        # Review holds.
        scopes = {}
        for r in requests:
            scopes[(r.agent_role, members.get(r.attempt_id))] = r
        # A stopped member scope settles that member alone where its role
        # declares so; the role's other work goes on.
        found = {
            status
            for (role, member), latest in scopes.items()
            if (status := _effort_scope_status(events, projection, latest))
            != "effort_stopped"
            or not _settles_member(role, member)
        }
        return next(
            (
                status
                for status in (
                    "effort_stopped",
                    "effort_review_direction_recorded",
                    "effort_review_pending",
                )
                if status in found
            ),
            None,
        )
    return _effort_scope_status(events, projection, requests[-1])


def _settles_member(agent_role, member_id) -> bool:
    return member_id is not None and role(agent_role).member_scope == "member"


def _effort_scope_status(events, projection, request):
    if request.pending:
        return "effort_review_pending"
    receipt = next(
        r for r in projection.receipts if r.request_event_id == request.request_event_id
    )
    if receipt.response.action == "stop":
        return "effort_stopped"
    if not any(
        e["event_type"] == "AttemptStarted"
        and e["payload"].get("continuation_effort_review_event_id")
        == receipt.response_event_id
        for e in events
    ):
        return "effort_review_direction_recorded"
    return None


def refuse_fresh_effort_allowance(
    events, *, agent_role, member_id=None, holds_from_event_id=None, checked_once=None
):
    """Hold a role's work with its effort review until the exact extension reopens it.

    A pending review, a stop answer or an unconsumed extension refuses a fresh
    Attempt of that role, in the stop's member scope, before its lifecycle
    writes anything.
    """
    if (
        effort_status(
            events,
            agent_role=agent_role,
            member_id=member_id,
            holds_from_event_id=holds_from_event_id,
            checked_once=checked_once,
        )
        is not None
    ):
        raise ValueError(f"{agent_role} effort stop requires its exact extension")


def validate_effort_terminal(events, terminal):
    request = next(
        (
            r
            for r in project_effort_review(events).requests
            if r.request_event_id == terminal["payload"]["source_event_id"]
        ),
        None,
    )
    if (
        request is None
        or request.attempt_id != terminal["payload"]["attempt_id"]
        or terminal["parent_ids"] != [request.request_event_id]
    ):
        raise ValueError("effort stop lacks its exact review")


def validate_effort_history(events, *, checked_once=None):
    projection = project_effort_review(events, checked_once=checked_once)
    terminals = [
        e
        for e in events
        if e["event_type"] == "AttemptStopped"
        and e["payload"]["code"] == "effort_exhausted"
    ]
    if len(terminals) != len(projection.requests):
        raise ValueError("effort requests and Attempt stops are not bijective")
    for terminal in terminals:
        validate_effort_terminal(events, terminal)
    return projection


def record_effort_stop(
    activity, exhausted, *, attempt_id, policy, retry_frontier_event_id
):
    from bio_neb.agentic_journey.attempt_journey import AttemptStopped

    request = activity.record(
        AgentEffortReviewRequested(
            attempt_id=attempt_id,
            agent_role=policy.agent_role,
            retry_frontier_event_id=retry_frontier_event_id,
            requests_used=exhausted.requests_used,
            request_limit=policy.model_requests,
            messages=exhausted.messages,
        )
    )
    stopped = activity.record(
        AttemptStopped(attempt_id=attempt_id, source_event_id=request["event_id"])
    )
    activity.succeeded(stopped)


def respond_effort_review(journey, request_event_id, response):
    events = read_evidence(journey)
    projection = project_effort_review(events)
    request, retained = evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=_validate_answer,
        refusal=AgentEffortReviewResponseRefused,
    )
    if retained is not None:
        return retained
    fact = AgentEffortReviewResponded(
        request_event_id=request_event_id, response=response
    )
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    event = journey.record(
        fact.event_type,
        payload,
        review_answer_parents(events, request_event_id),
    )
    return AgentEffortReviewReceipt(
        request_event_id=request_event_id,
        response_event_id=event["event_id"],
        response=response,
    )


def pending_effort_work(journey, response_event_id, *, agent_role):
    events = read_evidence(journey)
    projection = project_effort_review(events)
    receipts = [
        r
        for r in projection.receipts
        if r.response_event_id == response_event_id and r.response.action == "extend"
    ]
    consumed = any(
        e["event_type"] == "AttemptStarted"
        and e["payload"].get("continuation_effort_review_event_id") == response_event_id
        for e in events
    )
    if len(receipts) != 1 or consumed:
        raise ValueError("effort extension requires exact unused authority")
    receipt = receipts[0]
    request = next(
        r for r in projection.requests if r.request_event_id == receipt.request_event_id
    )
    if request.agent_role != agent_role:
        raise ValueError("effort extension changed Agent role")
    return (
        request,
        (read_messages_record(request.messages) if request.messages else None),
        extension_overrides(events, request, receipt.response),
    )
