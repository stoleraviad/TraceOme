"""Exact scoped review of Agent work its Attempt cannot continue.

One failure rule for every role (owner, #447 B): the ``failed_turns``-th reply in
a row in which nothing worked (every call refused or repeated, a text or
reasoning-only ending, a cut-off, a refused output) stops the Attempt, never the
Run. So does a model that ends without its result and a next request that would
exceed the model context window. The request retains the stopping evidence and
the Attempt's ordered fault history; one exact answer either starts a new Attempt
with a fresh count and no transcript or rejects the work.
"""

from typing import Annotated, Literal, get_args

from pydantic import Field, JsonValue, StrictInt, StrictStr

from bio_neb.agentic_journey.continuation import (
    review_answer_parents,
    validate_attempt_continuation,
    validate_review_answer_parentage,
)
from bio_neb.agentic_journey.continuation_policy import (
    COPIES_KEY,
    OUTPUT_ALONE_KEY,
    OUTPUT_KEY,
    REASONING_ONLY_KEY,
    TEXT_ENDING_KEY,
    TRUNCATION_KEY,
    UNOFFERED_KEY,
    ContinuationPolicy,
    model_output_faults,
)
from bio_neb.agentic_journey.model_transport import validate_model_truncation_source
from bio_neb.agentic_journey.role_declaration import AgentRole, role
from bio_neb.contracts import FrozenContract
from bio_neb.junction import evaluate_scoped_answer
from bio_neb.recording import read_evidence

AgentWorkStopCause = Literal[
    "failed_turns", "result_missing", "context_exhausted", "researcher_question"
]
STOP_CODES: frozenset[str] = frozenset(get_args(AgentWorkStopCause))
AgentWorkStatus = Literal[
    "agent_work_review_pending",
    "agent_work_review_direction_recorded",
    "agent_work_rejected",
]
_REFUSALS = frozenset({"ToolCallValidated", "AgentOutputValidated"})


def _is_fault(event) -> bool:
    """A refused action or output, a text ending, a reasoning-only reply, a
    cut-off completion or a routed reply's copies, unoffered calls or resends."""
    kind, payload = event["event_type"], event["payload"]
    return (
        (kind in _REFUSALS and payload["status"] == "invalid")
        or kind == "ModelCompletionTruncated"
        # A cut-off counts on its own truncation.
        or (
            kind == "AgentTurnRouted"
            and any(fault["key"] != "output_truncated" for fault in payload["faults"])
        )
        or (
            kind == "AgentActionTurnDispositionRecorded"
            and payload["disposition"] in {"text_refused", "reasoning_refused"}
        )
    )


class AgentWorkStop(Exception):
    """Agent work its allowances cannot continue; the Attempt stops for review."""

    cause: AgentWorkStopCause

    def __init__(self, reason: str, source_event_id: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.source_event_id = source_event_id


class AgentFailedTurns(AgentWorkStop):
    """The reply that reached its limit of failed turns in a row (#447 B).

    The next request is not sent. The source is that reply's last fault: its
    refused turn, truncation, routing or refused output.
    """

    cause = "failed_turns"


class AgentResearcherQuestion(AgentWorkStop):
    """The role asked its researcher question (``ask_researcher``, #447 Q8).

    The source is its ``ResearcherQuestionAsked``; the request shows what the
    model found so far, the conflict and the question.
    """

    cause = "researcher_question"

    def __init__(self, asked: dict):
        super().__init__(
            "the assistant asks the researcher a question", asked["event_id"]
        )
        self.asked = asked["payload"]


class AgentResultMissing(AgentWorkStop):
    """The model ended without a typed result, or the framework refused to go on."""

    cause = "result_missing"


class AgentContextExhausted(AgentWorkStop):
    """The next model request would exceed the context window; it was not sent.

    Retrying the same logical request would overflow again, so this is not the
    model availability review. The source is the refused ``ModelRequestFailed``.
    """

    cause = "context_exhausted"


class ContinueAgentWorkResponse(FrozenContract):
    action: Literal["continue"]
    instruction: StrictStr | None = Field(default=None, min_length=1, max_length=2000)
    # The next Attempt's number of failed turns in a row, granted by the
    # researcher; an answer naming none records no key.
    failed_turns: StrictInt | None = Field(
        default=None, ge=1, le=20, exclude_if=lambda value: value is None
    )


class DismissAgentWorkResponse(FrozenContract):
    """Settle only the stopped item (owner, #447: "the user need a way to
    apporved this and stop the item"); the rest go on."""

    action: Literal["dismiss"]
    reason: StrictStr = Field(min_length=1, max_length=2000)
    asserted_by: StrictStr = Field(min_length=1)


class RejectAgentWorkResponse(FrozenContract):
    action: Literal["reject"]
    reason: StrictStr = Field(min_length=1, max_length=2000)
    # An Investigation member's reject is its request's accounted rejection.
    asserted_by: StrictStr = Field(min_length=1)


AgentWorkReviewResponse = Annotated[
    ContinueAgentWorkResponse | RejectAgentWorkResponse | DismissAgentWorkResponse,
    Field(discriminator="action"),
]

_EFFECTS = {
    "continue": (
        "Start a new Attempt of this assignment from the current checked state with "
        "a fresh request budget and a fresh count of failed turns. An instruction is "
        "retained as a User assertion. failed_turns sets the new Attempt's number of "
        "failed turns in a row. The stopped Attempt is not resumed."
    ),
    "reject": (
        "Stop this assignment's Agent work. Completed work is retained and no "
        "further Attempt of this role starts."
    ),
}
# A rejected member stop of a member-scoped role settles that member alone
# and states its role's own effect (``RoleDeclaration``).
_RESPONSES = {
    "continue": ContinueAgentWorkResponse,
    "reject": RejectAgentWorkResponse,
    "dismiss": DismissAgentWorkResponse,
}


class AgentWorkReviewRequested(FrozenContract):
    event_type: Literal["AgentWorkReviewRequested"] = "AgentWorkReviewRequested"
    attempt_id: StrictStr
    agent_role: AgentRole
    cause: AgentWorkStopCause
    reason: StrictStr = Field(min_length=1, max_length=200)
    source_event_id: StrictStr
    # Every model-output fault of the Attempt, in order.
    fault_event_ids: tuple[StrictStr, ...]
    # A researcher question's own words; other causes record none.
    found_so_far: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    conflict: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    question: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class AgentWorkReviewResponded(FrozenContract):
    event_type: Literal["AgentWorkReviewResponded"] = "AgentWorkReviewResponded"
    request_event_id: StrictStr
    response: AgentWorkReviewResponse
    authority: Literal["researcher_direction"] = "researcher_direction"
    authorizes_execution: Literal[False] = False


class AgentWorkFaultView(FrozenContract):
    """One retained model-output fault; its own event owns it."""

    event_id: StrictStr
    event_type: Literal[
        "ToolCallValidated",
        "AgentOutputValidated",
        "AgentActionTurnDispositionRecorded",
        "ModelCompletionTruncated",
        "AgentTurnRouted",
    ]
    tool_name: StrictStr | None
    arguments: dict[str, JsonValue]
    errors: tuple[StrictStr, ...]
    # The correction its turn presented, or a refused output's repair; none for a
    # stopping truncation.
    feedback: StrictStr | None = None
    # Its correction key and its reply's place in the failed turns in a row.
    correction_key: StrictStr | None = None
    correction_count: StrictInt | None = None


class AgentWorkTruncationView(FrozenContract):
    """The cut-off completion; its ``ModelCompletionTruncated`` owns it."""

    event_id: StrictStr
    request_number: StrictInt
    requested_max_tokens: StrictInt | None
    completion_tokens: StrictInt | None


class AgentWorkStuckOn(FrozenContract):
    """What the stopping run of failed turns failed on most: a Tool, or a reply
    kind (a text ending, a reasoning-only reply, a cut-off). Read, never stored
    (owner, #447 Q2 iii: "a loop stuck tool or surface")."""

    surface: StrictStr
    failed_turns: StrictInt
    limit: StrictInt


class AgentWorkLegalAction(FrozenContract):
    action: Literal["continue", "reject", "dismiss"]
    effect: StrictStr
    response_schema: dict[str, JsonValue]


class AgentWorkReviewRequestView(AgentWorkReviewRequested):
    request_event_id: StrictStr
    pending: bool
    faults: tuple[AgentWorkFaultView, ...]
    # Only a stop on a cut-off reply names the cut-off completion.
    truncation: AgentWorkTruncationView | None = None
    # Only a failed-turns stop names what its run failed on most.
    stuck_on: AgentWorkStuckOn | None = None
    legal_actions: tuple[AgentWorkLegalAction, ...]


class AgentWorkReviewReceipt(FrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: AgentWorkReviewResponse


class AgentWorkReviewProjection(FrozenContract):
    requests: tuple[AgentWorkReviewRequestView, ...] = ()
    receipts: tuple[AgentWorkReviewReceipt, ...] = ()
    continued_response_event_ids: tuple[StrictStr, ...] = ()
    pending_request: AgentWorkReviewRequestView | None = None


class AgentWorkReviewResponseRefused(ValueError):
    def __init__(self, reason, request_event_id):
        self.code = f"agent_work_review_response_{reason}"
        self.request_event_id = request_event_id
        super().__init__(self.code)


def _settles_member(agent_role, member_id) -> bool:
    """Whether rejecting this scope's stop settles one member, as its role declares."""
    return member_id is not None and role(agent_role).member_scope == "member"


def _legal_actions(
    *, member_scoped: bool, agent_role: str, item: bool = False
) -> tuple[AgentWorkLegalAction, ...]:
    """Continue and reject always; dismiss where the stop names an item its role
    can settle alone (#447)."""
    declared = role(agent_role)
    effects = {
        **_EFFECTS,
        "reject": declared.member_reject_effect
        if member_scoped
        else _EFFECTS["reject"],
        "dismiss": declared.dismiss_effect,
    }
    actions = ("continue", "reject") + (
        ("dismiss",) if item and declared.dismiss_effect is not None else ()
    )
    return tuple(
        AgentWorkLegalAction(
            action=action,
            effect=effects[action],
            response_schema=_RESPONSES[action].model_json_schema(),
        )
        for action in actions
    )


def _validate_answer(request, response):
    # Both actions are always legal; neither grants domain or execution authority.
    if response.action not in {item.action for item in request.legal_actions}:
        raise AgentWorkReviewResponseRefused("illegal", request.request_event_id)


def _fault_event_ids(timeline, attempt_id) -> tuple[str, ...]:
    """The Attempt's ordered faults, across the provider continuations it resumes."""
    from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

    owned = {*prior_model_attempt_ids(timeline, attempt_id), attempt_id}
    return tuple(
        e["event_id"]
        for e in timeline
        if e["payload"].get("attempt_id") in owned and _is_fault(e)
    )


def _attempt_events(timeline, attempt_id):
    return [e for e in timeline if e["payload"].get("attempt_id") == attempt_id]


def _fault_views(prefix, fact) -> tuple[AgentWorkFaultView, ...]:
    by_id = {e["event_id"]: e for e in prefix}
    corrections = {
        fault.validation_event_id or fault.event_id: fault
        for fault in model_output_faults(prefix, attempt_id=fact.attempt_id)
    }
    # A refused action shows the correction its turn presented, a multi-action
    # selection included; a text ending or truncation its own.
    feedback = {
        validation_id: e["payload"].get("feedback")
        for e in prefix
        if e["event_type"] == "AgentActionTurnDispositionRecorded"
        for validation_id in e["payload"]["validation_event_ids"]
    }
    views = []
    for event_id in fact.fault_event_ids:
        event = by_id[event_id]
        payload = event["payload"]
        kind = event["event_type"]
        correction = corrections.get(event_id)
        views.append(
            AgentWorkFaultView(
                event_id=event_id,
                event_type=kind,
                tool_name=payload["tool_name"] if kind == "ToolCallValidated" else None,
                arguments=payload["arguments"]
                if kind == "ToolCallValidated"
                else payload["output"]
                if kind == "AgentOutputValidated"
                else {},
                errors=payload.get("errors", ()),
                feedback=feedback.get(event_id)
                if kind == "ToolCallValidated"
                else payload.get("feedback"),
                correction_key=None if correction is None else correction.key,
                correction_count=None if correction is None else correction.count,
            )
        )
    return tuple(views)


# A fault that names no Tool is named by its reply kind.
_REPLY_KINDS = {
    TEXT_ENDING_KEY: "a text ending",
    REASONING_ONLY_KEY: "a reasoning-only reply",
    TRUNCATION_KEY: "a cut-off reply",
    OUTPUT_KEY: "a refused output",
    COPIES_KEY: "copied calls",
    UNOFFERED_KEY: "calls to Tools not offered",
    OUTPUT_ALONE_KEY: "an output beside calls",
}


def _stuck_on(prefix, fact) -> AgentWorkStuckOn:
    """The Tool or reply kind most of the stopping run's failed turns failed on."""
    by_id = {e["event_id"]: e for e in prefix}
    (start, *_) = [
        e
        for e in prefix
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == fact.attempt_id
    ]
    policy = ContinuationPolicy.model_validate(start["payload"]["continuation_policy"])
    faults = model_output_faults(prefix, attempt_id=fact.attempt_id)
    run = []
    for fault in reversed(faults):
        if not fault.failed or (run and fault.count > run[-1].count):
            break
        run.append(fault)
    turns: dict[str, set[int]] = {}
    for fault in run:
        if fault.validation_event_id is not None:
            surface = f"`{by_id[fault.validation_event_id]['payload']['tool_name']}`"
        else:
            surface = _REPLY_KINDS.get(fault.key, fault.key)
        turns.setdefault(surface, set()).add(fault.count)
    surface, places = max(turns.items(), key=lambda item: len(item[1]))
    return AgentWorkStuckOn(
        surface=surface,
        failed_turns=len(places),
        limit=policy.limit(run[0].stage),
    )


def _truncation_view(event) -> AgentWorkTruncationView:
    payload = event["payload"]
    return AgentWorkTruncationView(
        event_id=event["event_id"],
        request_number=payload["request_number"],
        requested_max_tokens=payload["requested_max_tokens"],
        completion_tokens=payload["completion_tokens"],
    )


def _validate_request(prefix, fact):
    """The request names the exact stopping evidence of its own open Attempt."""
    from bio_neb.agentic_journey.attempt_journey import MODEL_REQUEST_TERMINALS

    starts = [
        e
        for e in prefix
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == fact.attempt_id
    ]
    if len(starts) != 1 or starts[0]["payload"]["agent_role"] != fact.agent_role:
        raise ValueError("Agent work review lacks its exact Attempt")
    owned = _attempt_events(prefix, fact.attempt_id)
    if any(
        e["event_type"] in {"AttemptSucceeded", "AttemptFailed", "AttemptStopped"}
        for e in owned
    ):
        raise ValueError("Agent work review follows a closed Attempt")
    if fact.fault_event_ids != _fault_event_ids(prefix, fact.attempt_id):
        raise ValueError("Agent work review changed its fault evidence")
    if fact.cause == "failed_turns":
        _validate_failed_turns(prefix, owned, fact)
    elif fact.cause == "researcher_question":
        asked = [e for e in owned if e["event_type"] == "ResearcherQuestionAsked"]
        if (
            not asked
            or asked[-1]["event_id"] != fact.source_event_id
            or (fact.found_so_far, fact.conflict, fact.question)
            != tuple(
                asked[-1]["payload"][name]
                for name in ("found_so_far", "conflict", "question")
            )
        ):
            raise ValueError("researcher-question stop lacks its exact question")
    elif fact.cause == "context_exhausted":
        terminals = [e for e in owned if e["event_type"] in MODEL_REQUEST_TERMINALS]
        if (
            not terminals
            or terminals[-1]["event_id"] != fact.source_event_id
            or terminals[-1]["event_type"] != "ModelRequestFailed"
            or terminals[-1]["payload"].get("failure_reason")
            != "context_window_exhausted"
        ):
            raise ValueError("context-exhausted stop lacks its exact refused request")
    else:
        responses = [e for e in owned if e["event_type"] == "ModelResponseRecorded"]
        if not responses or responses[-1]["event_id"] != fact.source_event_id:
            raise ValueError("missing-result stop lacks its exact model response")


def _validate_failed_turns(prefix, owned, fact):
    """The stop's source is the last fault of the Attempt's last reply, a failed
    turn at its limit; no request followed it."""
    (start, *_) = [e for e in owned if e["event_type"] == "AttemptStarted"]
    policy = ContinuationPolicy.model_validate(start["payload"]["continuation_policy"])
    faults = model_output_faults(prefix, attempt_id=fact.attempt_id)
    requests = [e for e in owned if e["event_type"] == "ModelRequestRecorded"]
    by_id = {e["event_id"]: index for index, e in enumerate(prefix)}
    if (
        not faults
        or faults[-1].event_id != fact.source_event_id
        or faults[-1].corrected(policy)
        or not requests
        or by_id[requests[-1]["event_id"]] > by_id[fact.source_event_id]
    ):
        raise ValueError("failed-turns stop lacks its exact stopping reply")
    if faults[-1].key == TRUNCATION_KEY:
        validate_model_truncation_source(prefix, prefix[by_id[fact.source_event_id]])


def concurrent_stop_allowed(timeline, prior_pending, attempt_id) -> bool:
    """Whether a role's earlier pending Review lawfully coexists with this stop.

    Siblings the coordinator ran together (#105) each stop on their own Review:
    the earlier Review is another Attempt's, opened after this Attempt started,
    or it holds another member's scope. A Review in this Attempt's scope opened
    before it started would have held it.
    """
    from bio_neb.agentic_journey.attempt_journey import (
        attempt_members,
        shares_hold_scope,
    )

    start = next(
        i
        for i, event in enumerate(timeline)
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == attempt_id
    )
    positions = {event["event_id"]: i for i, event in enumerate(timeline)}
    members = attempt_members(timeline)
    return all(
        request.attempt_id != attempt_id
        and (
            positions[request.request_event_id] > start
            or not shares_hold_scope(
                members.get(request.attempt_id), members.get(attempt_id)
            )
        )
        for request in prior_pending
    )


def project_agent_work_review(
    events, *, checked_once=None
) -> AgentWorkReviewProjection:
    """``checked_once`` is the reading owner's when ``events`` are its evidence."""
    from bio_neb.agentic_journey.attempt_journey import attempt_members, attempts_since

    timeline = list(events)
    members = attempt_members(timeline)
    by_id = {e["event_id"]: e for e in timeline}
    if len(by_id) != len(timeline):
        raise ValueError("Agent work history repeats an event identity")
    requests: list[AgentWorkReviewRequestView] = []
    receipts: list[AgentWorkReviewReceipt] = []
    continued: list[str] = []
    for index, event in enumerate(timeline):
        kind = event["event_type"]
        if kind == "AgentWorkReviewRequested":
            fact = AgentWorkReviewRequested.model_validate(event["payload"])
            prefix = timeline[:index]
            _validate_request(prefix, fact)
            pending = [
                r for r in requests if r.pending and r.agent_role == fact.agent_role
            ]
            if not concurrent_stop_allowed(prefix, pending, fact.attempt_id):
                raise ValueError("Agent work review opened while another is pending")
            requests.append(
                AgentWorkReviewRequestView(
                    **fact.model_dump(),
                    request_event_id=event["event_id"],
                    pending=True,
                    faults=_fault_views(prefix, fact),
                    truncation=_truncation_view(by_id[fact.source_event_id])
                    if by_id[fact.source_event_id]["event_type"]
                    == "ModelCompletionTruncated"
                    else None,
                    stuck_on=_stuck_on(prefix, fact)
                    if fact.cause == "failed_turns"
                    else None,
                    legal_actions=_legal_actions(
                        member_scoped=_settles_member(
                            fact.agent_role, members.get(fact.attempt_id)
                        ),
                        agent_role=fact.agent_role,
                        item=members.get(fact.attempt_id) is not None,
                    ),
                )
            )
        elif kind == "AgentWorkReviewResponded":
            fact = AgentWorkReviewResponded.model_validate(event["payload"])
            request, retained = evaluate_scoped_answer(
                requests,
                receipts,
                fact.request_event_id,
                fact.response,
                validate=_validate_answer,
                refusal=AgentWorkReviewResponseRefused,
            )
            validate_review_answer_parentage(timeline, event)
            if retained is not None:
                raise ValueError("Agent work answer changed its exact request")
            receipts.append(
                AgentWorkReviewReceipt(
                    request_event_id=fact.request_event_id,
                    response_event_id=event["event_id"],
                    response=fact.response,
                )
            )
            requests = [
                r.model_copy(update={"pending": False}) if r == request else r
                for r in requests
            ]
        elif kind == "AttemptStarted":
            validate_attempt_continuation(timeline, event, checked_once=checked_once)
            role = event["payload"]["agent_role"]
            answer_id = event["payload"].get("continuation_work_review_event_id")
            if answer_id is None:
                since = attempts_since(
                    timeline[:index], event["payload"].get("holds_from_event_id")
                )
                if (
                    _status(
                        [r for r in requests if since is None or r.attempt_id in since],
                        receipts,
                        continued,
                        role,
                        event["payload"].get("member_id"),
                        attempt_members(timeline[:index]),
                    )
                    is not None
                ):
                    raise ValueError(
                        f"{role} stopped Agent work requires its exact continue answer"
                    )
                continue
            receipt = next(
                (r for r in receipts if r.response_event_id == answer_id), None
            )
            request = next(
                (
                    r
                    for r in requests
                    if receipt is not None
                    and r.request_event_id == receipt.request_event_id
                ),
                None,
            )
            # The new Attempt's lifecycle start rests on the answer it consumes.
            lifecycle = by_id.get(event["parent_ids"][0])
            if (
                request is None
                or lifecycle is None
                or receipt.response.action != "continue"
                or answer_id in continued
                or request.agent_role != role
                or event["payload"].get("continuation_request_event_id") is not None
                or event["payload"].get("continuation_effort_review_event_id")
                is not None
                or not _grant_is_consumed(receipt.response, event["payload"])
            ):
                raise ValueError("Agent work continuation lacks exact unused authority")
            continued.append(answer_id)
    return AgentWorkReviewProjection(
        requests=tuple(requests),
        receipts=tuple(receipts),
        continued_response_event_ids=tuple(continued),
        pending_request=next((r for r in requests if r.pending), None),
    )


def _grant_is_consumed(response: ContinueAgentWorkResponse, started) -> bool:
    """Whether the Attempt a continue started records exactly the number it
    granted: its override and policy carry a grant, and nothing else does
    (Spec r9 F8)."""
    granted = response.failed_turns
    overrides = started["invocation_overrides"]
    if granted is None:
        return "failed_turns" not in overrides
    return overrides == {"failed_turns": granted} and started[
        "continuation_policy"
    ] == {"failed_turns": granted}


def _status(
    requests, receipts, continued, agent_role, member_id=None, members=None
) -> AgentWorkStatus | None:
    from bio_neb.agentic_journey.attempt_journey import shares_hold_scope

    scoped = [
        r
        for r in requests
        if (agent_role is None or r.agent_role == agent_role)
        and shares_hold_scope((members or {}).get(r.attempt_id), member_id)
    ]
    if not scoped:
        return None
    if member_id is None and members:
        # Each role and member scope has its own latest stop: a reject ends the
        # work, an answered continue re-enters, then a pending Review. A member's
        # reject settles that member alone where its role declares so; the role's
        # other work goes on.
        latest = {}
        for r in scoped:
            latest[(r.agent_role, members.get(r.attempt_id))] = r
        found = {
            status
            for (role, member), r in latest.items()
            if (status := _work_scope_status(r, receipts, continued))
            != "agent_work_rejected"
            or not _settles_member(role, member)
        }
        return next(
            (
                status
                for status in (
                    "agent_work_rejected",
                    "agent_work_review_direction_recorded",
                    "agent_work_review_pending",
                )
                if status in found
            ),
            None,
        )
    return _work_scope_status(scoped[-1], receipts, continued)


def _work_scope_status(request, receipts, continued) -> AgentWorkStatus | None:
    if request.pending:
        return "agent_work_review_pending"
    receipt = next(
        r for r in receipts if r.request_event_id == request.request_event_id
    )
    if receipt.response.action == "reject" or (
        receipt.response.action == "dismiss"
        and role(request.agent_role).member_scope == "member"
    ):
        # A member-scoped role's dismissed member is settled as rejected.
        return "agent_work_rejected"
    if receipt.response.action == "dismiss":
        # A role-scoped item is settled alone; the role goes on.
        return None
    if receipt.response_event_id not in continued:
        return "agent_work_review_direction_recorded"
    return None


def agent_work_status(
    events,
    *,
    agent_role=None,
    member_id=None,
    holds_from_event_id=None,
    checked_once=None,
) -> AgentWorkStatus | None:
    """The latest Agent work stop's state in a role, or in one member's scope;
    with ``holds_from_event_id``, only stops since that request (#408)."""
    from bio_neb.agentic_journey.attempt_journey import (
        attempt_members,
        attempts_since,
    )

    events = list(events)
    projection = project_agent_work_review(events, checked_once=checked_once)
    since = attempts_since(events, holds_from_event_id)
    return _status(
        [r for r in projection.requests if since is None or r.attempt_id in since],
        projection.receipts,
        projection.continued_response_event_ids,
        agent_role,
        member_id,
        attempt_members(events),
    )


def refuse_fresh_agent_work(
    events, *, agent_role, member_id=None, holds_from_event_id=None, checked_once=None
):
    """Hold a role's work with its stop until the exact continue answer reopens it.

    A pending review, a reject answer or an unconsumed continue answer refuses a
    fresh Attempt of that role, in the stop's member scope, before its lifecycle
    writes anything.
    """
    if (
        agent_work_status(
            events,
            agent_role=agent_role,
            member_id=member_id,
            holds_from_event_id=holds_from_event_id,
            checked_once=checked_once,
        )
        is not None
    ):
        raise ValueError(
            f"{agent_role} stopped Agent work requires its exact continue answer"
        )


def validate_agent_work_terminal(events, terminal):
    request = next(
        (
            r
            for r in project_agent_work_review(events).requests
            if r.request_event_id == terminal["payload"]["source_event_id"]
        ),
        None,
    )
    if (
        request is None
        or request.attempt_id != terminal["payload"]["attempt_id"]
        or request.cause != terminal["payload"]["code"]
        or terminal["parent_ids"] != [request.request_event_id]
    ):
        raise ValueError("Agent work stop lacks its exact review")


def validate_agent_work_history(events, *, checked_once=None):
    projection = project_agent_work_review(events, checked_once=checked_once)
    terminals = [
        e
        for e in events
        if e["event_type"] == "AttemptStopped" and e["payload"]["code"] in STOP_CODES
    ]
    if len(terminals) != len(projection.requests):
        raise ValueError("Agent work requests and Attempt stops are not bijective")
    for terminal in terminals:
        validate_agent_work_terminal(events, terminal)
    return projection


def record_agent_work_stop(
    activity, stop: AgentWorkStop, *, journey, attempt_id, agent_role
):
    """Retain the exact refused evidence, then stop the Attempt on that request."""
    from bio_neb.agentic_journey.attempt_journey import AttemptStopped

    timeline = read_evidence(journey)
    owned = _attempt_events(timeline, attempt_id)
    source = stop.source_event_id or next(
        (
            e["event_id"]
            for e in reversed(owned)
            if e["event_type"] == "ModelResponseRecorded"
        ),
        None,
    )
    if source is None:
        raise ValueError("Agent work stop lacks its exact evidence")
    asked = getattr(stop, "asked", None) or {}
    request = activity.record(
        AgentWorkReviewRequested(
            attempt_id=attempt_id,
            agent_role=agent_role,
            cause=stop.cause,
            reason=stop.reason,
            source_event_id=source,
            fault_event_ids=_fault_event_ids(timeline, attempt_id),
            found_so_far=asked.get("found_so_far"),
            conflict=asked.get("conflict"),
            question=asked.get("question"),
        )
    )
    stopped = activity.record(
        AttemptStopped(
            attempt_id=attempt_id, code=stop.cause, source_event_id=request["event_id"]
        )
    )
    activity.succeeded(stopped)


def respond_agent_work_review(journey, request_event_id, response):
    events = read_evidence(journey)
    projection = project_agent_work_review(events)
    _, retained = evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=_validate_answer,
        refusal=AgentWorkReviewResponseRefused,
    )
    if retained is not None:
        return retained
    fact = AgentWorkReviewResponded(
        request_event_id=request_event_id, response=response
    )
    payload = fact.model_dump(mode="json")
    payload.pop("event_type")
    event = journey.record(
        fact.event_type,
        payload,
        review_answer_parents(events, request_event_id),
    )
    return AgentWorkReviewReceipt(
        request_event_id=request_event_id,
        response_event_id=event["event_id"],
        response=response,
    )


def pending_agent_work(journey, response_event_id, *, agent_role):
    """The stopped request an exact unused continue answer reopens."""
    projection = project_agent_work_review(read_evidence(journey))
    receipts = [
        r
        for r in projection.receipts
        if r.response_event_id == response_event_id and r.response.action == "continue"
    ]
    if (
        len(receipts) != 1
        or response_event_id in projection.continued_response_event_ids
    ):
        raise ValueError("Agent work continuation requires exact unused authority")
    request = next(
        r
        for r in projection.requests
        if r.request_event_id == receipts[0].request_event_id
    )
    if request.agent_role != agent_role:
        raise ValueError("Agent work continuation changed Agent role")
    return request, receipts[0].response.instruction


def granted_failed_turns(journey, response_event_id) -> int | None:
    """The number of failed turns a continue answer grants its next Attempt, if any."""
    projection = project_agent_work_review(read_evidence(journey))
    (receipt,) = [
        r for r in projection.receipts if r.response_event_id == response_event_id
    ]
    if not isinstance(receipt.response, ContinueAgentWorkResponse):
        raise ValueError("only a continue answer grants failed turns")
    return receipt.response.failed_turns
