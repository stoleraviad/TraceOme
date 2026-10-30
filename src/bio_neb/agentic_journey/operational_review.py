"""Agent-scoped instances of the shared failure junction.

Policy and classification belong to failure; this owner binds a stopped operation
and the native pending transcript to one exact researcher direction.
"""

import asyncio
from typing import Any, Literal, Mapping

from pydantic import Field, JsonValue, StrictStr, model_validator

from bio_neb.agentic_journey.continuation import (
    review_answer_parents,
    validate_attempt_continuation,
    validate_review_answer_parentage,
)
from bio_neb.agentic_journey.model_turn import read_messages_record
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.contracts import FrozenContract
from bio_neb.failure import (
    Ask,
    FailureClass,
    ResolvedFailurePolicy,
    ReviewAnswer,
    operation_retry_decision,
)
from bio_neb.journey import FORK_ORIGIN_EVENT_TYPE, canonical_json_digest
from bio_neb.junction import evaluate_scoped_answer
from bio_neb.recording import read_evidence


class OperationalFailureRecorded(FrozenContract):
    event_type: Literal["OperationalFailureRecorded"] = "OperationalFailureRecorded"
    attempt_id: StrictStr
    started_event_id: StrictStr
    source_event_id: StrictStr | None
    failure_class: FailureClass
    original_failure_class: FailureClass
    family: Literal["source", "provider"]
    policy: ResolvedFailurePolicy
    contact_attempt: int = Field(ge=1)
    detail: StrictStr


class AgentFailureReviewRequested(FrozenContract):
    event_type: Literal["AgentFailureReviewRequested"] = "AgentFailureReviewRequested"
    attempt_id: StrictStr
    agent_role: StrictStr
    member_id: StrictStr
    retry_frontier_event_id: StrictStr
    request: Ask
    response_event_id: StrictStr
    validation_event_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    arguments: dict[str, JsonValue]
    bound_argument_names: tuple[str, ...] = ()
    messages: list[dict[str, JsonValue]]
    # The failure module's key for the cause (one keyed rule for every fault):
    # a later item stopped for the same cause while this Review is open joins
    # it, and one answer releases both.
    fault_key: StrictStr = Field(min_length=1)
    joins_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The item joined the open Review before its call contacted the source: its
    # Ask is the open Review's, and its call never started.
    joined_before_contact: bool = Field(
        default=False, exclude_if=lambda value: not value
    )


def operational_fault_key(fault: "OperationalFailureRecorded") -> str:
    """One operational stop's cause: its family, surface and failure class."""
    return f"{fault.family} {fault.policy.surface.name}: {fault.failure_class}"


def open_operational_review(
    events, fault_key: str | None = None, *, tool_name: str | None = None
) -> str | None:
    """The open operational Review a new item joins (G4.1): one with this
    cause's key, or, before contact, an open source Review on the one surface
    this source Tool contacts. A Tool that may contact several surfaces is not
    held before contact: it joins only by its own failure's key."""
    from bio_neb.agentic_journey.keyed_review import open_keyed_review
    from bio_neb.agentic_journey.source_contacts import contact_surfaces

    surfaces = None if tool_name is None else contact_surfaces(tool_name)
    return open_keyed_review(
        events,
        requested="AgentFailureReviewRequested",
        responded="AgentFailureReviewResponded",
        matches=lambda payload: (
            fault_key is None or payload.get("fault_key") == fault_key
        )
        and (
            surfaces is None
            or (
                len(surfaces) == 1
                and payload.get("fault_key", "").startswith(
                    f"source {next(iter(surfaces))}: "
                )
            )
        ),
    )


class AgentFailureReviewRequestView(AgentFailureReviewRequested):
    request_event_id: StrictStr
    pending: bool = True


class AgentFailureReviewResponded(FrozenContract):
    event_type: Literal["AgentFailureReviewResponded"] = "AgentFailureReviewResponded"
    request_event_id: StrictStr
    response: ReviewAnswer
    authority: Literal["researcher_direction"] = "researcher_direction"
    scientifically_verified: Literal[False] = False
    authorizes_execution: Literal[False] = False
    # The researcher's retry preset that gave this answer, when one did.
    preset_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


# A preset retries only a classified operational fault; an invariant holds the Run.
PresetFailureClass = Literal[
    "transient", "exhausted", "rejected", "worker_failed", "semantic"
]
PRESET_RETRY_LIMIT = 3


class AgentFailureRetryPreset(FrozenContract):
    """A researcher's `retry`, set ahead for one Agent role's failure Reviews.

    Each Review of this Run whose failure class it names, at one of its Tools
    (any Tool when it names none), is answered `retry` as it opens, up to
    ``count`` times. A newer preset for the role replaces it; a count of 0
    revokes it.
    """

    event_type: Literal["AgentFailureRetryPreset"] = "AgentFailureRetryPreset"
    agent_role: AgentRole
    failure_classes: tuple[PresetFailureClass, ...] = ()
    tool_names: tuple[StrictStr, ...] = ()
    count: int = Field(ge=0, le=PRESET_RETRY_LIMIT)
    by: StrictStr = Field(min_length=1)
    reason: StrictStr | None = Field(default=None, min_length=1, max_length=2000)
    authority: Literal["researcher_direction"] = "researcher_direction"

    @model_validator(mode="after")
    def bounded(self):
        if (self.count == 0) != (not self.failure_classes) or (
            self.count == 0 and self.tool_names
        ):
            raise ValueError("a preset names the classes it retries; a revocation none")
        if len(set(self.failure_classes)) != len(self.failure_classes) or len(
            set(self.tool_names)
        ) != len(self.tool_names):
            raise ValueError("a preset names each class and Tool once")
        return self


class AgentFailureRetryPresetView(AgentFailureRetryPreset):
    preset_event_id: StrictStr
    remaining: int = Field(ge=0, le=PRESET_RETRY_LIMIT)


class AgentFailureReviewReceipt(FrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: ReviewAnswer


class AgentFailureReviewRefused(ValueError):
    def __init__(self, reason, request_event_id):
        self.code = "operational_review_response_" + reason
        self.request_event_id = request_event_id
        super().__init__(self.code)


class AgentFailureReviewProjection(FrozenContract):
    requests: tuple[AgentFailureReviewRequestView, ...] = ()
    receipts: tuple[AgentFailureReviewReceipt, ...] = ()
    pending_request: AgentFailureReviewRequestView | None = None
    # Each Agent role's current retry preset; a revoked one remains with none left.
    retry_presets: tuple[AgentFailureRetryPresetView, ...] = ()


class OperationalStop(Exception):
    def __init__(self, failure, admitted, *, joins: Mapping[str, Any] | None = None):
        super().__init__(failure["payload"]["detail"])
        self.failure = failure
        self.admitted = admitted
        self.messages = []
        self.member_id = None
        # The open Review this item joins before its call contacts the source.
        self.joins = joins
        # Its Tool failed after an operator interrupt; see the record below.
        self.interrupted = False

    def retain_invocation(self, messages, member_id):
        """Bind the invocation's recorded pending transcript (JSON messages)."""
        self.messages = messages
        self.member_id = member_id


def record_operational_stop(context, stop, *, agent_role, close_attempt=True):
    import uuid

    from bio_neb.agentic_journey.attempt_journey import AttemptStopped

    call = stop.admitted.call
    fault_key = (
        operational_fault_key(
            OperationalFailureRecorded.model_validate(stop.failure["payload"])
        )
        if stop.joins is None
        else stop.joins["payload"]["fault_key"]
    )
    requested = context.activity.record(
        AgentFailureReviewRequested(
            attempt_id=context.attempt_id,
            agent_role=agent_role,
            member_id=stop.member_id or context.input_set_id,
            retry_frontier_event_id=context.causal_head,
            request=Ask(
                stage=agent_role,
                failure_event_id=stop.failure["event_id"],
                request_id=str(uuid.uuid4()),
                reason=str(stop),
                legal_actions=("retry", "amend", "stop"),
                amendable=("instruction",),
            ),
            response_event_id=stop.admitted.response_event_id,
            validation_event_id=stop.admitted.validation_event_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
            bound_argument_names=next(
                e["payload"]["bound_argument_names"]
                for e in read_evidence(context.journey)
                if e["event_id"] == stop.admitted.validation_event_id
            ),
            messages=stop.messages,
            fault_key=fault_key,
            joins_event_id=open_operational_review(
                read_evidence(context.journey), fault_key
            )
            if stop.joins is None
            else stop.joins["event_id"],
            joined_before_contact=stop.joins is not None,
        )
    )
    if close_attempt:
        terminal = context.activity.record(
            AttemptStopped(
                attempt_id=context.attempt_id,
                code="operational_failure",
                source_event_id=requested["event_id"],
            )
        )
        context.activity.succeeded(terminal)
    if stop.interrupted:
        # With its Review retained, the interrupt the failing Tool deferred ends
        # the work here, before any further decision of the assignment.
        raise asyncio.CancelledError
    return requested


def _validate_fault(events, fault, started):
    from bio_neb.agentic_journey.source_contacts import contact_surfaces, source_tools

    terminal = [
        e
        for e in events
        if e["event_type"]
        in {"ObservedToolExecutionSucceeded", "ObservedToolExecutionFailed"}
        and e["payload"]["started_event_id"] == fault.started_event_id
    ]
    expected_family = (
        "source" if started["payload"]["tool_name"] in source_tools() else "provider"
    )
    if (
        len(terminal) != 1
        or fault.family != expected_family
        or fault.policy.surface.scope != "tool"
        or fault.policy.surface.name
        not in contact_surfaces(started["payload"]["tool_name"])
        or fault.original_failure_class == "invariant"
        or fault.policy.effective.route(fault.failure_class) != "ask"
        or operation_retry_decision(
            fault.original_failure_class, fault.contact_attempt, fault.policy
        )
        != (fault.failure_class, None)
        # Only a source contact's later attempt may end otherwise: a breached
        # limit, or an answer after a timeout. Its source window is checked below.
        or (
            fault.original_failure_class != "transient"
            and fault.contact_attempt != 1
            and expected_family != "source"
        )
    ):
        raise ValueError("operational failure changed its class, family or policy")
    result_field = (
        "stage_result_event_id"
        if terminal[0]["event_type"] == "ObservedToolExecutionSucceeded"
        else "stage_failure_event_id"
    )
    if terminal[0]["payload"].get(result_field) != fault.source_event_id:
        raise ValueError("operational failure changed its exact source result")
    source = next((e for e in events if e["event_id"] == fault.source_event_id), None)
    result = source["payload"].get("result", {}) if source else {}
    classified = result.get("failure") or (
        result if result.get("outcome") == "operational_failure" else None
    )
    if classified is not None and (
        classified["failure_class"] != fault.failure_class
        or (classified.get("original_failure_class") or classified["failure_class"])
        != fault.original_failure_class
        or classified.get("contact_attempt", 1) != fault.contact_attempt
    ):
        raise ValueError("operational failure changed its typed classification")
    scope_ids = {
        e["event_id"]
        for e in events
        if e["event_type"] == "SourceContactScopeStarted"
        and e["payload"]["attempt_id"] == fault.attempt_id
        and e["payload"]["tool_call_id"] == started["payload"]["tool_call_id"]
    }
    failures = [
        e["payload"]
        for e in events
        if e["event_type"] == "SourceOperationFailureRecorded"
        and e["payload"]["scope_event_id"] in scope_ids
    ]
    if (not failures and fault.contact_attempt != 1) or (
        failures
        and any(
            failures[-1][name] != getattr(fault, name)
            for name in ("failure_class", "original_failure_class", "contact_attempt")
        )
    ):
        raise ValueError("operational failure changed its source contact window")
    if failures and failures[-1]["policy"] != fault.policy.model_dump(mode="json"):
        raise ValueError("operational failure changed its source policy")


def _validate_pending_transcript(events, fact, response, logical):
    read_messages_record(fact.messages)
    sent = logical["payload"]["compatible_adapter_evidence"]["request"]["messages"]
    suffix = fact.messages[len(sent) :]
    if fact.messages[: len(sent)] != sent or not suffix or len(suffix) > 2:
        raise ValueError("operational Review changed its pending transcript")

    def response_material(message):
        return {
            key: value
            for key, value in message.items()
            if key not in {"run_id", "conversation_id"}
        }

    from bio_neb.agentic_journey.transcript import kept_message

    returns = suffix[1].get("parts", []) if len(suffix) == 2 else []
    # The reply as the loop kept it: a routed reply holds only its kept calls, and
    # a refused single call stays only where its correction answers it.
    original = kept_message(
        events,
        response,
        frozenset(
            part["tool_call_id"]
            for part in returns
            if part.get("part_kind") == "tool-return"
        ),
    )
    if response_material(suffix[0]) != response_material(original):
        raise ValueError("operational Review changed its pending response")
    if len(suffix) == 2 and suffix[1].get("kind") != "request":
        raise ValueError("operational Review appended a non-request message")
    routed = [
        e["payload"]
        for e in events
        if e["event_type"] == "AgentTurnRouted"
        and e["payload"]["response_event_id"] == fact.response_event_id
    ]
    note = routed[0]["feedback"] if routed else None
    # A routed turn's note follows its returns, in the retry channel.
    if note is not None and returns:
        last = returns[-1]
        if (
            last.get("part_kind") != "retry-prompt"
            or last.get("tool_name") is not None
            or last.get("content") != note
        ):
            raise ValueError("operational Review lost its routed turn's note")
        returns = returns[:-1]
    expected = {}
    starts = {
        e["event_id"]: e["payload"]
        for e in events
        if e["event_type"] == "ObservedToolExecutionStarted"
        and e["payload"]["response_event_id"] == fact.response_event_id
    }
    for event in events:
        payload = event["payload"]
        if (
            event["event_type"] == "ObservedToolExecutionSucceeded"
            and payload["started_event_id"] in starts
        ):
            call = starts[payload["started_event_id"]]
            if call["tool_call_id"] != fact.tool_call_id:
                expected[call["tool_call_id"]] = (
                    call["tool_name"],
                    payload["model_visible_result_digest"],
                )
        if (
            event["event_type"] == "AgentActionTurnDispositionRecorded"
            and payload["response_event_id"] == fact.response_event_id
            and payload["feedback"] is not None
        ):
            for validation_id in payload["validation_event_ids"]:
                validation = next(
                    e["payload"] for e in events if e["event_id"] == validation_id
                )
                expected[validation["tool_call_id"]] = (
                    validation["tool_name"],
                    canonical_json_digest(payload["feedback"]),
                )
    observed = {}
    for part in returns:
        if part.get("part_kind") != "tool-return" or part["tool_call_id"] in observed:
            raise ValueError("operational Review changed its completed returns")
        observed[part["tool_call_id"]] = (
            part["tool_name"],
            canonical_json_digest(part["content"]),
        )
    if observed != expected:
        raise ValueError("operational Review lost completed or suppressed siblings")


def _validate_request(events, event, fact, fault, started, by_id):
    from bio_neb.agentic_journey.tool_execution import (
        ObservedToolExecutionStarted,
        validate_observed_tool_admission,
    )

    indexed = {e["event_id"]: (i, e) for i, e in enumerate(events)}
    validate_observed_tool_admission(
        indexed[started["event_id"]][0],
        ObservedToolExecutionStarted.model_validate(started["payload"]),
        indexed,
    )
    validation = by_id[fact.validation_event_id]["payload"]
    response = by_id[fact.response_event_id]
    contact = by_id[response["payload"]["physical_contact_started_event_id"]]
    logical = by_id[contact["payload"]["logical_request_event_id"]]
    attempt = next(
        e
        for e in events
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == fact.attempt_id
    )
    if (
        any(
            validation.get(name) != getattr(fact, name)
            for name in ("arguments", "tool_name", "tool_call_id")
        )
        or tuple(validation.get("bound_argument_names", ()))
        != fact.bound_argument_names
        or fact.agent_role != attempt["payload"]["agent_role"]
        or fact.member_id != logical["payload"].get("member_id")
        or len(attempt["parent_ids"]) != 1
        or by_id[attempt["parent_ids"][0]]["event_type"] != "activity_started"
        or by_id[attempt["parent_ids"][0]]["parent_ids"]
        != [fact.retry_frontier_event_id]
        or fact.request.stage != fact.agent_role
        or fact.request.legal_actions != ("retry", "amend", "stop")
        or fact.request.amendable != ("instruction",)
        or fact.request.retry_consequence != "reuse_intent"
        or fact.request.amendment_consequences != {}
        or fact.request.reason != fault.detail
    ):
        raise ValueError("operational Review changed its work or authority")
    _validate_fault(events[: indexed[event["event_id"]][0]], fault, started)
    _validate_pending_transcript(
        events[: indexed[event["event_id"]][0]], fact, response, logical
    )


def _validate_joined_before_contact(events, event, fact, by_id):
    """An item that joined the open Review before its call contacted the source:
    its admitted call never started, its Ask is the open Review's, and it joined
    exactly the Review open for that Tool (#477)."""
    index = events.index(event)
    lead = by_id.get(fact.joins_event_id)
    validation = by_id.get(fact.validation_event_id)
    response = by_id.get(fact.response_event_id)
    if (
        lead is None
        or validation is None
        or response is None
        or fact.joins_event_id
        != open_operational_review(events[:index], tool_name=fact.tool_name)
        or fact.fault_key != lead["payload"]["fault_key"]
        or fact.request.model_dump(exclude={"request_id"})
        != Ask.model_validate(lead["payload"]["request"]).model_dump(
            exclude={"request_id"}
        )
        or any(
            validation["payload"].get(name) != getattr(fact, name)
            for name in ("arguments", "tool_name", "tool_call_id")
        )
        or tuple(validation["payload"].get("bound_argument_names", ()))
        != fact.bound_argument_names
        or any(
            item["event_type"] == "ObservedToolExecutionStarted"
            and item["payload"]["validation_event_id"] == fact.validation_event_id
            for item in events[:index]
        )
    ):
        raise ValueError("operational join changed its open Review or its call")
    contact = by_id[response["payload"]["physical_contact_started_event_id"]]
    logical = by_id[contact["payload"]["logical_request_event_id"]]
    if fact.member_id != logical["payload"].get("member_id"):
        raise ValueError("operational join changed its member")
    _validate_pending_transcript(events[:index], fact, response, logical)


def _validate_continuation(events, event, requests, receipts, consumed, checked_once):
    validate_attempt_continuation(events, event, checked_once=checked_once)
    answer_id = event["payload"].get("continuation_operational_review_event_id")
    if answer_id is None:
        return
    receipt = next((r for r in receipts if r.response_event_id == answer_id), None)
    request = next(
        (
            r
            for r in requests
            if receipt and r.request_event_id == receipt.request_event_id
        ),
        None,
    )
    by_id = {e["event_id"]: e for e in events}
    if request is None or receipt.response.action == "stop" or answer_id in consumed:
        raise ValueError("operational continuation lacks its unconsumed direction")
    original = next(
        e
        for e in events
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == request.attempt_id
    )
    activity = by_id[event["parent_ids"][0]] if len(event["parent_ids"]) == 1 else None
    if (
        activity is None
        or activity["event_type"] != "activity_started"
        or any(
            event["payload"].get(key) != original["payload"].get(key)
            for key in (
                "agent_role",
                "input_set_id",
                "effective_invocation_policy",
                "invocation_policy_id",
                "invocation_overrides",
            )
        )
    ):
        raise ValueError(
            "operational continuation changed its scope or invocation policy"
        )
    consumed.add(answer_id)


def covering_preset(presets, request, failure_class):
    """The role's current retry preset that answers this Review, if any is left."""
    preset = next(
        (p for p in presets.values() if p.agent_role == request.agent_role), None
    )
    if (
        preset is None
        or preset.remaining == 0
        or failure_class not in preset.failure_classes
        or (preset.tool_names and request.tool_name not in preset.tool_names)
    ):
        return None
    return preset


def _record_preset(presets, event):
    fact = AgentFailureRetryPreset.model_validate(event["payload"])
    for preset_id, preset in list(presets.items()):
        if preset.agent_role == fact.agent_role:
            del presets[preset_id]
    presets[event["event_id"]] = AgentFailureRetryPresetView(
        **fact.model_dump(), preset_event_id=event["event_id"], remaining=fact.count
    )


def _validate_preset_answer(presets, request, answer, by_id):
    """A preset's answer is its role's current preset, still covering this Review."""
    failure = by_id[request.request.failure_event_id]["payload"]
    preset = covering_preset(presets, request, failure["failure_class"])
    if (
        preset is None
        or preset.preset_event_id != answer.preset_event_id
        or answer.response.action != "retry"
        or answer.response.by != preset.by
    ):
        raise ValueError("operational answer lacks its exact retry preset")
    presets[preset.preset_event_id] = preset.model_copy(
        update={"remaining": preset.remaining - 1}
    )


def project_operational_review(events, *, checked_once=None):
    """``checked_once`` is the reading owner's when ``events`` are its evidence."""
    events = list(events)
    by_id = {e["event_id"]: e for e in events}
    requests = []
    receipts = []
    consumed = set()
    presets = {}
    for event in events:
        if event["event_type"] == "AttemptStarted":
            _validate_continuation(
                events, event, requests, receipts, consumed, checked_once
            )
        if event["event_type"] == FORK_ORIGIN_EVENT_TYPE:
            # A preset is the Run's own: a Fork child's Reviews need its own.
            presets.clear()
        if event["event_type"] == "AgentFailureRetryPreset":
            _record_preset(presets, event)
            continue
        if event["event_type"] == "AgentFailureReviewResponded":
            answer = AgentFailureReviewResponded.model_validate(event["payload"])
            request, retained = evaluate_scoped_answer(
                requests,
                receipts,
                answer.request_event_id,
                answer.response,
                validate=_validate_answer,
                refusal=AgentFailureReviewRefused,
            )
            validate_review_answer_parentage(events, event)
            if retained is not None:
                raise ValueError("operational response changed its exact request")
            if answer.preset_event_id is not None:
                _validate_preset_answer(presets, request, answer, by_id)
            receipts.append(
                AgentFailureReviewReceipt(
                    request_event_id=answer.request_event_id,
                    response_event_id=event["event_id"],
                    response=answer.response,
                )
            )
            requests = [
                r.model_copy(update={"pending": False}) if r == request else r
                for r in requests
            ]
            continue
        if event["event_type"] != "AgentFailureReviewRequested":
            continue
        fact = AgentFailureReviewRequested.model_validate(event["payload"])
        if fact.joined_before_contact:
            _validate_joined_before_contact(events, event, fact, by_id)
            requests.append(
                AgentFailureReviewRequestView(
                    **fact.model_dump(), request_event_id=event["event_id"]
                )
            )
            continue
        failure = by_id.get(fact.request.failure_event_id)
        if failure is None or failure["event_type"] != "OperationalFailureRecorded":
            raise ValueError("operational Review lost its classified failure")
        fault = OperationalFailureRecorded.model_validate(failure["payload"])
        started = by_id.get(fault.started_event_id)
        if (
            fault.attempt_id != fact.attempt_id
            or started is None
            or started["event_type"] != "ObservedToolExecutionStarted"
            or any(
                started["payload"].get(k) != getattr(fact, k)
                for k in (
                    "attempt_id",
                    "response_event_id",
                    "validation_event_id",
                    "tool_call_id",
                    "tool_name",
                )
            )
            or events.index(failure) >= events.index(event)
        ):
            raise ValueError("operational Review changed its pending Tool")
        _validate_request(events, event, fact, fault, started, by_id)
        if fact.fault_key != operational_fault_key(
            fault
        ) or fact.joins_event_id != open_operational_review(
            events[: events.index(event)], fact.fault_key
        ):
            raise ValueError("operational Review changed its cause or its join")
        if any(
            r.request.failure_event_id == fact.request.failure_event_id
            for r in requests
        ):
            raise ValueError("operational failure requested two Reviews")
        requests.append(
            AgentFailureReviewRequestView(
                **fact.model_dump(), request_event_id=event["event_id"]
            )
        )
    return AgentFailureReviewProjection(
        requests=tuple(requests),
        receipts=tuple(receipts),
        # Each item of a Review is answered on its own.
        pending_request=next((r for r in requests if r.pending), None),
        retry_presets=tuple(presets.values()),
    )


def validate_operational_terminal(events, terminal):
    projection = project_operational_review(events)
    if not any(
        request.attempt_id == terminal["payload"]["attempt_id"]
        and request.request_event_id == terminal["payload"]["source_event_id"]
        and terminal["parent_ids"] == [request.request_event_id]
        for request in projection.requests
    ):
        raise ValueError("operational stop lost its exact Review")


def _validate_answer(request, response):
    if (
        response.request_id != request.request.request_id
        or response.action not in request.request.legal_actions
    ):
        raise AgentFailureReviewRefused("illegal", request.request_event_id)
    if response.action == "amend" and (
        response.field != "instruction"
        or not isinstance(response.value, str)
        or not response.value.strip()
    ):
        raise AgentFailureReviewRefused("illegal", request.request_event_id)


def evaluate_operational_review_response(projection, request_event_id, response):
    """Validate direction or acknowledge replay before constructing recovery work."""
    response = ReviewAnswer.model_validate(response)
    return evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=_validate_answer,
        refusal=AgentFailureReviewRefused,
    )


def answer_operational_review(journey, request_event_id, response):
    response = ReviewAnswer.model_validate(response)
    events = read_evidence(journey)
    projection = project_operational_review(events)
    request, retained = evaluate_operational_review_response(
        projection, request_event_id, response
    )
    if retained is not None:
        return retained, False
    return _record_answer(
        journey,
        events,
        AgentFailureReviewResponded(
            request_event_id=request_event_id, response=response
        ),
    ), True


def _record_answer(journey, events, fact):
    event = journey.record(
        fact.event_type,
        fact.model_dump(mode="json", exclude={"event_type"}),
        review_answer_parents(events, fact.request_event_id),
    )
    return AgentFailureReviewReceipt(
        request_event_id=fact.request_event_id,
        response_event_id=event["event_id"],
        response=fact.response,
    )


def preset_agent_failure_retry(journey, preset):
    """Record a researcher's retry preset; it answers Reviews that open later."""
    preset = AgentFailureRetryPreset.model_validate(preset)
    events = read_evidence(journey)
    project_operational_review(events)
    head = journey.head_event_id
    event = journey.record(
        preset.event_type,
        preset.model_dump(mode="json", exclude={"event_type"}),
        [] if head is None else [head],
    )
    return AgentFailureRetryPresetView(
        **preset.model_dump(), preset_event_id=event["event_id"], remaining=preset.count
    )


def answer_from_retry_preset(journey):
    """Answer each pending Agent failure Review a current retry preset covers.

    The answer is the ordinary `retry` of the Review's own record, naming the
    preset; the coordinator then continues the stopped work as it would after a
    researcher's own answer. Returns whether any Review was answered.
    """
    answered = False
    while True:
        events = read_evidence(journey)
        projection = project_operational_review(events)
        presets = {p.preset_event_id: p for p in projection.retry_presets}
        by_id = {e["event_id"]: e for e in events}
        covered = next(
            (
                (request, preset)
                for request in projection.requests
                if request.pending
                and (
                    preset := covering_preset(
                        presets,
                        request,
                        by_id[request.request.failure_event_id]["payload"][
                            "failure_class"
                        ],
                    )
                )
                is not None
            ),
            None,
        )
        if covered is None:
            return answered
        request, preset = covered
        _record_answer(
            journey,
            events,
            AgentFailureReviewResponded(
                request_event_id=request.request_event_id,
                response=ReviewAnswer(
                    request_id=request.request.request_id,
                    action="retry",
                    by=preset.by,
                    reason="Answered by the researcher's retry preset.",
                ),
                preset_event_id=preset.preset_event_id,
            ),
        )
        answered = True


def pending_operational_work(journey, response_event_id, *, agent_role):
    events = read_evidence(journey)
    projection = project_operational_review(events)
    receipt = next(
        (r for r in projection.receipts if r.response_event_id == response_event_id),
        None,
    )
    request = next(
        (
            r
            for r in projection.requests
            if receipt and r.request_event_id == receipt.request_event_id
        ),
        None,
    )
    if (
        request is None
        or request.agent_role != agent_role
        or receipt.response.action == "stop"
    ):
        raise ValueError("operational re-entry lacks its exact direction")
    if any(
        e["event_type"] == "AttemptStarted"
        and e["payload"].get("continuation_operational_review_event_id")
        == response_event_id
        for e in events
    ):
        raise ValueError("operational direction was already consumed")
    return request, receipt


def resumed_transcript(request):
    """The pending transcript a `retry` resumes, as framework messages.

    A routed turn's note, the pending request's last part, is owed after the
    stopped call's result: it leaves the pending request, and the admission
    session sends it once that result exists (ADR 0044's note).
    """
    messages = list(request.messages)
    pending = messages[-1]
    parts = pending.get("parts", [])
    if (
        pending.get("kind") == "request"
        and parts
        and parts[-1].get("part_kind") == "retry-prompt"
        and parts[-1].get("tool_name") is None
    ):
        messages[-1:] = [{**pending, "parts": parts[:-1]}] if parts[:-1] else []
    return read_messages_record(messages)


def operational_state(events, *, agent_role="planning_agent", checked_once=None):
    projection = project_operational_review(events, checked_once=checked_once)
    requests = {
        r.request_event_id: r for r in projection.requests if r.agent_role == agent_role
    }
    if any(r.pending for r in requests.values()):
        return "operational_review_pending"
    for receipt in reversed(projection.receipts):
        if receipt.request_event_id not in requests:
            continue
        if receipt.response.action == "stop":
            return "operational_stopped"
        if not any(
            e["event_type"] == "AttemptStarted"
            and e["payload"].get("continuation_operational_review_event_id")
            == receipt.response_event_id
            for e in events
        ):
            return "operational_review_direction_recorded"
    return None
