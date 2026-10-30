"""Observed model-endpoint availability review contracts and replay."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from bio_neb.agentic_journey.attempt_journey import (
    MODEL_REQUEST_TERMINALS,
    AttemptCancelled,
    AttemptFailed,
    ModelTurnCancelled,
    NebiusModelEndpointEvidence,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.continuation import (
    pending_continuation,
    validate_attempt_continuation,
    validate_review_answer_parentage,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.contracts import FrozenContract
from bio_neb.failure import ModelContactPolicy
from bio_neb.journey import Journey, JourneyEvent
from bio_neb.recording import read_evidence

ModelAvailabilityExhaustionReason = Literal[
    "request_rejected",
    "malformed_response",
    "automatic_retry_exhausted",
    "retry_after_over_cap",
    "wait_budget_exhausted",
    "contact_stalled",
]


def _uuid(value: str, name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be canonical")
    return value


class ModelAvailabilityReviewRequested(FrozenContract):
    event_type: Literal["ModelAvailabilityReviewRequested"] = (
        "ModelAvailabilityReviewRequested"
    )
    attempt_id: StrictStr
    agent_role: AgentRole
    member_id: StrictStr | None = None
    retry_frontier_event_id: StrictStr
    # The item's own failed request; None for an item that joined the open
    # Review at its next model call, before sending (orchestrator answer).
    logical_request_event_id: StrictStr | None = None
    logical_failure_event_id: StrictStr | None = None
    request_number: StrictInt | None = Field(default=None, gt=0)
    nebius_model_endpoint: NebiusModelEndpointEvidence
    physical_contact_count: StrictInt | None = Field(default=None, gt=0)
    exhaustion_reason: ModelAvailabilityExhaustionReason
    # The failure module's key for the cause: the endpoint's status and body,
    # or the exhaustion. While one Review is open, every other item's next
    # model call joins it (``joins_event_id``) without sending; an item
    # already failing joins it too. One answer releases all.
    fault_key: StrictStr = Field(min_length=1)
    provider_statement: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    joins_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator(
        "attempt_id",
        "retry_frontier_event_id",
        "logical_request_event_id",
        "logical_failure_event_id",
    )
    @classmethod
    def canonical_ids(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _uuid(value, info.field_name)

    @model_validator(mode="after")
    def joined_before_sending_names_no_request(self):
        unsent = self.logical_request_event_id is None
        if any(
            (value is None) != unsent
            for value in (
                self.logical_failure_event_id,
                self.request_number,
                self.physical_contact_count,
            )
        ) or (unsent and self.joins_event_id is None):
            raise ValueError("an availability request names its failed request")
        return self

    @property
    def joined_before_sending(self) -> bool:
        return self.logical_request_event_id is None


class RetryModelAvailabilityResponse(FrozenContract):
    action: Literal["retry"]


class AmendModelAvailabilityResponse(FrozenContract):
    action: Literal["amend"]
    instruction: StrictStr = Field(min_length=1, max_length=2_000)

    @field_validator("instruction")
    @classmethod
    def canonical_instruction(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("instruction must be a non-empty canonical string")
        return value


class StopModelAvailabilityResponse(FrozenContract):
    action: Literal["stop"]
    reason: StrictStr = Field(max_length=2_000)

    @field_validator("reason")
    @classmethod
    def canonical_reason(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("reason must be a non-empty canonical string")
        return value


ModelAvailabilityReviewResponse = Annotated[
    RetryModelAvailabilityResponse
    | AmendModelAvailabilityResponse
    | StopModelAvailabilityResponse,
    Field(discriminator="action"),
]


class ModelAvailabilityReviewResponded(FrozenContract):
    event_type: Literal["ModelAvailabilityReviewResponded"] = (
        "ModelAvailabilityReviewResponded"
    )
    request_event_id: StrictStr
    response: ModelAvailabilityReviewResponse
    authority: Literal["researcher_direction"] = "researcher_direction"
    scientifically_verified: Literal[False] = False
    authorizes_execution: Literal[False] = False

    @field_validator("request_event_id")
    @classmethod
    def canonical_request(cls, value: str) -> str:
        return _uuid(value, "request_event_id")


class ModelAvailabilityStopped(FrozenContract):
    event_type: Literal["ModelAvailabilityStopped"] = "ModelAvailabilityStopped"
    request_event_id: StrictStr
    response_event_id: StrictStr
    reason: StrictStr

    @field_validator("request_event_id", "response_event_id")
    @classmethod
    def canonical_ids(cls, value: str, info: Any) -> str:
        return _uuid(value, info.field_name)


class ModelAvailabilityContactView(FrozenContract):
    ordinal: StrictInt = Field(gt=0)
    status_code: StrictInt | None = None
    failure_reason: StrictStr | None = None
    completion_certainty: StrictStr
    policy_disposition: StrictStr
    retry_after_disposition: StrictStr | None = None
    retry_after_seconds: float | None = None


class ModelAvailabilityDelayView(FrozenContract):
    seconds: float = Field(ge=0, allow_inf_nan=False)
    source: Literal[
        "retry_after",
        "exponential_jitter",
        "malformed_response",
    ]
    completed: bool


class ModelAvailabilityReviewRequestView(FrozenContract):
    request_event_id: StrictStr
    attempt_id: StrictStr
    agent_role: AgentRole
    member_id: StrictStr | None = None
    retry_frontier_event_id: StrictStr
    logical_request_event_id: StrictStr | None = None
    logical_failure_event_id: StrictStr | None = None
    request_number: StrictInt | None = None
    nebius_model_endpoint: NebiusModelEndpointEvidence
    contact_policy: ModelContactPolicy = Field(default_factory=ModelContactPolicy)
    contacts: tuple[ModelAvailabilityContactView, ...] = ()
    delays: tuple[ModelAvailabilityDelayView, ...]
    exhaustion_reason: ModelAvailabilityExhaustionReason
    pending: bool
    # The cause, in the provider's words when the endpoint answered, and the open
    # Review this one waits on (G4.2).
    fault_key: StrictStr
    provider_statement: StrictStr | None = None
    joins_event_id: StrictStr | None = None

    @property
    def joined_before_sending(self) -> bool:
        return self.logical_request_event_id is None

    @field_validator(
        "request_event_id",
        "attempt_id",
        "retry_frontier_event_id",
        "logical_request_event_id",
        "logical_failure_event_id",
    )
    @classmethod
    def canonical_ids(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _uuid(value, info.field_name)


class ModelAvailabilityReviewResponseReceipt(FrozenContract):
    status: Literal["accepted"] = "accepted"
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: ModelAvailabilityReviewResponse

    @field_validator("request_event_id", "response_event_id")
    @classmethod
    def canonical_ids(cls, value: str, info: Any) -> str:
        return _uuid(value, info.field_name)


class ModelAvailabilityReviewProjection(FrozenContract):
    requests: tuple[ModelAvailabilityReviewRequestView, ...]
    receipts: tuple[ModelAvailabilityReviewResponseReceipt, ...]
    pending_request: ModelAvailabilityReviewRequestView | None
    stop_event_id: StrictStr | None = None


class ModelAvailabilityReviewResponseRefused(ValueError):
    def __init__(self, code: str, request_event_id: str):
        super().__init__(code)
        self.code = code
        self.request_event_id = request_event_id
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class ModelAvailabilityReviewPending(Exception):
    """Control signal emitted only after a logical request closes for review."""

    def __init__(
        self,
        *,
        request_event_id: str,
        failure_event_id: str,
        request_number: int,
        physical_contact_count: int,
        exhaustion_reason: ModelAvailabilityExhaustionReason,
        nebius_model_endpoint: NebiusModelEndpointEvidence,
        fault_key: str | None = None,
        provider_statement: str | None = None,
    ):
        super().__init__(exhaustion_reason)
        self.request_event_id = request_event_id
        self.failure_event_id = failure_event_id
        self.request_number = request_number
        self.physical_contact_count = physical_contact_count
        self.exhaustion_reason = exhaustion_reason
        self.nebius_model_endpoint = nebius_model_endpoint
        # The cause's key: the endpoint's status and body, or the exhaustion.
        self.fault_key = fault_key or availability_fault_key(exhaustion_reason)
        self.provider_statement = provider_statement
        self.member_id = None
        # The open Review this item joins at its next call, sending nothing.
        self.joins_before_sending: str | None = None

    @classmethod
    def joining(cls, lead: Mapping[str, Any]) -> "ModelAvailabilityReviewPending":
        """An item's next model call that joins the open Review ``lead``."""
        payload = lead["payload"]
        pending = cls.__new__(cls)
        Exception.__init__(pending, payload["exhaustion_reason"])
        pending.request_event_id = None
        pending.failure_event_id = None
        pending.request_number = None
        pending.physical_contact_count = None
        pending.exhaustion_reason = payload["exhaustion_reason"]
        pending.nebius_model_endpoint = NebiusModelEndpointEvidence.model_validate(
            payload["nebius_model_endpoint"]
        )
        pending.fault_key = payload["fault_key"]
        pending.provider_statement = payload.get("provider_statement")
        pending.member_id = None
        pending.joins_before_sending = lead["event_id"]
        return pending


def availability_fault_key(exhaustion_reason: str) -> str:
    """The key of an exhaustion no answer body names (G4.1): its reason."""
    return f"model endpoint: {exhaustion_reason}"


def open_availability_review(
    events, fault_key: str | None = None, *, endpoint: Mapping[str, Any] | None = None
) -> str | None:
    """The open model endpoint Review a new item joins (G4.1): one with this
    cause's key, or, before sending, any open one on this endpoint."""
    from bio_neb.agentic_journey.keyed_review import open_keyed_review

    return open_keyed_review(
        events,
        requested="ModelAvailabilityReviewRequested",
        responded="ModelAvailabilityReviewResponded",
        matches=lambda payload: (
            fault_key is None or payload.get("fault_key") == fault_key
        )
        and (endpoint is None or payload.get("nebius_model_endpoint") == endpoint),
    )


def record_model_availability_review_pending(
    attempt_activity: Activity,
    pending: ModelAvailabilityReviewPending,
    *,
    attempt_id: str,
    agent_role: AgentRole,
    retry_frontier_event_id: str,
    close_attempt: bool = True,
) -> str:
    """Close one exhausted Attempt while deliberately leaving its Run open.

    An open Review with the same cause takes this one as joined: the Run asks
    once, and that one answer releases both (owner, #476 alignment G4.1).
    """
    from bio_neb.recording import evidence_view

    joins = pending.joins_before_sending or open_availability_review(
        evidence_view(attempt_activity.journey), pending.fault_key
    )
    if pending.joins_before_sending is not None and close_attempt:
        raise ValueError("an item that joined before sending holds its own Attempt")
    requested = attempt_activity.record(
        ModelAvailabilityReviewRequested(
            attempt_id=attempt_id,
            agent_role=agent_role,
            member_id=pending.member_id,
            retry_frontier_event_id=retry_frontier_event_id,
            logical_request_event_id=pending.request_event_id,
            logical_failure_event_id=pending.failure_event_id,
            request_number=pending.request_number,
            nebius_model_endpoint=pending.nebius_model_endpoint,
            physical_contact_count=pending.physical_contact_count,
            exhaustion_reason=pending.exhaustion_reason,
            fault_key=pending.fault_key,
            provider_statement=pending.provider_statement,
            joins_event_id=joins,
        )
    )
    if close_attempt:
        failed = attempt_activity.record(
            AttemptFailed(
                attempt_id=attempt_id,
                code="model_availability_exhausted",
                source_event_id=pending.failure_event_id,
            )
        )
        attempt_activity.failed(failed)
    return requested["event_id"]


def record_model_attempt_cancelled(
    attempt_activity: Activity,
    journey: Journey,
    *,
    attempt_id: str,
) -> None:
    """Cancel the Attempt on the exact source the interrupt met.

    An open contact or retry delay recorded its own cancellation. With nothing in
    flight, every model request has its outcome and the turn boundary is the source.
    """
    owned = [
        event
        for event in read_evidence(journey)
        if event.get("payload", {}).get("attempt_id") == attempt_id
    ]
    sources = [
        event
        for event in owned
        if event.get("event_type")
        in {"ModelPhysicalContactCancelled", "ModelAvailabilityRetryDelayCancelled"}
    ]
    requests = sum(e["event_type"] == "ModelRequestRecorded" for e in owned)
    if not sources and requests == sum(
        e["event_type"] in MODEL_REQUEST_TERMINALS for e in owned
    ):
        sources = [
            attempt_activity.record(
                ModelTurnCancelled(attempt_id=attempt_id, model_request_count=requests)
            )
        ]
    if len(sources) != 1:
        raise RuntimeError("model cancellation lacks its exact lifecycle source")
    cancelled = attempt_activity.record(
        AttemptCancelled(
            attempt_id=attempt_id,
            source_event_id=sources[0]["event_id"],
        )
    )
    attempt_activity.cancelled(cancelled)


def _as_event(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def _fact(event: Mapping[str, Any], owner: type[FrozenContract]) -> Any:
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


def _has_ancestor(
    by_id: Mapping[str, Mapping[str, Any]], ancestor_id: str, descendant_id: str
) -> bool:
    pending = [descendant_id]
    seen: set[str] = set()
    while pending:
        selected = pending.pop()
        if selected == ancestor_id:
            return True
        if selected in seen:
            continue
        seen.add(selected)
        event = by_id.get(selected)
        if event is not None:
            pending.extend(event.get("parent_ids", []))
    return False


# How early the event loop may fire a timer that cut a send off at its limit.
_TIMER_TOLERANCE = 0.001


def validate_model_contact_window(events, request, terminal=None, *, cancellation=None):
    """Check the retained physical sequence against its recorded effective window.

    Both recovery views and ordinary/cancelled readers use this owner. A retained
    random delay proves its allowed range, never the random sample that chose it.
    """
    from bio_neb.failure import ModelContactPolicy

    policy = ModelContactPolicy.model_validate(
        request["payload"].get("contact_policy", {})
    )
    after_stall = request["payload"].get("after_stall_event_id") is not None
    # The refusal this request carried: the same refusal of it asks (G1.5).
    shown_key = request["payload"].get("after_rejection_fault_key")
    positions = {event["event_id"]: index for index, event in enumerate(events)}
    physical_types = {
        "ModelPhysicalContactStarted",
        "ModelPhysicalContactFailed",
        "ModelPhysicalContactCancelled",
        "ModelAvailabilityRetryScheduled",
        "ModelAvailabilityRetryDelayCompleted",
        "ModelAvailabilityRetryDelayCancelled",
    }
    # Concurrent Attempts interleave: the window holds this logical request's
    # physical contacts and its siblings'; each reads only its own.
    physical = [
        e
        for e in events
        if e["event_type"] in physical_types
        and e["payload"].get("logical_request_event_id") in {request["event_id"], None}
    ]
    state, ordinal, waited = "start", 1, 0.0
    started = failed = scheduled = completed = None
    malformed_responses = 0
    for event in physical:
        if validate_attempt_fact(event) is None:
            raise ValueError("invalid physical model fact")
        p = event["payload"]
        if (
            any(
                p.get(k) != request["payload"][k]
                for k in ("attempt_id", "request_number")
            )
            or p.get("logical_request_event_id") != request["event_id"]
        ):
            raise ValueError("physical contact changed its logical request")
        kind = event["event_type"]
        if state == "start" and kind == "ModelPhysicalContactStarted":
            if p["contact_ordinal"] != ordinal or p.get(
                "retry_delay_completed_event_id"
            ) != (completed["event_id"] if completed else None):
                raise ValueError("physical contact changed its ordinal or delay")
            source = completed or request
            if positions.get(source["event_id"], -1) >= positions[event["event_id"]]:
                raise ValueError(
                    "physical contact is detached from its request or delay"
                )
            started, state = event, "contact"
        elif state == "contact" and kind in {
            "ModelPhysicalContactFailed",
            "ModelPhysicalContactCancelled",
        }:
            if (
                p["contact_ordinal"] != ordinal
                or p["contact_started_event_id"] != started["event_id"]
            ):
                raise ValueError("physical failure/cancellation changed its start")
            failed = event
            if kind == "ModelPhysicalContactFailed":
                _validate_failure_body(events, event, started)
            if p.get("failure_reason") == "malformed_response":
                malformed_responses += 1
            # A stall is the loop's own cut-off: its monotonic seconds reach the
            # window's limit, within the event loop's timer.
            if (
                p.get("failure_reason") == "contact_stalled"
                and p["contact_seconds"] < policy.stall_seconds - _TIMER_TOLERANCE
            ):
                raise ValueError("stalled contact ended before its stall limit")
            state = "cancelled" if kind.endswith("Cancelled") else "failure"
        elif state == "failure" and kind == "ModelAvailabilityRetryScheduled":
            f = failed["payload"]
            if (
                f["policy_disposition"] != "automatic_retry_scheduled"
                or p["failed_contact_event_id"] != failed["event_id"]
                or p["failed_contact_ordinal"] != ordinal
                or p["next_contact_ordinal"] != ordinal + 1
            ):
                raise ValueError("retry schedule changed its failure")
            expected = _verdict(
                failed,
                policy,
                ordinal,
                waited,
                malformed_responses,
                after_stall,
                shown_key,
            )
            if (
                expected.policy_disposition != "automatic_retry_scheduled"
                or expected.delay_seconds != p["delay_seconds"]
                or expected.delay_source != p["delay_source"]
            ):
                raise ValueError("retry schedule violates its effective policy")
            scheduled, state = event, "delay"
        elif state == "delay" and kind in {
            "ModelAvailabilityRetryDelayCompleted",
            "ModelAvailabilityRetryDelayCancelled",
        }:
            if p["retry_scheduled_event_id"] != scheduled["event_id"] or any(
                p[k] != scheduled["payload"][k]
                for k in (
                    "delay_seconds",
                    "failed_contact_ordinal",
                    "next_contact_ordinal",
                )
            ):
                raise ValueError("retry delay changed its schedule")
            if kind.endswith("Cancelled"):
                state = "cancelled"
            else:
                waited += p["delay_seconds"]
                ordinal += 1
                completed, state = event, "start"
        else:
            raise ValueError("physical contact facts are reordered or duplicated")
    if cancellation is not None:
        if (
            state != "cancelled"
            or not physical
            or physical[-1]["event_id"] != cancellation["payload"]["source_event_id"]
        ):
            raise ValueError("cancelled contact lacks exact cancellation source")
        return
    _validate_contact_terminal(
        terminal,
        state,
        ordinal,
        started,
        failed,
        _verdict(
            failed,
            policy,
            ordinal,
            waited,
            malformed_responses,
            after_stall,
            shown_key,
        )
        if state == "failure"
        else None,
        policy,
    )


def _validate_failure_body(events, failed, started):
    """An HTTP failure names its own contact's retained answer, and its key is
    that answer's status and digest (#477 G4.2; Standards r1 ST1). The words are
    re-read from the bytes by the transport reader."""
    from bio_neb.failure import contact_fault_key, model_fault_key

    f = failed["payload"]
    pointer = f.get("response_event_id")
    response = next(
        (
            event
            for event in events
            if event["event_type"] == "ModelTransportResponseRecorded"
            and event["payload"]["contact_started_event_id"] == started["event_id"]
        ),
        None,
    )
    if pointer is None:
        # No answer retained: the cause is the status, or how the contact failed.
        if f.get("provider_statement") is not None or f.get("fault_key") not in (
            None,
            contact_fault_key(
                validate_attempt_fact(failed).failure, f.get("http_status_code")
            ),
        ):
            raise ValueError("model failure names a cause without its answer")
        return
    if (
        response is None
        or response["event_id"] != pointer
        or f.get("http_status_code") != response["payload"]["status_code"]
        or f.get("fault_key")
        != model_fault_key(
            response["payload"]["status_code"],
            response["payload"]["response_body_sha256"],
        )
    ):
        raise ValueError("model failure changed its retained answer or its cause")


def _verdict(  # noqa: PLR0913 - the replayed window's state at one contact
    failed, policy, ordinal, waited, malformed_responses, after_stall, shown_key
):
    """The failure package's verdict on a retained failed contact (#443)."""
    from bio_neb.failure import model_contact_verdict

    fact = validate_attempt_fact(failed)
    f = failed["payload"]
    return model_contact_verdict(
        policy,
        failure=fact.failure,
        contact=ordinal,
        waited=waited,
        jitter=f.get("retry_jitter", 0.5),
        http_status_code=f.get("http_status_code"),
        retry_after_disposition=f.get("retry_after_disposition"),
        retry_after_seconds=f.get("retry_after_seconds"),
        malformed_responses=malformed_responses,
        after_stall=after_stall,
        rejection_key=f.get("fault_key") if f.get("provider_statement") else None,
        shown_key=shown_key,
        rejoins=f.get("open_review_event_id") is not None,
    )


def validate_attempt_cancellation(
    events, cancellation, model_requests, model_terminals
):
    """Check an Attempt's ``AttemptCancelled`` (or None) against its exact source.

    ``events`` hold the Attempt's facts in recorded order. An open request's
    contact or retry delay is its own source, ending that request's window. With
    nothing in flight every request has its outcome, and one ``ModelTurnCancelled``
    after the last of them counts them; no other Attempt retains one.
    """
    turns = [e for e in events if e["event_type"] == "ModelTurnCancelled"]
    if cancellation is None:
        if turns:
            raise ValueError("a turn cancellation belongs only to a cancelled Attempt")
        return
    position = {event["event_id"]: index for index, event in enumerate(events)}
    source_id = cancellation["payload"]["source_event_id"]
    if (
        any(
            event["event_id"] not in position
            for event in (cancellation, *model_requests, *model_terminals)
        )
        or position.get(source_id, len(events)) >= position[cancellation["event_id"]]
    ):
        raise ValueError("Attempt cancellation lacks its exact model source")
    if len(model_terminals) < len(model_requests):
        if turns or len(model_terminals) != len(model_requests) - 1:
            raise ValueError("Attempt cancellation lacks its exact model source")
        validate_model_contact_window(
            events[
                position[model_requests[-1]["event_id"]] : position[
                    cancellation["event_id"]
                ]
                + 1
            ],
            model_requests[-1],
            cancellation=cancellation,
        )
        return
    source = events[position[source_id]]
    if (
        turns != [source]
        or source["payload"]["model_request_count"] != len(model_requests)
        or (
            model_terminals
            and position[source_id] < position[model_terminals[-1]["event_id"]]
        )
    ):
        raise ValueError("Attempt cancellation lacks its exact model source")


def _validate_contact_terminal(  # noqa: PLR0913 - the replayed window's state at its end
    terminal, state, ordinal, started, failed, verdict, policy
):
    if terminal is None:
        raise ValueError("logical request lacks its terminal")
    t = terminal["payload"]
    if terminal["event_type"] == "ModelResponseRecorded":
        if (
            state != "contact"
            or t["physical_contact_ordinal"] != ordinal
            or t["physical_contact_started_event_id"] != started["event_id"]
        ):
            raise ValueError("response changed its physical contact")
        return
    if state != "failure":
        raise ValueError("logical failure lacks its physical terminal")
    f = failed["payload"]
    if f["policy_disposition"] != verdict.policy_disposition:
        raise ValueError("logical failure departs from its contact verdict")
    if terminal["event_type"] == "ModelRequestRejected":
        if (
            verdict.policy_disposition != "rejection_shown"
            or t["physical_contact_count"] != ordinal
            or t["last_physical_contact_event_id"] != failed["event_id"]
            or any(
                t[k] != f.get(k)
                for k in (
                    "response_event_id",
                    "http_status_code",
                    "fault_key",
                    "provider_statement",
                )
            )
        ):
            raise ValueError("refused request changed its physical terminal")
        return
    if terminal["event_type"] == "ModelRequestStalled":
        if (
            verdict.policy_disposition != "stall_noted"
            or t["physical_contact_count"] != ordinal
            or t["last_physical_contact_event_id"] != failed["event_id"]
            or t["stall_seconds"] != policy.stall_seconds
        ):
            raise ValueError("stalled request changed its physical terminal")
        return
    if verdict.policy_disposition not in {
        "availability_review_requested",
        "terminal_failure",
    }:
        raise ValueError("logical failure has no exhausted contact window")
    if (
        t["physical_contact_count"] != ordinal
        or t["last_physical_contact_event_id"] != failed["event_id"]
        or any(
            t.get(k) != f.get(k)
            for k in (
                "error_type",
                "http_status_code",
                "failure_reason",
                "response_event_id",
                "fault_key",
                "provider_statement",
            )
        )
        or t.get("availability_exhaustion_reason") != verdict.exhaustion_reason
    ):
        raise ValueError("logical failure changed its physical terminal")


def validate_provider_continuations(timeline, requests, responses, checked_once=None):
    """A new Attempt may inherit work only through its exact retained retry answer."""
    by_id = {e["event_id"]: e for e in timeline}
    starts = {
        e["payload"]["attempt_id"]: e
        for e in timeline
        if e["event_type"] == "AttemptStarted"
    }
    consumed = set()
    for attempt_id, start in starts.items():
        binding = validate_attempt_continuation(
            timeline, start, checked_once=checked_once
        )
        continuation_id = start["payload"].get("continuation_request_event_id")
        model_requests = [
            e
            for e in timeline
            if e["event_type"] == "ModelRequestRecorded"
            and e["payload"]["attempt_id"] == attempt_id
        ]
        direction_id = start["payload"].get("continuation_availability_review_event_id")
        causal_answer = None if binding is None else binding.answer["event_id"]
        if direction_id is not None:
            matches = [
                (identity, request)
                for identity, request in requests
                if identity in responses and responses[identity][0] == direction_id
            ]
            if (
                len(matches) != 1
                or direction_id != causal_answer
                or direction_id in consumed
            ):
                raise ValueError(
                    "model availability continuation lacks exact authority"
                )
            identity, request = matches[0]
            response = responses[identity][1].response
            if (
                response.action == "stop"
                or request.agent_role != start["payload"]["agent_role"]
            ):
                raise ValueError(
                    "model availability continuation changed its direction"
                )
            if any(
                e["payload"].get("member_id") != request.member_id
                for e in model_requests
            ):
                raise ValueError(
                    "model availability continuation changed its member scope"
                )
            original = starts[request.attempt_id]["payload"]
            if any(
                start["payload"].get(key) != original.get(key)
                for key in (
                    "agent_role",
                    "input_set_id",
                    "instruction_identity",
                    "effective_invocation_policy",
                    "invocation_policy_id",
                    "invocation_overrides",
                )
            ):
                raise ValueError(
                    "model availability continuation changed invocation policy"
                )
            if response.action == "amend":
                if continuation_id is not None:
                    raise ValueError(
                        "model amendment cannot inherit pending model work"
                    )
                consumed.add(direction_id)
            elif continuation_id != request.logical_request_event_id:
                raise ValueError("model retry lost its pending request")
        elif causal_answer in {value[0] for value in responses.values()}:
            raise ValueError(
                "model availability continuation lost its direction binding"
            )
        if continuation_id is None:
            if any(
                e["payload"].get("continuation_request_event_id")
                for e in model_requests
            ):
                raise ValueError("model continuation lacks its Attempt binding")
            continue
        answer_id = direction_id
        matches = [
            (request_id, req)
            for request_id, req in requests
            if req.logical_request_event_id == continuation_id
            and request_id in responses
            and responses[request_id][0] == answer_id
            and responses[request_id][1].response.action == "retry"
        ]
        if len(matches) != 1 or answer_id in consumed:
            raise ValueError("provider continuation lacks one exact retry authority")
        consumed.add(answer_id)
        prior_request = by_id[continuation_id]
        prior_start = starts[prior_request["payload"]["attempt_id"]]
        if any(
            start["payload"].get(k) != prior_start["payload"].get(k)
            for k in (
                "agent_role",
                "input_set_id",
                "instruction_identity",
                "effective_invocation_policy",
                "invocation_policy_id",
                "invocation_overrides",
            )
        ):
            raise ValueError("provider continuation changed its retained invocation")
        if (
            not model_requests
            or model_requests[0]["payload"].get("continuation_request_event_id")
            != continuation_id
            or any(
                e["payload"].get("continuation_request_event_id")
                for e in model_requests[1:]
            )
        ):
            raise ValueError(
                "provider continuation changed its pending request binding"
            )
        first = model_requests[0]["payload"]
        if first["compatible_adapter_evidence"]["request"] != prior_request["payload"][
            "compatible_adapter_evidence"
        ]["request"] or first.get("contact_policy") != prior_request["payload"].get(
            "contact_policy"
        ):
            raise ValueError(
                "provider continuation changed pending messages, Tools or policy"
            )


def _operational_retry_preserves_frontier(timeline, started, lifecycle, request):
    """An exact Tool retry may inherit its stopped Attempt's model frontier."""
    if started is None or lifecycle is None:
        return False
    answer_id = started["payload"].get("continuation_operational_review_event_id")
    if answer_id is None:
        return False
    from bio_neb.agentic_journey.operational_review import project_operational_review

    # Reuse the operational owner's answer, fault, transcript and invocation
    # checks. A coincidental ancestor or another member's answer grants nothing.
    operational = project_operational_review(timeline[: timeline.index(started) + 1])
    receipt = next(
        (item for item in operational.receipts if item.response_event_id == answer_id),
        None,
    )
    if receipt is None or receipt.response.action != "retry":
        return False
    origin = next(
        item
        for item in operational.requests
        if item.request_event_id == receipt.request_event_id
    )
    return (
        origin.retry_frontier_event_id == request.retry_frontier_event_id
        and origin.agent_role == request.agent_role
        and origin.member_id == request.member_id
    )


def project_model_availability_review(  # noqa: C901 - Existing orchestration complexity; contract-base migration preserves this flow.
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    checked_once=None,
) -> ModelAvailabilityReviewProjection:
    """Rebuild every availability junction and reject hostile correlation."""

    timeline = [_as_event(event) for event in events]
    by_id: dict[str, dict[str, Any]] = {}
    requests: list[tuple[str, ModelAvailabilityReviewRequested]] = []
    responses: dict[str, tuple[str, ModelAvailabilityReviewResponded]] = {}
    pending_ids: set[str] = set()
    stop_event_id: str | None = None
    stopped_answers: set[str] = set()
    for event in timeline:
        event_id = _uuid(event.get("event_id"), "event_id")
        if event_id in by_id:
            raise ValueError("Journey repeats an event identity")
        if not isinstance(event.get("parent_ids"), list) or any(
            parent not in by_id for parent in event["parent_ids"]
        ):
            raise ValueError("Journey event has invalid backward parents")
        by_id[event_id] = event
        if event.get("event_type") == "ModelAvailabilityReviewRequested":
            request = _fact(event, ModelAvailabilityReviewRequested)
            logical_request = by_id.get(request.logical_request_event_id)
            logical_failure = by_id.get(request.logical_failure_event_id)
            started_events = [
                item
                for item in timeline
                if item.get("event_type") == "AttemptStarted"
                and item.get("payload", {}).get("attempt_id") == request.attempt_id
            ]
            started = started_events[0] if len(started_events) == 1 else None
            lifecycle = (
                by_id.get(started["parent_ids"][0])
                if started is not None and len(started.get("parent_ids", [])) == 1
                else None
            )
            prior_retry_request = (
                next(
                    (
                        prior
                        for prior_id, prior in requests
                        if (retained := responses.get(prior_id)) is not None
                        and retained[0]
                        == started["payload"].get(
                            "continuation_availability_review_event_id"
                        )
                    ),
                    None,
                )
                if lifecycle is not None
                else None
            )
            frontier_valid = (
                lifecycle is not None
                and lifecycle.get("event_type") == "activity_started"
                and (
                    lifecycle.get("parent_ids") == [request.retry_frontier_event_id]
                    or (
                        prior_retry_request is not None
                        and prior_retry_request.retry_frontier_event_id
                        == request.retry_frontier_event_id
                    )
                    or _operational_retry_preserves_frontier(
                        timeline, started, lifecycle, request
                    )
                )
            )
            request_index = timeline.index(event)
            logical_request_index = (
                timeline.index(logical_request) if logical_request is not None else -1
            )
            logical_failure_index = (
                timeline.index(logical_failure) if logical_failure is not None else -1
            )
            attempt_terminals = [
                item
                for item in timeline[request_index + 1 :]
                if item.get("event_type") == "AttemptFailed"
                and item.get("payload", {}).get("attempt_id") == request.attempt_id
            ]
            # An Attempt the stop ends closes on the shared exact terminal (the
            # planner of a collection keeps it open and holds the Review in its
            # completion instead); either closure names this exact Review.
            closure_valid = (
                len(attempt_terminals) == 1
                and attempt_terminals[0]["payload"].get("code")
                == "model_availability_exhausted"
                and attempt_terminals[0]["payload"].get("source_event_id")
                == request.logical_failure_event_id
            ) or (
                request.agent_role != "planning_agent"
                and any(
                    item["event_type"] == "InvestigationAttemptCompleted"
                    and item["payload"]["attempt_id"] == request.attempt_id
                    and event_id in item["payload"].get("held_review_event_ids", ())
                    for item in timeline[request_index + 1 :]
                )
            )
            if request.joined_before_sending:
                # The item joined the open Review at its next model call and sent
                # nothing: the cause and the words are the open Review's.
                lead = by_id.get(request.joins_event_id)
                if (
                    started is None
                    or started["payload"].get("agent_role") != request.agent_role
                    or not frontier_valid
                    or request.retry_frontier_event_id not in by_id
                    or request.agent_role == "planning_agent"
                    or not closure_valid
                    or lead is None
                    or request.joins_event_id
                    != open_availability_review(
                        timeline[:request_index],
                        endpoint=request.nebius_model_endpoint.model_dump(mode="json"),
                    )
                    or (
                        lead["payload"]["fault_key"],
                        lead["payload"].get("provider_statement"),
                        lead["payload"]["exhaustion_reason"],
                    )
                    != (
                        request.fault_key,
                        request.provider_statement,
                        request.exhaustion_reason,
                    )
                ):
                    raise ValueError("availability join changed its open Review")
                requests.append((event_id, request))
                pending_ids.add(event_id)
                continue
            if (
                logical_request is None
                or logical_request.get("event_type") != "ModelRequestRecorded"
                or logical_failure is None
                or logical_failure.get("event_type") != "ModelRequestFailed"
                or logical_request["payload"].get("attempt_id") != request.attempt_id
                or logical_failure["payload"].get("attempt_id") != request.attempt_id
                or logical_request["payload"].get("request_number")
                != request.request_number
                or logical_failure["payload"].get("request_number")
                != request.request_number
                or logical_failure["payload"].get("availability_exhaustion_reason")
                != request.exhaustion_reason
                or logical_failure["payload"].get("physical_contact_count")
                != request.physical_contact_count
                or logical_request["payload"].get("nebius_model_endpoint")
                != request.nebius_model_endpoint.model_dump(mode="json")
                or started is None
                or started["payload"].get("agent_role") != request.agent_role
                or not frontier_valid
                or request.retry_frontier_event_id not in by_id
                or not logical_request_index < logical_failure_index < request_index
                or not closure_valid
                or logical_request["payload"].get("member_id") != request.member_id
                or not _has_ancestor(
                    by_id,
                    request.logical_request_event_id,
                    request.logical_failure_event_id,
                )
                or not _has_ancestor(
                    {**by_id, event_id: event},
                    request.logical_failure_event_id,
                    event_id,
                )
            ):
                raise ValueError("availability request changed its logical request")
            # The cause is the logical failure's own; a Review joins the open one
            # with the same cause, exactly when one is open.
            failure_payload = logical_failure["payload"]
            if (
                request.fault_key
                != (
                    failure_payload.get("fault_key")
                    or availability_fault_key(request.exhaustion_reason)
                )
                or request.provider_statement
                != failure_payload.get("provider_statement")
                or request.joins_event_id
                != open_availability_review(timeline[:request_index], request.fault_key)
                # A contact that rejoined names the Review it rejoined.
                or any(
                    item.get("event_type") == "ModelPhysicalContactFailed"
                    and item["event_id"]
                    == failure_payload.get("last_physical_contact_event_id")
                    and item["payload"].get("open_review_event_id")
                    not in {None, request.joins_event_id}
                    for item in timeline[:request_index]
                )
            ):
                raise ValueError("availability request changed its cause or its join")
            requests.append((event_id, request))
            pending_ids.add(event_id)
        elif event.get("event_type") == "ModelAvailabilityReviewResponded":
            response = _fact(event, ModelAvailabilityReviewResponded)
            validate_review_answer_parentage(timeline, event)
            if (
                response.request_event_id not in pending_ids
                or response.request_event_id in responses
            ):
                raise ValueError(
                    "availability response does not target the pending request"
                )
            responses[response.request_event_id] = (event_id, response)
            pending_ids.remove(response.request_event_id)
        elif event.get("event_type") == "ModelAvailabilityStopped":
            rejected = _fact(event, ModelAvailabilityStopped)
            retained = responses.get(rejected.request_event_id)
            if (
                retained is None
                or rejected.response_event_id in stopped_answers
                or retained[0] != rejected.response_event_id
                or not isinstance(retained[1].response, StopModelAvailabilityResponse)
                or retained[1].response.reason != rejected.reason
                or event.get("parent_ids") != [rejected.response_event_id]
            ):
                raise ValueError("availability rejection changed researcher direction")
            stopped_answers.add(rejected.response_event_id)
            # The role's declaration says what a stop ends (ADR 0040): a role whose
            # domain settles its stopped members ends only that member (ADR 0039);
            # any other stop ends the Run's work.
            from bio_neb.agentic_journey.role_declaration import ROLES

            stopped = dict(requests)[rejected.request_event_id]
            if not ROLES[stopped.agent_role].settles_member_stops:
                stop_event_id = event_id

    validate_provider_continuations(timeline, requests, responses, checked_once)
    views: list[ModelAvailabilityReviewRequestView] = []
    for request_event_id, request in requests:
        if request.joined_before_sending:
            views.append(
                ModelAvailabilityReviewRequestView(
                    request_event_id=request_event_id,
                    attempt_id=request.attempt_id,
                    agent_role=request.agent_role,
                    member_id=request.member_id,
                    retry_frontier_event_id=request.retry_frontier_event_id,
                    nebius_model_endpoint=request.nebius_model_endpoint,
                    delays=(),
                    exhaustion_reason=request.exhaustion_reason,
                    pending=request_event_id in pending_ids,
                    fault_key=request.fault_key,
                    provider_statement=request.provider_statement,
                    joins_event_id=request.joins_event_id,
                )
            )
            continue
        request_index = next(
            index
            for index, item in enumerate(timeline)
            if item["event_id"] == request.logical_request_event_id
        )
        failure_index = next(
            index
            for index, item in enumerate(timeline)
            if item["event_id"] == request.logical_failure_event_id
        )
        if (
            validate_attempt_fact(timeline[request_index]) is None
            or validate_attempt_fact(timeline[failure_index]) is None
        ):
            raise ValueError("availability logical model evidence is invalid")
        window = timeline[request_index + 1 : failure_index]
        starts = [
            item
            for item in window
            if item.get("event_type") == "ModelPhysicalContactStarted"
            and item["payload"].get("attempt_id") == request.attempt_id
            and item["payload"].get("request_number") == request.request_number
        ]
        failures = [
            item
            for item in window
            if item.get("event_type") == "ModelPhysicalContactFailed"
            and item["payload"].get("attempt_id") == request.attempt_id
            and item["payload"].get("request_number") == request.request_number
        ]
        if (
            len(starts) != request.physical_contact_count
            or len(failures) != request.physical_contact_count
            or [item["payload"].get("contact_ordinal") for item in starts]
            != list(range(1, request.physical_contact_count + 1))
            or [item["payload"].get("contact_ordinal") for item in failures]
            != list(range(1, request.physical_contact_count + 1))
            or failures[-1]["payload"].get("policy_disposition")
            != "availability_review_requested"
            or any(
                start["payload"].get("logical_request_event_id")
                != request.logical_request_event_id
                or failure["payload"].get("logical_request_event_id")
                != request.logical_request_event_id
                or failure["payload"].get("contact_started_event_id")
                != start["event_id"]
                or timeline.index(start) >= timeline.index(failure)
                for start, failure in zip(starts, failures)  # noqa: B905 - Preserve existing zip truncation; contract-base migration adds no options.
            )
        ):
            raise ValueError("availability physical-contact grammar is invalid")
        scheduled = [
            item
            for item in window
            if item.get("event_type") == "ModelAvailabilityRetryScheduled"
            and item["payload"].get("logical_request_event_id")
            == request.logical_request_event_id
        ]
        completed = [
            item
            for item in window
            if item.get("event_type") == "ModelAvailabilityRetryDelayCompleted"
            and item["payload"].get("logical_request_event_id")
            == request.logical_request_event_id
        ]
        cancelled_delays = [
            item
            for item in window
            if item.get("event_type") == "ModelAvailabilityRetryDelayCancelled"
            and item["payload"].get("logical_request_event_id")
            == request.logical_request_event_id
        ]
        if any(
            validate_attempt_fact(item) is None
            for item in (*starts, *failures, *scheduled, *completed, *cancelled_delays)
        ):
            raise ValueError("availability physical model evidence is invalid")
        validate_model_contact_window(
            timeline[request_index : failure_index + 1],
            timeline[request_index],
            timeline[failure_index],
        )
        views.append(
            ModelAvailabilityReviewRequestView(
                request_event_id=request_event_id,
                attempt_id=request.attempt_id,
                agent_role=request.agent_role,
                member_id=request.member_id,
                retry_frontier_event_id=request.retry_frontier_event_id,
                logical_request_event_id=request.logical_request_event_id,
                logical_failure_event_id=request.logical_failure_event_id,
                request_number=request.request_number,
                nebius_model_endpoint=request.nebius_model_endpoint,
                contact_policy=ModelContactPolicy.model_validate(
                    timeline[request_index]["payload"].get("contact_policy", {})
                ),
                contacts=tuple(
                    ModelAvailabilityContactView(
                        ordinal=item["payload"]["contact_ordinal"],
                        status_code=item["payload"].get("http_status_code"),
                        failure_reason=item["payload"].get("failure_reason"),
                        completion_certainty=item["payload"]["completion_certainty"],
                        policy_disposition=item["payload"]["policy_disposition"],
                        retry_after_disposition=item["payload"].get(
                            "retry_after_disposition"
                        ),
                        retry_after_seconds=item["payload"].get("retry_after_seconds"),
                    )
                    for item in failures
                ),
                delays=tuple(
                    ModelAvailabilityDelayView(
                        seconds=item["payload"]["delay_seconds"],
                        source=item["payload"]["delay_source"],
                        completed=True,
                    )
                    for item in scheduled
                ),
                exhaustion_reason=request.exhaustion_reason,
                pending=request_event_id in pending_ids,
                fault_key=request.fault_key,
                provider_statement=request.provider_statement,
                joins_event_id=request.joins_event_id,
            )
        )
    receipts = tuple(
        ModelAvailabilityReviewResponseReceipt(
            request_event_id=request_id,
            response_event_id=response_id,
            response=response.response,
        )
        for request_id, _request in requests
        if (retained := responses.get(request_id)) is not None
        for response_id, response in (retained,)
    )
    # Each item of a Review is answered on its own.
    pending = next((item for item in views if item.pending), None)
    return ModelAvailabilityReviewProjection(
        requests=tuple(views),
        receipts=receipts,
        pending_request=pending,
        stop_event_id=stop_event_id,
    )


def evaluate_model_availability_review_response(
    projection: ModelAvailabilityReviewProjection,
    request_event_id: str,
    response: ModelAvailabilityReviewResponse,
) -> ModelAvailabilityReviewResponseReceipt | None:
    _uuid(request_event_id, "request_event_id")
    from bio_neb.junction import evaluate_scoped_answer

    _, retained = evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=lambda request, answer: None,
        refusal=lambda reason, identity: ModelAvailabilityReviewResponseRefused(
            f"model_availability_review_response_{reason}", identity
        ),
    )
    return retained


def model_availability_direction(
    journey: Journey,
    response_event_id: str,
    *,
    agent_role: AgentRole,
) -> ModelAvailabilityReviewRequestView:
    events = read_evidence(journey)
    pending_continuation(events, response_event_id, agent_role=agent_role)
    projection = project_model_availability_review(events)
    receipts = [
        receipt
        for receipt in projection.receipts
        if receipt.response_event_id == response_event_id
        and receipt.response.action in {"retry", "amend"}
    ]
    if len(receipts) != 1:
        raise ValueError("model availability retry is not exact")
    request = next(
        item
        for item in projection.requests
        if item.request_event_id == receipts[0].request_event_id
    )
    if request.agent_role != agent_role:
        raise ValueError("model availability retry changed Agent role")
    return request, receipts[0]


def model_availability_retry_request(journey, response_event_id, *, agent_role):
    request, receipt = model_availability_direction(
        journey, response_event_id, agent_role=agent_role
    )
    if receipt.response.action != "retry":
        raise ValueError("model availability retry requires exact retry direction")
    return request


def pending_model_work(journey, response_event_id, *, agent_role):
    """Load exact pending messages/policy after validating its one retry answer."""
    from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
    from bio_neb.agentic_journey.model_turn import read_messages_record

    request = model_availability_retry_request(
        journey, response_event_id, agent_role=agent_role
    )
    events = read_evidence(journey)
    logical = next(
        e for e in events if e["event_id"] == request.logical_request_event_id
    )
    started = next(
        e
        for e in events
        if e["event_type"] == "AttemptStarted"
        and e["payload"]["attempt_id"] == request.attempt_id
    )
    messages = read_messages_record(
        logical["payload"]["compatible_adapter_evidence"]["request"]["messages"]
    )
    return (
        logical,
        messages,
        AgentInvocationOverrides.model_validate(
            started["payload"]["invocation_overrides"]
        ),
    )


def prior_model_attempt_ids(events, attempt_id):
    """Trace only exact provider continuations, never ordinary clarification."""
    by_id = {e["event_id"]: e for e in events}
    starts = {
        e["payload"]["attempt_id"]: e
        for e in events
        if e["event_type"] == "AttemptStarted"
    }
    previous = []
    seen = {attempt_id}
    while (start := starts.get(attempt_id)) is not None:
        operational_answer = start["payload"].get(
            "continuation_operational_review_event_id"
        )
        if operational_answer is not None:
            answer = by_id[operational_answer]
            if answer["payload"]["response"]["action"] != "retry":
                break
            request = by_id[answer["payload"]["request_event_id"]]
            attempt_id = request["payload"]["attempt_id"]
            if attempt_id in seen:
                raise ValueError("operational retry repeats an Attempt")
            previous.append(attempt_id)
            seen.add(attempt_id)
            continue
        request_id = start["payload"].get("continuation_request_event_id")
        if request_id is None:
            answer_id = start["payload"].get("continuation_effort_review_event_id")
            if answer_id is None:
                break
            answer = by_id.get(answer_id)
            request = (
                by_id.get(answer["payload"]["request_event_id"]) if answer else None
            )
            if (
                answer is None
                or answer["event_type"] != "AgentEffortReviewResponded"
                or request is None
                or request["event_type"] != "AgentEffortReviewRequested"
            ):
                raise ValueError("effort continuation lost its exact request")
            attempt_id = request["payload"]["attempt_id"]
            if attempt_id in seen:
                raise ValueError("effort continuation repeats an Attempt")
            previous.append(attempt_id)
            seen.add(attempt_id)
            continue
        request = by_id.get(request_id)
        if request is None or request["event_type"] != "ModelRequestRecorded":
            raise ValueError("provider continuation lost its pending request")
        attempt_id = request["payload"]["attempt_id"]
        if attempt_id in seen:
            raise ValueError("provider continuation repeats an Attempt")
        previous.append(attempt_id)
        seen.add(attempt_id)
    return tuple(reversed(previous))


__all__ = [
    name
    for name in globals()
    if name.startswith("ModelAvailability")
    or name.startswith("RejectModel")
    or name.startswith("RetryModel")
]
