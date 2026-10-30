"""Incremental Journey observation around Pydantic AI model requests."""

from __future__ import annotations

import asyncio
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import count
from typing import Any, Awaitable, Callable, Mapping, Sequence

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import ToolDefinition
from pydantic_core import PydanticSerializationError

from bio_neb.agentic_journey.agent_work_review import AgentContextExhausted
from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    ModelAvailabilityRetryDelayCancelled,
    ModelAvailabilityRetryDelayCompleted,
    ModelAvailabilityRetryScheduled,
    ModelPhysicalContactCancelled,
    ModelPhysicalContactFailed,
    ModelPhysicalContactStarted,
    ModelRequestFailed,
    ModelRequestRecorded,
    ModelRequestRejected,
    ModelRequestStalled,
    ModelResponseRecorded,
    NebiusModelEndpointEvidence,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewPending,
)
from bio_neb.agentic_journey.model_settings import ModelSettingsSelection
from bio_neb.agentic_journey.model_transport import (
    TRANSPORT_MEDIA_TYPE,
    ModelTransportRequestRecorded,
    ModelTransportResponseRecorded,
    PreparedModelTransportRequest,
    ReceivedModelTransportResponse,
    observe_model_transport_requests,
    observe_model_transport_responses,
)
from bio_neb.agentic_journey.model_turn import messages_record
from bio_neb.failure import (
    Invariant,
    ModelContactPolicy,
    ModelContactStalled,
    ModelRequestRefusedShown,
    ModelResponseMalformed,
    contact_fault_key,
    model_contact_failure,
    model_contact_verdict,
    model_retry_after,
    provider_statement,
)
from bio_neb.recording import ContactScope, physical_contact

REQUEST_STAGE: ContextVar[str | None] = ContextVar("request_stage", default=None)
"""The stage of the request being sent, set by the Agent assembly around it."""

_TOOL_DEFINITIONS = TypeAdapter(list[ToolDefinition])
_INSTRUCTION_PARTS = TypeAdapter(list[InstructionPart] | None)


def _contact_failure(error: Exception):
    """Translate framework errors at the Agent edge; the failure package reads the rest."""
    from openai import APIConnectionError
    from pydantic_ai.exceptions import ModelAPIError

    if isinstance(error, ModelHTTPError):
        status = error.status_code
        lawful = isinstance(status, int) and not isinstance(status, bool)
        return "http", status if lawful and 100 <= status <= 599 else None
    if isinstance(error, ModelAPIError) and isinstance(
        error.__cause__, APIConnectionError
    ):
        error = error.__cause__
    return model_contact_failure(error), None


@dataclass(frozen=True)
class ReceivedReply:
    """A retained reply as admission reads it, with its logical request's facts.

    ``resends`` counts the contacts of the request whose reply never arrived (a
    gateway or request timeout, or a dropped connection whose completion is
    unknown) and that were sent again before this reply arrived; the failure
    package's verdict marks each (``ContactVerdict.resend``).
    """

    response_event_id: str
    request_number: int
    requested_max_tokens: int | None
    resends: int


class JourneyObservedModel(WrapperModel):
    """Retain one event before contact and one after every model response."""

    def __init__(
        self,
        wrapped: Any,
        *,
        parent_activity: Activity,
        attempt_id: str,
        failure_source: AttemptFailureSource,
        nebius_model_endpoint: NebiusModelEndpointEvidence | Mapping[str, Any],
        settings_selection: ModelSettingsSelection | None = None,
        agent_role: str = "planning_agent",
        member_id=None,
        structured_output_session: Any | None = None,
        availability_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        contact_policy: ModelContactPolicy,
        continuation_request: Mapping[str, Any] | None = None,
        availability_jitter: Callable[[], float] | None = None,
        availability_clock: Callable[[], float] = time.time,
        contact_clock: Callable[[], float] = time.monotonic,
        before_request: Callable[[], None] | None = None,
        released: bool = False,
        run_evidence: Callable[[], Sequence[Mapping[str, Any]]] = tuple,
    ):
        self._member_id = member_id or (lambda: None)
        super().__init__(wrapped)
        self._parent_activity = parent_activity
        self._attempt_id = attempt_id
        self._failure_source = failure_source
        self._nebius_model_endpoint = NebiusModelEndpointEvidence.model_validate(
            nebius_model_endpoint
        )
        self._agent_role = agent_role
        # Released by an availability answer: its first request passes the hold.
        self._released = released
        # The Run's checked evidence, where open keyed Reviews are read;
        # none, no Review is open.
        self._run_evidence = run_evidence
        self._settings_selection = settings_selection
        self._structured_output_session = structured_output_session
        self._availability_sleep = availability_sleep
        # The failure module resolved it for the Attempt's role; no fallback.
        self._contact_policy = contact_policy
        self._continuation_request = continuation_request
        self._availability_jitter = availability_jitter or random.random
        self._availability_clock = availability_clock
        self._contact_clock = contact_clock
        self._before_request = before_request
        self.request_count = 0
        self.last_reply: ReceivedReply | None = None
        self._active_contact_started_event_id: str | None = None
        self._active_contact_sent_at = 0.0
        # The stall the next request follows and notes.
        self._stall_event_id: str | None = None
        # The refusal the next request follows and carries, and its fault
        # key.
        self._rejection: tuple[str, str] | None = None
        # The current contact's retained answer body: its event and bytes.
        self._contact_response: tuple[Mapping[str, Any], bytes] | None = None

    async def _physical_contact(
        self,
        activity: Activity,
        *,
        messages: Any,
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
        request_number: int,
        contact_ordinal: int,
        logical_request_event_id: str,
        retry_delay_completed_event_id: str | None,
    ) -> Any:
        contact_started = activity.record_contact(
            ModelPhysicalContactStarted(
                attempt_id=self._attempt_id,
                request_number=request_number,
                logical_request_event_id=logical_request_event_id,
                contact_ordinal=contact_ordinal,
                transport_evidence_expected=getattr(
                    self.wrapped, "records_model_transport_requests", False
                ),
                response_body_expected=getattr(
                    self.wrapped, "records_model_transport_responses", False
                ),
                retry_delay_completed_event_id=retry_delay_completed_event_id,
            )
        )
        self._active_contact_started_event_id = contact_started["event_id"]
        self._active_contact_sent_at = self._contact_clock()
        self._contact_response = None

        def retain_transport(prepared: PreparedModelTransportRequest) -> None:
            activity.record_with_artifact(
                ModelTransportRequestRecorded.from_prepared(
                    prepared,
                    attempt_id=self._attempt_id,
                    request_number=request_number,
                    logical_request_event_id=logical_request_event_id,
                    contact_started_event_id=contact_started["event_id"],
                    contact_ordinal=contact_ordinal,
                ),
                prepared.request_body,
                media_type=TRANSPORT_MEDIA_TYPE,
            )

        received: list[ReceivedModelTransportResponse] = []

        def retain_responses() -> None:
            # Retained after the send, never inside it: the SDK wraps anything
            # raised in its send as a connection fault, and a failure to record
            # is ours, not the provider's (ADR 0031).
            while received:
                body = received.pop(0)
                retained = activity.record_with_artifact(
                    ModelTransportResponseRecorded.from_received(
                        body,
                        attempt_id=self._attempt_id,
                        request_number=request_number,
                        logical_request_event_id=logical_request_event_id,
                        contact_started_event_id=contact_started["event_id"],
                        contact_ordinal=contact_ordinal,
                    ),
                    body.body,
                    media_type=body.media_type,
                )
                self._contact_response = (retained, body.body)

        with physical_contact(
            ContactScope(
                attempt_id=self._attempt_id,
                request_id=logical_request_event_id,
                evidence_event_id=contact_started["event_id"],
            ),
            "model.request",
        ):
            with (
                observe_model_transport_requests(retain_transport),
                observe_model_transport_responses(received.append),
            ):
                pending = asyncio.create_task(
                    self.wrapped.request(
                        messages, model_settings, model_request_parameters
                    )
                )
            try:
                # asyncio.wait, not a bare await: cancelling this caller must not
                # cancel the request implicitly; the handler below cancels it once.
                # No reply within the stall limit cuts the send off.
                stall_seconds = self._contact_policy.stall_seconds
                done, _ = await asyncio.wait({pending}, timeout=stall_seconds)
                if not done:
                    pending.cancel()
                    try:
                        await pending
                    except BaseException:
                        pass
                    # Whatever arrived before the cut-off is kept with the contact.
                    retain_responses()
                    raise ModelContactStalled(stall_seconds)
                try:
                    response = await pending
                finally:
                    retain_responses()
                return response, contact_started["event_id"]
            except asyncio.CancelledError:
                pending.cancel()
                try:
                    await pending
                except BaseException:
                    pass
                retain_responses()
                cancelled = activity.record_contact(
                    ModelPhysicalContactCancelled(
                        attempt_id=self._attempt_id,
                        request_number=request_number,
                        logical_request_event_id=logical_request_event_id,
                        contact_started_event_id=contact_started["event_id"],
                        contact_ordinal=contact_ordinal,
                    )
                )
                activity.cancelled(cancelled)
                raise

    def _join_open_review(self) -> None:
        """While one item's model endpoint Review is open, another Investigation
        item's next model call joins it and sends nothing (orchestrator answer
        under the owner's grant, #477 G4.1): the endpoint sees each cause once."""
        from bio_neb.agentic_journey.model_availability import (
            open_availability_review,
        )

        if self._agent_role != "investigation_planner" or (
            self._released and self.request_count == 0
        ):
            return
        events = self._run_evidence()
        lead_id = open_availability_review(
            events, endpoint=self._nebius_model_endpoint.model_dump(mode="json")
        )
        if lead_id is None:
            return
        lead = next(event for event in events if event["event_id"] == lead_id)
        raise ModelAvailabilityReviewPending.joining(lead)

    # Keep the logical request, its physical window and response admission under
    # one recording owner. Failure classification and retry decisions live in the failure package.
    async def request(  # noqa: C901, PLR0915 - One ordered recording lifecycle, with shared policy decisions.
        self,
        messages,
        model_settings,
        model_request_parameters: ModelRequestParameters,
    ):
        if self._before_request is not None:
            self._before_request()
        self._join_open_review()
        self.request_count += 1
        request_number = self.request_count
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings, model_request_parameters
        )
        request_fact = ModelRequestRecorded(
            member_id=self._member_id(),
            attempt_id=self._attempt_id,
            request_number=request_number,
            nebius_model_endpoint=self._nebius_model_endpoint,
            settings_selection=self._settings_selection,
            contact_policy=self._contact_policy,
            continuation_request_event_id=self._continuation_request["event_id"]
            if request_number == 1 and self._continuation_request is not None
            else None,
            stage=REQUEST_STAGE.get(),
            after_stall_event_id=self._stall_event_id,
            after_rejection_event_id=None
            if self._rejection is None
            else self._rejection[0],
            after_rejection_fault_key=None
            if self._rejection is None
            else self._rejection[1],
            compatible_adapter_evidence={
                "model_name": self.model_name,
                "model_system": self.system,
                "request": {
                    "messages": messages_record(messages),
                    "instruction_parts": _INSTRUCTION_PARTS.dump_python(
                        prepared_parameters.instruction_parts, mode="json"
                    ),
                    "model_settings": {
                        key: value
                        for key, value in (prepared_settings or {}).items()
                        if value is not None
                    },
                    "function_tools": _TOOL_DEFINITIONS.dump_python(
                        prepared_parameters.function_tools, mode="json"
                    ),
                    "output_tools": _TOOL_DEFINITIONS.dump_python(
                        prepared_parameters.output_tools, mode="json"
                    ),
                    "allow_text_output": prepared_parameters.allow_text_output,
                    "output_mode": prepared_parameters.output_mode,
                },
            },
        )
        if request_number == 1 and self._continuation_request is not None:
            prior = self._continuation_request["payload"]
            if (
                request_fact.compatible_adapter_evidence.request.model_dump(mode="json")
                != prior["compatible_adapter_evidence"]["request"]
            ):
                raise Invariant(
                    "pending_request_changed",
                    "provider retry changed its exact pending request",
                )
        activity = self._parent_activity.start_child(
            adapter="pydantic_ai_model",
            stage="model_contact",
        )
        request_event = activity.record(request_fact)
        after_stall, self._stall_event_id = self._stall_event_id is not None, None
        # The refusal this request carries: the same refusal of it asks (G1.5).
        shown_key = None if self._rejection is None else self._rejection[1]
        self._rejection = None
        response = None
        retry_delay_completed_event_id = None
        waited = 0.0
        malformed_responses = 0
        resends = 0
        contact_ordinals = count(1)
        for contact_ordinal in contact_ordinals:
            try:
                response, contact_started_event_id = await self._physical_contact(
                    activity,
                    messages=messages,
                    model_settings=model_settings,
                    model_request_parameters=model_request_parameters,
                    request_number=request_number,
                    contact_ordinal=contact_ordinal,
                    logical_request_event_id=request_event["event_id"],
                    retry_delay_completed_event_id=retry_delay_completed_event_id,
                )
                try:
                    response_fact = ModelResponseRecorded(
                        attempt_id=self._attempt_id,
                        request_number=request_number,
                        physical_contact_ordinal=contact_ordinal,
                        physical_contact_started_event_id=contact_started_event_id,
                        completion_certainty="response_received",
                        policy_disposition="response_admitted",
                        nebius_model_endpoint=self._nebius_model_endpoint,
                        compatible_adapter_evidence={
                            "model_name": self.model_name,
                            "model_system": self.system,
                            "response": {"message": messages_record([response])[0]},
                        },
                    )
                except (ValidationError, PydanticSerializationError) as error:
                    raise ModelResponseMalformed() from error
            except asyncio.CancelledError:
                raise
            except Exception as error:
                contact_started_event_id = self._active_contact_started_event_id
                if contact_started_event_id is None:
                    raise Invariant(
                        "contact_start_unretained",
                        "physical contact start was not retained",
                    ) from error
                contact_seconds = max(
                    0.0, self._contact_clock() - self._active_contact_sent_at
                )
                failure, http_status_code = _contact_failure(error)
                error_type = type(error).__name__
                retry_after_disposition = None
                retry_after_seconds = None
                rate_limit_headers = None
                if failure == "malformed":
                    malformed_responses += 1
                    error_type = (
                        type(error.__cause__).__name__
                        if error.__cause__ is not None
                        else type(error).__name__
                    )
                if isinstance(error, ModelHTTPError):
                    error_type = "ModelHTTPError"
                    (
                        retry_after_disposition,
                        retry_after_seconds,
                    ) = model_retry_after(
                        getattr(error, "headers", None), now=self._availability_clock()
                    )
                    rate_limit_headers = _rate_limit_headers(error.headers)
                # The endpoint's answer body is read, never classified. Every
                # failed contact has a cause: the status and the exact body of a
                # retained answer, else what it lacks (r2 SP4); words decide only
                # whether the model can be shown it.
                response_event_id = statement = digest = None
                if failure == "http" and self._contact_response is not None:
                    answered, body = self._contact_response
                    response_event_id = answered["event_id"]
                    statement = provider_statement(body)
                    digest = answered["payload"]["response_body_sha256"]
                fault_key = contact_fault_key(failure, http_status_code, digest)
                # A cause whose Review is still open recurs: rejoin it at once.
                open_review = None
                if fault_key is not None:
                    from bio_neb.agentic_journey.model_availability import (
                        open_availability_review,
                    )

                    open_review = open_availability_review(
                        self._run_evidence(), fault_key
                    )
                retry_jitter = self._availability_jitter()
                verdict = model_contact_verdict(
                    self._contact_policy,
                    failure=failure,
                    contact=contact_ordinal,
                    waited=waited,
                    jitter=retry_jitter,
                    http_status_code=http_status_code,
                    retry_after_disposition=retry_after_disposition,
                    retry_after_seconds=retry_after_seconds,
                    malformed_responses=malformed_responses,
                    after_stall=after_stall,
                    rejection_key=fault_key
                    if response_event_id is not None and statement is not None
                    else None,
                    shown_key=shown_key,
                    rejoins=open_review is not None,
                )
                delay_seconds = verdict.delay_seconds
                delay_source = verdict.delay_source
                exhaustion_reason = verdict.exhaustion_reason
                failure_reason = verdict.failure_reason
                contact_failed = activity.record_contact(
                    ModelPhysicalContactFailed(
                        attempt_id=self._attempt_id,
                        request_number=request_number,
                        logical_request_event_id=request_event["event_id"],
                        contact_started_event_id=contact_started_event_id,
                        contact_ordinal=contact_ordinal,
                        failure_class=verdict.failure_class,
                        retry_jitter=retry_jitter,
                        error_type=error_type,
                        contact_seconds=contact_seconds,
                        http_status_code=http_status_code,
                        failure_reason=failure_reason,
                        completion_certainty=verdict.completion_certainty,
                        policy_disposition=verdict.policy_disposition,
                        retry_after_disposition=retry_after_disposition,
                        retry_after_seconds=retry_after_seconds,
                        rate_limit_headers=rate_limit_headers,
                        response_event_id=response_event_id,
                        fault_key=fault_key,
                        provider_statement=statement,
                        open_review_event_id=open_review,
                    )
                )
                resends += verdict.resend
                if verdict.policy_disposition == "rejection_shown":
                    # The request ends; the next carries the provider's own words
                    # and is one more failed turn.
                    rejected = activity.record(
                        ModelRequestRejected(
                            attempt_id=self._attempt_id,
                            request_number=request_number,
                            physical_contact_count=contact_ordinal,
                            last_physical_contact_event_id=contact_failed["event_id"],
                            response_event_id=response_event_id,
                            http_status_code=http_status_code,
                            provider_statement=statement,
                            fault_key=fault_key,
                            nebius_model_endpoint=self._nebius_model_endpoint,
                        )
                    )
                    activity.succeeded(rejected)
                    self._rejection = (rejected["event_id"], fault_key)
                    raise ModelRequestRefusedShown(
                        http_status_code, statement
                    ) from error
                if verdict.policy_disposition == "stall_noted":
                    # The request ends without a reply; the next carries the
                    # note and names this stall.
                    stalled = activity.record(
                        ModelRequestStalled(
                            attempt_id=self._attempt_id,
                            request_number=request_number,
                            physical_contact_count=contact_ordinal,
                            last_physical_contact_event_id=contact_failed["event_id"],
                            stall_seconds=self._contact_policy.stall_seconds,
                            nebius_model_endpoint=self._nebius_model_endpoint,
                        )
                    )
                    # The request ended as the window decided: its note continues it.
                    activity.succeeded(stalled)
                    self._stall_event_id = stalled["event_id"]
                    raise
                if delay_seconds is not None:
                    assert delay_source is not None
                    retry_scheduled = activity.record_contact(
                        ModelAvailabilityRetryScheduled(
                            attempt_id=self._attempt_id,
                            request_number=request_number,
                            logical_request_event_id=request_event["event_id"],
                            failed_contact_event_id=contact_failed["event_id"],
                            failed_contact_ordinal=contact_ordinal,
                            next_contact_ordinal=contact_ordinal + 1,
                            delay_seconds=delay_seconds,
                            delay_source=delay_source,
                        )
                    )
                    try:
                        await self._availability_sleep(delay_seconds)
                    except asyncio.CancelledError:
                        cancelled = activity.record_contact(
                            ModelAvailabilityRetryDelayCancelled(
                                attempt_id=self._attempt_id,
                                request_number=request_number,
                                logical_request_event_id=request_event["event_id"],
                                retry_scheduled_event_id=retry_scheduled["event_id"],
                                failed_contact_ordinal=contact_ordinal,
                                next_contact_ordinal=contact_ordinal + 1,
                                delay_seconds=delay_seconds,
                            )
                        )
                        activity.cancelled(cancelled)
                        raise
                    delay_completed = activity.record_contact(
                        ModelAvailabilityRetryDelayCompleted(
                            attempt_id=self._attempt_id,
                            request_number=request_number,
                            logical_request_event_id=request_event["event_id"],
                            retry_scheduled_event_id=retry_scheduled["event_id"],
                            failed_contact_ordinal=contact_ordinal,
                            next_contact_ordinal=contact_ordinal + 1,
                            delay_seconds=delay_seconds,
                        )
                    )
                    waited += delay_seconds
                    retry_delay_completed_event_id = delay_completed["event_id"]
                    continue
                failed = activity.record(
                    ModelRequestFailed(
                        attempt_id=self._attempt_id,
                        request_number=request_number,
                        physical_contact_count=contact_ordinal,
                        last_physical_contact_event_id=contact_failed["event_id"],
                        availability_exhaustion_reason=exhaustion_reason,
                        nebius_model_endpoint=self._nebius_model_endpoint,
                        error_type=error_type,
                        http_status_code=http_status_code,
                        failure_reason=failure_reason,
                        response_event_id=response_event_id,
                        provider_statement=statement,
                        fault_key=fault_key,
                    ),
                )
                self._failure_source.capture(failed)
                activity.failed(failed)
                if exhaustion_reason is not None:
                    raise ModelAvailabilityReviewPending(
                        request_event_id=request_event["event_id"],
                        failure_event_id=failed["event_id"],
                        request_number=request_number,
                        physical_contact_count=contact_ordinal,
                        exhaustion_reason=exhaustion_reason,
                        nebius_model_endpoint=self._nebius_model_endpoint,
                        fault_key=fault_key,
                        provider_statement=statement,
                    ) from error
                if failure == "not_sent":
                    # Resending would overflow again: stop for the Agent work review.
                    raise AgentContextExhausted(
                        str(error), source_event_id=failed["event_id"]
                    ) from error
                raise
            else:
                break
        assert response is not None
        received = activity.record(response_fact)
        self.last_reply = ReceivedReply(
            response_event_id=received["event_id"],
            request_number=request_number,
            requested_max_tokens=(prepared_settings or {}).get("max_tokens"),
            resends=resends,
        )
        activity.succeeded(received)
        return response

    def observe_outputs(
        self, response_event_id: str, routed: Any, output_tools: Sequence[Any]
    ) -> None:
        """Hand a reply's output calls, as routing kept them, to its session.

        Admission routes every reply first (ADR 0044): an output call's copy has
        collapsed and a cut-off reply's partial call is gone. ``output_tools``
        are the output Tools the request showed, as the Model turn reads them.
        """
        if self._structured_output_session is not None:
            self._structured_output_session.observe_response(
                response_event_id, routed, output_tools
            )

    def output_refusal(self, arguments: Any):
        """The role's faults for schema-refused output arguments, from its
        session, or None to leave the schema's (#448 D1)."""
        if self._structured_output_session is None:
            return None
        return self._structured_output_session.arguments_refusal(arguments)

    def output_repair(self):
        """The current member's output repair count after its latest refusal, which
        a correction sent after the record reads (#373 E3); None with none."""
        if self._structured_output_session is None:
            return None
        return self._structured_output_session.recorded_repair()


def _rate_limit_headers(headers) -> dict[str, str] | None:
    """The provider's ``x-ratelimit-*`` headers, lower-cased; ``None`` with none."""
    selected = {
        str(name).lower(): str(value)
        for name, value in (headers or {}).items()
        if str(name).lower().startswith("x-ratelimit-")
    }
    return dict(sorted(selected.items())) or None
