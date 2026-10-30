"""Typed Journey facts for one registered-input inspection Attempt."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationPolicy,
    ContinuationPolicy,
)
from bio_neb.agentic_journey.model_settings import ModelSettingsSelection
from bio_neb.agentic_journey.model_transport import (
    ModelTransportRequestRecorded,
    ModelTransportResponseRecorded,
)
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.agentic_journey.runtime_config import NebiusModelEndpointIdentity
from bio_neb.failure import (
    FailureClass,
    ModelContactPolicy,
    model_contact_reading,
)


@dataclass
class AttemptFailureSource:
    """Transient handoff of the exact typed leaf recorded by an Attempt component."""

    event_id: str | None = None
    event_type: str | None = None

    def capture(self, event: Mapping[str, Any]) -> None:
        self.event_id = event["event_id"]
        self.event_type = event["event_type"]


class _AttemptFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ModelEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


NebiusModelEndpointEvidence = NebiusModelEndpointIdentity


class CompatibleAdapterRequest(_ModelEvidence):
    """Exact Pydantic-AI-normalized request fields, not raw HTTP evidence."""

    messages: list[dict[str, JsonValue]]
    instruction_parts: list[dict[str, JsonValue]] | None
    model_settings: dict[str, JsonValue]
    function_tools: list[dict[str, JsonValue]]
    output_tools: list[dict[str, JsonValue]]
    allow_text_output: StrictBool
    output_mode: StrictStr


class CompatibleAdapterRequestEvidence(_ModelEvidence):
    model_name: StrictStr
    model_system: StrictStr
    request: CompatibleAdapterRequest


class CompatibleAdapterResponse(_ModelEvidence):
    """Exact Pydantic-AI-normalized response message, not raw HTTP evidence."""

    message: dict[str, JsonValue]


class CompatibleAdapterResponseEvidence(_ModelEvidence):
    model_name: StrictStr
    model_system: StrictStr
    response: CompatibleAdapterResponse


class AttemptStarted(_AttemptFact):
    event_type: Literal["AttemptStarted"] = "AttemptStarted"
    attempt_id: StrictStr
    agent_role: AgentRole = "planning_agent"
    input_set_id: StrictStr
    instruction_identity: StrictStr | None = None
    continuation_policy: ContinuationPolicy | None = None
    continuation_policy_id: StrictStr | None = None
    effective_invocation_policy: AgentInvocationPolicy | None = None
    invocation_policy_id: StrictStr | None = None
    invocation_overrides: dict[str, JsonValue] | None = None
    continuation_request_event_id: StrictStr | None = None
    continuation_effort_review_event_id: StrictStr | None = None
    continuation_availability_review_event_id: StrictStr | None = None
    continuation_work_review_event_id: StrictStr | None = None
    continuation_operational_review_event_id: StrictStr | None = None
    # The one collection member this Attempt works on. Its stops hold only
    # that member's work; with none, they hold the whole role. Additive:
    # absent when unset.
    member_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The researcher's request this Attempt's work answers: stops of Attempts
    # started before it do not hold it. Additive: absent when unset.
    holds_from_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("attempt_id")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("attempt_id must be canonical")
        return value


class ModelRequestRecorded(_AttemptFact):
    event_type: Literal["ModelRequestRecorded"] = "ModelRequestRecorded"
    attempt_id: StrictStr
    member_id: StrictStr | None = None
    request_number: StrictInt = Field(gt=0)
    nebius_model_endpoint: NebiusModelEndpointEvidence
    compatible_adapter_evidence: CompatibleAdapterRequestEvidence
    settings_selection: ModelSettingsSelection | None = None
    contact_policy: ModelContactPolicy = Field(default_factory=ModelContactPolicy)
    continuation_request_event_id: StrictStr | None = None
    # The registry stage the request was sent in; its settings bind to its entry.
    stage: StrictStr | None = None
    # The ModelRequestStalled this request follows: it carries the stall note, and
    # a stall of its own asks the availability Review.
    after_stall_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The ModelRequestRejected this request follows: it carries the provider's
    # words, and the same refusal of its own asks the Review.
    after_rejection_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # That refusal's fault key: the same key on this request's own refusal asks.
    after_rejection_fault_key: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def compatible_model_matches_selection(self) -> "ModelRequestRecorded":
        if (
            self.settings_selection is not None
            and self.settings_selection.model != self.nebius_model_endpoint.model
        ):
            raise ValueError("request settings selection belongs to another model")
        if (
            self.compatible_adapter_evidence.model_system == "openai"
            and self.compatible_adapter_evidence.model_name
            != self.nebius_model_endpoint.model
        ):
            raise ValueError("compatible model name changed the Nebius selection")
        return self


# not_sent: the Adapter refused the request before any bytes left the process.
ModelContactCompletion = Literal["response_received", "completion_unknown", "not_sent"]
ModelFailureReason = Literal[
    "authentication_rejected",
    "authorization_rejected",
    "request_timeout",
    "request_too_large",
    "rate_limited",
    "request_rejected",
    "provider_failure",
    "http_failure",
    "non_http_failure",
    "malformed_response",
    "context_window_exhausted",
    "contact_stalled",
]
ModelRetryAfterDisposition = Literal[
    "valid",
    "missing",
    "unusable",
    "over_cap",
]


class ModelPhysicalContactStarted(_AttemptFact):
    event_type: Literal["ModelPhysicalContactStarted"] = "ModelPhysicalContactStarted"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    contact_ordinal: StrictInt = Field(gt=0)
    transport_evidence_expected: StrictBool = Field(
        default=False, exclude_if=lambda value: not value
    )
    # The contact keeps the raw body of any HTTP response it receives.
    response_body_expected: StrictBool = Field(
        default=False, exclude_if=lambda value: not value
    )
    retry_delay_completed_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    completion_certainty: Literal["completion_unknown"] = "completion_unknown"
    policy_disposition: Literal["pending"] = "pending"

    @model_validator(mode="after")
    def retry_source_matches_ordinal(self) -> "ModelPhysicalContactStarted":
        if (self.contact_ordinal > 1) != (
            self.retry_delay_completed_event_id is not None
        ):
            raise ValueError("second physical contact requires its completed delay")
        return self


class ModelPhysicalContactFailed(_AttemptFact):
    event_type: Literal["ModelPhysicalContactFailed"] = "ModelPhysicalContactFailed"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    contact_started_event_id: StrictStr
    contact_ordinal: StrictInt = Field(gt=0)
    failure_class: FailureClass = "transient"
    retry_jitter: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    error_type: StrictStr
    # The contact's own duration on a monotonic clock, from its send to its
    # failure: a stall's cut-off and a fast 504 read differently.
    contact_seconds: float = Field(ge=0, allow_inf_nan=False)
    http_status_code: StrictInt | None = Field(
        default=None, ge=100, le=599, exclude_if=lambda value: value is None
    )
    failure_reason: ModelFailureReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The provider's ``x-ratelimit-*`` response headers, lower-cased, when an
    # HTTP failure carried any; no other header is retained.
    rate_limit_headers: dict[StrictStr, StrictStr] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    completion_certainty: ModelContactCompletion
    policy_disposition: Literal[
        "automatic_retry_scheduled",
        "stall_noted",
        "rejection_shown",
        "availability_review_requested",
        "terminal_failure",
    ]
    retry_after_disposition: ModelRetryAfterDisposition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retry_after_seconds: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, exclude_if=lambda value: value is None
    )
    # The contact's retained answer body (ModelTransportResponseRecorded), when an
    # HTTP failure carried one: every model endpoint error points to it.
    response_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The failure module's key for that body's status and bytes
    # (``failure.model_fault_key``), and the provider's words in it, if any.
    fault_key: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    provider_statement: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The open Review with that key when the contact failed: the cause recurs,
    # so the item rejoins it instead of being shown the words again.
    open_review_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def failure(self) -> str:
        """How the contact failed, as the failure package's classifier reads it."""
        if self.error_type == "ModelHTTPError":
            return "http"
        if self.error_type == "ModelContactStalled":
            return "stalled"
        if self.completion_certainty == "response_received":
            return "malformed"
        if self.completion_certainty == "not_sent":
            return "not_sent"
        return "connection" if self.failure_class == "transient" else "unexpected"

    @model_validator(mode="after")
    def coherent_safe_failure(self) -> "ModelPhysicalContactFailed":
        if self.error_type != "ModelHTTPError" and self.http_status_code is not None:
            raise ValueError("a non-HTTP contact cannot carry an HTTP status")
        if (
            self.failure_class,
            self.failure_reason,
            self.completion_certainty,
        ) != model_contact_reading(self.failure, self.http_status_code):
            raise ValueError("physical contact classification is incoherent")
        if (
            self.policy_disposition not in _LAWFUL_DISPOSITIONS[self.failure_class]
            or (self.policy_disposition == "stall_noted" and self.failure != "stalled")
            or (
                self.policy_disposition == "rejection_shown"
                and self.response_event_id is None
            )
        ):
            raise ValueError("model failure disposition exceeds its class")
        if (self.retry_after_disposition == "valid") != (
            self.retry_after_seconds is not None
        ):
            raise ValueError("bounded Retry-After evidence is incoherent")
        if self.error_type == "ModelHTTPError":
            if self.retry_after_disposition is None:
                raise ValueError("HTTP failure lacks safe Retry-After disposition")
        elif self.retry_after_disposition is not None:
            raise ValueError("non-HTTP failure cannot claim Retry-After evidence")
        return self


# What each class of failure may lead to: only a transient failure retries or
# ends its request with the stall note; a refusal asks; our fault or an
# unsendable request is terminal.
_LAWFUL_DISPOSITIONS = {
    "transient": {
        "automatic_retry_scheduled",
        "stall_noted",
        "rejection_shown",
        "availability_review_requested",
    },
    "rejected": {"rejection_shown", "availability_review_requested"},
    "semantic": {"automatic_retry_scheduled", "availability_review_requested"},
    "exhausted": {"terminal_failure"},
    "invariant": {"terminal_failure"},
}


class ModelPhysicalContactCancelled(_AttemptFact):
    event_type: Literal["ModelPhysicalContactCancelled"] = (
        "ModelPhysicalContactCancelled"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    contact_started_event_id: StrictStr
    contact_ordinal: StrictInt = Field(gt=0)
    completion_certainty: Literal["completion_unknown"] = "completion_unknown"
    policy_disposition: Literal["cancelled"] = "cancelled"


class ModelAvailabilityRetryScheduled(_AttemptFact):
    event_type: Literal["ModelAvailabilityRetryScheduled"] = (
        "ModelAvailabilityRetryScheduled"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    failed_contact_event_id: StrictStr
    failed_contact_ordinal: StrictInt = Field(default=1, gt=0)
    next_contact_ordinal: StrictInt = Field(default=2, gt=1)
    delay_seconds: float = Field(ge=0, allow_inf_nan=False)
    delay_source: Literal[
        "retry_after",
        "exponential_jitter",
        "malformed_response",
    ]


class ModelAvailabilityRetryDelayCompleted(_AttemptFact):
    event_type: Literal["ModelAvailabilityRetryDelayCompleted"] = (
        "ModelAvailabilityRetryDelayCompleted"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    retry_scheduled_event_id: StrictStr
    failed_contact_ordinal: StrictInt = Field(default=1, gt=0)
    next_contact_ordinal: StrictInt = Field(default=2, gt=1)
    delay_seconds: float = Field(ge=0, allow_inf_nan=False)


class ModelAvailabilityRetryDelayCancelled(_AttemptFact):
    event_type: Literal["ModelAvailabilityRetryDelayCancelled"] = (
        "ModelAvailabilityRetryDelayCancelled"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    retry_scheduled_event_id: StrictStr
    failed_contact_ordinal: StrictInt = Field(default=1, gt=0)
    next_contact_ordinal: StrictInt = Field(default=2, gt=1)
    delay_seconds: float = Field(ge=0, allow_inf_nan=False)


class ModelTurnCancelled(_AttemptFact):
    """An interrupt between turns: every model request has its outcome, none is open."""

    event_type: Literal["ModelTurnCancelled"] = "ModelTurnCancelled"
    attempt_id: StrictStr
    model_request_count: StrictInt = Field(ge=0)


class ModelResponseRecorded(_AttemptFact):
    event_type: Literal["ModelResponseRecorded"] = "ModelResponseRecorded"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    physical_contact_ordinal: StrictInt = Field(gt=0)
    physical_contact_started_event_id: StrictStr
    completion_certainty: Literal["response_received"]
    policy_disposition: Literal["response_admitted"]
    nebius_model_endpoint: NebiusModelEndpointEvidence
    compatible_adapter_evidence: CompatibleAdapterResponseEvidence

    @model_validator(mode="after")
    def compatible_model_matches_selection(self) -> "ModelResponseRecorded":
        if (
            self.compatible_adapter_evidence.model_system == "openai"
            and self.compatible_adapter_evidence.model_name
            != self.nebius_model_endpoint.model
        ):
            raise ValueError("compatible model name changed the Nebius selection")
        return self


class ModelCompletionTruncated(_AttemptFact):
    """A completion the endpoint stopped on our own ceiling, not a contract defect.

    Retained between the admitted response and any output validation so that a
    truncated completion reads as a truncation rather than as a schema or
    output-validator failure. `finish_reason` is trustworthy in this direction
    only: it separates truncation from refusal, and is not a success signal.
    """

    event_type: Literal["ModelCompletionTruncated"] = "ModelCompletionTruncated"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    response_event_id: StrictStr
    finish_reason: Literal["length"] = "length"
    requested_max_tokens: StrictInt | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    completion_tokens: StrictInt | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    # This truncation's count in the Attempt, and the correction the retry sends
    # in place of the cut-off reply; no correction when it stops the Attempt.
    correction_count: StrictInt | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    feedback: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ModelRequestFailed(_AttemptFact):
    event_type: Literal["ModelRequestFailed"] = "ModelRequestFailed"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    physical_contact_count: StrictInt = Field(gt=0)
    last_physical_contact_event_id: StrictStr
    availability_exhaustion_reason: (
        Literal[
            "request_rejected",
            "malformed_response",
            "automatic_retry_exhausted",
            "retry_after_over_cap",
            "wait_budget_exhausted",
            "contact_stalled",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    nebius_model_endpoint: NebiusModelEndpointEvidence
    error_type: StrictStr
    http_status_code: StrictInt | None = Field(
        default=None, ge=100, le=599, exclude_if=lambda value: value is None
    )
    failure_reason: ModelFailureReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The endpoint's answer body and its own words, and the failure module's
    # key for the cause (status and body, G4.1).
    response_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    provider_statement: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fault_key: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def coherent_http_diagnostics(self) -> "ModelRequestFailed":
        if self.http_status_code is not None:
            if (
                self.error_type != "ModelHTTPError"
                or self.failure_reason
                != model_contact_reading("http", self.http_status_code)[1]
            ):
                raise ValueError("model HTTP failure diagnostics are incoherent")
        elif self.error_type == "ModelHTTPError":
            if self.failure_reason is not None:
                raise ValueError("statusless HTTP failure claimed a failure class")
        elif self.failure_reason not in {
            None,
            "non_http_failure",
            "malformed_response",
            "context_window_exhausted",
            "contact_stalled",
        }:
            raise ValueError("non-HTTP request failure classification is incoherent")
        if (
            self.failure_reason == "context_window_exhausted"
            and self.availability_exhaustion_reason is not None
        ):
            raise ValueError(
                "a malformed or locally refused request cannot exhaust availability retry"
            )
        return self


# Each logical model request ends in exactly one of these (a stall ends a
# request without a reply, and the Attempt goes on).
MODEL_REQUEST_TERMINALS = frozenset(
    {
        "ModelResponseRecorded",
        "ModelRequestFailed",
        "ModelRequestStalled",
        "ModelRequestRejected",
    }
)


class ModelRequestStalled(_AttemptFact):
    """A request that ended without a reply: its send stalled and was cut off.

    The Attempt goes on: its next request carries the stall note and names this
    event, and a stall of that request asks the availability Review (owner, #443
    Q-A). Nothing of the stalled reply was received.
    """

    event_type: Literal["ModelRequestStalled"] = "ModelRequestStalled"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    physical_contact_count: StrictInt = Field(gt=0)
    last_physical_contact_event_id: StrictStr
    stall_seconds: float = Field(gt=0, allow_inf_nan=False)
    nebius_model_endpoint: NebiusModelEndpointEvidence


class ModelRequestRejected(_AttemptFact):
    """A request the model endpoint refused with a 4xx and a body, shown to the
    model (owner, #476 alignment G1.5 B; #477).

    The Attempt goes on: its next request carries the provider's words and names
    this event, and is one more failed turn; the same refusal of that request asks
    the Run-level availability Review. ``response_event_id`` is the retained body.
    """

    event_type: Literal["ModelRequestRejected"] = "ModelRequestRejected"
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    physical_contact_count: StrictInt = Field(gt=0)
    last_physical_contact_event_id: StrictStr
    response_event_id: StrictStr
    http_status_code: StrictInt = Field(ge=400, le=499)
    provider_statement: StrictStr = Field(min_length=1)
    fault_key: StrictStr = Field(min_length=1)
    nebius_model_endpoint: NebiusModelEndpointEvidence


class ToolCallValidated(_AttemptFact):
    continuation_request_event_id: StrictStr | None = None
    event_type: Literal["ToolCallValidated"] = "ToolCallValidated"
    attempt_id: StrictStr
    tool_name: StrictStr
    tool_call_id: StrictStr
    arguments: dict[str, JsonValue]
    status: Literal["valid", "invalid"]
    errors: Tuple[StrictStr, ...]
    # The effective arguments include the request's server-bound values; these
    # names say which arguments the request bound. Older Journeys bound none.
    bound_argument_names: Tuple[StrictStr, ...] = ()
    # The model wrote the arguments inside a Markdown fence, which the Model turn
    # stripped; written only when true.
    arguments_fenced: StrictBool = Field(
        default=False, exclude_if=lambda value: not value
    )
    # The top-level arguments the Model turn read as the object their JSON text
    # held, in the order sent; written only when some were.
    arguments_decoded: Tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class AgentActionTurnDispositionRecorded(_AttemptFact):
    """One response's admission decision, or one call's in a multi-call response.

    It links the exact retained validations. A response with no Tool call is a
    text ending: it is refused with no validation and presents its own correction.
    """

    event_type: Literal["AgentActionTurnDispositionRecorded"] = (
        "AgentActionTurnDispositionRecorded"
    )
    attempt_id: StrictStr
    response_event_id: StrictStr
    validation_event_ids: Tuple[StrictStr, ...]
    # ``multi_call_*`` records one call of a multi-call response.
    disposition: Literal[
        "singleton_admitted",
        "singleton_refused",
        "multi_call_admitted",
        "multi_call_refused",
        "text_refused",
        "reasoning_refused",
    ]
    # The correction this turn presents: the retry prompt the model receives, or,
    # when the refusal stops the Attempt, the feedback its Review shows. Older
    # Journeys recorded none.
    feedback: StrictStr | None = None
    # A refused call or text ending: its fault key and that key's count in the
    # Attempt; a count beyond the allowance stopped the Attempt.
    correction_key: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    correction_count: StrictInt | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def cardinality_matches_disposition(self) -> "AgentActionTurnDispositionRecorded":
        one_call = self.disposition.startswith(("singleton_", "multi_call_"))
        if (self.correction_count is not None) != (
            self.disposition
            in {
                "singleton_refused",
                "multi_call_refused",
                "text_refused",
                "reasoning_refused",
            }
        ) or (self.correction_key is None) != (self.correction_count is None):
            raise ValueError("only a refused turn counts its fault key")
        if self.feedback is not None and self.disposition in {
            "singleton_admitted",
            "multi_call_admitted",
        }:
            raise ValueError("only a refused turn presents feedback")
        if self.disposition in {"text_refused", "reasoning_refused"}:
            if self.validation_event_ids or self.feedback is None:
                raise ValueError(
                    "a reply without a call has no validation and one correction"
                )
        elif (
            not self.validation_event_ids
            or len(set(self.validation_event_ids)) != len(self.validation_event_ids)
            or (one_call and len(self.validation_event_ids) != 1)
            or (not one_call and len(self.validation_event_ids) <= 1)
        ):
            raise ValueError("action disposition has incoherent validation cardinality")
        for name, value in (
            ("response_event_id", self.response_event_id),
            *(("validation_event_id", item) for item in self.validation_event_ids),
        ):
            try:
                parsed = uuid.UUID(value)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a UUID") from error
            if str(parsed) != value:
                raise ValueError(f"{name} must be canonical")
        return self


class TurnFaultCount(_AttemptFact):
    """One reply fault: its key and that key's count in the Attempt."""

    key: StrictStr = Field(min_length=1)
    count: StrictInt = Field(gt=0)


class AgentTurnRouted(_AttemptFact):
    """How admission reshaped one reply before any of its calls ran (ADR 0044).

    The raw reply stays in its ``ModelResponseRecorded``. The kept reply holds the
    calls of ``run_order``, in that order: each copy of an earlier identical call
    is collapsed onto it, each distinct unoffered call is refused by its own
    validation, each output call sent beside a call that runs is refused
    (``refused_output_call_ids``), and a cut-off reply's partial last call is
    dropped. ``faults`` are the reply's faults with their counts, and
    ``feedback`` the one note sent after the turn's results (none for a reply
    only reordered, nor for a resent reply that held no call: its correction
    comes alone, #444).
    """

    event_type: Literal["AgentTurnRouted"] = "AgentTurnRouted"
    attempt_id: StrictStr
    response_event_id: StrictStr
    run_order: Tuple[StrictStr, ...]
    # Each collapsed call and the call it copies.
    copies: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    unoffered_validation_event_ids: Tuple[StrictStr, ...] = ()
    # Written only when an output was refused.
    refused_output_call_ids: Tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    dropped_call_id: StrictStr | None = None
    faults: Tuple[TurnFaultCount, ...] = ()
    feedback: StrictStr | None = None

    @model_validator(mode="after")
    def coherent_routing(self) -> "AgentTurnRouted":
        named = [
            *self.run_order,
            *self.copies,
            *self.refused_output_call_ids,
            *([self.dropped_call_id] if self.dropped_call_id else []),
        ]
        if len(set(named)) != len(named) or any(
            target not in self.run_order and target in self.copies
            for target in self.copies.values()
        ):
            raise ValueError("routed reply names a call twice")
        if len({fault.key for fault in self.faults}) != len(self.faults):
            raise ValueError("routed reply counts one fault key twice")
        # A resent reply with no call is counted, and its correction comes alone.
        silent = not (
            self.run_order
            or self.copies
            or self.unoffered_validation_event_ids
            or self.refused_output_call_ids
            or self.dropped_call_id
        ) and all(fault.key == "model_resend" for fault in self.faults)
        if (self.feedback is None) != (not self.faults or silent):
            raise ValueError("a routed reply's note names exactly its faults")
        return self


class ToolExecutionFailed(_AttemptFact):
    event_type: Literal["ToolExecutionFailed"] = "ToolExecutionFailed"
    attempt_id: StrictStr
    tool_name: Literal[
        "inspect_input_metadata",
        "verify_sample_attestation",
        "resolve_taxon",
        "discover_reference_assemblies",
        "describe_reference",
        "describe_reference_companion",
        "search_prebuilt_indexes",
        "read_prebuilt_index",
        "suggest_assembly_with_index",
        "find_paper",
        "retrieve_paper",
        "list_compatible_analysis_recipes",
        "reconsider_recipe_choice",
        "lookup_compatible_resource_profiles",
    ] = "inspect_input_metadata"
    error_type: StrictStr


class AgentOutputValidated(_AttemptFact):
    event_type: Literal["AgentOutputValidated"] = "AgentOutputValidated"
    attempt_id: StrictStr
    agent_role: AgentRole = "planning_agent"
    status: Literal["valid", "invalid"]
    output: dict[str, JsonValue]
    errors: Tuple[StrictStr, ...]
    # A refused output's repair text as the model receives it, or, when repair is
    # exhausted, would have received. Older Journeys and valid outputs record none.
    feedback: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def only_a_refused_output_presents_feedback(self) -> "AgentOutputValidated":
        if self.feedback is not None and self.status != "invalid":
            raise ValueError("only a refused output presents repair feedback")
        return self


class SliceCheckpointReached(_AttemptFact):
    event_type: Literal["SliceCheckpointReached"] = "SliceCheckpointReached"
    attempt_id: StrictStr
    checkpoint: Literal[
        "registered_inputs_inspected",
        "reference_identity_progress",
        "planning_hitl_requested",
        "execution_planning_ready",
        "target_answer_read",
        "assembly_suggested",
        "paper_review_requested",
        # The plan's values cite a recorded paper still to read: the
        # coordinator reads it next; the checkpoint names the plan.
        "paper_reading_due",
    ] = "registered_inputs_inspected"
    input_set_id: StrictStr
    inspection_event_id: StrictStr | None = None
    proposal_plan_event_id: StrictStr | None = None
    reference_evidence_event_id: StrictStr | None = None
    planning_hitl_request_event_id: StrictStr | None = None
    execution_planning_evidence_event_id: StrictStr | None = None
    # The read of a researcher's target answer; left out when empty, so
    # earlier checkpoints keep their bytes.
    target_answer_read_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The suggestion of another assembly the Attempt ended at, for the coordinator's
    # question; left out when empty.
    assembly_suggestion_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The paper Review a retrieval's dead end asked; left out when empty.
    paper_review_request_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def checkpoint_lineage(self) -> "SliceCheckpointReached":
        if (self.checkpoint == "assembly_suggested") != (
            self.assembly_suggestion_event_id is not None
        ):
            raise ValueError("a suggestion checkpoint names exactly its suggestion")
        if (self.checkpoint == "target_answer_read") != (
            self.target_answer_read_event_id is not None
        ):
            raise ValueError("a target-answer checkpoint names exactly its read")
        if (self.checkpoint == "paper_review_requested") != (
            self.paper_review_request_event_id is not None
        ):
            raise ValueError("a paper-Review checkpoint names exactly its request")
        if self.checkpoint == "registered_inputs_inspected":
            if (
                self.inspection_event_id is None
                or self.proposal_plan_event_id is not None
                or self.reference_evidence_event_id is not None
                or self.planning_hitl_request_event_id is not None
                or self.execution_planning_evidence_event_id is not None
            ):
                raise ValueError(
                    "inspection checkpoint cannot reference a proposal plan"
                )
        elif self.checkpoint == "planning_hitl_requested":
            if (
                self.inspection_event_id is not None
                or self.proposal_plan_event_id is None
                or self.reference_evidence_event_id is not None
                or self.planning_hitl_request_event_id is None
                or self.execution_planning_evidence_event_id is not None
            ):
                raise ValueError("Planning HITL checkpoint requires its exact request")
        elif self.checkpoint == "execution_planning_ready":
            if (
                self.inspection_event_id is not None
                or self.proposal_plan_event_id is None
                or self.reference_evidence_event_id is not None
                or self.planning_hitl_request_event_id is not None
                or self.execution_planning_evidence_event_id is None
            ):
                raise ValueError(
                    "execution-planning checkpoint requires exact evidence"
                )
        elif self.checkpoint == "paper_reading_due":
            if (
                self.inspection_event_id is not None
                or self.proposal_plan_event_id is None
                or self.reference_evidence_event_id is not None
                or self.planning_hitl_request_event_id is not None
                or self.execution_planning_evidence_event_id is not None
            ):
                raise ValueError("a paper-reading checkpoint names exactly its plan")
        elif self.checkpoint in {
            "target_answer_read",
            "paper_review_requested",
            "assembly_suggested",
        }:
            # A paper Review may be asked before any plan: the cited paper is found
            # first, in the plan stage.
            if (
                self.inspection_event_id is not None
                or (
                    self.proposal_plan_event_id is None
                    and self.checkpoint != "paper_review_requested"
                )
                or self.reference_evidence_event_id is not None
                or self.planning_hitl_request_event_id is not None
                or self.execution_planning_evidence_event_id is not None
            ):
                raise ValueError("an answer checkpoint requires its exact source")
        elif (
            self.inspection_event_id is not None
            or self.proposal_plan_event_id is None
            or self.reference_evidence_event_id is None
            or self.planning_hitl_request_event_id is not None
            or self.execution_planning_evidence_event_id is not None
        ):
            raise ValueError("reference checkpoint requires its exact Tool evidence")
        return self


class AttemptSucceeded(_AttemptFact):
    event_type: Literal["AttemptSucceeded"] = "AttemptSucceeded"
    attempt_id: StrictStr
    checkpoint_event_id: StrictStr


class AttemptStopped(_AttemptFact):
    event_type: Literal["AttemptStopped"] = "AttemptStopped"
    attempt_id: StrictStr
    # The code names the review its source is: the effort review, or the Agent
    # work review for failed turns at their limit, a missing typed result or an
    # exhausted model context window.
    code: Literal[
        "operational_failure",
        "effort_exhausted",
        "failed_turns",
        "result_missing",
        "context_exhausted",
        "researcher_question",
    ] = "effort_exhausted"
    source_event_id: StrictStr


class AttemptFailed(_AttemptFact):
    """An Attempt's failure by its true class (#216).

    A model or Tool fault or an invalid model output keeps its code; anything
    else is our bug, an ``invariant`` naming its id.
    """

    event_type: Literal["AttemptFailed"] = "AttemptFailed"
    attempt_id: StrictStr
    code: Literal[
        "agent_run_failed",
        "agent_output_invalid",
        "invariant",
        "model_availability_exhausted",
    ]
    invariant_id: StrictStr | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    source_event_id: StrictStr

    @model_validator(mode="after")
    def typed_invariant(self):
        if (self.code == "invariant") != (self.invariant_id is not None):
            raise ValueError("an invariant, and only an invariant, names its id")
        return self

    @field_validator("source_event_id")
    @classmethod
    def canonical_source_reference(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("source_event_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("source_event_id must be canonical")
        return value


# A retained model, Tool or output fault keeps its class; anything else is our bug.
_FAILURE_CODES = {
    "ModelRequestFailed": "agent_run_failed",
    "ToolExecutionFailed": "agent_run_failed",
    "ObservedToolExecutionFailed": "agent_run_failed",
    "AgentOutputValidated": "agent_output_invalid",
}


def record_attempt_failure(context, error: BaseException) -> dict[str, Any]:
    """Retain an Attempt's failure by its true class (#216) and close its activity.

    A model or Tool fault or an invalid output cites its retained source; anything
    else is an ``invariant`` citing the Attempt's last retained response, or else its
    start. The role then raises the failure module's ``Invariant``, which holds the
    Run (ADR 0031)."""
    from bio_neb.failure import Invariant
    from bio_neb.recording import read_evidence

    source = context.failure_source
    code = _FAILURE_CODES.get(source.event_type, "invariant")
    invariant_id = None
    source_event_id = source.event_id
    if isinstance(error, Invariant) and source.event_type == (
        "ObservedToolExecutionFailed"
    ):
        # A Tool's own bug keeps its id and cites the Tool's failure.
        code, invariant_id = "invariant", error.invariant_id
    elif code == "invariant":
        invariant_id = (
            error.invariant_id
            if isinstance(error, Invariant)
            else "attempt_unhandled_error"
        )
        source_event_id = next(
            event["event_id"]
            for event in reversed(read_evidence(context.journey))
            if event["event_type"] in {"ModelResponseRecorded", "AttemptStarted"}
            and event["payload"]["attempt_id"] == context.attempt_id
        )
    failed = context.activity.record(
        AttemptFailed(
            attempt_id=context.attempt_id,
            code=code,
            invariant_id=invariant_id,
            source_event_id=source_event_id,
        )
    )
    context.activity.failed(failed)
    return failed


class AttemptCancelled(_AttemptFact):
    event_type: Literal["AttemptCancelled"] = "AttemptCancelled"
    attempt_id: StrictStr
    source_event_id: StrictStr

    @field_validator("source_event_id")
    @classmethod
    def canonical_source_reference(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("source_event_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("source_event_id must be canonical")
        return value


class ToolExecutionRefused(_AttemptFact):
    """A Tool that could not produce its fact, shown to the model (owner, #477 C8).

    Its declared refusal (``failure.ToolRefused``), or once its undeclared failure
    (``failure.TOOL_FAILED``), is the call's result; ``failed_event_id`` is the
    execution's mechanical terminal. The Attempt continues, and the reply that
    holds it is one more failed turn when nothing else of it worked. ``fault_key``
    is the failure module's key: the same undeclared key again holds the Run.
    """

    event_type: Literal["ToolExecutionRefused"] = "ToolExecutionRefused"
    attempt_id: StrictStr
    failed_event_id: StrictStr
    reason: StrictStr = Field(min_length=1)
    statement: StrictStr = Field(min_length=1)
    fault_key: StrictStr = Field(min_length=1)
    # Where to go next and the reply's place in the failed turns in a row, as the
    # failure module wrote them (``failure.execution_refused_view``).
    next: StrictStr = Field(min_length=1)
    count: StrictStr = Field(min_length=1)

    def model_view(self) -> dict[str, str]:
        """The call's result as the model was shown it: the failure module's
        rendering, recorded with the fact (#478 G1.4)."""
        return {
            "status": "refused",
            "reason": self.reason,
            "statement": self.statement,
            "next": self.next,
            "count": self.count,
        }


_FACT_TYPES = {
    fact.model_fields["event_type"].default: fact
    for fact in (
        AttemptStarted,
        ModelRequestRecorded,
        ModelTransportRequestRecorded,
        ModelTransportResponseRecorded,
        ModelPhysicalContactStarted,
        ModelPhysicalContactFailed,
        ModelPhysicalContactCancelled,
        ModelAvailabilityRetryScheduled,
        ModelAvailabilityRetryDelayCompleted,
        ModelAvailabilityRetryDelayCancelled,
        ModelTurnCancelled,
        ModelResponseRecorded,
        ModelCompletionTruncated,
        ModelRequestFailed,
        ModelRequestStalled,
        ModelRequestRejected,
        ToolCallValidated,
        AgentTurnRouted,
        AgentActionTurnDispositionRecorded,
        ToolExecutionFailed,
        ToolExecutionRefused,
        AgentOutputValidated,
        SliceCheckpointReached,
        AttemptSucceeded,
        AttemptStopped,
        AttemptFailed,
        AttemptCancelled,
    )
}


def attempt_members(events) -> dict[str, str | None]:
    """Each Attempt's recorded member scope (``None``: the whole role)."""
    return {
        event["payload"]["attempt_id"]: event["payload"].get("member_id")
        for event in events
        if event.get("event_type") == "AttemptStarted"
    }


def attempts_since(events, event_id: str | None) -> frozenset[str] | None:
    """The Attempts started after ``event_id``, whose stops can hold work that
    answers it; None (every Attempt) when no request bounds the hold (#408)."""
    if event_id is None:
        return None
    ids = [event["event_id"] for event in events]
    if event_id not in ids:
        raise ValueError("an Attempt's hold names no earlier request")
    return frozenset(
        event["payload"]["attempt_id"]
        for event in events[ids.index(event_id) :]
        if event.get("event_type") == "AttemptStarted"
    )


def shares_hold_scope(held_member: str | None, member: str | None) -> bool:
    """Whether a stop in one member scope holds work in another.

    A role-wide stop or start shares every scope; two member scopes share
    only when they name the same member (#105).
    """
    return held_member is None or member is None or held_member == member


_ACTIVITY_TERMINALS = frozenset(
    {"activity_succeeded", "activity_failed", "activity_cancelled"}
)


def attempt_lifecycle_window(
    timeline: list[dict[str, Any]], started: dict[str, Any]
) -> tuple[int, int]:
    """The positions of an Attempt's lifecycle start and its one lifecycle terminal."""
    begin, terminals = _lifecycle_terminals(timeline, started)
    if len(terminals) != 1:
        raise ValueError("Attempt activity lacks one exact lifecycle terminal")
    return begin, terminals[0] + 1


def _lifecycle_terminals(
    timeline: list[dict[str, Any]], started: dict[str, Any]
) -> tuple[int, list[int]]:
    """An Attempt's lifecycle start position and its lifecycle terminals' positions."""
    if len(started["parent_ids"]) != 1:
        raise ValueError("Attempt start must have one lifecycle parent")
    lifecycle_id = started["parent_ids"][0]
    begin = next(
        (i for i, event in enumerate(timeline) if event["event_id"] == lifecycle_id),
        None,
    )
    if begin is None or timeline[begin]["event_type"] != "activity_started":
        raise ValueError("Attempt start lacks lifecycle evidence")
    activity_id = timeline[begin]["payload"]["activity_id"]
    return begin, [
        index
        for index, event in enumerate(timeline[begin + 1 :], begin + 1)
        if event["event_type"] in _ACTIVITY_TERMINALS
        and event["payload"]["activity_id"] == activity_id
    ]


def attempt_events(
    timeline: list[dict[str, Any]], started: dict[str, Any]
) -> list[dict[str, Any]]:
    """The Attempt's own events, in Journey order, by identity rather than position.

    Concurrent Attempts interleave (#105): an event is this Attempt's when it
    names its ``attempt_id``, or, naming none (an activity's lifecycle), when an
    owned event or activity is its cause. A sibling's events are never selected.
    """
    return _owned_events(
        timeline, started, *attempt_lifecycle_window(timeline, started)
    )


def attempt_events_so_far(
    timeline: list[dict[str, Any]], started: dict[str, Any]
) -> list[dict[str, Any]]:
    """The Attempt's own events, also while it is open (no lifecycle terminal yet).

    The same lifecycle rules hold: a start without one lifecycle parent, a
    parent that is no lifecycle start, or more than one terminal is refused.
    """
    begin, terminals = _lifecycle_terminals(timeline, started)
    if len(terminals) > 1:
        raise ValueError("Attempt activity lacks one exact lifecycle terminal")
    end = terminals[0] + 1 if terminals else len(timeline)
    return _owned_events(timeline, started, begin, end)


def _owned_events(
    timeline: list[dict[str, Any]], started: dict[str, Any], begin: int, end: int
) -> list[dict[str, Any]]:
    attempt_id = started["payload"]["attempt_id"]
    lifecycle = timeline[begin]
    owned = {lifecycle["event_id"]}
    activities = {lifecycle["payload"]["activity_id"]}
    selected = [lifecycle]
    for event in timeline[begin + 1 : end]:
        payload = event.get("payload", {})
        if "attempt_id" in payload:
            mine = payload["attempt_id"] == attempt_id
        elif event["event_type"].startswith("activity_"):
            mine = payload.get("activity_id") in activities or any(
                parent in owned for parent in event["parent_ids"]
            )
        else:
            mine = bool(event["parent_ids"]) and all(
                parent in owned for parent in event["parent_ids"]
            )
        if mine:
            owned.add(event["event_id"])
            if event["event_type"] == "activity_started":
                activities.add(payload["activity_id"])
            selected.append(event)
    return selected


def validate_attempt_fact(
    event: Mapping[str, Any],
) -> (
    _AttemptFact | ModelTransportRequestRecorded | ModelTransportResponseRecorded | None
):
    owner = _FACT_TYPES.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )
