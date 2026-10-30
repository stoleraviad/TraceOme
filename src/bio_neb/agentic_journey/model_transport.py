"""Retain actual model transport preparation without giving adapters a Journey."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.model_settings import ModelContextBudget
from bio_neb.content import ContentId
from bio_neb.failure import provider_statement
from bio_neb.journey import ArtifactReference, Journey

TRANSPORT_MEDIA_TYPE = "application/json"


def validate_model_truncation_source(
    selected: list[dict[str, Any]], source: Mapping[str, Any]
) -> None:
    """Keep one typed truncation tied to the actual completed model response."""
    from bio_neb.agentic_journey.attempt_journey import ModelCompletionTruncated

    fact = ModelCompletionTruncated.model_validate(source["payload"])
    responses = [
        event
        for event in selected
        if event["event_type"] == "ModelResponseRecorded"
        and event["event_id"] == fact.response_event_id
    ]
    requests = [
        event
        for event in selected
        if event["event_type"] == "ModelRequestRecorded"
        and event["payload"]["request_number"] == fact.request_number
        and event["payload"]["attempt_id"] == fact.attempt_id
    ]
    if len(responses) != 1 or len(requests) != 1:
        raise ValueError("model truncation lost its exact request or response")
    response, request = responses[0], requests[0]
    message = response["payload"]["compatible_adapter_evidence"]["response"]["message"]
    closures = [
        event
        for event in selected
        if event["event_type"] == "activity_succeeded"
        and event["payload"].get("result_event_id") == response["event_id"]
        and event["payload"].get("stage") == "model_contact"
        and event["parent_ids"] == [response["event_id"]]
    ]
    if (
        response["payload"]["request_number"] != fact.request_number
        or response["payload"]["attempt_id"] != fact.attempt_id
        or not selected.index(request)
        < selected.index(response)
        < selected.index(source)
        or len(closures) != 1
        # Admission records it first, as it routes the reply (ADR 0044).
        or source["parent_ids"] != [closures[0]["event_id"]]
        or message.get("finish_reason") != "length"
        or fact.requested_max_tokens
        != request["payload"]["compatible_adapter_evidence"]["request"][
            "model_settings"
        ].get("max_tokens")
        or fact.completion_tokens != message.get("usage", {}).get("output_tokens")
    ):
        raise ValueError("model truncation changed its exact response or budget")


@dataclass(frozen=True)
class PreparedModelTransportRequest:
    """The two actual SDK builds; neither their headers nor credentials cross here."""

    unbudgeted_body: bytes
    request_body: bytes
    context_budget: ModelContextBudget


@dataclass(frozen=True)
class ReceivedModelTransportResponse:
    """One HTTP response as the provider sent it: status, media type and raw body.

    The body is the exact bytes, before the SDK parses anything, so a lost call
    can be traced to its layer (#421).
    """

    status_code: int
    media_type: str
    body: bytes


_observer: ContextVar[Callable[[PreparedModelTransportRequest], None] | None] = (
    ContextVar("model_transport_observer", default=None)
)
_response_observer: ContextVar[
    Callable[[ReceivedModelTransportResponse], None] | None
] = ContextVar("model_transport_response_observer", default=None)


@contextmanager
def observe_model_transport_requests(
    observer: Callable[[PreparedModelTransportRequest], None],
) -> Iterator[None]:
    """Bind only the current task and tasks created inside this scope."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def model_transport_request_prepared(request: PreparedModelTransportRequest) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(request)


@contextmanager
def observe_model_transport_responses(
    observer: Callable[[ReceivedModelTransportResponse], None],
) -> Iterator[None]:
    """Bind only the current task and tasks created inside this scope."""
    token = _response_observer.set(observer)
    try:
        yield
    finally:
        _response_observer.reset(token)


def model_transport_response_received(response: ReceivedModelTransportResponse) -> None:
    observer = _response_observer.get()
    if observer is not None:
        observer(response)


class ModelTransportRequestRecorded(BaseModel):
    """Prepared bytes bound to a physical contact, not proof of provider receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: Literal["ModelTransportRequestRecorded"] = (
        "ModelTransportRequestRecorded"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    contact_started_event_id: StrictStr
    contact_ordinal: StrictInt = Field(gt=0)
    unbudgeted_body_byte_count: StrictInt = Field(gt=0)
    unbudgeted_body_sha256: StrictStr
    request_body_byte_count: StrictInt = Field(gt=0)
    request_body_sha256: StrictStr
    requested_max_tokens: StrictInt | None = Field(default=None, gt=0)
    final_max_completion_tokens: StrictInt | None = Field(default=None, gt=0)
    context_budget: ModelContextBudget

    @field_validator(
        "attempt_id", "logical_request_event_id", "contact_started_event_id"
    )
    @classmethod
    def canonical_event_identity(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("model transport event identity must be canonical")
        return value

    @field_validator("unbudgeted_body_sha256", "request_body_sha256")
    @classmethod
    def canonical_digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("model transport digest must be lowercase SHA256")
        return value

    @model_validator(mode="after")
    def coherent_budget(self) -> ModelTransportRequestRecorded:
        # The allowance never exceeds the room the body's estimate leaves; a
        # smaller recorded one is safer and is not refused here.
        if (self.requested_max_tokens is None) != (
            self.final_max_completion_tokens is None
        ) or (
            self.final_max_completion_tokens is not None
            and self.final_max_completion_tokens
            > self.context_budget.completion_allowance(
                self.requested_max_tokens, self.unbudgeted_body_byte_count
            )
        ):
            raise ValueError("model transport final allowance exceeds its budget")
        if self.requested_max_tokens == self.final_max_completion_tokens and (
            self.unbudgeted_body_byte_count != self.request_body_byte_count
            or self.unbudgeted_body_sha256 != self.request_body_sha256
        ):
            raise ValueError("unadjusted model transport changed its request bytes")
        return self

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedModelTransportRequest,
        *,
        attempt_id: str,
        request_number: int,
        logical_request_event_id: str,
        contact_started_event_id: str,
        contact_ordinal: int,
    ) -> ModelTransportRequestRecorded:
        return cls(
            attempt_id=attempt_id,
            request_number=request_number,
            logical_request_event_id=logical_request_event_id,
            contact_started_event_id=contact_started_event_id,
            contact_ordinal=contact_ordinal,
            unbudgeted_body_byte_count=len(prepared.unbudgeted_body),
            unbudgeted_body_sha256=hashlib.sha256(prepared.unbudgeted_body).hexdigest(),
            request_body_byte_count=len(prepared.request_body),
            request_body_sha256=hashlib.sha256(prepared.request_body).hexdigest(),
            requested_max_tokens=json.loads(prepared.unbudgeted_body).get(
                "max_completion_tokens"
            ),
            final_max_completion_tokens=json.loads(prepared.request_body).get(
                "max_completion_tokens"
            ),
            context_budget=prepared.context_budget,
        )


class ModelTransportResponseRecorded(BaseModel):
    """The raw bytes one physical contact received, beside its request body.

    Bound to the same contact as ``ModelTransportRequestRecorded``; an error
    status keeps its body too. Not proof of what the adapter read from it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: Literal["ModelTransportResponseRecorded"] = (
        "ModelTransportResponseRecorded"
    )
    attempt_id: StrictStr
    request_number: StrictInt = Field(gt=0)
    logical_request_event_id: StrictStr
    contact_started_event_id: StrictStr
    contact_ordinal: StrictInt = Field(gt=0)
    status_code: StrictInt = Field(ge=100, le=599)
    media_type: StrictStr = Field(min_length=1)
    response_body_byte_count: StrictInt = Field(ge=0)
    response_body_sha256: StrictStr

    @field_validator(
        "attempt_id", "logical_request_event_id", "contact_started_event_id"
    )
    @classmethod
    def canonical_event_identity(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("model transport event identity must be canonical")
        return value

    @field_validator("response_body_sha256")
    @classmethod
    def canonical_digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("model transport digest must be lowercase SHA256")
        return value

    @classmethod
    def from_received(
        cls,
        received: ReceivedModelTransportResponse,
        *,
        attempt_id: str,
        request_number: int,
        logical_request_event_id: str,
        contact_started_event_id: str,
        contact_ordinal: int,
    ) -> ModelTransportResponseRecorded:
        return cls(
            attempt_id=attempt_id,
            request_number=request_number,
            logical_request_event_id=logical_request_event_id,
            contact_started_event_id=contact_started_event_id,
            contact_ordinal=contact_ordinal,
            status_code=received.status_code,
            media_type=received.media_type,
            response_body_byte_count=len(received.body),
            response_body_sha256=hashlib.sha256(received.body).hexdigest(),
        )


_CONTACT_OUTCOMES = frozenset(
    {
        "ModelResponseRecorded",
        "ModelRequestFailed",
        "ModelRequestStalled",
        "ModelRequestRejected",
        "ModelPhysicalContactFailed",
        "ModelPhysicalContactCancelled",
    }
)


def validate_model_transport_requests(
    events: Sequence[Mapping[str, Any]], *, journey: Journey | None = None
) -> None:
    """Check retained bindings; absence on old or non-transport Models stays absence."""
    by_id = {event["event_id"]: (index, event) for index, event in enumerate(events)}
    seen_contacts: set[str] = set()
    for index, event in enumerate(events):
        if event.get("event_type") != "ModelTransportRequestRecorded":
            continue
        fact = ModelTransportRequestRecorded.model_validate(event["payload"])
        request_index, request = by_id.get(fact.logical_request_event_id, (-1, {}))
        contact_index, contact = by_id.get(fact.contact_started_event_id, (-1, {}))
        rp, cp = request.get("payload", {}), contact.get("payload", {})
        if (
            request.get("event_type") != "ModelRequestRecorded"
            or contact.get("event_type") != "ModelPhysicalContactStarted"
            or not 0 <= request_index < contact_index < index
            or fact.contact_started_event_id in seen_contacts
            or (rp.get("attempt_id"), rp.get("request_number"))
            != (fact.attempt_id, fact.request_number)
            or (
                cp.get("attempt_id"),
                cp.get("request_number"),
                cp.get("contact_ordinal"),
            )
            != (fact.attempt_id, fact.request_number, fact.contact_ordinal)
            or cp.get("logical_request_event_id") != fact.logical_request_event_id
            or any(
                prior.get("event_type") in _CONTACT_OUTCOMES
                and prior.get("payload", {}).get("attempt_id") == fact.attempt_id
                and prior.get("payload", {}).get("request_number")
                == fact.request_number
                for prior in events[contact_index + 1 : index]
            )
        ):
            raise ValueError("model transport changed its open physical contact")
        seen_contacts.add(fact.contact_started_event_id)
        settings = rp["compatible_adapter_evidence"]["request"]["model_settings"]
        if settings.get("max_tokens") != fact.requested_max_tokens or rp.get(
            "settings_selection", {}
        ).get("context_budget") != fact.context_budget.model_dump(mode="json"):
            raise ValueError("model transport changed its selected request budget")
        refs = event.get("artifact_references", ())
        if len(refs) != 1:
            raise ValueError("model transport requires its exact body artifact")
        ref = ArtifactReference.model_validate(refs[0])
        if (
            ref.source_event_id != event["event_id"]
            or ref.journey_id != event["journey_id"]
            or ref.byte_count != fact.request_body_byte_count
            or ContentId.parse(ref.artifact_id).digest != fact.request_body_sha256
            or ref.media_type != TRANSPORT_MEDIA_TYPE
        ):
            raise ValueError("model transport changed its body artifact identity")
        if journey is not None:
            body = json.loads(journey.read_artifact(ref))
            if (
                body.get("model") != rp["nebius_model_endpoint"]["model"]
                or body.get("max_completion_tokens") != fact.final_max_completion_tokens
            ):
                raise ValueError("model transport body changed its model or allowance")
    # A failed/cancelled preparation may have no completed SDK build. A response
    # from a transport-aware Adapter, however, must keep its actual prepared bytes.
    for event in events:
        if event.get("event_type") != "ModelResponseRecorded":
            continue
        contact_id = event["payload"].get("physical_contact_started_event_id")
        _, contact = by_id.get(contact_id, (-1, {}))
        if (
            contact.get("payload", {}).get("transport_evidence_expected") is True
            and contact_id not in seen_contacts
        ):
            raise ValueError(
                "model response lost its expected transport request evidence"
            )


def validate_model_transport_responses(
    events: Sequence[Mapping[str, Any]], *, journey: Journey | None = None
) -> None:
    """Check each retained response body against the one contact it names.

    It follows that contact's request body and comes before the contact's
    outcome. A contact that expected one keeps it for its response; Journeys
    recorded before #421 expected none, so absence there stays absence.
    """
    by_id = {event["event_id"]: (index, event) for index, event in enumerate(events)}
    requested = {
        event["payload"]["contact_started_event_id"]: index
        for index, event in enumerate(events)
        if event.get("event_type") == "ModelTransportRequestRecorded"
    }
    seen_contacts: set[str] = set()
    for index, event in enumerate(events):
        if event.get("event_type") != "ModelTransportResponseRecorded":
            continue
        fact = ModelTransportResponseRecorded.model_validate(event["payload"])
        contact_index, contact = by_id.get(fact.contact_started_event_id, (-1, {}))
        cp = contact.get("payload", {})
        if (
            contact.get("event_type") != "ModelPhysicalContactStarted"
            or cp.get("response_body_expected") is not True
            or not 0 <= contact_index < index
            or fact.contact_started_event_id in seen_contacts
            or (
                cp.get("transport_evidence_expected") is True
                and not contact_index
                < requested.get(fact.contact_started_event_id, index)
                < index
            )
            or (
                cp.get("attempt_id"),
                cp.get("request_number"),
                cp.get("contact_ordinal"),
                cp.get("logical_request_event_id"),
            )
            != (
                fact.attempt_id,
                fact.request_number,
                fact.contact_ordinal,
                fact.logical_request_event_id,
            )
            or any(
                prior.get("event_type") in _CONTACT_OUTCOMES
                and prior.get("payload", {}).get("attempt_id") == fact.attempt_id
                and prior.get("payload", {}).get("request_number")
                == fact.request_number
                for prior in events[contact_index + 1 : index]
            )
        ):
            raise ValueError("model transport response changed its physical contact")
        seen_contacts.add(fact.contact_started_event_id)
        refs = event.get("artifact_references", ())
        if len(refs) != 1:
            raise ValueError("model transport response requires its exact body")
        ref = ArtifactReference.model_validate(refs[0])
        if (
            ref.source_event_id != event["event_id"]
            or ref.journey_id != event["journey_id"]
            or ref.byte_count != fact.response_body_byte_count
            or ContentId.parse(ref.artifact_id).digest != fact.response_body_sha256
            or ref.media_type != fact.media_type
        ):
            raise ValueError("model transport changed its response body identity")
        body = None
        if journey is not None:
            # The read verifies the retained bytes against their content address.
            body = journey.read_artifact(ref)
        outcome = next(
            (
                later
                for later in events[index + 1 :]
                if later.get("event_type") in _CONTACT_OUTCOMES
                and later.get("payload", {}).get("attempt_id") == fact.attempt_id
                and later.get("payload", {}).get("request_number")
                == fact.request_number
            ),
            None,
        )
        op = (outcome or {}).get("payload", {})
        if (outcome or {}).get("event_type") == "ModelResponseRecorded" and (
            op.get("physical_contact_started_event_id") != fact.contact_started_event_id
            or not 200 <= fact.status_code < 300
        ):
            raise ValueError("model transport response changed its HTTP status")
        if (outcome or {}).get("event_type") == "ModelPhysicalContactFailed" and (
            op.get("contact_started_event_id") != fact.contact_started_event_id
            or op.get("http_status_code") not in (None, fact.status_code)
        ):
            raise ValueError("model transport response changed its HTTP status")
        if (
            (outcome or {}).get("event_type") == "ModelPhysicalContactFailed"
            and op.get("http_status_code") is not None
            and (
                op.get("response_event_id") != event["event_id"]
                or (
                    body is not None
                    and op.get("provider_statement") != provider_statement(body)
                )
            )
        ):
            # The failure's pointer and words are this answer's own.
            raise ValueError("model failure changed its retained answer's words")
    # A received response keeps its body: an admitted one, an HTTP failure or a
    # malformed reply alike (``response_received``); a send with no response has none.
    for event in events:
        payload = event.get("payload", {})
        if event.get("event_type") == "ModelResponseRecorded":
            contact_id = payload["physical_contact_started_event_id"]
        elif (
            event.get("event_type") == "ModelPhysicalContactFailed"
            and payload["completion_certainty"] == "response_received"
        ):
            contact_id = payload["contact_started_event_id"]
        else:
            continue
        _, contact = by_id.get(contact_id, (-1, {}))
        if (
            contact.get("payload", {}).get("response_body_expected") is True
            and contact_id not in seen_contacts
        ):
            raise ValueError("model response lost its expected raw response body")
