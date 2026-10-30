"""Typed Journey facts for registered FASTQ inputs and metadata inspection."""

from __future__ import annotations

import os
import uuid
from typing import Any, Literal, Mapping, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.runtime_config import RetainedRuntimeSelections
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Ask, ReviewAnswer
from bio_neb.input_inspection.tool import (
    InputMetadataResult,
    RegisteredInputPath,
    registered_locator,
)
from bio_neb.journey import Journey, has_ancestor


class _JourneyFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegisteredFastqInput(_JourneyFact):
    role: StrictStr
    registered_path: StrictStr

    @field_validator("role")
    @classmethod
    def canonical_role(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("role must be a non-empty canonical string")
        return value

    @field_validator("registered_path")
    @classmethod
    def exact_locator(cls, value: str) -> str:
        return registered_locator(value)


# Execution runs one forward/reverse pair: one sample on one lane. Registration
# refuses any other role set; several lanes per sample come later (lane M).
PAIRED_FASTQ_ROLE_SETS = (
    frozenset({"read_1", "read_2"}),
    frozenset({"forward_read", "reverse_read"}),
)


def paired_fastq_roles(roles) -> bool:
    """Whether these FASTQ roles are exactly one complete forward/reverse pair."""
    roles = tuple(roles)
    return len(roles) == 2 and frozenset(roles) in PAIRED_FASTQ_ROLE_SETS


class JourneyStarted(_JourneyFact):
    event_type: Literal["JourneyStarted"] = "JourneyStarted"


class ResearchRequestRegistered(_JourneyFact):
    event_type: Literal["ResearchRequestRegistered"] = "ResearchRequestRegistered"
    prompt: StrictStr
    prompt_file: StrictStr

    @field_validator("prompt_file")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not value or not os.path.isabs(value):
            raise ValueError("prompt_file must be absolute")
        return value


class RunInputsRegistered(_JourneyFact):
    event_type: Literal["RunInputsRegistered"] = "RunInputsRegistered"
    input_set_id: StrictStr
    run_root: StrictStr
    # The file a Run was begun with, if any; its resolved selections below are the
    # only configuration later work reads. Without one, no key is recorded.
    runtime_config_file: StrictStr | None = Field(
        default=None, exclude_if=lambda path: path is None
    )
    runtime_selections: RetainedRuntimeSelections
    inputs: Tuple[RegisteredFastqInput, ...] = Field(min_length=1)
    # Retained key only: every registration records it empty. Supplied prepared
    # reference files were never admitted; the prepared index is a launch choice.
    prepared_reference_inputs: tuple[RegisteredInputPath, ...] = Field(
        default=(), max_length=0
    )

    @field_validator("input_set_id")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("input_set_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("input_set_id must be canonical")
        return value

    @field_validator("run_root", "runtime_config_file")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and (not value or not os.path.isabs(value)):
            raise ValueError("operational paths must be absolute")
        return value

    @model_validator(mode="after")
    def unique_roles(self) -> "RunInputsRegistered":
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("registered FASTQ roles must be unique")
        return self


class InputMetadataInspected(_JourneyFact):
    event_type: Literal["InputMetadataInspected"] = "InputMetadataInspected"
    selection_event_id: StrictStr
    tool_name: Literal["inspect_input_metadata"] = "inspect_input_metadata"
    input_set_id: StrictStr
    result: InputMetadataResult

    @model_validator(mode="after")
    def matching_identity(self) -> "InputMetadataInspected":
        if self.result.input_set_id != self.input_set_id:
            raise ValueError("inspection result identity changed")
        return self


_FACT_TYPES = {
    "JourneyStarted": JourneyStarted,
    "ResearchRequestRegistered": ResearchRequestRegistered,
    "RunInputsRegistered": RunInputsRegistered,
    "InputMetadataInspected": InputMetadataInspected,
}


def fact_payload(fact: _JourneyFact) -> dict[str, Any]:
    value = fact.model_dump(mode="json")
    value.pop("event_type")
    return value


def validate_fact(event: Mapping[str, Any]) -> _JourneyFact | None:
    owner = _FACT_TYPES.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


def record_fact(
    journey: Journey, fact: _JourneyFact, parent_ids: Tuple[str, ...] = ()
) -> dict[str, Any]:
    return journey.record(
        fact.event_type,
        fact_payload(fact),
        list(parent_ids),
    )


class InputInspectionErrored(_JourneyFact):
    event_type: Literal["InputInspectionErrored"] = "InputInspectionErrored"
    selection_event_id: str
    error_type: str


class InputInspectionReviewRequested(_JourneyFact):
    event_type: Literal["InputInspectionReviewRequested"] = (
        "InputInspectionReviewRequested"
    )
    request: "Ask"


class InputInspectionReviewAnswered(_JourneyFact):
    event_type: Literal["InputInspectionReviewAnswered"] = (
        "InputInspectionReviewAnswered"
    )
    request_event_id: str
    response: "ReviewAnswer"


class InputReviewRequest(FrozenContract):
    request_event_id: str
    request: Ask
    pending: bool


class InputReviewReceipt(FrozenContract):
    request_event_id: str
    response_event_id: str
    response: ReviewAnswer


class InputSelection(FrozenContract):
    input_set_id: str
    source_event_id: str
    inputs: tuple[RegisteredInputPath, ...]
    fastq_roles: tuple[str, ...]
    inspections: tuple[dict[str, Any], ...] = ()
    requests: tuple[InputReviewRequest, ...] = ()
    receipts: tuple[InputReviewReceipt, ...] = ()
    correction_pending: bool = False
    stopped: bool = False


def validate_input_answer(request, response):
    if (
        response.request_id != request.request.request_id
        or response.action not in request.request.legal_actions
    ):
        raise ValueError("input correction changed its exact request or legal action")
    if response.action == "amend":
        if response.field not in request.request.amendable or not isinstance(
            response.value, str
        ):
            raise ValueError(
                "input correction requires one declared role and absolute path"
            )
        RegisteredInputPath(role=response.field, registered_path=response.value)


def project_input_selection(events) -> InputSelection:
    """Interpret original selection, exact human amendment and ordered observations."""
    from bio_neb.input_inspection.tool import (
        InputMetadataInspectionTool,
        InspectInputMetadataInput,
    )
    from bio_neb.junction import evaluate_scoped_answer

    registration = next(e for e in events if e["event_type"] == "RunInputsRegistered")
    fact = RunInputsRegistered.model_validate(
        {"event_type": registration["event_type"], **registration["payload"]}
    )
    selection_id = registration["event_id"]
    inputs = tuple(
        RegisteredInputPath(role=i.role, registered_path=i.registered_path)
        for i in fact.inputs
    )
    inspections, requests, receipts = [], [], []
    correction_pending = stopped = False
    for event in events:
        kind = event["event_type"]
        if kind == "InputMetadataInspected":
            inspected = InputMetadataInspected.model_validate(
                {"event_type": kind, **event["payload"]}
            )
            if stopped or (inspections and not correction_pending):
                raise ValueError(
                    "input reinspection lacks an exact correction direction"
                )
            if (
                inspected.selection_event_id != selection_id
                or inspected.input_set_id != fact.input_set_id
            ):
                raise ValueError("inspection changed its selected input provenance")
            if not has_ancestor(
                events, ancestor_id=selection_id, descendant_id=event["event_id"]
            ):
                raise ValueError("inspection is outside selection lineage")
            selected = InspectInputMetadataInput(
                input_set_id=fact.input_set_id, inputs=inputs
            )
            if inspected.result.observation is not None:
                expected = InputMetadataInspectionTool().derive_from_observation(
                    selected, inspected.result.observation
                )
                if expected != inspected.result:
                    raise ValueError(
                        "inspection facts do not derive from selected observation"
                    )
            inspections.append(event)
            correction_pending = False
        elif kind == "InputInspectionErrored":
            error = InputInspectionErrored.model_validate(
                {"event_type": kind, **event["payload"]}
            )
            if error.selection_event_id != selection_id or stopped:
                raise ValueError("input error changed its current selection")
            stopped = True
            correction_pending = False
        elif kind == "InputInspectionReviewRequested":
            request = InputInspectionReviewRequested.model_validate(
                {"event_type": kind, **event["payload"]}
            ).request
            if (
                not inspections
                or inspections[-1]["payload"]["result"]["status"] != "failed"
                or request.failure_event_id != inspections[-1]["event_id"]
            ):
                raise ValueError("input review lacks its latest observed failure")
            if event["parent_ids"] != [request.failure_event_id] or any(
                r.pending for r in requests
            ):
                raise ValueError(
                    "input review changed lineage or duplicated pending work"
                )
            if (
                request.stage != "input_metadata"
                or request.amendable != tuple(i.role for i in inputs)
                or request.legal_actions != ("retry", "amend", "abort")
            ):
                raise ValueError("input review changed its correction scope")
            requests.append(
                InputReviewRequest(
                    request_event_id=event["event_id"], request=request, pending=True
                )
            )
        elif kind == "InputInspectionReviewAnswered":
            answer = InputInspectionReviewAnswered.model_validate(
                {"event_type": kind, **event["payload"]}
            )
            pending, replay = evaluate_scoped_answer(
                requests,
                receipts,
                answer.request_event_id,
                answer.response,
                validate=validate_input_answer,
                refusal=lambda code, identity: ValueError(
                    f"input review {code}: {identity}"
                ),
            )
            if replay is not None or event["parent_ids"] != [answer.request_event_id]:
                raise ValueError("input review repeated its answer or changed lineage")
            if any(
                e["event_type"] == "ExecutionProposalSubmitted"
                for e in events[: events.index(event)]
            ):
                raise ValueError("input correction cannot amend reviewed execution")
            if answer.response.action == "amend":
                inputs = tuple(
                    i.model_copy(update={"registered_path": answer.response.value})
                    if i.role == answer.response.field
                    else i
                    for i in inputs
                )
            InspectInputMetadataInput(input_set_id=fact.input_set_id, inputs=inputs)
            selection_id = event["event_id"]
            correction_pending = answer.response.action != "abort"
            stopped = not correction_pending
            requests = [
                r.model_copy(update={"pending": False})
                if r.request_event_id == answer.request_event_id
                else r
                for r in requests
            ]
            receipts.append(
                InputReviewReceipt(
                    request_event_id=answer.request_event_id,
                    response_event_id=event["event_id"],
                    response=answer.response,
                )
            )
    return InputSelection(
        input_set_id=fact.input_set_id,
        source_event_id=selection_id,
        inputs=inputs,
        fastq_roles=tuple(i.role for i in fact.inputs),
        inspections=tuple(inspections),
        requests=tuple(requests),
        receipts=tuple(receipts),
        correction_pending=correction_pending,
        stopped=stopped,
    )


def answer_input_review(journey, request_event_id, response):
    """Use the shared junction decision; validate amendments before the authority write."""
    from bio_neb.junction import evaluate_scoped_answer

    state = journey.checked_view("input_selection")
    pending, replay = evaluate_scoped_answer(
        state.requests,
        state.receipts,
        request_event_id,
        response,
        validate=validate_input_answer,
        refusal=lambda code, identity: ValueError(f"input review {code}: {identity}"),
    )
    if replay is not None:
        return replay
    return record_fact(
        journey,
        InputInspectionReviewAnswered(
            request_event_id=request_event_id, response=response
        ),
        (request_event_id,),
    )
