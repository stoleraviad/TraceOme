"""Journey-owned contracts and pure legality for sequential Planning HITL."""

from __future__ import annotations

import copy
import re
import uuid
from functools import cache
from typing import Annotated, Any, Iterable, Literal, Mapping, Tuple, Union

from pydantic import (
    Field,
    JsonValue,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.contracts import FrozenContract
from bio_neb.journey import JourneyEvent, ReadOnlyJsonObject, read_only_json
from bio_neb.junction import evaluate_scoped_answer

PlanningHitlAction = Literal["confirm", "select", "clarify", "reject", "stop"]
PlanningHitlRefusalCode = Literal[
    "planning_hitl_response_stale",
    "planning_hitl_response_conflict",
    "planning_hitl_response_illegal",
]

_PLAN_ID = re.compile(r"^plan_[0-9a-f]{64}$")
_REQUIREMENT_ID = re.compile(r"^requirement_[0-9a-f]{32}$")
_SUBJECT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


def _canonical_uuid(value: str, name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be canonical")
    return value


def _canonical_text(value: str, name: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


class _PlanRequirementLocator(FrozenContract):
    plan_event_id: StrictStr
    plan_id: StrictStr
    requirement_id: StrictStr
    subject_key: StrictStr

    @field_validator("plan_event_id")
    @classmethod
    def canonical_plan_event(cls, value: str) -> str:
        return _canonical_uuid(value, "plan_event_id")

    @field_validator("plan_id")
    @classmethod
    def canonical_plan(cls, value: str) -> str:
        if not _PLAN_ID.fullmatch(value):
            raise ValueError("plan_id must be a canonical plan identity")
        return value

    @field_validator("requirement_id")
    @classmethod
    def canonical_requirement(cls, value: str) -> str:
        if not _REQUIREMENT_ID.fullmatch(value):
            raise ValueError("requirement_id must be a canonical requirement identity")
        return value

    @field_validator("subject_key")
    @classmethod
    def canonical_subject(cls, value: str) -> str:
        if not _SUBJECT_KEY.fullmatch(value):
            raise ValueError("subject_key must be lowercase snake case")
        return value


class PlanningHitlCandidate(FrozenContract):
    """One exact candidate presented for researcher selection, not a verified fact."""

    candidate_id: StrictStr = Field(max_length=300)
    asserted_value: StrictStr = Field(max_length=2_000)
    display_label: StrictStr = Field(max_length=300)
    evidence_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("candidate_id", "asserted_value", "display_label")
    @classmethod
    def canonical_text(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @field_validator("evidence_event_ids")
    @classmethod
    def canonical_evidence(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("candidate evidence event identities must be unique")
        for value in values:
            _canonical_uuid(value, "evidence_event_id")
        return values


class PlanningHitlCandidateRecommendation(FrozenContract):
    kind: Literal["candidate"] = "candidate"
    candidate_id: StrictStr = Field(max_length=300)
    evidence_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("candidate_id")
    @classmethod
    def canonical_candidate(cls, value: str) -> str:
        return _canonical_text(value, "candidate_id")

    @field_validator("evidence_event_ids")
    @classmethod
    def canonical_evidence(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("recommendation evidence event identities must be unique")
        for value in values:
            _canonical_uuid(value, "evidence_event_id")
        return values


class PlanningHitlFactRecommendation(FrozenContract):
    kind: Literal["fact"] = "fact"
    asserted_value: StrictStr = Field(max_length=2_000)
    evidence_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("asserted_value")
    @classmethod
    def canonical_value(cls, value: str) -> str:
        return _canonical_text(value, "asserted_value")

    @field_validator("evidence_event_ids")
    @classmethod
    def canonical_evidence(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("recommendation evidence event identities must be unique")
        for value in values:
            _canonical_uuid(value, "evidence_event_id")
        return values


PlanningHitlRecommendation = Annotated[
    Union[PlanningHitlCandidateRecommendation, PlanningHitlFactRecommendation],
    Field(discriminator="kind"),
]


class PlanningHitlRequested(_PlanRequirementLocator):
    """A backend-projectable review junction with one exact retained owner."""

    event_type: Literal["PlanningHitlRequested"] = "PlanningHitlRequested"
    attempt_id: StrictStr
    owner: Literal["agent_attempt", "requirement_resolution", "assembly_suggestion"] = (
        Field(
            default="agent_attempt",
            exclude_if=lambda value: value == "agent_attempt",
        )
    )
    issue: StrictStr = Field(max_length=1_000)
    question: StrictStr = Field(max_length=2_000)
    evidence_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)
    candidates: Tuple[PlanningHitlCandidate, ...] = ()
    recommendation: PlanningHitlRecommendation | None = None

    @field_validator("attempt_id")
    @classmethod
    def canonical_attempt(cls, value: str) -> str:
        return _canonical_uuid(value, "attempt_id")

    @field_validator("issue", "question")
    @classmethod
    def canonical_text(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @field_validator("evidence_event_ids")
    @classmethod
    def canonical_evidence(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("request evidence event identities must be unique")
        for value in values:
            _canonical_uuid(value, "evidence_event_id")
        return values

    @model_validator(mode="after")
    def exact_candidates_and_recommendation(self) -> "PlanningHitlRequested":
        candidates = {
            candidate.candidate_id: candidate for candidate in self.candidates
        }
        if len(candidates) != len(self.candidates):
            raise ValueError("Planning HITL candidate identities must be unique")
        request_evidence = set(self.evidence_event_ids)
        if any(
            not set(candidate.evidence_event_ids) <= request_evidence
            for candidate in self.candidates
        ):
            raise ValueError("candidate evidence must be included in request evidence")
        if self.recommendation is not None:
            if not set(self.recommendation.evidence_event_ids) <= request_evidence:
                raise ValueError(
                    "recommendation evidence must be included in request evidence"
                )
            if isinstance(self.recommendation, PlanningHitlCandidateRecommendation):
                candidate = candidates.get(self.recommendation.candidate_id)
                if candidate is None:
                    raise ValueError(
                        "recommended candidate must be presented by the request"
                    )
                if (
                    candidate.evidence_event_ids
                    != self.recommendation.evidence_event_ids
                ):
                    raise ValueError("recommended candidate evidence must remain exact")
        return self


class ConfirmPlanningHitlResponse(FrozenContract):
    action: Literal["confirm"]


class SelectPlanningHitlResponse(FrozenContract):
    action: Literal["select"]
    candidate_id: StrictStr = Field(max_length=300)
    instruction: StrictStr | None = Field(default=None, max_length=2_000)

    @field_validator("candidate_id")
    @classmethod
    def canonical_candidate(cls, value: str) -> str:
        return _canonical_text(value, "candidate_id")

    @field_validator("instruction")
    @classmethod
    def canonical_instruction(cls, value: str | None) -> str | None:
        return value if value is None else _canonical_text(value, "instruction")


class ClarifyPlanningHitlResponse(FrozenContract):
    action: Literal["clarify"]
    instruction: StrictStr = Field(min_length=1, max_length=2_000)

    @field_validator("instruction")
    @classmethod
    def canonical_instruction(cls, value: str) -> str:
        return _canonical_text(value, "instruction")


class _ReasonResponse(FrozenContract):
    action: str
    reason: StrictStr = Field(min_length=1, max_length=2_000)

    @field_validator("reason")
    @classmethod
    def canonical_reason(cls, value: str) -> str:
        return _canonical_text(value, "reason")


class StopPlanningHitlResponse(_ReasonResponse):
    """Researcher confirmation that required evidence cannot be supplied."""

    action: Literal["stop"]


class RejectPlanningHitlResponse(_ReasonResponse):
    """Resolve only this scoped request, retaining its issue and prior evidence."""

    action: Literal["reject"]


PlanningHitlResponse = Annotated[
    Union[
        ConfirmPlanningHitlResponse,
        SelectPlanningHitlResponse,
        ClarifyPlanningHitlResponse,
        StopPlanningHitlResponse,
        RejectPlanningHitlResponse,
    ],
    Field(discriminator="action"),
]


class PlanningHitlResponded(FrozenContract):
    """One accepted response; its information remains an unverified user assertion."""

    event_type: Literal["PlanningHitlResponded"] = "PlanningHitlResponded"
    request_event_id: StrictStr
    response: PlanningHitlResponse
    authority: Literal["user_assertion"] = "user_assertion"
    scientifically_verified: Literal[False] = False
    authorizes_execution: Literal[False] = False

    @field_validator("request_event_id")
    @classmethod
    def canonical_request(cls, value: str) -> str:
        return _canonical_uuid(value, "request_event_id")


def assembly_question_ids(journey_id: str, suggestion_event_id: str) -> tuple[str, str]:
    """The coordinator's question on one suggestion of another assembly (#459 D4): its
    Attempt id and its event id, derived from the suggestion it presents."""
    return tuple(
        str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{journey_id}:{suggestion_event_id}:{kind}")
        )
        for kind in ("assembly-question", "assembly-review")
    )


def _validate_resolution_question(event, event_id, request, parent_ids, by_id) -> None:
    """A question the coordinator opens on an unsettled target (#342): it owns a
    scientific-investigation requirement, descends from one causal head and has its
    derived identity."""
    plan = by_id[request.plan_event_id]["payload"]["plan"]
    matches = [
        item
        for item in plan.get("requirements", [])
        if item.get("requirement_id") == request.requirement_id
    ]
    if len(matches) != 1 or matches[0].get("kind") != "scientific_investigation":
        raise ValueError("requirement-resolution Review does not own a target")
    if len(parent_ids) != 1:
        raise ValueError("requirement-resolution Review lacks one causal head")
    expected_attempt_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{event['journey_id']}:{request.requirement_id}:"
                f"{parent_ids[0]}:resolution"
            ),
        )
    )
    expected_request_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (f"{event['journey_id']}:{request.requirement_id}:{parent_ids[0]}:review"),
        )
    )
    if request.attempt_id != expected_attempt_id or event_id != expected_request_id:
        raise ValueError("requirement-resolution Review identity is not exact")


def _validate_assembly_question(event, request, parent_ids, by_id) -> None:
    """A question the coordinator opens on a suggestion of another assembly's index
    (#459 D4, with no model turn): it owns the reference requirement, presents exactly
    one suggestion, descends from one causal head and has its derived identity."""
    plan = by_id[request.plan_event_id]["payload"]["plan"]
    matches = [
        item
        for item in plan["requirements"]
        if item["requirement_id"] == request.requirement_id
    ]
    suggestions = [
        evidence
        for evidence in request.evidence_event_ids
        if by_id[evidence]["event_type"] == "AssemblyWithIndexSuggested"
    ]
    if (
        len(matches) != 1
        or matches[0]["kind"] != "reference_assembly_identity"
        or len(suggestions) != 1
        or len(parent_ids) != 1
    ):
        raise ValueError("an assembly question presents no one suggestion")
    if (request.attempt_id, event["event_id"]) != assembly_question_ids(
        event["journey_id"], suggestions[0]
    ):
        raise ValueError("an assembly question's identity is not exact")


class PlanningHitlLegalActionView(FrozenContract):
    action: PlanningHitlAction
    response_schema: ReadOnlyJsonObject


class PlanningHitlRequestView(_PlanRequirementLocator):
    request_event_id: StrictStr
    attempt_id: StrictStr
    owner: Literal["agent_attempt", "requirement_resolution", "assembly_suggestion"] = (
        Field(
            default="agent_attempt",
            exclude_if=lambda value: value == "agent_attempt",
        )
    )
    issue: StrictStr
    question: StrictStr
    evidence_event_ids: Tuple[StrictStr, ...]
    candidates: Tuple[PlanningHitlCandidate, ...]
    recommendation: PlanningHitlRecommendation | None
    legal_actions: Tuple[PlanningHitlLegalActionView, ...] = Field(min_length=1)
    pending: bool

    @field_validator("request_event_id", "attempt_id")
    @classmethod
    def canonical_identity(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)


class PlanningHitlUserAssertion(_PlanRequirementLocator):
    request_event_id: StrictStr
    response_event_id: StrictStr
    action: PlanningHitlAction
    selected_candidate_id: StrictStr | None = None
    asserted_value: StrictStr
    instruction: StrictStr | None = None
    evidence_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)
    authority: Literal["user_assertion"] = "user_assertion"
    scientifically_verified: Literal[False] = False
    authorizes_execution: Literal[False] = False

    @field_validator("request_event_id", "response_event_id")
    @classmethod
    def canonical_identity(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)


class PlanningHitlResponseReceipt(_PlanRequirementLocator):
    status: Literal["accepted"] = "accepted"
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: PlanningHitlResponse
    assertion: PlanningHitlUserAssertion

    @field_validator("request_event_id", "response_event_id")
    @classmethod
    def canonical_identity(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)

    @model_validator(mode="after")
    def exact_receipt(self) -> "PlanningHitlResponseReceipt":
        if (
            self.assertion.request_event_id != self.request_event_id
            or self.assertion.response_event_id != self.response_event_id
            or self.assertion.plan_event_id != self.plan_event_id
            or self.assertion.plan_id != self.plan_id
            or self.assertion.requirement_id != self.requirement_id
            or self.assertion.subject_key != self.subject_key
            or self.assertion.action != self.response.action
        ):
            raise ValueError("Planning HITL receipt identity changed")
        return self


def _presentation_request(requests):
    pending = [r for r in requests if r.pending]
    return next(
        (r for r in pending if r.owner == "agent_attempt"),
        pending[0] if pending else None,
    )


class PlanningHitlProjection(FrozenContract):
    requests: Tuple[PlanningHitlRequestView, ...]
    receipts: Tuple[PlanningHitlResponseReceipt, ...]
    pending_request: PlanningHitlRequestView | None
    run_aborted: bool = False

    def for_requirements(self, requirement_ids: set[str]) -> "PlanningHitlProjection":
        """An isolated domain view; a sibling question grants or blocks no authority."""
        requests = tuple(
            r for r in self.requests if r.requirement_id in requirement_ids
        )
        request_ids = {r.request_event_id for r in requests}
        return PlanningHitlProjection(
            requests=requests,
            receipts=tuple(
                r for r in self.receipts if r.request_event_id in request_ids
            ),
            pending_request=_presentation_request(requests),
            run_aborted=self.run_aborted,
        )

    @property
    def pending_requests(self) -> Tuple[PlanningHitlRequestView, ...]:
        return tuple(request for request in self.requests if request.pending)

    @model_validator(mode="after")
    def exact_pending_selection(self) -> "PlanningHitlProjection":
        pending = tuple(request for request in self.requests if request.pending)
        if _presentation_request(pending) != self.pending_request:
            raise ValueError("Planning HITL pending projection is inconsistent")
        return self


class PlanningHitlResponseDisposition(FrozenContract):
    status: Literal["accept", "identical"]
    retained_receipt: PlanningHitlResponseReceipt | None = None

    @model_validator(mode="after")
    def retained_only_when_identical(self) -> "PlanningHitlResponseDisposition":
        if (self.status == "identical") != (self.retained_receipt is not None):
            raise ValueError("only an identical response has a retained receipt")
        return self


class PlanningHitlResponseRefused(ValueError):
    def __init__(self, code: PlanningHitlRefusalCode, request_event_id: str):
        super().__init__(code)
        self.code = code
        self.request_event_id = request_event_id
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


@cache
def _action_schema(action: PlanningHitlAction) -> dict[str, JsonValue]:
    """The response model's validation schema, fixed by code; callers copy it."""
    owner = {
        "confirm": ConfirmPlanningHitlResponse,
        "select": SelectPlanningHitlResponse,
        "clarify": ClarifyPlanningHitlResponse,
        "stop": StopPlanningHitlResponse,
        "reject": RejectPlanningHitlResponse,
    }[action]
    return read_only_json(owner.model_json_schema(mode="validation"))


def _response_schema(
    action: PlanningHitlAction, candidates: Tuple[PlanningHitlCandidate, ...]
) -> dict[str, JsonValue]:
    schema = copy.deepcopy(_action_schema(action))
    schema.pop("title", None)
    if action == "select":
        schema["properties"]["candidate_id"] = {
            "enum": [candidate.candidate_id for candidate in candidates],
            "type": "string",
        }
    return schema


def planning_hitl_request_view(
    request_event_id: str,
    request: PlanningHitlRequested,
    *,
    pending: bool,
) -> PlanningHitlRequestView:
    """Derive exact legal actions and schemas from one retained request."""

    _canonical_uuid(request_event_id, "request_event_id")
    actions: list[PlanningHitlAction] = []
    if request.recommendation is not None:
        actions.append("confirm")
    if request.candidates:
        actions.append("select")
    actions.extend(("clarify", "reject", "stop"))
    return PlanningHitlRequestView(
        request_event_id=request_event_id,
        attempt_id=request.attempt_id,
        owner=request.owner,
        plan_event_id=request.plan_event_id,
        plan_id=request.plan_id,
        requirement_id=request.requirement_id,
        subject_key=request.subject_key,
        issue=request.issue,
        question=request.question,
        evidence_event_ids=request.evidence_event_ids,
        candidates=request.candidates,
        recommendation=request.recommendation,
        legal_actions=tuple(
            PlanningHitlLegalActionView(
                action=action,
                response_schema=_response_schema(action, request.candidates),
            )
            for action in actions
        ),
        pending=pending,
    )


def make_planning_hitl_responded(
    request_event_id: str, response: PlanningHitlResponse
) -> PlanningHitlResponded:
    """Construct the only Journey response fact accepted by the shared handler. Code
    reads no value out of the answer: the stage that uses it types it (#472 Q2 a)."""

    return PlanningHitlResponded(request_event_id=request_event_id, response=response)


def _answer(
    request: PlanningHitlRequestView, response: PlanningHitlResponse
) -> tuple[str, str | None, str | None, Tuple[str, ...]]:
    """The asserted value, selected candidate, instruction and evidence of an answer."""

    candidates = {candidate.candidate_id: candidate for candidate in request.candidates}
    if isinstance(response, ConfirmPlanningHitlResponse):
        recommendation = request.recommendation
        assert recommendation is not None
        if isinstance(recommendation, PlanningHitlCandidateRecommendation):
            candidate = candidates[recommendation.candidate_id]
            return (
                candidate.asserted_value,
                candidate.candidate_id,
                None,
                recommendation.evidence_event_ids,
            )
        return (
            recommendation.asserted_value,
            None,
            None,
            recommendation.evidence_event_ids,
        )
    if isinstance(response, SelectPlanningHitlResponse):
        candidate = candidates[response.candidate_id]
        return (
            candidate.asserted_value,
            candidate.candidate_id,
            response.instruction,
            candidate.evidence_event_ids,
        )
    if isinstance(response, ClarifyPlanningHitlResponse):
        return (
            response.instruction,
            None,
            response.instruction,
            request.evidence_event_ids,
        )
    assert isinstance(response, (StopPlanningHitlResponse, RejectPlanningHitlResponse))
    return response.reason, None, response.reason, request.evidence_event_ids


def _assertion(
    request: PlanningHitlRequestView,
    response_event_id: str,
    responded: PlanningHitlResponded,
) -> PlanningHitlUserAssertion:
    response = responded.response
    asserted_value, selected_candidate_id, instruction, evidence_event_ids = _answer(
        request, response
    )
    return PlanningHitlUserAssertion(
        request_event_id=request.request_event_id,
        response_event_id=response_event_id,
        plan_event_id=request.plan_event_id,
        plan_id=request.plan_id,
        requirement_id=request.requirement_id,
        subject_key=request.subject_key,
        action=response.action,
        selected_candidate_id=selected_candidate_id,
        asserted_value=asserted_value,
        instruction=instruction,
        evidence_event_ids=evidence_event_ids,
    )


def planning_hitl_response_receipt(
    request: PlanningHitlRequestView,
    response_event_id: str,
    responded: PlanningHitlResponded,
) -> PlanningHitlResponseReceipt:
    """Project one response event as its durable application receipt."""

    validate_planning_hitl_response(request, responded.response)
    return PlanningHitlResponseReceipt(
        request_event_id=request.request_event_id,
        response_event_id=response_event_id,
        plan_event_id=request.plan_event_id,
        plan_id=request.plan_id,
        requirement_id=request.requirement_id,
        subject_key=request.subject_key,
        response=responded.response,
        assertion=_assertion(request, response_event_id, responded),
    )


def validate_planning_hitl_response(
    request: PlanningHitlRequestView, response: PlanningHitlResponse
) -> None:
    """Apply the request-derived legality represented by the backend schemas."""

    legal = {item.action for item in request.legal_actions}
    if response.action not in legal:
        raise PlanningHitlResponseRefused(
            "planning_hitl_response_illegal", request.request_event_id
        )
    if isinstance(
        response, SelectPlanningHitlResponse
    ) and response.candidate_id not in {
        candidate.candidate_id for candidate in request.candidates
    }:
        raise PlanningHitlResponseRefused(
            "planning_hitl_response_illegal", request.request_event_id
        )


def evaluate_planning_hitl_response(
    projection: PlanningHitlProjection,
    request_event_id: str,
    response: PlanningHitlResponse,
) -> PlanningHitlResponseDisposition:
    """Classify a proposed response without graph state or Journey mutation."""

    _canonical_uuid(request_event_id, "request_event_id")
    _, retained = evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=validate_planning_hitl_response,
        refusal=lambda reason, identity: PlanningHitlResponseRefused(
            f"planning_hitl_response_{reason}", identity
        ),
    )
    if retained is not None:
        return PlanningHitlResponseDisposition(
            status="identical", retained_receipt=retained
        )
    if projection.run_aborted:
        raise PlanningHitlResponseRefused(
            "planning_hitl_response_not_pending", request_event_id
        )
    return PlanningHitlResponseDisposition(status="accept")


def _as_event(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def _fact(event: Mapping[str, Any], owner: type[FrozenContract]) -> FrozenContract:
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


def _validate_plan_requirement(
    request: PlanningHitlRequested,
    *,
    ancestors: set[str],
    by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    if request.plan_event_id not in ancestors:
        raise ValueError("Planning HITL plan event is outside its ancestry")
    plan_event = by_id[request.plan_event_id]
    if plan_event.get("event_type") != "ProposalRequirementsIdentified":
        raise ValueError("Planning HITL plan event has the wrong fact type")
    payload = plan_event.get("payload")
    plan = payload.get("plan") if isinstance(payload, Mapping) else None
    if not isinstance(plan, Mapping) or plan.get("plan_id") != request.plan_id:
        raise ValueError("Planning HITL request changed the immutable plan identity")
    requirements = plan.get("requirements")
    if not isinstance(requirements, list):
        raise ValueError("Planning HITL plan lacks its requirement collection")
    matches = [
        requirement
        for requirement in requirements
        if isinstance(requirement, Mapping)
        and requirement.get("requirement_id") == request.requirement_id
    ]
    if len(matches) != 1:
        raise ValueError("Planning HITL request lacks its exact plan requirement")
    if matches[0].get("subject_key") != request.subject_key:
        raise ValueError("Planning HITL request changed the requirement subject")


def project_planning_hitl(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> PlanningHitlProjection:
    """Fold requests and responses from Journey evidence without live dependencies."""

    timeline = [_as_event(event) for event in events]
    by_id: dict[str, dict[str, Any]] = {}
    ancestors_by_id: dict[str, set[str]] = {}
    requests: dict[str, PlanningHitlRequested] = {}
    responses: dict[str, tuple[str, PlanningHitlResponded]] = {}
    request_order: list[str] = []
    pending_request_ids: set[str] = set()
    run_aborted = False

    for event in timeline:
        event_id = _canonical_uuid(event.get("event_id"), "event_id")
        if event_id in by_id:
            raise ValueError("Journey repeats an event identity")
        parent_ids = event.get("parent_ids")
        if not isinstance(parent_ids, list):
            raise ValueError("Journey parent identities must be a list")
        if any(parent_id not in by_id for parent_id in parent_ids):
            raise ValueError("Journey event has a non-backward parent")
        ancestors = set(parent_ids)
        for parent_id in parent_ids:
            ancestors.update(ancestors_by_id[parent_id])
        ancestors_by_id[event_id] = ancestors
        by_id[event_id] = event

        if event.get("event_type") == "RunFailed":
            run_aborted = True
        if run_aborted and event.get("event_type") in {
            "PlanningHitlRequested",
            "PlanningHitlResponded",
        }:
            raise ValueError("Planning Review cannot continue an aborted Run")
        if event.get("event_type") == "PlanningHitlRequested":
            request = _fact(event, PlanningHitlRequested)
            if any(
                (requests[p].plan_id, requests[p].requirement_id)
                == (request.plan_id, request.requirement_id)
                for p in pending_request_ids
            ):
                raise ValueError(
                    "Planning HITL request created while another request is pending for this scope"
                )
            assert isinstance(request, PlanningHitlRequested)
            _validate_plan_requirement(request, ancestors=ancestors, by_id=by_id)
            if not set(request.evidence_event_ids) <= ancestors:
                raise ValueError(
                    "Planning HITL request evidence is outside its ancestry"
                )
            if request.owner == "agent_attempt":
                attempt_starts = [
                    ancestor_id
                    for ancestor_id in ancestors
                    if by_id[ancestor_id].get("event_type") == "AttemptStarted"
                    and by_id[ancestor_id].get("payload", {}).get("attempt_id")
                    == request.attempt_id
                ]
                if len(attempt_starts) != 1:
                    raise ValueError(
                        "Planning HITL request lacks its exact owning Attempt"
                    )
            elif request.owner == "assembly_suggestion":
                _validate_assembly_question(event, request, parent_ids, by_id)
            else:
                _validate_resolution_question(
                    event, event_id, request, parent_ids, by_id
                )
            requests[event_id] = request
            request_order.append(event_id)
            pending_request_ids.add(event_id)
        elif event.get("event_type") == "PlanningHitlResponded":
            responded = _fact(event, PlanningHitlResponded)
            assert isinstance(responded, PlanningHitlResponded)
            request_id = responded.request_event_id
            if request_id not in pending_request_ids or request_id not in requests:
                raise ValueError(
                    "Planning HITL response does not target the exact pending request"
                )
            if request_id not in ancestors:
                raise ValueError(
                    "Planning HITL response is outside its request ancestry"
                )
            if request_id in responses:
                raise ValueError("Planning HITL request has more than one response")
            request_view = planning_hitl_request_view(
                request_id, requests[request_id], pending=True
            )
            validate_planning_hitl_response(request_view, responded.response)
            responses[request_id] = (event_id, responded)
            pending_request_ids.remove(request_id)

    request_views = tuple(
        planning_hitl_request_view(
            request_id,
            requests[request_id],
            pending=request_id in pending_request_ids,
        )
        for request_id in request_order
    )
    views_by_id = {view.request_event_id: view for view in request_views}
    receipts = tuple(
        planning_hitl_response_receipt(
            views_by_id[request_id], response_event_id, responded
        )
        for request_id in request_order
        if (retained := responses.get(request_id)) is not None
        for response_event_id, responded in (retained,)
    )
    return PlanningHitlProjection(
        requests=request_views,
        receipts=receipts,
        pending_request=_presentation_request(request_views),
        run_aborted=run_aborted,
    )


__all__ = [
    "ClarifyPlanningHitlResponse",
    "ConfirmPlanningHitlResponse",
    "PlanningHitlCandidate",
    "PlanningHitlCandidateRecommendation",
    "PlanningHitlFactRecommendation",
    "PlanningHitlLegalActionView",
    "PlanningHitlProjection",
    "PlanningHitlRequestView",
    "PlanningHitlRequested",
    "PlanningHitlResponded",
    "PlanningHitlResponse",
    "PlanningHitlResponseDisposition",
    "PlanningHitlResponseReceipt",
    "PlanningHitlResponseRefused",
    "PlanningHitlUserAssertion",
    "SelectPlanningHitlResponse",
    "StopPlanningHitlResponse",
    "RejectPlanningHitlResponse",
    "evaluate_planning_hitl_response",
    "make_planning_hitl_responded",
    "planning_hitl_request_view",
    "planning_hitl_response_receipt",
    "project_planning_hitl",
    "validate_planning_hitl_response",
]
