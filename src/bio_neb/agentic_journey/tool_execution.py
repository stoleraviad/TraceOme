"""Linked admission and mechanical execution observation for typed Tools."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from bio_neb.agentic_journey.action_admission import (
    request_bound_arguments,
    validated_model_call,
)
from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.lifecycle import (
    Activity,
    ActivityCancelled,
    ActivityFailed,
    ActivityStarted,
    ActivitySucceeded,
    validate_activity_fact,
)
from bio_neb.agentic_journey.role_declaration import (
    ROLES,
    AgentRole,
    AttemptClosure,
    role,
)
from bio_neb.content import ContentId, shown_to_model
from bio_neb.failure import (
    TOOL_FAILED,
    FailureClass,
    FailureSurface,
    Invariant,
    ProviderError,
    ResolvedFailurePolicy,
    ToolRefused,
    operation_retry_decision,
    tool_failure_holds_at_once,
    tool_fault_key,
)
from bio_neb.journey import canonical_json_digest
from bio_neb.providers.source_operations import SourceSurface
from bio_neb.recording import (
    ProcessingPass,
    contact_evidence_anchor,
    current_processing_pass,
    is_contact_evidence,
    record_contact_evidence,
)

if TYPE_CHECKING:
    from bio_neb.agentic_journey.action_admission import (
        ActionAdmissionSession,
        ProposedAgentAction,
        TurnDisposition,
    )


T = TypeVar("T")
_JSON_VALUE = TypeAdapter(Any)


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


def _canonical_digest(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("digest must be lowercase sha256")
    return value


class _ObservedToolFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ToolOperationFailureRecorded(_ObservedToolFact):
    event_type: Literal["ToolOperationFailureRecorded"] = "ToolOperationFailureRecorded"
    started_event_id: StrictStr
    policy: ResolvedFailurePolicy
    contact_attempt: int = Field(ge=1)
    failure_class: FailureClass
    original_failure_class: FailureClass
    detail: StrictStr
    retry_delay_seconds: float | None = Field(default=None, ge=0)


_operation_owner: ContextVar[tuple[ProcessingPass | None, str] | None] = ContextVar(
    "observed_tool_operation_owner", default=None
)


def record_tool_operation_failure(error, ordinal, delay, policy):
    """Retain a classified adapter attempt under its admitted Tool contact."""
    owner = _operation_owner.get()
    if owner is not None:
        interval, started_event_id = owner
        if interval is None or current_processing_pass() is not interval:
            raise RuntimeError("Tool operation lost its owning processing Pass")
        record_contact_evidence(
            interval.journey,
            ToolOperationFailureRecorded(
                started_event_id=started_event_id,
                policy=policy,
                contact_attempt=ordinal,
                failure_class=error.failure_class,
                original_failure_class=error.evidence["original_failure_class"],
                detail=error.detail,
                retry_delay_seconds=delay,
            ),
        )


class ObservedToolExecutionStarted(_ObservedToolFact):
    event_type: Literal["ObservedToolExecutionStarted"] = "ObservedToolExecutionStarted"
    attempt_id: StrictStr
    response_event_id: StrictStr
    validation_event_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    adapter_identity: StrictStr
    continuation_request_event_id: StrictStr | None = None

    @field_validator("attempt_id", "tool_call_id", "tool_name", "adapter_identity")
    @classmethod
    def exact_nonempty_text(cls, value: str, info: Any) -> str:
        if not value or value.strip() != value:
            raise ValueError(f"{info.field_name} must be exact non-empty text")
        return value

    @field_validator("response_event_id", "validation_event_id")
    @classmethod
    def canonical_event_reference(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)


class ObservedToolExecutionSucceeded(_ObservedToolFact):
    event_type: Literal["ObservedToolExecutionSucceeded"] = (
        "ObservedToolExecutionSucceeded"
    )
    attempt_id: StrictStr
    started_event_id: StrictStr
    stage_result_event_id: StrictStr
    model_visible_result_digest: StrictStr
    artifact_digests: tuple[StrictStr, ...]
    artifact_source_event_ids: tuple[StrictStr, ...]
    # The model was shown the result with its control characters as spaces
    # (``content.shown_to_model``), and the digest covers that view; written only when
    # true, so a result recorded before the guard reopens as it was.
    result_guarded: StrictBool = Field(
        default=False, exclude_if=lambda value: not value
    )

    @field_validator("attempt_id")
    @classmethod
    def exact_attempt_id(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("attempt_id must be exact non-empty text")
        return value

    @field_validator("started_event_id", "stage_result_event_id")
    @classmethod
    def canonical_event_reference(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)

    @field_validator("model_visible_result_digest")
    @classmethod
    def canonical_result_digest(cls, value: str) -> str:
        return _canonical_digest(value)

    @field_validator("artifact_source_event_ids")
    @classmethod
    def canonical_artifact_sources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _canonical_uuid(value, "artifact_source_event_id") for value in values
        )

    @field_validator("artifact_digests")
    @classmethod
    def canonical_artifact_digests(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("artifact digests must be unique")
        return tuple(_canonical_digest(value) for value in values)


class ObservedToolExecutionFailed(_ObservedToolFact):
    event_type: Literal["ObservedToolExecutionFailed"] = "ObservedToolExecutionFailed"
    attempt_id: StrictStr
    started_event_id: StrictStr
    mechanical_error_type: StrictStr
    stage_failure_event_id: StrictStr | None = None

    @field_validator("attempt_id", "mechanical_error_type")
    @classmethod
    def exact_nonempty_text(cls, value: str, info: Any) -> str:
        if not value or value.strip() != value:
            raise ValueError(f"{info.field_name} must be exact non-empty text")
        return value

    @field_validator("started_event_id", "stage_failure_event_id")
    @classmethod
    def canonical_event_reference(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _canonical_uuid(value, info.field_name)


@dataclass(frozen=True)
class ShownResult:
    """A Tool's return: the model view it shows and the recorded event the view is
    built from (#478 G1.3). The assembly returns it as the framework's
    ``ToolReturn``: the view is sent, the event id is kept beside it and never sent."""

    view: Mapping[str, Any]
    event_id: str


class ToolExecutionRefusedResult(Exception):
    """Carries a refused execution's model view out of the domain Tool to the
    assembly, which returns it as the call's result, its recorded refusal's id
    beside it (#478 G1.3)."""

    def __init__(self, view: Mapping[str, Any], event_id: str):
        super().__init__(view["statement"])
        self.shown = ShownResult(dict(view), event_id)


def _no_stage_failure(_error: BaseException) -> Mapping[str, Any] | None:
    return None


@dataclass(frozen=True)
class ToolExecutionAdapter(Generic[T]):
    """Stage-owned evidence bridge around one typed Tool operation."""

    identity: str
    stage_result_event: Callable[[T], Mapping[str, Any]]
    artifact_digests: Callable[[T], Sequence[str]]
    operational_failure: Callable[[T], ProviderError | None] = lambda _result: None
    stage_failure_event: Callable[[BaseException], Mapping[str, Any] | None] = (
        _no_stage_failure
    )


# Keys a model view never shows, at any depth: hashes, sizes, timings, costs and
# internal ids. The Journey keeps them with the whole fact.
_INTERNAL_KEYS = frozenset(
    {
        "assessment",
        "assessment_id",
        "attempt_id",
        "byte_count",
        "cost_usd",
        "elapsed_time_ms",
        "evidence_id",
        "grounding",
        "observed_at",
        "raw_artifact_id",
        "raw_response_byte_count",
        "raw_response_sha256",
        "result_id",
        "sha256",
        "source",
        "source_id",
        "tool_call_id",
    }
)


def _without_internal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_internal(item)
            for key, item in value.items()
            if key not in _INTERNAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_internal(item) for item in value]
    return value


def _result_status(payload: Mapping[str, Any]) -> str:
    return payload["result"]["status"]


def _no_statement(_payload: Mapping[str, Any]) -> str | None:
    return None


def _copied_result(payload: Mapping[str, Any]) -> Any:
    return {
        key: value
        for key, value in payload["result"].items()
        if key not in ("status", "summary")
    }


def _nothing_more(_payload: Mapping[str, Any]) -> Any:
    return {}


def _recorded(_payload: Mapping[str, Any]) -> str:
    return "recorded"


def _asked(_payload: Mapping[str, Any]) -> str:
    return "asked"


@dataclass(frozen=True)
class ModelView:
    """What one Tool row shows the model of its recorded fact (#478 G1.3).

    The shared builder (``tool_model_view``) shapes every row's view alike: its
    ``status``; the ``statement`` of what happened, in the source's words when it
    has any; the fact's copied ``result`` and each source answer's own body,
    without hashes, timings or internal ids; the ids the model cites (``cite``
    names the argument it cites this fact by); and one ``next`` step, read from
    the status. The Journey keeps the whole fact.
    """

    next_step: Mapping[str, str] | Callable[[str, Mapping[str, Any]], str]
    status: Callable[[Mapping[str, Any]], str] = _result_status
    statement: Callable[[Mapping[str, Any]], str | None] = _no_statement
    result: Callable[[Mapping[str, Any]], Any] = _copied_result
    cite: str | None = None
    # The Journey search shows recorded events as they are: it is the model's
    # reader of the Journey, so its view keeps their own fields.
    scrubbed: bool = True

    def next(self, status: str, payload: Mapping[str, Any]) -> str:
        if callable(self.next_step):
            return self.next_step(status, payload)
        return self.next_step[status if status in self.next_step else "*"]


def recorded_view(
    next_step: Mapping[str, str] | Callable[[str, Mapping[str, Any]], str],
    *,
    result: Callable[[Mapping[str, Any]], Any] = _nothing_more,
    cite: str | None = None,
) -> ModelView:
    """The view of a Tool that records the model's own choice or request."""
    return ModelView(
        next_step=next_step,
        status=_recorded,
        result=result,
        cite=cite,
    )


_RESEARCHER_ANSWERS = (
    "The Attempt ends at this question; the researcher's answer comes back as your "
    "next instruction."
)


def asked_view(next_step: str = _RESEARCHER_ANSWERS) -> ModelView:
    """The view of a Tool that asks the researcher: the Attempt ends at it."""
    return ModelView(
        next_step={"*": next_step},
        status=_asked,
        result=_nothing_more,
    )


def tool_model_view(view: ModelView, event: Mapping[str, Any]) -> dict[str, Any]:
    """The one shape every Tool's result reaches the model in, from its fact."""
    payload = event["payload"]
    status = view.status(payload)
    shown: dict[str, Any] = {"status": status}
    if (statement := view.statement(payload)) is not None:
        shown["statement"] = statement
    result = view.result(payload)
    if result := _without_internal(result) if view.scrubbed else result:
        shown["result"] = result
    if "source_answers" in payload:
        shown["source_answers"] = [
            {
                "operation": answer["operation"],
                "request": answer["request_identity"],
                "status_code": answer["status_code"],
                "shown": answer["shown"],
                "body": answer["body"],
            }
            for answer in payload["source_answers"]
        ]
    if view.cite is not None:
        shown["cite"] = {view.cite: event["event_id"]}
    shown["next"] = view.next(status, payload)
    return shown


# An owner's once-per-owner check (``Journey.derived_once``): key, then check.
CheckedOnce = Callable[[tuple[str, str], Callable[[], None]], None]
ToolCompatibilityDisposition = Literal[
    "legacy_tool_execution_failed_retained",
    "generic_mechanical_failure_only",
    "investigation_source_failure_retained",
]
ToolFailureErrorCorrelation = Literal[
    "matches_terminal_mechanical_error",
    "stage_error_preserved_then_wrapped_as_RuntimeError",
]


@dataclass(frozen=True)
class ToolPrecondition:
    """One declared reason admission refuses a stage Tool's call before contact.

    ``slot`` is the argument or checked state the reason is about; ``explanation``
    says in plain words why the call cannot run now and what to do instead.
    ``met_by`` names the Tools whose call can meet it, and ``unmet_by`` those
    whose call makes it unmet: in one reply, the first run before this Tool's
    call and the second after it (``dependency_order``).
    """

    code: str
    slot: str
    explanation: str
    met_by: tuple[str, ...] = ()
    unmet_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegativeResult:
    """A domain result that is negative but lawful: it goes back to the model.

    The Tool's recorded fact of ``event_type`` holds ``field`` with one of
    ``values`` somewhere in its payload. No terminal ends an Attempt on it; an
    operational fault is classified at the adapter edge instead (ADR 0031).
    """

    event_type: str
    field: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class FunctionToolContract:
    """One durable row in the complete observed Function Tool inventory.

    The row is also the Tool's registry entry for the default agentic loop: the
    coordinator stages that show it, few-shot examples rendered into its
    description, and the preconditions admission refuses it with. What the Tool is
    for is model wording: each role's instruction blocks state it (#442).
    """

    tool_name: str
    agent_role: AgentRole
    typed_input: tuple[str, ...]
    typed_output: str
    adapter_identity: str
    stage_result_event_type: str
    stage_failure_event_type: str | None
    stage_failure_error_field: str | None
    stage_failure_error_correlation: ToolFailureErrorCorrelation | None
    model_view: ModelView
    compatibility_disposition: ToolCompatibilityDisposition
    stages: tuple[str, ...] = ()
    # The source a lookup Tool reads and the assemblies it serves, shown to the model
    # before its purpose.
    source_tag: str | None = None
    examples: tuple[str, ...] = ()
    preconditions: tuple[ToolPrecondition, ...] = ()
    # A Tool that opens a researcher question: the Attempt ends at it, so the later
    # calls of its response are refused.
    opens_question: bool = False
    negative_results: tuple[NegativeResult, ...] = ()
    # A source Tool's contact surfaces and its own per-call limits; the failure
    # package's defaults apply where it declares none.
    source_surface: SourceSurface | None = None
    # Arguments that only shape how much of one result comes back (``max_hits``); a
    # Tool's own repeat check compares calls without them.
    result_shaping: tuple[str, ...] = ()
    lifecycle: Literal["ObservedToolSession"] = "ObservedToolSession"
    domain_event_ownership: Literal["stage_owned"] = "stage_owned"
    admission: Literal["each_call_admitted_before_execution"] = (
        "each_call_admitted_before_execution"
    )
    reopen: Literal["exact_return_or_attempt_terminal"] = (
        "exact_return_or_attempt_terminal"
    )

    @property
    def complementary_recognized_tool_names(self) -> frozenset[str]:
        """Return every recognized Tool owned by another role."""

        return function_tool_names() - function_tool_names(self.agent_role)


def model_visible(result: Any) -> tuple[Any, bool]:
    """A Tool's return as the model receives it, and whether the guard changed it.

    Every function Tool's return reaches the model through this one view
    (``PydanticActionAdmissionCapability.after_tool_execute``), and its digest
    is the one ``ObservedToolExecutionSucceeded`` records. A ``ShownResult`` is
    shown as its view; its event id is never sent.
    """
    if isinstance(result, ShownResult):
        shown, guarded = model_visible(result.view)
        return replace(result, view=shown), guarded
    recorded = _JSON_VALUE.dump_python(result, mode="json")
    shown = shown_to_model(recorded)
    return shown, shown != recorded


@cache
def _model_views() -> dict[str, ModelView]:
    return {item.tool_name: item.model_view for item in function_tool_contracts()}


@cache
def _result_shaping() -> dict[str, frozenset[str]]:
    shaping: dict[str, frozenset[str]] = {}
    for item in function_tool_contracts():
        shaping[item.tool_name] = shaping.get(item.tool_name, frozenset()) | frozenset(
            item.result_shaping
        )
    return shaping


def result_shaping_of(tool_name: str) -> frozenset[str]:
    """The arguments the Tool's row declares result-shaping; none for a Tool no row
    names."""
    return _result_shaping().get(tool_name, frozenset())


def model_view_of(tool_name: str) -> ModelView:
    """The model view the Tool's row declares; every role's row of one Tool
    declares the same view."""
    return _model_views()[tool_name]


def function_tool_contract(
    tool_name: str,
    agent_role: AgentRole,
    typed_input: tuple[str, ...],
    typed_output: str,
    adapter_identity: str,
    stage_result_event_type: str,
    stage_failure_event_type: str | None,
    stage_failure_error_field: str | None,
    model_view: ModelView,
    compatibility_disposition: ToolCompatibilityDisposition,
    stage_failure_error_correlation: ToolFailureErrorCorrelation = (
        "matches_terminal_mechanical_error"
    ),
    *,
    stages: tuple[str, ...] = (),
    source_tag: str | None = None,
    examples: tuple[str, ...] = (),
    preconditions: tuple[ToolPrecondition, ...] = (),
    opens_question: bool = False,
    negative_results: tuple[NegativeResult, ...] = (),
    result_shaping: tuple[str, ...] = (),
    source_surface: SourceSurface | None = None,
) -> FunctionToolContract:
    return FunctionToolContract(
        source_surface=source_surface,
        stages=stages,
        source_tag=source_tag,
        examples=examples,
        preconditions=preconditions,
        opens_question=opens_question,
        negative_results=negative_results,
        result_shaping=result_shaping,
        tool_name=tool_name,
        agent_role=agent_role,
        typed_input=typed_input,
        typed_output=typed_output,
        adapter_identity=adapter_identity,
        stage_result_event_type=stage_result_event_type,
        stage_failure_event_type=stage_failure_event_type,
        stage_failure_error_field=stage_failure_error_field,
        stage_failure_error_correlation=(
            stage_failure_error_correlation
            if stage_failure_error_field is not None
            else None
        ),
        model_view=model_view,
        compatibility_disposition=compatibility_disposition,
    )


QUESTION_OPENED = ToolPrecondition(
    "researcher_question_opened",
    "researcher question",
    "An earlier call in this response opened this researcher question; the Attempt ends there, so this call was not run. Send it again after the answer if it is still needed.",
)


# Each domain declares its Tools' rows beside their handlers, bound values and
# refusal facts (`TOOL_ROWS`); the loop assembles them into the one inventory
# (ADR 0040). The imports are deferred: the domains import the row types above.
@cache
def tool_row_modules() -> tuple[ModuleType, ...]:
    """Every module declaring Tool rows, in the inventory's order."""
    from bio_neb.agentic_journey import assignment_context, planning_assignment
    from bio_neb.input_inspection import binding
    from bio_neb.investigation import investigation_agent, vcf_table_reading
    from bio_neb.paper_intake import tools as paper_tools
    from bio_neb.paper_reading import tools as paper_reading
    from bio_neb.parabricks import execution_planning_agent
    from bio_neb.reference_discovery import (
        reference_facts_agent,
        reference_identity_agent,
        reference_tools,
    )
    from bio_neb.reporting import explanation_agent

    return (
        binding,
        reference_identity_agent,
        reference_facts_agent,
        reference_tools,
        paper_tools,
        execution_planning_agent,
        planning_assignment,
        investigation_agent,
        assignment_context,
        vcf_table_reading,
        explanation_agent,
        paper_reading,
    )


@cache
def _inventory() -> tuple[FunctionToolContract, ...]:
    return tuple(row for module in tool_row_modules() for row in module.TOOL_ROWS)


def function_tool_contracts(
    agent_role: AgentRole | None = None,
) -> tuple[FunctionToolContract, ...]:
    """Return the complete inventory, optionally restricted to one Agent role."""

    if agent_role is None:
        return _inventory()
    return tuple(item for item in _inventory() if item.agent_role == agent_role)


def function_tool_names(agent_role: AgentRole | None = None) -> frozenset[str]:
    """Return recognized observed Tool names from the durable inventory."""

    return frozenset(item.tool_name for item in function_tool_contracts(agent_role))


def _closure(agent_role: object) -> AttemptClosure | None:
    """The role's declared closure facts; None for a name no role declares."""
    declared = ROLES.get(agent_role) if isinstance(agent_role, str) else None
    return None if declared is None else declared.closure


def question_tool_names(agent_role: str) -> frozenset[str]:
    """The role's Tools that open a researcher question; a role with none is never
    told to ask the researcher (I-63)."""

    return frozenset(
        item.tool_name
        for item in function_tool_contracts(agent_role)
        if item.opens_question
    )


def stage_tool_names(agent_role: AgentRole, stage: str) -> frozenset[str]:
    """Every registered Tool the stage shows; the Journey search Tool is in every stage."""

    if stage not in role(agent_role).stages:
        raise ValueError(f"{agent_role} has no stage {stage!r}")
    return frozenset(
        item.tool_name
        for item in function_tool_contracts(agent_role)
        if stage in item.stages
    )


def dependency_order(
    agent_role: AgentRole, calls: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    """One reply's calls, ``(tool_name, tool_call_id)``, in the order they run.

    The sent order, except that each call first runs the reply's calls its Tool's
    declared preconditions wait on: a call of a Tool that can meet one of them
    (``met_by``) runs before it, and a call of a Tool that would make one unmet
    (``unmet_by``) runs after it. Calls of one Tool keep their sent order, and a
    cycle keeps the sent order.
    """
    contracts = {item.tool_name: item for item in function_tool_contracts(agent_role)}
    before: dict[str, list[str]] = {call_id: [] for _, call_id in calls}
    for tool_name, call_id in calls:
        contract = contracts.get(tool_name)
        for precondition in contract.preconditions if contract else ():
            for other_tool, other_id in calls:
                if other_tool == tool_name:
                    continue
                if other_tool in precondition.met_by:
                    before[call_id].append(other_id)
                if other_tool in precondition.unmet_by:
                    before[other_id].append(call_id)
    sent = {call_id: index for index, (_, call_id) in enumerate(calls)}
    order: list[str] = []
    entered: set[str] = set()

    def run(call_id: str) -> None:
        if call_id in entered:
            return
        entered.add(call_id)
        for earlier in sorted(set(before[call_id]), key=sent.__getitem__):
            run(earlier)
        order.append(call_id)

    for _, call_id in calls:
        run(call_id)
    return tuple(order)


def _holds(value: Any, field: str, values: tuple[str, ...]) -> bool:
    if isinstance(value, Mapping):
        return value.get(field) in values or any(
            _holds(item, field, values) for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(_holds(item, field, values) for item in value)
    return False


def negative_domain_result(agent_role: AgentRole, event: Mapping[str, Any]) -> bool:
    """Whether a recorded fact is a negative domain result the role's registry declares.

    Documented reader: no product path calls it; tests and inspection read retained evidence through it.
    """

    return any(
        negative.event_type == event.get("event_type")
        and _holds(event.get("payload"), negative.field, negative.values)
        for contract in function_tool_contracts(agent_role)
        for negative in contract.negative_results
    )


@dataclass(frozen=True)
class _AdmittedCall:
    response_event_id: str
    validation_event_id: str
    call: ProposedAgentAction
    continuation_request_event_id: str | None = None


class ObservedToolSession:
    """Own admitted lookup, exact-once invocation, and linked generic success."""

    def __init__(
        self,
        *,
        attempt_id: str,
        parent_activity: Activity,
        admission_session: ActionAdmissionSession,
        failure_source: AttemptFailureSource,
        observed_tool_names: frozenset[str],
    ) -> None:
        self._attempt_id = attempt_id
        self._parent_activity = parent_activity
        self._admission_session = admission_session
        self._failure_source = failure_source
        self._observed_tool_names = observed_tool_names
        self._admitted: dict[str, _AdmittedCall] = {}
        self._executed: set[str] = set()
        # The researcher question this invocation asked; the Attempt ends at it.
        self.question_asked: dict[str, Any] | None = None

    def restore_pending(self, request):
        self._admission_session.restore_pending(request)
        self._admitted[request.tool_call_id] = _AdmittedCall(
            response_event_id=request.response_event_id,
            validation_event_id=self._admission_session.admitted_validation_event_id(
                request.tool_call_id
            ),
            call=self._admission_session.admitted_call(request.tool_call_id),
            continuation_request_event_id=request.request_event_id,
        )

    def refuse_without_call(
        self,
        response_event_id: str,
        offered_tool_names: frozenset[str],
        unusable: str | None = None,
    ) -> None:
        self._admission_session.refuse_without_call(
            response_event_id, offered_tool_names, unusable
        )

    def route_reply(self, *args, **kwargs):
        return self._admission_session.route_reply(*args, **kwargs)

    def admit_turn(
        self,
        response_event_id: str,
        calls: Sequence[ProposedAgentAction],
        exposed_tools: frozenset[str],
        *,
        bound_arguments: Mapping[str, Mapping[str, Any]] | None = None,
        routed: bool = False,
    ) -> TurnDisposition | None:
        """Admit the turn's calls and remember each admitted Tool for execution."""

        observed = tuple(
            call for call in calls if call.tool_name in self._observed_tool_names
        )
        if not calls:
            return None
        disposition = self._admission_session.admit_turn(
            response_event_id=response_event_id,
            actions=calls,
            exposed_tool_names=exposed_tools,
            bound_arguments=bound_arguments,
            routed=routed,
        )
        if disposition == "singleton_admitted" and observed:
            self._remember(response_event_id, observed[0])
        return disposition

    def _remember(self, response_event_id: str, call: ProposedAgentAction) -> None:
        self._admitted[call.tool_call_id] = _AdmittedCall(
            response_event_id=response_event_id,
            validation_event_id=self._admission_session.admitted_validation_event_id(
                call.tool_call_id
            ),
            call=replace(
                call,
                arguments=self._admission_session.admitted_arguments(call.tool_call_id),
            ),
        )

    def admitted_arguments(self, tool_call_id: str) -> dict[str, Any]:
        return self._admission_session.admitted_arguments(tool_call_id)

    def ordered_call(self, tool_call_id: str) -> bool:
        return self._admission_session.ordered_call(tool_call_id)

    def ordered_arguments(self, tool_call_id: str) -> tuple[dict[str, Any], bool]:
        return self._admission_session.ordered_arguments(tool_call_id)

    def admit_call(self, tool_call_id: str) -> dict[str, Any]:
        """Admit a multi-call response's call in its turn and remember it to run."""

        response_event_id = self._admission_session.ordered_response_event_id()
        arguments = self._admission_session.admit_call(tool_call_id)
        call = self._admission_session.admitted_call(tool_call_id)
        if call.tool_name in self._observed_tool_names:
            self._remember(response_event_id, call)
        return arguments

    def close_response(self, checkpoint: str) -> None:
        """Refuse the calls a multi-call response still holds; none of them runs."""

        return self._admission_session.close_response(checkpoint)

    @property
    def turn_note(self) -> str | None:
        """The routed turn's note, owed after its results."""
        return self._admission_session.turn_note

    def note_after(self, call_ids: Sequence[str]) -> None:
        self._admission_session.note_after(call_ids)

    def owed_note(self, call_ids: Sequence[str]) -> str | None:
        return self._admission_session.owed_note(call_ids)

    def show_note(self, slot: str) -> None:
        self._admission_session.show_note(slot)

    def execute(
        self,
        tool_call_id: str,
        typed_operation: Callable[[], T],
        adapter: ToolExecutionAdapter[T],
    ) -> T:
        """Invoke one admitted typed operation and retain its exact success link."""

        try:
            admitted = self._admitted[tool_call_id]
        except KeyError as error:
            raise RuntimeError(
                "Tool action was not admitted before execution"
            ) from error
        if tool_call_id in self._executed:
            raise RuntimeError("admitted Tool action was already executed")
        self._executed.add(tool_call_id)
        self._join_open_source_review(admitted)
        started = self._parent_activity.record_contact(
            ObservedToolExecutionStarted(
                attempt_id=self._attempt_id,
                response_event_id=admitted.response_event_id,
                validation_event_id=admitted.validation_event_id,
                tool_call_id=tool_call_id,
                tool_name=admitted.call.tool_name,
                adapter_identity=adapter.identity,
                continuation_request_event_id=admitted.continuation_request_event_id,
            )
        )
        operation_completed = False
        try:
            token = _operation_owner.set(
                (current_processing_pass(), started["event_id"])
            )
            try:
                result = typed_operation()
            finally:
                _operation_owner.reset(token)
            operation_completed = True
            result_event = adapter.stage_result_event(result)
            result_event_id = result_event.get("event_id")
            if not isinstance(result_event_id, str):
                raise RuntimeError(
                    "typed Tool operation did not retain its stage result"
                )
            view = tool_model_view(model_view_of(admitted.call.tool_name), result_event)
            shown, guarded = model_visible(view)
            result_digest = canonical_json_digest(shown)
            artifact_digests = tuple(adapter.artifact_digests(result))
            self._parent_activity.record_contact(
                ObservedToolExecutionSucceeded(
                    attempt_id=self._attempt_id,
                    started_event_id=started["event_id"],
                    stage_result_event_id=result_event_id,
                    model_visible_result_digest=result_digest,
                    result_guarded=guarded,
                    artifact_digests=artifact_digests,
                    artifact_source_event_ids=self._parent_activity.retained_artifact_sources(
                        artifact_digests, result_event
                    ),
                )
            )
        except Exception as error:
            terminal_error = error
            stage_failure_event_id = None
            if not operation_completed:
                try:
                    stage_failure = adapter.stage_failure_event(error)
                    stage_failure_event_id = (
                        stage_failure.get("event_id")
                        if stage_failure is not None
                        else None
                    )
                    if stage_failure_event_id is not None and not isinstance(
                        stage_failure_event_id, str
                    ):
                        raise RuntimeError(
                            "typed Tool failure evidence lacks an event identity"
                        )
                except Exception as failure_bridge_error:
                    terminal_error = failure_bridge_error
                    stage_failure_event_id = None
            failed = self._parent_activity.record_contact(
                ObservedToolExecutionFailed(
                    attempt_id=self._attempt_id,
                    started_event_id=started["event_id"],
                    mechanical_error_type=type(terminal_error).__name__,
                    stage_failure_event_id=stage_failure_event_id,
                )
            )
            if terminal_error is error and not operation_completed:
                # Declared, or once undeclared: the model is shown why and the
                # Attempt goes on.
                if isinstance(error, ToolRefused):
                    self._refuse(failed, admitted, error, error.reason, error.statement)
                if stage_failure_event_id is None and self._shown_once(admitted, error):
                    self._refuse(
                        failed,
                        admitted,
                        error,
                        TOOL_FAILED,
                        f"the Tool failed: {type(error).__name__}",
                    )
            if stage_failure_event_id is None:
                self._failure_source.capture(failed)
            if terminal_error is not error:
                raise terminal_error from error
            if isinstance(error, ProviderError):
                self._stop_operation(error, admitted, started, stage_failure_event_id)
            if (shown := self._earlier_showing(admitted, error)) is not None:
                # The same fault again is ours: one Run-level hold names both.
                raise Invariant(
                    "tool_failed_again",
                    f"{tool_fault_key(admitted.call.tool_name, error)} failed again",
                    evidence={
                        "fault_key": tool_fault_key(admitted.call.tool_name, error),
                        "shown_event_id": shown,
                        "failed_event_id": failed["event_id"],
                    },
                ) from error
            raise
        operational_error = adapter.operational_failure(result)
        if operational_error is not None:
            self._stop_operation(operational_error, admitted, started, result_event_id)
        # The model is shown its row's view of the fact; the recorded event's id
        # rides beside it and is never sent.
        return ShownResult(view, result_event_id)

    def _join_open_source_review(self, admitted) -> None:
        """While a source's operational Review is open, another item's call to
        that source Tool joins it before contact (orchestrator answer under the
        owner's grant, #477 G4.1). An item an answer released runs its first
        call past the hold of the still-open Review (#477 6483312280)."""
        from bio_neb.agentic_journey.operational_review import (
            OperationalStop,
            open_operational_review,
        )
        from bio_neb.agentic_journey.source_contacts import source_tools

        if admitted.call.tool_name not in source_tools():
            return
        events = self._admission_session.evidence()
        lead_id = open_operational_review(events, tool_name=admitted.call.tool_name)
        if lead_id is None:
            return
        own = [
            event
            for event in events
            if event.get("payload", {}).get("attempt_id") == self._attempt_id
        ]
        if any(
            event["event_type"] == "AttemptStarted"
            and event["payload"].get("continuation_operational_review_event_id")
            for event in own
        ) and not any(
            event["event_type"] == "ObservedToolExecutionStarted" for event in own
        ):
            return
        by_id = {event["event_id"]: event for event in events}
        lead = by_id[lead_id]
        raise OperationalStop(
            by_id[lead["payload"]["request"]["failure_event_id"]],
            admitted,
            joins=lead,
        )

    def _earlier_showing(self, admitted, error) -> str | None:
        """The Run's earlier shown failure with this undeclared fault's key."""
        key = tool_fault_key(admitted.call.tool_name, error)
        return next(
            (
                event["event_id"]
                for event in reversed(self._admission_session.evidence())
                if event["event_type"] == "ToolExecutionRefused"
                and event["payload"]["reason"] == TOOL_FAILED
                and event["payload"]["fault_key"] == key
            ),
            None,
        )

    def _shown_once(self, admitted, error) -> bool:
        """Whether an undeclared failure goes to the model (owner, #477 C8): not a
        source or provider stop, not our fault on recorded data, and not a key the
        Run has shown already."""
        if isinstance(error, ProviderError) or tool_failure_holds_at_once(error):
            return False
        return self._earlier_showing(admitted, error) is None

    def _refuse(self, failed, admitted, error, reason, statement):
        from bio_neb.agentic_journey.attempt_journey import ToolExecutionRefused

        view = self._admission_session.execution_refusal_view(reason, statement)
        refused = ToolExecutionRefused(
            attempt_id=self._attempt_id,
            failed_event_id=failed["event_id"],
            reason=reason,
            statement=statement,
            fault_key=tool_fault_key(admitted.call.tool_name, error),
            next=view["next"],
            count=view["count"],
        )
        recorded = self._parent_activity.record(refused)
        self._admission_session.release_refused(admitted.call.tool_call_id)
        raise ToolExecutionRefusedResult(
            refused.model_view(), recorded["event_id"]
        ) from error

    def _stop_operation(self, error, admitted, started, source_event_id):
        from bio_neb.agentic_journey.operational_review import (
            OperationalFailureRecorded,
            OperationalStop,
        )
        from bio_neb.agentic_journey.source_contacts import source_tools
        from bio_neb.failure import policy_for

        # The policy its contact's retries resolved: the same surface and Run (an
        # operation's own surface when the Tool's contacts run on several).
        policy = policy_for(
            FailureSurface(
                scope="tool",
                name=error.evidence.get("surface", admitted.call.tool_name),
            )
        )
        fault = self._parent_activity.record(
            OperationalFailureRecorded(
                attempt_id=self._attempt_id,
                started_event_id=started["event_id"],
                source_event_id=source_event_id,
                failure_class=error.failure_class,
                original_failure_class=error.evidence.get(
                    "original_failure_class", error.failure_class
                ),
                family="source"
                if admitted.call.tool_name in source_tools()
                else "provider",
                policy=policy,
                contact_attempt=error.evidence.get("contact_attempt", 1),
                detail=error.detail,
            )
        )
        raise OperationalStop(fault, admitted) from error


_OBSERVED_FACT_TYPES = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        ObservedToolExecutionStarted,
        ObservedToolExecutionSucceeded,
        ObservedToolExecutionFailed,
        ToolOperationFailureRecorded,
    )
}


def _reopen_error(detail: str) -> ValueError:
    return ValueError(f"observed Tool execution {detail}")


def _contains_exact(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_exact(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_exact(item, expected) for item in value)
    return False


@dataclass(frozen=True)
class ToolResultLineage:
    """What the loop admitted behind one Tool result.

    The Tool execution's start, the validated call and the logical model
    request that asked for it; a role reads what that request told the model
    from its own recorded facts (#444).
    """

    started: Mapping[str, Any]
    validation: Mapping[str, Any]
    request: Mapping[str, Any]


def tool_result_lineage(
    result_event_id: str,
    timeline: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> ToolResultLineage | None:
    """The lineage behind the one receipt naming ``result_event_id`` as its
    stage result; ``None`` without exactly one receipt. ``by_id`` indexes
    ``timeline`` when the caller already holds one."""
    receipts = [
        item
        for item in timeline
        if item["event_type"] == "ObservedToolExecutionSucceeded"
        and item["payload"]["stage_result_event_id"] == result_event_id
    ]
    if len(receipts) != 1:
        return None
    by_id = by_id or {item["event_id"]: item for item in timeline}
    return _lineage(by_id[receipts[0]["payload"]["started_event_id"]]["payload"], by_id)


def tool_call_lineage(
    attempt_id: str,
    tool_call_id: str,
    timeline: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> ToolResultLineage | None:
    """The lineage of a call still executing, from its start: what a result it is
    about to record will name, read before that result exists (#476 C8)."""
    starts = [
        item["payload"]
        for item in timeline
        if item["event_type"] == "ObservedToolExecutionStarted"
        and item["payload"]["attempt_id"] == attempt_id
        and item["payload"]["tool_call_id"] == tool_call_id
    ]
    return _lineage(starts[0], by_id) if len(starts) == 1 else None


def _lineage(started, by_id) -> ToolResultLineage:
    response = by_id[started["response_event_id"]]["payload"]
    contact = by_id[response["physical_contact_started_event_id"]]
    request = by_id[contact["payload"]["logical_request_event_id"]]
    return ToolResultLineage(
        started, by_id[started["validation_event_id"]]["payload"], request
    )


def validate_attempt_closure(
    timeline: Sequence[Mapping[str, Any]],
    attempt_start: Mapping[str, Any],
    terminal_event: Mapping[str, Any],
) -> None:
    """The Attempt's own activity closes once, right after its terminal fact.

    The activity is the one its ``AttemptStarted`` follows, in the role's
    stage; it closes as a cancellation, a failure (the role's failure fact or
    ``AttemptFailed``) or a success, naming ``terminal_event`` as its parent.
    The terminal is whatever its domain recorded, a member question included;
    ``_attempt_closed_after_tool_success`` asks the narrower reopen question,
    whether the Attempt stopped after a Tool's success with one of the role's
    stopping facts.
    """
    error = ValueError("Attempt closure changed its exact terminal")
    closure = _closure(attempt_start["payload"].get("agent_role"))
    if closure is None:
        raise _reopen_error("names an Attempt role without a Tool closure")
    activity_starts = [
        event
        for event in timeline
        if event.get("event_type") == "activity_started"
        and attempt_start.get("parent_ids") == [event["event_id"]]
        and event.get("payload", {}).get("stage") == closure.stage
    ]
    if len(activity_starts) != 1:
        raise error
    activity_start = activity_starts[0]
    closures = [
        event
        for event in timeline
        if event.get("event_type")
        in {"activity_succeeded", "activity_failed", "activity_cancelled"}
        and event.get("payload", {}).get("activity_id")
        == activity_start["payload"]["activity_id"]
    ]
    if len(closures) != 1:
        raise error
    closed = closures[0]
    failed = terminal_event["event_type"] in {closure.failure_type, "AttemptFailed"}
    cancelled = terminal_event["event_type"] == "AttemptCancelled"
    try:
        validate_activity_fact(activity_start)
        validate_activity_fact(closed)
    except ValueError as invalid:
        raise error from invalid
    if (
        closed["event_type"]
        != (
            "activity_cancelled"
            if cancelled
            else "activity_failed"
            if failed
            else "activity_succeeded"
        )
        or closed["payload"].get("stage") != closure.stage
        or closed["payload"].get(
            "cancellation_event_id"
            if cancelled
            else "failure_event_id"
            if failed
            else "result_event_id"
        )
        != terminal_event["event_id"]
        or closed.get("parent_ids") != [terminal_event["event_id"]]
        or timeline.index(closed) <= timeline.index(terminal_event)
    ):
        raise error


def _attempt_closed_after_tool_success(
    timeline: tuple[Mapping[str, Any], ...],
    *,
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
    tool_start_index: int,
    tool_terminal_index: int,
    attempt_id: str,
    stopped: bool,
) -> bool:
    """Recognize an exact outer Attempt terminal that stops Agent continuation.

    The closure belongs to the Attempt that made the call, whichever role's Tool
    inventory is being reopened: a planning Attempt calls Tools registered under
    other roles, and a Tool may be registered under several roles. A stop for
    review (effort or Agent work) closes through ``AttemptStopped`` for every role.
    """

    starts = [
        event
        for event in timeline
        if event.get("event_type") == "AttemptStarted"
        and event.get("payload", {}).get("attempt_id") == attempt_id
    ]
    if len(starts) != 1:
        raise _reopen_error("lost its owning Attempt")
    closure = _closure(starts[0]["payload"].get("agent_role"))
    if closure is None:
        raise _reopen_error("names an Attempt role without a Tool closure")
    if stopped:
        closure = closure._replace(success_type="AttemptStopped")
    lifecycle: list[tuple[int, Mapping[str, Any], Any]] = []
    try:
        for index, event in enumerate(timeline):
            fact = validate_activity_fact(event)
            if fact is not None:
                lifecycle.append((index, event, fact))
    except (TypeError, ValueError, ValidationError) as error:
        raise _reopen_error("changed its stopped Attempt closure") from error

    # The Attempt's own activity closes right after its terminal fact: success
    # (or a review stop), failure, or an interrupt's cancellation.
    closings = (
        (ActivitySucceeded, "result_event_id", closure.success_type),
        (ActivityFailed, "failure_event_id", closure.failure_type),
        (ActivityCancelled, "cancellation_event_id", "AttemptCancelled"),
    )
    for terminal_index, terminal_event, terminal in lifecycle:
        kind = next((c for c in closings if isinstance(terminal, c[0])), None)
        if (
            kind is None
            or kind[2] is None
            or terminal_index <= tool_terminal_index
            or terminal.stage != closure.stage
        ):
            continue
        reference_id = getattr(terminal, kind[1])
        reference_entry = indexed_events.get(reference_id)
        if reference_entry is None:
            continue
        reference_index, reference_event = reference_entry
        identity = (
            terminal.activity_id,
            terminal.adapter,
            terminal.stage,
            terminal.attempt,
        )
        matching_lifecycle = [
            (index, event, fact)
            for index, event, fact in lifecycle
            if (
                fact.activity_id,
                fact.adapter,
                fact.stage,
                fact.attempt,
            )
            == identity
        ]
        starts = [
            (index, event)
            for index, event, fact in matching_lifecycle
            if isinstance(fact, ActivityStarted)
        ]
        terminals = [
            (index, event)
            for index, event, fact in matching_lifecycle
            if isinstance(fact, (ActivitySucceeded, ActivityFailed, ActivityCancelled))
        ]
        if (
            len(starts) == 1
            and starts[0][0] < tool_start_index
            and terminals == [(terminal_index, terminal_event)]
            # Its terminal's own parent, not its position: a sibling Attempt
            # may write between them.
            and tool_terminal_index < reference_index < terminal_index
            and terminal_event.get("parent_ids") == [reference_id]
            and (
                reference_event.get("event_type") == kind[2]
                or (
                    isinstance(kind[2], frozenset)
                    and reference_event.get("event_type") in kind[2]
                )
            )
            and reference_event.get("payload", {}).get("attempt_id") == attempt_id
        ):
            return True
    return False


def validate_observed_tool_admission(
    start_index: int,
    start: ObservedToolExecutionStarted,
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> None:
    """Reopen only the admitted call; later results and Attempt closure are separate."""

    response_entry = indexed_events.get(start.response_event_id)
    validation_entry = indexed_events.get(start.validation_event_id)
    if response_entry is None or validation_entry is None:
        raise _reopen_error("lost its response or validation identity")
    response_index, response = response_entry
    validation_index, validation = validation_entry
    validate_attempt_fact(response)
    validate_attempt_fact(validation)
    validation_payload = validation.get("payload", {})
    if (
        response.get("event_type") != "ModelResponseRecorded"
        or (
            response.get("payload", {}).get("attempt_id") != start.attempt_id
            and start.continuation_request_event_id is None
        )
        or validation.get("event_type") != "ToolCallValidated"
        or validation_payload.get("attempt_id") != start.attempt_id
        or validation_payload.get("tool_call_id") != start.tool_call_id
        or validation_payload.get("tool_name") != start.tool_name
        or validation_payload.get("status") != "valid"
        or validation_payload.get("errors") != []
        or not response_index < validation_index < start_index
    ):
        raise _reopen_error("changed its admitted response or validation")

    if start.continuation_request_event_id is not None:
        request_entry = indexed_events.get(start.continuation_request_event_id)
        attempt = next(
            (
                e
                for _, e in indexed_events.values()
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["attempt_id"] == start.attempt_id
            ),
            None,
        )
        answer_entry = (
            indexed_events.get(
                attempt["payload"].get("continuation_operational_review_event_id")
            )
            if attempt
            else None
        )
        if (
            request_entry is None
            or request_entry[1]["event_type"] != "AgentFailureReviewRequested"
            or answer_entry is None
            or answer_entry[1]["event_type"] != "AgentFailureReviewResponded"
            or answer_entry[1]["payload"]["request_event_id"]
            != start.continuation_request_event_id
            or answer_entry[1]["payload"]["response"]["action"] != "retry"
            or validation_payload.get("continuation_request_event_id")
            != start.continuation_request_event_id
            or not request_entry[0] < answer_entry[0] < validation_index
            or any(
                request_entry[1]["payload"][key] != validation_payload[key]
                for key in (
                    "tool_call_id",
                    "tool_name",
                    "arguments",
                    "bound_argument_names",
                )
            )
            or request_entry[1]["payload"]["response_event_id"]
            != start.response_event_id
        ):
            raise _reopen_error("changed its pending operational continuation")
    elif validation_payload.get("continuation_request_event_id") is not None:
        raise _reopen_error("lost its operational continuation")

    try:
        parts = response["payload"]["compatible_adapter_evidence"]["response"][
            "message"
        ]["parts"]
    except (KeyError, TypeError) as error:
        raise _reopen_error("lost its exact model call") from error
    calls = [
        part
        for part in parts
        if isinstance(part, Mapping)
        and part.get("part_kind") == "tool-call"
        and part.get("tool_call_id") == start.tool_call_id
    ]
    # The response names its physical contact, which names its logical request.
    contact = indexed_events.get(
        response.get("payload", {}).get("physical_contact_started_event_id")
    )
    request = (
        indexed_events.get(
            contact[1].get("payload", {}).get("logical_request_event_id")
        )
        if contact is not None
        else None
    )
    if (
        request is None
        or request[1].get("event_type") != "ModelRequestRecorded"
        or not request[0] < response_index
    ):
        raise _reopen_error("lost its exact model request")
    try:
        bound = request_bound_arguments(
            request[1]["payload"]["compatible_adapter_evidence"]["request"][
                "function_tools"
            ]
        )
    except (KeyError, TypeError) as error:
        raise _reopen_error("lost its exact model request") from error
    admitted = (
        validated_model_call(calls[0], validation_payload, bound)
        if len(calls) == 1
        else None
    )
    if admitted is None or admitted.arguments_invalid_json:
        raise _reopen_error("changed its exact model call")

    dispositions = [
        (index, event)
        for index, event in indexed_events.values()
        if event.get("event_type") == "AgentActionTurnDispositionRecorded"
        and event.get("payload", {}).get("attempt_id") == start.attempt_id
        and event.get("payload", {}).get("response_event_id") == start.response_event_id
        and event.get("payload", {}).get("validation_event_ids")
        == [start.validation_event_id]
    ]
    if (
        len(dispositions) != 1
        or dispositions[0][1].get("payload", {}).get("disposition")
        not in _ADMITTED_TURNS
        or not validation_index < dispositions[0][0] < start_index
    ):
        raise _reopen_error("changed its exact admission disposition")
    validate_attempt_fact(dispositions[0][1])


_ADMITTED_TURNS = frozenset({"singleton_admitted", "multi_call_admitted"})


def _observed_execution_facts(timeline: Sequence[Mapping[str, Any]]):
    """Parse each retained mechanical fact before checking cross-record identity."""
    observed: list[
        tuple[
            int,
            Mapping[str, Any],
            ObservedToolExecutionStarted
            | ObservedToolExecutionSucceeded
            | ObservedToolExecutionFailed
            | ToolOperationFailureRecorded,
        ]
    ] = []
    for index, event in enumerate(timeline):
        owner = _OBSERVED_FACT_TYPES.get(event.get("event_type"))
        if owner is None:
            continue
        try:
            retained = {
                "event_type": event["event_type"],
                **event.get("payload", {}),
            }
            if owner is ObservedToolExecutionSucceeded and isinstance(
                retained.get("artifact_digests"), list
            ):
                retained["artifact_digests"] = tuple(retained["artifact_digests"])
            if owner is ObservedToolExecutionSucceeded and isinstance(
                retained.get("artifact_source_event_ids"), list
            ):
                retained["artifact_source_event_ids"] = tuple(
                    retained["artifact_source_event_ids"]
                )
            fact = owner.model_validate(retained)
        except (KeyError, TypeError, ValidationError) as error:
            raise _reopen_error("contains an invalid retained fact") from error
        observed.append((index, event, fact))
    return observed


def _validate_execution_failure(
    *,
    start_index: int,
    start: ObservedToolExecutionStarted,
    terminal_index: int,
    terminal: ObservedToolExecutionFailed,
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
    expected_stage_failure_types: Mapping[str, str],
    expected_stage_failure_error_fields: Mapping[str, str],
    expected_stage_failure_error_correlations: Mapping[str, ToolFailureErrorCorrelation]
    | None,
) -> None:
    """Bind mechanical failure to its exact stage-owned failure when present."""
    if terminal.stage_failure_event_id is None:
        return
    stage_failure_entry = indexed_events.get(terminal.stage_failure_event_id)
    if stage_failure_entry is None:
        raise _reopen_error("lost its stage-owned failure")
    stage_failure_index, stage_failure = stage_failure_entry
    stage_failure_payload = stage_failure.get("payload", {})
    failure_error_field = expected_stage_failure_error_fields.get(start.tool_name)
    failure_error_correlation = (expected_stage_failure_error_correlations or {}).get(
        start.tool_name, "matches_terminal_mechanical_error"
    )
    stage_failure_error = (
        stage_failure_payload.get(failure_error_field)
        if failure_error_field is not None
        else None
    )
    failure_error_changed = False
    if failure_error_field is not None:
        if failure_error_correlation == "matches_terminal_mechanical_error":
            failure_error_changed = (
                stage_failure_error != terminal.mechanical_error_type
            )
        elif (
            failure_error_correlation
            == "stage_error_preserved_then_wrapped_as_RuntimeError"
        ):
            failure_error_changed = not (
                isinstance(stage_failure_error, str)
                and bool(stage_failure_error)
                and terminal.mechanical_error_type == "RuntimeError"
            )
        else:
            failure_error_changed = True
    from bio_neb.agentic_journey.source_contacts import contact_surfaces

    if (
        stage_failure.get("event_type")
        != expected_stage_failure_types.get(start.tool_name)
        or stage_failure_payload.get("attempt_id") != start.attempt_id
        # A Tool whose contacts run on several operations names the one that failed.
        or stage_failure_payload.get(
            "operation", stage_failure_payload.get("tool_name")
        )
        not in contact_surfaces(start.tool_name)
        or failure_error_changed
        or not start_index < stage_failure_index < terminal_index
    ):
        raise _reopen_error("changed its stage-owned failure")


def _validate_execution_result(
    *,
    timeline: Sequence[Mapping[str, Any]],
    start_index: int,
    start: ObservedToolExecutionStarted,
    terminal_index: int,
    terminal: ObservedToolExecutionSucceeded,
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
    expected_stage_result_types: Mapping[str, str],
    checked_once: CheckedOnce | None,
) -> None:
    """Check the exact typed result, model payload, and retained artifact sources."""
    stage_result_entry = indexed_events.get(terminal.stage_result_event_id)
    if stage_result_entry is None:
        raise _reopen_error("lost its stage result")
    stage_result_index, stage_result = stage_result_entry
    stage_payload = stage_result.get("payload", {})
    if (
        stage_result.get("event_type")
        != expected_stage_result_types.get(start.tool_name)
        or (
            stage_payload.get("attempt_id") is not None
            and stage_payload.get("attempt_id") != start.attempt_id
        )
        or not start_index < stage_result_index < terminal_index
    ):
        raise _reopen_error("changed its exact stage result")
    if start.tool_name == "search_journey":
        from bio_neb.agentic_journey.assignment_context import (
            validate_search_receipt,
        )

        def check_receipt():
            validate_search_receipt(
                timeline[:stage_result_index],
                stage_result,
                indexed_events[start.validation_event_id][1]["payload"]["arguments"],
            )

        # A receipt's re-read is fixed by the admitted prefix it names, so an
        # owner checks it once; an independent open checks it again.
        if checked_once is None:
            check_receipt()
        else:
            checked_once(("search_receipt", stage_result["event_id"]), check_receipt)
    try:
        visible = tool_model_view(model_view_of(start.tool_name), stage_result)
    except (KeyError, TypeError) as error:
        raise _reopen_error("changed its model view") from error
    if terminal.result_guarded:
        shown = shown_to_model(visible)
        if shown == visible:
            raise _reopen_error("claims a guard that changed nothing")
        visible = shown
    result_digest = canonical_json_digest(visible)
    if result_digest != terminal.model_visible_result_digest:
        raise _reopen_error("changed its canonical result digest")

    if len(terminal.artifact_source_event_ids) != len(terminal.artifact_digests):
        raise _reopen_error("lost its exact artifact sources")
    for source_id, digest in zip(
        terminal.artifact_source_event_ids, terminal.artifact_digests, strict=True
    ):
        source_entry = indexed_events.get(source_id)
        if source_entry is None:
            raise _reopen_error("lost its retained artifact source")
        source_index, source = source_entry
        if (
            not start_index < source_index <= stage_result_index
            or source.get("payload", {}).get("attempt_id") != start.attempt_id
            or not any(
                reference.get("artifact_id") == str(ContentId(digest))
                and reference.get("source_event_id") == source_id
                for reference in source.get("artifact_references", ())
            )
            or not _contains_exact(stage_payload, digest)
            or not _contains_exact(stage_payload, source_id)
        ):
            raise _reopen_error("changed its retained artifact identity")


def _validate_execution_return(
    *,
    timeline: Sequence[Mapping[str, Any]],
    start_index: int,
    start: ObservedToolExecutionStarted,
    terminal_index: int,
    terminal: ObservedToolExecutionSucceeded,
    indexed_events: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> None:
    """Require the same-call return in continuation or an exact Attempt closure."""
    response = indexed_events[start.response_event_id][1]
    contact = indexed_events[response["payload"]["physical_contact_started_event_id"]][
        1
    ]
    logical = indexed_events[contact["payload"]["logical_request_event_id"]][1]
    member_id = logical["payload"].get("member_id")
    later_requests = [
        event
        for event in timeline[terminal_index + 1 :]
        if event.get("event_type") == "ModelRequestRecorded"
        and event.get("payload", {}).get("attempt_id") == start.attempt_id
        and event.get("payload", {}).get("member_id") == member_id
    ]
    held = next(
        (
            event
            for event in timeline[terminal_index + 1 :]
            if event["event_type"] == "AgentFailureReviewRequested"
            and event["payload"]["attempt_id"] == start.attempt_id
            and event["payload"]["response_event_id"] == start.response_event_id
        ),
        None,
    )
    stops = [
        event
        for event in timeline[terminal_index + 1 :]
        if event.get("event_type") == "AttemptStopped"
        and event.get("payload", {}).get("attempt_id") == start.attempt_id
    ]
    if not later_requests and held is not None:
        from bio_neb.agentic_journey.operational_review import (
            project_operational_review,
        )

        project_operational_review(timeline)
        fault = indexed_events[held["payload"]["request"]["failure_event_id"]][1]
        if fault["payload"]["source_event_id"] == terminal.stage_result_event_id:
            return
        messages = held["payload"]["messages"]
    elif not later_requests and len(stops) == 1:
        from bio_neb.agentic_journey.agent_work_review import (
            validate_agent_work_terminal,
        )
        from bio_neb.agentic_journey.effort import validate_effort_terminal

        stopped = stops[0]
        effort = stopped["payload"]["code"] == "effort_exhausted"
        from bio_neb.agentic_journey.operational_review import (
            validate_operational_terminal,
        )

        operational = stopped["payload"]["code"] == "operational_failure"
        validator = (
            validate_operational_terminal
            if operational
            else validate_effort_terminal
            if effort
            else validate_agent_work_terminal
        )
        validator(timeline, stopped)
        if not _attempt_closed_after_tool_success(
            timeline,
            indexed_events=indexed_events,
            tool_start_index=start_index,
            tool_terminal_index=terminal_index,
            attempt_id=start.attempt_id,
            stopped=True,
        ):
            raise _reopen_error("changed its review stop closure")
        if not effort and not operational:
            # The Agent work stop retains no continuation transcript to check.
            return
        review = indexed_events[stopped["payload"]["source_event_id"]][1]
        if operational:
            fault = indexed_events[review["payload"]["request"]["failure_event_id"]][1]
            if fault["payload"]["source_event_id"] == terminal.stage_result_event_id:
                return
        messages = review["payload"]["messages"]
    elif not later_requests:
        closed_without_continuing = _attempt_closed_after_tool_success(
            timeline,
            indexed_events=indexed_events,
            tool_start_index=start_index,
            tool_terminal_index=terminal_index,
            attempt_id=start.attempt_id,
            stopped=False,
        )
        if not closed_without_continuing:
            raise _reopen_error("lost its next model request")
        return
    else:
        try:
            messages = later_requests[0]["payload"]["compatible_adapter_evidence"][
                "request"
            ]["messages"]
        except (KeyError, TypeError) as error:
            raise _reopen_error("lost its next model request") from error
    tool_returns = [
        part
        for message in messages
        if isinstance(message, Mapping)
        for part in message.get("parts", ())
        if isinstance(part, Mapping)
        and part.get("part_kind") == "tool-return"
        and part.get("tool_call_id") == start.tool_call_id
    ]
    if (
        len(tool_returns) != 1
        or tool_returns[0].get("tool_name") != start.tool_name
        or canonical_json_digest(tool_returns[0].get("content"))
        != terminal.model_visible_result_digest
        # The fact's id rides beside the view and is never sent.
        or tool_returns[0].get("metadata")
        != {"event_id": terminal.stage_result_event_id}
    ):
        raise _reopen_error("changed its same-call model return")


def validate_tool_failure_source(
    events: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    terminal: Mapping[str, Any],
    attempt_id: str,
) -> None:
    """Bind an Attempt terminal to its mechanical Tool failure by exact anchor.

    The journal failure has no semantic parent; the terminal follows the same
    anchor with no semantic fact between them (ADR 0034, journey-path "Two records").
    """
    payload = source.get("payload", {})
    source_index, terminal_index = events.index(source), events.index(terminal)
    if (
        source.get("event_type") != "ObservedToolExecutionFailed"
        or payload.get("attempt_id") != attempt_id
        or payload.get("stage_failure_event_id") is not None
        or source_index >= terminal_index
        or not all(map(is_contact_evidence, events[source_index + 1 : terminal_index]))
        or terminal.get("parent_ids") != [contact_evidence_anchor(events, source)]
    ):
        raise ValueError("Attempt failure changed its exact mechanical Tool source")


def recorded_within_call(
    event: Mapping[str, Any],
    started: Mapping[str, Any],
    events_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Whether a stage fact belongs to one admitted call by exact semantic lineage.

    A failed call has no aggregate result naming what it recorded. The rule follows
    exact recorded parent pointers only, never position or writer order: the first
    admission disposition on the fact's single-parent lineage must be the call's own.
    """
    call = started.get("payload", {})
    parents = event.get("parent_ids", ())
    seen: set[str] = set()
    while len(parents) == 1 and parents[0] not in seen:
        seen.add(parents[0])
        parent = events_by_id.get(parents[0])
        if parent is None:
            return False
        if parent.get("event_type") == "AgentActionTurnDispositionRecorded":
            admission = parent.get("payload", {})
            return (
                admission.get("attempt_id"),
                admission.get("response_event_id"),
                admission.get("validation_event_ids"),
            ) == (
                call.get("attempt_id"),
                call.get("response_event_id"),
                [call.get("validation_event_id")],
            )
        parents = parent.get("parent_ids", ())
    return False


def _validate_operation_failures(timeline, observed, starts, terminals):
    """Each retry remains inside one Tool execution and its retained policy."""
    starts_by_id = {event["event_id"]: (index, fact) for index, event, fact in starts}
    failures = {}
    for index, _, fact in observed:
        if not isinstance(fact, ToolOperationFailureRecorded):
            continue
        start = starts_by_id.get(fact.started_event_id)
        prior = failures.setdefault(fact.started_event_id, [])
        routed, ceiling = operation_retry_decision(
            fact.original_failure_class, fact.contact_attempt, fact.policy
        )
        if (
            start is None
            or start[0] >= index
            or fact.policy.surface
            != FailureSurface(scope="tool", name=start[1].tool_name)
            or any(
                terminal.started_event_id == fact.started_event_id and end <= index
                for end, _, terminal in terminals
            )
            or fact.contact_attempt != len(prior) + 1
            or fact.original_failure_class == "invariant"
            or (
                prior
                and (
                    prior[-1].retry_delay_seconds is None
                    or prior[-1].policy != fact.policy
                )
            )
            or fact.failure_class != routed
            or (ceiling is None) != (fact.retry_delay_seconds is None)
            or (ceiling is not None and not 0 <= fact.retry_delay_seconds <= ceiling)
        ):
            raise _reopen_error("failure changed its scoped policy or attempt")
        prior.append(fact)

    for started_event_id, records in failures.items():
        fault = records[-1]
        operational = [
            event["payload"]
            for event in timeline
            if event.get("event_type") == "OperationalFailureRecorded"
            and event.get("payload", {}).get("started_event_id") == started_event_id
        ]
        if operational and (
            len(operational) != 1
            or fault.retry_delay_seconds is not None
            or any(
                operational[0].get(field) != fault.model_dump(mode="json")[field]
                for field in (
                    "policy",
                    "contact_attempt",
                    "failure_class",
                    "original_failure_class",
                    "detail",
                )
            )
        ):
            raise _reopen_error("failure changed its terminal operation fault")


def _held_cause_names() -> frozenset[str]:
    """The exception types that hold at once, never shown (#477 C8): every
    ``Invariant``, ``JourneyValidationError`` and ``OSError`` type."""
    from bio_neb.journey import JourneyValidationError

    names: set[str] = set()
    pending = [Invariant, JourneyValidationError, OSError]
    while pending:
        owner = pending.pop()
        names.add(owner.__name__)
        pending.extend(owner.__subclasses__())
    return frozenset(names)


def validate_tool_refusals(events: Sequence[Mapping[str, Any]]) -> None:
    """Each ``ToolExecutionRefused`` follows from its own failed call (#477 C8;
    Standards r1 ST2).

    It names an earlier ``ObservedToolExecutionFailed`` of its own Attempt with no
    stage failure, at most once. A declared refusal's terminal is ``ToolRefused``
    and its key is the Tool and its reason; an undeclared one's cause is no class
    that holds at once, its statement and key name that cause, and its key is
    shown once in the Run (Standards r2 ST3).
    """
    by_id = {event["event_id"]: (index, event) for index, event in enumerate(events)}
    seen: set[str] = set()
    shown_keys: set[str] = set()
    held = None
    for index, event in enumerate(events):
        if event.get("event_type") != "ToolExecutionRefused":
            continue
        refusal = event["payload"]
        failed_index, failed = by_id.get(refusal["failed_event_id"], (-1, {}))
        if (
            failed.get("event_type") != "ObservedToolExecutionFailed"
            or not 0 <= failed_index < index
            or refusal["failed_event_id"] in seen
        ):
            raise _reopen_error("refusal lost its exact failed call")
        seen.add(refusal["failed_event_id"])
        terminal = failed["payload"]
        _, started = by_id.get(terminal["started_event_id"], (-1, {}))
        cause = terminal["mechanical_error_type"]
        if (
            started.get("event_type") != "ObservedToolExecutionStarted"
            or terminal["attempt_id"] != refusal["attempt_id"]
            or terminal.get("stage_failure_event_id") is not None
        ):
            raise _reopen_error("refusal lost its exact failed call")
        tool = started["payload"]["tool_name"]
        if refusal["reason"] == TOOL_FAILED:
            held = _held_cause_names() if held is None else held
            lawful = (
                cause not in held
                and cause != "ToolRefused"
                and refusal["statement"] == f"the Tool failed: {cause}"
                and refusal["fault_key"] == f"{tool}: {cause}"
                and refusal["fault_key"] not in shown_keys
            )
            shown_keys.add(refusal["fault_key"])
        else:
            lawful = (
                cause == "ToolRefused"
                and refusal["fault_key"] == f"{tool}: {refusal['reason']}"
            )
        if not lawful:
            raise _reopen_error("refusal changed its cause, reason or key")


def validate_observed_tool_executions(
    events: Iterable[Mapping[str, Any]],
    *,
    expected_adapter_identities: Mapping[str, str],
    separately_validated_tool_names: frozenset[str],
    expected_stage_result_types: Mapping[str, str],
    expected_stage_failure_types: Mapping[str, str],
    expected_stage_failure_error_fields: Mapping[str, str],
    expected_stage_failure_error_correlations: Mapping[str, ToolFailureErrorCorrelation]
    | None = None,
    checked_once: CheckedOnce | None = None,
) -> None:
    """Reopen every retained generic execution chain without Adapter contact.

    ``checked_once`` is the reading owner's once-per-owner check, when there is one.
    """

    timeline = tuple(events)
    observed = _observed_execution_facts(timeline)
    validate_tool_refusals(timeline)
    indexed_events: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, event in enumerate(timeline):
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise _reopen_error("contains an event without exact identity")
        if event_id in indexed_events:
            raise _reopen_error("contains a repeated event identity")
        indexed_events[event_id] = (index, event)

    all_starts = [
        (index, event, fact)
        for index, event, fact in observed
        if isinstance(fact, ObservedToolExecutionStarted)
    ]
    all_started_event_ids = {event["event_id"] for _, event, _ in all_starts}
    all_terminals = [
        (index, event, fact)
        for index, event, fact in observed
        if isinstance(
            fact, (ObservedToolExecutionSucceeded, ObservedToolExecutionFailed)
        )
    ]
    _validate_operation_failures(timeline, observed, all_starts, all_terminals)
    known_tool_names = (
        frozenset(expected_adapter_identities) | separately_validated_tool_names
    )
    if any(fact.tool_name not in known_tool_names for _, _, fact in all_starts):
        raise _reopen_error("contains an unrecognized observed Tool")
    if any(
        fact.started_event_id not in all_started_event_ids
        for _, _, fact in all_terminals
    ):
        raise _reopen_error("has a terminal without its exact start")

    starts = [
        (index, event, fact)
        for index, event, fact in all_starts
        if fact.tool_name in expected_adapter_identities
    ]
    if not starts:
        return
    selected_started_event_ids = {event["event_id"] for _, event, _ in starts}
    terminals = [
        (index, event, fact)
        for index, event, fact in all_terminals
        if fact.started_event_id in selected_started_event_ids
    ]
    if len({(fact.attempt_id, fact.tool_call_id) for _, _, fact in starts}) != len(
        starts
    ):
        raise _reopen_error("repeated an executed call identity")

    for start_index, start_event, start in starts:
        expected_adapter = expected_adapter_identities.get(start.tool_name)
        if expected_adapter is None or start.adapter_identity != expected_adapter:
            raise _reopen_error("changed its Adapter identity")

        validate_observed_tool_admission(start_index, start, indexed_events)

        matching_terminals = [
            (index, event, fact)
            for index, event, fact in terminals
            if fact.started_event_id == start_event["event_id"]
        ]
        if len(matching_terminals) != 1:
            raise _reopen_error("does not have exactly one terminal")
        terminal_index, _, terminal = matching_terminals[0]
        if terminal.attempt_id != start.attempt_id or terminal_index <= start_index:
            raise _reopen_error("changed its terminal order or Attempt")

        if isinstance(terminal, ObservedToolExecutionFailed):
            _validate_execution_failure(
                start_index=start_index,
                start=start,
                terminal_index=terminal_index,
                terminal=terminal,
                indexed_events=indexed_events,
                expected_stage_failure_types=expected_stage_failure_types,
                expected_stage_failure_error_fields=expected_stage_failure_error_fields,
                expected_stage_failure_error_correlations=expected_stage_failure_error_correlations,
            )
            continue

        _validate_execution_result(
            timeline=timeline,
            start_index=start_index,
            start=start,
            terminal_index=terminal_index,
            terminal=terminal,
            indexed_events=indexed_events,
            expected_stage_result_types=expected_stage_result_types,
            checked_once=checked_once,
        )

        _validate_execution_return(
            timeline=timeline,
            start_index=start_index,
            start=start,
            terminal_index=terminal_index,
            terminal=terminal,
            indexed_events=indexed_events,
        )

    migrated_attempts = {fact.attempt_id for _, _, fact in starts}
    for validation_index, validation in enumerate(timeline):
        payload = validation.get("payload", {})
        if (
            validation.get("event_type") != "ToolCallValidated"
            or payload.get("attempt_id") not in migrated_attempts
            or payload.get("tool_name") not in expected_adapter_identities
        ):
            continue
        matching_starts = [
            fact
            for _, _, fact in starts
            if fact.validation_event_id == validation.get("event_id")
        ]
        turns = [
            event.get("payload", {}).get("disposition")
            for index, event in enumerate(timeline)
            if event.get("event_type") == "AgentActionTurnDispositionRecorded"
            and event.get("payload", {}).get("attempt_id") == payload.get("attempt_id")
            and event.get("payload", {}).get("validation_event_ids")
            == [validation.get("event_id")]
            and event.get("payload", {}).get("disposition") in _ADMITTED_TURNS
            and validation_index < index
        ]
        if (turns and len(matching_starts) != 1) or (not turns and matching_starts):
            raise _reopen_error("changed pre-contact admission or execution")


def validate_observed_function_tool_role(
    events: Iterable[Mapping[str, Any]],
    *,
    agent_role: AgentRole,
    checked_once: CheckedOnce | None = None,
) -> None:
    """Reopen one role from the durable Function Tool contract."""

    contracts = tuple(item for item in _inventory() if item.agent_role == agent_role)
    if not contracts:
        raise ValueError(f"unknown observed Function Tool role: {agent_role}")
    _validate_observed_contracts(events, contracts, checked_once)


def validate_observed_function_tools(
    events: Iterable[Mapping[str, Any]], *, checked_once: CheckedOnce | None = None
) -> None:
    """Reopen every registered role's observed Function Tools from the one registry."""

    _validate_observed_contracts(events, _inventory(), checked_once)


def _validate_observed_contracts(
    events: Iterable[Mapping[str, Any]],
    contracts: tuple[FunctionToolContract, ...],
    checked_once: CheckedOnce | None,
) -> None:
    validate_observed_tool_executions(
        events,
        expected_adapter_identities={
            item.tool_name: item.adapter_identity for item in contracts
        },
        separately_validated_tool_names=function_tool_names()
        - frozenset(item.tool_name for item in contracts),
        expected_stage_result_types={
            item.tool_name: item.stage_result_event_type for item in contracts
        },
        expected_stage_failure_types={
            item.tool_name: item.stage_failure_event_type
            for item in contracts
            if item.stage_failure_event_type is not None
        },
        expected_stage_failure_error_fields={
            item.tool_name: item.stage_failure_error_field
            for item in contracts
            if item.stage_failure_error_field is not None
        },
        expected_stage_failure_error_correlations={
            item.tool_name: item.stage_failure_error_correlation
            for item in contracts
            if item.stage_failure_error_correlation is not None
        },
        checked_once=checked_once,
    )


__all__ = [
    "FunctionToolContract",
    "ShownResult",
    "ObservedToolExecutionFailed",
    "ObservedToolExecutionStarted",
    "ObservedToolExecutionSucceeded",
    "ObservedToolSession",
    "ToolExecutionAdapter",
    "ToolFailureErrorCorrelation",
    "function_tool_contracts",
    "function_tool_names",
    "validate_observed_function_tool_role",
    "validate_observed_function_tools",
    "validate_observed_tool_executions",
    "validate_tool_failure_source",
]
