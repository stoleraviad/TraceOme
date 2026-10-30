"""Native Agent assembly and invocation shared by planning and scientific assignments."""

import asyncio
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from inspect import signature, unwrap
from typing import Annotated, Any, Generic, NamedTuple, TypeVar, get_type_hints

import anyio
from pydantic import Field, ValidationError
from pydantic_ai import (
    Agent,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    Tool,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage, UsageLimits

from bio_neb.agentic_journey.action_admission import (
    ATTEMPT_ENDED,
    QUESTION_OPENED,
    ActionAdmissionSession,
    PydanticActionAdmissionCapability,
    grounded_tool_schema,
)
from bio_neb.agentic_journey.action_policy import (
    PreconditionUnmet,
    ProposedAgentAction,
)
from bio_neb.agentic_journey.agent_work_review import AgentResearcherQuestion
from bio_neb.agentic_journey.assignment_context import (
    SEARCH_FILTER_ITEMS,
    SEARCH_TEXT_CHARACTERS,
    SEARCH_TOOL_NAME,
    JourneyPayloadSlice,
    JourneySearchRead,
    assignment_admission_errors,
    assignment_repeat_identity,
    journey_search_query,
    search_journey,
    search_refusal_facts,
)
from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
from bio_neb.agentic_journey.continuation_policy import (
    RESEND_KEY,
    AgentInvocationPolicy,
    current_turn,
)
from bio_neb.agentic_journey.effort import AgentEffortExhausted
from bio_neb.agentic_journey.injection import (
    InjectionPolicy,
    InjectionSignals,
    ToolResult,
    injected_pattern,
    injection_messages,
    latest_reply,
    stage_request_number,
)
from bio_neb.agentic_journey.invocation_observation import (
    inspect_agent,
    inspection_event,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_observation import (
    REQUEST_STAGE,
    JourneyObservedModel,
)
from bio_neb.agentic_journey.model_settings import (
    ModelEndpointPolicy,
)
from bio_neb.agentic_journey.model_turn import messages_record, read_messages_record
from bio_neb.agentic_journey.operational_review import OperationalStop
from bio_neb.agentic_journey.researcher_question import (
    ASK_RESEARCHER,
    ResearcherQuestionAsked,
)
from bio_neb.agentic_journey.role_declaration import AgentRole, role
from bio_neb.agentic_journey.role_instructions import RoleInstructions
from bio_neb.agentic_journey.tool_execution import (
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolExecutionRefusedResult,
    ToolPrecondition,
    dependency_order,
    function_tool_contracts,
    function_tool_names,
    stage_tool_names,
)
from bio_neb.agentic_journey.transcript import Pending, reply_between, request_parts
from bio_neb.content import canonical_json
from bio_neb.failure import (
    CorrectionWording,
    Invariant,
    ModelContactStalled,
    ModelRequestRefusedShown,
    OutputFaults,
    OutputRefused,
    ProviderRejected,
    ReplyStalled,
    model_value_text,
    render_correction,
    sentence_with_facts,
)
from bio_neb.recording import evidence_view, read_evidence

Dependencies = TypeVar("Dependencies")
_PENDING_INSTRUCTIONS: ContextVar[tuple[str | None, int] | None] = ContextVar(
    "pending_agent_instructions", default=None
)


@contextmanager
def _pending_instructions(message_history, requests_used):
    pending = message_history[-1] if message_history else None
    binding = (
        (pending.instructions, requests_used)
        if isinstance(pending, ModelRequest)
        else None
    )
    token = _PENDING_INSTRUCTIONS.set(binding)
    try:
        yield
    finally:
        _PENDING_INSTRUCTIONS.reset(token)


@dataclass(frozen=True)
class ActionAdmissionSpec(Generic[Dependencies]):
    """The domain part of action admission; the shared session owns the rest."""

    validator: Callable[[Dependencies, ProposedAgentAction], tuple[str, ...]]
    # Repair text per Tool, shown only in that Tool's own refusal feedback.
    repair_guidance: Mapping[str, str] = field(default_factory=dict)
    # The domain refuses a repeated call of its own Tools before contact with this
    # code (``assignment_repeat_identity``); a search's repeats are only counted.
    repeat_error: str | None = None
    # The typed facts a refusal reason's sentence names, computed once by the
    # domain that refuses: ``(deps, action, code) -> {field: value}``. A
    # ``value`` fact is the checked state a row's slot names, shown in place of
    # the call that was sent.
    refusal_facts: (
        Callable[[Any, ProposedAgentAction, str], Mapping[str, Any]] | None
    ) = None
    # Whose earlier calls the loop guard reads (``repeated_calls.CallHistory``): the
    # Attempt's by default; a role whose work spans Attempts supplies its own.
    call_history: Callable[[list, str], Any] | None = None


@dataclass(frozen=True)
class AssignmentOutput:
    """A typed structured outcome and its domain validator."""

    output_type: type
    name: str
    description: str
    validator: Callable[["ToolContext[Any]", Any], Any]


class _ModelToolSchemas(AbstractCapability[Any]):
    """Show every offered Tool through the selected model's schema projection."""

    def __init__(self, endpoint_policy: ModelEndpointPolicy) -> None:
        self._endpoint_policy = endpoint_policy

    async def prepare_tools(self, ctx, tool_defs):
        return [self._endpoint_policy.project_tool_schema(item) for item in tool_defs]

    async def prepare_output_tools(self, ctx, tool_defs):
        return [self._endpoint_policy.project_tool_schema(item) for item in tool_defs]


# Failures a Tool raised after an interrupt reached it. The framework turns the
# raise into its cancellation, so the Tool's Attempt binds this list and the
# failure takes its own path; `record_operational_stop` then interrupts.
INTERRUPTED_TOOL_FAILURES: ContextVar[list[Exception] | None] = ContextVar(
    "interrupted_tool_failures", default=None
)


class _ToolRunsToCompletion(AbstractCapability[Any]):
    """An interrupt takes effect between turns, never inside a running Tool.

    A Tool runs in a worker thread that a cancellation cannot stop. Its Attempt
    waits for it, so the Tool's own facts are retained before the cancellation
    is recorded; a second interrupt still stops at once. A Tool that fails hands
    its failure to the Attempt (``INTERRUPTED_TOOL_FAILURES``).
    """

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        running = asyncio.ensure_future(handler(args))
        try:
            return await asyncio.shield(running)
        except ToolExecutionRefusedResult as refused:
            # A Tool that could not produce its fact answers with its
            # refusal; the Attempt goes on.
            return refused.shown
        except asyncio.CancelledError:
            # The framework's task group repeats its cancellation until the
            # interrupted task leaves; the shielded scope waits the Tool out.
            with anyio.CancelScope(shield=True):
                await asyncio.wait({running})
            failure = None if running.cancelled() else running.exception()
            failures = INTERRUPTED_TOOL_FAILURES.get()
            if isinstance(failure, Exception) and failures is not None:
                failures.append(failure)
            raise


class _StageRequest(AbstractCapability[Any]):
    """Each request's stage, read once, selects its settings, Tool choice and record.

    The settings and Tool choice were merged once, into the Attempt's invocation
    policy; the coordinator's current stage only selects its entry. A stage with
    no configured choice sends none: nothing forces a call by default (owner,
    #476 alignment G1.2 B). The framework refuses a static choice, so this is
    derived per step.
    """

    def __init__(
        self,
        policy: AgentInvocationPolicy,
        stage: Callable[[Any], str],
    ) -> None:
        self._policy = policy
        self._stage = stage
        self._current: tuple[tuple[str, int], str] | None = None

    def request_stage(self, ctx) -> str:
        request = (ctx.run_id, ctx.run_step)
        if self._current is None or self._current[0] != request:
            self._current = (request, self._stage(ctx.deps))
        return self._current[1]

    def get_model_settings(self):
        def settings(ctx):
            stage = self.request_stage(ctx)
            selected = self._policy.stage_model_settings(stage)
            # A stage's entry keeps its explicit omissions (None) over the settings
            # the adapter was built with; without one the role's values are sent.
            values = (
                selected.model_dump()
                if stage in self._policy.stage_settings
                else selected.request_settings()
            )
            choice = self._policy.stage_tool_choice(stage)
            if choice is not None:
                values["tool_choice"] = choice
            return values

        return settings

    async def wrap_model_request(self, ctx, *, request_context, handler):
        token = REQUEST_STAGE.set(self.request_stage(ctx))
        try:
            return await handler(request_context)
        finally:
            REQUEST_STAGE.reset(token)


class _StallNote(AbstractCapability[Any]):
    """A stalled or refused request ends with no reply; the next request carries
    its note.

    The note is the failure package's correction, so the model learns its last
    reply never arrived and sends a shorter one instead of the same (owner, #443
    Q-A, Q-B), or reads the endpoint's refusal in the provider's words (#477
    G1.5). Each is the first in a row: the same again asks the availability
    Review before it gets here.
    """

    def __init__(self, wording: CorrectionWording) -> None:
        self._wording = wording

    async def on_model_request_error(self, ctx, *, request_context, error):
        if isinstance(error, ModelContactStalled):
            raise ModelRetry(
                render_correction(
                    ReplyStalled(seconds=error.stall_seconds),
                    self._wording,
                )
            ) from error
        if isinstance(error, ModelRequestRefusedShown):
            # The endpoint refused the request: the model reads the provider's
            # own words and may change what it sends.
            raise ModelRetry(
                render_correction(
                    ProviderRejected(status=error.status, statement=error.statement),
                    self._wording,
                )
            ) from error
        raise error


class _Injection:
    """The one Injection slot's speaker: what switched-on patterns would say now.

    Each message is a name-tagged user part of the pending request, so the
    framework history, the request record and the model hold the same bytes, and
    reopen replays it with the request. The transcript composer places them
    (ADR 0048).
    """

    def __init__(
        self,
        policy: InjectionPolicy | None,
        agent_role: str,
        stage: Callable[[RunContext[Any]], str],
        *,
        events: Callable[[], Sequence[Mapping[str, Any]]],
        attempt_id: str,
        limit: Callable[[str], int | None],
    ) -> None:
        self._policy = policy
        self._agent_role = agent_role
        self._stage = stage
        self._events = events
        self._attempt_id = attempt_id
        self._limit = limit

    def notices(
        self, ctx: RunContext[Any], messages: Sequence[Any]
    ) -> tuple[str, tuple[str, ...]]:
        """The request's stage and the messages its patterns would send."""
        stage = self._stage(ctx)
        if self._policy is None:
            return stage, ()
        pending = messages[-1]
        events = self._events()
        signals = InjectionSignals(
            agent_role=self._agent_role,
            stage=stage,
            request_number=ctx.usage.requests + 1,
            tool_results=sum(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            ),
            sent=Counter(
                name
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
                and (name := injected_pattern(part.content))
            ),
            stage_request_number=stage_request_number(events, self._attempt_id, stage),
            reply=latest_reply(events, self._attempt_id, limit=self._limit),
            results=tuple(
                ToolResult(part.tool_name, part.content)
                for part in pending.parts
                if isinstance(part, ToolReturnPart)
            ),
        )
        return stage, tuple(injection_messages(self._policy, signals))


class _ComposedRequest(AbstractCapability[Any]):
    """Each request the framework sends is the transcript composer's answer.

    Admission says what it owes (a refused call's answer in its slot, the routed
    turn's note), the Injection slot what it would say; the composer orders them
    with the framework's results and keeps two requests from following each
    other (``transcript``). A request that already carries an injected message is
    a resumed one and is sent as recorded.
    """

    def __init__(
        self,
        owed: Callable[
            [RunContext[Any], ModelResponse | None],
            tuple[ToolReturnPart | None, str | None],
        ],
        injection: _Injection,
        *,
        attempt_id: str,
    ) -> None:
        self._owed = owed
        self._injection = injection
        self._attempt_id = attempt_id

    async def before_model_request(self, ctx, request_context):
        messages = list(request_context.messages)
        if len(messages) > 1:
            # Only the tail can need it: a stalled request, then its retry.
            previous, pending_record = messages_record(messages[-2:])
            between = reply_between(previous, pending_record)
            if between is not None:
                messages[-1:-1] = read_messages_record([between])
                request_context.messages = messages
        pending = messages[-1] if messages else None
        if not isinstance(pending, ModelRequest) or any(
            isinstance(part, UserPromptPart) and injected_pattern(part.content)
            for part in pending.parts
        ):
            return request_context
        reply = (
            messages[-2]
            if len(messages) > 1 and isinstance(messages[-2], ModelResponse)
            else None
        )
        answer, note = self._owed(ctx, reply)
        stage, notices = self._injection.notices(ctx, messages)
        (pending_record,) = messages_record([pending])
        composed = request_parts(
            Pending(
                tuple(pending_record["parts"]),
                None
                if answer is None
                else messages_record([ModelRequest(parts=[answer])])[0]["parts"][0],
                note,
                notices,
            )
        )
        (sent,) = read_messages_record([{**pending_record, "parts": composed}])
        pending.parts = sent.parts
        for part in pending.parts:
            if isinstance(part, UserPromptPart) and (
                pattern := injected_pattern(part.content)
            ):
                inspection_event(
                    "Injected notice",
                    attempt_id=self._attempt_id,
                    pattern=pattern,
                    stage=stage,
                    text=part.content,
                )
        return request_context


def _stopping_count(
    policy: AgentInvocationPolicy, key: str, events, attempt_id: str
) -> int | None:
    """The failed turn in a row at which the Attempt stops, in the stage of the
    latest reply's request; a resend has none here (the model-contact window
    bounds it)."""
    if key == RESEND_KEY:
        return None
    return current_turn(events, attempt_id=attempt_id, policy=policy.continuation)[1]


def missing_tool_purposes(
    role_text: RoleInstructions,
    tool_names: Iterable[str],
    output: AssignmentOutput | None = None,
) -> tuple[str, ...]:
    """Names of Tools whose rendered description states no purpose.

    A function Tool's description starts with the purpose the role's instruction
    blocks state for it, and the output Tool's is the assignment's own; without
    one a Tool reaches the model as a bare name and schema.
    """
    descriptions = {name: role_text.tool_purpose(name) for name in tool_names}
    if output is not None:
        descriptions[output.name] = output.description
    return tuple(
        sorted(name for name, text in descriptions.items() if not (text or "").strip())
    )


# The strict reading's refusals of a scalar the lax reading would have coerced;
# its other differences (a JSON array read where a tuple is declared) coerce
# nothing the record and the handler could disagree on.
_COERCED_SCALAR_TYPES = frozenset(
    {"int_type", "float_type", "bool_type", "string_type"}
)


def _argument_refusals(arguments: Any, items: Sequence[Mapping]) -> tuple[str, ...]:
    """The validator's errors as refusal codes, each named as the call sent it."""
    return tuple(
        "tool_arguments_invalid:"
        + _argument_location(arguments, item["loc"])
        + ":"
        + item["type"]
        + ":"
        + item["msg"]
        for item in items
        if not _short_for_its_items(item, items)
    )


def _argument_location(arguments: Any, loc: Sequence[Any]) -> str:
    """An error location as the sent arguments name it.

    A tagged union adds its tag to the location (a requirement's ``kind`` value,
    a knowledge ``state``): a part the sent object holds as a value, not a key, is
    left out.
    """
    parts = []
    value = arguments
    for part in loc:
        if isinstance(value, Mapping) and part not in value and part in value.values():
            continue
        parts.append(str(part))
        if isinstance(value, Mapping):
            value = value.get(part)
        elif (
            isinstance(value, list | tuple)
            and isinstance(part, int)
            and part < len(value)
        ):
            value = value[part]
        else:
            value = None
    return ".".join(parts)


def _stage_list(stages: tuple[str, ...]) -> str:
    return " or ".join(f"`{stage}`" for stage in stages)


class _StagedTools:
    """One invocation's registry view: stage visibility, preconditions and reasons.

    The role's registry rows place every Tool in its stages; the surface supplies
    the current stage and each Tool's server-bound values, raising
    ``PreconditionUnmet`` when the Tool cannot run now.
    """

    def __init__(
        self,
        assignment: AgentRole,
        tools: Mapping[str, Callable[..., Any]],
        stage: Callable[[Any], str],
        bound_values: Callable[[Any, str], Any],
        repeat_error: str | None,
        role_text: RoleInstructions,
        refusal_facts: Callable[[Any, ProposedAgentAction, str], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self._role_text = role_text
        self._refusal_facts = refusal_facts
        self._contracts = {
            item.tool_name: item for item in function_tool_contracts(assignment)
        }
        self._stage_tools = {
            name: stage_tool_names(assignment, name) & frozenset(tools)
            for name in role(assignment).stages
        }
        self._tools = frozenset(tools)
        self._stage = stage
        self._bound_values = bound_values
        self._repeat_error = repeat_error
        self._unmet: dict[str, PreconditionUnmet] = {}

    def check_registered(self) -> None:
        """Refuse an assembled Tool that no registry row places in a stage."""
        staged = frozenset().union(*self._stage_tools.values())
        if unstaged := self._tools - staged:
            raise RuntimeError(
                "assembled Tool has no registered stage: " + ", ".join(sorted(unstaged))
            )

    def description(self, name: str, tool: Tool[Any]) -> str:
        """The stated purpose, the handler's call guidance, then the examples.

        The role's instruction blocks state what the Tool is for, in the model's
        own words; the handler's docstring keeps only how to call it.
        """
        contract = self._contracts[name]
        return "\n\n".join(
            part
            for part in (
                contract.source_tag and f"[{contract.source_tag}]",
                self._role_text.tool_purpose(name),
                (tool.description or "").strip(),
                "\n".join(f"Example: {example}" for example in contract.examples),
            )
            if part
        )

    def prepare(self, name: str):
        def prepare(ctx, definition):
            if name not in self._stage_tools[self._stage(ctx.deps)]:
                return None
            try:
                current = self._bound_values(ctx.deps, name)
            except PreconditionUnmet:
                # Shown unbound: admission refuses the call with its reason.
                return grounded_tool_schema(definition, constants={})
            return grounded_tool_schema(
                definition,
                constants=current.constants,
                enums=current.enums,
            )

        return prepare

    def waiting(self, deps: Any, names: frozenset[str]) -> dict[str, str]:
        """The offered stage Tools whose precondition does not hold now, each with
        the reason its refusal gives.

        ADR 0040 keeps them offered; each request's tools-now view and a refusal's
        offer both read this one source (#372 W5; #478 G1.1).
        """
        waiting = {}
        for name in sorted(names):
            if name not in self._contracts or not self._contracts[name].stages:
                continue
            try:
                self._bound_values(deps, name)
            except PreconditionUnmet as unmet:
                waiting[name] = self._waiting_reason(deps, name, unmet)
        return waiting

    def tools_now(self, deps: Any, stage: str, also: frozenset[str]) -> dict[str, Any]:
        """What the model can call on this request, and what waits and why.

        ``also`` are the Tools every stage offers beside the stage's own: the
        Journey search and the role's output Tool.
        """
        offered = self._stage_tools[stage] | also
        waiting = self.waiting(deps, offered)
        return {
            "callable_now": sorted(offered - waiting.keys()),
            "waiting": [
                {"tool": name, "reason": reason} for name, reason in waiting.items()
            ],
        }

    def _waiting_reason(self, deps: Any, name: str, unmet: PreconditionUnmet) -> str:
        """The sentence a call of ``name`` would be refused with now."""
        action = ProposedAgentAction(
            tool_name=name, tool_call_id=f"waiting:{name}", arguments={}
        )
        self._unmet[action.tool_call_id] = unmet
        try:
            return self.explain(deps, action, unmet.code) or unmet.code
        finally:
            del self._unmet[action.tool_call_id]

    def precondition(self, deps: Any, action: ProposedAgentAction) -> str | None:
        """The stage Tool's unmet precondition code, kept for its explanation."""
        if (
            action.tool_name not in self._contracts
            or not self._contracts[action.tool_name].stages
        ):
            return None
        try:
            current = self._bound_values(deps, action.tool_name)
        except PreconditionUnmet as unmet:
            self._unmet[action.tool_call_id] = unmet
            return unmet.code
        for name, value in current.constants.items():
            if action.arguments.get(name) != value:
                # Earlier calls in this response may have changed the lawful
                # binding. Keep request evidence exact and refuse before effects;
                # the next request will carry the current server-bound values.
                unmet = PreconditionUnmet(f"{name}:bound_value_mismatch", value)
                self._unmet[action.tool_call_id] = unmet
                return unmet.code
        # An offered choice holds as a bound constant does, for every Tool, with
        # or without a provider grammar (sweep P3).
        for name, offered in current.enums.items():
            if name in action.arguments and action.arguments[name] not in offered:
                unmet = PreconditionUnmet(f"{name}:not_offered", tuple(offered))
                self._unmet[action.tool_call_id] = unmet
                return unmet.code
        return None

    def explain(
        self,
        deps: Any,
        action: ProposedAgentAction,
        code: str,
        response_value: str | None = None,
    ) -> str | None:
        """The plain-words sentence for a refusal code: slot, value and reason.

        A code is explained by the refused Tool's own registry row, or else by the
        role row that declares it. ``<key>:<code>`` is a reason about the argument
        item that ``key`` names, such as one proposal requirement.
        ``response_value`` is what a response-level refusal is about: the question
        Tool an earlier call opened, or the Tools the response named unoffered.
        """
        contract = self._contracts.get(action.tool_name)
        if code.endswith(":bound_value_mismatch"):
            unmet = self._unmet.get(action.tool_call_id)
            if unmet is not None and unmet.code == code:
                name = code.removesuffix(":bound_value_mismatch")
                return (
                    f"Argument `{name}` has effective value "
                    f"{model_value_text(action.arguments.get(name))}, but its current "
                    f"server-bound value is {model_value_text(unmet.value)}. "
                    "Call again on the next request and omit server-bound arguments; "
                    "an earlier call may have changed their state."
                )
        if code.endswith(":not_offered"):
            unmet = self._unmet.get(action.tool_call_id)
            if unmet is not None and unmet.code == code:
                name = code.removesuffix(":not_offered")
                return (
                    f"Argument `{name}` is "
                    f"{model_value_text(action.arguments.get(name))}, which this Tool "
                    f"does not offer; it offers {model_value_text(list(unmet.value))}."
                )
        if code == QUESTION_OPENED:
            declared = self._declared(contract, code)
            if declared is None or response_value is None:
                return None
            return (
                f"`{declared.slot}` is {model_value_text(response_value)}: "
                + declared.explanation
            )
        if code == ATTEMPT_ENDED:
            return (
                f"`Attempt checkpoint` is {model_value_text(response_value)}: an "
                "earlier call in this response ended the Attempt there, so this call "
                "was not run."
            )
        if code == "tool_name_unavailable":
            if contract is None or not contract.stages:
                return None
            current = self._stage(deps)
            # A Tool of the current stage that this invocation does not assemble
            # is refused for its own reason alone (the other assembly's lookup
            # Tool).
            stage = (
                ""
                if current in contract.stages
                else f"It works in the {_stage_list(contract.stages)} stage; "
                f"the current stage is `{current}`."
            )
            # Why the Tool cannot run yet: its own unmet precondition, when declared.
            try:
                self._bound_values(deps, action.tool_name)
            except PreconditionUnmet as unmet:
                declared = self._declared(contract, unmet.code)
                if declared is not None:
                    value = (
                        " has no value"
                        if unmet.value is None
                        else f" is {model_value_text(unmet.value)}"
                    )
                    reason = self._sentence(deps, action, unmet.code, declared, unmet)
                    return " ".join(
                        filter(None, (stage, f"`{declared.slot}`{value}: {reason}"))
                    )
            return stage or None
        if code == self._repeat_error:
            # A row that declares the repeat states it in its own words.
            declared = self._declared(contract, code)
            return (
                declared.explanation
                if declared is not None
                else "This exact call was already admitted in this Attempt; "
                "repeating it adds no evidence."
            )
        key, _, reason = code.rpartition(":")
        declared = self._declared(contract, reason)
        if declared is None:
            return None
        unmet = self._unmet.get(action.tool_call_id)
        if unmet is not None and unmet.code != code:
            unmet = None
        facts = self._facts(deps, action, code, unmet)
        if key:
            slot = f"`{declared.slot}` of `{key}`"
            value = _keyed_value(action.arguments, key, declared.slot)
        else:
            slot = f"`{declared.slot}`"
            if declared.slot in action.arguments:
                value = model_value_text(action.arguments[declared.slot])
            elif unmet is not None:
                # A precondition is about checked state, which may be absent.
                value = None if unmet.value is None else model_value_text(unmet.value)
            elif "value" in facts:
                # A domain refusal about checked state: the role names its value.
                value = (
                    None if facts["value"] is None else model_value_text(facts["value"])
                )
            else:
                # A domain refusal of the whole call: its slot is what was sent.
                value = model_value_text(dict(action.arguments))
        return (
            slot
            + (f" is {value}" if value is not None else " has no value")
            + f": {sentence_with_facts(declared.explanation, facts)}"
        )

    def _sentence(
        self,
        deps: Any,
        action: ProposedAgentAction,
        code: str,
        declared: ToolPrecondition,
        unmet: PreconditionUnmet | None,
    ) -> str:
        """The row's sentence with the facts its refusal holds (#372 D2)."""
        return sentence_with_facts(
            declared.explanation, self._facts(deps, action, code, unmet)
        )

    def _facts(
        self,
        deps: Any,
        action: ProposedAgentAction,
        code: str,
        unmet: PreconditionUnmet | None,
    ) -> dict[str, Any]:
        """The unmet precondition's own facts, else those the role computes for
        the code; a role's ``value`` fact is its slot's checked state."""
        facts = dict(unmet.facts) if unmet is not None and unmet.facts else {}
        if not facts and self._refusal_facts is not None:
            facts = dict(self._refusal_facts(deps, action, code))
        return facts

    def _declared(self, contract: Any, code: str) -> ToolPrecondition | None:
        rows = (contract, *self._contracts.values()) if contract else ()
        return next(
            (item for row in rows for item in row.preconditions if item.code == code),
            None,
        )


def _short_for_its_items(item: Mapping[str, Any], items: list) -> bool:
    """A length error whose items failed on their own: their errors say why, and
    "at least 1 item, not 0" would deny the item that was sent (#409, #380 S3)."""
    loc = tuple(item["loc"])
    return item["type"] == "too_short" and any(
        len(other["loc"]) > len(loc) and tuple(other["loc"][: len(loc)]) == loc
        for other in items
    )


def _keyed_value(arguments: Any, key: str, field: str) -> str | None:
    """The ``field`` of the argument item that ``key`` names, as the model reads it."""
    if isinstance(arguments, Mapping):
        if key in arguments.values() and field in arguments:
            return model_value_text(arguments[field])
        items = arguments.values()
    elif isinstance(arguments, list):
        items = arguments
    else:
        return None
    return next(
        (
            value
            for item in items
            if (value := _keyed_value(item, key, field)) is not None
        ),
        None,
    )


@dataclass(frozen=True)
class ToolContext(Generic[Dependencies]):
    """The domain's dependencies and exact admitted call; no framework state."""

    deps: Dependencies
    tool_call_id: str


class OutputRepair(Exception):
    """A domain validator's recorded refusal and model-facing correction: its
    reason, or its faults for the failure package to word (#448)."""

    def __init__(self, reason: str, faults: OutputFaults | None = None) -> None:
        super().__init__(reason)
        self.faults = faults


def _bind_tool(function):
    """Keep the domain's argument schema while translating its call context."""

    @wraps(function)
    def bound(ctx, *args, **kwargs):
        return function(ToolContext(ctx.deps, ctx.tool_call_id), *args, **kwargs)

    # Delegated Tools keep their original domain signature through @wraps. Both
    # Python's signature and type-hint readers must see the translated context,
    # with all Annotated argument constraints and descriptions preserved.
    annotations = get_type_hints(unwrap(function), include_extras=True)
    parameters = list(signature(function).parameters.values())
    annotations[parameters[0].name] = RunContext[Any]
    bound.__annotations__ = annotations
    bound.__signature__ = signature(function).replace(
        parameters=[
            parameter.replace(
                annotation=annotations.get(parameter.name, parameter.annotation)
            )
            for parameter in parameters
        ],
        return_annotation=annotations.get(
            "return", signature(function).return_annotation
        ),
    )
    return bound


class AssembledAgent(NamedTuple, Generic[Dependencies]):
    agent: Agent[Dependencies, Any]
    deps: Dependencies


def journey_search_handler(
    search: Callable[[RunContext[Any], Mapping[str, Any]], dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """The shared Journey search's offered signature, around one invocation's read."""

    def journey_search(
        ctx: RunContext[Any],
        text: Annotated[
            str | None, Field(min_length=1, max_length=SEARCH_TEXT_CHARACTERS)
        ] = None,
        event_ids: Annotated[
            tuple[str, ...], Field(max_length=SEARCH_FILTER_ITEMS)
        ] = (),
        kinds: Annotated[tuple[str, ...], Field(max_length=SEARCH_FILTER_ITEMS)] = (),
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        before_event_id: str | None = None,
        payload_slice: JourneyPayloadSlice | None = None,
    ) -> dict[str, Any]:
        """text matches retained events case-insensitively; kinds and event_ids
        narrow them; limit keeps the newest matches and before_event_id pages
        back. Set payload_slice with the event_id of a payload_reference, then
        follow next_offset; slice offsets count Unicode characters in canonical
        JSON, and a slice read takes no text, event_ids, kinds or limit. A search
        grants no effect authority.
        """
        return search(
            ctx,
            {
                "text": text,
                "event_ids": event_ids,
                "kinds": kinds,
                "limit": limit,
                "before_event_id": before_event_id,
                "payload_slice": payload_slice,
            },
        )

    return journey_search


def _researcher_question(observed, parent_activity, attempt_id, assignment):
    """The shared question's one handler, bound per invocation (#447 Q8)."""

    def ask_researcher(
        ctx: RunContext[Any],
        found_so_far: Annotated[str, Field(min_length=1, max_length=2000)],
        conflict: Annotated[str, Field(min_length=1, max_length=2000)],
        question: Annotated[str, Field(min_length=1, max_length=2000)],
    ) -> dict[str, Any]:
        """found_so_far is what your reads or results show so far; conflict is
        what stops you, in one sentence; question is your one intent question for
        the researcher. Asking ends this Attempt."""
        event = None

        def record():
            nonlocal event
            event = parent_activity.record(
                ResearcherQuestionAsked(
                    attempt_id=attempt_id,
                    agent_role=assignment,
                    found_so_far=found_so_far,
                    conflict=conflict,
                    question=question,
                )
            )
            return event

        result = observed.execute(
            ctx.tool_call_id,
            record,
            ToolExecutionAdapter(
                identity="researcher_question",
                stage_result_event=lambda _result: event,
                artifact_digests=lambda _result: (),
            ),
        )
        observed.question_asked = event
        return result

    return ask_researcher


def assemble_agent(
    model: JourneyObservedModel,
    *,
    assignment: AgentRole,
    instructions: str | Callable[[], str],
    tools: Mapping[str, Callable[..., Any]],
    stage: Callable[[Dependencies], str],
    bound_values: Callable[[Dependencies, str], Any],
    admission: ActionAdmissionSpec[Dependencies],
    policy: AgentInvocationPolicy,
    model_policy: ModelEndpointPolicy,
    role_text: RoleInstructions,
    journey: Any,
    attempt_id: str,
    parent_activity: Activity,
    failure_source: AttemptFailureSource,
    dependencies: Callable[[ActionAdmissionSession, ObservedToolSession], Dependencies],
    output: AssignmentOutput | None = None,
    search_scope_event_id: str | None = None,
    context: Callable[[], Mapping[str, Any]] | None = None,
) -> AssembledAgent[Dependencies]:
    """Build one invocation's Agent with the shared admission, schema and Journey search.

    Surfaces supply instructions, the Attempt's role text (``role_text``, whose
    blocks state each Tool's purpose), domain Tools, the coordinator's current stage,
    each Tool's server-bound values, an optional typed outcome and their admission
    validator. The role's Tool registry places every Tool in its stages: each
    request shows every Tool of the current stage, read again per request, and
    ``bound_values`` raising ``PreconditionUnmet`` refuses a call at admission
    with the registry's explanation instead of hiding the Tool. Each call creates
    the invocation's own admission and observed Tool sessions over the Journey's
    checked evidence, physical contacts included; ``dependencies`` receives them to
    build the domain state. ``search_scope_event_id`` names the retained fact
    listing the only events the Journey search may read; admission refuses any
    other event before the read. Each request's instructions end with one JSON
    view: the role's ``context`` and the Tools callable now and those waiting,
    each with the reason its refusal would give (#478 G1.1).
    """

    def refusal_facts(deps: Any, action: ProposedAgentAction, code: str):
        # The shared search names its own refusal's facts for every role.
        if action.tool_name == SEARCH_TOOL_NAME:
            return search_refusal_facts(
                read_evidence(journey),
                action.arguments,
                code,
                scope_event_id=search_scope_event_id,
            )
        if admission.refusal_facts is None:
            return {}
        return admission.refusal_facts(deps, action, code)

    # A role that declares the shared question's row offers it; the assembly
    # binds its one handler below, as it binds the Journey search.
    if ASK_RESEARCHER in function_tool_names(assignment):
        tools = {**tools, ASK_RESEARCHER: None}
    staged = _StagedTools(
        assignment,
        tools,
        stage,
        bound_values,
        admission.repeat_error,
        role_text,
        refusal_facts,
    )
    argument_validators = {}

    def argument_errors(action: ProposedAgentAction) -> tuple[str, ...]:
        # Each Tool's own Pydantic validator checks the effective arguments once,
        # before contact; no surface keeps a copy of the argument structure.
        validator = argument_validators[action.tool_name]
        try:
            validator.validate_python(dict(action.arguments))
        except ValidationError as error:
            return _argument_refusals(
                action.arguments, error.errors(include_url=False, include_input=False)
            )
        # Read again as the JSON they are, strictly: a scalar of the wrong JSON
        # type (a number or a flag sent as text) is refused with the schema's
        # correction rather than coerced after it is recorded.
        try:
            validator.validate_json(canonical_json(dict(action.arguments)), strict=True)
        except ValidationError as error:
            return _argument_refusals(
                action.arguments,
                [
                    item
                    for item in error.errors(include_url=False, include_input=False)
                    if item["type"] in _COERCED_SCALAR_TYPES
                ],
            )
        return ()

    session = ActionAdmissionSession(
        # The owner's shared read-only evidence, never a per-call history copy
        # (ADR 0032); the session copies what it keeps.
        prior_events=evidence_view(journey),
        events=lambda: evidence_view(journey),
        attempt_id=attempt_id,
        parent_activity=parent_activity,
        failure_source=failure_source,
        eligible_tool_names=function_tool_names(assignment),
        action_validator=lambda action: (
            # The shared question opens once: the Attempt ends at it.
            ("researcher_question_already_asked",)
            if observed.question_asked is not None
            else ()
        )
        if action.tool_name == ASK_RESEARCHER
        else assignment_admission_errors(
            action,
            events=lambda: evidence_view(journey),
            domain_errors=lambda proposed: admission.validator(deps, proposed),
            search_scope_event_id=search_scope_event_id,
        ),
        argument_validator=argument_errors,
        action_fingerprint=None
        if admission.repeat_error is None
        else assignment_repeat_identity,
        admitted_repeat_error=admission.repeat_error,
        continuation=policy.continuation,
        resend_limit=policy.model_contact.contacts - 1,
        wording=model_policy.correction_wording,
        repair_guidance=admission.repair_guidance,
        precondition=lambda action: staged.precondition(deps, action),
        **(
            {}
            if admission.call_history is None
            else {"call_history": admission.call_history}
        ),
        explain=lambda action, code: staged.explain(
            deps, action, code, session.response_value(action.tool_call_id)
        ),
        output_tool_names=frozenset() if output is None else frozenset({output.name}),
        waiting=lambda names: frozenset(staged.waiting(deps, names)),
        question_tool_names=frozenset(
            item.tool_name
            for item in function_tool_contracts(assignment)
            if item.opens_question
        ),
        order=lambda calls: dependency_order(assignment, calls),
    )
    observed = session.observe_tools(function_tool_names(assignment))
    deps = dependencies(session, observed)

    def search(ctx: RunContext[Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = journey_search_query(**arguments, scope_event_id=search_scope_event_id)
        event = None

        def retrieve():
            nonlocal event
            event = parent_activity.record(
                JourneySearchRead(
                    attempt_id=attempt_id,
                    query=query,
                    result=search_journey(journey, **query),
                )
            )
            return event

        return observed.execute(
            ctx.tool_call_id,
            retrieve,
            ToolExecutionAdapter(
                identity="journey_read",
                stage_result_event=lambda _result: event,
                artifact_digests=lambda _result: (),
            ),
        )

    journey_search = journey_search_handler(search)

    if undocumented := missing_tool_purposes(
        role_text, (*tools, SEARCH_TOOL_NAME), output
    ):
        raise RuntimeError("offered Tool states no purpose: " + ", ".join(undocumented))
    staged.check_registered()
    agent_tools = [
        Tool(
            function,
            name=name,
            description=staged.description(name, Tool(function, name=name)),
            sequential=True,
            prepare=staged.prepare(name),
        )
        for name, domain_function in tools.items()
        if name != ASK_RESEARCHER
        for function in (_bind_tool(domain_function),)
    ]
    if ASK_RESEARCHER in tools:
        ask_researcher = _researcher_question(
            observed, parent_activity, attempt_id, assignment
        )
        agent_tools.append(
            Tool(
                ask_researcher,
                name=ASK_RESEARCHER,
                description=staged.description(
                    ASK_RESEARCHER, Tool(ask_researcher, name=ASK_RESEARCHER)
                ),
                sequential=True,
                prepare=staged.prepare(ASK_RESEARCHER),
            )
        )
    agent_tools.append(
        Tool(
            journey_search,
            name=SEARCH_TOOL_NAME,
            description=staged.description(
                SEARCH_TOOL_NAME, Tool(journey_search, name=SEARCH_TOOL_NAME)
            ),
            sequential=True,
            prepare=lambda _ctx, definition: grounded_tool_schema(
                definition, constants={}
            ),
        )
    )
    argument_validators.update(
        {tool.name: tool.function_schema.validator for tool in agent_tools}
    )

    def current_instructions(ctx: RunContext[Dependencies]) -> str | None:
        # The framework resolves instructions before checking its request limit.
        # At that unsent boundary retain the last context, so the effort receipt
        # is the exact pending Tool continuation. The first resumed request also
        # retains that binding; later requests refresh findings normally.
        pending = _PENDING_INSTRUCTIONS.get()
        if pending is not None and ctx.usage.requests == pending[1]:
            return pending[0]
        limit = ctx.usage_limits.request_limit
        if limit is not None and ctx.usage.requests >= limit:
            for message in reversed(ctx.messages[:-1]):
                if isinstance(message, ModelRequest):
                    return message.instructions
        # Sent as the framework retains them (it strips the joined text), so a
        # resumed request resends the exact instructions of its pending one.
        text = (instructions() if callable(instructions) else instructions).strip()
        view = {
            **(context() if context is not None else {}),
            **staged.tools_now(ctx.deps, stage_request.request_stage(ctx), always),
        }
        return f"{text}\n{canonical_json(view)}"

    always = frozenset({SEARCH_TOOL_NAME}) | (
        frozenset() if output is None else frozenset({output.name})
    )

    stage_request = _StageRequest(policy, stage)
    admitting = PydanticActionAdmissionCapability(
        observed,
        received=lambda _ctx: model.last_reply,
        observe_outputs=model.observe_outputs,
        wording=model_policy.correction_wording,
        output_tool_names=frozenset() if output is None else frozenset({output.name}),
        output_repair=model.output_repair,
        output_refusal=model.output_refusal,
    )
    agent = Agent(
        model,
        name=assignment,
        deps_type=type(deps),
        instructions=current_instructions,
        tools=agent_tools,
        output_type=str
        if output is None
        else ToolOutput(
            output.output_type,
            name=output.name,
            description=output.description,
            # The failure rule counts a refused output as a failed turn; the
            # framework's own count is sized never to bind first.
            max_retries=policy.framework_retry_allowance,
        ),
        # The continuation check stops the Attempt at its limit of failed turns
        # in a row. Corrections also reach the global output counter, sized never
        # to bind first; an admitted Tool gets no hidden framework retry.
        retries={
            "tools": policy.tool_retry,
            "output": policy.framework_retry_allowance,
        },
        capabilities=[
            _ModelToolSchemas(model_policy),
            _ToolRunsToCompletion(),
            stage_request,
            admitting,
            _StallNote(model_policy.correction_wording),
            _ComposedRequest(
                admitting.owed,
                # The request's one stage read, which its record names too. The
                # owner's shared read-only evidence, never a per-request copy.
                _Injection(
                    policy.injection,
                    assignment,
                    stage_request.request_stage,
                    events=lambda: evidence_view(journey),
                    attempt_id=attempt_id,
                    limit=lambda key: _stopping_count(
                        policy, key, evidence_view(journey), attempt_id
                    ),
                ),
                attempt_id=attempt_id,
            ),
        ],
    )

    if output is not None:

        def validate_output(ctx: RunContext[Dependencies], value: Any) -> Any:
            try:
                return output.validator(ToolContext(ctx.deps, ctx.tool_call_id), value)
            except OutputRepair as error:
                raise ModelRetry(
                    render_correction(
                        OutputRefused(
                            reason=None if error.faults is not None else str(error),
                            faults=error.faults,
                            fault=model.output_repair(),
                            tool_name=output.name,
                        ),
                        model_policy.correction_wording,
                    )
                ) from error

        agent.output_validator(validate_output)
    return AssembledAgent(agent, deps)


class _CompletedWork(Exception):
    def __init__(self, result: Any):
        self.result = result


def _pending_transcript(messages, corrections, note=None):
    """An operational stop's native transcript, as JSON for its Review.

    Each refused sibling call's correction is returned in the pending request,
    followed by the routed turn's note, as the transcript composer orders them.
    """
    messages = list(messages)
    if corrections or note is not None:
        # A copy of the pending request: the framework's own history is unchanged.
        pending = (
            messages.pop()
            if isinstance(messages[-1], ModelRequest)
            else ModelRequest(parts=[])
        )
        returns = [
            ToolReturnPart(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                content=feedback,
            )
            for call, feedback in corrections
        ]
        (record,) = messages_record(
            [replace(pending, parts=[*pending.parts, *returns])]
        )
        messages.append(
            {
                **record,
                # No notice joins a held transcript.
                "parts": request_parts(Pending(tuple(record["parts"]), note=note)),
            }
        )
        return [*messages_record(messages[:-1]), messages[-1]]
    return messages_record(messages)


def retained_usage(message_history=()) -> RunUsage:
    """Physical retry windows do not consume a second reasoning allowance."""
    return RunUsage(
        requests=sum(isinstance(message, ModelResponse) for message in message_history)
    )


async def invoke_agent(
    agent,
    prompt,
    *,
    deps,
    policy,
    journey_id,
    attempt_id,
    member_id=None,
    message_history=None,
    usage=None,
    completed_work: Callable[[], Any | None] | None = None,
    deferred_call_id: str | None = None,
):
    """Run the framework loop; domain completion may stop after an actual Tool.

    A domain result does not require another model request to establish what its
    deterministic operation already proved. No framework nodes are scheduled or
    polled here. Scientific result admission remains with the supplied domain.
    """
    hooks = Hooks()

    @hooks.on.after_tool_execute
    def after_question(context, *, call, tool_def, args, result):
        # The researcher question ends the Attempt once its result is recorded:
        # the calls its reply still holds are refused, and the Agent work Review
        # opens.
        if tool_def.name != ASK_RESEARCHER:
            return result
        asked = context.deps.observed_tool_session.question_asked
        if asked is not None:
            context.deps.observed_tool_session.close_response("researcher_question")
            raise AgentResearcherQuestion(asked)
        return result

    if completed_work is not None:

        @hooks.on.after_tool_execute
        def after_tool(context, *, call, tool_def, args, result):
            completed = completed_work()
            if completed is not None:
                raise _CompletedWork(completed)
            return result

    effective_usage = (
        usage if usage is not None else retained_usage(message_history or ())
    )
    pending = message_history[-1] if message_history else None
    if isinstance(pending, ModelRequest):
        requests_used = effective_usage.requests

        @hooks.on.before_model_request
        def retain_pending_timestamp(ctx, request_context):
            # Resuming without a prompt rebuilds the pending ModelRequest. Its
            # original timestamp belongs to the logical request; contact events
            # date each resend. Preserve it here so framework history, evidence
            # and the model all receive the same request.
            if ctx.usage.requests == requests_used:
                request_context.messages[-1].timestamp = pending.timestamp
            return request_context

    with (
        _pending_instructions(message_history, effective_usage.requests),
        capture_run_messages() as messages,
        inspect_agent(
            agent,
            journey_id=str(journey_id),
            attempt_id=attempt_id,
            member_id=member_id,
        ),
    ):
        try:
            # A fresh invocation can close from checked retained state. A
            # pending transcript still follows the exact continuation contract.
            if message_history is None and completed_work is not None:
                completed = completed_work()
                if completed is not None:
                    raise _CompletedWork(completed)
            try:
                return await agent.run(
                    prompt,
                    message_history=message_history,
                    deps=deps,
                    usage=effective_usage,
                    usage_limits=UsageLimits(request_limit=policy.model_requests),
                    capabilities=[hooks],
                    deferred_tool_results=DeferredToolResults(
                        approvals={deferred_call_id: True}
                    )
                    if deferred_call_id is not None
                    else None,
                )
            except UserError as error:
                # The framework refused how we drove it (a history or resume
                # shape it cannot run): our bug, held under its own name.
                raise Invariant("agent_framework_refused", str(error)) from error
            except asyncio.CancelledError:
                # Outside the framework's scope, a Tool that failed after the
                # interrupt raises its own failure; an operational stop delivers
                # the interrupt once its Review is recorded.
                failures = INTERRUPTED_TOOL_FAILURES.get()
                if not failures:
                    raise
                failure = failures[0]
                if isinstance(failure, OperationalStop):
                    failure.interrupted = True
                raise failure from None
        except OperationalStop as stop:
            corrections = deps.observed_tool_session.close_response(
                "operational_failure"
            )
            stop.retain_invocation(
                _pending_transcript(
                    messages, corrections, deps.observed_tool_session.turn_note
                ),
                member_id,
            )
            raise
        except UsageLimitExceeded:
            if effective_usage.requests < policy.model_requests:
                raise
            raise AgentEffortExhausted(
                messages or message_history or (), effective_usage.requests
            ) from None
        except _CompletedWork as completed:
            inspection_event(
                "Journey semantic completion",
                journey_id=str(journey_id),
                attempt_id=attempt_id,
                checkpoint=completed.result.checkpoint,
                framework_end=False,
            )
            return completed.result
