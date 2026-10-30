"""Per-Attempt admission retention and Pydantic AI function-action interception."""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from pydantic import ValidationError
from pydantic_ai import (
    ModelRetry,
    RunContext,
    ToolFailed,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolValidation
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.action_policy import (
    ActionFingerprint,
    ActionPrecondition,
    ActionValidator,
    ProposedAgentAction,
    action_admission_errors,
    call_identity,
    effective_arguments,
)
from bio_neb.agentic_journey.attempt_journey import (
    AgentActionTurnDispositionRecorded,
    AgentTurnRouted,
    AttemptFailureSource,
    ModelCompletionTruncated,
    ToolCallValidated,
    TurnFaultCount,
)
from bio_neb.agentic_journey.continuation_policy import (
    COPIES_KEY,
    OUTPUT_ALONE_KEY,
    REASONING_ONLY_KEY,
    RESEND_KEY,
    TEXT_ENDING_KEY,
    TRUNCATION_KEY,
    UNOFFERED_KEY,
    ContinuationPolicy,
    correction_key,
    current_turn,
)
from bio_neb.agentic_journey.invocation_observation import (
    inspection_event,
    turn_progress,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_settings import reply_content, reply_quirks
from bio_neb.agentic_journey.model_turn import (
    ModelArguments,
    ModelCall,
    ModelTurn,
    UnusableReply,
    messages_record,
    object_arguments,
    output_schema_errors,
    read_arguments,
    read_messages_record,
    read_turn,
)
from bio_neb.agentic_journey.repeated_calls import (
    REPEATED_CALL,
    CallHistoryReader,
    RepeatProgress,
    attempt_call_history,
    repeat_key,
    repeated_call_of,
)
from bio_neb.agentic_journey.structured_output import SeveralOutputs
from bio_neb.agentic_journey.transcript import kept_reply, reply_retry
from bio_neb.failure import (
    ActionRefused,
    CorrectionWording,
    FaultCount,
    Invariant,
    OutputFaults,
    OutputRefused,
    ReasoningOnly,
    RepeatedCall,
    ReplyCutOff,
    TextEnding,
    TurnFaults,
    execution_refused_view,
    fault_trace,
    names_waiting,
    render_correction,
)

if TYPE_CHECKING:
    from bio_neb.agentic_journey.model_observation import ReceivedReply
    from bio_neb.agentic_journey.tool_execution import ObservedToolSession


BOUND_ARGUMENTS_METADATA_KEY = "bound_arguments"


def _without_null_default(schema: Any) -> Any:
    """Render each ``X | None = None`` field as plain ``X``, recursively.

    Omission still validates to ``None`` and the null branch is no longer offered.
    """
    if isinstance(schema, list):
        return [_without_null_default(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    rendered = {key: _without_null_default(value) for key, value in schema.items()}
    branches = rendered.get("anyOf")
    if (
        "default" in rendered
        and rendered["default"] is None
        and isinstance(branches, list)
        and {"type": "null"} in branches
    ):
        rendered.pop("default")
        remaining = [branch for branch in branches if branch != {"type": "null"}]
        rendered.pop("anyOf")
        if len(remaining) == 1 and isinstance(remaining[0], dict):
            rendered = {**remaining[0], **rendered}
        else:
            rendered["anyOf"] = remaining
    return rendered


def grounded_tool_schema(
    definition: ToolDefinition,
    *,
    constants: Mapping[str, Any],
    enums: Mapping[str, Sequence[Any]] | None = None,
) -> ToolDefinition:
    """Expose a sequential Tool with only the arguments the model still chooses.

    The bound constants are server-known: they leave the schema and ride in
    ``metadata``, which the recorded request retains and the model never receives.
    Admission fills them. Enums stay as the model's real choices.
    """
    schema = copy.deepcopy(definition.parameters_json_schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise Invariant(
            "tool_schema_ungrounded", "grounded Tool schema has no properties"
        )
    for name in (*constants, *(enums or {})):
        if name not in properties:
            raise Invariant(
                "tool_schema_ungrounded",
                "grounded Tool schema and handler arguments disagree",
            )
    for name in constants:
        properties.pop(name)
    for name, values in (enums or {}).items():
        # A binder with no admissible value raises ``PreconditionUnmet``, so its
        # Tool waits with that reason: an empty enum no call can satisfy breaks
        # the provider's Tool grammar.
        if not values:
            raise Invariant(
                "tool_choice_ungrounded",
                f"{definition.name}.{name} is offered as an empty choice",
            )
        if len({_json_kind(value) for value in values}) != 1 or _json_kind(
            values[0]
        ) not in ("string", "integer", "number", "boolean"):
            raise Invariant(
                "tool_choice_ungrounded",
                f"{definition.name}.{name} is offered as a choice of mixed types "
                "or of no scalar type",
            )
        properties[name] = {"enum": list(values)}
    if "required" in schema:
        schema["required"] = [
            name for name in schema["required"] if name not in constants
        ]
    schema["additionalProperties"] = False
    return replace(
        definition,
        parameters_json_schema=_without_null_default(schema),
        metadata={
            **(definition.metadata or {}),
            BOUND_ARGUMENTS_METADATA_KEY: dict(constants),
        },
        sequential=True,
        # Never strict: one strict Tool puts every Tool under the provider's grammar,
        # which fixes the argument order the model may write.
        strict=False,
    )


def _json_kind(value: Any) -> str:
    """A choice value's JSON type; a choice is of one scalar type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def output_tool_schema(definition: ToolDefinition) -> ToolDefinition:
    """Expose an output Tool with the same optional-field rendering as function Tools."""

    return replace(
        definition,
        parameters_json_schema=_without_null_default(definition.parameters_json_schema),
        strict=False,
    )


def request_bound_arguments(
    function_tools: Sequence[ToolDefinition | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return each offered Tool's server-bound values from one request's definitions.

    Accepts live definitions or their recorded form. Every offered Tool records its
    bound values, possibly none; a definition without that record is refused.
    """

    bound: dict[str, dict[str, Any]] = {}
    for tool in function_tools:
        name, metadata = (
            (tool.name, tool.metadata)
            if isinstance(tool, ToolDefinition)
            else (tool["name"], tool.get("metadata"))
        )
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get(BOUND_ARGUMENTS_METADATA_KEY), Mapping
        ):
            raise ValueError(f"offered Tool {name} records no bound arguments")
        bound[name] = dict(metadata[BOUND_ARGUMENTS_METADATA_KEY])
    return bound


RefusalExplanation = Callable[[ProposedAgentAction, str], str | None]


def validated_model_call(
    call: Mapping[str, Any],
    validation: Mapping[str, Any],
    bound_arguments: Mapping[str, Mapping[str, Any]],
) -> ProposedAgentAction | None:
    """Return a retained model call as admitted, or None unless it is the validated call.

    The one reopen check of the exact identity chain: the response's call name and id
    equal its ``ToolCallValidated`` record, and its arguments plus the values its
    request bound for that Tool equal the recorded effective arguments and names.
    ``bound_arguments`` comes from ``request_bound_arguments`` over the request the
    response answers; the returned action carries the effective arguments.
    """

    # Written only when some were (``ToolCallValidated.arguments_decoded``).
    decoded = tuple(validation.get("arguments_decoded", ()))
    read = read_arguments(call.get("args"), decoded)
    bound = {} if read.malformed else bound_arguments.get(call.get("tool_name"), {})
    arguments = effective_arguments(read.value, bound)
    if (
        call.get("tool_name") != validation.get("tool_name")
        or call.get("tool_call_id") != validation.get("tool_call_id")
        or arguments != validation.get("arguments")
        # Written only when true (``ToolCallValidated.arguments_fenced``).
        or read.fenced != validation.get("arguments_fenced", False)
        or read.decoded != decoded
        or "bound_argument_names" not in validation
        or sorted(bound) != list(validation["bound_argument_names"])
    ):
        return None
    return ProposedAgentAction(
        tool_name=call["tool_name"],
        tool_call_id=call["tool_call_id"],
        arguments=arguments,
        arguments_invalid_json=read.malformed,
        arguments_fenced=read.fenced,
        arguments_decoded=read.decoded,
    )


# A cut-off reply that began a call as text, as the model's adapter reads it.
TEXT_CALL_CUT_OFF = "text_form_tool_call_cut_off"
# A blank run this long in a reply is named in its cut-off note.
_BLANK_RUN_NAMED = 1000
_BLANK = re.compile(r"\s+")


def _blank_run(text: str) -> int:
    """A reply's longest run of blank characters, when long enough to name."""
    run = max((len(match) for match in _BLANK.findall(text)), default=0)
    return run if run >= _BLANK_RUN_NAMED else 0


TurnDisposition = Literal[
    "singleton_admitted",
    "singleton_refused",
    "multi_call_admitted",
    "multi_call_refused",
    "text_refused",
    "reasoning_refused",
]

# A call after a question-opening call in the same response is refused with this.
QUESTION_OPENED = "researcher_question_opened"
# The calls after the one whose result ended the Attempt are refused with this.
ATTEMPT_ENDED = "attempt_ended_earlier"


@dataclass(frozen=True)
class TurnRouting:
    """A reply as admission routed it (ADR 0044), before any call runs.

    ``run_order`` holds the kept calls, function and output alike, in the order
    they run; ``actions`` the kept function calls as admission reads them, and
    ``note`` the one note the turn sends after its results.
    """

    run_order: tuple[ModelCall, ...]
    actions: tuple[ProposedAgentAction, ...]
    note: str | None


def _proposed(call: ModelCall) -> ProposedAgentAction:
    return ProposedAgentAction(
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        arguments=call.arguments.value,
        arguments_invalid_json=call.arguments.malformed,
        arguments_fenced=call.arguments.fenced,
        arguments_decoded=call.arguments.decoded,
    )


def _route(
    turn: ModelTurn,
    offered: frozenset[str],
    output_tools: frozenset[str],
    order: Callable[[Sequence[tuple[str, str]]], tuple[str, ...]],
):
    """Collapse copies, separate unoffered calls and order the kept calls.

    A copy has the Tool and canonical arguments of an earlier call of the reply,
    an output call's included. A cut-off reply's partial last call is dropped
    before anything else. An output must come alone: output calls are kept, as
    sent, only when no function call of the reply is left to run; otherwise they
    are refused.
    """
    seen: dict[str, str] = {}
    copies: dict[str, str] = {}
    distinct: list[ModelCall] = []
    outputs: list[ModelCall] = []
    for call in turn.complete_calls:
        identity = call_identity(_proposed(call))
        if identity in seen:
            copies[call.tool_call_id] = seen[identity]
            continue
        seen[identity] = call.tool_call_id
        (outputs if call.tool_name in output_tools else distinct).append(call)
    unoffered = [call for call in distinct if call.tool_name not in offered]
    kept = {call.tool_call_id: call for call in distinct if call.tool_name in offered}
    ordered = [
        kept[call_id]
        for call_id in order(
            [(call.tool_name, call_id) for call_id, call in kept.items()]
        )
    ]
    if ordered:
        return ordered, [], outputs, copies, unoffered
    return ordered, outputs, [], copies, unoffered


@dataclass
class _OrderedResponse:
    """A multi-call response whose calls are admitted one by one as they run."""

    response_event_id: str
    pending: list[ProposedAgentAction]
    exposed_tool_names: frozenset[str]
    bound_arguments: Mapping[str, Mapping[str, Any]]
    question: str | None = None


def refused_validation_ids(events: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """The admitted calls whose Tool refused them, shown to the model (#477 C8)."""
    refused_failures = {
        event["payload"]["failed_event_id"]
        for event in events
        if event["event_type"] == "ToolExecutionRefused"
    }
    started = {
        event["payload"]["started_event_id"]
        for event in events
        if event["event_type"] == "ObservedToolExecutionFailed"
        and event["event_id"] in refused_failures
    }
    return frozenset(
        event["payload"]["validation_event_id"]
        for event in events
        if event["event_type"] == "ObservedToolExecutionStarted"
        and event["event_id"] in started
    )


class ActionAdmissionSession:
    """Per-Attempt policy session that routes each reply and admits its calls.

    Every reply is read before any call runs (``route_reply``): copies collapse,
    unoffered calls are refused, a cut-off reply's partial last call is dropped
    and the kept calls are ordered by their declared dependencies. A plain
    single call keeps its whole-response decision before any contact. Each call
    of a routed or multi-call reply is admitted and recorded immediately before
    it runs, against the state its earlier calls left: a refused call returns its
    correction as its own Tool result, and once a call opens a researcher
    question or ends the Attempt, the remaining calls are refused.
    """

    def __init__(
        self,
        *,
        attempt_id: str,
        parent_activity: Activity,
        prior_events: Sequence[Mapping[str, Any]] = (),
        events: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        failure_source: AttemptFailureSource,
        eligible_tool_names: frozenset[str],
        action_validator: ActionValidator,
        argument_validator: ActionValidator | None = None,
        action_fingerprint: ActionFingerprint | None = None,
        admitted_repeat_error: str | None = None,
        continuation: ContinuationPolicy,
        resend_limit: int = 0,
        wording: CorrectionWording,
        repair_guidance: Mapping[str, str] | None = None,
        precondition: ActionPrecondition | None = None,
        explain: RefusalExplanation | None = None,
        question_tool_names: frozenset[str] = frozenset(),
        order: Callable[[Sequence[tuple[str, str]]], tuple[str, ...]] | None = None,
        output_tool_names: frozenset[str] = frozenset(),
        waiting: Callable[[frozenset[str]], frozenset[str]] | None = None,
        call_history: CallHistoryReader = attempt_call_history,
    ) -> None:
        self._attempt_id = attempt_id
        # Whose earlier calls a call repeats: the Attempt's by default, a role's own
        # work history when it supplies one.
        self._call_history = call_history
        self._parent_activity = parent_activity
        self._events = events or (lambda: prior_events)
        self._failure_source = failure_source
        self._eligible_tool_names = eligible_tool_names
        self._action_validator = action_validator
        self._argument_validator = argument_validator
        self._action_fingerprint = action_fingerprint
        self._admitted_repeat_error = admitted_repeat_error
        self._continuation = continuation
        self._resend_limit = resend_limit
        self._order = order or (lambda calls: tuple(call_id for _, call_id in calls))
        self._wording = wording
        self._repair_guidance = dict(repair_guidance or {})
        self._precondition = precondition
        self._explain = explain
        self._question_tool_names = question_tool_names
        # A refusal offers the output Tools too; ``waiting`` names the offered
        # Tools whose precondition does not hold now.
        self._output_tool_names = output_tool_names
        self._waiting = waiting
        self._response_values: dict[str, str] = {}
        self._ordered: _OrderedResponse | None = None
        # The routed turn's note, owed after its results, and its inspection facts.
        self.turn_note: str | None = None
        self._note_trace: dict[str, Any] | None = None
        # The kept calls whose results the note follows, until it is sent.
        self._noted_calls: tuple[str, ...] | None = None
        # Each call stopped as a repeat, and the earlier call it repeats.
        self._repeats: dict[str, str] = {}
        self._admitted_action_fingerprints: set[str] = set()
        # Every admitted call's identity in the Attempt: the loop guard reads
        # evidence only for a call with one of them.
        self._admitted_identities: set[str] = set()
        self._admitted_validation_event_ids: dict[str, str] = {}
        self._admitted_arguments: dict[str, dict[str, Any]] = {}
        self._admitted_calls: dict[str, ProposedAgentAction] = {}
        self._prior_validations = {
            event["event_id"]: event["payload"]
            for event in prior_events
            if event["event_type"] == "ToolCallValidated"
        }
        self._restore(prior_events)

    def restore_pending(self, request):
        """Restore the exact already-admitted call after checked retry authority."""
        admitted = ToolCallValidated.model_validate(
            self._prior_validations[request.validation_event_id]
        )
        action = ProposedAgentAction(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
        )
        validation = self._parent_activity.record(
            ToolCallValidated(
                attempt_id=self._attempt_id,
                tool_name=action.tool_name,
                tool_call_id=action.tool_call_id,
                arguments=action.arguments,
                status="valid",
                errors=(),
                bound_argument_names=request.bound_argument_names,
                continuation_request_event_id=request.request_event_id,
                arguments_fenced=admitted.arguments_fenced,
                arguments_decoded=admitted.arguments_decoded,
            )
        )
        self._admit(request.response_event_id, validation, action, "singleton_admitted")
        # The stopped call's reply still owes its routed note after the
        # replayed result; the pending request no longer carries it.
        routed = next(
            (
                event["payload"]
                for event in self._events()
                if event["event_type"] == "AgentTurnRouted"
                and event["payload"]["response_event_id"] == request.response_event_id
            ),
            None,
        )
        if routed is not None and routed["feedback"] is not None:
            self.turn_note = routed["feedback"]
            self._noted_calls = tuple(routed["run_order"])
            self._note_trace = fault_trace(
                self._routed_faults(routed, request.response_event_id), self._wording
            )

    def _routed_faults(
        self, routed: Mapping[str, Any], response_event_id: str
    ) -> TurnFaults:
        """A retained routed turn's faults as its note named them, rebuilt from its
        record and its raw reply, for the restored note's inspection (#373)."""
        events = self._events()
        by_id = {event["event_id"]: event for event in events}
        message = by_id[response_event_id]["payload"]["compatible_adapter_evidence"][
            "response"
        ]["message"]
        tools = {
            part["tool_call_id"]: part["tool_name"]
            for part in message["parts"]
            if part["part_kind"] == "tool-call"
        }
        counts = {fault["key"]: fault["count"] for fault in routed["faults"]}

        def counted(key: str) -> FaultCount | None:
            if key not in counts:
                return None
            if key == RESEND_KEY:
                return FaultCount(counts[key], self._resend_limit)
            return FaultCount(counts[key], self._reply_limit(response_event_id))

        copies = Counter((routed.get("copies") or {}).values())
        validations = [
            by_id[event_id]["payload"]
            for event_id in routed.get("unoffered_validation_event_ids", ())
        ]
        truncation = next(
            (
                event["payload"]
                for event in events
                if event["event_type"] == "ModelCompletionTruncated"
                and event["payload"]["response_event_id"] == response_event_id
            ),
            None,
        )
        content = reply_content(message.get("provider_details"))
        text = (
            "".join(p["content"] for p in message["parts"] if p["part_kind"] == "text")
            if content is None
            else content
        )
        return TurnFaults(
            copies=tuple(
                (tools[call_id], times + 1) for call_id, times in copies.items()
            ),
            copies_fault=counted(COPIES_KEY),
            unoffered=tuple(
                ActionRefused(
                    tool_name=validation["tool_name"],
                    errors=tuple(validation["errors"]),
                    arguments=validation["arguments"],
                    offered=frozenset(),
                    dropped=True,
                )
                for validation in validations
            ),
            unoffered_fault=counted(UNOFFERED_KEY),
            outputs=tuple(
                tools[call_id] for call_id in routed.get("refused_output_call_ids", ())
            ),
            outputs_fault=counted(OUTPUT_ALONE_KEY),
            cut_off=None
            if truncation is None
            else ReplyCutOff(
                output_tokens=truncation.get("completion_tokens")
                or truncation.get("requested_max_tokens")
                or 0,
                offered=frozenset(),
                calls=len(tools),
                text_call=TEXT_CALL_CUT_OFF
                in reply_quirks(message.get("provider_details")),
                blank_run=_blank_run(text),
            ),
            dropped_tool=None
            if routed.get("dropped_call_id") is None
            else tools[routed["dropped_call_id"]],
            cut_off_fault=counted(TRUNCATION_KEY),
            sent=counts[RESEND_KEY] + 1 if RESEND_KEY in counts else 0,
            resend_fault=counted(RESEND_KEY),
        )

    def note_after(self, call_ids: Sequence[str]) -> None:
        """Owe the turn's note after the results of these kept calls."""
        self._noted_calls = tuple(call_ids)

    def owed_note(self, call_ids: Sequence[str]) -> str | None:
        """The note owed after these calls' results, given once."""
        if self._noted_calls is None or tuple(call_ids) != self._noted_calls:
            return None
        self._noted_calls = None
        return self.turn_note

    def show_note(self, slot: str) -> None:
        """Show the routed turn's note to development inspection as it is sent."""
        inspection_event(
            "Fault message",
            attempt_id=self._attempt_id,
            slot=slot,
            text=self.turn_note,
            **(self._note_trace or {}),
        )

    def _restore(self, events):
        """Recover the Attempt's admitted call identities, every member's (#302).

        The domain's repeat refusals recover only along an exact provider continuation.
        """
        from bio_neb.agentic_journey.model_availability import prior_model_attempt_ids

        prior_ids = prior_model_attempt_ids(events, self._attempt_id)
        owned = {*prior_ids, self._attempt_id}
        # The loop guard reads the work's whole call history.
        called = self._call_history(list(events), self._attempt_id).attempt_ids
        by_id = {event["event_id"]: event for event in events}
        refused = refused_validation_ids(events)
        for event in events:
            payload = event.get("payload", {})
            if payload.get("attempt_id") not in owned | called:
                continue
            if (
                event["event_type"] == "AgentOutputValidated"
                and payload["status"] == "valid"
                and payload["attempt_id"] in owned
            ):
                # A completed collection member starts the next member's repeat
                # identity; correction counts stay with the Attempt.
                self._admitted_action_fingerprints.clear()
            elif event["event_type"] == "AgentActionTurnDispositionRecorded":
                disposition = payload["disposition"]
                if (
                    disposition in {"singleton_admitted", "multi_call_admitted"}
                    and payload["validation_event_ids"][0] not in refused
                ):
                    validation = by_id[payload["validation_event_ids"][0]]["payload"]
                    action = ProposedAgentAction(
                        tool_name=validation["tool_name"],
                        tool_call_id=validation["tool_call_id"],
                        arguments=validation["arguments"],
                    )
                    self._admitted_identities.add(call_identity(action))
                    if payload["attempt_id"] not in owned:
                        # A call of the work's history before this Attempt's own:
                        # only the loop guard reads it.
                        continue
                    if prior_ids and self._action_fingerprint is not None:
                        fingerprint = self._action_fingerprint(action)
                        if fingerprint is not None:
                            self._admitted_action_fingerprints.add(fingerprint)

    def admitted_validation_event_id(self, tool_call_id: str) -> str:
        """Return the retained validation that admitted this dispatched action."""

        try:
            return self._admitted_validation_event_ids[tool_call_id]
        except KeyError as error:
            raise Invariant(
                "tool_dispatched_unadmitted",
                "Tool action was not admitted before dispatch",
            ) from error

    def admitted_arguments(self, tool_call_id: str) -> dict[str, Any]:
        """Return the effective arguments admitted for this dispatched action."""

        try:
            return dict(self._admitted_arguments[tool_call_id])
        except KeyError as error:
            raise Invariant(
                "tool_dispatched_unadmitted",
                "Tool action was not admitted before dispatch",
            ) from error

    def response_value(self, tool_call_id: str) -> str | None:
        """What a response-level refusal of this call is about.

        The question Tool an earlier call opened, the checkpoint an earlier call's
        result ended the Attempt at, or the Tools the response named that its
        request did not offer.
        """

        return self._response_values.get(tool_call_id)

    def release_refused(self, tool_call_id: str) -> None:
        """A call whose Tool refused it did nothing: calling it again is no repeat
        (#477 C8)."""
        action = self._admitted_calls[tool_call_id]
        self._admitted_identities.discard(call_identity(action))
        if self._action_fingerprint is not None:
            fingerprint = self._action_fingerprint(action)
            if fingerprint is not None:
                self._admitted_action_fingerprints.discard(fingerprint)

    def execution_refusal_view(self, reason: str, statement: str) -> dict[str, str]:
        """A refused execution's view, by the failure module, with this reply's
        place in the failed turns in a row (#478 G1.4, F8)."""
        return execution_refused_view(
            reason, statement, self._turn_count(), self._wording
        )

    def evidence(self) -> Sequence[Mapping[str, Any]]:
        """The Run's checked evidence as this admission reads it now."""
        return self._events()

    def observe_tools(self, tool_names: frozenset[str]) -> ObservedToolSession:
        """Bind execution observation to this admission's exact Attempt identity."""
        from bio_neb.agentic_journey.tool_execution import ObservedToolSession

        return ObservedToolSession(
            attempt_id=self._attempt_id,
            parent_activity=self._parent_activity,
            admission_session=self,
            failure_source=self._failure_source,
            observed_tool_names=tool_names,
        )

    def route_reply(
        self,
        reply: ReceivedReply,
        turn: ModelTurn,
        exposed_tool_names: frozenset[str],
        output_tool_names: frozenset[str] = frozenset(),
        *,
        bound_arguments: Mapping[str, Mapping[str, Any]] | None = None,
        quirks: tuple[str, ...] = (),
        text: str = "",
    ) -> TurnRouting | None:
        """Read the whole reply before any call runs; None when it runs as sent.

        Copies collapse onto the first identical call, each distinct unoffered
        call is refused by its own validation, an output sent beside a call that
        runs is refused, a cut-off reply's partial last call is dropped and the
        kept calls are put in dependency order. A reply's copies, unoffered calls,
        refused outputs, cut-off and timed-out resends are its faults: the turn's
        one note names them, each with the reply's place in the failed turns in a
        row; when nothing of the reply works, the next request's check stops the
        Attempt at its limit (#447 B). ``quirks`` and
        ``text`` are what the model's adapter read in the raw reply; a cut-off's
        note names a call begun as text and a long blank run (#373).
        """
        response_event_id = reply.response_event_id
        ordered, outputs, refused_outputs, copies, unoffered = _route(
            turn, exposed_tool_names, output_tool_names, self._order
        )
        sent = 1 + reply.resends
        kept = [call for call in turn.complete_calls if call in ordered]
        self.turn_note = None
        self._noted_calls = None
        faulted = copies or unoffered or refused_outputs or turn.cut_off or sent > 1
        if not (faulted or ordered != kept):
            return None
        counts: dict[str, FaultCount] = {}
        place = self._turn_count()

        def count(key: str) -> FaultCount:
            counts[key] = place
            return place

        unoffered_errors = [
            (
                "tool_name_unavailable"
                if call.tool_name in self._eligible_tool_names
                else "tool_name_ineligible",
            )
            for call in unoffered
        ]
        refusals = [
            replace(
                self._refusal(_proposed(call), errors, exposed_tool_names),
                dropped=True,
            )
            for call, errors in zip(unoffered, unoffered_errors, strict=True)
        ]
        sent_times = {}
        for call_id in copies.values():
            sent_times[call_id] = sent_times.get(call_id, 1) + 1
        by_id = {call.tool_call_id: call for call in turn.calls}
        offered = exposed_tool_names | output_tool_names
        faults = TurnFaults(
            copies=tuple(
                (by_id[call_id].tool_name, times)
                for call_id, times in sent_times.items()
            ),
            copies_fault=count(COPIES_KEY) if copies else None,
            unoffered=tuple(refusals),
            unoffered_fault=count(UNOFFERED_KEY) if unoffered else None,
            outputs=tuple(call.tool_name for call in refused_outputs),
            outputs_fault=count(OUTPUT_ALONE_KEY) if refused_outputs else None,
            cut_off=ReplyCutOff(
                output_tokens=turn.output_tokens
                if turn.output_tokens is not None
                else reply.requested_max_tokens or 0,
                offered=offered,
                calls=len(turn.calls),
                text_call=TEXT_CALL_CUT_OFF in quirks,
                blank_run=_blank_run(text),
            )
            if turn.cut_off
            else None,
            dropped_tool=None if turn.partial is None else turn.partial.tool_name,
            cut_off_fault=count(TRUNCATION_KEY) if turn.cut_off else None,
            sent=sent if sent > 1 else 0,
            resend_fault=FaultCount(sent - 1, self._resend_limit) if sent > 1 else None,
        )
        # A reply with no call gets its correction alone: the model cannot act
        # on a note beside it.
        note = (
            render_correction(faults, self._wording)
            if faulted and (turn.calls or turn.cut_off)
            else None
        )
        if turn.cut_off:
            self._parent_activity.record(
                ModelCompletionTruncated(
                    attempt_id=self._attempt_id,
                    request_number=reply.request_number,
                    response_event_id=response_event_id,
                    requested_max_tokens=reply.requested_max_tokens,
                    completion_tokens=turn.output_tokens,
                    correction_count=counts[TRUNCATION_KEY].count,
                    feedback=note,
                )
            )
        validations = [
            self._validate(
                _proposed(call), exposed_tool_names, bound_arguments, refused=errors
            )[1]["event_id"]
            for call, errors in zip(unoffered, unoffered_errors, strict=True)
        ]
        run_order = (*ordered, *outputs)
        routed = self._parent_activity.record(
            AgentTurnRouted(
                attempt_id=self._attempt_id,
                response_event_id=response_event_id,
                run_order=tuple(call.tool_call_id for call in run_order),
                copies=copies,
                unoffered_validation_event_ids=tuple(validations),
                refused_output_call_ids=tuple(
                    call.tool_call_id for call in refused_outputs
                ),
                dropped_call_id=None
                if turn.partial is None
                else turn.partial.tool_call_id,
                faults=tuple(
                    TurnFaultCount(key=key, count=counted.count)
                    for key, counted in (
                        *counts.items(),
                        *(
                            ((RESEND_KEY, faults.resend_fault),)
                            if faults.resend_fault
                            else ()
                        ),
                    )
                ),
                feedback=note,
            )
        )
        del routed
        # The note is shown to development inspection where it is sent; a turn
        # whose output completes its member sends none (ADR 0044).
        self._note_trace = None if note is None else fault_trace(faults, self._wording)
        self.turn_note = note
        return TurnRouting(
            run_order=run_order,
            actions=tuple(_proposed(call) for call in ordered),
            note=note,
        )

    def admit_turn(
        self,
        response_event_id: str,
        actions: Sequence[ProposedAgentAction],
        exposed_tool_names: frozenset[str],
        *,
        bound_arguments: Mapping[str, Mapping[str, Any]] | None = None,
        routed: bool = False,
    ) -> TurnDisposition | None:
        """Admit a reply's calls in their run order, each before any contact.

        ``bound_arguments`` are the request's server-bound values per exposed Tool.
        Each call is validated, recorded and, when admitted, dispatched with its
        effective arguments. A routed reply's calls, or a reply's several calls,
        are admitted one by one as they run; once a call opens a researcher
        question, the later calls of the reply are refused.
        """

        if not actions:
            return None
        if len(actions) == 1 and not routed:
            action, validation = self._validate(
                actions[0], exposed_tool_names, bound_arguments
            )
            if validation["payload"]["errors"]:
                # The whole response is refused; the model sends it again.
                raise ModelRetry(
                    self._refuse_call(
                        response_event_id,
                        validation,
                        action,
                        exposed_tool_names,
                        "singleton_refused",
                    )
                )
            self._admit(response_event_id, validation, action, "singleton_admitted")
            return "singleton_admitted"

        self._ordered = _OrderedResponse(
            response_event_id=response_event_id,
            pending=list(actions),
            exposed_tool_names=exposed_tool_names,
            bound_arguments=dict(bound_arguments or {}),
        )
        return None

    def ordered_response_event_id(self) -> str:
        return self._ordered.response_event_id

    def admitted_call(self, tool_call_id: str) -> ProposedAgentAction:
        """The admitted call with its effective arguments."""

        return self._admitted_calls[tool_call_id]

    def ordered_call(self, tool_call_id: str) -> bool:
        """Whether this call waits in a multi-call response for its own admission."""

        return self._ordered is not None and any(
            item.tool_call_id == tool_call_id for item in self._ordered.pending
        )

    def ordered_arguments(self, tool_call_id: str) -> tuple[dict[str, Any], bool]:
        """A waiting call's effective arguments and whether their shape is invalid.

        The framework validates every call of a response before the first runs; a
        call whose shape admission will refuse skips that validation, so its
        refusal is recorded in its turn rather than spent as a framework retry.
        """
        ordered = self._ordered
        proposed = next(
            item for item in ordered.pending if item.tool_call_id == tool_call_id
        )
        bound = (
            {}
            if proposed.arguments_invalid_json
            else dict(ordered.bound_arguments.get(proposed.tool_name, {}))
        )
        arguments = effective_arguments(proposed.arguments, bound)
        invalid = proposed.arguments_invalid_json or bool(
            self._argument_validator
            and self._argument_validator(replace(proposed, arguments=arguments))
        )
        return dict(arguments), invalid

    def admit_call(self, tool_call_id: str) -> dict[str, Any]:
        """Admit the next call of a multi-call response immediately before it runs.

        Returns its effective arguments, or raises ``ToolFailed`` with its correction
        when it is refused; the same refused call beyond its allowance stops the
        Attempt. Its admission reads the state the earlier calls left.
        """
        ordered = self._ordered
        if ordered is None or not ordered.pending:
            raise Invariant(
                "multi_call_order_lost",
                "no call of a multi-call response waits for admission",
            )
        if ordered.pending[0].tool_call_id != tool_call_id:
            raise Invariant(
                "multi_call_order_lost",
                "multi-call response ran out of its sent order",
            )
        proposed = ordered.pending.pop(0)
        if ordered.question is not None:
            self._response_values[proposed.tool_call_id] = ordered.question
        action, validation = self._validate(
            proposed,
            ordered.exposed_tool_names,
            ordered.bound_arguments,
            refused=(QUESTION_OPENED,) if ordered.question is not None else None,
        )
        if validation["payload"]["errors"]:
            raise ToolFailed(
                self._refuse_call(
                    ordered.response_event_id,
                    validation,
                    action,
                    ordered.exposed_tool_names,
                    "multi_call_refused",
                )
            )
        self._admit(
            ordered.response_event_id, validation, action, "multi_call_admitted"
        )
        if action.tool_name in self._question_tool_names:
            ordered.question = action.tool_name
        return dict(action.arguments)

    def close_response(self, checkpoint: str):
        """Refuse the calls a response still holds once an earlier call ended it.

        A question an earlier call opened is their reason; otherwise the checkpoint
        the Attempt ended at. They are recorded before the Attempt's terminal and
        never run.
        """
        corrections = []
        ordered = self._ordered
        if ordered is None:
            return ()
        reason = QUESTION_OPENED if ordered.question is not None else ATTEMPT_ENDED
        while ordered.pending:
            proposed = ordered.pending.pop(0)
            self._response_values[proposed.tool_call_id] = (
                ordered.question or checkpoint
            )
            action, validation = self._validate(
                proposed,
                ordered.exposed_tool_names,
                ordered.bound_arguments,
                refused=(reason,),
            )
            feedback = self._refuse_call(
                ordered.response_event_id,
                validation,
                action,
                ordered.exposed_tool_names,
                "multi_call_refused",
                stop=False,
            )
            corrections.append((action, feedback))
        self._ordered = None
        return tuple(corrections)

    def _refuse_call(
        self,
        response_event_id: str,
        validation: Mapping[str, Any],
        action: ProposedAgentAction,
        exposed_tool_names: frozenset[str],
        disposition: TurnDisposition,
        *,
        stop: bool = True,
    ) -> str:
        """Record one refused call and return its correction.

        It names its reply's place in the failed turns in a row; when nothing of
        the reply works, the next request's check stops the Attempt at its limit
        (#447 B). A call refused because the Attempt already ended (``stop``
        false) is recorded and held.
        """
        errors = tuple(validation["payload"]["errors"])
        turn = self._turn_count()
        if errors == (REPEATED_CALL,):
            key = repeat_key(action.tool_name, self._repeats[action.tool_call_id])
            earlier = self._earlier_view(self._repeats[action.tool_call_id])
            correction = RepeatedCall(
                tool_name=action.tool_name,
                count=turn.count,
                cap=turn.limit,
                offered=frozenset(exposed_tool_names) | self._output_tool_names,
                earlier_status=earlier["status"],
                earlier_next=earlier["next"],
                waiting=self._waiting_now(exposed_tool_names),
            )
        else:
            key = correction_key(action.tool_name, errors, dict(action.arguments))
            correction = replace(
                self._refusal(action, errors, exposed_tool_names), fault=turn
            )
        feedback = render_correction(correction, self._wording)
        self._spend_correction(
            response_event_id, (validation["event_id"],), disposition, key, feedback
        )
        if not stop:
            # A call refused because an earlier call ended the response: recorded
            # and held with the Attempt's end, never read next.
            self._inspect(correction, feedback, "held", name="Held correction")
        elif turn.count < turn.limit:
            # Only a correction the model may read next is shown; the stopping
            # reply's is never sent. A refused call's correction is its own
            # slot's.
            self._inspect(correction, feedback, "call_slot")
        if errors == (REPEATED_CALL,):
            self._show_repeat(action, validation)
        return feedback

    def _earlier_view(self, validation_event_id: str) -> Mapping[str, Any]:
        """The model view the earlier call of a repeat was shown (#478 G1.4)."""
        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        events = self.evidence()
        by_id = {event["event_id"]: event for event in events}
        started = next(
            event
            for event in events
            if event["event_type"] == "ObservedToolExecutionStarted"
            and event["payload"]["validation_event_id"] == validation_event_id
        )
        succeeded = next(
            event
            for event in events
            if event["event_type"] == "ObservedToolExecutionSucceeded"
            and event["payload"]["started_event_id"] == started["event_id"]
        )
        return tool_model_view(
            model_view_of(started["payload"]["tool_name"]),
            by_id[succeeded["payload"]["stage_result_event_id"]],
        )

    def _refusal(
        self,
        action: ProposedAgentAction,
        errors: tuple[str, ...],
        exposed_tool_names: frozenset[str],
    ) -> ActionRefused:
        """One refused call's correction facts, with the role's sentence per reason."""
        return ActionRefused(
            tool_name=action.tool_name,
            errors=errors,
            arguments=action.arguments,
            offered=frozenset(exposed_tool_names) | self._output_tool_names,
            waiting=self._waiting_now(exposed_tool_names),
            explanations={
                code: sentence
                for code in errors
                if self._explain is not None
                and (sentence := self._explain(action, code))
            },
            guidance=self._repair_guidance.get(action.tool_name),
        )

    def _waiting_now(self, exposed_tool_names: frozenset[str]) -> frozenset[str]:
        """The exposed Tools that cannot run now, read once per refusal, and only
        when the model's wording renders them."""
        if self._waiting is None or not names_waiting(self._wording):
            return frozenset()
        return self._waiting(frozenset(exposed_tool_names))

    def _show_repeat(self, action, validation) -> None:
        """Show a repeat live and to development inspection; never stored."""
        progress = RepeatProgress(
            attempt_id=self._attempt_id,
            tool_name=action.tool_name,
            repeat=self._turn_count().count,
            cap=self._turn_count().limit,
            source_event_id=validation["event_id"],
        )
        turn_progress(progress)
        inspection_event(
            "Repeated call",
            journey_id=validation["journey_id"],
            **progress.model_dump(mode="json"),
        )

    def _validate(
        self,
        proposed: ProposedAgentAction,
        exposed_tool_names: frozenset[str],
        bound_arguments: Mapping[str, Mapping[str, Any]] | None,
        *,
        refused: tuple[str, ...] | None = None,
    ) -> tuple[ProposedAgentAction, dict[str, Any]]:
        """Retain one call's validation over its effective arguments.

        ``refused`` replaces the admission gates with a response-level reason.
        """
        bound = (
            {}
            if proposed.arguments_invalid_json
            else dict((bound_arguments or {}).get(proposed.tool_name, {}))
        )
        action = replace(
            proposed, arguments=effective_arguments(proposed.arguments, bound)
        )
        errors = (
            refused
            if refused is not None
            else action_admission_errors(
                action,
                eligible_tool_names=self._eligible_tool_names,
                exposed_tool_names=exposed_tool_names,
                action_validator=self._action_validator,
                action_fingerprint=self._action_fingerprint,
                admitted_action_fingerprints=frozenset(
                    self._admitted_action_fingerprints
                ),
                admitted_repeat_error=self._admitted_repeat_error,
                bound_arguments=bound,
                argument_validator=self._argument_validator,
                precondition=self._precondition,
            )
        )
        if (
            not errors
            and refused is None
            and call_identity(action) in self._admitted_identities
        ):
            # The loop guard: a lawful call repeating an earlier one with nothing
            # changed since is stopped before contact (ADR 0044).
            earlier = repeated_call_of(
                self._events(),
                attempt_id=self._attempt_id,
                action=action,
                history=self._call_history,
            )
            if earlier is not None:
                self._repeats[action.tool_call_id] = earlier
                errors = (REPEATED_CALL,)
        validation = self._parent_activity.record(
            ToolCallValidated(
                attempt_id=self._attempt_id,
                tool_name=action.tool_name,
                tool_call_id=action.tool_call_id,
                arguments=dict(action.arguments),
                status="invalid" if errors else "valid",
                errors=errors,
                bound_argument_names=tuple(sorted(bound)),
                arguments_fenced=action.arguments_fenced,
                arguments_decoded=action.arguments_decoded,
            )
        )
        return action, validation

    def _admit(
        self,
        response_event_id: str,
        validation: Mapping[str, Any],
        action: ProposedAgentAction,
        disposition: TurnDisposition,
    ) -> None:
        """Record one admitted call and keep it for dispatch and repeat identity."""
        self._record_disposition(
            response_event_id, (validation["event_id"],), disposition
        )
        self._admitted_validation_event_ids[action.tool_call_id] = validation[
            "event_id"
        ]
        self._admitted_arguments[action.tool_call_id] = dict(action.arguments)
        self._admitted_calls[action.tool_call_id] = action
        self._admitted_identities.add(call_identity(action))
        if self._action_fingerprint is not None:
            fingerprint = self._action_fingerprint(action)
            if fingerprint is not None:
                self._admitted_action_fingerprints.add(fingerprint)

    def refuse_without_call(
        self,
        response_event_id: str,
        offered_tool_names: frozenset[str],
        unusable: UnusableReply | None = None,
    ) -> NoReturn:
        """A response with no Tool call is no result: correct it under its own key.

        A reply with nothing to act on (only reasoning, nothing at all, a call
        stub) is not a text ending: no call arrived, and it counts under the
        reasoning-only key and stop, with a sentence naming what came back
        (#421; owner, #444 Q4, Q5). Its correction comes alone: a routed turn's
        note on a reply with no call is not sent (#444).
        """
        if unusable is not None:
            disposition, key = "reasoning_refused", REASONING_ONLY_KEY
            correction = ReasoningOnly(
                offered_tool_names, fault=self._turn_count(), reply=unusable
            )
        else:
            disposition, key = "text_refused", TEXT_ENDING_KEY
            correction = TextEnding(offered_tool_names, fault=self._turn_count())
        feedback = render_correction(correction, self._wording)
        self._spend_correction(response_event_id, (), disposition, key, feedback)
        if correction.fault.count < correction.fault.limit:
            # The stopping reply's correction is never sent, so never shown.
            self._inspect(correction, feedback, "retry")
        raise ModelRetry(reply_retry(feedback, None))

    def _turn_count(self) -> FaultCount:
        """This reply's place in the failed turns in a row, if it fails, and the
        place that stops the Attempt: every fault of one reply names both (#447 B)."""
        count, limit = current_turn(
            self._events(), attempt_id=self._attempt_id, policy=self._continuation
        )
        return FaultCount(count, limit)

    def _reply_limit(self, response_event_id: str) -> int:
        """The limit of failed turns in a row in the stage of the request a
        retained reply answered."""
        stage = None
        for event in self._events():
            if event["event_type"] == "ModelRequestRecorded":
                stage = event["payload"].get("stage")
            if event["event_id"] == response_event_id:
                break
        return self._continuation.limit(stage)

    def _inspect(
        self, correction, text: str, slot: str, *, name: str = "Fault message"
    ) -> None:
        """Show each correction to development inspection; never stored (#373)."""
        inspection_event(
            name,
            attempt_id=self._attempt_id,
            slot=slot,
            text=text,
            **fault_trace(correction, self._wording),
        )

    def _spend_correction(
        self,
        response_event_id: str,
        validation_event_ids: tuple[str, ...],
        disposition: TurnDisposition,
        key: str,
        feedback: str,
    ) -> None:
        """Record the refused turn with its key and its reply's place in the
        failed turns in a row."""
        self._record_disposition(
            response_event_id,
            validation_event_ids,
            disposition,
            feedback,
            correction=(key, self._turn_count().count),
        )

    def _record_disposition(
        self,
        response_event_id: str,
        validation_event_ids: tuple[str, ...],
        disposition: TurnDisposition,
        feedback: str | None = None,
        *,
        correction: tuple[str, int] | None = None,
    ) -> None:
        key, count = correction or (None, None)
        self._parent_activity.record(
            AgentActionTurnDispositionRecorded(
                attempt_id=self._attempt_id,
                response_event_id=response_event_id,
                validation_event_ids=validation_event_ids,
                disposition=disposition,
                feedback=feedback,
                correction_key=key,
                correction_count=count,
            )
        )


def _keep(response: ModelResponse, run: Sequence[str] | None) -> None:
    """Give the reply the framework keeps the parts the composer keeps of it.

    When no call of a reply is left to run, the framework keeps the reply it
    handed admission, so its kept form replaces it before the retry.
    """
    (kept,) = read_messages_record([kept_reply(messages_record([response])[0], run)])
    response.parts = kept.parts
    response.finish_reason = kept.finish_reason
    response.provider_details = kept.provider_details


class PydanticActionAdmissionCapability(AbstractCapability[Any]):
    """Thin public Pydantic AI 2.25 hook around an admission session."""

    def __init__(
        self,
        session: ObservedToolSession | Callable[[RunContext[Any]], ObservedToolSession],
        *,
        received: Callable[[RunContext[Any]], ReceivedReply | None],
        observe_outputs: Callable[[str, ModelResponse, Sequence[ToolDefinition]], None],
        wording: CorrectionWording,
        output_tool_names: frozenset[str] = frozenset(),
        output_repair: Callable[[], FaultCount | None] = lambda: None,
        output_refusal: Callable[[Any], OutputFaults | None] = lambda _arguments: None,
    ) -> None:
        self._session = session
        # The role's faults for schema-refused arguments, from its output
        # session; None leaves the schema's.
        self._output_refusal = output_refusal
        self._received = received
        # The member's output repair count once its refusal is recorded.
        self._output_repair = output_repair
        # Records the reply's output calls once routing has kept them.
        self._observe_outputs = observe_outputs
        self._wording = wording
        self._output_tool_names = output_tool_names
        self._owed_answer: ToolReturnPart | None = None

    def _current(self, ctx: RunContext[Any]) -> ObservedToolSession:
        return self._session(ctx) if callable(self._session) else self._session

    async def prepare_output_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Render output Tools with the same optional-field rule as function Tools."""

        return [output_tool_schema(definition) for definition in tool_defs]

    async def before_output_validate(
        self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
    ) -> Any:
        """Give the output Tool its arguments as the Model turn reads them.

        A fenced call is validated without its fence, and an object argument
        sent as JSON text as that object; a call that is no JSON object is
        refused with the retry rendering's repair, so the framework neither
        reads a fence of its own nor writes a repair.
        """
        definition = output_context.tool_def
        read = read_arguments(
            output,
            ()
            if definition is None
            else object_arguments(definition.parameters_json_schema),
        )
        if read.malformed:
            self._refuse_output(
                output_schema_errors(output_context.output_type, read), read
            )
        return read.text

    async def on_output_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
        error: ValidationError | ModelRetry,
    ) -> Any:
        """Send a schema refusal as the retry rendering writes it."""
        if isinstance(error, ValidationError):
            self._refuse_output(
                error.errors(include_url=False, include_context=False),
                read_arguments(output),
            )
        raise error

    def _refuse_output(self, schema_errors, read: ModelArguments) -> NoReturn:
        # ``read`` is the Model turn's reading (ADR 0043: one reader); a call that
        # is no JSON object keeps the schema's correction.
        faults = None if read.malformed else self._output_refusal(read.value)
        raise ModelRetry(
            render_correction(
                OutputRefused(
                    schema_errors=None if faults is not None else tuple(schema_errors),
                    faults=faults,
                    fault=self._output_repair(),
                    tool_name=next(iter(self._output_tool_names), None),
                ),
                self._wording,
            )
        )

    async def before_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
    ) -> Any:
        """Give a function call its effective arguments for validation.

        The model's call part stays unchanged in history; only validation and the
        handler receive the server-bound values admission filled in. A call of a
        multi-call response is admitted later, as it runs; one whose shape
        admission will refuse skips the framework's validation.
        """

        session = self._current(ctx)
        if session.ordered_call(call.tool_call_id):
            arguments, invalid = session.ordered_arguments(call.tool_call_id)
            if invalid:
                raise SkipToolValidation(arguments)
            return arguments
        return session.admitted_arguments(call.tool_call_id)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Admit a multi-call response's call in its turn, just before it runs."""

        session = self._current(ctx)
        if session.ordered_call(call.tool_call_id):
            session.admit_call(call.tool_call_id)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Return a Tool's result to the model as the model is shown it (#422): its
        model view, with the recorded event's id kept beside it and never sent
        (#478 G1.3)."""
        from bio_neb.agentic_journey.tool_execution import ShownResult, model_visible

        shown = model_visible(result)[0]
        if isinstance(shown, ShownResult):
            return ToolReturn(
                return_value=shown.view, metadata={"event_id": shown.event_id}
            )
        return shown

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Route the whole reply before any call runs; the kept reply is returned.

        The kept reply, not the raw one, enters the framework's history and runs.
        When none of its calls is left to run, the framework keeps the reply this
        hook was handed, so the kept reply's parts replace it before the retry.
        """
        reply = self._received(ctx)
        if reply is None:
            raise Invariant(
                "reply_unretained",
                "model response was not retained before action admission",
            )
        event_id = reply.response_event_id

        parameters = request_context.model_request_parameters
        turn = read_turn(
            response, (*parameters.function_tools, *parameters.output_tools)
        )
        session = self._current(ctx)
        offered = frozenset(item.name for item in parameters.function_tools)
        outputs = frozenset(item.name for item in parameters.output_tools)
        bound = request_bound_arguments(parameters.function_tools)
        details = response.provider_details
        content = reply_content(details)
        routing = session.route_reply(
            reply,
            turn,
            offered,
            outputs,
            bound_arguments=bound,
            quirks=reply_quirks(details),
            text="".join(
                part.content for part in response.parts if part.part_kind == "text"
            )
            if content is None
            else content,
        )
        if not turn.calls and not turn.cut_off:
            # Kept as the composer keeps it: a stub without its fragment, a reply
            # with nothing to send padded.
            _keep(response, None)
            session.refuse_without_call(event_id, offered | outputs, turn.unusable)
        if routing is None:
            self._outputs(session, event_id, response, response, parameters)
            try:
                session.admit_turn(
                    event_id,
                    [
                        _proposed(call)
                        for call in turn.calls
                        if call.tool_name not in self._output_tool_names
                    ],
                    offered,
                    bound_arguments=bound,
                )
            except ModelRetry as refused:
                # A refused single call did not run. Kept in its reply, it is
                # answered by its correction in its own slot.
                (call,) = (
                    call
                    for call in turn.calls
                    if call.tool_name not in self._output_tool_names
                )
                _keep(response, [call.tool_call_id])
                self._owed_answer = ToolReturnPart(
                    tool_name=call.tool_name,
                    content=refused.message,
                    tool_call_id=call.tool_call_id,
                    outcome="failed",
                )
                raise
            return response
        (kept,) = read_messages_record(
            [
                kept_reply(
                    messages_record([response])[0],
                    [call.tool_call_id for call in routing.run_order],
                )
            ]
        )
        if not routing.run_order:
            _keep(response, ())
            session.show_note("retry")
            raise ModelRetry(reply_retry(None, routing.note))
        self._outputs(
            session,
            event_id,
            kept,
            response,
            parameters,
            note=routing.note,
        )
        session.admit_turn(
            event_id, routing.actions, offered, bound_arguments=bound, routed=True
        )
        if routing.note is not None:
            session.note_after([call.tool_call_id for call in routing.run_order])
        return kept

    def _outputs(
        self, session, event_id, reply, response, parameters, *, note=None
    ) -> None:
        """Record the reply's output calls as routing kept them.

        Several output calls in one reply, where the model's policy corrects them,
        leave the kept reply with none: the correction, then any routed note, is
        the next request's reason (#373 E4).
        """
        try:
            self._observe_outputs(event_id, reply, parameters.output_tools)
        except SeveralOutputs as several:
            _keep(response, ())
            if note is not None:
                session.show_note("retry")
            raise ModelRetry(reply_retry(several.correction, note)) from None

    def owed(
        self, ctx: RunContext[Any], reply: ModelResponse | None
    ) -> tuple[ToolReturnPart | None, str | None]:
        """What admission owes the next request, for the composer to place.

        A refused single call kept in its reply is answered by its correction in
        its own slot (#373), once. A routed turn's note is owed after the results
        of its kept calls, which ``reply`` holds; a retried stop's reply,
        restored from its transcript, owes it the same way.
        """
        answer, self._owed_answer = self._owed_answer, None
        session = self._current(ctx)
        note = (
            None
            if reply is None
            else session.owed_note([call.tool_call_id for call in reply.tool_calls])
        )
        if note is not None:
            session.show_note("after_results")
        return answer, note


__all__ = [
    "action_admission_errors",
    "output_tool_schema",
    "request_bound_arguments",
    "validated_model_call",
    "ActionAdmissionSession",
    "PydanticActionAdmissionCapability",
    "ProposedAgentAction",
]
