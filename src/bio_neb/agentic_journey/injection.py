"""The one Injection slot's pattern registry (ADR 0048).

A pattern names when it speaks (its trigger, read from signals the loop already
has) and what it says (its message). The selected model's endpoint policy switches
patterns on; policy config tunes them per role or stage. An injected message is a
name-tagged user message in the pending request, so the framework history, the
Journey's request record and the model read the same bytes. A pattern only writes
text: it never ends an Attempt or writes an answer.

The registry ships no pattern; a model's patterns arrive with its endpoint policy's work.
This module imports no ``pydantic_ai``.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, StrictBool, ValidationError
from pydantic_core import InitErrorDetails

from bio_neb.agentic_journey.model_turn import read_arguments


class PatternTuning(BaseModel):
    """One pattern's tunable values; ``on`` switches it for a scope."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    on: StrictBool = False


@dataclass(frozen=True)
class ReplyCall:
    """One call of the latest reply, as the model sent it."""

    tool_name: str
    # The arguments as the Model turn read them, the call reasoning moved out.
    arguments: Mapping[str, Any]
    # The call's own reasoning, when the model's adapter moved one out.
    reasoning: str | None
    # Whether the called Tool's schema takes any argument.
    takes_arguments: bool


@dataclass(frozen=True)
class ReplyRefusal:
    """One refused call of the latest reply, with its key's count in the Attempt."""

    tool_name: str
    # The arguments as the model sent them, never the effective ones.
    arguments: Mapping[str, Any]
    codes: tuple[str, ...]
    count: int
    # The count at which the Attempt stops, or None when the key has no limit.
    limit: int | None
    # A refused single call: the kept reply holds no call, so the history
    # the model reads next does not show it.
    whole_reply: bool


@dataclass(frozen=True)
class LatestReply:
    """The Attempt's latest reply as recorded, and what the loop did with it."""

    # The quirks the model's adapter read in it (``model_settings.ReplyQuirk``).
    quirks: tuple[str, ...]
    # Its text as received: the wire content when the adapter's reading changed it.
    text: str
    calls: tuple[ReplyCall, ...]
    # It ended in text, with no Tool call, and was corrected.
    text_ended: bool
    refusals: tuple[ReplyRefusal, ...]
    # A routed reply's fault keys and their counts in the Attempt (ADR 0044).
    faults: Mapping[str, int]


@dataclass(frozen=True)
class ToolResult:
    """One Tool result the pending request carries, as the model reads it."""

    tool_name: str
    content: Any


@dataclass(frozen=True)
class InjectionSignals:
    """What a trigger may read about the pending request."""

    agent_role: str
    stage: str
    # The pending request's number in the Attempt, counting retained requests.
    request_number: int
    # Tool results the Attempt's history holds, the pending request's included.
    tool_results: int
    # Earlier injected messages in the history, by pattern name.
    sent: Mapping[str, int]
    # The pending request's number among the Attempt's requests in its stage.
    stage_request_number: int
    # The latest reply, or None before the first.
    reply: LatestReply | None
    # The Tool results the pending request carries.
    results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class InjectionPattern:
    """A registered pattern: its tuning shape and its message, or None to stay silent."""

    name: str
    tuning: type[PatternTuning]
    message: Callable[[InjectionSignals, Any], str | None]


PATTERNS: dict[str, InjectionPattern] = {}

_TAG = "[injection:{name}]\n"


def _registered_pattern(name: str) -> str:
    if name not in PATTERNS:
        raise ValueError("not a registered injection pattern")
    return name


PatternName = Annotated[str, AfterValidator(_registered_pattern)]


def _valid_tunings(value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    # Each entry is checked against its pattern's own shape, and a refusal names the
    # entry's path; omissions stay omitted.
    errors = []
    for name, tuning in value.items():
        try:
            PATTERNS[name].tuning.model_validate(tuning)
        except ValidationError as error:
            errors.extend(
                InitErrorDetails(
                    type=item["type"],
                    loc=(name, *item["loc"]),
                    input=item["input"],
                    ctx=item.get("ctx", {}),
                )
                for item in error.errors()
            )
    if errors:
        raise ValidationError.from_exception_data("injection", errors)
    return value


InjectionTunings = Annotated[
    dict[PatternName, dict[str, Any]], AfterValidator(_valid_tunings)
]
"""A scope's sparse pattern entries: an omitted key keeps the wider scope's value."""


class InjectionPolicy(BaseModel):
    """The patterns an Attempt's requests switch on, with their tuning, resolved once.

    ``patterns`` serve every request whose stage has no entry of its own;
    ``stage_patterns`` serve the rest. Only switched-on patterns are listed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    patterns: dict[str, dict[str, Any]]
    stage_patterns: dict[str, dict[str, dict[str, Any]]] = {}

    def for_stage(self, stage: str | None) -> Mapping[str, dict[str, Any]]:
        entry = self.stage_patterns.get(stage) if stage is not None else None
        return self.patterns if entry is None else entry

    def switched_on(self) -> frozenset[str]:
        """Every pattern some request of the Attempt may send."""
        return frozenset(self.patterns).union(*self.stage_patterns.values())


def _resolved(
    switched_on: Iterable[str], layers: Iterable[Mapping[str, Mapping[str, Any]] | None]
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {name: {"on": True} for name in switched_on}
    for layer in layers:
        for name, tuning in (layer or {}).items():
            values[name] = {**values.get(name, {}), **tuning}
    # JSON values, so the policy compares equal to its retained record.
    return {
        name: dumped
        for name, tuning in sorted(values.items())
        if (
            dumped := PATTERNS[name]
            .tuning.model_validate(tuning)
            .model_dump(mode="json")
        )["on"]
    }


def resolve_injection(
    switched_on: Iterable[str],
    global_layer: Mapping[str, Mapping[str, Any]] | None,
    role_layer: Mapping[str, Mapping[str, Any]] | None,
    stage_layers: Mapping[str, Mapping[str, Mapping[str, Any]] | None],
) -> InjectionPolicy | None:
    """The model's switched-on set, then global, role and stage; the narrowest wins.

    None when no request of the role switches a pattern on, so the Attempt's
    recorded policy is unchanged.
    """
    switched_on = tuple(switched_on)
    patterns = _resolved(switched_on, (global_layer, role_layer))
    stage_patterns = {
        stage: _resolved(switched_on, (global_layer, role_layer, layer))
        for stage, layer in stage_layers.items()
        if layer
    }
    if not patterns and not any(stage_patterns.values()):
        return None
    return InjectionPolicy(patterns=patterns, stage_patterns=stage_patterns)


_REFUSED = frozenset({"singleton_refused", "multi_call_refused"})
_REPLY_RECORDS = frozenset(
    {
        "ModelRequestRecorded",
        "ModelResponseRecorded",
        "AgentActionTurnDispositionRecorded",
        "AgentTurnRouted",
        "ToolCallValidated",
    }
)


def latest_reply(
    events: Sequence[Mapping[str, Any]],
    attempt_id: str,
    *,
    limit: Callable[[str], int | None],
) -> LatestReply | None:
    """The Attempt's latest recorded reply and what its turn records say of it.

    Read from the raw reply as recorded (a routed reply's kept copy carries no
    provider details, #336) and the loop's own dispositions and routing for it.
    A call takes arguments when its request showed the Tool a field besides the
    call reasoning; server-bound fields are not shown. ``limit`` gives a
    correction key's stopping count.
    """
    from bio_neb.agentic_journey.model_settings import (
        CALL_REASONING_ARGUMENT,
        reply_content,
        reply_quirks,
    )

    own = [
        event
        for event in events
        if event["event_type"] in _REPLY_RECORDS
        and event["payload"]["attempt_id"] == attempt_id
    ]
    replies = [
        i for i, e in enumerate(own) if e["event_type"] == "ModelResponseRecorded"
    ]
    if not replies:
        return None
    response = own[replies[-1]]
    shown = next(
        e["payload"]["compatible_adapter_evidence"]["request"]
        for e in reversed(own[: replies[-1]])
        if e["event_type"] == "ModelRequestRecorded"
    )
    fields = {
        tool["name"]: set(tool["parameters_json_schema"]["properties"])
        - {CALL_REASONING_ARGUMENT}
        for tool in (*shown["function_tools"], *shown["output_tools"])
    }
    message = response["payload"]["compatible_adapter_evidence"]["response"]["message"]
    parts = message["parts"]
    reasoning = {
        owner: part["content"]
        for part in parts
        if part["part_kind"] == "thinking"
        and (owner := (part["provider_details"] or {}).get("tool_call_id"))
    }

    def of_reply(event_type: str) -> list[Mapping[str, Any]]:
        return [
            event["payload"]
            for event in own
            if event["event_type"] == event_type
            and event["payload"]["response_event_id"] == response["event_id"]
        ]

    dispositions = of_reply("AgentActionTurnDispositionRecorded")
    validations = {
        event["event_id"]: event["payload"]
        for event in own
        if event["event_type"] == "ToolCallValidated"
    }
    # Each call as sent; a validation records the effective arguments instead.
    sent = {
        part["tool_call_id"]: read_arguments(part["args"]).value
        for part in parts
        if part["part_kind"] == "tool-call"
    }
    text = reply_content(message["provider_details"])
    return LatestReply(
        quirks=reply_quirks(message["provider_details"]),
        text="".join(p["content"] for p in parts if p["part_kind"] == "text")
        if text is None
        else text,
        calls=tuple(
            ReplyCall(
                tool_name=part["tool_name"],
                arguments=sent[part["tool_call_id"]],
                reasoning=reasoning.get(part["tool_call_id"]),
                takes_arguments=bool(fields.get(part["tool_name"])),
            )
            for part in parts
            if part["part_kind"] == "tool-call"
        ),
        text_ended=any(item["disposition"] == "text_refused" for item in dispositions),
        refusals=tuple(
            ReplyRefusal(
                tool_name=validation["tool_name"],
                arguments=sent[validation["tool_call_id"]],
                codes=tuple(validation["errors"]),
                count=item["correction_count"],
                limit=limit(item["correction_key"]),
                whole_reply=item["disposition"] == "singleton_refused",
            )
            for item in dispositions
            if item["disposition"] in _REFUSED
            for validation in (validations[i] for i in item["validation_event_ids"])
        ),
        faults={
            fault["key"]: fault["count"]
            for routed in of_reply("AgentTurnRouted")
            for fault in routed["faults"]
        },
    )


def stage_request_number(
    events: Iterable[Mapping[str, Any]], attempt_id: str, stage: str
) -> int:
    """The pending request's number among the Attempt's requests in ``stage``."""
    return 1 + sum(
        event["event_type"] == "ModelRequestRecorded"
        and event["payload"]["attempt_id"] == attempt_id
        and event["payload"]["stage"] == stage
        for event in events
    )


def tagged(name: str, text: str) -> str:
    """An injected message as the model reads it: its pattern's tag, then the text."""
    return _TAG.format(name=name) + text


def injected_pattern(content: Any) -> str | None:
    """The pattern that wrote an injected message, or None for any other content."""
    if not isinstance(content, str) or not content.startswith("[injection:"):
        return None
    name, separator, _ = content.removeprefix("[injection:").partition("]\n")
    return name if separator else None


def injection_messages(
    policy: InjectionPolicy | None, signals: InjectionSignals
) -> tuple[str, ...]:
    """The tagged messages the pending request receives, in pattern-name order."""
    if policy is None:
        return ()
    messages = []
    for name, tuning in policy.for_stage(signals.stage).items():
        pattern = PATTERNS[name]
        text = pattern.message(signals, pattern.tuning.model_validate(tuning))
        if text is not None:
            messages.append(tagged(name, text))
    return tuple(messages)


__all__ = [
    "PATTERNS",
    "InjectionPattern",
    "InjectionPolicy",
    "InjectionSignals",
    "LatestReply",
    "ReplyCall",
    "ReplyRefusal",
    "ToolResult",
    "InjectionTunings",
    "PatternName",
    "PatternTuning",
    "injected_pattern",
    "injection_messages",
    "latest_reply",
    "resolve_injection",
    "stage_request_number",
    "tagged",
]
