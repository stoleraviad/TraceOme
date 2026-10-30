"""The Model turn: the one reader of model output.

Every path that reads what a model wrote reads it here, the same way: a reply's
calls and their arguments, function and output calls alike, a reply cut off on
its output ceiling, and a retained transcript. Fenced JSON is read the same
everywhere: the fence is stripped, the arguments are accepted, and the caller
records that they were fenced. A top-level argument the Tool declares as an
object, sent as JSON text holding one object, is read as that object, and the
caller records its name (#422).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import pydantic_core
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

# One Markdown code fence that is the whole text: an optional info word, then
# the body. A fence inside other text is not stripped.
_FENCE = re.compile(r"\A\s*```[A-Za-z0-9_+.-]*\s*(.*?)\s*```\s*\Z", re.DOTALL)


def unfenced(text: str) -> str | None:
    """The body of the one Markdown fence that is the whole text, or None."""
    match = _FENCE.match(text)
    return match.group(1) if match else None


@dataclass(frozen=True)
class ModelArguments:
    """One call's arguments as read.

    ``value`` is the JSON object the model wrote, or ``{"invalid_json": <raw>}``
    when it wrote no JSON object. ``fenced`` says the object was written inside a
    Markdown fence, which the reading stripped. ``text`` is the JSON text read,
    without its fence, or the object the model sent.
    """

    value: dict[str, Any]
    text: str | dict[str, Any] | None
    malformed: bool = False
    fenced: bool = False
    # The top-level arguments read as the object their JSON text held, in the
    # order sent; ``value`` and ``text`` hold the objects.
    decoded: tuple[str, ...] = ()


def read_arguments(raw: Any, objects: Collection[str] = ()) -> ModelArguments:
    """Read one call's raw arguments: an object, JSON text, or fenced JSON text.

    ``objects`` names the top-level arguments the Tool declares as an object
    (``object_arguments``), or, on reopen, those the record says were read from
    JSON text. Such an argument sent as JSON text holding one object is read as
    that object; any other string, invalid JSON text included, stays as sent.
    """
    if raw is None or isinstance(raw, dict):
        return _decoded(ModelArguments(dict(raw or {}), raw), objects)
    if isinstance(raw, str):
        body = unfenced(raw)
        text = raw if body is None else body
        value = _json_object(text)
        if value is not None:
            return _decoded(
                ModelArguments(value, text, fenced=body is not None), objects
            )
        return ModelArguments({"invalid_json": raw}, text, malformed=True)
    return ModelArguments({"invalid_json": raw}, None, malformed=True)


def _decoded(arguments: ModelArguments, objects: Collection[str]) -> ModelArguments:
    value = dict(arguments.value)
    decoded = []
    for name, item in arguments.value.items():
        if name in objects and isinstance(item, str):
            read = _json_object(item) if item else None
            if read is not None:
                value[name] = read
                decoded.append(name)
    if not decoded:
        return arguments
    text = (
        value
        if isinstance(arguments.text, dict)
        else pydantic_core.to_json(value).decode()
    )
    return replace(arguments, value=value, text=text, decoded=tuple(decoded))


def object_arguments(schema: Mapping[str, Any]) -> frozenset[str]:
    """The top-level arguments a Tool's JSON schema declares as an object.

    An object, a ``$ref`` to one, or either one or null (``anyOf[object, null]``).
    """
    definitions = schema.get("$defs") or {}

    def is_object(item: Any) -> bool:
        if not isinstance(item, Mapping):
            return False
        reference = item.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            return is_object(definitions.get(reference.removeprefix("#/$defs/")))
        kind = item.get("type")
        if kind == "object" or (
            isinstance(kind, list) and set(kind) in ({"object"}, {"object", "null"})
        ):
            return True
        members = [
            member
            for member in item.get("anyOf") or item.get("oneOf") or ()
            if not (isinstance(member, Mapping) and member.get("type") == "null")
        ]
        return len(members) == 1 and is_object(members[0])

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return frozenset()
    return frozenset(name for name, item in properties.items() if is_object(item))


def output_schema_errors(
    output_type: type[BaseModel], arguments: ModelArguments
) -> list[ErrorDetails] | None:
    """The output type's errors for an output call's arguments as read, or none.

    JSON text is validated as JSON, an object as Python, as the framework
    validates an output Tool's arguments.
    """
    try:
        if isinstance(arguments.text, str):
            output_type.model_validate_json(arguments.text or "{}")
        else:
            output_type.model_validate(arguments.text or {})
    except ValidationError as error:
        return error.errors(include_url=False, include_context=False)
    return None


def _json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return {}
    try:
        value = pydantic_core.from_json(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class ModelCall:
    """One call a reply made, in the order sent."""

    tool_name: str
    tool_call_id: str
    arguments: ModelArguments


UnusableReply = Literal["reasoning_only", "empty_reply", "call_stub"]
"""A reply with no call that carries nothing to act on (#421, #444).

``reasoning_only``: reasoning and nothing else. ``empty_reply``: no call, no text
and no reasoning. ``call_stub``: the provider ended the reply on a tool call, yet
no call arrived, only a text fragment (#387 celegans-a1's ``"The"``).
"""


@dataclass(frozen=True)
class ModelTurn:
    """One model reply as read: its calls and how it ended.

    A reply ``cut_off`` on its output ceiling keeps every call it held, the
    partial last one included, as read; the loop decides what a cut-off means.
    An ``unusable`` reply held no call and nothing to act on, whatever finish
    reason the provider gave (#395's six said ``tool_calls``); a reply that ended
    in text of its own accord is a text ending, not unusable.
    """

    calls: tuple[ModelCall, ...]
    cut_off: bool
    output_tokens: int | None
    unusable: UnusableReply | None = None

    @property
    def partial(self) -> ModelCall | None:
        """The call generation stopped inside: a cut-off reply's last call.

        Nothing in the reply shows whether that call ended. The compatible wire
        sends the calls as a list after any text, and the provider closes a cut
        call's arguments: #326's two cut-off replies ended in ``{}`` and in a
        search without the limit each of its copies carried. So the last call
        of a cut-off reply is never run; its note asks for it again (ADR 0044).
        """
        return self.calls[-1] if self.cut_off and self.calls else None

    @property
    def complete_calls(self) -> tuple[ModelCall, ...]:
        """The calls the model finished writing, in the order sent."""
        return self.calls[:-1] if self.partial is not None else self.calls


def messages_record(messages: Any) -> list[dict[str, Any]]:
    """A transcript of framework messages as the Journey retains it (JSON)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")


def read_messages_record(record: Any) -> list[Any]:
    """A retained transcript read back into framework messages.

    A record the framework's message schema cannot read is refused (ValueError).
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.validate_python(record)


def retry_part(content: str) -> dict[str, Any]:
    """A retry part the loop writes (no Tool), as the Journey retains it."""
    from pydantic_ai.messages import RetryPromptPart

    return _request_part(RetryPromptPart(content=content))


def prompt_part(content: str) -> dict[str, Any]:
    """A user prompt part, as the Journey retains it."""
    from pydantic_ai.messages import UserPromptPart

    return _request_part(UserPromptPart(content))


def empty_reply(timestamp: Any) -> dict[str, Any]:
    """A reply that sent nothing, as empty text, dated ``timestamp`` (#444)."""
    from pydantic_ai.messages import ModelResponse, TextPart

    (record,) = messages_record([ModelResponse(parts=[TextPart("")])])
    return {**record, "timestamp": timestamp}


def _request_part(part: Any) -> dict[str, Any]:
    from pydantic_ai.messages import ModelRequest

    (record,) = messages_record([ModelRequest(parts=[part])])
    return record["parts"][0]


def as_received(correction: str, tool_name: str | None) -> str:
    """A correction as the model's next request shows it, framed by the framework.

    The retry rendering writes the correction; the framework's retry channel
    frames it for the call it answers.
    """
    from pydantic_ai.messages import RetryPromptPart

    return RetryPromptPart(content=correction, tool_name=tool_name).model_response()


def read_turn(response: Any, tools: Iterable[Any] = ()) -> ModelTurn:
    """Read one framework reply against the Tool definitions its request showed.

    Each definition's object arguments are read as objects when sent as JSON
    text (``object_arguments``); a call of a Tool not shown is read as sent.
    """
    from pydantic_ai.messages import ToolCallPart

    objects = {
        tool.name: object_arguments(tool.parameters_json_schema) for tool in tools
    }
    usage = getattr(response, "usage", None)
    calls = tuple(
        ModelCall(
            part.tool_name,
            part.tool_call_id,
            read_arguments(part.args, objects.get(part.tool_name, ())),
        )
        for part in response.parts
        if isinstance(part, ToolCallPart)
    )
    finish = getattr(response, "finish_reason", None)
    cut_off = finish == "length"
    return ModelTurn(
        calls=calls,
        cut_off=cut_off,
        output_tokens=getattr(usage, "output_tokens", None),
        unusable=None if calls or cut_off else _unusable(response.parts, finish),
    )


def _unusable(parts: Sequence[Any], finish: str | None) -> UnusableReply | None:
    """What a reply with no call and no cut-off held, when nothing to act on."""
    from pydantic_ai.messages import TextPart, ThinkingPart

    if any(isinstance(part, TextPart) and part.content.strip() for part in parts):
        return "call_stub" if finish == "tool_call" else None
    if any(isinstance(part, ThinkingPart) and part.content.strip() for part in parts):
        return "reasoning_only"
    return "empty_reply"


def read_turn_record(record: Mapping[str, Any]) -> ModelTurn:
    """Read one retained reply the way its live turn was read."""
    (response,) = read_messages_record([record])
    return read_turn(response)


__all__ = [
    "ModelArguments",
    "ModelCall",
    "ModelTurn",
    "UnusableReply",
    "as_received",
    "empty_reply",
    "prompt_part",
    "retry_part",
    "messages_record",
    "object_arguments",
    "output_schema_errors",
    "read_arguments",
    "read_messages_record",
    "read_turn",
    "read_turn_record",
    "unfenced",
]
