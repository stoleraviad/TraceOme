"""Concrete Nebius model endpoint Adapter over pinned compatible libraries."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Mapping

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, Timeout
from openai._models import FinalRequestOptions
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters, get_user_agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.model_settings import (
    CALL_REASONING_ARGUMENT,
    REPLY_READING,
    CallReasoningPlacement,
    ModelContextBudget,
    ModelEndpointPolicy,
    ReplyQuirk,
    forces_a_call,
)
from bio_neb.agentic_journey.model_transport import (
    PreparedModelTransportRequest,
    ReceivedModelTransportResponse,
    model_transport_request_prepared,
    model_transport_response_received,
)
from bio_neb.agentic_journey.model_turn import (
    ModelArguments,
    read_arguments,
    read_turn,
    unfenced,
)
from bio_neb.agentic_journey.runtime_config import (
    LoadedRuntimeConfig,
    NebiusModelEndpointIdentity,
    NebiusModelEndpointSelection,
    RuntimeConfigError,
    untranslatable_setting_paths,
)
from bio_neb.failure import Invariant, ModelContextExhausted, ModelResponseMalformed

# How long we are willing to hold one connection open, not a cost lever. The client
# default is ten minutes, which is a number nothing in this product chose; a role that
# needs longer than that to answer should be answered, not cut off. Connect stays short
# so an unreachable endpoint still fails fast. This does not make a long completion
# reachable on its own: a historical endpoint request received a gateway 504 near
# ten minutes. That observation is not a current deadline guarantee.
_MODEL_CONTACT_TIMEOUT = Timeout(timeout=1800.0, connect=5.0)


# Budget inputs are the selected model's configured assumptions. The complete
# serialized UTF-8 body's estimate (ModelContextBudget.input_estimate) stands for
# its input tokens, plus template room.
class _BudgetedNebiusClient(AsyncOpenAI):
    """Apply the input budget after the pinned SDK has serialized the whole request."""

    def __init__(self, *, context_budget: ModelContextBudget, **kwargs):
        super().__init__(**kwargs)
        self._context_budget = context_budget

    def _build_request(
        self, options: FinalRequestOptions, *, retries_taken: int = 0
    ) -> httpx.Request:
        request = super()._build_request(options, retries_taken=retries_taken)
        if options.url != "/chat/completions" or options.method.lower() != "post":
            return request
        unbudgeted = request.content
        body = json.loads(request.content)
        if "max_completion_tokens" not in body:
            model_transport_request_prepared(
                PreparedModelTransportRequest(
                    unbudgeted, request.content, self._context_budget
                )
            )
            return request

        requested_maximum = body["max_completion_tokens"]
        allowance = self._context_budget.completion_allowance(
            requested_maximum, len(request.content)
        )
        if allowance <= 0:
            # Classified at this edge (ADR 0031): nothing has been sent.
            raise ModelContextExhausted(
                input_estimate=self._context_budget.input_estimate(
                    len(request.content)
                ),
                safety_margin=self._context_budget.serialized_body_safety_margin,
                context_window=self._context_budget.context_window,
            )
        if allowance == requested_maximum:
            model_transport_request_prepared(
                PreparedModelTransportRequest(
                    unbudgeted, request.content, self._context_budget
                )
            )
            return request

        # The first SDK build includes messages, reasoning, repair history, Tool
        # definitions and extra-body merging. Copy its JSON and change only the wire
        # allowance; caller/effective settings retain their requested maximum.
        body["max_completion_tokens"] = allowance
        adjusted = options.model_copy(update={"json_data": body, "extra_json": None})
        request = super()._build_request(adjusted, retries_taken=retries_taken)
        model_transport_request_prepared(
            PreparedModelTransportRequest(
                unbudgeted, request.content, self._context_budget
            )
        )
        return request

    async def _send_request(
        self, request: httpx.Request, *, stream: bool, **kwargs
    ) -> httpx.Response:
        """Keep the provider's raw body before the SDK reads anything (#421).

        Any HTTP status: an error body is evidence of its layer too.
        """
        response = await super()._send_request(request, stream=stream, **kwargs)
        if request.url.path.endswith("/chat/completions") and request.method == "POST":
            body = await response.aread()
            media_type = response.headers.get("content-type", "")
            model_transport_response_received(
                ReceivedModelTransportResponse(
                    response.status_code,
                    media_type.split(";")[0].strip().lower()
                    or "application/octet-stream",
                    body,
                )
            )
        return response


def _wire_tool(definition: ToolDefinition) -> dict[str, Any]:
    function = {
        "name": definition.name,
        "description": definition.description or "",
        "parameters": definition.parameters_json_schema,
    }
    if definition.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _unshapeable(detail: str) -> Invariant:
    # Configuration read and adapter construction refuse every configured value
    # shaping cannot carry, so one reaching a request is our bug.
    return Invariant("model_request_invalid", detail)


def _wire_tool_choice(choice: Any, parameters: ModelRequestParameters) -> Any:
    """Render the resolved choice; an unresolved one carries no key.

    Nothing forces a call by default, whatever the output shape (owner, #476
    alignment G1.2 B): a role that must end with an output Tool is told so by
    its correction, not by a provider grammar.
    """
    if choice is None:
        return None
    if not isinstance(choice, str):
        raise _unshapeable(f"tool_choice {choice!r} is not a mode or Tool name")
    if choice in ("required", "auto"):
        return choice
    if choice in {
        item.name for item in (*parameters.function_tools, *parameters.output_tools)
    }:
        return {"type": "function", "function": {"name": choice}}
    raise _unshapeable(f"tool_choice {choice!r} is not an offered Tool or mode")


def _grammar(request: Mapping[str, Any]) -> list[str]:
    """What in a shaped request puts it under the provider's grammar."""
    found = [
        f"strict Tool {item['function']['name']}"
        for item in request.get("tools", ())
        if item["function"].get("strict")
    ]
    if forces_a_call(request.get("tool_choice")):
        found.append(f"forced tool_choice {request['tool_choice']!r}")
    if "response_format" in request:
        found.append("response_format")
    return found


def shape_request(
    policy: ModelEndpointPolicy,
    settings: Mapping[str, Any],
    parameters: ModelRequestParameters,
) -> dict[str, Any]:
    """The request's compatible client parameters, less its messages and model.

    A pure step over typed resolved settings, the stage's Tools and the resolved
    tool choice. Each setting goes where the model's translation table puts it;
    an explicit None omits it. What it cannot shape is refused as an Invariant.
    """
    values = dict(settings)
    choice = values.pop("tool_choice", None)
    try:
        typed = policy.settings_selection.validate_settings(
            {name: value for name, value in values.items() if value is not None}
        )
    except ValueError as error:
        raise _unshapeable(str(error)) from error
    if untranslatable := policy.untranslatable_settings(typed):
        raise _unshapeable("untranslatable settings: " + ", ".join(untranslatable))
    tools = [
        _wire_tool(item)
        for item in (*parameters.function_tools, *parameters.output_tools)
    ]
    request: dict[str, Any] = {}
    for name, value in typed.request_settings().items():
        if name == "parallel_tool_calls" and not tools:
            continue  # The wire takes it only beside Tools.
        for (*parents, field), wire_value in policy.wire_fields(name, value).items():
            target = request
            for parent in parents:
                target = target.setdefault(parent, {})
            target[field] = wire_value
    if not tools:
        if choice is not None:
            raise _unshapeable("tool_choice without an offered Tool")
        return request
    request["tools"] = tools
    if (rendered := _wire_tool_choice(choice, parameters)) is not None:
        request["tool_choice"] = rendered
    if policy.grammar_evidence is None and (grammar := _grammar(request)):
        # A grammar fixes argument order and refuses schema keywords the model
        # was shown; only a model with recorded support may be sent one.
        raise Invariant(
            "model_request_grammar",
            f"{policy.model} declares no grammar support: " + ", ".join(grammar),
        )
    return request


CALL_REASONING_PART = "call_reasoning"
"""The id of the thinking part holding one call's reasoning, just before that call."""

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
# A reply that opens like a text-form Tool call, even one cut off at the cap.
_TEXT_CALL_START = re.compile(
    r'\s*(?:```(?:json)?\s*)?(?:<tool_call>|<function=|\[?\s*\{\s*"name"\s*:)'
)
_TOOL_CALL_TAGS = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", re.DOTALL)
_FUNCTION_TAGS = re.compile(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", re.DOTALL)
_PARAMETER_TAGS = re.compile(r"\s*<parameter=([^>\s]+)>(.*?)</parameter>\s*", re.DOTALL)


def _whole(pattern: re.Pattern, text: str) -> list[re.Match] | None:
    """Every match of a pattern that alone makes up the whole text, or None."""
    matches, position = [], 0
    while position < len(text):
        match = pattern.match(text, position)
        if match is None or match.end() == position:
            return None
        matches.append(match)
        position = match.end()
    return matches or None


def _json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _json_calls(text: str) -> list[tuple[str, str]] | None:
    value = _json_value(text)
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        arguments = item.get("arguments", item.get("parameters", {}))
        # A string is kept as written, so admission refuses malformed or
        # non-object arguments as it refuses them in a real call.
        calls.append(
            (
                item["name"],
                arguments if isinstance(arguments, str) else _arguments_text(arguments),
            )
        )
    return calls or None


def _text_form_calls(text: str) -> list[tuple[str, str]] | None:
    """The Tool calls, with their argument text, a whole reply text writes, or None.

    Forms read: a JSON call or array of calls with ``name`` and ``arguments`` or
    ``parameters`` (seen on Nebius, #279 S4 and #282 s5), optionally fenced or
    in ``<tool_call>`` tags, and ``<function=name><parameter=key>`` tags.
    """
    text = text.strip()
    if (body := unfenced(text)) is not None:
        text = body
    if blocks := _whole(_TOOL_CALL_TAGS, text):
        calls = []
        for block in blocks:
            if (inner := _text_form_calls(block.group(1))) is None:
                return None
            calls.extend(inner)
        return calls
    if functions := _whole(_FUNCTION_TAGS, text):
        calls = []
        for function in functions:
            body = function.group(2).strip()
            parameters = _whole(_PARAMETER_TAGS, body) if body else []
            if parameters is None:
                return None
            calls.append(
                (
                    function.group(1),
                    _arguments_text(
                        {
                            item.group(1): _json_value(item.group(2).strip())
                            for item in parameters
                        }
                    ),
                )
            )
        return calls
    return _json_calls(text)


def _without_think_tags(text: str) -> tuple[list[str], str]:
    """Reasoning written into content between think tags, and the text left over.

    A closing tag with no opening one ends reasoning written straight into
    content; an opening tag with no closing one starts reasoning cut off there.
    Text that is a whole text-form call is data: a tag inside its arguments
    stays as written.
    """
    thinking: list[str] = []
    rest: list[str] = []
    while True:
        if _text_form_calls(text) is not None:
            rest.append(text)
            break
        opening, closing = text.find(_THINK_OPEN), text.find(_THINK_CLOSE)
        if opening == -1 and closing == -1:
            rest.append(text)
            break
        if opening != -1 and (closing == -1 or opening < closing):
            rest.append(text[:opening])
            text = text[opening + len(_THINK_OPEN) :]
            closing = text.find(_THINK_CLOSE)
            if closing == -1:
                thinking.append(text)
                break
        thinking.append(text[:closing])
        text = text[closing + len(_THINK_CLOSE) :]
    return [item.strip() for item in thinking if item.strip()], "".join(rest)


# The Tools the request being answered showed, by name; its reply is read against them.
_SHOWN_TOOLS: ContextVar[Mapping[str, ToolDefinition] | None] = ContextVar(
    "shown_tools", default=None
)
_NAME_SPELLING = re.compile(r"[A-Za-z0-9_-]+")
_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def _name_words(name: str) -> tuple[str, ...]:
    """A name's words, whatever its case, `_`/`-` separators or camelCase.

    `eventIds` is `event_ids`. A name with any other character has no words, so
    it matches nothing: `event.ids` and `élimit` are not spellings of a name.
    """
    if not _NAME_SPELLING.fullmatch(name):
        return ()
    return tuple(word.lower() for word in _NAME_WORD.findall(name))


def _shown_names(
    arguments: Mapping[str, Any], definition: ToolDefinition
) -> dict[str, str]:
    """Each sent argument name to fix, mapped to the shown name it spells (#242).

    A name the Tool did not show is fixed only when exactly one shown name the
    call did not send has the same words, and no other sent name claims it.
    Anything else is left for admission to refuse. Top-level names only.
    """
    shown = definition.parameters_json_schema.get("properties")
    if not isinstance(shown, Mapping):
        return {}
    unsent: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for name in shown:
        if name not in arguments and (words := _name_words(name)):
            unsent[words].append(name)
    claims: dict[str, list[str]] = defaultdict(list)
    for sent in arguments:
        if sent not in shown and len(matches := unsent.get(_name_words(sent), ())) == 1:
            claims[matches[0]].append(sent)
    return {sent[0]: name for name, sent in claims.items() if len(sent) == 1}


def _with_shown_argument_names(parts) -> tuple[list[Any], bool]:
    """Each call with the argument names its Tool showed; whether any was fixed."""
    shown_tools = _SHOWN_TOOLS.get() or {}
    read: list[Any] = []
    fixed = False
    for part in parts:
        definition = (
            shown_tools.get(part.tool_name) if isinstance(part, ToolCallPart) else None
        )
        reading = _arguments(part) if definition is not None else None
        names = _shown_names(reading.value, definition) if reading is not None else {}
        if not names:
            read.append(part)
            continue
        arguments = {names.get(key, key): value for key, value in reading.value.items()}
        # The call keeps its fence, so admission records it.
        read.append(
            replace(part, args=_arguments_text(arguments, fenced=reading.fenced))
        )
        fixed = True
    return read, fixed


def _arguments(call: ToolCallPart) -> ModelArguments | None:
    read = read_arguments(call.args)
    # Admission refuses malformed arguments as it always has.
    return None if read.malformed else read


def _arguments_text(arguments: Mapping[str, Any], *, fenced: bool = False) -> str:
    """Arguments as the call carries them; fenced arguments stay fenced (#222)."""
    # Replay must send JSON strings: Nebius refuses object arguments (422, S3).
    text = json.dumps(arguments, ensure_ascii=False)
    return f"```json\n{text}\n```" if fenced else text


def _replayed_call(
    call: ToolCallPart,
    reasoning: str | None,
    placement: CallReasoningPlacement,
) -> ToolCallPart:
    """A call as replayed: its arguments as the Model turn read them (#222).

    A fenced call is sent as the JSON inside its fence, never as the framework's
    invalid-JSON wrapper; a call whose reasoning was moved out gets it back where
    the model was asked to write it. Malformed arguments replay as before.
    """
    read = _arguments(call)
    if read is None:
        return call
    if reasoning is None or placement == "none":
        return replace(call, args=read.text) if read.fenced else call
    added = {CALL_REASONING_ARGUMENT: reasoning}
    ordered = (
        {**added, **read.value} if placement == "first" else {**read.value, **added}
    )
    return replace(call, args=_arguments_text(ordered))


class NebiusModelEndpointAdapter(OpenAIChatModel):
    """Own the compatible client, request policy, and Tool adaptation for Nebius."""

    records_model_transport_requests = True
    records_model_transport_responses = True

    def __init__(
        self, loaded: LoadedRuntimeConfig, selected: NebiusModelEndpointSelection
    ):
        if (
            selected != loaded.selections.model_for_policy(loaded.invocation_policy)
            or selected.settings_selection
            != loaded.invocation_policy.settings_selection
        ):
            raise RuntimeConfigError("runtime_config_changed", ("model_reference",))
        self.nebius_model_endpoint = NebiusModelEndpointIdentity(
            provider="nebius",
            model=selected.model,
            contacted_url=selected.endpoint,
        )
        self.settings_selection = selected.settings_selection
        # The Attempt's bound policy; the adapter never looks a model up.
        self.endpoint_policy = loaded.model
        if untranslatable := untranslatable_setting_paths(
            self.endpoint_policy, loaded.invocation_policy
        ):
            raise RuntimeConfigError("runtime_config_invalid", untranslatable)
        # OpenAI names below are compatible wire grammar, not product/provider ownership.
        client = _BudgetedNebiusClient(
            context_budget=self.settings_selection.context_budget,
            base_url=selected.endpoint,
            api_key=loaded.credential,
            max_retries=0,
            timeout=_MODEL_CONTACT_TIMEOUT,
        )
        super().__init__(
            selected.model,
            provider=OpenAIProvider(openai_client=client),
            settings=loaded.invocation_policy.model_settings.request_settings(),
        )

    async def request(
        self,
        messages: list[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        shown = _SHOWN_TOOLS.set(
            {
                item.name: item
                for item in (
                    *model_request_parameters.function_tools,
                    *model_request_parameters.output_tools,
                )
            }
        )
        try:
            return await super().request(
                messages, model_settings, model_request_parameters
            )
        finally:
            _SHOWN_TOOLS.reset(shown)

    def _process_response(self, response):
        # The compatible SDK deliberately does not validate successful bodies.
        # Classify only provider payload shape failures at this adapter boundary.
        if not isinstance(response, ChatCompletion):
            raise ModelResponseMalformed("model response must be a chat completion")
        # Validate before framework normalization reads choices/message fields.
        processed = super()._process_response(self._validate_completion(response))
        return self._read_reply(processed, response.choices[0])

    def _read_reply(self, processed: ModelResponse, choice) -> ModelResponse:
        """The reply as this model's adapter reads its wire quirks (#241).

        ``finish_reason`` is read before content: a cut-off reply keeps its text,
        never read for calls, even when it opens like a text-form call; the loop
        routes its parsed calls (ADR 0044). Each quirk is recorded, with the raw
        wire fields the reading changed.
        """
        policy = self.endpoint_policy
        quirks: list[ReplyQuirk] = []
        raw: dict[str, Any] = {}
        parts = list(processed.parts)
        content = choice.message.content or ""
        if choice.message.refusal:
            # The framework keeps a refusal as a reply with no parts; its text is
            # never read for calls or reasoning.
            return processed
        if policy.reads_reply_quirks:
            if _THINK_OPEN in content or _THINK_CLOSE in content:
                # Read again from the wire: the framework splits tags anywhere,
                # inside a text-form call's arguments too.
                thinking, unread = _without_think_tags(content)
                parts = [
                    *(
                        part
                        for part in parts
                        if isinstance(part, ThinkingPart) and part.id != "content"
                    ),
                    *(
                        ThinkingPart(item, id="content", provider_name=self.system)
                        for item in thinking
                    ),
                    *([TextPart(unread)] if unread.strip() else []),
                    *(part for part in parts if isinstance(part, ToolCallPart)),
                ]
                if unread != content:
                    quirks.append("think_tags")
                    raw["content"] = choice.message.content
                content = unread
            if choice.finish_reason == "length":
                if _TEXT_CALL_START.match(content):
                    quirks.append("text_form_tool_call_cut_off")
            elif (
                choice.finish_reason in ("stop", "tool_calls")
                and not any(isinstance(part, ToolCallPart) for part in parts)
                and (calls := _text_form_calls(content))
            ):
                # Only a completed reply's text becomes calls; any other ending
                # (a content filter) keeps it as text.
                parts = [
                    *(part for part in parts if not isinstance(part, TextPart)),
                    *(ToolCallPart(name, arguments) for name, arguments in calls),
                ]
                quirks.append("text_form_tool_call")
                raw["content"] = choice.message.content
        fixed = split = False
        if policy.fixes_argument_names:
            # Before the reasoning split, so a misspelled reasoning name is split too.
            parts, fixed = _with_shown_argument_names(parts)
            quirks.extend(["argument_name"] if fixed else [])
        if policy.call_reasoning != "none":
            parts, missing, split = self._split_call_reasoning(parts)
            quirks.extend(["call_reasoning_missing"] if missing else [])
        if (fixed or split) and choice.message.tool_calls:
            raw["tool_calls"] = [
                call.model_dump(mode="json", exclude_unset=True)
                for call in choice.message.tool_calls
            ]
        if unusable := read_turn(replace(processed, parts=parts)).unusable:
            # Marked as it arrived, for every model, with the Model turn's
            # reading.
            quirks.append(unusable)
        if not quirks and not raw:
            return processed
        return replace(
            processed,
            parts=parts,
            provider_details={
                **(processed.provider_details or {}),
                REPLY_READING: {"quirks": quirks, "raw": raw},
            },
        )

    def _split_call_reasoning(self, parts):
        """Move each call's reasoning argument to its own part, just before it.

        Domain Tools and admission never see the argument; the model's reasoning
        for the call stays recorded as a thinking part (owner, #241 readiness).
        """
        read: list[Any] = []
        missing = split = False
        for part in parts:
            reading = _arguments(part) if isinstance(part, ToolCallPart) else None
            if reading is None:
                read.append(part)
                continue
            arguments = dict(reading.value)
            if CALL_REASONING_ARGUMENT not in arguments:
                missing = True
                read.append(part)
                continue
            reasoning = arguments.pop(CALL_REASONING_ARGUMENT)
            read.append(
                ThinkingPart(
                    reasoning
                    if isinstance(reasoning, str)
                    else json.dumps(reasoning, ensure_ascii=False),
                    id=CALL_REASONING_PART,
                    provider_name=self.system,
                    provider_details={
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                    },
                )
            )
            # The call keeps its fence, so admission records it.
            read.append(
                replace(part, args=_arguments_text(arguments, fenced=reading.fenced))
            )
            split = True
        return read, missing, split

    def _map_model_response(self, message: ModelResponse):
        """Replay a reply as the model wrote it: each call as read, with its reasoning.

        Reasoning read out of content tags replays in the reasoning field, never
        as tags in content. A kept reply with nothing to send already carries the
        transcript composer's empty text (#444).
        """
        policy = self.endpoint_policy
        reasoning = {
            part.provider_details["tool_call_id"]: part.content
            for part in message.parts
            if isinstance(part, ThinkingPart) and part.id == CALL_REASONING_PART
        }
        parts = []
        for part in message.parts:
            if isinstance(part, ThinkingPart) and part.id == CALL_REASONING_PART:
                continue
            if (
                isinstance(part, ThinkingPart)
                and part.id == "content"
                and policy.reads_reply_quirks
            ):
                part = replace(part, id="reasoning_content")
            if isinstance(part, ToolCallPart):
                part = _replayed_call(
                    part, reasoning.get(part.tool_call_id), policy.call_reasoning
                )
            parts.append(part)
        return super()._map_model_response(replace(message, parts=parts))

    def _validate_completion(self, response):
        try:
            validated = super()._validate_completion(response)
        except ValidationError as error:
            raise ModelResponseMalformed(
                "model response failed its contract"
            ) from error
        if not validated.choices:
            raise ModelResponseMalformed("model response has no completion choice")
        return validated

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # The framework provider treats our supplied SDK client as externally
        # owned. Its context exit alone does not release this adapter's client.
        try:
            await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            await self.client.close()

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings, parameters = super().prepare_request(
            model_settings, model_request_parameters
        )
        # Shaped here too, so an unshapeable request is refused before it is
        # recorded or sent. Explicit None survives repeated preparation, so merging
        # model defaults cannot restore a parameter the caller omitted.
        shape_request(self.endpoint_policy, settings or {}, parameters)
        return settings, parameters

    async def _completions_create(
        self, messages, stream, model_settings, model_request_parameters
    ):
        """Transport only: send the shaped request with the mapped messages."""
        if stream:
            raise _unshapeable("a streamed model request has no shaped form")
        request = shape_request(
            self.endpoint_policy, model_settings, model_request_parameters
        )
        mapped = await self._map_messages(
            messages, model_request_parameters, model_settings=model_settings
        )
        try:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=mapped,
                stream=False,
                extra_headers={"User-Agent": get_user_agent()},
                **request,
            )
        except APIStatusError as error:
            # The framework's error contract, which failure classification reads.
            if error.status_code >= 400:
                raise ModelHTTPError(
                    status_code=error.status_code,
                    model_name=self.model_name,
                    body=error.body,
                    headers=dict(error.response.headers),
                ) from error
            raise ModelAPIError(
                model_name=self.model_name, message=error.message
            ) from error
        except APIConnectionError as error:
            raise ModelAPIError(
                model_name=self.model_name, message=error.message
            ) from error


__all__ = [
    "NebiusModelEndpointAdapter",
    "NebiusModelEndpointIdentity",
    "shape_request",
]
