"""A scripted model endpoint at the HTTP layer; no network contact.

The production factory builds the real Nebius Adapter and only its HTTP transport
is replaced (`httpx.MockTransport`), so every scripted request crosses request
shaping, the context budget and the transport record (owner, #221). A script keeps
the framework's `FunctionModel` signature, `respond(messages, info)`: the sent body
is read back into messages and an `AgentInfo`, and the returned `ModelResponse` is
answered as a chat completion. A raised `ModelHTTPError` is answered as that HTTP
status, and an `httpx.TransportError` or `OSError` (`ConnectionError`, `TimeoutError`)
is a connection fault; the model request raises any other exception as the script
raised it.

The read-back is the wire's view, not the framework's: Tool results arrive as their
sent text (parsed when it is JSON), a correction as a `RetryPromptPart` holding its
sent text, a refused call's correction in its slot as a failed `ToolReturnPart`
holding its sent text (the wire's `{"error": ...}`), and `info.output_tools` holds the
offered Tools the assembly declares as outputs. `corrections` reads both kinds.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.adapters import create_nebius_model_endpoint
from bio_neb.agentic_journey.runtime_config import (
    RuntimeConfiguration,
    resolve_runtime_config,
)
from bio_neb.investigation.investigation_agent import INVESTIGATION_OUTPUT
from bio_neb.investigation.vcf_table_reading import TABLE_READING_OUTPUT
from bio_neb.reporting.explanation_agent import COMBINATION_OUTPUT, MEMBER_OUTPUT

OUTPUT_TOOL_NAMES = frozenset(
    {
        INVESTIGATION_OUTPUT.name,
        TABLE_READING_OUTPUT.name,
        MEMBER_OUTPUT.name,
        COMBINATION_OUTPUT.name,
    }
)
_RETRY_SUFFIX = "\n\nFix the errors and try again."
_FEEDBACK_PREFIX = "Validation feedback:\n"
_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "tool_call": "tool_calls",
    "content_filter": "content_filter",
}

Respond = Callable[[list[ModelMessage], AgentInfo], Any]


def _tool_result(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _request_part(message: dict[str, Any], tool_names: dict[str, str]):
    content = message.get("content") or ""
    if message["role"] == "tool":
        call_id = message["tool_call_id"]
        if content.endswith(_RETRY_SUFFIX):
            return RetryPromptPart(
                content=content.removesuffix(_RETRY_SUFFIX),
                tool_name=tool_names.get(call_id),
                tool_call_id=call_id,
            )
        result = _tool_result(content)
        if isinstance(result, dict) and list(result) == ["error"]:
            # A failed Tool result: a refused call answered in its own slot.
            return ToolReturnPart(
                tool_name=tool_names.get(call_id, ""),
                content=result["error"],
                tool_call_id=call_id,
                outcome="failed",
            )
        return ToolReturnPart(
            tool_name=tool_names.get(call_id, ""),
            content=result,
            tool_call_id=call_id,
        )
    if content.endswith(_RETRY_SUFFIX):
        return RetryPromptPart(
            content=content.removesuffix(_RETRY_SUFFIX).removeprefix(_FEEDBACK_PREFIX)
        )
    return UserPromptPart(content=content)


def corrections(message: ModelMessage) -> list[str]:
    """The corrections a read-back request carries, in order: each retry part and
    each refused call's failed result in its slot (#373; every model, #445)."""
    return [
        part.content
        for part in message.parts
        if isinstance(part, RetryPromptPart)
        or (isinstance(part, ToolReturnPart) and part.outcome == "failed")
    ]


def wire_messages(body: dict[str, Any]) -> tuple[list[ModelMessage], str | None]:
    """The sent messages as framework messages, and the joined instructions."""
    sent = list(body["messages"])
    leading = []
    while sent and sent[0]["role"] in {"system", "developer"}:
        leading.append(sent.pop(0)["content"])
    instructions = "\n\n".join(leading) or None
    messages: list[ModelMessage] = []
    tool_names: dict[str, str] = {}
    parts: list[Any] = []
    for message in sent:
        if message["role"] != "assistant":
            parts.append(_request_part(message, tool_names))
            continue
        if parts:
            messages.append(ModelRequest(parts=parts, instructions=instructions))
            parts = []
        response: list[Any] = []
        if message.get("reasoning_content"):
            response.append(ThinkingPart(content=message["reasoning_content"]))
        if message.get("content"):
            response.append(TextPart(content=message["content"]))
        for call in message.get("tool_calls") or ():
            tool_names[call["id"]] = call["function"]["name"]
            response.append(
                ToolCallPart(
                    call["function"]["name"],
                    call["function"]["arguments"],
                    tool_call_id=call["id"],
                )
            )
        messages.append(ModelResponse(parts=response))
    if parts:
        messages.append(ModelRequest(parts=parts, instructions=instructions))
    return messages, instructions


def wire_info(body: dict[str, Any], instructions: str | None) -> AgentInfo:
    """The offered Tools and settings the body carries, as the framework names them."""
    function_tools, output_tools = [], []
    for item in body.get("tools") or ():
        function = item["function"]
        definition = ToolDefinition(
            name=function["name"],
            description=function.get("description") or None,
            parameters_json_schema=function["parameters"],
            strict=function.get("strict"),
            kind="output" if function["name"] in OUTPUT_TOOL_NAMES else "function",
        )
        (output_tools if definition.kind == "output" else function_tools).append(
            definition
        )
    settings = {
        key: value
        for key, value in body.items()
        if key not in {"messages", "model", "stream", "tools"}
    }
    return AgentInfo(
        function_tools=function_tools,
        allow_text_output=not output_tools,
        output_tools=output_tools,
        model_settings=settings,
        model_request_parameters=ModelRequestParameters(
            function_tools=function_tools,
            output_tools=output_tools,
            allow_text_output=not output_tools,
        ),
        instructions=instructions,
    )


def completion(response: ModelResponse, *, model: str) -> dict[str, Any]:
    """A chat completion carrying the scripted response."""
    text = "".join(
        part.content for part in response.parts if isinstance(part, TextPart)
    )
    reasoning = "".join(
        part.content for part in response.parts if isinstance(part, ThinkingPart)
    )
    calls = [
        {
            "id": part.tool_call_id,
            "type": "function",
            # Text arguments go out verbatim, fenced or malformed ones included.
            "function": {
                "name": part.tool_name,
                "arguments": part.args
                if isinstance(part.args, str)
                else json.dumps(part.args or {}),
            },
        }
        for part in response.parts
        if isinstance(part, ToolCallPart)
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    finish = _FINISH_REASONS.get(
        response.finish_reason or "", "tool_calls" if calls else "stop"
    )
    usage = response.usage
    return {
        "id": "scripted-completion",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


class ScriptFailure(BaseException):
    """A script's own error, carried past the client's connection-fault handling.

    The compatible client turns any exception inside its transport into a
    connection fault; the Adapter's request re-raises the script's error instead,
    so a scripted model failure, a script bug or an unexpected contact fails as
    the script raised it.
    """

    def __init__(self, error: Exception):
        super().__init__(error)
        self.error = error


class ScriptedModelEndpoint:
    """A `NebiusModelEndpointFactory` whose Adapters answer from a script.

    `adapter` builds the real Adapter (the production factory by default; a
    test may pass an Adapter subclass that injects a fault). `bodies` keeps every
    sent request body in order, across Adapters.
    """

    def __init__(self, respond: Respond, *, adapter=create_nebius_model_endpoint):
        self.respond = respond
        self.adapter = adapter
        self.bodies: list[dict[str, Any]] = []
        self.adapters: list[Any] = []

    def __call__(self, loaded, selected):
        adapter = self.adapter(loaded, selected)
        adapter.client._client = httpx.AsyncClient(
            transport=self.open_transport(), follow_redirects=False
        )
        send = adapter.request

        async def request(*args, **kwargs):
            try:
                return await send(*args, **kwargs)
            except ScriptFailure as failure:
                # The model request raises the script's own error, as it was raised.
                raise failure.error from None

        adapter.request = request
        self.adapters.append(adapter)
        return adapter

    def open_transport(self) -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        messages, instructions = wire_messages(body)
        try:
            response = self.respond(messages, wire_info(body, instructions))
            if inspect.isawaitable(response):
                response = await response
        except ModelHTTPError as error:
            return httpx.Response(
                error.status_code,
                headers=error.headers or {},
                json=error.body if error.body is not None else {"message": str(error)},
            )
        except (httpx.TransportError, OSError):
            raise
        except Exception as error:
            raise ScriptFailure(error) from error
        return httpx.Response(200, json=completion(response, model=body["model"]))


def scripted_adapter(respond: Respond, *, model: str = "Qwen/Qwen3.5-397B-A17B"):
    """One real Adapter for a seam that wraps a model directly, with no Attempt.

    Its runtime selection is the default one for `model`, with a dummy credential.
    """
    configuration = RuntimeConfiguration.model_validate(
        {
            "contract_version": "1.0",
            "heartbeat_interval_seconds": 30,
            "agent_invocation": {"model_reference": "qwen"},
            "model_roster": {
                "qwen": {
                    "provider": "nebius",
                    "endpoint": "https://model.example.test/v1",
                    "model": model,
                    "credential_env": "SCRIPTED_MODEL_KEY",
                }
            },
        }
    )
    loaded = resolve_runtime_config(
        configuration, {"SCRIPTED_MODEL_KEY": "offline"}, agent_role="planning_agent"
    )
    return ScriptedModelEndpoint(respond)(
        loaded, configuration.model_for_policy(loaded.invocation_policy)
    )
