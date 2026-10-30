import asyncio
import copy
import json
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    AgentModelSettings,
)
from bio_neb.agentic_journey.model_transport import observe_model_transport_requests
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.agentic_journey.runtime_config import (
    LoadedRuntimeConfig,
    RuntimeConfiguration,
)
from bio_neb.failure import Invariant, ModelContextExhausted, model_contact_failure
from bio_neb.providers.nebius_model_endpoint import (
    NebiusModelEndpointAdapter,
    shape_request,
)
from tests.model_lifetime_support import LoopTransport


def loaded_config(
    *, reference: str = "qwen", model_settings: dict | None = None
) -> LoadedRuntimeConfig:
    from bio_neb.agentic_journey.runtime_config import resolve_runtime_config

    public_config = RuntimeConfiguration.model_validate(
        {
            "contract_version": "1.0",
            "heartbeat_interval_seconds": 3,
            "model_roster": {
                "qwen": {
                    "provider": "nebius",
                    "endpoint": "https://API.Example.TEST",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "credential_env": "MODEL_KEY",
                },
                "nemotron": {
                    "provider": "nebius",
                    "endpoint": "https://API.Example.TEST",
                    "model": "nvidia/Nemotron-3.5-Super-VL",
                    "credential_env": "SECOND_KEY",
                },
            },
            "agent_invocation": {
                "model_settings": model_settings or {},
                "scopes": {"role": {"planning_agent": {"model_reference": reference}}},
            },
        }
    )
    return resolve_runtime_config(
        public_config,
        {"MODEL_KEY": "dummy-key", "SECOND_KEY": "second-key"},
        agent_role="planning_agent",
    )


def adapter_for_reference(reference: str = "qwen") -> NebiusModelEndpointAdapter:
    loaded = loaded_config(reference=reference)
    return NebiusModelEndpointAdapter(
        loaded, loaded.selections.model_for_policy(loaded.invocation_policy)
    )


def tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        parameters_json_schema={"type": "object"},
    )


def parameters(
    *, functions: list[ToolDefinition], outputs: list[ToolDefinition] | None = None
) -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=functions,
        output_tools=outputs or [],
        allow_text_output=False,
        output_mode="tool",
    )


class NebiusModelEndpointAdapterTests(unittest.TestCase):
    def test_explicit_omission_survives_repeated_preparation(self):
        adapter = adapter_for_reference()
        request_parameters = parameters(functions=[])
        inherited, _ = adapter.prepare_request({}, request_parameters)
        self.assertEqual(inherited["max_tokens"], 245760)
        omitted, adjusted = adapter.prepare_request(
            {"max_tokens": None}, request_parameters
        )
        again, _ = adapter.prepare_request(omitted, adjusted)
        self.assertIsNone(omitted["max_tokens"])
        self.assertEqual(again, omitted)

    def test_construction_has_canonical_identity_retry_and_contact_timeout(self):
        adapter = adapter_for_reference()

        self.assertEqual(str(adapter.client.base_url), "https://api.example.test/v1/")
        self.assertEqual(adapter.client.max_retries, 0)
        # Half an hour to answer, five seconds to be reachable. The client's own
        # ten-minute default is a ceiling nothing in this product chose.
        self.assertEqual(adapter.client.timeout.read, 1800.0)
        self.assertEqual(adapter.client.timeout.write, 1800.0)
        self.assertEqual(adapter.client.timeout.pool, 1800.0)
        self.assertEqual(adapter.client.timeout.connect, 5.0)
        self.assertEqual(
            adapter.nebius_model_endpoint.model_dump(mode="json"),
            {
                "provider": "nebius",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "contacted_url": "https://api.example.test/v1",
            },
        )

    def test_construction_refuses_a_setting_the_model_cannot_translate(self):
        # Configuration read applies the same translation table; construction
        # checks it again before any client exists (the configuration fault
        # path).
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            model_endpoint_policy,
            model_registry,
            use_model_registry,
        )
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfigError,
            resolve_runtime_config,
        )

        loaded = loaded_config(model_settings={"top_k": 5})
        selected = loaded.selections.model_for_policy(loaded.invocation_policy)
        policy = model_endpoint_policy(selected.model)
        untranslating = replace(
            policy,
            wire_settings={
                name: field
                for name, field in policy.wire_settings.items()
                if name != "top_k"
            },
        )
        # The Attempt binds its entry when it resolves; the adapter reads it.
        with use_model_registry(model_registry().replacing(untranslating)):
            bound = resolve_runtime_config(
                loaded.selections,
                {"MODEL_KEY": "dummy-key", "SECOND_KEY": "second-key"},
                agent_role="planning_agent",
            )
        with self.assertRaises(RuntimeConfigError) as refused:
            NebiusModelEndpointAdapter(bound, selected)
        self.assertEqual(
            (refused.exception.code, refused.exception.violations),
            ("runtime_config_invalid", ("agent_invocation.model_settings.top_k",)),
        )
        self.assertEqual(
            NebiusModelEndpointAdapter(loaded, selected).endpoint_policy, policy
        )

    def test_effective_policy_is_idempotent_and_refuses_invalid_values(self):
        adapter = adapter_for_reference()
        request_parameters = parameters(functions=[tool("inspect")])

        first = adapter.prepare_request({"max_tokens": 245760}, request_parameters)
        second = adapter.prepare_request(first[0], first[1])

        self.assertEqual(first, second)
        self.assertEqual(
            first[0],
            {
                "parallel_tool_calls": False,
                "max_tokens": 245760,
            },
        )
        enabled, _ = adapter.prepare_request(
            {"parallel_tool_calls": True}, request_parameters
        )
        self.assertIs(enabled["parallel_tool_calls"], True)
        self.assertEqual(
            adapter.prepare_request(enabled, request_parameters)[0], enabled
        )
        for settings in (
            {"parallel_tool_calls": "true"},
            {"parallel_tool_calls": 1},
            {"temperature": "0.5"},
            {"max_tokens": 0},
            {"max_tokens": 262144},
            {"extra_body": {"top_k": 1}},
            # A Tool choice is a mode or an offered Tool's name.
            {"tool_choice": "none"},
            {"tool_choice": ["inspect"]},
            {"tool_choice": "unoffered"},
        ):
            # Configuration read refuses every such source, so a request
            # carrying one is our bug: refused before it is recorded or sent.
            with self.subTest(settings=settings), self.assertRaises(Invariant):
                adapter.prepare_request(settings, request_parameters)
            with self.subTest(shaped=settings), self.assertRaises(Invariant) as bug:
                shape_request(
                    adapter.endpoint_policy,
                    {"parallel_tool_calls": False, **settings},
                    request_parameters,
                )
            self.assertEqual(bug.exception.invariant_id, "model_request_invalid")

    def test_tool_choice_renders_only_the_resolved_value(self):
        policy = adapter_for_reference().endpoint_policy

        def choice(settings, request_parameters):
            return shape_request(policy, settings, request_parameters).get(
                "tool_choice", "absent"
            )

        text_allowed = ModelRequestParameters(
            function_tools=[tool("inspect"), tool("search_journey")],
            allow_text_output=True,
            output_mode="text",
        )
        output_shape = parameters(
            functions=[tool("function")], outputs=[tool("result")]
        )
        # Unresolved carries no key whatever the output shape: nothing forces a
        # call by default.
        self.assertEqual(choice({}, text_allowed), "absent")
        self.assertEqual(
            choice({}, parameters(functions=[tool("search_journey")])), "absent"
        )
        self.assertEqual(choice({}, output_shape), "absent")
        self.assertEqual(choice({"tool_choice": "auto"}, text_allowed), "auto")
        self.assertEqual(choice({"tool_choice": "auto"}, output_shape), "auto")
        no_tools = ModelRequestParameters(allow_text_output=True, output_mode="text")
        self.assertEqual(choice({}, no_tools), "absent")
        with self.assertRaises(Invariant):
            choice({"tool_choice": "auto"}, no_tools)

    def test_no_request_carries_a_grammar_its_model_does_not_declare(self):
        """Owner, #476 alignment G1.2 B: a strict Tool, a forced choice or a
        response format puts the request under a provider grammar; only a model
        entry that records grammar support may send one."""
        from dataclasses import replace

        policy = adapter_for_reference().endpoint_policy
        self.assertIsNone(policy.grammar_evidence)
        strict = replace(tool("inspect"), strict=True)
        output_shape = parameters(
            functions=[tool("function")], outputs=[tool("result")]
        )
        for settings, request_parameters in (
            ({"tool_choice": "required"}, output_shape),
            ({"tool_choice": "result"}, output_shape),
            ({"tool_choice": "function"}, output_shape),
            ({}, parameters(functions=[strict])),
            ({"tool_choice": "auto"}, parameters(functions=[tool("a"), strict])),
        ):
            with self.subTest(settings=settings), self.assertRaises(Invariant) as bug:
                shape_request(policy, settings, request_parameters)
            self.assertEqual(bug.exception.invariant_id, "model_request_grammar")
        declared = replace(policy, grammar_evidence="test: declared support")
        self.assertEqual(
            shape_request(declared, {"tool_choice": "required"}, output_shape)[
                "tool_choice"
            ],
            "required",
        )
        self.assertEqual(
            shape_request(declared, {"tool_choice": "result"}, output_shape)[
                "tool_choice"
            ],
            {"type": "function", "function": {"name": "result"}},
        )
        self.assertIs(
            shape_request(declared, {}, parameters(functions=[strict]))["tools"][0][
                "function"
            ]["strict"],
            True,
        )

    def test_a_forced_choice_needs_recorded_grammar_evidence(self):
        """Owner, #476 alignment G1.2 B: a model entry may declare a forced
        choice only with recorded evidence that its endpoint forces without a
        grammar."""
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
        )

        for forced in ("required", "inspect"):
            with self.subTest(forced=forced), self.assertRaises(ValueError):
                replace(NEMOTRON_SUPER_VL_ENDPOINT_POLICY, tool_choice=forced)
        replace(
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            tool_choice="required",
            grammar_evidence="test: declared support",
        )


class NebiusOutputCeilingTests(unittest.TestCase):
    def test_selected_model_allows_smaller_positive_maximum_with_strict_values(self):
        adapter = adapter_for_reference()
        self.assertEqual(
            adapter.settings_selection.context_budget.output_ceiling, 245760
        )
        self.assertEqual(
            AgentModelSettings(max_tokens=245760).request_settings(),
            {
                "parallel_tool_calls": False,
                "max_tokens": 245760,
            },
        )
        for allowed in (1, 32768, 245759, 245760):
            with self.subTest(max_tokens=allowed):
                settings, _ = adapter.prepare_request(
                    {"max_tokens": allowed}, parameters(functions=[])
                )
                self.assertEqual(settings["max_tokens"], allowed)
        for refused in (0, -1, True, "245760", 1.5):
            with self.subTest(max_tokens=refused), self.assertRaises(ValidationError):
                AgentModelSettings(max_tokens=refused)

    def test_sampling_contract_refuses_extras_and_invalid_ranges(self):
        for settings in (
            {"temperature": -0.1},
            {"temperature": float("nan")},
            {"temperature": True},
            {"top_p": 0},
            {"top_p": 1.1},
            {"top_k": 0},
            {"top_k": 1.5},
            {"seed": True},
            {"seed": "1"},
            {"extra_body": {"top_k": 1}},
            {"tool_choice": "required"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                AgentModelSettings.model_validate(settings)

    def test_a_forced_choice_is_refused_for_a_model_without_grammar_support(self):
        """Owner, #476 alignment G1.2 B: configuration read refuses a forced
        choice at every scope for a model with no recorded grammar support."""
        configured = loaded_config().selections.model_dump(mode="json")
        stage_path = ("agent_invocation", "scopes", "stage", "reference", "tool_choice")
        for value, scopes, global_values, location in (
            (
                "required",
                {},
                {"tool_choice": "required"},
                stage_path[:1] + stage_path[-1:],
            ),
            (
                "required",
                {"stage": {"reference": {"tool_choice": "required"}}},
                {},
                stage_path,
            ),
            (
                "resolve_taxon",
                {"stage": {"reference": {"tool_choice": "resolve_taxon"}}},
                {},
                stage_path,
            ),
        ):
            configured["agent_invocation"] = {
                "model_reference": "qwen",
                "scopes": scopes,
                **global_values,
            }
            with self.subTest(value=value, location=location):
                with self.assertRaises(ValidationError) as raised:
                    RuntimeConfiguration.model_validate(configured)
                self.assertIn(
                    (location, "tool_choice_needs_grammar"),
                    [(e["loc"], e["type"]) for e in raised.exception.errors()],
                )

    def test_only_valid_offered_tool_choice_values_are_accepted(self):
        # tool_choice is a scoped policy setting, not a model request setting.
        # Forced values are read here under a Qwen entry that records grammar
        # support.
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            model_endpoint_policy,
            model_registry,
            use_model_registry,
        )

        with use_model_registry(
            model_registry().replacing(
                replace(model_endpoint_policy(QWEN), grammar_evidence="test")
            )
        ):
            self._offered_tool_choice_values()

    def _offered_tool_choice_values(self):
        selection = adapter_for_reference().settings_selection
        with self.assertRaisesRegex(ValueError, "unsupported.*tool_choice"):
            selection.validate_settings({"tool_choice": "required"})
        configured = loaded_config().selections.model_dump(mode="json")

        def validate(scopes, **global_values):
            configured["agent_invocation"] = {
                "model_reference": "qwen",
                "scopes": scopes,
                **global_values,
            }
            return RuntimeConfiguration.model_validate(configured)

        def refusal(scopes, **global_values):
            with self.assertRaises(ValidationError) as raised:
                validate(scopes, **global_values)
            return [(e["loc"], e["type"]) for e in raised.exception.errors()]

        for scope, name in (("role", "planning_agent"), ("stage", "reference")):
            with self.subTest(model_settings=scope):
                self.assertEqual(
                    refusal(
                        {scope: {name: {"model_settings": {"tool_choice": "required"}}}}
                    ),
                    [
                        (
                            (
                                "agent_invocation",
                                "scopes",
                                scope,
                                name,
                                "model_settings",
                                "tool_choice",
                            ),
                            "extra_forbidden",
                        )
                    ],
                )
        # Accepted: a mode, or a Tool the stage offers (a registered stage Tool
        # or the search), resolved for that stage with its source.
        for value in ("required", "auto", "resolve_taxon", "search_journey"):
            with self.subTest(accepted=value):
                policy = validate(
                    {"stage": {"reference": {"tool_choice": value}}}
                ).agent_invocation_policy("planning_agent", AgentInvocationOverrides())
                entry = policy.stage_settings["reference"]
                self.assertEqual(
                    (entry.tool_choice, entry.settings_sources["tool_choice"]),
                    (value, "stage"),
                )
                self.assertEqual(policy.stage_tool_choice("reference"), value)
                self.assertIsNone(policy.stage_tool_choice("companion"))
        # One merge: global, then role, then stage; each stage records its source.
        merged = validate(
            {
                "role": {"planning_agent": {"tool_choice": "required"}},
                "stage": {"reference": {"tool_choice": "resolve_taxon"}},
            },
            tool_choice="auto",
        )
        planning = merged.agent_invocation_policy(
            "planning_agent", AgentInvocationOverrides()
        )
        investigation = merged.agent_invocation_policy(
            "investigation_planner", AgentInvocationOverrides()
        )
        resolved = {
            (role, stage): (entry.tool_choice, entry.settings_sources["tool_choice"])
            for role, policy in (
                ("planning_agent", planning),
                ("investigation_planner", investigation),
            )
            for stage, entry in policy.stage_settings.items()
        }
        self.assertEqual(
            resolved[("planning_agent", "reference")], ("resolve_taxon", "stage")
        )
        self.assertEqual(
            resolved[("planning_agent", "companions")], ("required", "role")
        )
        self.assertEqual(
            resolved[("investigation_planner", "source_reading")], ("auto", "global")
        )
        self.assertEqual(
            {stage for role, stage in resolved if role == "planning_agent"},
            set(declared_role("planning_agent").stages),
        )
        # Unset everywhere: the one default is recorded once, globally, and
        # every stage resolves it from there.
        unset = validate({}).agent_invocation_policy(
            "planning_agent", AgentInvocationOverrides()
        )
        self.assertEqual(
            {
                entry.settings_sources["tool_choice"]
                for entry in unset.stage_settings.values()
            },
            {"global"},
        )
        self.assertEqual(unset.stage_tool_choice("reference"), "auto")
        unset_invocation = validate({}).model_dump(mode="json")["agent_invocation"]
        self.assertEqual(unset_invocation["tool_choice"], "auto")
        self.assertEqual(unset_invocation["scopes"]["role"], {})
        # Refused at configuration read, by the path that set it: a Tool the
        # stage does not offer, including through a wider scope, ...
        stage_path = ("agent_invocation", "scopes", "stage", "reference", "tool_choice")
        for unoffered in ("fetch_refsnp", "compile_proposal"):
            with self.subTest(unoffered=unoffered):
                self.assertEqual(
                    refusal({"stage": {"reference": {"tool_choice": unoffered}}}),
                    [(stage_path, "tool_choice_not_offered")],
                )
        # Investigation's stage offers its output Tool.
        reading = validate(
            {"stage": {"source_reading": {"tool_choice": "compile_proposal"}}}
        ).agent_invocation_policy("investigation_planner", AgentInvocationOverrides())
        self.assertEqual(
            reading.stage_tool_choice("source_reading"), "compile_proposal"
        )
        self.assertIn(
            (
                ("agent_invocation", "scopes", "role", "planning_agent", "tool_choice"),
                "tool_choice_not_offered",
            ),
            refusal({"role": {"planning_agent": {"tool_choice": "resolve_taxon"}}}),
        )
        self.assertIn(
            (("agent_invocation", "tool_choice"), "tool_choice_not_offered"),
            refusal({}, tool_choice="resolve_taxon"),
        )
        # ... a value that is neither a mode nor a registered Tool, ...
        for value in ("none", ["resolve_taxon"], "unregistered_tool"):
            with self.subTest(refused=value):
                refused = refusal({"stage": {"reference": {"tool_choice": value}}})
                self.assertEqual({loc[:5] for loc, _ in refused}, {stage_path})
        self.assertEqual(
            refusal({"stage": {"reference": {"tool_choice": None}}}),
            [(stage_path[:-1], "value_error")],
        )
        # ... and any per-Tool entry, until a Tool-owned model call exists.
        self.assertEqual(
            refusal({"tool": {"resolve_taxon": {"tool_choice": "required"}}}),
            [
                (
                    (
                        "agent_invocation",
                        "scopes",
                        "tool",
                        "resolve_taxon",
                        "tool_choice",
                    ),
                    "value_error",
                )
            ],
        )


class NebiusRequestBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_context_awaits_its_supplied_client_transport_release(self):
        respond = self.respond

        class CompletionTransport(LoopTransport):
            async def handle_async_request(self, request):
                await super().handle_async_request(request)
                return respond(request)

        await self.adapter.client.close()
        transport = CompletionTransport()
        self.adapter.client._client = httpx.AsyncClient(transport=transport)
        async with self.adapter:
            await self.adapter.request(
                [ModelRequest(parts=[UserPromptPart(content="controlled lifetime")])],
                {},
                parameters(functions=[]),
            )
        self.assertEqual((transport.requests, transport.close_started), (1, 1))
        self.assertTrue(transport.closed)
        self.assertTrue(self.adapter.client.is_closed())

    async def test_optional_sampling_maps_to_wire_without_raw_body_access_or_state_leak(
        self,
    ):
        messages = [ModelRequest(parts=[UserPromptPart(content="sample")])]
        request_parameters = parameters(functions=[tool("inspect")])
        sampling = {
            "temperature": 0.6,
            "top_p": 0.5,
            "top_k": 1,
            "seed": 7,
            "max_tokens": 32768,
        }
        await self.adapter.request(messages, sampling, request_parameters)
        body = json.loads(self.requests[-1].content)
        for name in ("temperature", "top_p", "top_k", "seed"):
            self.assertEqual(body[name], sampling[name])
        self.assertEqual(body["max_completion_tokens"], 32768)
        self.assertNotIn("extra_body", body)
        self.assertEqual(sampling["top_k"], 1)
        await self.adapter.request(
            messages,
            {"temperature": None, "top_p": None, "top_k": None, "seed": None},
            request_parameters,
        )
        body = json.loads(self.requests[-1].content)
        for name in ("temperature", "top_p", "top_k", "seed"):
            self.assertNotIn(name, body)
        with self.assertRaises(Invariant):
            await self.adapter.request(
                messages,
                {"extra_body": {"top_k": 1, "max_completion_tokens": 1}},
                request_parameters,
            )
        self.assertEqual(len(self.requests), 2)

    async def test_explicit_roster_reference_changes_only_wire_model_and_credential(
        self,
    ):
        messages = [ModelRequest(parts=[UserPromptPart(content="inspect sample")])]
        request_parameters = parameters(functions=[tool("inspect")])
        await self.adapter.request(messages, {}, request_parameters)
        first = self.requests[-1]
        alternate = adapter_for_reference("nemotron")
        await alternate.client._client.aclose()
        alternate.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond), follow_redirects=False
        )
        try:
            await alternate.request(messages, {}, request_parameters)
        finally:
            await alternate.client.close()
        second = self.requests[-1]
        self.assertEqual(
            json.loads(second.content),
            {**json.loads(first.content), "model": NEMOTRON_SUPER_VL},
        )
        self.assertEqual(first.headers["authorization"], "Bearer dummy-key")
        self.assertEqual(second.headers["authorization"], "Bearer second-key")
        self.assertEqual(first.url, second.url)

    async def asyncSetUp(self):
        self.requests = []
        self.adapter = adapter_for_reference()
        await self.adapter.client._client.aclose()
        self.adapter.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond), follow_redirects=False
        )

    async def asyncTearDown(self):
        await self.adapter.client.close()

    def respond(self, request):
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "synthetic-response",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3.5-397B-A17B",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def test_a_forced_tool_choice_is_refused_before_it_is_sent(self):
        """Owner, #476 alignment G1.2 B: a model with no recorded grammar support
        is never sent a forced choice; ``auto`` reaches the wire unchanged."""
        messages = [ModelRequest(parts=[UserPromptPart(content="inspect sample")])]
        request_parameters = ModelRequestParameters(
            function_tools=[tool("inspect"), tool("search_journey")],
            allow_text_output=True,
            output_mode="text",
        )

        with self.assertRaises(Invariant) as bug:
            await self.adapter.request(
                messages, {"tool_choice": "required"}, request_parameters
            )
        self.assertEqual(bug.exception.invariant_id, "model_request_grammar")
        self.assertEqual(self.requests, [])
        await self.adapter.request(
            messages, {"tool_choice": "auto"}, request_parameters
        )

        body = json.loads(self.requests[-1].content)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(
            [item["function"]["name"] for item in body["tools"]],
            ["inspect", "search_journey"],
        )

    async def test_unforced_request_offering_one_tool_carries_no_tool_choice(self):
        # A stage showing only search_journey is unforced: the adapter
        # renders the resolved value and adds no choice of its own.
        messages = [ModelRequest(parts=[UserPromptPart(content="inspect sample")])]
        request_parameters = ModelRequestParameters(
            function_tools=[tool("search_journey")],
            allow_text_output=True,
            output_mode="text",
        )

        await self.adapter.request(messages, {}, request_parameters)

        body = json.loads(self.requests[-1].content)
        self.assertNotIn("tool_choice", body)
        self.assertEqual(
            [item["function"]["name"] for item in body["tools"]], ["search_journey"]
        )

    async def test_short_request_keeps_requested_ceiling_and_omission(self):
        messages = [ModelRequest(parts=[UserPromptPart(content="inspect sample")])]
        request_parameters = parameters(functions=[tool("inspect")])
        prepared = []
        for settings in (
            {},
            {"max_tokens": 245760},
            {"max_tokens": None},
            {"parallel_tool_calls": True},
        ):
            with observe_model_transport_requests(prepared.append):
                await self.adapter.request(messages, settings, request_parameters)
            body = json.loads(self.requests[-1].content)
            self.assertEqual(prepared[-1].request_body, self.requests[-1].content)
            self.assertEqual(prepared[-1].unbudgeted_body, self.requests[-1].content)
            self.assertIs(
                body["parallel_tool_calls"], settings.get("parallel_tool_calls", False)
            )
            self.assertEqual(body["messages"][0]["content"], "inspect sample")
            if settings.get("max_tokens", 245760) is not None:
                self.assertEqual(body["max_completion_tokens"], 245760)
            else:
                self.assertNotIn("max_completion_tokens", body)
                self.assertNotIn("max_tokens", body)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(prepared), 4)

    async def test_repaired_request_budgets_full_utf8_body_without_state_leak(self):
        settings = {"max_tokens": 245760}
        messages = [
            ModelRequest(parts=[UserPromptPart(content="inspect sample")]),
            ModelResponse(
                parts=[
                    ThinkingPart(content="reasoning Ω " * 2000),
                    ToolCallPart(
                        tool_name="result",
                        args={"candidate": "sample"},
                        tool_call_id="synthetic-call",
                    ),
                ]
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="repair 漢 " * 2000,
                        tool_name="result",
                        tool_call_id="synthetic-call",
                    )
                ]
            ),
        ]
        output = tool("result")
        output.parameters_json_schema = {
            "type": "object",
            "description": "schema Ω " * 2000,
            "properties": {"candidate": {"type": "string"}},
        }
        request_parameters = parameters(functions=[], outputs=[output])
        originals = copy.deepcopy((messages, settings, request_parameters))

        # Capture the same synthetic input through the SDK without budgeting to
        # prove that the only outgoing JSON change is the completion allowance.
        with patch.object(
            type(self.adapter.client), "_build_request", AsyncOpenAI._build_request
        ):
            await self.adapter.request(messages, settings, request_parameters)
        unbudgeted = self.requests.pop()
        prepared = []
        with observe_model_transport_requests(prepared.append):
            await self.adapter.request(messages, settings, request_parameters)

        request = self.requests[-1]
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].unbudgeted_body, unbudgeted.content)
        self.assertEqual(prepared[0].request_body, request.content)
        body = json.loads(request.content)
        # Two body bytes per estimated token, rounded up.
        expected = min(245760, 262144 - -(-len(unbudgeted.content) // 2) - 8192)
        self.assertGreaterEqual(expected, 100000)
        self.assertLess(expected, 245760)
        self.assertEqual(body["max_completion_tokens"], expected)
        self.assertEqual(
            body, {**json.loads(unbudgeted.content), "max_completion_tokens": expected}
        )
        self.assertGreater(len(request.content), len(request.content.decode("utf-8")))
        self.assertIn("reasoning Ω", json.dumps(body["messages"], ensure_ascii=False))
        self.assertIn("repair 漢", body["messages"][-1]["content"])
        self.assertIn("schema Ω", json.dumps(body["tools"], ensure_ascii=False))
        self.assertEqual((messages, settings, request_parameters), originals)
        self.assertEqual(len(self.requests), 1)

        await self.adapter.request(
            [ModelRequest(parts=[UserPromptPart(content="another sample")])],
            settings,
            parameters(functions=[tool("inspect")]),
        )
        self.assertEqual(
            json.loads(self.requests[-1].content)["max_completion_tokens"], 245760
        )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(prepared), 1)

    async def test_exhausted_context_refuses_before_transport(self):
        messages = [ModelRequest(parts=[UserPromptPart(content="x" * 2 * 262144)])]
        settings = {"max_tokens": 245760}
        request_parameters = parameters(functions=[tool("inspect")])
        # The same SDK build without budgeting supplies the expected estimate.
        with patch.object(
            type(self.adapter.client), "_build_request", AsyncOpenAI._build_request
        ):
            await self.adapter.request(messages, settings, request_parameters)
        unbudgeted = self.requests.pop()
        prepared = []
        with (
            self.assertRaises(ModelContextExhausted) as refused,
            observe_model_transport_requests(prepared.append),
        ):
            await self.adapter.request(messages, settings, request_parameters)
        exhausted = refused.exception
        self.assertEqual(
            (
                exhausted.input_estimate,
                exhausted.safety_margin,
                exhausted.context_window,
            ),
            (-(-len(unbudgeted.content) // 2), 8192, 262144),
        )
        self.assertEqual(model_contact_failure(exhausted), "not_sent")
        self.assertEqual(self.requests, [])
        self.assertEqual(prepared, [])

    async def test_request_58_sized_body_keeps_completion_room(self):
        # A recorded Qwen request: a 254,315-byte body that the provider counted
        # as about 105k prompt tokens was refused when bytes stood for tokens.
        # At two bytes per token it estimates about 127k and is sent.
        messages = [ModelRequest(parts=[UserPromptPart(content="x" * 254000)])]
        prepared = []
        with observe_model_transport_requests(prepared.append):
            await self.adapter.request(
                messages,
                {"max_tokens": 245760},
                parameters(functions=[tool("inspect")]),
            )

        body_bytes = len(prepared[0].unbudgeted_body)
        self.assertGreater(body_bytes, 254000)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            json.loads(self.requests[0].content)["max_completion_tokens"],
            262144 - -(-body_bytes // 2) - 8192,
        )

    async def test_transport_observers_are_task_local_and_restore_after_cancellation(
        self,
    ):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def respond(request):
            if json.loads(request.content)["messages"][0]["content"] == "cancel me":
                entered.set()
                await release.wait()
            return self.respond(request)

        await self.adapter.client._client.aclose()
        self.adapter.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        cancelled_evidence, successful_evidence, outer_evidence = [], [], []

        async def send(prompt, evidence):
            with observe_model_transport_requests(evidence.append):
                await self.adapter.request(
                    [ModelRequest(parts=[UserPromptPart(content=prompt)])],
                    {},
                    parameters(functions=[]),
                )

        with observe_model_transport_requests(outer_evidence.append):
            cancelled = asyncio.create_task(send("cancel me", cancelled_evidence))
            await asyncio.wait_for(entered.wait(), timeout=3)
            await send("complete me", successful_evidence)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            await self.adapter.request(
                [ModelRequest(parts=[UserPromptPart(content="outer scope")])],
                {},
                parameters(functions=[]),
            )

        for evidence, prompt in (
            (cancelled_evidence, "cancel me"),
            (successful_evidence, "complete me"),
            (outer_evidence, "outer scope"),
        ):
            self.assertEqual(len(evidence), 1)
            self.assertEqual(
                json.loads(evidence[0].request_body)["messages"][0]["content"], prompt
            )


QWEN = "Qwen/Qwen3.5-397B-A17B"
NEMOTRON_ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
NEMOTRON_SUPER_VL = "nvidia/Nemotron-3.5-Super-VL"


class ModelFamilyWireTests(unittest.IsolatedAsyncioTestCase):
    """Each listed model's adapter defaults, as the bytes its requests carry."""

    async def send(
        self, model: str, settings: dict, content: str = "inspect sample", **invocation
    ) -> bytes:
        from bio_neb.agentic_journey.runtime_config import resolve_runtime_config

        loaded = resolve_runtime_config(
            RuntimeConfiguration.model_validate(
                {
                    "contract_version": "1.0",
                    "heartbeat_interval_seconds": 3,
                    "model_roster": {
                        "selected": {
                            "provider": "nebius",
                            "endpoint": "https://api.example.test",
                            "model": model,
                            "credential_env": "MODEL_KEY",
                        }
                    },
                    "agent_invocation": {"model_reference": "selected", **invocation},
                }
            ),
            {"MODEL_KEY": "dummy-key"},
            agent_role="planning_agent",
        )
        adapter = NebiusModelEndpointAdapter(
            loaded, loaded.selections.model_for_policy(loaded.invocation_policy)
        )
        sent = []

        def respond(request):
            sent.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "synthetic-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "synthetic"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        await adapter.client._client.aclose()
        adapter.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        )
        try:
            await adapter.request(
                [ModelRequest(parts=[UserPromptPart(content=content)])],
                settings,
                parameters(functions=[tool("inspect")]),
            )
        finally:
            await adapter.client.close()
        (request,) = sent
        return request.content

    async def test_qwen_request_bytes_are_unchanged_by_the_nemotron_profile(self):
        # Product regression: captured from the adapter at NemoEval c4cc9a85, before
        # the Nemotron family profile existed. Qwen's adapter stays as it was, less
        # the forced Tool choice no request carries by default any more.
        tools = (
            '"tools":[{"type":"function","function":'
            '{"name":"inspect","description":"inspect","parameters":{"type":"object",'
            '"properties":{},"additionalProperties":false}}}]'
        )
        head = (
            '{"messages":[{"role":"user","content":"inspect sample"}],'
            '"model":"Qwen/Qwen3.5-397B-A17B","max_completion_tokens":245760,'
            '"parallel_tool_calls":false,'
        )
        self.assertEqual(
            await self.send(QWEN, {}), (head + '"stream":false,' + tools + "}").encode()
        )
        self.assertEqual(
            await self.send(
                QWEN, {"temperature": 0.6, "top_p": 0.5, "top_k": 4, "seed": 7}
            ),
            (
                head
                + '"seed":7,"stream":false,"temperature":0.6,'
                + tools
                + ',"top_p":0.5,"top_k":4}'
            ).encode(),
        )

    async def test_nemotron_request_carries_the_family_defaults_at_their_wire_paths(
        self,
    ):
        body = json.loads(await self.send(NEMOTRON_ULTRA, {}))
        self.assertEqual(body["model"], NEMOTRON_ULTRA)
        self.assertEqual((body["temperature"], body["top_p"]), (1.0, 0.95))
        self.assertEqual(
            body["chat_template_kwargs"],
            {
                "enable_thinking": True,
                "medium_effort": True,
                "force_nonempty_content": True,
            },
        )
        # Only the template controls are new; the rest of the request is unchanged.
        for name in ("reasoning", "force_nonempty_content", "enable_thinking"):
            self.assertNotIn(name, body)
        self.assertEqual(body["max_completion_tokens"], 245760)
        self.assertNotIn("tool_choice", body)
        self.assertIs(body["parallel_tool_calls"], False)

    async def test_nemotron_super_vl_sends_no_sampling_or_template_controls(self):
        # The provider decides sampling and thinking. The reply limit stays the
        # family's: sent without one, Nebius cut replies at 8,192 tokens (human-a1
        # planning).
        body = json.loads(await self.send(NEMOTRON_SUPER_VL, {}))
        self.assertEqual(body["model"], NEMOTRON_SUPER_VL)
        for name in ("temperature", "top_p", "top_k", "seed", "chat_template_kwargs"):
            self.assertNotIn(name, body)
        self.assertEqual(body["max_completion_tokens"], 245760)

    async def test_nemotron_requests_carry_its_own_window_and_qwens_allowance(self):
        # Nemotron 3 Ultra's window on Nebius is 1,048,576 tokens; its reply
        # allowance stays 245,760, decided apart from the window. A body
        # Qwen's 262,144 window refuses is sent with the whole allowance.
        long_input = "x" * 2 * 262144
        with self.assertRaises(ModelContextExhausted) as qwen_refused:
            await self.send(QWEN, {}, long_input)
        self.assertEqual(qwen_refused.exception.context_window, 262144)
        body = json.loads(await self.send(NEMOTRON_ULTRA, {}, long_input))
        self.assertEqual(body["max_completion_tokens"], 245760)
        # Only past its own window is a Nemotron request refused before contact.
        with self.assertRaises(ModelContextExhausted) as nemotron_refused:
            await self.send(NEMOTRON_ULTRA, {}, "x" * 2 * 1048576)
        self.assertEqual(nemotron_refused.exception.context_window, 1048576)

    async def test_nemotron_reasoning_writes_both_template_keys_for_each_value(self):
        for reasoning, thinking, medium in (
            ("off", False, False),
            ("medium", True, True),
            ("on", True, False),
        ):
            with self.subTest(reasoning=reasoning):
                # Configured globally, the way a runtime tunes the family default.
                body = json.loads(
                    await self.send(
                        NEMOTRON_ULTRA,
                        {},
                        model_settings={
                            "reasoning": reasoning,
                            "force_nonempty_content": False,
                            "top_k": 20,
                        },
                    )
                )
                self.assertEqual(
                    body["chat_template_kwargs"],
                    {
                        "enable_thinking": thinking,
                        "medium_effort": medium,
                        "force_nonempty_content": False,
                    },
                )
                self.assertEqual(body["top_k"], 20)
        # An explicit None omits a template control like any other setting.
        body = json.loads(
            await self.send(
                NEMOTRON_ULTRA, {"reasoning": None, "force_nonempty_content": None}
            )
        )
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual((body["temperature"], body["top_p"]), (1.0, 0.95))

    async def test_a_configured_null_template_control_is_omitted_from_the_body(self):
        omitted = {"reasoning": None, "force_nonempty_content": None}
        for scope, invocation in (
            ("global", {"model_settings": omitted}),
            (
                "role",
                {"scopes": {"role": {"planning_agent": {"model_settings": omitted}}}},
            ),
        ):
            with self.subTest(scope=scope):
                body = json.loads(await self.send(NEMOTRON_ULTRA, {}, **invocation))
                self.assertNotIn("chat_template_kwargs", body)
                self.assertEqual((body["temperature"], body["top_p"]), (1.0, 0.95))


def _proposal_tool() -> ToolDefinition:
    return ToolDefinition(
        name="record_proposal_requirements",
        description="Record the requirements.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "requirements": {"type": "array", "items": {"type": "object"}},
                "note": {"type": "string"},
            },
            "required": ["requirements"],
            "additionalProperties": False,
        },
    )


def _placed_policy(placement: str):
    """Nemotron's endpoint policy at another call-reasoning placement (#402)."""
    from dataclasses import replace

    from bio_neb.agentic_journey.model_settings import NEMOTRON_ULTRA_ENDPOINT_POLICY

    return replace(NEMOTRON_ULTRA_ENDPOINT_POLICY, call_reasoning=placement)


@contextmanager
def _nemotron_placement(placement: str):
    """Nemotron's adapter built at another placement; its default is `none` (#402)."""
    from bio_neb.agentic_journey.model_settings import (
        model_registry,
        use_model_registry,
    )

    with use_model_registry(model_registry().replacing(_placed_policy(placement))):
        yield


class CallReasoningProjectionTests(unittest.TestCase):
    """A family's placement shows every Tool with a required call-reasoning field,
    first or last, or with none (#241, #402)."""

    def test_the_last_placement_puts_a_required_reasoning_string_after_the_arguments(
        self,
    ):
        offered = _proposal_tool()
        projected = _placed_policy("last").project_tool_schema(offered)
        schema = projected.parameters_json_schema
        # S1: a top-level sibling declared last came last in every reply.
        self.assertEqual(
            list(schema["properties"]), ["requirements", "note", "reasoning"]
        )
        self.assertEqual(
            schema["properties"]["reasoning"],
            {
                "type": "string",
                "description": "After the arguments: why you chose these values.",
            },
        )
        self.assertEqual(schema["required"], ["requirements", "reasoning"])
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(
            (projected.name, projected.description),
            ("record_proposal_requirements", "Record the requirements."),
        )
        # The offered definition itself is not changed: the assembly offers it
        # again on the next request.
        self.assertEqual(
            offered.parameters_json_schema, _proposal_tool().parameters_json_schema
        )

    def test_the_first_placement_puts_the_reasoning_field_before_the_arguments(self):
        policy = _placed_policy("first")
        schema = policy.project_tool_schema(_proposal_tool()).parameters_json_schema
        self.assertEqual(
            list(schema["properties"]), ["reasoning", "requirements", "note"]
        )
        self.assertEqual(
            schema["properties"]["reasoning"]["description"],
            "Before the arguments: why you choose these values.",
        )
        self.assertEqual(schema["required"], ["reasoning", "requirements"])

    def test_a_tool_without_properties_gains_only_the_reasoning_field(self):
        schema = (
            _placed_policy("last")
            .project_tool_schema(tool("inspect"))
            .parameters_json_schema
        )
        self.assertEqual(
            (schema["type"], list(schema["properties"]), schema["required"]),
            ("object", ["reasoning"], ["reasoning"]),
        )

    def test_a_tool_that_already_has_a_reasoning_argument_is_refused(self):
        clash = ToolDefinition(
            name="explain",
            parameters_json_schema={
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "explain already has a reasoning argument"
        ):
            _placed_policy("last").project_tool_schema(clash)

    def test_qwen_tools_are_shown_as_offered(self):
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy

        offered = _proposal_tool()
        self.assertIs(model_endpoint_policy(QWEN).project_tool_schema(offered), offered)

    def test_nemotron_tools_carry_no_reasoning_argument(self):
        # Every recorded runaway sat in that free-text argument; with it removed
        # a probe saw 0 runaways in 32 replies.
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy

        policy = model_endpoint_policy(NEMOTRON_ULTRA)
        self.assertEqual(policy.call_reasoning, "none")
        for offered in (_proposal_tool(), tool("inspect")):
            self.assertIs(policy.project_tool_schema(offered), offered)


def _journey_search_tool() -> ToolDefinition:
    """`search_journey` as Nemotron is shown it: its arguments, with no call reasoning."""
    return ToolDefinition(
        name="search_journey",
        description="Search the Journey.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "event_ids": {"type": "array", "items": {"type": "string"}},
                "before_event_id": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    )


def _wire_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class NemotronReplyReadingTests(unittest.IsolatedAsyncioTestCase):
    """Nemotron's wire quirks are read in its adapter before the loop sees a reply (#241).

    Replies are recorded Nemotron shapes (#279 S4, #282 s5) answered through the
    real adapter's HTTP transport, as the endpoint would send them.
    """

    async def exchange(
        self,
        model: str,
        message: dict,
        *,
        finish_reason: str = "stop",
        history: list | None = None,
        tools: list[ToolDefinition] | None = None,
        placement: str | None = None,
    ):
        """One request through the real adapter; its reply and the sent body.

        ``placement`` builds Nemotron's adapter at that call-reasoning placement.
        """
        with _nemotron_placement(placement) if placement else nullcontext():
            return await self._exchange(model, message, finish_reason, history, tools)

    async def _exchange(self, model, message, finish_reason, history, tools):
        from bio_neb.agentic_journey.runtime_config import resolve_runtime_config

        loaded = resolve_runtime_config(
            RuntimeConfiguration.model_validate(
                {
                    "contract_version": "1.0",
                    "heartbeat_interval_seconds": 3,
                    "model_roster": {
                        "selected": {
                            "provider": "nebius",
                            "endpoint": "https://api.example.test",
                            "model": model,
                            "credential_env": "MODEL_KEY",
                        }
                    },
                    "agent_invocation": {"model_reference": "selected"},
                }
            ),
            {"MODEL_KEY": "dummy-key"},
            agent_role="planning_agent",
        )
        adapter = NebiusModelEndpointAdapter(
            loaded, loaded.selections.model_for_policy(loaded.invocation_policy)
        )
        sent = []

        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "recorded-shape",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", **message},
                            "finish_reason": finish_reason,
                        }
                    ],
                },
            )

        await adapter.client._client.aclose()
        adapter.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        )
        try:
            response = await adapter.request(
                [
                    ModelRequest(parts=[UserPromptPart(content="inspect sample")]),
                    *(history or ()),
                ],
                {},
                parameters(functions=tools or [tool("search_journey")]),
            )
        finally:
            await adapter.client.close()
        (body,) = sent
        return response, body

    def kinds(self, response):
        return [
            (type(part).__name__, getattr(part, "id", None)) for part in response.parts
        ]

    async def test_call_reasoning_moves_to_a_recorded_part_before_its_call(self):
        arguments = '{"text": "DUROC-07", "limit": 5, "reasoning": "Find the sample."}'
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "reasoning_content": "The sample is named in the request.",
                "tool_calls": [_wire_call("call-1", "search_journey", arguments)],
            },
            finish_reason="tool_calls",
            placement="last",
        )
        self.assertEqual(
            self.kinds(response),
            [
                ("ThinkingPart", "reasoning_content"),
                ("ThinkingPart", "call_reasoning"),
                ("ToolCallPart", None),
            ],
        )
        thinking, reasoning, call = response.parts
        self.assertEqual(thinking.content, "The sample is named in the request.")
        self.assertEqual(reasoning.content, "Find the sample.")
        self.assertEqual(
            reasoning.provider_details,
            {"tool_call_id": "call-1", "tool_name": "search_journey"},
        )
        # Domain admission reads arguments without it, still as a JSON string.
        self.assertEqual(call.tool_call_id, "call-1")
        self.assertIsInstance(call.args, str)
        self.assertEqual(json.loads(call.args), {"text": "DUROC-07", "limit": 5})
        # The raw call stays recorded; splitting the field is not a quirk.
        self.assertEqual(
            response.provider_details["reply_reading"],
            {
                "quirks": [],
                "raw": {
                    "tool_calls": [_wire_call("call-1", "search_journey", arguments)]
                },
            },
        )

    async def test_nemotron_calls_are_read_with_no_call_reasoning_part(self):
        # The record drops the `call_reasoning` part; the model's own
        # reasoning_content stays.
        arguments = '{"text": "DUROC-07", "limit": 5}'
        response, body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "reasoning_content": "The sample is named in the request.",
                "tool_calls": [_wire_call("call-1", "search_journey", arguments)],
            },
            finish_reason="tool_calls",
        )
        self.assertEqual(
            self.kinds(response),
            [("ThinkingPart", "reasoning_content"), ("ToolCallPart", None)],
        )
        self.assertEqual(response.parts[1].args, arguments)
        self.assertNotIn("reply_reading", response.provider_details or {})
        # Shown with no reasoning argument, under no forced Tool choice.
        self.assertNotIn("tool_choice", body)
        self.assertNotIn(
            "reasoning", body["tools"][0]["function"]["parameters"]["properties"]
        )

    async def test_fenced_call_arguments_are_read_as_the_unfenced_call(self):
        # The adapter reads a call's arguments with the Model turn's one
        # reader, so a fenced call has its reasoning split like any other call,
        # and the call stays fenced for admission to record.
        from bio_neb.agentic_journey.model_turn import read_arguments

        arguments = '{"text": "DUROC-07", "limit": 5, "reasoning": "Find the sample."}'
        fenced = "```json\n" + arguments + "\n```"
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "tool_calls": [_wire_call("call-1", "search_journey", fenced)],
            },
            finish_reason="tool_calls",
            placement="last",
        )
        reasoning, call = response.parts
        self.assertEqual(reasoning.content, "Find the sample.")
        read = read_arguments(call.args)
        self.assertEqual(
            (read.value, read.fenced), ({"text": "DUROC-07", "limit": 5}, True)
        )
        # The fenced wire call stays recorded as received.
        self.assertEqual(
            response.provider_details["reply_reading"],
            {
                "quirks": [],
                "raw": {"tool_calls": [_wire_call("call-1", "search_journey", fenced)]},
            },
        )

    async def test_a_call_without_its_reasoning_is_kept_and_recorded_as_a_quirk(self):
        from bio_neb.agentic_journey.model_settings import reply_quirks

        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "tool_calls": [_wire_call("call-1", "search_journey", '{"limit": 5}')],
            },
            finish_reason="tool_calls",
            placement="last",
        )
        self.assertEqual(self.kinds(response), [("ToolCallPart", None)])
        self.assertEqual(response.parts[0].args, '{"limit": 5}')
        self.assertEqual(
            response.provider_details["reply_reading"],
            {"quirks": ["call_reasoning_missing"], "raw": {}},
        )
        # The quirk is what an injection trigger reads from the recorded reply.
        self.assertEqual(
            reply_quirks(response.provider_details), ("call_reasoning_missing",)
        )

    async def test_a_misspelled_argument_name_reaches_admission_as_shown(self):
        # A name that spells a shown argument differently is fixed before
        # admission; the call as sent stays recorded.
        arguments = '{"eventIds": ["e-1"], "Limit": 1}'
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "tool_calls": [_wire_call("call-1", "search_journey", arguments)],
            },
            finish_reason="tool_calls",
            tools=[_journey_search_tool()],
        )
        (call,) = response.parts
        self.assertEqual(json.loads(call.args), {"event_ids": ["e-1"], "limit": 1})
        self.assertEqual(
            response.provider_details["reply_reading"],
            {
                "quirks": ["argument_name"],
                "raw": {
                    "tool_calls": [_wire_call("call-1", "search_journey", arguments)]
                },
            },
        )

    async def test_a_name_the_shim_cannot_place_reaches_admission_as_sent(self):
        # Only a different spelling of one unsent shown name is fixed; every
        # other name is left for admission to refuse, unrecorded as a fix.
        for sent in (
            {"note": "x"},  # no shown name has its words
            {"limit": 1, "Limit": 2},  # the shown name was sent too
            {"eventIds": ["e-1"], "EventIds": ["e-2"]},  # two names claim one
            {"event_id": "e-1"},  # other words: no guess at a meaning
            # Characters outside case, `_`, `-` and camelCase are not a spelling.
            {"event.ids": ["e-1"]},
            {"event/ids": ["e-1"]},
            {"$limit": 1},
            {"limit!": 1},
            {"élimit": 1},
        ):
            with self.subTest(sent=sent):
                arguments = json.dumps(sent)
                response, _body = await self.exchange(
                    NEMOTRON_ULTRA,
                    {
                        "content": None,
                        "tool_calls": [
                            _wire_call("call-1", "search_journey", arguments)
                        ],
                    },
                    finish_reason="tool_calls",
                    tools=[_journey_search_tool()],
                )
                (call,) = response.parts
                self.assertEqual(call.args, arguments)
                self.assertNotIn("reply_reading", response.provider_details or {})

    async def test_qwen_calls_reach_admission_with_the_names_sent(self):
        arguments = '{"eventIds": ["e-1"], "Limit": 1}'
        response, _body = await self.exchange(
            QWEN,
            {
                "content": None,
                "tool_calls": [_wire_call("call-1", "search_journey", arguments)],
            },
            finish_reason="tool_calls",
            tools=[_journey_search_tool()],
        )
        (call,) = response.parts
        self.assertEqual(call.args, arguments)
        self.assertNotIn("reply_reading", response.provider_details or {})

    async def test_a_text_form_call_in_content_becomes_a_real_call(self):
        # A recorded call, completed (there it was cut off at the cap), as
        # Nemotron is now shown its Tools: with no call reasoning.
        content = (
            '[{"name": "search_journey", "parameters": {"event_ids": '
            '["4fe152a7-dffd-4691-8ea2-2e3cac82aaf5"], "limit": 1}}]  \n'
        )
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": content, "reasoning_content": "Select the recipe."},
        )
        self.assertEqual(
            self.kinds(response),
            [("ThinkingPart", "reasoning_content"), ("ToolCallPart", None)],
        )
        thinking, call = response.parts
        self.assertEqual(thinking.content, "Select the recipe.")
        self.assertEqual(call.tool_name, "search_journey")
        self.assertEqual(
            json.loads(call.args),
            {"event_ids": ["4fe152a7-dffd-4691-8ea2-2e3cac82aaf5"], "limit": 1},
        )
        # The read is the quirk the loop's correction reads on the next request.
        self.assertEqual(
            response.provider_details["reply_reading"],
            {"quirks": ["text_form_tool_call"], "raw": {"content": content}},
        )

    async def test_the_last_placement_splits_a_text_form_call_s_reasoning(self):
        content = (
            '[{"name": "search_journey", "parameters": {"limit": 1, '
            '"reasoning": "Read the recipe candidate."}}]'
        )
        response, _body = await self.exchange(
            NEMOTRON_ULTRA, {"content": content}, placement="last"
        )
        reasoning, call = response.parts
        self.assertEqual(json.loads(call.args), {"limit": 1})
        self.assertEqual(
            (reasoning.id, reasoning.content, reasoning.provider_details),
            (
                "call_reasoning",
                "Read the recipe candidate.",
                {"tool_call_id": call.tool_call_id, "tool_name": "search_journey"},
            ),
        )

    async def test_other_text_forms_become_calls_and_plain_text_stays_text(self):
        for content, expected in (
            (
                '<tool_call>{"name": "search_journey", "arguments": '
                '"{\\"limit\\": 2}"}</tool_call>',
                [("search_journey", {"limit": 2})],
            ),
            (
                '```json\n{"name": "search_journey", "arguments": {"limit": 3}}\n```',
                [("search_journey", {"limit": 3})],
            ),
            (
                "<function=search_journey><parameter=limit>4</parameter>"
                "<parameter=text>DUROC-07</parameter></function>",
                [("search_journey", {"limit": 4, "text": "DUROC-07"})],
            ),
            ("The sample is DUROC-07.", None),
            ('I will call {"name": "search_journey"} next.', None),
        ):
            with self.subTest(content=content):
                response, _body = await self.exchange(
                    NEMOTRON_ULTRA, {"content": content}
                )
                calls = [
                    (part.tool_name, json.loads(part.args))
                    for part in response.parts
                    if isinstance(part, ToolCallPart)
                ]
                if expected is None:
                    self.assertEqual(calls, [])
                    self.assertEqual(response.parts[-1].content, content)
                    self.assertIsNone(
                        (response.provider_details or {}).get("reply_reading")
                    )
                else:
                    self.assertEqual(calls, expected)
                    self.assertEqual(
                        response.provider_details["reply_reading"]["quirks"],
                        ["text_form_tool_call"],
                    )

    async def test_a_text_form_call_cut_off_at_the_cap_stays_a_truncation(self):
        # A recorded shape: a JSON call in content, then whitespace to the cap.
        content = (
            '[{"name": "record_proposal_requirements", "parameters": '
            '{"requirements": [{"subject_key": "research_goal"}]}}]' + " \t\n" * 50
        )
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": content, "reasoning_content": "Plan."},
            finish_reason="length",
        )
        # finish_reason is read before content: no call is made of a cut-off reply.
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(
            self.kinds(response),
            [("ThinkingPart", "reasoning_content"), ("TextPart", None)],
        )
        self.assertEqual(response.parts[1].content, content)
        self.assertEqual(
            response.provider_details["reply_reading"],
            {"quirks": ["text_form_tool_call_cut_off"], "raw": {}},
        )

    async def test_think_tags_leave_content_as_thinking(self):
        for content, thinking, text in (
            ("<think>Check the taxon.</think>Done.", ["Check the taxon."], "Done."),
            # Reasoning written into content, closed but never opened.
            ("Check the taxon.\n</think>\n\nDone.", ["Check the taxon."], "\n\nDone."),
            # Opened and cut off before it closed.
            ("Done.<think>Check the", ["Check the"], "Done."),
        ):
            with self.subTest(content=content):
                response, _body = await self.exchange(
                    NEMOTRON_ULTRA, {"content": content}
                )
                self.assertEqual(
                    [
                        part.content
                        for part in response.parts
                        if isinstance(part, ThinkingPart)
                    ],
                    thinking,
                )
                self.assertEqual(
                    [
                        part.content
                        for part in response.parts
                        if not isinstance(part, ThinkingPart)
                    ],
                    [text],
                )
                self.assertEqual(
                    response.provider_details["reply_reading"],
                    {"quirks": ["think_tags"], "raw": {"content": content}},
                )

    async def test_reasoning_content_becomes_a_thinking_part(self):
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": "Done.", "reasoning_content": "Check the taxon."},
        )
        self.assertEqual(
            [(type(part).__name__, part.content) for part in response.parts],
            [("ThinkingPart", "Check the taxon."), ("TextPart", "Done.")],
        )
        self.assertIsNone((response.provider_details or {}).get("reply_reading"))

    async def test_replay_returns_each_call_with_its_reasoning_as_a_json_string(self):
        recorded, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "reasoning_content": "Look it up.",
                "tool_calls": [
                    _wire_call(
                        "call-1",
                        "search_journey",
                        '{"limit": 5, "reasoning": "Find the sample."}',
                    ),
                    _wire_call("call-2", "search_journey", '{"limit": 6}'),
                ],
            },
            finish_reason="tool_calls",
            placement="last",
        )
        thought = ModelResponse(
            parts=[
                ThinkingPart("Old reasoning", id="content", provider_name="openai"),
                TextPart("Done."),
            ]
        )
        _reply, body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": "Done."},
            history=[
                recorded,
                ModelRequest(parts=[UserPromptPart(content="continue")]),
                thought,
                ModelRequest(parts=[UserPromptPart(content="again")]),
            ],
            placement="last",
        )
        replayed = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(
            replayed[0],
            {
                "role": "assistant",
                "reasoning_content": "Look it up.",
                "content": None,
                "tool_calls": [
                    # S3: Nebius refuses object arguments; the model's own order.
                    _wire_call(
                        "call-1",
                        "search_journey",
                        '{"limit": 5, "reasoning": "Find the sample."}',
                    ),
                    _wire_call("call-2", "search_journey", '{"limit": 6}'),
                ],
            },
        )
        # Reasoning read out of tags replays as reasoning, never as tags.
        self.assertEqual(
            replayed[1],
            {
                "role": "assistant",
                "reasoning_content": "Old reasoning",
                "content": "Done.",
            },
        )

    async def test_a_reply_recorded_at_last_replays_without_its_reasoning(self):
        # A Fork from a parent recorded at `last` replays each call as read,
        # with no argument the Tools no longer show.
        recorded, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {
                "content": None,
                "tool_calls": [
                    _wire_call(
                        "call-1",
                        "search_journey",
                        '{"limit": 5, "reasoning": "Find the sample."}',
                    )
                ],
            },
            finish_reason="tool_calls",
            placement="last",
        )
        _reply, body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": "Done."},
            history=[recorded, ModelRequest(parts=[UserPromptPart(content="again")])],
        )
        (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(
            assistant["tool_calls"],
            [_wire_call("call-1", "search_journey", '{"limit": 5}')],
        )

    async def test_a_refused_or_filtered_reply_is_never_read_for_calls(self):
        call = '{"name": "search_journey", "parameters": {"limit": 1}}'
        response, _body = await self.exchange(
            NEMOTRON_ULTRA,
            {"content": call, "refusal": "Cannot comply."},
            finish_reason="content_filter",
        )
        # The framework keeps a refusal as a reply with no parts, as for Qwen.
        self.assertEqual(
            (response.parts, response.finish_reason), ([], "content_filter")
        )
        self.assertNotIn("reply_reading", response.provider_details)
        response, _body = await self.exchange(
            NEMOTRON_ULTRA, {"content": call}, finish_reason="content_filter"
        )
        self.assertEqual(self.kinds(response), [("TextPart", None)])
        self.assertEqual(response.parts[0].content, call)
        self.assertNotIn("reply_reading", response.provider_details or {})

    async def test_a_text_form_call_keeps_its_argument_text_for_admission(self):
        from bio_neb.agentic_journey.model_turn import read_arguments

        content = (
            '[{"name": "search_journey", "arguments": "{broken"}, '
            '{"name": "search_journey", "arguments": "{\\"limit\\": 2}"}, '
            '{"name": "search_journey", "arguments": [1]}]'
        )
        response, _body = await self.exchange(NEMOTRON_ULTRA, {"content": content})
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        self.assertEqual(
            [part.args for part in calls], ["{broken", '{"limit": 2}', "[1]"]
        )
        # The Model turn's one reader, admission's, refuses the malformed and
        # non-object ones.
        self.assertEqual(
            [
                (read.value, read.malformed)
                for read in (read_arguments(part.args) for part in calls)
            ],
            [
                ({"invalid_json": "{broken"}, True),
                ({"limit": 2}, False),
                ({"invalid_json": "[1]"}, True),
            ],
        )
        self.assertEqual(
            response.provider_details["reply_reading"],
            {"quirks": ["text_form_tool_call"], "raw": {"content": content}},
        )

    async def test_think_tags_inside_a_text_form_call_are_its_data(self):
        call = (
            '{"name": "search_journey", "parameters": {"text": "</think> and <think>", '
            '"reasoning": "Quote <think> tags."}}'
        )
        for content, thinking in (
            (call, []),
            ("<think>Plan.</think>\n" + call, ["Plan."]),
        ):
            with self.subTest(content=content):
                response, _body = await self.exchange(
                    NEMOTRON_ULTRA, {"content": content}, placement="last"
                )
                self.assertEqual(
                    [
                        part.content
                        for part in response.parts
                        if isinstance(part, ThinkingPart) and part.id == "content"
                    ],
                    thinking,
                )
                reasoning, call_part = response.parts[-2:]
                self.assertEqual(reasoning.content, "Quote <think> tags.")
                self.assertEqual(
                    json.loads(call_part.args), {"text": "</think> and <think>"}
                )
                self.assertEqual(
                    response.provider_details["reply_reading"]["quirks"],
                    ["think_tags", "text_form_tool_call"]
                    if thinking
                    else ["text_form_tool_call"],
                )

    async def test_qwen_replies_are_read_as_the_compatible_wire_gives_them(self):
        # Product regression: Qwen's adapter does no Nemotron reading.
        content = '[{"name": "search_journey", "parameters": {"limit": 1}}]'
        response, _body = await self.exchange(QWEN, {"content": content})
        self.assertEqual(self.kinds(response), [("TextPart", None)])
        arguments = '{"limit": 5, "reasoning": "Find the sample."}'
        response, _body = await self.exchange(
            QWEN,
            {
                "content": None,
                "tool_calls": [_wire_call("call-1", "search_journey", arguments)],
            },
            finish_reason="tool_calls",
        )
        self.assertEqual(self.kinds(response), [("ToolCallPart", None)])
        self.assertEqual(response.parts[0].args, arguments)
        self.assertNotIn("reply_reading", response.provider_details or {})


class CallReasoningFirstReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_first_placement_replays_reasoning_before_the_arguments(self):
        with _nemotron_placement("first"):
            reading = NemotronReplyReadingTests()
            recorded, _body = await reading.exchange(
                NEMOTRON_ULTRA,
                {
                    "content": None,
                    "tool_calls": [
                        _wire_call(
                            "call-1",
                            "search_journey",
                            '{"reasoning": "Find it.", "limit": 5}',
                        )
                    ],
                },
                finish_reason="tool_calls",
            )
            self.assertEqual(json.loads(recorded.parts[1].args), {"limit": 5})
            _reply, body = await reading.exchange(
                NEMOTRON_ULTRA,
                {"content": "Done."},
                history=[
                    recorded,
                    ModelRequest(parts=[UserPromptPart(content="continue")]),
                ],
            )
        (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            '{"reasoning": "Find it.", "limit": 5}',
        )


class ReasoningOnlyReplayTests(unittest.IsolatedAsyncioTestCase):
    """#395's recorded reasoning-only replies through the real adapter (#421).

    Each said ``finish_reason`` ``tool_calls`` with no call and empty content,
    only reasoning. The adapter reads it as reasoning only, and the next request
    shows the transcript composer's kept reply as an assistant message with
    empty content (#444).
    """

    async def test_each_recorded_reply_is_read_and_replayed_as_reasoning_only(self):
        from pathlib import Path

        from bio_neb.agentic_journey.model_turn import (
            messages_record,
            read_messages_record,
            read_turn,
        )
        from bio_neb.agentic_journey.transcript import kept_reply

        for name in ("issue395-base-4", "issue395-base-5"):
            with self.subTest(name):
                recorded = json.loads(
                    (
                        Path(__file__).parent
                        / "fixtures/reasoning_only"
                        / f"{name}.json"
                    ).read_text()
                )
                (choice,) = recorded["choices"]
                message = {
                    key: value
                    for key, value in choice["message"].items()
                    if key != "role"
                }
                reading = NemotronReplyReadingTests()
                reply, _body = await reading.exchange(
                    QWEN, message, finish_reason=choice["finish_reason"]
                )
                turn = read_turn(reply)
                self.assertEqual(
                    (turn.calls, turn.cut_off, turn.unusable),
                    ((), False, "reasoning_only"),
                )
                self.assertEqual(
                    reply.provider_details["reply_reading"]["quirks"],
                    ["reasoning_only"],
                )
                (kept,) = read_messages_record(
                    [kept_reply(messages_record([reply])[0])]
                )
                _next, body = await reading.exchange(
                    QWEN,
                    {"content": "Done."},
                    history=[
                        kept,
                        ModelRequest(parts=[RetryPromptPart(content="Send the call.")]),
                    ],
                )
                (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
                self.assertEqual(
                    (assistant["content"], assistant.get("tool_calls")), ("", None)
                )
                self.assertIn(choice["message"]["reasoning"], assistant.values())

    async def test_a_kept_reply_whose_calls_were_refused_is_sent_empty(self):
        # Admission keeps a refused call's reply without the call, so only its
        # reasoning is left. Dropping it from Qwen's next request left two user
        # messages back to back; the composer keeps it with empty content like
        # any reply with nothing to send.
        from bio_neb.agentic_journey.model_turn import (
            messages_record,
            read_messages_record,
        )
        from bio_neb.agentic_journey.transcript import kept_reply

        received, _body = await NemotronReplyReadingTests().exchange(
            QWEN,
            {
                "content": None,
                "reasoning": "Inspect the sample.",
                "tool_calls": [_wire_call("call-1", "search_journey", "{}")],
            },
            finish_reason="tool_calls",
        )
        (kept,) = read_messages_record([kept_reply(messages_record([received])[0], ())])
        self.assertEqual([part.part_kind for part in kept.parts], ["thinking", "text"])
        _next, body = await NemotronReplyReadingTests().exchange(
            QWEN,
            {"content": "Done."},
            history=[
                kept,
                ModelRequest(parts=[RetryPromptPart(content="Call refused.")]),
            ],
        )
        (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(
            (assistant["content"], assistant.get("tool_calls")), ("", None)
        )

    async def test_a_resent_reasoning_only_reply_replays_from_its_retained_record(
        self,
    ):
        # After a resend the reply is routed with nothing to run; its history is
        # rebuilt from the Journey and keeps it as received.
        from bio_neb.agentic_journey.model_turn import (
            messages_record,
            read_messages_record,
        )
        from bio_neb.agentic_journey.transcript import kept_message

        reading = NemotronReplyReadingTests()
        reply, _body = await reading.exchange(
            QWEN,
            {"content": None, "reasoning": "Now the call."},
            finish_reason="tool_calls",
        )
        response = {
            "event_id": "response",
            "event_type": "ModelResponseRecorded",
            "payload": {
                "compatible_adapter_evidence": {
                    "response": {"message": messages_record([reply])[0]}
                }
            },
        }
        routed = {
            "event_type": "AgentTurnRouted",
            "payload": {"response_event_id": "response", "run_order": []},
        }
        refused = {
            "event_type": "AgentActionTurnDispositionRecorded",
            "payload": {
                "response_event_id": "response",
                "disposition": "reasoning_refused",
            },
        }
        (kept,) = read_messages_record(
            [kept_message([response, routed, refused], response)]
        )
        _next, body = await reading.exchange(
            QWEN,
            {"content": "Done."},
            history=[
                kept,
                ModelRequest(parts=[RetryPromptPart(content="Send the call.")]),
            ],
        )
        (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(
            (assistant["content"], assistant["reasoning"]), ("", "Now the call.")
        )

    async def test_a_text_reply_still_reads_as_text(self):
        from bio_neb.agentic_journey.model_turn import read_turn

        reply, _body = await NemotronReplyReadingTests().exchange(
            QWEN, {"content": "Finished.", "reasoning": "All done."}
        )
        self.assertIsNone(read_turn(reply).unusable)


if __name__ == "__main__":
    unittest.main()


class ModelRunDefaultsTests(unittest.TestCase):
    """A model's own Run defaults bind when a fresh configuration is read (#387)."""

    def document(self, model: str, **invocation) -> dict:
        return {
            "contract_version": "1.0",
            "heartbeat_interval_seconds": 3,
            "model_roster": {
                "selected": {
                    "provider": "nebius",
                    "endpoint": "https://api.example.test",
                    "model": model,
                    "credential_env": "MODEL_KEY",
                }
            },
            "agent_invocation": {"model_reference": "selected", **invocation},
        }

    def registered(self, model: str, **invocation):
        from bio_neb.agentic_journey.runtime_config import (
            retained_runtime_selections,
        )

        fresh = RuntimeConfiguration.model_validate(self.document(model, **invocation))
        return fresh, retained_runtime_selections(
            fresh.runtime_selections().model_dump(mode="json")
        )

    def resolved(self, selections):
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )

        policy = selections.agent_invocation_policy(
            "investigation_planner", AgentInvocationOverrides()
        )
        return (
            policy.stage_tool_choice("source_reading"),
            policy.continuation.failed_turns,
        )

    def test_super_vl_gets_auto_and_the_role_default_failure_rule(self):
        # The Tool choice both passing Runs used is the model's own; its failure
        # rule is the global role default's.
        for selections in self.registered(NEMOTRON_SUPER_VL):
            self.assertEqual(self.resolved(selections), ("auto", 4))

    def test_a_stated_value_wins_and_the_rest_keep_the_models(self):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )

        # A forced choice needs its model's recorded grammar support.
        supported = replace(NEMOTRON_SUPER_VL_ENDPOINT_POLICY, grammar_evidence="test")
        with use_model_registry(model_registry().replacing(supported)):
            for selections in self.registered(
                NEMOTRON_SUPER_VL, tool_choice="required", failed_turns=6
            ):
                self.assertEqual(self.resolved(selections), ("required", 6))
        for selections in self.registered(NEMOTRON_SUPER_VL, failed_turns=2):
            self.assertEqual(self.resolved(selections), ("auto", 2))

    def test_a_registered_run_keeps_its_defaults_when_the_model_entry_changes(self):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )

        _, registered = self.registered(NEMOTRON_SUPER_VL)
        changed = replace(
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            tool_choice="required",
            failed_turns=7,
            grammar_evidence="test",
        )
        with use_model_registry(model_registry().replacing(changed)):
            # A fresh Run reads the changed entry; the registered one does not.
            for fresh in self.registered(NEMOTRON_SUPER_VL):
                self.assertEqual(self.resolved(fresh), ("required", 7))
            self.assertEqual(self.resolved(registered), ("auto", 4))

    def test_qwen_and_ultra_take_the_one_default_and_state_no_failure_rule(self):
        # The per-model allowances are retired; an entry may state its own
        # number of failed turns (none does), which a fresh Run then takes.
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            model_endpoint_policy,
            model_registry,
            use_model_registry,
        )

        for model in (QWEN, NEMOTRON_ULTRA):
            entry = model_endpoint_policy(model)
            with self.subTest(model=model):
                self.assertIsNone(entry.tool_choice)
                self.assertIsNone(entry.failed_turns)
                for selections in self.registered(model):
                    self.assertEqual(self.resolved(selections), ("auto", 4))
                owned = replace(entry, failed_turns=2)
                with use_model_registry(model_registry().replacing(owned)):
                    for selections in self.registered(model):
                        self.assertEqual(self.resolved(selections), ("auto", 2))

    def test_the_run_file_records_the_tool_choice_once(self):
        """Owner, #476 alignment G1.2 B ("we can do this once no need to each
        call"): the effective choice is recorded once, globally; a role carries
        one only where the model it selects states another."""
        from dataclasses import replace

        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.model_settings import (
            model_endpoint_policy,
            model_registry,
            use_model_registry,
        )
        from bio_neb.agentic_journey.runtime_config import retained_runtime_selections

        def registered(run_model: str, planner_model: str):
            document = self.document(run_model)
            document["model_roster"]["planner"] = {
                **document["model_roster"]["selected"],
                "model": planner_model,
            }
            document["agent_invocation"]["scopes"] = {
                "role": {"investigation_planner": {"model_reference": "planner"}}
            }
            fresh = RuntimeConfiguration.model_validate(document)
            recorded = fresh.runtime_selections().model_dump(mode="json")
            registered = retained_runtime_selections(recorded)
            choices = {
                (
                    role,
                    selections.agent_invocation_policy(
                        role, AgentInvocationOverrides()
                    ).stage_tool_choice(stage),
                )
                for selections in (fresh, registered)
                for role, stage in (
                    ("investigation_planner", "source_reading"),
                    ("table_reader", "vcf_table_reading"),
                )
            }
            return recorded["agent_invocation"], choices

        for run_model, planner_model in (
            (QWEN, NEMOTRON_SUPER_VL),
            (NEMOTRON_SUPER_VL, QWEN),
        ):
            with self.subTest(run_model=run_model):
                invocation, choices = registered(run_model, planner_model)
                self.assertEqual(invocation["tool_choice"], "auto")
                self.assertEqual(
                    invocation["scopes"]["role"],
                    {"investigation_planner": {"model_reference": "planner"}},
                )
                self.assertEqual(
                    choices,
                    {("investigation_planner", "auto"), ("table_reader", "auto")},
                )
        forced = replace(
            model_endpoint_policy(QWEN), tool_choice="required", grammar_evidence="test"
        )
        with use_model_registry(model_registry().replacing(forced)):
            invocation, choices = registered(NEMOTRON_SUPER_VL, QWEN)
        self.assertEqual(invocation["tool_choice"], "auto")
        self.assertEqual(
            invocation["scopes"]["role"]["investigation_planner"]["tool_choice"],
            "required",
        )
        self.assertEqual(
            choices,
            {("investigation_planner", "required"), ("table_reader", "auto")},
        )

    def test_malformed_scopes_reach_the_typed_refusal_unchanged(self):
        # Standards r3 / Spec r3: the fill reads only valid shapes and never
        # repairs or trips over one the validators must refuse.
        from bio_neb.agentic_journey.runtime_config import describe_runtime_drift

        for scopes in (
            [],
            ["bad"],
            {"role": [1]},
            {"role": 1},
            {"role": {"planning_agent": []}},
        ):
            with self.subTest(scopes=scopes):
                document = self.document(NEMOTRON_SUPER_VL, scopes=scopes)
                with self.assertRaises(ValidationError):
                    RuntimeConfiguration.model_validate(document)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "runtime.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    self.assertFalse(describe_runtime_drift(path).valid)

    def test_drift_reads_the_models_own_defaults(self):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )
        from bio_neb.agentic_journey.runtime_config import describe_runtime_drift

        def drift(**invocation):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.json"
                path.write_text(
                    json.dumps(self.document(NEMOTRON_SUPER_VL, **invocation)),
                    encoding="utf-8",
                )
                return {
                    item.path.removeprefix("agent_invocation."): (
                        item.default,
                        item.default_source,
                    )
                    for item in describe_runtime_drift(path).stale_but_valid
                    if item.path.removeprefix("agent_invocation.")
                    in ("tool_choice", "failed_turns")
                }

        self.assertEqual(drift(tool_choice="auto"), {})
        # A forced choice needs recorded grammar support; under it a stated
        # "required" reads against the one default.
        supported = replace(NEMOTRON_SUPER_VL_ENDPOINT_POLICY, grammar_evidence="test")
        with use_model_registry(model_registry().replacing(supported)):
            self.assertEqual(
                drift(tool_choice="required"),
                {"tool_choice": ("auto", "model_default")},
            )
        # With no entry stating one, the role default's number is the default: a
        # file stating it reads no drift, another number reads against it.
        self.assertEqual(drift(failed_turns=4), {})
        self.assertEqual(
            drift(failed_turns=6), {"failed_turns": (4, "product_default")}
        )
        # An entry that states its own failure rule (none does) is the default
        # a stated number is read against.
        entry = replace(NEMOTRON_SUPER_VL_ENDPOINT_POLICY, failed_turns=6)
        with use_model_registry(model_registry().replacing(entry)):
            self.assertEqual(drift(tool_choice="auto", failed_turns=6), {})
            self.assertEqual(
                drift(failed_turns=3), {"failed_turns": (6, "model_default")}
            )
