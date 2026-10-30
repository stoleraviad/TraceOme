"""Every offered Tool and output schema, as each listed model's wire carries it.

Owner, #476 alignment G1.2 B (#477): no request reaches the provider under a
grammar unless the model's policy declares it. This projects every role's offered
Tools and output Tools through the assembly's renderers, the model's schema
projection and the adapter's request shaping, offline, and fails on a strict Tool,
an empty choice or a schema keyword the model's policy does not declare.
"""

import asyncio
import unittest
from typing import Any, get_args

from pydantic_ai import Agent, Tool, ToolOutput
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from bio_neb.agentic_journey.action_admission import (
    grounded_tool_schema,
    output_tool_schema,
)
from bio_neb.agentic_journey.assignment_context import SEARCH_TOOL_NAME
from bio_neb.agentic_journey.invocation import (
    _bind_tool,
    _ModelToolSchemas,
    _researcher_question,
    journey_search_handler,
)
from bio_neb.agentic_journey.model_settings import model_registry
from bio_neb.agentic_journey.researcher_question import ASK_RESEARCHER
from bio_neb.agentic_journey.role_declaration import AgentRole
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.agentic_journey.runtime_config import (
    RuntimeConfiguration,
    resolve_runtime_config,
)
from bio_neb.agentic_journey.tool_execution import function_tool_names
from bio_neb.providers.nebius_model_endpoint import (
    NebiusModelEndpointAdapter,
    shape_request,
)


def _role_tools() -> dict[
    str, tuple[dict[str, Any], tuple[tuple[str, type, str], ...]]
]:
    """Each role's domain Tools and output Tools, as its assembly passes them."""
    from bio_neb.agentic_journey.planning_assignment import _planning_tools
    from bio_neb.investigation.investigation_agent import (
        INVESTIGATION_OUTPUT,
        INVESTIGATION_TOOLS,
    )
    from bio_neb.investigation.vcf_table_reading import TABLE_READING_OUTPUT
    from bio_neb.paper_reading.reading import ANSWER, PLAN
    from bio_neb.paper_reading.tools import READING_TOOLS
    from bio_neb.reporting.explanation_agent import COMBINATION_OUTPUT, MEMBER_OUTPUT

    def outputs(*items):
        return tuple((item.name, item.output_type, item.description) for item in items)

    return {
        "planning_agent": (_planning_tools(), ()),
        "investigation_planner": (INVESTIGATION_TOOLS, outputs(INVESTIGATION_OUTPUT)),
        "table_reader": ({}, outputs(TABLE_READING_OUTPUT)),
        "report_explainer": ({}, outputs(MEMBER_OUTPUT, COMBINATION_OUTPUT)),
        "paper_reader": (READING_TOOLS, outputs(PLAN, ANSWER)),
    }


class _Captured(Exception):
    pass


class _OutputRendering(AbstractCapability[Any]):
    """The assembly's output-Tool rendering (``PydanticActionAdmissionCapability``)."""

    async def prepare_output_tools(self, ctx, tool_defs):
        return [output_tool_schema(definition) for definition in tool_defs]


def _offered(policy, role: str, tools: dict, output) -> ModelRequestParameters:
    """One request's Tools for the role, as its assembly prepares them."""
    agent_tools = [
        Tool(
            _bind_tool(function),
            name=name,
            sequential=True,
            prepare=lambda _ctx, definition: grounded_tool_schema(
                definition, constants={}
            ),
        )
        for name, function in tools.items()
    ]
    if ASK_RESEARCHER in function_tool_names(role):
        agent_tools.append(
            Tool(
                _researcher_question(None, None, None, role),
                name=ASK_RESEARCHER,
                sequential=True,
                prepare=lambda _ctx, definition: grounded_tool_schema(
                    definition, constants={}
                ),
            )
        )
    agent_tools.append(
        Tool(
            journey_search_handler(lambda _ctx, _arguments: {}),
            name=SEARCH_TOOL_NAME,
            sequential=True,
            prepare=lambda _ctx, definition: grounded_tool_schema(
                definition, constants={}
            ),
        )
    )
    captured: list[AgentInfo] = []

    def capture(_messages, info: AgentInfo):
        captured.append(info)
        raise _Captured

    agent = Agent(
        FunctionModel(capture),
        tools=agent_tools,
        output_type=str
        if output is None
        else ToolOutput(output[1], name=output[0], description=output[2]),
        capabilities=[_ModelToolSchemas(policy), _OutputRendering()],
    )
    try:
        asyncio.run(agent.run("offline projection"))
    except _Captured:
        pass
    (info,) = captured
    return ModelRequestParameters(
        function_tools=info.function_tools,
        output_tools=info.output_tools,
        allow_text_output=info.allow_text_output,
        output_mode="text" if output is None else "tool",
    )


_SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_VALUES = ("items", "additionalProperties", "not", "contains")


def _schema_findings(schema: Any, path: str = "") -> tuple[set[str], list[str]]:
    """Every keyword a schema uses, and each empty choice, with its path."""
    keywords: set[str] = set()
    empty: list[str] = []
    if not isinstance(schema, dict):
        return keywords, empty
    keywords.update(schema)
    for name in ("enum", *_SCHEMA_LISTS):
        if name in schema and not schema[name]:
            empty.append(f"{path}/{name}")
    nested = [
        (f"{path}/{name}/{key}", value)
        for name in _SCHEMA_MAPS
        for key, value in (schema.get(name) or {}).items()
    ]
    nested += [
        (f"{path}/{name}/{index}", value)
        for name in _SCHEMA_LISTS
        for index, value in enumerate(schema.get(name) or ())
    ]
    nested += [
        (f"{path}/{name}", schema[name]) for name in _SCHEMA_VALUES if name in schema
    ]
    for where, value in nested:
        inner_keywords, inner_empty = _schema_findings(value, where)
        keywords |= inner_keywords
        empty += inner_empty
    return keywords, empty


def _wire(model: str, role: str, parameters: ModelRequestParameters) -> dict:
    """The request a fresh Run's adapter shapes for the role: its framework
    preparation, then ``shape_request``, which refuses as ``model_request_grammar``
    a strict Tool or a forced choice the model does not declare."""
    configuration = RuntimeConfiguration.model_validate(
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
    )
    loaded = resolve_runtime_config(
        configuration, {"MODEL_KEY": "offline"}, agent_role=role
    )
    policy = loaded.invocation_policy
    adapter = NebiusModelEndpointAdapter(
        loaded, loaded.selections.model_for_policy(policy)
    )
    try:
        stage = declared_role(role).stages[0]
        settings, prepared = adapter.prepare_request(
            {"tool_choice": policy.stage_tool_choice(stage)}, parameters
        )
        return shape_request(adapter.endpoint_policy, settings or {}, prepared)
    finally:
        asyncio.run(adapter.client.close())


class OfferedWireSchemaTests(unittest.TestCase):
    def test_every_role_is_projected_with_every_tool_it_offers(self):
        # A Tool a role registers but this projection misses would escape it.
        tools = _role_tools()
        self.assertEqual(set(tools), set(get_args(AgentRole)))
        for role, (domain, _outputs) in tools.items():
            with self.subTest(role=role):
                self.assertLessEqual(
                    function_tool_names(role),
                    set(domain) | {ASK_RESEARCHER, SEARCH_TOOL_NAME},
                )

    def test_no_offered_schema_carries_a_grammar_or_an_undeclared_keyword(self):
        """Owner, #476 alignment G1.2 B: every offered Tool and output schema, on
        every listed model's wire, carries no strict Tool, no forced choice, no
        empty choice and only the keywords the model's policy declares."""
        registry = model_registry()
        for model, policy in registry.entries.items():
            for role, (tools, outputs) in _role_tools().items():
                for output in outputs or (None,):
                    with self.subTest(
                        model=model, role=role, output=output and output[0]
                    ):
                        wire = _wire(model, role, _offered(policy, role, tools, output))
                        for item in wire["tools"]:
                            function = item["function"]
                            self.assertNotIn("strict", function, function["name"])
                            keywords, empty = _schema_findings(
                                function["parameters"], function["name"]
                            )
                            self.assertEqual(empty, [])
                            self.assertLessEqual(
                                keywords,
                                policy.schema_keywords,
                                f"{function['name']} uses undeclared keywords",
                            )
