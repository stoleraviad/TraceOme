"""One default agentic loop: registry stages, visible stage Tools, explained refusals."""

import copy
import inspect
import itertools
import json
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.agentic_journey import planning_assignment
from bio_neb.agentic_journey.action_admission import request_bound_arguments
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.agentic_journey.tool_execution import (
    function_tool_contracts,
    stage_tool_names,
)
from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.content import canonical_json, canonical_json_digest
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tests.application_fakes import _planning_ready, _registered
from tests.planning_assignment_fakes import PlanningComposition
from tests.reference_fakes import ReferenceAdapter


def _assert_only_the_hold_appended(test, journey_file, before, code):
    """No answer is recorded: the Journey gains only the held Pass (#214)."""
    content = Path(journey_file).read_bytes()
    test.assertEqual(content[: len(before)], before)
    appended = [json.loads(line) for line in content[len(before) :].splitlines()]
    test.assertEqual(
        [(e["event_type"], e["payload"].get("disposition")) for e in appended],
        [("PassStarted", None), ("PassClosed", "held")],
    )
    test.assertEqual(appended[-1]["payload"]["failure"]["invariant_id"], code)


HISTORY = "search_journey"


def _requests(journey_file):
    return [
        e["payload"]["compatible_adapter_evidence"]["request"]
        for e in read_evidence(Journey.open(Path(journey_file)))
        if e["event_type"] == "ModelRequestRecorded"
    ]


class _PrematureChoice(PlanningComposition):
    """Records a reference choice before any discovery, every time it can."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if "record_reference_choice" in {tool.name for tool in info.function_tools}:
            self.calls.append("record_reference_choice")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "record_reference_choice",
                        {"candidate_id": "GCF_000001405.40", "reason": "Too early."},
                        tool_call_id=f"premature-{len(self.calls)}",
                    )
                ]
            )
        return await super().respond(messages, info)


class _PrematureQuestion(PlanningComposition):
    """Asks the reference question once before the proposal plan exists."""

    asked = False

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        shown = {tool.name for tool in info.function_tools}
        if not self.asked and "record_proposal_requirements" in shown:
            self.asked = True
            return ModelResponse(
                parts=[ToolCallPart("request_planning_hitl", {}, tool_call_id="early")]
            )
        return await super().respond(messages, info)


def _as_the_domain_defines(info):
    """The offered Tools without a placement's call-reasoning argument (#241).

    The scripted model chooses its domain arguments from each Tool as the domain
    defines it; a scenario adds its reasoning beside them when it means to. At
    Nemotron's `none` placement the Tools are already as defined (#402).
    """
    from dataclasses import replace

    def offered(definition):
        schema = definition.parameters_json_schema
        if "reasoning" not in schema.get("properties", {}):
            return definition
        properties = dict(schema["properties"])
        properties.pop("reasoning")
        return replace(
            definition,
            parameters_json_schema={
                **schema,
                "properties": properties,
                "required": [n for n in schema["required"] if n != "reasoning"],
            },
        )

    return replace(info, function_tools=[offered(t) for t in info.function_tools])


class _TextFormBrokenCall(PlanningComposition):
    """Writes its first call as text with malformed arguments (#241, Spec r1 F2)."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, TextPart

        if not self.calls:
            self.calls.append("text-form")
            return ModelResponse(
                parts=[TextPart('[{"name": "search_journey", "arguments": "{broken"}]')]
            )
        return await super().respond(messages, _as_the_domain_defines(info))


class _FencedCalls(PlanningComposition):
    """Writes every call's arguments as JSON text inside a Markdown fence (#222)."""

    def __init__(self, receipt, paths):
        super().__init__(receipt, paths)
        self.sent = {}

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        for part in response.parts:
            arguments = part.args_as_dict()
            self.sent[part.tool_call_id] = arguments
            part.args = "```json\n" + json.dumps(arguments) + "\n```"
        return response


class _UngroundedQuote(PlanningComposition):
    """Quotes words the research request never says in one requirement, once."""

    quoted = None

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        part = response.parts[0]
        if part.tool_name == "record_proposal_requirements" and self.quoted is None:
            requirement = next(
                item
                for item in part.args["requirements"]
                if item["knowledge"].get("state") == "asserted"
            )
            requirement["knowledge"]["quote"] = "words nobody wrote"
            self.quoted = requirement["subject_key"]
        return response


class _ValueNotInQuote(PlanningComposition):
    """Types a sample value its quote does not state, once."""

    keyed = None

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        part = response.parts[0]
        if part.tool_name == "record_proposal_requirements" and self.keyed is None:
            requirement = next(
                item
                for item in part.args["requirements"]
                if item["kind"] == "sample_identity"
            )
            requirement["knowledge"]["value"] = "sample-002"
            self.keyed = requirement["subject_key"]
        return response


class _ForeignHistoryRead(PlanningComposition):
    """Reads an event of another Run once, then works as scripted."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if not self.calls:
            self.calls.append("foreign-read")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        HISTORY,
                        {"event_ids": ["foreign-event"]},
                        tool_call_id="foreign",
                    )
                ]
            )
        return await super().respond(messages, info)


class _SliceWithPageLimit(PlanningComposition):
    """Reads one of this Run's own events by slice with a page limit beside it,
    as #107's live Run did (``ToolCallValidated`` ``7096528e``), then plans."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if not self.calls:
            self.calls.append("slice-with-limit")
            first = self.journey.read()[0]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        HISTORY,
                        {
                            "limit": 10,
                            "payload_slice": {
                                "event_id": first["event_id"],
                                "length": 2000,
                            },
                        },
                        tool_call_id="slice-with-limit",
                    )
                ]
            )
        return await super().respond(messages, info)


class _HistoryThenPlan(PlanningComposition):
    """Sends a history read and the proposal plan in one response, once."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if self.calls:
            return await super().respond(messages, info)
        self.calls.append(HISTORY)
        (plan,) = (await super().respond(messages, info)).parts
        return ModelResponse(
            parts=[
                ToolCallPart(
                    HISTORY,
                    {"kinds": ["ResearchRequestRegistered"]},
                    tool_call_id="pair-1",
                ),
                ToolCallPart(plan.tool_name, plan.args, tool_call_id="pair-2"),
            ]
        )


class _QuestionThenRead(PlanningComposition):
    """Sends a researcher question and a history read after it."""

    question = None

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        response = await super().respond(messages, info)
        (part,) = response.parts
        questions = {
            c.tool_name
            for c in function_tool_contracts("planning_agent")
            if c.opens_question
        }
        if part.tool_name not in questions:
            return response
        self.question = part.tool_name
        return ModelResponse(
            parts=[
                part,
                ToolCallPart(
                    HISTORY,
                    {"kinds": ["ResearchRequestRegistered"]},
                    tool_call_id="after-question",
                ),
            ]
        )


class _ReadThenCall(PlanningComposition):
    """Sends a history read and one scripted call in one response, once."""

    def __init__(self, receipt, paths, call):
        super().__init__(receipt, paths)
        self.second = call

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if self.calls:
            return await super().respond(messages, info)
        self.calls.append("pair")
        name, arguments = self.second
        if arguments is None:
            # The scripted model's own next call, with its grounded arguments.
            (part,) = (await super().respond(messages, info)).parts
            name, arguments = part.tool_name, part.args
        return ModelResponse(
            parts=[
                ToolCallPart(
                    HISTORY,
                    {"kinds": ["ResearchRequestRegistered"]},
                    tool_call_id="read-first",
                ),
                ToolCallPart(name, arguments, tool_call_id="call-second"),
            ]
        )


class _ThenSearch(PlanningComposition):
    """Adds a Journey search after the scripted call named ``first``, once."""

    def __init__(self, receipt, paths, first):
        super().__init__(receipt, paths)
        self.first = first
        self.sent = False

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        response = await super().respond(messages, info)
        (part,) = response.parts
        if self.sent or part.tool_name != self.first:
            return response
        self.sent = True
        return ModelResponse(
            parts=[
                part,
                ToolCallPart(
                    HISTORY,
                    {"kinds": ["ResearchRequestRegistered"]},
                    tool_call_id="after-first",
                ),
            ]
        )


def _events(journey_file):
    return read_evidence(Journey.open(Path(journey_file)))


def _refusals(journey_file, tool_name):
    events = read_evidence(Journey.open(Path(journey_file)))
    validations = {
        e["event_id"]: e["payload"]
        for e in events
        if e["event_type"] == "ToolCallValidated"
        and e["payload"]["tool_name"] == tool_name
        and e["payload"]["status"] == "invalid"
    }
    turns = [
        e["payload"]
        for e in events
        if e["event_type"] == "AgentActionTurnDispositionRecorded"
        and set(e["payload"]["validation_event_ids"]) & set(validations)
    ]
    started = {
        e["payload"]["validation_event_id"]
        for e in events
        if e["event_type"] == "ObservedToolExecutionStarted"
    }
    return list(validations.values()), turns, started & set(validations)


class ContinuationProgressTests(unittest.TestCase):
    """Allowance decisions on synthetic checked evidence; no source admission proof."""

    def test_new_receipts_advance_even_when_domain_result_kind_repeats(self):
        from bio_neb.agentic_journey.continuation_policy import (
            ContinuationPolicy,
            validate_continuation,
        )
        from bio_neb.failure import Invariant

        policy = ContinuationPolicy(failed_turns=4)
        events = []

        def record(kind, **payload):
            events.append(
                {
                    "event_id": f"event-{len(events)}",
                    "event_type": kind,
                    "parent_ids": [],
                    "payload": {"attempt_id": "attempt", **payload},
                }
            )
            return events[-1]["event_id"]

        binding = {
            "continuation_policy": policy.model_dump(mode="json"),
            "continuation_policy_id": policy.policy_id,
        }
        record("AttemptStarted", **binding)
        # Investigation's additional domain binding is not a progress entitlement.
        record("InvestigationAttemptBound", **binding)
        for number in (1, 2):
            record("ModelRequestRecorded")
            # Each receipt links its call, as the repeat count reads it.
            call = {"tool_call_id": f"lookup-{number}", "tool_name": "fetch_refsnp"}
            validation = record(
                "ToolCallValidated", arguments={"rsid": f"rs{number}"}, **call
            )
            started = record(
                "ObservedToolExecutionStarted", validation_event_id=validation, **call
            )
            result = record("RefSnpToolResultRecorded")
            record(
                "ObservedToolExecutionSucceeded",
                started_event_id=started,
                stage_result_event_id=result,
            )
        validate_continuation(
            events, attempt_id="attempt", policy=policy, next_request=True
        )

        # Neither a domain fact nor a bare history read proves execution finished.
        for unreceipted in ("RefSnpToolResultRecorded", "JourneySearchRead"):
            with self.subTest(unreceipted=unreceipted):
                prefix = events[:3] + [
                    {
                        "event_id": "unreceipted",
                        "event_type": unreceipted,
                        "parent_ids": [],
                        "payload": {"attempt_id": "attempt"},
                    }
                ]
                # Sending would be our bug, held under its own id.
                with self.assertRaises(Invariant) as held:
                    validate_continuation(
                        prefix, attempt_id="attempt", policy=policy, next_request=True
                    )
                self.assertEqual(
                    held.exception.invariant_id, "continuation_lacks_progress"
                )


class ToolRegistryTests(unittest.TestCase):
    def test_free_text_rules_are_in_the_schema_the_model_reads(self):
        # A text argument the model chooses shows its own rule (not blank, no
        # surrounding whitespace, a reason's length) in the schema it is
        # offered, not only in a Python validator.
        from typing import Annotated

        from pydantic import Field, TypeAdapter, ValidationError
        from pydantic_ai import Tool

        from bio_neb.agentic_journey.action_admission import grounded_tool_schema
        from bio_neb.agentic_journey.invocation import _bind_tool
        from bio_neb.agentic_journey.planning_assignment import _planning_tools

        tools = _planning_tools()

        def offered(name):
            definition = Tool(_bind_tool(tools[name]), name=name).tool_def
            return grounded_tool_schema(definition, constants={}).parameters_json_schema

        def text_rules(schema, names):
            found = []

            def walk(node):
                if isinstance(node, dict):
                    for key, value in (node.get("properties") or {}).items():
                        if key in names:
                            found.append((key, value))
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(schema)
            return found

        def refuses_blank_and_padded(rule):
            pattern = rule.get("pattern")
            self.assertIsNotNone(pattern, rule)
            # Checked with the validators' own regex engine, which also reads
            # the pattern the model is shown as JSON Schema does.
            shown = TypeAdapter(Annotated[str, Field(pattern=pattern)])
            for refused in ("", " ", " padded", "padded ", "padded\n"):
                with self.assertRaises(ValidationError, msg=repr(refused)):
                    shown.validate_python(refused)
            for accepted in ("x", "two words", "two\nlines"):
                shown.validate_python(accepted)

        requirements = text_rules(
            offered("record_proposal_requirements"),
            {"meaning", "quote", "input_role", "requirement"},
        )
        self.assertEqual(
            {name for name, _ in requirements},
            {"meaning", "quote", "input_role", "requirement"},
        )
        for name, rule in requirements:
            with self.subTest(argument=name):
                refuses_blank_and_padded(rule)
        for tool in (
            "record_reference_choice",
            "record_supplied_reference",
            "reconsider_recipe_choice",
        ):
            with self.subTest(tool=tool):
                rule = offered(tool)["properties"]["reason"]
                refuses_blank_and_padded(rule)
                self.assertEqual((rule["minLength"], rule["maxLength"]), (1, 2000))

    def test_the_canonical_text_rule_reaches_every_model_in_words(self):
        # Nemotron never answered a request whose Tool schemas carried the
        # canonical-text pattern (no reply in 300 s twice; 4.7 s and 7.3 s for the
        # same body without it). Every model reads the rule in words, and the
        # Tool's own validator keeps checking the pattern before contact.
        import json

        from pydantic import ValidationError
        from pydantic_ai import Tool

        from bio_neb.agentic_journey.action_admission import grounded_tool_schema
        from bio_neb.agentic_journey.action_policy import CANONICAL_TEXT_PATTERN
        from bio_neb.agentic_journey.invocation import _bind_tool
        from bio_neb.agentic_journey.model_settings import (
            CANONICAL_TEXT_WORDS,
            model_endpoint_policy,
        )
        from bio_neb.agentic_journey.planning_assignment import _planning_tools

        tools = _planning_tools()
        for model in ("Qwen/Qwen3.5-397B-A17B", "nvidia/Nemotron-3-Ultra-550b-a55b"):
            policy = model_endpoint_policy(model)
            for name in ("record_proposal_requirements", "record_reference_choice"):
                with self.subTest(model=model, tool=name):
                    tool = Tool(_bind_tool(tools[name]), name=name)
                    definition = grounded_tool_schema(tool.tool_def, constants={})
                    checked = json.dumps(definition.parameters_json_schema)
                    shown = json.dumps(
                        policy.project_tool_schema(definition).parameters_json_schema
                    )
                    rules = checked.count(json.dumps(CANONICAL_TEXT_PATTERN)[1:-1])
                    self.assertGreater(rules, 0)
                    self.assertNotIn(json.dumps(CANONICAL_TEXT_PATTERN)[1:-1], shown)
                    self.assertEqual(shown.count(CANONICAL_TEXT_WORDS), rules)
                    # Other patterns still reach the model as patterns.
                    self.assertEqual(
                        shown.count('"pattern"'), checked.count('"pattern"') - rules
                    )
        reason = Tool(
            _bind_tool(tools["record_reference_choice"]), name="record_reference_choice"
        ).function_schema.validator
        with self.assertRaises(ValidationError):
            reason.validate_python({"candidate_id": "x", "reason": " padded"})

    def test_every_planning_tool_declares_stages_purpose_examples_and_preconditions(
        self,
    ):
        from bio_neb.agentic_journey.role_instructions import role_instructions

        stages = declared_role("planning_agent").stages
        self.assertTrue(stages)
        # The purpose is the role's own instruction block.
        selected = role_instructions("planning_agent")
        for contract in function_tool_contracts("planning_agent"):
            with self.subTest(tool=contract.tool_name):
                self.assertTrue(selected.tool_purpose(contract.tool_name).strip())
                if contract.tool_name == HISTORY:
                    # The one history Tool is in every stage (0030 Retained context).
                    self.assertEqual(contract.stages, ())
                    continue
                self.assertTrue(contract.stages)
                self.assertLessEqual(set(contract.stages), set(stages))
                self.assertTrue(contract.examples)
                self.assertTrue(contract.preconditions)
                for precondition in contract.preconditions:
                    self.assertTrue(precondition.code.strip())
                    self.assertTrue(precondition.slot.strip())
                    self.assertTrue(precondition.explanation.strip())

    def test_every_planning_refusal_code_is_declared_with_its_explanation(self):
        # The assembly explains a code from the refused Tool's row, else from the
        # role row declaring it; a code no row declares would reach the model bare.
        import re

        root = Path(__file__).resolve().parents[1] / "src" / "bio_neb"
        raised = set()
        for module in (
            "agentic_journey/planning_assignment.py",
            "reference_discovery/reference_tools.py",
            "parabricks/execution_planning_action_policy.py",
            "parabricks/proposal_information.py",
            "agentic_journey/assignment_context.py",
        ):
            text = (root / module).read_text(encoding="utf-8")
            raised |= set(re.findall(r'PreconditionUnmet\(\s*"([a-z_]+)"', text))
            # The companion action names its code through a table.
            raised |= set(re.findall(r'": "(companion_[a-z_]+)"', text))
            # Domain validators return their codes; keyed ones end ":<code>".
            raised |= set(re.findall(r'return \(\s*"([a-z_]+)",\s*\)', text))
            raised |= set(
                re.findall(r'errors\.append\(f?"(?:\{[^}]*\}:)?([a-z_]+)"', text)
            )
        raised -= {"tool_not_eligible"}  # no planning Tool reaches it
        declared = {
            precondition.code
            for contract in function_tool_contracts("planning_agent")
            for precondition in contract.preconditions
        }
        self.assertGreater(len(raised), 45)
        self.assertEqual(raised - declared, set())

    def test_the_researcher_question_tool_is_in_every_planning_stage(self):
        for stage in declared_role("planning_agent").stages:
            with self.subTest(stage=stage):
                self.assertIn(
                    "request_planning_hitl", stage_tool_names("planning_agent", stage)
                )

    def test_preconditions_name_only_their_role_s_tools_to_order_a_reply(self):
        from bio_neb.agentic_journey.tool_execution import dependency_order

        for role in ("planning_agent", "investigation_planner"):
            names = {c.tool_name for c in function_tool_contracts(role)}
            for contract in function_tool_contracts(role):
                for precondition in contract.preconditions:
                    with self.subTest(role=role, code=precondition.code):
                        self.assertLessEqual(
                            {*precondition.met_by, *precondition.unmet_by}, names
                        )
        # A recorded reply in the order sent: the sample is attested before the
        # taxon lookup, the taxon is resolved before a choice stands, and the
        # supplied choice is recorded before it is inspected.
        sent = [
            ("resolve_taxon", "1"),
            ("record_supplied_reference", "2"),
            ("inspect_reference", "3"),
            ("attest_sample", "5"),
            ("search_journey", "6"),
        ]
        self.assertEqual(
            dependency_order("planning_agent", sent), ("5", "1", "2", "3", "6")
        )
        # A question opens after the reply's other calls; with no edge the sent
        # order holds.
        self.assertEqual(
            dependency_order(
                "planning_agent",
                [("request_planning_hitl", "q"), ("attest_sample", "a")],
            ),
            ("a", "q"),
        )
        self.assertEqual(
            dependency_order(
                "planning_agent", [("search_journey", "s"), ("attest_sample", "a")]
            ),
            ("s", "a"),
        )

    def test_each_stage_s_instructions_name_only_the_tools_it_offers(self):
        from bio_neb.agentic_journey.role_instructions import (
            role_instructions,
            stage_text,
        )

        text = role_instructions("planning_agent").text
        tools = {c.tool_name for c in function_tool_contracts("planning_agent")}
        self.assertIn("Prefer one call per reply.", text)
        self.assertNotIn("in the order sent", text)
        for stage in declared_role("planning_agent").stages:
            with self.subTest(stage=stage):
                named = {tool for tool in tools if tool in stage_text(text, stage)}
                self.assertLessEqual(
                    named, stage_tool_names("planning_agent", stage) | {HISTORY}
                )

    def test_each_stage_has_its_own_tool_set(self):
        # A recorded request's Tool set names its stage.
        sets = [
            stage_tool_names("planning_agent", stage)
            for stage in declared_role("planning_agent").stages
        ]
        self.assertEqual(len(sets), len(set(sets)))

    def test_investigation_has_one_stage_with_every_source_tool(self):
        (stage,) = declared_role("investigation_planner").stages
        self.assertEqual(
            stage_tool_names("investigation_planner", stage),
            frozenset(
                c.tool_name
                for c in function_tool_contracts("investigation_planner")
                if c.tool_name != HISTORY
            ),
        )

    def test_the_attempt_view_reads_the_recorded_assignment_not_a_role_name(self):
        from bio_neb.agentic_journey import turn_reopen
        from bio_neb.agentic_journey.agent_roles import AttemptReopen
        from bio_neb.application import queries

        for projector in (
            queries._project_one_attempt_evidence,
            turn_reopen.validate_action_turns,
            turn_reopen._validate_admitted_action,
            turn_reopen._prior_action_fingerprints,
        ):
            with self.subTest(projector=projector.__name__):
                self.assertNotIn('"planning_agent"', inspect.getsource(projector))
        reopen = planning_assignment.planning_assignment().reopen
        self.assertIsInstance(reopen, AttemptReopen)
        self.assertIs(reopen.admission, planning_assignment.PLANNING_ADMISSION)
        self.assertTrue(reopen.compiled_checkpoint)

    def test_the_shared_assembly_takes_no_offer(self):
        from bio_neb.agentic_journey.invocation import assemble_agent

        self.assertNotIn("offer", inspect.signature(assemble_agent).parameters)
        self.assertFalse(hasattr(planning_assignment, "_offer"))


class EmptyChoiceTests(unittest.TestCase):
    """A binder with no admissible value raises its declared precondition, so
    its Tool waits with that reason and is never offered an empty choice (#478,
    sweep Pattern C rule 1)."""

    def _waits(self, role, tool, bind, code):
        from bio_neb.agentic_journey.action_policy import PreconditionUnmet

        with self.assertRaises(PreconditionUnmet) as unmet:
            bind()
        self.assertEqual(unmet.exception.code, code)
        (row,) = [c for c in function_tool_contracts(role) if c.tool_name == tool]
        self.assertIn(code, {item.code for item in row.preconditions})

    def test_a_member_with_every_word_read_has_no_target_reading_to_offer(self):
        from bio_neb.investigation import investigation_agent

        with patch.object(investigation_agent, "_unread_words", return_value={}):
            self._waits(
                "investigation_planner",
                "record_target_reading",
                lambda: investigation_agent.investigation_tool_values(
                    SimpleNamespace(), "record_target_reading"
                ),
                "target_words_all_read",
            )

    def test_an_open_discovery_with_no_page_has_no_choice_to_offer(self):
        from bio_neb.reference_discovery import reference_tools

        view = SimpleNamespace(status="reference_candidates", discovery_event_ids=())
        with (
            patch.object(reference_tools, "current_choices", return_value=[]),
            patch.object(
                reference_tools, "applicable_reference_receipt", return_value=None
            ),
            patch.object(reference_tools, "answered_choice_receipt", return_value=None),
        ):
            self._waits(
                "planning_agent",
                "record_reference_choice",
                lambda: reference_tools._choice_values(SimpleNamespace(), view),
                "reference_discovery_not_open",
            )

    def test_no_researcher_statement_has_no_supplied_reference_to_offer(self):
        from bio_neb.reference_discovery import reference_tools

        view = SimpleNamespace(description_event_id=None)
        with (
            patch.object(reference_tools, "current_choices", return_value=[]),
            patch.object(reference_tools, "answered_choice_receipt", return_value=None),
            patch.object(reference_tools, "planning_evidence", return_value=[]),
            patch.object(
                reference_tools,
                "reference_requirement",
                return_value=SimpleNamespace(requirement_id="reference"),
            ),
            # No cited paper's reading keeps a version to cite either.
            patch.object(reference_tools, "_paper_versions", return_value=()),
        ):
            self._waits(
                "planning_agent",
                "record_supplied_reference",
                lambda: reference_tools._supplied_values(
                    SimpleNamespace(journey=None), view
                ),
                "researcher_statement_missing",
            )

    def test_a_run_with_no_organism_record_has_no_ploidy_citation_to_offer(self):
        from bio_neb.parabricks import execution_planning_action_policy as policy

        step = SimpleNamespace(
            work=SimpleNamespace(work_item_id="w"),
            basis_event_id="b",
            disposition=None,
            compatible=[{"candidate_id": "c"}],
        )
        with (
            patch.object(policy, "_recipe_step", return_value=step),
            patch.object(policy, "_organism_records", return_value={}),
        ):
            self._waits(
                "planning_agent",
                "select_recipe_candidate",
                lambda: policy._select_recipe_values(SimpleNamespace(), None),
                "organism_record_missing",
            )


class DefaultLoopTests(unittest.TestCase):
    def test_every_request_renders_every_tool_of_one_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _composition = _planning_ready(Path(directory))
            requests = _requests(receipt.journey_file)
        stage_sets = {
            stage: stage_tool_names("planning_agent", stage) | {HISTORY}
            for stage in declared_role("planning_agent").stages
        }
        seen = set()
        self.assertGreater(len(requests), 1)
        for number, request in enumerate(requests, start=1):
            with self.subTest(request=number):
                offered = frozenset(tool["name"] for tool in request["function_tools"])
                stage = next(
                    (name for name, tools in stage_sets.items() if tools == offered),
                    None,
                )
                self.assertIsNotNone(stage, sorted(offered))
                seen.add(stage)
                self.assertIn("request_planning_hitl", offered)
                self.assertEqual(request["model_settings"].get("tool_choice"), "auto")
        self.assertGreater(len(seen), 1)

    def test_every_offered_function_tool_reaches_the_model_through_its_projection(self):
        # The selected model's endpoint policy projects each Tool definition the
        # model is shown; main's projection is identity, so a stand-in
        # projection proves the assembly applies it to every offered Tool.
        from dataclasses import replace
        from unittest.mock import patch

        from bio_neb.agentic_journey.model_settings import ModelEndpointPolicy

        def project(_policy, definition):
            return replace(definition, description=definition.description + " [P]")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(ModelEndpointPolicy, "project_tool_schema", project),
        ):
            receipt, _composition = _planning_ready(Path(directory))
            requests = _requests(receipt.journey_file)
        self.assertGreater(len(requests), 1)
        for number, request in enumerate(requests, start=1):
            with self.subTest(request=number):
                self.assertTrue(
                    all(
                        tool["description"].endswith(" [P]")
                        for tool in request["function_tools"]
                    ),
                    [tool["name"] for tool in request["function_tools"]],
                )

    def test_a_fenced_nemotron_call_keeps_its_fence_fact_through_the_reasoning_split(
        self,
    ):
        # The adapter moves a fenced call's reasoning out and the call stays
        # fenced, so admission records the fence and reopen checks it against
        # the recorded call. The split runs at `last`.
        from dataclasses import replace
        from unittest.mock import patch

        from pydantic_ai.messages import ToolCallPart

        from bio_neb.agentic_journey import model_settings
        from bio_neb.agentic_journey.model_turn import read_arguments
        from bio_neb.application.queries import project_journey_events
        from tests.planning_assignment_fakes import PlanningComposition

        scripted = PlanningComposition.respond
        nemotron = "nvidia/Nemotron-3-Ultra-550b-a55b"
        at_last = replace(
            model_settings.NEMOTRON_ULTRA_ENDPOINT_POLICY, call_reasoning="last"
        )

        async def respond_fenced(composition, messages, info):
            response = await scripted(
                composition, messages, _as_the_domain_defines(info)
            )
            return replace(
                response,
                parts=[
                    replace(
                        part,
                        args="```json\n"
                        + json.dumps(
                            {
                                **part.args_as_dict(),
                                "reasoning": f"Why {part.tool_call_id}.",
                            }
                        )
                        + "\n```",
                    )
                    if isinstance(part, ToolCallPart)
                    else part
                    for part in response.parts
                ],
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(PlanningComposition, "respond", respond_fenced),
            model_settings.use_model_registry(
                model_settings.model_registry().replacing(at_last)
            ),
        ):
            receipt, composition = _planning_ready(Path(directory), model=nemotron)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            validations = [e for e in events if e["event_type"] == "ToolCallValidated"]
            self.assertGreater(len(validations), 1)
            for validation in validations:
                with self.subTest(call=validation["payload"]["tool_call_id"]):
                    self.assertIs(validation["payload"]["arguments_fenced"], True)
                    self.assertNotIn("reasoning", validation["payload"]["arguments"])
            # The recorded call stays fenced without its reasoning.
            for event in events:
                if event["event_type"] != "ModelResponseRecorded":
                    continue
                for part in event["payload"]["compatible_adapter_evidence"]["response"][
                    "message"
                ]["parts"]:
                    if part["part_kind"] == "tool-call":
                        read = read_arguments(part["args"])
                        self.assertTrue(read.fenced, part["args"])
                        self.assertNotIn("reasoning", read.value)
            # Replay sends each earlier call as read: plain JSON with its reasoning.
            replayed = [
                call["function"]["arguments"]
                for body in composition.model_endpoint.bodies
                for message in body["messages"]
                if message["role"] == "assistant"
                for call in message.get("tool_calls") or ()
            ]
            self.assertTrue(replayed)
            for arguments in replayed:
                self.assertNotIn("INVALID_JSON", arguments)
                self.assertEqual(list(json.loads(arguments))[-1], "reasoning")
            # Reopen checks the fence fact against the recorded call.
            hostile = copy.deepcopy(events)
            next(e for e in hostile if e["event_type"] == "ToolCallValidated")[
                "payload"
            ].pop("arguments_fenced")
            with self.assertRaisesRegex(
                ValueError, "observed Tool execution changed its exact model call"
            ):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_nemotron_calls_carry_no_call_reasoning_in_requests_or_records(self):
        # Nemotron is shown no reasoning argument, its records keep no
        # `call_reasoning` part, and its calls are admitted.
        with tempfile.TemporaryDirectory() as directory:
            receipt, composition = _planning_ready(
                Path(directory), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        requests = [
            e["payload"]["compatible_adapter_evidence"]["request"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ]
        self.assertGreater(len(requests), 1)
        for request in requests:
            for definition in request["function_tools"]:
                self.assertNotIn(
                    "reasoning",
                    definition["parameters_json_schema"].get("properties", {}),
                    definition["name"],
                )
        for body in composition.model_endpoint.bodies:
            self.assertEqual(body["tool_choice"], "auto")
            for item in body["tools"]:
                self.assertNotIn(
                    "reasoning", item["function"]["parameters"].get("properties", {})
                )
        parts = [
            part
            for e in events
            if e["event_type"] == "ModelResponseRecorded"
            for part in e["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
        ]
        self.assertEqual(
            sum(part["part_kind"] == "tool-call" for part in parts),
            len(composition.calls),
        )
        self.assertNotIn(
            "call_reasoning", {part.get("id") for part in parts if "id" in part}
        )
        self.assertGreater(
            sum(e["event_type"] == "ToolCallValidated" for e in events), 1
        )

    def test_the_last_placement_s_call_reasoning_never_reaches_admission(self):
        # At `last` the projection asks every call for a reasoning argument; the
        # adapter moves it to a recorded thinking part before admission.
        from dataclasses import replace
        from unittest.mock import patch

        from pydantic_ai.messages import ToolCallPart

        from bio_neb.agentic_journey import model_settings
        from tests.planning_assignment_fakes import PlanningComposition

        scripted = PlanningComposition.respond
        nemotron = "nvidia/Nemotron-3-Ultra-550b-a55b"
        at_last = replace(
            model_settings.NEMOTRON_ULTRA_ENDPOINT_POLICY, call_reasoning="last"
        )

        async def respond_with_reasoning(composition, messages, info):
            response = await scripted(
                composition, messages, _as_the_domain_defines(info)
            )
            return replace(
                response,
                parts=[
                    replace(
                        part,
                        args={
                            **part.args_as_dict(),
                            "reasoning": f"Why {part.tool_call_id}.",
                        },
                    )
                    if isinstance(part, ToolCallPart)
                    else part
                    for part in response.parts
                ],
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(PlanningComposition, "respond", respond_with_reasoning),
            model_settings.use_model_registry(
                model_settings.model_registry().replacing(at_last)
            ),
        ):
            receipt, composition = _planning_ready(Path(directory), model=nemotron)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        requests = [
            e["payload"]["compatible_adapter_evidence"]["request"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ]
        self.assertGreater(len(requests), 1)
        for request in requests:
            for definition in request["function_tools"]:
                schema = definition["parameters_json_schema"]
                self.assertEqual(
                    schema["required"][-1], "reasoning", definition["name"]
                )
                self.assertEqual(schema["properties"]["reasoning"]["type"], "string")
        # The Journey writes canonical JSON; the sent Tools keep the placement.
        for body in composition.model_endpoint.bodies:
            for item in body["tools"]:
                parameters = item["function"]["parameters"]
                self.assertEqual(
                    list(parameters["properties"])[-1], "reasoning", item["function"]
                )
        calls = 0
        for event in events:
            if event["event_type"] != "ModelResponseRecorded":
                continue
            parts = event["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
            for before, part in zip(parts, parts[1:], strict=False):
                if part["part_kind"] != "tool-call":
                    continue
                calls += 1
                self.assertNotIn("reasoning", json.loads(part["args"]))
                self.assertEqual(
                    (before["part_kind"], before["id"], before["content"]),
                    ("thinking", "call_reasoning", f"Why {part['tool_call_id']}."),
                )
                self.assertEqual(
                    before["provider_details"],
                    {
                        "tool_call_id": part["tool_call_id"],
                        "tool_name": part["tool_name"],
                    },
                )
        self.assertEqual(calls, len(composition.calls))

        # `reasoning` also names the typed request setting; settings are not calls.
        settings = {"model_settings", "settings_sources", "baseline"}

        def reasoning_keys(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in settings:
                        continue
                    if key == "reasoning":
                        yield path
                    yield from reasoning_keys(item, (*path, key))
            elif isinstance(value, list):
                for item in value:
                    yield from reasoning_keys(item, path)

        # Only the offered Tool schemas name it: no admission, Tool or domain
        # record carries a reasoning argument.
        self.assertEqual(
            {
                (event["event_type"], *path[:4])
                for event in events
                for path in reasoning_keys(event["payload"])
            },
            {
                (
                    "ModelRequestRecorded",
                    "compatible_adapter_evidence",
                    "request",
                    "function_tools",
                    "parameters_json_schema",
                )
            },
        )
        # Replay sends each earlier call with its reasoning, as a JSON string.
        replayed = [
            call["function"]["arguments"]
            for body in composition.model_endpoint.bodies
            for message in body["messages"]
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or ()
        ]
        self.assertTrue(replayed)
        for arguments in replayed:
            self.assertIsInstance(arguments, str)
            self.assertEqual(list(json.loads(arguments))[-1], "reasoning")

    def test_a_misspelled_nemotron_argument_is_admitted_with_both_forms_recorded(
        self,
    ):
        # Nemotron's adapter fixes an argument name that spells a shown one
        # differently before admission (ADR 0034). The Journey keeps the call as
        # sent in the recorded reply and as fixed in the admitted call.
        from dataclasses import replace
        from unittest.mock import patch

        from pydantic_ai.messages import ToolCallPart

        from tests.planning_assignment_fakes import PlanningComposition

        scripted = PlanningComposition.respond
        misspelled = []

        async def respond_misspelled_once(composition, messages, info):
            response = await scripted(
                composition, messages, _as_the_domain_defines(info)
            )
            parts = []
            for part in response.parts:
                if isinstance(part, ToolCallPart):
                    arguments = part.args_as_dict()
                    if not misspelled and "kinds" in arguments:
                        arguments["Kinds"] = arguments.pop("kinds")
                        misspelled.append(part.tool_call_id)
                    part = replace(part, args=arguments)
                parts.append(part)
            return replace(response, parts=parts)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(PlanningComposition, "respond", respond_misspelled_once),
        ):
            receipt, _composition = _planning_ready(
                Path(directory), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        (call_id,) = misspelled
        (validated,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == call_id
        ]
        self.assertEqual(validated["tool_name"], "search_journey")
        self.assertIn("kinds", validated["arguments"])
        self.assertNotIn("Kinds", validated["arguments"])
        (reply,) = [
            e["payload"]["compatible_adapter_evidence"]["response"]["message"]
            for e in events
            if e["event_type"] == "ModelResponseRecorded"
            and any(
                part.get("tool_call_id") == call_id
                for part in e["payload"]["compatible_adapter_evidence"]["response"][
                    "message"
                ]["parts"]
            )
        ]
        reading = reply["provider_details"]["reply_reading"]
        self.assertEqual(reading["quirks"], ["argument_name"])
        (sent,) = reading["raw"]["tool_calls"]
        self.assertEqual(sent["id"], call_id)
        self.assertEqual(
            set(json.loads(sent["function"]["arguments"])),
            {"Kinds", "limit"},
        )

    def test_two_spellings_of_one_nemotron_call_collapse_as_one_copy(self):
        # The name shim runs in the adapter before routing: once the names are
        # fixed, the two calls are one call sent twice, so routing keeps the
        # first and names the other a copy. The raw reply keeps both.
        from dataclasses import replace
        from unittest.mock import patch

        from pydantic_ai.messages import ToolCallPart

        from tests.planning_assignment_fakes import PlanningComposition

        scripted = PlanningComposition.respond
        doubled = []

        async def respond_doubled_once(composition, messages, info):
            response = await scripted(
                composition, messages, _as_the_domain_defines(info)
            )
            parts = []
            for part in response.parts:
                if isinstance(part, ToolCallPart):
                    arguments = part.args_as_dict()
                    if not doubled and "kinds" in arguments:
                        first = f"{part.tool_call_id}-spelled"
                        misspelled = dict(arguments)
                        misspelled["Kinds"] = misspelled.pop("kinds")
                        parts.append(replace(part, args=misspelled, tool_call_id=first))
                        doubled.extend([first, part.tool_call_id])
                    part = replace(part, args=arguments)
                parts.append(part)
            return replace(response, parts=parts)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(PlanningComposition, "respond", respond_doubled_once),
        ):
            receipt, _composition = _planning_ready(
                Path(directory), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        first, copy = doubled
        (routed,) = [
            e["payload"] for e in events if e["event_type"] == "AgentTurnRouted"
        ]
        self.assertEqual(routed["copies"], {copy: first})
        self.assertIn(first, routed["run_order"])
        self.assertEqual(
            [(f["key"], f["count"]) for f in routed["faults"]], [("copied_calls", 1)]
        )
        (validated,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] in doubled
        ]
        self.assertEqual(validated["tool_call_id"], first)
        self.assertIn("kinds", validated["arguments"])
        (reply,) = [
            e["payload"]["compatible_adapter_evidence"]["response"]["message"]
            for e in events
            if e["event_type"] == "ModelResponseRecorded"
            and e["event_id"] == routed["response_event_id"]
        ]
        reading = reply["provider_details"]["reply_reading"]
        self.assertEqual(reading["quirks"], ["argument_name"])
        self.assertEqual(
            [
                sorted(json.loads(call["function"]["arguments"]))
                for call in reading["raw"]["tool_calls"]
                if call["id"] in doubled
            ],
            [["Kinds", "limit"], ["kinds", "limit"]],
        )

    def test_a_text_form_call_with_malformed_arguments_is_refused_by_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            composition = _TextFormBrokenCall(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            refused, turns, started = _refusals(receipt.journey_file, HISTORY)
        # The Nemotron adapter made it a real call; admission refused it as it
        # refuses any malformed call, and the loop went on.
        self.assertEqual(
            [(v["errors"], v["arguments"]) for v in refused],
            [(["tool_arguments_invalid_json"], {"invalid_json": "{broken"})],
        )
        self.assertEqual(len(turns), 1)
        self.assertIn("tool_arguments_invalid_json", turns[0]["feedback"])
        self.assertEqual(started, set())
        self.assertGreater(len(composition.calls), 1)

    def test_fenced_call_arguments_are_admitted_as_written_and_recorded(self):
        # Fenced JSON is stripped, accepted and recorded on every path; the same
        # scripted Run with plain arguments is the reference.
        def run(composition_type):
            with tempfile.TemporaryDirectory() as directory:
                receipt, paths = _registered(Path(directory))
                composition = composition_type(receipt, paths)
                with use_run_advancement_composition(composition):
                    status = advance(ResumeRun(journey_file=receipt.journey_file))
                events = _events(receipt.journey_file)
                # The Run reads back: reopen re-reads every recorded call.
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            validations = {
                e["event_id"]: e["payload"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
            }
            started = {
                e["payload"]["validation_event_id"]
                for e in events
                if e["event_type"] == "ObservedToolExecutionStarted"
            }
            return composition, status, validations, started

        _, plain_status, plain, _ = run(PlanningComposition)
        composition, status, fenced, started = run(_FencedCalls)
        # Replay shows the model each admitted call as read, never as invalid.
        replayed = [
            (call["id"], call["function"]["arguments"])
            for body in composition.model_endpoint.bodies
            for message in body["messages"]
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or ()
        ]
        self.assertTrue(replayed)
        for call_id, arguments in replayed:
            with self.subTest(replayed=call_id):
                self.assertEqual(json.loads(arguments), composition.sent[call_id])
        self.assertEqual(status.status, plain_status.status)
        self.assertGreater(len(fenced), 1)
        self.assertEqual(
            [(v["tool_name"], v["status"]) for v in fenced.values()],
            [(v["tool_name"], v["status"]) for v in plain.values()],
        )
        for event_id, validation in fenced.items():
            with self.subTest(call=validation["tool_call_id"]):
                self.assertEqual(validation["status"], "valid")
                sent = composition.sent[validation["tool_call_id"]]
                self.assertEqual({k: validation["arguments"][k] for k in sent}, sent)
                self.assertIs(validation["arguments_fenced"], True)
                self.assertIn(event_id, started)
        self.assertTrue(all("arguments_fenced" not in v for v in plain.values()))

    def test_a_search_result_shows_each_control_character_as_one_space(self):
        # A raw tab copied from a result into a call's JSON string made the
        # endpoint drop the call. The model is shown each control character but
        # the line break as one space; the receipt stays as read.
        prompt = (
            "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.\n"
            "Table:\tgene\tvalue\r\x0bend\x7f."
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), prompt_text=prompt)
            # The scripted model's first call reads the request by a search.
            composition = PlanningComposition(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
            # Reopen re-derives the digest of the view the model was shown.
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        # The first Tool result the model receives is that search's.
        shown = next(
            json.loads(message["content"])
            for body in composition.model_endpoint.bodies
            for message in body["messages"]
            if message["role"] == "tool"
        )
        (request,) = [
            event["payload"]["prompt"]
            for event in shown["result"]["events"]
            if event["event_type"] == "ResearchRequestRegistered"
        ]
        self.assertEqual(
            request,
            "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.\n"
            "Table: gene value  end .",
        )
        (read,) = [e for e in events if e["event_type"] == "JourneySearchRead"]
        (recorded,) = [
            event["payload"]["prompt"]
            for event in read["payload"]["result"]["events"]
            if event["event_type"] == "ResearchRequestRegistered"
        ]
        self.assertEqual(recorded, prompt)
        (succeeded,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
            and e["payload"]["stage_result_event_id"] == read["event_id"]
        ]
        self.assertIs(succeeded["result_guarded"], True)
        self.assertEqual(
            succeeded["model_visible_result_digest"], canonical_json_digest(shown)
        )
        # Every other result held no control character and records no flag.
        self.assertTrue(
            all(
                "result_guarded" not in e["payload"]
                for e in events
                if e["event_type"] == "ObservedToolExecutionSucceeded"
                and e["payload"]["stage_result_event_id"] != read["event_id"]
            )
        )
        # Reopen re-derives the shown view only where the record says the
        # guard changed it, and refuses either flag the view contradicts.
        for label, change, refusal in (
            ("flag dropped", "drop", "canonical result digest"),
            ("flag on an unchanged result", "add", "guard that changed nothing"),
        ):
            changed = copy.deepcopy(events)
            for event in changed:
                if event["event_type"] != "ObservedToolExecutionSucceeded":
                    continue
                guarded = event["payload"]["stage_result_event_id"] == read["event_id"]
                if change == "drop" and guarded:
                    del event["payload"]["result_guarded"]
                elif change == "add" and not guarded:
                    event["payload"]["result_guarded"] = True
            with (
                self.subTest(label),
                self.assertRaisesRegex(ValueError, refusal),
            ):
                project_journey_events(changed, journey_file=receipt.journey_file)
        # A search recorded before the guard showed the model its raw view,
        # digested raw and carrying no flag; it reopens as it was.
        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        earlier = copy.deepcopy(events)
        raw = tool_model_view(model_view_of("search_journey"), read)
        for event in earlier:
            payload = event["payload"]
            if payload.get("stage_result_event_id") == read["event_id"]:
                del payload["result_guarded"]
                payload["model_visible_result_digest"] = canonical_json_digest(raw)
            for message in (
                payload.get("compatible_adapter_evidence", {})
                .get("request", {})
                .get("messages", ())
            ):
                for part in message.get("parts", ()):
                    if (
                        part.get("part_kind") == "tool-return"
                        and part["content"] == shown
                    ):
                        part["content"] = raw
        project_journey_events(earlier, journey_file=receipt.journey_file)

    def test_no_tool_result_shows_the_model_a_control_character(self):
        # One guard covers every function Tool's result. The assembly source
        # names its assembly with a tab, so a domain Tool's own result
        # (``discover_assemblies``) carries one; its recorded fact keeps it.
        control = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")

        class TabbedAssemblies(ReferenceAdapter):
            def _assembly(self, accession, ncbi_tax_id, strain):
                record = super()._assembly(accession, ncbi_tax_id, strain)
                return record.model_copy(update={"assembly_name": "GRCh38\t.p14"})

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = TabbedAssemblies()
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        results = {
            message["tool_call_id"]: json.loads(message["content"])
            for body in composition.model_endpoint.bodies
            for message in body["messages"]
            if message["role"] == "tool"
        }
        calls = {
            e["payload"]["tool_call_id"]: e["payload"]["tool_name"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
        }
        self.assertIn("discover_assemblies", {calls[i] for i in results})
        for call_id, result in results.items():
            with self.subTest(tool=calls[call_id]):
                self.assertEqual(
                    [text for text in strings(result) if control.search(text)], []
                )
        (discovered,) = [
            result
            for call_id, result in results.items()
            if calls[call_id] == "discover_assemblies"
        ]
        self.assertIn('"GRCh38 .p14"', json.dumps(discovered))
        (recorded,) = [
            e for e in events if e["event_type"] == "ReferenceAssembliesRecorded"
        ]
        self.assertIn('"GRCh38\\t.p14"', json.dumps(recorded["payload"]))

    def test_each_request_carries_its_stage_settings_with_their_sources(self):
        # One merge: model default, global, role, then stage. A stage with
        # settings of its own changes only its requests; each request records its
        # stage, and the Run reads back with every request bound to that entry.
        invocation = {
            "model_settings": {"temperature": 0.5},
            "scopes": {
                "role": {"planning_agent": {"model_settings": {"top_p": 0.9}}},
                "stage": {
                    "reference": {
                        "model_settings": {"temperature": 0.2, "max_tokens": None}
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            receipt, _composition = _planning_ready(
                Path(directory), agent_invocation=invocation
            )
            events = _events(receipt.journey_file)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        started = [
            e["payload"]["effective_invocation_policy"]
            for e in events
            if e["event_type"] == "AttemptStarted"
        ]
        self.assertTrue(started)
        for policy in started:
            self.assertEqual(
                policy["settings_sources"],
                {
                    "parallel_tool_calls": "model_default",
                    "max_tokens": "model_default",
                    "temperature": "global",
                    "top_p": "role",
                },
            )
            self.assertEqual(
                policy["stage_settings"]["reference"]["settings_sources"],
                {
                    "parallel_tool_calls": "model_default",
                    "max_tokens": "stage",
                    "temperature": "stage",
                    "top_p": "role",
                    "tool_choice": "global",
                },
            )
        stage_sets = {
            stage: stage_tool_names("planning_agent", stage) | {HISTORY}
            for stage in declared_role("planning_agent").stages
        }
        role_values = {
            "parallel_tool_calls": False,
            "max_tokens": 245760,
            "temperature": 0.5,
            "top_p": 0.9,
            "tool_choice": "auto",
        }
        reference_values = {
            "parallel_tool_calls": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "tool_choice": "auto",
        }
        stages = []
        for event in events:
            if event["event_type"] != "ModelRequestRecorded":
                continue
            payload = event["payload"]
            request = payload["compatible_adapter_evidence"]["request"]
            offered = frozenset(tool["name"] for tool in request["function_tools"])
            with self.subTest(request=payload["request_number"]):
                self.assertEqual(stage_sets[payload["stage"]], offered)
                self.assertEqual(
                    request["model_settings"],
                    reference_values
                    if payload["stage"] == "reference"
                    else role_values,
                )
            stages.append(payload["stage"])
        self.assertIn("reference", stages)
        self.assertGreater(len(set(stages)), 1)

    def test_a_configured_stage_tool_choice_is_what_its_requests_send(self):
        # One Tool-choice policy: a stage's configured choice replaces the
        # Run's one recorded choice for its requests only; the adapter renders
        # it and the Run reads back with every request bound to its stage's
        # choice.
        invocation = {"scopes": {"stage": {"reference": {"tool_choice": "auto"}}}}
        with tempfile.TemporaryDirectory() as directory:
            receipt, composition = _planning_ready(
                Path(directory), agent_invocation=invocation
            )
            events = _events(receipt.journey_file)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        policies = [
            e["payload"]["effective_invocation_policy"]
            for e in events
            if e["event_type"] == "AttemptStarted"
        ]
        self.assertTrue(policies)
        for policy in policies:
            self.assertEqual(
                {
                    stage: (
                        entry["tool_choice"],
                        entry["settings_sources"]["tool_choice"],
                    )
                    for stage, entry in policy["stage_settings"].items()
                },
                {
                    stage: ("auto", "stage" if stage == "reference" else "global")
                    for stage in declared_role("planning_agent").stages
                },
            )
        requests = [
            e["payload"] for e in events if e["event_type"] == "ModelRequestRecorded"
        ]
        bodies = composition.model_endpoint.bodies
        self.assertEqual(len(bodies), len(requests))
        sent = [
            (
                request["stage"],
                request["compatible_adapter_evidence"]["request"]["model_settings"][
                    "tool_choice"
                ],
                body["tool_choice"],
            )
            for request, body in zip(requests, bodies, strict=True)
        ]
        self.assertIn(("reference", "auto", "auto"), sent)
        self.assertEqual(
            {(stage, recorded, wire) for stage, recorded, wire in sent},
            {(stage, "auto", "auto") for stage, _, _ in sent},
        )
        self.assertGreater(len({stage for stage, _, _ in sent}), 1)

    def test_a_premature_call_is_refused_with_tool_slot_value_and_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _PrematureChoice(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        self.assertEqual(result.status, "agent_work_review_pending")
        validations = {
            e["event_id"]: e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "record_reference_choice"
        }
        self.assertEqual(len(validations), 4)
        self.assertEqual(
            {tuple(v["errors"]) for v in validations.values()},
            {("reference_discovery_not_open",)},
        )
        turns = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and set(e["payload"]["validation_event_ids"]) & set(validations)
        ]
        # Each precondition refusal is a failed turn in a row; the fourth stops.
        # The key's call identity follows its reasons.
        self.assertEqual(
            [
                (t["correction_key"].split(" @ ")[0], t["correction_count"])
                for t in turns
            ],
            [
                ("record_reference_choice: reference_discovery_not_open", count)
                for count in range(1, 5)
            ],
        )
        feedback = turns[0]["feedback"]
        self.assertIn("`record_reference_choice` was not run", feedback)
        # No discovery has run, so the slot has no value to show (it was the
        # internal reference status).
        self.assertIn("`reference discovery` has no value", feedback)
        self.assertIn("discover the candidate references first", feedback)
        self.assertNotIn("not offered now", feedback)

    def test_an_attempt_ends_at_its_pending_researcher_question(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        self.assertEqual(result.status, "planning_hitl_pending")
        asked = next(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "PlanningHitlRequested"
        )
        later = [e["event_type"] for e in events[asked:]]
        self.assertNotIn("ModelRequestRecorded", later)
        self.assertIn("AttemptSucceeded", later)

    def test_an_absent_slot_is_named_as_having_no_value(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _PrematureQuestion(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(
                receipt.journey_file, "request_planning_hitl"
            )
        self.assertEqual(validations[0]["errors"], ["reference_work_not_open"])
        self.assertEqual(started, set())
        self.assertIn(
            "Tool `request_planning_hitl` was not run: it waits on its step. Call an "
            "allowed Tool now.\n- `proposal plan` has no value: Reference work opens "
            "once the proposal requirements are recorded.",
            turns[0]["feedback"],
        )

    def test_a_domain_grounding_refusal_names_the_item_its_value_and_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _UngroundedQuote(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(
                receipt.journey_file, "record_proposal_requirements"
            )
        key = composition.quoted
        self.assertEqual(
            validations[0]["errors"], [f"{key}:request_assertion_ungrounded"]
        )
        self.assertEqual(started, set())
        feedback = turns[0]["feedback"]
        self.assertIn(f"`knowledge` of `{key}` is {{", feedback)
        self.assertIn('"quote": "words nobody wrote"', feedback)
        self.assertIn(
            "Its quote must copy words of the research request exactly", feedback
        )

    def test_a_typed_value_its_quote_does_not_state_is_refused_in_plain_words(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ValueNotInQuote(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(
                receipt.journey_file, "record_proposal_requirements"
            )
        key = composition.keyed
        self.assertEqual(
            validations[0]["errors"], [f"{key}:asserted_value_not_in_quote"]
        )
        self.assertEqual(started, set())
        feedback = turns[0]["feedback"]
        self.assertIn(f"`knowledge` of `{key}` is {{", feedback)
        self.assertIn('"value": "sample-002"', feedback)
        self.assertIn("Its typed value must be one its quote states", feedback)

    def test_a_placeholder_rsid_on_a_coordinate_only_site_is_refused_naming_both_forms(
        self,
    ):
        # The R240 shape: a site the request gives only by coordinates, sent
        # with an invented rsID.
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text="Analyze rs1234 for sample-001 Homo sapiens and produce VCF. "
                "Does it carry V:2129317 C>T?",
            )
            composition = PlanningComposition(receipt, paths)
            composition.scientific_intent = "V:2129317 C>T"
            composition.scientific_rsid = "rs429358"
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(
                receipt.journey_file, "record_proposal_requirements"
            )
        self.assertEqual(
            validations[0]["errors"],
            ["post_execution_variant:asserted_value_not_in_quote"],
        )
        self.assertEqual(started, set())
        feedback = turns[0]["feedback"]
        self.assertIn("`knowledge` of `post_execution_variant` is {", feedback)
        self.assertIn('"value": "rs429358"', feedback)
        self.assertIn("Its typed value must be one its quote states", feedback)
        # The site is typed from its quote.
        self.assertIn(
            '{"state": "asserted", "quote": "V:2129317 C>T", '
            '"site": {"contig": "V", "position": 2129317, "ref": "C", "alt": "T"}}',
            feedback,
        )
        self.assertIn(
            "an rsID the request does not give is to_find, never a placeholder",
            feedback,
        )
        self.assertNotIn("leave the value out", feedback)

    def test_the_offered_requirements_tool_shows_a_coordinate_only_site_and_no_placeholder(
        self,
    ):
        # The offered Tool teaches both forms in its example and in the
        # investigation value's description, as the model received them.
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            with use_run_advancement_composition(PlanningComposition(receipt, paths)):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        offered = [
            tool
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
            for tool in e["payload"]["compatible_adapter_evidence"]["request"][
                "function_tools"
            ]
            if tool["name"] == "record_proposal_requirements"
        ]
        self.assertTrue(offered)
        for tool in offered:
            description = tool["description"]
            self.assertIn(
                '"knowledge": {"state": "asserted", "quote": "V:2129317 C>T", '
                '"site": {"contig": "V", "position": 2129317, "ref": "C", "alt": "T"}}',
                description,
            )
            self.assertIn(
                "an rsID the request does not give is to_find, never a placeholder",
                description,
            )
            fields = tool["parameters_json_schema"]["$defs"][
                "AssertedInvestigationDraft"
            ]["properties"]
            value = fields["value"]["description"]
            # A coordinate-only site is typed in its own field.
            site = fields["site"]["description"]
            self.assertIn("V:2129317 C>T", site)
            self.assertIn("The server checks that the quote holds each of them.", site)
            self.assertIn(
                "an rsID the request does not give is to_find, never a placeholder",
                value,
            )
            for text in (description, value, site):
                self.assertNotIn("leave the value out", text)
                self.assertNotIn("omit", text.lower())

    def test_a_refused_whole_call_shows_what_was_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ForeignHistoryRead(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(receipt.journey_file, HISTORY)
        self.assertEqual(validations[0]["errors"], ["search_outside_run"])
        self.assertEqual(started, set())
        self.assertIn(
            '`query` is {"event_ids": ["foreign-event"]}: The search must name events',
            turns[0]["feedback"],
        )

    def test_a_slice_read_with_a_page_limit_is_refused_for_its_own_reason(self):
        # The event is this Run's own: the refusal names the limit beside the
        # slice, never "outside this Run".
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _SliceWithPageLimit(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(receipt.journey_file, HISTORY)
        self.assertEqual(validations[0]["errors"], ["search_slice_with_page_filter"])
        self.assertEqual(started, set())
        feedback = turns[0]["feedback"]
        self.assertIn("search_slice_with_page_filter", feedback)
        # The slot is the slice, and the sentence names the filter sent beside
        # it.
        self.assertIn("`payload_slice` is {", feedback)
        self.assertIn(
            "A payload slice read takes no other filter; you also sent `limit`. "
            "Send payload_slice alone.",
            feedback,
        )
        self.assertNotIn("inside this Run's retained Journey", feedback)

    def test_planning_lacking_only_preparation_information_closes_before_contact(
        self,
    ):
        from bio_neb.application.planning_read import (
            current_planning,
            planning_evidence,
        )
        from bio_neb.parabricks.planning_readiness import readiness_of

        with tempfile.TemporaryDirectory() as directory:
            receipt, _composition = _planning_ready(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            gap = readiness_of(current_planning(journey), planning_evidence(journey))
            closes = planning_assignment.planning_closes_before_contact(journey)
        # Only the researcher can supply launch settings: the coordinator asks
        # through the preparation Review, so no Attempt asks the model.
        self.assertEqual(gap.code, "execution_configuration_required")
        self.assertTrue(closes)
        source = inspect.getsource(planning_assignment)
        self.assertNotIn("proposal_output_question_ineligible", source)


class OrderedCallsTests(unittest.TestCase):
    def test_two_calls_in_one_response_run_in_order_with_their_own_receipts(self):
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                agent_invocation={"model_settings": {"parallel_tool_calls": True}},
            )
            composition = _HistoryThenPlan(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
            # Reopen reconstructs the ordered calls from the retained records.
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            self.assertTrue(
                all(
                    request["model_settings"]["parallel_tool_calls"] is True
                    for request in _requests(receipt.journey_file)
                )
            )
        validations = [e for e in events if e["event_type"] == "ToolCallValidated"][:2]
        self.assertEqual(
            [
                (v["payload"]["tool_call_id"], v["payload"]["status"])
                for v in validations
            ],
            [("pair-1", "valid"), ("pair-2", "valid")],
        )
        turns = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
        ][:2]
        self.assertEqual(
            [(t["disposition"], t["validation_event_ids"]) for t in turns],
            [("multi_call_admitted", [v["event_id"]]) for v in validations],
        )
        started = [
            e["payload"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionStarted"
        ][:2]
        self.assertEqual(
            [s["validation_event_id"] for s in started],
            [v["event_id"] for v in validations],
        )
        succeeded = [
            e["payload"]["started_event_id"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
        ]
        starts = [
            e["event_id"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionStarted"
        ][:2]
        self.assertEqual(succeeded[:2], starts)
        self.assertNotIn(
            "AgentActionReviewRequested", {e["event_type"] for e in events}
        )

    def test_a_call_after_a_question_opening_call_is_refused_with_its_reason(self):
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _QuestionThenRead(receipt, paths)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            validations, turns, started = _refusals(receipt.journey_file, HISTORY)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertEqual(validations[-1]["tool_call_id"], "after-question")
        self.assertEqual(validations[-1]["errors"], ["researcher_question_opened"])
        self.assertEqual(started, set())
        self.assertEqual(turns[-1]["disposition"], "multi_call_refused")
        self.assertIn(
            f'`researcher question` is "{composition.question}": An earlier call',
            turns[-1]["feedback"],
        )
        self.assertNotIn(
            "AgentActionReviewRequested", {e["event_type"] for e in events}
        )

    def _pair(self, call):
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ReadThenCall(receipt, paths, call)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            # Reopen checks each call's recorded decision and correction.
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
            requests = _requests(receipt.journey_file)
        validations = {
            e["payload"]["tool_call_id"]: e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] in {"read-first", "call-second"}
        }
        turns = {
            e["payload"]["validation_event_ids"][0]: e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
        }
        started = {
            e["payload"]["validation_event_id"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionStarted"
        }
        return (
            {
                call_id: (
                    validation["payload"]["errors"],
                    # An unoffered call's decision is its reply's routing.
                    turns.get(validation["event_id"]),
                    validation["event_id"] in started,
                )
                for call_id, validation in validations.items()
            },
            [part for message in requests[1]["messages"] for part in message["parts"]],
        )

    def _pair_journey(self, call, **registration):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), **registration)
            composition = _ReadThenCall(receipt, paths, call)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            return result, _events(receipt.journey_file), receipt.journey_file

    def test_a_refusal_after_an_earlier_call_ran_is_not_a_failed_turn(self):
        # A reply in which a call ran is progress, so its later refusal is
        # corrected even when one failed turn would stop the Attempt.
        from bio_neb.application.queries import project_journey_events

        result, events, journey_file = self._pair_journey(
            ("request_planning_hitl", {}),
            agent_invocation={"failed_turns": 1},
        )
        project_journey_events(events, journey_file=journey_file)
        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertFalse(
            any(e["event_type"] == "AgentWorkReviewRequested" for e in events)
        )
        validations = {
            e["payload"]["tool_call_id"]: e
            for e in events
            if e["event_type"] == "ToolCallValidated"
        }
        started = {
            e["payload"]["validation_event_id"]
            for e in events
            if e["event_type"] == "ObservedToolExecutionStarted"
        }
        # The earlier admitted call ran; the later call was refused and corrected.
        self.assertIn(validations["read-first"]["event_id"], started)
        self.assertNotIn(validations["call-second"]["event_id"], started)
        self.assertEqual(
            validations["call-second"]["payload"]["errors"],
            ["reference_work_not_open"],
        )
        (refused,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["validation_event_ids"]
            == [validations["call-second"]["event_id"]]
        ]
        self.assertEqual(
            (refused["disposition"], refused["correction_count"]),
            ("multi_call_refused", 1),
        )
        self.assertTrue(refused["feedback"])

    def _hostile(self, mutate, message):
        from bio_neb.application.queries import project_journey_events

        _result, events, journey_file = self._pair_journey(
            ("record_proposal_requirements", None)
        )
        project_journey_events(events, journey_file=journey_file)
        hostile = copy.deepcopy(events)
        mutate(hostile)
        with self.assertRaisesRegex(ValueError, message):
            project_journey_events(hostile, journey_file=journey_file)

    def test_reopen_refuses_a_batched_order_as_a_broken_causal_chain(self):
        def batch_admission(events):
            # Move the second call's validation and disposition before the first
            # call's execution, as a whole-response admission would record them.
            second = next(
                i
                for i, e in enumerate(events)
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "call-second"
            )
            moved = events[second : second + 2]
            del events[second : second + 2]
            first_turn = next(
                i
                for i, e in enumerate(events)
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "multi_call_admitted"
            )
            moved[0]["parent_ids"] = [events[first_turn]["event_id"]]
            events[first_turn + 1 : first_turn + 1] = moved

        # The Attempt's single causal chain already binds each call's turn.
        self._hostile(batch_admission, "causal Journey chain")

    def test_ordered_calls_must_each_run_before_the_next_is_admitted(self):
        from bio_neb.agentic_journey.turn_reopen import (
            _validate_ordered_calls,
        )
        from bio_neb.agentic_journey.turn_reopen import (
            compatible_model_request as _compatible_model_request,
        )

        _result, events, _journey_file = self._pair_journey(
            ("record_proposal_requirements", None)
        )
        response = next(
            e
            for e in events
            if e["event_type"] == "ModelResponseRecorded"
            and any(
                part.get("tool_call_id") == "call-second"
                for part in e["payload"]["compatible_adapter_evidence"]["response"][
                    "message"
                ]["parts"]
            )
        )
        attempt = [
            e
            for e in events
            if e["payload"].get("attempt_id") == response["payload"]["attempt_id"]
        ]
        request = next(
            e
            for e in attempt
            if e["event_type"] == "ModelRequestRecorded"
            and e["payload"]["request_number"] == response["payload"]["request_number"]
        )
        later = [
            e
            for e in attempt
            if e["event_type"] == "ModelRequestRecorded"
            and attempt.index(e) > attempt.index(response)
        ]
        calls = [
            part
            for part in response["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
            if part.get("part_kind") == "tool-call"
        ]

        def read(selected):
            turn = [e for e in selected if selected.index(e) > selected.index(response)]
            tools = _compatible_model_request(request)["function_tools"]
            return _validate_ordered_calls(
                response,
                calls,
                [e for e in turn if e["event_type"] == "ToolCallValidated"][:2],
                [
                    e
                    for e in turn
                    if e["event_type"] == "AgentActionTurnDispositionRecorded"
                ][:2],
                role="planning_agent",
                reopen=planning_assignment.planning_assignment().reopen,
                exposed_tool_names={tool["name"] for tool in tools},
                bound_arguments=request_bound_arguments(tools),
                timeline=events,
                selected=selected,
                admitted_action_fingerprints=set(),
                next_request=later[0] if later else None,
            )

        read(attempt)
        # The same retained records, with the second call's admission placed before
        # the first call ran, as a whole-response admission recorded them.
        second = next(
            i
            for i, e in enumerate(attempt)
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "call-second"
        )
        batched = list(attempt)
        moved = batched[second : second + 2]
        del batched[second : second + 2]
        first_start = next(
            i
            for i, e in enumerate(batched)
            if e["event_type"] == "ObservedToolExecutionStarted"
        )
        batched[first_start:first_start] = moved
        with self.assertRaisesRegex(ValueError, "not admitted in the order sent"):
            read(batched)

    def test_reopen_does_not_replay_a_refused_call_s_errors(self):
        from bio_neb.application.queries import project_journey_events

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _QuestionThenRead(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
        project_journey_events(events, journey_file=receipt.journey_file)
        changed = copy.deepcopy(events)
        refused = next(
            e
            for e in changed
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "after-question"
        )
        refused["payload"]["errors"] = ["another_reason"]
        project_journey_events(changed, journey_file=receipt.journey_file)

    def test_reopen_refuses_an_admitted_call_that_never_ran(self):
        def drop_first_run(events):
            first = next(
                e["event_id"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "read-first"
            )
            start = next(
                e
                for e in events
                if e["event_type"] == "ObservedToolExecutionStarted"
                and e["payload"]["validation_event_id"] == first
            )
            events[:] = [
                e
                for e in events
                if e is not start
                and e["payload"].get("started_event_id") != start["event_id"]
            ]

        self._hostile(drop_first_run, "observed Tool execution|did not run")

    def test_reopen_refuses_a_whole_refusal_with_a_changed_lineage(self):
        from bio_neb.application.queries import project_journey_events

        _result, events, journey_file = self._pair_journey(
            ("record_reference_choice", {"candidate_id": "x", "reason": "Early."})
        )
        project_journey_events(events, journey_file=journey_file)
        turns = [
            i
            for i, e in enumerate(events)
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
        ]
        for name, change in (
            ("response", lambda turn: {"response_event_id": str(uuid.uuid4())}),
            (
                "validation",
                lambda turn: {
                    "validation_event_ids": events[turns[1]]["payload"][
                        "validation_event_ids"
                    ]
                },
            ),
        ):
            with self.subTest(changed=name):
                hostile = copy.deepcopy(events)
                hostile[turns[0]]["payload"].update(change(hostile[turns[0]]))
                with self.assertRaises(ValueError):
                    project_journey_events(hostile, journey_file=journey_file)

    def test_a_failing_first_call_is_shown_and_the_same_failure_again_holds(self):
        """Owner, #477 C8 (alignment gate 2026-10-28): the failed first call is
        shown to the model as its result; the same Tool failing the same way
        again holds the Run, and the hold keeps the cause."""
        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey
        from tests.reference_fakes import ReferenceAdapter

        class Failing(ReferenceAdapter):
            def discover_assemblies(self, ncbi_tax_id, query):
                self.contact_count += 1
                raise RuntimeError("unexpected adapter fault")

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ThenSearch(receipt, paths, "discover_assemblies")
            composition.reference_adapter = Failing()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            # Contact-free reopen reads the failed calls' own evidence.
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            events = _events(receipt.journey_file)
        self.assertTrue(composition.sent)
        types = [e["event_type"] for e in events]
        self.assertIn("TaxonResolutionRecorded", types)
        (shown,) = (e for e in events if e["event_type"] == "ToolExecutionRefused")
        self.assertEqual(
            (shown["payload"]["statement"], shown["payload"]["fault_key"]),
            ("the Tool failed: RuntimeError", "discover_assemblies: RuntimeError"),
        )
        failed = next(e for e in events if e["event_type"] == "AttemptFailed")
        self.assertEqual(
            (failed["payload"]["code"], failed["payload"]["invariant_id"]),
            ("invariant", "tool_failed_again"),
        )
        # The failed Attempt holds the Run and the hold keeps the cause.
        self.assertEqual(review.status, "run_held")
        self.assertEqual(
            (
                refused.exception.failure.invariant_id,
                refused.exception.failure.evidence["code"],
            ),
            ("planning_attempt_failed", "invariant"),
        )
        self.assertRegex(refused.exception.failure.cause, r"^Invariant@")
        self.assertNotIn("RunFailed", types)

    def test_reopen_refuses_a_tool_refusal_with_an_invariant_cause(self):
        """ADR 0031/0034 and owner C8 (#477 gate): a refusal follows from its own
        failed call; an invariant holds and is never shown (Standards r1 ST2)."""
        import copy
        import uuid

        from bio_neb.application.advancement import (
            ResumeRun,
            RunAdvancementRefused,
            advance,
            use_run_advancement_composition,
        )
        from bio_neb.application.queries import project_journey_events
        from tests.application_fakes import _registered
        from tests.reference_fakes import ReferenceAdapter

        class Fails(ReferenceAdapter):
            def discover_assemblies(self, ncbi_tax_id, query):
                self.contact_count += 1
                raise RuntimeError("synthetic offline adapter fault")

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ThenSearch(receipt, paths, "discover_assemblies")
            composition.reference_adapter = Fails()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused),
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            project_journey_events(events, journey_file=receipt.journey_file)
            (shown,) = [e for e in events if e["event_type"] == "ToolExecutionRefused"]

            def hostile(change):
                changed = copy.deepcopy(events)
                change(changed)
                return changed

            def failed_terminal(changed):
                return next(
                    e
                    for e in changed
                    if e["event_id"] == shown["payload"]["failed_event_id"]
                )["payload"]

            def refusal(changed):
                return next(
                    e for e in changed if e["event_type"] == "ToolExecutionRefused"
                )

            def duplicate(changed):
                twin = copy.deepcopy(refusal(changed))
                twin["event_id"] = str(uuid.uuid4())
                changed.insert(changed.index(refusal(changed)) + 1, twin)

            for name, change in (
                (
                    "invariant cause",
                    lambda c: failed_terminal(c).update(
                        mechanical_error_type="Invariant"
                    ),
                ),
                (
                    "changed key",
                    lambda c: refusal(c)["payload"].update(
                        fault_key="discover_assemblies: ValueError"
                    ),
                ),
                ("duplicate", duplicate),
            ):
                with self.subTest(change=name), self.assertRaises(ValueError):
                    project_journey_events(
                        hostile(change), journey_file=receipt.journey_file
                    )

    def test_a_refused_call_returns_its_correction_while_the_admitted_one_runs(self):
        calls, parts = self._pair(("request_planning_hitl", {}))
        errors, turn, ran = calls["call-second"]
        self.assertEqual(errors, ["reference_work_not_open"])
        self.assertFalse(ran)
        self.assertEqual(turn["disposition"], "multi_call_refused")
        self.assertEqual(
            turn["correction_key"].split(" @ ")[0],
            "request_planning_hitl: reference_work_not_open",
        )
        self.assertTrue(calls["read-first"][2])
        returns = {
            part["tool_call_id"]: part
            for part in parts
            if part["part_kind"] == "tool-return"
        }
        self.assertEqual(returns["call-second"]["content"], turn["feedback"])
        self.assertEqual(returns["call-second"]["outcome"], "failed")
        self.assertIn("read-first", returns)

    def test_an_unoffered_call_is_refused_on_its_own_while_the_offered_call_runs(
        self,
    ):
        calls, parts = self._pair(
            ("record_reference_choice", {"candidate_id": "x", "reason": "Early."})
        )
        errors, turn, ran = calls["call-second"]
        self.assertEqual(errors, ["tool_name_unavailable"])
        self.assertIsNone(turn)
        self.assertFalse(ran)
        self.assertEqual(calls["read-first"][0], [])
        self.assertTrue(calls["read-first"][2])
        # The kept reply holds only the call that ran; the note follows its result.
        (reply,) = [
            part["tool_call_id"] for part in parts if part["part_kind"] == "tool-call"
        ]
        self.assertEqual(reply, "read-first")
        returned = [
            part["part_kind"] for part in parts if part["part_kind"] != "tool-call"
        ]
        self.assertEqual(returned[-2:], ["tool-return", "retry-prompt"])
        note = parts[-1]["content"]
        self.assertIn("not offered now", note)
        self.assertIn("`record_reference_choice`", note)
        self.assertIn("failed turn 1 of 4 in a row", note)


class _Search(PlanningComposition):
    """Searches the Journey once with scripted arguments, then works as scripted."""

    def __init__(self, receipt, paths, arguments):
        super().__init__(receipt, paths)
        self.arguments = arguments

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if self.calls:
            return await super().respond(messages, info)
        self.calls.append("search")
        arguments = (
            self.arguments(self.journey.read())
            if callable(self.arguments)
            else self.arguments
        )
        # A list is one reply's several searches, the first named "search".
        return ModelResponse(
            parts=[
                ToolCallPart(HISTORY, item, tool_call_id="search" + "-again" * number)
                for number, item in enumerate(
                    arguments if isinstance(arguments, list) else [arguments]
                )
            ]
        )


class JourneySearchTests(unittest.TestCase):
    def _search(self, arguments, **registration):
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), **registration)
            composition = _Search(receipt, paths, arguments)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            self.journey_file = receipt.journey_file
            return _events(receipt.journey_file), _requests(receipt.journey_file)

    def test_the_assembly_offers_search_journey_and_no_history_read(self):
        from bio_neb.agentic_journey import assignment_context

        _events_, requests = self._search({"text": "sample"})
        for request in requests:
            offered = {tool["name"] for tool in request["function_tools"]}
            self.assertIn(HISTORY, offered)
            self.assertNotIn("read_assignment_history", offered)
        self.assertFalse(hasattr(assignment_context, "read_assignment_history"))

    def test_a_text_search_returns_bounded_recorded_matches(self):
        events, requests = self._search({"text": "SAMPLE", "limit": 2})
        (read,) = [e for e in events if e["event_type"] == "JourneySearchRead"]
        query, result = read["payload"]["query"], read["payload"]["result"]
        self.assertEqual(query["text"], "SAMPLE")
        self.assertEqual(query["limit"], 2)
        self.assertTrue(result["events"])
        self.assertLessEqual(len(result["events"]), 2)
        retained = {e["event_id"]: e for e in events}
        for event in result["events"]:
            # A transcript is returned by reference; the retained event matched.
            self.assertIn("sample", json.dumps(retained[event["event_id"]]).lower())
        # The model receives the search's view: the recorded result as recorded,
        # with the read's id beside it, never sent.
        (returned,) = [
            part
            for message in requests[1]["messages"]
            for part in message["parts"]
            if part.get("tool_call_id") == "search"
            and part["part_kind"] == "tool-return"
        ]
        self.assertEqual(returned["content"]["result"], result)
        self.assertEqual(returned["content"]["status"], "read")
        self.assertEqual(returned["metadata"], {"event_id": read["event_id"]})
        self.assertIn(
            read["event_id"],
            {
                e["payload"]["stage_result_event_id"]
                for e in events
                if e["event_type"] == "ObservedToolExecutionSucceeded"
            },
        )

    def test_a_slice_counts_the_characters_of_the_text_the_model_is_shown(self):
        # A slice cuts the recorded canonical JSON, where a tab is already the
        # two characters ``\\t``; a raw DEL is shown as one space, so the
        # offsets and counts the model reads address the text it is shown.
        prompt = "Analyze rs1234 for sample-001 Homo sapiens.\tend\x7f."

        def request_slice(events):
            (request,) = [
                e for e in events if e["event_type"] == "ResearchRequestRegistered"
            ]
            # Five characters before the DEL, for ten.
            offset = canonical_json(request["payload"]).index("\x7f") - 5
            return {
                "payload_slice": {
                    "event_id": request["event_id"],
                    "offset": offset,
                    "length": 10,
                }
            }

        events, requests = self._search(request_slice, prompt_text=prompt)
        (read,) = [e for e in events if e["event_type"] == "JourneySearchRead"]
        recorded = read["payload"]["result"]["payload_slice"]
        (returned,) = [
            part["content"]["result"]["payload_slice"]
            for message in requests[1]["messages"]
            for part in message["parts"]
            if part.get("tool_call_id") == "search"
            and part["part_kind"] == "tool-return"
        ]
        (request,) = [
            e for e in events if e["event_type"] == "ResearchRequestRegistered"
        ]
        text = canonical_json(request["payload"])
        at = text.index("\x7f") - 5
        self.assertEqual(recorded["text"], '\\tend\x7f.","')
        self.assertEqual(text[at : at + 10], recorded["text"])
        self.assertEqual(
            (returned["text"], returned["total_characters"], returned["next_offset"]),
            ('\\tend .","', len(text), at + 10),
        )
        self.assertNotIn("\x7f", json.dumps(returned, ensure_ascii=False))

    def test_an_object_argument_sent_as_json_text_is_read_as_that_object(self):
        # Under "auto" a nullable top-level object argument arrived as JSON text
        # (6 of 6). The Model turn reads it as the object before validation and
        # records the name it read; reopen reads it the same way.
        def sliced_as_text(events):
            (request,) = [
                e for e in events if e["event_type"] == "ResearchRequestRegistered"
            ]
            return {
                "payload_slice": json.dumps(
                    {"event_id": request["event_id"], "length": 10}
                )
            }

        events, _requests_ = self._search(sliced_as_text)
        # Reopen reads the call as the record says it was read, and refuses a
        # record whose decoded names the reply's arguments contradict.
        for label, names in (("names dropped", None), ("name added", ["text"])):
            changed = copy.deepcopy(events)
            for event in changed:
                if (
                    event["event_type"] == "ToolCallValidated"
                    and event["payload"]["tool_call_id"] == "search"
                ):
                    if names is None:
                        del event["payload"]["arguments_decoded"]
                    else:
                        event["payload"]["arguments_decoded"] += names
            with (
                self.subTest(label),
                self.assertRaisesRegex(ValueError, "changed its exact model call"),
            ):
                project_journey_events(changed, journey_file=self.journey_file)
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        (request,) = [
            e for e in events if e["event_type"] == "ResearchRequestRegistered"
        ]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(
            validation["arguments"]["payload_slice"],
            {"event_id": request["event_id"], "length": 10},
        )
        self.assertEqual(validation["arguments_decoded"], ["payload_slice"])
        (read,) = [e for e in events if e["event_type"] == "JourneySearchRead"]
        self.assertEqual(read["payload"]["result"]["payload_slice"]["offset"], 0)

    def test_a_call_and_its_copy_sent_as_json_text_collapse_and_reopen(self):
        # Routing compares calls as the Model turn reads them, so the same slice
        # sent as an object and again as JSON text is one call sent twice; the
        # reopen of the routed reply reads them the same way.
        def twice(events):
            (request,) = [
                e for e in events if e["event_type"] == "ResearchRequestRegistered"
            ]
            piece = {"event_id": request["event_id"], "length": 10}
            return [{"payload_slice": piece}, {"payload_slice": json.dumps(piece)}]

        events, _requests_ = self._search(twice)
        (routed,) = [
            e["payload"] for e in events if e["event_type"] == "AgentTurnRouted"
        ]
        self.assertEqual(routed["copies"], {"search-again": "search"})
        self.assertEqual(routed["run_order"], ["search"])

    def test_a_string_argument_holding_json_stays_a_string(self):
        events, _requests_ = self._search({"text": '{"kinds": []}'})
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["arguments"]["text"], '{"kinds": []}')
        self.assertNotIn("arguments_decoded", validation)

    def test_an_object_argument_sent_as_invalid_json_text_stays_a_schema_error(self):
        events, _requests_ = self._search({"payload_slice": '{"event_id": '})
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        self.assertEqual(validation["status"], "invalid")
        self.assertEqual(validation["arguments"]["payload_slice"], '{"event_id": ')
        self.assertTrue(
            validation["errors"][0].startswith("tool_arguments_invalid:payload_slice"),
            validation["errors"],
        )
        self.assertNotIn("arguments_decoded", validation)

    def test_a_number_sent_as_text_is_refused_at_admission_not_coerced(self):
        """#476 (owner, 2026-10-28, G3): admission records exactly what the Tool
        receives, so a wrong JSON type is refused with the schema's correction
        rather than coerced after it is recorded (sweep A1-A3)."""
        events, _requests_ = self._search({"text": "x", "limit": "10"})
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        self.assertEqual(validation["status"], "invalid")
        self.assertTrue(
            validation["errors"][0].startswith("tool_arguments_invalid:limit:int_type"),
            validation["errors"],
        )
        self.assertNotIn("JourneySearchRead", {e["event_type"] for e in events})

    def test_a_blank_or_null_optional_argument_is_recorded_as_absent(self):
        """#476 (owner, 2026-10-28, G3): an omitted, null or blank optional value
        is one canonical "absent" in the recorded call, and the Journey reopens."""
        events, _requests_ = self._search(
            {"text": "x", "before_event_id": "", "payload_slice": None}
        )
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["arguments"], {"text": "x"})

    def test_an_unbounded_search_is_refused_before_it_runs(self):
        events, _requests_ = self._search({"text": "x" * 201})
        (validation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_call_id"] == "search"
        ]
        self.assertEqual(validation["status"], "invalid")
        self.assertTrue(
            validation["errors"][0].startswith("tool_arguments_invalid:text")
        )
        self.assertNotIn("JourneySearchRead", {e["event_type"] for e in events})


def _stage_findings(instructions):
    """The stage findings view a request's instructions carried."""
    return json.loads(instructions[instructions.index('{"authority":') :])[
        "stage_findings"
    ]


class StageFindingsTests(unittest.TestCase):
    def test_the_findings_view_fills_as_tools_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = {e["event_id"]: e for e in _events(receipt.journey_file)}
            requests = _requests(receipt.journey_file)
        views = [_stage_findings(text) for text in composition.instructions_seen]
        self.assertEqual(len(views), len(requests))
        stage_sets = {
            stage: stage_tool_names("planning_agent", stage) | {HISTORY}
            for stage in declared_role("planning_agent").stages
        }
        grew = False
        for number, (view, request) in enumerate(
            zip(views, requests, strict=True), start=1
        ):
            with self.subTest(request=number):
                offered = frozenset(t["name"] for t in request["function_tools"])
                # The view describes the very stage whose Tools the request shows.
                self.assertEqual(stage_sets[view["stage"]], offered)
                for finding in view["findings"]:
                    # A slot row cites the plan that records the slot.
                    self.assertEqual(
                        events[finding["source_event_id"]]["event_type"],
                        "ProposalRequirementsIdentified"
                        if finding["kind"] == "fact_slot"
                        else finding["kind"],
                    )
                for output in view["verified_outputs"]:
                    for event_id in output["evidence_event_ids"]:
                        self.assertIn(event_id, events)
        for earlier, later in itertools.pairwise(views):
            if earlier["stage"] != later["stage"]:
                # A stage closes only on its verified output.
                self.assertIn(
                    earlier["stage"], [o["stage"] for o in later["verified_outputs"]]
                )
                continue
            before = {f["source_event_id"] for f in earlier["findings"]}
            after = {f["source_event_id"] for f in later["findings"]}
            self.assertLessEqual(before, after)
            grew = grew or before < after
        self.assertTrue(grew)
        self.assertGreater(len({view["stage"] for view in views}), 2)
        slots = {
            finding["subject_key"]: (finding.get("value"), finding["status"])
            for view in views
            if view["stage"] == "reference"
            for finding in view["findings"]
            if finding["kind"] == "fact_slot"
        }
        self.assertEqual(
            slots["sample_identity"],
            ("sample-001", "asserted by researcher, unverified"),
        )
        self.assertIn("taxon_identity", slots)

    def test_a_closing_stage_output_the_server_cannot_verify_keeps_its_stage_open(
        self,
    ):
        from unittest import mock

        from bio_neb.agentic_journey.attempt_terminal import (
            ContinueAttempt,
            SliceReachedTerminal,
        )
        from tests.planning_assignment_fakes import prepare_configured_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, *_rest = prepare_configured_run(Path(directory))
            events = _events(receipt.journey_file)
        ready = next(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "SliceCheckpointReached"
            and e["payload"]["checkpoint"] == "execution_planning_ready"
        )
        prefix, attempt_id = events[:ready], events[ready]["payload"]["attempt_id"]
        closed = planning_assignment.assignment_terminal(prefix, attempt_id)
        self.assertIsInstance(closed, SliceReachedTerminal)
        # The real terminal caller closes the stage only on its verified output:
        # an output whose citation the server cannot verify keeps the stage open.
        with mock.patch.object(
            planning_assignment.ExecutionPlanningStageOutput,
            "evidence_types",
            {"selection_event_ids": frozenset({"ProposalRequirementsIdentified"})},
        ):
            held = planning_assignment.assignment_terminal(prefix, attempt_id)
            view = planning_assignment.planning_stage_findings(
                SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
            )
        self.assertIsInstance(held, ContinueAttempt)
        self.assertEqual(view.stage, "execution_planning")
        self.assertEqual(view.output, "refused")
        self.assertEqual(view.refusal.slot, "selection_event_ids")
        self.assertIn("ProposalRequirementsIdentified", view.refusal.explanation)

    def test_a_stage_output_the_server_cannot_verify_is_refused(self):
        from bio_neb.agentic_journey.planning_assignment import ReferenceStageOutput
        from bio_neb.agentic_journey.stage_findings import verify_stage_output

        with tempfile.TemporaryDirectory() as directory:
            receipt, _composition = _planning_ready(Path(directory))
            events = _events(receipt.journey_file)
        plan = next(
            e for e in events if e["event_type"] == "ProposalRequirementsIdentified"
        )
        description = next(
            e for e in events if e["event_type"] == "ReferenceDescriptionRecorded"
        )
        verified = ReferenceStageOutput(
            assembly_accession="GCF_000001405.40",
            reference_event_id=description["event_id"],
        )
        self.assertIsNone(verify_stage_output(verified, events))
        wrong_kind = verify_stage_output(
            verified.model_copy(update={"reference_event_id": plan["event_id"]}),
            events,
        )
        self.assertEqual(wrong_kind.code, "stage_output_unverified")
        self.assertEqual(wrong_kind.slot, "reference_event_id")
        self.assertEqual(wrong_kind.value, plan["event_id"])
        self.assertIn("ReferenceDescriptionRecorded", wrong_kind.explanation)
        disagreeing = verify_stage_output(
            verified.model_copy(update={"assembly_accession": "GCF_000000000.1"}),
            events,
        )
        self.assertEqual(disagreeing.slot, "assembly_accession")
        self.assertEqual(disagreeing.value, "GCF_000000000.1")


class _TaxonAdapter:
    """A scripted NCBI taxonomy answer: no record, or an operational fault."""

    @staticmethod
    def build(kind):
        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
            TaxonomyAdapterObservation,
        )
        from tests.reference_fakes import ReferenceAdapter, reference_source

        class Adapter(ReferenceAdapter):
            def resolve_taxon(self, query):
                self.contact_count += 1
                if kind == "fault":
                    raise ReferenceAdapterError(
                        "transport_error", "NCBI is unreachable.", retryable=False
                    )
                return TaxonomyAdapterObservation(
                    records=(),
                    source=reference_source("ncbi_taxonomy", f"taxonomy/{query}"),
                )

        return Adapter()


class _OnceUnavailable(ReferenceAdapter):
    def __init__(self):
        super().__init__()
        self.queries = []

    def resolve_taxon(self, query):
        self.queries.append(query)
        if len(self.queries) == 1:
            raise ReferenceAdapterError(
                "http_error", "Service rejected request.", retryable=False
            )
        return super().resolve_taxon(query)


class TurnFailureRecordTests(unittest.TestCase):
    """Each Agent turn failure reaches the failure owner's record (#371): a
    recoverable one its Review, our bug a held Run under its own id."""

    def test_a_tool_invariant_keeps_its_own_id_on_the_attempt_record(self):
        from bio_neb.application.advancement import RunAdvancementRefused
        from bio_neb.failure import Invariant

        class Broken(ReferenceAdapter):
            def resolve_taxon(self, query):
                raise Invariant("taxon_reader_broken", "the taxon reader lost its row")

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = Broken()
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            events = _events(receipt.journey_file)
        (failed,) = [e for e in events if e["event_type"] == "AttemptFailed"]
        (tool_failure,) = [
            e for e in events if e["event_type"] == "ObservedToolExecutionFailed"
        ]
        self.assertEqual(
            (
                failed["payload"]["code"],
                failed["payload"]["invariant_id"],
                failed["payload"]["source_event_id"],
            ),
            ("invariant", "taxon_reader_broken", tool_failure["event_id"]),
        )
        self.assertEqual(review.status, "run_held")
        self.assertEqual(refused.exception.failure.evidence["code"], "invariant")

    def test_a_framework_refusal_is_held_under_its_own_invariant_id(self):
        import asyncio

        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.test import TestModel

        from bio_neb.agentic_journey.invocation import invoke_agent
        from bio_neb.failure import Invariant

        # A new prompt on a history whose last reply still waits for its call's
        # result is a shape the framework refuses to run.
        history = [ModelResponse(parts=[ToolCallPart("lookup", {}, "waiting")])]
        with self.assertRaises(Invariant) as held:
            asyncio.run(
                invoke_agent(
                    Agent(TestModel()),
                    "go on",
                    deps=None,
                    policy=SimpleNamespace(model_requests=3),
                    journey_id="journey",
                    attempt_id="attempt",
                    message_history=history,
                )
            )
        self.assertEqual(held.exception.invariant_id, "agent_framework_refused")
        self.assertIn("unprocessed tool calls", str(held.exception))

    def test_the_framework_s_own_output_limit_binding_first_opens_the_work_review(
        self,
    ):
        from unittest.mock import PropertyMock, patch

        from pydantic_ai.messages import ModelResponse, TextPart

        from bio_neb.agentic_journey.continuation_policy import AgentInvocationPolicy

        class EndsInText(PlanningComposition):
            async def respond(self, messages, info):
                if "attest_sample" in {tool.name for tool in info.function_tools}:
                    return ModelResponse(parts=[TextPart("Finished.")])
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            # A bounded fault: the framework's global output counter, sized never
            # to bind first, binds at the first text ending.
            with (
                patch.object(
                    AgentInvocationPolicy,
                    "framework_retry_allowance",
                    new_callable=PropertyMock,
                    return_value=0,
                ),
                use_run_advancement_composition(EndsInText(receipt, paths)),
            ):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
        self.assertEqual(result.status, "agent_work_review_pending")
        self.assertEqual(reopened.status, "agent_work_review_pending")
        (request,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentWorkReviewRequested"
        ]
        self.assertEqual(request["cause"], "result_missing")
        self.assertIn("Exceeded maximum output retries (0)", request["reason"])
        self.assertNotIn("AttemptFailed", [e["event_type"] for e in events])


class _RejectsTaxon(ReferenceAdapter):
    """Rejects the first ``rejections`` taxon resolutions."""

    def __init__(self, rejections):
        super().__init__()
        self.rejections = rejections
        self.queries = []

    def resolve_taxon(self, query):
        self.queries.append(query)
        if len(self.queries) <= self.rejections:
            raise ReferenceAdapterError(
                "http_error", "Service rejected request.", retryable=False
            )
        return super().resolve_taxon(query)


class RetryPresetTests(unittest.TestCase):
    """A researcher's retry preset answers an Agent failure Review as it opens,
    through the Review's own record, within its bound (#371 D2)."""

    def _run(self, rejections, *presets):
        from bio_neb.application.commands import (
            RecordRetryPresetCommand,
            record_retry_preset,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, paths = _registered(Path(directory.name))
        for preset in presets:
            record_retry_preset(
                RecordRetryPresetCommand(
                    journey_file=receipt.journey_file, preset=preset
                )
            )
        composition = PlanningComposition(receipt, paths)
        composition.reference_adapter = _RejectsTaxon(rejections)
        with use_run_advancement_composition(composition):
            result = advance(ResumeRun(journey_file=receipt.journey_file))
        self.journey_file = receipt.journey_file
        self.composition = composition
        return result, _events(receipt.journey_file), composition.reference_adapter

    @staticmethod
    def _preset(**fields):
        return {
            "agent_role": "planning_agent",
            "failure_classes": ["rejected"],
            "count": 1,
            "by": "test researcher",
            **fields,
        }

    def test_a_preset_retry_answers_the_review_and_the_run_continues(self):
        result, events, source = self._run(
            1, self._preset(tool_names=["resolve_taxon"])
        )

        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertEqual(source.queries, ["Homo sapiens", "Homo sapiens"])
        (preset,) = [e for e in events if e["event_type"] == "AgentFailureRetryPreset"]
        (request,) = [
            e for e in events if e["event_type"] == "AgentFailureReviewRequested"
        ]
        (answer,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentFailureReviewResponded"
        ]
        # The ordinary answer of the Review's own record, naming its preset.
        self.assertEqual(
            (
                answer["request_event_id"],
                answer["response"]["action"],
                answer["response"]["by"],
                answer["authority"],
                answer["preset_event_id"],
            ),
            (
                request["event_id"],
                "retry",
                "test researcher",
                "researcher_direction",
                preset["event_id"],
            ),
        )
        retained = Path(self.journey_file).read_bytes()
        reopened = review_journey(ReviewJourneyQuery(journey_file=self.journey_file))
        self.assertEqual(reopened.status, "planning_hitl_pending")
        (view,) = reopened.operational_review.retry_presets
        self.assertEqual(view.remaining, 0)
        self.assertEqual(Path(self.journey_file).read_bytes(), retained)

    def test_a_preset_set_while_a_review_waits_answers_it_at_the_next_resume(self):
        from bio_neb.application.commands import (
            RecordRetryPresetCommand,
            record_retry_preset,
        )

        stopped, _, source = self._run(1)
        self.assertEqual(stopped.status, "operational_review_pending")
        record_retry_preset(
            RecordRetryPresetCommand(
                journey_file=self.journey_file, preset=self._preset()
            )
        )
        with use_run_advancement_composition(self.composition):
            resumed = advance(ResumeRun(journey_file=self.journey_file))
        self.assertEqual(resumed.status, "planning_hitl_pending")
        self.assertEqual(len(source.queries), 2)

    def test_a_preset_answers_an_investigation_member_s_review(self):
        from bio_neb.agentic_journey.operational_review import (
            answer_from_retry_preset,
            preset_agent_failure_retry,
            project_operational_review,
        )
        from bio_neb.application.queries import read_investigation
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SourceResponsesOnly,
            run_attempt,
        )

        class RejectsOnce(SourceResponsesOnly):
            def fetch_refsnp(self, rsid):
                if not self.operations:
                    self.operations.append("fetch_refsnp:" + rsid)
                    raise ProviderError("rejected", "Source denied this request.")
                return super().fetch_refsnp(rsid)

        with tempfile.TemporaryDirectory() as directory:
            model, source = SequentialAdaptiveModel(), RejectsOnce()
            stopped, journey = run_attempt(Path(directory) / "run", model, source)
            self.assertEqual(stopped.status, "operational_review_pending")
            preset_agent_failure_retry(
                journey, self._preset(agent_role="investigation_planner")
            )
            self.assertTrue(answer_from_retry_preset(journey))
            self.assertFalse(answer_from_retry_preset(journey))
            (receipt,) = project_operational_review(read_evidence(journey)).receipts
            resumed = investigation_assignment(
                model_factory=model.model, source_adapter_factory=lambda: source
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                operational_review_response_event_id=receipt.response_event_id,
            )
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(source.operations[:2], ["fetch_refsnp:rs429358"] * 2)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, resumed
            )

    def test_a_review_past_the_preset_s_count_waits_for_the_researcher(self):
        result, events, source = self._run(2, self._preset())

        self.assertEqual(result.status, "operational_review_pending")
        self.assertEqual(len(source.queries), 2)
        answers = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentFailureReviewResponded"
        ]
        self.assertEqual(len(answers), 1)
        self.assertEqual(
            len(
                [e for e in events if e["event_type"] == "AgentFailureReviewRequested"]
            ),
            2,
        )

    def test_a_review_the_preset_does_not_cover_waits_for_the_researcher(self):
        for name, preset in (
            ("another class", self._preset(failure_classes=["transient"])),
            ("another Tool", self._preset(tool_names=["discover_assemblies"])),
            ("another role", self._preset(agent_role="investigation_planner")),
            (
                "revoked",
                [
                    self._preset(),
                    {"agent_role": "planning_agent", "count": 0, "by": "r"},
                ],
            ),
        ):
            with self.subTest(preset=name):
                presets = preset if isinstance(preset, list) else [preset]
                result, events, source = self._run(1, *presets)
                self.assertEqual(result.status, "operational_review_pending")
                self.assertEqual(len(source.queries), 1)
                self.assertNotIn(
                    "AgentFailureReviewResponded", [e["event_type"] for e in events]
                )

    def test_reopen_refuses_a_preset_answer_its_preset_does_not_give(self):
        from bio_neb.application.queries import project_journey_events

        _, events, _ = self._run(1, self._preset())
        project_journey_events(events, journey_file=self.journey_file)

        def payload(hostile, event_type):
            (event,) = [e for e in hostile if e["event_type"] == event_type]
            return event["payload"]

        changes = {
            "another class": lambda h: payload(h, "AgentFailureRetryPreset").update(
                failure_classes=["transient"]
            ),
            "spent": lambda h: payload(h, "AgentFailureRetryPreset").update(
                count=0, failure_classes=[]
            ),
            "another author": lambda h: payload(h, "AgentFailureReviewResponded")[
                "response"
            ].update(by="someone else"),
            "unknown preset": lambda h: payload(
                h, "AgentFailureReviewResponded"
            ).update(preset_event_id=str(uuid.uuid4())),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                hostile = copy.deepcopy(events)
                change(hostile)
                with self.assertRaisesRegex(
                    ValueError, "operational answer lacks its exact retry preset"
                ):
                    project_journey_events(hostile, journey_file=self.journey_file)

    def test_a_preset_is_refused_where_another_review_s_re_entry_reads_the_head(
        self,
    ):
        # Finalization's case is PostExecutionAnswerTests' (it owns that Run).
        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionReviewAnswer,
            ObserveRun,
            PlanningReviewAnswer,
            observe,
        )
        from bio_neb.application.commands import (
            RecordRetryPresetCommand,
            RetryPresetRefused,
            record_retry_preset,
        )
        from bio_neb.parabricks.execution_proposal import ReviseExecutionResponse
        from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as tmp:
            receipt, composition, _ = configured_planning_run(Path(tmp))
            with use_run_advancement_composition(composition):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
                while (
                    review.current_review is not None
                    and review.current_review.kind == "planning"
                ):
                    question = review.planning_hitl.pending_request
                    review = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=question.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=question.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                self.assertEqual(review.status, "execution_hitl_pending")
                before = Path(receipt.journey_file).read_bytes()
                with self.assertRaises(RetryPresetRefused) as refusal:
                    record_retry_preset(
                        RecordRetryPresetCommand(
                            journey_file=receipt.journey_file, preset=self._preset()
                        )
                    )
                self.assertEqual(
                    refusal.exception.code, "retry_preset_placement_unsupported"
                )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                request = review.execution_hitl.pending_request
                # The Review's own lawful revision still re-enters exactly.
                revised = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=ReviseExecutionResponse(
                                action="revise",
                                proposal_id=request.proposal.proposal_id,
                                instruction="Revalidate the execution policy.",
                                revalidation_scope="execution_planning",
                            ),
                        ),
                    )
                )
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)), revised
            )
            self.assertNotEqual(revised.status, "execution_hitl_pending")

    def test_a_parent_s_preset_never_answers_a_fork_child_s_review(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.application.commands import (
            RecordRetryPresetCommand,
            record_retry_preset,
        )
        from bio_neb.application.fork import ForkRun, fork_run
        from bio_neb.failure import ProviderError
        from tests.planning_assignment_fakes import configured_planning_run
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )
        from tests.terminal_journey_fakes import EVIDENCE_REQUEST, execute_terminal

        def preset(journey_file, by):
            record_retry_preset(
                RecordRetryPresetCommand(
                    journey_file=journey_file,
                    preset=self._preset(agent_role="investigation_planner", by=by),
                )
            )

        def registered_with_preset(*args, **kwargs):
            # The parent's own preset, set before its first work.
            registered = configured_planning_run(*args, **kwargs)
            preset(registered[0].journey_file, "parent researcher")
            return registered

        class RejectsOnce(SequentialSourceResponses):
            calls = 0

            def fetch_refsnp(self, rsid):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("rejected", "Source denied this request.")
                return super().fetch_refsnp(rsid)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "tests.planning_assignment_fakes.configured_planning_run",
                side_effect=registered_with_preset,
            ):
                parent = execute_terminal(
                    root / "parent", investigation_request=EVIDENCE_REQUEST
                )
            for name, child_preset in (("parent's only", False), ("child's own", True)):
                with self.subTest(preset=name):
                    child = fork_run(
                        ForkRun(
                            parent_journey_file=parent,
                            child_run_root=str(root / name.replace(" ", "-")),
                        )
                    ).child_journey_file
                    if child_preset:
                        preset(child, "child researcher")
                    model = SequentialAdaptiveModel()
                    source = RejectsOnce(assembly_body=ASSEMBLY_BODY)
                    with use_run_advancement_dependencies(
                        environ={"LOCAL_MODEL_KEY": "offline"},
                        investigation_model_factory=model.model,
                        investigation_source_adapter_factory=lambda source=source: source,
                    ):
                        result = advance(ResumeRun(journey_file=child))
                    answers = [
                        e["payload"]
                        for e in _events(child)
                        if e["event_type"] == "AgentFailureReviewResponded"
                    ]
                    if child_preset:
                        (answer,) = answers
                        self.assertEqual(answer["response"]["by"], "child researcher")
                        self.assertEqual(source.calls, 2)
                        self.assertNotEqual(result.status, "operational_review_pending")
                    else:
                        self.assertEqual(answers, [])
                        self.assertEqual(source.calls, 1)
                        self.assertEqual(result.status, "operational_review_pending")

    def test_a_preset_is_bounded_and_never_retries_an_invariant(self):
        from pydantic import ValidationError

        from bio_neb.agentic_journey.operational_review import AgentFailureRetryPreset

        for name, preset in (
            ("invariant", self._preset(failure_classes=["invariant"])),
            ("past the bound", self._preset(count=4)),
            ("no class", self._preset(failure_classes=[])),
            ("revocation with a class", self._preset(count=0)),
            ("repeated class", self._preset(failure_classes=["rejected", "rejected"])),
            ("unknown role", self._preset(agent_role="any_agent")),
        ):
            with self.subTest(preset=name), self.assertRaises(ValidationError):
                AgentFailureRetryPreset.model_validate(preset)

    def test_the_cli_records_a_preset_and_prints_it(self):
        import contextlib
        import io

        from bio_neb.cli import main

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, _ = _registered(Path(directory.name))
        preset_file = Path(directory.name) / "preset.json"
        preset_file.write_text(json.dumps(self._preset(count=2)), encoding="utf-8")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main(
                [
                    "retry-preset",
                    "--journey-file",
                    receipt.journey_file,
                    "--preset-file",
                    str(preset_file),
                ]
            )
        self.assertEqual(code, 0)
        shown = json.loads(printed.getvalue())
        self.assertEqual(
            (shown["agent_role"], shown["count"], shown["remaining"]),
            ("planning_agent", 2, 2),
        )
        (preset,) = [
            e
            for e in _events(receipt.journey_file)
            if e["event_type"] == "AgentFailureRetryPreset"
        ]
        self.assertEqual(shown["preset_event_id"], preset["event_id"])


class NegativeResultTests(unittest.TestCase):
    def test_source_amendment_is_a_fresh_instruction_and_replay_does_no_work(self):
        from bio_neb.application.advancement import AnswerReview, JunctionReviewAnswer

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            source = _OnceUnavailable()
            composition.reference_adapter = source
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                request = stopped.current_review
                amendment = AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    answer=JunctionReviewAnswer(
                        kind="junction",
                        response={
                            "request_id": request.request.request.request_id,
                            "action": "amend",
                            "field": "instruction",
                            "value": "Reconsider this source under my supplied human taxon assertion.",
                            "by": "researcher",
                        },
                    ),
                )
                result = advance(amendment)
                before = Path(receipt.journey_file).read_bytes()
                self.assertEqual(advance(amendment), result)
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            events = _events(receipt.journey_file)
            starts = [e for e in events if e["event_type"] == "AttemptStarted"]
            self.assertEqual(
                starts[-1]["payload"]["effective_invocation_policy"],
                starts[-2]["payload"]["effective_invocation_policy"],
            )
            logical = next(
                e
                for e in events
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == starts[-1]["payload"]["attempt_id"]
            )
            messages = logical["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]
            self.assertTrue(
                any(
                    "Reconsider this source" in str(part.get("content", ""))
                    for message in messages
                    for part in message["parts"]
                )
            )
            self.assertFalse(
                any(
                    part["part_kind"] == "tool-return"
                    for message in messages
                    for part in message["parts"]
                )
            )
            self.assertIsNone(logical["payload"]["continuation_request_event_id"])
            self.assertEqual(source.queries, ["Homo sapiens"])
            self.assertEqual(result.status, "planning_hitl_pending")

    def test_operational_review_reopen_rejects_changed_work_policy_and_transcript(self):
        from bio_neb.agentic_journey.operational_review import (
            project_operational_review,
        )

        _, events, _ = self._run("fault")
        self.assertTrue(project_operational_review(events).pending_request)
        cases = (
            ("AgentFailureReviewRequested", "arguments", {"query": "another species"}),
            ("AgentFailureReviewRequested", "agent_role", "investigation_planner"),
            ("AgentFailureReviewRequested", "member_id", "different-member"),
            ("AgentFailureReviewRequested", "messages", []),
            ("AgentFailureReviewRequested", "bound_argument_names", ["foreign"]),
            ("OperationalFailureRecorded", "family", "provider"),
            ("OperationalFailureRecorded", "failure_class", "invariant"),
            ("OperationalFailureRecorded", "source_event_id", None),
            ("OperationalFailureRecorded", "contact_attempt", 2),
        )
        for kind, field, value in cases:
            with self.subTest(kind=kind, field=field):
                changed = copy.deepcopy(events)
                next(e for e in changed if e["event_type"] == kind)["payload"][
                    field
                ] = value
                with self.assertRaises(ValueError):
                    project_operational_review(changed)

    def test_an_operational_stop_records_its_pending_correction_in_the_transcript(
        self,
    ):
        """Characterization (#219): the recorded pending transcript is the sent
        request, the stopped response and one appended request returning the
        refused sibling's correction, exactly as the Review projection reads it."""
        from bio_neb.agentic_journey.operational_review import (
            project_operational_review,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ThenSearch(receipt, paths, "resolve_taxon")
            composition.reference_adapter = _TaxonAdapter.build("fault")
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            events = _events(receipt.journey_file)
        self.assertEqual(result.status, "operational_review_pending")
        (requested,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentFailureReviewRequested"
        ]
        messages = requested["messages"]
        pending = project_operational_review(events).pending_request
        self.assertEqual(pending.messages, messages)
        sent = [
            e["payload"]["compatible_adapter_evidence"]["request"]["messages"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ][-1]
        self.assertEqual(messages[: len(sent)], sent)
        stopped, correction = copy.deepcopy(messages[len(sent) :])
        self.assertEqual(stopped["kind"], "response")
        self.assertEqual(
            [
                (part["part_kind"], part["tool_name"], part["tool_call_id"])
                for part in stopped["parts"]
            ],
            [
                ("tool-call", "resolve_taxon", "assignment-4"),
                ("tool-call", HISTORY, "after-first"),
            ],
        )
        (refused,) = [
            e["payload"]["feedback"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["disposition"] == "multi_call_refused"
        ]
        self.assertTrue(
            refused.startswith(
                "Refused before execution (attempt_ended_earlier).\n"
                "Tool `search_journey` was not run."
            )
        )
        (part,) = correction["parts"]
        self.assertIsInstance(part.pop("timestamp"), str)
        self.assertEqual(
            correction,
            {
                "conversation_id": None,
                "instructions": None,
                "kind": "request",
                "metadata": None,
                "parts": [
                    {
                        "content": refused,
                        "metadata": None,
                        "outcome": "success",
                        "part_kind": "tool-return",
                        "tool_call_id": "after-first",
                        "tool_kind": None,
                        "tool_name": HISTORY,
                    }
                ],
                "run_id": None,
                "state": "complete",
                "timestamp": None,
            },
        )

    def _run(self, kind):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = _TaxonAdapter.build(kind)
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            return result, _events(receipt.journey_file), composition

    def test_a_negative_domain_result_returns_to_the_model_and_the_attempt_continues(
        self,
    ):
        result, events, composition = self._run("none")
        types = [e["event_type"] for e in events]
        taxon = types.index("TaxonResolutionRecorded")
        self.assertEqual(events[taxon]["payload"]["result"]["resolution"], "not_found")
        # The source negative is available to the next real model request.
        self.assertIn("ModelRequestRecorded", types[taxon:])
        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertEqual(composition.calls[-1], "request_planning_hitl")
        from bio_neb.agentic_journey.tool_execution import negative_domain_result

        self.assertTrue(negative_domain_result("planning_agent", events[taxon]))
        # The Tool ran: an answer that does not fit is a successful execution with
        # the result's own status, and the model reads that status and one next
        # step (G1.3).
        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        (ran,) = [
            e
            for e in events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
            and e["payload"]["stage_result_event_id"] == events[taxon]["event_id"]
        ]
        view = tool_model_view(model_view_of("resolve_taxon"), events[taxon])
        self.assertEqual(view["status"], "not_found")
        self.assertIn("`resolve_taxon`", view["next"])

    def test_transient_source_faults_retry_before_recording_one_result(self):
        from unittest.mock import patch

        from bio_neb.reference_discovery.reference_identity_tool import (
            ReferenceAdapterError,
        )
        from tests.reference_fakes import ReferenceAdapter

        class BrieflyUnavailable(ReferenceAdapter):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def resolve_taxon(self, query):
                self.calls += 1
                if self.calls < 3:
                    raise ReferenceAdapterError(
                        "http_error", "Temporarily unavailable.", retryable=True
                    )
                return super().resolve_taxon(query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            source = BrieflyUnavailable()
            composition.reference_adapter = source
            with (
                patch("random.uniform", side_effect=lambda low, high: high),
                patch("time.sleep") as sleep,
                use_run_advancement_composition(composition),
            ):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "planning_hitl_pending")
            self.assertEqual(source.calls, 3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])
            events = _events(receipt.journey_file)
            self.assertEqual(
                sum(e["event_type"] == "TaxonResolutionRecorded" for e in events), 1
            )
            self.assertEqual(
                sum(
                    e["event_type"] == "SourceOperationFailureRecorded" for e in events
                ),
                2,
            )
            # One default policy asks; with no entry nothing tunes it (D6).
            for event in events:
                if event["event_type"] == "SourceOperationFailureRecorded":
                    policy = event["payload"]["policy"]
                    self.assertEqual(
                        policy["surface"], {"scope": "tool", "name": "resolve_taxon"}
                    )
                    self.assertEqual(policy["surface_overrides"], {})
                    self.assertEqual(policy["effective"]["semantic"], "ask")
            self.assertNotIn(
                "AgentFailureReviewRequested", [e["event_type"] for e in events]
            )

    def test_a_tool_entry_tunes_only_that_tools_contact_retries(self):
        # A Tool surface's entry, recorded with its surface.
        from unittest.mock import patch

        class BrieflyUnavailable(ReferenceAdapter):
            def __init__(self):
                super().__init__()
                self.faults = {"resolve_taxon": 2, "discover_assemblies": 1}

            def _fault(self, name):
                if self.faults[name]:
                    self.faults[name] -= 1
                    raise ReferenceAdapterError(
                        "http_error", "Temporarily unavailable.", retryable=True
                    )

            def resolve_taxon(self, query):
                self._fault("resolve_taxon")
                return super().resolve_taxon(query)

            def discover_assemblies(self, ncbi_tax_id, query):
                self._fault("discover_assemblies")
                return super().discover_assemblies(ncbi_tax_id, query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                failure={
                    "scopes": {
                        "tool": {"resolve_taxon": {"operation": {"backoff_seconds": 3}}}
                    }
                },
            )
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = BrieflyUnavailable()
            with (
                patch("random.uniform", side_effect=lambda low, high: high),
                patch("time.sleep") as sleep,
                use_run_advancement_composition(composition),
            ):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "planning_hitl_pending")
            # The tuned Tool backs off from 3 seconds; the other keeps 1.
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [3, 6, 1])
            recorded = [
                e["payload"]["policy"]
                for e in _events(receipt.journey_file)
                if e["event_type"] == "SourceOperationFailureRecorded"
            ]
            self.assertEqual(
                [(p["surface"]["name"], p["surface_overrides"]) for p in recorded],
                [
                    ("resolve_taxon", {"backoff_seconds": 3}),
                    ("resolve_taxon", {"backoff_seconds": 3}),
                    ("discover_assemblies", {}),
                ],
            )
            self.assertEqual(recorded[0]["effective"]["backoff_seconds"], 3)
            self.assertEqual(recorded[2]["effective"]["backoff_seconds"], 1)

    def test_source_recovery_holds_on_unavailable_assignment_before_answer_write(self):
        from dataclasses import replace

        from bio_neb.application.advancement import (
            AnswerReview,
            JunctionReviewAnswer,
            RunAdvancementRefused,
        )

        class UnavailableAssignment(PlanningComposition):
            unavailable = False

            def planning_assignment(self, journey):
                assignment = super().planning_assignment(journey)
                return (
                    replace(assignment, agent_role="retired_planning_assignment")
                    if self.unavailable
                    else assignment
                )

        for action in ("retry", "amend"):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, paths = _registered(Path(directory))
                composition = UnavailableAssignment(receipt, paths)
                composition.reference_adapter = _OnceUnavailable()
                with use_run_advancement_composition(composition):
                    stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                    request = stopped.current_review
                    composition.unavailable = True
                    response = {
                        "request_id": request.request.request.request_id,
                        "action": action,
                        "by": "researcher",
                    }
                    if action == "amend":
                        response.update(
                            field="instruction", value="Reconsider this source."
                        )
                    before = Path(receipt.journey_file).read_bytes()
                    contacts = tuple(composition.reference_adapter.queries)
                    with self.assertRaisesRegex(
                        ValueError, "operational_review_response_illegal"
                    ):
                        advance(
                            AnswerReview(
                                journey_file=receipt.journey_file,
                                request_event_id=request.request_event_id,
                                answer=JunctionReviewAnswer(
                                    kind="junction",
                                    response={
                                        "request_id": request.request.request.request_id,
                                        "action": "amend",
                                        "field": "model",
                                        "value": "another endpoint",
                                        "by": "researcher",
                                    },
                                ),
                            )
                        )
                    self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                    with self.assertRaisesRegex(
                        RunAdvancementRefused, "agent_reentry_unavailable"
                    ):
                        advance(
                            AnswerReview(
                                journey_file=receipt.journey_file,
                                request_event_id=request.request_event_id,
                                answer=JunctionReviewAnswer(
                                    kind="junction", response=response
                                ),
                            )
                        )
                    _assert_only_the_hold_appended(
                        self,
                        receipt.journey_file,
                        before,
                        "agent_reentry_unavailable",
                    )
                    self.assertEqual(
                        tuple(composition.reference_adapter.queries), contacts
                    )

    def test_model_recovery_holds_on_unavailable_assignment_before_answer_write(self):
        from dataclasses import replace

        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import (
            AnswerReview,
            ModelAvailabilityReviewAnswer,
            RunAdvancementRefused,
        )

        class UnavailableAssignment(PlanningComposition):
            unavailable = False

            async def respond(self, messages, info):
                raise ModelHTTPError(402, "offline")

            def planning_assignment(self, journey):
                assignment = super().planning_assignment(journey)
                return (
                    replace(assignment, agent_role="retired_planning_assignment")
                    if self.unavailable
                    else assignment
                )

        for action in ("retry", "amend"):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, paths = _registered(Path(directory))
                composition = UnavailableAssignment(receipt, paths)
                with use_run_advancement_composition(composition):
                    stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                    request = stopped.current_review
                    composition.unavailable = True
                    response = {"action": action}
                    if action == "amend":
                        response["instruction"] = "Reconsider this request."
                    before = Path(receipt.journey_file).read_bytes()
                    with self.assertRaisesRegex(
                        RunAdvancementRefused, "agent_reentry_unavailable"
                    ):
                        advance(
                            AnswerReview(
                                journey_file=receipt.journey_file,
                                request_event_id=request.request_event_id,
                                answer=ModelAvailabilityReviewAnswer(
                                    kind="model_availability", response=response
                                ),
                            )
                        )
                    _assert_only_the_hold_appended(
                        self,
                        receipt.journey_file,
                        before,
                        "agent_reentry_unavailable",
                    )

    def test_source_retry_or_amend_followed_by_model_rejection_reopens_exact_review(
        self,
    ):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import AnswerReview, JunctionReviewAnswer
        from bio_neb.application.queries import ObserveRun, observe

        class SourceThenModelFailure(PlanningComposition):
            # The first refusal is shown to the model; the same one again asks
            # the availability Review.
            rejections = 0
            reject_next = False

            async def respond(self, messages, info):
                if (
                    self.reject_next or len(self.reference_adapter.queries) >= 2
                ) and self.rejections < 2:
                    self.rejections += 1
                    raise ModelHTTPError(402, "offline")
                return await super().respond(messages, info)

        for action in ("retry", "amend"):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, paths = _registered(Path(directory))
                composition = SourceThenModelFailure(receipt, paths)
                composition.reference_adapter = _OnceUnavailable()
                with use_run_advancement_composition(composition):
                    stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                    request = stopped.current_review
                    response = {
                        "request_id": request.request.request.request_id,
                        "action": action,
                        "by": "researcher",
                    }
                    if action == "amend":
                        response.update(
                            field="instruction", value="Reconsider this source."
                        )
                        composition.reject_next = True
                    resumed = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=request.request_event_id,
                            answer=JunctionReviewAnswer(
                                kind="junction", response=response
                            ),
                        )
                    )
                self.assertEqual(resumed.status, "model_availability_review_pending")
                self.assertEqual(
                    composition.reference_adapter.queries,
                    ["Homo sapiens"] * (2 if action == "retry" else 1),
                )
                pending = resumed.model_availability_review.pending_request
                self.assertEqual(pending.exhaustion_reason, "request_rejected")
                self.assertEqual(len(pending.contacts), 1)
                before = Path(receipt.journey_file).read_bytes()
                self.assertEqual(
                    observe(ObserveRun(journey_file=receipt.journey_file)), resumed
                )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                events = _events(receipt.journey_file)
                self.assertNotIn("RunFailed", [event["event_type"] for event in events])
                self.assertEqual(
                    sum(
                        event["event_type"] == "ModelAvailabilityReviewRequested"
                        for event in events
                    ),
                    1,
                )

                from bio_neb.agentic_journey.model_availability import (
                    project_model_availability_review,
                )

                changed = copy.deepcopy(events)
                failure_review = next(
                    event
                    for event in changed
                    if event["event_type"] == "ModelAvailabilityReviewRequested"
                )
                failure_review["payload"]["retry_frontier_event_id"] = changed[0][
                    "event_id"
                ]
                with self.assertRaisesRegex(
                    ValueError, "availability request changed its logical request"
                ):
                    project_model_availability_review(changed)

    def test_source_stop_reaches_one_finalization_without_more_contacts(self):
        from bio_neb.application.advancement import AnswerReview, JunctionReviewAnswer

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            source = _OnceUnavailable()
            composition.reference_adapter = source
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                before = _events(receipt.journey_file)
                review = stopped.current_review
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=review.request_event_id,
                        answer=JunctionReviewAnswer(
                            kind="junction",
                            response={
                                "state": "answered",
                                "request_id": review.request.request.request_id,
                                "action": "stop",
                                "by": "test researcher",
                            },
                        ),
                    )
                )
            self.assertEqual(source.queries, ["Homo sapiens"])
            self.assertEqual(resumed.status, "finalization_review_pending")
            after = _events(receipt.journey_file)
            self.assertNotIn("RunFailed", [e["event_type"] for e in after])
            self.assertEqual(
                sum(e["event_type"] == "FinalizationBasisCompiled" for e in after), 1
            )
            old_completed = [
                e for e in before if e["event_type"] == "ObservedToolExecutionSucceeded"
            ]
            for event in old_completed:
                self.assertEqual(
                    sum(e["event_id"] == event["event_id"] for e in after), 1
                )

    def test_exact_source_retry_accounts_for_ordered_siblings(self):
        from bio_neb.application.advancement import AnswerReview, JunctionReviewAnswer

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = _ThenSearch(receipt, paths, "resolve_taxon")
            source = _OnceUnavailable()
            composition.reference_adapter = source
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                before = _events(receipt.journey_file)
                review = stopped.current_review
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=review.request_event_id,
                        answer=JunctionReviewAnswer(
                            kind="junction",
                            response={
                                "state": "answered",
                                "request_id": review.request.request.request_id,
                                "action": "retry",
                                "by": "test researcher",
                            },
                        ),
                    )
                )
            suppressed = [
                e
                for e in before
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "after-first"
            ]
            self.assertEqual(len(suppressed), 1)
            self.assertEqual(
                suppressed[0]["payload"]["errors"], ["attempt_ended_earlier"]
            )
            self.assertEqual(source.queries, ["Homo sapiens", "Homo sapiens"])
            self.assertEqual(resumed.status, "planning_hitl_pending")
            after = _events(receipt.journey_file)
            old_completed = [
                e for e in before if e["event_type"] == "ObservedToolExecutionSucceeded"
            ]
            for event in old_completed:
                self.assertEqual(
                    sum(e["event_id"] == event["event_id"] for e in after), 1
                )

    def test_an_operational_fault_waits_at_a_scoped_failure_review(self):
        result, events, _composition = self._run("fault")
        types = [e["event_type"] for e in events]
        taxon = types.index("TaxonResolutionRecorded")
        self.assertEqual(events[taxon]["payload"]["result"]["status"], "failed")
        self.assertNotIn("ModelRequestRecorded", types[taxon:])
        self.assertEqual(result.status, "operational_review_pending")
        self.assertEqual(result.current_review.kind, "junction")
        self.assertEqual(
            [action.action for action in result.current_review.legal_actions],
            ["retry", "amend", "stop"],
        )
        self.assertNotIn("RunFailed", types)


# Every source surface's effective per-call limits, unchanged by moving each Tool's
# declaration onto its row (the table `limits_for` read at ddfc9578).
_MIB = 1024 * 1024
_EFFECTIVE_LIMITS = {
    # The whole assembly report: the companion's and the sequence names'.
    "describe_companion": (30, 256 * _MIB),
    "ncbi_sequence_names": (30, 256 * _MIB),
    "download_paper": (16, 4 * _MIB),
    # A cited paper's download, the planning Tool's own call.
    "retrieve_paper": (16, 4 * _MIB),
    "europe_pmc_full_text": (30, 64 * _MIB),
    "europe_pmc_supplementary_files": (30, 64 * _MIB),
    "ncbi_genome_annotation": (30, 256 * _MIB),
    "ncbi_hgvs_contextuals": (30, 16 * _MIB),
    "ncbi_spdi_equivalents": (30, 16 * _MIB),
    "ncbi_taxon_assemblies": (30, 16 * _MIB),
    "pmc_open_data_file": (30, 64 * _MIB),
}
_DEFAULT_LIMITS = (30, 4 * _MIB)
_SOURCE_SURFACES = {
    "discover_assemblies",
    "eva_clustered_variant",
    "eva_clustered_variants_at",
    "eva_contig_alias",
    "eva_submitted_variants",
    "europe_pmc_search",
    "fetch_assembly_report",
    "fetch_eva_variant",
    "fetch_refsnp",
    # A cited paper's Europe PMC search.
    "find_paper",
    "galaxy_prebuilt_index",
    "inspect_reference",
    "read_prebuilt_index",
    "locate_variant",
    "ncbi_reference_span",
    "ncbi_spdi_rsids",
    "pmc_open_data_listing",
    # A typed gene change's placement, its GETs on their operations' surfaces.
    "place_gene_change",
    "resolve_taxon",
    "search_assemblies",
    "search_prebuilt_indexes",
    # A suggested assembly's known targets checked on their lookups' operations.
    "suggest_assembly_with_index",
    *_EFFECTIVE_LIMITS,
}
_ASSEMBLY_ACCESSION = re.compile(r"GC[AF]_\d{9}\.\d+")


class DomainToolRowTests(unittest.TestCase):
    """Each domain owns its Tools' rows beside their handlers; the loop assembles one
    inventory (ADR 0040, #447)."""

    def test_each_tool_row_sits_beside_its_handler_in_its_domain(self):
        import ast

        from bio_neb.agentic_journey import tool_execution
        from bio_neb.agentic_journey.researcher_question import (
            ASK_RESEARCHER,
            researcher_question_row,
        )

        assembled = []
        for module in tool_execution.tool_row_modules():
            name = module.__name__
            for row in module.TOOL_ROWS:
                assembled.append(row)
                with self.subTest(module=name, tool=row.tool_name):
                    if row.tool_name == ASK_RESEARCHER:
                        # The shared question: its domain declares the row, the
                        # assembly binds its one handler.
                        self.assertEqual(
                            row,
                            researcher_question_row(row.agent_role, row.stages),
                        )
                    else:
                        self.assertTrue(callable(getattr(module, row.tool_name)))
        self.assertCountEqual(assembled, function_tool_contracts())
        # The loop keeps the row types, the assembly, execution and reopen: it
        # declares no domain row.
        source = Path(inspect.getfile(tool_execution)).read_text(encoding="utf-8")
        calls = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("function_tool_contract", calls)
        # Its one row constructor is the factory every domain calls.
        constructors = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FunctionToolContract"
        ]
        self.assertEqual(len(constructors), 1)

    def test_a_source_tool_declares_its_limits_on_its_row(self):
        from bio_neb.agentic_journey import source_contacts
        from bio_neb.failure import FailureSurface, FailureTuning, limits_for
        from bio_neb.providers.paper_sources import PAPER_DOWNLOAD_SURFACES

        self.assertFalse(hasattr(source_contacts, "TOOL_LIMITS"))
        self.assertFalse(hasattr(source_contacts, "_TOOL_SURFACES"))
        rows = {
            row.tool_name: row.source_surface
            for row in function_tool_contracts()
            if row.source_surface is not None
        }
        # NCBI's companion GETs keep their own operation's bound; the genomic FASTA's
        # index Tools read Galaxy on theirs, and a read NCBI's report on its.
        self.assertEqual(rows["describe_companion"].operations, {"describe_companion"})
        self.assertEqual(
            rows["search_prebuilt_indexes"].operations, {"galaxy_prebuilt_index"}
        )
        self.assertEqual(
            rows["read_prebuilt_index"].operations,
            {"galaxy_prebuilt_index", "describe_companion"},
        )
        # The paper download is no Agent Tool: its own module declares it.
        self.assertEqual(PAPER_DOWNLOAD_SURFACES["download_paper"].limits.contacts, 16)
        self.assertEqual(source_contacts.source_tools(), _SOURCE_SURFACES)
        for name in sorted(_SOURCE_SURFACES):
            with self.subTest(surface=name):
                effective = limits_for(
                    FailureSurface(scope="tool", name=name), FailureTuning()
                ).effective
                self.assertEqual(
                    (effective.contacts, effective.response_bytes),
                    _EFFECTIVE_LIMITS.get(name, _DEFAULT_LIMITS),
                )

    def test_the_discovery_examples_name_where_each_value_comes_from(self):
        # Every Run is shown these examples, so they carry no organism's TaxId,
        # strain or assembly name, only where the value comes from (ADR 0040's
        # Registry).
        (row,) = (
            c
            for c in function_tool_contracts("planning_agent")
            if c.tool_name == "discover_assemblies"
        )

        self.assertEqual(len(row.examples), 3)
        for example in row.examples:
            with self.subTest(example=example):
                self.assertIn('"taxid": <the resolved TaxId>', example)
                self.assertNotRegex(example, r"\d")
        self.assertIn('"assembly_version": "all_assemblies"', row.examples[2])

    def test_no_tool_text_names_a_concrete_assembly(self):
        # A static example names where its value comes from; the Run's own value is
        # in its evidence, so a C. elegans Run is never shown the human accession.
        # Bound values (enums, metadata) are the Run's.
        from tests.application_fakes import _ready_journey
        from tests.sequential_investigation_support import (
            SourceResponsesOnly,
            _AdaptiveInvestigationModel,
            run_attempt,
        )

        def texts(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"description", "examples"}:
                        yield from (item if isinstance(item, list) else [item])
                    elif key not in {"metadata", "enum", "const", "default"}:
                        yield from texts(item)
            elif isinstance(value, list):
                for item in value:
                    yield from texts(item)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runner").mkdir()
            (root / "member").mkdir()
            receipt, _ = _ready_journey(root / "runner")
            _, investigation = run_attempt(
                root / "member", _AdaptiveInvestigationModel(), SourceResponsesOnly()
            )
            requests = [
                *_requests(receipt.journey_file),
                *_requests(investigation.path),
            ]
        shown = {
            (tool["name"], text)
            for request in requests
            for tool in (*request["function_tools"], *request["output_tools"])
            for text in texts(tool)
            if isinstance(text, str)
        }
        self.assertIn("fetch_assembly_report", {name for name, _ in shown})
        self.assertIn("describe_companion", {name for name, _ in shown})
        self.assertEqual(
            {(name, text) for name, text in shown if _ASSEMBLY_ACCESSION.search(text)},
            set(),
        )


if __name__ == "__main__":
    unittest.main()
