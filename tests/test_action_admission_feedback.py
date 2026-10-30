"""Shared action admission: bound arguments and the correction each refusal presents.

Migrated by #172 from the retired Agent action Review module; these cases test
current admission behavior, not the retired Review.
"""

from __future__ import annotations

import copy
import unittest
import uuid

from bio_neb.agentic_journey.continuation_policy import ContinuationPolicy
from bio_neb.failure import CorrectionWording


class ActionAdmissionArgumentTests(unittest.TestCase):
    def test_bound_arguments_leave_the_schema_and_ride_in_unsent_metadata(self):
        from pydantic_ai.tools import ToolDefinition

        from bio_neb.agentic_journey.action_admission import grounded_tool_schema

        original = ToolDefinition(
            name="bounded_choice",
            description="Choose with retained evidence.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "selection": {"type": "string"},
                    "optional_selection": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "candidate": {"type": "string"},
                    "reason": {"type": "string", "minLength": 1},
                    "before": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Optional page bound.",
                    },
                    "nested": {
                        "anyOf": [{"$ref": "#/$defs/Slice"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["selection", "optional_selection", "candidate", "reason"],
                "$defs": {
                    "Slice": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "purpose": {
                                "anyOf": [
                                    {"enum": ["a", "b"], "type": "string"},
                                    {"type": "null"},
                                ],
                                "default": None,
                            },
                        },
                        "required": ["event_id"],
                    }
                },
            },
        )
        unchanged = copy.deepcopy(original)
        constants = {"selection": 'quoted "selection" β', "optional_selection": None}
        grounded = grounded_tool_schema(
            original, constants=constants, enums={"candidate": ("first", "second")}
        )
        schema = grounded.parameters_json_schema

        # Bound values are absent from what the model sees and ride in the
        # definition's metadata, which the request records but never sends.
        self.assertEqual(grounded.description, original.description)
        self.assertEqual(
            set(schema["properties"]), {"candidate", "reason", "before", "nested"}
        )
        self.assertEqual(schema["required"], ["candidate", "reason"])
        self.assertEqual(grounded.metadata, {"bound_arguments": constants})
        self.assertEqual(
            schema["properties"]["candidate"], {"enum": ["first", "second"]}
        )
        self.assertEqual(
            schema["properties"]["reason"], {"type": "string", "minLength": 1}
        )
        # A true optional is omittable and renders no null branch or default.
        self.assertEqual(
            schema["properties"]["before"],
            {"type": "string", "description": "Optional page bound."},
        )
        self.assertEqual(schema["properties"]["nested"], {"$ref": "#/$defs/Slice"})
        self.assertEqual(
            schema["$defs"]["Slice"]["properties"]["purpose"],
            {"enum": ["a", "b"], "type": "string"},
        )
        self.assertNotIn("null", repr(schema))
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(grounded.sequential)
        self.assertEqual(original, unchanged)
        unbound = grounded_tool_schema(original, constants={})
        self.assertEqual(unbound.metadata, {"bound_arguments": {}})
        self.assertEqual(
            unbound.parameters_json_schema["required"],
            original.parameters_json_schema["required"],
        )

    def test_an_empty_or_untyped_choice_is_refused_never_shown(self):
        """ADR 0040's callable-now refinement (#478, binder rule): a choice with no
        admissible value waits on its binder's reason, so the grounded schema
        refuses an empty or untyped choice as our bug."""
        # An empty enum no call can satisfy made the provider refuse the whole
        # request's Tool grammar. Its binder raises
        # ``PreconditionUnmet`` instead, so the Tool shows as waiting; a
        # choice that reaches the schema is non-empty and of one JSON type.
        from pydantic_ai.tools import ToolDefinition

        from bio_neb.agentic_journey.action_admission import grounded_tool_schema
        from bio_neb.failure import Invariant

        original = ToolDefinition(
            name="read_words",
            description="Read the words not yet read.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "source_event_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["source_event_id"],
            },
        )
        for values, check in (
            ((), "an empty choice"),
            (("a", 1), "a choice of mixed types"),
            ((None,), "of no scalar type"),
            (({"a": 1},), "of no scalar type"),
        ):
            with self.subTest(values=values):
                with self.assertRaises(Invariant) as refused:
                    grounded_tool_schema(
                        original, constants={}, enums={"source_event_id": values}
                    )
                self.assertEqual(
                    refused.exception.invariant_id, "tool_choice_ungrounded"
                )
                self.assertIn(check, str(refused.exception))
                self.assertIn("read_words.source_event_id", str(refused.exception))
        typed = grounded_tool_schema(
            original, constants={}, enums={"source_event_id": ("a", "b")}
        )
        self.assertEqual(
            typed.parameters_json_schema["properties"]["source_event_id"],
            {"enum": ["a", "b"]},
        )

    def test_no_offered_tool_asks_the_provider_for_strict_grammar(self):
        # A strict Tool put every Tool under the provider's grammar, which fixed the
        # argument order: a call begun with `source` could no longer name
        # `sequences`.
        from pydantic_ai.tools import ToolDefinition

        from bio_neb.agentic_journey.action_admission import (
            grounded_tool_schema,
            output_tool_schema,
        )

        definition = ToolDefinition(
            name="fully_required",
            description="Every argument required.",
            parameters_json_schema={
                "type": "object",
                "properties": {"accession": {"type": "string"}},
                "required": ["accession"],
                "additionalProperties": False,
            },
        )
        self.assertIs(grounded_tool_schema(definition, constants={}).strict, False)
        self.assertIs(output_tool_schema(definition).strict, False)

    def test_admission_fills_omitted_bound_arguments_and_refuses_contradictions(self):
        from pydantic_ai import ModelRetry

        from bio_neb.agentic_journey.action_admission import (
            ActionAdmissionSession,
            ProposedAgentAction,
        )
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from tests.agent_attempt_fakes import FakeActivity, FakeJourney

        journey = FakeJourney([])
        domain_seen = []

        def domain(action):
            domain_seen.append(dict(action.arguments))
            return ()

        session = ActionAdmissionSession(
            wording=CorrectionWording(),
            attempt_id=str(uuid.uuid4()),
            parent_activity=FakeActivity(journey),
            failure_source=AttemptFailureSource(),
            eligible_tool_names=frozenset({"pick"}),
            action_validator=domain,
            continuation=ContinuationPolicy(failed_turns=4),
        )
        bound = {"pick": {"basis_event_id": "basis-1", "asset_event_id": None}}

        def admit(call_id, arguments):
            return session.admit_turn(
                str(uuid.uuid4()),
                [ProposedAgentAction("pick", call_id, arguments)],
                frozenset({"pick"}),
                bound_arguments=bound,
            )

        # "None" for a bound null is refused before the domain or any contact;
        # the record keeps what the model sent.
        with self.assertRaises(ModelRetry) as refused:
            admit("call-none", {"asset_event_id": "None", "reason": "r"})
        self.assertIn("asset_event_id:bound_value_mismatch", str(refused.exception))
        # An omitted bound key is filled from the request's binding.
        self.assertEqual(admit("call-omit", {"reason": "r"}), "singleton_admitted")
        # A bound key sent anyway is admitted when exact.
        self.assertEqual(
            admit("call-exact", {"basis_event_id": "basis-1", "reason": "s"}),
            "singleton_admitted",
        )

        validations = [
            event["payload"]
            for event in journey.timeline
            if event["event_type"] == "ToolCallValidated"
        ]
        effective = {"basis_event_id": "basis-1", "asset_event_id": None}
        self.assertEqual(
            [
                (v["arguments"], v["bound_argument_names"], v["errors"])
                for v in validations
            ],
            [
                (
                    {
                        "asset_event_id": "None",
                        "reason": "r",
                        "basis_event_id": "basis-1",
                    },
                    ["asset_event_id", "basis_event_id"],
                    ["asset_event_id:bound_value_mismatch"],
                ),
                (
                    {**effective, "reason": "r"},
                    ["asset_event_id", "basis_event_id"],
                    [],
                ),
                (
                    {**effective, "reason": "s"},
                    ["asset_event_id", "basis_event_id"],
                    [],
                ),
            ],
        )
        self.assertEqual(
            domain_seen, [{**effective, "reason": "r"}, {**effective, "reason": "s"}]
        )
        # Dispatch receives the admitted effective arguments, not the literal call.
        self.assertEqual(
            session.admitted_arguments("call-omit"), {**effective, "reason": "r"}
        )

    def test_reopen_rebuilds_effective_arguments_from_the_recorded_request(self):
        from bio_neb.agentic_journey.action_admission import (
            request_bound_arguments,
            validated_model_call,
        )

        call = {"tool_name": "pick", "tool_call_id": "call-1", "args": '{"reason":"r"}'}
        request_tools = [
            {"name": "pick", "metadata": {"bound_arguments": {"basis_event_id": "b"}}},
            {"name": "search_journey", "metadata": {"bound_arguments": {}}},
        ]
        bound = request_bound_arguments(request_tools)
        self.assertEqual(bound, {"pick": {"basis_event_id": "b"}, "search_journey": {}})
        validation = {
            "tool_name": "pick",
            "tool_call_id": "call-1",
            "arguments": {"reason": "r", "basis_event_id": "b"},
            "bound_argument_names": ["basis_event_id"],
        }
        action = validated_model_call(call, validation, bound)
        assert action is not None
        self.assertEqual(dict(action.arguments), validation["arguments"])
        # A validation that disagrees with what the request bound is not that call.
        for forged in (
            {**validation, "arguments": {"reason": "r", "basis_event_id": "x"}},
            {**validation, "bound_argument_names": []},
            {**validation, "arguments": {"reason": "r"}},
        ):
            with self.subTest(forged=forged):
                self.assertIsNone(validated_model_call(call, forged, bound))
        # Every offered Tool records its bound values, possibly none; a
        # definition without that record is refused, not read as unbound.
        with self.assertRaisesRegex(
            ValueError, "offered Tool pick records no bound arguments"
        ):
            request_bound_arguments([{"name": "pick", "metadata": None}])


class AsCalledArgumentTests(unittest.TestCase):
    def test_an_absent_bound_value_is_filled_in_for_null_and_blank_too(self):
        """#476 (owner, 2026-10-28, G3; Spec r1 SP1): omitted, null and blank are
        one absent value before a bound value is filled in; a real conflicting
        value is kept for admission to refuse."""
        from bio_neb.agentic_journey.action_policy import effective_arguments

        binding = {"input_set_id": "input-1"}
        self.assertEqual(effective_arguments({}, binding), binding)
        for value in (None, ""):
            with self.subTest(sent=value):
                self.assertEqual(
                    effective_arguments({"input_set_id": value}, binding), binding
                )
        self.assertEqual(
            effective_arguments({"input_set_id": "input-2"}, binding),
            {"input_set_id": "input-2"},
        )


# Every counted correction ends with its count line; these sessions allow six
# failed turns in a row and each refusal is the first.
COUNT_1_OF_6 = (
    "\nCount: failed turn 1 of 6 in a row. At 6 the Attempt stops and the "
    "researcher reviews it."
)


class CorrectionFeedbackTests(unittest.TestCase):
    """The shared session builds each correction from the fault and the current stage.

    Each names the Tool, the argument or slot, its value and the reason.
    """

    OFFER = frozenset({"pick", "search_journey"})

    def setUp(self):
        from tests.agent_attempt_fakes import FakeJourney

        self.journey = FakeJourney([])

    def _session(self, errors=(), **options):
        from bio_neb.agentic_journey.action_admission import ActionAdmissionSession
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from tests.agent_attempt_fakes import FakeActivity

        return ActionAdmissionSession(
            wording=CorrectionWording(),
            attempt_id=str(uuid.uuid4()),
            parent_activity=FakeActivity(self.journey),
            failure_source=AttemptFailureSource(),
            eligible_tool_names=frozenset({"pick", "drop", "search_journey"}),
            action_validator=lambda _action: errors,
            continuation=ContinuationPolicy(failed_turns=6),
            **options,
        )

    def _feedback(self, session, tool_name, arguments=None, bound=None):
        from pydantic_ai import ModelRetry

        from bio_neb.agentic_journey.action_admission import ProposedAgentAction

        with self.assertRaises(ModelRetry) as refused:
            session.admit_turn(
                str(uuid.uuid4()),
                [ProposedAgentAction(tool_name, "call", arguments or {})],
                self.OFFER,
                bound_arguments=bound,
            )
        return str(refused.exception)

    def test_a_tool_that_does_not_exist_names_the_offer(self):
        feedback = self._feedback(self._session(), "select_companion_asset")
        self.assertEqual(
            feedback,
            "Refused before execution (tool_name_ineligible).\n"
            "Tool `select_companion_asset`: no such tool.\n"
            "Allowed: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_a_tool_not_offered_now_names_the_offer(self):
        feedback = self._feedback(self._session(), "drop")
        self.assertEqual(
            feedback,
            "Refused before execution (tool_name_unavailable).\n"
            "Tool `drop`: not offered now.\n"
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_invalid_arguments_name_the_offending_argument(self):
        feedback = self._feedback(
            self._session(),
            "pick",
            {"asset_event_id": "None", "reason": "r"},
            bound={"pick": {"asset_event_id": None}},
        )
        self.assertEqual(
            feedback,
            "Refused before execution (asset_event_id:bound_value_mismatch).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            '- Field `asset_event_id`: you sent "None".\n'
            "  Allowed form: omit it; the server supplies it.\n"
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_a_structure_error_names_its_argument(self):
        # The reason drops pydantic's "Value error, " prefix; the code keeps
        # the message as recorded.
        error = "tool_arguments_invalid:requirements.1:value_error:Value error, bad"
        feedback = self._feedback(self._session((error,)), "pick", {"reason": "r"})
        self.assertEqual(
            feedback,
            "Refused before execution (tool_arguments_invalid).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            "- Field `requirements.1`: you sent nothing.\n"
            "  Allowed form: bad.\n"
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_a_whole_call_error_names_the_argument_it_is_about(self):
        # A model-level error has an empty location; it names the call's one argument
        # and its value, never "``" and "nothing".
        error = (
            "tool_arguments_invalid::value_error:Value error, requirement subject "
            "keys must be unique; repeated: goal"
        )
        feedback = self._feedback(
            self._session((error,)), "pick", {"requirements": [{"subject_key": "goal"}]}
        )
        self.assertIn(
            '- Field `requirements`: you sent [{"subject_key": "goal"}].\n'
            "  Allowed form: requirement subject keys must be unique; repeated: goal.\n",
            feedback,
        )

    def test_a_structure_error_shows_the_value_sent(self):
        error = "tool_arguments_invalid:requirements.0.kind:enum:Input should be 'a'"
        feedback = self._feedback(
            self._session((error,)), "pick", {"requirements": [{"kind": "b"}]}
        )
        self.assertIn(
            '- Field `requirements.0.kind`: you sent "b".\n'
            "  Allowed form: Input should be 'a'.\n",
            feedback,
        )

    def test_an_unmet_precondition_names_its_slot_value_and_reason(self):
        explained = []

        def explain(action, code):
            explained.append((action.tool_name, code))
            return '`reference discovery` is "sample_identity": discover first.'

        session = self._session(
            precondition=lambda action: (
                "reference_discovery_not_open" if action.tool_name == "pick" else None
            ),
            explain=explain,
        )
        self.assertEqual(
            self._feedback(session, "pick", {"reason": "r"}),
            "Refused before execution (reference_discovery_not_open).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            '- `reference discovery` is "sample_identity": discover first.\n'
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )
        self.assertEqual(explained, [("pick", "reference_discovery_not_open")])

    def test_an_unmet_precondition_precedes_bound_values_and_arguments(self):
        session = self._session(
            ("domain_refusal",),
            precondition=lambda _action: "not_yet",
            argument_validator=lambda _action: ("tool_arguments_invalid:x:missing:m",),
        )
        feedback = self._feedback(
            session, "pick", {"asset": "wrong"}, bound={"pick": {"asset": "right"}}
        )
        self.assertTrue(feedback.startswith("Refused before execution (not_yet).\n"))

    def test_a_tool_of_another_stage_names_its_stage(self):
        session = self._session(
            explain=lambda _action, code: (
                "It works in the `reference` stage; the current stage is `companions`."
                if code == "tool_name_unavailable"
                else None
            )
        )
        self.assertEqual(
            self._feedback(session, "drop"),
            "Refused before execution (tool_name_unavailable).\n"
            "Tool `drop`: not offered now. It works in the `reference` stage; the "
            "current stage is `companions`.\n"
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_a_domain_refusal_keeps_its_reason_and_the_offer(self):
        feedback = self._feedback(
            self._session(("taxon_query_ungrounded",)), "pick", {"reason": "r"}
        )
        self.assertEqual(
            feedback,
            "Refused before execution (taxon_query_ungrounded).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_repair_guidance_reaches_only_its_refused_offered_tool(self):
        from pydantic_ai import ModelRetry

        from bio_neb.agentic_journey.action_admission import ProposedAgentAction

        guidance = {"pick": "For pick: name a reason.", "drop": "For drop: never."}
        session = self._session(("pick_reason_missing",), repair_guidance=guidance)
        self.assertEqual(
            self._feedback(session, "pick", {"reason": ""}),
            "Refused before execution (pick_reason_missing).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            "Allowed now: one of `pick`, `search_journey`.\n"
            "For pick: name a reason." + COUNT_1_OF_6,
        )
        # The un-offered Tool's own guidance is not shown, nor another Tool's.
        for refused in ("drop", "search_journey"):
            with self.subTest(refused=refused):
                feedback = self._feedback(session, refused)
                self.assertNotIn("For pick", feedback)
                self.assertNotIn("For drop", feedback)
        # Malformed arguments take precedence over the offer check; the un-offered
        # Tool's guidance still stays out.
        with self.assertRaises(ModelRetry) as refused:
            session.admit_turn(
                str(uuid.uuid4()),
                [ProposedAgentAction("drop", "call", {"invalid_json": "{"}, True)],
                self.OFFER,
            )
        self.assertNotIn("For drop", str(refused.exception))

    def test_malformed_arguments_say_so(self):
        from pydantic_ai import ModelRetry

        from bio_neb.agentic_journey.action_admission import ProposedAgentAction

        with self.assertRaises(ModelRetry) as refused:
            self._session().admit_turn(
                str(uuid.uuid4()),
                [ProposedAgentAction("pick", "call", {"invalid_json": "{"}, True)],
                self.OFFER,
            )
        self.assertEqual(
            str(refused.exception),
            "Refused before execution (tool_arguments_invalid_json).\n"
            "Tool `pick` was not run. Fix each item, then call it again.\n"
            "- Arguments: not one JSON object.\n"
            "  Allowed form: one JSON object of argument names to values, for "
            'example {"name": "value"}.\n'
            "Allowed now: one of `pick`, `search_journey`." + COUNT_1_OF_6,
        )

    def test_the_offer_names_the_output_tool_and_the_tools_that_wait(self):
        # Every refusal lists the output Tool as offered (members never read
        # `compile_proposal`), and the hook names the Tools whose step is not
        # current, so the writer can tell callable from waiting.
        session = self._session(
            ("pick_reason_missing",),
            output_tool_names=frozenset({"record_plan"}),
            waiting=lambda names: frozenset(names) & {"pick"},
        )
        self.assertEqual(
            self._feedback(session, "pick", {"reason": ""}),
            "Refused before execution (pick_reason_missing).\n"
            "Tool `pick` was not run: it waits on its step. Call an allowed Tool now.\n"
            "Allowed now: one of `record_plan`, `search_journey`.\n"
            "Waiting on their step, not callable now: `pick`." + COUNT_1_OF_6,
        )

    def test_a_reason_sentence_names_the_facts_its_refusal_holds(self):
        # D2: a refusal's typed facts are rendered into its sentence once, for
        # every model; a Tool list reads as Tools, an empty one as none.
        from bio_neb.failure import sentence_with_facts

        self.assertEqual(
            sentence_with_facts(
                "Add the missing kinds: {missing}; call {tools}.",
                {"missing": ("taxon_identity", "resource_profile"), "tools": ()},
            ),
            "Add the missing kinds: `taxon_identity`, `resource_profile`; call none.",
        )
        self.assertEqual(
            sentence_with_facts("The step is {step}.", {"step": "recipe validation"}),
            "The step is recipe validation.",
        )
        with self.assertRaises(KeyError):
            sentence_with_facts("Call {tools}.", {})

    def test_a_truncated_reply_is_told_to_call_an_offered_tool(self):
        from bio_neb.failure import ReplyCutOff, render_correction

        self.assertEqual(
            render_correction(ReplyCutOff(32768, self.OFFER), CorrectionWording()),
            "Your reply was cut at 32768 tokens before any tool call.\n"
            "Allowed form: a short reply that calls one of `pick`, `search_journey`.",
        )


class ModelCorrectionWordingTests(unittest.TestCase):
    """Each model's correction wording through the one writer (#222, #242, #336, #341).

    Every correction kind is rendered from one fault in each model's endpoint
    policy wording: the one text names the field, the value sent and the allowed
    form, one fact per line, for every model (#445).
    """

    OFFER = frozenset({"pick", "search_journey"})
    ERRORS = (
        {"type": "missing", "loc": ("sample",), "msg": "Field required", "input": {}},
    )
    DUMPED = (
        "```json\n[\n  {\n"
        '    "type": "missing",\n    "loc": [\n      "sample"\n    ],\n'
        '    "msg": "Field required",\n    "input": {}\n  }\n]\n```'
    )

    def _faults(self):
        from bio_neb.failure import (
            ActionRefused,
            FaultCount,
            OutputCalls,
            OutputFaults,
            OutputRefused,
            RepeatedCall,
            ReplyCutOff,
            TextEnding,
            TurnFaults,
        )

        invalid = "tool_arguments_invalid:requirements.0.kind:enum:Input should be 'a'"
        unoffered = ActionRefused(
            "drop",
            ("tool_name_unavailable",),
            {},
            self.OFFER,
            explanations={
                "tool_name_unavailable": "It works in the `reference` stage; "
                "the current stage is `input`."
            },
        )
        return {
            "tool_unknown": ActionRefused(
                "select_companion_asset", ("tool_name_ineligible",), {}, self.OFFER
            ),
            "tool_unoffered": unoffered,
            "argument_server_bound": ActionRefused(
                "pick",
                ("asset_event_id:bound_value_mismatch",),
                {"asset_event_id": "None"},
                self.OFFER,
            ),
            # The invocation's own sentence for a server-bound value.
            "argument_server_bound_explained": ActionRefused(
                "pick",
                ("asset_event_id:bound_value_mismatch",),
                {"asset_event_id": "x"},
                self.OFFER,
                explanations={
                    "asset_event_id:bound_value_mismatch": "Argument "
                    '`asset_event_id` has effective value "x", but its current '
                    'server-bound value is "y".'
                },
            ),
            # A precondition's sentence and the Tool's guidance.
            "precondition": ActionRefused(
                "pick",
                ("sample_unattested",),
                {},
                self.OFFER,
                explanations={
                    "sample_unattested": "The sample is not attested yet; call "
                    "attest_sample first."
                },
                guidance="For pick: preserve valid fields.",
            ),
            "argument_invalid": ActionRefused(
                "pick",
                (invalid,),
                {"requirements": [{"kind": "b"}]},
                self.OFFER,
                guidance="For pick: preserve valid fields.",
            ),
            "arguments_not_object": ActionRefused(
                "pick", ("tool_arguments_invalid_json",), {}, self.OFFER
            ),
            "truncation": ReplyCutOff(32768, self.OFFER),
            "text_ending": TextEnding(self.OFFER),
            "output_schema": OutputRefused(schema_errors=self.ERRORS),
            "output_refused": OutputRefused(reason="The plan names no sample."),
            # A role's refusal as facts: codes, one sentence each, fields.
            "output_faults": OutputRefused(
                faults=OutputFaults(
                    codes=(
                        "claims:extra_forbidden",
                        "explanation_qualifications_missing:7",
                    ),
                    sentences=(
                        "claims is no field of this output: remove it.",
                        "Qualification 7 (gap) is in no limitation.",
                    ),
                    fields=("answer", "answer_references", "limitations"),
                ),
                tool_name="record_explanation",
            ),
            # The earlier call's own status and next step, from its model
            # view.
            "repeated_call": RepeatedCall(
                "pick", 2, 3, self.OFFER, "picked", "Call record_plan with it."
            ),
            # A refusal whose Tool waits on its step, beside an output Tool.
            "precondition_waiting": ActionRefused(
                "pick",
                ("step_not_current",),
                {},
                self.OFFER | {"record_plan"},
                explanations={"step_not_current": "Its step comes later."},
                waiting=frozenset({"pick"}),
            ),
            "repeated_call_waiting": RepeatedCall(
                "search_journey",
                1,
                3,
                self.OFFER | {"record_plan"},
                "read",
                "Read the matched events above and cite their event ids.",
                waiting=frozenset({"pick"}),
            ),
            "copied_calls": TurnFaults(
                copies=(("pick", 3),), copies_fault=FaultCount(1, 6)
            ),
            "unoffered_calls": TurnFaults(
                unoffered=(unoffered,), unoffered_fault=FaultCount(1, 6)
            ),
            "output_not_alone": TurnFaults(
                outputs=("record_plan",), outputs_fault=FaultCount(1, 6)
            ),
            "cut_off_partial": TurnFaults(
                cut_off=ReplyCutOff(221319, self.OFFER, calls=3),
                dropped_tool="pick",
                cut_off_fault=FaultCount(1, 2),
            ),
            "cut_off_no_call": TurnFaults(
                cut_off=ReplyCutOff(32768, self.OFFER), cut_off_fault=FaultCount(1, 2)
            ),
            "model_resend": TurnFaults(sent=2, resend_fault=FaultCount(1, 5)),
            # The fault notices' facts folded into the one correction.
            "precondition_last": ActionRefused(
                "pick",
                ("sample_unattested",),
                {},
                self.OFFER,
                explanations={"sample_unattested": "The sample is not attested yet."},
                fault=FaultCount(5, 6),
            ),
            "tool_unknown_dropped": ActionRefused(
                "read_it",
                ("tool_name_ineligible",),
                {"path": "x"},
                self.OFFER,
                dropped=True,
            ),
            "placeholder": ActionRefused(
                "pick",
                (
                    "tool_arguments_invalid:asset_event_id:string_pattern_mismatch:"
                    "String should match a UUID",
                ),
                {"asset_event_id": "<EVENT_ID>"},
                self.OFFER,
            ),
            "literal_value": ActionRefused(
                "pick",
                ("request_assertion_ungrounded",),
                {"sample": "hg001"},
                self.OFFER,
                explanations={
                    "request_assertion_ungrounded": "`hg001` is not in the request."
                },
            ),
            "text_ending_last": TextEnding(self.OFFER, FaultCount(5, 6)),
            "cut_off_text_call": TurnFaults(
                cut_off=ReplyCutOff(32768, self.OFFER, text_call=True, blank_run=1500),
                cut_off_fault=FaultCount(1, 2),
            ),
            "output_calls": OutputCalls("record_plan", 2, FaultCount(1, 5)),
            "output_refused_last": OutputRefused(
                reason="The plan names no sample.", fault=FaultCount(1, 2)
            ),
        }

    def _rendered(self, model: str) -> dict[str, str]:
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy
        from bio_neb.failure import render_correction

        wording = model_endpoint_policy(model).correction_wording
        return {
            kind: render_correction(fault, wording)
            for kind, fault in self._faults().items()
        }

    def _assert_each(self, rendered: dict[str, str], expected: dict[str, str]):
        self.assertEqual(set(rendered), set(expected))
        for kind, text in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(rendered[kind], text)

    def test_nemotron_names_the_field_the_value_sent_and_the_allowed_form(self):
        offered = "`pick`, `search_journey`"
        fix = "Tool `pick` was not run. Fix each item, then call it again.\n"
        unoffered = (
            "Refused before execution (tool_name_unavailable).\n"
            "Tool `drop`: not offered now. It works in the `reference` stage; "
            f"the current stage is `input`.\nAllowed now: one of {offered}."
        )
        truncation = (
            "Your reply was cut at 32768 tokens before any tool call.\nAllowed "
            f"form: a short reply that calls one of {offered}."
        )

        # The stop rule is each fault's firm line at its last correction.
        def last(kind):
            del kind  # every fault is one more failed turn
            return (
                "\nLast correction: one more failed turn stops the Attempt for review."
            )

        # Every counted correction ends with its count line.
        def count(n, limit):
            return (
                f"\nCount: failed turn {n} of {limit} in a row. At {limit} the "
                "Attempt stops and the researcher reviews it."
            )

        same_arguments = "This call sends the same arguments, so it adds no evidence.\n"
        text_ending = (
            "Your reply ended in text without a tool call; text is not a result, so "
            "nothing happened.\nA step happens only through a tool call: make the "
            f"call that does it.\nAllowed form: a tool call to one of {offered}."
        )
        self._assert_each(
            self._rendered("nvidia/Nemotron-3-Ultra-550b-a55b"),
            {
                "tool_unknown": "Refused before execution (tool_name_ineligible).\n"
                "Tool `select_companion_asset`: no such tool.\n"
                f"Allowed: one of {offered}.",
                "tool_unoffered": unoffered,
                "argument_server_bound": "Refused before execution "
                "(asset_event_id:bound_value_mismatch).\n"
                + fix
                + '- Field `asset_event_id`: you sent "None".\n'
                "  Allowed form: omit it; the server supplies it.\n"
                f"Allowed now: one of {offered}.",
                # One fact per line, codes only in the refused line.
                "argument_server_bound_explained": "Refused before execution "
                "(asset_event_id:bound_value_mismatch).\n"
                + fix
                + '- Field `asset_event_id`: you sent "x".\n'
                "  Allowed form: omit it; the server supplies it.\n"
                f"Allowed now: one of {offered}.",
                "precondition": "Refused before execution (sample_unattested).\n"
                + fix
                + "- The sample is not attested yet; call attest_sample first.\n"
                f"Allowed now: one of {offered}.\nFor pick: preserve valid fields.",
                "argument_invalid": "Refused before execution "
                "(tool_arguments_invalid).\n"
                + fix
                + '- Field `requirements.0.kind`: you sent "b".\n'
                "  Allowed form: Input should be 'a'.\n"
                f"Allowed now: one of {offered}.\nFor pick: preserve valid fields.",
                "arguments_not_object": "Refused before execution "
                "(tool_arguments_invalid_json).\n"
                + fix
                + "- Arguments: not one JSON object.\n"
                "  Allowed form: one JSON object of argument names to values, "
                'for example {"name": "value"}.\n'
                f"Allowed now: one of {offered}.",
                "truncation": truncation,
                "text_ending": text_ending,
                "output_schema": "Your output has 1 error. In each, `loc` is the "
                "field, `input` is what you sent and `msg` is the allowed form:\n"
                + self.DUMPED,
                "output_refused": "Output refused: The plan names no sample.",
                "output_faults": "Output refused (claims:extra_forbidden, "
                "explanation_qualifications_missing:7).\n"
                "- claims is no field of this output: remove it.\n"
                "- Qualification 7 (gap) is in no limitation.\n"
                "Allowed form: `record_explanation` again with its fields answer, "
                "answer_references, limitations, each fault above fixed.",
                "repeated_call": "Refused before execution (repeated_call).\n"
                "Tool `pick` was not run: your earlier `pick` call with the same "
                "arguments ended `picked`, and nothing has changed since. Its next "
                "step: Call record_plan with it.\n"
                + same_arguments
                + f"Allowed now: one of {offered}."
                + count(2, 3)
                + last("repeat"),
                # Only the Tools callable now are allowed; the others wait.
                "precondition_waiting": "Refused before execution "
                "(step_not_current).\nTool `pick` was not run: it waits on its step. "
                "Call an allowed Tool now.\n- Its step comes later.\n"
                "Allowed now: one of `record_plan`, `search_journey`.\n"
                "Waiting on their step, not callable now: `pick`.",
                "repeated_call_waiting": "Refused before execution (repeated_call).\n"
                "Tool `search_journey` was not run: your earlier `search_journey` "
                "call with the same arguments ended `read`, and nothing has changed "
                "since. Its next step: Read the matched events above and cite their "
                "event ids.\n"
                + same_arguments
                + "Allowed now: one of `record_plan`, `search_journey`.\nWaiting on "
                "their step, not callable now: `pick`." + count(1, 3),
                # The turn's one note: one fact per line, each item ending in its
                # count, and in its firm line at the last allowed count.
                "copied_calls": "Copied calls: you sent `pick` 3 times.\n"
                "Kept: the first of each; the copies were dropped.\n"
                "Allowed form: each distinct call once per reply." + count(1, 6),
                "unoffered_calls": "Unoffered calls: this stage does not offer "
                "these Tools; each was refused before execution.\n"
                + unoffered
                + count(1, 6),
                "output_not_alone": "Output `record_plan`: sent with other calls "
                "in one reply; not accepted.\nAllowed form: `record_plan` alone in "
                "its own reply, after you read this turn's results." + count(1, 6),
                "cut_off_partial": "Your reply was cut at 221319 tokens; it held 3 "
                "tool calls.\nLast call `pick`: it reached the ceiling and was "
                "dropped.\nAllowed form: call `pick` again in a later reply if you "
                "still need it." + count(1, 2) + last("cut-off"),
                "cut_off_no_call": truncation + count(1, 2) + last("cut-off"),
                "model_resend": "No reply arrived: the provider timed out or "
                "dropped the connection.\nSent: this request 2 times.\n"
                "Count: resend fault 1 of 5.",
                # The notices' facts, each a line of its fault's one correction.
                "precondition_last": "Refused before execution (sample_unattested).\n"
                + fix
                + "- The sample is not attested yet.\n"
                f"Allowed now: one of {offered}." + count(5, 6) + last("same-refusal"),
                # E2: the issue with its values, and what the dropped call sent.
                "tool_unknown_dropped": "Refused before execution "
                "(tool_name_ineligible).\nTool `read_it`: no such tool.\n"
                'You sent it with {"path": "x"}.\n'
                f"Allowed: one of {offered}.",
                "placeholder": "Refused before execution (tool_arguments_invalid).\n"
                + fix
                + '- Field `asset_event_id`: you sent the placeholder "<EVENT_ID>".\n'
                "  Allowed form: the value another call returns. Call that tool "
                "first, read its result, then send the real value in a later "
                f"reply.\nAllowed now: one of {offered}.",
                # The per-surface table's Nemotron row (literal_guard, E5).
                "literal_value": "Refused before execution "
                "(request_assertion_ungrounded).\n"
                + fix
                + "- `hg001` is not in the request.\n"
                f"Allowed now: one of {offered}.\nExact values: copy each value "
                "and quote from the request character for character, keeping its "
                "spelling, case, spaces and punctuation.",
                "text_ending_last": text_ending + count(5, 6) + last("text-ending"),
                "cut_off_text_call": "Your reply was cut at 32768 tokens before any "
                "tool call.\nIt began a tool call as text in the message. Send each "
                "call in the tool-call field, with no text before it.\nIt held a run "
                "of 1500 blank characters. End each reply right after its last tool "
                f"call.\nAllowed form: a short reply that calls one of {offered}."
                + count(1, 2)
                + last("cut-off"),
                # E4: several outputs in one reply.
                "output_calls": "Outputs: you sent `record_plan` 2 times in one "
                "reply; none was accepted.\nAllowed form: one `record_plan` call "
                "alone in its own reply." + count(1, 5),
                # E3 for an output refusal at the member's last repair.
                "output_refused_last": "Output refused: The plan names no sample."
                + count(1, 2)
                + last("output"),
            },
        )

    def test_every_registered_model_reads_the_one_text(self):
        # One default wording, complete for every model; an entry states only
        # what differs, and none does today.
        from bio_neb.agentic_journey.model_settings import model_registry

        nemotron = self._rendered("nvidia/Nemotron-3-Ultra-550b-a55b")
        for model in model_registry().entries:
            with self.subTest(model=model):
                self.assertEqual(self._rendered(model), nemotron)


class FailedTurnCountTests(unittest.TestCase):
    """Every fault of one reply names that reply's place in the failed turns in a
    row (owner, #447 B); its key still names the call: Tool, arguments and reasons.

    #379's s2 spike (Journey 7091711a, reply 6): six distinct `request_planning_hitl`
    calls refused for the same reason in one reply are one failed turn, not six.
    """

    REFUSAL = ("reference_identity_resolved",)
    OFFER = frozenset({"request_planning_hitl"})

    def setUp(self):
        from bio_neb.agentic_journey.action_admission import ActionAdmissionSession
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from tests.agent_attempt_fakes import FakeActivity, FakeJourney
        from tests.reference_fakes import append

        self.journey = FakeJourney([])
        self.attempt_id = str(uuid.uuid4())
        self.policy = ContinuationPolicy(failed_turns=6)
        append(
            self.journey.timeline,
            "attempt-started",
            "AttemptStarted",
            {
                "attempt_id": self.attempt_id,
                "continuation_policy": self.policy.model_dump(mode="json"),
                "continuation_policy_id": self.policy.policy_id,
            },
        )
        self.session = ActionAdmissionSession(
            wording=CorrectionWording(),
            attempt_id=self.attempt_id,
            parent_activity=FakeActivity(self.journey),
            failure_source=AttemptFailureSource(),
            eligible_tool_names=self.OFFER,
            action_validator=lambda _action: self.REFUSAL,
            continuation=self.policy,
            events=lambda: self.journey.timeline,
        )

    def _reply(self, requirement_ids):
        from bio_neb.agentic_journey.action_admission import ProposedAgentAction
        from tests.reference_fakes import append

        append(
            self.journey.timeline,
            f"request-{uuid.uuid4()}",
            "ModelRequestRecorded",
            {"attempt_id": self.attempt_id},
        )
        actions = [
            ProposedAgentAction(
                "request_planning_hitl",
                f"call-{index}",
                {"requirement_id": requirement_id},
            )
            for index, requirement_id in enumerate(requirement_ids)
        ]
        self.session.admit_turn(str(uuid.uuid4()), actions, self.OFFER)
        return [action.tool_call_id for action in actions]

    def _dispositions(self):
        return [
            (event["payload"]["correction_key"], event["payload"]["correction_count"])
            for event in self.journey.timeline
            if event["event_type"] == "AgentActionTurnDispositionRecorded"
        ]

    def test_distinct_calls_refused_in_one_reply_are_one_failed_turn(self):
        from pydantic_ai import ToolFailed

        requirement_ids = [
            "analysis_recipe",
            "companion_asset",
            "resource_profile",
            "rs7412",
            "rs429358",
            "rs1801133",
        ]
        for call_id in self._reply(requirement_ids):
            with self.assertRaises(ToolFailed):
                self.session.admit_call(call_id)

        dispositions = self._dispositions()
        self.assertEqual([count for _key, count in dispositions], [1] * 6)
        self.assertEqual(len({key for key, _count in dispositions}), 6)

    def test_the_limit_th_failed_turn_in_a_row_stops_before_the_next_request(self):
        from pydantic_ai import ToolFailed

        from bio_neb.agentic_journey.agent_work_review import AgentFailedTurns
        from bio_neb.agentic_journey.continuation_policy import (
            failed_turns_spent,
            validate_continuation,
        )

        def spent():
            return failed_turns_spent(
                self.journey.timeline, attempt_id=self.attempt_id, policy=self.policy
            )

        for _ in range(6):
            self.assertFalse(spent())
            for call_id in self._reply(["rs7412", "rs429358"]):
                # Admission records and corrects; it never stops the Attempt.
                with self.assertRaises(ToolFailed):
                    self.session.admit_call(call_id)

        counts = [count for _key, count in self._dispositions()]
        self.assertEqual(counts, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        self.assertTrue(spent())
        with self.assertRaises(AgentFailedTurns) as stopped:
            validate_continuation(
                self.journey.timeline,
                attempt_id=self.attempt_id,
                policy=self.policy,
                next_request=True,
            )
        self.assertEqual(stopped.exception.cause, "failed_turns")
        self.assertEqual(
            stopped.exception.source_event_id,
            [
                event["event_id"]
                for event in self.journey.timeline
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
            ][-1],
        )


class OutputRefusalReadingTests(unittest.TestCase):
    """A role's own refusal of a schema-refused output reads the arguments as
    the Model turn reads them (ADR 0043: one reader; Standards ST2 r1, #448)."""

    def _refuse(self, output):
        import asyncio

        from pydantic import ValidationError
        from pydantic_ai.exceptions import ModelRetry

        from bio_neb.agentic_journey.action_admission import (
            PydanticActionAdmissionCapability,
        )
        from bio_neb.failure import OutputFaults
        from bio_neb.reporting.explanation_draft import MemberDraft

        read = []

        def refusal(arguments):
            read.append(arguments)
            return OutputFaults(
                codes=("answer:missing",),
                sentences=("answer is missing: add it.",),
                fields=tuple(MemberDraft.model_fields),
            )

        capability = PydanticActionAdmissionCapability(
            lambda _ctx: None,
            received=lambda _ctx: None,
            observe_outputs=lambda *_args: None,
            wording=CorrectionWording(),
            output_tool_names=frozenset({"record_explanation"}),
            output_refusal=refusal,
        )
        try:
            MemberDraft.model_validate({})
        except ValidationError as error:
            invalid = error
        with self.assertRaises(ModelRetry) as raised:
            asyncio.run(
                capability.on_output_validate_error(
                    None, output_context=None, output=output, error=invalid
                )
            )
        return read, str(raised.exception)

    def test_each_supported_form_reaches_the_role_as_the_model_turn_reads_it(self):
        from bio_neb.agentic_journey.model_turn import read_arguments

        for output in ("", '{"answer": "x"}', '```json\n{"answer": "x"}\n```', {}):
            with self.subTest(output=output):
                read, correction = self._refuse(output)
                self.assertEqual(read, [read_arguments(output).value])
                self.assertTrue(
                    correction.startswith("Output refused (answer:missing).")
                )

    def test_a_call_that_is_no_json_object_keeps_the_schemas_correction(self):
        read, correction = self._refuse("not json")
        self.assertEqual(read, [])
        self.assertTrue(correction.startswith("Your output has"))
