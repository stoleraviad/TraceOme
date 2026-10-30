"""The registered injection patterns the Nemotron profile switches on (#243, ADR 0048).

Each pattern's decision is tested through ``injection_messages`` with signals built
here; the slot's reading of those signals from a real Attempt is tested through the
planning assignment with a scripted Nemotron.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from bio_neb.agentic_journey import (  # noqa: F401
    injection,
    injection_patterns,
    model_settings,
)
from bio_neb.agentic_journey.injection import (
    InjectionSignals,
    LatestReply,
    ReplyCall,
    ReplyRefusal,
    ToolResult,
    injection_messages,
    resolve_injection,
)
from bio_neb.agentic_journey.model_settings import (
    NEMOTRON_ULTRA_ENDPOINT_POLICY,
    model_registry,
    use_model_registry,
)
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint

NEMOTRON = NEMOTRON_ULTRA_ENDPOINT_POLICY.settings_selection.model


def _reply(**fields):
    values = {
        "quirks": (),
        "text": "",
        "calls": (),
        "text_ended": False,
        "refusals": (),
        "faults": MappingProxyType({}),
    }
    return LatestReply(**{**values, **fields})


def _signals(**fields):
    values = {
        "agent_role": "planning_agent",
        "stage": "reference",
        "request_number": 3,
        "tool_results": 2,
        "sent": MappingProxyType({}),
        "stage_request_number": 2,
        "reply": _reply(),
        "results": (),
    }
    return InjectionSignals(**{**values, **fields})


def _call(tool="discover_assemblies", arguments=None, reasoning="Why.", takes=True):
    return ReplyCall(
        tool_name=tool,
        arguments=arguments if arguments is not None else {"taxon": "9606"},
        reasoning=reasoning,
        takes_arguments=takes,
    )


def _refusal(codes, *, tool="record_reference_choice", count=1, limit=3, whole=False):
    return ReplyRefusal(
        tool_name=tool,
        arguments={"candidate_id": "assembly:GCF_000001405.40"},
        codes=tuple(codes),
        count=count,
        limit=limit,
        whole_reply=whole,
    )


def _spoken(name, signals, **tuning):
    """The one pattern's message for these signals, untagged, or None."""
    policy = resolve_injection((name,), {name: tuning} if tuning else None, None, {})
    messages = injection_messages(policy, signals)
    if not messages:
        return None
    (message,) = messages
    tag = f"[injection:{name}]\n"
    assert message.startswith(tag), message
    return message.removeprefix(tag)


def _notice(*lines):
    return "<loop_notice>\n" + "\n".join(lines) + "\n</loop_notice>"


class _PatternTestCase(unittest.TestCase):
    def assertSilent(self, name, signals, **tuning):
        self.assertIsNone(_spoken(name, signals, **tuning))


class PatternDecisionTests(_PatternTestCase):
    def test_action_commit_follows_a_reply_that_ended_in_text(self):
        expected = _notice(
            "Your last reply described the next step in text, so nothing happened.",
            "A step happens only through a tool call: make the call that does it.",
        )
        ended = _signals(reply=_reply(text="I will now record it.", text_ended=True))
        self.assertEqual(_spoken("action_commit", ended), expected)
        # It names no ask Tool: a role without one cannot ask.
        self.assertNotIn("researcher", expected)
        self.assertSilent("action_commit", _signals(reply=_reply(text="Done.")))
        self.assertSilent("action_commit", _signals(reply=None))
        # Loose cap: two notices per Attempt by default, tunable per scope.
        self.assertSilent(
            "action_commit", _signals(reply=ended.reply, sent={"action_commit": 2})
        )
        self.assertIsNotNone(
            _spoken(
                "action_commit",
                _signals(reply=ended.reply, sent={"action_commit": 2}),
                max_sends=3,
            )
        )

    def test_plan_review_opens_each_plan_stage_once(self):
        expected = _notice(
            "Plan review, before you call a tool in this stage:",
            "- Keep each value the request states exactly as written.",
            "- A fact the request does not state is one to find: never fill it "
            "with a guess or a placeholder.",
            "- Order the calls: a call that needs another call's result comes "
            "after it, in a later reply.",
        )
        for stage in ("proposal_requirements", "source_reading"):
            with self.subTest(stage=stage):
                first = _signals(stage=stage, stage_request_number=1, reply=None)
                self.assertEqual(_spoken("plan_review", first), expected)
                self.assertSilent(
                    "plan_review", _signals(stage=stage, stage_request_number=2)
                )
        self.assertSilent(
            "plan_review", _signals(stage="reference", stage_request_number=1)
        )
        self.assertIsNotNone(
            _spoken(
                "plan_review",
                _signals(stage="reference", stage_request_number=1),
                stages=["reference"],
            )
        )

    def test_literal_guard_follows_a_refusal_of_a_value_the_request_does_not_state(
        self,
    ):
        expected = _notice(
            "Exact values: copy each value and quote from the request character "
            "for character.",
            "Keep its spelling, case, spaces and punctuation; do not shorten, "
            "expand or reword it.",
        )
        for code in (
            "sample_hg002:request_assertion_ungrounded",
            "variant_rs7412:asserted_value_not_in_quote",
        ):
            with self.subTest(code=code):
                refused = _signals(
                    reply=_reply(
                        refusals=(
                            _refusal([code], tool="record_proposal_requirements"),
                        )
                    )
                )
                self.assertEqual(_spoken("literal_guard", refused), expected)
        other = _signals(reply=_reply(refusals=(_refusal(["sample_unattested"]),)))
        self.assertSilent("literal_guard", other)

    def test_result_suffix_names_the_next_page_of_a_partial_result(self):
        page = {
            "journey_id": "j",
            "events": [{"event_id": "e-41"}, {"event_id": "e-42"}],
            "has_earlier_matches": True,
        }
        self.assertEqual(
            _spoken(
                "result_suffix",
                _signals(results=(ToolResult("search_journey", page),)),
            ),
            _notice(
                "The `search_journey` result above is one page: earlier matches exist.",
                'To read them, call `search_journey` again with before_event_id "e-41"; '
                "skip it if this page answers your question.",
            ),
        )
        piece = {
            "payload_slice": {
                "event_id": "e-7",
                "offset": 0,
                "text": "...",
                "total_characters": 9000,
                "next_offset": 4000,
            }
        }
        self.assertEqual(
            _spoken(
                "result_suffix",
                _signals(results=(ToolResult("search_journey", piece),)),
            ),
            _notice(
                "The `search_journey` result above is one slice of a longer payload: "
                "4000 of 9000 characters.",
                'To read on, call `search_journey` with payload_slice {"event_id": '
                '"e-7", "offset": 4000}; skip it if this slice answers your question.',
            ),
        )
        whole = {**page, "has_earlier_matches": False}
        last = {"payload_slice": {**piece["payload_slice"], "next_offset": None}}
        for content in (whole, last, "plain text result"):
            with self.subTest(content=content):
                self.assertSilent(
                    "result_suffix",
                    _signals(results=(ToolResult("search_journey", content),)),
                )

    def test_repeat_call_follows_the_loop_guard_s_refusal_with_next_step_guidance(self):
        repeated = _signals(
            reply=_reply(
                refusals=(_refusal(["repeated_call"], tool="discover_assemblies"),)
            )
        )
        self.assertEqual(
            _spoken("repeat_call", repeated),
            _notice(
                "`discover_assemblies` already ran with these arguments; its result "
                "is in the history above.",
                "Next: use that result. Change the arguments only to get different "
                "evidence, or call another tool.",
            ),
        )
        self.assertSilent(
            "repeat_call",
            _signals(reply=_reply(refusals=(_refusal(["sample_unattested"]),))),
        )

    def test_progress_notice_speaks_past_its_loose_thresholds_and_then_every_few_requests(
        self,
    ):
        def at(request, results=2, sent=0):
            return _spoken(
                "progress_notice",
                _signals(
                    request_number=request,
                    tool_results=results,
                    sent={"progress_notice": sent} if sent else {},
                ),
            )

        # First after request 4, then every 8.
        self.assertEqual(
            at(5),
            _notice(
                "Progress: this is request 5 of this Attempt, with 2 tool results so far.",
                "If you have the facts this stage needs, make the call that completes it.",
                "If a fact cannot be found, ask the researcher with the question tool "
                "instead of searching again.",
            ),
        )
        self.assertEqual(
            [n for n in range(1, 30) if at(n, sent=1) is not None], [5, 13, 21, 29]
        )
        # Many Tool results speak once before the request threshold.
        self.assertIsNotNone(at(3, results=25))
        self.assertIsNone(at(4, results=26, sent=1))
        self.assertIsNone(at(3, results=24))

    def test_no_notice_names_a_question_tool_to_a_role_that_has_none(self):
        # I-63: the line pointing at the question tool is spoken only to a role
        # whose Tools open a researcher question; the others read the notice
        # without it.
        from bio_neb.agentic_journey.role_declaration import ROLES
        from bio_neb.agentic_journey.tool_execution import question_tool_names

        line = "ask the researcher with the question tool"
        # Every role has one: the post-approval roles share
        # `ask_researcher`, so the notice names it to each of them too.
        asks = {
            "planning_agent": True,
            "investigation_planner": True,
            "table_reader": True,
            "report_explainer": True,
            "paper_reader": True,
        }
        self.assertEqual(
            {name: bool(question_tool_names(name)) for name in ROLES}, asks
        )
        for name, has_one in asks.items():
            with self.subTest(role=name):
                spoken = _spoken(
                    "progress_notice",
                    _signals(agent_role=name, request_number=5, tool_results=2),
                )
                self.assertIn(
                    "If you have the facts this stage needs, make the call that "
                    "completes it.",
                    spoken,
                )
                self.assertEqual(line in spoken, has_one)

    def test_hint_as_user_follows_the_same_refusal_from_its_second_time(self):
        def refused(count, codes=("reference_discovery_not_open",), limit=3):
            return _signals(
                reply=_reply(refusals=(_refusal(codes, count=count, limit=limit),))
            )

        self.assertSilent("hint_as_user", refused(1))
        self.assertEqual(
            _spoken("hint_as_user", refused(2)),
            _notice(
                "The same refusal again: `record_reference_choice` "
                "(reference_discovery_not_open), 2 times in this Attempt; at 3 the "
                "Attempt stops.",
                "The fix is in that call's refusal above: change what it names, "
                "then call `record_reference_choice` again.",
            ),
        )
        # Argument codes are named once, by kind, never by their full message.
        self.assertIn(
            "(tool_arguments_invalid, discovery_event_id:bound_value_mismatch)",
            _spoken(
                "hint_as_user",
                refused(
                    2,
                    codes=(
                        "tool_arguments_invalid:requirements.0:missing:Field required",
                        "tool_arguments_invalid:requirements.1:missing:Field required",
                        "discovery_event_id:bound_value_mismatch",
                    ),
                ),
            ),
        )
        self.assertNotIn(
            "Attempt stops", _spoken("hint_as_user", refused(2, limit=None))
        )
        # Several keys refused in one reply: the most repeated one is named.
        several = _signals(
            reply=_reply(
                refusals=tuple(
                    _refusal(["reference_identity_resolved"], count=count, limit=6)
                    for count in (2, 3, 5)
                )
            )
        )
        self.assertIn("5 times in this Attempt; at 6", _spoken("hint_as_user", several))
        # The loop guard's own refusal has its own pattern.
        self.assertSilent("hint_as_user", refused(2, codes=("repeated_call",)))


class QuirkAndGapPatternTests(_PatternTestCase):
    def test_each_adapter_quirk_gets_its_own_next_turn_guidance(self):
        expected = {
            "text_form_tool_call": _notice(
                "Your last reply wrote its tool call as text in the message; it was "
                "read this time.",
                "Send each tool call in the tool-call field, with no call text in "
                "the message.",
            ),
            "text_form_tool_call_cut_off": _notice(
                "Your last reply wrote a tool call as text and was cut off.",
                "Send the next call in the tool-call field, with no text before it.",
            ),
            "think_tags": _notice(
                "Your last reply put think tags in the message text.",
                "Keep the message to the tool call, with no think tags.",
            ),
            "call_reasoning_missing": _notice(
                "A call in your last reply had no `reasoning` argument.",
                "Every tool call carries `reasoning`: one sentence on why you chose "
                "these values.",
            ),
            "argument_name": _notice(
                "An argument name in your last reply was spelled differently from "
                "the tool's schema; it was fixed this time.",
                "Use each argument name exactly as the tool's schema spells it.",
            ),
        }
        for quirk, notice in expected.items():
            with self.subTest(quirk=quirk):
                quirky = _signals(reply=_reply(quirks=(quirk,)))
                self.assertEqual(_spoken(quirk, quirky), notice)
                self.assertSilent(quirk, _signals())
                self.assertSilent(quirk, _signals(reply=quirky.reply, sent={quirk: 2}))

    def test_whitespace_runaway_follows_a_reply_with_a_long_blank_run(self):
        runaway = _signals(reply=_reply(text='{"name": "x"}' + " \n" * 600))
        self.assertEqual(
            _spoken("whitespace_runaway", runaway),
            _notice(
                "Your last reply held a run of 1200 blank characters.",
                "End each reply right after its last tool call.",
            ),
        )
        self.assertSilent("whitespace_runaway", _signals(reply=_reply(text=" " * 999)))
        self.assertIsNotNone(
            _spoken(
                "whitespace_runaway", _signals(reply=_reply(text=" " * 50)), min_run=50
            )
        )

    def test_refused_call_context_shows_a_whole_reply_refusal_the_history_drops(self):
        refused = _signals(
            reply=_reply(
                refusals=(
                    _refusal(["sample_unattested"], tool="resolve_taxon", whole=True),
                )
            )
        )
        self.assertEqual(
            _spoken("refused_call_context", refused),
            _notice(
                "Your last call is not in the history above: it was refused before "
                "it ran.",
                'You sent `resolve_taxon` with {"candidate_id": '
                '"assembly:GCF_000001405.40"}.',
            ),
        )
        kept = _signals(reply=_reply(refusals=(_refusal(["sample_unattested"]),)))
        self.assertSilent("refused_call_context", kept)

    def test_empty_arguments_follows_a_call_that_put_its_values_in_its_reasoning(self):
        reasoning = (
            "The plan: sample HG002, taxon 9606, reference GCF_000001405.40, "
            "variants rs7412 and rs429358."
        )
        empty = _signals(reply=_reply(calls=(_call("record_plan", {}, reasoning),)))
        self.assertEqual(
            _spoken("empty_arguments", empty),
            _notice(
                "Your call to `record_plan` sent no arguments and put its values "
                "in `reasoning`.",
                "Put each value in its own argument field; `reasoning` only says "
                "why you chose them.",
            ),
        )
        for call in (
            _call("attest_sample", {}, reasoning, takes=False),
            _call("record_plan", {}, "Short."),
            _call("record_plan", {"sample": "HG002"}, reasoning),
        ):
            with self.subTest(call=call):
                self.assertSilent(
                    "empty_arguments", _signals(reply=_reply(calls=(call,)))
                )

    def test_placeholder_argument_names_the_placeholder_and_the_step_order(self):
        placeholder = _signals(
            reply=_reply(
                calls=(
                    _call("discover_assemblies"),
                    _call(
                        "record_reference_choice",
                        {
                            "candidate_id": "assembly:GCF_000001405.40",
                            "discovery_event_id": "<DISCOVERY_EVENT_ID>",
                        },
                    ),
                )
            )
        )
        self.assertEqual(
            _spoken("placeholder_argument", placeholder),
            _notice(
                "Your call to `record_reference_choice` sent the placeholder "
                '"<DISCOVERY_EVENT_ID>" in `discovery_event_id`.',
                "A value another call returns exists only after that call's result: "
                "call it first, read its result, then send the real value in a "
                "later reply.",
            ),
        )
        for value in ("V:2129317 C>T", "a <b> c", "<>"):
            with self.subTest(value=value):
                self.assertSilent(
                    "placeholder_argument",
                    _signals(reply=_reply(calls=(_call(arguments={"quote": value}),))),
                )
        nested = _signals(
            reply=_reply(calls=(_call("record_plan", {"items": [{"rsid": "<RSID>"}]}),))
        )
        self.assertIn("in `items.0.rsid`", _spoken("placeholder_argument", nested))

    def test_a_blank_string_id_is_an_unfilled_value(self):
        # A blank id is unfilled, like "<EVENT_ID>"; a blank text field is just
        # empty.
        for arguments, location in (
            ({"discovery_event_id": ""}, "discovery_event_id"),
            ({"candidate_id": "  "}, "candidate_id"),
            ({"item_ids": ["item-1", ""]}, "item_ids.1"),
            ({"items": [{"id": ""}]}, "items.0.id"),
        ):
            with self.subTest(arguments=arguments):
                spoken = _spoken(
                    "placeholder_argument",
                    _signals(reply=_reply(calls=(_call("pick", arguments),))),
                )
                self.assertIn(f"in `{location}`", spoken)
        for arguments in ({"note": ""}, {"reason": "   "}, {"paid": ""}):
            with self.subTest(arguments=arguments):
                self.assertSilent(
                    "placeholder_argument",
                    _signals(reply=_reply(calls=(_call("pick", arguments),))),
                )


# A test pattern that repeats the text-ending correction's first line.
_ECHO = injection.InjectionPattern(
    name="echo_correction",
    tuning=injection.PatternTuning,
    message=lambda signals, _tuning: (
        "Your reply ended in text without a tool call; text is not a result."
        if signals.reply is not None and signals.reply.text_ended
        else None
    ),
)


class SlotSignalTests(unittest.TestCase):
    """The slot reads each signal from the Attempt's own record (scripted Nemotron).

    Since #373 (owner E6, on the N15 spike's evidence) a request that carries a
    correction gets no notice on Nemotron, and the folded fault-answering
    patterns never speak: the correction is the fault's one message.
    """

    def speak(self, why: dict, placement: str | None = None, first=None):
        """Each request's injected tags, and the last request's own parts and the
        reply its history keeps before them.

        ``why`` is the call reasoning each scripted call sends; ``placement``
        builds Nemotron's adapter at that call-reasoning placement (#402);
        ``first`` replaces the first reply.
        """
        search = {"text": "registered input", **why}
        replies = [
            # No call reasoning, and a Tool this stage does not offer (routed).
            first
            or ModelResponse(
                parts=[
                    ToolCallPart(
                        "record_proposal_requirements", {"requirements": []}, "c1"
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("I will record the requirements now.")]),
            # A text-form call the adapter reads, then the same call again.
            ModelResponse(
                parts=[
                    TextPart(
                        '<tool_call>{"name": "search_journey", "arguments": '
                        + json.dumps(search)
                        + "}</tool_call>"
                    )
                ]
            ),
            ModelResponse(parts=[ToolCallPart("search_journey", search, "c4")]),
            ModelResponse(parts=[ToolCallPart("search_journey", search, "c5")]),
        ]
        sent = []

        def respond(messages, info):
            sent.append(messages)
            return replies[len(sent) - 1]

        invocation = {
            "model_requests": 5,
            "injection": {"echo_correction": {"on": True}},
        }
        endpoint = ScriptedModelEndpoint(respond)
        registry = model_registry()
        if placement is not None:
            registry = registry.replacing(
                replace(NEMOTRON_ULTRA_ENDPOINT_POLICY, call_reasoning=placement)
            )
        with (
            patch.dict(injection.PATTERNS, {_ECHO.name: _ECHO}),
            use_model_registry(registry),
            tempfile.TemporaryDirectory() as directory,
        ):
            receipt, _ = _registered(
                Path(directory), model=NEMOTRON, agent_invocation=invocation
            )
            journey = Journey.open(Path(receipt.journey_file))
            prepare_input_correction(journey)
            result = planning_assignment(model_factory=endpoint).run(
                journey, environ={"LOCAL_MODEL_KEY": "secret"}
            )
            events = read_evidence(journey)
        self.assertEqual(result.status, "effort_review_pending")
        requests = [
            e["payload"] for e in events if e["event_type"] == "ModelRequestRecorded"
        ]
        spoken = []
        for request in requests:
            pending = request["compatible_adapter_evidence"]["request"]["messages"][-1]
            spoken.append(
                [
                    (index, part["content"].partition("\n")[0])
                    for index, part in enumerate(pending["parts"])
                    if part["part_kind"] == "user-prompt"
                    and part["content"].startswith("[injection:")
                ]
            )
            with self.subTest(request=request["request_number"]):
                # Each injected message reached the model as a user message.
                wire = endpoint.bodies[request["request_number"] - 1]["messages"]
                for _, tag in spoken[-1]:
                    self.assertTrue(
                        any(
                            m["role"] == "user" and m["content"].startswith(tag + "\n")
                            for m in wire
                        )
                    )
        *_, kept, last = requests[-1]["compatible_adapter_evidence"]["request"][
            "messages"
        ]
        return spoken, last["parts"], kept["parts"]

    def _assert_the_repeat_is_answered_in_its_slot(self, parts, kept):
        # The loop guard refused the single repeated call: the call stays in
        # history and its correction is its own failed result, the request's one
        # part; refused_call_context and repeat_call no longer speak.
        (part,) = parts
        self.assertEqual(
            (part["part_kind"], part["tool_call_id"], part["outcome"]),
            ("tool-return", "c4", "failed"),
        )
        # It restates the earlier call's status and next step.
        self.assertIn(
            "Tool `search_journey` was not run: your earlier `search_journey` call "
            "with the same arguments ended `read`, and nothing has changed since. "
            "Its next step: Read the matched events above",
            part["content"],
        )
        self.assertEqual(
            [p.get("tool_call_id") for p in kept if p["part_kind"] == "tool-call"],
            ["c4"],
        )

    def test_nemotron_s_patterns_read_the_attempt_s_record_and_speak_in_its_requests(
        self,
    ):
        # Nemotron's placement is `none`: its calls carry no reasoning.
        spoken, parts, kept = self.speak({})
        self.assertEqual(
            spoken,
            [
                [],
                # The routed call's note is the request's correction: no notice.
                [],
                # The text ending's correction is its one message: no
                # action_commit, and no echo of it.
                [],
                # After the Tool result the read text-form call produced: the
                # finished text-form call was read, run and corrected.
                [(1, "[injection:text_form_tool_call]")],
                # The loop guard refused a single call: its one correction.
                [],
            ],
        )
        self._assert_the_repeat_is_answered_in_its_slot(parts, kept)

    def test_a_missing_call_reasoning_is_spoken_at_the_last_placement(self):
        # A lawful first call sent without its reasoning runs; the raw reply's
        # quirk is read from the record (the kept reply carries no provider
        # details) and spoken after its result.
        spoken, parts, kept = self.speak(
            {"reasoning": "Find the inputs."},
            placement="last",
            first=ModelResponse(
                parts=[
                    ToolCallPart("search_journey", {"text": "registered sample"}, "c1")
                ]
            ),
        )
        self.assertEqual(
            spoken,
            [
                [],
                [(1, "[injection:call_reasoning_missing]")],
                [],
                [(1, "[injection:text_form_tool_call]")],
                [],
            ],
        )
        self._assert_the_repeat_is_answered_in_its_slot(parts, kept)


class LatestReplyTests(unittest.TestCase):
    """``latest_reply`` reads a reply's calls against the schema its request showed."""

    def test_a_call_takes_arguments_only_where_its_request_showed_a_field_besides_reasoning(
        self,
    ):
        def tool(name, properties):
            return {
                "name": name,
                "parameters_json_schema": {"type": "object", "properties": properties},
            }

        reasoning = {"reasoning": {"type": "string"}}
        # Server-bound fields are not shown: only `reasoning` is.
        request = {
            "event_id": "q1",
            "event_type": "ModelRequestRecorded",
            "payload": {
                "attempt_id": "a1",
                "stage": "execution_planning",
                "compatible_adapter_evidence": {
                    "request": {
                        "function_tools": [
                            tool("list_compatible_analysis_recipes", reasoning),
                            tool(
                                "select_recipe_candidate",
                                {"candidate_id": {}, **reasoning},
                            ),
                        ],
                        "output_tools": [tool("record_plan", {"plan": {}})],
                    }
                },
            },
        }
        parts = [
            {
                "part_kind": "tool-call",
                "tool_name": name,
                "tool_call_id": name,
                "args": "{}",
            }
            for name in (
                "list_compatible_analysis_recipes",
                "select_recipe_candidate",
                "record_plan",
            )
        ]
        response = {
            "event_id": "r1",
            "event_type": "ModelResponseRecorded",
            "payload": {
                "attempt_id": "a1",
                "compatible_adapter_evidence": {
                    "response": {"message": {"parts": parts, "provider_details": None}}
                },
            },
        }
        reply = injection.latest_reply([request, response], "a1", limit=lambda _k: 3)
        self.assertEqual(
            [(call.tool_name, call.takes_arguments) for call in reply.calls],
            [
                ("list_compatible_analysis_recipes", False),
                ("select_recipe_candidate", True),
                ("record_plan", True),
            ],
        )

    def test_a_refused_call_shows_the_arguments_the_model_sent_not_the_effective_ones(
        self,
    ):
        # The validation records effective arguments, server-bound values merged in;
        # the notice restores the call as sent (Standards r1 S1). A bound field the
        # model did send is kept, wrong value included.
        request = {
            "event_id": "q1",
            "event_type": "ModelRequestRecorded",
            "payload": {
                "attempt_id": "a1",
                "stage": "execution_planning",
                "compatible_adapter_evidence": {
                    "request": {"function_tools": [], "output_tools": []}
                },
            },
        }
        sent = {"candidate_id": "unknown", "basis_event_id": "made-up"}
        response = {
            "event_id": "r1",
            "event_type": "ModelResponseRecorded",
            "payload": {
                "attempt_id": "a1",
                "compatible_adapter_evidence": {
                    "response": {
                        "message": {
                            "parts": [
                                {
                                    "part_kind": "tool-call",
                                    "tool_name": "select_recipe_candidate",
                                    "tool_call_id": "c1",
                                    "args": json.dumps(sent),
                                }
                            ],
                            "provider_details": None,
                        }
                    }
                },
            },
        }
        validation = {
            "event_id": "v1",
            "event_type": "ToolCallValidated",
            "payload": {
                "attempt_id": "a1",
                "tool_name": "select_recipe_candidate",
                "tool_call_id": "c1",
                "arguments": {**sent, "work_item_id": "work-1"},
                "errors": ["basis_event_id:bound_value_mismatch"],
            },
        }
        disposition = {
            "event_id": "d1",
            "event_type": "AgentActionTurnDispositionRecorded",
            "payload": {
                "attempt_id": "a1",
                "response_event_id": "r1",
                "validation_event_ids": ["v1"],
                "disposition": "singleton_refused",
                "correction_key": "select_recipe_candidate: basis_event_id:bound_value_mismatch",
                "correction_count": 1,
            },
        }
        reply = injection.latest_reply(
            [request, response, validation, disposition], "a1", limit=lambda _k: 3
        )
        self.assertEqual(
            _spoken("refused_call_context", _signals(reply=reply)),
            _notice(
                "Your last call is not in the history above: it was refused before "
                "it ran.",
                'You sent `select_recipe_candidate` with {"candidate_id": '
                '"unknown", "basis_event_id": "made-up"}.',
            ),
        )

    def test_action_commit_stays_silent_after_a_reasoning_only_reply(self):
        # A reply that held only reasoning described nothing in text; the
        # notice that says it did must not speak for it.
        request = {
            "event_id": "q1",
            "event_type": "ModelRequestRecorded",
            "payload": {
                "attempt_id": "a1",
                "stage": "execution_planning",
                "compatible_adapter_evidence": {
                    "request": {"function_tools": [], "output_tools": []}
                },
            },
        }
        response = {
            "event_id": "r1",
            "event_type": "ModelResponseRecorded",
            "payload": {
                "attempt_id": "a1",
                "compatible_adapter_evidence": {
                    "response": {
                        "message": {
                            "parts": [
                                {
                                    "part_kind": "thinking",
                                    "content": "Now the call.",
                                    "provider_details": None,
                                }
                            ],
                            "provider_details": None,
                        }
                    }
                },
            },
        }

        def reply_after(disposition, key):
            refused = {
                "event_id": "d1",
                "event_type": "AgentActionTurnDispositionRecorded",
                "payload": {
                    "attempt_id": "a1",
                    "response_event_id": "r1",
                    "validation_event_ids": [],
                    "disposition": disposition,
                    "correction_key": key,
                    "correction_count": 1,
                },
            }
            return injection.latest_reply(
                [request, response, refused], "a1", limit=lambda _k: 3
            )

        self.assertIsNone(
            _spoken(
                "action_commit",
                _signals(reply=reply_after("reasoning_refused", "reasoning_only")),
            )
        )
        self.assertIsNotNone(
            _spoken(
                "action_commit",
                _signals(reply=reply_after("text_refused", "text_ending")),
            )
        )


class RegistryTests(unittest.TestCase):
    def test_nemotron_switches_every_registered_pattern_on_and_each_run_retains_them(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        roster = next(
            e["payload"]["runtime_selections"]["model_roster"]
            for e in events
            if "runtime_selections" in e["payload"]
        )
        self.assertEqual(
            roster["nemotron"]["settings_selection"]["injection_patterns"],
            list(injection.PATTERNS),
        )
        # The roster's other entry, Qwen, switches none on.
        self.assertNotIn("injection_patterns", roster["qwen"]["settings_selection"])

    def test_every_named_pattern_is_registered_with_a_tunable_send_shape(self):
        self.assertEqual(
            set(injection.PATTERNS),
            {
                "action_commit",
                "plan_review",
                "literal_guard",
                "result_suffix",
                "repeat_call",
                "progress_notice",
                "hint_as_user",
                "text_form_tool_call",
                "text_form_tool_call_cut_off",
                "think_tags",
                "call_reasoning_missing",
                "argument_name",
                "whitespace_runaway",
                "refused_call_context",
                "empty_arguments",
                "placeholder_argument",
            },
        )


if __name__ == "__main__":
    unittest.main()
