"""Each fault reaches Nemotron as one message from the failure module (#373).

Faulty Nemotron turns of the shapes #380's and #389's live Runs recorded (a
repeated single call, a text ending, an unknown Tool, a cut-off text-form call
with a blank runaway, several outputs in one reply) are replayed through the real
planning and Investigation loops with a scripted model. For each fault the next
request carries exactly one model-visible part: the refused call's own failed
result in its slot, or one retry part when no call exists. No notice stacks on
it, the folded notices' lines are in it, the firm line comes at the last failed
turn before the limit, then the stop. Every model reads them the same way (#445).
"""

import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import RequestUsage

from bio_neb.agentic_journey import action_admission, injection, invocation
from bio_neb.agentic_journey.injection_patterns import FOLDED, NAMES
from bio_neb.agentic_journey.model_settings import (
    NEMOTRON_ULTRA_ENDPOINT_POLICY,
    QWEN_ENDPOINT_POLICY,
)
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    run_attempt,
)

NEMOTRON = NEMOTRON_ULTRA_ENDPOINT_POLICY.settings_selection.model
QWEN = QWEN_ENDPOINT_POLICY.settings_selection.model
SEARCH = {"text": "registered input"}


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def _pending(request):
    """The request's own parts: what the model reads after its last reply."""
    return request["compatible_adapter_evidence"]["request"]["messages"][-1]["parts"]


def _kept(request):
    """The reply the request's history keeps just before its own parts."""
    return request["compatible_adapter_evidence"]["request"]["messages"][-2]["parts"]


_EVERY_REQUEST = injection.InjectionPattern(
    name="test_every_request",
    tuning=injection.PatternTuning,
    message=lambda signals, _tuning: "Test notice: every request.",
)


class _Inspected:
    """Captures development inspection events; nothing is stored."""

    def __init__(self):
        self.events = []

    def __call__(self, name, **attributes):
        self.events.append((name, attributes))


def _plan(model, replies, *, invocation_settings=None):
    """One planning Attempt on ``model`` answering ``replies`` in order, with a
    test pattern that would speak on every request, reopened by the reader."""
    sent = []

    def respond(messages, info):
        sent.append(messages)
        return replies[len(sent) - 1]

    endpoint = ScriptedModelEndpoint(respond)
    inspected = _Inspected()
    settings = {
        # The budget ends the Attempt once the script is spent.
        "model_requests": len(replies),
        "injection": {"test_every_request": {"on": True}},
        **(invocation_settings or {}),
    }
    with (
        patch.dict(injection.PATTERNS, {_EVERY_REQUEST.name: _EVERY_REQUEST}),
        patch.object(action_admission, "inspection_event", inspected),
        patch.object(invocation, "inspection_event", inspected),
        tempfile.TemporaryDirectory() as directory,
    ):
        receipt, _ = _registered(
            Path(directory), model=model, agent_invocation=settings
        )
        journey = Journey.open(Path(receipt.journey_file))
        prepare_input_correction(journey)
        result = planning_assignment(model_factory=endpoint).run(
            journey, environ={"LOCAL_MODEL_KEY": "secret"}
        )
        events = read_evidence(journey)
        # Reopen reads the whole Journey through the public reader.
        review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
    return result, events, endpoint.bodies, inspected.events


class NemotronFaultMessageTests(unittest.TestCase):
    """#380 S2 a2's repeated `locate_variant`, #389's text ending and an unknown
    Tool, as planning turns on Nemotron (owner option B, E2 A, E3)."""

    @classmethod
    def setUpClass(cls):
        replies = [
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c1")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
            ModelResponse(parts=[TextPart("I will record the requirements now.")]),
            ModelResponse(parts=[ToolCallPart("read_it", {"path": "x"}, "c4")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c5")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c6")]),
        ]
        # Five failed turns in a row after the first lawful call: the fifth stops.
        # The budget leaves one request beyond the script, so the stop is the
        # failed turns' and not the effort budget's.
        cls.result, cls.events, cls.bodies, cls.inspected = _plan(
            NEMOTRON,
            replies,
            invocation_settings={"failed_turns": 5, "model_requests": 7},
        )
        cls.requests = [e["payload"] for e in _of(cls.events, "ModelRequestRecorded")]
        cls.turns = {
            turn["payload"]["response_event_id"]: turn["payload"]
            for turn in _of(cls.events, "AgentActionTurnDispositionRecorded")
        }

    def _feedback(self, disposition):
        return [
            turn["feedback"]
            for turn in self.turns.values()
            if turn["disposition"] == disposition
        ]

    def test_every_fault_is_one_part_and_no_notice_stacks_on_it(self):
        # Requests 3-6 each follow a fault; request 2 follows a lawful call.
        faulted = self.requests[2:]
        self.assertEqual([len(_pending(request)) for request in faulted], [1, 1, 1, 1])
        self.assertEqual(
            [part["part_kind"] for part in _pending(self.requests[1])],
            ["tool-return", "user-prompt"],
        )
        for request in faulted:
            self.assertFalse(
                any(part["part_kind"] == "user-prompt" for part in _pending(request)),
                request["request_number"],
            )

    def test_a_repeated_single_call_is_answered_in_its_own_slot(self):
        repeats = self._feedback("singleton_refused")[:2]
        for request, call_id, feedback in zip(
            (self.requests[2], self.requests[5]), ("c2", "c5"), repeats, strict=True
        ):
            (part,) = _pending(request)
            self.assertEqual(
                (
                    part["part_kind"],
                    part["tool_call_id"],
                    part["outcome"],
                    part["content"],
                ),
                ("tool-return", call_id, "failed", feedback),
            )
            # The call stays in the kept reply it answers.
            self.assertEqual([p.get("tool_call_id") for p in _kept(request)], [call_id])
        # On the wire it is a Tool message, with no user turn and no "Fix the
        # errors and try again." after it (E3's check).
        wire = self.bodies[2]["messages"][-1]
        self.assertEqual((wire["role"], wire["tool_call_id"]), ("tool", "c2"))
        self.assertNotIn("try again", wire["content"])

    def test_the_repeat_s_correction_holds_the_folded_next_step(self):
        first, _, _ = self._feedback("singleton_refused")
        # The earlier call's own status and next step, from its model
        # view.
        self.assertIn(
            "your earlier `search_journey` call with the same arguments ended "
            "`read`, and nothing has changed since. Its next step: Read the matched "
            "events above and cite their event ids; a search grants no effect "
            "authority.\nThis call sends the same arguments, so it adds no evidence.\n",
            first,
        )
        # The count is the reply's place in the failed turns in a row.
        self.assertIn("Count: failed turn 1 of 5 in a row.", first)
        self.assertNotIn("Last correction", first)

    def test_the_firm_line_comes_at_the_last_allowed_count_then_the_stop(self):
        _, last, _ = self._feedback("singleton_refused")
        self.assertTrue(
            last.endswith(
                "Allowed now: one of `inspect_input_metadata`, `search_journey`.\n"
                "Waiting on their step, not callable now: `request_planning_hitl`.\n"
                "Count: failed turn 4 of 5 in a row. At 5 the Attempt stops and the "
                "researcher reviews it.\n"
                "Last correction: one more failed turn stops the Attempt for review."
            ),
            last,
        )
        self.assertEqual(self.result.status, "agent_work_review_pending")
        (review,) = _of(self.events, "AgentWorkReviewRequested")
        self.assertEqual(review["payload"]["cause"], "failed_turns")
        self.assertEqual(len(self.requests), 6)

    def test_a_text_ending_is_one_user_retry_with_the_commit_line(self):
        (feedback,) = self._feedback("text_refused")
        (part,) = _pending(self.requests[3])
        self.assertEqual(
            (part["part_kind"], part["tool_name"], part["content"]),
            ("retry-prompt", None, feedback),
        )
        self.assertIn(
            "A step happens only through a tool call: make the call that does it.",
            feedback,
        )
        # No question Tool is named to roles that have none; the researcher
        # appears only in the count line that ends it.
        text, count = feedback.rsplit("\n", 1)
        self.assertEqual(
            count,
            "Count: failed turn 2 of 5 in a row. At 5 the Attempt stops and the "
            "researcher reviews it.",
        )
        self.assertNotIn("researcher", text)

    def test_an_unknown_tool_is_named_with_what_was_sent(self):
        (routed,) = _of(self.events, "AgentTurnRouted")
        note = routed["payload"]["feedback"]
        (part,) = _pending(self.requests[4])
        self.assertEqual((part["part_kind"], part["content"]), ("retry-prompt", note))
        # The specific issue with its values (E2), and the trace of what the
        # dropped call sent, since the history no longer holds it (F3).
        self.assertIn(
            'Tool `read_it`: no such tool.\nYou sent it with {"path": "x"}.\n', note
        )
        self.assertNotIn("Stop rule", note)

    def test_each_message_is_shown_to_development_inspection(self):
        messages = [
            attributes for name, attributes in self.inspected if name == "Fault message"
        ]
        self.assertEqual(
            [(m["slot"], m["fault_kinds"]) for m in messages],
            [
                ("call_slot", ["repeated_call"]),
                ("retry", ["text_ending"]),
                ("retry", ["unoffered_calls"]),
                # The fifth failed turn stops the Attempt: its correction is
                # recorded but never sent, so none is shown.
                ("call_slot", ["repeated_call"]),
            ],
        )
        self.assertEqual(messages[3]["last_correction"], ["repeated_call"])
        self.assertEqual(
            messages[3]["templates"],
            ["failed_turn", "last_correction", "repeated_call"],
        )
        self.assertEqual(
            messages[2]["templates"],
            ["fault", "refused", "sent_call", "tool_unknown", "unoffered_calls"],
        )
        notices = [a for name, a in self.inspected if name == "Injected notice"]
        self.assertEqual([n["pattern"] for n in notices], ["test_every_request"] * 2)


class LiveInspectionTests(unittest.TestCase):
    """#373's live round (Spec LS1, LS2): development inspection shows only the
    corrections the model reads next, each naming its Tool."""

    def test_a_call_refused_when_an_earlier_call_ended_the_attempt_is_held(self):
        # Run 1 `0332f9da`: the reply's first call ends the Attempt, so the call
        # after it is refused and held, never read next.
        replies = [
            ModelResponse(
                parts=[
                    ToolCallPart("inspect_input_metadata", {}, "ends"),
                    ToolCallPart("search_journey", SEARCH, "held"),
                ]
            )
        ]
        result, events, _, inspected = _plan(NEMOTRON, replies)
        self.assertEqual(result.status, "inspection_succeeded")
        (turn,) = [
            t["payload"]
            for t in _of(events, "AgentActionTurnDispositionRecorded")
            if t["payload"]["disposition"] == "multi_call_refused"
        ]
        self.assertEqual(
            [name for name, a in inspected if a.get("text") == turn["feedback"]],
            ["Held correction"],
        )
        self.assertNotIn("Fault message", [name for name, _ in inspected])

    def test_a_copied_call_s_note_names_the_copied_tool(self):
        # Run 1 `a3f65a5d`: the copied `record_supplied_reference`.
        replies = [
            ModelResponse(
                parts=[
                    ToolCallPart("search_journey", SEARCH, "c1"),
                    ToolCallPart("search_journey", SEARCH, "c1-copy"),
                ]
            ),
            ModelResponse(parts=[ToolCallPart("inspect_input_metadata", {}, "c2")]),
        ]
        _, events, _, inspected = _plan(NEMOTRON, replies)
        (routed,) = _of(events, "AgentTurnRouted")
        (shown,) = [
            a for name, a in inspected if a.get("text") == routed["payload"]["feedback"]
        ]
        self.assertEqual(
            (shown["fault_kinds"], shown["tools"]),
            (["copied_calls"], ["search_journey"]),
        )


class TraceRowProvenanceTests(unittest.TestCase):
    """Spec live r2: a trace names only the per-surface rows whose line the
    correction holds; a group of Tools reads the "*" row, as the renderer does."""

    def test_rows_are_the_ones_that_wrote_the_text(self):
        from bio_neb.failure import (
            FaultCount,
            FaultTuning,
            TurnFaults,
            fault_trace,
            render_correction,
        )

        wording = NEMOTRON_ULTRA_ENDPOINT_POLICY.correction_wording.model_copy(
            update={
                "surfaces": {
                    "copied_calls": {
                        tool: FaultTuning(line=f"ROW {tool}")
                        for tool in ("*", "search_journey", "resolve_taxon")
                    }
                }
            }
        )
        for copies, row in (
            ((("search_journey", 2),), "search_journey"),
            ((("search_journey", 2), ("resolve_taxon", 2)), "*"),
        ):
            with self.subTest(copies=copies):
                faults = TurnFaults(copies=copies, copies_fault=FaultCount(1, 6))
                text = render_correction(faults, wording)
                trace = fault_trace(faults, wording)
                self.assertEqual(
                    [line for line in text.splitlines() if line.startswith("ROW ")],
                    [f"ROW {row}"],
                )
                self.assertEqual(trace["surface_rows"], [f"copied_calls / {row}"])
                self.assertEqual(trace["tools"], sorted(tool for tool, _ in copies))


class EffortBoundaryTests(unittest.TestCase):
    """An effort boundary right after a refusal answered in its slot: the held
    continuation keeps the call and its answer, and resumes from them."""

    def test_the_held_continuation_keeps_the_call_its_slot_answers(self):
        from bio_neb.agentic_journey.effort import (
            ExtendAgentEffortResponse,
            project_effort_review,
            respond_effort_review,
        )

        replies = [
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c1")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
            ModelResponse(parts=[ToolCallPart("inspect_input_metadata", {}, "c3")]),
        ]
        sent = []

        def respond(messages, info):
            sent.append(messages)
            return replies[len(sent) - 1]

        endpoint = ScriptedModelEndpoint(respond)
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(
                Path(directory), model=NEMOTRON, agent_invocation={"model_requests": 2}
            )
            journey = Journey.open(Path(receipt.journey_file))
            prepare_input_correction(journey)
            assignment = planning_assignment(model_factory=endpoint)
            held = assignment.run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(held.status, "effort_review_pending")
            pending = project_effort_review(read_evidence(journey))
            (answer,) = _pending(
                {
                    "compatible_adapter_evidence": {
                        "request": {"messages": pending.pending_request.messages}
                    }
                }
            )
            self.assertEqual(
                (answer["part_kind"], answer["tool_call_id"], answer["outcome"]),
                ("tool-return", "c2", "failed"),
            )
            response = respond_effort_review(
                journey,
                pending.pending_request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=1),
            )
            resumed = assignment.run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                effort_review_response_event_id=response.response_event_id,
            )
            # The held request was sent as held: the call, then its answer.
            self.assertEqual(resumed.status, "inspection_succeeded")
            self.assertEqual(
                [type(part).__name__ for part in sent[2][-1].parts],
                ["ToolReturnPart"],
            )
            self.assertEqual(
                [
                    p.tool_call_id
                    for p in sent[2][-2].parts
                    if hasattr(p, "tool_call_id")
                ],
                ["c2"],
            )
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))


class CorrectionGuardTests(unittest.TestCase):
    """A correction can quote what the model sent; the model reads it with each
    control character but the line break as one space, in either slot (#422 Spec
    r1 F1). The refused call's recorded errors keep what was sent."""

    def test_a_refusal_quoting_a_raw_tab_is_read_with_a_space(self):
        control = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")
        for model in (NEMOTRON, QWEN):
            with self.subTest(model=model):
                _, events, _, _ = _plan(
                    model,
                    [
                        ModelResponse(
                            parts=[
                                ToolCallPart(
                                    "search_journey",
                                    {"unexpected\tkey": "x"},
                                    "bad-search",
                                )
                            ]
                        ),
                        ModelResponse(parts=[TextPart("Stop here.")]),
                    ],
                )
                (validation,) = [
                    e["payload"]
                    for e in _of(events, "ToolCallValidated")
                    if e["payload"]["tool_call_id"] == "bad-search"
                ]
                self.assertTrue(
                    any("unexpected\tkey" in error for error in validation["errors"]),
                    validation["errors"],
                )
                (turn,) = [
                    t["payload"]
                    for t in _of(events, "AgentActionTurnDispositionRecorded")
                    if t["payload"]["disposition"] == "singleton_refused"
                ]
                request = _of(events, "ModelRequestRecorded")[1]["payload"]
                (part,) = [
                    p for p in _pending(request) if p["part_kind"] == "tool-return"
                ]
                self.assertEqual(part["content"], turn["feedback"])
                self.assertIn("unexpected key", part["content"])
                self.assertIsNone(control.search(part["content"]), part["content"])


class ConfiguredFoldedPatternTests(unittest.TestCase):
    """A configuration that switches a folded pattern on is lawful, and the
    pattern stays silent for every model: its line is in the correction (#445 Q1)."""

    def test_a_switched_on_repeat_call_never_speaks(self):
        replies = [
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c1")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
            ModelResponse(parts=[ToolCallPart("inspect_input_metadata", {}, "c3")]),
        ]
        for model in (NEMOTRON, QWEN):
            with self.subTest(model=model):
                _, events, _, _ = _plan(
                    model,
                    replies,
                    invocation_settings={"injection": {"repeat_call": {"on": True}}},
                )
                request = [e["payload"] for e in _of(events, "ModelRequestRecorded")][2]
                self.assertEqual(
                    [
                        (part["part_kind"], part["outcome"])
                        for part in _pending(request)
                    ],
                    [("tool-return", "failed")],
                )


class CutOffTests(unittest.TestCase):
    """#379 request 3 and #380 S3 a1/a2: a reply cut off inside a text-form call
    after a blank runaway reached Nemotron as the note plus two notices."""

    def test_a_cut_off_text_call_and_blank_run_are_lines_of_its_one_note(self):
        blank = " " * 1500
        replies = [
            ModelResponse(
                parts=[TextPart(blank + '<tool_call>{"name": "search_journey", "ar')],
                finish_reason="length",
                usage=RequestUsage(output_tokens=32768),
            ),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
        ]
        _, events, _, inspected = _plan(
            NEMOTRON, replies, invocation_settings={"failed_turns": 2}
        )
        (truncation,) = _of(events, "ModelCompletionTruncated")
        note = truncation["payload"]["feedback"]
        request = [e["payload"] for e in _of(events, "ModelRequestRecorded")][1]
        (part,) = _pending(request)
        self.assertEqual((part["part_kind"], part["content"]), ("retry-prompt", note))
        self.assertIn(
            "It began a tool call as text in the message. Send each call in the "
            "tool-call field, with no text before it.\n",
            note,
        )
        self.assertIn("It held a run of 1500 blank characters.", note)
        # The limit is two failed turns in a row: the first cut-off is the last
        # correction.
        self.assertTrue(
            note.endswith(
                "Count: failed turn 1 of 2 in a row. At 2 the Attempt stops and the "
                "researcher reviews it.\nLast correction: one more failed turn stops "
                "the Attempt for review."
            ),
            note,
        )
        # Neither cut-off notice speaks; only the test pattern does, never beside
        # the note.
        self.assertEqual(
            {a["pattern"] for name, a in inspected if name == "Injected notice"},
            {"test_every_request"},
        )


class _SendsItsOutputTwice(SequentialAdaptiveModel):
    """#389 item 6 and #372 a2: the member's output sent twice in one reply,
    once; every later reply is the scripted planner's own."""

    def __init__(self):
        super().__init__()
        self.doubled = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        outputs = {tool.name for tool in info.output_tools}
        if not self.doubled and calls and calls[0].tool_name in outputs:
            self.doubled += 1
            (call,) = calls
            # Two distinct outputs: an identical copy would collapse (ADR 0044).
            sent = call.args_as_dict()
            first = {
                **sent,
                "outcome": "unresolved"
                if sent["outcome"] == "complete"
                else "complete",
            }
            return ModelResponse(
                parts=[
                    ToolCallPart(call.tool_name, first, "twice-1"),
                    ToolCallPart(call.tool_name, call.args, "twice-2"),
                ]
            )
        return response


class _SendsTwoOutputsBesideAnUnofferedCall(_SendsItsOutputTwice):
    """The doubled output reply also calls a Tool its stage does not offer, so
    routing writes a note beside the outputs' correction (#444 Standards r1 ST1)."""

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        if self.doubled == 1 and any(
            isinstance(p, ToolCallPart) and p.tool_call_id == "twice-1"
            for p in response.parts
        ):
            self.doubled += 1
            response = ModelResponse(
                parts=[
                    *response.parts,
                    ToolCallPart("describe_companion", {}, "unoffered-1"),
                ]
            )
        return response


class _SendsAnUnreadableOutputFirst(SequentialAdaptiveModel):
    """The member's first output leaves out its required `refsnp_event_id` (a
    schema refusal), or with ``validator`` marks resolved sources unresolved (the
    domain validator's refusal, #372 D3; #478 G2.1 retired the draft's copied
    fields); every later reply is the scripted planner's own."""

    def __init__(self, validator=False):
        super().__init__()
        self.refused = 0
        self.validator = validator
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        outputs = {tool.name for tool in info.output_tools}
        if not self.refused and calls and calls[0].tool_name in outputs:
            self.refused += 1
            (call,) = calls
            arguments = call.args_as_dict()
            if self.validator:
                arguments.update(outcome="unresolved")
            else:
                arguments.pop("refsnp_event_id")
            return ModelResponse(
                parts=[ToolCallPart(call.tool_name, arguments, "unreadable")]
            )
        return response


class OutputRepairFirmLineTests(unittest.TestCase):
    """Spec r1 S1 (E3): an output refusal at the last failed turn before the limit
    ends in the firm line, recorded and sent byte for byte the same."""

    def test_the_last_repair_s_refusal_carries_the_firm_line(self):
        for validator in (False, True):
            with self.subTest(validator=validator):
                self._check(validator)

    def _check(self, validator):
        from bio_neb.agentic_journey.model_turn import as_received

        inspected = _Inspected()
        with (
            patch(
                "bio_neb.agentic_journey.structured_output.inspection_event", inspected
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            _, journey = run_attempt(
                Path(directory).resolve() / "run",
                _SendsAnUnreadableOutputFirst(validator),
                SourceResponsesOnly(),
                model_name=NEMOTRON,
                continuation_policy={"failed_turns": 2},
            )
            events = read_evidence(journey)
        # Spec r2 S2: the trace names the output Tool and its last correction.
        (shown,) = [a for name, a in inspected.events if name == "Fault message"]
        self.assertEqual(
            (shown["fault_kinds"], shown["tools"], shown["last_correction"]),
            (["output_refused"], ["compile_proposal"], ["output_refused"]),
        )
        (refused,) = [
            e["payload"]
            for e in _of(events, "AgentOutputValidated")
            if e["payload"]["status"] == "invalid"
        ]
        self.assertTrue(
            refused["feedback"].endswith(
                "Last correction: one more failed turn stops the Attempt for review."
                "\n\nFix the errors and try again."
            ),
            refused["feedback"],
        )
        sent = [
            part
            for e in _of(events, "ModelRequestRecorded")
            for part in _pending(e["payload"])
            if part["part_kind"] == "retry-prompt"
        ]
        (part,) = sent
        self.assertEqual(
            as_received(part["content"], part["tool_name"]), refused["feedback"]
        )


class _OutputBesideAnUnknownTool(SequentialAdaptiveModel):
    """The member's output sent beside a call to a Tool that does not exist: the
    unknown call is refused and the output completes the member (ADR 0044)."""

    def __init__(self):
        super().__init__()
        self.model = ScriptedModelEndpoint(self.respond)
        self.sent = False

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        outputs = {tool.name for tool in info.output_tools}
        if not self.sent and calls and calls[0].tool_name in outputs:
            self.sent = True
            return ModelResponse(
                parts=[ToolCallPart("read_it", {"path": "x"}, "unknown"), *calls]
            )
        return response


class UnsentNoteTests(unittest.TestCase):
    """Spec r2 S2: a routed note whose output completes its member is never sent,
    so development inspection shows none."""

    def test_a_member_completing_turn_shows_no_note(self):
        inspected = _Inspected()
        with (
            patch.object(action_admission, "inspection_event", inspected),
            tempfile.TemporaryDirectory() as directory,
        ):
            _, journey = run_attempt(
                Path(directory).resolve() / "run",
                _OutputBesideAnUnknownTool(),
                SourceResponsesOnly(),
                model_name=NEMOTRON,
            )
            events = read_evidence(journey)
        (routed,) = _of(events, "AgentTurnRouted")
        self.assertIsNotNone(routed["payload"]["feedback"])
        self.assertNotIn(
            routed["payload"]["feedback"],
            [a["text"] for name, a in inspected.events if name == "Fault message"],
        )


class SeveralOutputsTests(unittest.TestCase):
    """E4: several outputs in one reply are a counted output repair with a
    correction, for every model (#445)."""

    def _run(self, model_name):
        model = _SendsItsOutputTwice()
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory).resolve() / "run",
                model,
                SourceResponsesOnly(),
                model_name=model_name,
            )
            events = read_evidence(journey)
        return projection, events

    def test_nemotron_reads_one_correction_and_its_member_completes(self):
        projection, events = self._run(NEMOTRON)
        twice = [
            e["payload"]
            for e in _of(events, "AgentOutputValidated")
            if e["payload"]["errors"] == ["multiple_output_calls"]
        ]
        (validation,) = twice
        correction = validation["feedback"]
        self.assertTrue(correction.startswith("Outputs: you sent `"), correction)
        self.assertIn("2 times in one reply; none was accepted.", correction)
        requests = [e["payload"] for e in _of(events, "ModelRequestRecorded")]
        after = next(
            request
            for request in requests
            if any(part.get("content") == correction for part in _pending(request))
        )
        (part,) = _pending(after)
        self.assertEqual((part["part_kind"], part["tool_name"]), ("retry-prompt", None))
        # The kept reply holds neither output call.
        self.assertEqual([p for p in _kept(after) if p["part_kind"] == "tool-call"], [])
        self.assertTrue(
            all(
                lineage.accepted_output
                for lineage in projection.attempts[-1].structured_output_lineages
            )
        )

    def test_the_correction_and_a_routed_note_are_the_composers_one_retry(self):
        # The composer joins them, live and on reopen.
        import copy

        from bio_neb.agentic_journey.transcript import reply_retry
        from bio_neb.agentic_journey.turn_reopen import validate_action_turns
        from bio_neb.application.queries import _recorded_assignment

        model = _SendsTwoOutputsBesideAnUnofferedCall()
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory).resolve() / "run",
                model,
                SourceResponsesOnly(),
                model_name=NEMOTRON,
            )
            events = read_evidence(journey)
        (validation,) = [
            e["payload"]
            for e in _of(events, "AgentOutputValidated")
            if e["payload"]["errors"] == ["multiple_output_calls"]
        ]
        (routed,) = [
            e["payload"]
            for e in _of(events, "AgentTurnRouted")
            if e["payload"]["unoffered_validation_event_ids"]
        ]
        expected = reply_retry(validation["feedback"], routed["feedback"])
        self.assertEqual(expected, validation["feedback"] + "\n\n" + routed["feedback"])
        (after,) = [
            e
            for e in _of(events, "ModelRequestRecorded")
            if any(part.get("content") == expected for part in _pending(e["payload"]))
        ]
        (part,) = _pending(after["payload"])
        self.assertEqual((part["part_kind"], part["tool_name"]), ("retry-prompt", None))

        def reopen(timeline):
            attempt_id = routed["attempt_id"]
            selected = [
                e for e in timeline if e["payload"].get("attempt_id") == attempt_id
            ]
            validate_action_turns(
                selected,
                role="investigation_planner",
                assignment=_recorded_assignment(timeline, attempt_id),
                timeline=timeline,
            )()

        reopen(events)
        hostile = copy.deepcopy(events)
        changed = next(e for e in hostile if e["event_id"] == after["event_id"])
        _pending(changed["payload"])[-1]["content"] = validation["feedback"]
        with self.assertRaisesRegex(ValueError, "exact correction"):
            reopen(hostile)

    def test_its_correction_is_shown_to_development_inspection(self):
        model = _SendsItsOutputTwice()
        inspected = _Inspected()
        with (
            patch(
                "bio_neb.agentic_journey.structured_output.inspection_event", inspected
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            run_attempt(
                Path(directory).resolve() / "run",
                model,
                SourceResponsesOnly(),
                model_name=NEMOTRON,
            )
        (shown,) = [a for name, a in inspected.events if name == "Fault message"]
        self.assertEqual(
            (shown["slot"], shown["fault_kinds"], shown["templates"]),
            ("retry", ["output_calls"], ["failed_turn", "output_calls"]),
        )

    def test_reopen_refuses_a_request_after_an_uncorrected_duplicate(self):
        # Standards r1 S1: the terminal rule moved from the lineage to the
        # continuation check, which reads the correction from the feedback.
        import copy

        from bio_neb.agentic_journey.agent_work_review import AgentResultMissing
        from bio_neb.agentic_journey.continuation_policy import (
            ContinuationPolicy,
            validate_continuation,
        )

        _, events = self._run(NEMOTRON)
        (twice,) = [
            e
            for e in _of(events, "AgentOutputValidated")
            if e["payload"]["errors"] == ["multiple_output_calls"]
        ]
        attempt_id = twice["payload"]["attempt_id"]
        (started,) = [
            e
            for e in _of(events, "AttemptStarted")
            if e["payload"]["attempt_id"] == attempt_id
        ]
        policy = ContinuationPolicy.model_validate(
            started["payload"]["continuation_policy"]
        )
        held = events[: events.index(twice) + 1]
        # The corrected fault permits the next request.
        validate_continuation(
            held, attempt_id=attempt_id, policy=policy, next_request=True
        )
        # A record whose duplicate lost its correction stops at once for review:
        # no correction can be sent, whatever its place in the failed turns in a
        # row.
        uncorrected = copy.deepcopy(held)
        uncorrected[-1]["payload"]["feedback"] = None
        with self.assertRaises(AgentResultMissing):
            validate_continuation(
                uncorrected, attempt_id=attempt_id, policy=policy, next_request=True
            )
        # A retained request after it does not reopen.
        request = next(
            e
            for e in _of(events, "ModelRequestRecorded")
            if e["payload"]["attempt_id"] == attempt_id
        )
        with self.assertRaisesRegex(ValueError, "continuation lacks useful progress"):
            validate_continuation(
                [*uncorrected, request], attempt_id=attempt_id, policy=policy
            )


_MINTED = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{64}"
    r"|\d{4}-\d\d-\d\dT[0-9:.]+(?:Z|[+-]\d\d:\d\d)?"
    # A recorded event's size: its registration names the Run's model.
    r"|(?<='byte_count': )\d+"
)


def _read(request):
    """What the model reads after its last reply: the kept reply, then its own
    parts, with each Run's minted ids and digests masked."""
    return [
        [
            (
                part["part_kind"],
                part.get("tool_name"),
                _MINTED.sub("<id>", str(part.get("content"))),
                part.get("outcome"),
            )
            for part in parts
        ]
        for parts in (_kept(request), _pending(request))
    ]


class OneLoopEveryModelTests(unittest.TestCase):
    """Every model runs the loop on one rule set (#445, #439 A5): the same faults
    give the same requests under a Nemotron-shaped and a Qwen-shaped policy."""

    def test_the_same_faults_give_every_model_the_same_requests(self):
        self.maxDiff = None
        replies = [
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c1")]),
            # A repeated single call: answered in its own slot, no notice beside it.
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
            # A text ending: one retry part.
            ModelResponse(parts=[TextPart("I will do it.")]),
            ModelResponse(parts=[ToolCallPart("inspect_input_metadata", {}, "c4")]),
        ]
        read = {}
        for model in (NEMOTRON, QWEN):
            _, events, _, _ = _plan(model, replies)
            read[model] = [
                _read(e["payload"]) for e in _of(events, "ModelRequestRecorded")[1:]
            ]
        self.assertEqual(read[QWEN], read[NEMOTRON])
        _, slot, ended = read[QWEN]
        self.assertEqual(
            [(kind, name, outcome) for kind, name, _, outcome in slot[1]],
            [("tool-return", "search_journey", "failed")],
        )
        self.assertEqual(
            [(kind, name) for kind, name, _, _ in ended[1]], [("retry-prompt", None)]
        )

    def test_several_outputs_are_one_counted_correction_for_every_model(self):
        for model in (NEMOTRON, QWEN):
            with self.subTest(model=model):
                projection, events = SeveralOutputsTests()._run(model)
                (validation,) = [
                    e["payload"]
                    for e in _of(events, "AgentOutputValidated")
                    if e["payload"]["errors"] == ["multiple_output_calls"]
                ]
                self.assertTrue(
                    validation["feedback"].startswith("Outputs: you sent `")
                )
                self.assertEqual(_of(events, "AgentWorkReviewRequested"), [])

    def test_every_registered_model_fills_every_template(self):
        from bio_neb.agentic_journey.model_settings import model_registry
        from bio_neb.failure import CorrectionWording

        for policy in (CorrectionWording(), *model_registry().entries.values()):
            wording = getattr(policy, "correction_wording", policy)
            with self.subTest(policy=getattr(policy, "model", "defaults")):
                self.assertEqual(
                    [name for name, value in wording.model_dump().items() if not value],
                    [],
                )


_HUMAN_A2 = (
    Path(__file__).parent / "fixtures" / "issue387" / "human-a2-11c8261d-replies.json"
)


class HumanA2ReplayTests(unittest.TestCase):
    """#387 human-a2's planning Attempt 11c8261d, replayed through the one loop.

    Its twelve recorded replies are loaded and sent in order to a fresh Run whose
    inputs were inspected for real, as human-a2's were (seven are consumed before
    the Attempt stops for review); no identity is mapped, so calls naming
    that Run's own ids or quotes are refused as this Run decides. Both policies
    read the same requests, and no notice joins a request that carries a
    correction (#445).
    """

    def _replay(self, model):
        from bio_neb.agentic_journey.model_turn import read_messages_record
        from bio_neb.input_inspection.commands import inspect_inputs
        from bio_neb.input_inspection.local_files import LocalFilesAdapter

        recorded = json.loads(_HUMAN_A2.read_text())["replies"]
        replies = [read_messages_record([item["message"]])[0] for item in recorded]
        sent = []

        def respond(messages, info):
            sent.append(messages)
            return replies[len(sent) - 1]

        with (
            patch.dict(injection.PATTERNS, {_EVERY_REQUEST.name: _EVERY_REQUEST}),
            tempfile.TemporaryDirectory() as directory,
        ):
            receipt, _ = _registered(
                Path(directory),
                model=model,
                agent_invocation={
                    "model_requests": len(replies),
                    "injection": {"test_every_request": {"on": True}},
                },
            )
            journey = Journey.open(Path(receipt.journey_file))
            inspect_inputs(journey, adapter=LocalFilesAdapter())
            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            events = read_evidence(journey)
            # Reopen asks the composer the same question of every request.
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        return result, events

    def test_every_model_reads_the_same_requests_for_the_recorded_replies(self):
        self.maxDiff = None
        read, statuses = {}, {}
        for model in (NEMOTRON, QWEN):
            result, events = self._replay(model)
            statuses[model] = result.status
            requests = [e["payload"] for e in _of(events, "ModelRequestRecorded")]
            read[model] = [_read(request) for request in requests[1:]]
            for request in requests:
                parts = _pending(request)
                if any(
                    part["part_kind"] == "retry-prompt"
                    or (
                        part["part_kind"] == "tool-return"
                        and part["outcome"] == "failed"
                    )
                    for part in parts
                ):
                    self.assertFalse(
                        any(part["part_kind"] == "user-prompt" for part in parts), parts
                    )
        self.assertEqual(read[QWEN], read[NEMOTRON])
        self.assertEqual(statuses[QWEN], statuses[NEMOTRON])
        self.assertGreater(len(read[QWEN]), 1)


class FoldedTraceTests(unittest.TestCase):
    """Q1 (g), Q3 (a): the folded patterns stay silent, and the correction's own
    inspection record names the folded lines it carried."""

    def test_each_folded_pattern_is_named_where_its_line_is_written(self):
        from bio_neb.failure import (
            ActionRefused,
            FaultCount,
            RepeatedCall,
            ReplyCutOff,
            TextEnding,
            fault_trace,
        )

        wording = NEMOTRON_ULTRA_ENDPOINT_POLICY.correction_wording
        offered = frozenset({"pick"})
        cases = {
            "action_commit": TextEnding(offered, FaultCount(1, 4)),
            "repeat_call": RepeatedCall("pick", 1, 3, offered, "picked", "Go on."),
            "hint_as_user": TextEnding(offered, FaultCount(3, 4)),
            "text_form_tool_call_cut_off": ReplyCutOff(10, offered, text_call=True),
            "whitespace_runaway": ReplyCutOff(10, offered, blank_run=40),
            "literal_guard": ActionRefused(
                "pick", ("quote:request_assertion_ungrounded",), {}, offered
            ),
            "refused_call_context": ActionRefused(
                "pick", ("tool_name_ineligible",), {"a": 1}, offered, dropped=True
            ),
        }
        self.assertEqual(set(cases), FOLDED)
        for pattern, correction in cases.items():
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, fault_trace(correction, wording)["folded"])
        self.assertEqual(
            fault_trace(
                ActionRefused("pick", ("tool_name_ineligible",), {}, offered), wording
            )["folded"],
            [],
        )


class FaultOwnerContractTests(unittest.TestCase):
    """No fault message is written outside the failure owner (#373 acceptance 1,
    ADR 0044's review check made a test)."""

    def test_every_folded_name_is_registered_and_nemotron_still_switches_it_on(self):
        self.assertLessEqual(FOLDED, set(NAMES))
        self.assertEqual(
            FOLDED,
            {
                "action_commit",
                "literal_guard",
                "repeat_call",
                "hint_as_user",
                "text_form_tool_call_cut_off",
                "whitespace_runaway",
                "refused_call_context",
            },
        )
        # Retained Journeys and their Forks resolve from the retained list.
        self.assertLessEqual(
            FOLDED,
            set(NEMOTRON_ULTRA_ENDPOINT_POLICY.settings_selection.injection_patterns),
        )

    def test_the_slot_sends_no_folded_pattern_and_nothing_beside_a_correction(self):
        # Every pattern switched on, over a run that hits each folded trigger.
        replies = [
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c1")]),
            ModelResponse(parts=[ToolCallPart("search_journey", SEARCH, "c2")]),
            ModelResponse(parts=[TextPart("I will do it.")]),
            ModelResponse(parts=[ToolCallPart("inspect_input_metadata", {}, "c4")]),
        ]
        _, events, _, _ = _plan(
            NEMOTRON,
            replies,
            invocation_settings={"injection": {name: {"on": True} for name in NAMES}},
        )
        sent = Counter()
        for request in _of(events, "ModelRequestRecorded"):
            parts = _pending(request["payload"])
            names = [
                injection.injected_pattern(part.get("content"))
                for part in parts
                if part["part_kind"] == "user-prompt"
            ]
            sent.update(filter(None, names))
            carries = any(
                part["part_kind"] == "retry-prompt"
                or (part["part_kind"] == "tool-return" and part["outcome"] == "failed")
                for part in parts
            )
            if carries:
                self.assertEqual(list(filter(None, names)), [], parts)
        self.assertEqual(set(sent) & FOLDED, set())


if __name__ == "__main__":
    unittest.main()
