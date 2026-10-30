"""Admission reads every reply before any call runs (#336, ADR 0044).

The two runaway replies of #326's live Nemotron Run (Journey 9c7e7e49, 2,632 and
3,503 calls) are replayed through the real planning loop; only the model and
source adapters are scripted (``tests.turn_routing_replays`` keeps the replay).
"""

import asyncio
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.usage import RequestUsage

from bio_neb.agentic_journey import action_admission, injection
from bio_neb.application.advancement import (
    AnswerReview,
    JunctionReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.investigation.investigation import project_investigation
from bio_neb.investigation.investigation_commands import BeginInvestigationCommand
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition
from tests.reference_fakes import ReferenceAdapter
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    SourceResponsesOnly,
    begin_in_pass,
    members_one_at_a_time,
    run_attempt,
)
from tests.terminal_journey_fakes import forked_child
from tests.turn_routing_replays import PROMPT as _PROMPT
from tests.turn_routing_replays import Replays as _Replays
from tests.turn_routing_replays import of as _of
from tests.turn_routing_replays import recorded_reply, replay_326


class ReplayOf326Tests(unittest.TestCase):
    def _replay(self, reply, **registration):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return replay_326(Path(directory.name), reply, **registration)

    def _ran(self, turn):
        """The Tools of the turn's calls that started, in the order they ran."""
        validations = {
            e["event_id"]: e["payload"] for e in _of(turn, "ToolCallValidated")
        }
        return [
            validations[e["payload"]["validation_event_id"]]["tool_name"]
            for e in _of(turn, "ObservedToolExecutionStarted")
        ]

    def _assert_routed(self, reply, *, cut_off):
        events, response, routed, turn, after = self._replay(reply)
        payload = routed["payload"]
        calls = [part for part in reply.parts if isinstance(part, ToolCallPart)]
        by_id = {part.tool_call_id: part for part in calls}

        # Copies collapse: each distinct call ran once.
        self.assertEqual(
            len(payload["copies"]), len(calls) - len(payload["run_order"]) - 1 - cut_off
        )
        # The unoffered companion lookup was refused on its own, before any call ran.
        (unoffered,) = [
            e
            for e in _of(events, "ToolCallValidated")
            if e["event_id"] in payload["unoffered_validation_event_ids"]
        ]
        self.assertEqual(
            (unoffered["payload"]["tool_name"], unoffered["payload"]["errors"]),
            ("describe_companion", ["tool_name_unavailable"]),
        )
        self.assertLess(events.index(unoffered), events.index(routed))
        # Distinct lawful calls run in one turn, in dependency order: the sample is
        # attested before the taxon lookup and the supplied choice before inspection.
        ran = self._ran(turn)
        expected = [
            "attest_sample",
            "resolve_taxon",
            "record_supplied_reference",
            "inspect_reference",
            "search_journey",
        ]
        self.assertEqual(ran, expected if cut_off else [*expected, "search_journey"])
        self.assertEqual(
            [by_id[call_id].tool_name for call_id in payload["run_order"]], ran
        )
        # A cut-off reply's partial last call is dropped, never validated.
        self.assertEqual(
            payload["dropped_call_id"], calls[-1].tool_call_id if cut_off else None
        )
        self.assertEqual(
            calls[-1].tool_call_id
            in {e["payload"]["tool_call_id"] for e in _of(events, "ToolCallValidated")},
            not cut_off,
        )
        # One note names the faults with their reply's place in the failed turns
        # in a row and the default limit of four, after the turn's results.
        note = payload["feedback"]
        self.assertIn("Copied calls:", note)
        self.assertIn("failed turn 1 of 4 in a row", note)
        self.assertIn("`describe_companion`", note)
        # The cut-off is one more item of the same note, counted the same way.
        self.assertEqual("cut at" in note, cut_off)
        keys = [(f["key"], f["count"]) for f in payload["faults"]]
        self.assertEqual(
            keys,
            [("copied_calls", 1), ("unoffered_calls", 1)]
            + ([("output_truncated", 1)] if cut_off else []),
        )
        truncations = _of(events, "ModelCompletionTruncated")
        self.assertEqual(
            [
                (t["payload"]["correction_count"], t["payload"]["feedback"])
                for t in truncations
            ],
            [(1, note)] if cut_off else [],
        )
        self.assertIsNotNone(after)
        sent = after["payload"]["compatible_adapter_evidence"]["request"]["messages"]
        # The kept reply holds only the calls that ran; the raw reply stays recorded.
        (kept,) = [
            message
            for message in sent
            if message["kind"] == "response"
            and any(part.get("tool_call_id") in by_id for part in message["parts"])
        ]
        self.assertEqual(
            [
                part["tool_call_id"]
                for part in kept["parts"]
                if part["part_kind"] == "tool-call"
            ],
            list(payload["run_order"]),
        )
        # The raw reply's wire reading stays with the raw reply, so later requests
        # do not carry its thousands of calls again.
        self.assertIsNone(kept["provider_details"])
        self.assertLess(len(json.dumps(sent)), 200_000)
        self.assertEqual(
            sum(
                part["part_kind"] == "tool-call"
                for part in response["payload"]["compatible_adapter_evidence"][
                    "response"
                ]["message"]["parts"]
            ),
            len(calls),
        )
        last = sent[-1]["parts"]
        self.assertEqual([part["part_kind"] for part in last][-1], "retry-prompt")
        self.assertEqual(last[-1]["content"], note)

    def test_reply_4_as_completed_runs_each_distinct_lawful_call_once_in_order(self):
        self._assert_routed(recorded_reply(4, finish_reason="stop"), cut_off=False)

    def test_reply_4_cut_off_runs_its_complete_calls_and_drops_its_partial_call(self):
        self._assert_routed(recorded_reply(4), cut_off=True)

    def test_reply_5_cut_off_is_routed_the_same_way(self):
        self._assert_routed(recorded_reply(5), cut_off=True)


class NemotronRoutedNoteTests(unittest.TestCase):
    """#326's reply 4 on the Nemotron runtime: the one note is rendered from
    Nemotron's wording, slot by slot (#242 fitted to #336's slots, #356)."""

    _replay = ReplayOf326Tests._replay

    def test_the_routed_note_is_nemotrons_rendering_and_the_next_requests_last_part(
        self,
    ):
        events, _, routed, _, after = self._replay(
            recorded_reply(4), model="nvidia/Nemotron-3-Ultra-550b-a55b"
        )
        # Only the Tools callable before the sample is attested are allowed; the
        # stage's others wait on their step. The paper Tools are the plan and paper
        # stages', never the reference stage's.
        offered = (
            "`attest_sample`, `record_supplied_reference`, `search_journey`.\n"
            "Waiting on their step, not callable now: `discover_assemblies`, "
            "`inspect_reference`, `record_reference_choice`, "
            "`request_planning_hitl`, `resolve_taxon`"
        )
        note = (
            "Copied calls: you sent `search_journey` 2626 times.\n"
            "Kept: the first of each; the copies were dropped.\n"
            "Allowed form: each distinct call once per reply.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it.\n"
            "Unoffered calls: this stage does not offer these Tools; each was "
            "refused before execution.\n"
            "Refused before execution (tool_name_unavailable).\n"
            "Tool `describe_companion`: not offered now. It works in the `companions` "
            "stage; the current stage is `reference`. `verified reference` has no "
            "value: Companion work opens once the reference is verified.\n"
            # The routed turn dropped the call: its trace.
            'You sent it with {"asset": "genomic_fasta", "accession": '
            '"GCF_000001405.40"}.\n'
            f"Allowed now: one of {offered}.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it.\n"
            "Your reply was cut at 221319 tokens; it held 2632 tool calls.\n"
            "Last call `search_journey`: it reached the ceiling and was dropped.\n"
            "Allowed form: call `search_journey` again in a later reply if you still "
            "need it.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it."
        )

        self.assertEqual(routed["payload"]["feedback"], note)
        self.assertEqual(
            [(f["key"], f["count"]) for f in routed["payload"]["faults"]],
            [("copied_calls", 1), ("unoffered_calls", 1), ("output_truncated", 1)],
        )
        sent = after["payload"]["compatible_adapter_evidence"]["request"]["messages"]
        last = sent[-1]["parts"][-1]
        self.assertEqual((last["part_kind"], last["content"]), ("retry-prompt", note))
        # The cut-off counts on its own truncation fact, with the same note.
        (truncation,) = _of(events, "ModelCompletionTruncated")
        self.assertEqual(truncation["payload"]["feedback"], note)


def _call(name, arguments, call_id):
    return ToolCallPart(name, arguments, tool_call_id=call_id)


INSPECT = '{"accession": "GCF_000001405.40"}'


class RoutedTurnTests(unittest.TestCase):
    """One scripted reply at the first reference-stage request, routed by the real
    planning loop and reopened by the public reader (#336 round 1)."""

    def _route(self, *parts, finish_reason="stop"):
        reply = ModelResponse(
            parts=list(parts),
            finish_reason=finish_reason,
            usage=RequestUsage(output_tokens=100),
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, paths = _registered(Path(directory.name), prompt_text=_PROMPT)
        with use_run_advancement_composition(_Replays(receipt, paths, reply)):
            advance(ResumeRun(journey_file=receipt.journey_file))
        review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        events = read_evidence(Journey.open(Path(receipt.journey_file)))
        ids = {part.tool_call_id for part in parts if isinstance(part, ToolCallPart)}
        (response,) = [
            e
            for e in _of(events, "ModelResponseRecorded")
            if ids
            & {
                part.get("tool_call_id")
                for part in e["payload"]["compatible_adapter_evidence"]["response"][
                    "message"
                ]["parts"]
            }
        ]
        (routed,) = [
            e
            for e in _of(events, "AgentTurnRouted")
            if e["payload"]["response_event_id"] == response["event_id"]
        ]
        turn = events[events.index(routed) + 1 :]
        after = _of(turn, "ModelRequestRecorded")[0]
        turn = turn[: turn.index(after)]
        self.journey_file = receipt.journey_file
        return events, routed["payload"], turn, after

    def _validated(self, events, call_id):
        return [
            e["payload"]["errors"]
            for e in _of(events, "ToolCallValidated")
            if e["payload"]["tool_call_id"] == call_id
        ]

    def test_a_cut_off_reply_s_last_call_is_dropped_even_when_it_parses(self):
        # The provider closes a cut call's JSON (recorded replies ended in
        # `{}` and in a search without its limit), and the wire sends text ahead
        # of the calls, so nothing shows the last call ended; the note asks for
        # it again.
        events, routed, turn, _ = self._route(
            _call("attest_sample", "{}", "w-1"),
            _call(
                "search_journey",
                '{"kinds": ["ResearchRequestRegistered"], "limit": 19}',
                "w-2",
            ),
            finish_reason="length",
        )

        self.assertEqual(routed["run_order"], ["w-1"])
        self.assertEqual(routed["dropped_call_id"], "w-2")
        self.assertEqual(self._validated(events, "w-2"), [])
        self.assertEqual(
            routed["feedback"],
            "Your reply was cut at 100 tokens; it held 2 tool calls.\n"
            "Last call `search_journey`: it reached the ceiling and was dropped.\n"
            "Allowed form: call `search_journey` again in a later reply if you still "
            "need it.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it.",
        )

    def test_the_note_claims_nothing_admission_then_refuses(self):
        # The kept copy of an inspection is refused before any reference choice,
        # and the cut call is dropped: nothing runs, and the note says only what
        # routing did.
        events, routed, turn, after = self._route(
            _call("inspect_reference", INSPECT, "c-1"),
            _call("inspect_reference", INSPECT, "c-2"),
            _call("search_journey", "{}", "c-3"),
            finish_reason="length",
        )

        self.assertEqual(routed["copies"], {"c-2": "c-1"})
        self.assertEqual(routed["dropped_call_id"], "c-3")
        self.assertEqual(self._validated(events, "c-1"), [["reference_choice_absent"]])
        self.assertEqual(_of(turn, "ObservedToolExecutionStarted"), [])
        note = routed["feedback"]
        self.assertNotRegex(note, r"\bran\b")
        self.assertEqual(
            note,
            "Copied calls: you sent `inspect_reference` 2 times.\n"
            "Kept: the first of each; the copies were dropped.\n"
            "Allowed form: each distinct call once per reply.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it.\n"
            "Your reply was cut at 100 tokens; it held 3 tool calls.\n"
            "Last call `search_journey`: it reached the ceiling and was dropped.\n"
            "Allowed form: call `search_journey` again in a later reply if you still "
            "need it.\n"
            "Count: failed turn 1 of 4 in a row. At 4 the Attempt stops and the "
            "researcher reviews it.",
        )
        # The refusal is the call's own result; the note follows it, once.
        parts = after["payload"]["compatible_adapter_evidence"]["request"]["messages"][
            -1
        ]["parts"]
        self.assertEqual(
            [(p["part_kind"], p.get("tool_name")) for p in parts],
            [("tool-return", "inspect_reference"), ("retry-prompt", None)],
        )
        self.assertIn("was not run", parts[0]["content"])
        self.assertEqual(parts[-1]["content"], note)

    def test_reopen_refuses_a_changed_duplicated_or_displaced_note(self):
        events, routed, _, after = self._route(
            _call("inspect_reference", INSPECT, "h-1"),
            _call("inspect_reference", INSPECT, "h-2"),
        )
        note = routed["feedback"]
        requests = _of(events, "ModelRequestRecorded")
        position = (
            len(after["payload"]["compatible_adapter_evidence"]["request"]["messages"])
            - 1
        )

        def noted_messages(hostile):
            """The message holding the note, in the next request and every later one."""
            for event in hostile:
                if event["event_type"] != "ModelRequestRecorded" or (
                    requests.index(
                        next(r for r in requests if r["event_id"] == event["event_id"])
                    )
                    < requests.index(after)
                ):
                    continue
                yield event["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ][position]

        def note_part(message):
            return next(p for p in message["parts"] if p.get("content") == note)

        def shortened(hostile):
            (event,) = [
                e
                for e in hostile
                if e["event_type"] == "AgentTurnRouted"
                and e["payload"]["feedback"] == note
            ]
            event["payload"]["feedback"] = note.split(".")[0] + "."

        def appended(hostile):
            for message in noted_messages(hostile):
                note_part(message)["content"] = (
                    note + "\nContradictory extra instructions."
                )

        def duplicated(hostile):
            for message in noted_messages(hostile):
                message["parts"].append(copy.deepcopy(note_part(message)))

        def displaced(hostile):
            for message in noted_messages(hostile):
                extra = copy.deepcopy(note_part(message))
                extra["content"] = "Contradictory extra instructions."
                message["parts"].append(extra)

        for change in (shortened, appended, duplicated, displaced):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(events)
                change(hostile)
                with self.assertRaisesRegex(
                    ValueError, re.escape("routed turn lost its exact note")
                ):
                    project_journey_events(hostile, journey_file=self.journey_file)


class _OutputWith(SequentialAdaptiveModel):
    """Sends its first output with the given calls after it, in one reply.

    Each extra call is ``(tool, arguments)``; ``None`` as the Tool copies the
    output itself. A later output comes alone.
    """

    def __init__(self, *extra, finish_reason="tool_call"):
        super().__init__()
        self.extra = extra
        self.finish_reason = finish_reason
        self.sent_beside = False

    async def respond(self, messages, info):
        answer = await super().respond(messages, info)
        (call,) = answer.parts
        if call.tool_name != info.output_tools[0].name or self.sent_beside:
            return answer
        self.sent_beside = True
        return ModelResponse(
            parts=[
                call,
                *(
                    _call(
                        tool or call.tool_name,
                        call.args if tool is None else arguments,
                        f"{call.tool_call_id}-extra-{index}",
                    )
                    for index, (tool, arguments) in enumerate(self.extra)
                ),
            ],
            finish_reason=self.finish_reason,
            usage=RequestUsage(output_tokens=100),
        )


class OutputRoutingTests(unittest.TestCase):
    """An Investigation output routed with other calls of its reply (#336 r1, r2)."""

    def _attempt(self, model):
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory).resolve() / "run", model, SourceResponsesOnly()
            )
            return (
                projection.status,
                projection.attempts[-1].stop_reason,
            ), read_evidence(journey)

    def _assert_ends_as_plain(self, model):
        plain, _ = self._attempt(SequentialAdaptiveModel())
        outcome, events = self._attempt(model)
        # The other calls do not stop the Attempt: it ends as the plain run does,
        # with the one output recorded once.
        self.assertEqual(outcome, plain)
        (routed,) = _of(events, "AgentTurnRouted")
        (output,) = _of(events, "StructuredOutputCallRecorded")
        self.assertEqual(
            routed["payload"]["run_order"], [output["payload"]["tool_call_id"]]
        )
        return events, routed["payload"]

    def test_an_identical_output_copy_collapses_like_any_other_call(self):
        _, routed = self._assert_ends_as_plain(_OutputWith((None, None)))

        ((copy_id, original),) = routed["copies"].items()
        self.assertEqual(copy_id, original + "-extra-0")
        self.assertEqual(routed["faults"], [{"key": "copied_calls", "count": 1}])

    def test_an_output_beside_unoffered_calls_completes(self):
        for extra in (
            [("unoffered_probe", {})],
            [("unoffered_probe", {}), ("other_probe", {"x": 1})],
        ):
            with self.subTest(unoffered=len(extra)):
                events, routed = self._assert_ends_as_plain(_OutputWith(*extra))

                refused = [
                    e["payload"]["tool_name"]
                    for e in _of(events, "ToolCallValidated")
                    if e["event_id"] in routed["unoffered_validation_event_ids"]
                ]
                self.assertEqual(refused, [tool for tool, _ in extra])
                self.assertEqual(
                    routed["faults"], [{"key": "unoffered_calls", "count": 1}]
                )

    def test_a_cut_off_output_reply_keeps_its_output(self):
        events, routed = self._assert_ends_as_plain(
            _OutputWith(
                ("unoffered_probe", {}),
                ("search_journey", {}),
                finish_reason="length",
            )
        )

        self.assertTrue(routed["dropped_call_id"].endswith("-extra-1"))
        self.assertEqual(
            [f["key"] for f in routed["faults"]],
            ["unoffered_calls", "output_truncated"],
        )

    def _refused_beside(self, model):
        """The Attempt ends as the plain run does; its first output was refused
        at routing and only the later one, sent alone, is recorded."""
        plain, _ = self._attempt(SequentialAdaptiveModel())
        outcome, events = self._attempt(model)
        self.assertEqual(outcome, plain)
        # The Investigation reader opens the Journey.
        project_investigation(events)
        (routed,) = _of(events, "AgentTurnRouted")
        (response,) = [
            e
            for e in _of(events, "ModelResponseRecorded")
            if e["event_id"] == routed["payload"]["response_event_id"]
        ]
        raw = [
            part["tool_call_id"]
            for part in response["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
            if part.get("tool_name") == "compile_proposal"
        ]
        # Each output the reply sent was refused, or collapsed onto one that was.
        self.assertEqual(
            [
                *routed["payload"]["refused_output_call_ids"],
                *routed["payload"]["copies"],
            ],
            raw,
        )
        (output,) = _of(events, "StructuredOutputCallRecorded")
        self.assertNotIn(output["payload"]["tool_call_id"], raw)
        note = routed["payload"]["feedback"]
        self.assertIn(
            "alone in its own reply, after you read this turn's results", note
        )
        # The turn's note is the last part of the request after the routed turn,
        # behind the kept calls' own results.
        later = _of(events[events.index(routed) :], "ModelRequestRecorded")[0]
        parts = later["payload"]["compatible_adapter_evidence"]["request"]["messages"][
            -1
        ]["parts"]
        self.assertEqual(
            [p.get("tool_call_id") for p in parts[:-1]], routed["payload"]["run_order"]
        )
        self.assertEqual(parts[-1]["content"], note)
        return events, routed["payload"], parts[:-1]

    def test_an_output_beside_a_refused_call_is_refused_with_one_note(self):
        events, routed, results = self._refused_beside(
            _OutputWith(("fetch_assembly_report", "{not json"))
        )

        (call_id,) = routed["run_order"]
        (validation,) = [
            e["payload"]
            for e in _of(events, "ToolCallValidated")
            if e["payload"]["tool_call_id"] == call_id
        ]
        self.assertEqual(validation["errors"], ["tool_arguments_invalid_json"])
        self.assertEqual(routed["faults"], [{"key": "output_not_alone", "count": 1}])
        # The refused call's own correction is its result.
        self.assertIn("tool_arguments_invalid_json", json.dumps(results))

    def test_a_lawful_call_beside_a_refused_output_runs_and_its_result_is_sent(self):
        # The owner's condition: no call of the reply is lost.
        events, routed, results = self._refused_beside(
            _OutputWith(("search_journey", {"text": "rs429358", "limit": 5}))
        )

        (call_id,) = routed["run_order"]
        (started,) = [
            e["event_id"]
            for e in _of(events, "ObservedToolExecutionStarted")
            if e["payload"]["tool_call_id"] == call_id
        ]
        ran = [
            e
            for e in _of(events, "ObservedToolExecutionSucceeded")
            if e["payload"]["started_event_id"] == started
        ]
        self.assertEqual(len(ran), 1)
        ((result,),) = [[p for p in results if p["tool_call_id"] == call_id]]
        self.assertEqual(
            (result["part_kind"], result.get("outcome")), ("tool-return", "success")
        )

    def test_a_routed_reply_refuses_its_output_and_admits_each_call_as_it_runs(self):
        events, routed, results = self._refused_beside(
            _OutputWith(
                (None, None),
                ("search_journey", {"text": "rs429358", "limit": 5}),
                ("fetch_assembly_report", "{not json"),
            )
        )

        search, fetch = routed["run_order"]
        ((copy_id, output_id),) = routed["copies"].items()
        self.assertEqual(routed["refused_output_call_ids"], [output_id])
        self.assertEqual(
            routed["faults"],
            [
                {"key": "copied_calls", "count": 1},
                {"key": "output_not_alone", "count": 1},
            ],
        )
        # Each kept call was admitted as it ran: the search ran, the malformed
        # fetch was refused with its own correction as its result.
        turns = {
            e["payload"]["validation_event_ids"][0]: e["payload"]["disposition"]
            for e in _of(events, "AgentActionTurnDispositionRecorded")
            if e["payload"]["response_event_id"] == routed["response_event_id"]
        }
        by_call = {
            e["payload"]["tool_call_id"]: turns[e["event_id"]]
            for e in _of(events, "ToolCallValidated")
            if e["event_id"] in turns
        }
        self.assertEqual(
            by_call, {search: "multi_call_admitted", fetch: "multi_call_refused"}
        )
        self.assertEqual([p.get("outcome") for p in results], ["success", "failed"])

    def test_an_output_beside_a_call_that_runs_is_not_a_failed_turn(self):
        # A reply in which a call ran is progress, so the refused output beside
        # it never stops the Attempt, even at a limit of one failed turn; the
        # output comes again alone and the Attempt ends as plain.
        plain, _ = self._attempt(SequentialAdaptiveModel())
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory).resolve() / "run",
                _OutputWith(("search_journey", {"text": "rs429358", "limit": 5})),
                SourceResponsesOnly(),
                continuation_policy={"failed_turns": 1},
            )
            events = read_evidence(journey)

        self.assertEqual(
            (projection.status, projection.attempts[-1].stop_reason), plain
        )
        self.assertEqual(_of(events, "AgentWorkReviewRequested"), [])
        (routed,) = _of(events, "AgentTurnRouted")
        self.assertEqual(
            routed["payload"]["faults"], [{"key": "output_not_alone", "count": 1}]
        )
        # The reply's call ran; its output was refused unrecorded and sent again.
        after = events[events.index(routed) :]
        self.assertEqual(len(_of(after, "ObservedToolExecutionStarted")), 1)
        self.assertEqual(len(_of(events, "StructuredOutputCallRecorded")), 1)
        project_investigation(events)

    def test_the_output_reader_refuses_an_unnamed_record_before_the_routed_turn(self):
        events, routed = self._assert_ends_as_plain(
            _OutputWith(("unoffered_probe", {}), ("other_probe", {}))
        )
        hostile = copy.deepcopy(events)
        (turn,) = _of(hostile, "AgentTurnRouted")
        turn["payload"]["unoffered_validation_event_ids"] = routed[
            "unoffered_validation_event_ids"
        ][1:]

        with self.assertRaisesRegex(
            ValueError, "structured-output call changed its model response"
        ):
            project_investigation(hostile)

    def test_the_output_reader_refuses_a_record_of_an_output_its_turn_refused(self):
        events, routed = self._assert_ends_as_plain(
            _OutputWith(("unoffered_probe", {}))
        )
        hostile = copy.deepcopy(events)
        (turn,) = _of(hostile, "AgentTurnRouted")
        turn["payload"]["refused_output_call_ids"] = turn["payload"]["run_order"]
        turn["payload"]["run_order"] = []

        with self.assertRaisesRegex(
            ValueError, "structured-output call was refused by its routed turn"
        ):
            project_investigation(hostile)


class OutputReopenTests(unittest.TestCase):
    """The application's reopen reads an Investigation reply routed around its
    output, in a full Journey whose three members run in one Attempt."""

    def _begin(self, model):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "requests": [
                {
                    "request": f"Investigate {rsid} in sample-001.",
                    "target": {"rsid": rsid},
                    "normalized_question_action": "observe_variant",
                }
                for rsid in SequentialAdaptiveModel.rsids
            ],
        }
        journey = forked_child(root, investigation_request=request)
        request_file = root / "investigation-request.json"
        request_file.write_text(json.dumps(request), encoding="utf-8")
        pending = begin_in_pass(
            BeginInvestigationCommand(
                journey_file=str(journey.path), request_files=(str(request_file),)
            ),
            environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
            model_factory=lambda loaded, selected: model.model(loaded, selected),
            source_adapter_factory=lambda: SequentialSourceResponses(
                assembly_body=ASSEMBLY_BODY
            ),
        )
        # Every member reached the Investigation Review: no member stopped.
        self.assertIsNotNone(pending.pending_request)
        events = read_evidence(Journey.open(journey.path))
        project_journey_events(events, journey_file=str(journey.path))
        self.journey_file = str(journey.path)
        (routed,) = _of(events, "AgentTurnRouted")
        return events, routed["payload"]

    def test_reopen_reads_an_output_refused_beside_a_call(self):
        _, routed = self._begin(_OutputWith(("fetch_assembly_report", "{not json")))

        self.assertEqual(len(routed["refused_output_call_ids"]), 1)

    def test_reopen_refuses_a_refused_output_kept_unnamed_or_unkeyed(self):
        events, _ = self._begin(_OutputWith(("fetch_assembly_report", "{not json")))

        def kept(payload):
            # Kept beside the call it ran with, under another fault's key.
            payload["run_order"] = [
                *payload["refused_output_call_ids"],
                *payload["run_order"],
            ]
            payload.pop("refused_output_call_ids")
            payload["faults"] = [{"key": "unoffered_calls", "count": 1}]

        def unnamed(payload):
            payload.pop("refused_output_call_ids")

        def unkeyed(payload):
            payload["faults"] = [{"key": "unoffered_calls", "count": 1}]

        for change in (kept, unnamed, unkeyed):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(events)
                (turn,) = _of(hostile, "AgentTurnRouted")
                change(turn["payload"])
                with self.assertRaisesRegex(
                    ValueError, "routed reply changed its exact raw reply"
                ):
                    project_journey_events(hostile, journey_file=self.journey_file)

    # Members in order, so the routed-reply check meets the forgery first.
    @members_one_at_a_time()
    def test_reopen_refuses_an_output_refused_with_no_call_beside_it(self):
        events, _ = self._begin(_OutputWith(("unoffered_probe", {})))
        hostile = copy.deepcopy(events)
        (turn,) = _of(hostile, "AgentTurnRouted")
        payload = turn["payload"]
        payload["refused_output_call_ids"] = payload["run_order"]
        payload["run_order"] = []
        payload["faults"] = [
            *payload["faults"],
            {"key": "output_not_alone", "count": 1},
        ]

        with self.assertRaisesRegex(
            ValueError, "routed reply changed its exact raw reply"
        ):
            project_journey_events(hostile, journey_file=self.journey_file)

    def test_reopen_reads_a_lawful_call_run_beside_a_refused_output(self):
        _, routed = self._begin(
            _OutputWith(("search_journey", {"text": "rs429358", "limit": 5}))
        )

        self.assertEqual(routed["faults"], [{"key": "output_not_alone", "count": 1}])

    def test_reopen_reads_an_output_completed_beside_an_unoffered_call(self):
        # The output completes its member, so no request of that member follows
        # to carry the routed turn's note.
        _, routed = self._begin(_OutputWith(("unoffered_probe", {})))

        self.assertEqual(routed["faults"], [{"key": "unoffered_calls", "count": 1}])


class _TimesOutOnce(PlanningComposition):
    """Works as scripted; the gateway times out once on the first reference request."""

    def __init__(self, receipt, paths):
        super().__init__(receipt, paths)
        self.timed_out = False

    async def respond(self, messages, info):
        offered = {tool.name for tool in info.function_tools}
        if "attest_sample" in offered and not self.timed_out:
            self.timed_out = True
            raise ModelHTTPError(504, "scripted-model", "gateway timeout")
        return await super().respond(messages, info)


class _Stalls(PlanningComposition):
    """Works as scripted; the first ``stalls`` reference requests never answer."""

    def __init__(self, receipt, paths, stalls):
        super().__init__(receipt, paths)
        self.stalls = stalls

    async def respond(self, messages, info):
        offered = {tool.name for tool in info.function_tools}
        if "attest_sample" in offered and self.stalls:
            self.stalls -= 1
            await asyncio.sleep(3600)
        return await super().respond(messages, info)


class _StallsFirst(PlanningComposition):
    """Works as scripted; the Attempt's very first request never answers."""

    def __init__(self, receipt, paths):
        super().__init__(receipt, paths)
        self.stalled = False

    async def respond(self, messages, info):
        if not self.stalled:
            self.stalled = True
            await asyncio.sleep(3600)
        return await super().respond(messages, info)


# The stall limit, shortened so a send that never answers is cut off at once.
_STALL_FAST = {
    "scopes": {"role": {"planning_agent": {"model_contact": {"stall_seconds": 1.0}}}}
}


# A recorded C. elegans Run: each logical request that had a failed send, by member and
# request, with each send's outcome and seconds from send to answer (contact journals,
# read once offline). Every 504 came about 600 s in.
_CELEGANS_A1_CHAINS = (
    (
        "7df8a326",
        "8948ddd1",
        "504@601 504@601 504@599 504@599 504@601 504@602 504@600 504@602 504@599 504@602 200@2",
    ),
    (
        "ff378e48",
        "3ddc5411",
        "504@601 504@601 504@599 504@599 504@603 504@601 504@600 200@1",
    ),
    ("a55ce37f", "c1bafd4a", "504@601 504@601 504@599 200@362"),
    (
        "ff378e48",
        "6488213e",
        "504@601 504@598 504@598 504@601 504@602 504@600 504@599 504@602 504@602 504@599 504@602 cancelled",
    ),
    ("7df8a326", "aa2ba416", "504@600 200@1"),
    ("7df8a326", "27e0d5a0", "504@600 200@1"),
    ("7df8a326", "5de678d1", "504@601 504@600 504@600 504@599 200@3"),
    ("7df8a326", "cb719d67", "504@599 200@1"),
)


def _stalls_under_the_limit(outcomes: str, limit: float = 180) -> int:
    """How many sends in a row, from the first, gave no reply within the limit."""
    count = 0
    for outcome in outcomes.split():
        status, _, seconds = outcome.partition("@")
        if status == "200" and float(seconds) <= limit:
            break
        count += 1
    return count


class TurnFaultTests(unittest.TestCase):
    def setUp(self):
        # The availability backoff waits zero seconds.
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    def test_a_resend_after_a_gateway_timeout_is_counted_and_named_in_the_note(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), prompt_text=_PROMPT)
            with use_run_advancement_composition(_TimesOutOnce(receipt, paths)):
                advance(ResumeRun(journey_file=receipt.journey_file))
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

        (failed,) = _of(events, "ModelPhysicalContactFailed")
        self.assertEqual(
            (
                failed["payload"]["http_status_code"],
                failed["payload"]["policy_disposition"],
            ),
            (504, "automatic_retry_scheduled"),
        )
        (routed,) = _of(events, "AgentTurnRouted")
        response = next(
            e for e in events if e["event_id"] == routed["payload"]["response_event_id"]
        )
        self.assertEqual(response["payload"]["physical_contact_ordinal"], 2)
        self.assertEqual(
            routed["payload"]["faults"], [{"key": "model_resend", "count": 1}]
        )
        note = routed["payload"]["feedback"]
        self.assertIn("Sent: this request 2 times.\nCount: resend fault 1 of 1.", note)
        # Its one call ran, and the note followed its result.
        (call,) = routed["payload"]["run_order"]
        later = _of(events[events.index(routed) :], "ModelRequestRecorded")[0]
        parts = later["payload"]["compatible_adapter_evidence"]["request"]["messages"][
            -1
        ]["parts"]
        self.assertEqual(
            [(p["part_kind"], p.get("tool_call_id")) for p in parts][:1],
            [("tool-return", call)],
        )
        self.assertEqual(parts[-1]["content"], note)

    def test_a_stalled_reply_ends_its_request_and_the_next_carries_the_note(self):
        # No reply after the stall limit cuts the send off; the request is not
        # resent byte for byte, and the next one tells the model its reply never
        # arrived.
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory), prompt_text=_PROMPT, failure=_STALL_FAST
            )
            with use_run_advancement_composition(_Stalls(receipt, paths, 1)):
                advance(ResumeRun(journey_file=receipt.journey_file))
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

        (failed,) = _of(events, "ModelPhysicalContactFailed")
        self.assertEqual(
            (
                failed["payload"]["error_type"],
                failed["payload"]["failure_reason"],
                failed["payload"]["policy_disposition"],
                failed["payload"]["completion_certainty"],
            ),
            (
                "ModelContactStalled",
                "contact_stalled",
                "stall_noted",
                "completion_unknown",
            ),
        )
        self.assertGreaterEqual(failed["payload"]["contact_seconds"], 1.0)
        (stalled,) = _of(events, "ModelRequestStalled")
        self.assertEqual(
            stalled["payload"]["last_physical_contact_event_id"], failed["event_id"]
        )
        self.assertEqual(_of(events, "ModelRequestFailed"), [])
        following = _of(events[events.index(stalled) :], "ModelRequestRecorded")[0]
        self.assertEqual(
            following["payload"]["after_stall_event_id"], stalled["event_id"]
        )
        last = following["payload"]["compatible_adapter_evidence"]["request"][
            "messages"
        ][-1]["parts"][-1]
        self.assertEqual(last["part_kind"], "retry-prompt")
        self.assertIn(
            "No reply arrived within 1 s: it was cut off",
            last["content"],
        )
        self.assertIn("Count: stall fault 1 of 2.", last["content"])
        # The Attempt went on past the stall: the planning reply that followed ran.
        self.assertTrue(
            _of(events, "AgentTurnRouted") or _of(events, "ToolCallValidated")
        )

    def test_celegans_a1s_runaways_end_within_two_sends_per_request(self):
        # The ticket's replay: under the stall rule each recorded chain of
        # 10-minute 504s becomes at most two sends of at most the stall limit, then
        # the availability Review; a chain whose second send answered goes on.
        for member, request, outcomes in _CELEGANS_A1_CHAINS:
            stalls = _stalls_under_the_limit(outcomes)
            with self.subTest(member=member, request=request, stalls=stalls):
                with tempfile.TemporaryDirectory() as directory:
                    receipt, paths = _registered(
                        Path(directory), prompt_text=_PROMPT, failure=_STALL_FAST
                    )
                    with use_run_advancement_composition(
                        _Stalls(receipt, paths, stalls)
                    ):
                        advance(ResumeRun(journey_file=receipt.journey_file))
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    )
                    events = read_evidence(Journey.open(Path(receipt.journey_file)))
                failed = _of(events, "ModelPhysicalContactFailed")
                self.assertEqual(len(failed), min(stalls, 2))
                reviews = _of(events, "ModelAvailabilityReviewRequested")
                self.assertEqual(len(reviews), 1 if stalls >= 2 else 0)
                self.assertEqual(len(_of(events, "ModelRequestStalled")), 1)

    def test_reopen_refuses_a_changed_stall_note_or_stall_timing(self):
        # Reopen re-renders the note from the Attempt's wording and replays the
        # stall's cut-off from its retained seconds.
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory), prompt_text=_PROMPT, failure=_STALL_FAST
            )
            with use_run_advancement_composition(_Stalls(receipt, paths, 1)):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            project_journey_events(events, journey_file=receipt.journey_file)

            def note(hostile):
                (stalled,) = _of(hostile, "ModelRequestStalled")
                for request in _of(hostile, "ModelRequestRecorded"):
                    if request["payload"].get("after_stall_event_id"):
                        evidence = request["payload"]["compatible_adapter_evidence"]
                        part = evidence["request"]["messages"][-1]["parts"][-1]
                        part["content"] = (
                            "Send the same reply again without changing it."
                        )

            def kind(hostile):
                # The retry part kept, its message no longer a request.
                for request in _of(hostile, "ModelRequestRecorded"):
                    if request["payload"].get("after_stall_event_id"):
                        evidence = request["payload"]["compatible_adapter_evidence"]
                        evidence["request"]["messages"][-1]["kind"] = "response"

            def empty(hostile):
                # The reader accepts an empty request; it holds no note.
                for request in _of(hostile, "ModelRequestRecorded"):
                    if request["payload"].get("after_stall_event_id"):
                        evidence = request["payload"]["compatible_adapter_evidence"]
                        evidence["request"]["messages"][-1]["parts"] = []

            def seconds(hostile):
                (failed,) = _of(hostile, "ModelPhysicalContactFailed")
                failed["payload"]["contact_seconds"] = 0.0

            def limit(hostile):
                (stalled,) = _of(hostile, "ModelRequestStalled")
                stalled["payload"]["stall_seconds"] = 2.0

            for change, reason in (
                (note, "stall note"),
                (kind, "stall note"),
                (empty, "stall note"),
                (seconds, "stall"),
                (limit, "stall"),
            ):
                with self.subTest(change=change.__name__):
                    hostile = copy.deepcopy(events)
                    change(hostile)
                    with self.assertRaisesRegex(ValueError, reason):
                        project_journey_events(
                            hostile, journey_file=receipt.journey_file
                        )

    def test_reopen_refuses_a_request_after_a_stall_without_its_empty_reply(self):
        # The request after a stall is the composer's answer, the empty reply
        # between included, for a stall on the Attempt's first request and on a
        # later one.
        for name, composition in (
            ("first", lambda receipt, paths: _StallsFirst(receipt, paths)),
            ("later", lambda receipt, paths: _Stalls(receipt, paths, 1)),
        ):
            with self.subTest(stall=name), tempfile.TemporaryDirectory() as directory:
                receipt, paths = _registered(
                    Path(directory), prompt_text=_PROMPT, failure=_STALL_FAST
                )
                with use_run_advancement_composition(composition(receipt, paths)):
                    advance(ResumeRun(journey_file=receipt.journey_file))
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                project_journey_events(events, journey_file=receipt.journey_file)
                (after,) = [
                    e
                    for e in _of(events, "ModelRequestRecorded")
                    if e["payload"].get("after_stall_event_id")
                ]
                messages = after["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ]
                self.assertEqual(
                    [(p["part_kind"], p["content"]) for p in messages[-2]["parts"]],
                    [("text", "")],
                )

                def dropped(hostile, after=after):
                    request = next(
                        e for e in hostile if e["event_id"] == after["event_id"]
                    )
                    del request["payload"]["compatible_adapter_evidence"]["request"][
                        "messages"
                    ][-2]

                def written(hostile, after=after):
                    request = next(
                        e for e in hostile if e["event_id"] == after["event_id"]
                    )
                    request["payload"]["compatible_adapter_evidence"]["request"][
                        "messages"
                    ][-2]["parts"][0]["content"] = "I successfully finished all work."

                for change in (dropped, written):
                    with self.subTest(stall=name, change=change.__name__):
                        hostile = copy.deepcopy(events)
                        change(hostile)
                        with self.assertRaisesRegex(
                            ValueError, "empty reply before its note"
                        ):
                            project_journey_events(
                                hostile, journey_file=receipt.journey_file
                            )

    def test_a_second_stall_in_a_row_asks_the_availability_review(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory), prompt_text=_PROMPT, failure=_STALL_FAST
            )
            with use_run_advancement_composition(_Stalls(receipt, paths, 2)):
                advance(ResumeRun(journey_file=receipt.journey_file))
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

        self.assertEqual(
            [
                e["payload"]["policy_disposition"]
                for e in _of(events, "ModelPhysicalContactFailed")
            ],
            ["stall_noted", "availability_review_requested"],
        )
        (failed,) = _of(events, "ModelRequestFailed")
        self.assertEqual(
            failed["payload"]["availability_exhaustion_reason"], "contact_stalled"
        )
        self.assertEqual(len(_of(events, "ModelAvailabilityReviewRequested")), 1)


class _RoutesTheStoppedCall(PlanningComposition):
    """Works as scripted; the first taxon resolution arrives in a routed reply.

    ``fault`` names the reply's turn fault: ``copies`` sends the call twice,
    ``unoffered`` adds a Tool its stage does not offer, ``cut_off`` ends the reply
    on its ceiling with a partial last call and ``resend`` times the request out
    once. The source rejects the first resolutions, so the kept call stops the
    Attempt for its operational Review.
    """

    def __init__(self, receipt, paths, fault, *, rejections=1):
        super().__init__(receipt, paths)
        self.fault = fault
        self.routed = False
        self.timed_out = False
        self.sent = []
        self.reference_adapter = _Rejects(rejections)

    async def respond(self, messages, info):
        self.sent.append(messages)
        response = await super().respond(messages, info)
        (call,) = response.parts
        if self.routed or call.tool_name != "resolve_taxon":
            return response
        if self.fault == "resend" and not self.timed_out:
            self.timed_out = True
            raise ModelHTTPError(504, "scripted-model", "gateway timeout")
        self.routed = True
        extra = {
            "copies": [_call(call.tool_name, call.args, "copy-of-taxon")],
            "unoffered": [_call("describe_companion", "{}", "unoffered")],
            "cut_off": [_call("search_journey", "{}", "partial")],
            "resend": [],
        }[self.fault]
        return ModelResponse(
            parts=[call, *extra],
            finish_reason="length" if self.fault == "cut_off" else "stop",
            usage=RequestUsage(output_tokens=100),
        )


class _Rejects(ReferenceAdapter):
    """Rejects the first ``rejections`` taxon resolutions."""

    def __init__(self, rejections):
        super().__init__()
        self.rejections = rejections
        self.taxon_queries = 0

    def resolve_taxon(self, query):
        self.taxon_queries += 1
        if self.taxon_queries <= self.rejections:
            raise ReferenceAdapterError(
                "http_error", "Service rejected request.", retryable=False
            )
        return super().resolve_taxon(query)


class RoutedRetryTests(unittest.TestCase):
    """A routed reply's operational stop answered `retry` re-runs the stopped call
    and continues (#371): the turn's note follows the replayed call's result."""

    def setUp(self):
        # The availability backoff waits zero seconds.
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    @staticmethod
    def _retry(journey_file, review):
        return advance(
            AnswerReview(
                journey_file=journey_file,
                request_event_id=review.request_event_id,
                answer=JunctionReviewAnswer(
                    kind="junction",
                    response={
                        "request_id": review.request.request.request_id,
                        "action": "retry",
                        "by": "test researcher",
                    },
                ),
            )
        )

    def test_a_replayed_call_that_stops_again_keeps_the_note_owed(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), prompt_text=_PROMPT)
            composition = _RoutesTheStoppedCall(receipt, paths, "copies", rejections=2)
            with use_run_advancement_composition(composition):
                first = advance(ResumeRun(journey_file=receipt.journey_file))
                again = self._retry(receipt.journey_file, first.current_review)
                # The replay stopped before any model request: the second
                # Review's pending request still ends in the unsent note.
                self.assertEqual(again.status, "operational_review_pending")
                sent = len(composition.sent)
                resumed = self._retry(receipt.journey_file, again.current_review)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        (routed,) = _of(events, "AgentTurnRouted")
        note = routed["payload"]["feedback"]
        for stop in _of(events, "AgentFailureReviewRequested"):
            last = stop["payload"]["messages"][-1]["parts"][-1]
            self.assertEqual(
                (last["part_kind"], last.get("content")), ("retry-prompt", note)
            )
        self.assertEqual(len(_of(events, "AgentFailureReviewRequested")), 2)
        self.assertEqual(composition.reference_adapter.taxon_queries, 3)
        self.assertEqual(resumed.status, "planning_hitl_pending")
        parts = composition.sent[sent][-1].parts
        self.assertEqual(
            [part.content for part in parts if part.part_kind == "retry-prompt"],
            [note],
        )
        self.assertEqual(parts[-1].content, note)

    def test_reopen_refuses_a_resumed_request_that_changed_its_note(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory), prompt_text=_PROMPT)
            composition = _RoutesTheStoppedCall(receipt, paths, "copies")
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                self._retry(receipt.journey_file, stopped.current_review)
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        (stop,) = _of(events, "AgentFailureReviewRequested")
        resumed = _of(events[events.index(stop) :], "ModelRequestRecorded")[0]

        def last_parts(hostile):
            (event,) = [e for e in hostile if e["event_id"] == resumed["event_id"]]
            return event["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ][-1]["parts"]

        changes = {
            "dropped": lambda parts: parts.pop(),
            "duplicated": lambda parts: parts.append(copy.deepcopy(parts[-1])),
            "altered": lambda parts: parts[-1].update(content="Another note."),
        }
        project_journey_events(events, journey_file=receipt.journey_file)
        for name, change in changes.items():
            with self.subTest(change=name):
                hostile = copy.deepcopy(events)
                change(last_parts(hostile))
                with self.assertRaisesRegex(
                    ValueError, re.escape("routed turn lost its exact note")
                ):
                    project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_a_retry_after_a_routed_reply_replays_the_stopped_call_and_sends_its_note(
        self,
    ):
        for fault in ("copies", "unoffered", "cut_off", "resend"):
            with (
                self.subTest(fault=fault),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, paths = _registered(Path(directory), prompt_text=_PROMPT)
                composition = _RoutesTheStoppedCall(receipt, paths, fault)
                with use_run_advancement_composition(composition):
                    stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                    self.assertEqual(stopped.status, "operational_review_pending")
                    review = stopped.current_review
                    sent = len(composition.sent)
                    resumed = self._retry(receipt.journey_file, review)
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                (routed,) = _of(events, "AgentTurnRouted")
                note = routed["payload"]["feedback"]
                self.assertIsNotNone(note)
                # The stopped call ran again, and the Attempt went on to its
                # next model request instead of holding the Run.
                self.assertEqual(composition.reference_adapter.taxon_queries, 2)
                self.assertEqual(resumed.status, "planning_hitl_pending")
                self.assertEqual(_of(events, "AttemptFailed"), [])
                (stop,) = _of(events, "AgentFailureReviewRequested")
                stopped_call = stop["payload"]["tool_call_id"]
                starts = {
                    e["event_id"]
                    for e in _of(events, "ObservedToolExecutionStarted")
                    if e["payload"]["tool_call_id"] == stopped_call
                }
                # The stopped run ended in its classified failure; the replay
                # is the call's second run.
                stopped_run, replayed = [
                    e
                    for e in _of(events, "ObservedToolExecutionSucceeded")
                    if e["payload"]["started_event_id"] in starts
                ]
                self.assertLess(events.index(stopped_run), events.index(stop))
                after = _of(events[events.index(replayed) :], "ModelRequestRecorded")
                parts = after[0]["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ][-1]["parts"]
                # The note is the next request's last part, after the replayed
                # call's result, once.
                self.assertEqual(
                    (parts[-2]["part_kind"], parts[-2]["tool_call_id"]),
                    ("tool-return", stopped_call),
                )
                self.assertEqual(
                    (
                        parts[-1]["part_kind"],
                        parts[-1]["tool_name"],
                        parts[-1]["content"],
                    ),
                    ("retry-prompt", None, note),
                )
                self.assertEqual(
                    sum(
                        isinstance(part.get("content"), str) and note in part["content"]
                        for message in after[0]["payload"][
                            "compatible_adapter_evidence"
                        ]["request"]["messages"]
                        for part in message.get("parts", [])
                    ),
                    1,
                )
                # The model read it last, too.
                self.assertEqual(composition.sent[sent][-1].parts[-1].content, note)
                retained = Path(receipt.journey_file).read_bytes()
                self.assertEqual(
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    ).status,
                    "planning_hitl_pending",
                )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), retained)

    def test_a_retry_after_an_output_refused_beside_the_stopped_call_sends_its_note(
        self,
    ):
        from bio_neb.agentic_journey.operational_review import (
            answer_operational_review,
            project_operational_review,
        )
        from bio_neb.application.queries import read_investigation
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        class OutputBesideTheLookup(SequentialAdaptiveModel):
            sent_beside = False

            async def respond(self, messages, info):
                self.sent = getattr(self, "sent", [])
                self.sent.append(messages)
                answer = await super().respond(messages, info)
                (call,) = answer.parts
                if call.tool_name != "fetch_refsnp" or self.sent_beside:
                    return answer
                self.sent_beside = True
                return ModelResponse(
                    parts=[call, _call(info.output_tools[0].name, "{}", "early")]
                )

        class RejectsOnce(SourceResponsesOnly):
            def fetch_refsnp(self, rsid):
                if not self.operations:
                    self.operations.append("fetch_refsnp:" + rsid)
                    raise ProviderError("rejected", "Source denied this request.")
                return super().fetch_refsnp(rsid)

        with tempfile.TemporaryDirectory() as directory:
            model = OutputBesideTheLookup()
            source = RejectsOnce()
            _, journey = run_attempt(Path(directory) / "run", model, source)
            (request,) = project_operational_review(read_evidence(journey)).requests
            receipt, _ = answer_operational_review(
                journey,
                request.request_event_id,
                {
                    "request_id": request.request.request_id,
                    "action": "retry",
                    "by": "test researcher",
                },
            )
            sent = len(model.sent)
            investigation_assignment(
                model_factory=model.model, source_adapter_factory=lambda: source
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                operational_review_response_event_id=receipt.response_event_id,
            )
            events = read_evidence(journey)
            (routed,) = _of(events, "AgentTurnRouted")
            self.assertEqual(routed["payload"]["refused_output_call_ids"], ["early"])
            note = routed["payload"]["feedback"]
            self.assertEqual(source.operations[:2], ["fetch_refsnp:rs429358"] * 2)
            self.assertEqual(_of(events, "InvestigationAttemptFailed"), [])
            # The replayed lookup's result, then the note, end the next request.
            parts = model.sent[sent][-1].parts
            self.assertEqual(
                [(part.part_kind, part.tool_call_id) for part in parts][-2],
                ("tool-return", request.tool_call_id),
            )
            self.assertEqual(
                (parts[-1].part_kind, parts[-1].tool_name, parts[-1].content),
                ("retry-prompt", None, note),
            )
            retained = journey.path.read_bytes()
            read_investigation(Journey.open(journey.path))
            self.assertEqual(journey.path.read_bytes(), retained)


def _speaks_every_request(signals, tuning):
    return "Test notice: keep to this stage's Tools."


_ALWAYS = injection.InjectionPattern(
    name="test_always", tuning=injection.PatternTuning, message=_speaks_every_request
)


class InjectedRoutedRetryTests(unittest.TestCase):
    """Lane N's Injection slot (ADR 0048) with #371's restored note: on the request
    after a retried routed reply, the note is sent once, last (#379), and no
    notice joins the request that carries it, for every model (#373, #445)."""

    setUp = RoutedRetryTests.setUp

    def _retried(self, fault, model=None):
        """The request after a retried routed reply's replayed call: its parts,
        the note, the stopped call and every message of that request."""
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(
                Path(directory),
                prompt_text=_PROMPT,
                agent_invocation={"injection": {_ALWAYS.name: {"on": True}}},
                **({} if model is None else {"model": model}),
            )
            composition = _RoutesTheStoppedCall(receipt, paths, fault)
            with use_run_advancement_composition(composition):
                stopped = advance(ResumeRun(journey_file=receipt.journey_file))
                self.assertEqual(stopped.status, "operational_review_pending")
                resumed = RoutedRetryTests._retry(
                    receipt.journey_file, stopped.current_review
                )
            self.assertEqual(resumed.status, "planning_hitl_pending")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            (routed,) = _of(events, "AgentTurnRouted")
            note = routed["payload"]["feedback"]
            (stop,) = _of(events, "AgentFailureReviewRequested")
            stopped_call = stop["payload"]["tool_call_id"]
            starts = {
                e["event_id"]
                for e in _of(events, "ObservedToolExecutionStarted")
                if e["payload"]["tool_call_id"] == stopped_call
            }
            replayed = [
                e
                for e in _of(events, "ObservedToolExecutionSucceeded")
                if e["payload"]["started_event_id"] in starts
            ][-1]
            after = _of(events[events.index(replayed) :], "ModelRequestRecorded")
            messages = after[0]["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]
            # The note is sent once, and the retained Run reopens.
            self.assertEqual(
                sum(
                    isinstance(part.get("content"), str) and note in part["content"]
                    for message in messages
                    for part in message.get("parts", [])
                ),
                1,
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                "planning_hitl_pending",
            )
        return messages[-1]["parts"], note, stopped_call

    def test_the_default_model_s_restored_note_is_sent_alone_after_a_retry(self):
        self.enterContext(patch.dict(injection.PATTERNS, {_ALWAYS.name: _ALWAYS}))
        for fault in ("copies", "unoffered", "cut_off", "resend"):
            with self.subTest(fault=fault):
                parts, note, stopped_call = self._retried(fault)
                self.assertEqual(
                    [part["part_kind"] for part in parts[-2:]],
                    ["tool-return", "retry-prompt"],
                )
                self.assertEqual(parts[-2]["tool_call_id"], stopped_call)
                self.assertEqual(
                    (parts[-1]["tool_name"], parts[-1]["content"]), (None, note)
                )
                self.assertFalse(
                    any(injection.injected_pattern(p.get("content")) for p in parts)
                )

    def test_nemotron_s_restored_note_is_sent_alone_after_a_retry(self):
        # On Nemotron no notice stacks on the request carrying the note, and the
        # restored note shows the same inspection facts as an ordinary send.
        self.enterContext(patch.dict(injection.PATTERNS, {_ALWAYS.name: _ALWAYS}))
        common = {"slot": "after_results", "last": []}
        expected = {
            "copies": common
            | {
                "fault_kinds": ["copied_calls"],
                "tools": ["resolve_taxon"],
                "templates": ["copied_calls", "fault"],
            },
            "unoffered": common
            | {
                "fault_kinds": ["unoffered_calls"],
                "tools": ["describe_companion"],
                "templates": [
                    "fault",
                    "refused",
                    "sent_call",
                    "tool_unoffered",
                    "unoffered_calls",
                ],
            },
            "cut_off": common
            | {
                "fault_kinds": ["cut_off"],
                "tools": ["search_journey"],
                "templates": ["cut_off_partial", "fault"],
            },
            "resend": common
            | {
                "fault_kinds": ["model_resend"],
                "tools": [],
                "templates": ["fault", "model_resend"],
            },
        }
        for fault in ("copies", "unoffered", "cut_off", "resend"):
            with self.subTest(fault=fault):
                shown = []
                with patch.object(
                    action_admission,
                    "inspection_event",
                    lambda name, shown=shown, **facts: shown.append((name, facts)),
                ):
                    parts, note, stopped_call = self._retried(
                        fault, model="nvidia/Nemotron-3-Ultra-550b-a55b"
                    )
                # The stop came before the note's first send, so the restored
                # send is its only one; it names the Tools and wording an
                # ordinary send would (Spec r3 S2).
                (restored,) = [
                    facts for name, facts in shown if facts.get("text") == note
                ]
                self.assertEqual(
                    {
                        key: restored[key]
                        for key in ("slot", "fault_kinds", "tools", "templates")
                    }
                    | {"last": restored["last_correction"]},
                    expected[fault],
                )
                self.assertEqual(
                    [part["part_kind"] for part in parts[-2:]],
                    ["tool-return", "retry-prompt"],
                )
                self.assertEqual(parts[-2]["tool_call_id"], stopped_call)
                self.assertEqual(
                    (parts[-1]["tool_name"], parts[-1]["content"]), (None, note)
                )
                self.assertFalse(
                    any(part["part_kind"] == "user-prompt" for part in parts)
                )


if __name__ == "__main__":
    unittest.main()
