"""The transcript composer decides what the model reads next (#444, ADR 0044).

Given a reply and what waits for the next request, it returns the kept reply and
the ordered parts of that request, with no Run. #387 celegans-a1's three ordering
runaways are its worked examples: a broken call stub replayed as text, the resend
note joined to a correction the model cannot act on, and two user messages back
to back after a dropped reply.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from bio_neb.agentic_journey import transcript
from bio_neb.agentic_journey.model_turn import (
    messages_record,
    read_messages_record,
    read_turn,
)


def _reply(*parts, finish_reason=None):
    return ModelResponse(parts=list(parts), finish_reason=finish_reason)


class UnusableReplyTests(unittest.TestCase):
    """The Model turn reader names each reply that carries nothing to act on."""

    def test_each_reply_with_no_call_is_read_by_what_it_held(self):
        cases = {
            # celegans-a1 7df8a326 request 1: the provider said a call came; only
            # "The" did.
            "call_stub": _reply(TextPart("The"), finish_reason="tool_call"),
            "empty_reply": _reply(finish_reason="tool_call"),
            "reasoning_only": _reply(
                ThinkingPart("Now the lookup."), finish_reason="tool_call"
            ),
        }
        for kind, reply in cases.items():
            with self.subTest(kind):
                turn = read_turn(reply)
                self.assertEqual((turn.calls, turn.unusable), ((), kind))

    def test_a_blank_reply_is_empty_and_a_finished_text_is_a_text_ending(self):
        self.assertEqual(
            read_turn(_reply(TextPart("  \n"), ThinkingPart(" "))).unusable,
            "empty_reply",
        )
        self.assertIsNone(
            read_turn(_reply(TextPart("Done."), finish_reason="stop")).unusable
        )

    def test_a_reply_with_a_call_or_cut_off_is_never_unusable(self):
        self.assertIsNone(
            read_turn(
                _reply(ToolCallPart("search_journey", "{}"), finish_reason="tool_call")
            ).unusable
        )
        self.assertIsNone(
            read_turn(_reply(TextPart("The"), finish_reason="length")).unusable
        )


def _record(reply):
    return messages_record([reply])[0]


def _parts(record):
    return [
        (part["part_kind"], part.get("content", part.get("tool_call_id")))
        for part in record["parts"]
    ]


class KeptReplyTests(unittest.TestCase):
    """What the next request shows of a reply: never a stub, never nothing."""

    def test_a_call_stub_is_kept_without_its_fragment_and_sends_empty_content(self):
        # celegans-a1: 14 stubs kept as text became echoes of the feedback.
        stub = _record(_reply(TextPart("The"), finish_reason="tool_call"))
        self.assertEqual(_parts(transcript.kept_reply(stub)), [("text", "")])

    def test_an_empty_or_reasoning_only_reply_is_padded_like_reasoning_only(self):
        empty = _record(_reply(finish_reason="tool_call"))
        self.assertEqual(_parts(transcript.kept_reply(empty)), [("text", "")])
        thought = _record(_reply(ThinkingPart("Now."), finish_reason="tool_call"))
        self.assertEqual(
            _parts(transcript.kept_reply(thought)), [("thinking", "Now."), ("text", "")]
        )

    def test_a_finished_text_ending_and_a_reply_that_runs_as_sent_are_unchanged(self):
        for reply in (
            _reply(TextPart("Done."), finish_reason="stop"),
            _reply(
                ThinkingPart("Look it up."),
                ToolCallPart("search_journey", "{}", tool_call_id="c1"),
            ),
        ):
            with self.subTest(reply.parts[0].part_kind):
                record = _record(reply)
                self.assertEqual(transcript.kept_reply(record), record)

    def test_a_routed_reply_keeps_only_its_kept_calls_in_run_order(self):
        record = _record(
            _reply(
                TextPart("Two lookups."),
                ToolCallPart("b", "{}", tool_call_id="b1"),
                ToolCallPart("a", "{}", tool_call_id="a1"),
                ToolCallPart("b", "{}", tool_call_id="b2"),
            )
        )
        kept = transcript.kept_reply(record, ("a1", "b1"))
        self.assertEqual(
            _parts(kept),
            [("text", "Two lookups."), ("tool-call", "a1"), ("tool-call", "b1")],
        )
        self.assertIsNone(kept["provider_details"])

    def test_a_reply_whose_calls_did_not_run_and_held_no_text_is_padded(self):
        # Qwen's next request once dropped such a reply, so two user
        # messages followed each other.
        record = _record(
            _reply(
                ThinkingPart("Inspect the sample."),
                ToolCallPart("search_journey", "{}", tool_call_id="c1"),
            )
        )
        self.assertEqual(
            _parts(transcript.kept_reply(record, ())),
            [("thinking", "Inspect the sample."), ("text", "")],
        )

    def test_every_kept_reply_reads_back_as_a_framework_reply(self):
        stub = transcript.kept_reply(
            _record(_reply(TextPart("public"), finish_reason="tool_call"))
        )
        (reply,) = read_messages_record([stub])
        self.assertEqual([part.content for part in reply.parts], [""])


def _part(part):
    """A request part as the Journey retains it."""
    return messages_record([ModelRequest(parts=[part])])[0]["parts"][0]


def _result(name, call_id, content="ok", outcome="success"):
    return _part(ToolReturnPart(name, content, tool_call_id=call_id, outcome=outcome))


def _retry(text):
    return _part(RetryPromptPart(text))


def _request(*parts):
    return messages_record([ModelRequest(parts=list(parts))])[0]


def _shape(parts):
    """Each part by kind and the text the model reads in it."""
    return [(part["part_kind"], part["content"]) for part in parts]


_NOTICE = "[injection:plan_review]\n<loop_notice>\nPlan first.\n</loop_notice>"
_FOLDED = "[injection:action_commit]\n<loop_notice>\nMake the call.\n</loop_notice>"


class RequestOrderTests(unittest.TestCase):
    """The ordered parts of the next request, by the one slot rule (ADR 0044,
    0048): each fault is one message, and nothing joins a correction (#445)."""

    def compose(self, **pending):
        return _shape(transcript.request_parts(transcript.Pending(**pending)))

    def test_the_turns_note_follows_its_results_and_comes_last(self):
        self.assertEqual(
            self.compose(
                parts=(_result("a", "c1"), _result("b", "c2")),
                note="Copied calls dropped.",
            ),
            [
                ("tool-return", "ok"),
                ("tool-return", "ok"),
                ("retry-prompt", "Copied calls dropped."),
            ],
        )

    def test_no_notice_joins_a_request_that_carries_a_correction(self):
        for pending in (
            {"parts": (_result("a", "c1"),), "note": "Copied calls dropped."},
            {"parts": (_retry("Send the call."),)},
            {"parts": (_result("a", "c1", "Refused.", outcome="failed"),)},
        ):
            with self.subTest(pending=pending):
                self.assertEqual(
                    self.compose(**pending, notices=(_NOTICE,)),
                    self.compose(**pending),
                )

    def test_a_notice_follows_clean_results_but_a_folded_one_never_speaks(self):
        self.assertEqual(
            self.compose(parts=(_result("a", "c1"),), notices=(_NOTICE, _FOLDED)),
            [("tool-return", "ok"), ("user-prompt", _NOTICE)],
        )

    def test_a_refused_single_call_is_answered_in_its_own_slot(self):
        answer = _result("a", "c1", "Refused: a.", outcome="failed")
        self.assertEqual(
            self.compose(parts=(_retry("Refused: a."),), answer=answer),
            [("tool-return", "Refused: a.")],
        )
        from bio_neb.failure import Invariant

        with self.assertRaisesRegex(Invariant, "correction did not follow"):
            self.compose(parts=(_retry("Other."),), answer=answer)


class ReplyRetryTests(unittest.TestCase):
    """The one retry part of a reply none of whose calls run (#444 ST1)."""

    def test_a_correction_comes_first_and_the_note_after_it(self):
        self.assertEqual(
            transcript.reply_retry("Outputs refused.", "Copied calls dropped."),
            "Outputs refused.\n\nCopied calls dropped.",
        )
        self.assertEqual(
            transcript.reply_retry(None, "Calls refused."), "Calls refused."
        )
        self.assertEqual(
            transcript.reply_retry("Send the call.", None), "Send the call."
        )
        with self.assertRaisesRegex(ValueError, "correction or a note"):
            transcript.reply_retry(None, None)


class SentHistoryTests(unittest.TestCase):
    """Two user messages never follow each other in what is sent (#444)."""

    def test_a_stalled_request_is_followed_by_an_empty_reply_then_its_note(self):
        asked = _request(ToolReturnPart("a", "ok", tool_call_id="c1"))
        noted = _request(RetryPromptPart("Your reply did not arrive."))
        between = transcript.reply_between(asked, noted)
        self.assertEqual(
            (between["kind"], _shape(between["parts"])), ("response", [("text", "")])
        )
        # The empty reply is dated when its request was, so reopen rebuilds it.
        self.assertEqual(between["timestamp"], asked["timestamp"])

    def test_a_resumed_calls_results_stay_in_the_turn_of_its_request(self):
        pending = _request(ToolReturnPart("a", "ok", tool_call_id="c1"))
        resumed = _request(ToolReturnPart("b", "ok", tool_call_id="c2"))
        self.assertIsNone(transcript.reply_between(pending, resumed))

    def test_a_retry_after_a_reply_needs_nothing_between(self):
        reply = messages_record([ModelResponse(parts=[TextPart("")])])[0]
        retry = _request(RetryPromptPart("Send the call."))
        self.assertIsNone(transcript.reply_between(reply, retry))


_CELEGANS_A1 = (
    Path(__file__).parent
    / "fixtures"
    / "issue387"
    / "celegans-a1-7df8a326-replies.json"
)


class CelegansA1ReplayTests(unittest.TestCase):
    """#387 celegans-a1's member 7df8a326, replayed through the real loop (#444).

    Its eleven recorded replies (nine call stubs, two empty replies) are sent in
    order on Nemotron Ultra through the real adapter; each request the recording
    resent after a provider failure fails once here first, so the reply comes
    with a resend fault. Every request body is read as sent.
    """

    def setUp(self):
        # The availability backoff waits zero seconds.
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    def replay(self):
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_ULTRA_ENDPOINT_POLICY,
        )
        from bio_neb.agentic_journey.planning_assignment import planning_assignment
        from bio_neb.application.queries import ReviewJourneyQuery, review_journey
        from bio_neb.journey import Journey
        from bio_neb.recording import read_evidence
        from tests.application_fakes import _registered
        from tests.input_correction_fakes import prepare_input_correction
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        recorded = json.loads(_CELEGANS_A1.read_text())["replies"]
        replies = [read_messages_record([item["message"]])[0] for item in recorded]
        failed = set()

        def respond(messages, info):
            index = endpoint.answered
            if recorded[index]["physical_contact_ordinal"] > 1 and index not in failed:
                failed.add(index)
                raise ModelHTTPError(504, "scripted-model", "gateway timeout")
            endpoint.answered += 1
            return replies[index]

        endpoint = ScriptedModelEndpoint(respond)
        endpoint.answered = 0
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, _ = _registered(
            Path(directory.name),
            model=NEMOTRON_ULTRA_ENDPOINT_POLICY.settings_selection.model,
            # The replay is about how each request is composed, not when the
            # Attempt stops: the failure rule must not end it early.
            agent_invocation={
                "model_requests": len(replies),
                "failed_turns": len(replies),
            },
        )
        journey = Journey.open(Path(receipt.journey_file))
        prepare_input_correction(journey)
        planning_assignment(model_factory=endpoint).run(
            journey, environ={"LOCAL_MODEL_KEY": "secret"}
        )
        # Reopen asks the composer the same question of every request.
        review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        self.journey_file = receipt.journey_file
        return recorded, endpoint, read_evidence(journey)

    def test_no_stub_is_replayed_no_note_joins_a_correction_no_user_follows_a_user(
        self,
    ):
        _recorded, endpoint, events = self.replay()
        bodies = endpoint.bodies
        self.assertGreater(endpoint.answered, 3)
        for number, body in enumerate(bodies, start=1):
            with self.subTest(body=number):
                roles = [message["role"] for message in body["messages"]]
                after_first_reply = (
                    roles[roles.index("assistant") :] if ("assistant" in roles) else []
                )
                # No user message follows another once the model has replied.
                self.assertNotIn(
                    ("user", "user"),
                    list(zip(after_first_reply, after_first_reply[1:], strict=False)),
                )
                for message in body["messages"]:
                    if message["role"] == "assistant":
                        # A stub's fragment is never sent back; an empty reply is
                        # sent as empty content, like a reasoning-only one.
                        self.assertEqual(message.get("content"), "")
                    if message["role"] == "user" and "no tool call" in str(
                        message.get("content")
                    ):
                        # The resend note never joins a no-call correction.
                        self.assertNotIn("No reply arrived", message["content"])
        resent = [
            e
            for e in events
            if e["event_type"] == "AgentTurnRouted"
            and any(f["key"] == "model_resend" for f in e["payload"]["faults"])
        ]
        self.assertTrue(resent)
        self.assertTrue(all(e["payload"]["feedback"] is None for e in resent))
        corrections = [
            e["payload"]["feedback"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
        ]
        self.assertTrue(any("fragment" in text for text in corrections))
        self.assertTrue(any("came back empty" in text for text in corrections))

    def test_reopen_refuses_each_request_the_composer_would_not_write(self):
        # The live hooks wrote these requests; reopen asks the composer the same
        # question of each, so a request it would not write is refused.
        import copy

        from bio_neb.application.queries import project_journey_events

        _recorded, _endpoint, events = self.replay()
        project_journey_events(events, journey_file=self.journey_file)

        def later(hostile):
            # The third request: after a stub whose correction is the retry.
            requests = [e for e in hostile if e["event_type"] == "ModelRequestRecorded"]
            return requests[2]["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ]

        def stub_replayed(hostile):
            later(hostile)[-2]["parts"][0]["content"] = "The"

        def reply_dropped(hostile):
            del later(hostile)[-2]

        def note_joined(hostile):
            retry = later(hostile)[-1]["parts"][-1]
            retry["content"] += "\n\nNo reply arrived: the provider timed out."

        def notice_beside_correction(hostile):
            later(hostile)[-1]["parts"].insert(
                0,
                {
                    "part_kind": "user-prompt",
                    "content": _NOTICE,
                    "timestamp": later(hostile)[-1]["parts"][-1]["timestamp"],
                },
            )

        for change, reason in (
            (stub_replayed, "reply it keeps"),
            (reply_dropped, "retry right after a request"),
            (note_joined, "exact correction"),
            (notice_beside_correction, "order the composer gives"),
        ):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(events)
                change(hostile)
                with self.assertRaisesRegex(ValueError, reason):
                    project_journey_events(hostile, journey_file=self.journey_file)


if __name__ == "__main__":
    unittest.main()


class UnusableReplyCorrectionTests(unittest.TestCase):
    """Each unusable reply's correction says what came back (owner, #444 Q4, Q5).

    All three count under the reasoning-only key; the model reads which one
    happened, so it does not send the same reply again.
    """

    OFFERED = frozenset({"locate_variant", "search_journey"})

    def test_each_kind_has_its_own_sentence_in_every_wording(self):
        from bio_neb.agentic_journey.model_settings import model_registry
        from bio_neb.failure import CorrectionWording, ReasoningOnly, render_correction

        expected = {
            "empty_reply": "came back empty",
            "call_stub": "fragment",
            "reasoning_only": "reasoning",
        }
        for name, wording in (
            ("default", CorrectionWording()),
            *(
                (model, policy.correction_wording)
                for model, policy in model_registry().entries.items()
            ),
        ):
            texts = {
                kind: render_correction(
                    ReasoningOnly(self.OFFERED, reply=kind), wording
                )
                for kind in expected
            }
            for kind, words in expected.items():
                with self.subTest(wording=name, kind=kind):
                    self.assertIn(words, texts[kind])
                    self.assertIn("`locate_variant`", texts[kind])
            self.assertEqual(len(set(texts.values())), 3)
