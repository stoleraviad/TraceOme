"""The one Injection slot in the shared Agent assembly (ADR 0048).

A test registers its own pattern: the product registry ships none, so the slot is
observed through the real planning assignment with a scripted model.
"""

import copy
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)

from bio_neb.agentic_journey import injection, model_settings
from bio_neb.agentic_journey.effort import (
    ExtendAgentEffortResponse,
    project_effort_review,
    respond_effort_review,
)
from bio_neb.agentic_journey.model_settings import model_registry, use_model_registry
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.commands import (
    RegistrationError,
    RespondModelAvailabilityReviewCommand,
    respond_model_availability_review,
)
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _planning_ready, _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint

NOTICE = "Test notice: this is the second request."


class _AtRequest(injection.PatternTuning):
    request_number: int = 2


def _notice_at_request(signals, tuning):
    return NOTICE if signals.request_number == tuning.request_number else None


TEST_PATTERN = injection.InjectionPattern(
    name="test_notice", tuning=_AtRequest, message=_notice_at_request
)


class _AfterRequests(injection.PatternTuning):
    after_requests: int = 16


def _progress_notice(signals, tuning):
    if signals.request_number <= tuning.after_requests:
        return None
    return f"Notice: this is request {signals.request_number} of this Attempt."


CAP_PATTERN = injection.InjectionPattern(
    name="test_cap", tuning=_AfterRequests, message=_progress_notice
)


def _with_test_pattern():
    return patch.dict(
        injection.PATTERNS,
        {pattern.name: pattern for pattern in (TEST_PATTERN, CAP_PATTERN)},
    )


def _sent_injections(messages):
    """Each framework request's injected messages, in the order the model read them."""
    return [
        [
            part.content
            for part in message.parts
            if isinstance(part, UserPromptPart)
            and injection.injected_pattern(part.content)
        ]
        for message in messages
        if hasattr(message, "instructions")
    ]


def _events(journey_file):
    return read_evidence(Journey.open(Path(journey_file)))


def _requests(events):
    return [e["payload"] for e in events if e["event_type"] == "ModelRequestRecorded"]


def _user_texts(message):
    return [
        part["content"]
        for part in message["parts"]
        if part["part_kind"] == "user-prompt" and isinstance(part["content"], str)
    ]


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?")
_FILE_IDENTITY = re.compile(r"(inode|_ns|device)([^0-9a-z]{1,8})\d+")
_DIGEST = re.compile(r"[0-9a-f]{16,64}")


def _masked(bodies, root):
    """A Run's wire bodies with its own values masked: its root, ids in first-seen
    order, times, file identities and the digests derived from them."""
    text = json.dumps(bodies, sort_keys=True).replace(str(root), "<root>")
    text = _FILE_IDENTITY.sub(
        lambda m: m.group(1) + m.group(2) + "<n>", _TIME.sub("<t>", text)
    )
    ids = {}
    text = _UUID.sub(lambda m: ids.setdefault(m.group(0), f"<id{len(ids)}>"), text)
    return _DIGEST.sub("<digest>", text)


def _retained(events):
    """The stored bytes a pattern switched off must leave unchanged."""
    registration = next(e for e in events if "runtime_selections" in e["payload"])
    return {
        "attempts": [
            {
                key: e["payload"][key]
                for key in ("effective_invocation_policy", "invocation_policy_id")
            }
            for e in events
            if e["event_type"] == "AttemptStarted"
        ],
        "registration": registration["payload"]["runtime_selections"],
    }


class InjectionSlotTests(unittest.TestCase):
    def test_with_no_pattern_on_qwen_s_records_and_requests_are_unchanged(self):
        # A pattern switched off leaves every stored byte and every sent request as
        # a Run with no injection entry at all, and no Qwen record names the slot.
        # The registered roster also holds Nemotron's entry, which switches its
        # patterns on; Qwen's own entry and Attempts name none.
        switched_off = {"injection": {"test_notice": {"on": False}}}
        runs = {}
        for name, invocation in (("plain", None), ("off", switched_off)):
            with _with_test_pattern(), tempfile.TemporaryDirectory() as directory:
                receipt, composition = _planning_ready(
                    Path(directory), agent_invocation=invocation
                )
                events = _events(receipt.journey_file)
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
                bodies = _masked(composition.model_endpoint.bodies, directory)
            runs[name] = (events, bodies)
        plain, off = (runs[name][0] for name in ("plain", "off"))
        self.assertEqual(_retained(off)["attempts"], _retained(plain)["attempts"])
        registered = _retained(off)["registration"]
        self.assertEqual(
            registered["agent_invocation"].pop("injection"),
            {"test_notice": {"on": False}},
        )
        self.assertEqual(registered, _retained(plain)["registration"])
        # Every body the model received: messages, Tools and settings.
        self.assertGreater(len(runs["plain"][1]), 1000)
        self.assertEqual(runs["off"][1], runs["plain"][1])
        self.assertNotIn("[injection:", runs["off"][1])
        qwen = _retained(plain)["registration"]["model_roster"]["qwen"]
        self.assertNotIn("injection", set(_keys(_retained(plain)["attempts"])))
        self.assertNotIn("injection_patterns", set(_keys(qwen)))
        for request in _requests(plain):
            messages = request["compatible_adapter_evidence"]["request"]["messages"]
            self.assertFalse(
                any(
                    text.startswith("[injection:")
                    for message in messages
                    for text in _user_texts(message)
                )
            )

    def test_nemotron_with_every_pattern_switched_off_sends_what_it_sent_with_none_on(
        self,
    ):
        # Nemotron switches its patterns on. Switched off in config, each
        # request is the one a Nemotron with no pattern on sends, byte for byte.
        nemotron = model_settings.NEMOTRON_ULTRA_ENDPOINT_POLICY
        none_on = replace(
            nemotron,
            settings_selection=nemotron.settings_selection.model_copy(
                update={"injection_patterns": ()}
            ),
        )
        all_off = {
            "injection": {
                name: {"on": False}
                for name in nemotron.settings_selection.injection_patterns
            }
        }
        bodies = {}
        for name, entries, invocation in (
            ("none_on", (none_on,), None),
            ("all_off", (), all_off),
        ):
            with (
                use_model_registry(model_registry().replacing(*entries)),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, composition = _planning_ready(
                    Path(directory),
                    model=nemotron.settings_selection.model,
                    agent_invocation=invocation,
                )
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
                bodies[name] = _masked(composition.model_endpoint.bodies, directory)
        self.assertGreater(len(bodies["none_on"]), 1000)
        self.assertEqual(bodies["all_off"], bodies["none_on"])
        self.assertNotIn("[injection:", bodies["all_off"])

    def test_a_pattern_switched_on_in_config_injects_one_tagged_recorded_message(self):
        tagged = "[injection:test_notice]\n" + NOTICE
        invocation = {"injection": {"test_notice": {"on": True}}}
        with _with_test_pattern(), tempfile.TemporaryDirectory() as directory:
            receipt, composition = _planning_ready(
                Path(directory), agent_invocation=invocation
            )
            events = _events(receipt.journey_file)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        requests = _requests(events)
        first_attempt = requests[0]["attempt_id"]
        attempt_requests = [r for r in requests if r["attempt_id"] == first_attempt]
        self.assertGreater(len(attempt_requests), 2)
        for request in attempt_requests:
            messages = request["compatible_adapter_evidence"]["request"]["messages"]
            injected = [
                (index, text)
                for index, message in enumerate(messages)
                for text in _user_texts(message)
                if text.startswith("[injection:")
            ]
            with self.subTest(request=request["request_number"]):
                if request["request_number"] < 2:
                    self.assertEqual(injected, [])
                else:
                    # Sent once, in request 2's own message, and kept in history.
                    self.assertEqual(injected, [(2, tagged)])
        second = attempt_requests[1]["compatible_adapter_evidence"]["request"]
        wire = composition.model_endpoint.bodies[1]["messages"]
        self.assertIn({"role": "user", "content": tagged}, wire)
        self.assertEqual(
            [m for m in wire if m.get("role") == "user"][-1]["content"], tagged
        )
        self.assertEqual(_user_texts(second["messages"][-1])[-1], tagged)
        started = [
            e["payload"]["effective_invocation_policy"]
            for e in events
            if e["event_type"] == "AttemptStarted"
        ]
        self.assertEqual(
            started[0]["injection"],
            {
                "patterns": {"test_notice": {"on": True, "request_number": 2}},
                "stage_patterns": {},
            },
        )

    def test_a_cap_sends_a_notice_and_the_request_budget_still_ends_the_attempt(self):
        # Caps are notices (R3-3): the model keeps working after each one, and the
        # shared budget, not the pattern, stops the Attempt at its effort Review.
        invocation = {
            "model_requests": 3,
            "injection": {"test_cap": {"on": True, "after_requests": 1}},
        }
        notice = "[injection:test_cap]\nNotice: this is request {} of this Attempt."
        seen = []

        def respond(messages, info):
            seen.append(list(messages))
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {"text": f"registered input {len(seen)}"},
                        tool_call_id=f"search-{len(seen)}",
                    )
                ]
            )

        with _with_test_pattern(), tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory), agent_invocation=invocation)
            journey = Journey.open(Path(receipt.journey_file))
            prepare_input_correction(journey)
            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            stopped_after = len(seen)
            events = read_evidence(journey)
            # The held request's injected parts are bound to the Attempt's policy:
            # a pattern it does not switch on, or a second notice, is refused.
            for name, forged in (
                ("unswitched", "[injection:test_notice]\n" + NOTICE),
                ("repeated", notice.format(4)),
            ):
                hostile = copy.deepcopy(events)
                held = next(
                    e["payload"]["messages"]
                    for e in hostile
                    if e["event_type"] == "AgentEffortReviewRequested"
                )
                held[-1]["parts"].append(
                    {"part_kind": "user-prompt", "content": forged}
                )
                with (
                    self.subTest(forged=name),
                    self.assertRaisesRegex(
                        ValueError,
                        "effort transcript changed its pending Tool continuation",
                    ),
                ):
                    project_effort_review(hostile)
            reviewed = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            accepted = respond_effort_review(
                journey,
                result.agent_effort_review.pending_request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=1),
            )
            extended = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                effort_review_response_event_id=accepted.response_event_id,
            )
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        self.assertEqual((result.status, stopped_after), ("effort_review_pending", 3))
        self.assertEqual((extended.status, len(seen)), ("effort_review_pending", 4))
        self.assertEqual(reviewed.status, "effort_review_pending")
        # The extension sends the held request as the Review recorded it: its
        # notice once, not a second one from the slot.
        self.assertEqual(_sent_injections(seen[3])[-1], [notice.format(4)])
        self.assertEqual(
            _sent_injections(seen[2]), [[], [notice.format(2)], [notice.format(3)]]
        )
        self.assertFalse(any(e["event_type"] == "AttemptSucceeded" for e in events))
        # The unsent request the effort Review holds carries its own notice.
        pending = result.agent_effort_review.pending_request.messages[-1]["parts"]
        self.assertEqual(
            [p["content"] for p in pending if p["part_kind"] == "user-prompt"],
            [notice.format(4)],
        )

    def test_a_stage_scope_switches_a_pattern_for_that_stage_s_requests_only(self):
        invocation = {
            "scopes": {
                "stage": {
                    "reference": {
                        "injection": {"test_cap": {"on": True, "after_requests": 0}}
                    }
                }
            }
        }
        with _with_test_pattern(), tempfile.TemporaryDirectory() as directory:
            receipt, _ = _planning_ready(Path(directory), agent_invocation=invocation)
            events = _events(receipt.journey_file)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        spoken = []
        for request in _requests(events):
            pending = request["compatible_adapter_evidence"]["request"]["messages"][-1]
            injected = [
                text for text in _user_texts(pending) if text.startswith("[injection:")
            ]
            spoken.append((request["stage"], len(injected)))
        self.assertIn(("reference", 1), spoken)
        self.assertGreater(len({stage for stage, _ in spoken}), 1)
        self.assertEqual(
            {(stage, count) for stage, count in spoken},
            {(stage, int(stage == "reference")) for stage, _ in spoken},
        )
        (policy, *_) = [
            e["payload"]["effective_invocation_policy"]["injection"]
            for e in events
            if e["event_type"] == "AttemptStarted"
        ]
        self.assertEqual(
            policy,
            {
                "patterns": {},
                "stage_patterns": {
                    "reference": {"test_cap": {"on": True, "after_requests": 0}}
                },
            },
        )

    def test_a_provider_retry_resends_the_injected_request_as_recorded(self):
        # The pending request already holds its injected message: the retry sends
        # those bytes, and the slot adds no second message (ADR 0048, reopen).
        invocation = {"injection": {"test_notice": {"on": True, "request_number": 1}}}
        received = []

        async def unavailable(messages, info):
            # The scripted endpoint rebuilds the wire's messages with new times.
            received.append(
                [
                    [(part.part_kind, part.content) for part in message.parts]
                    for message in messages
                ]
            )
            # A 5xx over its Retry-After cap asks at once; a 4xx with words is
            # shown to the model first.
            raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})

        def attempt():
            return planning_assignment(model_factory=ScriptedModelEndpoint(unavailable))

        with _with_test_pattern(), tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(
                Path(directory).resolve(), agent_invocation=invocation
            )
            journey = Journey.open(Path(receipt.journey_file))
            prepare_input_correction(journey)
            pending = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            request = pending.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    response={"action": "retry"},
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=attempt(),
            )
            reviewed = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            recorded = [
                r["compatible_adapter_evidence"]["request"]["messages"]
                for r in _requests(_events(receipt.journey_file))
            ]
        tagged = "[injection:test_notice]\n" + NOTICE
        self.assertEqual(reviewed.status, "model_availability_review_pending")
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0], recorded[1])
        self.assertEqual(
            [t for t in _user_texts(recorded[1][-1]) if t.startswith("[injection:")],
            [tagged],
        )
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0], received[1])
        self.assertEqual(
            [content for _, content in received[1][-1] if content == tagged], [tagged]
        )

    def test_the_model_endpoint_policy_switches_its_patterns_on_and_a_role_scope_one_off(
        self,
    ):
        # The selected model's endpoint policy switches patterns on; the Run retains
        # that set with the model's baseline, and config tunes it per scope.
        qwen = model_settings.QWEN_ENDPOINT_POLICY
        switched = replace(
            qwen,
            settings_selection=qwen.settings_selection.model_copy(
                update={"injection_patterns": ("test_notice",)}
            ),
        )
        role_off = {
            "scopes": {
                "role": {
                    "planning_agent": {"injection": {"test_notice": {"on": False}}}
                }
            }
        }
        runs = {}
        with (
            _with_test_pattern(),
            use_model_registry(model_registry().replacing(switched)),
        ):
            for name, invocation in (("switched", None), ("role_off", role_off)):
                with tempfile.TemporaryDirectory() as directory:
                    receipt, _ = _planning_ready(
                        Path(directory), agent_invocation=invocation
                    )
                    runs[name] = _events(receipt.journey_file)
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    )
        tagged = "[injection:test_notice]\n" + NOTICE
        for name, expected in (("switched", [tagged]), ("role_off", [])):
            events = runs[name]
            registered = _retained(events)["registration"]["model_roster"]["qwen"]
            with self.subTest(run=name):
                self.assertEqual(
                    registered["settings_selection"]["injection_patterns"],
                    ["test_notice"],
                )
                second = [r for r in _requests(events) if r["request_number"] == 2][0][
                    "compatible_adapter_evidence"
                ]["request"]["messages"][-1]
                self.assertEqual(
                    [t for t in _user_texts(second) if t.startswith("[injection:")],
                    expected,
                )
        started = [
            e["payload"]["effective_invocation_policy"]
            for e in runs["role_off"]
            if e["event_type"] == "AttemptStarted"
        ]
        self.assertNotIn("injection", started[0])

    def test_configuration_refuses_an_unknown_pattern_a_bad_tuning_and_a_tool_entry(
        self,
    ):
        cases = {
            "unknown": {"injection": {"no_such_pattern": {"on": True}}},
            "tuning": {"injection": {"test_cap": {"after_requests": "many"}}},
            "tool": {
                "scopes": {
                    "tool": {
                        "search_journey": {"injection": {"test_cap": {"on": True}}}
                    }
                }
            },
        }
        refused = {}
        for name, invocation in cases.items():
            with (
                self.subTest(case=name),
                _with_test_pattern(),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(RegistrationError) as raised,
            ):
                _registered(Path(directory), agent_invocation=invocation)
            refused[name] = (raised.exception.code, raised.exception.fields)
        self.assertEqual(
            refused,
            {
                "unknown": (
                    "runtime_selections_invalid",
                    ("agent_invocation.injection.no_such_pattern.[key]",),
                ),
                "tuning": (
                    "runtime_selections_invalid",
                    ("agent_invocation.injection.test_cap.after_requests",),
                ),
                "tool": (
                    "runtime_selections_invalid",
                    ("agent_invocation.scopes.tool.search_journey.injection",),
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
