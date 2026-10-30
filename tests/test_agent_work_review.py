"""The Agent work review is an exact scoped request with exact answers.

A real planning Attempt stops on its first refused action (a limit of one failed
turn in a row); the controlled model is the only substitute. Hostile reopen checks
edit copies of that retained evidence, never the Journey.
"""

import copy
import tempfile
import unittest
import uuid
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart

from bio_neb.agentic_journey.action_admission import (
    ActionAdmissionSession,
    ProposedAgentAction,
)
from bio_neb.agentic_journey.agent_work_review import (
    AgentFailedTurns,
    AgentWorkReviewResponseRefused,
    AgentWorkStop,
    ContinueAgentWorkResponse,
    RejectAgentWorkResponse,
    granted_failed_turns,
    project_agent_work_review,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.attempt_journey import (
    AgentOutputValidated,
    AttemptFailureSource,
)
from bio_neb.agentic_journey.continuation_policy import (
    ContinuationPolicy,
    validate_continuation,
)
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.failure import CorrectionWording
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.agent_attempt_fakes import FakeActivity, FakeJourney
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint, corrections
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    _AdaptiveInvestigationModel,
    _IndependentCorrectionsModel,
    _MalformedOutputModel,
    run_attempt,
)

_ENVIRON = {"LOCAL_MODEL_KEY": "secret"}


class _Model:
    """Refuses until ``valid`` is set, then inspects the registered input set."""

    def __init__(self, input_set_id):
        self.input_set_id = input_set_id
        self.valid = False
        self.calls = 0
        # The admission feedback each request carried, newest last.
        self.feedback = []

    def respond(self, messages, info):
        del info
        self.feedback.extend(corrections(messages[-1]))
        self.calls += 1
        arguments = {"input_set_id": self.input_set_id if self.valid else "wrong"}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_input_metadata",
                    arguments,
                    tool_call_id=f"call-{self.calls}",
                )
            ]
        )

    def assignment(self):
        return planning_assignment(model_factory=ScriptedModelEndpoint(self.respond))


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def _semantic(events):
    """Journey facts without the processing-Pass journal markers."""
    return [e for e in events if e["event_type"] not in {"PassStarted", "PassClosed"}]


class AgentWorkReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        receipt, _ = _registered(
            Path(directory.name).resolve(),
            agent_invocation={"failed_turns": 1},
        )
        self.journey = Journey.open(Path(receipt.journey_file))
        prepare_input_correction(self.journey)
        input_set_id = review_journey(
            ReviewJourneyQuery(journey_file=receipt.journey_file)
        ).registered_input_set.input_set_id
        self.model = _Model(input_set_id)
        stopped = self.model.assignment().run(self.journey, environ=_ENVIRON)
        self.assertEqual(stopped.status, "agent_work_review_pending")
        self.request = stopped.agent_work_review.pending_request

    def test_the_review_shows_the_feedback_of_its_exhausting_refusal(self):
        (refusal,) = self.request.faults
        # The stop's source is the stopping reply's refused turn; the fault is
        # the refused call's validation that turn names.
        (source,) = [
            e
            for e in read_evidence(self.journey)
            if e["event_id"] == self.request.source_event_id
        ]
        self.assertEqual(
            (source["event_type"], source["payload"]["validation_event_ids"]),
            ("AgentActionTurnDispositionRecorded", [refusal.event_id]),
        )
        self.assertEqual(
            (refusal.correction_key.split(" @ ")[0], refusal.correction_count),
            ("inspect_input_metadata: input_set_id:bound_value_mismatch", 1),
        )
        # The first failed turn was the limit, so the model never saw its
        # correction; the reviewer does.
        self.assertEqual(self.model.feedback, [])
        # A server-bound value's refusal keeps the server-bound form.
        self.assertEqual(
            refusal.feedback,
            "Refused before execution (input_set_id:bound_value_mismatch).\n"
            "Tool `inspect_input_metadata` was not run. Fix each item, then call it "
            "again.\n"
            '- Field `input_set_id`: you sent "wrong".\n'
            "  Allowed form: omit it; the server supplies it.\n"
            "Allowed now: one of `inspect_input_metadata`, `search_journey`.\n"
            "Waiting on their step, not callable now: `request_planning_hitl`.\n"
            # Its place in the count.
            "Count: failed turn 1 of 1 in a row. At 1 the Attempt stops and the "
            "researcher reviews it.",
        )
        reopened = project_agent_work_review(read_evidence(self.journey))
        self.assertEqual(reopened.pending_request.faults, self.request.faults)

    def test_identical_replay_is_a_noop_and_other_answers_are_refused(self):
        continue_ = ContinueAgentWorkResponse(action="continue")
        receipt = respond_agent_work_review(
            self.journey, self.request.request_event_id, continue_
        )
        answered = read_evidence(self.journey)
        self.assertEqual(
            respond_agent_work_review(
                self.journey, self.request.request_event_id, continue_
            ),
            receipt,
        )
        refusals = (
            (
                self.request.request_event_id,
                RejectAgentWorkResponse(
                    action="reject",
                    reason="Different answer.",
                    asserted_by="researcher",
                ),
                "agent_work_review_response_conflict",
            ),
            (str(uuid.uuid4()), continue_, "agent_work_review_response_stale"),
        )
        for request_event_id, response, code in refusals:
            with (
                self.subTest(code=code),
                self.assertRaises(AgentWorkReviewResponseRefused) as refused,
            ):
                respond_agent_work_review(self.journey, request_event_id, response)
            self.assertEqual(refused.exception.code, code)
        self.assertEqual(read_evidence(self.journey), answered)

    def test_a_continue_may_grant_any_role_its_failed_turns(self):
        # A continue may grant the next Attempt's number of failed turns in a row
        # to every role, a Tool-driven one included; a continue naming none
        # records no key, so earlier answers read back exactly.
        self.assertEqual(self.request.agent_role, "planning_agent")
        self.assertEqual(
            ContinueAgentWorkResponse(action="continue").model_dump(mode="json"),
            {"action": "continue", "instruction": None},
        )
        for refused in (0, 21):
            with self.subTest(failed_turns=refused), self.assertRaises(ValidationError):
                ContinueAgentWorkResponse(action="continue", failed_turns=refused)
        answer = respond_agent_work_review(
            self.journey,
            self.request.request_event_id,
            ContinueAgentWorkResponse(action="continue", failed_turns=4),
        )
        self.assertEqual(
            granted_failed_turns(self.journey, answer.response_event_id), 4
        )
        self.model.valid = True
        resumed = self.model.assignment().run(
            self.journey,
            environ=_ENVIRON,
            work_review_response_event_id=answer.response_event_id,
        )
        self.assertEqual(resumed.status, "inspection_succeeded")
        continued = _of(read_evidence(self.journey), "AttemptStarted")[-1]["payload"]
        self.assertEqual(continued["invocation_overrides"], {"failed_turns": 4})
        self.assertEqual(continued["continuation_policy"], {"failed_turns": 4})

    def test_reopen_refuses_a_changed_request_or_answer(self):
        events = read_evidence(self.journey)
        (request,) = _of(events, "AgentWorkReviewRequested")
        (response,) = _of(events, "ModelResponseRecorded")
        hostile = {}

        changed = copy.deepcopy(events)
        _of(changed, "AgentWorkReviewRequested")[0]["payload"]["fault_event_ids"] = []
        hostile["fault evidence"] = (changed, "changed its fault evidence")

        changed = copy.deepcopy(events)
        _of(changed, "AgentWorkReviewRequested")[0]["payload"]["source_event_id"] = (
            response["event_id"]
        )
        hostile["stopping evidence"] = (changed, "exact stopping reply")

        changed = copy.deepcopy(events)
        repeated = copy.deepcopy(request)
        repeated["event_id"] = str(uuid.uuid4())
        changed.insert(changed.index(request) + 1, repeated)
        hostile["second pending request"] = (changed, "another is pending")

        answer = {
            "event_id": str(uuid.uuid4()),
            "event_type": "AgentWorkReviewResponded",
            "parent_ids": [response["event_id"]],
            "payload": {
                "request_event_id": request["event_id"],
                "response": {"action": "continue", "instruction": None},
                "authority": "researcher_direction",
                "authorizes_execution": False,
            },
        }
        hostile["answer off its request"] = (
            [*copy.deepcopy(events), answer],
            "changed its exact request",
        )
        for label, (timeline, message) in hostile.items():
            with (
                self.subTest(hostile=label),
                self.assertRaisesRegex(ValueError, message),
            ):
                project_agent_work_review(timeline)

    def test_continue_reopens_the_role_once_and_a_fresh_attempt_is_held(self):
        held = copy.deepcopy(read_evidence(self.journey))
        fresh = copy.deepcopy(_of(held, "AttemptStarted")[-1])
        fresh["event_id"] = str(uuid.uuid4())
        fresh["payload"]["attempt_id"] = str(uuid.uuid4())
        with self.assertRaisesRegex(ValueError, "exact continue answer"):
            project_agent_work_review([*held, fresh])
        with self.assertRaisesRegex(ValueError, "exact continue answer"):
            self.model.assignment().run(self.journey, environ=_ENVIRON)

        answer = respond_agent_work_review(
            self.journey,
            self.request.request_event_id,
            ContinueAgentWorkResponse(action="continue"),
        )
        self.model.valid = True
        resumed = self.model.assignment().run(
            self.journey,
            environ=_ENVIRON,
            work_review_response_event_id=answer.response_event_id,
        )
        self.assertEqual(resumed.status, "inspection_succeeded")
        events = read_evidence(self.journey)
        continued = _of(events, "AttemptStarted")[-1]
        self.assertEqual(
            continued["payload"]["continuation_work_review_event_id"],
            answer.response_event_id,
        )
        self.assertEqual(
            project_agent_work_review(events).continued_response_event_ids,
            (answer.response_event_id,),
        )
        second = copy.deepcopy(continued)
        second["event_id"] = str(uuid.uuid4())
        second["payload"]["attempt_id"] = str(uuid.uuid4())
        with self.assertRaisesRegex(ValueError, "exact unused authority"):
            project_agent_work_review([*events, second])
        with self.assertRaisesRegex(ValueError, "exact unused authority"):
            self.model.assignment().run(
                self.journey,
                environ=_ENVIRON,
                work_review_response_event_id=answer.response_event_id,
            )
        self.assertEqual(_semantic(read_evidence(self.journey)), _semantic(events))
        self.assertEqual(self.model.calls, 2)


class SentCorrectionTests(unittest.TestCase):
    """With two failed turns in a row allowed, the Review shows what the model
    was told."""

    def test_each_refusal_shows_the_feedback_its_turn_presented(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(
                Path(directory).resolve(),
                agent_invocation={"failed_turns": 2},
            )
            journey = Journey.open(Path(receipt.journey_file))
            prepare_input_correction(journey)
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            model = _Model(input_set_id)
            stopped = model.assignment().run(journey, environ=_ENVIRON)
            self.assertEqual(stopped.status, "agent_work_review_pending")
            first, last = stopped.agent_work_review.pending_request.faults
            # The first refusal's feedback is the retry prompt the model received.
            self.assertEqual(model.feedback, [first.feedback])
            self.assertEqual(last.feedback, _at_the_limit(first.feedback, 2))
            self.assertIn("`inspect_input_metadata` was not run.", first.feedback)


def _firm(kind):
    """The firm line a correction ends in at the last failed turn before the
    limit (#373, #447 B)."""
    del kind  # every fault is one more failed turn
    return "\nLast correction: one more failed turn stops the Attempt for review."


def _count(count, limit):
    """The count line a correction ends in (#478 G1.4)."""
    return (
        f"Count: failed turn {count} of {limit} in a row. At {limit} the Attempt "
        "stops and the researcher reviews it."
    )


def _at_the_limit(first, limit):
    """The exhausting refusal's feedback from the one before it: the same text at
    the next count, past its firm line."""
    before = _count(limit - 1, limit)
    if before not in first:
        raise AssertionError(f"no count line {before!r} in {first!r}")
    return first.replace(_firm("output"), "").replace(before, _count(limit, limit))


def _repairs_received(messages):
    """The repair text each retry prompt of the newest request presents."""
    return [
        part.model_response()
        for part in messages[-1].parts
        if isinstance(part, RetryPromptPart)
    ]


class _ReceivingMalformedOutputModel(_MalformedOutputModel):
    def __init__(self):
        super().__init__()
        self.received = []

    async def respond(self, messages, info):
        self.received.extend(_repairs_received(messages))
        return await super().respond(messages, info)


class _ReceivingFencedOutputModel(_ReceivingMalformedOutputModel):
    """Wraps the malformed output in Markdown fences, which the framework strips."""

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        (part,) = response.parts
        part.args = "```json\n" + part.args + "\n```"
        return response


class _ReceivingUngroundedOutputModel(_IndependentCorrectionsModel):
    """Re-types a source fact (`one_based_position`) into its first two outputs."""

    def __init__(self):
        super().__init__(wrong_actions=0, wrong_outputs=2)
        self.received = []

    async def respond(self, messages, info):
        self.received.extend(_repairs_received(messages))
        return await super().respond(messages, info)


class _ReceivingUnresolvedOutputModel(_AdaptiveInvestigationModel):
    """Marks its first two outputs unresolved though their cited sources resolved:
    a well-formed draft the domain validator refuses."""

    def __init__(self):
        super().__init__()
        self.received = []
        self.refused = 0

    async def respond(self, messages, info):
        self.received.extend(_repairs_received(messages))
        response = await super().respond(messages, info)
        (part,) = response.parts
        if part.tool_name == "compile_proposal" and self.refused < 2:
            self.refused += 1
            arguments = part.args_as_dict()
            arguments["outcome"] = "unresolved"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "compile_proposal",
                        arguments,
                        tool_call_id=f"unresolved-{self.refused}",
                    )
                ]
            )
        return response


class OutputRepairFeedbackTests(unittest.TestCase):
    """With two failed turns in a row allowed, the Review shows what each refused
    output was told."""

    def _refused_outputs(self, directory, model):
        projection, journey = run_attempt(
            Path(directory).resolve() / "run",
            model,
            SourceResponsesOnly(),
            continuation_policy={"failed_turns": 2},
        )
        self.assertEqual(projection.attempts[-1].stop_reason, "failed_turns")
        faults = project_agent_work_review(
            read_evidence(journey)
        ).pending_request.faults
        self.assertEqual(
            [fault.event_type for fault in faults], ["AgentOutputValidated"] * 2
        )
        return faults

    def test_a_schema_refusal_shows_the_framework_repair_the_model_received(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _ReceivingMalformedOutputModel()
            first, last = self._refused_outputs(directory, model)
        # The first refusal's feedback is the repair the next request carried, at
        # the member's last repair; the exhausting one shows what the model would
        # have received, past its firm line.
        self.assertEqual(model.received, [first.feedback])
        self.assertEqual(last.feedback, _at_the_limit(first.feedback, 2))
        self.assertIn(_count(1, 2) + _firm("output"), first.feedback)
        self.assertIn("Your output has 1 error.", first.feedback)
        # The draft type requires only the cited source and the branch.
        self.assertIn('"refsnp_event_id"', first.feedback)

    def test_a_fenced_schema_refusal_shows_the_repair_of_the_stripped_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _ReceivingFencedOutputModel()
            first, last = self._refused_outputs(directory, model)
        self.assertEqual(model.received, [first.feedback])
        self.assertEqual(last.feedback, _at_the_limit(first.feedback, 2))
        self.assertIn('"refsnp_event_id"', first.feedback)

    def test_an_ungrounded_output_shows_the_validator_repair_the_model_received(self):
        # The copy refusals (model_output_changed_source_facts) are retired: the
        # domain validator's refusal is now of the outcome the cited sources do
        # not support.
        with tempfile.TemporaryDirectory() as directory:
            model = _ReceivingUnresolvedOutputModel()
            first, last = self._refused_outputs(directory, model)
        self.assertEqual(model.received, [first.feedback])
        self.assertEqual(last.feedback, _at_the_limit(first.feedback, 2))
        self.assertEqual(
            last.feedback,
            "Output refused: Investigation output is not source-grounded: "
            'complete_sources_marked_unresolved; outcome: expected="complete"\n'
            + _count(2, 2)
            + "\n\nFix the errors and try again.",
        )

    def test_a_re_typed_source_fact_is_refused_as_a_field_the_output_does_not_take(
        self,
    ):
        # Code fills every copied field from the cited facts, so a draft that
        # re-types one (`one_based_position`) is refused by the draft's schema as
        # extra_forbidden, where it was once a copy refusal.
        with tempfile.TemporaryDirectory() as directory:
            model = _ReceivingUngroundedOutputModel()
            first, last = self._refused_outputs(directory, model)
        self.assertEqual(model.received, [first.feedback])
        # Each output re-types its own position: the first 1, the second 2.
        self.assertEqual(
            last.feedback,
            _at_the_limit(first.feedback, 2).replace('"input": 1', '"input": 2'),
        )
        self.assertEqual(
            last.feedback,
            "Your output has 1 error. In each, `loc` is the field, `input` is what "
            "you sent and `msg` is the allowed form:\n```json\n[\n  {\n"
            '    "type": "extra_forbidden",\n    "loc": [\n'
            '      "one_based_position"\n    ],\n'
            '    "msg": "Extra inputs are not permitted",\n    "input": 2\n  }\n]\n'
            "```\n" + _count(2, 2) + "\n\nFix the errors and try again.",
        )
        self.assertNotIn("model_output_changed_source_facts", first.feedback)

    def test_each_output_repair_is_written_in_the_selected_models_wording(self):
        # The failure module's retry rendering writes the schema repair and the
        # validator's repair; the Review shows exactly what the model received.
        from dataclasses import replace

        from bio_neb.agentic_journey import model_settings

        model = "Qwen/Qwen3.5-397B-A17B"
        selected = replace(
            model_settings.model_endpoint_policy(model),
            correction_wording=CorrectionWording(
                output_schema="SCHEMA[{count}{plural}] {errors}",
                output_refused="OUTPUT REFUSED: {reason}",
            ),
        )
        suffix = "\n\nFix the errors and try again."
        with model_settings.use_model_registry(
            model_settings.model_registry().replacing(selected)
        ):
            with tempfile.TemporaryDirectory() as directory:
                schema = _ReceivingMalformedOutputModel()
                first, last = self._refused_outputs(directory, schema)
            self.assertEqual(schema.received, [first.feedback])
            self.assertEqual(last.feedback, _at_the_limit(first.feedback, 2))
            self.assertTrue(first.feedback.startswith("SCHEMA[1] ["), first.feedback)
            self.assertIn('"refsnp_event_id"', first.feedback)
            self.assertTrue(first.feedback.endswith(suffix))
            with tempfile.TemporaryDirectory() as directory:
                validator = _ReceivingUnresolvedOutputModel()
                first, last = self._refused_outputs(directory, validator)
            self.assertEqual(validator.received, [first.feedback])
            self.assertEqual(
                last.feedback,
                "OUTPUT REFUSED: Investigation output is not source-grounded: "
                'complete_sources_marked_unresolved; outcome: expected="complete"\n'
                + _count(2, 2)
                + suffix,
            )

    def test_only_a_refused_output_records_repair_feedback(self):
        fields = {"attempt_id": "a", "output": {}, "errors": ()}
        # A valid output keeps the older payload shape, with no feedback key.
        self.assertNotIn(
            "feedback",
            AgentOutputValidated(status="valid", **fields).model_dump(mode="json"),
        )
        with self.assertRaisesRegex(ValueError, "only a refused output"):
            AgentOutputValidated(status="valid", feedback="retry", **fields)


class AdmissionStopCauseTests(unittest.TestCase):
    """Each refusal cause is a failed turn; at the limit the next request's check
    raises the one shared stop on the stopping reply's refused turn (#447 B)."""

    def test_every_refusal_cause_stops_on_its_refused_turn(self):
        action = ProposedAgentAction(
            tool_name="probe_tool", tool_call_id="call", arguments={"value": "a"}
        )
        causes = {
            # The first failed turn is the limit.
            "first refusal at the limit": (1, {}, ("probe_invalid",), 0, 0),
            # The same refused call is corrected once; its next refusal stops.
            "same call refused twice": (2, {}, ("probe_invalid",), 0, 1),
            # An admitted repeat is an ordinary failed turn.
            "admitted repeat": (
                2,
                {
                    "action_fingerprint": lambda proposed: proposed.tool_name,
                    "admitted_repeat_error": "probe_repeated",
                },
                (),
                1,
                1,
            ),
        }
        for label, (limit, options, errors, admitted, corrected) in causes.items():
            with self.subTest(cause=label):
                attempt_id = str(uuid.uuid4())
                policy = ContinuationPolicy(failed_turns=limit)
                timeline = []

                def record(event_type, timeline=timeline, attempt_id=attempt_id):
                    timeline.append(
                        {
                            "event_id": str(uuid.uuid4()),
                            "event_type": event_type,
                            "payload": {"attempt_id": attempt_id},
                        }
                    )

                record("AttemptStarted")
                timeline[-1]["payload"].update(
                    continuation_policy=policy.model_dump(mode="json"),
                    continuation_policy_id=policy.policy_id,
                )
                session = ActionAdmissionSession(
                    wording=CorrectionWording(),
                    continuation=policy,
                    events=lambda timeline=timeline: timeline,
                    attempt_id=attempt_id,
                    parent_activity=FakeActivity(FakeJourney(timeline)),
                    failure_source=AttemptFailureSource(),
                    eligible_tool_names=frozenset({"probe_tool"}),
                    action_validator=lambda _action, errors=errors: errors,
                    **options,
                )
                for _ in range(admitted):
                    record("ModelRequestRecorded")
                    session.admit_turn(
                        str(uuid.uuid4()), (action,), frozenset({"probe_tool"})
                    )
                    record("ObservedToolExecutionSucceeded")
                for _ in range(corrected + 1):
                    record("ModelRequestRecorded")
                    # Admission records each refusal and its correction; it
                    # never stops the Attempt itself.
                    with self.assertRaises(ModelRetry):
                        session.admit_turn(
                            str(uuid.uuid4()), (action,), frozenset({"probe_tool"})
                        )
                with self.assertRaises(AgentFailedTurns) as stopped:
                    validate_continuation(
                        timeline,
                        attempt_id=attempt_id,
                        policy=policy,
                        next_request=True,
                    )
                refused = [
                    e
                    for e in timeline
                    if e["event_type"] == "ToolCallValidated"
                    and e["payload"]["status"] == "invalid"
                ]
                turns = [
                    e
                    for e in timeline
                    if e["event_type"] == "AgentActionTurnDispositionRecorded"
                    and e["payload"]["disposition"] == "singleton_refused"
                ]
                self.assertEqual(len(refused), corrected + 1)
                self.assertEqual(
                    [turn["payload"]["correction_count"] for turn in turns],
                    list(range(1, limit + 1)),
                )
                self.assertIsInstance(stopped.exception, AgentWorkStop)
                self.assertEqual(
                    (stopped.exception.cause, stopped.exception.source_event_id),
                    ("failed_turns", turns[-1]["event_id"]),
                )


if __name__ == "__main__":
    unittest.main()
