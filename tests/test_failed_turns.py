"""One failure rule for every Agent role: failed turns in a row (owner, #447 B).

A reply in which nothing worked (every call refused or repeated, a text or
reasoning-only ending, a cut-off, a refused output) is a failed turn; a reply in
which a Tool ran or an output was valid resets the count. At the limit (default
4) the Attempt stops for the Agent work Review. The number is tuned by value:
the model entry, the runtime file, a role scope or a stage scope.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationDefaults,
    ContinuationPolicy,
    StageInvocation,
    ToolInvocation,
)
from bio_neb.agentic_journey.role_declaration import ROLE_DEFAULT
from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition


class _PrematureChoice(PlanningComposition):
    """Records a reference choice the server refuses, whenever it is offered."""

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


class _RefusedBetweenProgress(PlanningComposition):
    """Three refused choices, then one call that does the stage's work; again."""

    async def respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        self.turn = getattr(self, "turn", 0) + 1
        offered = {tool.name for tool in info.function_tools}
        if "record_reference_choice" in offered and self.turn % 4:
            self.calls.append("refused")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "record_reference_choice",
                        {"candidate_id": "GCF_000001405.40", "reason": "Too early."},
                        tool_call_id=f"refused-{self.turn}",
                    )
                ]
            )
        return await super().respond(messages, info)


def _run(composition_type, **registered):
    with tempfile.TemporaryDirectory() as directory:
        receipt, paths = _registered(Path(directory), **registered)
        composition = composition_type(receipt, paths)
        with use_run_advancement_composition(composition):
            result = advance(ResumeRun(journey_file=receipt.journey_file))
        events = read_evidence(Journey.open(Path(receipt.journey_file)))
    return result, events, composition


def _refused_choices(events):
    return [
        e
        for e in events
        if e["event_type"] == "ToolCallValidated"
        and e["payload"]["tool_name"] == "record_reference_choice"
        and e["payload"]["status"] == "invalid"
    ]


class FailedTurnRuleTests(unittest.TestCase):
    def test_the_default_is_four_and_each_scope_tunes_it_by_value(self):
        self.assertEqual(ROLE_DEFAULT.failed_turns, 4)
        defaults = AgentInvocationDefaults(
            model_reference="qwen",
            failed_turns=5,
            scopes={
                "role": {"planning_agent": {"failed_turns": 6}},
                "stage": {"reference": {"failed_turns": 3}},
            },
        )
        self.assertEqual(defaults.failed_turns, 5)
        self.assertEqual(defaults.scopes.role["planning_agent"].failed_turns, 6)
        self.assertEqual(defaults.scopes.stage["reference"].failed_turns, 3)
        # A request does not know which Tool the model will call: no Tool scope.
        with self.assertRaises(ValidationError):
            ToolInvocation(failed_turns=2)
        self.assertEqual(StageInvocation(failed_turns=2).failed_turns, 2)
        # The four allowances it replaces are gone from every scope.
        for name in (
            "same_call_correction",
            "repeated_call",
            "truncation_retry",
            "output_repair",
        ):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                AgentInvocationDefaults(model_reference="qwen", **{name: 1})
        policy = ContinuationPolicy(failed_turns=6, stage_failed_turns={"reference": 3})
        self.assertEqual(policy.limit("reference"), 3)
        self.assertEqual(policy.limit("companions"), 6)
        self.assertEqual(policy.limit(None), 6)

    def test_four_failed_turns_in_a_row_stop_for_the_agent_work_review(self):
        result, events, composition = _run(_PrematureChoice)
        self.assertEqual(result.status, "agent_work_review_pending")
        (request,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentWorkReviewRequested"
        ]
        self.assertEqual(request["cause"], "failed_turns")
        # The Review names what the run was stuck on, read only.
        view = result.current_review.request
        self.assertEqual(
            (view.stuck_on.surface, view.stuck_on.failed_turns, view.stuck_on.limit),
            ("`record_reference_choice`", 4, 4),
        )
        self.assertNotIn("stuck_on", request)
        refused = _refused_choices(events)
        # The fourth failed turn stops: three corrections were sent.
        self.assertEqual(len(refused), 4)
        self.assertEqual(composition.calls.count("record_reference_choice"), 4)
        counts = [
            e["payload"]["correction_count"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["disposition"] == "singleton_refused"
        ]
        self.assertEqual(counts, [1, 2, 3, 4])

    def test_the_limit_wins_when_it_lands_on_the_last_budgeted_request(self):
        # The search, the plan, then four refused choices: the sixth reply is the
        # fourth failed turn and spends the budget. The framework's request limit
        # fires before the continuation check; the failed turns still stop.
        result, events, _ = _run(
            _PrematureChoice, agent_invocation={"model_requests": 6}
        )
        self.assertEqual(result.status, "agent_work_review_pending")
        self.assertNotIn(
            "AgentEffortReviewRequested", {e["event_type"] for e in events}
        )
        self.assertEqual(
            sum(e["event_type"] == "ModelRequestRecorded" for e in events), 6
        )

    def test_a_reply_that_does_work_resets_the_count(self):
        result, events, _ = _run(_RefusedBetweenProgress)
        # Three failed turns never reach four: the Attempt is not stopped, even
        # after far more refusals of the same call than any one allowance held.
        self.assertNotIn("AgentWorkReviewRequested", {e["event_type"] for e in events})
        self.assertNotEqual(result.status, "agent_work_review_pending")
        self.assertGreaterEqual(len(_refused_choices(events)), 6)

    def test_a_stage_scope_tunes_its_own_requests(self):
        result, events, _ = _run(
            _RefusedBetweenProgress,
            agent_invocation={"scopes": {"stage": {"reference": {"failed_turns": 3}}}},
        )
        self.assertEqual(result.status, "agent_work_review_pending")
        # One refusal, a reply that does work, then the third in a row stops.
        self.assertEqual(
            [
                e["payload"]["correction_count"]
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "singleton_refused"
            ],
            [1, 1, 2, 3],
        )
        # The notices name the same stopping number the stage's requests stop at
        # (Spec r1 SP2).
        from types import SimpleNamespace

        from bio_neb.agentic_journey.continuation_policy import RESEND_KEY
        from bio_neb.agentic_journey.invocation import _stopping_count

        (started,) = [e for e in events if e["event_type"] == "AttemptStarted"][:1]
        policy = SimpleNamespace(
            continuation=ContinuationPolicy.model_validate(
                started["payload"]["continuation_policy"]
            )
        )
        attempt_id = started["payload"]["attempt_id"]
        self.assertEqual(_stopping_count(policy, "text_ending", events, attempt_id), 3)
        self.assertIsNone(_stopping_count(policy, RESEND_KEY, events, attempt_id))
        # Another stage's number leaves the reference stage's requests at four.
        result, events, _ = _run(
            _RefusedBetweenProgress,
            agent_invocation={"scopes": {"stage": {"companions": {"failed_turns": 1}}}},
        )
        self.assertNotIn("AgentWorkReviewRequested", {e["event_type"] for e in events})


if __name__ == "__main__":
    unittest.main()
