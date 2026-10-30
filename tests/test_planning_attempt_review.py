from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    AgentInvocationPolicy,
)
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import PlanningComposition

_ENVIRON = {"LOCAL_MODEL_KEY": "secret"}
# The first action of the scripted planning composition, so a chosen candidate
# is exactly what the continuing Attempt proposes next.
_HISTORY_ARGUMENTS = {"kinds": ["ResearchRequestRegistered"], "limit": 20}
# Recognized but not exposed before proposal requirements exist.
_TAXON_ARGUMENTS = {"term": "Homo sapiens"}
_CONTINUED_CALLS = [
    "search_journey",
    "record_proposal_requirements",
    "attest_sample",
    "resolve_taxon",
    "discover_assemblies",
    "record_reference_choice",
    "inspect_reference",
    "describe_companion",
    "list_compatible_analysis_recipes",
    "request_recipe_eligibility_assertion",
]


def _inspected_run(root: Path, **registration):
    receipt, paths = _registered(root, **registration)
    journey = Journey.open(Path(receipt.journey_file))
    inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=str(journey.path)), journey=journey
    )
    return receipt, paths


def _insert_unreceipted_fact(events, fact, *, before):
    """Retain a re-ided copy of ``fact`` in ``before``'s causal place, with no receipt."""
    forged = {
        **copy.deepcopy(fact),
        "event_id": str(uuid.uuid4()),
        "parent_ids": before["parent_ids"],
    }
    before["parent_ids"] = [forged["event_id"]]
    events.insert(events.index(before), forged)


class PlanningAttemptReviewTests(unittest.TestCase):
    def test_reopen_uses_recorded_policy_and_refuses_hostile_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            journey_file, review, _ = self._pending_review(
                Path(directory).resolve(),
                invocation_overrides={"model_requests": 200},
            )
            before = Path(journey_file).read_bytes()
            runtime_file = Path(review.registered_input_set.runtime_config_file)
            changed = json.loads(runtime_file.read_text())
            changed["agent_invocation"] = {
                "model_settings": {"max_tokens": 245760},
                "failed_turns": 7,
            }
            runtime_file.write_text(json.dumps(changed))
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=journey_file)).attempt,
                review.attempt,
            )
            self.assertEqual(Path(journey_file).read_bytes(), before)
            timeline = read_evidence(Journey.open(Path(journey_file)))
            for missing in ("agent_invocation", "max_tokens", "model_availability"):
                hostile = copy.deepcopy(timeline)
                selections = next(
                    event["payload"]["runtime_selections"]
                    for event in hostile
                    if event["event_type"] == "RunInputsRegistered"
                )
                if missing == "agent_invocation":
                    del selections[missing]
                elif missing == "max_tokens":
                    del selections["model_roster"]["qwen"]["settings_selection"][
                        "baseline"
                    ][missing]
                else:
                    del selections["agent_invocation"]["exhaustion"][missing]
                with (
                    self.subTest(missing_default=missing),
                    self.assertRaises(ValueError),
                ):
                    project_journey_events(hostile, journey_file=journey_file)
            for mutation in (
                "identity",
                "override",
                "request",
                "incomplete",
                "forged_effective",
                "settings_selection",
            ):
                with self.subTest(mutation=mutation):
                    hostile = copy.deepcopy(timeline)
                    start = next(
                        e["payload"]
                        for e in hostile
                        if e["event_type"] == "AttemptStarted"
                        and e["payload"]["agent_role"] == "planning_agent"
                    )
                    if mutation == "identity":
                        start["invocation_policy_id"] = "invocation-policy:" + "0" * 64
                    elif mutation == "override":
                        start["invocation_overrides"] = {"model_requests": 150}
                    elif mutation == "incomplete":
                        del start["effective_invocation_policy"]["loop_retry"]
                    elif mutation == "forged_effective":
                        start["effective_invocation_policy"]["continuation"][
                            "failed_turns"
                        ] = 9
                        start["continuation_policy"]["failed_turns"] = 9
                        forged = AgentInvocationPolicy.model_validate(
                            start["effective_invocation_policy"]
                        )
                        start["invocation_policy_id"] = forged.policy_id
                        start["continuation_policy_id"] = forged.continuation.policy_id
                    else:
                        request = next(
                            e["payload"]
                            for e in hostile
                            if e["event_type"] == "ModelRequestRecorded"
                            and e["payload"]["attempt_id"] == start["attempt_id"]
                        )
                        # The transport record binds the request budget and its
                        # maximum; a field it does not bind reaches the
                        # invocation binding.
                        if mutation == "settings_selection":
                            request["settings_selection"]["accepted_settings"].remove(
                                "seed"
                            )
                        else:
                            settings = request["compatible_adapter_evidence"][
                                "request"
                            ]["model_settings"]
                            settings["parallel_tool_calls"] = not settings[
                                "parallel_tool_calls"
                            ]
                    with self.assertRaisesRegex(ValueError, "invocation"):
                        project_journey_events(hostile, journey_file=journey_file)

    def test_unsupported_policy_refuses_before_model_construction(self):
        settings = (
            {"failed_turns": 0},
            {"loop_retry": 1},
            {"tool_retry": 1},
            {"failed_turns": -1},
            {"failed_turns": True},
            # The retired per-kind allowances are no override.
            {"output_repair": 1},
            {"action_correction": 0},
            {"same_call_correction": 0},
            {"repeated_call": 2},
            {"truncation_retry": 0},
            {"exhaustion": {"singleton_action": "terminal"}},
            {"exhaustion": {"repeated_multi_action": "terminal"}},
            {"exhaustion": {"model_availability": "terminal"}},
            {"mandatory_stage_review": False},
            {"mandatory_stage_review": True},
            {"model_settings": {"temperature": "0.1"}},
            {"model_settings": {"parallel_tool_calls": "true"}},
            {"model_settings": {"max_tokens": 0}},
            {"failed_turns": None},
            {"eligibility": "always"},
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _inspected_run(Path(directory).resolve())
            journey_file = receipt.journey_file
            before = Path(journey_file).read_bytes()

            def unexpected_model(_, _selected):
                self.fail("unsupported invocation constructed a model")

            for value in settings:
                with self.subTest(value=value):
                    with self.assertRaises(ValidationError):
                        planning_assignment(model_factory=unexpected_model).run(
                            Journey.open(Path(journey_file)),
                            environ=_ENVIRON,
                            invocation_overrides=AgentInvocationOverrides.model_validate(
                                value
                            ),
                        )
                    self.assertEqual(Path(journey_file).read_bytes(), before)

    def test_public_override_changes_only_declared_invocation(self):
        # Every reply is a failed turn: the failure rule's number of requests.
        for overrides, expected_requests in (
            (
                # A model setting is configuration only; an Attempt's
                # recorded amendments carry the failure rule's number and the
                # request budget.
                {"failed_turns": 2, "model_requests": 200},
                2,
            ),
            ({}, 4),
        ):
            with (
                self.subTest(overrides=overrides),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, _ = _inspected_run(Path(directory).resolve())
                journey_file = receipt.journey_file

                async def invalid_action(messages, info):
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "resolve_taxon",
                                _TAXON_ARGUMENTS,
                                tool_call_id=str(uuid.uuid4()),
                            )
                        ]
                    )

                invocation = AgentInvocationOverrides.model_validate_json(
                    AgentInvocationOverrides.model_validate(overrides).model_dump_json()
                )
                review = planning_assignment(
                    model_factory=ScriptedModelEndpoint(invalid_action)
                ).run(
                    Journey.open(Path(journey_file)),
                    environ=_ENVIRON,
                    invocation_overrides=invocation,
                )
                self.assertEqual(review.status, "agent_work_review_pending")
                self.assertIsNone(
                    review.registered_input_set.runtime_selections.agent_invocation.failed_turns
                )
                self.assertEqual(review.current_review.request.cause, "failed_turns")
                self.assertEqual(review.attempt.invocation_overrides, overrides)
                requests = [
                    e
                    for e in review.attempt.model_events
                    if e.event_type == "ModelRequestRecorded"
                ]
                self.assertEqual(len(requests), expected_requests)
                # The Run's one recorded choice rides with the override.
                expected_settings = {
                    "parallel_tool_calls": False,
                    "max_tokens": 245760,
                    "tool_choice": "auto",
                }
                self.assertEqual(
                    requests[0].payload["compatible_adapter_evidence"]["request"][
                        "model_settings"
                    ],
                    expected_settings,
                )
                self.assertEqual(
                    review_journey(
                        ReviewJourneyQuery(journey_file=journey_file)
                    ).attempt,
                    review.attempt,
                )

    def test_public_invocation_retains_defaults_through_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            journey_file, review, _ = self._pending_review(Path(directory).resolve())
            policy = review.attempt.effective_invocation_policy
            # One failure rule for every role, the role default's.
            self.assertEqual(
                policy.continuation.model_dump(mode="json"), {"failed_turns": 4}
            )
            self.assertEqual(policy.loop_retry, 0)
            self.assertEqual(policy.tool_retry, 0)
            self.assertEqual(policy.exhaustion.failed_turns, "review")
            self.assertNotIn("repeated_multi_action", policy.exhaustion.model_dump())
            self.assertEqual(policy.exhaustion.model_availability, "review")
            self.assertEqual(review.attempt.invocation_overrides, {})
            requests = [
                e
                for e in review.attempt.model_events
                if e.event_type == "ModelRequestRecorded"
            ]
            self.assertEqual(
                requests[0].payload["compatible_adapter_evidence"]["request"][
                    "model_settings"
                ],
                {
                    "parallel_tool_calls": False,
                    "max_tokens": 245760,
                    "tool_choice": "auto",
                },
            )
            reopened = review_journey(ReviewJourneyQuery(journey_file=journey_file))
            self.assertEqual(reopened.attempt.effective_invocation_policy, policy)

    class MixedMultiModel:
        def __init__(self) -> None:
            self.calls = 0
            self.model = ScriptedModelEndpoint(self.respond)

        async def respond(self, messages, info):
            del messages, info
            self.calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        _HISTORY_ARGUMENTS,
                        tool_call_id=f"history-{self.calls}",
                    ),
                    ToolCallPart(
                        "resolve_taxon",
                        _TAXON_ARGUMENTS,
                        tool_call_id=f"unavailable-{self.calls}",
                    ),
                ]
            )

    def _pending_review(
        self,
        root: Path,
        composition_type=PlanningComposition,
        *,
        invocation_overrides=None,
        **registration,
    ):
        receipt, paths = _inspected_run(root, **registration)
        journey_file = receipt.journey_file
        scripted = self.MixedMultiModel()
        composition = composition_type(receipt, paths)
        review = planning_assignment(
            model_factory=scripted.model,
            reference_adapter=composition.reference_adapter,
            companion_adapter=composition.companion_adapter,
            galaxy_adapter=composition.galaxy_adapter,
        ).run(
            Journey.open(Path(journey_file)),
            environ=_ENVIRON,
            invocation_overrides=AgentInvocationOverrides.model_validate(
                invocation_overrides or {}
            ),
        )
        # Each mixed reply's unoffered call is refused on its own and its search
        # runs once: the first reply ran a Tool, so it is no failed turn; every
        # later one repeats the search with nothing changed and refuses the
        # unoffered call, so nothing of it works. The ``failed_turns``-th such
        # reply in a row stops the Attempt for the Agent work review.
        continuation = review.attempt.effective_invocation_policy.continuation
        self.assertEqual(review.status, "agent_work_review_pending")
        self.assertEqual(review.current_review.request.cause, "failed_turns")
        self.assertEqual(scripted.calls, continuation.failed_turns + 1)
        self.assertEqual(
            [event.event_type for event in review.attempt.events].count(
                "JourneySearchRead"
            ),
            1,
        )
        self.assertFalse(
            any(
                event.event_type
                in {"TaxonResolutionRecorded", "AgentActionReviewRequested"}
                for event in review.attempt.events
            )
        )
        return journey_file, review, composition

    def test_a_mixed_reply_refuses_its_unoffered_call_before_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            journey_file, _review, composition = self._pending_review(
                Path(directory).resolve()
            )
            self.assertEqual(composition.reference_adapter.contact_count, 0)
            events = read_evidence(Journey.open(Path(journey_file)))
            first = [
                event["payload"]
                for event in events
                if event["event_type"] == "ToolCallValidated"
            ][:2]
            # The unoffered call is refused while its reply is routed, before the
            # offered search is admitted and runs.
            self.assertEqual(
                [(v["tool_name"], v["errors"]) for v in first],
                [
                    ("resolve_taxon", ["tool_name_unavailable"]),
                    ("search_journey", []),
                ],
            )
            self.assertEqual(
                [
                    event["payload"]["disposition"]
                    for event in events
                    if event["event_type"] == "AgentActionTurnDispositionRecorded"
                ][:2],
                ["multi_call_admitted", "multi_call_refused"],
            )

    def test_unreceipted_tool_fact_in_successful_planning_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _inspected_run(Path(directory).resolve())
            journey_file = receipt.journey_file
            continued = PlanningComposition(receipt, paths)
            continued.planning_assignment(Journey.open(Path(journey_file))).run(
                Journey.open(Path(journey_file)), environ=_ENVIRON
            )
            retained = read_evidence(Journey.open(Path(journey_file)))
            project_journey_events(retained, journey_file=journey_file)
            forgery = copy.deepcopy(retained)
            history = next(
                event for event in forgery if event["event_type"] == "JourneySearchRead"
            )
            self.assertIn(
                "AttemptSucceeded",
                [
                    event["event_type"]
                    for event in forgery
                    if event["payload"].get("attempt_id")
                    == history["payload"]["attempt_id"]
                ],
            )
            successor = next(
                event
                for event in forgery
                if event["parent_ids"] == [history["event_id"]]
            )
            _insert_unreceipted_fact(forgery, history, before=successor)
            with self.assertRaisesRegex(
                ValueError, "Tool fact is not bound to exactly one terminal receipt"
            ):
                project_journey_events(forgery, journey_file=journey_file)


if __name__ == "__main__":
    unittest.main()
