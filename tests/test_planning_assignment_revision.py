"""A real planning assignment revises an Agent choice after deterministic conflict."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from bio_neb.application.advancement import (
    AnswerReview,
    PlanningReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
from bio_neb.recording import read_evidence
from tests.agent_attempt_fakes import forge_request_binding
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition, eligible_tools
from tests.planning_fakes import policy
from tests.reference_fakes import (
    PREBUILT_INDEX_TOOLS,
    recipe_selection_arguments,
    reference_call,
)


class RevisingComposition(PlanningComposition):
    def __init__(self, receipt, paths):
        super().__init__(receipt, paths)
        self.catalogue = CapabilitySnapshotAdapter(
            policy(
                recipes=2, resources=2, resource_recipe_ids=("germline-1", "germline-1")
            )
        )
        self.read_conflict = False

    def planning_assignment(self, journey):
        self.journey = journey
        from bio_neb.agentic_journey.planning_assignment import planning_assignment
        from tests.model_endpoint_fakes import ScriptedModelEndpoint

        return planning_assignment(
            reference_adapter=self.reference_adapter,
            companion_adapter=self.companion_adapter,
            galaxy_adapter=self.galaxy_adapter,
            recipe_adapter=self.catalogue,
            model_factory=ScriptedModelEndpoint(self.respond),
        )

    async def respond(self, messages, info):
        tools = eligible_tools(self.journey, info.function_tools)
        if "reconsider_recipe_choice" in tools:
            if not self.read_conflict:
                self.read_conflict = True
                name = "search_journey"
                arguments = {
                    "kinds": [
                        "PlanningCandidateSelected",
                        "ResourceObservationRecorded",
                    ],
                    "limit": 20,
                }
            else:
                name = "reconsider_recipe_choice"
                properties = tools[name].parameters_json_schema["properties"]
                arguments = {
                    key: value["const"]
                    for key, value in properties.items()
                    if "const" in value
                }
                arguments["reason"] = (
                    "The selected recipe has no compatible resource profile; reconsider the other validated recipe."
                )
        elif "select_recipe_candidate" in tools:
            name = "select_recipe_candidate"
            properties = tools[name].parameters_json_schema["properties"]
            arguments = {
                key: value["const"]
                for key, value in properties.items()
                if "const" in value
            } | recipe_selection_arguments(self.journey, tools[name])
            choices = properties.get("candidate_id", {}).get("enum")
            if choices:
                latest = next(
                    e
                    for e in reversed(self.journey.read())
                    if e["event_type"] == "RecipeObservationRecorded"
                )
                wanted = "germline-1" if self.read_conflict else "germline-0"
                arguments["candidate_id"] = next(
                    candidate["candidate_id"]
                    for candidate in latest["payload"]["result"]["candidates"]
                    if candidate["recipe_id"] == wanted
                )
        else:
            return await super().respond(messages, info)
        self.calls.append(name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
                )
            ]
        )


def _answer_first(observation):
    request = observation.planning_hitl.pending_request
    return AnswerReview(
        journey_file=observation.journey_file,
        request_event_id=request.request_event_id,
        answer=PlanningReviewAnswer(
            kind="planning",
            response=SelectPlanningHitlResponse(
                action="select", candidate_id=request.candidates[0].candidate_id
            ),
        ),
    )


class HumanSelectedComposition(RevisingComposition):
    async def respond(self, messages, info):
        tools = eligible_tools(self.journey, info.function_tools)
        if "request_recipe_selection" in tools and "select_recipe_candidate" in tools:
            name = "request_recipe_selection"
            arguments = {
                key: value["const"]
                for key, value in tools[name]
                .parameters_json_schema["properties"]
                .items()
                if "const" in value
            }
            self.calls.append(name)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"assignment-{len(self.calls)}"
                    )
                ]
            )
        assert "reconsider_recipe_choice" not in tools
        # The genomic FASTA's index Tools stay callable after its choice.
        if (
            set(tools) - set(PREBUILT_INDEX_TOOLS) == {"search_journey"}
            and reference_call(Journey.open(Path(self.journey.path)), tools) is None
        ):
            # The researcher's recipe cannot be reconsidered and no resource is
            # compatible with it: nothing is left for the model to call.
            self.calls.append("text")
            return ModelResponse(
                parts=[
                    TextPart(
                        "The researcher-selected recipe has no compatible resource "
                        "profile, and that selection cannot be reconsidered here."
                    )
                ]
            )
        return await super().respond(messages, info)


class PlanningAssignmentRevisionTests(unittest.TestCase):
    def test_conflicting_resource_evidence_revises_recipe_and_retains_history(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        receipt, paths = _registered(tmp_path)
        composition = RevisingComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            # The reference companions settle without a question, so the first
            # question is the recipe eligibility; its answer lets the Agent select,
            # meet the conflicting resource evidence and reconsider.
            eligibility = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(
                eligibility.planning_hitl.pending_request.subject_key, "analysis_recipe"
            )
            retained_before = Journey.open(Path(receipt.journey_file)).read()
            revised = advance(_answer_first(eligibility))
        assert revised.status == "planning_hitl_pending"
        self.assertEqual(
            revised.planning_hitl.pending_request.subject_key, "resource_profile"
        )
        assert composition.read_conflict
        assert composition.catalogue.contact_count == 5
        assert composition.reference_adapter.contact_count == 3
        assert composition.companion_adapter.contact_count == 1
        owner = Journey.open(Path(receipt.journey_file))
        events = owner.read()
        assert events[: len(retained_before)] == retained_before
        reason = next(
            e for e in events if e["event_type"] == "PlanningChoiceReconsidered"
        )
        selections = [
            e
            for e in events
            if e["event_type"] == "PlanningCandidateSelected"
            and e["payload"]["kind"] == "recipe"
        ]
        assert len(selections) == 2
        assert (
            selections[0]["payload"]["candidate_id"]
            != selections[1]["payload"]["candidate_id"]
        )
        assert (
            selections[0]["payload"]["basis_event_id"]
            != selections[1]["payload"]["basis_event_id"]
        )
        assert (
            reason["payload"]["prior_selection_event_id"] == selections[0]["event_id"]
        )
        assert (
            revised.execution_planning.selected_recipe_bindings[0].selection_event_id
            == selections[1]["event_id"]
        )
        resource_facts = [
            e for e in events if e["event_type"] == "ResourceObservationRecorded"
        ]
        assert len(resource_facts) == 2
        assert resource_facts[0]["payload"]["result"]["outcome"] == "observed"
        assert all(
            candidate["compatibility"] == "incompatible"
            for candidate in resource_facts[0]["payload"]["result"]["candidates"]
        )
        assert (
            resource_facts[1]["payload"]["tool_input"]["work_item"]["recipe_event_id"]
            == selections[1]["event_id"]
        )
        history = [e for e in events if e["event_type"] == "JourneySearchRead"][-1]
        assert selections[0] in history["payload"]["result"]["events"]
        assert resource_facts[0] in history["payload"]["result"]["events"]
        assert len(
            [
                e
                for e in read_evidence(owner)
                if e["event_type"] == "ModelRequestRecorded"
            ]
        ) == len(composition.calls)
        assert not any(e["event_type"] == "ExecutionProposalSubmitted" for e in events)
        assert (
            review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).execution_planning
            == revised.execution_planning
        )

    def test_reconsideration_reopens_only_against_its_recorded_conflict(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        receipt, paths = _registered(Path(temporary.name))
        composition = RevisingComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            companions = advance(ResumeRun(journey_file=receipt.journey_file))
            advance(_answer_first(advance(_answer_first(companions))))
        assert "reconsider_recipe_choice" in composition.calls
        retained = read_evidence(Journey.open(Path(receipt.journey_file)))
        project_journey_events(retained, journey_file=receipt.journey_file)

        # Recorded relationships prove the reason; an ancestral event of another
        # kind cannot stand in for the conflicting resource evidence.
        hostile = copy.deepcopy(retained)
        forged = next(
            e for e in hostile if e["event_type"] == "PlanningChoiceReconsidered"
        )["payload"]
        authority = next(
            e for e in hostile if e["event_id"] == forged["authority_event_id"]
        )["payload"]
        call = next(
            part
            for e in hostile
            if e["event_type"] == "ModelResponseRecorded"
            for part in e["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
            if part.get("tool_call_id") == authority["tool_call_id"]
        )
        arguments = (
            json.loads(call["args"]) if isinstance(call["args"], str) else call["args"]
        )
        for recorded in (forged, authority["arguments"], arguments):
            recorded["conflicting_evidence_event_id"] = forged[
                "prior_selection_event_id"
            ]
        if isinstance(call["args"], str):
            call["args"] = json.dumps(arguments)
        forge_request_binding(
            hostile,
            next(e for e in hostile if e["event_id"] == forged["authority_event_id"]),
        )
        with self.assertRaisesRegex(
            ValueError,
            "recipe reconsideration lacks its recorded conflict with an Agent choice",
        ):
            project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_conflict_cannot_override_exact_human_recipe_choice(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        receipt, paths = _registered(tmp_path)
        composition = HumanSelectedComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            # The reference companions settle without a question, so the first
            # question is the recipe eligibility and the second the recipe choice.
            eligibility = advance(ResumeRun(journey_file=receipt.journey_file))
            request = eligibility.planning_hitl.pending_request
            self.assertEqual(request.subject_key, "analysis_recipe")
            selection = advance(_answer_first(eligibility))
            request = selection.planning_hitl.pending_request
            self.assertEqual(request.subject_key, "analysis_recipe")
            latest = next(
                e
                for e in reversed(composition.journey.read())
                if e["event_type"] == "RecipeObservationRecorded"
            )
            candidate_id = next(
                c["candidate_id"]
                for c in latest["payload"]["result"]["candidates"]
                if c["recipe_id"] == "germline-0"
            )
            candidate = next(
                c for c in request.candidates if c.asserted_value == candidate_id
            )
            answer = AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=SelectPlanningHitlResponse(
                        action="select", candidate_id=candidate.candidate_id
                    ),
                ),
            )
            stopped = advance(answer)
        # With nothing left to call, the model's text answers spend their
        # allowance and the Attempt stops for the Agent work Review.
        self.assertEqual(stopped.status, "agent_work_review_pending")
        events = Journey.open(Path(receipt.journey_file)).read()
        # The incompatible resource result returns to the model; no terminal ends
        # the Attempt on it (K5).
        resource = max(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "ResourceObservationRecorded"
        )
        assert "SliceCheckpointReached" not in {
            e["event_type"] for e in events[resource:]
        }, stopped.status
        selected = [
            e
            for e in events
            if e["event_type"] == "PlanningCandidateSelected"
            and e["payload"]["kind"] == "recipe"
        ]
        assert len(selected) == 1
        assert selected[0]["payload"]["authority"] == "researcher_selection"
        assert not any(
            e["event_type"] in {"PlanningChoiceReconsidered", "RunFailed"}
            for e in events
        )
        assert "reconsider_recipe_choice" not in composition.calls
        assert not composition.read_conflict
