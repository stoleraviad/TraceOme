from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bio_neb.application.planning_read import read_planning_events
from bio_neb.application.queries import (
    project_journey_events,
    read_planning,
    read_pre_execution,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_tools import reference_tool_values
from tests.agent_attempt_fakes import forge_request_binding
from tests.planning_assignment_fakes import PlanningComposition, eligible_tools
from tests.reference_query_fakes import (
    RefiningPlanning,
    build_query_timeline,
    identity,
    planning_run,
    run_planning_attempt,
    select_first_candidate,
)


def _response_parts(event):
    return event["payload"]["compatible_adapter_evidence"]["response"]["message"][
        "parts"
    ]


def _request(event):
    return event["payload"]["compatible_adapter_evidence"]["request"]


def _rebind_forged_capability_result(timeline, completed):
    """Keep a coherent forged envelope so semantic replay must reject the claim:
    the forged fact's own model view, as the Tool would have shown it (#478)."""
    from bio_neb.agentic_journey.tool_execution import model_view_of, tool_model_view

    observed = next(
        event
        for event in timeline
        if event["event_type"] == "ObservedToolExecutionSucceeded"
        and event["payload"]["stage_result_event_id"] == completed["event_id"]
    )
    started = next(
        event
        for event in timeline
        if event["event_id"] == observed["payload"]["started_event_id"]
    )
    visible = tool_model_view(model_view_of(started["payload"]["tool_name"]), completed)
    observed["payload"]["model_visible_result_digest"] = hashlib.sha256(
        canonical_json(visible).encode("utf-8")
    ).hexdigest()
    for event in timeline:
        if event["event_type"] != "ModelRequestRecorded":
            continue
        for message in _request(event).get("messages", []):
            for part in message.get("parts", []):
                if (
                    part.get("part_kind") == "tool-return"
                    and part.get("tool_call_id") == started["payload"]["tool_call_id"]
                ):
                    part["content"] = copy.deepcopy(visible)


class ReferenceExplorationTests(unittest.TestCase):
    def test_changed_inspection_query_preserves_choice_and_reopens_verified_basis(self):
        class Exploring(RefiningPlanning):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if (
                    self.calls[-1] == "inspect_reference"
                    and self.calls.count("inspect_reference") == 1
                ):
                    response.parts[0].args["accession"] = "GCF_000000001.1"
                return response

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition = planning_run(
                Path(directory).resolve(), Exploring
            )
            run_planning_attempt(journey_file, composition)
            owner = Journey.open(Path(journey_file))
            view = read_planning(owner)
            inspections = [
                f
                for f in view.reference.facts
                if f.event_type == "ReferenceDescriptionRecorded"
            ]
            self.assertEqual(
                [f.result["resolution"] for f in inspections],
                ["source_disagreement", "resolved"],
            )
            self.assertEqual(
                [f.result["assembly_accession"] for f in inspections],
                ["GCF_000000001.1", "GCF_000009999.1"],
            )
            self.assertEqual(len(view.reference.choices), 1)
            self.assertEqual(
                view.reference.verified_reference.assembly_accession, "GCF_000009999.1"
            )
            self.assertEqual(composition.calls.count("attest_sample"), 1)
            self.assertEqual(composition.calls.count("resolve_taxon"), 1)
            self.assertEqual(composition.calls.count("record_reference_choice"), 1)
            events = read_evidence(owner)
            self.assertEqual(read_planning_events(events), view)
            forged = copy.deepcopy(events)
            first = next(
                e for e in forged if e["event_type"] == "ReferenceDescriptionRecorded"
            )
            first["payload"]["result"]["resolution"] = "resolved"
            with self.assertRaisesRegex(
                ValueError, "differs from the selected identity"
            ):
                read_planning_events(forged)
            forged = copy.deepcopy(events)
            first = next(
                e for e in forged if e["event_type"] == "ReferenceDescriptionRecorded"
            )
            first["payload"]["tool_input"]["assembly_accession"] = "GCF_000009999.1"
            with self.assertRaisesRegex(ValueError, "changed its selected assembly"):
                read_planning_events(forged)
            self.assertFalse(
                any(e["event_type"] == "ExecutionBound" for e in read_evidence(owner))
            )


class ReferenceIdentityQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One real planning Run serves every replay case; each takes its own copy.
        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.journey_file, cls.retained, cls.stages = build_query_timeline(
            Path(directory).resolve()
        )

    def timeline(self, stage="final"):
        return copy.deepcopy(self.retained[: self.stages[stage]])

    def test_agent_can_choose_a_supported_candidate_without_human_selection(self):
        timeline = self.timeline()
        discovery_index = next(
            i
            for i, e in enumerate(timeline)
            if e["event_type"] == "ReferenceAssembliesRecorded"
        )
        prefix = timeline[: discovery_index + 1]
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: prefix))
        offer = reference_tool_values(deps, "record_reference_choice")
        self.assertIsNotNone(offer)
        # The model names the page it cites; any of its rows is a choice, checked
        # by membership with the researcher's words.
        self.assertEqual(offer.enums["discovery_event_id"], (prefix[-1]["event_id"],))
        rows = prefix[-1]["payload"]["result"]["rows"]
        self.assertEqual(
            [row["assembly_accession"] for row in rows],
            ["GCF_000001405.40", "GCF_000009999.1"],
        )

    def test_owned_planning_reads_active_prefix_and_retains_history_until_refresh(self):
        active = {}

        class ReadsDuringAttempt(RefiningPlanning):
            async def respond(self, messages, info):
                if not active and (
                    "lookup_compatible_resource_profiles"
                    in eligible_tools(self.journey, info.function_tools)
                ):
                    active_owner = Journey.open(owner.path)
                    active["selections"] = read_planning(
                        active_owner
                    ).execution.selections
                    try:
                        read_pre_execution(active_owner)
                    except ValueError as error:
                        active["pre_execution_error"] = error
                    active["historical"] = read_planning(owner)
                    active["owner_is_current"] = owner.is_current()
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition = planning_run(
                Path(directory).resolve(), ReadsDuringAttempt
            )
            pending = run_planning_attempt(journey_file, composition)
            select_first_candidate(journey_file, pending.planning_hitl.pending_request)
            owner = Journey.open(Path(journey_file))
            historical = read_planning(owner)
            self.assertEqual(historical.execution.selections, ())
            self.assertEqual(
                historical.hitl.receipts[-1].response_event_id,
                owner.read()[-1]["event_id"],
            )
            self.assertIsInstance(historical.plan.plan.requirements, tuple)

            run_planning_attempt(journey_file, composition)
            self.assertEqual(
                [selection.kind for selection in active["selections"]], ["recipe"]
            )
            self.assertIsInstance(active["pre_execution_error"], ValueError)
            self.assertEqual(active["historical"], historical)
            self.assertFalse(active["owner_is_current"])
            self.assertEqual(read_planning(owner), historical)
            self.assertFalse(owner.is_current())

            owner.refresh()
            complete = read_pre_execution(owner)
            self.assertEqual(len(complete.history.attempts), 2)
            self.assertEqual(
                complete.planning.reference.verified_reference.assembly_accession,
                "GCF_000009999.1",
            )
            self.assertEqual(
                [
                    selection.kind
                    for selection in complete.planning.execution.selections
                ],
                ["recipe", "resource"],
            )
            self.assertEqual(
                complete.planning, read_planning_events(read_evidence(owner))
            )
            complete.planning.reference.facts[-1].result.clear()
            complete.history.attempts[-1].events[0].payload.clear()
            fresh = read_pre_execution(owner)
            self.assertEqual(
                fresh.planning.reference.facts[-1].result["status"], "observed"
            )
            self.assertTrue(fresh.history.attempts[-1].events[0].payload)
            self.assertEqual(historical.execution.selections, ())

    def test_planning_rejects_recognized_malformed_facts_and_ignores_unrelated_events(
        self,
    ):
        timeline = self.timeline()
        plan_index = next(
            i
            for i, e in enumerate(timeline)
            if e["event_type"] == "ProposalRequirementsIdentified"
        )
        early = timeline[: plan_index + 1]
        unrelated = copy.deepcopy(early[-1])
        unrelated.update(
            event_id=identity("unrelated-planning-read"),
            event_type="ResearcherViewed",
            parent_ids=[early[-1]["event_id"]],
            payload={"note": [1, True]},
        )
        self.assertIsNone(
            read_planning_events([*early, unrelated]).reference.verified_reference
        )
        for name in (
            "SampleAttestationRecorded",
            "CompanionObservationRecorded",
            "RecipeObservationRecorded",
        ):
            with self.subTest(family=name), self.assertRaises(ValueError):
                read_planning_events([*early, {**unrelated, "event_type": name}])

    def test_review_accepts_compiled_terminal_without_model_receipt(self) -> None:
        timeline = self.timeline()
        review = project_journey_events(timeline, journey_file=self.journey_file)
        self.assertEqual(review.status, "execution_planning_ready")
        self.assertEqual(
            [attempt.agent_role for attempt in review.attempts],
            ["planning_agent", "planning_agent"],
        )
        self.assertFalse(
            any(
                event.event_type == "AgentOutputValidated"
                for attempt in review.attempts
                for event in attempt.validation_events
            )
        )
        self.assertFalse(
            any(
                _request(event)["output_tools"]
                for event in timeline
                if event["event_type"] == "ModelRequestRecorded"
            )
        )
        validated_call_ids = {
            event["payload"]["tool_call_id"]
            for event in timeline
            if event["event_type"] == "ToolCallValidated"
        }
        self.assertTrue(
            all(
                part["tool_call_id"] in validated_call_ids
                for event in timeline
                if event["event_type"] == "ModelResponseRecorded"
                for part in _response_parts(event)
            )
        )

    def test_review_accepts_requirement_grounded_hitl_before_sample_identity(
        self,
    ) -> None:
        class UnknownSample(PlanningComposition):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                call = response.parts[0]
                if call.tool_name == "record_proposal_requirements":
                    sample = next(
                        item
                        for item in call.args["requirements"]
                        if item["kind"] == "sample_identity"
                    )
                    sample["knowledge"] = {
                        "state": "to_find",
                        "evidence_event_ids": [self.registered.request.source_event_id],
                    }
                    sample["acquisition_route"] = "ask_planning_hitl"
                return response

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition = planning_run(
                Path(directory).resolve(), UnknownSample
            )
            run_planning_attempt(journey_file, composition)
            timeline = read_evidence(Journey.open(Path(journey_file)))
            plan_event = next(
                event
                for event in timeline
                if event["event_type"] == "ProposalRequirementsIdentified"
            )
            sample_requirement = next(
                requirement
                for requirement in plan_event["payload"]["plan"]["requirements"]
                if requirement["kind"] == "sample_identity"
            )
            review = project_journey_events(timeline, journey_file=journey_file)
            self.assertEqual(review.status, "planning_hitl_pending")
            pending = review.planning_hitl.pending_request
            self.assertEqual(
                pending.requirement_id, sample_requirement["requirement_id"]
            )
            self.assertEqual(pending.evidence_event_ids, (plan_event["event_id"],))
            self.assertFalse(
                any(
                    event["event_type"] == "SampleAttestationRecorded"
                    for event in timeline
                )
            )
            self.assertEqual(composition.reference_adapter.contact_count, 0)

    def test_review_projects_multi_tool_attempt_hitl_and_final_identity(self) -> None:
        expectations = (
            ("pending", "planning_hitl_pending"),
            ("responded", "execution_planning_progress"),
            ("final", "execution_planning_ready"),
        )
        for stage, status in expectations:
            with self.subTest(stage=stage):
                review = project_journey_events(
                    self.timeline(stage), journey_file=self.journey_file
                )
                self.assertEqual(review.status, status)
        final = project_journey_events(self.timeline(), journey_file=self.journey_file)
        self.assertEqual(len(final.attempts), 2)
        reference_attempt = final.attempts[0]
        valid_actions = [
            item
            for item in reference_attempt.validation_events
            if item.event_type == "ToolCallValidated"
            and item.payload["status"] == "valid"
        ]
        self.assertEqual(
            [item.payload["tool_name"] for item in valid_actions],
            [
                "search_journey",
                "record_proposal_requirements",
                "attest_sample",
                "resolve_taxon",
                "discover_assemblies",
                "discover_assemblies",
                "record_reference_choice",
                "inspect_reference",
                "describe_companion",
                "search_prebuilt_indexes",
                "record_prebuilt_index_choice",
                "list_compatible_analysis_recipes",
                "request_recipe_eligibility_assertion",
            ],
        )
        discoveries = [
            item
            for item in valid_actions
            if item.payload["tool_name"] == "discover_assemblies"
        ]
        self.assertEqual(discoveries[0].payload["arguments"], {"taxid": 9606})
        self.assertEqual(
            discoveries[1].payload["arguments"]["search_text"], ["isolate-B"]
        )
        self.assertEqual(len(reference_attempt.checkpoint_events), 1)
        self.assertEqual(len(final.planning_hitl.requests), 1)
        self.assertEqual(len(final.planning_hitl.receipts), 1)
        self.assertEqual(
            final.reference_identity.verified_reference.assembly_accession,
            "GCF_000009999.1",
        )
        self.assertIsNone(final.failure_code)

    def test_review_rejects_forged_reference_checkpoint_evidence(self) -> None:
        checkpoint = self.timeline()
        selected_checkpoint = next(
            event
            for event in checkpoint
            if event["event_type"] == "SliceCheckpointReached"
            and event["payload"]["checkpoint"] == "planning_hitl_requested"
        )
        selected_checkpoint["payload"]["planning_hitl_request_event_id"] = identity(
            "forged-evidence"
        )
        with self.assertRaisesRegex(
            ValueError, "checkpoint changed its compiled terminal source"
        ):
            project_journey_events(checkpoint, journey_file=self.journey_file)

        # The inspection's receipt names a fact another call recorded.
        borrowed = self.timeline()
        description = next(
            event
            for event in borrowed
            if event["event_type"] == "ReferenceDescriptionRecorded"
        )
        next(
            event
            for event in borrowed
            if event["event_type"] == "ObservedToolExecutionSucceeded"
            and event["payload"]["stage_result_event_id"] == description["event_id"]
        )["payload"]["stage_result_event_id"] = next(
            event["event_id"]
            for event in borrowed
            if event["event_type"] == "ReferenceAssembliesRecorded"
        )
        with self.assertRaisesRegex(ValueError, "changed its (exact )?stage result"):
            project_journey_events(borrowed, journey_file=self.journey_file)

    def test_review_rejects_reference_fact_owned_by_foreign_attempt(self) -> None:
        wrong_attempt = self.timeline()
        next(
            event
            for event in wrong_attempt
            if event["event_type"] == "ReferenceDescriptionRecorded"
        )["payload"]["attempt_id"] = identity("foreign-attempt")
        with self.assertRaisesRegex(
            ValueError, "observed Tool execution changed its exact stage result"
        ):
            project_journey_events(wrong_attempt, journey_file=self.journey_file)

    def test_review_rejects_forged_nonancestral_selection(self) -> None:
        selection = self.timeline()
        forged = identity("forged-selection")
        description = next(
            event
            for event in selection
            if event["event_type"] == "ReferenceDescriptionRecorded"
        )
        description["payload"]["tool_input"]["selected"]["selection_event_id"] = forged
        description["payload"]["result"]["selected"]["selection_event_id"] = forged
        _rebind_forged_capability_result(selection, description)
        # No choice or researcher answer is that selection.
        with self.assertRaisesRegex(
            ValueError, "reference selection lacks its exact Planning HITL assertion"
        ):
            project_journey_events(selection, journey_file=self.journey_file)

    def test_review_rejects_second_pending_planning_hitl_request(self) -> None:
        second_pending = self.timeline("pending")
        retained_request = next(
            event
            for event in reversed(second_pending)
            if event["event_type"] == "PlanningHitlRequested"
        )
        duplicate = copy.deepcopy(retained_request)
        duplicate.update(
            event_id=identity("second-pending"),
            parent_ids=[second_pending[-1]["event_id"]],
        )
        second_pending.append(duplicate)
        with self.assertRaisesRegex(ValueError, "another request is pending"):
            project_journey_events(second_pending, journey_file=self.journey_file)

    def test_review_rejects_forged_cross_attempt_action(self) -> None:
        unchanged = self.timeline()
        validations = [
            event
            for event in unchanged
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["tool_name"] == "list_compatible_analysis_recipes"
        ]
        model_calls = [
            part
            for event in unchanged
            if event["event_type"] == "ModelResponseRecorded"
            for part in _response_parts(event)
            if part.get("tool_name") == "list_compatible_analysis_recipes"
        ]
        self.assertNotEqual(
            validations[0]["payload"]["attempt_id"],
            validations[1]["payload"]["attempt_id"],
        )
        validations[1]["payload"]["arguments"] = copy.deepcopy(
            validations[0]["payload"]["arguments"]
        )
        model_calls[1]["args"] = copy.deepcopy(model_calls[0]["args"])
        forge_request_binding(unchanged, validations[1])
        # A fresh Attempt is outside the repeat scope (an exact provider
        # continuation); the forged call is refused by its retained domain result.
        with self.assertRaisesRegex(
            ValueError, "assignment domain result changed its admitted arguments"
        ):
            project_journey_events(unchanged, journey_file=self.journey_file)

    def test_review_requires_each_tool_result_in_the_later_model_request(self) -> None:
        timeline = self.timeline()
        second_request = next(
            event
            for event in timeline
            if event["event_type"] == "ModelRequestRecorded"
            and event["payload"]["request_number"] == 2
        )

        missing = copy.deepcopy(timeline)
        missing_request = next(
            event
            for event in missing
            if event["event_id"] == second_request["event_id"]
        )
        _request(missing_request)["messages"] = []
        with self.assertRaisesRegex(
            ValueError, "observed Tool execution changed its same-call model return"
        ):
            project_journey_events(missing, journey_file=self.journey_file)

        tampered = copy.deepcopy(timeline)
        tampered_request = next(
            event
            for event in tampered
            if event["event_id"] == second_request["event_id"]
        )
        tool_return = next(
            part
            for message in _request(tampered_request)["messages"]
            for part in message.get("parts", [])
            if part.get("part_kind") == "tool-return"
        )
        tool_return["content"] = {"status": "tampered"}
        with self.assertRaisesRegex(
            ValueError, "observed Tool execution changed its same-call model return"
        ):
            project_journey_events(tampered, journey_file=self.journey_file)

    def test_review_rejects_an_exact_repeat_inside_one_attempt(self) -> None:
        repeated = self.timeline("pending")
        discoveries = [
            event
            for event in repeated
            if event["event_type"] == "ReferenceAssembliesRecorded"
        ]
        validations = [
            event
            for event in repeated
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["tool_name"] == "discover_assemblies"
        ]
        model_calls = [
            part
            for event in repeated
            if event["event_type"] == "ModelResponseRecorded"
            for part in _response_parts(event)
            if part.get("tool_name") == "discover_assemblies"
        ]
        discoveries[1]["payload"]["tool_input"] = copy.deepcopy(
            discoveries[0]["payload"]["tool_input"]
        )
        discoveries[1]["payload"]["result"] = copy.deepcopy(
            discoveries[0]["payload"]["result"]
        )
        validations[1]["payload"]["arguments"] = copy.deepcopy(
            validations[0]["payload"]["arguments"]
        )
        validations[1]["payload"]["bound_argument_names"] = copy.deepcopy(
            validations[0]["payload"]["bound_argument_names"]
        )
        model_calls[1]["args"] = copy.deepcopy(model_calls[0]["args"])
        forge_request_binding(repeated, validations[1])
        _rebind_forged_capability_result(repeated, discoveries[1])
        with self.assertRaisesRegex(
            ValueError,
            "admitted action fails its recorded admission gates: planning_action_repeated",
        ):
            project_journey_events(repeated, journey_file=self.journey_file)


if __name__ == "__main__":
    unittest.main()
