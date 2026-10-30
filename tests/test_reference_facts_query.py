from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from bio_neb.application.planning_read import read_planning_events
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from tests.reference_query_fakes import (
    RefiningPlanning,
    identity,
    planning_run,
    run_planning_attempt,
)


class CompanionExplorationTests(unittest.TestCase):
    def test_changed_companion_query_recovers_without_changing_reference(self):
        class Exploring(RefiningPlanning):
            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls[-1] == "describe_companion":
                    response.parts[0].args["accession"] = (
                        "GCF_000000001.1"
                        if self.calls.count("describe_companion") == 1
                        else "GCF_000009999.1"
                    )
                return response

        with tempfile.TemporaryDirectory() as directory:
            journey_file, composition = planning_run(
                Path(directory).resolve(), Exploring
            )
            run_planning_attempt(journey_file, composition)
            before = composition.companion_adapter.contact_count
            view = review_journey(ReviewJourneyQuery(journey_file=journey_file))
            facts = view.reference_facts.facts
            self.assertEqual(
                [f.result["outcome"] for f in facts], ["incompatible", "verified"]
            )
            self.assertEqual(
                [f.tool_input["assembly_accession"] for f in facts],
                ["GCF_000000001.1", "GCF_000009999.1"],
            )
            self.assertEqual(view.reference_facts.status, "reference_facts")
            self.assertEqual(view.reference_facts.no_progress_work_item_ids, ())
            events = read_evidence(Journey.open(Path(journey_file)))
            forged = copy.deepcopy(events)
            first = next(
                e for e in forged if e["event_type"] == "CompanionObservationRecorded"
            )
            first["payload"]["result"]["outcome"] = "verified"
            first["payload"]["result"]["candidates"][0]["compatibility"] = "compatible"
            with self.assertRaisesRegex(
                ValueError, "verified companion requires one compatible candidate"
            ):
                read_planning_events(forged)
            self.assertEqual(before, 2)
            self.assertEqual(composition.companion_adapter.contact_count, before)
            self.assertEqual(composition.calls.count("inspect_reference"), 1)
            self.assertEqual(composition.calls.count("record_reference_choice"), 1)
            self.assertFalse(
                any(
                    e["event_type"] == "ExecutionBound"
                    for e in read_evidence(Journey.open(Path(journey_file)))
                )
            )


class ReferenceFactsQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One real planning Attempt serves both cases; each replays its own copy.
        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.journey_file, cls.composition = planning_run(Path(directory).resolve())
        cls.review = run_planning_attempt(cls.journey_file, cls.composition)
        cls.retained = read_evidence(Journey.open(Path(cls.journey_file)))

    def test_composed_projection_reaches_reference_facts_and_remains_incomplete(self):
        journey_file, review = self.journey_file, self.review
        retained = Path(journey_file).read_bytes()
        self.assertEqual(
            review_journey(ReviewJourneyQuery(journey_file=journey_file)), review
        )
        self.assertEqual(Path(journey_file).read_bytes(), retained)
        self.assertEqual(review.status, "planning_hitl_pending")
        self.assertNotEqual(review.status, "proposal_complete")
        self.assertEqual(review.reference_facts.status, "reference_facts")
        works = review.reference_facts.work_items
        self.assertEqual(tuple(work.role for work in works), ("genomic_fasta",))
        self.assertEqual(
            review.reference_facts.completed_work_item_ids,
            tuple(work.work_item_id for work in works),
        )
        self.assertEqual(self.composition.calls.count("inspect_reference"), 1)
        self.assertEqual(self.composition.calls.count("describe_companion"), len(works))
        self.assertEqual(self.composition.companion_adapter.contact_count, len(works))
        self.assertFalse(
            any(
                e.event_type == "AgentOutputValidated"
                for e in review.attempts[-1].validation_events
            )
        )
        self.assertEqual(review.attempts[-1].agent_role, "planning_agent")
        policy = review.attempts[-1].effective_invocation_policy
        # One failure rule for every role, the role default's.
        self.assertEqual(
            policy.continuation.model_dump(mode="json"), {"failed_turns": 4}
        )
        self.assertEqual(
            policy.model_settings.request_settings(),
            {"parallel_tool_calls": False, "max_tokens": 245760},
        )
        self.assertEqual(review.attempts[-1].invocation_policy_id, policy.policy_id)
        timeline = self.retained
        facts = [
            e
            for e in timeline
            if e["payload"].get("attempt_id") == review.attempts[-1].attempt_id
        ]
        self.assertEqual(
            sum(e["event_type"] == "ModelRequestRecorded" for e in facts),
            len(self.composition.calls),
        )
        self.assertEqual(
            sum(e["event_type"] == "ModelResponseRecorded" for e in facts),
            len(self.composition.calls),
        )
        self.assertEqual(sum(e["event_type"] == "AttemptSucceeded" for e in facts), 1)
        observations = [
            e for e in facts if e["event_type"] == "CompanionObservationRecorded"
        ]
        self.assertEqual(len(observations), len(works))
        for observation, work in zip(observations, works, strict=True):
            self.assertEqual(
                observation["payload"]["tool_input"]["work_item"]["work_item_id"],
                work.work_item_id,
            )
            result = observation["payload"]["result"]
            self.assertEqual(result["outcome"], "verified")
            self.assertEqual(
                result["candidates"][0]["asset_id"], "source-owned-opaque-id"
            )
            # Each description is its own call's receipt and model-visible result.
            succeeded = next(
                e
                for e in facts
                if e["event_type"] == "ObservedToolExecutionSucceeded"
                and e["payload"]["stage_result_event_id"] == observation["event_id"]
            )
            started = next(
                e
                for e in facts
                if e["event_id"] == succeeded["payload"]["started_event_id"]
            )
            self.assertEqual(started["payload"]["tool_name"], "describe_companion")
            self.assertEqual(started["payload"]["adapter_identity"], "ncbi_datasets")
            # A source lookup returns its sources' answers beside it; the fake
            # companion adapter makes no GET, so it shows none.
            self.assertEqual(observation["payload"]["source_answers"], [])
            from bio_neb.agentic_journey.tool_execution import (
                model_view_of,
                tool_model_view,
            )

            # Its row's model view of the fact.
            visible = tool_model_view(model_view_of("describe_companion"), observation)
            self.assertEqual(visible["source_answers"], [])
            self.assertEqual(
                succeeded["payload"]["model_visible_result_digest"],
                hashlib.sha256(canonical_json(visible).encode()).hexdigest(),
            )

    def test_query_rejects_hostile_companion_tool_evidence(self):
        journey_file, timeline = self.journey_file, self.retained
        first_start = next(
            event
            for event in timeline
            if event["event_type"] == "ObservedToolExecutionStarted"
            and event["payload"]["tool_name"] == "describe_companion"
        )
        first_success = next(
            event
            for event in timeline
            if event["event_type"] == "ObservedToolExecutionSucceeded"
            and event["payload"]["started_event_id"] == first_start["event_id"]
        )

        changed_adapter = copy.deepcopy(timeline)
        next(
            event
            for event in changed_adapter
            if event["event_id"] == first_start["event_id"]
        )["payload"]["adapter_identity"] = "hostile_adapter"
        with self.assertRaisesRegex(ValueError, "changed its Adapter identity"):
            project_journey_events(changed_adapter, journey_file=journey_file)

        changed_digest = copy.deepcopy(timeline)
        next(
            event
            for event in changed_digest
            if event["event_id"] == first_success["event_id"]
        )["payload"]["model_visible_result_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "canonical result digest"):
            project_journey_events(changed_digest, journey_file=journey_file)

        changed_return = copy.deepcopy(timeline)
        first_success_index = next(
            index
            for index, event in enumerate(changed_return)
            if event["event_id"] == first_success["event_id"]
        )
        next_request = next(
            event
            for event in changed_return[first_success_index + 1 :]
            if event["event_type"] == "ModelRequestRecorded"
            and event["payload"]["attempt_id"] == first_start["payload"]["attempt_id"]
        )
        tool_return = next(
            part
            for message in next_request["payload"]["compatible_adapter_evidence"][
                "request"
            ]["messages"]
            for part in message.get("parts", [])
            if part.get("part_kind") == "tool-return"
            and part.get("tool_call_id") == first_start["payload"]["tool_call_id"]
        )
        tool_return["content"] = {"status": "hostile"}
        with self.assertRaisesRegex(ValueError, "same-call model return"):
            project_journey_events(changed_return, journey_file=journey_file)

        unknown_start = copy.deepcopy(timeline)
        next(
            event
            for event in unknown_start
            if event["event_id"] == first_start["event_id"]
        )["payload"]["tool_name"] = "unknown_observed_tool"
        with self.assertRaisesRegex(ValueError, "unrecognized observed Tool"):
            project_journey_events(unknown_start, journey_file=journey_file)

        orphan_terminal = copy.deepcopy(timeline)
        next(
            event
            for event in orphan_terminal
            if event["event_id"] == first_success["event_id"]
        )["payload"]["started_event_id"] = identity("orphan-companion-start")
        with self.assertRaisesRegex(ValueError, "terminal without its exact start"):
            project_journey_events(orphan_terminal, journey_file=journey_file)


if __name__ == "__main__":
    unittest.main()
