"""The real adaptive caller's journal lineage remains independently checkable."""

import copy
import tempfile
import unittest
from pathlib import Path

from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import project_journey_events
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition


class ToolContactReopenTests(unittest.TestCase):
    def test_actual_assignment_reopen_rejects_wrong_call_response_result_and_return(
        self,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        receipt, paths = _registered(tmp_path)
        composition = PlanningComposition(receipt, paths)
        with use_run_advancement_composition(composition):
            accepted = advance(ResumeRun(journey_file=receipt.journey_file))
        journey = Journey.open(Path(receipt.journey_file))
        assert not any(
            event["event_type"].startswith("ObservedToolExecution")
            for event in journey.read()
        )
        events = read_evidence(journey)
        starts = [
            event
            for event in events
            if event["event_type"] == "ObservedToolExecutionStarted"
        ]
        successes = [
            event
            for event in events
            if event["event_type"] == "ObservedToolExecutionSucceeded"
        ]
        assert len(starts) == len(successes) == len(composition.calls)
        assert (
            project_journey_events(events, journey_file=receipt.journey_file).status
            == accepted.status
        )
        for changed in ("call", "response", "result", "digest", "return"):
            hostile = copy.deepcopy(events)
            start = next(
                event for event in hostile if event["event_id"] == starts[0]["event_id"]
            )
            terminal = next(
                event
                for event in hostile
                if event["event_type"] == "ObservedToolExecutionSucceeded"
                and event["payload"]["started_event_id"] == start["event_id"]
            )
            if changed == "call":
                start["payload"]["tool_call_id"] = starts[1]["payload"]["tool_call_id"]
            elif changed == "response":
                start["payload"]["response_event_id"] = starts[1]["payload"][
                    "response_event_id"
                ]
            elif changed == "result":
                terminal["payload"]["stage_result_event_id"] = successes[1]["payload"][
                    "stage_result_event_id"
                ]
            elif changed == "digest":
                terminal["payload"]["model_visible_result_digest"] = "a" * 64
            else:
                request = next(
                    event
                    for event in hostile[hostile.index(terminal) + 1 :]
                    if event["event_type"] == "ModelRequestRecorded"
                )
                returned = next(
                    part
                    for message in request["payload"]["compatible_adapter_evidence"][
                        "request"
                    ]["messages"]
                    for part in message["parts"]
                    if part["part_kind"] == "tool-return"
                    and part["tool_call_id"] == start["payload"]["tool_call_id"]
                )
                returned["content"] = {"borrowed": "another Tool result"}
            with self.assertRaises(ValueError):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_investigation_artifact_identity_cannot_be_borrowed_from_another_tool(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        from bio_neb.investigation.investigation import project_investigation
        from tests.sequential_investigation_support import (
            SourceResponsesOnly,
            _AdaptiveInvestigationModel,
            run_attempt,
        )

        model = _AdaptiveInvestigationModel()
        sources = SourceResponsesOnly()
        projection, journey = run_attempt(tmp_path, model, sources)
        assert projection.status == "review_pending"
        assert sources.contact_count == 2
        reopened = Journey.open(journey.path)
        events = read_evidence(reopened)
        assert project_investigation(events) == projection
        terminals = [
            event
            for event in events
            if event["event_type"] == "ObservedToolExecutionSucceeded"
        ]
        assert len(terminals) == 2
        assert all(event["payload"]["artifact_source_event_ids"] for event in terminals)
        for field in ("artifact_source_event_ids", "artifact_digests"):
            hostile = copy.deepcopy(events)
            target = next(
                event
                for event in hostile
                if event["event_id"] == terminals[0]["event_id"]
            )
            target["payload"][field] = terminals[1]["payload"][field]
            with self.assertRaises(ValueError):
                project_investigation(hostile)

        from bio_neb.agentic_journey.source_contacts import source_contact_units

        source_contact_units(events)
        hostile = copy.deepcopy(events)
        scopes = [
            event
            for event in hostile
            if event["event_type"] == "SourceContactScopeStarted"
        ]
        assert len(scopes) == 2
        scopes[0]["payload"]["tool_call_id"] = scopes[1]["payload"]["tool_call_id"]
        with self.assertRaises(ValueError):
            source_contact_units(hostile)
