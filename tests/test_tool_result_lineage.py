"""The loop's lineage query behind a Tool result and its Attempt closure (#437, B5).

The Investigation reads a Tool result's admitted call and Attempt closure through
``tool_execution`` instead of walking the timeline itself (ADR 0044: the loop
owns its turn contract), and the member context behind it as its own recorded
fact, never parsed back from a prompt (#444).
"""

import copy
import tempfile
import unittest
from pathlib import Path

from bio_neb.agentic_journey.tool_execution import (
    tool_result_lineage,
    validate_attempt_closure,
)
from bio_neb.investigation.investigation import (
    project_investigation,
    validate_investigation_fact,
)
from bio_neb.investigation.investigation_commands import BeginInvestigationCommand
from bio_neb.investigation.member_context import (
    member_context_fact,
    member_request,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    begin_in_pass,
    collection_request,
)
from tests.terminal_journey_fakes import forked_child


class ToolResultLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        document = collection_request(2)
        journey = forked_child(root, investigation_request=document)
        request_file = root / "investigation-request.json"
        request_file.write_text(canonical_json(document), encoding="utf-8")
        begin_in_pass(
            BeginInvestigationCommand(
                journey_file=str(journey.path), request_files=(str(request_file),)
            ),
            environ={"LOCAL_MODEL_KEY": "offline-test-value"},
            model_factory=SequentialAdaptiveModel().model,
            source_adapter_factory=SourceResponsesOnly,
        )
        cls.events = read_evidence(Journey.open(journey.path))

    def results(self):
        return [e for e in self.events if e["event_type"] == "RefSnpToolResultRecorded"]

    def test_a_result_reads_its_admitted_call_and_member_context(self):
        results = self.results()
        self.assertEqual(len(results), 2)
        by_id = {e["event_id"]: e for e in self.events}
        for result in results:
            payload = result["payload"]
            lineage = tool_result_lineage(result["event_id"], self.events)
            with self.subTest(member=payload["collection_member_id"]):
                self.assertEqual(lineage.started["attempt_id"], payload["attempt_id"])
                self.assertEqual(
                    lineage.started["tool_call_id"], payload["tool_call_id"]
                )
                self.assertEqual(lineage.validation["tool_name"], "fetch_refsnp")
                self.assertEqual(lineage.request["event_type"], "ModelRequestRecorded")
                # The member context behind the request is its own recorded fact.
                fact = member_context_fact(
                    lineage.request, self.events, validate_investigation_fact
                )
                self.assertEqual(
                    fact.collection_member_id, payload["collection_member_id"]
                )
                self.assertEqual(
                    member_request(fact, by_id, validate_investigation_fact).rsid,
                    payload["query"],
                )

    def test_a_result_without_exactly_one_receipt_has_no_lineage(self):
        result = self.results()[0]
        receipt = next(
            e
            for e in self.events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
            and e["payload"]["stage_result_event_id"] == result["event_id"]
        )
        doubled = [*self.events, {**copy.deepcopy(receipt), "event_id": "copy"}]
        without = [e for e in self.events if e is not receipt]

        self.assertIsNone(tool_result_lineage(result["event_id"], doubled))
        self.assertIsNone(tool_result_lineage(result["event_id"], without))

    def test_an_attempt_closes_once_right_after_its_terminal(self):
        starts = [
            e
            for e in self.events
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["agent_role"] == "investigation_planner"
        ]
        self.assertEqual(len(starts), 2)
        for start in starts:
            attempt_id = start["payload"]["attempt_id"]
            terminal = next(
                e
                for e in self.events
                if e["event_type"] == "InvestigationAttemptCompleted"
                and e["payload"]["attempt_id"] == attempt_id
            )
            validate_attempt_closure(self.events, start, terminal)
            closure = next(
                e
                for e in self.events
                if e["event_type"] == "activity_succeeded"
                and e["payload"].get("result_event_id") == terminal["event_id"]
            )
            hostile = copy.deepcopy(self.events)
            moved = next(e for e in hostile if e["event_id"] == closure["event_id"])
            moved["parent_ids"] = [start["event_id"]]
            with self.subTest(attempt=attempt_id):
                with self.assertRaisesRegex(
                    ValueError, "Attempt closure changed its exact terminal"
                ):
                    validate_attempt_closure(hostile, start, terminal)
                with self.assertRaisesRegex(
                    ValueError,
                    "Investigation Attempt closure changed its exact terminal",
                ):
                    project_investigation(hostile)


if __name__ == "__main__":
    unittest.main()
