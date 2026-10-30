"""An Investigation member's context is a recorded fact of pointers (#444).

Owner, #444 readiness: Q1 a new investigation Journey event, held and not
rebuilt each turn; Q2 ids into the immutable Journey plus the Tools the member
may and may not use; Q3 the model reads the same bytes as before, and reopen
refuses a request whose context differs from the fact rendered again.
"""

import copy
import tempfile
import unittest
from pathlib import Path

from bio_neb.agentic_journey.role_instructions import role_instructions
from bio_neb.investigation.investigation import (
    project_investigation,
    validate_investigation_fact,
)
from bio_neb.investigation.investigation_commands import BeginInvestigationCommand
from bio_neb.investigation.member_context import (
    InvestigationMemberContextRecorded,
    rendered_text,
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


def _of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


class MemberContextTests(unittest.TestCase):
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

    def first_request(self, fact_event):
        fact = fact_event["payload"]
        return next(
            e
            for e in self.events[self.events.index(fact_event) :]
            if e["event_type"] == "ModelRequestRecorded"
            and e["payload"]["attempt_id"] == fact["attempt_id"]
            and e["payload"]["member_id"] == fact["collection_member_id"]
        )

    def test_each_member_records_its_context_once_and_its_request_carries_it(self):
        facts = _of(self.events, "InvestigationMemberContextRecorded")
        self.assertEqual(len(facts), 2)
        by_id = {e["event_id"]: e for e in self.events}
        for event in facts:
            fact = validate_investigation_fact(event)
            self.assertIsInstance(fact, InvestigationMemberContextRecorded)
            with self.subTest(member=fact.collection_member_id):
                # Pointers, not copies: the Investigation, the bound work, the
                # member's own results and answers by id.
                self.assertEqual(
                    by_id[fact.started_event_id]["event_type"], "InvestigationStarted"
                )
                self.assertEqual(
                    by_id[fact.bound_event_id]["event_type"],
                    "InvestigationAttemptBound",
                )
                self.assertNotIn("user_assertion", event["payload"])
                started = next(
                    e
                    for e in self.events
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"]["attempt_id"] == fact.attempt_id
                )
                role, version, form = started["payload"]["instruction_identity"].split(
                    ":"
                )
                text = rendered_text(
                    fact,
                    by_id,
                    role_instructions(role, version=version, form=form),
                    validate_investigation_fact,
                )
                parts = self.first_request(event)["payload"][
                    "compatible_adapter_evidence"
                ]["request"]["messages"][-1]["parts"]
                self.assertEqual(
                    [p["content"] for p in parts if p["content"] == text], [text]
                )

    def test_reopen_refuses_a_context_that_differs_from_its_fact(self):
        (fact, _second) = _of(self.events, "InvestigationMemberContextRecorded")
        project_investigation(self.events)

        def prompt_changed(hostile):
            original = self.first_request(fact)
            request = next(e for e in hostile if e["event_id"] == original["event_id"])
            part = next(
                p
                for p in request["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ][-1]["parts"]
                if p["part_kind"] == "user-prompt"
            )
            part["content"] = part["content"].replace("fetch_assembly_report", "x")

        def tools_changed(hostile):
            changed = next(e for e in hostile if e["event_id"] == fact["event_id"])
            changed["payload"]["forbidden_operations"] = []

        def unknown_attempt(hostile):
            changed = next(e for e in hostile if e["event_id"] == fact["event_id"])
            changed["payload"]["attempt_id"] = "00000000-0000-4000-8000-000000000444"

        def later_bound(hostile):
            # Another Attempt's bound work, recorded later.
            changed = next(e for e in hostile if e["event_id"] == fact["event_id"])
            (_first, second) = [
                e for e in hostile if e["event_type"] == "InvestigationAttemptBound"
            ]
            changed["payload"]["bound_event_id"] = second["event_id"]

        for change, reason in (
            (unknown_attempt, "member context names no bound Investigation Attempt"),
            (later_bound, "member context points outside its Attempt's basis"),
        ):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(self.events)
                change(hostile)
                with self.assertRaisesRegex(ValueError, reason):
                    project_investigation(hostile)

        for change in (prompt_changed, tools_changed):
            with self.subTest(change=change.__name__):
                hostile = copy.deepcopy(self.events)
                change(hostile)
                with self.assertRaisesRegex(ValueError, "member context"):
                    project_investigation(hostile)


if __name__ == "__main__":
    unittest.main()


class UnrenderableContextTests(unittest.TestCase):
    def test_a_context_that_cannot_render_holds_the_run_before_any_request(self):
        # A pointer that does not resolve is a typed hold under its own name,
        # never coordinator_unhandled_error.
        from unittest.mock import patch

        from bio_neb.application.advancement import RunAdvancementRefused

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            document = collection_request(1)
            journey = forked_child(root, investigation_request=document)
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(document), encoding="utf-8")
            with (
                patch(
                    "bio_neb.investigation.investigation_assignment."
                    "latest_source_results",
                    return_value={
                        "RefSnpToolResultRecorded": {"event_id": "missing-source"}
                    },
                ),
                self.assertRaises(RunAdvancementRefused) as held,
            ):
                begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    ),
                    environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                    model_factory=SequentialAdaptiveModel().model,
                    source_adapter_factory=SourceResponsesOnly,
                )
            events = read_evidence(Journey.open(journey.path))

        self.assertEqual(held.exception.code, "member_context_unrenderable")
        self.assertEqual(_of(events, "InvestigationMemberContextRecorded"), [])
        planner = {
            e["payload"]["attempt_id"]
            for e in _of(events, "AttemptStarted")
            if e["payload"]["agent_role"] == "investigation_planner"
        }
        self.assertTrue(planner)
        self.assertFalse(
            any(
                e["payload"]["attempt_id"] in planner
                for e in _of(events, "ModelRequestRecorded")
            )
        )


class MissingContextTests(unittest.TestCase):
    """Every member's requests hold a recorded context, even a member that makes
    no source call and stops at the Agent work Review (#444 Spec r2 SP4)."""

    def run_member(self, directory, **patches):
        from types import SimpleNamespace

        from pydantic_ai.messages import ModelResponse, TextPart

        from tests.model_endpoint_fakes import ScriptedModelEndpoint
        from tests.sequential_investigation_support import run_attempt

        endpoint = ScriptedModelEndpoint(
            lambda messages, info: ModelResponse(parts=[TextPart("I have no call.")])
        )
        projection, journey = run_attempt(
            Path(directory), SimpleNamespace(model=endpoint), SourceResponsesOnly()
        )
        self.assertEqual(projection.status, "agent_work_review_pending")
        return read_evidence(Journey.open(journey.path))

    def test_a_producer_that_omits_the_context_is_refused(self):
        from unittest.mock import patch

        from bio_neb.agentic_journey.lifecycle import Activity

        record = Activity.record

        def omit(activity, fact):
            if fact.event_type == "InvestigationMemberContextRecorded":
                return None
            return record(activity, fact)

        # The Attempt's own reading of its Journey refuses it as it closes.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(Activity, "record", omit),
            self.assertRaisesRegex(ValueError, "no recorded member context"),
        ):
            self.run_member(directory)

    def test_a_stopped_member_whose_context_is_removed_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            events = self.run_member(directory)
        project_investigation(events)
        hostile = copy.deepcopy(events)
        (fact,) = _of(hostile, "InvestigationMemberContextRecorded")
        hostile.remove(fact)
        for event in hostile:
            event["parent_ids"] = [
                ancestor
                for parent in event["parent_ids"]
                for ancestor in (
                    fact["parent_ids"] if parent == fact["event_id"] else [parent]
                )
            ]
        with self.assertRaisesRegex(ValueError, "no recorded member context"):
            project_investigation(hostile)
