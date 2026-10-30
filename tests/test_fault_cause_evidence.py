"""Model endpoint and Tool faults: their causes, their evidence and their holds.

#477: a body-backed 4xx is shown once, an open cause is held and keyed by its
status and body, and every shown fault reopens bound to its own evidence.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import project_journey_events
from bio_neb.failure import ModelContactPolicy, model_contact_verdict, model_fault_key
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import PlanningComposition
from tests.reference_fakes import ReferenceAdapter
from tests.sequential_investigation_support import SourceResponsesOnly, run_attempt


class ModelEndpointCauseTests(unittest.TestCase):
    """Owner, #476 alignment G1.5 B and G4.1, with the orchestrator answers and
    the owner amendment 6483312280 under the grant (#477); Spec r1 SP1-SP3."""

    def test_empty_http_answers_still_have_distinct_status_and_body_keys(self):
        """G4.1: different statuses must open different Reviews, even without words."""
        import asyncio
        import hashlib

        import httpx

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )

        class EmptyAnswers(ScriptedModelEndpoint):
            def __init__(self):
                super().__init__(lambda *_: None)
                self.arrivals = 0
                self.ready = None

            async def handle(self, request):
                if self.ready is None:
                    self.ready = asyncio.Event()
                status = 400 + self.arrivals
                self.arrivals += 1
                if self.arrivals == 2:
                    self.ready.set()
                await self.ready.wait()
                return httpx.Response(status, content=b"")

        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            predecessor_ids=("offline-request",),
            requests=tuple(
                InvestigationRequestItem(request_text=f"Investigate {rsid}", rsid=rsid)
                for rsid in ("rs429358", "rs7412")
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            model = EmptyAnswers()
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                SourceResponsesOnly(),
                assertion=assertion,
                investigation_concurrency={"planner_members": 2, "table_chunks": 4},
            )
            reviews = project_model_availability_review(read_evidence(journey)).requests
            self.assertEqual(model.arrivals, 2)
            self.assertEqual(
                {r.fault_key for r in reviews},
                {
                    model_fault_key(status, hashlib.sha256(b"").hexdigest())
                    for status in (400, 401)
                },
            )
            self.assertTrue(all(r.joins_event_id is None for r in reviews))

    def test_each_body_backed_4xx_is_shown_once(self):
        """Owner G1.5: every 4xx with a body is shown once before a Review."""
        for status in (400, 408, 429):
            with self.subTest(status=status):
                verdict = model_contact_verdict(
                    ModelContactPolicy(),
                    failure="http",
                    contact=1,
                    waited=0,
                    jitter=0,
                    http_status_code=status,
                    rejection_key=model_fault_key(status, "a" * 64),
                )
                self.assertEqual(verdict.policy_disposition, "rejection_shown")

    def test_released_5xx_cause_rejoins_without_another_physical_retry(self):
        """Owner G4.1: a released item re-sends once and rejoins on recurrence."""
        from unittest.mock import patch

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        calls = []

        async def fail(messages, info):
            calls.append(messages)
            raise ModelHTTPError(503, "offline", body={"message": "same outage"})

        source = SourceResponsesOnly()
        model = ScriptedModelEndpoint(fail)
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            predecessor_ids=("offline-request",),
            requests=tuple(
                InvestigationRequestItem(request_text=f"Investigate {rsid}", rsid=rsid)
                for rsid in ("rs429358", "rs7412")
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                source,
                assertion=assertion,
            )
            first, joined = project_model_availability_review(
                read_evidence(journey)
            ).requests
            self.assertTrue(joined.joined_before_sending)
            before = len(calls)
            with patch("bio_neb.application.queries.review_journey"):
                respond_model_availability_review(
                    RespondModelAvailabilityReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=first.request_event_id,
                        response={"action": "retry"},
                    ),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    attempt=investigation_assignment(
                        model_factory=model, source_adapter_factory=lambda: source
                    ),
                )
            self.assertEqual(len(calls) - before, 1)

    def test_body_backed_429_reaches_the_next_request_as_a_failed_turn(self):
        """Owner G1.5: real HTTP path shows the provider body to the next request."""
        calls = []

        async def respond(messages, info):
            calls.append(messages)
            if len(calls) == 1:
                raise ModelHTTPError(
                    429,
                    "offline",
                    body={"message": "Change the supplied request"},
                    headers={"retry-after": "0"},
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele?",
                            "missing_information": "The allele is unspecified.",
                            "asks_about": "target",
                        },
                        tool_call_id=f"question-{len(calls)}",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                SourceResponsesOnly(),
            )
            rejected = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRejected"
            ]
            self.assertEqual(len(rejected), 1)

    def test_released_connection_failure_rejoins_after_one_send(self):
        """Owner amendment 6483312280 (Spec r2 SP4): a released item's connection
        fault on a still-open cause sends once and rejoins."""
        from unittest.mock import patch

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        calls = []

        async def unavailable(messages, info):
            calls.append(messages)
            raise httpx.ConnectError("same unavailable endpoint")

        source = SourceResponsesOnly()
        model = ScriptedModelEndpoint(unavailable)
        assertion = UserAssertion.create_collection(
            asserted_by="researcher",
            predecessor_ids=("offline-request",),
            requests=tuple(
                InvestigationRequestItem(request_text=f"Investigate {rsid}", rsid=rsid)
                for rsid in ("rs429358", "rs7412")
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                source,
                assertion=assertion,
            )
            first, joined = project_model_availability_review(
                read_evidence(journey)
            ).requests
            self.assertTrue(joined.joined_before_sending)
            before = len(calls)
            with patch("bio_neb.application.queries.review_journey"):
                respond_model_availability_review(
                    RespondModelAvailabilityReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=first.request_event_id,
                        response={"action": "retry"},
                    ),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    attempt=investigation_assignment(
                        model_factory=model, source_adapter_factory=lambda: source
                    ),
                )
            self.assertEqual(len(calls) - before, 1)


class RefusalReaderTests(unittest.TestCase):
    """Owner C8 (#477 gate), ADR 0031/0034; Standards r2 ST3/ST4: an undeclared
    fault is shown once per Run, and a refused result names its own Tool."""

    def test_refused_return_names_its_own_tool(self):
        """ADR 0034: a completed call's result cannot name a different Tool."""

        class FailsOnce(ReferenceAdapter):
            failures = 0

            def discover_assemblies(self, ncbi_tax_id, query):
                self.failures += 1
                if self.failures == 1:
                    raise RuntimeError("synthetic offline fault")
                return super().discover_assemblies(ncbi_tax_id, query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = FailsOnce()
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            project_journey_events(events, journey_file=receipt.journey_file)
            by_id = {e["event_id"]: e for e in events}
            (refusal,) = [
                e for e in events if e["event_type"] == "ToolExecutionRefused"
            ]
            failed = by_id[refusal["payload"]["failed_event_id"]]
            start = by_id[failed["payload"]["started_event_id"]]
            call_id = start["payload"]["tool_call_id"]
            hostile = copy.deepcopy(events)
            for request in hostile:
                if request["event_type"] != "ModelRequestRecorded":
                    continue
                for message in request["payload"]["compatible_adapter_evidence"][
                    "request"
                ]["messages"]:
                    for part in message.get("parts", []):
                        if (
                            part.get("part_kind") == "tool-return"
                            and part.get("tool_call_id") == call_id
                        ):
                            part["tool_name"] = "a_different_tool"
            with self.assertRaises(ValueError):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_the_same_undeclared_key_cannot_be_shown_twice(self):
        """Owner C8: the same undeclared Tool-fault key again holds immediately."""

        class FailsTwice(ReferenceAdapter):
            failures = 0

            def discover_assemblies(self, ncbi_tax_id, query):
                self.failures += 1
                if self.failures == 1:
                    raise RuntimeError("synthetic first fault")
                if self.failures == 2:
                    raise ValueError("synthetic distinct fault")
                return super().discover_assemblies(ncbi_tax_id, query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = FailsTwice()
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            project_journey_events(events, journey_file=receipt.journey_file)
            hostile = copy.deepcopy(events)
            by_id = {e["event_id"]: e for e in hostile}
            first, second = [
                e["payload"]
                for e in hostile
                if e["event_type"] == "ToolExecutionRefused"
            ]
            failed = by_id[second["failed_event_id"]]
            call_id = by_id[failed["payload"]["started_event_id"]]["payload"][
                "tool_call_id"
            ]
            failed["payload"]["mechanical_error_type"] = "RuntimeError"
            second["fault_key"] = first["fault_key"]
            second["statement"] = first["statement"]
            for request in hostile:
                if request["event_type"] != "ModelRequestRecorded":
                    continue
                for message in request["payload"]["compatible_adapter_evidence"][
                    "request"
                ]["messages"]:
                    for part in message.get("parts", []):
                        if (
                            part.get("part_kind") == "tool-return"
                            and part.get("tool_call_id") == call_id
                        ):
                            part["content"] = {
                                "status": "refused",
                                "reason": second["reason"],
                                "statement": second["statement"],
                            }
            with self.assertRaises(ValueError):
                project_journey_events(hostile, journey_file=receipt.journey_file)


class ToolRecoveryIntegrationTests(unittest.TestCase):
    def test_a_shown_failure_can_recover_to_a_successful_attempt(self):
        """Owner C8: a shown fault permits the ordinary caller to continue."""
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_composition,
        )
        from tests.application_fakes import _registered
        from tests.planning_assignment_fakes import PlanningComposition
        from tests.reference_fakes import ReferenceAdapter

        class FailsOnce(ReferenceAdapter):
            failed = False

            def discover_assemblies(self, ncbi_tax_id, query):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("synthetic offline adapter fault")
                return super().discover_assemblies(ncbi_tax_id, query)

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.reference_adapter = FailsOnce()
            with use_run_advancement_composition(composition):
                result = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(result.status, "planning_hitl_pending")
