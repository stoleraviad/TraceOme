from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.usage import RequestUsage

from bio_neb.agentic_journey.agent_roles import role_instructions
from bio_neb.agentic_journey.agent_work_review import (
    ContinueAgentWorkResponse,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.attempt_journey import (
    ModelAvailabilityRetryDelayCancelled,
    ModelAvailabilityRetryScheduled,
    ModelPhysicalContactCancelled,
    ModelPhysicalContactFailed,
    ModelPhysicalContactStarted,
    ModelRequestRecorded,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationOverrides,
    ContinuationPolicy,
)
from bio_neb.agentic_journey.effort import (
    ExtendAgentEffortResponse,
    respond_effort_review,
)
from bio_neb.agentic_journey.instruction_evidence import INSTRUCTION_MEDIA_TYPE
from bio_neb.agentic_journey.lifecycle import ActivityLifecycle
from bio_neb.agentic_journey.model_availability import (
    RetryModelAvailabilityResponse,
    record_model_attempt_cancelled,
)
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.commands import (
    RespondModelAvailabilityReviewCommand,
    respond_model_availability_review,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.devtools.agent_inspection import LocalInspection
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.journey import Journey
from bio_neb.providers.nebius_model_endpoint import NebiusModelEndpointAdapter
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint, corrections
from tests.planning_assignment_fakes import PlanningComposition
from tests.reference_query_fakes import registered_attempt_start


# The B023 suppressions below mark pre-existing loop-variable captures: every flagged
# closure is built and consumed inside the same iteration, so none of them is a live defect,
# and binding 10 of them by default argument would change signatures that the scripted model
# and the review fakes inspect.
def _count(count, limit=4):
    """The count line every counted correction ends in (#478 G1.4)."""
    return (
        f"\nCount: failed turn {count} of {limit} in a row. At {limit} the Attempt "
        "stops and the researcher reviews it."
    )


def _availability_failure_model():
    # A 5xx keeps the contact window: a 4xx with words is shown to the model
    # once instead (Spec r1 SP1).
    async def respond(messages, info):
        del messages, info
        raise ModelHTTPError(
            503,
            "availability-test",
            body={"message": "the provider is unavailable"},
            headers={},
        )

    return ScriptedModelEndpoint(respond)


def _availability_frontier(
    root: Path,
    *,
    agent_invocation=None,
    failure=None,
    model: str = "Qwen/Qwen3.5-397B-A17B",
):
    receipt, _ = _registered(
        root,
        model=model,
        agent_invocation=agent_invocation,
        failure=failure,
    )
    journey = Journey.open(Path(receipt.journey_file))
    prepare_input_correction(journey)
    return receipt, journey


class AgentReentryResolverTests(unittest.TestCase):
    """The coordinator's resolver keys re-entry on the stopped Attempt's record."""

    @staticmethod
    def _started(attempt_id, agent_role):
        return {
            "event_type": "AttemptStarted",
            "payload": {"attempt_id": attempt_id, "agent_role": agent_role},
        }

    def test_recorded_assignment_selects_the_coordinator_work_carrying_it(self):
        from bio_neb.application.agent_reentry import (
            coordinator_assignments,
            reentry_assignment,
        )

        for recorded in ("planning_agent", "investigation_planner"):
            with self.subTest(recorded=recorded):
                events = [self._started("stopped", recorded)]
                selected = reentry_assignment(
                    events, "stopped", coordinator_assignments()
                )
                self.assertEqual(selected.agent_role, recorded)
                self.assertEqual(
                    selected.activity_stage,
                    {
                        "planning_agent": "planning_attempt",
                        "investigation_planner": "investigation_planning_attempt",
                    }[recorded],
                )

    def test_unknown_or_inexact_recorded_assignment_is_refused(self):
        from bio_neb.application.agent_reentry import (
            coordinator_assignments,
            reentry_assignment,
        )

        cases = {
            "no Agent work re-enters the recorded assignment 'execution_planning_agent'": [
                self._started("stopped", "execution_planning_agent")
            ],
            "does not record one exact assignment": [
                self._started("stopped", "planning_agent"),
                self._started("stopped", "planning_agent"),
            ],
            "does not record one exact assignment$": [
                self._started("another", "planning_agent")
            ],
        }
        for message, events in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                reentry_assignment(events, "stopped", coordinator_assignments())


def _with_stop_cause(events, cause):
    """The events with the Agent work stop's cause and its Attempt's code forged."""
    changed = copy.deepcopy(events)
    for event in changed:
        if event["event_type"] == "AgentWorkReviewRequested":
            event["payload"]["cause"] = cause
        elif event["event_type"] == "AttemptStopped":
            event["payload"]["code"] = cause
    return changed


class AgentApplicationRecoveryIntegrationTests(unittest.TestCase):
    def test_public_retry_records_received_messages_and_instruction_parts(self):
        received = []

        async def unavailable(messages, info):
            raise ModelHTTPError(401, "offline")

        class ReceivingModel(NebiusModelEndpointAdapter):
            async def request(self, messages, model_settings, model_request_parameters):
                _, prepared = self.prepare_request(
                    model_settings, model_request_parameters
                )
                received.append(
                    {
                        "messages": ModelMessagesTypeAdapter.dump_python(
                            messages, mode="json"
                        ),
                        "instruction_parts": [
                            asdict(part) for part in prepared.instruction_parts
                        ],
                    }
                )
                return await super().request(
                    messages, model_settings, model_request_parameters
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())

            def attempt():
                return planning_assignment(
                    model_factory=ScriptedModelEndpoint(
                        unavailable, adapter=ReceivingModel
                    )
                )

            pending = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            request = pending.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    response={"action": "retry"},
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=attempt(),
            )
            reviewed = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(reviewed.status, "model_availability_review_pending")
            # Each Attempt shows the refusal once, then asks; the retry
            # resends the exact pending request.
            self.assertEqual(len(received), 4)
            self.assertEqual(received[1], received[2])
            self.assertEqual(len(received[0]["instruction_parts"]), 1)
            self.assertIs(received[0]["instruction_parts"][0]["dynamic"], True)
            requests = [
                event["payload"]["compatible_adapter_evidence"]["request"]
                for event in read_evidence(Journey.open(Path(receipt.journey_file)))
                if event["event_type"] == "ModelRequestRecorded"
            ]
            self.assertEqual(len(requests), 4)
            for sent, recorded in zip(received, requests, strict=True):
                self.assertEqual(recorded["messages"], sent["messages"])
                self.assertEqual(
                    recorded.get("instruction_parts"), sent["instruction_parts"]
                )

    def setUp(self):
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    def test_rejected_model_request_stops_at_finalization_without_run_failure(self):
        from bio_neb.application.advancement import (
            AnswerReview,
            ModelAvailabilityReviewAnswer,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())

            async def unavailable(messages, info):
                raise ModelHTTPError(402, "unavailable")

            pending = planning_assignment(
                model_factory=ScriptedModelEndpoint(unavailable)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            request = pending.model_availability_review.pending_request
            self.assertEqual(request.exhaustion_reason, "request_rejected")
            self.assertEqual(len(request.contacts), 1)
            self.assertEqual(
                {a.action for a in pending.current_review.request.legal_actions},
                {"retry", "amend", "stop"},
            )
            ended = advance(
                AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    answer=ModelAvailabilityReviewAnswer(
                        kind="model_availability",
                        response={"action": "stop", "reason": "Pause this work."},
                    ),
                )
            )
            self.assertEqual(ended.status, "finalization_review_pending")
            self.assertEqual(
                ended.finalization.pending_request.basis.outcome,
                "model_availability_stopped",
            )
            self.assertFalse(
                any(
                    e["event_type"] == "RunFailed"
                    for e in read_evidence(Journey.open(journey.path))
                )
            )
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                ended,
            )

    def test_model_amendment_starts_fresh_instruction_without_changing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            calls = []

            async def unavailable(messages, info):
                calls.append((messages, info.instructions))
                raise ModelHTTPError(402, "unavailable")

            attempt = planning_assignment(
                model_factory=ScriptedModelEndpoint(unavailable)
            )
            pending = attempt.run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            before = journey.path.read_bytes()
            with self.assertRaises(ValueError):
                attempt.run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(journey.path.read_bytes(), before)
            response = respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=pending.model_availability_review.pending_request.request_event_id,
                    response={
                        "action": "amend",
                        "instruction": "Inspect the registered input files first.",
                    },
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=attempt,
            )
            # Each Attempt shows the refusal once, then asks.
            self.assertEqual(len(calls), 4)
            self.assertIn(
                "Inspect the registered input files first.",
                ModelMessagesTypeAdapter.dump_json(calls[-1][0]).decode(),
            )
            events = read_evidence(Journey.open(journey.path))
            starts = [
                e["payload"] for e in events if e["event_type"] == "AttemptStarted"
            ]
            self.assertIsNone(starts[-1]["continuation_request_event_id"])
            self.assertEqual(
                starts[-1]["continuation_availability_review_event_id"],
                response.response_event_id,
            )
            self.assertEqual(
                starts[-1]["effective_invocation_policy"],
                starts[0]["effective_invocation_policy"],
            )
            requests = [
                e["payload"]
                for e in events
                if e["event_type"] == "ModelRequestRecorded"
            ]
            self.assertEqual([r["request_number"] for r in requests], [1, 2, 1, 2])
            self.assertIsNone(requests[-1]["continuation_request_event_id"])
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))

        # Amendment never silently admits a policy/settings edit.
        with self.assertRaises(ValueError):
            RespondModelAvailabilityReviewCommand(
                journey_file="/not-opened",
                request_event_id=str(uuid.uuid4()),
                response={
                    "action": "amend",
                    "instruction": "Use another endpoint.",
                    "model": "another",
                },
            )

    def test_provider_retry_preserves_multi_call_correction_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            calls = 0

            async def mixed_then_outage(messages, info):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})
                # The unoffered call is refused on its own and the search runs
                # once; its later identical calls are repeats with nothing changed.
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "search_journey",
                            {"kinds": ["ResearchRequestRegistered"]},
                            tool_call_id=f"search-{calls}",
                        ),
                        ToolCallPart(
                            "resolve_taxon",
                            {"term": "Homo sapiens"},
                            tool_call_id=f"discover-{calls}",
                        ),
                    ]
                )

            def attempt():
                return planning_assignment(
                    model_factory=ScriptedModelEndpoint(mixed_then_outage)
                )

            pending = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            request = pending.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    response={"action": "retry"},
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=attempt(),
            )
            reviewed = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            # The provider continuation keeps counting the failed turns: the
            # first reply ran the search, so the 4th repeat in a row (the 4th
            # failed turn) stops the Attempt for the Agent work review.
            self.assertEqual(reviewed.status, "agent_work_review_pending")
            self.assertEqual(calls, 6)
            events = Journey.open(Path(receipt.journey_file)).read()
            turns = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            self.assertEqual(
                [(t["disposition"], t.get("correction_count")) for t in turns],
                [
                    ("multi_call_admitted", None),
                    ("multi_call_refused", 1),
                    ("multi_call_refused", 2),
                    ("multi_call_refused", 3),
                    ("multi_call_refused", 4),
                ],
            )
            self.assertEqual(
                [
                    fault["count"]
                    for e in events
                    if e["event_type"] == "AgentTurnRouted"
                    for fault in e["payload"]["faults"]
                    if fault["key"] == "unoffered_calls"
                ],
                [1, 1, 2, 3, 4],
            )
            (stop,) = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentWorkReviewRequested"
            ]
            self.assertEqual(stop["cause"], "failed_turns")
            self.assertFalse(any(e["event_type"] == "RunFailed" for e in events))

    def test_provider_retry_preserves_completed_reference_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            inspect_registered_inputs(
                InspectRegisteredInputsCommand(journey_file=str(journey.path)),
                journey=journey,
            )
            composition = PlanningComposition(receipt, paths)
            adapter = composition.reference_adapter
            respond = composition.respond
            calls = 0

            async def progressing_then_outage(messages, info):
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})
                return await respond(messages, info)

            with patch.object(composition, "respond", progressing_then_outage):
                pending = composition.planning_assignment(journey).run(
                    journey, environ=composition.environ
                )
            self.assertEqual(
                composition.calls,
                [
                    "search_journey",
                    "record_proposal_requirements",
                    "attest_sample",
                    "resolve_taxon",
                    "discover_assemblies",
                ],
            )
            self.assertEqual(adapter.contact_count, 2)
            request = pending.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    response={"action": "retry"},
                ),
                environ=composition.environ,
                attempt=composition.planning_assignment(
                    Journey.open(Path(receipt.journey_file))
                ),
            )
            result = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertIsNone(result.model_availability_review.pending_request)
            self.assertIsNotNone(result.reference_identity.verified_reference)
            self.assertEqual(adapter.contact_count, 3)
            requests = [
                e
                for e in Journey.open(Path(receipt.journey_file)).read()
                if e["event_type"] == "ModelRequestRecorded"
            ]
            resumed = next(
                e for e in requests if e["payload"].get("continuation_request_event_id")
            )
            original = next(
                e
                for e in requests
                if e["event_id"] == resumed["payload"]["continuation_request_event_id"]
            )
            self.assertEqual(
                resumed["payload"]["compatible_adapter_evidence"]["request"],
                original["payload"]["compatible_adapter_evidence"]["request"],
            )

    def test_a_full_default_window_leaves_application_pending_with_exact_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            with patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0,
            ):
                result = planning_assignment(
                    model_factory=_availability_failure_model()
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            self.assertEqual(result.status, "model_availability_review_pending")
            pending = result.model_availability_review.pending_request
            # The default window: two sends.
            self.assertEqual(len(pending.contacts), 2)
            self.assertEqual(len(pending.delays), 1)
            self.assertEqual(pending.contact_policy.contacts, 2)
            self.assertEqual(pending.contact_policy.max_wait_seconds, 60)
            self.assertFalse(
                any(e["event_type"] == "RunFailed" for e in journey.read())
            )

    def test_a_role_model_contact_entry_sets_only_that_roles_window(self):
        # Model contact is the failure module's per-role tuning.
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.runtime_config import RuntimeSelections
        from bio_neb.failure import ModelContactPolicy

        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(),
                failure={
                    "scopes": {
                        "role": {"planning_agent": {"model_contact": {"contacts": 3}}}
                    }
                },
            )
            with patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0,
            ):
                result = planning_assignment(
                    model_factory=_availability_failure_model()
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            self.assertEqual(result.status, "model_availability_review_pending")
            pending = result.model_availability_review.pending_request
            self.assertEqual(len(pending.contacts), 3)
            self.assertEqual(pending.contact_policy.contacts, 3)
            self.assertEqual(pending.contact_policy.max_wait_seconds, 60)
            events = journey.read()
            (started,) = [e for e in events if e["event_type"] == "AttemptStarted"]
            self.assertEqual(
                started["payload"]["effective_invocation_policy"][
                    "model_contact_sources"
                ]["contacts"],
                "role",
            )
            # Each request binds to its Attempt's resolved window on reading.
            (request,) = [
                e for e in events if e["event_type"] == "ModelRequestRecorded"
            ]
            self.assertEqual(request["payload"]["contact_policy"]["contacts"], 3)
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(review.status, "model_availability_review_pending")
            # The other role keeps the default window.
            (registration,) = [
                e for e in events if e["event_type"] == "RunInputsRegistered"
            ]
            selections = RuntimeSelections.model_validate(
                registration["payload"]["runtime_selections"]
            )
            self.assertEqual(
                selections.agent_invocation_policy(
                    "investigation_planner", AgentInvocationOverrides()
                ).model_contact,
                ModelContactPolicy(),
            )

    def test_an_availability_retry_without_its_credential_holds_on_each_resume(
        self,
    ):
        # The retry's configuration failure holds the Run; each resume with the
        # credential still missing retains one more, and the Run reads held.
        from bio_neb.application.advancement import (
            AnswerReview,
            ModelAvailabilityReviewAnswer,
            RunAdvancementRefused,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())

            def unavailable(messages, info):
                # A 5xx over its Retry-After cap asks at once; a 4xx with words
                # would be shown to the model first.
                raise ModelHTTPError(
                    503, "policy-proof", headers={"retry-after": "999"}
                )

            first = planning_assignment(
                model_factory=ScriptedModelEndpoint(unavailable)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            request = first.model_availability_review.pending_request
            composition = PlanningComposition(receipt, {})
            composition.environ = {}
            # The researcher answers retry through the coordinator, as the CLI does.
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as answered,
            ):
                advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=ModelAvailabilityReviewAnswer(
                            kind="model_availability",
                            response=RetryModelAvailabilityResponse(action="retry"),
                        ),
                    )
                )
            self.assertEqual(
                (answered.exception.code, answered.exception.status),
                ("model_credential_unavailable", "run_held"),
            )
            with (
                use_run_advancement_composition(composition),
                self.assertRaises(RunAdvancementRefused) as refused,
            ):
                advance(ResumeRun(journey_file=receipt.journey_file))
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            types = [
                e["event_type"]
                for e in read_evidence(Journey.open(Path(receipt.journey_file)))
            ]
        self.assertEqual(
            (refused.exception.code, refused.exception.status, reopened.status),
            ("model_credential_unavailable", "run_held", "run_held"),
        )
        self.assertEqual(types.count("RuntimeConfigurationFailed"), 2)
        self.assertEqual(types.count("AttemptStarted"), 1)
        self.assertNotIn("RunFailed", types)

    def test_provider_retry_preserves_each_roles_exact_invocation_policy(
        self,
    ):
        role = "planning_agent"
        with tempfile.TemporaryDirectory() as directory:
            # The omitted ceiling is the role's configured setting; the
            # Attempt's own recorded amendment is its failed turns, which
            # replaces the configured number.
            receipt, journey = _availability_frontier(
                Path(directory).resolve(),
                agent_invocation={
                    "failed_turns": 1,
                    "scopes": {
                        "role": {role: {"model_settings": {"max_tokens": None}}}
                    },
                },
            )

            def unavailable(messages, info):
                raise ModelHTTPError(
                    503, "policy-proof", headers={"retry-after": "999"}
                )

            first = planning_assignment(
                model_factory=ScriptedModelEndpoint(unavailable)
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                invocation_overrides=AgentInvocationOverrides(failed_turns=2),
            )
            self.assertEqual(first.status, "model_availability_review_pending")
            request = first.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    response=RetryModelAvailabilityResponse(action="retry"),
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=planning_assignment(
                    model_factory=ScriptedModelEndpoint(unavailable)
                ),
            )
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            effective = reopened.attempt.effective_invocation_policy
            self.assertEqual(effective.agent_role, role)
            self.assertEqual(effective.continuation, ContinuationPolicy(failed_turns=2))
            self.assertEqual(
                effective.model_settings.max_tokens,
                None,
            )
            self.assertEqual(
                reopened.attempt.invocation_overrides,
                first.attempt.invocation_overrides,
            )
            self.assertEqual(
                next(
                    a
                    for a in reopened.attempts
                    if a.attempt_id == first.attempt.attempt_id
                ),
                first.attempt,
            )
            self.assertEqual(
                first.attempt.invocation_policy_id,
                reopened.attempt.invocation_policy_id,
            )
            for attempt in (first.attempt, reopened.attempt):
                requests = [
                    e
                    for e in attempt.model_events
                    if e.event_type == "ModelRequestRecorded"
                ]
                self.assertEqual(len(requests), 1)
                # The policy's settings, plus the Run's one recorded Tool
                # choice.
                self.assertEqual(
                    requests[0].payload["compatible_adapter_evidence"]["request"][
                        "model_settings"
                    ],
                    {
                        **attempt.effective_invocation_policy.model_settings.request_settings(),
                        "tool_choice": "auto",
                    },
                )

    def test_public_role_cancellation_preserves_contact_and_disposable_independence(
        self,
    ):
        role = "planning_agent"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, journey = _availability_frontier(root)
            first_event = len(read_evidence(journey))
            calls = 0

            async def cancel(messages, info):
                nonlocal calls
                del messages, info
                calls += 1
                raise asyncio.CancelledError()

            path = root / "cancelled-spans.jsonl"
            with (
                LocalInspection(path) as capture,
                self.assertRaises(asyncio.CancelledError),
            ):
                planning_assignment(model_factory=ScriptedModelEndpoint(cancel)).run(
                    journey, environ={"LOCAL_MODEL_KEY": "secret"}
                )
            self.assertEqual(capture.errors, [])
            self.assertEqual(calls, 1)
            events = read_evidence(journey)[first_event:]
            for event_type in (
                "ModelRequestRecorded",
                "ModelPhysicalContactStarted",
                "ModelPhysicalContactCancelled",
                "AttemptCancelled",
            ):
                self.assertEqual(
                    sum(event["event_type"] == event_type for event in events), 1
                )
            self.assertFalse(
                any(
                    event["event_type"]
                    in {
                        "ModelResponseRecorded",
                        "ObservedToolExecutionStarted",
                        "AttemptFailed",
                        "AttemptSucceeded",
                    }
                    for event in events
                )
            )
            result = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(
                (result.attempt.agent_role, result.attempt.outcome),
                (role, "cancelled"),
            )
            retained = journey.path.read_bytes()
            path.unlink()
            del capture
            self.assertEqual(
                review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file)),
                result,
            )
            self.assertEqual(journey.path.read_bytes(), retained)

    def test_companion_operational_failure_reviews_before_another_model_request(self):
        from tests.reference_fakes import AssemblyReportFake

        class FailingCompanionAdapter(AssemblyReportFake):
            contact_count = 0
            contacts = 0

            def describe_companions(self, assembly_accession, role):
                del assembly_accession, role
                self.contacts += 1
                raise ReferenceAdapterError(
                    "http_error",
                    "Companion source unavailable",
                    retryable=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            receipt, paths = _registered(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            inspect_registered_inputs(
                InspectRegisteredInputsCommand(journey_file=str(journey.path)),
                journey=journey,
            )
            composition = PlanningComposition(receipt, paths)
            adapter = FailingCompanionAdapter()
            composition.companion_adapter = adapter
            result = composition.planning_assignment(journey).run(
                journey, environ=composition.environ
            )
            self.assertEqual(result.status, "operational_review_pending")
            # The failed companion contact is the Attempt's last model request.
            self.assertEqual(
                composition.calls,
                [
                    "search_journey",
                    "record_proposal_requirements",
                    "attest_sample",
                    "resolve_taxon",
                    "discover_assemblies",
                    "record_reference_choice",
                    "inspect_reference",
                    "describe_companion",
                ],
            )
            self.assertEqual(adapter.contacts, 1)
            retained = journey.path.read_bytes()
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(reopened, result)
            self.assertEqual(journey.path.read_bytes(), retained)
            attempt = reopened.attempts[-1]
            self.assertEqual(attempt.outcome, "stopped")
            self.assertEqual(
                [item.payload["checkpoint"] for item in attempt.checkpoint_events],
                [],
            )

            from bio_neb.application.queries import ObserveRun, observe

            with use_run_advancement_composition(composition):
                ended = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(ended.status, "operational_review_pending")
            self.assertEqual(ended.current_review, result.current_review)
            self.assertEqual(adapter.contacts, 1)
            self.assertEqual(
                observe(ObserveRun(journey_file=receipt.journey_file)), ended
            )

    def test_truncation_on_the_last_permitted_request_reaches_the_effort_review(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(), agent_invocation={"model_requests": 1}
            )
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            seen = []

            def respond(messages, info):
                seen.append(messages)
                if len(seen) == 1:
                    return ModelResponse(
                        parts=[TextPart("looping")],
                        usage=RequestUsage(input_tokens=10, output_tokens=32768),
                        finish_reason="length",
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="after-extension",
                        )
                    ]
                )

            def attempt():
                return planning_assignment(model_factory=ScriptedModelEndpoint(respond))

            # The retry is due, but the request allowance is spent: the unchanged
            # effort review holds the pending retry and the stopped work.
            stopped = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual((stopped.status, len(seen)), ("effort_review_pending", 1))
            (truncation,) = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "ModelCompletionTruncated"
            ]
            feedback = truncation["payload"]["feedback"]
            self.assertIsNotNone(feedback)
            request = stopped.agent_effort_review.pending_request
            self.assertEqual(request.requests_used, 1)
            self.assertEqual(
                [part["part_kind"] for part in request.messages[-1]["parts"]],
                ["retry-prompt"],
            )
            self.assertEqual(request.messages[-1]["parts"][0]["content"], feedback)
            self.assertNotIn("looping", json.dumps(request.messages))
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                "effort_review_pending",
            )
            accepted = respond_effort_review(
                journey,
                request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=1),
            )
            resumed = attempt().run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                effort_review_response_event_id=accepted.response_event_id,
            )
            # The extension sends the retry: the correction, never the cut-off reply.
            self.assertEqual((resumed.status, len(seen)), ("inspection_succeeded", 2))
            # The framework sends the retry in the cut-off request's turn.
            self.assertIn(
                ("retry-prompt", feedback),
                [
                    (part.part_kind, getattr(part, "content", None))
                    for part in seen[1][-1].parts
                ],
            )
            self.assertNotIn(
                "looping",
                ModelMessagesTypeAdapter.dump_json(seen[1]).decode(),
            )

    def test_text_only_response_is_corrected_instead_of_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            seen = []

            def respond(messages, info):
                seen.append((messages, info))
                if len(seen) == 1:
                    return ModelResponse(parts=[TextPart("Finished.")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="after-correction",
                        )
                    ]
                )

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code, len(seen)),
                ("inspection_succeeded", None, 2),
            )
            offered = sorted(tool.name for tool in seen[0][1].function_tools)
            correction = (
                "Your reply ended in text without a tool call; text is not a result, "
                "so nothing happened.\n"
                "A step happens only through a tool call: make the call that does "
                "it.\nAllowed form: a tool call to one of "
                + ", ".join(f"`{name}`" for name in offered)
                + "."
                + _count(1)
            )
            retry = seen[1][0][-1].parts[-1]
            self.assertEqual(
                (retry.part_kind, retry.content), ("retry-prompt", correction)
            )
            events = read_evidence(journey)
            (turn,) = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "text_refused"
            ]
            self.assertEqual(
                (turn["correction_key"], turn["correction_count"], turn["feedback"]),
                ("text_ending", 1, correction),
            )
            self.assertEqual([item.outcome for item in result.attempts], ["succeeded"])
            self.assertFalse(
                any(
                    e["event_type"]
                    in {"AgentWorkReviewRequested", "AttemptFailed", "RunFailed"}
                    for e in events
                )
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )

    def test_a_reasoning_only_reply_is_corrected_under_its_own_key(self):
        # A reply with reasoning and no call and no text is not a text
        # ending; its correction says the call did not arrive.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            seen = []

            def respond(messages, info):
                seen.append((messages, info))
                if len(seen) == 1:
                    return ModelResponse(
                        parts=[ThinkingPart("I have what I need; now the call.")],
                        finish_reason="tool_call",
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="after-correction",
                        )
                    ]
                )

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code, len(seen)),
                ("inspection_succeeded", None, 2),
            )
            offered = sorted(tool.name for tool in seen[0][1].function_tools)
            correction = (
                "Your reply held only reasoning; no tool call arrived, so nothing "
                "happened.\nAllowed form: a tool call to one of "
                + ", ".join(f"`{name}`" for name in offered)
                + "."
                + _count(1)
            )
            retry = seen[1][0][-1].parts[-1]
            self.assertEqual(
                (retry.part_kind, retry.content), ("retry-prompt", correction)
            )
            # The model's next request shows it its reasoning-only reply.
            self.assertEqual(
                [part.part_kind for part in seen[1][0][-2].parts], ["thinking"]
            )
            events = read_evidence(journey)
            (turn,) = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] != "singleton_admitted"
            ]
            self.assertEqual(
                (
                    turn["disposition"],
                    turn["correction_key"],
                    turn["correction_count"],
                    turn["feedback"],
                ),
                ("reasoning_refused", "reasoning_only", 1, correction),
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                result.status,
            )
            # Reopen re-reads the reply: a text ending's disposition or key on a
            # reasoning-only reply, or the reverse key, is refused.
            for disposition, key in (
                ("text_refused", "text_ending"),
                ("text_refused", "reasoning_only"),
                ("reasoning_refused", "text_ending"),
            ):
                changed = copy.deepcopy(events)
                payload = next(
                    e["payload"]
                    for e in changed
                    if e["event_type"] == "AgentActionTurnDispositionRecorded"
                    and e["payload"]["disposition"] == "reasoning_refused"
                )
                payload.update(disposition=disposition, correction_key=key)
                with (
                    self.subTest(disposition=disposition, key=key),
                    self.assertRaisesRegex(ValueError, "refused turn|correction key"),
                ):
                    project_journey_events(changed, journey_file=receipt.journey_file)
            # A Journey recorded before response bodies were kept read the same
            # reply as a text ending; it still reopens.
            dropped = {
                e["event_id"]: e["parent_ids"]
                for e in events
                if e["event_type"] == "ModelTransportResponseRecorded"
            }
            older = copy.deepcopy([e for e in events if e["event_id"] not in dropped])
            for event in older:
                event["parent_ids"] = [
                    parent
                    for item in event["parent_ids"]
                    for parent in dropped.get(item, [item])
                ]
                event["payload"].pop("response_body_expected", None)
                if event["payload"].get("disposition") == "reasoning_refused":
                    event["payload"].update(
                        disposition="text_refused", correction_key="text_ending"
                    )
            project_journey_events(older, journey_file=receipt.journey_file)

    def test_every_correction_is_written_in_the_selected_models_wording(self):
        # The failure module's retry rendering writes every correction the loop
        # sends, in the wording of the selected model's endpoint policy.
        from dataclasses import replace

        from bio_neb.agentic_journey import model_settings
        from bio_neb.failure import CorrectionWording

        model = "Qwen/Qwen3.5-397B-A17B"
        wording = CorrectionWording(
            refused="REFUSED[{codes}] ",
            not_run="NOT-RUN[{tool}] {details}OFFERED[{offered}]{guidance}",
            arguments_not_object="NOT-OBJECT. ",
            truncation="CUT[{output_tokens}] OFFERED[{offered}]",
            text_ending="TEXT OFFERED[{offered}]",
            fault="FAULT[{kind} {count}/{limit}]",
            failed_turn="FAULT[turn {count}/{limit}]",
            cut_off_no_call="{correction} {fault}",
            last_correction="LAST[{kind}]",
        )
        selected = replace(
            model_settings.model_endpoint_policy(model), correction_wording=wording
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            script = [
                ModelResponse(parts=[TextPart("Finished.")]),
                ModelResponse(
                    parts=[TextPart("a reply that ran out")],
                    usage=RequestUsage(output_tokens=7),
                    finish_reason="length",
                ),
                ModelResponse(
                    parts=[ToolCallPart("inspect_input_metadata", "{broken", "bad")]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="good",
                        )
                    ]
                ),
            ]
            received, offered = [], []

            def respond(messages, info):
                # The newest correction this request carries, a retry part or a
                # refused call's failed result in its slot; a cut-off reply is
                # not kept, so its request also carries the earlier correction.
                received.append(corrections(messages[-1])[-1:])
                offered.append(
                    ", ".join(f"`{tool.name}`" for tool in info.function_tools)
                )
                return script[len(received) - 1]

            with model_settings.use_model_registry(
                model_settings.model_registry().replacing(selected)
            ):
                result = planning_assignment(
                    model_factory=ScriptedModelEndpoint(respond)
                ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
        self.assertEqual(result.status, "inspection_succeeded")
        self.assertEqual(len(set(offered)), 1, offered)
        tools = offered[0]
        self.assertEqual(
            received,
            [
                [],
                # Every counted correction ends with its count line.
                [f"TEXT OFFERED[{tools}]\nFAULT[turn 1/4]"],
                # The text ending was the first failed turn in a row, so the
                # cut-off is the second of the default four.
                [f"CUT[7] OFFERED[{tools}] FAULT[turn 2/4]"],
                [
                    # The third failed turn in a row: the last correction before
                    # the fourth stops the Attempt.
                    "REFUSED[tool_arguments_invalid_json] "
                    f"NOT-RUN[inspect_input_metadata] NOT-OBJECT. OFFERED[{tools}]"
                    "\nFAULT[turn 3/4]\nLAST[same-refusal]"
                ],
            ],
        )

    def test_a_cut_off_reply_runs_its_complete_calls_once_and_drops_its_partial(self):
        # Admission reads the cut-off reply (ADR 0044): its copy collapses, its
        # complete call runs once and its partial last call is dropped, never
        # corrected. The note says what was cut.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            complete = json.dumps({"input_set_id": input_set_id})
            script = [
                ModelResponse(
                    parts=[
                        ToolCallPart("inspect_input_metadata", complete, "copy-1"),
                        ToolCallPart("inspect_input_metadata", complete, "copy-2"),
                        ToolCallPart("inspect_input_metadata", '{"input_se', "cut"),
                    ],
                    usage=RequestUsage(output_tokens=9),
                    finish_reason="length",
                ),
            ]
            received = []

            def respond(messages, info):
                received.append(messages)
                return script[len(received) - 1]

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            events = read_evidence(journey)
            review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
        self.assertEqual(result.status, "inspection_succeeded")
        self.assertEqual(len(received), 1)
        self.assertEqual(
            [
                e["payload"]["tool_call_id"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
            ],
            ["copy-1"],
        )
        (routed,) = [
            e["payload"] for e in events if e["event_type"] == "AgentTurnRouted"
        ]
        self.assertEqual(
            (routed["run_order"], routed["copies"], routed["dropped_call_id"]),
            (["copy-1"], {"copy-2": "copy-1"}, "cut"),
        )
        (truncation,) = [
            e["payload"]
            for e in events
            if e["event_type"] == "ModelCompletionTruncated"
        ]
        self.assertEqual(
            (truncation["correction_count"], truncation["feedback"]),
            (1, routed["feedback"]),
        )
        self.assertIn(
            "Your reply was cut at 9 tokens; it held 3 tool calls.\n"
            "Last call `inspect_input_metadata`: it reached the ceiling and was "
            "dropped.\nAllowed form: call `inspect_input_metadata` again in a later "
            "reply if you still need it." + _count(1),
            routed["feedback"],
        )

    def test_a_nemotron_reply_of_only_think_tags_is_reasoning_only(self):
        # Nemotron's reasoning arrives in content tags; a reply of nothing else
        # holds no call and no text, and reads Nemotron's own correction.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            seen = []

            def respond(messages, info):
                seen.append((messages, info))
                if len(seen) == 1:
                    return ModelResponse(
                        parts=[TextPart("<think>Now the call.</think>")],
                        finish_reason="tool_call",
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata", {"input_set_id": input_set_id}
                        )
                    ]
                )

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code, len(seen)),
                ("inspection_succeeded", None, 2),
            )
            offered = ", ".join(
                f"`{name}`"
                for name in sorted(tool.name for tool in seen[0][1].function_tools)
            )
            retry = seen[1][0][-1].parts[-1]
            self.assertEqual(
                (retry.part_kind, retry.content),
                (
                    "retry-prompt",
                    "Your reply held only reasoning; no tool call arrived, so "
                    f"nothing happened.\nAllowed form: a tool call to one of {offered}."
                    + _count(1),
                ),
            )
            # The reply replays as reasoning, never as tags in its content.
            self.assertEqual(
                [(p.part_kind, p.content) for p in seen[1][0][-2].parts],
                [("thinking", "Now the call.")],
            )
            (turn,) = [
                e["payload"]
                for e in read_evidence(journey)
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] != "singleton_admitted"
            ]
            self.assertEqual(
                (turn["disposition"], turn["correction_key"]),
                ("reasoning_refused", "reasoning_only"),
            )

    def test_a_refused_nemotron_call_reads_nemotron_wording_from_the_one_writer(self):
        # On the Nemotron runtime every correction the loop sends is the retry
        # rendering in Nemotron's wording: field, value sent, allowed form.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(), model="nvidia/Nemotron-3-Ultra-550b-a55b"
            )
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            # Nemotron's calls carry no call reasoning.
            good = {"input_set_id": input_set_id}
            script = [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata", {**good, "note": "x"}, "extra"
                        )
                    ]
                ),
                ModelResponse(
                    parts=[ToolCallPart("inspect_input_metadata", "{broken", "bad")]
                ),
                ModelResponse(parts=[TextPart("Finished.")]),
                ModelResponse(
                    parts=[ToolCallPart("inspect_input_metadata", good, "good")]
                ),
            ]
            received, offered = [], []

            def respond(messages, info):
                # A refused call's correction is its own failed result, in its
                # slot; a text ending's is the retry part.
                sent = [
                    (part.part_kind, part.content)
                    for part in messages[-1].parts
                    if part.part_kind == "retry-prompt"
                ] + [
                    (part.part_kind, part.content)
                    for part in messages[-1].parts
                    if part.part_kind == "tool-return" and part.outcome == "failed"
                ]
                received.append([item for pair in sent[-1:] for item in pair])
                offered.append(
                    ", ".join(f"`{tool.name}`" for tool in info.function_tools)
                )
                return script[len(received) - 1]

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
        self.assertEqual(result.status, "inspection_succeeded")
        self.assertEqual(len(set(offered)), 1, offered)
        tools = offered[0]
        self.assertEqual(
            tools,
            "`inspect_input_metadata`, `request_planning_hitl`, `search_journey`",
        )
        # Only the Tools callable at the input stage are allowed; the question
        # Tool waits on its step.
        allowed = (
            "`inspect_input_metadata`, `search_journey`.\n"
            "Waiting on their step, not callable now: `request_planning_hitl`"
        )
        # The refused line names an argument code by its kind.
        extra = "tool_arguments_invalid"
        self.assertEqual(
            received,
            [
                [],
                [
                    "tool-return",
                    f"Refused before execution ({extra}).\n"
                    "Tool `inspect_input_metadata` was not run. Fix each item, then "
                    "call it again.\n"
                    '- Field `note`: you sent "x".\n'
                    "  Allowed form: Extra inputs are not permitted.\n"
                    f"Allowed now: one of {allowed}." + _count(1),
                ],
                [
                    "tool-return",
                    "Refused before execution (tool_arguments_invalid_json).\n"
                    "Tool `inspect_input_metadata` was not run. Fix each item, then "
                    "call it again.\n"
                    "- Arguments: not one JSON object.\n"
                    "  Allowed form: one JSON object of argument names to values, "
                    'for example {"name": "value"}.\n'
                    f"Allowed now: one of {allowed}." + _count(2),
                ],
                [
                    "retry-prompt",
                    "Your reply ended in text without a tool call; text is not a "
                    "result, so nothing happened.\nA step happens only through a tool "
                    "call: make the call that does it.\nAllowed form: a tool call to "
                    f"one of {tools}."
                    # The third failed turn in a row: the firm line before the
                    # fourth stops the Attempt.
                    + _count(3)
                    + "\nLast correction: one more failed turn stops the Attempt "
                    "for review.",
                ],
            ],
        )

    def test_a_resent_reasoning_only_reply_waits_at_the_effort_review_and_replays(
        self,
    ):
        # After a resend the reply is routed; its retained history keeps it as
        # received, so the effort Review holds and the extension still shows the
        # model its reasoning-only reply.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(), agent_invocation={"model_requests": 1}
            )
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            seen = []

            def respond(messages, info):
                seen.append(messages)
                if len(seen) == 1:
                    raise ModelHTTPError(
                        status_code=504,
                        model_name="scripted",
                        body={"message": "gateway timeout"},
                    )
                if len(seen) == 2:
                    return ModelResponse(
                        parts=[ThinkingPart("Now the call.")],
                        finish_reason="tool_call",
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="after-extension",
                        )
                    ]
                )

            def attempt():
                return planning_assignment(model_factory=ScriptedModelEndpoint(respond))

            stopped = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual((stopped.status, len(seen)), ("effort_review_pending", 2))
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                "effort_review_pending",
            )
            accepted = respond_effort_review(
                journey,
                stopped.agent_effort_review.pending_request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=1),
            )
            resumed = attempt().run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                effort_review_response_event_id=accepted.response_event_id,
            )
            self.assertEqual((resumed.status, len(seen)), ("inspection_succeeded", 3))
            self.assertEqual(
                [(p.part_kind, p.content) for p in seen[2][-2].parts],
                [("thinking", "Now the call.")],
            )

    def test_repeated_reasoning_only_replies_stop_as_failed_turns(self):
        # A reasoning-only reply is not "ended in text"; the fourth in a row is the
        # failed turn that stops the Attempt, its reply the source.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())

            def respond(messages, info):
                del messages, info
                return ModelResponse(
                    parts=[ThinkingPart("Now I will call the Tool.")],
                    finish_reason="tool_call",
                )

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code),
                ("agent_work_review_pending", None),
            )
            request = result.agent_work_review.pending_request
            events = read_evidence(journey)
            response = [e for e in events if e["event_type"] == "ModelResponseRecorded"]
            turns = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            self.assertEqual(
                (request.cause, request.source_event_id, len(response)),
                ("failed_turns", turns[-1]["event_id"], 4),
            )
            self.assertEqual(len(response), 4)
            self.assertEqual(
                [
                    (item.event_id, item.correction_key, item.correction_count)
                    for item in request.faults
                ],
                [
                    (turn["event_id"], "reasoning_only", count)
                    for count, turn in enumerate(turns, start=1)
                ],
            )
            self.assertEqual(
                {turn["payload"]["disposition"] for turn in turns},
                {"reasoning_refused"},
            )
            (stopped,) = [e for e in events if e["event_type"] == "AttemptStopped"]
            self.assertEqual(stopped["payload"]["code"], "failed_turns")
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(
                (
                    reopened.status,
                    reopened.failure_code,
                    reopened.agent_work_review.pending_request.cause,
                ),
                ("agent_work_review_pending", None, "failed_turns"),
            )
            # The cause is bound to the reply that stopped: not "ended in text".
            with self.assertRaisesRegex(
                ValueError, "missing-result stop lacks its exact model response"
            ):
                project_journey_events(
                    _with_stop_cause(events, "result_missing"),
                    journey_file=receipt.journey_file,
                )

    def test_repeated_text_ending_stops_for_review_after_three_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            input_set_id = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).registered_input_set.input_set_id
            calls = 0

            def respond(messages, info):
                nonlocal calls
                del messages, info
                calls += 1
                if calls <= 4:
                    return ModelResponse(parts=[TextPart("Finished.")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {"input_set_id": input_set_id},
                            tool_call_id="after-continue",
                        )
                    ]
                )

            def attempt():
                return planning_assignment(model_factory=ScriptedModelEndpoint(respond))

            result = attempt().run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code, calls),
                ("agent_work_review_pending", None, 4),
            )
            request = result.agent_work_review.pending_request
            events = read_evidence(journey)
            response = [e for e in events if e["event_type"] == "ModelResponseRecorded"]
            turns = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            # The fourth text ending in a row is the failed turn that stops.
            self.assertEqual(
                (request.cause, request.source_event_id, len(response)),
                ("failed_turns", turns[-1]["event_id"], 4),
            )
            self.assertEqual(
                [
                    (item.event_id, item.correction_key, item.correction_count)
                    for item in request.faults
                ],
                [
                    (turn["event_id"], "text_ending", count)
                    for count, turn in enumerate(turns, start=1)
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "missing-result stop lacks its exact model response"
            ):
                project_journey_events(
                    _with_stop_cause(events, "result_missing"),
                    journey_file=receipt.journey_file,
                )
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(
                (reopened.status, reopened.failure_code),
                ("agent_work_review_pending", None),
            )
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(action="continue"),
            )
            resumed = attempt().run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                work_review_response_event_id=answer.response_event_id,
            )
            self.assertEqual((resumed.status, calls), ("inspection_succeeded", 5))
            self.assertEqual(
                [item.outcome for item in resumed.attempts],
                ["stopped", "succeeded"],
            )
            self.assertFalse(
                any(
                    e["event_type"] in {"AttemptFailed", "RunFailed"}
                    for e in read_evidence(journey)
                )
            )

    def test_role_policy_exhaustion_stops_for_review(self):
        # A Run configured with one failed turn stops at the first reply in which
        # nothing worked, before any correction is sent.
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(
                Path(directory).resolve(),
                agent_invocation={"failed_turns": 1},
            )
            calls = 0

            def respond(messages, info):
                nonlocal calls
                del messages
                calls += 1
                tool = info.function_tools[0]
                # Omitted bound arguments are filled; an undeclared one is invalid.
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool.name,
                            '{"undeclared_argument":true}',
                            tool_call_id=f"invalid-{calls}",
                        )
                    ]
                )

            attempt = planning_assignment(model_factory=ScriptedModelEndpoint(respond))
            result = attempt.run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
            self.assertEqual(
                (result.status, result.failure_code, calls),
                ("agent_work_review_pending", None, 1),
            )
            attempt_events = [
                event
                for event in read_evidence(journey)
                if event["payload"].get("attempt_id") == result.attempt.attempt_id
            ]
            self.assertEqual(
                sum(
                    event["event_type"] == "ModelPhysicalContactStarted"
                    for event in attempt_events
                ),
                1,
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    for event in attempt_events
                )
            )
            (refused,) = [
                event
                for event in attempt_events
                if event["event_type"] == "ToolCallValidated"
            ]
            (turn,) = [
                event
                for event in attempt_events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            self.assertEqual(
                turn["payload"]["validation_event_ids"], [refused["event_id"]]
            )
            request = result.agent_work_review.pending_request
            self.assertEqual(
                (request.cause, request.source_event_id, request.reason),
                (
                    "failed_turns",
                    turn["event_id"],
                    "the model's replies failed their limit of turns in a row",
                ),
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                "agent_work_review_pending",
            )

    def test_public_registration_rejects_new_correction_entitlements(self):
        # The per-kind allowances are retired; only failed_turns remains.
        for key in (
            "output_repair",
            "same_call_correction",
            "repeated_call",
            "truncation_retry",
        ):
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaisesRegex(ValueError, "runtime_selections_invalid"),
            ):
                _registered(
                    Path(directory).resolve(),
                    agent_invocation={"scopes": {"role": {"planning_agent": {key: 1}}}},
                )

    def test_public_registration_rejects_a_retired_role(self):
        with tempfile.TemporaryDirectory() as directory:
            # An otherwise lawful number; only the retired role name is refused.
            values = {"failed_turns": 1}
            with self.assertRaisesRegex(ValueError, "runtime_selections_invalid"):
                _registered(
                    Path(directory).resolve(),
                    agent_invocation={
                        "scopes": {"role": {"execution_planning_agent": values}}
                    },
                )

    def test_input_configured_failed_turns_and_retained_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            # Two refused replies are corrected; the third failed turn would stop.
            receipt, _ = _registered(
                Path(directory).resolve(),
                agent_invocation={"failed_turns": 3},
            )
            journey = Journey.open(Path(receipt.journey_file))
            before = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            calls = 0

            def respond(messages, info):
                nonlocal calls
                del messages, info
                calls += 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "inspect_input_metadata",
                            {
                                "input_set_id": str(uuid.uuid4())
                                if calls <= 2
                                else before.registered_input_set.input_set_id
                            },
                            tool_call_id=f"input-{calls}",
                        )
                    ]
                )

            result = planning_assignment(
                model_factory=ScriptedModelEndpoint(respond)
            ).run(
                prepare_input_correction(journey), environ={"LOCAL_MODEL_KEY": "secret"}
            )
            self.assertEqual(result.status, "inspection_succeeded")
            self.assertEqual(calls, 3)
            Path(before.registered_input_set.runtime_config_file).write_text(
                "{}", encoding="utf-8"
            )
            reopened = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(reopened.status, result.status)
            for attack in (
                "missing-policy",
                "missing-id",
                "wrong-id",
                "coherent-enlargement",
            ):
                with self.subTest(attack=attack):
                    events = copy.deepcopy(
                        read_evidence(Journey.open(Path(receipt.journey_file)))
                    )
                    binding = next(
                        event["payload"]
                        for event in events
                        if event["event_type"] == "AttemptStarted"
                    )
                    if attack == "missing-policy":
                        binding.pop("continuation_policy")
                    elif attack == "missing-id":
                        binding.pop("continuation_policy_id")
                    elif attack == "wrong-id":
                        binding["continuation_policy_id"] = "continuation-policy:wrong"
                    else:
                        enlarged = ContinuationPolicy(failed_turns=4)
                        binding["continuation_policy"] = enlarged.model_dump(
                            mode="json"
                        )
                        binding["continuation_policy_id"] = enlarged.policy_id
                    with self.assertRaises(ValueError):
                        project_journey_events(
                            events, journey_file=receipt.journey_file
                        )

    # Known pre-existing complexity; this test is a candidate for splitting.
    def test_role_local_correction_availability_and_caller_budgets_are_independent(  # noqa: C901, PLR0915 - complexity 22 / 107 statements; see above
        self,
    ):
        for order in ("availability-first", "correction-first"):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                receipt, _ = _registered(root)
                input_set_id = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).registered_input_set.input_set_id
                initial_calls = 0

                async def initial(messages, info):
                    nonlocal initial_calls
                    del messages, info
                    initial_calls += 1
                    if order == "correction-first" and initial_calls == 1:  # noqa: B023
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    "inspect_input_metadata",
                                    {"input_set_id": "wrong"},
                                    tool_call_id="correction-before-availability",
                                )
                            ]
                        )
                    raise ModelHTTPError(
                        429,
                        "ordering-fault",
                        body={},
                        headers={},
                    )

                journey = Journey.open(Path(receipt.journey_file))
                pending = planning_assignment(
                    model_factory=ScriptedModelEndpoint(initial)
                ).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
                request = pending.model_availability_review.pending_request
                self.assertIsNotNone(request)

                retry_calls = 0

                async def retry(messages, info):
                    nonlocal retry_calls
                    del messages, info
                    retry_calls += 1
                    if order == "availability-first" and retry_calls == 1:  # noqa: B023
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    "inspect_input_metadata",
                                    {"input_set_id": "wrong"},
                                    tool_call_id="correction-after-availability",
                                )
                            ]
                        )
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "inspect_input_metadata",
                                {"input_set_id": input_set_id},  # noqa: B023
                                tool_call_id="ordering-final",
                            )
                        ]
                    )

                respond_model_availability_review(
                    RespondModelAvailabilityReviewCommand(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        response={"action": "retry"},
                    ),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                    attempt=planning_assignment(
                        model_factory=ScriptedModelEndpoint(retry)
                    ),
                )
                final = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                )
                self.assertEqual(final.status, "inspection_succeeded")
                journey = Journey.open(Path(receipt.journey_file))
                events = read_evidence(journey)
                self.assertEqual(
                    len(
                        [
                            event
                            for event in events
                            if event["event_type"] == "ModelPhysicalContactStarted"
                        ]
                    ),
                    4,
                )
                review_index = next(
                    index
                    for index, event in enumerate(events)
                    if event["event_type"] == "ModelAvailabilityReviewRequested"
                )
                correction_index = next(
                    index
                    for index, event in enumerate(events)
                    if event["event_type"] == "ToolCallValidated"
                    and event["payload"]["status"] == "invalid"
                )
                if order == "availability-first":
                    self.assertLess(review_index, correction_index)
                else:
                    self.assertLess(correction_index, review_index)
        agent_role = "planning_agent"
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())

            def model_factory(_loaded, _selected):
                return _availability_failure_model()(_loaded, _selected)

            def make_attempt():
                return planning_assignment(model_factory=model_factory)

            attempt = make_attempt()

            review = attempt.run(journey, environ={"LOCAL_MODEL_KEY": "secret"})

            self.assertEqual(review.status, "model_availability_review_pending")
            self.assertIsNone(review.failure_code)
            self.assertEqual(review.attempts[-1].agent_role, agent_role)
            self.assertEqual(review.attempts[-1].outcome, "failed")
            pending = review.model_availability_review.pending_request
            self.assertEqual(pending.agent_role, agent_role)
            self.assertEqual(len(pending.contacts), 2)
            self.assertEqual(
                pending.exhaustion_reason,
                "automatic_retry_exhausted",
            )
            retained = Journey.open(Path(receipt.journey_file)).read()
            self.assertFalse(
                any(event["event_type"] == "RunFailed" for event in retained)
            )

            first_frontier = pending.retry_frontier_event_id
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=receipt.journey_file,
                    request_event_id=pending.request_event_id,
                    response=RetryModelAvailabilityResponse(action="retry"),
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=make_attempt(),
            )
            recurring = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(recurring.status, "model_availability_review_pending")
            self.assertEqual(
                recurring.model_availability_review.pending_request.agent_role,
                agent_role,
            )
            self.assertEqual(
                recurring.model_availability_review.pending_request.retry_frontier_event_id,
                first_frontier,
            )
            self.assertEqual(
                [attempt.agent_role for attempt in recurring.attempts[-2:]],
                [agent_role, agent_role],
            )
            attempt_ids = {attempt.attempt_id for attempt in recurring.attempts[-2:]}
            self.assertEqual(
                len(
                    [
                        event
                        for event in read_evidence(
                            Journey.open(Path(receipt.journey_file))
                        )
                        if event["event_type"] == "ModelPhysicalContactStarted"
                        and event["payload"].get("attempt_id") in attempt_ids
                    ]
                ),
                4,
            )
            timeline = read_evidence(Journey.open(Path(receipt.journey_file)))
            response = next(
                event
                for event in timeline
                if event["event_type"] == "ModelAvailabilityReviewResponded"
                and event["payload"].get("request_event_id") == pending.request_event_id
            )
            rewired_bridge = copy.deepcopy(timeline)
            next(
                event
                for event in rewired_bridge
                if event["event_id"] == response["event_id"]
            )["parent_ids"] = [timeline[0]["event_id"]]
            with self.assertRaises(ValueError):
                project_journey_events(
                    rewired_bridge, journey_file=receipt.journey_file
                )

    def test_cancelled_attempt_replays_its_frontier_without_logical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            registered = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            registration = journey.read()[-1]
            selection = registration["payload"]["runtime_selections"]["model_roster"][
                "qwen"
            ]
            endpoint = {
                "provider": selection["provider"],
                "model": selection["model"],
                "contacted_url": selection["endpoint"],
            }
            attempt_id = str(uuid.uuid4())
            lifecycle = ActivityLifecycle(journey, causal_head=registration["event_id"])
            attempt_activity = lifecycle.start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            instructions = role_instructions("planning_agent")
            attempt_activity.record_with_artifact(
                registered_attempt_start(
                    journey.read(),
                    attempt_id=attempt_id,
                    agent_role="planning_agent",
                    input_set_id=registered.registered_input_set.input_set_id,
                ).model_copy(update={"instruction_identity": instructions.identity}),
                instructions.content,
                media_type=INSTRUCTION_MEDIA_TYPE,
            )
            contact_activity = attempt_activity.start_child(
                adapter="pydantic_ai_model",
                stage="model_contact",
            )
            request = contact_activity.record(
                ModelRequestRecorded(
                    attempt_id=attempt_id,
                    request_number=1,
                    nebius_model_endpoint=endpoint,
                    compatible_adapter_evidence={
                        "model_name": "cancelled-test",
                        "model_system": "function",
                        "request": {
                            "messages": [],
                            "model_settings": {
                                "parallel_tool_calls": False,
                                "max_tokens": 245760,
                            },
                            "function_tools": [],
                            "output_tools": [],
                            "allow_text_output": True,
                            "output_mode": "text",
                            "instruction_parts": None,
                        },
                    },
                )
            )
            started = contact_activity.record(
                ModelPhysicalContactStarted(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    contact_ordinal=1,
                )
            )
            cancelled = contact_activity.record(
                ModelPhysicalContactCancelled(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    contact_started_event_id=started["event_id"],
                    contact_ordinal=1,
                )
            )
            contact_activity.cancelled(cancelled)
            record_model_attempt_cancelled(
                attempt_activity, journey, attempt_id=attempt_id
            )

            replayed = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

            # A cancelled Attempt does not end the Run: it reads the frontier the
            # Attempt left, for a fresh Attempt to continue.
            self.assertEqual(replayed.status, "inputs_registered")
            self.assertIsNone(replayed.failure_code)
            self.assertEqual(replayed.attempts[-1].outcome, "cancelled")
            model_events = {event.event_type for event in replayed.attempts[-1].events}
            self.assertIn("ModelPhysicalContactCancelled", model_events)
            self.assertIn("AttemptCancelled", model_events)
            self.assertNotIn("ModelRequestFailed", model_events)
            self.assertFalse(
                any(
                    event["event_type"]
                    in {"ModelAvailabilityReviewRequested", "RunFailed"}
                    for event in journey.read()
                )
            )

    def test_full_replay_rejects_tool_validation_before_model_response(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, journey = _availability_frontier(Path(directory).resolve())
            planning_assignment(model_factory=_availability_failure_model()).run(
                prepare_input_correction(journey), environ={"LOCAL_MODEL_KEY": "secret"}
            )
            hostile = json.loads(json.dumps(read_evidence(journey)))
            started_index = next(
                index
                for index, event in enumerate(hostile)
                if event["event_type"] == "ModelPhysicalContactStarted"
            )
            started = hostile[started_index]
            tool_event_id = str(uuid.uuid4())
            hostile.insert(
                started_index + 1,
                {
                    "event_id": tool_event_id,
                    "event_type": "ToolCallValidated",
                    "parent_ids": [started["payload"]["logical_request_event_id"]],
                    "payload": {
                        "attempt_id": started["payload"]["attempt_id"],
                        "tool_name": "inspect_input_metadata",
                        "tool_call_id": "hostile-before-response",
                        "arguments": {
                            "input_set_id": review_journey(
                                ReviewJourneyQuery(journey_file=receipt.journey_file)
                            ).registered_input_set.input_set_id
                        },
                        "status": "valid",
                        "errors": [],
                    },
                },
            )
            # Preserve semantic causal order; journal contacts are parentless.
            next_semantic = next(
                event
                for event in hostile[started_index + 2 :]
                if event.get("parent_ids")
                == [started["payload"]["logical_request_event_id"]]
            )
            next_semantic["parent_ids"] = [tool_event_id]

            with self.assertRaisesRegex(
                ValueError, "outside a retained model response"
            ):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_cancelled_retry_delay_replay_binds_exact_scheduled_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            registered = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            registration = journey.read()[-1]
            selection = registration["payload"]["runtime_selections"]["model_roster"][
                "qwen"
            ]
            endpoint = {
                "provider": selection["provider"],
                "model": selection["model"],
                "contacted_url": selection["endpoint"],
            }
            attempt_id = str(uuid.uuid4())
            lifecycle = ActivityLifecycle(journey, causal_head=registration["event_id"])
            attempt_activity = lifecycle.start(
                adapter="pydantic_ai",
                stage="planning_attempt",
            )
            instructions = role_instructions("planning_agent")
            attempt_activity.record_with_artifact(
                registered_attempt_start(
                    journey.read(),
                    attempt_id=attempt_id,
                    agent_role="planning_agent",
                    input_set_id=registered.registered_input_set.input_set_id,
                ).model_copy(update={"instruction_identity": instructions.identity}),
                instructions.content,
                media_type=INSTRUCTION_MEDIA_TYPE,
            )
            contact_activity = attempt_activity.start_child(
                adapter="pydantic_ai_model",
                stage="model_contact",
            )
            request = contact_activity.record(
                ModelRequestRecorded(
                    attempt_id=attempt_id,
                    request_number=1,
                    nebius_model_endpoint=endpoint,
                    compatible_adapter_evidence={
                        "model_name": "cancelled-delay-test",
                        "model_system": "function",
                        "request": {
                            "messages": [],
                            "model_settings": {
                                "parallel_tool_calls": False,
                                "max_tokens": 245760,
                            },
                            "function_tools": [],
                            "output_tools": [],
                            "allow_text_output": True,
                            "output_mode": "text",
                            "instruction_parts": None,
                        },
                    },
                )
            )
            started = contact_activity.record(
                ModelPhysicalContactStarted(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    contact_ordinal=1,
                )
            )
            failed = contact_activity.record(
                ModelPhysicalContactFailed(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    contact_started_event_id=started["event_id"],
                    contact_ordinal=1,
                    error_type="ModelHTTPError",
                    contact_seconds=1.0,
                    http_status_code=503,
                    failure_reason="provider_failure",
                    completion_certainty="response_received",
                    policy_disposition="automatic_retry_scheduled",
                    retry_after_disposition="missing",
                )
            )
            scheduled = contact_activity.record(
                ModelAvailabilityRetryScheduled(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    failed_contact_event_id=failed["event_id"],
                    # The default 30-second base at the recorded 0.5 jitter.
                    delay_seconds=15,
                    delay_source="exponential_jitter",
                )
            )
            cancelled = contact_activity.record(
                ModelAvailabilityRetryDelayCancelled(
                    attempt_id=attempt_id,
                    request_number=1,
                    logical_request_event_id=request["event_id"],
                    retry_scheduled_event_id=scheduled["event_id"],
                    delay_seconds=15,
                )
            )
            contact_activity.cancelled(cancelled)
            record_model_attempt_cancelled(
                attempt_activity, journey, attempt_id=attempt_id
            )
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                ).status,
                "inputs_registered",
            )
            hostile = json.loads(json.dumps(journey.read()))
            next(
                event
                for event in hostile
                if event["event_type"] == "ModelAvailabilityRetryDelayCancelled"
            )["payload"]["delay_seconds"] = 3
            with self.assertRaisesRegex(ValueError, "changed its schedule"):
                project_journey_events(hostile, journey_file=receipt.journey_file)

    def test_full_agentic_composition_stops_at_model_availability_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = _registered(root)

            async def fail(messages, info):
                del messages, info
                raise ConnectionError("provider failure text must not escape")

            class FailingComposition:
                environ = {"LOCAL_MODEL_KEY": "secret"}

                def planning_assignment(self, _journey):
                    return planning_assignment(
                        model_factory=ScriptedModelEndpoint(fail)
                    )

            with use_run_advancement_composition(FailingComposition()):
                review = advance(ResumeRun(journey_file=receipt.journey_file))

            self.assertEqual(review.status, "model_availability_review_pending")
            self.assertEqual(
                [attempt.agent_role for attempt in review.attempts],
                ["planning_agent"],
            )
            self.assertIsNone(review.failure_code)
            self.assertEqual(review.attempts[-1].outcome, "failed")
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
            self.assertEqual(
                len(
                    [
                        event
                        for event in events
                        if event["event_type"] == "ModelPhysicalContactStarted"
                    ]
                ),
                2,
            )
            self.assertNotIn("provider failure text", json.dumps(events))
            self.assertNotIn("secret", json.dumps(events))
            self.assertFalse(
                any(
                    item.event_type
                    in {"AgenticExecutionReadmitted", "ExecutionConfigurationAccepted"}
                    for item in review.lineage
                )
            )


if __name__ == "__main__":
    unittest.main()
