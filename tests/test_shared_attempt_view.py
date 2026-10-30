"""Every registered Agent role reads back through the one checked Attempt view."""

from __future__ import annotations

import copy
import json
import os
import signal
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.advancement import (
    AnswerReview,
    BeginInvestigation,
    ExecutionReviewAnswer,
    ResumeRun,
    RunAdvancementRefused,
    advance,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import ObserveRun, observe, project_journey_events
from bio_neb.failure import ProviderError
from bio_neb.investigation.investigation_commands import BeginInvestigationCommand
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import operator_sigint
from tests.execution_approval import approve_execution
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.run_prerequisites import RunPrerequisites, State
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    SequentialSourceResponses,
    begin_in_pass,
    forked_child,
    members_one_at_a_time,
)

_SOURCE_RESULTS = {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
_RECEIPTS = {"ObservedToolExecutionStarted", "ObservedToolExecutionSucceeded"}
_FAULT = "offline planner fault"


class _SearchesFirst:
    """Read the Run's own history once, then plan from typed source results."""

    def __init__(self) -> None:
        from tests.sequential_investigation_support import (
            _AdaptiveInvestigationModel,
        )

        self.planner = _AdaptiveInvestigationModel()
        self.searched = False
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        if not self.searched:
            self.searched = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {"kinds": ["InvestigationStarted"]},
                        tool_call_id="read-scientific-history",
                    )
                ]
            )
        return await self.planner.respond(messages, info)


class _SearchesAndReadsTogether(_SearchesFirst):
    """Send the history read and the first RefSNP lookup in one response."""

    async def respond(self, messages, info):
        if not self.searched:
            self.searched = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {"kinds": ["InvestigationStarted"]},
                        tool_call_id="read-scientific-history",
                    ),
                    ToolCallPart(
                        "fetch_refsnp",
                        {"rsid": "rs429358"},
                        tool_call_id="read-refsnp",
                    ),
                ]
            )
        return await self.planner.respond(messages, info)


class _PrematureOrderedCalls:
    """Ask for two assembly reports before any RefSNP record, every time."""

    def __init__(self) -> None:
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        early = {"assembly_accession": "GCF_000001405.40", "refsnp_event_id": "missing"}
        return ModelResponse(
            parts=[
                ToolCallPart("fetch_assembly_report", early, tool_call_id="early-a"),
                ToolCallPart("fetch_assembly_report", early, tool_call_id="early-b"),
            ]
        )


class _AssemblySourceFaults(SequentialSourceResponses):
    """Answer RefSNP lookups; the assembly report source faults."""

    def fetch_assembly_report(self, assembly_accession):
        self.operations.append(f"fetch_assembly_report:{assembly_accession}")
        raise OSError(_FAULT)


class _FailsAfterFirstSource(_SearchesFirst):
    """After its history read and one RefSNP record, the planner faults."""

    async def respond(self, messages, info):
        if self.planner.calls == 1:
            raise RuntimeError(_FAULT)
        return await super().respond(messages, info)


def _approved_with_three_requests(root: Path) -> tuple[str, Path]:
    """A verified Run whose plan names three rsIDs from its registered request file.

    The approval stops the coordinator before Investigation begins, so the Run
    rests at its after-Execution child point; the caller begins Investigation with
    the same request file (#342).
    """
    from tests.planning_assignment_fakes import (
        configured_provider,
        prepare_configured_run,
    )
    from tests.sequential_investigation_support import SequentialAdaptiveModel
    from tests.terminal_journey_fakes import stopped_before_investigation

    document = {
        "schema_version": "bio-neb.investigation-request/2",
        "asserted_by": "researcher:owner",
        "requests": [
            {
                "request": f"Investigate {rsid} in sample HG001.",
                "target": {"rsid": rsid},
            }
            for rsid in SequentialAdaptiveModel.rsids
        ],
    }
    receipt, review, _ = prepare_configured_run(root, investigation_request=document)
    request = review.execution_hitl.pending_request
    with (
        use_run_advancement_dependencies(
            environ={},
            parabricks_infrastructure_factory=configured_provider(
                receipt.journey_file, []
            ),
        ),
        stopped_before_investigation(),
    ):
        advance(
            AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                answer=ExecutionReviewAnswer(
                    kind="execution",
                    response=approve_execution(request.proposal),
                ),
            )
        )
    requests = root / "investigation-request.json"
    requests.write_text(json.dumps(document), encoding="utf-8")
    return receipt.journey_file, requests


class _InterruptedRejection(SequentialSourceResponses):
    """The operator's SIGINT reaches the first RefSNP lookup, which the source then
    rejects; ``operations`` keeps every lookup asked for."""

    def fetch_refsnp(self, rsid):
        if not self.operations:
            self.operations.append(f"fetch_refsnp:{rsid}")
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)  # the coordinator is cancelled while this Tool runs
            raise ProviderError("rejected", "offline Investigation rejection")
        return super().fetch_refsnp(rsid)


class _InterruptedSources(SequentialSourceResponses):
    """The operator's SIGINT arrives while the first RefSNP lookup runs."""

    interrupted: list[str] = []

    def fetch_refsnp(self, rsid):
        if not self.interrupted:
            self.interrupted.append(rsid)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)  # the coordinator is cancelled while this Tool runs
        return super().fetch_refsnp(rsid)


def _investigated_run(
    root: Path,
    *,
    model_type=None,
    sources_type=None,
    expected_status="investigation_review_pending",
) -> tuple[Path, None]:
    """Approve Execution of a Run whose prompt asks a scientific question.

    The approval's Pass verifies Execution and begins Investigation from the
    original prompt, as the retained live a3 Run did; by default it ends at the
    Investigation Review. A planner fault that ends the Run reaches Finalization
    (#273), so its callers expect ``finalization_review_pending``.
    """
    from tests.planning_assignment_fakes import (
        configured_provider,
        prepare_configured_run,
    )

    receipt, review, _ = prepare_configured_run(
        root,
        scientific_intent="Investigate rs429358 in this sample.",
        # The plan types the rsID it quotes (no code reads it from words).
        scientific_rsid="rs429358",
    )
    request = review.execution_hitl.pending_request
    model = (model_type or _SearchesFirst)()
    constructed: list = []
    with (
        use_run_advancement_dependencies(
            environ={"LOCAL_MODEL_KEY": "not-retained"},
            investigation_model_factory=model.model,
            investigation_source_adapter_factory=lambda: (
                sources_type or SequentialSourceResponses
            )(assembly_body=ASSEMBLY_BODY),
        ),
        use_run_advancement_dependencies(
            parabricks_infrastructure_factory=configured_provider(
                receipt.journey_file, constructed
            )
        ),
    ):
        try:
            result = advance(
                AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    answer=ExecutionReviewAnswer(
                        kind="execution",
                        response=approve_execution(request.proposal),
                    ),
                )
            )
        except RunAdvancementRefused as held:
            # A planner fault holds the Run for its fix (R2).
            if held.status != "run_held":
                raise
            result = observe(ObserveRun(journey_file=receipt.journey_file))
    if result.status != expected_status:
        raise AssertionError(f"Run stopped at {result.status}")
    return Path(receipt.journey_file), None


_INVESTIGATED = State("investigated", _investigated_run)


def _insert_unreceipted_copy(events: list[dict], fact: dict) -> None:
    """Retain a re-ided copy of ``fact`` in its successor's causal place, unreceipted."""
    successor = next(e for e in events if e["parent_ids"] == [fact["event_id"]])
    forged = {
        **copy.deepcopy(fact),
        "event_id": str(uuid.uuid4()),
        "parent_ids": successor["parent_ids"],
    }
    successor["parent_ids"] = [forged["event_id"]]
    events.insert(events.index(successor), forged)


class SharedAttemptViewTests(RunPrerequisites, unittest.TestCase):
    def setUp(self):
        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(
                target, side_effect=AssertionError("offline reader forbids network")
            )
            guard.start()
            self.addCleanup(guard.stop)

    def _investigated(self) -> tuple[Path, list[dict]]:
        journey_file = self.prerequisite(_INVESTIGATED).journey_file
        return journey_file, read_evidence(Journey.open(journey_file))

    def _investigation_failed(self) -> tuple[Path, list[dict]]:
        journey_file = self.prerequisite(
            _INVESTIGATED,
            model_type=_FailsAfterFirstSource,
            expected_status="run_held",
        ).journey_file
        return journey_file, read_evidence(Journey.open(journey_file))

    def _refuses_unreceipted_history_read(self, journey_file, events, outcome):
        attempt = observe(ObserveRun(journey_file=str(journey_file))).attempts[-1]
        self.assertEqual(
            (attempt.agent_role, attempt.outcome), ("investigation_planner", outcome)
        )
        forgery = copy.deepcopy(events)
        fact = next(
            e
            for e in forgery
            if e["event_type"] == "JourneySearchRead"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        )
        _insert_unreceipted_copy(forgery, fact)
        with self.assertRaisesRegex(
            ValueError, "Tool fact is not bound to exactly one terminal receipt"
        ):
            project_journey_events(forgery, journey_file=str(journey_file))

    def test_observe_presents_the_investigation_attempt_with_its_exact_receipts(self):
        journey_file, events = self._investigated()
        before = journey_file.read_bytes()

        review = observe(ObserveRun(journey_file=str(journey_file)))

        started = [e for e in events if e["event_type"] == "AttemptStarted"]
        self.assertEqual(
            [attempt.attempt_id for attempt in review.attempts],
            [e["payload"]["attempt_id"] for e in started],
        )
        investigation = review.attempts[-1]
        self.assertEqual(investigation.agent_role, "investigation_planner")
        self.assertEqual(
            {attempt.agent_role for attempt in review.attempts[:-1]},
            {"planning_agent"},
        )
        self.assertEqual(investigation.outcome, "succeeded")
        owned = [
            e
            for e in events
            if e["payload"].get("attempt_id") == investigation.attempt_id
        ]
        # One rsID: its RefSNP record and its assembly report, each answered once.
        receipts = [
            e for e in owned if e["event_type"] == "ObservedToolExecutionSucceeded"
        ]
        source_results = [
            item
            for item in investigation.tool_events
            if item.event_type in _SOURCE_RESULTS
        ]
        self.assertEqual(
            sorted(item.event_type for item in source_results),
            ["AssemblyReportToolResultRecorded", "RefSnpToolResultRecorded"],
        )
        by_id = {e["event_id"]: e for e in events}
        self.assertEqual(
            {item.source_event_id for item in source_results},
            {
                e["payload"]["stage_result_event_id"]
                for e in receipts
                if by_id[e["payload"]["stage_result_event_id"]]["event_type"]
                in _SOURCE_RESULTS
            },
        )
        # The collection's proposal Review closes after its members'
        # Attempts; it is no Tool fact of the member's Attempt.
        [review] = [
            e
            for e in events
            if e["event_type"] == "InvestigationReviewRequested"
            and e["payload"].get("question") is None
        ]
        self.assertNotIn(
            review["event_id"],
            {item.source_event_id for item in investigation.tool_events},
        )
        self.assertEqual(
            [item.source_event_id for item in investigation.receipt_events],
            [e["event_id"] for e in owned if e["event_type"] in _RECEIPTS],
        )
        self.assertEqual(journey_file.read_bytes(), before)

    def test_an_interrupt_during_a_source_tool_holds_the_run_until_resume(self):
        # The Investigation Attempt is cancelled at the turn boundary after its
        # running source Tool finishes; the Run is held, reads, and `resume`
        # continues the Investigation with one fresh Attempt.
        _InterruptedSources.interrupted = []
        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            root = Path(directory).resolve()
            with self.assertRaises(KeyboardInterrupt):
                _investigated_run(root, sources_type=_InterruptedSources)
            (journey_file,) = root.rglob("journey.jsonl")
            journey = Journey.open(journey_file)
            events = read_evidence(journey)
            closed = journey.latest_event(("PassClosed",))
            held = observe(ObserveRun(journey_file=str(journey_file)))
            with use_run_advancement_dependencies(
                environ={"LOCAL_MODEL_KEY": "not-retained"},
                investigation_model_factory=_SearchesFirst().model,
                investigation_source_adapter_factory=lambda: SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            ):
                resumed = advance(ResumeRun(journey_file=str(journey_file)))

        self.assertEqual(_InterruptedSources.interrupted, ["rs429358"])
        (cancelled,) = [e for e in events if e["event_type"] == "AttemptCancelled"]
        owned = [
            e["event_type"]
            for e in events
            if e["payload"].get("attempt_id") == cancelled["payload"]["attempt_id"]
        ]
        self.assertEqual(
            owned[-3:],
            [
                "ObservedToolExecutionSucceeded",
                "ModelTurnCancelled",
                "AttemptCancelled",
            ],
        )
        self.assertEqual(
            (
                closed["payload"]["disposition"],
                closed["payload"]["failure"]["failure_class"],
            ),
            ("held", "interrupted"),
        )
        self.assertEqual(held.status, "run_held")
        self.assertEqual(
            (held.attempts[-1].agent_role, held.attempts[-1].outcome),
            ("investigation_planner", "cancelled"),
        )
        self.assertEqual(resumed.status, "investigation_review_pending")
        investigation = [
            a.outcome
            for a in resumed.attempts
            if a.agent_role == "investigation_planner"
        ]
        self.assertEqual(investigation, ["cancelled", "succeeded"])

    # One member at a time, so the interrupt meets only the first; the
    # concurrent case is test_an_interrupt_cancels_every_member_in_flight.
    @members_one_at_a_time()
    def test_a_source_rejected_after_an_interrupt_keeps_its_review_and_stops(self):
        # Investigation keeps the first member's operational Review inside
        # its Attempt. The interrupt still ends the command right after that
        # Review, before another member or its allowance guard, and the
        # Review holds the member after `resume`.
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from tests.sequential_investigation_support import SequentialAdaptiveModel

        # Resumed, the two other members each reach the held source within their
        # own allowance (one request is enough) and join its Review before
        # contact.
        cases = (
            (
                {},
                "operational_review_pending",
                ["cancelled", "succeeded", "succeeded"],
                2,
            ),
            (
                {"model_requests": 1},
                "operational_review_pending",
                ["cancelled", "succeeded", "succeeded"],
                2,
            ),
        )
        for overrides, resumed_status, outcomes, worked in cases:
            with (
                self.subTest(overrides=overrides),
                tempfile.TemporaryDirectory() as directory,
                operator_sigint(),
            ):
                root = Path(directory).resolve()
                journey_file, requests = _approved_with_three_requests(root)
                sources = _InterruptedRejection(assembly_body=ASSEMBLY_BODY)
                with (
                    use_run_advancement_dependencies(
                        environ={"LOCAL_MODEL_KEY": "not-retained"},
                        investigation_model_factory=SequentialAdaptiveModel().model,
                        investigation_source_adapter_factory=lambda s=sources: s,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    advance(
                        BeginInvestigation(
                            journey_file=journey_file,
                            request_files=(str(requests),),
                            invocation_overrides=AgentInvocationOverrides(**overrides),
                        )
                    )
                held = observe(ObserveRun(journey_file=journey_file))
                resume_sources = SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                with use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "not-retained"},
                    investigation_model_factory=SequentialAdaptiveModel().model,
                    investigation_source_adapter_factory=lambda s=resume_sources: s,
                ):
                    resumed = advance(ResumeRun(journey_file=journey_file))
                events = read_evidence(Journey.open(Path(journey_file)))

                # Only the rejected first lookup ran in the interrupted command.
                self.assertEqual(sources.operations, ["fetch_refsnp:rs429358"])
                kinds = [e["event_type"] for e in events]
                (fault,) = [
                    e for e in events if e["event_type"] == "OperationalFailureRecorded"
                ]
                self.assertEqual(fault["payload"]["failure_class"], "rejected")
                # While its Review is open the source is held: on resume each
                # other member's lookup joins it before contact.
                self.assertEqual(kinds.count("AgentFailureReviewRequested"), 1 + worked)
                self.assertEqual(held.status, "run_held")
                self.assertEqual(
                    (held.attempts[-1].agent_role, held.attempts[-1].outcome),
                    ("investigation_planner", "cancelled"),
                )
                # The Review holds only its member; `resume` works the others.
                self.assertEqual(resumed.status, resumed_status)
                self.assertEqual(
                    [
                        a.outcome
                        for a in resumed.attempts
                        if a.agent_role == "investigation_planner"
                    ],
                    outcomes,
                )
                self.assertIn("junction", [r.kind for r in resumed.pending_reviews])
                self.assertEqual(
                    [
                        o
                        for o in resume_sources.operations
                        if o.startswith("fetch_refsnp")
                    ],
                    [],
                )

    def test_an_interrupt_cancels_every_member_in_flight(self):
        # Per member: members run together; the operator's SIGINT
        # cancels every member Attempt in flight, the hold names each, and
        # `resume` plans each cancelled member again.
        import asyncio

        from tests.sequential_investigation_support import SequentialAdaptiveModel

        class InterruptWhenAllInFlight(SequentialAdaptiveModel):
            def __init__(self) -> None:
                super().__init__()
                self.in_flight, self.sent = 0, False

            async def respond(self, messages, info):  # noqa: ANN001, ANN201
                self.in_flight += 1
                for _ in range(200):
                    if self.in_flight >= 3 or self.sent:
                        break
                    await asyncio.sleep(0.01)
                if not self.sent:
                    self.sent = True
                    os.kill(os.getpid(), signal.SIGINT)
                await asyncio.sleep(1)
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory, operator_sigint():
            root = Path(directory).resolve()
            journey_file, requests = _approved_with_three_requests(root)
            with (
                use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "not-retained"},
                    investigation_model_factory=InterruptWhenAllInFlight().model,
                    investigation_source_adapter_factory=lambda: (
                        SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                    ),
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                advance(
                    BeginInvestigation(
                        journey_file=journey_file, request_files=(str(requests),)
                    )
                )
            journey = Journey.open(Path(journey_file))
            events = read_evidence(journey)
            held = observe(ObserveRun(journey_file=journey_file))
            closed = journey.latest_event(("PassClosed",))
            with use_run_advancement_dependencies(
                environ={"LOCAL_MODEL_KEY": "not-retained"},
                investigation_model_factory=SequentialAdaptiveModel().model,
                investigation_source_adapter_factory=lambda: SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            ):
                resumed = advance(ResumeRun(journey_file=journey_file))

        cancelled = [e for e in events if e["event_type"] == "AttemptCancelled"]
        self.assertEqual(len(cancelled), 3)
        self.assertEqual(held.status, "run_held")
        failure = closed["payload"]["failure"]
        self.assertEqual(failure["failure_class"], "interrupted")
        self.assertEqual(
            sorted(failure["evidence"]["cancelled_attempt_ids"]),
            sorted(e["payload"]["attempt_id"] for e in cancelled),
        )
        self.assertEqual(resumed.status, "investigation_review_pending")
        planner = [
            a for a in resumed.attempts if a.agent_role == "investigation_planner"
        ]
        self.assertEqual(
            sorted(a.outcome for a in planner), ["cancelled"] * 3 + ["succeeded"] * 3
        )

    def test_unreceipted_tool_fact_in_successful_investigation_is_refused(self):
        journey_file, events = self._investigated()
        self._refuses_unreceipted_history_read(journey_file, events, "succeeded")

    def test_unreceipted_tool_fact_in_failed_investigation_is_refused(self):
        journey_file, events = self._investigation_failed()
        failed = [e for e in events if e["event_type"] == "InvestigationAttemptFailed"]
        self.assertEqual(
            [e["payload"]["code"] for e in failed], ["model_contact_failed"]
        )
        self._refuses_unreceipted_history_read(journey_file, events, "failed")

    def test_ordered_investigation_calls_reopen_and_a_borrowed_admission_refuses(
        self,
    ):
        journey_file = self.prerequisite(
            _INVESTIGATED, model_type=_SearchesAndReadsTogether
        ).journey_file
        events = read_evidence(Journey.open(journey_file))

        attempt = observe(ObserveRun(journey_file=str(journey_file))).attempts[-1]

        self.assertEqual(
            (attempt.agent_role, attempt.outcome),
            ("investigation_planner", "succeeded"),
        )
        turns = [
            e
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        ]
        first, second = turns[:2]
        self.assertEqual(
            [first["payload"]["disposition"], second["payload"]["disposition"]],
            ["multi_call_admitted", "multi_call_admitted"],
        )
        self.assertEqual(
            first["payload"]["response_event_id"],
            second["payload"]["response_event_id"],
        )
        # The second call claims the first call's admission.
        forgery = copy.deepcopy(events)
        borrowed = next(e for e in forgery if e["event_id"] == second["event_id"])
        borrowed["payload"]["validation_event_ids"] = first["payload"][
            "validation_event_ids"
        ]
        with self.assertRaisesRegex(
            ValueError, "changed its exact admission disposition"
        ):
            project_journey_events(forgery, journey_file=str(journey_file))

    def test_exhausted_ordered_refusals_reopen_as_a_stopped_investigation_attempt(
        self,
    ):
        journey_file = self.prerequisite(
            _INVESTIGATED,
            model_type=_PrematureOrderedCalls,
            expected_status="agent_work_review_pending",
        ).journey_file
        events = read_evidence(Journey.open(journey_file))

        review = observe(ObserveRun(journey_file=str(journey_file)))

        attempt = review.attempts[-1]
        self.assertEqual(
            (attempt.agent_role, attempt.outcome), ("investigation_planner", "stopped")
        )
        turns = [
            e["payload"]
            for e in events
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        ]
        # Each reply's second identical call collapses onto its first and the
        # kept call is refused: each reply is a failed turn, and the fourth in a
        # row stops the Attempt before the next request.
        self.assertEqual(
            [(t["disposition"], t["correction_count"]) for t in turns],
            [("multi_call_refused", count) for count in range(1, 5)],
        )
        routed = [
            e["payload"]["faults"]
            for e in events
            if e["event_type"] == "AgentTurnRouted"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        ]
        self.assertEqual(
            routed, [[{"key": "copied_calls", "count": n}] for n in range(1, 5)]
        )
        self.assertEqual(attempt.receipt_events, ())

    def test_failed_source_call_binds_its_stage_failure_in_the_shared_view(self):
        journey_file = self.prerequisite(
            _INVESTIGATED,
            sources_type=_AssemblySourceFaults,
            expected_status="run_held",
        ).journey_file
        events = read_evidence(Journey.open(journey_file))

        attempt = observe(ObserveRun(journey_file=str(journey_file))).attempts[-1]

        self.assertEqual(
            (attempt.agent_role, attempt.outcome), ("investigation_planner", "failed")
        )
        [failed] = [
            e
            for e in events
            if e["event_type"] == "ObservedToolExecutionFailed"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        ]
        [stage_failure] = [
            e for e in events if e["event_type"] == "InvestigationSourceContactFailed"
        ]
        self.assertEqual(
            failed["payload"]["stage_failure_event_id"], stage_failure["event_id"]
        )
        self.assertIn(
            stage_failure["event_id"],
            {item.source_event_id for item in attempt.tool_events},
        )
        self.assertIn(
            failed["event_id"],
            {item.source_event_id for item in attempt.receipt_events},
        )

    def test_fork_child_presents_inherited_and_own_attempts_in_the_shared_view(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = {
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in sample-001.",
                "target": {"rsid": "rs429358"},
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=document
                ).journey_file,
            )
            request = root / "request.json"
            request.write_text(json.dumps(document), encoding="utf-8")
            model = _SearchesFirst()
            begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request),),
                ),
                environ={"LOCAL_MODEL_KEY": "not-retained"},
                model_factory=model.model,
                source_adapter_factory=lambda: SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            events = read_evidence(Journey.open(journey.path))

            review = observe(ObserveRun(journey_file=str(journey.path)))

            starts = [e for e in events if e["event_type"] == "AttemptStarted"]
            started = [e for e in starts if e["journey_id"] == journey.journey_id]
            self.assertEqual(len(started), 1)
            # The inherited planning Attempts keep their order and parent identity.
            self.assertEqual(
                [item.attempt_id for item in review.attempts],
                [e["payload"]["attempt_id"] for e in starts],
            )
            self.assertTrue(starts[:-1])
            self.assertEqual(starts[-1], started[0])
            attempt = review.attempts[-1]
            self.assertEqual(
                (attempt.attempt_id, attempt.agent_role, attempt.outcome),
                (
                    started[0]["payload"]["attempt_id"],
                    "investigation_planner",
                    "succeeded",
                ),
            )
            # The history read and the two source results, each with its receipt.
            self.assertEqual(
                sorted(
                    item.event_type
                    for item in attempt.tool_events
                    if item.event_type != "ToolCallValidated"
                ),
                [
                    "AssemblyReportToolResultRecorded",
                    "JourneySearchRead",
                    "RefSnpToolResultRecorded",
                ],
            )
            self.assertEqual(
                [item.event_type for item in attempt.receipt_events].count(
                    "ObservedToolExecutionSucceeded"
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main()
