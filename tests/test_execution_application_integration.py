from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.tool_execution import function_tool_contracts
from bio_neb.application.advancement import (
    AnswerReview,
    ExecutionReviewAnswer,
    PlanningReviewAnswer,
    ResumeRun,
    RunAdvancementRefused,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import (
    ObserveRun,
    observe,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.execution_proposal import (
    RejectExecutionResponse,
    ReviseExecutionResponse,
)
from bio_neb.parabricks.planning_hitl import (
    SelectPlanningHitlResponse,
)
from bio_neb.parabricks.result import read_completed_parabricks_result
from tests.application_fakes import (
    _ready_journey,
)
from tests.execution_approval import approval_json, approve_execution
from tests.explainer_fakes import explained_by_default
from tests.prerequisite_builds import measured_build
from tests.run_prerequisites import RunPrerequisites, State

_CAPTURED_REQUEST = {
    "asserted_by": "researcher:owner",
    "request": "Investigate rs429358 in this sample.",
    "target": {"rsid": "rs429358"},
}


def _request_file(root):
    request_file = root / "investigation-request.json"
    request_file.write_text(
        canonical_json(
            {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": "Investigate rs429358 in this sample.",
                        "target": {"rsid": "rs429358"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return request_file


def _pass_count(journey_file):
    return sum(
        event["event_type"] == "PassStarted"
        for event in Journey.open(Path(journey_file)).read()
    )


def _preparation_review_run(directory):
    """A planned Run without launch settings, stopped at its preparation Review.

    The supplied configuration extends the accepted capability with the tool the
    planned recipe needs, so supplying it is lawful.
    """
    from bio_neb.content import canonical_json_digest
    from bio_neb.parabricks.accepted_pre_execution import (
        _accepted_capability_snapshot,
    )
    from bio_neb.parabricks.capability_policy import (
        CapabilitySnapshot,
        ExecutionRuntimeSelection,
    )
    from tests.planning_assignment_fakes import configured_planning_run

    receipt, composition, configuration = configured_planning_run(
        directory, supply_configuration=False
    )
    base = _accepted_capability_snapshot(Journey.open(Path(receipt.journey_file)))
    original = base.model_dump(mode="json", exclude={"snapshot_id"})
    configured = configuration.runtime_selection.capability
    original["tools"].append(
        next(
            t.model_dump(mode="json") for t in configured.tools if t.tool_id == "python"
        )
    )
    original["tools"].sort(key=lambda t: t["tool_id"])
    for resource in original["resources"]:
        resource["tool_ids"] = sorted(set(resource["tool_ids"]) | {"python"})
    original["runtime_profiles"] = [
        profile.model_dump(mode="json") for profile in configured.runtime_profiles
    ]
    extended = CapabilitySnapshot(
        snapshot_id=canonical_json_digest(original), **original
    )
    configuration = configuration.model_copy(
        update={
            "runtime_selection": ExecutionRuntimeSelection(
                capability=extended, profile_ids=("h100",)
            )
        }
    )
    composition.capability = base
    with use_run_advancement_composition(composition):
        review = advance(ResumeRun(journey_file=receipt.journey_file))
        while (
            review.current_review is not None
            and review.current_review.kind == "planning"
        ):
            question = review.current_review.request
            review = advance(
                AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=question.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select",
                            candidate_id=question.candidates[0].candidate_id,
                        ),
                    ),
                )
            )
    return receipt, composition, configuration, review


class ExecutionApplicationIntegrationTests(RunPrerequisites, unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(
                target, side_effect=AssertionError("offline caller forbids network")
            )
            guard.start()
            self.addCleanup(guard.stop)

    def test_coordinator_opens_the_preparation_review_and_accepts_exact_information(
        self,
    ):
        import socket
        from unittest.mock import patch

        from bio_neb.application.advancement import answer_review_from_json
        from bio_neb.content import canonical_json_digest

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                socket, "create_connection", side_effect=AssertionError("network")
            ),
        ):
            receipt, composition, configuration, review = _preparation_review_run(
                Path(directory).resolve()
            )
            with use_run_advancement_composition(composition):
                self.assertEqual(review.current_review.kind, "preparation")
                request_id = review.current_review.request_event_id
                # Planning closed when only preparation information was missing:
                # no model request follows the last planning fact, and the
                # coordinator opened the Review after the Attempt ended.
                events = Journey.open(Path(receipt.journey_file)).read()
                kinds = [e["event_type"] for e in events]
                last_request = max(
                    i for i, k in enumerate(kinds) if k == "ModelRequestRecorded"
                )
                last_selection = max(
                    i for i, k in enumerate(kinds) if k == "PlanningCandidateSelected"
                )
                self.assertLess(last_request, last_selection)
                opened = next(e for e in events if e["event_id"] == request_id)
                closed = next(
                    e
                    for e in reversed(events)
                    if e["event_type"] == "SliceCheckpointReached"
                )
                self.assertEqual(
                    closed["payload"]["checkpoint"], "execution_planning_ready"
                )
                self.assertEqual(
                    opened["payload"]["attempt_id"], closed["payload"]["attempt_id"]
                )
                self.assertGreater(events.index(opened), events.index(closed))
                self.assertEqual(
                    (opened["payload"]["code"], opened["payload"]["missing_fields"]),
                    (
                        "execution_configuration_required",
                        ["launch", "runtime_selection"],
                    ),
                )
                before = Path(receipt.journey_file).read_bytes()
                # Nobody types the reference byte identity; the answer refuses it.
                bad = configuration.model_dump(mode="json")
                bad["launch"]["accepted_plain_fasta"] = {
                    "byte_count": 1,
                    "sha256": "0" * 64,
                }
                with self.assertRaises(ValueError):
                    advance(
                        answer_review_from_json(
                            review,
                            request_id,
                            json.dumps({"action": "supply", "configuration": bad}),
                        )
                    )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                changed = configuration.model_dump(mode="json")
                capability = changed["runtime_selection"]["capability"]
                next(t for t in capability["tools"] if t["tool_id"] == "pbrun")[
                    "version"
                ] = "unobserved-version"
                capability["runtime_profiles"][0]["h100"]["pbrun_version"] = (
                    "unobserved-version"
                )
                capability["snapshot_id"] = canonical_json_digest(
                    {k: v for k, v in capability.items() if k != "snapshot_id"}
                )
                with self.assertRaises(ValueError):
                    advance(
                        answer_review_from_json(
                            review,
                            request_id,
                            json.dumps({"action": "supply", "configuration": changed}),
                        )
                    )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                answer = answer_review_from_json(
                    review,
                    request_id,
                    json.dumps(
                        {
                            "action": "supply",
                            "configuration": configuration.model_dump(mode="json"),
                        }
                    ),
                )
                ready = advance(answer)
                self.assertEqual(ready.current_review.kind, "execution")
                before = Path(receipt.journey_file).read_bytes()
                self.assertEqual(advance(answer), ready)
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                self.assertEqual(
                    ready.execution_hitl.pending_request.proposal.configuration,
                    configuration,
                )

    def test_a_preparation_request_with_false_lineage_refuses_on_reopen(self):
        # The coordinator's pending request names the planning Attempt that closed
        # ready, cites the current plan and descends from the causal head.
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, review = _preparation_review_run(Path(directory).resolve())
            path = Path(receipt.journey_file)
            original = path.read_bytes()
            events = Journey.open(path).read()
            request_id = review.current_review.request_event_id
            first_attempt = next(
                e["payload"]["attempt_id"]
                for e in events
                if e["event_type"] == "AttemptStarted"
            )
            registration = next(
                e["event_id"]
                for e in events
                if e["event_type"] == "RunInputsRegistered"
            )
            mutations = (
                (
                    "payload",
                    "attempt_id",
                    first_attempt,
                    "preparation request does not follow its closing planning Attempt",
                ),
                (
                    "payload",
                    "evidence_event_ids",
                    [registration],
                    "preparation request does not cite the current plan",
                ),
                (
                    None,
                    "parent_ids",
                    [registration],
                    "preparation request does not descend from its causal head",
                ),
            )
            for where, field, value, reason in mutations:
                with self.subTest(field=field):
                    hostile = copy.deepcopy(events)
                    request = next(e for e in hostile if e["event_id"] == request_id)
                    (request["payload"] if where else request)[field] = value
                    try:
                        path.write_text(
                            "".join(canonical_json(e) + "\n" for e in hostile)
                        )
                        with self.assertRaisesRegex(ValueError, reason):
                            observe(ObserveRun(journey_file=receipt.journey_file))
                    finally:
                        path.write_bytes(original)
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=receipt.journey_file)
                ).current_review.request_event_id,
                request_id,
            )

    def test_a_revision_after_the_preparation_review_reenters_planning(self):
        # The coordinator's preparation request sits between planning Attempts;
        # a revised proposal still re-enters planning through it.
        from bio_neb.application.advancement import answer_review_from_json

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, configuration, review = _preparation_review_run(
                Path(directory).resolve()
            )
            with use_run_advancement_composition(composition):
                ready = advance(
                    answer_review_from_json(
                        review,
                        review.current_review.request_event_id,
                        json.dumps(
                            {
                                "action": "supply",
                                "configuration": configuration.model_dump(mode="json"),
                            }
                        ),
                    )
                )
                original = ready.execution_hitl.pending_request
                before = len(composition.calls)
                revised = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=original.request_event_id,
                        answer=ExecutionReviewAnswer(
                            kind="execution",
                            response=ReviseExecutionResponse(
                                action="revise",
                                proposal_id=original.proposal.proposal_id,
                                instruction="Revalidate recipe and resource evidence.",
                                revalidation_scope="execution_planning",
                            ),
                        ),
                    )
                )
                while revised.current_review.kind == "planning":
                    question = revised.current_review.request
                    revised = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=question.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=question.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
            self.assertGreater(len(composition.calls), before)
            self.assertEqual(revised.current_review.kind, "execution")
            self.assertEqual(
                revised.execution_hitl.pending_request.proposal.supersedes_proposal_id,
                original.proposal.proposal_id,
            )

    def test_real_information_review_approval_dispatches_exact_content_once(self):
        import socket
        from unittest.mock import patch

        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            configured_provider,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("offline test contacted network"),
            ),
        ):
            # The failure section is recorded at registration.
            h100 = {"operation": {"retry_attempts": 2}}
            receipt, composition, configuration = configured_planning_run(
                Path(directory).resolve(),
                failure={"scopes": {"stage": {"h100": h100}}},
            )
            calls = []
            with (
                use_run_advancement_composition(composition),
                use_run_advancement_dependencies(
                    parabricks_infrastructure_factory=configured_provider(
                        receipt.journey_file, calls
                    )
                ),
            ):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
                while (
                    review.current_review is not None
                    and review.current_review.kind == "planning"
                ):
                    question = review.current_review.request
                    review = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=question.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=question.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                self.assertEqual(
                    review.status,
                    "execution_hitl_pending",
                    (review.current_review, composition.calls, review.failure_code),
                )
                request = review.execution_hitl.pending_request
                self.assertEqual(request.proposal.configuration, configuration)
                model_calls = len(composition.calls)
                answer = AnswerReview(
                    journey_file=receipt.journey_file,
                    request_event_id=request.request_event_id,
                    answer=ExecutionReviewAnswer(
                        kind="execution",
                        response=approve_execution(request.proposal),
                    ),
                )
                composition.environ = {}
                # The verified Execution hands over to Investigation in the same
                # call; Investigation resolves the Run's registered selections,
                # whose credential this test withdrew: the fault is retained before
                # the start and holds the Run.
                with self.assertRaises(RunAdvancementRefused) as held:
                    advance(answer)
                self.assertEqual(
                    (held.exception.status, held.exception.code),
                    ("run_held", "model_credential_unavailable"),
                )
                result = observe(ObserveRun(journey_file=receipt.journey_file))
                self.assertEqual(
                    read_completed_parabricks_result(receipt.journey_file).disposition,
                    "succeeded",
                )
                types = [
                    e["event_type"]
                    for e in Journey.open(Path(receipt.journey_file)).read()
                ]
                self.assertNotIn("InvestigationStarted", types)
                self.assertEqual(types.count("RuntimeConfigurationFailed"), 1)
                # Execution tunes each stage from the Run's recorded section.
                recorded = {
                    step["stage"]: step["failure_policy"]
                    for journal in Journey.open(
                        Path(receipt.journey_file)
                    ).run_directory.glob("contacts/*/journal.jsonl")
                    for line in journal.read_text().splitlines()
                    if (
                        step := (json.loads(line).get("domain") or {}).get(
                            "payload", {}
                        )
                    ).get("failure_policy")
                }
                self.assertEqual(
                    set(recorded),
                    # No prebuilt index is bound: the Run prepares on CPU.
                    {"input_storage", "cpu_preparation", "h100"},
                )
                self.assertEqual(
                    recorded["h100"]["surface_overrides"], h100["operation"]
                )
                self.assertEqual(recorded["h100"]["effective"]["retry_attempts"], 2)
                self.assertEqual(recorded["input_storage"]["surface_overrides"], {})
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(composition.calls), model_calls)
                self.assertEqual(
                    calls[0].identity.proposal_id, request.proposal.proposal_id
                )
                self.assertEqual(calls[0].launch, configuration.launch)
                before = Path(receipt.journey_file).read_bytes()
                with patch.object(
                    Journey, "_load_events", side_effect=Journey._load_events
                ) as loads:
                    self.assertEqual(advance(answer), result)
                # Routing an executed Run opens its Journey independently once.
                self.assertEqual(loads.call_count, 1)
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
                self.assertEqual(len(calls), 1)

    def _scripted_investigation(self, investigation_model=None):
        """Fresh Investigation model and sources for this case; nothing else is scripted."""
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        self.investigation_model = investigation_model or SequentialAdaptiveModel()
        self.investigation_sources = SequentialSourceResponses(
            assembly_body=ASSEMBLY_BODY
        )
        self.investigation_dependencies = lambda: use_run_advancement_dependencies(
            environ={"LOCAL_MODEL_KEY": "not-retained"},
            investigation_model_factory=(self.investigation_model.model),
            investigation_source_adapter_factory=lambda: self.investigation_sources,
        )

    def _request_needed(self):
        """The restored Run waiting for an Investigation request after Execution."""
        baseline = self.prerequisite(_REQUEST_NEEDED)
        self._scripted_investigation()
        return SimpleNamespace(journey_file=str(baseline.journey_file)), baseline.value

    @measured_build("execution-application:verified-execution")
    def _verified_execution(
        self,
        root,
        investigation_request,
        *,
        scientific_intent=None,
        scientific_rsid="rs429358",
        investigation_model=None,
        old_journey=False,
    ):
        from contextlib import nullcontext

        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.planning_assignment_fakes import (
            configured_provider,
            planned_before_required_targets,
            prepare_configured_run,
        )
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        # An old Journey, planned before plans named targets, names none (its
        # request-needed path is kept); a current Run's plan always names
        # targets.
        with planned_before_required_targets() if old_journey else nullcontext():
            receipt, review, _ = prepare_configured_run(
                root,
                investigation_request=investigation_request,
                scientific_intent=scientific_intent,
                scientific_rsid=scientific_rsid,
            )
        request = review.execution_hitl.pending_request
        calls = []
        self.investigation_model = investigation_model or SequentialAdaptiveModel()
        self.investigation_sources = SequentialSourceResponses(
            assembly_body=ASSEMBLY_BODY
        )
        self.investigation_dependencies = lambda: use_run_advancement_dependencies(
            environ={"LOCAL_MODEL_KEY": "not-retained"},
            investigation_model_factory=(self.investigation_model.model),
            investigation_source_adapter_factory=lambda: self.investigation_sources,
        )
        with (
            self.investigation_dependencies(),
            use_run_advancement_dependencies(
                parabricks_infrastructure_factory=configured_provider(
                    receipt.journey_file, calls
                )
            ),
        ):
            self.passes_before_approval = _pass_count(receipt.journey_file)
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
        self.assertEqual(len(calls), 1)
        return receipt, result

    def test_verified_execution_without_a_captured_request_reports_request_needed(
        self,
    ):
        from bio_neb.investigation.investigation_request import (
            investigation_request_document_schema,
        )
        from bio_neb.parabricks.result import ParabricksExecutionOutcome

        with tempfile.TemporaryDirectory() as directory:
            receipt, result = self._verified_execution(
                Path(directory).resolve(), None, old_journey=True
            )
            retained = Path(receipt.journey_file).read_bytes()
            observed = observe(ObserveRun(journey_file=receipt.journey_file))
            with self.investigation_dependencies():
                resumed = advance(ResumeRun(journey_file=receipt.journey_file))
            unchanged = Path(receipt.journey_file).read_bytes() == retained
            event_types = [
                e["event_type"] for e in Journey.open(Path(receipt.journey_file)).read()
            ]

        self.assertIsInstance(result, ParabricksExecutionOutcome)
        self.assertEqual(result.disposition, "succeeded")
        self.assertEqual(result.status, "investigation_request_needed")
        self.assertEqual(
            result.schema_version, "bio-neb.investigation-request-needed/1"
        )
        self.assertEqual(
            [action.model_dump(mode="json") for action in result.legal_actions],
            [
                {
                    "action": "supply_investigation_request",
                    "supplied_field": "request_file",
                    "effect": "begin_investigation_with_supplied_request",
                    "request_schema": investigation_request_document_schema(),
                }
            ],
        )
        self.assertEqual(observed, result)
        self.assertEqual(resumed, result)
        self.assertTrue(unchanged)
        self.assertNotIn("InvestigationStarted", event_types)
        self.assertEqual(self.investigation_model.calls, 0)
        self.assertEqual(self.investigation_sources.contact_count, 0)

    def _assert_collection_review(self, journey_file, result):
        events = Journey.open(Path(journey_file)).read()
        event_types = [event["event_type"] for event in events]
        self.assertEqual(result.status, "investigation_review_pending")
        self.assertEqual(result.current_review.kind, "investigation")
        self.assertEqual(event_types.count("InvestigationStarted"), 1)
        self.assertEqual(event_types.count("InvestigationReviewRequested"), 1)
        self.assertEqual(
            result.current_review.request_event_id,
            next(
                event["event_id"]
                for event in events
                if event["event_type"] == "InvestigationReviewRequested"
            ),
        )
        self.assertNotIn("InvestigationAuthorityGranted", event_types)
        self.assertFalse(
            any(kind.startswith("InvestigationObservation") for kind in event_types)
        )
        self.assertEqual(self.investigation_model.calls, 3)
        self.assertEqual(self.investigation_sources.contact_count, 2)
        self.assertNotIn("not-retained", canonical_json(events))
        # Reopen shows the same stop and repeats no effect.
        retained = Path(journey_file).read_bytes()
        self.assertEqual(observe(ObserveRun(journey_file=journey_file)), result)
        with self.investigation_dependencies():
            resumed = advance(ResumeRun(journey_file=journey_file))
        self.assertEqual(resumed.status, "investigation_review_pending")
        self.assertEqual(
            resumed.current_review.request_event_id,
            result.current_review.request_event_id,
        )
        self.assertEqual(Path(journey_file).read_bytes(), retained)
        self.assertEqual(self.investigation_model.calls, 3)
        self.assertEqual(self.investigation_sources.contact_count, 2)

    def test_verified_execution_with_a_captured_request_continues_to_collection_review(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, result = self._verified_execution(
                Path(directory).resolve(),
                {
                    "asserted_by": "researcher:owner",
                    "request": "Investigate rs429358 in this sample.",
                    "target": {"rsid": "rs429358"},
                },
            )
            # The approval's call closes its Pass at the after-Execution child point
            # (ADR 0031) and begins Investigation in a fresh Pass.
            self.assertEqual(
                _pass_count(receipt.journey_file), self.passes_before_approval + 2
            )
            events = Journey.open(Path(receipt.journey_file)).read()
            event_types = [e["event_type"] for e in events]
            self.assertIn("InvestigationRequestRegistered", event_types)
            verified = (
                len(event_types) - 1 - event_types[::-1].index("ExecutionPathRecorded")
            )
            self.assertEqual(
                [
                    (e["event_type"], e["payload"].get("disposition"))
                    for e in events[verified + 1 : verified + 3]
                ],
                [("PassClosed", "completed"), ("PassStarted", None)],
            )
            self.assertEqual(event_types.index("InvestigationStarted"), verified + 3)
            self._assert_collection_review(receipt.journey_file, result)

    def test_fork_from_the_after_execution_point_reaches_investigation_without_execution(
        self,
    ):
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.application.fork import ForkRun, fork_run
        from bio_neb.parabricks.result import ParabricksExecutionOutcome

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = self._verified_execution(root, _CAPTURED_REQUEST)
            parent = Journey.open(Path(receipt.journey_file))
            events = parent.read()
            verified = (
                len(events)
                - 1
                - [e["event_type"] for e in events][::-1].index("ExecutionPathRecorded")
            )
            point = events[verified + 1]
            parent_bytes = Path(receipt.journey_file).read_bytes()
            model_calls = self.investigation_model.calls

            fork = fork_run(
                ForkRun(
                    parent_journey_file=receipt.journey_file,
                    child_run_root=str(root / "child"),
                    point_event_id=point["event_id"],
                )
            )
            idle = observe(ObserveRun(journey_file=fork.child_journey_file))

            def no_execution(*_args, **_kwargs):
                raise AssertionError("the child reran Execution")

            with (
                self.investigation_dependencies(),
                use_run_advancement_dependencies(
                    parabricks_infrastructure_factory=no_execution
                ),
            ):
                resumed = advance(ResumeRun(journey_file=fork.child_journey_file))
            child = Journey.open(Path(fork.child_journey_file))
            own = [
                e["event_type"]
                for e in child.read()
                if e["journey_id"] == child.journey_id
            ]
            parent_after = Path(receipt.journey_file).read_bytes()

        self.assertEqual(
            (point["event_type"], point["payload"]["disposition"]),
            ("PassClosed", "completed"),
        )
        self.assertEqual(fork.origin.point_event_id, point["event_id"])
        # The idle child observes as the parent did at that point.
        self.assertIsInstance(idle, ParabricksExecutionOutcome)
        self.assertEqual(idle.disposition, "succeeded")
        self.assertEqual(resumed.status, "investigation_review_pending")
        self.assertEqual(own[:3], ["RunForked", "PassStarted", "InvestigationStarted"])
        self.assertNotIn("ExecutionPathRecorded", own)
        self.assertNotIn("ExecutionHitlResponded", own)
        self.assertEqual(parent_after, parent_bytes)
        self.assertGreater(self.investigation_model.calls, model_calls)

    def test_the_researchers_goal_targets_and_order_reach_every_member(self):
        # A recorded Run asked "Tell me whether HG001 carries rs429358, rs7412
        # and rs4149056." Its Investigation got rs4149056 first, and every member
        # was told the original intent was "rs4149056".
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        from bio_neb.application.advancement import (
            use_run_advancement_composition,
            use_run_advancement_dependencies,
        )
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            configured_provider,
        )
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        goal = "Tell me whether HG001 carries rs429358, rs7412 and rs4149056."

        class Members(SequentialAdaptiveModel):
            """Keeps each member's assignment context as the model receives it."""

            def __init__(self):
                super().__init__()
                self.contexts = {}

            async def respond(self, messages, info):
                for message in ModelMessagesTypeAdapter.dump_python(
                    messages, mode="json"
                ):
                    for part in message["parts"]:
                        if part["part_kind"] != "user-prompt" or not isinstance(
                            part["content"], str
                        ):
                            continue
                        try:
                            context = json.loads(part["content"])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(context, dict) and "user_assertion" in context:
                            self.contexts[context["collection_member_id"]] = context
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, composition, _ = configured_planning_run(
                root, scientific_intent=goal, scientific_rsid=None
            )
            scripted = composition.respond

            async def respond(messages, info):
                # The model records the goal and the three targets in its own
                # order (reversed), under keys whose sorting reorders them again.
                response = await scripted(messages, info)
                part = response.parts[0]
                if part.tool_name == "record_proposal_requirements":
                    requirements = [
                        item
                        for item in part.args["requirements"]
                        if item["kind"] != "scientific_investigation"
                    ]
                    next(
                        item for item in requirements if item["kind"] == "research_goal"
                    )["knowledge"] = {"state": "asserted", "quote": goal}
                    for key, rsid in (
                        ("target_c", "rs4149056"),
                        ("target_b", "rs7412"),
                        ("target_a", "rs429358"),
                    ):
                        requirements.append(
                            {
                                "kind": "scientific_investigation",
                                "subject_key": key,
                                "domain": "scientific",
                                "meaning": f"Whether HG001 carries {rsid}.",
                                "knowledge": {
                                    "state": "asserted",
                                    "value": rsid,
                                    "quote": rsid,
                                },
                                "acquisition_route": "validate_with_tool",
                                "validation": {
                                    "kind": "tool_evidence",
                                    "requirement": "Observe it after Execution.",
                                },
                                "investigation_purpose": "known_variant",
                            }
                        )
                    part.args["requirements"] = requirements
                return response

            composition.respond = respond
            members = Members()
            with use_run_advancement_composition(composition):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
                while (
                    review.current_review is not None
                    and review.current_review.kind == "planning"
                ):
                    question = review.current_review.request
                    review = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=question.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=question.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                approval = review.execution_hitl.pending_request
                with use_run_advancement_dependencies(
                    investigation_model_factory=members.model,
                    investigation_source_adapter_factory=lambda: (
                        SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                    ),
                    parabricks_infrastructure_factory=configured_provider(
                        receipt.journey_file, []
                    ),
                ):
                    result = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=approval.request_event_id,
                            answer=ExecutionReviewAnswer(
                                kind="execution",
                                response=approve_execution(approval.proposal),
                            ),
                        )
                    )
            assertion = next(
                e
                for e in Journey.open(Path(receipt.journey_file)).read()
                if e["event_type"] == "InvestigationStarted"
            )["payload"]["assertion"]

        self.assertEqual(result.status, "investigation_review_pending")
        # The researcher's order and words; the goal travels beside them.
        self.assertEqual(
            [(item["request_text"], item["rsid"]) for item in assertion["requests"]],
            [
                ("rs429358", "rs429358"),
                ("rs7412", "rs7412"),
                ("rs4149056", "rs4149056"),
            ],
        )
        self.assertEqual(assertion["research_goals"], [goal])
        self.assertEqual(assertion["asserted_by"], "researcher")
        # Every member's original intent is the goal; each keeps its own request.
        self.assertEqual(len(members.contexts), 3)
        self.assertEqual(
            sorted(
                context["user_assertion"]["rsid"]
                for context in members.contexts.values()
            ),
            ["rs4149056", "rs429358", "rs7412"],
        )
        for context in members.contexts.values():
            with self.subTest(member=context["user_assertion"]["rsid"]):
                self.assertEqual(
                    context["assignment"]["original_intent"]["research_goal"], [goal]
                )
                self.assertNotIn("requests", context["user_assertion"])

    def test_original_scientific_intent_reaches_collection_without_request_file(self):
        from tests.sequential_investigation_support import _AdaptiveInvestigationModel

        question = "Investigate rs429358 in this sample."
        with tempfile.TemporaryDirectory() as directory:
            receipt, result = self._verified_execution(
                Path(directory).resolve(),
                None,
                scientific_intent=question,
                # The plan types the rsID its quote gives (no code reads it).
                scientific_rsid="rs429358",
                investigation_model=_AdaptiveInvestigationModel(),
            )
            self._assert_collection_review(receipt.journey_file, result)
            journey = Journey.open(Path(receipt.journey_file))
            captured = result.investigation_request
            self.assertEqual(captured.request.items[0].request_text, question)
            # The plan typed the quote's rsID (a retired fallback).
            self.assertEqual(captured.request.items[0].rsid, "rs429358")
            self.assertEqual(
                result.investigation.fact_slots[0].requested_value, "rs429358"
            )
            self.assertIsNone(captured.request_file)
            prompt = next(
                e
                for e in journey.read()
                if e["event_type"] == "ResearchRequestRegistered"
            )
            self.assertEqual(
                prompt["payload"]["prompt"],
                "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.\n"
                + question,
            )
            self.assertIn(prompt["event_id"], captured.source_event_ids)
            started = next(
                e for e in journey.read() if e["event_type"] == "InvestigationStarted"
            )
            self.assertIn(
                prompt["event_id"], started["payload"]["assertion"]["predecessor_ids"]
            )
            plan = next(
                e
                for e in journey.read()
                if e["event_type"] == "ProposalRequirementsIdentified"
            )
            self.assertIn(
                plan["event_id"], started["payload"]["assertion"]["predecessor_ids"]
            )

            from bio_neb.investigation.investigation import (
                UserAssertion,
                project_investigation,
            )
            from bio_neb.recording import read_evidence

            evidence = read_evidence(journey)
            prefix = copy.deepcopy(evidence[: evidence.index(started) + 1])
            original = started["payload"]["assertion"]
            prefix[-1]["payload"]["assertion"] = UserAssertion.create(
                asserted_by=original["asserted_by"],
                request_text=original["request_text"],
                rsid=original["rsid"],
                predecessor_ids=tuple(original["predecessor_ids"][:2]),
            ).model_dump(mode="json")
            with self.assertRaisesRegex(ValueError, "request origin"):
                project_investigation(prefix)

    def test_answered_investigation_review_resumes_investigation_not_execution(self):
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.effort import ExtendAgentEffortResponse
        from bio_neb.application.advancement import (
            AgentEffortReviewAnswer,
            BeginInvestigation,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # A verified Run whose plan names the request, resting at its
            # after-Execution child point; it begins under a small effort limit.
            receipt = SimpleNamespace(
                journey_file=str(
                    self.prerequisite(
                        "executed",
                        investigation_request=_CAPTURED_REQUEST,
                        writable=True,
                    ).journey_file
                )
            )
            self._scripted_investigation()
            with self.investigation_dependencies():
                stopped = advance(
                    BeginInvestigation(
                        journey_file=receipt.journey_file,
                        request_files=(str(_request_file(root)),),
                        invocation_overrides=AgentInvocationOverrides(model_requests=2),
                    )
                )
                self.assertEqual(stopped.status, "effort_review_pending")
                request = stopped.agent_effort_review.pending_request
                # The recorded Execution authority must not claim the answered
                # Investigation Attempt's re-entry.
                resumed = advance(
                    AnswerReview(
                        journey_file=receipt.journey_file,
                        request_event_id=request.request_event_id,
                        answer=AgentEffortReviewAnswer(
                            kind="agent_effort",
                            response=ExtendAgentEffortResponse(
                                action="extend", additional_requests=2
                            ),
                        ),
                    )
                )
            self.assertEqual(resumed.status, "investigation_review_pending")
            self.assertEqual(resumed.current_review.kind, "investigation")
            # The first resumed request keeps its pre-stop findings
            # (invocation.current_instructions); it cites the assembly report by
            # the source_event_id its Tool return carries, so no repair is spent.
            self.assertEqual(self.investigation_model.calls, 3)

    def test_answered_member_agent_work_reject_closes_the_collection_review(self):
        """The native answer of a member's Agent work reject settles that member and
        advances to the collection's Review, as a member question's reject does
        (#399: live, the answer stopped at investigation_work_ready)."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.agentic_journey.agent_work_review import RejectAgentWorkResponse
        from bio_neb.application.advancement import (
            AgentWorkReviewAnswer,
            BeginInvestigation,
        )
        from tests.model_endpoint_fakes import ScriptedModelEndpoint
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            _current_member_messages,
        )

        scientific = SequentialAdaptiveModel()

        async def respond(messages, info):
            request, *_ = _current_member_messages(messages)
            if request.get("rsid") == "rs7412":
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "compile_proposal",
                            {"outcome": "complete"},
                            tool_call_id=f"bad-{uuid.uuid4().hex}",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request_file = root / "investigation-request.json"
            request_file.write_text(
                canonical_json(
                    {
                        "schema_version": "bio-neb.investigation-request/2",
                        "asserted_by": "researcher:owner",
                        "requests": [
                            {
                                "request": f"Investigate {rsid} in this sample.",
                                "target": {"rsid": rsid},
                            }
                            for rsid in ("rs429358", "rs7412")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            receipt, _ = self._request_needed()
            self._scripted_investigation(
                SimpleNamespace(model=ScriptedModelEndpoint(respond))
            )
            with self.investigation_dependencies():
                stopped = advance(
                    BeginInvestigation(
                        journey_file=str(receipt.journey_file),
                        request_files=(str(request_file),),
                    )
                )
                self.assertEqual(stopped.current_review.kind, "agent_work")
                request = stopped.current_review.request
                reviewed = advance(
                    AnswerReview(
                        journey_file=str(receipt.journey_file),
                        request_event_id=request.request_event_id,
                        answer=AgentWorkReviewAnswer(
                            kind="agent_work",
                            response=RejectAgentWorkResponse(
                                action="reject",
                                reason="Go on without rs7412.",
                                asserted_by="researcher:owner",
                            ),
                        ),
                    )
                )
            self.assertEqual(reviewed.status, "investigation_review_pending")
            self.assertEqual(reviewed.current_review.kind, "investigation")
            proposal = reviewed.investigation.proposals[-1]
            self.assertEqual(
                [item.target.one_based_position for item in proposal.work_items],
                [44908684],
            )
            rejected = next(
                member
                for member in proposal.request_members
                if member.request.rsid == "rs7412"
            )
            self.assertEqual(
                rejected.rejection.request_event_id, request.request_event_id
            )

    def test_supplied_request_at_request_needed_reaches_collection_review(self):
        from bio_neb.application.advancement import BeginInvestigation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, result = self._request_needed()
            self.assertEqual(result.status, "investigation_request_needed")
            request_file = _request_file(root)
            passes = _pass_count(receipt.journey_file)
            with self.investigation_dependencies():
                supplied = advance(
                    BeginInvestigation(
                        journey_file=receipt.journey_file,
                        request_files=(str(request_file),),
                    )
                )
            self.assertEqual(_pass_count(receipt.journey_file), passes + 1)
            self._assert_collection_review(receipt.journey_file, supplied)
            frozen = Path(receipt.journey_file).read_bytes()
            self.assertEqual(
                advance(
                    BeginInvestigation(
                        journey_file=receipt.journey_file,
                        request_files=(str(request_file),),
                    )
                ),
                supplied,
            )
            self.assertEqual(Path(receipt.journey_file).read_bytes(), frozen)
            request_file.write_text(
                json.dumps(
                    {
                        "asserted_by": "researcher:owner",
                        "request": "Recheck rs429358.",
                        "target": {"rsid": "rs429358"},
                    }
                )
            )
            from bio_neb.investigation.investigation_commands import (
                InvestigationApplicationRefused,
            )

            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                advance(
                    BeginInvestigation(
                        journey_file=receipt.journey_file,
                        request_files=(str(request_file),),
                    )
                )
            self.assertEqual(conflict.exception.code, "investigation_request_conflict")
            self.assertEqual(Path(receipt.journey_file).read_bytes(), frozen)

    def test_cli_investigation_begin_supplies_the_request_inside_one_pass(self):
        import io
        import os
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch

        from bio_neb.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, result = self._request_needed()
            self.assertEqual(result.status, "investigation_request_needed")
            request_file = _request_file(root)
            passes = _pass_count(receipt.journey_file)
            output = io.StringIO()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "not-retained"}),
                redirect_stdout(output),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "investigation-begin",
                        "--journey-file",
                        receipt.journey_file,
                        "--request-file",
                        str(request_file),
                    ],
                    investigation_model_factory=(self.investigation_model.model),
                    investigation_source_adapter_factory=(
                        lambda: self.investigation_sources
                    ),
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output.getvalue())["status"], "investigation_review_pending"
            )
            events = Journey.open(Path(receipt.journey_file)).read()
            appended = events[
                next(
                    index
                    for index, event in enumerate(events)
                    if event["event_type"] == "InvestigationStarted"
                )
                - 1 :
            ]
            # The Pass opens before the first Investigation write and closes last.
            self.assertEqual(appended[0]["event_type"], "PassStarted")
            self.assertEqual(appended[-1]["event_type"], "PassClosed")
            self.assertEqual(_pass_count(receipt.journey_file), passes + 1)
            self.assertEqual(self.investigation_model.calls, 3)
            self.assertEqual(self.investigation_sources.contact_count, 2)

    def test_cli_answer_dispatches_and_replays_without_a_launch_document(self):
        import io
        import os
        import socket
        from contextlib import redirect_stderr, redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import patch

        from bio_neb.cli import main
        from bio_neb.parabricks.result import read_completed_parabricks_result
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            configured_provider,
        )
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SourceResponsesOnly,
            _AdaptiveInvestigationModel,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                socket, "create_connection", side_effect=AssertionError("network")
            ),
        ):
            root = Path(directory).resolve()
            _, composition, _ = configured_planning_run(root)
            with use_run_advancement_composition(composition):
                with (
                    redirect_stdout(io.StringIO()) as begun,
                    redirect_stderr(io.StringIO()),
                ):
                    code = main(
                        [
                            "begin",
                            "--run-root",
                            str(root / "cli-runs"),
                            "--prompt-file",
                            str(root / "prompt.txt"),
                            "--runtime-config",
                            str(root / "runtime.json"),
                            "--fastq",
                            "read_1=" + str(root / "r1.fastq"),
                            "--fastq",
                            "read_2=" + str(root / "r2.fastq"),
                        ],
                        parabricks_infrastructure_factory=lambda *_: self.fail(
                            "pre-review construction"
                        ),
                    )
                self.assertEqual(code, 0)
                receipt = SimpleNamespace(
                    journey_file=json.loads(begun.getvalue())["journey_file"]
                )
                review = observe(ObserveRun(journey_file=receipt.journey_file))
                while (
                    review.current_review and review.current_review.kind == "planning"
                ):
                    q = review.current_review.request
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "answer",
                                    "--journey-file",
                                    receipt.journey_file,
                                    "--request-event-id",
                                    q.request_event_id,
                                    "--response-json",
                                    json.dumps(
                                        {
                                            "action": "select",
                                            "candidate_id": q.candidates[
                                                0
                                            ].candidate_id,
                                        }
                                    ),
                                ],
                                parabricks_infrastructure_factory=lambda *_: self.fail(
                                    "pre-review construction"
                                ),
                            ),
                            0,
                        )
                    review = observe(ObserveRun(journey_file=receipt.journey_file))
                before = Path(receipt.journey_file).read_bytes()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        main(
                            ["resume", "--journey-file", receipt.journey_file],
                            parabricks_infrastructure_factory=lambda *_: self.fail(
                                "pending review construction"
                            ),
                        ),
                        0,
                    )
                self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            request = review.execution_hitl.pending_request
            response = Path(directory) / "response.json"
            response.write_text(json.dumps(approval_json(request.proposal)))
            argv = [
                "answer",
                "--journey-file",
                receipt.journey_file,
                "--request-event-id",
                request.request_event_id,
                "--response-file",
                str(response),
            ]
            calls = []
            output = io.StringIO()
            # The answer's call verifies Execution and begins Investigation from
            # the plan's target, with offline model and source adapters.
            investigation = {
                "investigation_model_factory": _AdaptiveInvestigationModel().model,
                "investigation_source_adapter_factory": lambda: SourceResponsesOnly(
                    assembly_body=ASSEMBLY_BODY
                ),
            }
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "not-retained"}),
                redirect_stdout(output),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        argv,
                        parabricks_infrastructure_factory=configured_provider(
                            receipt.journey_file, calls
                        ),
                        **investigation,
                    ),
                    0,
                )
            view = json.loads(output.getvalue())
            self.assertEqual(
                read_completed_parabricks_result(receipt.journey_file).disposition,
                "succeeded",
            )
            self.assertEqual(view["status"], "investigation_review_pending")
            before = Path(receipt.journey_file).read_bytes()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "not-retained"}),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        argv,
                        parabricks_infrastructure_factory=lambda *_: self.fail(
                            "replayed effect"
                        ),
                        investigation_model_factory=lambda *_: self.fail(
                            "replayed model"
                        ),
                    ),
                    0,
                )
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            self.assertEqual(len(calls), 1)

    def test_approval_replay_after_prebinding_refusal_does_not_retry_dispatch(self):
        from unittest.mock import patch

        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.parabricks.plan import ExecutionCompositionRefused

        baseline = self.prerequisite("review-pending")
        receipt = SimpleNamespace(journey_file=str(baseline.journey_file))
        request = baseline.value.execution_hitl.pending_request
        answer = AnswerReview(
            journey_file=receipt.journey_file,
            request_event_id=request.request_event_id,
            answer=ExecutionReviewAnswer(
                kind="execution",
                response=approve_execution(request.proposal),
            ),
        )
        with (
            use_run_advancement_dependencies(
                parabricks_infrastructure_factory=lambda *_: self.fail(
                    "unbound execution contacted provider"
                )
            ),
            patch(
                "bio_neb.parabricks.operation.plan_execution",
                side_effect=ExecutionCompositionRefused(
                    "lineage", "injected_prebinding_refusal"
                ),
            ) as compose,
        ):
            with self.assertRaises(ExecutionCompositionRefused) as refused:
                advance(answer)
            self.assertEqual(
                (refused.exception.code, refused.exception.details),
                ("lineage", ("injected_prebinding_refusal",)),
            )
            events = Journey.open(Path(receipt.journey_file)).read()
            self.assertEqual(
                sum(e["event_type"] == "ExecutionHitlResponded" for e in events), 1
            )
            self.assertFalse(
                any(e["event_type"] == "ExecutionPathRecorded" for e in events)
            )
            before = Path(receipt.journey_file).read_bytes()
            advance(answer)
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            self.assertEqual(compose.call_count, 1)

    def test_dispatch_interruption_replay_conflict_and_stale_answers_do_not_repeat_effects(
        self,
    ):
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from bio_neb.parabricks.execution_proposal import ExecutionHitlResponseRefused

        baseline = self.prerequisite("review-pending")
        receipt = SimpleNamespace(journey_file=str(baseline.journey_file))
        request = baseline.value.execution_hitl.pending_request
        answer = AnswerReview(
            journey_file=receipt.journey_file,
            request_event_id=request.request_event_id,
            answer=ExecutionReviewAnswer(
                kind="execution",
                response=approve_execution(request.proposal),
            ),
        )
        calls = []

        def unavailable(*args):
            calls.append(args)
            raise ValueError(
                "scripted infrastructure unavailable before provider contacts"
            )

        with use_run_advancement_dependencies(
            parabricks_infrastructure_factory=unavailable
        ):
            stale = answer.model_copy(update={"request_event_id": str(uuid.uuid4())})
            before = Path(receipt.journey_file).read_bytes()
            with self.assertRaises(ExecutionHitlResponseRefused) as refused:
                advance(stale)
            self.assertEqual(refused.exception.code, "execution_hitl_response_stale")
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            # The untyped infrastructure fault holds the Run.
            with self.assertRaisesRegex(
                RunAdvancementRefused, "coordinator_unhandled_error"
            ) as held:
                advance(answer)
            self.assertIn("scripted infrastructure", str(held.exception.__cause__))
            before = Path(receipt.journey_file).read_bytes()
            advance(answer)
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
            conflict = answer.model_copy(
                update={
                    "answer": ExecutionReviewAnswer(
                        kind="execution",
                        response=RejectExecutionResponse(
                            action="reject",
                            proposal_id=request.proposal.proposal_id,
                        ),
                    )
                }
            )
            with self.assertRaises(ExecutionHitlResponseRefused) as refused:
                advance(conflict)
            self.assertEqual(refused.exception.code, "execution_hitl_response_conflict")
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
        self.assertEqual(len(calls), 1)

    def test_preparation_requires_execution_settings_before_opening_review(self):
        from bio_neb.parabricks.preparation import prepare_execution

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _ready_journey(Path(directory).resolve())
            journey = Journey.open(Path(receipt.journey_file))
            before = journey.path.read_bytes()
            prepared = prepare_execution(journey)
            self.assertEqual(prepared.status, "missing_information")
            self.assertEqual(prepared.code, "execution_configuration_required")
            self.assertEqual(journey.path.read_bytes(), before)

    def test_exact_approval_is_the_only_retained_authority_fact(self):
        from bio_neb.parabricks.planning_commands import (
            RespondExecutionHitlCommand,
            respond_execution_hitl,
        )

        baseline = self.prerequisite("review-pending")
        receipt = SimpleNamespace(journey_file=str(baseline.journey_file))
        request = baseline.value.execution_hitl.pending_request
        # The approve action binds this exact proposal.
        approve = next(
            item for item in request.legal_actions if item.action == "approve"
        )
        self.assertEqual(
            approve.response_schema["properties"]["proposal_id"]["const"],
            request.proposal.proposal_id,
        )
        receipt_of_answer = respond_execution_hitl(
            RespondExecutionHitlCommand(
                journey_file=receipt.journey_file,
                request_event_id=request.request_event_id,
                response=approve_execution(request.proposal),
            )
        )
        events = Journey.open(Path(receipt.journey_file)).read()
        self.assertFalse(
            any(e["event_type"] == "ExecutionAuthorityGranted" for e in events)
        )
        answer = next(e for e in events if e["event_type"] == "ExecutionHitlResponded")
        self.assertTrue(answer["payload"]["authorizes_execution"])
        self.assertEqual(receipt_of_answer.response_event_id, answer["event_id"])
        # The compiled package is retained once, exactly as the proposal binds it.
        compiled = [e for e in events if e["event_type"] == "ExecutionPackageCompiled"]
        self.assertEqual(len(compiled), 1)
        self.assertEqual(
            compiled[0]["payload"]["package"],
            request.proposal.execution_package.model_dump(mode="json"),
        )
        retained = observe(ObserveRun(journey_file=receipt.journey_file))
        self.assertEqual(
            retained.execution_hitl.authorities[0].approval_event_id,
            answer["event_id"],
        )

    def test_the_model_has_no_preparation_question_tool(self):
        # The coordinator opens the preparation Review itself.
        from bio_neb.agentic_journey.tool_execution import stage_tool_names

        contracts = function_tool_contracts()
        names = {item.tool_name for item in contracts}
        self.assertEqual(
            len({(i.agent_role, i.tool_name) for i in contracts}), len(contracts)
        )
        self.assertNotIn("execute_approved_parabricks", names)
        self.assertNotIn("request_preparation_information", names)
        self.assertNotIn(
            "PreparationInformationRequested",
            {item.stage_result_event_type for item in contracts},
        )
        self.assertNotIn(
            "request_preparation_information",
            stage_tool_names("planning_agent", "execution_planning"),
        )
        for item in contracts:
            self.assertLessEqual(item.complementary_recognized_tool_names, names)

    def test_a_listed_model_selection_is_carried_through_the_adaptive_assignment(
        self,
    ):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            QWEN_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )
        from tests.planning_assignment_fakes import prepare_configured_run

        # A model the registry lists, here a test entry with Qwen's plain adapter
        # (an unlisted model is refused).
        minimax = replace(
            QWEN_ENDPOINT_POLICY,
            settings_selection=QWEN_ENDPOINT_POLICY.settings_selection.model_copy(
                update={"model": "MiniMaxAI/MiniMax-M3"}
            ),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            use_model_registry(model_registry().replacing(minimax)),
        ):
            receipt, review, _ = prepare_configured_run(
                Path(directory).resolve(), model="MiniMaxAI/MiniMax-M3"
            )
            events = Journey.open(Path(receipt.journey_file)).read()
            requests = [e for e in events if e["event_type"] == "ModelRequestRecorded"]
            self.assertTrue(requests)
            self.assertEqual(
                {e["payload"]["nebius_model_endpoint"]["model"] for e in requests},
                {"MiniMaxAI/MiniMax-M3"},
            )
            self.assertEqual(
                {a.agent_role for a in review.attempts}, {"planning_agent"}
            )

    def test_current_planning_selection_and_role_corruption_refuse_on_reopen(self):
        baseline = self.prerequisite("review-pending")
        receipt = SimpleNamespace(journey_file=str(baseline.journey_file))
        path = Path(receipt.journey_file)
        original = path.read_bytes()
        events = Journey.open(path).read()
        mutations = (
            ("PlanningCandidateSelected", "authority_event_id", str(uuid.uuid4())),
            ("ToolCallValidated", "attempt_id", str(uuid.uuid4())),
            ("AttemptStarted", "agent_role", "investigation_planner"),
        )
        for kind, field, value in mutations:
            with self.subTest(kind=kind, field=field):
                hostile = copy.deepcopy(events)
                next(e for e in hostile if e["event_type"] == kind)["payload"][
                    field
                ] = value
                try:
                    path.write_text("".join(canonical_json(e) + "\n" for e in hostile))
                    with self.assertRaises(ValueError):
                        observe(ObserveRun(journey_file=receipt.journey_file))
                finally:
                    path.write_bytes(original)
        self.assertEqual(
            observe(ObserveRun(journey_file=receipt.journey_file)).status,
            "execution_hitl_pending",
        )

    def test_full_planning_binds_discovered_source_identity_without_remote_receipt(
        self,
    ):
        from tests.planning_assignment_fakes import prepare_configured_run
        from tests.reference_fakes import SOURCE_MD5

        with tempfile.TemporaryDirectory() as directory:
            receipt, review, _ = prepare_configured_run(Path(directory).resolve())
            proposal = review.execution_hitl.pending_request.proposal
            self.assertEqual(
                proposal.reference_source_admission.expected_checksum, SOURCE_MD5
            )
            self.assertEqual(
                proposal.execution_package.reference.source_locator,
                "https://ftp.ncbi.nlm.nih.gov/genomes/all/offline/"
                "GCF_000001405.40_genomic.fna.gz",
            )
            self.assertFalse(
                any(
                    e["event_type"] == "ReferenceSourceAcquired"
                    for e in Journey.open(Path(receipt.journey_file)).read()
                )
            )

    def test_full_adaptive_composition_stops_at_genuine_planning_review(self):
        from tests.planning_assignment_fakes import configured_planning_run

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _ = configured_planning_run(Path(directory).resolve())
            with use_run_advancement_composition(composition):
                review = advance(ResumeRun(journey_file=receipt.journey_file))
            self.assertEqual(review.current_review.kind, "planning")
            self.assertIsNone(
                review.execution_hitl.pending_request if review.execution_hitl else None
            )
            self.assertFalse(
                any(
                    e["event_type"] == "ExecutionProposalSubmitted"
                    for e in Journey.open(Path(receipt.journey_file)).read()
                )
            )

    def test_recipe_revision_reopens_same_assignment_and_keeps_prior_proposal(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.planning_assignment_fakes import (
            configured_planning_run,
            configured_provider,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, composition, _ = configured_planning_run(Path(directory).resolve())

            def answer_questions(review):
                while (
                    review.current_review and review.current_review.kind == "planning"
                ):
                    q = review.current_review.request
                    review = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=q.request_event_id,
                            answer=PlanningReviewAnswer(
                                kind="planning",
                                response=SelectPlanningHitlResponse(
                                    action="select",
                                    candidate_id=q.candidates[0].candidate_id,
                                ),
                            ),
                        )
                    )
                return review

            calls = []
            with (
                use_run_advancement_composition(composition),
                use_run_advancement_dependencies(
                    parabricks_infrastructure_factory=configured_provider(
                        receipt.journey_file, calls
                    )
                ),
            ):
                original = answer_questions(
                    advance(ResumeRun(journey_file=receipt.journey_file))
                ).execution_hitl.pending_request
                before = len(composition.calls)
                revised = answer_questions(
                    advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=original.request_event_id,
                            answer=ExecutionReviewAnswer(
                                kind="execution",
                                response=ReviseExecutionResponse(
                                    action="revise",
                                    proposal_id=original.proposal.proposal_id,
                                    instruction="Revalidate recipe and resource evidence.",
                                    revalidation_scope="execution_planning",
                                ),
                            ),
                        )
                    )
                )
                current = revised.execution_hitl.pending_request
                self.assertEqual(
                    current.proposal.supersedes_proposal_id,
                    original.proposal.proposal_id,
                )
                self.assertGreater(len(composition.calls), before)
                self.assertFalse(calls)
                events = Journey.open(Path(receipt.journey_file)).read()
                self.assertEqual(
                    next(
                        e["payload"]["proposal"]
                        for e in events
                        if e["event_id"] == original.request_event_id
                    ),
                    original.proposal.model_dump(mode="json"),
                )
                # The approval verifies Execution and begins Investigation from
                # the plan's target, with offline model and source adapters.
                from bio_neb.parabricks.result import read_completed_parabricks_result
                from tests.sequential_investigation_support import (
                    ASSEMBLY_BODY,
                    SourceResponsesOnly,
                    _AdaptiveInvestigationModel,
                )

                with use_run_advancement_dependencies(
                    investigation_model_factory=_AdaptiveInvestigationModel().model,
                    investigation_source_adapter_factory=lambda: SourceResponsesOnly(
                        assembly_body=ASSEMBLY_BODY
                    ),
                ):
                    outcome = advance(
                        AnswerReview(
                            journey_file=receipt.journey_file,
                            request_event_id=current.request_event_id,
                            answer=ExecutionReviewAnswer(
                                kind="execution",
                                response=approve_execution(current.proposal),
                            ),
                        )
                    )
                self.assertEqual(
                    read_completed_parabricks_result(receipt.journey_file).disposition,
                    "succeeded",
                )
                self.assertEqual(outcome.status, "investigation_review_pending")
                self.assertEqual(len(calls), 1)

    def test_reject_is_terminal_and_reopens_without_provider_construction(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies

        baseline = self.prerequisite("review-pending")
        receipt = SimpleNamespace(journey_file=str(baseline.journey_file))
        pending = baseline.value.execution_hitl.pending_request
        answer = AnswerReview(
            journey_file=receipt.journey_file,
            request_event_id=pending.request_event_id,
            answer=ExecutionReviewAnswer(
                kind="execution",
                response=RejectExecutionResponse(
                    action="reject", proposal_id=pending.proposal.proposal_id
                ),
            ),
        )
        with use_run_advancement_dependencies(
            parabricks_infrastructure_factory=lambda *_: self.fail("rejected execution")
        ):
            rejected = advance(answer)
            self.assertEqual(rejected.status, "finalization_review_pending")
            self.assertEqual(
                rejected.finalization.pending_request.basis.outcome,
                "execution_rejected",
            )
            before = Path(receipt.journey_file).read_bytes()
            self.assertEqual(advance(answer), rejected)
            self.assertEqual(Path(receipt.journey_file).read_bytes(), before)
        self.assertFalse(rejected.execution_hitl.authorities)
        # The one retained decision answers the exact request with this response.
        (decision,) = (
            e
            for e in Journey.open(Path(receipt.journey_file)).read()
            if e["event_type"] == "ExecutionHitlResponded"
        )
        self.assertEqual(decision["parent_ids"], [pending.request_event_id])
        self.assertEqual(
            decision["payload"]["request_event_id"], pending.request_event_id
        )
        self.assertEqual(
            decision["payload"]["response"],
            answer.answer.response.model_dump(mode="json"),
        )
        self.assertEqual(
            observe(ObserveRun(journey_file=receipt.journey_file)), rejected
        )


def _request_needed_after_execution(root):
    """Build: an old Journey (planned before #342, no target) approves one
    Execution without a captured request; it verifies and waits for the request."""
    from bio_neb.application.advancement import use_run_advancement_dependencies
    from tests.planning_assignment_fakes import (
        configured_provider,
        planned_before_required_targets,
        prepare_configured_run,
    )

    with planned_before_required_targets():
        receipt, review, _ = prepare_configured_run(root)
    approval = review.execution_hitl.pending_request
    calls = []
    with use_run_advancement_dependencies(
        parabricks_infrastructure_factory=configured_provider(
            receipt.journey_file, calls
        ),
    ):
        result = advance(
            AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=approval.request_event_id,
                answer=ExecutionReviewAnswer(
                    kind="execution",
                    response=approve_execution(approval.proposal),
                ),
            )
        )
    if len(calls) != 1 or result.status != "investigation_request_needed":
        raise AssertionError("the baseline did not wait for an Investigation request")
    return Path(receipt.journey_file), result


_REQUEST_NEEDED = State(
    "request-needed-after-execution", _request_needed_after_execution
)


def _investigation_review_after_execution(root):
    """Build: approve one Execution, which verifies and opens Investigation review."""
    from bio_neb.application.advancement import use_run_advancement_dependencies
    from tests.planning_assignment_fakes import (
        configured_provider,
        prepare_configured_run,
    )
    from tests.sequential_investigation_support import (
        ASSEMBLY_BODY,
        SequentialAdaptiveModel,
        SequentialSourceResponses,
    )

    receipt, review, _ = prepare_configured_run(
        root, investigation_request=_CAPTURED_REQUEST
    )
    approval = review.execution_hitl.pending_request
    model = SequentialAdaptiveModel()
    sources = SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
    calls = []
    with use_run_advancement_dependencies(
        environ={"LOCAL_MODEL_KEY": "not-retained"},
        investigation_model_factory=model.model,
        investigation_source_adapter_factory=lambda: sources,
        parabricks_infrastructure_factory=configured_provider(
            receipt.journey_file, calls
        ),
    ):
        result = advance(
            AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=approval.request_event_id,
                answer=ExecutionReviewAnswer(
                    kind="execution",
                    response=approve_execution(approval.proposal),
                ),
            )
        )
    if len(calls) != 1 or result.status != "investigation_review_pending":
        raise AssertionError("the baseline did not reach Investigation review")
    request = result.current_review.request
    return Path(receipt.journey_file), (
        request.request_event_id,
        request.proposal.proposal_id,
    )


_INVESTIGATION_REVIEW = State(
    "investigation-review-after-execution", _investigation_review_after_execution
)


class ExecutionHoldTests(RunPrerequisites, unittest.TestCase):
    """Our bug in an Execution stage holds the Run; ``resume`` after the fix
    re-enters that stage in a fresh Execution pass (#218, readiness Q2)."""

    def setUp(self):
        from unittest.mock import patch

        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(
                target, side_effect=AssertionError("offline caller forbids network")
            )
            guard.start()
            self.addCleanup(guard.stop)

    def _approved(self):
        """An approved, unexecuted Run and its fixed and buggy infrastructures."""
        from collections import deque

        from bio_neb.providers.ports import Receipt
        from tests.h100_stage_fakes import h100_provider
        from tests.planning_assignment_fakes import configured_provider

        baseline = self.prerequisite("approved")
        approved = baseline.value
        journey_file = str(baseline.journey_file)
        fixed = configured_provider(journey_file, [])

        def unreadable_receipt(provider, plan):
            provider.scripts["h100", "verify_outputs"] = deque(
                [Receipt(values={"bytes": 2, "sha256": "0" * 64})]
            )

        def unreadable(plan, journal):
            return h100_provider(
                fixed(plan, journal), plan, approved, amend=unreadable_receipt
            )

        return journey_file, fixed, unreadable

    def _advance(self, intent, infrastructure, bug=None):
        from contextlib import nullcontext

        from bio_neb.application.advancement import use_run_advancement_dependencies

        with (
            bug or nullcontext(),
            use_run_advancement_dependencies(
                parabricks_infrastructure_factory=infrastructure
            ),
        ):
            return advance(intent)

    def test_an_execution_invariant_from_either_runner_site_holds_the_run(self):
        from dataclasses import replace
        from unittest.mock import patch

        from bio_neb.journey_path import JourneyPath
        from bio_neb.parabricks import h100_stage

        conclusion = h100_stage.H100.path_fact

        def invalid_conclusion(receipts):
            fact = conclusion(receipts)
            return fact.model_copy(update={"rests_on": ("unknown-prerequisite",)})

        for site, invariant_id in (
            ("step", "worker_receipt_read_invalid"),
            ("path_fact", "stage_path_fact_invalid"),
        ):
            with self.subTest(site=site):
                journey_file, fixed, unreadable = self._approved()
                with self.assertRaises(RunAdvancementRefused) as held:
                    if site == "step":
                        self._advance(ResumeRun(journey_file=journey_file), unreadable)
                    else:
                        self._advance(
                            ResumeRun(journey_file=journey_file),
                            fixed,
                            patch.object(
                                h100_stage,
                                "H100",
                                replace(h100_stage.H100, path_fact=invalid_conclusion),
                            ),
                        )
                state = JourneyPath.open(Path(journey_file)).state
                failed_id = state.latest_event_ids["StageFailed"]
                self.assertEqual(held.exception.code, invariant_id)
                self.assertEqual(held.exception.status, "run_held")
                self.assertEqual(
                    held.exception.failure.evidence["stage_failed_event_id"], failed_id
                )
                # Recorded and held, never ended: no outcome, no RunFailed.
                self.assertIsNone(state.outcome)
                self.assertEqual(state.failures[-1].invariant_id, invariant_id)
                events = Journey.open(Path(journey_file)).read()
                closed = [e for e in events if e["event_type"] == "PassClosed"][-1]
                self.assertEqual(closed["payload"]["disposition"], "held")
                self.assertEqual(
                    closed["payload"]["failure"]["evidence"]["stage_failed_event_id"],
                    failed_id,
                )
                self.assertNotIn("RunFailed", [e["event_type"] for e in events])
                self.assertEqual(
                    observe(ObserveRun(journey_file=journey_file)).status, "run_held"
                )

    def test_resume_after_the_fix_reenters_the_held_stage_in_a_fresh_pass(self):
        from bio_neb.journey_path import JourneyPath

        journey_file, fixed, unreadable = self._approved()
        resume = ResumeRun(journey_file=journey_file)
        with self.assertRaises(RunAdvancementRefused) as first:
            self._advance(resume, unreadable)
        self.assertEqual(first.exception.code, "worker_receipt_read_invalid")
        held = JourneyPath.open(Path(journey_file)).state
        failed_id = held.held_failure_id
        # Before the fix, resume re-enters once and holds again; the new pass's
        # failure rests only on the one it resolved.
        with self.assertRaises(RunAdvancementRefused) as again_refused:
            self._advance(resume, unreadable)
        self.assertEqual(again_refused.exception.code, "worker_receipt_read_invalid")
        again = JourneyPath.open(Path(journey_file))
        repeated = again.read()[-1]
        self.assertEqual(repeated.event_type, "StageFailed")
        self.assertEqual(repeated.pass_, held.pass_ + 1)
        self.assertEqual(repeated.rests_on, (failed_id,))
        self.assertEqual(again.state.held_failure_id, repeated.event_id)
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.sequential_investigation_support import (
            ASSEMBLY_BODY,
            SourceResponsesOnly,
            _AdaptiveInvestigationModel,
        )

        # The verified Execution begins Investigation from the plan's
        # target, with offline model and source adapters.
        with use_run_advancement_dependencies(
            environ={"LOCAL_MODEL_KEY": "not-retained"},
            investigation_model_factory=_AdaptiveInvestigationModel().model,
            investigation_source_adapter_factory=lambda: SourceResponsesOnly(
                assembly_body=ASSEMBLY_BODY
            ),
        ):
            resumed = self._advance(resume, fixed)
        self.assertEqual(resumed.status, "investigation_review_pending")
        after = JourneyPath.open(Path(journey_file))
        self.assertEqual(after.state.disposition, "succeeded")
        self.assertEqual(after.state.pass_, held.pass_ + 2)
        reentry = next(e for e in after.read() if e.pass_ == held.pass_ + 2)
        self.assertEqual(reentry.event_type, "H100ArtifactsVerified")
        self.assertIn(repeated.event_id, reentry.rests_on)
        self.assertFalse(after.state.unresolved_failure_ids)
        self.assertNotIn(
            "RunFailed",
            [e["event_type"] for e in Journey.open(Path(journey_file)).read()],
        )


class PostExecutionAnswerTests(RunPrerequisites, unittest.TestCase):
    """Native answers on an Execution Journey reach each Review's own owner."""

    def setUp(self):
        from unittest.mock import patch

        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(
                target, side_effect=AssertionError("offline caller forbids network")
            )
            guard.start()
            self.addCleanup(guard.stop)

    def investigation_review(self):
        baseline = self.prerequisite(_INVESTIGATION_REVIEW)
        return str(baseline.journey_file), *baseline.value

    def cli(self, argv, *, stdin=""):
        """Run the native CLI offline; any model or Execution dispatch fails."""
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch

        from bio_neb.cli import main

        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(stdin)),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            code = main(
                argv,
                parabricks_infrastructure_factory=lambda *_: self.fail(
                    "a Review answer dispatched Execution"
                ),
                investigation_model_factory=lambda *_: self.fail(
                    "a Review answer contacted a model"
                ),
            )
        return code, output.getvalue(), errors.getvalue()

    def stop(self, proposal_id):
        return json.dumps(
            {
                "action": "stop",
                "proposal_id": proposal_id,
                "reason": "Stop after the verified Execution.",
            }
        )

    def answers(self, journey_file):
        return [
            event["payload"]
            for event in Journey.open(Path(journey_file)).read()
            if event["event_type"] == "InvestigationReviewResponded"
        ]

    def test_cli_answer_records_investigation_and_finalization_reviews_after_execution(
        self,
    ):
        journey_file, request_event_id, proposal_id = self.investigation_review()
        code, _, errors = self.cli(
            ["answer", "--journey-file", journey_file]
            + ["--request-event-id", request_event_id]
            + ["--response-json", self.stop(proposal_id)]
        )
        self.assertEqual(code, 0, errors)
        stopped = observe(ObserveRun(journey_file=journey_file))
        self.assertEqual(stopped.status, "finalization_review_pending")
        finalization = stopped.current_review.request
        # The published report is explained by default.
        with explained_by_default():
            code, _, errors = self.cli(
                ["answer", "--journey-file", journey_file]
                + ["--request-event-id", finalization.request_event_id]
                + [
                    "--response-json",
                    json.dumps(
                        {
                            "action": "approve",
                            "basis_digest": finalization.basis.basis_digest,
                            "head_event_id": finalization.request_event_id,
                        }
                    ),
                ]
            )
        self.assertEqual(code, 0, errors)

        answers = self.answers(journey_file)
        self.assertEqual(
            [(a["request_event_id"], a["response"]["action"]) for a in answers],
            [(request_event_id, "stop")],
        )
        recorded = [
            event["payload"]["receipt"]
            for event in Journey.open(Path(journey_file)).read()
            if event["event_type"] == "FinalizationReviewRecorded"
        ]
        self.assertEqual(
            [(r["request_event_id"], r["response"]["action"]) for r in recorded],
            [(finalization.request_event_id, "approve")],
        )
        self.assertEqual(
            observe(ObserveRun(journey_file=journey_file)).status,
            "grounded_report_published",
        )

    def test_a_retry_preset_is_refused_at_a_finalization_review(self):
        # A preset there would become the head the exact approval binds.
        journey_file, request_event_id, proposal_id = self.investigation_review()
        code, _, errors = self.cli(
            ["answer", "--journey-file", journey_file]
            + ["--request-event-id", request_event_id]
            + ["--response-json", self.stop(proposal_id)]
        )
        self.assertEqual(code, 0, errors)
        preset = Path(journey_file).parent / "retry-preset.json"
        preset.write_text(
            json.dumps(
                {
                    "agent_role": "investigation_planner",
                    "failure_classes": ["rejected"],
                    "count": 1,
                    "by": "researcher",
                }
            ),
            encoding="utf-8",
        )
        before = Path(journey_file).read_bytes()
        code, _, errors = self.cli(
            ["retry-preset", "--journey-file", journey_file]
            + ["--preset-file", str(preset)]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(
            json.loads(errors)["code"], "retry_preset_placement_unsupported"
        )
        self.assertEqual(Path(journey_file).read_bytes(), before)
        finalization = observe(ObserveRun(journey_file=journey_file)).current_review
        # The published report is explained by default.
        with explained_by_default():
            code, _, errors = self.cli(
                ["answer", "--journey-file", journey_file]
                + ["--request-event-id", finalization.request.request_event_id]
                + [
                    "--response-json",
                    json.dumps(
                        {
                            "action": "approve",
                            "basis_digest": finalization.request.basis.basis_digest,
                            "head_event_id": finalization.request.request_event_id,
                        }
                    ),
                ]
            )
        self.assertEqual(code, 0, errors)
        self.assertEqual(
            observe(ObserveRun(journey_file=journey_file)).status,
            "grounded_report_published",
        )

    def test_cli_answer_refuses_an_unknown_request_id_after_execution(self):
        journey_file, _, _ = self.investigation_review()
        before = Path(journey_file).read_bytes()
        unknown = str(uuid.uuid4())
        code, output, errors = self.cli(
            ["answer", "--journey-file", journey_file]
            + ["--request-event-id", unknown]
            + ["--response-json", json.dumps({"action": "approve"})]
        )
        self.assertEqual((code, output), (2, ""))
        refusal = json.loads(errors)
        self.assertEqual(
            (refusal["frontier"], refusal["code"], refusal["request_event_id"]),
            ("refused", "review_answer_request_unknown", unknown),
        )
        self.assertEqual(Path(journey_file).read_bytes(), before)

    def test_answer_and_interactive_resume_record_the_same_answer_after_execution(
        self,
    ):
        recorded = {}
        for transport in ("answer", "interactive"):
            with self.subTest(transport=transport):
                journey_file, request_event_id, proposal_id = (
                    self.investigation_review()
                )
                response = self.stop(proposal_id)
                argv = (
                    ["answer", "--journey-file", journey_file]
                    + ["--request-event-id", request_event_id]
                    + ["--response-json", response]
                    if transport == "answer"
                    else ["resume", "--journey-file", journey_file, "--interactive"]
                )
                code, _, errors = self.cli(argv, stdin=response + "\n")
                self.assertEqual(code, 0, errors)
                recorded[transport] = (
                    self.answers(journey_file),
                    observe(ObserveRun(journey_file=journey_file)).status,
                )
        self.assertEqual(len(recorded["answer"][0]), 1)
        self.assertEqual(recorded["answer"][0][0]["request_event_id"], request_event_id)
        self.assertEqual(recorded["answer"], recorded["interactive"])


if __name__ == "__main__":
    unittest.main()
