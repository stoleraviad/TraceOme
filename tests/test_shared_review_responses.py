"""Connected public Review responses preserve each stage's exact authority."""

import io
import json
import os
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.advancement import (
    RunAdvancementRefused,
    advance,
    review_response_from_json,
    use_run_advancement_composition,
)
from bio_neb.application.queries import ObserveRun, observe, read_investigation
from bio_neb.application.run_interaction import run_interaction
from bio_neb.cli import main
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    continue_investigation,
)
from bio_neb.investigation.investigation_observation_commands import (
    investigation_observe,
)
from bio_neb.journey import Journey
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    compile_finalization,
)
from tests.explainer_fakes import explained_by_default
from tests.finalization_fakes import evidence_journey
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    THREE_MEMBER_REQUEST,
    ObservationEvidence,
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    amend_to_partial,
    approved_collection_observation,
    collection_request,
)
from tests.sequential_investigation_support import (
    SequentialObservationAdapter as _SequentialObservationAdapter,
)
from tests.sequential_investigation_support import (
    retained_three_member_parent as _retained_three_member_parent,
)
from tests.terminal_journey_fakes import EVIDENCE_REQUEST
from tests.vcf_lookup_fakes import TableReaderModel, run_table_reading

TABLE_READING_ENVIRON = {"LOCAL_MODEL_KEY": "secret-value-not-retained"}


class InvestigationComposition:
    def continue_investigation(self, observation):
        return continue_investigation(
            ContinueInvestigationCommand(journey_file=observation.journey_file),
            environ={"LOCAL_MODEL_KEY": "offline-test-value"},
            model_factory=lambda loaded, selected: SequentialAdaptiveModel().model(
                loaded, selected
            ),
            source_adapter_factory=lambda: SequentialSourceResponses(
                assembly_body=ASSEMBLY_BODY
            ),
        )


class SharedReviewResponseTests(RunPrerequisites, unittest.TestCase):
    def test_a_selected_approval_observes_once_and_its_replay_writes_nothing(self):
        # The approval's continuation supplies the Run's own inputs and
        # observes; replaying the consumed answer writes nothing and fetches
        # nothing again.
        from bio_neb.application.advancement import use_run_advancement_dependencies
        from tests.observation_input_fakes import (
            RUN_VCF,
            TABLE_READING_ENVIRON,
            RunOutputStore,
        )

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, _, _, _ = approved_collection_observation(
                Path(temporary).resolve(),
                cardinality=3,
                approve_count=1,
                forked=True,
                parent_file=self.prerequisite(
                    "executed",
                    investigation_request=collection_request(3),
                    output_payloads={"VCF": RUN_VCF},
                ).journey_file,
            )
            before = observe(ObserveRun(journey_file=str(journey.path)))
            request = before.investigation.pending_request
            command = review_response_from_json(
                before,
                request.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "proposal_id": request.proposal.proposal_id,
                        "work_item_ids": [
                            before.investigation.pending_work_item_ids[0]
                        ],
                    }
                ),
            )
            store = RunOutputStore()
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=TableReaderModel().model,
                investigation_vcf_source_factory=store,
            ):
                answered = advance(command)
                frozen = journey.path.read_bytes()
                replay = advance(command)
            self.assertEqual(len(replay.investigation.authorities), 2)
            self.assertEqual(replay, answered)
            self.assertEqual(journey.path.read_bytes(), frozen)
            # Both grants observed from one download each of the Run's own VCF
            # and BAM, both made by the first grant.
            self.assertEqual(
                len(read_investigation(Journey.open(journey.path)).collections), 2
            )
            self.assertEqual(len(store.gets), 2)

    def test_selected_answer_automatically_observes_only_approved_members(self):
        from bio_neb.application.advancement import use_run_advancement_dependencies

        with tempfile.TemporaryDirectory() as temporary:
            inp, journey, first, descriptor, prepared = approved_collection_observation(
                Path(temporary).resolve(),
                cardinality=3,
                approve_count=1,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            # The Finalization below is the researcher's explicit partial
            # choice; completion otherwise waits for every member.
            amend_to_partial(journey)
            adapter = _SequentialObservationAdapter(prepared, descriptor, inp.sample_id)
            reader = ObservationEvidence(descriptor)
            investigation_observe(prepared, adapter=adapter, evidence_reader=reader)
            before = observe(ObserveRun(journey_file=str(journey.path)))
            request = before.investigation.pending_request
            selected = before.investigation.pending_work_item_ids[0]
            command = review_response_from_json(
                before,
                request.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "proposal_id": request.proposal.proposal_id,
                        "work_item_ids": [selected],
                    }
                ),
            )
            model_requests = sum(
                e["event_type"] == "ModelRequestRecorded"
                for e in Journey.open(journey.path).read()
            )
            queries_before = len(
                [
                    argv
                    for argv in adapter.calls
                    if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file")
                ]
            )
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=TableReaderModel().model,
                investigation_observation_adapter=adapter,
                investigation_observation_evidence_reader=reader,
            ):
                completed = advance(command)
                frozen = journey.path.read_bytes()
                self.assertEqual(advance(command), completed)
                self.assertEqual(journey.path.read_bytes(), frozen)
            self.assertEqual(completed.status, "finalization_review_pending")
            # The completed collection closes its Pass at the before-Finalization
            # child point (ADR 0031); Finalization compiles in a fresh Pass.
            events = Journey.open(journey.path).read()
            compiled = next(
                index
                for index, event in enumerate(events)
                if event["event_type"] == "FinalizationBasisCompiled"
            )
            # After the lookups every item's window is read and the VCF table is
            # read in chunks, then the fold Finalization reads (ADR 0047).
            self.assertEqual(
                [
                    (e["event_type"], e["payload"].get("disposition"))
                    for e in events[compiled - 3 : compiled]
                ],
                [
                    ("InvestigationItemEvidenceCollected", None),
                    ("PassClosed", "completed"),
                    ("PassStarted", None),
                ],
            )
            self.assertEqual(completed.investigation.authorities[0], first)
            self.assertEqual(len(completed.investigation.pending_work_item_ids), 1)
            windows = [
                e
                for e in events
                if e["event_type"] == "InvestigationItemWindowRecorded"
            ]
            chunks = [
                e for e in events if e["event_type"] == "InvestigationTableChunkRead"
            ]
            chunk_attempts = {e["payload"]["attempt_id"] for e in chunks}
            new_requests = [
                e for e in events if e["event_type"] == "ModelRequestRecorded"
            ][model_requests:]
            # Observation and windows contact no model; only the table reading
            # does, one request for the one chunk holding both items.
            self.assertEqual(len(windows), 2)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(
                {e["payload"]["attempt_id"] for e in new_requests}, chunk_attempts
            )
            self.assertEqual(len(new_requests), len(chunks))
            queries = [
                argv
                for argv in adapter.calls
                if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file")
            ]
            self.assertEqual(len(queries), queries_before + 1 + len(windows))
            evidence = next(
                e
                for e in reversed(Journey.open(journey.path).read())
                if e["event_type"] == "InvestigationEvidenceRecorded"
            )
            self.assertEqual(evidence["payload"]["work_item"]["work_item_id"], selected)

    def test_fork_from_the_before_finalization_point_compiles_without_observing(self):
        from bio_neb.application.advancement import (
            ResumeRun,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.fork import ForkRun, fork_run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, _, descriptor, prepared = approved_collection_observation(
                root,
                cardinality=2,
                approve_count=1,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(2)
                ).journey_file,
            )
            adapter = _SequentialObservationAdapter(prepared, descriptor, inp.sample_id)
            reader = ObservationEvidence(descriptor)
            investigation_observe(prepared, adapter=adapter, evidence_reader=reader)
            before = observe(ObserveRun(journey_file=str(journey.path)))
            request = before.investigation.pending_request
            command = review_response_from_json(
                before,
                request.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "proposal_id": request.proposal.proposal_id,
                        "work_item_ids": list(
                            before.investigation.pending_work_item_ids
                        ),
                    }
                ),
            )
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=TableReaderModel().model,
                investigation_observation_adapter=adapter,
                investigation_observation_evidence_reader=reader,
            ):
                parent = advance(command)
            self.assertEqual(parent.status, "finalization_review_pending")
            events = Journey.open(journey.path).read()
            compiled = next(
                index
                for index, event in enumerate(events)
                if event["event_type"] == "FinalizationBasisCompiled"
            )
            point = events[compiled - 2]
            self.assertEqual(
                (point["event_type"], point["payload"]["disposition"]),
                ("PassClosed", "completed"),
            )
            frozen = journey.path.read_bytes()
            calls = len(adapter.calls)

            fork = fork_run(
                ForkRun(
                    parent_journey_file=str(journey.path),
                    child_run_root=str(root / "finalization-child"),
                    point_event_id=point["event_id"],
                )
            )
            idle = observe(ObserveRun(journey_file=fork.child_journey_file))

            def no_observation(*_args, **_kwargs):
                raise AssertionError("the child observed again")

            with use_run_advancement_dependencies(
                investigation_observation_adapter=no_observation,
                investigation_observation_evidence_reader=no_observation,
            ):
                resumed = advance(ResumeRun(journey_file=fork.child_journey_file))
            child = Journey.open(Path(fork.child_journey_file))
            own = [
                e["event_type"]
                for e in child.read()
                if e["journey_id"] == child.journey_id
            ]
            self.assertEqual(journey.path.read_bytes(), frozen)

        self.assertEqual(idle.status, "investigation_evidence_ready")
        self.assertEqual(idle.run_provenance, "causal_child_or_fork")
        self.assertEqual(resumed.status, "finalization_review_pending")
        self.assertEqual(
            resumed.current_review.request.basis.selected_work_item_ids,
            parent.current_review.request.basis.selected_work_item_ids,
        )
        self.assertEqual(
            own, ["RunForked", "PassStarted", "FinalizationBasisCompiled", "PassClosed"]
        )
        self.assertEqual(len(adapter.calls), calls)

    def test_cli_finalization_response_publishes_exact_approved_basis(self):
        for interactive in (False, True):
            with (
                self.subTest(interactive=interactive),
                tempfile.TemporaryDirectory() as temporary,
            ):
                journey = evidence_journey(
                    Path(temporary).resolve(),
                    parent_file=self.prerequisite(
                        "executed", investigation_request=EVIDENCE_REQUEST
                    ).journey_file,
                )
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
                observation = observe(ObserveRun(journey_file=str(journey.path)))
                review = observation.current_review
                frozen = journey.path.read_bytes()
                observed = io.StringIO()

                def forbidden_factory(*_args):
                    raise AssertionError("observation constructed a dependency")

                with redirect_stdout(observed):
                    self.assertEqual(
                        main(
                            ["observe", "--journey-file", str(journey.path)],
                            investigation_model_factory=forbidden_factory,
                            investigation_source_adapter_factory=forbidden_factory,
                        ),
                        0,
                    )
                self.assertEqual(
                    json.loads(observed.getvalue()),
                    run_interaction(observation).model_dump(mode="json"),
                )
                self.assertEqual(journey.path.read_bytes(), frozen)
                response = json.dumps(
                    {
                        "action": "approve",
                        "basis_digest": review.request.basis.basis_digest,
                        "head_event_id": review.request_event_id,
                    }
                )
                argv = (
                    ["resume", "--journey-file", str(journey.path), "--interactive"]
                    if interactive
                    else [
                        "answer",
                        "--journey-file",
                        str(journey.path),
                        "--request-event-id",
                        review.request_event_id,
                        "--response-json",
                        response,
                    ]
                )
                output, errors = io.StringIO(), io.StringIO()
                with (
                    redirect_stdout(output),
                    redirect_stderr(errors),
                    patch("sys.stdin", io.StringIO(response + "\n")),
                    explained_by_default(),
                ):
                    code = main(argv)
                self.assertEqual(code, 0, errors.getvalue())
                rendered = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual(rendered["status"], "grounded_report_published")
                self.assertEqual(
                    rendered["grounded_report"]["report"]["finalization_basis_digest"],
                    review.request.basis.basis_digest,
                )

    def test_pending_finalization_blocks_new_investigation_direction_but_allows_exact_replay(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            inp, journey, _, descriptor, command = approved_collection_observation(
                Path(temporary).resolve(),
                cardinality=3,
                forked=True,
                approve_count=1,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            # The Finalization below is the researcher's explicit partial
            # choice; completion otherwise waits for every member.
            amend_to_partial(journey)
            investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=ObservationEvidence(descriptor),
            )
            run_table_reading(journey.path)
            compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            view = observe(ObserveRun(journey_file=str(journey.path)))
            request = view.investigation.pending_request
            retained = view.investigation.review_receipts[-1]
            # The answered approval closed its Review; its exact replay names
            # that answered request, not the Review it opened.
            self.assertNotEqual(retained.request_event_id, request.request_event_id)
            duplicate = review_response_from_json(
                view, retained.request_event_id, retained.response.model_dump_json()
            )
            frozen = journey.path.read_bytes()
            self.assertEqual(advance(duplicate), view)
            self.assertEqual(journey.path.read_bytes(), frozen)
            new_approval = review_response_from_json(
                view,
                request.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "proposal_id": request.proposal.proposal_id,
                        "work_item_ids": [view.investigation.pending_work_item_ids[0]],
                    }
                ),
            )
            # The pending Finalization Review is the current one, so a new
            # Investigation direction is not a current answer.
            with self.assertRaises(ValueError) as blocked:
                advance(new_approval)
            self.assertEqual(blocked.exception.code, "review_answer_not_current")
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_finalization_binding_retains_frontier_and_exact_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey = evidence_journey(
                Path(temporary).resolve(),
                parent_file=self.prerequisite(
                    "executed", investigation_request=EVIDENCE_REQUEST
                ).journey_file,
            )
            compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            observation = observe(ObserveRun(journey_file=str(journey.path)))
            review = observation.current_review
            self.assertEqual(review.command_frontier, "finalization")
            self.assertEqual(
                tuple(action.action for action in review.legal_actions),
                ("approve", "correct", "stop"),
            )
            response = {
                "action": "approve",
                "basis_digest": review.request.basis.basis_digest,
                "head_event_id": review.request_event_id,
            }
            command = review_response_from_json(
                observation, review.request_event_id, json.dumps(response)
            )
            self.assertIsInstance(command, ApproveFinalizationCommand)
            before = journey.path.read_bytes()
            with self.assertRaises(ValueError):
                advance(command.model_copy(update={"head_event_id": str(uuid.uuid4())}))
            self.assertEqual(journey.path.read_bytes(), before)
            with explained_by_default():
                approved = advance(command)
                self.assertEqual(approved.status, "grounded_report_published")
                before = journey.path.read_bytes()
                self.assertEqual(advance(command), approved)
            self.assertEqual(journey.path.read_bytes(), before)

    def test_finalization_correction_requires_new_basis_approval_and_stop_grants_none(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = evidence_journey(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=EVIDENCE_REQUEST
                ).journey_file,
            )
            compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            view = observe(ObserveRun(journey_file=str(journey.path)))
            review = view.current_review
            candidate = review.request.basis.candidate_evidence[0]
            correction_file = root / "correction.json"
            correction_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-correction/1",
                        "asserted_by": "researcher:owner",
                        "reason": "Exclude this evidence from report membership.",
                        "requested_changes": [
                            {
                                "action": "set_report_membership",
                                "member_id": candidate.candidate_id,
                                "included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            command = review_response_from_json(
                view,
                review.request_event_id,
                json.dumps(
                    {
                        "action": "correct",
                        "basis_digest": review.request.basis.basis_digest,
                        "head_event_id": review.request_event_id,
                        "correction_file": str(correction_file),
                    }
                ),
            )
            corrected = advance(command)
            self.assertEqual(corrected.status, "finalization_review_pending")
            self.assertIsNone(corrected.finalization.report_basis)
            self.assertNotEqual(
                corrected.current_review.request_event_id, review.request_event_id
            )
            self.assertNotEqual(
                corrected.current_review.request.basis.basis_digest,
                review.request.basis.basis_digest,
            )
            frozen = journey.path.read_bytes()
            self.assertEqual(advance(command), corrected)
            self.assertEqual(journey.path.read_bytes(), frozen)
            with self.assertRaises(ValueError):
                advance(
                    ApproveFinalizationCommand(
                        journey_file=str(journey.path),
                        request_event_id=review.request_event_id,
                        basis_digest=review.request.basis.basis_digest,
                        head_event_id=review.request_event_id,
                    )
                )
            self.assertEqual(journey.path.read_bytes(), frozen)
            reason_file = root / "reason.json"
            reason_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-stop/1",
                        "asserted_by": "researcher:owner",
                        "reason": "Stop without publishing this report.",
                    }
                ),
                encoding="utf-8",
            )
            pending = corrected.current_review
            stop = review_response_from_json(
                corrected,
                pending.request_event_id,
                json.dumps(
                    {
                        "action": "stop",
                        "basis_digest": pending.request.basis.basis_digest,
                        "head_event_id": pending.request_event_id,
                        "reason_file": str(reason_file),
                    }
                ),
            )
            stopped = advance(stop)
            self.assertEqual(stopped.status, "finalization_stopped")
            self.assertIsNone(stopped.finalization.report_basis)
            self.assertIsNone(stopped.grounded_report.report)
            frozen = journey.path.read_bytes()
            self.assertEqual(advance(stop), stopped)
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_partial_finalization_publishes_exact_basis_with_investigation_pending(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            inp, journey, _, descriptor, command = approved_collection_observation(
                Path(temporary).resolve(),
                cardinality=3,
                forked=True,
                approve_count=2,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            # The Finalization below is the researcher's explicit partial
            # choice; completion otherwise waits for every member.
            amend_to_partial(journey)
            investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=ObservationEvidence(descriptor),
            )
            run_table_reading(journey.path)
            compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            view = observe(ObserveRun(journey_file=str(journey.path)))
            review = view.current_review
            self.assertEqual(review.kind, "finalization")
            self.assertEqual(len(view.investigation.pending_work_item_ids), 1)
            command = review_response_from_json(
                view,
                review.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "basis_digest": review.request.basis.basis_digest,
                        "head_event_id": review.request_event_id,
                    }
                ),
            )
            with explained_by_default():
                completed = advance(command)
            self.assertEqual(completed.status, "grounded_report_published")
            self.assertEqual(
                completed.investigation.pending_work_item_ids,
                view.investigation.pending_work_item_ids,
            )
            self.assertEqual(
                completed.grounded_report.report.collection_status, "partial"
            )
            self.assertEqual(len(completed.grounded_report.report.work_items), 3)
            interaction = run_interaction(completed)
            self.assertEqual(interaction.frontier, "report_published")
            self.assertEqual(
                interaction.investigation_evidence.combine_status, "partial"
            )
            self.assertEqual(len(interaction.investigation_members), 3)
            pending = interaction.investigation_members[-1]
            self.assertEqual(pending.state.status, "approval_pending")
            self.assertIsNone(pending.recorded)
            for member in interaction.investigation_members[:2]:
                self.assertIsNotNone(member.recorded)
                self.assertEqual(
                    member.recorded.evidence.evidence_state.observation_model_contact_count,
                    0,
                )
            frozen = journey.path.read_bytes()
            with explained_by_default():
                self.assertEqual(advance(command), completed)
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_investigation_amendment_normalizes_then_requires_fresh_selected_approval(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            _, journey, _, _, _ = approved_collection_observation(
                Path(temporary).resolve(),
                cardinality=3,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            # The Finalization below is the researcher's explicit partial
            # choice; completion otherwise waits for every member.
            amend_to_partial(journey)
            view = observe(ObserveRun(journey_file=str(journey.path)))
            original = view.investigation
            request = original.review_requests[-1]
            items = request.proposal.work_items
            response = {
                "action": "revise",
                "proposal_id": request.proposal.proposal_id,
                "asserted_by": "researcher:owner",
                "operations": [
                    {
                        "operation": "amend",
                        "work_item_id": items[0].work_item_id,
                        "request": {
                            "rsid": items[0].target.aliases[0],
                            "request_text": f"Recheck {items[0].target.aliases[0]} in this sample.",
                        },
                    },
                    *[
                        {"operation": "keep", "work_item_id": item.work_item_id}
                        for item in items[1:]
                    ],
                ],
            }
            command = review_response_from_json(
                view, request.request_event_id, json.dumps(response)
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline-test-value"}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "answer",
                        "--journey-file",
                        str(journey.path),
                        "--request-event-id",
                        request.request_event_id,
                        "--response-json",
                        json.dumps(response),
                    ],
                    investigation_model_factory=lambda loaded, selected: (
                        SequentialAdaptiveModel().model(loaded, selected)
                    ),
                    investigation_source_adapter_factory=lambda: SequentialSourceResponses(
                        assembly_body=ASSEMBLY_BODY
                    ),
                )
            self.assertEqual(code, 0, stderr.getvalue())
            continued = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(
                json.loads(stdout.getvalue()),
                run_interaction(continued).model_dump(mode="json"),
            )
            frozen = journey.path.read_bytes()
            self.assertEqual(advance(command), continued)
            self.assertEqual(journey.path.read_bytes(), frozen)
            self.assertEqual(continued.current_review.kind, "investigation")
            self.assertNotEqual(
                continued.current_review.request_event_id, request.request_event_id
            )
            self.assertEqual(len(continued.investigation.pending_work_item_ids), 1)
            self.assertEqual(continued.investigation.authorities, original.authorities)
            self.assertEqual(
                len(continued.investigation.attempts), len(original.attempts) + 1
            )

    def test_finalization_rework_normalizes_selected_change_and_duplicate_cannot_retry_failed_preparation(
        self,
    ):
        for fail_preparation in (False, True):
            with (
                self.subTest(fail_preparation=fail_preparation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                journey = _retained_three_member_parent(
                    root,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=THREE_MEMBER_REQUEST
                    ).journey_file,
                )
                run_table_reading(journey.path)
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
                view = observe(ObserveRun(journey_file=str(journey.path)))
                review = view.current_review
                proposal = view.investigation.proposals[-1]
                first, *others = proposal.work_items
                correction = {
                    "schema_version": "bio-neb.finalization-correction/1",
                    "asserted_by": "researcher:owner",
                    "reason": "Recheck the first variant before reporting.",
                    "requested_changes": [
                        {
                            "action": "change_observation",
                            "description": "Recheck the selected variant.",
                        }
                    ],
                    "investigation_revision": {
                        "action": "revise",
                        "proposal_id": proposal.proposal_id,
                        "asserted_by": "researcher:owner",
                        "operations": [
                            {
                                "operation": "amend",
                                "work_item_id": first.work_item_id,
                                "request": {
                                    "rsid": first.target.aliases[0],
                                    "request_text": f"Recheck {first.target.aliases[0]} in sample HG001.",
                                },
                            },
                            *[
                                {"operation": "keep", "work_item_id": item.work_item_id}
                                for item in others
                            ],
                        ],
                    },
                }
                path = root / "correction.json"
                path.write_text(json.dumps(correction), encoding="utf-8")
                command = review_response_from_json(
                    view,
                    review.request_event_id,
                    json.dumps(
                        {
                            "action": "correct",
                            "basis_digest": review.request.basis.basis_digest,
                            "head_event_id": review.request_event_id,
                            "correction_file": str(path),
                        }
                    ),
                )
                preparations = []

                class FailedPreparation:
                    def __init__(self, preparations):
                        self.preparations = preparations

                    def continue_investigation(self, observation):
                        def unavailable_model(_loaded, _selected):
                            self.preparations.append("model")
                            raise RuntimeError(
                                "selected model construction unavailable"
                            )

                        return continue_investigation(
                            ContinueInvestigationCommand(
                                journey_file=observation.journey_file
                            ),
                            environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                            model_factory=unavailable_model,
                        )

                composition = (
                    FailedPreparation(preparations)
                    if fail_preparation
                    else InvestigationComposition()
                )
                with use_run_advancement_composition(composition):
                    if fail_preparation:
                        # The untyped preparation fault holds the Run.
                        with self.assertRaisesRegex(
                            RunAdvancementRefused, "coordinator_unhandled_error"
                        ) as held:
                            advance(command)
                        self.assertIn(
                            "model construction unavailable",
                            str(held.exception.__cause__),
                        )
                        current = observe(ObserveRun(journey_file=str(journey.path)))
                        self.assertEqual(current.status, "run_held")
                        self.assertEqual(
                            current.investigation.status, "investigation_work_ready"
                        )
                    else:
                        current = advance(command)
                        self.assertEqual(current.current_review.kind, "investigation")
                        self.assertEqual(
                            len(current.investigation.pending_work_item_ids), 1
                        )
                        self.assertIsNone(current.finalization.report_basis)
                    frozen = journey.path.read_bytes()
                    self.assertEqual(advance(command), current)
                    self.assertEqual(journey.path.read_bytes(), frozen)
                self.assertEqual(preparations, ["model"] if fail_preparation else [])
