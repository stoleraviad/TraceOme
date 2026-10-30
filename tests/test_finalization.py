from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from bio_neb.agentic_journey.continuation_policy import FinalizationReviewPolicy
from bio_neb.application.fork import own_fork_origin
from bio_neb.application.queries import (
    ObserveRun,
    ReviewJourneyQuery,
    observe,
    review_journey,
)
from bio_neb.cli import main
from bio_neb.investigation.investigation_observation import (
    validate_investigation_collection_evidence_fact,
    validate_investigation_evidence_fact,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.reporting.finalization import (
    FinalizationBasis,
    FinalizationReturnToInvestigation,
    FinalizationReviewRecorded,
    FinalizationReviewRequested,
    InvestigationRequiredChange,
    validate_finalization_request_fact,
    validate_finalization_review_fact,
)
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    FinalizationApplicationRefused,
    InspectFinalizationCommand,
    approve_finalization,
    compile_finalization,
    inspect_finalization,
)
from tests.content_identity_vectors import expected_digest
from tests.finalization_fakes import (
    evidence_journey,
)
from tests.run_prerequisites import RunPrerequisites
from tests.terminal_journey_fakes import EVIDENCE_REQUEST
from tests.vcf_lookup_fakes import run_table_reading


def _texts(qualifications):
    """Each qualification's sentence (#448 D2: the basis keeps its ids too)."""
    return tuple(item.text for item in qualifications)


def _run(argv: list[str]) -> tuple[int, str, str]:
    output, error = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        exit_code = main(argv)
    return exit_code, output.getvalue(), error.getvalue()


def _copy_journey(journey: Journey, root: Path) -> Journey:
    copied = root / journey.journey_id
    shutil.copytree(journey.run_directory, copied)
    return Journey.open(copied / "journey.jsonl")


def _assert_invalid_reopen(test: unittest.TestCase, journey: Journey) -> None:
    before = journey.path.read_bytes()
    argv = ["observe", "--journey-file", str(journey.path), "--rich"]
    first = _run(argv)
    second = _run(argv)
    test.assertEqual(first, second)
    test.assertEqual(first[0], 2)
    test.assertEqual(json.loads(first[2])["code"], "journey_review_invalid")
    test.assertEqual(journey.path.read_bytes(), before)


def _assert_identity_vector(test, payload, identity_field, namespace, fields):
    # Public JSON is the input vector. Field membership is owned by this test,
    # including nested computed identities; no production selector is the oracle.
    names = fields.split()
    test.assertEqual(set(payload), {*names, identity_field})
    material = {name: payload[name] for name in names}
    test.assertEqual(
        payload[identity_field], namespace + ":" + expected_digest(material)
    )


def _assert_shared_finalization_review(test, journey, request):
    before = journey.path.read_bytes()
    view = observe(ObserveRun(journey_file=str(journey.path)))
    shared = view.current_review
    test.assertEqual(shared.kind, "finalization")
    test.assertEqual(shared.command_frontier, "finalization")
    test.assertEqual(shared.request_event_id, request["request_event_id"])
    test.assertEqual(shared.request.legal_actions, ("approve", "correct", "stop"))
    test.assertEqual(shared.request.model_dump(mode="json"), request)
    test.assertEqual(
        review_journey(
            ReviewJourneyQuery(journey_file=str(journey.path))
        ).current_review,
        shared,
    )
    competing = dict(view)
    competing_request = view.investigation.review_requests[-1].model_copy(
        update={"pending": True}
    )
    competing["investigation"] = view.investigation.model_copy(
        update={
            "pending_request": competing_request,
            "review_requests": (
                *view.investigation.review_requests[:-1],
                competing_request,
            ),
        }
    )
    with test.assertRaisesRegex(ValueError, "more than one pending Review"):
        type(view).model_validate(competing)
    exit_code, _, error = _run(
        [
            "answer",
            "--journey-file",
            str(journey.path),
            "--request-event-id",
            request["request_event_id"],
            "--response-json",
            '{"action":"approve"}',
        ]
    )
    test.assertEqual(exit_code, 2)
    test.assertEqual(json.loads(error)["code"], "command_invalid")
    with patch("sys.stdin", StringIO("")):
        resumed = _run(["resume", "--journey-file", str(journey.path), "--interactive"])
    test.assertEqual(resumed[0], 2)
    test.assertIn("exact legal Review response JSON", resumed[2])
    test.assertIn('"code":"command_invalid"', resumed[2])
    test.assertEqual(journey.path.read_bytes(), before)


class FinalizationPublicPathTests(RunPrerequisites, unittest.TestCase):
    def evidence_ready(self) -> Journey:
        """Observed evidence before Finalization, restored in place."""
        return Journey.open(self.prerequisite("retained-evidence").journey_file)

    def test_bound_execution_abort_finalizes_without_restarting_provider_work(self):
        from bio_neb.application.advancement import (
            AnswerReview,
            ExecutionPathReviewAnswer,
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.parabricks.operation import execute_parabricks
        from tests.explainer_fakes import ExplainerModel
        from tests.observation_input_fakes import TABLE_READING_ENVIRON
        from tests.parabricks_operation_fakes import (
            approved_input,
            input_storage_provider,
        )

        with tempfile.TemporaryDirectory() as directory:
            approved = approved_input(Path(directory).resolve())
            pending = execute_parabricks(
                approved, infrastructure_factory=input_storage_provider
            )
            # No prebuilt index is bound, so the Run prepares on CPU; the
            # unscripted CPU job fails into its stage-failure Review.
            self.assertEqual(pending.status, "execution_review_pending")
            answer = AnswerReview(
                journey_file=pending.journey_file,
                request_event_id=pending.request_event_id,
                answer=ExecutionPathReviewAnswer(
                    kind="execution_path", response={"action": "abort"}
                ),
            )
            with use_run_advancement_dependencies(
                planning_model_factory=lambda *_: self.fail(
                    "Finalization contacted a planning model"
                ),
                investigation_model_factory=lambda *_: self.fail(
                    "Finalization contacted an Investigation model"
                ),
                parabricks_infrastructure_factory=lambda *_: self.fail(
                    "aborted Execution restarted"
                ),
                # The published report is explained by default, by the
                # scripted explainer on the fixture's model credential.
                report_model_factory=ExplainerModel().model,
                environ=TABLE_READING_ENVIRON,
            ):
                ended = advance(answer)
                self.assertEqual(ended.status, "finalization_review_pending")
                basis = ended.finalization.pending_request.basis
                self.assertEqual(basis.outcome, "execution_failed")
                self.assertEqual(basis.work_items, ())
                # No VCF was made, so the basis states no method.
                self.assertIsNone(basis.method)
                self.assertTrue(
                    any(
                        entry.event_type == "ExecutionPathRecorded"
                        and entry.payload["event_type"] == "ExecutionOutcome"
                        for entry in basis.accounting
                    )
                )
                before = Path(pending.journey_file).read_bytes()
                self.assertEqual(advance(answer), ended)
                self.assertEqual(
                    advance(ResumeRun(journey_file=pending.journey_file)), ended
                )
                self.assertEqual(Path(pending.journey_file).read_bytes(), before)
                self.assertEqual(
                    observe(ObserveRun(journey_file=pending.journey_file)), ended
                )

                request = ended.finalization.pending_request
                approval = ApproveFinalizationCommand(
                    journey_file=pending.journey_file,
                    request_event_id=request.request_event_id,
                    basis_digest=basis.basis_digest,
                    head_event_id=request.request_event_id,
                )
                published = advance(approval)
                self.assertEqual(published.status, "grounded_report_published")
                self.assertEqual(
                    published.grounded_report.report.outcome, "execution_failed"
                )
                self.assertEqual(published.grounded_report.report.admissions, ())
                # An aborted Execution may have run; the readable page says only
                # that no verified VCF is recorded.
                from bio_neb.reporting.report_commands import (
                    ReadReadableReportCommand,
                    read_readable_report,
                )

                readable = read_readable_report(
                    ReadReadableReportCommand(
                        journey_file=pending.journey_file,
                        report_digest=published.grounded_report.report.report_digest,
                    )
                )
                self.assertIn(
                    "No verified VCF is recorded for this Run, so no method can be "
                    "stated.",
                    readable,
                )
                self.assertIn(
                    "The sequencing analysis did not finish; the report accounts for "
                    "the Run.",
                    readable,
                )
                before = Path(pending.journey_file).read_bytes()
                self.assertEqual(advance(approval), published)
                self.assertEqual(
                    advance(ResumeRun(journey_file=pending.journey_file)), published
                )
                self.assertEqual(
                    observe(ObserveRun(journey_file=pending.journey_file)), published
                )
                self.assertEqual(Path(pending.journey_file).read_bytes(), before)

    def test_all_investigation_requests_rejected_reach_the_same_finalization(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.application.advancement import (
            AnswerReview,
            BeginInvestigation,
            InvestigationReviewAnswer,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.investigation.investigation import RejectInvestigationResponse
        from tests.model_endpoint_fakes import ScriptedModelEndpoint
        from tests.sequential_investigation_support import (
            SourceResponsesOnly,
            forked_child,
        )

        async def respond(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele?",
                            "missing_information": "Allele unspecified.",
                            "asks_about": "target",
                        },
                        tool_call_id="question",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = {
                "asserted_by": "researcher",
                "requests": [
                    {"request": "Investigate rs429358", "target": {"rsid": "rs429358"}}
                ],
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request))
            sources = SourceResponsesOnly()
            with use_run_advancement_dependencies(
                environ={"LOCAL_MODEL_KEY": "offline"},
                investigation_model_factory=ScriptedModelEndpoint(respond),
                investigation_source_adapter_factory=lambda: sources,
            ):
                pending = advance(
                    BeginInvestigation(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    )
                )
                request = pending.current_review.request
                ended = advance(
                    AnswerReview(
                        journey_file=str(journey.path),
                        request_event_id=request.request_event_id,
                        answer=InvestigationReviewAnswer(
                            kind="investigation",
                            response=RejectInvestigationResponse(
                                action="reject",
                                collection_member_id=request.question.collection_member_id,
                                revision_number=request.question.revision_number,
                                asserted_by="researcher",
                                reason="Do not pursue this request.",
                            ),
                        ),
                    )
                )
            self.assertEqual(ended.status, "finalization_review_pending")
            self.assertEqual(ended.current_review.kind, "finalization")
            self.assertEqual(sources.operations, [])
            basis = ended.finalization.pending_request.basis
            self.assertEqual(basis.work_items, ())
            self.assertEqual(basis.admissions, ())
            text = canonical_json(basis.model_dump(mode="json"))
            for retained in (
                "Which allele?",
                "Do not pursue this request.",
                "Investigate rs429358",
                request.request_event_id,
            ):
                self.assertIn(retained, text)
            self.assertEqual(observe(ObserveRun(journey_file=str(journey.path))), ended)

    def test_input_abort_and_finalization_stop_reopen_without_a_report(self):
        from bio_neb.application.advancement import (
            AnswerReview,
            InputReviewAnswer,
            ResumeRun,
            advance,
        )
        from bio_neb.failure import ReviewAnswer
        from bio_neb.reporting.finalization_commands import StopFinalizationCommand
        from tests.application_fakes import _registered

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            registered, paths = _registered(root)
            paths["read_1"].unlink()
            pending = advance(ResumeRun(journey_file=registered.journey_file))
            self.assertEqual(pending.status, "input_review_pending")
            self.assertEqual(pending.finalization.status, "not_started")
            request = pending.current_review.request
            ended = advance(
                AnswerReview(
                    journey_file=registered.journey_file,
                    request_event_id=request.request_event_id,
                    answer=InputReviewAnswer(
                        response=ReviewAnswer(
                            request_id=request.request.request_id,
                            action="abort",
                            by="researcher",
                        )
                    ),
                )
            )
            self.assertEqual(ended.status, "finalization_review_pending")
            self.assertEqual(ended.attempts, ())
            pending = ended.finalization.pending_request
            self.assertEqual(pending.basis.outcome, "inspection_failed")
            self.assertIn(
                "input_missing", canonical_json(pending.basis.model_dump(mode="json"))
            )
            reason = root / "stop.json"
            reason.write_text(
                json.dumps({"asserted_by": "researcher", "reason": "Stop this basis."})
            )
            stopped = advance(
                StopFinalizationCommand(
                    journey_file=registered.journey_file,
                    request_event_id=pending.request_event_id,
                    basis_digest=pending.basis.basis_digest,
                    head_event_id=pending.request_event_id,
                    reason_file=str(reason),
                )
            )
            self.assertEqual(stopped.status, "finalization_stopped")
            self.assertIsNone(stopped.finalization.report_basis)
            self.assertEqual(stopped.grounded_report.status, "not_published")
            before = Path(registered.journey_file).read_bytes()
            self.assertEqual(
                advance(ResumeRun(journey_file=registered.journey_file)), stopped
            )
            self.assertEqual(Path(registered.journey_file).read_bytes(), before)
            self.assertEqual(
                observe(ObserveRun(journey_file=registered.journey_file)), stopped
            )

            hostile = _copy_journey(
                Journey.open(Path(registered.journey_file)), root / "hostile-accounting"
            )
            events = hostile.read()
            fact = next(
                event
                for event in events
                if event["event_type"] == "FinalizationBasisCompiled"
            )
            fact["payload"]["basis"]["accounting"][0]["payload"]["invented"] = (
                "not retained"
            )
            hostile.path.write_text(
                "".join(canonical_json(event) + "\n" for event in events)
            )
            before = hostile.path.read_bytes()
            argv = ["observe", "--journey-file", str(hostile.path), "--rich"]
            refused = _run(argv)
            self.assertEqual(_run(argv), refused)
            self.assertEqual(refused[0], 2)
            self.assertEqual(json.loads(refused[2])["code"], "command_invalid")
            self.assertEqual(hostile.path.read_bytes(), before)

    def test_accounting_only_basis_retains_request_and_stop_without_scientific_members(
        self,
    ):
        from types import SimpleNamespace

        from bio_neb.investigation.investigation import project_investigation
        from bio_neb.reporting.finalization import compile_finalization_basis

        journey_id = "00000000-0000-4000-8000-000000000001"
        request_id = "00000000-0000-4000-8000-000000000002"
        stop_id = "00000000-0000-4000-8000-000000000003"
        # Supplied compiler inputs: public coordinator eligibility is proved separately.
        timeline = [
            {
                "journey_id": journey_id,
                "event_id": request_id,
                "event_type": "ResearchRequestRegistered",
                "parent_ids": [],
                "payload": {
                    "prompt": "Explain the sample",
                    "prompt_file": "/supplied/request.txt",
                },
                "artifact_references": [],
            },
            {
                "journey_id": journey_id,
                "event_id": stop_id,
                "event_type": "AttemptStopped",
                "parent_ids": [request_id],
                "payload": {
                    "attempt_id": "00000000-0000-4000-8000-000000000004",
                    "code": "actions_refused",
                    "source_event_id": request_id,
                },
                "artifact_references": [],
            },
        ]
        read = SimpleNamespace(
            timeline=timeline, investigation=project_investigation([])
        )
        basis = compile_finalization_basis(
            SimpleNamespace(
                source_head_event=timeline[-1],
                collection=None,
                investigation=read,
                outcome="effort_stopped",
            )
        )
        self.assertEqual(basis.outcome, "effort_stopped")
        self.assertIsNone(basis.question)
        self.assertIsNone(basis.evidence_state_id)
        self.assertIsNone(basis.collection_status)
        self.assertEqual(basis.work_items, ())
        self.assertEqual(basis.selected_work_item_ids, ())
        self.assertEqual(basis.admissions, ())
        self.assertEqual(
            {item.source_event_id: item.payload for item in basis.accounting},
            {request_id: timeline[0]["payload"], stop_id: timeline[1]["payload"]},
        )
        self.assertEqual(len(basis.report_membership), 2)
        self.assertTrue(all(not item.admitted for item in basis.report_membership))
        from bio_neb.reporting.finalization import (
            ApproveFinalizationResponse,
            FinalizationCorrectionAssertion,
            FinalizationReviewReceipt,
            ReportBasis,
            SetReportMembershipChange,
            derive_finalization_correction,
        )
        from bio_neb.reporting.grounded_report import derive_grounded_report

        corrected = derive_finalization_correction(
            basis,
            FinalizationCorrectionAssertion(
                asserted_by="researcher",
                reason="Exclude the request text from presentation",
                requested_changes=(
                    SetReportMembershipChange(
                        member_id=next(
                            item.accounting_id
                            for item in basis.accounting
                            if item.source_event_id == request_id
                        ),
                        included=False,
                    ),
                ),
            ),
        )
        report = derive_grounded_report(
            ReportBasis(
                finalization_basis=corrected,
                approval=FinalizationReviewReceipt(
                    request_event_id="review",
                    response_head_event_id="review",
                    basis_digest=corrected.basis_digest,
                    source_head_event_id=stop_id,
                    response=ApproveFinalizationResponse(),
                ),
            ),
            journey_id=journey_id,
            source_head_event_id="approval",
        )
        self.assertIsNone(report.question)
        self.assertEqual(report.admissions, ())
        self.assertEqual(report.outcome, "effort_stopped")
        self.assertEqual(report.fact_slots, basis.fact_slots)
        self.assertEqual(
            tuple(item.value for item in report.accounting), basis.accounting
        )
        excluded = next(
            item
            for item in report.accounting
            if item.value.source_event_id == request_id
        )
        self.assertFalse(excluded.membership.included)
        self.assertFalse(excluded.membership.admitted)
        self.assertEqual(excluded.value.payload["prompt"], "Explain the sample")

    def test_selected_source_observations_survive_finalization_with_alternatives(self):
        from dataclasses import replace
        from types import SimpleNamespace

        from pydantic_ai.messages import ModelResponse, ToolCallPart

        from bio_neb.application.advancement import ResumeRun, advance
        from bio_neb.application.queries import read_investigation
        from bio_neb.recording import read_evidence
        from tests.finalization_fakes import _Sources
        from tests.model_endpoint_fakes import ScriptedModelEndpoint
        from tests.sequential_investigation_support import source_basis

        class Sources(_Sources):
            # The report names the accession it was asked for, so another
            # accession is a lawful later negative (a lookup of an rsID the
            # request does not name is refused).
            def fetch_assembly_report(self, assembly_accession):
                response = super().fetch_assembly_report(assembly_accession)
                return replace(
                    response,
                    body=response.body.replace(
                        b"GCF_000001405.40", assembly_accession.encode()
                    ),
                )

        calls = 0
        chosen = {}

        async def respond(messages, info):
            nonlocal calls, chosen
            calls += 1
            if calls in (1, 3):
                if calls == 3:
                    chosen = source_basis(info)
                name, args = "fetch_refsnp", {"rsid": "rs429358"}
            elif calls == 4:
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.39",
                    },
                )
            elif calls == 2:
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.40",
                    },
                )
            else:
                name, args = (
                    "compile_proposal",
                    {
                        **chosen,
                        "outcome": "complete",
                    },
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, canonical_json(args), tool_call_id=f"final-basis-{calls}"
                    )
                ]
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "tests.finalization_fakes._Planner",
                return_value=SimpleNamespace(model=ScriptedModelEndpoint(respond)),
            ),
            patch("tests.finalization_fakes._Sources", Sources),
        ):
            journey = evidence_journey(
                Path(directory).resolve(),
                parent_file=self.prerequisite(
                    "executed", investigation_request=EVIDENCE_REQUEST
                ).journey_file,
            )
            observations = [
                event
                for event in read_evidence(journey)
                if event["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            ]
            self.assertEqual(
                [event["payload"]["result"]["status"] for event in observations],
                ["resolved", "compatible", "resolved", "incompatible"],
            )
            finalized = advance(ResumeRun(journey_file=str(journey.path)))
            self.assertEqual(finalized.status, "finalization_review_pending")
            pending = finalized.finalization.pending_request
            self.assertIsNotNone(pending)
            basis = pending.basis
            selected_results = [
                observations[index]["payload"]["result"] for index in (0, 1)
            ]
            self.assertEqual(
                set(basis.work_items[0].source_ids),
                {result["source"]["source_id"] for result in selected_results},
            )
            self.assertEqual(
                set(basis.work_items[0].target.assessment_ids),
                {result["assessment"]["assessment_id"] for result in selected_results},
            )
            self.assertEqual(basis.work_items[0].target.one_based_position, 44908684)
            self.assertEqual(
                {item.source_id for item in basis.retrieved_sources},
                {
                    event["payload"]["result"]["source"]["source_id"]
                    for event in observations
                },
            )
            self.assertEqual(
                {item.assessment_id for item in basis.source_assessments},
                {
                    event["payload"]["result"]["assessment"]["assessment_id"]
                    for event in observations
                },
            )
            scientific = read_investigation(journey).investigation
            value = next(
                item
                for item in scientific.requirement_values
                if item.source == "tool_result"
            )
            self.assertEqual(value.lineage_event_ids, (observations[0]["event_id"],))
            slot = next(
                item for item in basis.fact_slots if item.kind == "requested_rsid"
            )
            self.assertEqual(
                (slot.status, slot.verified_value), ("verified", "rs429358")
            )
            frozen = journey.path.read_bytes()
            reopened = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(reopened.finalization.pending_request.basis, basis)
            self.assertEqual(journey.path.read_bytes(), frozen)
            self.assertEqual(calls, 5)

    def test_every_work_item_states_that_coverage_was_not_measured(self) -> None:
        # Replaces the archived coverage prototype's "truthful unknown" test: this
        # Run holds no local BAM (its inputs are supplied), so its own measurement
        # states each item's coverage once as not measured, with why, and no
        # coverage member or callability claim appears.
        from bio_neb.investigation.investigation_observation_commands import (
            investigation_observe,
        )
        from tests.sequential_investigation_support import (
            ObservationAdapter,
            ObservationEvidence,
            approved_collection_observation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, _, descriptor, command = (
                approved_collection_observation(root, cardinality=3)
            )
            investigation_observe(
                command,
                adapter=ObservationAdapter(
                    command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=ObservationEvidence(descriptor),
            )

            run_table_reading(journey.path)
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            ).pending_request

        assert pending is not None
        basis = pending.basis
        self.assertEqual(len(basis.work_items), 3)
        self.assertEqual(
            sorted(gap for gap in _texts(basis.gaps) if "overage" in gap),
            sorted(
                f"Coverage was not measured for Work item {item.work_item_id}"
                ": bam_unavailable (the Run holds no local BAM: no Execution made "
                "one, or its download was aborted)."
                for item in basis.work_items
            ),
        )
        self.assertFalse(any("callability" in gap for gap in _texts(basis.gaps)))
        self.assertNotIn("coverage_evidence", basis.model_dump(mode="json"))
        self.assertFalse(
            any(
                item.member_role == "coverage_evidence"
                for item in basis.report_membership
            )
        )

    def test_the_basis_states_how_the_run_s_vcf_was_made(self) -> None:
        # The method is copied from the Run's approved Execution and its verified
        # outputs, so the readable report never reads raw accounting.
        journey = self.evidence_ready()
        timeline = journey.read()
        path = [
            event["payload"]
            for event in timeline
            if event["event_type"] == "ExecutionPathRecorded"
        ]
        (bound,) = (p["payload"] for p in path if p["event_type"] == "ExecutionBound")
        (verified,) = (p for p in path if p["event_type"] == "H100ArtifactsVerified")
        package = bound["approved_input"]["proposal"]["execution_package"]
        (vcf,) = (
            item for item in verified["payload"]["pointers"] if item["role"] == "VCF"
        )

        pending = compile_finalization(
            CompileFinalizationCommand(journey_file=str(journey.path))
        ).pending_request

        assert pending is not None
        method = pending.basis.method
        assert method is not None
        self.assertEqual(method.sample_id, package["workflow"]["sample_id"])
        self.assertEqual(
            method.reference_accession, package["reference"]["assembly_accession"]
        )
        self.assertEqual(
            method.reference_source, package["reference"]["source_locator"]
        )
        self.assertEqual(
            method.reference_checksum,
            "md5:" + package["reference"]["expected_checksum"],
        )
        self.assertEqual(method.workflow_id, package["workflow"]["workflow_id"])
        self.assertEqual(method.command, ("pbrun", "germline"))
        self.assertEqual(
            (method.aligner, method.caller), ("fq2bam (BWA-MEM)", "HaplotypeCaller")
        )
        self.assertTrue(method.pbrun_version.startswith("pbrun: "))
        self.assertEqual(method.vcf_content_id, vcf["content_id"])
        self.assertEqual(
            method.source_event_ids,
            tuple(
                event["event_id"]
                for event in timeline
                if event["event_type"] == "ExecutionPathRecorded"
                and event["payload"]["event_type"]
                in {"ExecutionBound", "H100ArtifactsVerified"}
            ),
        )

    def test_a_run_on_its_requested_assembly_states_no_other(self) -> None:
        """Owner, #459 D4: only a Run switched to a suggested assembly states the
        assembly the researcher asked for (its method and report line,
        `test_prebuilt_index_tools.SwitchedMethodTests`); this Run states none."""
        from types import SimpleNamespace

        from bio_neb.reporting import finalization
        from bio_neb.reporting.readable_report import _method_lines

        method = finalization.run_method(self.evidence_ready().read())
        assert method is not None
        self.assertNotIn("requested_reference_accession", method.model_dump())
        self.assertNotIn("asked for", _method_lines(SimpleNamespace(method=method))[1])

    def test_diploid_calling_on_an_organism_judged_otherwise_is_the_run_s_limitation(
        self,
    ) -> None:
        # The method copies the planning model's judgment, and a judgment other
        # than diploid, which the researcher answered by choosing diploid calling,
        # is a limitation of the Run typed from it, never words.
        from unittest import mock

        from bio_neb.application.queries import read_completion
        from bio_neb.investigation.qualification_origins import (
            DiploidCallingJudgedOtherwise,
        )
        from bio_neb.reporting import finalization

        journey = self.evidence_ready()
        candidate = read_completion(journey).candidate
        assert candidate is not None
        method = finalization.run_method(journey.read())
        assert method is not None
        # The default human Run: judged diploid, so no ploidy limitation.
        self.assertEqual(method.organism_ploidy, "diploid")
        self.assertNotIn(
            "diploid_calling_judged_otherwise",
            [
                item.origin.code
                for item in finalization.compile_finalization_basis(
                    candidate
                ).limitations
            ],
        )
        for judgment in ("likely_non_diploid", "unknown"):
            # A derived input: the same Run's method with this judgment.
            judged = method.model_copy(update={"organism_ploidy": judgment})
            with (
                self.subTest(judgment=judgment),
                mock.patch.object(finalization, "run_method", return_value=judged),
            ):
                basis = finalization.compile_finalization_basis(candidate)
                (limitation,) = [
                    item
                    for item in basis.limitations
                    if item.origin == DiploidCallingJudgedOtherwise(judgment=judgment)
                ]
                self.assertEqual(
                    (limitation.work_item_ids, limitation.request_member_ids),
                    ((), ()),
                )

    def test_each_qualification_carries_the_work_items_it_concerns(self) -> None:
        # One qualification per sentence, with the id of every item it came
        # from, so the explainer reads ids, not the text.
        from bio_neb.investigation.investigation_observation_commands import (
            investigation_observe,
        )
        from tests.sequential_investigation_support import (
            ObservationAdapter,
            ObservationEvidence,
            approved_collection_observation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, _, descriptor, command = (
                approved_collection_observation(root, cardinality=3)
            )
            investigation_observe(
                command,
                adapter=ObservationAdapter(
                    command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=ObservationEvidence(descriptor),
            )
            run_table_reading(journey.path)
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            ).pending_request

        assert pending is not None
        basis = pending.basis
        items = sorted(item.work_item_id for item in basis.work_items)
        self.assertEqual(len(items), 3)
        by_text = {item.text: item for item in basis.limitations}
        # Every observed item's evidence state states this sentence; the basis
        # holds it once, concerning all three.
        shared = by_text[
            "Coverage, callability, biological absence, function, causality, "
            "and diagnosis are not inferred."
        ]
        self.assertEqual(
            (shared.work_item_ids, shared.request_member_ids), (tuple(items), ())
        )
        for item in items:
            (gap,) = [
                gap
                for gap in basis.gaps
                if gap.text.startswith(
                    f"Coverage was not measured for Work item {item}:"
                )
            ]
            self.assertEqual((gap.work_item_ids, gap.request_member_ids), ((item,), ()))

    def test_the_basis_states_each_fact_slot_of_the_run_s_plan(self) -> None:
        journey_file = self.prerequisite("planned-evidence").journey_file
        pending = compile_finalization(
            CompileFinalizationCommand(journey_file=str(journey_file))
        ).pending_request
        assert pending is not None
        slots = {
            item.kind: (
                item.status,
                item.verification_source,
                item.requested_value,
                item.mismatch,
                item.summary,
            )
            for item in pending.basis.fact_slots
        }
        self.assertEqual(
            slots["sample_identity"],
            (
                "asserted",
                None,
                "sample-001",
                False,
                "asserted by researcher, unverified",
            ),
        )
        self.assertEqual(slots["taxon_identity"][:2], ("verified", "NCBI Taxonomy"))
        self.assertEqual(
            slots["reference_assembly_identity"][:2], ("verified", "NCBI Datasets")
        )
        self.assertEqual(
            slots["requested_rsid"][:4], ("verified", "NCBI RefSNP", "rs429358", False)
        )
        self.assertIn(
            "fact_slots",
            pending.basis.model_dump(mode="json", exclude={"basis_digest"}),
        )

    def test_retained_owner_reuses_bytes_across_finalization_approval_and_refuses_stale_history(
        self,
    ):
        journey = self.evidence_ready()
        artifact_paths = {
            journey.artifact_path(reference)
            for event in journey.read()
            for reference in event["artifact_references"]
        }
        contents = {path.read_bytes() for path in artifact_paths}
        reads, hashes = [], []
        original_read, original_hash = Path.read_bytes, hashlib.sha256

        def read_bytes(path):
            if path in artifact_paths:
                reads.append(path)
            return original_read(path)

        def sha256(content=b"", *args, **kwargs):
            if content in contents:
                hashes.append(content)
            return original_hash(content, *args, **kwargs)

        with (
            patch.object(Path, "read_bytes", read_bytes),
            patch.object(hashlib, "sha256", sha256),
        ):
            owner = Journey.open(journey.path)
            compiled = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path)),
                journey=owner,
            )
            pending = compiled.pending_request
            assert pending is not None
            inspect_finalization(
                InspectFinalizationCommand(
                    journey_file=str(journey.path),
                    basis_digest=pending.basis.basis_digest,
                ),
                journey=owner,
            )
            command = ApproveFinalizationCommand(
                journey_file=str(journey.path),
                request_event_id=pending.request_event_id,
                basis_digest=pending.basis.basis_digest,
                head_event_id=pending.request_event_id,
            )
            approved = approve_finalization(command, journey=owner)
            self.assertIsNotNone(approved.report_basis)
            from bio_neb.application.queries import (
                read_investigation,
            )

            view = read_investigation(owner)
            # The Fork child's inherited first fact may carry no payload; the
            # isolation claim holds for any held event, so use the latest.
            view.timeline[-1]["payload"].clear()
            self.assertTrue(read_investigation(owner).timeline[-1]["payload"])
            from bio_neb.recording import is_contact_evidence, read_evidence

            read = read_investigation(owner)
            self.assertEqual(
                read.prefix_event_ids,
                tuple(event["event_id"] for event in read_evidence(owner)),
            )
            self.assertEqual(
                tuple(
                    event["event_id"]
                    for event in read.timeline
                    if not is_contact_evidence(event)
                ),
                owner.prefix_event_ids,
            )
            self.assertEqual(len(reads), len(artifact_paths))
            self.assertEqual(len(hashes), len(artifact_paths))
            fresh = Journey.open(journey.path)
            self.assertEqual(len(reads), 2 * len(artifact_paths))
            self.assertEqual(len(hashes), 2 * len(artifact_paths))
            self.assertEqual(read_investigation(fresh), read_investigation(owner))
            fresh.record(
                "UnrelatedSemanticEvent",
                {},
                parent_ids=[fresh.read()[-1]["event_id"]],
            )
            before = original_read(journey.path)
            with self.assertRaises(FinalizationApplicationRefused) as stale:
                approve_finalization(command, journey=owner)
            self.assertEqual(stale.exception.code, "finalization_journey_context_stale")
            self.assertEqual(original_read(journey.path), before)

    def test_public_compile_inspect_approve_reopen_is_digest_bound_and_contact_free(  # noqa: PLR0915 - Existing connected authority proof remains one scenario.
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = self.evidence_ready()
            self.assertIsNone(
                observe(ObserveRun(journey_file=str(journey.path))).current_review
            )
            source_head = journey.read()[-1]
            # The head is the item evidence fold (ADR 0047); the lookup precedes it.
            evidence_fact = next(
                fact
                for event in reversed(journey.read())
                if (fact := validate_investigation_evidence_fact(event)) is not None
            )
            observation_vectors = (
                (
                    evidence_fact.sample_observation,
                    "observation_id",
                    "sample-observation",
                    "schema_version investigation_authority_id proposal_id review_response_id "
                    "work_item_id sample_id assembly_accession assembly_name vcf_descriptor "
                    "materialization_receipt_id materialization_receipt_sha256 preparation "
                    "result run_provenance provenance_origin_event_id grants_scientific_truth",
                ),
                (
                    evidence_fact.candidate_evidence,
                    "candidate_id",
                    "candidate-evidence",
                    "schema_version investigation_authority_id work_item_id observation_id "
                    "outcome proposed_claim claim_scope limitations source_role scientifically_admitted",
                ),
                (
                    evidence_fact.assessment,
                    "assessment_id",
                    "observation-evidence-assessment",
                    "schema_version investigation_authority_id work_item_id candidate_id "
                    "observation_id outcome verdict work_item_disposition basis claim_scope "
                    "limitations contradictions applicability supports_admission assessor admits_claim",
                ),
                (
                    evidence_fact.admission,
                    "admission_id",
                    "admitted-observation-evidence",
                    "status investigation_authority_id work_item_id candidate_id assessment_id "
                    "observation_id admitted_claim claim_scope admission_authority",
                ),
                (
                    evidence_fact.evidence_state,
                    "evidence_state_id",
                    "investigation-evidence-state",
                    "schema_version status investigation_authority_id proposal_id review_response_id "
                    "work_item_id observation_id candidate_id assessment_id admission_id outcome "
                    "work_item_disposition evidence_admission limitations conflicts gaps "
                    "inherited_model_contact_count inherited_source_contact_count "
                    "observation_model_contact_count observation_source_contact_count "
                    "forbidden_contact_count run_provenance provenance_origin_event_id",
                ),
            )
            for value, field, namespace, fields in observation_vectors:
                with self.subTest(identity=namespace):
                    _assert_identity_vector(
                        self, value.model_dump(mode="json"), field, namespace, fields
                    )
            collection = next(
                fact
                for event in reversed(journey.read())
                if (fact := validate_investigation_collection_evidence_fact(event))
                is not None
            )
            _assert_identity_vector(
                self,
                collection.model_dump(mode="json"),
                "collection_state_id",
                "investigation-collection-evidence",
                "event_type command authority selected_work_item_ids collection_work_item_ids "
                "collection_proposal_id carried_revision_id previous_collection_state_id "
                "item_states combine_status work_policy work_policy_id",
            )
            origin_event_id = own_fork_origin(Journey.open(journey.path))[0]
            self.assertEqual(
                evidence_fact.sample_observation.run_provenance, "causal_child_or_fork"
            )
            self.assertEqual(
                evidence_fact.sample_observation.provenance_origin_event_id,
                origin_event_id,
            )
            hostile_parent = _copy_journey(journey, root / "hostile-parent")
            hostile_basis = _copy_journey(journey, root / "hostile-basis")
            hostile_artifact = _copy_journey(journey, root / "hostile-artifact")
            hostile_origin = _copy_journey(journey, root / "hostile-origin")
            hostile_timeline = hostile_origin.read()
            hostile_timeline[0]["payload"] = {}
            hostile_origin.path.write_text(
                "".join(canonical_json(event) + "\n" for event in hostile_timeline),
                encoding="utf-8",
            )
            hostile_origin_bytes = hostile_origin.path.read_bytes()
            with self.assertRaises(ValueError):
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(hostile_origin.path))
                )
            self.assertEqual(hostile_origin.path.read_bytes(), hostile_origin_bytes)
            compile_argv = [
                "finalization-compile",
                "--journey-file",
                str(journey.path),
            ]

            exit_code, stdout, stderr = _run(compile_argv)

            self.assertEqual(exit_code, 0, stderr)
            self.assertEqual(stderr, "")
            compiled = json.loads(stdout)
            finalization = compiled["finalization"]
            _assert_shared_finalization_review(
                self, journey, finalization["pending_request"]
            )
            self.assertEqual(compiled["status"], "finalization_review_pending")
            self.assertEqual(
                finalization["pending_request"]["review_policy"],
                {"stage": "finalization", "approval": "mandatory"},
            )
            from bio_neb.reporting.finalization import project_finalization

            for hostile_policy in (
                {},
                {"stage": "finalization"},
                {"approval": "mandatory"},
                {"stage": "execution", "approval": "mandatory"},
                {"stage": "finalization", "approval": "optional"},
            ):
                hostile = copy.deepcopy(Journey.open(journey.path).read())
                hostile[-1]["payload"]["review_policy"] = hostile_policy
                with (
                    self.subTest(review_policy=hostile_policy),
                    self.assertRaises(ValueError),
                ):
                    project_finalization(hostile)
            basis = finalization["pending_request"]["basis"]
            _assert_identity_vector(
                self,
                basis,
                "basis_digest",
                "finalization-basis",
                "schema_version journey_id source_head_event_id outcome accounting evidence_state_id "
                "selected_work_item_ids collection_item_states collection_status question "
                "work_items retrieved_sources source_assessments sample_observations "
                "candidate_evidence observation_assessments admissions item_evidence "
                "conflicts limitations gaps evidence_pointers "
                "report_membership inherited_model_contact_count inherited_source_contact_count "
                "finalization_external_contact_count run_provenance provenance_origin_event_id "
                "supersedes_basis_digest correction_request_ids fact_slots "
                "collection_view run_health method",
            )
            for pointer in basis["evidence_pointers"]:
                _assert_identity_vector(
                    self,
                    pointer,
                    "pointer_id",
                    "finalization-evidence-pointer",
                    "role locator content_id sha256 source_event_id",
                )
            query = evidence_fact.sample_observation.result.query_invocation
            assert query is not None
            invocation_pointer = next(
                pointer
                for pointer in basis["evidence_pointers"]
                if pointer["role"] == "process_invocation:" + query.purpose
                and pointer["locator"]
                == "argv:"
                + json.dumps(
                    list(query.argv), ensure_ascii=False, separators=(",", ":")
                )
            )
            invocation_json = query.model_dump(mode="json")
            # A query's receipt keeps the regions file it read.
            invocation_fields = (
                "purpose argv return_code stdout_byte_count "
                "stdout_sha256 stderr_byte_count stderr_sha256 regions_file"
            ).split()
            self.assertEqual(set(invocation_json), set(invocation_fields))
            invocation_material = {
                name: invocation_json[name] for name in invocation_fields
            }
            self.assertEqual(
                invocation_pointer["content_id"],
                "process-invocation:"
                + hashlib.sha256(
                    json.dumps(
                        invocation_material,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(basis["source_head_event_id"], source_head["event_id"])
            self.assertEqual(basis["evidence_state_id"], collection.collection_state_id)
            self.assertEqual(basis["question"]["rsid"], "rs429358")
            target = basis["work_items"][0]["target"]
            self.assertEqual(
                (
                    target["assembly_accession"],
                    target["assembly_name"],
                    target["contig"],
                    target["one_based_position"],
                    target["ref"],
                    target["alt"],
                    basis["sample_observations"][0]["sample_id"],
                ),
                (
                    "GCF_000001405.40",
                    "GRCh38.p14",
                    "NC_000019.10",
                    44908684,
                    "T",
                    "C",
                    "sample-001",
                ),
            )
            self.assertEqual(basis["candidate_evidence"][0]["outcome"], "no_record")
            self.assertEqual(basis["admissions"][0]["status"], "admitted")
            self.assertEqual(
                basis["observation_assessments"][0]["work_item_disposition"],
                "unresolved",
            )
            self.assertEqual(basis["schema_version"], "bio-neb.finalization-basis/7")
            self.assertEqual(basis["run_provenance"], "causal_child_or_fork")
            self.assertEqual(basis["provenance_origin_event_id"], origin_event_id)
            self.assertNotIn("non_fresh_development_clone", basis)
            request_fact = validate_finalization_request_fact(
                Journey.open(journey.path).read()[-1]
            )
            assert request_fact is not None
            contradictory_basis = request_fact.basis.model_copy(
                update={"provenance_origin_event_id": None}
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                contradictory_basis.complete_stably_ordered_inventory()
            self.assertEqual(
                (
                    basis["inherited_model_contact_count"],
                    basis["inherited_source_contact_count"],
                    basis["finalization_external_contact_count"],
                ),
                # Three planner contacts and the table reader's one.
                (4, 2, 0),
            )
            self.assertIn(
                (
                    "Coverage, callability, biological absence, function, causality, "
                    "and diagnosis are not inferred."
                ),
                [item["text"] for item in basis["limitations"]],
            )
            self.assertEqual(
                [item["pointer_id"] for item in basis["evidence_pointers"]],
                sorted(item["pointer_id"] for item in basis["evidence_pointers"]),
            )
            self.assertIn(
                "journey_artifact:RunForked",
                {item["role"] for item in basis["evidence_pointers"]},
            )
            self.assertNotIn("secret-value-not-retained", json.dumps(basis))
            self.assertTrue(
                all(item["included"] for item in basis["report_membership"])
            )
            self.assertTrue(
                all(item["payload_retained"] for item in basis["report_membership"])
            )
            concise_argv = ["observe", "--journey-file", str(journey.path)]
            exit_code, concise_stdout, stderr = _run(concise_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            concise = json.loads(concise_stdout)
            self.assertEqual(concise["status"], "finalization_review_pending")
            self.assertEqual(concise["frontier"], "review_required")
            self.assertEqual(concise["review"]["kind"], "finalization")
            self.assertEqual(
                concise["finalization"]["pending_request"]["basis"]["basis_digest"],
                basis["basis_digest"],
            )

            compiled_event = Journey.open(journey.path).read()[-1]
            compiled_fact = validate_finalization_request_fact(compiled_event)
            assert compiled_fact is not None
            unrelated = hostile_parent.record(
                "HostileUnrelatedAppend",
                {"does_not_supply_investigation_evidence": True},
                [source_head["event_id"]],
            )
            false_parent_payload = compiled_fact.basis.model_dump(
                mode="json", exclude_computed_fields=True
            )
            false_parent_payload["source_head_event_id"] = unrelated["event_id"]
            false_parent_basis = FinalizationBasis.model_validate_json(
                canonical_json(false_parent_payload), strict=True
            )
            false_parent_request = FinalizationReviewRequested(
                review_policy=FinalizationReviewPolicy(),
                source_head_event_id=unrelated["event_id"],
                basis=false_parent_basis,
            )
            hostile_parent.record(
                false_parent_request.event_type,
                false_parent_request.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [unrelated["event_id"]],
            )
            _assert_invalid_reopen(self, hostile_parent)

            fabricated_payload = compiled_fact.basis.model_dump(
                mode="json", exclude_computed_fields=True
            )
            fabricated_payload["report_membership"][0]["included"] = False
            fabricated_basis = FinalizationBasis.model_validate_json(
                canonical_json(fabricated_payload), strict=True
            )
            fabricated_request = FinalizationReviewRequested(
                review_policy=FinalizationReviewPolicy(),
                source_head_event_id=source_head["event_id"],
                basis=fabricated_basis,
            )
            hostile_basis.record(
                fabricated_request.event_type,
                fabricated_request.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [source_head["event_id"]],
            )
            _assert_invalid_reopen(self, hostile_basis)

            hostile_artifact.record(
                compiled_fact.event_type,
                compiled_fact.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [source_head["event_id"]],
            )
            artifact_reference = next(
                reference
                for event in hostile_artifact.read()
                for reference in event["artifact_references"]
            )
            artifact_path = hostile_artifact.artifact_path(artifact_reference)
            artifact_path.write_bytes(b"hostile-artifact-corruption")
            corrupted = artifact_path.read_bytes()
            _assert_invalid_reopen(self, hostile_artifact)
            self.assertEqual(artifact_path.read_bytes(), corrupted)
            retained_after_compile = journey.path.read_bytes()

            replay = _run(compile_argv)
            self.assertEqual(replay, (0, stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_compile)
            digest = basis["basis_digest"]
            inspect_argv = [
                "finalization-inspect",
                "--journey-file",
                str(journey.path),
                "--basis-digest",
                digest,
            ]
            self.assertEqual(_run(inspect_argv), (0, stdout, ""))

            pending = finalization["pending_request"]
            request_head = Journey.open(journey.path).read()[-1]["event_id"]
            approve_argv = [
                "finalization-approve",
                "--journey-file",
                str(journey.path),
                "--request-event-id",
                pending["request_event_id"],
                "--basis-digest",
                digest,
                "--head-event-id",
                request_head,
            ]
            exit_code, approved_stdout, stderr = _run(approve_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            approved = json.loads(approved_stdout)
            self.assertEqual(approved["status"], "finalization_approved")
            report_basis = approved["finalization"]["report_basis"]
            _assert_identity_vector(
                self,
                report_basis["approval"],
                "response_id",
                "finalization-review-response",
                "request_event_id response_head_event_id basis_digest source_head_event_id response",
            )
            expected_report_material = {
                "finalization_basis_digest": basis["basis_digest"],
                "approval_response_id": report_basis["approval"]["response_id"],
            }
            self.assertEqual(
                report_basis["report_basis_digest"],
                "report-basis:"
                + hashlib.sha256(
                    json.dumps(
                        expected_report_material,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(report_basis["finalization_basis"]["basis_digest"], digest)
            self.assertTrue(
                report_basis["report_basis_digest"].startswith("report-basis:")
            )
            self.assertEqual(report_basis["finalization_external_contact_count"], 0)
            exit_code, concise_stdout, stderr = _run(concise_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            concise = json.loads(concise_stdout)
            self.assertEqual(concise["status"], "finalization_approved")
            self.assertEqual(concise["frontier"], "checkpoint")
            self.assertNotIn("review", concise)
            self.assertEqual(
                concise["finalization"]["report_basis"]["report_basis_digest"],
                report_basis["report_basis_digest"],
            )
            retained_after_approval = journey.path.read_bytes()

            reopen_argv = ["observe", "--journey-file", str(journey.path), "--rich"]
            first_reopen = _run(reopen_argv)
            second_reopen = _run(reopen_argv)
            self.assertEqual(first_reopen, second_reopen)
            self.assertEqual(first_reopen, (0, approved_stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_approval)

            for field in (
                "schema_version",
                "source_kind",
                "continuation_id",
                "committed_candidate_sha",
            ):
                with self.subTest(origin_field=field):
                    hostile = _copy_journey(journey, root / f"origin-{field}")
                    events = hostile.read()
                    events[0]["payload"][field] = "changed-origin-binding"
                    hostile.path.write_text(
                        "".join(canonical_json(event) + "\n" for event in events),
                        encoding="utf-8",
                    )
                    hostile_bytes = hostile.path.read_bytes()
                    calls = (
                        lambda hostile=hostile: compile_finalization(
                            CompileFinalizationCommand(journey_file=str(hostile.path))
                        ),
                        lambda hostile=hostile: inspect_finalization(
                            InspectFinalizationCommand(
                                journey_file=str(hostile.path), basis_digest=digest
                            )
                        ),
                        lambda hostile=hostile: approve_finalization(
                            ApproveFinalizationCommand(
                                journey_file=str(hostile.path),
                                request_event_id=pending["request_event_id"],
                                basis_digest=digest,
                                head_event_id=request_head,
                            )
                        ),
                    )
                    for call in calls:
                        with self.assertRaises(ValueError):
                            call()
                        self.assertEqual(hostile.path.read_bytes(), hostile_bytes)

    def test_public_correct_with_retained_membership_supersedes_basis_and_new_research_returns_to_investigation(  # noqa: PLR0915 - Existing connected correction proof remains one scenario.
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = self.evidence_ready()
            exit_code, compiled_stdout, stderr = _run(
                ["finalization-compile", "--journey-file", str(journey.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            compiled = json.loads(compiled_stdout)
            pending = compiled["finalization"]["pending_request"]
            basis = pending["basis"]
            candidate = basis["candidate_evidence"][0]
            correction_file = root / "membership-correction.json"
            correction_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-correction/1",
                        "asserted_by": "researcher:owner",
                        "reason": (
                            "Résumé: keep unresolved no_record evidence visible but exclude it "
                            "from exact Report membership."
                        ),
                        "requested_changes": [
                            {
                                "action": "set_report_membership",
                                "member_id": candidate["candidate_id"],
                                "included": False,
                            }
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            correct_argv = [
                "finalization-correct",
                "--journey-file",
                str(journey.path),
                "--request-event-id",
                pending["request_event_id"],
                "--basis-digest",
                basis["basis_digest"],
                "--head-event-id",
                Journey.open(journey.path).read()[-1]["event_id"],
                "--correction-file",
                str(correction_file),
            ]
            hostile_superseding = _copy_journey(journey, root / "hostile-superseding")

            exit_code, corrected_stdout, stderr = _run(correct_argv)

            self.assertEqual((exit_code, stderr), (0, ""))
            corrected = json.loads(corrected_stdout)
            self.assertEqual(corrected["status"], "finalization_review_pending")
            finalization = corrected["finalization"]
            self.assertEqual(len(finalization["bases"]), 2)
            superseding = finalization["pending_request"]["basis"]
            self.assertEqual(
                superseding["supersedes_basis_digest"], basis["basis_digest"]
            )
            self.assertEqual(
                superseding["source_head_event_id"], basis["source_head_event_id"]
            )
            self.assertEqual(
                superseding["evidence_state_id"], basis["evidence_state_id"]
            )
            self.assertEqual(
                superseding["candidate_evidence"], basis["candidate_evidence"]
            )
            self.assertEqual(superseding["limitations"], basis["limitations"])
            self.assertEqual(
                superseding["evidence_pointers"], basis["evidence_pointers"]
            )
            membership = {
                item["member_id"]: item for item in superseding["report_membership"]
            }
            self.assertFalse(membership[candidate["candidate_id"]]["included"])
            self.assertTrue(membership[candidate["candidate_id"]]["payload_retained"])
            self.assertEqual(
                (
                    superseding["inherited_model_contact_count"],
                    superseding["inherited_source_contact_count"],
                    superseding["finalization_external_contact_count"],
                ),
                (4, 2, 0),
            )
            corrected_event = Journey.open(journey.path).read()[-1]
            corrected_fact = validate_finalization_review_fact(corrected_event)
            assert corrected_fact is not None
            correction = corrected_fact.receipt.response.correction
            _assert_identity_vector(
                self,
                correction.model_dump(mode="json"),
                "correction_id",
                "finalization-correction",
                "schema_version asserted_by reason requested_changes investigation_revision",
            )
            _assert_identity_vector(
                self,
                corrected_fact.receipt.model_dump(mode="json"),
                "response_id",
                "finalization-review-response",
                "request_event_id response_head_event_id basis_digest source_head_event_id response",
            )
            fabricated_payload = corrected_fact.superseding_basis.model_dump(
                mode="json", exclude_computed_fields=True
            )
            for item in fabricated_payload["report_membership"]:
                if item["member_id"] == candidate["candidate_id"]:
                    item["included"] = True
                elif item["included"]:
                    item["included"] = False
                    break
            fabricated_basis = FinalizationBasis.model_validate_json(
                canonical_json(fabricated_payload), strict=True
            )
            fabricated_review = FinalizationReviewRecorded(
                review_policy=FinalizationReviewPolicy(),
                receipt=corrected_fact.receipt,
                superseding_basis=fabricated_basis,
            )
            hostile_superseding.record(
                fabricated_review.event_type,
                fabricated_review.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [pending["request_event_id"]],
            )
            _assert_invalid_reopen(self, hostile_superseding)
            exit_code, concise_stdout, stderr = _run(
                ["observe", "--journey-file", str(journey.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            concise = json.loads(concise_stdout)
            self.assertEqual(concise["status"], "finalization_review_pending")
            self.assertEqual(concise["review"]["kind"], "finalization")
            self.assertEqual(
                concise["finalization"]["pending_request"]["basis"]["basis_digest"],
                superseding["basis_digest"],
            )
            retained_after_correction = journey.path.read_bytes()
            self.assertEqual(_run(correct_argv), (0, corrected_stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_correction)

            next_pending = finalization["pending_request"]
            research_file = root / "new-research-correction.json"
            research_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-correction/1",
                        "asserted_by": "researcher:owner",
                        "reason": (
                            "Coverage evidence is required before deciding what "
                            "no_record means."
                        ),
                        "requested_changes": [
                            {
                                "action": "new_research",
                                "description": (
                                    "Establish retained coverage and callability "
                                    "evidence for the target locus."
                                ),
                            }
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            return_argv = [
                "finalization-correct",
                "--journey-file",
                str(journey.path),
                "--request-event-id",
                next_pending["request_event_id"],
                "--basis-digest",
                superseding["basis_digest"],
                "--head-event-id",
                Journey.open(journey.path).read()[-1]["event_id"],
                "--correction-file",
                str(research_file),
            ]
            hostile_return = _copy_journey(journey, root / "hostile-return")

            exit_code, returned_stdout, stderr = _run(return_argv)

            self.assertEqual((exit_code, stderr), (0, ""))
            returned = json.loads(returned_stdout)
            self.assertEqual(
                returned["status"], "finalization_returned_to_investigation"
            )
            finalization = returned["finalization"]
            self.assertIsNone(finalization["pending_request"])
            self.assertIsNone(finalization["report_basis"])
            self.assertEqual(len(finalization["bases"]), 2)
            self.assertEqual(
                finalization["return_to_investigation"]["reason_code"],
                "new_research_required",
            )
            self.assertEqual(
                finalization["return_to_investigation"]["basis_digest"],
                superseding["basis_digest"],
            )
            returned_event = Journey.open(journey.path).read()[-1]
            returned_fact = validate_finalization_review_fact(returned_event)
            assert returned_fact is not None
            fabricated_return = FinalizationReturnToInvestigation(
                basis_digest=returned_fact.return_to_investigation.basis_digest,
                correction_id=returned_fact.return_to_investigation.correction_id,
                reason_code=returned_fact.return_to_investigation.reason_code,
                requested_changes=(
                    InvestigationRequiredChange(
                        action="new_research",
                        description="A different request was not recorded by the researcher.",
                    ),
                ),
            )
            fabricated_return_review = FinalizationReviewRecorded(
                review_policy=FinalizationReviewPolicy(),
                receipt=returned_fact.receipt,
                return_to_investigation=fabricated_return,
            )
            hostile_return.record(
                fabricated_return_review.event_type,
                fabricated_return_review.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [next_pending["request_event_id"]],
            )
            _assert_invalid_reopen(self, hostile_return)
            exit_code, concise_stdout, stderr = _run(
                ["observe", "--journey-file", str(journey.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            concise = json.loads(concise_stdout)
            self.assertEqual(
                concise["status"], "finalization_returned_to_investigation"
            )
            self.assertEqual(concise["frontier"], "checkpoint")
            self.assertEqual(
                concise["finalization"]["return_to_investigation"]["requested_changes"],
                finalization["return_to_investigation"]["requested_changes"],
            )
            retained_after_return = journey.path.read_bytes()
            self.assertEqual(_run(return_argv), (0, returned_stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_return)
            reopen_argv = ["observe", "--journey-file", str(journey.path), "--rich"]
            first_reopen = _run(reopen_argv)
            second_reopen = _run(reopen_argv)
            self.assertEqual(first_reopen, second_reopen)
            self.assertEqual(first_reopen, (0, returned_stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_return)

            # The same retained evidence again, restored: the corrected Run
            # above is not read after this point.
            retained = self.evidence_ready()
            exit_code, retained_compiled_stdout, stderr = _run(
                ["finalization-compile", "--journey-file", str(retained.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            retained_pending = json.loads(retained_compiled_stdout)["finalization"][
                "pending_request"
            ]
            _assert_shared_finalization_review(self, retained, retained_pending)
            retained_return_argv = [
                "finalization-correct",
                "--journey-file",
                str(retained.path),
                "--request-event-id",
                retained_pending["request_event_id"],
                "--basis-digest",
                retained_pending["basis"]["basis_digest"],
                "--head-event-id",
                retained_pending["request_event_id"],
                "--correction-file",
                str(research_file),
            ]
            exit_code, retained_returned_stdout, stderr = _run(retained_return_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            self.assertEqual(
                json.loads(retained_returned_stdout)["status"],
                "finalization_returned_to_investigation",
            )
            exit_code, retained_concise_stdout, stderr = _run(
                ["observe", "--journey-file", str(retained.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            retained_concise = json.loads(retained_concise_stdout)
            self.assertEqual(
                retained_concise["status"],
                "finalization_returned_to_investigation",
            )
            self.assertEqual(retained_concise["frontier"], "checkpoint")
            self.assertIn("return_to_investigation", retained_concise["finalization"])

    def test_public_stop_and_invalid_responses_are_atomic_and_create_no_report_basis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = self.evidence_ready()
            exit_code, compiled_stdout, stderr = _run(
                ["finalization-compile", "--journey-file", str(journey.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            compiled = json.loads(compiled_stdout)
            pending = compiled["finalization"]["pending_request"]
            basis = pending["basis"]
            reason_file = root / "stop-reason.json"
            reason_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-stop/1",
                        "asserted_by": "researcher:owner",
                        "reason": (
                            "Résumé: do not create a Report basis from this evidence state."
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            stop_argv = [
                "finalization-stop",
                "--journey-file",
                str(journey.path),
                "--request-event-id",
                pending["request_event_id"],
                "--basis-digest",
                basis["basis_digest"],
                "--head-event-id",
                Journey.open(journey.path).read()[-1]["event_id"],
                "--reason-file",
                str(reason_file),
            ]

            exit_code, stopped_stdout, stderr = _run(stop_argv)

            self.assertEqual((exit_code, stderr), (0, ""))
            stopped = json.loads(stopped_stdout)
            self.assertEqual(stopped["status"], "finalization_stopped")
            stop_fact = validate_finalization_review_fact(
                Journey.open(journey.path).read()[-1]
            )
            assert stop_fact is not None
            _assert_identity_vector(
                self,
                stop_fact.receipt.response.stop.model_dump(mode="json"),
                "stop_id",
                "finalization-stop",
                "schema_version asserted_by reason",
            )
            _assert_identity_vector(
                self,
                stop_fact.receipt.model_dump(mode="json"),
                "response_id",
                "finalization-review-response",
                "request_event_id response_head_event_id basis_digest source_head_event_id response",
            )
            finalization = stopped["finalization"]
            self.assertIsNone(finalization["pending_request"])
            self.assertIsNone(finalization["report_basis"])
            self.assertEqual(
                finalization["stop"]["basis_digest"], basis["basis_digest"]
            )
            self.assertEqual(
                finalization["stop"]["finalization_external_contact_count"], 0
            )
            exit_code, concise_stdout, stderr = _run(
                ["observe", "--journey-file", str(journey.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            concise = json.loads(concise_stdout)
            self.assertEqual(concise["status"], "finalization_stopped")
            self.assertEqual(concise["frontier"], "terminal")
            self.assertEqual(
                concise["finalization"]["stop"]["basis_digest"],
                basis["basis_digest"],
            )
            retained_after_stop = journey.path.read_bytes()
            self.assertEqual(_run(stop_argv), (0, stopped_stdout, ""))
            self.assertEqual(journey.path.read_bytes(), retained_after_stop)
            conflicting_approve = [
                "finalization-approve",
                "--journey-file",
                str(journey.path),
                "--request-event-id",
                pending["request_event_id"],
                "--basis-digest",
                basis["basis_digest"],
                "--head-event-id",
                pending["request_event_id"],
            ]
            exit_code, _, stderr = _run(conflicting_approve)
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                json.loads(stderr)["code"], "finalization_review_response_conflict"
            )
            self.assertEqual(journey.path.read_bytes(), retained_after_stop)
            self.assertEqual(
                _run(["finalization-compile", "--journey-file", str(journey.path)]),
                (0, stopped_stdout, ""),
            )
            self.assertEqual(journey.path.read_bytes(), retained_after_stop)

            # Each block restores the evidence; no block reads an earlier Run.
            invalid = self.evidence_ready()
            exit_code, invalid_compiled_stdout, stderr = _run(
                ["finalization-compile", "--journey-file", str(invalid.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            invalid_compiled = json.loads(invalid_compiled_stdout)
            invalid_pending = invalid_compiled["finalization"]["pending_request"]
            invalid_basis = invalid_pending["basis"]
            before_invalid = invalid.path.read_bytes()
            malformed_file = root / "malformed-correction.json"
            malformed_file.write_text("{not-json", encoding="utf-8")
            invalid_common = [
                "finalization-correct",
                "--journey-file",
                str(invalid.path),
                "--request-event-id",
                invalid_pending["request_event_id"],
                "--basis-digest",
                invalid_basis["basis_digest"],
                "--head-event-id",
                invalid_pending["request_event_id"],
                "--correction-file",
            ]
            exit_code, _, stderr = _run([*invalid_common, str(malformed_file)])
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                json.loads(stderr)["code"], "finalization_correction_invalid"
            )
            self.assertEqual(invalid.path.read_bytes(), before_invalid)

            candidate_id = invalid_basis["candidate_evidence"][0]["candidate_id"]
            noop_file = root / "noop-correction.json"
            noop_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.finalization-correction/1",
                        "asserted_by": "researcher:owner",
                        "reason": (
                            "This repeats the exact retained membership decision."
                        ),
                        "requested_changes": [
                            {
                                "action": "set_report_membership",
                                "member_id": candidate_id,
                                "included": True,
                            }
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            exit_code, _, stderr = _run([*invalid_common, str(noop_file)])
            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(stderr)["code"], "finalization_correction_noop")
            self.assertEqual(invalid.path.read_bytes(), before_invalid)

            current = Journey.open(invalid.path)
            current.record(
                "HostileUnrelatedAppend",
                {"does_not_authorize_finalization": True},
                [current.read()[-1]["event_id"]],
            )
            after_hostile_append = invalid.path.read_bytes()
            stale_approve = [
                "finalization-approve",
                "--journey-file",
                str(invalid.path),
                "--request-event-id",
                invalid_pending["request_event_id"],
                "--basis-digest",
                invalid_basis["basis_digest"],
                "--head-event-id",
                Journey.open(invalid.path).read()[-1]["event_id"],
            ]
            exit_code, _, stderr = _run(stale_approve)
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                json.loads(stderr)["code"], "finalization_review_binding_stale"
            )
            self.assertEqual(invalid.path.read_bytes(), after_hostile_append)
            reopen_argv = ["observe", "--journey-file", str(invalid.path), "--rich"]
            first_reopen = _run(reopen_argv)
            second_reopen = _run(reopen_argv)
            self.assertEqual(first_reopen, second_reopen)
            self.assertEqual((first_reopen[0], first_reopen[2]), (0, ""))
            reopened = json.loads(first_reopen[1])
            self.assertEqual(reopened["status"], "finalization_review_pending")
            self.assertEqual(
                reopened["finalization"]["pending_request"]["basis"]["basis_digest"],
                invalid_basis["basis_digest"],
            )
            self.assertEqual(invalid.path.read_bytes(), after_hostile_append)

            retained = self.evidence_ready()
            exit_code, retained_compiled_stdout, stderr = _run(
                ["finalization-compile", "--journey-file", str(retained.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            retained_pending = json.loads(retained_compiled_stdout)["finalization"][
                "pending_request"
            ]
            retained_stop_argv = [
                "finalization-stop",
                "--journey-file",
                str(retained.path),
                "--request-event-id",
                retained_pending["request_event_id"],
                "--basis-digest",
                retained_pending["basis"]["basis_digest"],
                "--head-event-id",
                retained_pending["request_event_id"],
                "--reason-file",
                str(reason_file),
            ]
            exit_code, retained_stopped_stdout, stderr = _run(retained_stop_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            self.assertEqual(
                json.loads(retained_stopped_stdout)["status"],
                "finalization_stopped",
            )
            exit_code, retained_concise_stdout, stderr = _run(
                ["observe", "--journey-file", str(retained.path)]
            )
            self.assertEqual((exit_code, stderr), (0, ""))
            retained_concise = json.loads(retained_concise_stdout)
            self.assertEqual(retained_concise["status"], "finalization_stopped")
            self.assertEqual(retained_concise["frontier"], "terminal")
            self.assertIn("stop", retained_concise["finalization"])
