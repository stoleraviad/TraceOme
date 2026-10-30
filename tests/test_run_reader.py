"""The Run reader: the Journey feed and the Run artifacts of one retained Run.

The oracle is the retained `journey.jsonl` itself, read line by line in the test.
"""

import hashlib
import json
import shutil
import tempfile
import unittest
import uuid
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from bio_neb.application.run_reader import (
    RunArtifactsUnavailable,
    journey_feed,
    read_journey_feed,
    read_run_artifacts,
)
from bio_neb.cli import main
from bio_neb.failure import ProviderError
from bio_neb.journey import Journey, JourneyEvent, JourneyValidationError
from bio_neb.journey_path import Outcome
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    approve_finalization,
    compile_finalization,
)
from bio_neb.reporting.report_commands import (
    PublishGroundedReportCommand,
    publish_grounded_report,
)
from tests.finalization_fakes import evidence_journey
from tests.parabricks_operation_fakes import (
    BAM_BYTES,
    VCF_BYTES,
    approved_input,
    input_storage_provider,
)
from tests.run_prerequisites import RunPrerequisites, State
from tests.terminal_journey_fakes import EVIDENCE_REQUEST


def _lines(journey_file: Path) -> list[dict]:
    return [json.loads(line) for line in journey_file.read_text().splitlines()]


def _held_pass(failure_class, invariant_id):
    return {
        "pass_id": str(uuid.uuid4()),
        "started_event_id": str(uuid.uuid4()),
        "disposition": "held",
        "journal_sha256": "0" * 64,
        "journal_bytes": 0,
        "accounting_error": None,
        "failure": {
            "failure_class": failure_class,
            "invariant_id": invariant_id,
            "cause": "KeyboardInterrupt",
            "evidence": {},
        },
    }


def _event(event_type: str, payload: dict) -> JourneyEvent:
    """A synthetic retained envelope for a pure projection case."""
    return JourneyEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        journey_id="3ac7744f-aab4-4102-ae64-e60d7cf4fadf",
        occurred_at="2026-10-14T09:00:00Z",
        payload=payload,
    )


def _cli(*argv: str) -> tuple[int, str, str]:
    output, error = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        code = main(list(argv))
    return code, output.getvalue(), error.getvalue()


def _head(journey_file: Path) -> str:
    return Journey.open(journey_file).read()[-1]["event_id"]


def _published(root: Path, parent_file: Path) -> tuple[Path, None]:
    """A Fork child of the executed Run, through Finalization to its Grounded report."""
    journey_file = evidence_journey(root, parent_file=parent_file).path
    pending = compile_finalization(
        CompileFinalizationCommand(journey_file=str(journey_file))
    ).pending_request
    approved = approve_finalization(
        ApproveFinalizationCommand(
            journey_file=str(journey_file),
            request_event_id=pending.request_event_id,
            basis_digest=pending.basis.basis_digest,
            head_event_id=_head(journey_file),
        )
    )
    publish_grounded_report(
        PublishGroundedReportCommand(
            journey_file=str(journey_file),
            report_basis_digest=approved.report_basis.report_basis_digest,
            head_event_id=_head(journey_file),
        )
    )
    return journey_file, None


_PUBLISHED = State(
    "published-report",
    _published,
    forks_from="executed",
    forks_from_choices={"investigation_request": EVIDENCE_REQUEST},
)


class JourneyFeedTests(RunPrerequisites, unittest.TestCase):
    def test_each_item_references_its_raw_event_and_hides_its_payload(self):
        journey_file = Path(self.prerequisite("executed").journey_file)
        events = _lines(journey_file)

        feed = read_journey_feed(str(journey_file), everything=True)

        self.assertEqual(feed.journey_file, str(journey_file))
        self.assertEqual(feed.journey_id, events[0]["journey_id"])
        self.assertEqual(
            [
                (item.sequence, item.event_id, item.event_type, item.occurred_at)
                for item in feed.items
            ],
            [
                (number, event["event_id"], event["event_type"], event["occurred_at"])
                for number, event in enumerate(events, start=1)
            ],
        )
        for item in feed.items:
            self.assertNotIn("payload", item.model_dump(mode="json"))

        shown = read_journey_feed(str(journey_file), everything=True, payloads=True)

        self.assertEqual(
            [item.payload for item in shown.items],
            [event["payload"] for event in events],
        )

    def test_the_default_view_shows_milestones_and_steps_and_all_shows_every_event(
        self,
    ):
        # Key and step items show by default; Agent, Pass and model mechanics are
        # quiet or noise, shown only with everything.
        journey_file = Path(self.prerequisite("executed").journey_file)
        events = _lines(journey_file)

        default = read_journey_feed(str(journey_file))
        everything = read_journey_feed(str(journey_file), everything=True)

        self.assertEqual(
            [item.event_id for item in everything.items],
            [event["event_id"] for event in events],
        )
        self.assertEqual(
            {item.event_type for item in default.items},
            {
                # key
                "JourneyStarted",
                "ResearchRequestRegistered",
                "ProposalRequirementsIdentified",
                "PlanningHitlRequested",
                "PlanningHitlResponded",
                "ExecutionProposalSubmitted",
                "ExecutionHitlResponded",
                "ExecutionPathRecorded",
                # step
                "RunInputsRegistered",
                "ParabricksConfigurationRegistered",
                "InputMetadataInspected",
                "SampleAttestationRecorded",
                "TaxonResolutionRecorded",
                "ReferenceAssembliesRecorded",
                "ReferenceChoiceRecorded",
                "ReferenceDescriptionRecorded",
                "CompanionObservationRecorded",
                # The genomic FASTA's index search and choice.
                "PrebuiltIndexesListed",
                "PrebuiltIndexChosen",
                "RecipeObservationRecorded",
                "PlanningCandidateSelected",
                "ResourceObservationRecorded",
                "ReferenceSourceAdmitted",
                "ExecutionPackageCompiled",
            },
        )
        self.assertEqual(
            [item.sequence for item in default.items],
            [
                item.sequence
                for item in everything.items
                if item.level in {"key", "step"}
            ],
        )
        level = {item.event_id: item.level for item in everything.items}
        expected = {
            "ExecutionPathRecorded": "key",
            "RunInputsRegistered": "step",
            "PassStarted": "quiet",
            "AttemptStarted": "quiet",
            "ToolCallValidated": "quiet",
            "ModelRequestRecorded": "noise",
            "ModelTransportRequestRecorded": "noise",
            "activity_started": "noise",
        }
        for event in events:
            if event["event_type"] in expected:
                self.assertEqual(
                    level[event["event_id"]],
                    expected[event["event_type"]],
                    event["event_type"],
                )

    def test_a_refused_agent_turn_is_quiet_and_an_admitted_one_is_noise(self):
        # A refused turn presents the correction the model receives.
        admitted, refused = journey_feed(
            [
                _event(
                    "AgentActionTurnDispositionRecorded",
                    {"disposition": "singleton_admitted"},
                ),
                _event(
                    "AgentActionTurnDispositionRecorded",
                    {"disposition": "multi_call_refused"},
                ),
            ]
        )

        self.assertEqual((admitted.level, refused.level), ("noise", "quiet"))

    def test_a_reasoning_only_reply_and_its_stop_are_told_as_their_own(self):
        # A reply with only reasoning is not told as a reply in text.
        refused, stopped = journey_feed(
            [
                _event(
                    "AgentActionTurnDispositionRecorded",
                    {"disposition": "reasoning_refused"},
                ),
                _event("AgentWorkReviewRequested", {"cause": "failed_turns"}),
            ]
        )

        # The stop is the one failure rule's, whatever the failed turns
        # were.
        self.assertEqual(
            [(item.level, item.text) for item in (refused, stopped)],
            [
                (
                    "quiet",
                    "The assistant reasoned but sent no action; asked again",
                ),
                (
                    "key",
                    "Question for you: nothing the assistant tried worked several "
                    "times in a row; continue?",
                ),
            ],
        )

    def test_a_routed_reply_with_faults_is_quiet_and_a_reordered_one_is_noise(self):
        # A routed reply with faults presents the note the model receives.
        reordered, faulted = journey_feed(
            [
                _event("AgentTurnRouted", {"faults": []}),
                _event(
                    "AgentTurnRouted",
                    {"faults": [{"key": "copied_calls", "count": 1}]},
                ),
            ]
        )

        self.assertEqual(
            [(item.level, item.text) for item in (reordered, faulted)],
            [
                ("noise", "The assistant's calls were put in the order they need"),
                (
                    "quiet",
                    "The assistant's reply was trimmed to its distinct offered calls",
                ),
            ],
        )

    def test_each_item_tells_its_event_in_one_plain_line_with_its_status(self):
        executed = read_journey_feed(
            str(self.prerequisite("executed").journey_file), everything=True
        ).items
        published = read_journey_feed(
            str(self.prerequisite(_PUBLISHED).journey_file), everything=True
        ).items

        for item in (*executed, *published):
            self.assertTrue(item.text, item.event_type)
            self.assertNotIn("\n", item.text, item.event_type)
            self.assertLessEqual(len(item.text), 100, item.text)
            self.assertNotEqual(item.text, item.event_type, "no wording")

        def told(items, event_type):
            return [
                (item.status, item.text)
                for item in items
                if item.event_type == event_type
            ]

        self.assertEqual(
            told(executed, "ExecutionProposalSubmitted"),
            [("question", "The analysis plan is ready for your approval")],
        )
        self.assertEqual(
            told(executed, "ExecutionHitlResponded"),
            [("answer", "You approved the execution plan")],
        )
        self.assertEqual(
            told(executed, "ExecutionPathRecorded"),
            [
                ("done", "The approved analysis was set up to run"),
                (
                    "verified",
                    "Your read files were copied to cloud storage and verified",
                ),
                ("verified", "Reference genome index prepared and verified"),
                ("verified", "GPU analysis results verified"),
                ("done", "The analysis run finished successfully"),
            ],
        )
        self.assertEqual(
            {status for status, _ in told(executed, "PlanningHitlRequested")},
            {"question"},
        )
        self.assertEqual(
            {status for status, _ in told(executed, "PlanningHitlResponded")},
            {"answer"},
        )
        self.assertEqual(
            told(published, "FinalizationBasisCompiled"),
            [("question", "Results are ready for your review before the report")],
        )
        self.assertEqual(
            told(published, "FinalizationReviewRecorded"),
            [("answer", "You reviewed the results")],
        )
        self.assertEqual(
            told(published, "GroundedReportPublished"),
            [("done", "Your report was published")],
        )

    def test_a_source_contact_names_the_source_it_reaches(self):
        # A non-human Run's lookups reach EVA, not NCBI.
        eva, ncbi, failed = journey_feed(
            [
                _event(
                    "SourcePhysicalContactStarted",
                    {
                        "scope_event_id": "s",
                        "request_identity": "https://www.ebi.ac.uk/eva/webservices/"
                        "identifiers/v1/clustered-variants/5334475172",
                        "method": "GET",
                    },
                ),
                _event(
                    "SourcePhysicalContactStarted",
                    {
                        "scope_event_id": "s",
                        "request_identity": "https://api.ncbi.nlm.nih.gov/variation/v0/"
                        "refsnp/429358",
                        "method": "GET",
                    },
                ),
                _event(
                    "SourceOperationFailureRecorded",
                    {
                        "policy": {
                            "surface": {"scope": "tool", "name": "eva_contig_alias"}
                        },
                        "detail": "EVA rejected its request with HTTP 503.",
                    },
                ),
            ]
        )

        self.assertEqual(
            (eva.text, ncbi.text, failed.text),
            (
                "Contacting EVA",
                "Contacting NCBI",
                "An EVA lookup failed: EVA rejected its request with HTTP 503",
            ),
        )

    def test_failures_and_new_event_types_are_told(self):
        failed, operational, unknown = journey_feed(
            [
                _event("RunFailed", {"code": "rejected_by_user"}),
                _event(
                    "OperationalFailureRecorded",
                    {
                        "family": "source",
                        # The surface whose policy routed it names the source.
                        "policy": {
                            "surface": {"scope": "tool", "name": "fetch_refsnp"}
                        },
                        "detail": "Official NCBI metadata returned HTTP 502.",
                    },
                ),
                _event("AnalysisStepInvented", {"anything": 1}),
            ]
        )

        self.assertEqual(
            (failed.status, failed.level, failed.text),
            ("failed", "key", "The Run ended because you rejected it"),
        )
        self.assertEqual(
            (operational.status, operational.level, operational.text),
            (
                "failed",
                "key",
                "An NCBI lookup failed: Official NCBI metadata returned HTTP 502",
            ),
        )
        self.assertEqual(
            (unknown.status, unknown.level, unknown.text),
            ("done", "quiet", "AnalysisStepInvented"),
        )

    def test_an_execution_review_that_asks_about_ploidy_says_so(self):
        # The Review asks before diploid calling a likely non-diploid organism.
        asks, states = journey_feed(
            [
                _event(
                    "ExecutionProposalSubmitted",
                    {"proposal": {"ploidy": {"question": "Is it diploid?"}}},
                ),
                _event("ExecutionProposalSubmitted", {"proposal": {"ploidy": {}}}),
            ]
        )

        self.assertEqual(
            (asks.status, asks.text),
            (
                "question",
                "The analysis plan is ready; it asks you to confirm diploid calling",
            ),
        )
        self.assertEqual(states.text, "The analysis plan is ready for your approval")

    def test_the_runs_health_and_each_items_depth_are_told(self):
        # The progress view shows the Run's own health measurement once its
        # BAM and VCF are local; never a gate.
        health, measured, unmeasured = journey_feed(
            [
                _event(
                    "RunHealthMeasured",
                    {
                        "schema_version": "bio-neb.run-health/2",
                        "collection_event_id": "c",
                        "bam_sha256": "b" * 64,
                        "vcf_sha256": "c" * 64,
                        "measurement": {
                            "manifest_sha256": "a" * 64,
                            "manifest_byte_count": 2048,
                            "reads": 40_366_123,
                            "mapped_percent": 99.85,
                            "mean_depth": 31.24,
                            "ti_tv": 2.05,
                            "covered_contigs": 24,
                            "covered_bases": 2_900_000_000,
                            "covered_fraction": 0.94,
                        },
                        "not_measured": None,
                    },
                ),
                _event(
                    "InvestigationItemCoverageRecorded",
                    {
                        "contig": "NC_000019.10",
                        "span_start": 44908684,
                        "span_end": 44908684,
                        "measurement": {"depth": 11, "meets_minimum_depth": True},
                        "not_measured": None,
                    },
                ),
                _event(
                    "InvestigationItemCoverageRecorded",
                    {
                        "contig": "chrUn",
                        "span_start": 5,
                        "span_end": 7,
                        "measurement": None,
                        "not_measured": {"code": "contig_unknown", "detail": "x"},
                    },
                ),
            ]
        )

        self.assertEqual(
            (health.status, health.level, health.text),
            (
                "verified",
                "key",
                "Health of your run: 40,366,123 reads, 99.85% mapped,"
                " mean depth 31.2, Ti/Tv 2.05, 24 sequences with reads",
            ),
        )
        self.assertEqual(
            (measured.status, measured.level, measured.text),
            (
                "done",
                "step",
                "Reads at NC_000019.10:44908684: 11 at its least-covered base",
            ),
        )
        self.assertEqual(
            (unmeasured.status, unmeasured.level, unmeasured.text),
            ("done", "step", "Reads at chrUn:5-7 not measured: contig_unknown"),
        )

    def test_a_health_not_measured_and_its_review_are_told_for_every_code(self):
        # Every code `HealthNotMeasuredCode` declares has its words, and the
        # failure's Review reads as a question and an answer.
        from typing import get_args

        from bio_neb.application.run_reader import HEALTH_NOT_MEASURED_WORDS
        from bio_neb.investigation.run_health import HealthNotMeasuredCode

        self.assertEqual(
            set(HEALTH_NOT_MEASURED_WORDS), set(get_args(HealthNotMeasuredCode))
        )
        lines = journey_feed(
            [
                *(
                    _event(
                        "RunHealthMeasured",
                        {
                            "schema_version": "bio-neb.run-health/2",
                            "collection_event_id": "c",
                            "measurement": None,
                            "not_measured": {"code": code, "detail": "x"},
                        },
                    )
                    for code in sorted(HEALTH_NOT_MEASURED_WORDS)
                ),
                _event(
                    "RunHealthMeasurementFailed",
                    {"collection_event_id": "c", "code": "timeout", "detail": "slow"},
                ),
                _event("RunHealthReviewRequested", {"collection_event_id": "c"}),
                _event(
                    "RunHealthReviewAnswered",
                    {"request_event_id": "r", "response": {"action": "abort"}},
                ),
            ]
        )
        self.assertEqual(
            [(line.status, line.text) for line in lines],
            [
                (
                    "verified",
                    "Health of your run: not measured: the run holds no local BAM",
                ),
                (
                    "verified",
                    "Health of your run: not measured: its measurement failed",
                ),
                ("failed", "Your run's health could not be measured: slow"),
                (
                    "question",
                    "Question for you: measure your run's health again, or go on?",
                ),
                ("answer", "Going on without your run's health"),
            ],
        )

    def test_a_merged_rsid_lookup_is_told(self):
        (merged,) = journey_feed(
            [
                _event(
                    "RefSnpToolResultRecorded",
                    {
                        "query": "rs60037639",
                        "result": {"status": "merged", "lookup_source": "ncbi_dbsnp"},
                    },
                )
            ]
        )

        self.assertEqual(
            merged.text, "rs60037639: dbSNP merged this rsID into a current one"
        )

    def test_an_interrupt_and_the_hold_it_leaves_are_told(self):
        turn, held, fault = journey_feed(
            [
                _event(
                    "ModelTurnCancelled",
                    {"attempt_id": str(uuid.uuid4()), "model_request_count": 2},
                ),
                _event("PassClosed", _held_pass("interrupted", None)),
                _event("PassClosed", _held_pass("invariant", "inspection_bug")),
            ]
        )

        self.assertEqual(
            (turn.status, turn.text),
            ("done", "The assistant's step was interrupted between turns"),
        )
        self.assertEqual(
            (held.status, held.text),
            ("done", "Work held after an interrupt; resume continues it"),
        )
        self.assertEqual(
            (fault.status, fault.text), ("done", "Work held after a problem")
        )

    def test_an_execution_outcome_keeps_its_own_disposition_as_its_status(self):
        # The current typed Outcome (journey_path.Outcome) records each disposition.
        items = journey_feed(
            [
                _event(
                    "ExecutionPathRecorded",
                    {
                        "event_type": "ExecutionOutcome",
                        "payload": Outcome(
                            disposition=disposition, contact_journal="journal.jsonl"
                        ).model_dump(mode="json"),
                    },
                )
                for disposition in ("succeeded", "failed", "incomplete")
            ]
        )

        self.assertEqual(
            [(item.status, item.text) for item in items],
            [
                ("done", "The analysis run finished successfully"),
                ("failed", "The analysis run failed"),
                ("done", "The analysis run paused before finishing"),
            ],
        )

    def test_a_payload_outside_its_current_contract_is_refused(self):
        # Retained Runs are evidence, not compatibility constraints (PROJECT.md): a
        # known event type is told from its current contract or not at all.
        for name, outside in (
            # A payload retained before its current contract: no `question`.
            ("missing field", {"issue": "An output is unresolved."}),
            ("wrong type", {"question": 42}),
        ):
            with self.subTest(name):
                with self.assertRaises(JourneyValidationError) as raised:
                    journey_feed(
                        [
                            _event("JourneyStarted", {}),
                            _event("PlanningHitlRequested", outside),
                        ]
                    )

                self.assertEqual(
                    str(raised.exception),
                    "Journey event on line 2 does not match the current"
                    " PlanningHitlRequested contract",
                )
        for event_type, outside in (
            # A value its line cannot format.
            ("ModelAvailabilityRetryScheduled", {"delay_seconds": "soon"}),
            # The field its level reads.
            ("AgentActionTurnDispositionRecorded", {"disposition": None}),
        ):
            with self.subTest(event_type):
                with self.assertRaises(JourneyValidationError) as raised:
                    journey_feed([_event(event_type, outside)])

                self.assertEqual(
                    str(raised.exception),
                    f"Journey event on line 1 does not match the current {event_type}"
                    " contract",
                )

    def test_items_carry_their_pass_and_payload_size_and_the_feed_its_head(self):
        # Fields the GUI reads: the Pass an item belongs to and how large its hidden
        # payload is; the head and count let a tail resume.
        journey_file = Path(self.prerequisite("executed").journey_file)
        events = _lines(journey_file)

        feed = read_journey_feed(str(journey_file))
        everything = read_journey_feed(str(journey_file), everything=True)

        self.assertEqual(
            (feed.event_count, feed.head_event_id),
            (len(events), events[-1]["event_id"]),
        )
        passes, expected = 0, []
        for event in events:
            passes += event["event_type"] == "PassStarted"
            expected.append(
                (
                    passes,
                    len(
                        json.dumps(
                            event["payload"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ),
                )
            )
        self.assertEqual(
            [(item.pass_number, item.payload_bytes) for item in everything.items],
            expected,
        )
        self.assertEqual(everything.items[0].pass_number, 0)

    def test_each_item_names_the_product_stage_of_its_event(self):
        # Product path stages (PROJECT.md): the request and its inputs, planning, the
        # Execution Review, Execution, then Investigation. Agent mechanics belong to the
        # stage they run in.
        journey_file = Path(self.prerequisite("planned-evidence").journey_file)
        events = _lines(journey_file)
        stages = {
            item.event_id: item.stage
            for item in read_journey_feed(str(journey_file), everything=True).items
        }

        def of(event_type):
            return [
                stages[event["event_id"]]
                for event in events
                if event["event_type"] == event_type
            ]

        self.assertEqual(of("ResearchRequestRegistered"), ["request"])
        self.assertEqual(of("InputMetadataInspected"), ["request"])
        self.assertEqual(of("ProposalRequirementsIdentified"), ["planning"])
        self.assertEqual(of("ExecutionProposalSubmitted"), ["execution_review"])
        self.assertEqual(of("ExecutionHitlResponded"), ["execution_review"])
        self.assertEqual(set(of("ExecutionPathRecorded")), {"execution"})
        self.assertEqual(of("InvestigationStarted"), ["investigation"])
        self.assertEqual(set(of("InvestigationReviewRequested")), {"investigation"})
        model_requests = of("ModelRequestRecorded")
        self.assertEqual(model_requests[0], "planning")
        self.assertEqual(model_requests[-1], "investigation")

    def test_a_forked_child_reads_from_its_request_to_its_published_report(self):
        journey_file = Path(self.prerequisite(_PUBLISHED).journey_file)
        items = read_journey_feed(str(journey_file), everything=True).items
        stage = {item.event_type: item.stage for item in items}

        self.assertEqual(items[0].event_type, "RunForked")
        self.assertEqual(items[0].stage, "request")
        self.assertEqual(stage["InvestigationStarted"], "investigation")
        self.assertEqual(stage["FinalizationBasisCompiled"], "finalization")
        self.assertEqual(stage["FinalizationReviewRecorded"], "finalization")
        self.assertEqual(stage["GroundedReportPublished"], "report")

    def test_a_half_written_last_line_waits_for_the_next_read(self):
        source = Path(self.prerequisite("executed").journey_file)
        with tempfile.TemporaryDirectory() as temporary:
            journey_file = Path(temporary) / "journey.jsonl"
            shutil.copyfile(source, journey_file)
            complete = len(_lines(journey_file))
            with journey_file.open("ab") as handle:
                handle.write(b'{"artifact_references": [], "contract_ver')

            feed = read_journey_feed(str(journey_file), everything=True)

            self.assertEqual(len(feed.items), complete)
            self.assertEqual(feed.items[-1].sequence, complete)


class RunArtifactsTests(RunPrerequisites, unittest.TestCase):
    def test_a_completed_run_lists_its_verified_files_with_roles_and_proofs(self):
        journey_file = Path(
            self.prerequisite(
                "executed", output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
            ).journey_file
        )
        events = _lines(journey_file)
        path_events = {
            event["payload"]["event_type"]: event["event_id"]
            for event in events
            if event["event_type"] == "ExecutionPathRecorded"
        }
        # The offline Execution's launch (tests/parabricks_operation_fakes.py).
        results = "s3://fresh-run-bucket/execution/operation/offline/"

        artifacts = read_run_artifacts(str(journey_file))

        self.assertEqual(artifacts.schema_version, "bio-neb.run-artifacts/1")
        self.assertEqual(
            (artifacts.journey_id, artifacts.journey_file, artifacts.run_root),
            (
                events[0]["journey_id"],
                str(journey_file),
                str(journey_file.parent.parent),
            ),
        )
        self.assertEqual(
            (artifacts.outcome_event_id, artifacts.verified_event_id),
            (path_events["ExecutionOutcome"], path_events["H100ArtifactsVerified"]),
        )
        self.assertEqual(
            (artifacts.bucket, artifacts.object_scope),
            ("fresh-run-bucket", "operation/offline"),
        )
        self.assertEqual(
            [(item.role, item.location) for item in artifacts.outputs],
            [
                ("aligned_bam", results + "aligned.bam"),
                ("VCF", results + "variants.vcf"),
                ("worker_receipt", results + "worker-receipt.json"),
                ("execution_manifest", results + "manifest.json"),
            ],
        )
        bam, vcf = artifacts.outputs[:2]
        for item, content in ((bam, BAM_BYTES), (vcf, VCF_BYTES)):
            self.assertEqual(
                (item.byte_count, item.sha256),
                (len(content), hashlib.sha256(content).hexdigest()),
            )
        self.assertEqual(
            [(item.role, item.producer_stage) for item in artifacts.inputs],
            [
                ("read_1", "input_storage"),
                ("read_2", "input_storage"),
                ("input_manifest", "input_storage"),
            ],
        )
        self.assertEqual(
            {item.role for item in artifacts.reference},
            {
                "genomic_fasta",
                "fasta_fai",
                "reference_dict",
                "bwa_amb",
                "bwa_ann",
                "bwa_bwt",
                "bwa_pac",
                "bwa_sa",
                "reference_source",
                "source_manifest",
                "prepared_reference_manifest",
                "worker_receipt",
            },
        )
        manifest = artifacts.inputs[-1]
        # A small document is retained in the Journey bundle by its digest.
        self.assertEqual(manifest.location, "artifacts/sha256/" + manifest.sha256)
        for item in (*artifacts.outputs, *artifacts.inputs, *artifacts.reference):
            self.assertEqual(item.grounding, "verified", item.role)
            self.assertRegex(item.proof_ref, r"^cj:1:[1-9][0-9]*$")
        self.assertEqual({item.producer_stage for item in artifacts.outputs}, {"h100"})
        self.assertEqual(
            {item.producer_stage for item in artifacts.reference}, {"cpu_preparation"}
        )

    def test_a_run_lists_the_vcf_and_index_its_own_observation_materialized(self):
        # Additive `local` entries name the Run's own VCF and its index with the
        # Journey's recorded identities and receipts; listing reads no file.
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from tests.observation_input_fakes import (
            RUN_VCF,
            TABLE_READING_ENVIRON,
            RunOutputStore,
        )
        from tests.sequential_investigation_support import (
            approved_collection_observation,
            collection_request,
        )
        from tests.terminal_journey_fakes import forked_child
        from tests.vcf_lookup_fakes import TableReaderModel

        parent = Path(
            self.prerequisite(
                "executed",
                investigation_request=collection_request(3),
                output_payloads={"VCF": RUN_VCF},
            ).journey_file
        )
        # A Run with no observation of its own keeps its earlier document.
        self.assertNotIn(
            "local", read_run_artifacts(str(parent)).model_dump(mode="json")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, journey, _, _, _ = approved_collection_observation(
                root / "observed",
                cardinality=3,
                approve_count=1,
                forked=True,
                parent_file=parent,
            )
            with use_run_advancement_dependencies(
                environ=TABLE_READING_ENVIRON,
                investigation_model_factory=TableReaderModel().model,
                investigation_vcf_source_factory=RunOutputStore(),
            ):
                advance(ResumeRun(journey_file=str(journey.path)))
            owned = journey.run_directory / "investigation"
            vcf = owned / "vcf" / "variants.vcf"
            (index,) = (owned / "derivation").rglob("*.csi")
            index_bytes = index.read_bytes()
            (preparation,) = (
                event["payload"]["sample_observation"]["preparation"]
                for event in _lines(journey.path)
                if event["event_type"] == "InvestigationEvidenceRecorded"
            )
            # The listing reads the Journey, not the files: it lists them gone.
            vcf.unlink()
            index.unlink()

            artifacts = read_run_artifacts(str(journey.path))

            self.assertEqual(artifacts.schema_version, "bio-neb.run-artifacts/1")
            self.assertEqual(
                [entry.model_dump(mode="json") for entry in artifacts.local],
                [
                    {
                        "role": "VCF",
                        "path": str(vcf),
                        "byte_count": len(RUN_VCF),
                        "sha256": hashlib.sha256(RUN_VCF).hexdigest(),
                        "receipt": str(
                            owned / "vcf" / "vcf-materialization-receipt.json"
                        ),
                    },
                    {
                        "role": "VCF_CSI",
                        "path": str(index),
                        "byte_count": len(index_bytes),
                        "sha256": hashlib.sha256(index_bytes).hexdigest(),
                        "receipt": preparation["receipt_id"],
                    },
                ],
            )
            # A Fork inherits the observation but not its files: none are its own.
            grandchild = forked_child(root / "fork", parent_file=journey.path)
            self.assertNotIn(
                "local",
                read_run_artifacts(str(grandchild.path)).model_dump(mode="json"),
            )

    def test_a_run_without_a_completed_execution_has_no_artifacts_yet(self):
        approved = self.prerequisite("approved").journey_file
        with tempfile.TemporaryDirectory() as directory:
            failed = approved_input(Path(directory).resolve() / "case")

            def collided(plan, journal):
                provider = input_storage_provider(plan, journal)
                provider.scripts["input_storage", "fresh_bucket"] = deque(
                    [ProviderError("semantic", "bucket_collision")]
                )
                return provider

            execute_parabricks(failed, infrastructure_factory=collided)

            for journey_file, code in (
                (approved, "parabricks_result_unavailable"),
                (failed.journey_file, "parabricks_execution_incomplete"),
            ):
                with self.subTest(code=code):
                    with self.assertRaises(RunArtifactsUnavailable) as raised:
                        read_run_artifacts(str(journey_file))
                    self.assertEqual(raised.exception.code, code)


class RunReaderCliTests(RunPrerequisites, unittest.TestCase):
    """The CLI prints the reader's documents; it reads no events itself."""

    def test_journey_feed_prints_the_default_view_or_every_event_with_payloads(self):
        journey_file = Path(self.prerequisite("executed").journey_file)
        events = _lines(journey_file)

        code, printed, error = _cli("journey-feed", "--journey-file", str(journey_file))

        self.assertEqual((code, error), (0, ""))
        feed = json.loads(printed)
        self.assertEqual(
            feed, read_journey_feed(str(journey_file)).model_dump(mode="json")
        )
        self.assertEqual(feed["schema_version"], "bio-neb.journey-feed/1")
        self.assertLess(len(feed["items"]), len(events))
        self.assertTrue(all("payload" not in item for item in feed["items"]))

        code, printed, error = _cli(
            "journey-feed", "--journey-file", str(journey_file), "--all", "--payloads"
        )

        self.assertEqual((code, error), (0, ""))
        self.assertEqual(
            [
                (item["event_id"], item["payload"])
                for item in json.loads(printed)["items"]
            ],
            [(event["event_id"], event["payload"]) for event in events],
        )

    def test_an_absent_or_unreadable_journey_is_refused_as_an_invalid_journey(self):
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "absent" / "journey.jsonl"
            malformed = Path(temporary) / "journey.jsonl"
            malformed.write_text('{"event_type": "JourneyStarted"}\n')

            for journey_file in (absent, malformed):
                for command in ("journey-feed", "run-artifacts"):
                    with self.subTest(command=command, journey_file=journey_file.name):
                        code, printed, error = _cli(
                            command, "--journey-file", str(journey_file)
                        )

                        self.assertEqual((code, printed), (2, ""))
                        refusal = json.loads(error)
                        self.assertEqual(refusal["code"], "journey_review_invalid")
                        self.assertEqual(refusal["fields"][0], "journey_file")

    def test_journey_feed_refuses_a_payload_outside_its_current_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            journey_file = Path(temporary) / "journey.jsonl"
            journey_file.write_text(
                "".join(
                    event.model_dump_json() + "\n"
                    for event in (
                        _event("JourneyStarted", {}),
                        _event("PlanningHitlRequested", {"question": 42}),
                    )
                )
            )

            code, printed, error = _cli(
                "journey-feed", "--journey-file", str(journey_file), "--all"
            )

            self.assertEqual((code, printed), (2, ""))
            refusal = json.loads(error)
            self.assertEqual(refusal["code"], "journey_review_invalid")
            self.assertEqual(refusal["fields"][0], "journey_file")

    def test_run_artifacts_prints_the_listing_or_refuses_a_run_without_one(self):
        executed = self.prerequisite("executed").journey_file
        approved = self.prerequisite("approved").journey_file

        code, printed, error = _cli("run-artifacts", "--journey-file", str(executed))

        self.assertEqual((code, error), (0, ""))
        self.assertEqual(
            json.loads(printed),
            read_run_artifacts(str(executed)).model_dump(mode="json"),
        )

        code, printed, error = _cli("run-artifacts", "--journey-file", str(approved))

        self.assertEqual((code, printed), (2, ""))
        self.assertEqual(
            json.loads(error),
            {
                "status": "failed",
                "code": "parabricks_result_unavailable",
                "fields": ["run-artifacts"],
            },
        )


if __name__ == "__main__":
    unittest.main()
