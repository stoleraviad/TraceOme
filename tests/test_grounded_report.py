from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.fork import own_fork_origin
from bio_neb.cli import main
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from bio_neb.reporting.finalization import project_finalization
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    CorrectFinalizationCommand,
    StopFinalizationCommand,
    approve_finalization,
    compile_finalization,
    correct_finalization,
    stop_finalization,
)
from bio_neb.reporting.grounded_report import (
    REPORT_MEDIA_TYPE,
    GroundedReport,
    GroundedReportPublished,
    derive_grounded_report,
)
from bio_neb.reporting.report_commands import (
    PublishGroundedReportCommand,
    ReopenGroundedReportCommand,
    publish_grounded_report,
    reopen_grounded_report,
)
from tests.finalization_fakes import evidence_journey
from tests.prerequisite_builds import measured_build
from tests.run_prerequisites import RunPrerequisites, State
from tests.terminal_journey_fakes import EVIDENCE_REQUEST
from tests.vcf_lookup_fakes import query_row


def _run(argv: list[str]) -> tuple[int, str, str]:
    output, error = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        exit_code = main(argv)
    return exit_code, output.getvalue(), error.getvalue()


@measured_build("grounded-report:approved-finalization")
def _approved_journey(
    root: Path, *, parent_file: Path, query_stdout: bytes = b""
) -> tuple[Journey, str, str]:
    journey = evidence_journey(root, query_stdout=query_stdout, parent_file=parent_file)
    pending = compile_finalization(
        CompileFinalizationCommand(journey_file=str(journey.path))
    ).pending_request
    assert pending is not None
    head_event_id = Journey.open(journey.path).read()[-1]["event_id"]
    approved = approve_finalization(
        ApproveFinalizationCommand(
            journey_file=str(journey.path),
            request_event_id=pending.request_event_id,
            basis_digest=pending.basis.basis_digest,
            head_event_id=head_event_id,
        )
    )
    assert approved.report_basis is not None
    return (
        Journey.open(journey.path),
        approved.report_basis.report_basis_digest,
        Journey.open(journey.path).read()[-1]["event_id"],
    )


def _approved_finalization(
    root: Path, parent_file: Path, *, query_stdout: bytes = b""
) -> tuple[Path, tuple[str, str]]:
    journey, basis_digest, approval_head = _approved_journey(
        root, parent_file=parent_file, query_stdout=query_stdout
    )
    return journey.path, (basis_digest, approval_head)


# The evidence child begins Investigation with EVIDENCE_REQUEST, so its parent
# was planned for that registered request.
_APPROVED_FINALIZATION = State(
    "approved-finalization",
    _approved_finalization,
    forks_from="executed",
    forks_from_choices={"investigation_request": EVIDENCE_REQUEST},
)


class GroundedReportPublicPathTests(RunPrerequisites, unittest.TestCase):
    def evidence_ready(self) -> Journey:
        """Observed evidence before Finalization, restored in place."""
        return Journey.open(self.prerequisite("retained-evidence").journey_file)

    def approved(
        self, *, query_stdout: bytes = b"", slot: str = "default"
    ) -> tuple[Journey, str, str]:
        """An approved Report basis forked from the shared parent, restored in place."""
        baseline = self.prerequisite(
            _APPROVED_FINALIZATION, slot=slot, query_stdout=query_stdout
        )
        return (Journey.open(baseline.journey_file), *baseline.value)

    def test_the_rendered_report_reads_its_answer_before_its_run_accounting(
        self,
    ) -> None:
        # ~900 lines of raw accounting opened the report before the question. The
        # researcher's sections come first; accounting and evidence pointers
        # close it, in that order.
        journey, basis_digest, approval_head = self.approved()

        exit_code, text, stderr = _run(
            [
                "grounded-report-publish",
                "--journey-file",
                str(journey.path),
                "--report-basis-digest",
                basis_digest,
                "--head-event-id",
                approval_head,
            ]
        )

        self.assertEqual((exit_code, stderr), (0, ""))
        headings = [
            line
            for line in text.splitlines()
            if line
            in {"Question", "Review authority", "Run accounting", "Evidence pointers"}
        ]
        self.assertEqual(
            headings,
            ["Question", "Review authority", "Run accounting", "Evidence pointers"],
        )
        self.assertTrue(text.split("\nEvidence pointers\n")[1].startswith("- "))

    def test_public_publish_reopen_is_deterministic_grounded_and_projection_equivalent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, basis_digest, approval_head = self.approved()
            before = journey.path.read_bytes()
            publish_argv = [
                "grounded-report-publish",
                "--journey-file",
                str(journey.path),
                "--report-basis-digest",
                basis_digest,
                "--head-event-id",
                approval_head,
            ]

            exit_code, published_cli, stderr = _run(publish_argv)

            self.assertEqual((exit_code, stderr), (0, ""))
            self.assertIn("Grounded report\n", published_cli)
            self.assertIn("Question\n", published_cli)
            self.assertIn("Requested and verified identities\n", published_cli)
            self.assertIn(
                "  requested: rs429358\n"
                "  verified: rs429358\n"
                "  status: verified\n"
                "  verification source: NCBI RefSNP\n"
                "  mismatch: false\n"
                "  assessment: verified by NCBI RefSNP",
                published_cli,
            )
            self.assertIn("rs429358", published_cli)
            self.assertIn("sample-001", published_cli)
            self.assertIn("NC_000019.10:44908684 T>C", published_cli)
            self.assertIn("outcome: no_record", published_cli)
            self.assertIn("disposition: unresolved", published_cli)
            self.assertIn("admission: admitted", published_cli)
            self.assertIn("membership: included", published_cli)
            self.assertIn(
                "run provenance: causal_child_or_fork",
                published_cli,
            )
            origin_event_id = own_fork_origin(Journey.open(journey.path))[0]
            self.assertIn(f"provenance origin event: {origin_event_id}", published_cli)
            self.assertNotIn("non-fresh-development-clone:", published_cli)
            self.assertIn("report external contact count: 0", published_cli)
            self.assertNotIn("secret-value-not-retained", published_cli)
            published = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=basis_digest,
                    head_event_id=approval_head,
                )
            )
            report = published.report
            assert report is not None
            self.assertEqual(report.schema_version, "bio-neb.grounded-report/7")
            self.assertEqual(report.run_provenance, "causal_child_or_fork")
            self.assertEqual(report.provenance_origin_event_id, origin_event_id)
            self.assertEqual(report.report_basis_digest, basis_digest)
            self.assertEqual(report.source_head_event_id, approval_head)
            self.assertEqual(report.report_external_contact_count, 0)
            contradictory_report = report.model_dump(
                mode="json", exclude_computed_fields=True
            )
            contradictory_report["provenance_origin_event_id"] = None
            with self.assertRaisesRegex(ValueError, "provenance"):
                GroundedReport.model_validate_json(
                    canonical_json(contradictory_report), strict=True
                )
            self.assertEqual(
                tuple(
                    sorted(item.membership.member_id for item in report.all_members())
                ),
                tuple(item.member_id for item in report.report_membership),
            )
            retained = Journey.open(journey.path)
            report_event = retained.read()[-1]
            self.assertEqual(report_event["event_type"], "GroundedReportPublished")
            self.assertEqual(len(report_event["artifact_references"]), 1)
            report_artifact = retained.read_artifact(
                report_event["artifact_references"][0]
            )
            self.assertEqual(report_artifact, report.canonical_bytes())
            # The public artifact supplies the vector, with test-owned identity
            # membership. Its newline-framed byte hash is a separate contract.
            artifact_document = json.loads(report_artifact)
            identity_fields = (
                "schema_version renderer_version journey_id source_head_event_id outcome accounting fact_slots "
                "report_basis_digest finalization_basis_digest evidence_state_id "
                "selected_work_item_ids collection_item_states collection_status "
                "review_authority question work_items retrieved_sources source_assessments "
                "sample_observations candidate_evidence observation_assessments admissions "
                "report_membership conflicts "
                "limitations gaps evidence_pointers inherited_model_contact_count "
                "inherited_source_contact_count report_external_contact_count "
                "run_provenance provenance_origin_event_id collection_view "
                "run_health method"
            ).split()
            self.assertEqual(
                set(artifact_document), {*identity_fields, "report_digest"}
            )
            report_material = {
                name: artifact_document[name] for name in identity_fields
            }
            expected_digest = hashlib.sha256(
                json.dumps(
                    report_material,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(report.report_digest, "grounded-report:" + expected_digest)
            self.assertTrue(report_artifact.endswith(b"\n"))
            self.assertNotEqual(
                hashlib.sha256(report_artifact).hexdigest(), expected_digest
            )
            self.assertNotIn("response_id", report_material["review_authority"])
            self.assertNotIn("pointer_id", report_material["evidence_pointers"][0])
            rich_argv = ["observe", "--journey-file", str(journey.path), "--rich"]
            exit_code, projection_json, stderr = _run(rich_argv)
            self.assertEqual((exit_code, stderr), (0, ""))
            projection = json.loads(projection_json)
            self.assertEqual(projection["status"], "grounded_report_published")
            self.assertEqual(
                projection["grounded_report"]["report"],
                report.model_dump(mode="json"),
            )
            reopen_argv = [
                "grounded-report-reopen",
                "--journey-file",
                str(journey.path),
                "--report-digest",
                report.report_digest,
            ]
            first_reopen = _run(reopen_argv)
            second_reopen = _run(reopen_argv)
            self.assertEqual(first_reopen, (0, published_cli, ""))
            self.assertEqual(second_reopen, first_reopen)
            self.assertNotEqual(journey.path.read_bytes(), before)
            after_publish = journey.path.read_bytes()
            self.assertEqual(_run(publish_argv), (0, published_cli, ""))
            self.assertEqual(journey.path.read_bytes(), after_publish)
            for state, content in (("approved", before), ("published", after_publish)):
                with self.subTest(origin_state=state):
                    destination = (
                        Path(temporary) / f"hostile-{state}" / journey.journey_id
                    )
                    shutil.copytree(journey.run_directory, destination)
                    path = destination / "journey.jsonl"
                    events = [json.loads(line) for line in content.splitlines()]
                    events[0]["payload"]["committed_candidate_sha"] = "b" * 40
                    path.write_text(
                        "".join(canonical_json(event) + "\n" for event in events),
                        encoding="utf-8",
                    )
                    hostile_bytes = path.read_bytes()
                    with self.assertRaises(ValueError):
                        publish_grounded_report(
                            PublishGroundedReportCommand(
                                journey_file=str(path),
                                report_basis_digest=basis_digest,
                                head_event_id=approval_head,
                            )
                        )
                    if state == "published":
                        with self.assertRaises(ValueError):
                            reopen_grounded_report(
                                ReopenGroundedReportCommand(
                                    journey_file=str(path),
                                    report_digest=report.report_digest,
                                )
                            )
                    self.assertEqual(path.read_bytes(), hostile_bytes)

    def test_public_render_preserves_positive_negative_partial_conflicted_gap_and_non_admitted_semantics(  # noqa: E501
        self,
    ) -> None:
        def row(genotype: str, **quality: str) -> bytes:
            return query_row(
                "NC_000019.10", 44908684, "T", "C", "sample-001", genotype, **quality
            )

        exact = row("0/1")
        contradicted = row("0/0")
        unavailable = row("./.")
        ambiguous = exact + exact
        # G1: a LowQual call keeps its quality in the report and is still admitted.
        low_quality = row("0/1", filters="LowQual", qual="3", gq="15", dp="5", ad="2,3")
        cases = (
            (exact, "exact_allele_observed", "supported", "admitted", None),
            (low_quality, "exact_allele_observed", "supported", "admitted", None),
            (
                contradicted,
                "target_allele_not_carried",
                "contradicted",
                "admitted",
                "The sample does not carry the requested target ALT",
            ),
            (
                unavailable,
                "target_genotype_unresolved",
                "unresolved",
                "not_admitted",
                "Automatic claim admission is unavailable",
            ),
            (
                ambiguous,
                "ambiguous_records",
                "partial",
                "not_admitted",
                "Automatic claim admission is unavailable",
            ),
            (b"", "no_record", "unresolved", "admitted", None),
        )
        for query, outcome, disposition, admission, visible in cases:
            with self.subTest(outcome=outcome):
                journey, basis_digest, approval_head = self.approved(query_stdout=query)
                report_basis = project_finalization(read_evidence(journey)).report_basis
                assert report_basis is not None
                if admission == "not_admitted":
                    decision = report_basis.finalization_basis.admissions[0]
                    decision_json = decision.model_dump(mode="json")
                    fields = (
                        "status investigation_authority_id work_item_id candidate_id "
                        "assessment_id observation_id reason claim_scope"
                    ).split()
                    self.assertEqual(set(decision_json), {*fields, "admission_id"})
                    material = {name: decision_json[name] for name in fields}
                    expected = hashlib.sha256(
                        json.dumps(
                            material,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    self.assertEqual(
                        decision.admission_id,
                        "observation-evidence-not-admitted:" + expected,
                    )
                exact_observation = report_basis.finalization_basis.sample_observations[
                    0
                ]
                source_result = exact_observation.result.model_dump(
                    mode="json", exclude_none=True
                )
                report_safe_fields = (
                    "schema_version",
                    "outcome",
                    "target",
                    "record",
                    "other_records",
                    "records",
                    "matching_records",
                    "genotype",
                    "code",
                    "violations",
                    "claim_scope",
                )
                expected_result = {
                    field: source_result[field]
                    for field in report_safe_fields
                    if field in source_result
                }
                result = _run(
                    [
                        "grounded-report-publish",
                        "--journey-file",
                        str(journey.path),
                        "--report-basis-digest",
                        basis_digest,
                        "--head-event-id",
                        approval_head,
                    ]
                )
                self.assertEqual((result[0], result[2]), (0, ""))
                published = Journey.open(journey.path).read()[-1]["payload"]["report"]
                observation = published["sample_observations"][0]["value"]
                self.assertEqual(observation["result"], expected_result)
                self.assertNotIn("outcome", observation)
                self.assertNotIn("claim", observation)
                self.assertIn(
                    "proposed_claim",
                    published["candidate_evidence"][0]["value"],
                )
                admission_value = published["admissions"][0]["value"]
                self.assertIn("claim_scope", admission_value)
                if admission == "admitted":
                    self.assertIn("admitted_claim", admission_value)
                else:
                    self.assertNotIn("admitted_claim", admission_value)
                self.assertIn(f"outcome: {outcome}", result[1])
                self.assertIn(f"disposition: {disposition}", result[1])
                self.assertIn(f"admission: {admission}", result[1])
                if visible is not None:
                    self.assertIn(visible, result[1])
                self.assertIn(
                    "Coverage, callability, biological absence, function, causality, "
                    "and diagnosis are not inferred.",
                    result[1],
                )

    def test_public_publish_refuses_non_authoritative_stale_or_mismatched_basis_before_mutation_and_is_idempotent(  # noqa: E501
        self,
    ) -> None:
        def artifacts(journey: Journey) -> tuple[tuple[str, bytes], ...]:
            root = journey.run_directory / "artifacts"
            return tuple(
                (path.relative_to(root).as_posix(), path.read_bytes())
                for path in sorted(root.rglob("*"))
                if path.is_file()
            )

        def refusal(argv: list[str], journey: Journey, code: str) -> None:
            before = journey.path.read_bytes()
            artifacts_before = artifacts(journey)
            result = _run(argv)
            self.assertEqual(result[0], 2)
            self.assertEqual(json.loads(result[2])["code"], code)
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(artifacts(journey), artifacts_before)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            absent = self.evidence_ready()
            refusal(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(absent.path),
                    "--report-basis-digest",
                    "report-basis:" + "0" * 64,
                    "--head-event-id",
                    absent.read()[-1]["event_id"],
                ],
                absent,
                "grounded_report_basis_absent",
            )

            unapproved = self.evidence_ready()
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(unapproved.path))
            ).pending_request
            assert pending is not None
            refusal(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(unapproved.path),
                    "--report-basis-digest",
                    "report-basis:" + "0" * 64,
                    "--head-event-id",
                    unapproved.read()[-1]["event_id"],
                ],
                unapproved,
                "grounded_report_basis_unapproved",
            )

            corrected = self.evidence_ready()
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(corrected.path))
            ).pending_request
            assert pending is not None
            correction_file = root / "correction.json"
            correction_file.write_text(
                canonical_json(
                    {
                        "schema_version": "bio-neb.finalization-correction/1",
                        "asserted_by": "researcher:owner",
                        "reason": "Retain but exclude one exact source assessment.",
                        "requested_changes": [
                            {
                                "action": "set_report_membership",
                                "member_id": pending.basis.source_assessments[
                                    0
                                ].assessment_id,
                                "included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            corrected_projection = correct_finalization(
                CorrectFinalizationCommand(
                    journey_file=str(corrected.path),
                    request_event_id=pending.request_event_id,
                    basis_digest=pending.basis.basis_digest,
                    head_event_id=Journey.open(corrected.path).read()[-1]["event_id"],
                    correction_file=str(correction_file),
                )
            )
            assert corrected_projection.pending_request is not None
            refusal(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(corrected.path),
                    "--report-basis-digest",
                    "report-basis:" + "0" * 64,
                    "--head-event-id",
                    Journey.open(corrected.path).read()[-1]["event_id"],
                ],
                corrected,
                "grounded_report_basis_corrected_unapproved",
            )

            stopped = self.evidence_ready()
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(stopped.path))
            ).pending_request
            assert pending is not None
            stop_file = root / "stop.json"
            stop_file.write_text(
                canonical_json(
                    {
                        "schema_version": "bio-neb.finalization-stop/1",
                        "asserted_by": "researcher:owner",
                        "reason": "Do not publish this basis.",
                    }
                ),
                encoding="utf-8",
            )
            stop_finalization(
                StopFinalizationCommand(
                    journey_file=str(stopped.path),
                    request_event_id=pending.request_event_id,
                    basis_digest=pending.basis.basis_digest,
                    head_event_id=Journey.open(stopped.path).read()[-1]["event_id"],
                    reason_file=str(stop_file),
                )
            )
            refusal(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(stopped.path),
                    "--report-basis-digest",
                    "report-basis:" + "0" * 64,
                    "--head-event-id",
                    Journey.open(stopped.path).read()[-1]["event_id"],
                ],
                stopped,
                "grounded_report_basis_stopped",
            )

            approved, basis_digest, approval_head = self.approved()
            base_argv = [
                "grounded-report-publish",
                "--journey-file",
                str(approved.path),
                "--report-basis-digest",
                basis_digest,
                "--head-event-id",
                approval_head,
            ]
            refusal(
                [*base_argv[:-3], "not-a-digest", *base_argv[-2:]],
                approved,
                "command_invalid",
            )
            refusal(
                [*base_argv[:-3], "report-basis:" + "0" * 64, *base_argv[-2:]],
                approved,
                "grounded_report_basis_mismatched",
            )
            refusal(
                [*base_argv[:-1], "00000000-0000-0000-0000-000000000000"],
                approved,
                "grounded_report_binding_stale",
            )
            # A second approved Run, so the first stays unpublished and current.
            stale, stale_basis_digest, stale_approval_head = self.approved(
                slot="stale-current-head"
            )
            unrelated = stale.record(
                "UnrelatedPostApprovalEvent",
                {"does_not_grant_report_authority": True},
                [stale_approval_head],
            )
            refusal(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(stale.path),
                    "--report-basis-digest",
                    stale_basis_digest,
                    "--head-event-id",
                    unrelated["event_id"],
                ],
                stale,
                "grounded_report_binding_stale",
            )
            first = _run(base_argv)
            self.assertEqual((first[0], first[2]), (0, ""))
            published_bytes = approved.path.read_bytes()
            self.assertEqual(_run(base_argv), first)
            self.assertEqual(approved.path.read_bytes(), published_bytes)

    def test_public_reopen_refuses_unverifiable_artifact_without_contact_or_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            approved, basis_digest, approval_head = self.approved()
            hostile_root = root / "hostile-lineage"
            shutil.copytree(approved.run_directory, hostile_root / approved.journey_id)
            hostile = Journey.open(hostile_root / approved.journey_id / "journey.jsonl")
            unrelated = hostile.record(
                "HostileUnrelatedAppend",
                {"does_not_grant_report_authority": True},
                [approval_head],
            )
            report_basis = project_finalization(read_evidence(hostile)).report_basis
            assert report_basis is not None
            hostile_report = derive_grounded_report(
                report_basis,
                journey_id=hostile.journey_id,
                source_head_event_id=unrelated["event_id"],
            )
            hostile_fact = GroundedReportPublished(report=hostile_report)
            hostile.record_with_artifact(
                hostile_fact.event_type,
                hostile_fact.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                hostile_report.canonical_bytes(),
                REPORT_MEDIA_TYPE,
                parent_ids=[unrelated["event_id"]],
            )
            hostile_before = hostile.path.read_bytes()
            with (
                patch(
                    "socket.create_connection", side_effect=AssertionError("contact")
                ),
                patch("subprocess.run", side_effect=AssertionError("dependency")),
            ):
                hostile_result = _run(
                    [
                        "grounded-report-reopen",
                        "--journey-file",
                        str(hostile.path),
                        "--report-digest",
                        hostile_report.report_digest,
                    ]
                )
            self.assertEqual(hostile_result[0], 2)
            self.assertEqual(
                json.loads(hostile_result[2])["code"],
                "grounded_report_basis_unverifiable",
            )
            self.assertEqual(hostile.path.read_bytes(), hostile_before)

            publish_result = _run(
                [
                    "grounded-report-publish",
                    "--journey-file",
                    str(approved.path),
                    "--report-basis-digest",
                    basis_digest,
                    "--head-event-id",
                    approval_head,
                ]
            )
            self.assertEqual((publish_result[0], publish_result[2]), (0, ""))
            retained = Journey.open(approved.path)
            report_event = retained.read()[-1]
            report_digest = report_event["payload"]["report"]["report_digest"]
            artifact_path = retained.artifact_path(
                report_event["artifact_references"][0]
            )
            artifact_path.write_bytes(b"hostile-artifact-corruption")
            corrupted = artifact_path.read_bytes()
            journey_before = approved.path.read_bytes()
            with (
                patch(
                    "socket.create_connection", side_effect=AssertionError("contact")
                ),
                patch("subprocess.run", side_effect=AssertionError("dependency")),
            ):
                corrupted_result = _run(
                    [
                        "grounded-report-reopen",
                        "--journey-file",
                        str(approved.path),
                        "--report-digest",
                        report_digest,
                    ]
                )
            self.assertEqual(corrupted_result[0], 2)
            self.assertEqual(
                json.loads(corrupted_result[2])["code"],
                "grounded_report_basis_unverifiable",
            )
            self.assertEqual(approved.path.read_bytes(), journey_before)
            self.assertEqual(artifact_path.read_bytes(), corrupted)
