"""Public contract tests for ordered sequential Investigation collections."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.application.advancement import (
    advance,
    review_response_from_json,
    use_run_advancement_dependencies,
)
from bio_neb.application.queries import (
    ObserveRun,
    observe,
    project_journey_events,
    read_investigation,
    read_table_reading_stage,
)
from bio_neb.investigation.investigation import (
    AddInvestigationMember,
    AdjustInvestigationSearch,
    AmendInvestigationMember,
    ApproveInvestigationFindingResponse,
    ApproveInvestigationResponse,
    AssemblyReportToolResult,
    ClarifyInvestigationMember,
    DeferInvestigationMember,
    EvidenceAssessment,
    InvestigationAuthority,
    InvestigationInput,
    InvestigationProposal,
    InvestigationRequestItem,
    InvestigationVcfIdentity,
    KeepInvestigationMember,
    RefSnpToolResult,
    RemoveInvestigationMember,
    ResolvedVariantCandidate,
    RetrievedSource,
    RetryInvestigationMember,
    ReviseInvestigationResponse,
    StatedPosition,
    UserAssertion,
    VariantTarget,
    WorkItem,
    compile_investigation_collection_proposal,
    project_investigation,
)
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    ContinueInvestigationCommand,
    InvestigationApplicationRefused,
    RespondInvestigationReviewCommand,
    continue_investigation,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
    InvestigationCollectionItemState,
    InvestigationObservationCommand,
    fold_investigation_collection,
)
from bio_neb.investigation.investigation_observation_commands import (
    investigation_observe,
)
from bio_neb.investigation.vcf_observation import CompletedProcess
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.execution_contact_accounting import (
    compile_execution_contact_evidence,
)
from bio_neb.recording import read_evidence
from bio_neb.reporting.finalization import (
    FinalizationCorrectionAssertion,
    InvestigationRequiredChange,
    SetReportMembershipChange,
    project_finalization,
)
from bio_neb.reporting.finalization_commands import (
    ApproveFinalizationCommand,
    CompileFinalizationCommand,
    CorrectFinalizationCommand,
    FinalizationApplicationRefused,
    approve_finalization,
    compile_finalization,
    correct_finalization,
)
from bio_neb.reporting.report_commands import (
    GroundedReportApplicationRefused,
    PublishGroundedReportCommand,
    ReopenGroundedReportCommand,
    publish_grounded_report,
    reopen_grounded_report,
)
from bio_neb.verified_artifacts import VerifiedObject
from tests.model_endpoint_fakes import corrections
from tests.run_prerequisites import RunPrerequisites, State
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    REFSNP_BODY,
    SourceResponsesOnly,
    amend_to_partial,
    begin_in_pass,
    collection_request,
    forked_child,
    members_one_at_a_time,
)
from tests.sequential_investigation_support import (
    ObservationAdapter as _Adapter,
)
from tests.sequential_investigation_support import (
    ObservationEvidence as _Evidence,
)
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel as _SequentialAdaptiveModel,
)
from tests.sequential_investigation_support import (
    SequentialObservationAdapter as _SequentialObservationAdapter,
)
from tests.sequential_investigation_support import (
    SequentialSourceResponses as _SequentialSourceResponses,
)
from tests.sequential_investigation_support import (
    approved_collection_observation as _approved_collection_observation,
)
from tests.sequential_investigation_support import (
    run_attempt as _run_attempt,
)
from tests.terminal_journey_fakes import EVIDENCE_REQUEST
from tests.vcf_lookup_fakes import (
    TableReaderModel,
    lookup_command,
    query_position,
    query_region,
    query_row,
    run_table_reading,
    vcf_header,
)


def _texts(qualifications):
    """Each qualification's sentence (#448 D2: the basis keeps its ids too)."""
    return tuple(item.text for item in qualifications)


def _item_windows(journey: Journey) -> int:
    """Each approved item's window, a VCF contact after its lookup (ADR 0047)."""
    return sum(
        event["event_type"] == "InvestigationItemWindowRecorded"
        for event in read_evidence(Journey.open(journey.path))
    )


def _post_execution_vcf_queries(journey: Journey) -> int | None:
    review = compile_execution_contact_evidence(
        journal_bytes=b"",
        binding=None,
        expected_execution_id="parabricks-execution:" + "a" * 64,
        journey_events=read_evidence(Journey.open(journey.path)),
        journey_binding="journey:" + "b" * 64,
    )
    return review.ledger.post_execution_vcf_query.count


def _target(*, position: int = 44908684) -> VariantTarget:
    material = {
        "assembly_accession": "GCF_000001405.40",
        "assembly_name": "GRCh38.p14",
        "contig": "NC_000019.10",
        "one_based_position": position,
        "ref": "T",
        "alt": "C",
        "aliases": ("rs429358",),
        "assessment_ids": (
            "evidence-assessment:" + "1" * 64,
            "evidence-assessment:" + "2" * 64,
        ),
    }
    digest = hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()
    return VariantTarget(target_id=f"variant-target:{digest}", **material)


def _source_pair(
    rsid: str, position: int
) -> tuple[RefSnpToolResult, AssemblyReportToolResult]:
    refsnp_source = RetrievedSource.create(
        operation="fetch_refsnp",
        request_identity=f"https://example.test/refsnp/{rsid}",
        status_code=200,
        observed_at=f"2026-08-28T00:00:{position % 60:02d}Z",
        elapsed_time_ms=1,
        raw_response=f"refsnp:{rsid}:{position}".encode(),
    )
    refsnp_assessment = EvidenceAssessment.create(
        source_id=refsnp_source.source_id,
        verdict="accepted",
        rationale="One exact assembly-bound biallelic SNV was retained.",
    )
    refsnp = RefSnpToolResult.create(
        requested_rsid=rsid,
        expected_assembly_accession="GCF_000001405.40",
        status="resolved",
        candidates=(
            ResolvedVariantCandidate(
                assembly_accession="GCF_000001405.40",
                assembly_name="GRCh38.p14",
                contig="NC_000019.10",
                one_based_position=position,
                ref="T",
                alt="C",
                aliases=(rsid,),
            ),
        ),
        conflicts=(),
        source=refsnp_source,
        assessment=refsnp_assessment,
        lookup_source="ncbi_dbsnp",
        eva="not_applicable",
        location="not_applicable",
    )
    assembly_source = RetrievedSource.create(
        operation="fetch_assembly_report",
        request_identity="https://example.test/assembly/GCF_000001405.40",
        status_code=200,
        observed_at=f"2026-08-28T00:01:{position % 60:02d}Z",
        elapsed_time_ms=1,
        raw_response=f"assembly:{rsid}:{position}".encode(),
    )
    assembly = AssemblyReportToolResult.create(
        expected_assembly_accession="GCF_000001405.40",
        expected_assembly_name="GRCh38.p14",
        status="compatible",
        observed_assembly_accession="GCF_000001405.40",
        observed_assembly_name="GRCh38.p14",
        conflicts=(),
        source=assembly_source,
        assessment=EvidenceAssessment.create(
            source_id=assembly_source.source_id,
            verdict="accepted",
            rationale="The exact assembly identity was retained.",
        ),
    )
    return refsnp, assembly


def _approve_issue(journey, review):
    """Approve one issue finding as checked, not verified (#359)."""
    return respond_investigation_review(
        RespondInvestigationReviewCommand(
            journey_file=str(journey.path),
            request_event_id=review.request_event_id,
            response=ApproveInvestigationFindingResponse(
                action="approve",
                collection_member_id=review.question.collection_member_id,
                revision_number=review.question.revision_number,
                asserted_by="researcher:owner",
            ),
        )
    )


class _RefusingSequentialObservationAdapter(_SequentialObservationAdapter):
    def __init__(self, *args, refused_query_index: int, **kwargs) -> None:  # noqa: ANN002,ANN003
        super().__init__(*args, **kwargs)
        self.refused_query_index = refused_query_index
        self.query_count = 0

    def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001, ANN201
        if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file"):
            query_index = self.query_count
            self.query_count += 1
            if query_index == self.refused_query_index:
                self.calls.append(tuple(argv))
                raise OSError("bounded sequential query refusal")
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


# The shared-observation failure is kept at four positions. The query inspects no
# prepared file (0029), so the sample identity of the query output is the one
# shared kind left.
_SHARED_OBSERVATION_FAILURES = (
    (1, 0, "sample"),
    (4, 0, "sample"),
    (4, 1, "sample"),
    (4, 3, "sample"),
)
_UNKNOWN_OBSERVATION_FAILURES = (
    (1, 0, "query"),
    (4, 0, "query"),
    (4, 1, "query"),
    (4, 3, "query"),
    (1, 0, "evidence"),
    (4, 0, "evidence"),
)
_REFUSAL_POSITIONS = ((1, 0), (4, 0), (4, 1), (4, 3))


def _approved_collection(root: Path, parent_file: Path, *, cardinality: int):
    investigation_input, journey, authority, descriptor, command = (
        _approved_collection_observation(
            root, cardinality=cardinality, forked=True, parent_file=parent_file
        )
    )
    return journey.path, (investigation_input, authority, descriptor, command)


_APPROVED_COLLECTION = State(
    "approved-collection",
    _approved_collection,
    forks_from="executed",
    # The parent's plan names the collection's request.
    forks_from_choices=lambda *, cardinality: {
        "investigation_request": collection_request(cardinality)
    },
)


class SequentialInvestigationPublicPathTests(RunPrerequisites, unittest.TestCase):
    def approved_collection(self, *, cardinality: int):
        """An approved collection forked from the shared parent, restored in place.

        Returns the builder's tuple with the Journey reopened from the restored
        file; each case still observes, replays and finalizes itself.
        """

        baseline = self.prerequisite(_APPROVED_COLLECTION, cardinality=cardinality)
        investigation_input, authority, descriptor, command = baseline.value
        return (
            investigation_input,
            Journey.open(baseline.journey_file),
            authority,
            descriptor,
            command,
        )

    def test_owned_journey_reads_survive_observation_and_later_approval(self) -> None:
        from bio_neb.application import investigation_read, queries
        from bio_neb.investigation.investigation_observation_commands import (
            InvestigationObservationApplicationRefused,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, authority, descriptor, command = (
                _approved_collection_observation(
                    root,
                    cardinality=3,
                    forked=True,
                    approve_count=2,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(3)
                    ).journey_file,
                )
            )
            owner = Journey.open(journey.path)
            paths = {
                owner.artifact_path(reference)
                for event in owner.read()
                for reference in event["artifact_references"]
            }
            self.assertTrue(paths)
            reads = []
            original_read = Path.read_bytes

            def read_bytes(path):
                if path in paths:
                    reads.append(path)
                return original_read(path)

            with (
                patch.object(Path, "read_bytes", read_bytes),
                patch.object(
                    investigation_read,
                    "project_investigation_evidence",
                    wraps=investigation_read.project_investigation_evidence,
                ) as assembled,
            ):
                view = queries.read_investigation(owner)
                queries.read_investigation(owner)
                self.assertEqual(assembled.call_count, 1)
                self.assertIsNone(view.evidence)
                first = investigation_observe(
                    command,
                    journey=owner,
                    adapter=_SequentialObservationAdapter(
                        command, descriptor, investigation_input.sample_id
                    ),
                    evidence_reader=_Evidence(descriptor),
                )
                self.assertEqual(assembled.call_count, 1)
                self.assertEqual(reads, [])
                view = queries.read_investigation(owner)
                self.assertEqual(assembled.call_count, 2)
                self.assertEqual(view.evidence, first)
                self.assertEqual(
                    [member.state.status for member in view.active_members],
                    ["completed", "completed", "approval_pending"],
                )
                self.assertIsNone(view.active_members[-1].recorded)
                self.assertEqual(len(view.recorded_members), 2)
                original_ids = tuple(
                    member.event["event_id"] for member in view.recorded_members
                )
                self.assertEqual(
                    tuple(
                        member.evidence.authority for member in view.recorded_members
                    ),
                    (authority, authority),
                )

                self.assertEqual(
                    [item.status for item in first.item_states],
                    ["completed", "completed", "approval_pending"],
                )
                pending = view.investigation.pending_request
                assert pending is not None
                approved = respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=pending.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=authority.proposal_id,
                            work_item_ids=view.investigation.pending_work_item_ids,
                        ),
                    )
                )
                self.assertEqual(len(reads), len(paths))  # independent approval owner
                later = command.model_copy(
                    update={"authority_id": approved.authority.authority_id}
                )
                adapter = _SequentialObservationAdapter(
                    later, descriptor, investigation_input.sample_id
                )
                with self.assertRaises(
                    InvestigationObservationApplicationRefused
                ) as stale:
                    investigation_observe(
                        later,
                        journey=owner,
                        adapter=adapter,
                        evidence_reader=_Evidence(descriptor),
                    )
                self.assertEqual(
                    stale.exception.code, "investigation_observation_context_stale"
                )
                self.assertEqual(adapter.calls, [])
                owner.refresh()
                self.assertEqual(
                    queries.read_investigation(owner).investigation.authority,
                    approved.authority,
                )
                second = investigation_observe(
                    later,
                    journey=owner,
                    adapter=adapter,
                    evidence_reader=_Evidence(descriptor),
                )
                self.assertEqual(len(reads), len(paths))
                self.assertEqual(second.item_states[:2], first.item_states[:2])
                self.assertEqual(
                    [item.status for item in second.item_states], ["completed"] * 3
                )
                self.assertEqual(queries.read_investigation(owner).evidence, second)
                latest = queries.read_investigation(owner)
                self.assertEqual(
                    tuple(
                        member.event["event_id"]
                        for member in latest.recorded_members[:2]
                    ),
                    original_ids,
                )
                self.assertEqual(
                    latest.recorded_members[0].evidence.authority, authority
                )
                self.assertEqual(
                    latest.recorded_members[-1].evidence.authority, approved.authority
                )
                self.assertEqual(len(latest.collections), 2)

                run_table_reading(owner)
                before = len(reads)
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(owner.path)),
                    journey=owner,
                )
                self.assertEqual(len(reads), before)

        _assert_investigation_read_snapshot(self)

    def test_observation_factory_failure_stops_once_after_receipt_validation(
        self,
    ) -> None:
        from bio_neb.investigation.investigation_observation_commands import (
            InvestigationObservationApplicationRefused,
        )

        with tempfile.TemporaryDirectory() as temporary:
            _, journey, _, descriptor, command = _approved_collection_observation(
                Path(temporary).resolve(),
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            factory_calls = []

            def unavailable_adapter():
                factory_calls.append("construct")
                raise RuntimeError("synthetic adapter initialization failure")

            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused, "vcf_identity_changed"
            ):
                investigation_observe(
                    command,
                    adapter_factory=unavailable_adapter,
                    evidence_reader=_Evidence(
                        descriptor.model_copy(update={"sha256": "0" * 64})
                    ),
                )
            self.assertEqual(factory_calls, [])
            from bio_neb.failure import Invariant

            # The shared preparation's fault is retained on its first item, then
            # holds the Run for the fix (R2).
            with self.assertRaises(Invariant) as held:
                investigation_observe(
                    command,
                    adapter_factory=unavailable_adapter,
                    evidence_reader=_Evidence(descriptor),
                )
            self.assertEqual(
                held.exception.invariant_id, "investigation_shared_observation_failed"
            )
            failed = read_investigation(Journey.open(journey.path)).evidence
            self.assertEqual(factory_calls, ["construct"])
            self.assertEqual(
                [item.status for item in failed.item_states],
                ["failed", "not_started", "not_started"],
            )
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_evidence,
                failed,
            )

    def test_owned_preparation_survives_later_selected_approval(self) -> None:
        self._assert_owned_preparation(use_factory=True)

    def test_eager_observation_dependencies_share_owned_preparation(self) -> None:
        self._assert_owned_preparation(use_factory=False)

    def _assert_owned_preparation(self, *, use_factory: bool) -> None:
        from bio_neb.investigation.investigation_observation_adapter import (
            DirectInvestigationObservationEvidence,
        )
        from bio_neb.investigation.investigation_observation_commands import (
            InvestigationObservationSession,
            MaterializationObservationEvidence,
        )
        from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter
        from bio_neb.investigation.observation_tools import (
            ObservationToolsReceipt,
            ProbedTool,
        )
        from bio_neb.investigation.vcf_observation import AttestedPlainVcf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            vcf_content = b"synthetic original VCF bytes"
            investigation_input, journey, authority, descriptor, command = (
                _approved_collection_observation(
                    root, cardinality=3, approve_count=2, vcf_content=vcf_content
                )
            )
            contents = {
                command.vcf_file: vcf_content,
                command.materialization_receipt_file: b"synthetic materialization receipt",
                command.bcftools_executable: b"synthetic executable identity",
            }
            # The Run's own typed tool probe names this executable.
            contents[command.bcftools_probe_receipt_file] = canonical_json(
                ObservationToolsReceipt(
                    tools=tuple(
                        ProbedTool(
                            name=name,
                            executable=executable,
                            executable_sha256=hashlib.sha256(identity).hexdigest(),
                            package="pysam",
                            package_version="0.24.1",
                            tool_version="1.24",
                            htslib_version="1.24",
                        )
                        for name, executable, identity in (
                            (
                                "bcftools",
                                command.bcftools_executable,
                                contents[command.bcftools_executable],
                            ),
                            ("samtools", str(root / "samtools"), b"samtools"),
                        )
                    )
                ).model_dump(mode="json")
            ).encode()
            for path, content in contents.items():
                Path(path).write_bytes(content)
                Path(path).chmod(
                    0o500 if path == command.bcftools_executable else 0o400
                )

            class ProcessResponses(LocalBcftoolsAdapter):
                def __init__(self):
                    super().__init__()
                    self.calls = []

                def process_response(self, argv, **kwargs):
                    argv = tuple(argv)
                    self.calls.append(argv)
                    step = lookup_command(argv)
                    output = b""
                    if step == "bgzf_derivation":
                        bgzf = str(Path(command.derivation_root) / "variants.vcf.gz")
                        contents[bgzf] = b"synthetic BGZF bytes"
                        os.write(kwargs["stdout"], contents[bgzf])
                    elif step == "csi_derivation":
                        contents[argv[-1] + ".csi"] = b"synthetic CSI bytes"
                        Path(argv[-1] + ".csi").write_bytes(contents[argv[-1] + ".csi"])
                    elif step == "header_inventory":
                        output = vcf_header("HG001")
                    else:
                        output = query_row(
                            "NC_000019.10",
                            query_position({argv[4]: Path(argv[4]).read_bytes()}, argv),
                            "T",
                            "C",
                            "HG001",
                            "0/1",
                        )
                    return subprocess.CompletedProcess(argv, 0, output, b"")

            adapter = ProcessResponses()

            class MaterializationEvidence(DirectInvestigationObservationEvidence):
                def materialization(self, *, receipt_file, vcf_file):
                    _, observation = self.files.read(receipt_file, retain_bytes=True)
                    return MaterializationObservationEvidence(
                        receipt_id="vcf-materialization-receipt:" + "7" * 64,
                        descriptor=descriptor,
                        attested_vcf=AttestedPlainVcf(
                            local_path=vcf_file,
                            byte_count=descriptor.byte_count,
                            sha256=descriptor.sha256,
                            acquisition_receipt_path=receipt_file,
                            acquisition_receipt_sha256=observation.sha256,
                        ),
                        receipt_file=observation,
                    )

            session = InvestigationObservationSession(
                adapter=adapter if not use_factory else None,
                adapter_factory=(lambda: adapter) if use_factory else None,
                evidence_reader=MaterializationEvidence(),
            )
            self.addCleanup(session.close)
            process_patch = patch(
                "bio_neb.investigation.local_bcftools.subprocess.run",
                side_effect=adapter.process_response,
            )
            process_patch.start()
            self.addCleanup(process_patch.stop)
            read_bytes = {}
            fdopen = os.fdopen

            def measured_open(*args, **kwargs):
                stream = fdopen(*args, **kwargs)
                path = os.readlink(f"/proc/self/fd/{stream.fileno()}")
                read = stream.read

                def measured_read(size=-1):
                    content = read(size)
                    read_bytes[path] = read_bytes.get(path, 0) + len(content)
                    return content

                stream.read = measured_read
                return stream

            measurement = patch("os.fdopen", side_effect=measured_open)
            measurement.start()
            self.addCleanup(measurement.stop)
            first = investigation_observe(command, session=session)
            self.assertEqual(
                [item.status for item in first.item_states],
                ["completed", "completed", "approval_pending"],
            )
            pending = project_investigation(read_evidence(Journey.open(journey.path)))
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=authority.proposal_id,
                        work_item_ids=pending.pending_work_item_ids,
                    ),
                )
            )
            later = command.model_copy(
                update={"authority_id": approved.authority.authority_id}
            )
            combined = investigation_observe(later, session=session)
            self.assertEqual(
                [item.status for item in combined.item_states], ["completed"] * 3
            )
            self.assertEqual(combined.item_states[:2], first.item_states[:2])
            steps = [lookup_command(call) for call in adapter.calls]
            self.assertEqual(steps.count("bgzf_derivation"), 1)
            self.assertEqual(steps.count("variant_query"), 3)
            facts = [
                event["payload"]
                for event in Journey.open(journey.path).read()
                if event["event_type"] == "InvestigationEvidenceRecorded"
            ]
            first_preparation = facts[0]["sample_observation"]["preparation"]
            later_preparation = facts[-1]["sample_observation"]["preparation"]
            self.assertNotEqual(
                first_preparation["continuation_id"],
                later_preparation["continuation_id"],
            )
            self.assertEqual(
                first_preparation["derived_bgzf"], later_preparation["derived_bgzf"]
            )
            self.assertEqual(investigation_observe(command, session=session), first)
            # Trusted reuse (0029): the attested VCF and the executable are never
            # read; each receipt and each derived file is read once.
            self.assertEqual(
                read_bytes,
                {
                    path: len(content)
                    for path, content in contents.items()
                    if path not in {command.vcf_file, command.bcftools_executable}
                },
            )
            measurement.stop()

    def test_structural_observation_failure_takes_precedence_over_unresolved_request_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "target": {
                            "rsid": "rs429358",
                            "position": {
                                "contig": "NC_000019.10",
                                "position": 1,
                                "ref": "T",
                                "alt": "C",
                            },
                        },
                        "request": "Observe rs429358 at NC_000019.10:1 T>C.",
                    },
                    {
                        "target": {"rsid": "rs7412"},
                        "request": "Observe rs7412 in HG001.",
                    },
                ],
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            asked = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                model_factory=lambda loaded, selected: _SequentialAdaptiveModel().model(
                    loaded, selected
                ),
                source_adapter_factory=lambda: _SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            # The unresolved request is an issue the researcher approves.
            pending = _approve_issue(journey, asked.pending_request)
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve", proposal_id=pending.proposals[-1].proposal_id
                    ),
                )
            )
            inp = approved.investigation_input
            descriptor = VerifiedObject(
                role="VCF",
                bucket=inp.vcf.bucket,
                object_key=inp.vcf.object_key,
                byte_count=inp.vcf.byte_count,
                sha256=inp.vcf.sha256,
            )
            command = InvestigationObservationCommand(
                journey_file=str(journey.path),
                authority_id=approved.authority.authority_id,
                vcf_file=str(root / "variants.vcf"),
                materialization_receipt_file=str(root / "materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )

            class FailedAdapter(_SequentialObservationAdapter):
                def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001,ANN201
                    if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file"):
                        raise RuntimeError("unclassified implementation failure")
                    return super().run(
                        argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                    )

            from bio_neb.failure import Invariant

            # The item's unclassified fault is retained on its item, then holds
            # the Run for the fix (R2); the Run is not ended.
            with self.assertRaises(Invariant) as held:
                investigation_observe(
                    command,
                    adapter=FailedAdapter(command, descriptor, inp.sample_id),
                    evidence_reader=_Evidence(descriptor),
                )
            self.assertEqual(held.exception.invariant_id, "investigation_item_failed")
            visible = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(visible.status, "investigation_evidence_ready")
            collection = read_investigation(Journey.open(journey.path)).evidence
            self.assertEqual(
                held.exception.evidence["work_item_ids"],
                [
                    i.work_item_id
                    for i in collection.item_states
                    if i.status == "failed"
                ],
            )
            self.assertEqual(visible.investigation.proposals, pending.proposals)
            before = journey.path.read_bytes()
            member = pending.proposals[-1].request_members[0]
            self.assertIsNotNone(member.issue_approval)
            # The approval closed the first Review; its fresh final Review still
            # declares the scoped revision opportunity the failure refuses.
            remaining = next(
                r
                for r in reversed(visible.investigation.review_requests)
                if r.proposal is not None
            )
            self.assertNotEqual(
                remaining.request_event_id, pending.pending_request.request_event_id
            )
            with self.assertRaises(InvestigationApplicationRefused) as refused:
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=remaining.request_event_id,
                        response=ReviseInvestigationResponse.model_validate(
                            {
                                "action": "revise",
                                "proposal_id": pending.proposals[-1].proposal_id,
                                "asserted_by": "researcher:owner",
                                "operations": [
                                    {
                                        "operation": "amend_request",
                                        "request_member_id": member.request_member_id,
                                        "request": {
                                            "rsid": "rs429358",
                                            "request_text": "Observe rs429358 in HG001.",
                                        },
                                    },
                                ],
                            }
                        ),
                    )
                )
            self.assertEqual(refused.exception.code, "investigation_rework_ineligible")
            self.assertEqual(journey.path.read_bytes(), before)

    def test_multiallelic_shared_rsid_preserves_all_assertions_and_per_alt_candidates(
        self,
    ) -> None:
        # A request naming one exact allele keeps that allele's candidate; one
        # naming none gets a candidate Work item per source ALT.
        body = json.loads(REFSNP_BODY)
        alleles = body["primary_snapshot_data"]["placements_with_allele"][0]["alleles"]
        other = copy.deepcopy(alleles[-1])
        other["allele"]["spdi"]["inserted_sequence"] = "G"
        alleles.append(other)
        for unnamed in (False, True):
            with (
                self.subTest(unnamed=unnamed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                requests = tuple(
                    InvestigationRequestItem(
                        rsid="rs429358",
                        request_text=f"Observe rs429358 as {alt or 'any allele'}.",
                        position=None
                        if alt is None
                        else StatedPosition(
                            contig="NC_000019.10", position=44908684, ref="T", alt=alt
                        ),
                    )
                    for alt in ("C", "G", "C", *((None,) if unnamed else ()))
                )
                assertion = UserAssertion.create_collection(
                    asserted_by="researcher:owner",
                    requests=requests,
                    predecessor_ids=("test-input",),
                )
                source = SourceResponsesOnly(
                    refsnp_body=json.dumps(body).encode(), assembly_body=ASSEMBLY_BODY
                )
                projected, journey = _run_attempt(
                    Path(temporary).resolve(),
                    _SequentialAdaptiveModel(),
                    source,
                    assertion=assertion,
                )
                proposal = projected.pending_request.proposal
                self.assertEqual(
                    [item.target.alt for item in proposal.work_items], ["C", "G"]
                )
                self.assertEqual(
                    tuple(member.request for member in proposal.request_members),
                    requests,
                )
                self.assertEqual(len(projected.review_requests), 1)
                self.assertEqual(proposal.unresolved_request_member_ids, ())
                candidate_c, candidate_g = (
                    (item.work_item_id,) for item in proposal.work_items
                )
                self.assertEqual(
                    [member.work_item_ids for member in proposal.request_members[:3]],
                    [candidate_c, candidate_g, candidate_c],
                )
                self.assertNotEqual(
                    proposal.request_members[0].request_member_id,
                    proposal.request_members[2].request_member_id,
                )
                if unnamed:
                    self.assertEqual(
                        proposal.request_members[-1].refsnp.status, "resolved"
                    )
                    self.assertEqual(
                        proposal.request_members[-1].work_item_ids,
                        candidate_c + candidate_g,
                    )
                self.assertEqual(
                    project_investigation(read_evidence(Journey.open(journey.path))),
                    projected,
                )
                hostile = copy.deepcopy(read_evidence(Journey.open(journey.path)))
                links = [
                    event
                    for event in hostile
                    if event["event_type"] == "StructuredOutputValidationLinked"
                ]
                links[-1]["payload"]["collection_member_id"] = links[0]["payload"][
                    "collection_member_id"
                ]
                with self.assertRaises(ValueError):
                    project_investigation(hostile)

    def test_wholly_unresolved_requests_have_review_without_approval_or_invented_work(
        self,
    ) -> None:
        # Each unresolved request asks its own issue Review; approving both
        # settles the collection as accounting, with no Work to approve.
        with tempfile.TemporaryDirectory() as temporary:
            assertion = UserAssertion.create_collection(
                asserted_by="researcher:owner",
                predecessor_ids=("test-input",),
                requests=tuple(
                    InvestigationRequestItem(
                        rsid=rsid,
                        request_text=f"Observe {rsid} at NC_000019.10:1 T>C.",
                        position=StatedPosition(
                            contig="NC_000019.10", position=1, ref="T", alt="C"
                        ),
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            model = _SequentialAdaptiveModel()
            sources = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            # Both members ask in the first batch, each its own Attempt.
            second, journey = _run_attempt(
                Path(temporary).resolve(), model, sources, assertion=assertion
            )
            issues = [r for r in second.review_requests if r.pending]
            self.assertEqual(len(issues), 2)
            self.assertTrue(all(r.question.issue.conflicts for r in issues))
            self.assertEqual(second.proposals, ())
            for review in issues:
                settled = respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=review.request_event_id,
                        response=ApproveInvestigationFindingResponse(
                            action="approve",
                            collection_member_id=review.question.collection_member_id,
                            revision_number=1,
                            asserted_by="researcher:owner",
                        ),
                    )
                )
            (proposal,) = settled.proposals
            self.assertEqual(proposal.work_items, ())
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            self.assertTrue(
                all(member.issue_approval for member in proposal.request_members)
            )
            # No proposal Review opens, so nothing can be approved or invented.
            self.assertIsNone(settled.pending_request)
            self.assertEqual(settled.authorities, ())
            self.assertEqual(second.status, "review_pending")

    def test_selected_request_clarification_preserves_observed_sibling_and_truth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "target": {
                            "rsid": "rs429358",
                            "position": {
                                "contig": "NC_000019.10",
                                "position": 1,
                                "ref": "T",
                                "alt": "C",
                            },
                        },
                        "request": "Observe rs429358 at NC_000019.10:1 T>C.",
                    },
                    {
                        "target": {"rsid": "rs7412"},
                        "request": "Observe rs7412 in HG001.",
                    },
                ],
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            asked = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                model_factory=lambda loaded, selected: _SequentialAdaptiveModel().model(
                    loaded, selected
                ),
                source_adapter_factory=lambda: _SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            # The unresolved request is an issue first approved as checked; a
            # later proposal revision may still check it again.
            pending = _approve_issue(journey, asked.pending_request)
            proposal = pending.pending_request.proposal
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve", proposal_id=proposal.proposal_id
                    ),
                )
            )
            original_grant = approved.authority
            inp = approved.investigation_input
            descriptor = VerifiedObject(
                role="VCF",
                bucket=inp.vcf.bucket,
                object_key=inp.vcf.object_key,
                byte_count=inp.vcf.byte_count,
                sha256=inp.vcf.sha256,
            )
            command = InvestigationObservationCommand(
                journey_file=str(journey.path),
                authority_id=original_grant.authority_id,
                vcf_file=str(root / "variants.vcf"),
                materialization_receipt_file=str(root / "materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )
            observed = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            original_state = observed.item_states[0]
            visible = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(
                visible.investigation.proposals[-1].request_members,
                proposal.request_members,
            )
            for coordinate in (1, 44908684):
                current = project_investigation(
                    read_evidence(Journey.open(journey.path))
                )
                member = next(
                    item
                    for item in current.proposals[-1].request_members
                    if not item.work_item_ids
                )
                # The approved issue blocks nothing, yet a revision may recheck it.
                self.assertIsNotNone(member.issue_approval)
                self.assertNotIn(
                    "unresolved_request_identity",
                    read_investigation(
                        Journey.open(journey.path)
                    ).work_policy.completion_blockers,
                )
                revised = respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=current.review_requests[-1].request_event_id,
                        response=ReviseInvestigationResponse.model_validate(
                            {
                                "action": "revise",
                                "proposal_id": current.proposals[-1].proposal_id,
                                "asserted_by": "researcher:owner",
                                "operations": [
                                    {
                                        "operation": "clarify_request",
                                        "request_member_id": member.request_member_id,
                                        "request": {
                                            "rsid": "rs429358",
                                            "request_text": f"Observe rs429358 at NC_000019.10:{coordinate} T>C.",
                                            "position": {
                                                "contig": "NC_000019.10",
                                                "position": coordinate,
                                                "ref": "T",
                                                "alt": "C",
                                            },
                                        },
                                    }
                                ],
                            }
                        ),
                    )
                )
                self.assertEqual(revised.authorities, (original_grant,))
                source = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
                clarified = continue_investigation(
                    ContinueInvestigationCommand(journey_file=str(journey.path)),
                    environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                    model_factory=lambda loaded,
                    selected: _SequentialAdaptiveModel().model(loaded, selected),
                    source_adapter_factory=lambda: source,  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                )
                if coordinate == 1:
                    # Still unresolved: the recheck is a new issue Review, which
                    # the researcher approves before the revision is proposed.
                    reissued = clarified.pending_request
                    self.assertNotEqual(reissued.question.issue, None)
                    self.assertEqual(reissued.question.revision_number, 2)
                    self.assertEqual(len(clarified.proposals), len(current.proposals))
                    clarified = _approve_issue(journey, reissued)
                self.assertEqual(
                    clarified.proposals[-1].work_items[0], proposal.work_items[0]
                )
                # The clarified member's stated position is placed first.
                self.assertEqual(
                    source.operations[0], "fetch_assembly_report:GCF_000001405.40"
                )
                self.assertNotIn("fetch_refsnp:rs7412", source.operations)
                self.assertEqual(clarified.authorities, (original_grant,))
                self.assertFalse(clarified.assertions[-1].scientifically_verified)
                self.assertEqual(
                    clarified.proposals[-1].unresolved_request_member_ids, ()
                )
                self.assertEqual(
                    any(
                        item.issue_approval is not None
                        for item in clarified.proposals[-1].request_members
                    ),
                    coordinate == 1,
                )
            current = clarified
            fresh = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=current.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=current.proposals[-1].proposal_id,
                        work_item_ids=current.pending_work_item_ids,
                    ),
                )
            )
            new_command = command.model_copy(
                update={
                    "authority_id": fresh.authority.authority_id,
                    "derivation_root": str(root / "clarified"),
                }
            )
            completed = investigation_observe(
                new_command,
                adapter=_SequentialObservationAdapter(
                    new_command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(completed.item_states[0], original_state)
            self.assertEqual(len(completed.item_states), 2)
            self.assertEqual(
                project_investigation(
                    read_evidence(Journey.open(journey.path))
                ).proposals[0],
                proposal,
            )

    def test_unresolved_request_reaches_review_without_stopping_ready_sibling(
        self,
    ) -> None:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        class CorrectedIdentityModel(_SequentialAdaptiveModel):
            async def respond(self, messages, info):  # noqa: ANN001, ANN201
                response = await super().respond(messages, info)
                arguments = response.parts[0].args_as_dict()
                refusal_visible = any(
                    "fetch_assembly_report_requires_resolved_refsnp_result" in str(text)
                    for message in messages
                    for text in corrections(message)
                )
                if arguments.get("outcome") == "unresolved" and not refusal_visible:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "fetch_assembly_report",
                                {
                                    "assembly_accession": "GCF_000001405.40",
                                    "refsnp_event_id": arguments["refsnp_event_id"],
                                },
                                tool_call_id=f"conflicted-assembly-{self.calls}",
                            )
                        ]
                    )
                return response

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher:owner",
                requests=(
                    InvestigationRequestItem(
                        rsid="rs429358",
                        request_text="Observe rs429358 at NC_000019.10:1 T>C.",
                        position=StatedPosition(
                            contig="NC_000019.10", position=1, ref="T", alt="C"
                        ),
                    ),
                    InvestigationRequestItem(
                        rsid="rs7412", request_text="Observe rs7412 in HG001."
                    ),
                ),
                predecessor_ids=("test-input",),
            )
            source = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            model = CorrectedIdentityModel()
            asked, journey = _run_attempt(root, model, source, assertion=assertion)
            # The unresolved request asks its own issue Review and its ready
            # sibling, its own Attempt in the same batch, is checked while the
            # issue waits.
            self.assertEqual(asked.status, "review_pending")
            issue = asked.pending_request
            self.assertEqual(issue.question.issue.verdict, "conflicted")
            checked = asked
            self.assertEqual(checked.proposals, ())
            self.assertEqual(len(checked.attempts), 2)
            projection = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=issue.request_event_id,
                    response=ApproveInvestigationFindingResponse(
                        action="approve",
                        collection_member_id=issue.question.collection_member_id,
                        revision_number=1,
                        asserted_by="researcher:owner",
                    ),
                )
            )
            self.assertEqual(projection.status, "review_pending")
            proposal = projection.pending_request.proposal
            self.assertEqual(len(proposal.work_items), 1)
            self.assertEqual(proposal.work_items[0].target.aliases, ("rs7412",))
            self.assertEqual(len(proposal.request_members), 2)
            self.assertEqual(proposal.request_members[0].work_item_ids, ())
            self.assertTrue(proposal.request_members[0].conflicts)
            self.assertEqual(proposal.request_members[0].refsnp.status, "conflicted")
            self.assertEqual(
                proposal.request_members[0].conflicts,
                proposal.request_members[0].refsnp.conflicts,
            )
            refused = [
                event["payload"]
                for event in journey.read()
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["status"] == "invalid"
            ]
            self.assertEqual(len(refused), 1)
            self.assertEqual(
                refused[0]["errors"],
                ["fetch_assembly_report_requires_resolved_refsnp_result"],
            )
            # The stated position is placed first, its sequence named by the model
            # after a sequence_unnamed result, and its REF is not the reference's;
            # the sibling's rsID lookup is unchanged.
            self.assertEqual(
                source.operations,
                [
                    "fetch_assembly_report:GCF_000001405.40",
                    "ncbi_sequence_names:GCF_000001405.40",
                    "fetch_assembly_report:GCF_000001405.40",
                    "ncbi_sequence_names:GCF_000001405.40",
                    "ncbi_reference_span:NC_000019.10:0:1",
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            from bio_neb.investigation.investigation_policy import (
                InvestigationPolicyAmendment,
            )
            from bio_neb.investigation.investigation_policy_commands import (
                AmendInvestigationPolicyCommand,
                amend_investigation_policy,
            )

            policy = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=proposal.proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=None,
                        asserted_by="researcher:owner",
                        reason="Keep identity unresolved while ready work proceeds.",
                        completion="require_terminal_members",
                    ),
                )
            )
            # An approved issue is checked, not verified, and blocks no work.
            self.assertNotIn("unresolved_request_identity", policy.completion_blockers)
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=projection.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve", proposal_id=proposal.proposal_id
                    ),
                )
            )
            self.assertEqual(
                approved.authority.authorized_work_items, proposal.work_items
            )
            # Every Work item is approved and the issue is settled: no Review holds.
            self.assertIsNone(approved.pending_request)
            self.assertEqual(
                approved.proposals[-1].request_members, proposal.request_members
            )

    def test_only_a_pass_closed_at_its_review_admits_a_policy_amendment(
        self,
    ) -> None:
        from bio_neb.agentic_journey.continuation_policy import (
            InvestigationWorkPolicy,
        )
        from bio_neb.investigation.investigation_policy import (
            InvestigationPolicyAmended,
            InvestigationPolicyAmendment,
            project_investigation_policy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            projection, journey = _run_attempt(
                Path(temporary).resolve(),
                _SequentialAdaptiveModel(),
                _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY),
            )
            self.assertEqual(projection.status, "review_pending")
            events = read_evidence(journey)
            self.assertEqual(
                (events[-1]["event_type"], events[-1]["payload"]["disposition"]),
                ("PassClosed", "waiting"),
            )
            fact = InvestigationPolicyAmended.create(
                amendment=InvestigationPolicyAmendment(
                    proposal_id=projection.pending_request.proposal.proposal_id,
                    expected_head_event_id=events[-1]["event_id"],
                    previous_policy_id=None,
                    asserted_by="researcher:owner",
                    reason="Require every member before Finalization.",
                    completion="require_terminal_members",
                ),
                defaults=InvestigationWorkPolicy(),
                held_work_item_ids=(),
            )
            amendment = {
                "event_type": fact.event_type,
                "event_id": "amendment",
                "payload": fact.model_dump(mode="json", exclude={"event_type"}),
                "parent_ids": [events[-1]["event_id"]],
            }
            policy = project_investigation_policy(
                [*events, amendment], projection, None
            )
            self.assertEqual(policy.policy_id, fact.policy_id)
            for disposition, failure in (
                (
                    "held",
                    {
                        "failure_class": "invariant",
                        "invariant_id": "coordinator_unhandled_error",
                        "cause": "RuntimeError",
                    },
                ),
                ("interrupted", None),
            ):
                closed = copy.deepcopy(events)
                closed[-1]["payload"]["disposition"] = disposition
                if failure is not None:
                    closed[-1]["payload"]["failure"] = failure
                self.assertFalse(
                    project_investigation_policy(
                        closed, projection, None
                    ).amendment_eligible
                )
                with self.assertRaisesRegex(
                    ValueError, "Investigation amendment changed its exact frontier"
                ):
                    project_investigation_policy([*closed, amendment], projection, None)

    def test_shared_rsid_assertions_resolve_and_deduplicate_by_exact_target(
        self,
    ) -> None:
        requests = tuple(
            InvestigationRequestItem(
                rsid="rs429358",
                request_text=f"Observe rs429358 at NC_000019.10:{position} T>C; {label}.",
                position=StatedPosition(
                    contig="NC_000019.10", position=position, ref="T", alt="C"
                ),
            )
            for position, label in (
                (44908684, "first"),
                (44908685, "second"),
                (44908684, "again"),
            )
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=requests,
            predecessor_ids=("test-input",),
        )
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="run-bucket",
                object_key="sample.vcf",
                byte_count=1,
                sha256="a" * 64,
            ),
            sample_id="HG001",
            assembly_accession="GCF_000001405.40",
        )
        pairs = tuple(
            _source_pair("rs429358", position)
            for position in (44908684, 44908685, 44908684)
        )
        proposal = compile_investigation_collection_proposal(
            investigation_input=investigation_input,
            assertion=assertion,
            source_results=pairs,
            revision_number=1,
            supersedes_proposal_id=None,
            predecessor_ids=tuple(
                dict.fromkeys(result.result_id for pair in pairs for result in pair)
            ),
        )
        self.assertEqual(
            [item.target.one_based_position for item in proposal.work_items],
            [44908684, 44908685],
        )
        self.assertEqual(
            tuple(member.request for member in proposal.request_members), requests
        )
        self.assertEqual(
            proposal.request_members[0].work_item_ids,
            proposal.request_members[2].work_item_ids,
        )
        self.assertEqual(proposal.unresolved_request_member_ids, ())

    def test_new_grants_refresh_pending_siblings_between_observation_batches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, first, descriptor, command = (
                _approved_collection_observation(
                    root,
                    cardinality=4,
                    forked=True,
                    approve_count=1,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(4)
                    ).journey_file,
                )
            )
            initial = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            partial = project_investigation(read_evidence(Journey.open(journey.path)))
            assert partial.pending_request is not None
            grants = []
            for work_item_id in partial.pending_work_item_ids:
                # Each approval closes its Review; the remaining members wait on
                # the new Review it opens.
                pending = project_investigation(
                    read_evidence(Journey.open(journey.path))
                ).pending_request
                assert pending is not None
                approved = respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=pending.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=first.proposal_id,
                            work_item_ids=(work_item_id,),
                        ),
                    )
                )
                grants.append(approved.authority)
            for index, grant in enumerate(reversed(grants)):
                assert grant is not None
                selected_command = command.model_copy(
                    update={
                        "authority_id": grant.authority_id,
                        "derivation_root": str(root / f"pending-{index}"),
                    }
                )
                recorded = investigation_observe(
                    selected_command,
                    adapter=_SequentialObservationAdapter(
                        selected_command, descriptor, investigation_input.sample_id
                    ),
                    evidence_reader=_Evidence(descriptor),
                )
                self.assertEqual(recorded.item_states[0], initial.item_states[0])
                self.assertEqual(
                    observe(
                        ObserveRun(journey_file=str(journey.path))
                    ).investigation_evidence,
                    recorded,
                )
                if index == 0:
                    self.assertEqual(
                        [item.status for item in recorded.item_states],
                        [
                            "completed",
                            "observation_pending",
                            "observation_pending",
                            "completed",
                        ],
                    )
                    self.assertEqual(recorded.combine_status, "partial")
            self.assertEqual(
                [item.status for item in recorded.item_states], ["completed"] * 4
            )

    def test_disjoint_grants_can_be_observed_in_either_order(self) -> None:
        for newest_first in (False, True):
            with (
                self.subTest(newest_first=newest_first),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                investigation_input, journey, first, descriptor, command = (
                    _approved_collection_observation(
                        root,
                        cardinality=4,
                        forked=True,
                        approve_count=2,
                        parent_file=self.prerequisite(
                            "executed", investigation_request=collection_request(4)
                        ).journey_file,
                    )
                )
                partial = project_investigation(
                    read_evidence(Journey.open(journey.path))
                )
                assert partial.pending_request is not None
                approved = respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=partial.pending_request.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=first.proposal_id,
                            work_item_ids=partial.pending_work_item_ids,
                        ),
                    )
                )
                assert approved.authority is not None
                grants = (
                    (approved.authority, first)
                    if newest_first
                    else (first, approved.authority)
                )
                previous = None
                for index, grant in enumerate(grants):
                    selected_command = command.model_copy(
                        update={
                            "authority_id": grant.authority_id,
                            "derivation_root": str(root / f"derived-{index}"),
                        }
                    )
                    recorded = investigation_observe(
                        selected_command,
                        adapter=_SequentialObservationAdapter(
                            selected_command, descriptor, investigation_input.sample_id
                        ),
                        evidence_reader=_Evidence(descriptor),
                    )
                    self.assertEqual(
                        recorded.previous_collection_state_id,
                        previous.collection_state_id if previous else None,
                    )
                    previous = recorded
                self.assertEqual(
                    [item.status for item in recorded.item_states], ["completed"] * 4
                )
                self.assertEqual(
                    observe(
                        ObserveRun(journey_file=str(journey.path))
                    ).investigation_evidence,
                    recorded,
                )
                # The collection was observed by direct step commands, outside a
                # coordinator Pass, so it has no closed point to Fork; Finalization
                # compiles on the Run itself (a Fork from a closed before-Finalization
                # point is covered in test_shared_review_responses).
                projection = observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(
                    projection.investigation.authorities, approved.authorities
                )
                run_table_reading(journey.path)
                finalization = compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
                assert finalization.pending_request is not None
                self.assertEqual(
                    len(finalization.pending_request.basis.sample_observations), 4
                )

    def test_policy_holds_pending_member_while_authorized_siblings_continue(
        self,
    ) -> None:
        from bio_neb.investigation.investigation_policy import (
            InvestigationPolicyAmendment,
        )
        from bio_neb.investigation.investigation_policy_commands import (
            AmendInvestigationPolicyCommand,
            amend_investigation_policy,
            inspect_investigation_policy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, grant, descriptor, command = _approved_collection_observation(
                root,
                cardinality=3,
                forked=True,
                approve_count=1,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            projection = project_investigation(
                read_evidence(Journey.open(journey.path))
            )
            held_id = projection.pending_work_item_ids[0]
            amendment = InvestigationPolicyAmendment(
                proposal_id=projection.proposals[-1].proposal_id,
                expected_head_event_id=Journey.open(journey.path).read()[-1][
                    "event_id"
                ],
                previous_policy_id=None,
                asserted_by="researcher:owner",
                reason="Hold this member while approved siblings proceed.",
                hold_work_item_ids=(held_id,),
                completion="require_terminal_members",
            )
            decision = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path), amendment=amendment
                )
            )
            self.assertEqual(decision.held_work_item_ids, (held_id,))
            self.assertEqual(
                decision.eligible_work_item_ids,
                (grant.authorized_work_items[0].work_item_id,),
            )
            with self.assertRaises(InvestigationApplicationRefused) as refused:
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=projection.pending_request.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=projection.proposals[-1].proposal_id,
                            work_item_ids=(held_id,),
                        ),
                    )
                )
            self.assertEqual(refused.exception.code, "investigation_work_item_held")
            observed = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(
                [item.status for item in observed.item_states],
                ["completed", "approval_pending", "approval_pending"],
            )
            decision = inspect_investigation_policy(str(journey.path))
            self.assertFalse(decision.finalization_eligible)
            self.assertIn("pending_members", decision.completion_blockers)
            with self.assertRaises(FinalizationApplicationRefused) as refused:
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
            self.assertEqual(refused.exception.code, "finalization_policy_ineligible")
            released = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=projection.proposals[-1].proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=decision.policy_id,
                        asserted_by="researcher:owner",
                        reason="Allow a partial reviewed basis.",
                        release_work_item_ids=(held_id,),
                        completion="allow_partial_with_limitations",
                    ),
                )
            )
            self.assertTrue(released.finalization_eligible)
            from bio_neb.application.queries import read_completion

            # The released amendment opens the VCF table stage; Finalization
            # reads its fold, recorded after the amendment (ADR 0047).
            self.assertIsNone(read_completion(Journey.open(journey.path)).candidate)
            run_table_reading(journey.path)
            completion = read_completion(Journey.open(journey.path))
            candidate = completion.candidate
            self.assertIsNotNone(candidate)
            self.assertEqual(
                candidate.source_head_event["event_type"],
                "InvestigationItemEvidenceCollected",
            )
            self.assertEqual(
                candidate.collection.event["event_type"],
                "InvestigationCollectionEvidenceRecorded",
            )
            self.assertNotEqual(
                candidate.source_head_event["event_id"],
                candidate.collection.event["event_id"],
            )
            self.assertEqual(
                candidate.collection.evidence.work_policy.completion,
                "require_terminal_members",
            )
            self.assertEqual(
                completion.investigation.work_policy.effective.completion,
                "allow_partial_with_limitations",
            )
            finalization = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            self.assertEqual(
                len(finalization.pending_request.basis.sample_observations), 1
            )
            self.assertEqual(
                finalization.pending_request.basis.collection_item_states,
                observed.item_states,
            )
            self.assertEqual(
                inspect_investigation_policy(str(journey.path)).policy_id,
                released.policy_id,
            )

    def test_policy_survives_selected_revision_and_release_preserves_evidence(
        self,
    ) -> None:
        from bio_neb.agentic_journey.continuation_policy import InvestigationWorkPolicy
        from bio_neb.investigation.investigation_policy import (
            InvestigationPolicyAmendment,
        )
        from bio_neb.investigation.investigation_policy_commands import (
            AmendInvestigationPolicyCommand,
            InvestigationPolicyApplicationRefused,
            amend_investigation_policy,
            inspect_investigation_policy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, grant, descriptor, command = _approved_collection_observation(
                root,
                cardinality=3,
                forked=True,
                approve_count=1,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            projection = project_investigation(
                read_evidence(Journey.open(journey.path))
            )
            held_id, removed_id = projection.pending_work_item_ids
            first = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=projection.proposals[-1].proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=None,
                        asserted_by="researcher:owner",
                        reason="Require terminal work and hold one pending member.",
                        hold_work_item_ids=(held_id,),
                    ),
                ),
                defaults=InvestigationWorkPolicy(completion="require_terminal_members"),
            )
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_policy,
                first,
            )
            observed = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            revised = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=projection.pending_request.request_event_id,
                    response=ReviseInvestigationResponse.model_validate(
                        {
                            "action": "revise",
                            "proposal_id": projection.proposals[-1].proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": [
                                {
                                    "operation": "remove"
                                    if item.work_item_id == removed_id
                                    else "keep",
                                    "work_item_id": item.work_item_id,
                                }
                                for item in projection.proposals[-1].work_items
                            ],
                        }
                    ),
                )
            )
            decision = inspect_investigation_policy(str(journey.path))
            self.assertEqual(decision.held_work_item_ids, (held_id,))
            self.assertEqual(decision.effective.completion, "require_terminal_members")
            with self.assertRaises(InvestigationApplicationRefused) as held:
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=revised.pending_request.request_event_id,
                        response=ApproveInvestigationResponse(
                            action="approve",
                            proposal_id=revised.proposals[-1].proposal_id,
                            work_item_ids=(held_id,),
                        ),
                    )
                )
            self.assertEqual(held.exception.code, "investigation_work_item_held")
            released = amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=revised.proposals[-1].proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=decision.policy_id,
                        asserted_by="researcher:owner",
                        reason="Release exactly the pending member.",
                        release_work_item_ids=(held_id,),
                    ),
                )
            )
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=revised.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=revised.proposals[-1].proposal_id,
                        work_item_ids=(held_id,),
                    ),
                )
            )
            second_command = command.model_copy(
                update={
                    "authority_id": approved.authority.authority_id,
                    "derivation_root": str(root / "later"),
                }
            )
            combined = investigation_observe(
                second_command,
                adapter=_SequentialObservationAdapter(
                    second_command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(combined.item_states[0], observed.item_states[0])
            self.assertEqual(len(combined.item_states), 2)
            self.assertTrue(
                inspect_investigation_policy(str(journey.path)).finalization_eligible
            )
            before = Journey.open(journey.path).read()
            before_evidence = read_evidence(Journey.open(journey.path))
            for field, value in (
                ("hold_work_item_ids", (grant.authorized_work_items[0].work_item_id,)),
                ("previous_policy_id", first.policy_id),
            ):
                invalid = dict(
                    proposal_id=revised.proposals[-1].proposal_id,
                    expected_head_event_id=before[-1]["event_id"],
                    previous_policy_id=released.policy_id,
                    asserted_by="researcher:owner",
                    reason="Invalid scope",
                    completion="allow_partial_with_limitations",
                )
                invalid[field] = value
                with self.assertRaises(
                    InvestigationPolicyApplicationRefused
                ) as refused:
                    amend_investigation_policy(
                        AmendInvestigationPolicyCommand(
                            journey_file=str(journey.path),
                            amendment=InvestigationPolicyAmendment(**invalid),
                        )
                    )
                self.assertEqual(
                    refused.exception.code, "investigation_policy_amendment_ineligible"
                )
                self.assertEqual(
                    refused.exception.disposition.kind, "caller_correction_and_reissue"
                )
                self.assertEqual(Journey.open(journey.path).read(), before)
            from bio_neb.investigation.investigation_policy import (
                project_investigation_policy,
            )

            self.assertEqual(
                project_investigation_policy(
                    before_evidence, project_investigation(before_evidence), combined
                ),
                inspect_investigation_policy(str(journey.path)),
            )
            for mutation, violation in (
                ("omitted_default", "retained Investigation policy must be complete"),
                ("collection_policy", "collection changed its retained work policy"),
                (
                    "amendment_head",
                    "Investigation amendment changed its exact frontier",
                ),
            ):
                changed = copy.deepcopy(before_evidence)
                if mutation == "collection_policy":
                    collection = next(
                        event
                        for event in changed
                        if event["event_type"]
                        == "InvestigationCollectionEvidenceRecorded"
                    )
                    collection["payload"]["work_policy"]["completion"] = (
                        "allow_partial_with_limitations"
                    )
                else:
                    policy_event = next(
                        event
                        for event in changed
                        if event["event_type"] == "InvestigationPolicyAmended"
                    )
                    if mutation == "omitted_default":
                        del policy_event["payload"]["defaults"]["unresolved_identity"]
                    else:
                        policy_event["parent_ids"] = []
                with self.assertRaisesRegex(ValueError, violation):
                    project_investigation_policy(
                        changed, project_investigation(changed), combined
                    )

    def test_recorded_collection_policy_is_complete_without_current_defaults(
        self,
    ) -> None:
        from bio_neb.investigation.investigation_policy import (
            project_investigation_policy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, _grant, descriptor, command = (
                _approved_collection_observation(
                    root,
                    cardinality=2,
                    forked=True,
                    approve_count=1,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(2)
                    ).journey_file,
                )
            )
            evidence = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            timeline = read_evidence(Journey.open(journey.path))
            investigation = project_investigation(timeline)
            # The default waits for every member: one is still pending.
            waiting = project_investigation_policy(timeline, investigation, evidence)
            self.assertEqual(waiting.effective.completion, "require_terminal_members")
            self.assertFalse(waiting.finalization_eligible)
            self.assertIn("pending_members", waiting.completion_blockers)
            # A complete literal retained policy models an invocation from a source
            # whose configured default differed. No amendment or current setting
            # may reinterpret that recorded completion decision.
            timeline[-1]["payload"]["work_policy"]["completion"] = (
                "allow_partial_with_limitations"
            )
            retained = project_investigation_policy(timeline, investigation, evidence)
            self.assertEqual(
                retained.effective.completion, "allow_partial_with_limitations"
            )
            self.assertTrue(retained.finalization_eligible)
            del timeline[-1]["payload"]["work_policy"]["other_work"]
            with self.assertRaises(ValueError):
                project_investigation_policy(timeline, investigation, evidence)

    def test_selected_approval_observes_eight_then_pending_two_and_reopens(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, first_grant, descriptor, first_command = (
                _approved_collection_observation(
                    root,
                    cardinality=10,
                    forked=True,
                    approve_count=8,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(10)
                    ).journey_file,
                )
            )
            partial = project_investigation(read_evidence(Journey.open(journey.path)))
            self.assertEqual(len(first_grant.authorized_work_items), 8)
            self.assertEqual(len(partial.pending_work_item_ids), 2)
            self.assertIsNotNone(partial.pending_request)
            first_adapter = _SequentialObservationAdapter(
                first_command, descriptor, investigation_input.sample_id
            )
            first = investigation_observe(
                first_command,
                adapter=first_adapter,
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(
                [item.status for item in first.item_states],
                ["completed"] * 8 + ["approval_pending"] * 2,
            )
            self.assertEqual(
                len(
                    [
                        call
                        for call in first_adapter.calls
                        if call[1:4]
                        == ("query", "--allow-undef-tags", "--regions-file")
                    ]
                ),
                8,
            )
            self.assertEqual(
                [
                    "{}:{}-{}".format(*query_region(first_adapter.files, call))
                    for call in first_adapter.calls
                    if call[1:4] == ("query", "--allow-undef-tags", "--regions-file")
                ],
                [
                    "NC_000019.10:44908684-44908684",
                    "NC_000019.10:44908822-44908822",
                    "NC_000019.10:21331549-21331549",
                    "NC_000019.10:14574-14574",
                    "NC_000019.10:15000-15000",
                    "NC_000019.10:15001-15001",
                    "NC_000019.10:15002-15002",
                    "NC_000019.10:15003-15003",
                ],
            )
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_evidence,
                first,
            )
            assert partial.pending_request is not None
            second = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=partial.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=first_grant.proposal_id,
                        work_item_ids=partial.pending_work_item_ids,
                    ),
                )
            )
            self.assertEqual(second.authorities[0], first_grant)
            self.assertEqual(second.pending_work_item_ids, ())
            second_command = first_command.model_copy(
                update={
                    "authority_id": second.authority.authority_id,
                    "derivation_root": str(root / "derived-later"),
                }
            )
            second_adapter = _SequentialObservationAdapter(
                second_command, descriptor, investigation_input.sample_id
            )
            combined = investigation_observe(
                second_command,
                adapter=second_adapter,
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(
                [item.status for item in combined.item_states], ["completed"] * 10
            )
            self.assertEqual(combined.item_states[:8], first.item_states[:8])
            self.assertEqual(
                len(
                    [
                        call
                        for call in second_adapter.calls
                        if call[1:4]
                        == ("query", "--allow-undef-tags", "--regions-file")
                    ]
                ),
                2,
            )
            self.assertEqual(
                [
                    "{}:{}-{}".format(*query_region(second_adapter.files, call))
                    for call in second_adapter.calls
                    if call[1:4] == ("query", "--allow-undef-tags", "--regions-file")
                ],
                ["NC_000019.10:15004-15004", "NC_000019.10:15005-15005"],
            )
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_evidence,
                combined,
            )
            frozen = journey.path.read_bytes()

            def no_contact():
                self.fail("retained command replay constructed an adapter")

            self.assertEqual(
                investigation_observe(first_command, adapter_factory=no_contact), first
            )
            self.assertEqual(
                investigation_observe(second_command, adapter_factory=no_contact),
                combined,
            )
            self.assertEqual(journey.path.read_bytes(), frozen)
            run_table_reading(journey.path)
            finalization = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            assert finalization.pending_request is not None
            self.assertEqual(len(finalization.pending_request.basis.work_items), 10)
            self.assertEqual(
                len(finalization.pending_request.basis.sample_observations), 10
            )
            timeline = Journey.open(journey.path).read()
            for mutation in (
                "grant",
                "retained_state",
                "pending_state",
                "orphan_grant",
            ):
                with self.subTest(mutation=mutation):
                    changed = copy.deepcopy(timeline)
                    if mutation == "grant":
                        grants = [
                            event
                            for event in changed
                            if event["event_type"] == "InvestigationAuthorityGranted"
                        ]
                        grants[-1]["payload"]["authority"] = first_grant.model_dump(
                            mode="json"
                        )
                    elif mutation == "orphan_grant":
                        changed = [
                            event
                            for event in changed
                            if event["event_type"] != "InvestigationAuthorityGranted"
                        ]
                    else:
                        collections = [
                            event
                            for event in changed
                            if event["event_type"]
                            == "InvestigationCollectionEvidenceRecorded"
                        ]
                        if mutation == "retained_state":
                            collections[-1]["payload"]["item_states"][0]["outcome"] = (
                                "invented"
                            )
                        else:
                            collections[0]["payload"]["item_states"][-1]["status"] = (
                                "not_started"
                            )
                    with self.assertRaises(ValueError):
                        project_journey_events(changed, journey_file=str(journey.path))

    def test_partial_approval_finalization_accounts_for_pending_and_requires_whole_basis_review(
        self,
    ) -> None:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from bio_neb.application.queries import (
            ReviewJourneyQuery,
            review_journey,
        )
        from bio_neb.cli import main

        def cli(*arguments):
            output, error = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                result = main(list(arguments))
            self.assertEqual((result, error.getvalue()), (0, ""))
            return json.loads(output.getvalue())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input, journey, _grant, descriptor, command = (
                _approved_collection_observation(
                    root,
                    cardinality=10,
                    forked=True,
                    approve_count=8,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(10)
                    ).journey_file,
                )
            )
            # Completion waits for every member unless the researcher chooses a
            # partial report; with that choice the table stage opens.
            amend_to_partial(journey)
            investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            run_table_reading(journey.path)
            original = observe(ObserveRun(journey_file=str(journey.path)))
            original_request = original.investigation.pending_request
            original_pending = original.investigation.pending_work_item_ids
            compiled = cli("finalization-compile", "--journey-file", str(journey.path))
            self.assertEqual(compiled["status"], "finalization_review_pending")
            frozen_review = journey.path.read_bytes()
            viewed = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(viewed.current_review.kind, "finalization")
            self.assertEqual(viewed.investigation.pending_request, original_request)
            self.assertEqual(
                viewed.investigation.pending_work_item_ids, original_pending
            )
            self.assertTrue(viewed.investigation.pending_request.pending)
            for changes in (
                {"pending_work_item_ids": original_pending[:-1]},
                {"status": "approved"},
                {
                    "pending_request": original_request.model_copy(
                        update={"pending": False}
                    )
                },
            ):
                hostile = dict(viewed)
                hostile["investigation"] = viewed.investigation.model_copy(
                    update=changes
                )
                with self.assertRaisesRegex(ValueError, "more than one pending Review"):
                    type(viewed).model_validate(hostile)
            self.assertEqual(
                review_journey(
                    ReviewJourneyQuery(journey_file=str(journey.path))
                ).current_review,
                viewed.current_review,
            )
            for flags in ((), ("--rich",)):
                reopened = cli("observe", "--journey-file", str(journey.path), *flags)
                self.assertEqual(reopened["status"], "finalization_review_pending")
            self.assertEqual(journey.path.read_bytes(), frozen_review)
            finalization = viewed.finalization
            pending = finalization.pending_request
            assert pending is not None
            self.assertEqual(len(pending.basis.work_items), 10)
            self.assertEqual(len(pending.basis.sample_observations), 8)
            self.assertEqual(
                [item.status for item in pending.basis.collection_item_states],
                ["completed"] * 8 + ["approval_pending"] * 2,
            )
            self.assertIsNone(finalization.report_basis)
            with self.assertRaises(GroundedReportApplicationRefused) as unapproved:
                publish_grounded_report(
                    PublishGroundedReportCommand(
                        journey_file=str(journey.path),
                        report_basis_digest="report-basis:" + "0" * 64,
                        head_event_id=pending.request_event_id,
                    )
                )
            self.assertEqual(
                unapproved.exception.code, "grounded_report_basis_unapproved"
            )
            approved = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    basis_digest=pending.basis.basis_digest,
                    head_event_id=pending.request_event_id,
                )
            )
            assert approved.report_basis is not None
            report = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=approved.report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            ).report
            assert report is not None
            self.assertEqual(len(report.work_items), 10)
            self.assertEqual(report.collection_status, "partial")
            retained_bases = observe(
                ObserveRun(journey_file=str(journey.path))
            ).finalization.bases
            current = project_investigation(read_evidence(Journey.open(journey.path)))
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=current.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=current.proposals[-1].proposal_id,
                        work_item_ids=current.pending_work_item_ids,
                    ),
                )
            )
            from bio_neb.investigation.investigation_policy import (
                InvestigationPolicyAmendment,
            )
            from bio_neb.investigation.investigation_policy_commands import (
                AmendInvestigationPolicyCommand,
                amend_investigation_policy,
            )

            amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=current.proposals[-1].proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        # The researcher's earlier partial choice is the policy now.
                        previous_policy_id=read_investigation(
                            Journey.open(journey.path)
                        ).work_policy.policy_id,
                        asserted_by="researcher:owner",
                        reason="A new approval is not new evidence.",
                        completion="allow_partial_with_limitations",
                    ),
                )
            )
            frozen = journey.path.read_bytes()
            with self.assertRaises(FinalizationApplicationRefused) as refused:
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
            self.assertEqual(refused.exception.code, "finalization_evidence_unchanged")
            self.assertEqual(journey.path.read_bytes(), frozen)
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))).finalization.bases,
                retained_bases,
            )

    def test_partial_selection_consumes_one_request_and_next_selection_has_fresh_scope(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            _, journey, authority, _, _ = _approved_collection_observation(
                Path(temporary).resolve(), cardinality=2, approve_count=1
            )
            projection = project_investigation(
                read_evidence(Journey.open(journey.path))
            )
            original = projection.review_receipts[-1]
            next_request = projection.pending_request
            self.assertNotEqual(
                next_request.request_event_id, original.request_event_id
            )
            frozen = journey.path.read_bytes()
            replay = RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=original.request_event_id,
                response=original.response,
            )
            self.assertEqual(respond_investigation_review(replay), projection)
            self.assertEqual(journey.path.read_bytes(), frozen)
            second = ApproveInvestigationResponse(
                action="approve",
                proposal_id=authority.proposal_id,
                work_item_ids=projection.pending_work_item_ids,
            )
            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                respond_investigation_review(
                    replay.model_copy(update={"response": second})
                )
            self.assertEqual(conflict.exception.code, "investigation_review_conflict")
            self.assertEqual(journey.path.read_bytes(), frozen)
            accepted = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=next_request.request_event_id,
                    response=second,
                )
            )
            self.assertEqual(accepted.authorities[0], authority)
            self.assertEqual(len(accepted.authorities), 2)
            self.assertEqual(accepted.pending_work_item_ids, ())

    def test_policy_cli_amends_with_runtime_defaults_and_inspects_the_retained_decision(
        self,
    ) -> None:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from bio_neb.cli import main
        from tests.proposal_information_fakes import runtime_configuration

        def cli(*arguments):
            output, error = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                result = main(list(arguments))
            self.assertEqual((result, error.getvalue()), (0, ""))
            return json.loads(output.getvalue())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _input, journey, _first, _descriptor, _command = (
                _approved_collection_observation(root, cardinality=3, approve_count=1)
            )
            partial = project_investigation(read_evidence(Journey.open(journey.path)))
            held_id = partial.pending_work_item_ids[0]
            # The amendment names no completion; the runtime file's non-default
            # completion is the default the CLI must supply.
            amendment = root / "amendment.json"
            amendment.write_text(
                json.dumps(
                    {
                        "proposal_id": partial.proposals[-1].proposal_id,
                        "expected_head_event_id": Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        "previous_policy_id": None,
                        "asserted_by": "researcher:owner",
                        "reason": "Require terminal work and hold one member.",
                        "hold_work_item_ids": [held_id],
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "runtime.json"
            runtime.write_text(
                json.dumps(
                    {
                        **runtime_configuration(),
                        "investigation_work": {
                            "completion": "require_terminal_members"
                        },
                    }
                ),
                encoding="utf-8",
            )
            amended = cli(
                "investigation-policy-amend",
                "--journey-file",
                str(journey.path),
                "--amendment-file",
                str(amendment),
                "--runtime-config",
                str(runtime),
            )
            self.assertEqual(amended["held_work_item_ids"], [held_id])
            self.assertEqual(
                amended["effective"]["completion"], "require_terminal_members"
            )
            self.assertEqual(
                cli("investigation-policy", "--journey-file", str(journey.path)),
                amended,
            )

    def test_partial_review_refuses_stale_overlapping_and_unknown_selections_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _input, journey, first, _descriptor, _command = (
                _approved_collection_observation(root, cardinality=4, approve_count=2)
            )
            partial = project_investigation(read_evidence(Journey.open(journey.path)))
            assert partial.pending_request is not None
            request_id = partial.pending_request.request_event_id
            frozen = journey.path.read_bytes()
            replay = RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=partial.review_receipts[0].request_event_id,
                response=ApproveInvestigationResponse(
                    action="approve",
                    proposal_id=first.proposal_id,
                    work_item_ids=tuple(
                        reversed(
                            [item.work_item_id for item in first.authorized_work_items]
                        )
                    ),
                ),
            )
            self.assertEqual(respond_investigation_review(replay), partial)
            for ids, proposal_id, selected_request, code in (
                (
                    (
                        first.authorized_work_items[0].work_item_id,
                        partial.pending_work_item_ids[0],
                    ),
                    first.proposal_id,
                    request_id,
                    "investigation_review_conflict",
                ),
                (
                    ("work-item:" + "0" * 64,),
                    first.proposal_id,
                    request_id,
                    "investigation_review_selection_invalid",
                ),
                (
                    partial.pending_work_item_ids,
                    "investigation-proposal:" + "0" * 64,
                    request_id,
                    "investigation_review_stale",
                ),
                (
                    partial.pending_work_item_ids,
                    first.proposal_id,
                    "stale-request",
                    "investigation_review_stale",
                ),
            ):
                with self.subTest(
                    ids=ids, proposal_id=proposal_id, request=selected_request
                ):
                    with self.assertRaises(InvestigationApplicationRefused) as refused:
                        respond_investigation_review(
                            RespondInvestigationReviewCommand(
                                journey_file=str(journey.path),
                                request_event_id=selected_request,
                                response=ApproveInvestigationResponse(
                                    action="approve",
                                    proposal_id=proposal_id,
                                    work_item_ids=ids,
                                ),
                            )
                        )
                    self.assertEqual(refused.exception.code, code)
                    self.assertEqual(journey.path.read_bytes(), frozen)

    def test_postapproval_selected_amend_carries_eight_results_and_grants(self) -> None:
        class ProblemAdapter(_SequentialObservationAdapter):
            def run(
                self,
                argv: tuple[str, ...],
                *,
                timeout_seconds: int,
                stdout_path: str | None = None,
            ) -> CompletedProcess:
                if argv[1:4] == (
                    "query",
                    "--allow-undef-tags",
                    "--regions-file",
                ) and "{}:{}-{}".format(*query_region(self.files, argv)) in {
                    "NC_000019.10:44908684-44908684",
                    "NC_000019.10:44908822-44908822",
                }:
                    self.calls.append(argv)
                    return CompletedProcess(0, b"", b"")
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, grant, descriptor, command = _approved_collection_observation(
                root,
                cardinality=10,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(10)
                ).journey_file,
            )
            first = investigation_observe(
                command,
                adapter=ProblemAdapter(command, descriptor, inp.sample_id),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(
                [item.status for item in first.item_states],
                ["unresolved"] * 2 + ["completed"] * 8,
            )
            run_table_reading(journey.path)
            initial_final = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            initial_pending = initial_final.pending_request
            assert initial_pending is not None
            initial_approval = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=initial_pending.request_event_id,
                    head_event_id=initial_pending.request_event_id,
                    basis_digest=initial_pending.basis.basis_digest,
                )
            )
            assert initial_approval.report_basis is not None
            first_report = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=initial_approval.report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            )
            assert first_report.report is not None
            original_bytes = journey.path.read_bytes()
            original = project_investigation(read_evidence(Journey.open(journey.path)))
            proposal = original.proposals[-1]
            response = RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=original.review_requests[-1].request_event_id,
                response=ReviseInvestigationResponse(
                    action="revise",
                    proposal_id=proposal.proposal_id,
                    asserted_by="researcher:owner",
                    operations=tuple(
                        AmendInvestigationMember(
                            operation="amend",
                            work_item_id=item.work_item_id,
                            request=InvestigationRequestItem(
                                rsid=item.target.aliases[0],
                                request_text=f"Recheck {item.target.aliases[0]} after an ordinary item problem.",
                            ),
                        )
                        if index < 2
                        else KeepInvestigationMember(
                            operation="keep", work_item_id=item.work_item_id
                        )
                        for index, item in enumerate(proposal.work_items)
                    ),
                ),
            )
            revised = respond_investigation_review(response)
            self.assertEqual(revised.status, "investigation_work_ready")
            self.assertEqual(respond_investigation_review(response), revised)
            model = _SequentialAdaptiveModel()
            sources = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            continued = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )
            self.assertEqual(
                continued.model_contact_count - original.model_contact_count, 6
            )
            self.assertEqual(
                continued.source_contact_count - original.source_contact_count, 4
            )
            # The revision plans each changed member in its own Attempt.
            self.assertEqual(len({a.revision_id for a in continued.attempts}), 2)
            self.assertEqual(
                continued.proposals[-1].work_items[2:], proposal.work_items[2:]
            )
            self.assertEqual(len(continued.pending_work_item_ids), 2)
            self.assertIn(grant, continued.authorities)
            pending = continued.pending_request
            assert pending is not None
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=pending.proposal.proposal_id,
                        work_item_ids=continued.pending_work_item_ids,
                    ),
                )
            )
            assert approved.authority is not None
            second_command = command.model_copy(
                update={
                    "authority_id": approved.authority.authority_id,
                    "derivation_root": str(root / "reworked"),
                }
            )
            adapter = _SequentialObservationAdapter(
                second_command, descriptor, inp.sample_id
            )
            second = investigation_observe(
                second_command, adapter=adapter, evidence_reader=_Evidence(descriptor)
            )
            self.assertEqual(second.item_states[2:], first.item_states[2:])
            self.assertEqual(
                _post_execution_vcf_queries(journey), 12 + _item_windows(journey)
            )
            self.assertEqual(
                len(
                    [
                        call
                        for call in adapter.calls
                        if call[1:4]
                        == ("query", "--allow-undef-tags", "--regions-file")
                    ]
                ),
                2,
            )
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_evidence,
                second,
            )
            self.assertEqual(investigation_observe(command), first)
            self.assertTrue(journey.path.read_bytes().startswith(original_bytes))
            from bio_neb.investigation.investigation_policy import (
                InvestigationPolicyAmendment,
            )
            from bio_neb.investigation.investigation_policy_commands import (
                AmendInvestigationPolicyCommand,
                amend_investigation_policy,
            )

            amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=approved.proposals[-1].proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=None,
                        asserted_by="researcher:owner",
                        reason="Complete reworked evidence before the next basis.",
                        completion="require_terminal_members",
                    ),
                )
            )
            run_table_reading(journey.path)
            final = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            self.assertEqual(final.status, "review_pending")
            self.assertEqual(len(final.bases), 2)
            self.assertEqual(len(final.bases[-1].work_items), 10)
            self.assertEqual(len(final.bases[-1].sample_observations), 10)
            self.assertEqual(final.bases[0], initial_pending.basis)
            self.assertEqual(
                reopen_grounded_report(
                    ReopenGroundedReportCommand(
                        journey_file=str(journey.path),
                        report_digest=first_report.report.report_digest,
                    )
                ),
                first_report,
            )
            final_pending = final.pending_request
            assert final_pending is not None
            with self.subTest(frontier="new_finalization_review"):
                self.assertEqual(
                    observe(ObserveRun(journey_file=str(journey.path))).status,
                    "finalization_review_pending",
                )
            hostile = copy.deepcopy(read_evidence(Journey.open(journey.path)))
            orphan = copy.deepcopy(
                next(
                    event
                    for event in hostile
                    if event["event_type"] == "FinalizationReviewRecorded"
                )
            )
            orphan["event_id"] = "00000000-0000-4000-8000-000000000999"
            first_basis_index = next(
                index
                for index, event in enumerate(hostile)
                if event["event_type"] == "FinalizationBasisCompiled"
            )
            hostile.insert(first_basis_index, orphan)
            with self.subTest(frontier="orphan_finalization_review"):
                with self.assertRaisesRegex(ValueError, "invalid backward parentage"):
                    project_finalization(hostile)
            with self.assertRaises(GroundedReportApplicationRefused) as superseded:
                publish_grounded_report(
                    PublishGroundedReportCommand(
                        journey_file=str(journey.path),
                        report_basis_digest=initial_approval.report_basis.report_basis_digest,
                        head_event_id=final_pending.request_event_id,
                    )
                )
            self.assertEqual(
                superseded.exception.code, "grounded_report_basis_corrected_unapproved"
            )
            final_approval = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=final_pending.request_event_id,
                    head_event_id=final_pending.request_event_id,
                    basis_digest=final_pending.basis.basis_digest,
                )
            )
            assert final_approval.report_basis is not None
            second_report = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=final_approval.report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            )
            self.assertNotEqual(second_report.report, first_report.report)
            self.assertEqual(
                reopen_grounded_report(
                    ReopenGroundedReportCommand(
                        journey_file=str(journey.path),
                        report_digest=first_report.report.report_digest,
                    )
                ),
                first_report,
            )

    def test_finalization_returns_selected_retry_and_defer_to_same_run(self) -> None:
        class UnresolvedAdapter(_Adapter):
            def run(
                self,
                argv: tuple[str, ...],
                *,
                timeout_seconds: int,
                stdout_path: str | None = None,
            ) -> CompletedProcess:
                if argv[1:4] == ("query", "--allow-undef-tags", "--regions-file"):
                    self.calls.append(argv)
                    return CompletedProcess(0, b"", b"")
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, grant, descriptor, command = _approved_collection_observation(
                root,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            first = investigation_observe(
                command,
                adapter=UnresolvedAdapter(command, descriptor, inp.sample_id),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(first.item_states[0].status, "unresolved")
            original = project_investigation(read_evidence(Journey.open(journey.path)))
            proposal = original.proposals[-1]
            run_table_reading(journey.path)
            final = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            pending = final.pending_request
            assert pending is not None
            revision = ReviseInvestigationResponse(
                action="revise",
                proposal_id=proposal.proposal_id,
                asserted_by="researcher:owner",
                operations=(
                    RetryInvestigationMember(
                        operation="retry",
                        work_item_id=proposal.work_items[0].work_item_id,
                    ),
                    DeferInvestigationMember(
                        operation="defer",
                        work_item_id=proposal.work_items[1].work_item_id,
                    ),
                    KeepInvestigationMember(
                        operation="keep",
                        work_item_id=proposal.work_items[2].work_item_id,
                    ),
                ),
            )
            correction = FinalizationCorrectionAssertion(
                asserted_by="researcher:owner",
                reason="Recheck the unresolved item; defer the other question.",
                requested_changes=(
                    InvestigationRequiredChange(
                        action="change_observation",
                        description="Retry the selected unresolved observation.",
                    ),
                ),
                investigation_revision=revision,
            )
            correction_file = root / "correction.json"
            correction_file.write_text(
                canonical_json(
                    correction.model_dump(mode="json", exclude_computed_fields=True)
                ),
                encoding="utf-8",
            )
            correction_command = CorrectFinalizationCommand(
                journey_file=str(journey.path),
                request_event_id=pending.request_event_id,
                head_event_id=pending.request_event_id,
                basis_digest=pending.basis.basis_digest,
                correction_file=str(correction_file),
            )
            before_correction = journey.path.read_bytes()
            for invalid_revision, code in (
                (
                    revision.model_copy(
                        update={"proposal_id": "investigation-proposal:" + "0" * 64}
                    ),
                    "finalization_rework_stale",
                ),
                (
                    revision.model_copy(
                        update={
                            "operations": (
                                RetryInvestigationMember(
                                    operation="retry",
                                    work_item_id="work-item:" + "0" * 64,
                                ),
                                *revision.operations[1:],
                            )
                        }
                    ),
                    "finalization_rework_invalid",
                ),
            ):
                invalid = correction.model_copy(
                    update={"investigation_revision": invalid_revision}
                )
                correction_file.write_text(
                    canonical_json(
                        invalid.model_dump(mode="json", exclude_computed_fields=True)
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(FinalizationApplicationRefused) as refused:
                    correct_finalization(correction_command)
                self.assertEqual(refused.exception.code, code)
                self.assertEqual(journey.path.read_bytes(), before_correction)
            correction_file.write_text(
                canonical_json(
                    correction.model_dump(mode="json", exclude_computed_fields=True)
                ),
                encoding="utf-8",
            )
            returned = correct_finalization(correction_command)
            self.assertEqual(returned.status, "returned_to_investigation")
            self.assertEqual(correct_finalization(correction_command), returned)
            rework = project_investigation(read_evidence(Journey.open(journey.path)))
            self.assertEqual(rework.status, "investigation_work_ready")
            hostile = copy.deepcopy(read_evidence(Journey.open(journey.path)))
            bridge = next(
                event
                for event in reversed(hostile)
                if event["event_type"] == "InvestigationReviewResponded"
            )
            bridge["payload"]["finalization_response_event_id"] = (
                "missing-finalization-response"
            )
            with self.assertRaisesRegex(
                ValueError,
                "Investigation rework changed Finalization correction lineage",
            ):
                project_investigation(hostile)
            self.assertEqual(
                rework.revisions[-1].deferred_work_item_ids,
                (proposal.work_items[1].work_item_id,),
            )
            continued = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                model_factory=lambda _loaded, _selected: (
                    _SequentialAdaptiveModel().model(_loaded, _selected)
                ),
                source_adapter_factory=lambda: _SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            self.assertEqual(len(continued.pending_work_item_ids), 1)
            retry = continued.proposals[-1].work_items[0]
            self.assertEqual(
                retry.previous_work_item_id, proposal.work_items[0].work_item_id
            )
            self.assertEqual(retry.revision_id, rework.revisions[-1].revision_id)
            review = continued.pending_request
            assert review is not None
            view = observe(ObserveRun(journey_file=str(journey.path)))
            approval = review_response_from_json(
                view,
                review.request_event_id,
                json.dumps(
                    {
                        "action": "approve",
                        "proposal_id": review.proposal.proposal_id,
                        "work_item_ids": [retry.work_item_id],
                    }
                ),
            )
            adapter = UnresolvedAdapter(command, descriptor, inp.sample_id)
            item_windows_before = _item_windows(journey)
            with use_run_advancement_dependencies(
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                investigation_model_factory=TableReaderModel().model,
                investigation_observation_adapter=adapter,
                investigation_observation_evidence_reader=_Evidence(descriptor),
            ):
                completed = advance(approval)
            second = completed.investigation_evidence
            # One lookup for the retried item, then each settled item's own query.
            self.assertEqual(
                len(
                    [
                        a
                        for a in adapter.calls
                        if a[1:4] == ("query", "--allow-undef-tags", "--regions-file")
                    ]
                ),
                1 + _item_windows(journey) - item_windows_before,
            )
            self.assertEqual(second.item_states[0].status, "unresolved")
            self.assertEqual(
                second.item_states[1].evidence_state_id,
                first.item_states[2].evidence_state_id,
            )
            run_table_reading(journey.path)
            next_final = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            self.assertEqual(next_final.status, "review_pending")
            self.assertEqual(next_final.bases[0], pending.basis)
            self.assertEqual(
                next_final.bases[-1].supersedes_basis_digest, pending.basis.basis_digest
            )
            next_pending = next_final.pending_request
            assert next_pending is not None
            approval = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=next_pending.request_event_id,
                    head_event_id=next_pending.request_event_id,
                    basis_digest=next_pending.basis.basis_digest,
                )
            )
            assert approval.report_basis is not None
            report = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=approval.report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            )
            assert report.report is not None
            self.assertEqual(
                reopen_grounded_report(
                    ReopenGroundedReportCommand(
                        journey_file=str(journey.path),
                        report_digest=report.report.report_digest,
                    )
                ),
                report,
            )

    def test_rework_preserves_pending_and_unobserved_sibling_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, first_grant, descriptor, command = (
                _approved_collection_observation(
                    root,
                    forked=True,
                    approve_count=1,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=collection_request(3)
                    ).journey_file,
                )
            )
            first = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            original = project_investigation(read_evidence(Journey.open(journey.path)))
            proposal = original.proposals[-1]
            request_id = original.review_requests[-1].request_event_id
            second_approval = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=proposal.proposal_id,
                        work_item_ids=(proposal.work_items[1].work_item_id,),
                    ),
                )
            )
            second_grant = second_approval.authority
            assert second_grant is not None
            revision_response = ReviseInvestigationResponse(
                action="revise",
                proposal_id=proposal.proposal_id,
                asserted_by="researcher:owner",
                operations=(
                    RetryInvestigationMember(
                        operation="retry",
                        work_item_id=proposal.work_items[0].work_item_id,
                    ),
                    *(
                        KeepInvestigationMember(
                            operation="keep", work_item_id=item.work_item_id
                        )
                        for item in proposal.work_items[1:]
                    ),
                ),
            )
            # The second approval closed its Review; the rework answers the
            # Review it opened for the remaining member.
            remaining = project_investigation(
                read_evidence(Journey.open(journey.path))
            ).pending_request
            assert remaining is not None
            self.assertNotEqual(remaining.request_event_id, request_id)
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=remaining.request_event_id,
                    response=revision_response,
                )
            )
            sibling_command = command.model_copy(
                update={
                    "authority_id": second_grant.authority_id,
                    "derivation_root": str(root / "sibling"),
                }
            )
            sibling = investigation_observe(
                sibling_command,
                adapter=_SequentialObservationAdapter(
                    sibling_command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(sibling.item_states[0], first.item_states[0])
            self.assertEqual(sibling.item_states[2].status, "approval_pending")
            continued = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                model_factory=lambda _loaded, _selected: (
                    _SequentialAdaptiveModel().model(_loaded, _selected)
                ),
                source_adapter_factory=lambda: _SequentialSourceResponses(
                    assembly_body=ASSEMBLY_BODY
                ),
            )
            self.assertEqual(len(continued.pending_work_item_ids), 2)
            pending = continued.pending_request
            assert pending is not None
            selected = continued.proposals[-1].work_items[0]
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=pending.proposal.proposal_id,
                        work_item_ids=(selected.work_item_id,),
                    ),
                )
            )
            assert approved.authority is not None
            retry_command = command.model_copy(
                update={
                    "authority_id": approved.authority.authority_id,
                    "derivation_root": str(root / "retry"),
                }
            )
            retried = investigation_observe(
                retry_command,
                adapter=_SequentialObservationAdapter(
                    retry_command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertEqual(retried.item_states[1], sibling.item_states[1])
            self.assertEqual(retried.item_states[2].status, "approval_pending")
            self.assertEqual(
                observe(
                    ObserveRun(journey_file=str(journey.path))
                ).investigation_evidence,
                retried,
            )
            self.assertEqual(investigation_observe(sibling_command), sibling)
            frozen = journey.path.read_bytes()
            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=request_id,
                        response=ReviseInvestigationResponse(
                            action="revise",
                            proposal_id=proposal.proposal_id,
                            asserted_by="another:owner",
                            operations=revision_response.operations,
                        ),
                    )
                )
            self.assertEqual(conflict.exception.code, "investigation_review_conflict")
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_postapproval_deferral_compiles_carried_basis_without_new_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inp, journey, grant, descriptor, command = _approved_collection_observation(
                root,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(3)
                ).journey_file,
            )
            first = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            before = project_investigation(read_evidence(Journey.open(journey.path)))
            run_table_reading(journey.path)
            original = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            pending = original.pending_request
            assert pending is not None
            approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    head_event_id=pending.request_event_id,
                    basis_digest=pending.basis.basis_digest,
                )
            )
            response = RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=before.review_requests[-1].request_event_id,
                response=ReviseInvestigationResponse(
                    action="revise",
                    proposal_id=before.proposals[-1].proposal_id,
                    asserted_by="researcher:owner",
                    operations=(
                        KeepInvestigationMember(
                            operation="keep",
                            work_item_id=grant.authorized_work_items[0].work_item_id,
                        ),
                        *(
                            DeferInvestigationMember(
                                operation="defer", work_item_id=item.work_item_id
                            )
                            for item in grant.authorized_work_items[1:]
                        ),
                    ),
                ),
            )
            revised = respond_investigation_review(response)
            self.assertEqual(revised.model_contact_count, before.model_contact_count)
            self.assertEqual(revised.source_contact_count, before.source_contact_count)
            self.assertEqual(revised.authorities, before.authorities)
            self.assertEqual(len(revised.attempts), len(before.attempts))
            run_table_reading(journey.path)
            final = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            self.assertEqual(final.status, "review_pending")
            self.assertEqual(len(final.bases[-1].work_items), 1)
            self.assertEqual(
                _post_execution_vcf_queries(journey), 3 + _item_windows(journey)
            )
            self.assertEqual(
                final.bases[-1].collection_item_states, first.item_states[:1]
            )
            self.assertEqual(final.bases[0], pending.basis)
            self.assertEqual(investigation_observe(command), first)

    def test_mixed_revision_contacts_only_changed_added_and_reopened_work(self) -> None:
        initial_rsids = (
            "rs429358",
            "rs7412",
            "rs4149056",
            "rs28503599",
            "rs1001",
            "rs1002",
        )

        class RevisionSourceResponses(_SequentialSourceResponses):
            positions = {
                **_SequentialSourceResponses.positions,
                "rs28503599": 14574,
                "rs1001": 1001,
                "rs1002": 1002,
                "rs1003": 1003,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": f"Investigate {rsid} in sample HG001.",
                        "target": {"rsid": rsid},
                    }
                    for rsid in initial_rsids
                ],
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(request), encoding="utf-8")
            initial_model = _SequentialAdaptiveModel()
            adapter = RevisionSourceResponses(assembly_body=ASSEMBLY_BODY)
            first = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                model_factory=initial_model.model,
                source_adapter_factory=lambda: adapter,
            )
            pending = first.pending_request
            assert pending is not None
            original = pending.proposal
            item_ids = tuple(item.work_item_id for item in original.work_items)
            before_revision_bytes = journey.path.read_bytes()
            revised = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=original.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            KeepInvestigationMember(
                                operation="keep", work_item_id=item_ids[0]
                            ),
                            AmendInvestigationMember(
                                operation="amend",
                                work_item_id=item_ids[1],
                                request=InvestigationRequestItem(
                                    request_text="Investigate amended rs7412 scope.",
                                    rsid="rs7412",
                                ),
                            ),
                            RemoveInvestigationMember(
                                operation="remove", work_item_id=item_ids[2]
                            ),
                            DeferInvestigationMember(
                                operation="defer", work_item_id=item_ids[3]
                            ),
                            ClarifyInvestigationMember(
                                operation="clarify",
                                work_item_id=item_ids[4],
                                request=InvestigationRequestItem(
                                    request_text="Investigate clarified rs1001 scope.",
                                    rsid="rs1001",
                                ),
                            ),
                            AdjustInvestigationSearch(
                                operation="search_adjustment",
                                work_item_id=item_ids[5],
                                request=InvestigationRequestItem(
                                    request_text="Investigate rs1002 with adjusted search.",
                                    rsid="rs1002",
                                ),
                            ),
                            AddInvestigationMember(
                                operation="add",
                                request=InvestigationRequestItem(
                                    request_text="Investigate added rs1003 scope.",
                                    rsid="rs1003",
                                ),
                            ),
                        ),
                    ),
                )
            )
            self.assertNotEqual(journey.path.read_bytes(), before_revision_bytes)
            self.assertEqual(revised.status, "investigation_work_ready")
            self.assertIsNone(revised.pending_request)
            self.assertIsNone(revised.authority)
            self.assertEqual(revised.model_contact_count, 18)
            self.assertEqual(revised.source_contact_count, 12)
            self.assertEqual(len(revised.revisions), 1)
            self.assertEqual(len(revised.revisions[0].work_requests), 4)

            accepted_revision_bytes = journey.path.read_bytes()
            retained_lines = accepted_revision_bytes.splitlines(keepends=True)
            self.assertEqual(
                json.loads(retained_lines[-1])["event_type"],
                "InvestigationAssertionRecorded",
            )
            journey.path.write_bytes(b"".join(retained_lines[:-1]))
            truncated_revision_bytes = journey.path.read_bytes()
            with self.assertRaises(ValueError):
                continue_investigation(
                    ContinueInvestigationCommand(journey_file=str(journey.path)),
                    environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                    model_factory=lambda _loaded, _selected: (_ for _ in ()).throw(
                        AssertionError("truncated revision contacted model factory")
                    ),
                    source_adapter_factory=lambda: (_ for _ in ()).throw(
                        AssertionError("truncated revision contacted source factory")
                    ),
                )
            self.assertEqual(journey.path.read_bytes(), truncated_revision_bytes)
            journey.path.write_bytes(accepted_revision_bytes)

            continuation_model = _SequentialAdaptiveModel()
            second = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                model_factory=continuation_model.model,
                source_adapter_factory=lambda: adapter,
            )
            self.assertEqual(second.status, "review_pending")
            self.assertEqual(second.model_contact_count, 30)
            self.assertEqual(second.source_contact_count, 20)
            # One Attempt per planned member, grouped by revision.
            by_revision: dict = {}
            for attempt in second.attempts:
                by_revision.setdefault(attempt.revision_id, []).append(attempt)
            initial, revised_members = by_revision.values()
            self.assertEqual(len(initial), 6)
            self.assertEqual(len(revised_members), 4)
            self.assertEqual(sum(len(a.model_contact_event_ids) for a in initial), 18)
            self.assertEqual(sum(len(a.source_contact_event_ids) for a in initial), 12)
            self.assertEqual(
                sum(len(a.model_contact_event_ids) for a in revised_members), 12
            )
            self.assertEqual(
                sum(len(a.source_contact_event_ids) for a in revised_members), 8
            )
            # One policy for every Attempt, compared by identity: its stage map
            # leaves the policy unhashable.
            self.assertEqual(
                {a.continuation_policy.policy_id for a in second.attempts},
                {initial[0].continuation_policy.policy_id},
            )
            self.assertEqual(len(revised_members[0].work_requests), 4)
            superseding = second.proposals[-1]
            self.assertEqual(superseding.revision_number, 2)
            self.assertEqual(superseding.supersedes_proposal_id, original.proposal_id)
            self.assertEqual(superseding.work_items[0], original.work_items[0])
            self.assertEqual(
                tuple(item.target.aliases[0] for item in superseding.work_items),
                ("rs429358", "rs7412", "rs1001", "rs1002", "rs1003"),
            )
            original_asserted = next(
                value
                for value in first.requirement_values
                if value.value == "rs429358" and value.status == "asserted"
            )
            carried_asserted = next(
                value
                for value in second.requirement_values
                if value.value == "rs429358" and value.status == "asserted"
            )
            self.assertEqual(carried_asserted, original_asserted)
            amended_asserted = next(
                value
                for value in second.requirement_values
                if value.value == "rs7412" and value.status == "asserted"
            )
            self.assertEqual(
                amended_asserted.lineage_event_ids,
                (json.loads(retained_lines[-1])["event_id"],),
            )
            self.assertNotEqual(
                amended_asserted.lineage_event_ids, original_asserted.lineage_event_ids
            )
            second_pending = second.pending_request
            assert second_pending is not None
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=second_pending.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve", proposal_id=superseding.proposal_id
                    ),
                )
            )
            self.assertEqual(approved.status, "approved")
            assert approved.authority is not None
            self.assertEqual(
                approved.authority.authorized_work_items, superseding.work_items
            )
            before_reopen = journey.path.read_bytes()
            timeline = read_evidence(Journey.open(journey.path))
            reopened = project_investigation(timeline)
            self.assertEqual(reopened, approved)
            self.assertEqual(journey.path.read_bytes(), before_reopen)

            hostile_policy = copy.deepcopy(timeline)
            revision_binding = [
                event
                for event in hostile_policy
                if event["event_type"] == "InvestigationAttemptBound"
            ][1]
            revision_binding["payload"]["continuation_policy"]["failed_turns"] += 1
            with self.assertRaises(ValueError):
                project_investigation(hostile_policy)

            hostile_revision = copy.deepcopy(timeline)
            revision_event = next(
                event
                for event in hostile_revision
                if event["event_type"] == "InvestigationCollectionRevisionRecorded"
            )
            revision_event["payload"]["revision"]["members"][0]["operation"] = "amend"
            with self.assertRaises(ValueError):
                project_investigation(hostile_revision)
            self.assertEqual(journey.path.read_bytes(), before_reopen)

    def test_remove_and_defer_revision_carries_kept_member_without_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input = InvestigationInput.create(
                vcf=InvestigationVcfIdentity.create(
                    bucket="accepted-run-bucket",
                    object_key="outputs/HG001.vcf.gz",
                    byte_count=1234,
                    sha256="a" * 64,
                ),
                sample_id="HG001",
                assembly_accession="GCF_000001405.40",
            )
            requests = tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in _SequentialAdaptiveModel.rsids
            )
            assertion = UserAssertion.create_collection(
                asserted_by="researcher:owner",
                requests=requests,
                predecessor_ids=(investigation_input.input_id,),
            )
            model = _SequentialAdaptiveModel()
            adapter = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)
            pending, journey = _run_attempt(
                root,
                model,
                adapter,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            first_request = pending.pending_request
            assert first_request is not None
            first_proposal = first_request.proposal
            before_bytes = journey.path.read_bytes()
            before_model_contacts = pending.model_contact_count
            before_source_contacts = pending.source_contact_count

            revised = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=first_request.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=first_proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            KeepInvestigationMember(
                                operation="keep",
                                work_item_id=first_proposal.work_items[0].work_item_id,
                            ),
                            RemoveInvestigationMember(
                                operation="remove",
                                work_item_id=first_proposal.work_items[1].work_item_id,
                            ),
                            DeferInvestigationMember(
                                operation="defer",
                                work_item_id=first_proposal.work_items[2].work_item_id,
                            ),
                        ),
                    ),
                )
            )

            self.assertNotEqual(journey.path.read_bytes(), before_bytes)
            self.assertEqual(revised.status, "review_pending")
            self.assertIsNone(revised.authority)
            # No member was planned again: only the first revision's Attempts.
            self.assertEqual({a.revision_id for a in revised.attempts}, {None})
            self.assertEqual(revised.model_contact_count, before_model_contacts)
            self.assertEqual(revised.source_contact_count, before_source_contacts)
            self.assertEqual(len(revised.proposals), 2)
            second_proposal = revised.proposals[-1]
            self.assertEqual(second_proposal.revision_number, 2)
            self.assertEqual(
                second_proposal.supersedes_proposal_id, first_proposal.proposal_id
            )
            self.assertEqual(
                second_proposal.work_items, (first_proposal.work_items[0],)
            )
            self.assertEqual(
                revised.pending_request.proposal if revised.pending_request else None,
                second_proposal,
            )
            reopened = project_investigation(read_evidence(Journey.open(journey.path)))
            self.assertEqual(reopened, revised)

    def test_collection_finalization_report_preserves_membership_status_and_contact_free_reopen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for cardinality in (1, 4):
                with self.subTest(cardinality=cardinality):
                    case_root = root / f"cardinality-{cardinality}"
                    (
                        investigation_input,
                        journey,
                        authority,
                        descriptor,
                        observation_command,
                    ) = _approved_collection_observation(
                        case_root, cardinality=cardinality
                    )
                    recorded = investigation_observe(
                        observation_command,
                        adapter=_SequentialObservationAdapter(
                            observation_command,
                            descriptor,
                            investigation_input.sample_id,
                        ),
                        evidence_reader=_Evidence(descriptor),
                    )
                    selected_ids = tuple(
                        item.work_item_id for item in authority.authorized_work_items
                    )
                    self.assertEqual(len(selected_ids), cardinality)
                    self.assertEqual(
                        isinstance(recorded, InvestigationCollectionEvidenceRecorded),
                        True,
                    )
                    run_table_reading(journey.path)
                    finalization = compile_finalization(
                        CompileFinalizationCommand(journey_file=str(journey.path))
                    )
                    pending = finalization.pending_request
                    assert pending is not None
                    basis = pending.basis
                    self.assertEqual(
                        {item.work_item_id for item in basis.work_items},
                        set(selected_ids),
                    )
                    self.assertEqual(basis.selected_work_item_ids, selected_ids)
                    self.assertEqual(basis.collection_status, "complete")
                    self.assertEqual(
                        tuple(item.status for item in basis.collection_item_states),
                        ("completed",) * cardinality,
                    )
                    excluded_member_id = None
                    if cardinality == 4:
                        excluded_member_id = basis.work_items[0].work_item_id
                        correction = FinalizationCorrectionAssertion(
                            asserted_by="researcher:owner",
                            reason="Exclude one member from report presentation.",
                            requested_changes=(
                                SetReportMembershipChange(
                                    member_id=excluded_member_id,
                                    included=False,
                                ),
                            ),
                        )
                        correction_file = case_root / "correction.json"
                        correction_file.write_text(
                            correction.model_dump_json(exclude_computed_fields=True),
                            encoding="utf-8",
                        )
                        corrected = correct_finalization(
                            CorrectFinalizationCommand(
                                journey_file=str(journey.path),
                                request_event_id=pending.request_event_id,
                                basis_digest=basis.basis_digest,
                                head_event_id=pending.request_event_id,
                                correction_file=str(correction_file),
                            )
                        )
                        pending = corrected.pending_request
                        assert pending is not None
                        basis = pending.basis
                        self.assertEqual(
                            basis.supersedes_basis_digest,
                            finalization.bases[0].basis_digest,
                        )
                        self.assertEqual(basis.selected_work_item_ids, selected_ids)
                        excluded = next(
                            item
                            for item in basis.report_membership
                            if item.member_id == excluded_member_id
                        )
                        self.assertFalse(excluded.included)
                    approved = approve_finalization(
                        ApproveFinalizationCommand(
                            journey_file=str(journey.path),
                            request_event_id=pending.request_event_id,
                            basis_digest=basis.basis_digest,
                            head_event_id=pending.request_event_id,
                        )
                    )
                    report_basis = approved.report_basis
                    assert report_basis is not None
                    approval_head = Journey.open(journey.path).read()[-1]["event_id"]
                    published = publish_grounded_report(
                        PublishGroundedReportCommand(
                            journey_file=str(journey.path),
                            report_basis_digest=report_basis.report_basis_digest,
                            head_event_id=approval_head,
                        )
                    )
                    report = published.report
                    assert report is not None
                    self.assertEqual(
                        {item.value.work_item_id for item in report.work_items},
                        set(selected_ids),
                    )
                    self.assertEqual(report.selected_work_item_ids, selected_ids)
                    self.assertEqual(report.collection_status, "complete")
                    if excluded_member_id is not None:
                        excluded = next(
                            item
                            for item in report.work_items
                            if item.value.work_item_id == excluded_member_id
                        )
                        self.assertFalse(excluded.membership.included)
                    before = journey.path.read_bytes()

                    class _ForbiddenEvidence:
                        def materialization(self, **_kwargs):  # noqa: ANN003,ANN201
                            raise AssertionError("reopen contacted evidence")

                        def runtime(self, **_kwargs):  # noqa: ANN003,ANN201
                            raise AssertionError("reopen contacted runtime")

                    def forbidden_adapter():  # noqa: ANN202
                        raise AssertionError("reopen scheduled a Tool")

                    replayed_observation = investigation_observe(
                        observation_command,
                        adapter_factory=forbidden_adapter,
                        evidence_reader=_ForbiddenEvidence(),
                    )
                    reopened = reopen_grounded_report(
                        ReopenGroundedReportCommand(
                            journey_file=str(journey.path),
                            report_digest=report.report_digest,
                        )
                    )

                    self.assertEqual(replayed_observation, recorded)
                    self.assertEqual(reopened, published)
                    self.assertEqual(journey.path.read_bytes(), before)
                    self.assertEqual(report.report_external_contact_count, 0)

    def test_public_observer_refuses_terminal_member_state_not_bound_to_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (
                investigation_input,
                journey,
                authority,
                descriptor,
                command,
            ) = _approved_collection_observation(root)
            recorded = investigation_observe(
                command,
                adapter=_SequentialObservationAdapter(
                    command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            self.assertIsInstance(recorded, InvestigationCollectionEvidenceRecorded)
            assert isinstance(recorded, InvestigationCollectionEvidenceRecorded)
            timeline = Journey.open(journey.path).read()
            forged_states = (
                InvestigationCollectionItemState(
                    work_item_id=recorded.item_states[0].work_item_id,
                    selected_ordinal=1,
                    status=recorded.item_states[0].status,
                    evidence_state_id=recorded.item_states[0].evidence_state_id,
                    outcome="invented-terminal-outcome",
                ),
                *recorded.item_states[1:],
            )
            forged = fold_investigation_collection(
                command=command,
                authority=authority,
                item_states=forged_states,
            )
            lines = journey.path.read_text(encoding="utf-8").splitlines()
            journey.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            Journey.open(journey.path).record(
                forged.event_type,
                forged.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [timeline[-2]["event_id"]],
            )

            with self.assertRaises(ValueError):
                observe(ObserveRun(journey_file=str(journey.path)))
            with self.assertRaises(FinalizationApplicationRefused) as refused:
                compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
            self.assertEqual(refused.exception.code, "finalization_basis_invalid")

    def test_collection_observation_isolates_first_middle_last_refusal_and_replays_without_contact(
        self,
    ) -> None:
        for cardinality, refused_index in _REFUSAL_POSITIONS:
            with self.subTest(cardinality=cardinality, refused_index=refused_index):
                investigation_input, journey, authority, descriptor, command = (
                    self.approved_collection(cardinality=cardinality)
                )
                evidence = _Evidence(descriptor)
                adapter = _RefusingSequentialObservationAdapter(
                    command,
                    descriptor,
                    investigation_input.sample_id,
                    refused_query_index=refused_index,
                )

                recorded = investigation_observe(
                    command, adapter=adapter, evidence_reader=evidence
                )

                self.assertIsInstance(recorded, InvestigationCollectionEvidenceRecorded)
                assert isinstance(recorded, InvestigationCollectionEvidenceRecorded)
                self.assertEqual(recorded.combine_status, "partial")
                self.assertEqual(
                    tuple(item.status for item in recorded.item_states),
                    tuple(
                        "refused" if index == refused_index else "completed"
                        for index in range(cardinality)
                    ),
                )
                self.assertEqual(
                    recorded.selected_work_item_ids,
                    tuple(
                        item.work_item_id for item in authority.authorized_work_items
                    ),
                )
                queries = [
                    call
                    for call in adapter.calls
                    if call[1:4] == ("query", "--allow-undef-tags", "--regions-file")
                ]
                self.assertEqual(len(queries), cardinality)
                self.assertEqual(len(set(queries)), cardinality)
                lifecycle = [
                    event["event_type"]
                    for event in Journey.open(journey.path).read()
                    if event["event_type"]
                    in {
                        "InvestigationWorkItemStarted",
                        "InvestigationEvidenceRecorded",
                    }
                ]
                self.assertEqual(
                    lifecycle,
                    [
                        "InvestigationWorkItemStarted",
                        "InvestigationEvidenceRecorded",
                    ]
                    * cardinality,
                )
                projected = observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(projected.status, "investigation_evidence_ready")
                self.assertEqual(projected.investigation_evidence, recorded)
                run_table_reading(journey.path)
                finalization = compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                )
                self.assertEqual(finalization.status, "review_pending")
                before = journey.path.read_bytes()
                evidence_calls = tuple(evidence.calls)
                adapter_calls = tuple(adapter.calls)

                reopened = investigation_observe(
                    command, adapter=adapter, evidence_reader=evidence
                )

                self.assertEqual(reopened, recorded)
                self.assertEqual(journey.path.read_bytes(), before)
                self.assertEqual(tuple(evidence.calls), evidence_calls)
                self.assertEqual(tuple(adapter.calls), adapter_calls)

    def test_unknown_observation_failure_finalizes_actual_collection_outcomes(
        self,
    ) -> None:
        for cardinality, failed_index, failure_phase in _UNKNOWN_OBSERVATION_FAILURES:
            with self.subTest(
                cardinality=cardinality,
                failed_index=failed_index,
                failure_phase=failure_phase,
            ):
                investigation_input, journey, authority, descriptor, command = (
                    self.approved_collection(cardinality=cardinality)
                )

                class FailingAdapter(_SequentialObservationAdapter):
                    query_count = 0

                    def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001,ANN201
                        if argv[1:4] == (
                            "query",
                            "--allow-undef-tags",
                            "--regions-file",
                        ):
                            index = self.query_count
                            self.query_count += 1
                            if index == failed_index:  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                                self.calls.append(tuple(argv))
                                raise RuntimeError(
                                    "unclassified implementation failure"
                                )
                        return super().run(
                            argv,
                            timeout_seconds=timeout_seconds,
                            stdout_path=stdout_path,
                        )

                adapter = FailingAdapter(
                    command, descriptor, investigation_input.sample_id
                )

                class FailingEvidence(_Evidence):
                    def materialization(self, **kwargs):  # noqa: ANN003,ANN201
                        if failure_phase == "evidence":  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                            self.calls.append("materialization")
                            raise RuntimeError("unclassified evidence reader failure")
                        return super().materialization(**kwargs)

                evidence = FailingEvidence(descriptor)
                from bio_neb.failure import Invariant

                shared = failure_phase == "evidence"
                # The fault is retained on its item and holds the Run (R2).
                with self.assertRaises(Invariant) as held:
                    investigation_observe(
                        command, adapter=adapter, evidence_reader=evidence
                    )
                self.assertEqual(
                    held.exception.invariant_id,
                    "investigation_shared_observation_failed"
                    if shared
                    else "investigation_item_failed",
                )
                recorded = read_investigation(Journey.open(journey.path)).evidence
                # An item's own fault leaves its siblings to run (R2); a
                # shared preparation fault stops the batch at its first item.
                self.assertEqual(
                    tuple(item.status for item in recorded.item_states),
                    ("failed",) + ("not_started",) * (cardinality - 1)
                    if shared
                    else ("completed",) * failed_index
                    + ("failed",)
                    + ("completed",) * (cardinality - failed_index - 1),
                )
                self.assertEqual(
                    recorded.item_states[failed_index].reason,
                    "observation_exception:RuntimeError",
                )
                self.assertEqual(adapter.query_count, 0 if shared else cardinality)
                self.assertEqual(
                    recorded.selected_work_item_ids,
                    tuple(
                        item.work_item_id for item in authority.authorized_work_items
                    ),
                )
                reopened = observe(ObserveRun(journey_file=str(journey.path)))
                # A failed item is accounted, never an ended Run.
                self.assertEqual(reopened.status, "investigation_evidence_ready")
                self.assertEqual(reopened.investigation_evidence, recorded)
                before = journey.path.read_bytes()
                calls = (tuple(adapter.calls), tuple(evidence.calls))
                self.assertEqual(
                    investigation_observe(
                        command, adapter=adapter, evidence_reader=evidence
                    ),
                    recorded,
                )
                self.assertEqual(journey.path.read_bytes(), before)
                from bio_neb.application.advancement import ResumeRun, advance

                if shared and cardinality > 1:
                    # Members that never ran hold completion (the safeguard).
                    waiting = advance(ResumeRun(journey_file=str(journey.path)))
                    self.assertEqual(waiting.status, "investigation_evidence_ready")
                    self.assertEqual(waiting.finalization.status, "not_started")
                    self.assertIn(
                        "pending_members",
                        read_investigation(
                            Journey.open(journey.path)
                        ).work_policy.completion_blockers,
                    )
                    continue
                from bio_neb.application.advancement import (
                    use_run_advancement_dependencies,
                )

                # The siblings' readings run before Finalization (ADR 0047).
                with use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                    investigation_model_factory=TableReaderModel().model,
                    investigation_observation_adapter=adapter,
                    investigation_observation_evidence_reader=evidence,
                ):
                    ended = advance(ResumeRun(journey_file=str(journey.path)))
                    self.assertEqual(ended.status, "finalization_review_pending")
                    basis = ended.finalization.pending_request.basis
                    self.assertEqual(basis.outcome, "investigation_evidence_ready")
                    self.assertEqual(basis.collection_item_states, recorded.item_states)
                    self.assertEqual(ended.investigation_evidence, recorded)
                    frozen = journey.path.read_bytes()
                    calls = (tuple(adapter.calls), tuple(evidence.calls))
                    self.assertEqual(
                        observe(ObserveRun(journey_file=str(journey.path))), ended
                    )
                    self.assertEqual(
                        advance(ResumeRun(journey_file=str(journey.path))), ended
                    )
                    self.assertEqual(journey.path.read_bytes(), frozen)
                    self.assertEqual(
                        (tuple(adapter.calls), tuple(evidence.calls)), calls
                    )

    def test_shared_observation_sample_failure_retains_evidence_and_stops(
        self,
    ) -> None:
        for cardinality, failed_index, failure_kind in _SHARED_OBSERVATION_FAILURES:
            with self.subTest(
                cardinality=cardinality,
                failed_index=failed_index,
                failure_kind=failure_kind,
            ):
                investigation_input, journey, authority, descriptor, command = (
                    self.approved_collection(cardinality=cardinality)
                )

                class ChangedInputAdapter(_SequentialObservationAdapter):
                    queries = 0

                    def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001,ANN201
                        result = super().run(
                            argv,
                            timeout_seconds=timeout_seconds,
                            stdout_path=stdout_path,
                        )
                        if argv[1:4] == (
                            "query",
                            "--allow-undef-tags",
                            "--regions-file",
                        ):
                            self.queries += 1
                            if (
                                failure_kind == "sample"  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                                and self.queries == failed_index + 1  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                            ):
                                return CompletedProcess(
                                    0,
                                    result.stdout.replace(
                                        investigation_input.sample_id.encode(),  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                                        b"wrong-sample",
                                    ),
                                    b"",
                                )
                        return result

                adapter = ChangedInputAdapter(
                    command, descriptor, investigation_input.sample_id
                )
                evidence = _Evidence(descriptor)
                from bio_neb.failure import Invariant

                # A shared fault stops the batch at its item and holds the Run
                # (R2); the collection is retained first.
                with self.assertRaises(Invariant) as held:
                    investigation_observe(
                        command, adapter=adapter, evidence_reader=evidence
                    )
                self.assertEqual(
                    held.exception.invariant_id,
                    "investigation_shared_observation_failed",
                )
                recorded = read_investigation(Journey.open(journey.path)).evidence
                self.assertEqual(
                    tuple(item.status for item in recorded.item_states),
                    ("completed",) * failed_index
                    + ("failed",)
                    + ("not_started",) * (cardinality - failed_index - 1),
                )
                self.assertEqual(
                    recorded.item_states[failed_index].reason,
                    "shared_observation:tool_failure",
                )
                self.assertIsNotNone(
                    recorded.item_states[failed_index].evidence_state_id
                )
                self.assertEqual(adapter.queries, failed_index + 1)
                reopened = observe(ObserveRun(journey_file=str(journey.path)))
                # A failed item is accounted, never an ended Run.
                self.assertEqual(reopened.status, "investigation_evidence_ready")
                self.assertEqual(reopened.investigation_evidence, recorded)
                before = journey.path.read_bytes()
                calls = (tuple(adapter.calls), tuple(evidence.calls))
                self.assertEqual(
                    investigation_observe(
                        command, adapter=adapter, evidence_reader=evidence
                    ),
                    recorded,
                )
                self.assertEqual(journey.path.read_bytes(), before)
                if failed_index + 1 < cardinality:
                    # Members the fault left unstarted hold completion until
                    # the researcher chooses a partial report.
                    amend_to_partial(journey, "Report past the shared fault.")
                run_table_reading(journey.path)
                calls = (tuple(adapter.calls), tuple(evidence.calls))
                # Finalization accounts the failed member's limitation; its
                # retained failure evidence is accounted, and members never
                # started have no observation.
                basis = compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                ).pending_request.basis
                failed_item = recorded.item_states[failed_index].work_item_id
                self.assertTrue(
                    any(
                        failed_item in limitation
                        for limitation in _texts(basis.limitations)
                    )
                )
                self.assertEqual(
                    {item.work_item_id for item in basis.sample_observations},
                    {
                        state.work_item_id
                        for state in recorded.item_states[: failed_index + 1]
                    },
                )
                self.assertEqual((tuple(adapter.calls), tuple(evidence.calls)), calls)

    def test_stopped_collection_finalizes_and_reports_without_manufactured_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (
                _investigation_input,
                journey,
                authority,
                _descriptor,
                command,
            ) = _approved_collection_observation(root)
            stopped = fold_investigation_collection(
                command=command,
                authority=authority,
                item_states=tuple(
                    InvestigationCollectionItemState(
                        work_item_id=item.work_item_id,
                        selected_ordinal=index,
                        status="stopped" if index == 1 else "not_started",
                        reason="owner_stop" if index == 1 else None,
                    )
                    for index, item in enumerate(
                        authority.authorized_work_items, start=1
                    )
                ),
            )
            self.assertEqual(stopped.combine_status, "stopped_before_start")
            timeline = Journey.open(journey.path).read()
            Journey.open(journey.path).record(
                stopped.event_type,
                stopped.model_dump(
                    mode="json",
                    exclude={"event_type"},
                    exclude_computed_fields=True,
                ),
                [timeline[-1]["event_id"]],
            )

            from bio_neb.investigation.investigation_policy import (
                project_investigation_policy,
            )

            # The safeguard: members that never ran are named and hold
            # Finalization, and its table stage, until the researcher explicitly
            # chooses a partial report.
            timeline = read_evidence(Journey.open(journey.path))
            waiting = project_investigation_policy(
                timeline, project_investigation(timeline), stopped
            )
            self.assertEqual(waiting.completion_blockers, ("pending_members",))
            self.assertEqual(
                [
                    item.work_item_id
                    for item in waiting.items
                    if item.continuation != "retained"
                ],
                [item.work_item_id for item in authority.authorized_work_items],
            )
            self.assertIsNone(read_table_reading_stage(Journey.open(journey.path)))
            amend_to_partial(journey, "Report the stopped collection as it stands.")
            run_table_reading(journey.path)
            finalization = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            pending = finalization.pending_request
            assert pending is not None
            basis = pending.basis
            self.assertEqual(basis.collection_status, "stopped_before_start")
            self.assertEqual(len(basis.work_items), 3)
            self.assertEqual(basis.sample_observations, ())
            self.assertEqual(basis.candidate_evidence, ())
            self.assertEqual(basis.observation_assessments, ())
            self.assertEqual(basis.admissions, ())
            # Only the Journey's accounting is pointed at; nothing was observed.
            self.assertEqual(
                [
                    pointer.role
                    for pointer in basis.evidence_pointers
                    if not pointer.role.startswith("journey_accounting:")
                ],
                [],
            )
            self.assertTrue(
                all(
                    any(
                        item.work_item_id in limitation
                        for limitation in _texts(basis.limitations)
                    )
                    for item in stopped.item_states
                )
            )
            approved = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.request_event_id,
                    basis_digest=basis.basis_digest,
                    head_event_id=pending.request_event_id,
                )
            )
            report_basis = approved.report_basis
            assert report_basis is not None
            published = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            )
            report = published.report
            assert report is not None
            self.assertEqual(report.collection_status, "stopped_before_start")
            self.assertEqual(len(report.work_items), 3)
            self.assertEqual(report.sample_observations, ())
            self.assertEqual(report.evidence_pointers, basis.evidence_pointers)
            self.assertEqual(report.limitations, basis.limitations)

    def test_total_collection_fold_isolates_failure_negative_and_stop_states(
        self,
    ) -> None:
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/HG001.vcf.gz",
                byte_count=1234,
                sha256="a" * 64,
            ),
            sample_id="HG001",
            assembly_accession="GCF_000001405.40",
        )
        requests = tuple(
            InvestigationRequestItem(
                request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
            )
            for rsid in _SequentialAdaptiveModel.rsids
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=requests,
            predecessor_ids=(investigation_input.input_id,),
        )
        pairs = tuple(
            _source_pair(rsid, position)
            for rsid, position in zip(
                _SequentialAdaptiveModel.rsids,
                _SequentialAdaptiveModel.positions,
                strict=True,
            )
        )
        proposal = compile_investigation_collection_proposal(
            investigation_input=investigation_input,
            assertion=assertion,
            source_results=pairs,
            revision_number=1,
            supersedes_proposal_id=None,
            predecessor_ids=tuple(
                result.result_id for pair in pairs for result in pair
            ),
        )
        authority = InvestigationAuthority.create(
            proposal_id=proposal.proposal_id,
            review_response_id="investigation-review-response:" + "b" * 64,
            authorized_work_items=proposal.work_items,
            predecessor_ids=(
                proposal.proposal_id,
                "investigation-review-response:" + "b" * 64,
            ),
        )
        command = InvestigationObservationCommand(
            journey_file="/evidence/journey.jsonl",
            authority_id=authority.authority_id,
            vcf_file="C:/evidence/variants.vcf",
            materialization_receipt_file="C:/evidence/materialization.json",
            bcftools_executable="C:/tools/bcftools",
            bcftools_probe_receipt_file="C:/evidence/probe.json",
            derivation_root="C:/evidence/derived",
        )

        def retained_state(index: int, status: str):  # noqa: ANN202
            return InvestigationCollectionItemState(
                work_item_id=proposal.work_items[index].work_item_id,
                selected_ordinal=index + 1,
                status=status,
                evidence_state_id="investigation-evidence-state:" + str(index + 1) * 64,
                outcome=status,
            )

        for failed_index in range(3):
            states = tuple(
                InvestigationCollectionItemState(
                    work_item_id=item.work_item_id,
                    selected_ordinal=index + 1,
                    status="failed",
                    reason="no_progress",
                )
                if index == failed_index
                else retained_state(index, "completed")
                for index, item in enumerate(proposal.work_items)
            )
            folded = fold_investigation_collection(
                command=command, authority=authority, item_states=states
            )
            self.assertEqual(folded.combine_status, "partial")
            self.assertEqual(folded.item_states[failed_index].reason, "no_progress")
            self.assertEqual(
                tuple(item.work_item_id for item in folded.item_states),
                tuple(item.work_item_id for item in proposal.work_items),
            )

        truthful = fold_investigation_collection(
            command=command,
            authority=authority,
            item_states=(
                retained_state(0, "negative"),
                retained_state(1, "unresolved"),
                retained_state(2, "completed"),
            ),
        )
        self.assertEqual(truthful.combine_status, "complete")
        stopped_before = fold_investigation_collection(
            command=command,
            authority=authority,
            item_states=tuple(
                InvestigationCollectionItemState(
                    work_item_id=item.work_item_id,
                    selected_ordinal=index + 1,
                    status="stopped",
                    reason="owner_stop",
                )
                for index, item in enumerate(proposal.work_items)
            ),
        )
        self.assertEqual(stopped_before.combine_status, "stopped_before_start")
        stopped_after = fold_investigation_collection(
            command=command,
            authority=authority,
            item_states=(
                retained_state(0, "completed"),
                InvestigationCollectionItemState(
                    work_item_id=proposal.work_items[1].work_item_id,
                    selected_ordinal=2,
                    status="stopped",
                    reason="owner_stop",
                ),
                InvestigationCollectionItemState(
                    work_item_id=proposal.work_items[2].work_item_id,
                    selected_ordinal=3,
                    status="not_started",
                ),
            ),
        )
        self.assertEqual(stopped_after.combine_status, "stopped_after_partial")

    def test_collection_observation_prepares_once_and_folds_selected_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            investigation_input = InvestigationInput.create(
                vcf=InvestigationVcfIdentity.create(
                    bucket="accepted-run-bucket",
                    object_key="outputs/HG001.vcf.gz",
                    byte_count=1234,
                    sha256="a" * 64,
                ),
                sample_id="HG001",
                assembly_accession="GCF_000001405.40",
            )
            assertion = UserAssertion.create_collection(
                asserted_by="researcher:owner",
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid} in sample HG001.",
                        rsid=rsid,
                    )
                    for rsid in _SequentialAdaptiveModel.rsids
                ),
                predecessor_ids=(investigation_input.input_id,),
            )
            pending, journey = _run_attempt(
                root / "planning",
                _SequentialAdaptiveModel(),
                _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            assert pending.pending_request is not None
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=pending.pending_request.proposal.proposal_id,
                    ),
                )
            )
            authority = approved.authority
            assert authority is not None
            descriptor = VerifiedObject(
                role="VCF",
                bucket=investigation_input.vcf.bucket,
                object_key=investigation_input.vcf.object_key,
                byte_count=investigation_input.vcf.byte_count,
                sha256=investigation_input.vcf.sha256,
            )
            command = InvestigationObservationCommand(
                journey_file=str(journey.path),
                authority_id=authority.authority_id,
                vcf_file=str(root / "variants.vcf"),
                materialization_receipt_file=str(root / "materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )
            evidence = _Evidence(descriptor)
            adapter = _SequentialObservationAdapter(
                command, descriptor, investigation_input.sample_id
            )

            recorded = investigation_observe(
                command, adapter=adapter, evidence_reader=evidence
            )

            self.assertIsInstance(recorded, InvestigationCollectionEvidenceRecorded)
            self.assertEqual(recorded.combine_status, "complete")
            self.assertEqual(
                recorded.selected_work_item_ids,
                tuple(item.work_item_id for item in authority.authorized_work_items),
            )
            self.assertEqual(
                tuple(item.status for item in recorded.item_states),
                ("completed", "completed", "completed"),
            )
            self.assertEqual(evidence.calls, ["materialization", "runtime"])
            queries = [
                call
                for call in adapter.calls
                if call[1:4] == ("query", "--allow-undef-tags", "--regions-file")
            ]
            self.assertEqual(
                tuple(
                    "{}:{}-{}".format(*query_region(adapter.files, call))
                    for call in queries
                ),
                (
                    "NC_000019.10:44908684-44908684",
                    "NC_000019.10:44908822-44908822",
                    "NC_000019.10:21331549-21331549",
                ),
            )

    def test_collection_attempt_runs_agent_and_tools_in_selected_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            investigation_input = InvestigationInput.create(
                vcf=InvestigationVcfIdentity.create(
                    bucket="accepted-run-bucket",
                    object_key="outputs/HG001.vcf.gz",
                    byte_count=1234,
                    sha256="a" * 64,
                ),
                sample_id="HG001",
                assembly_accession="GCF_000001405.40",
            )
            requested_order = ("rs4149056", "rs429358", "rs7412")
            requests = tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in requested_order
            )
            assertion = UserAssertion.create_collection(
                asserted_by="researcher:owner",
                requests=requests,
                predecessor_ids=(investigation_input.input_id,),
            )
            model = _SequentialAdaptiveModel()
            adapter = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)

            projection, _journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                investigation_input=investigation_input,
                assertion=assertion,
            )

            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(model.calls, 9)
            self.assertEqual(
                adapter.operations,
                [
                    "fetch_refsnp:rs4149056",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertEqual(
                tuple(
                    item.target.aliases[0]
                    for item in projection.proposals[0].work_items
                ),
                requested_order,
            )

    def test_collection_request_retains_duplicate_assertions_before_normalization(
        self,
    ) -> None:
        from bio_neb.investigation.investigation_request import (
            read_investigation_request,
        )

        with tempfile.TemporaryDirectory() as temporary:
            request_file = Path(temporary).resolve() / "request.json"
            requests = [
                {
                    "request": "Investigate rs429358 in HG001.",
                    "target": {"rsid": "rs429358"},
                },
                {
                    "request": "Check rs429358 again in HG001.",
                    "target": {"rsid": "rs429358"},
                },
            ]
            request_file.write_text(
                json.dumps({"asserted_by": "researcher:owner", "requests": requests}),
                encoding="utf-8",
            )
            read = read_investigation_request(str(request_file))
            self.assertEqual(
                [item.request_text for item in read.items],
                [item["request"] for item in requests],
            )
            self.assertEqual(
                [item.rsid for item in read.items], ["rs429358", "rs429358"]
            )

    def test_public_begin_rejects_missing_adjacent_rsid_before_activation(
        self,
    ) -> None:
        from bio_neb.investigation.investigation_request import (
            InvestigationRequestRefused,
            read_investigation_request,
        )

        requests = (
            (
                "On GCF_000001405.40 / GRCh38.p14 in sample HG001, observe the "
                "exact allele NC_000001.11:14574 A>G in the named immutable "
                "variant-only VCF.",
                "rs28503599",
            ),
            (
                "On GCF_000001405.40 / GRCh38.p14 in sample HG001, observe the "
                "exact allele NC_000001.11:14677 G>A in the named immutable "
                "variant-only VCF.",
                "rs201327123",
            ),
            (
                "On GCF_000001405.40 / GRCh38.p14 in sample HG001, observe the "
                "exact allele NC_000001.11:15118 A>G in the named immutable "
                "variant-only VCF.",
                "rs11580262",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = Journey.create(root / "run")
            journey.record("test_anchor", {})
            continuation_file = root / "continuation.json"
            continuation_file.write_text("{}", encoding="utf-8")
            request_file = root / "investigation-request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.investigation-request/2",
                        "asserted_by": "researcher:owner",
                        "requests": [
                            {
                                "request": request_text,
                                "target": {"rsid": rsid},
                                "normalized_question_action": "observe_variant",
                            }
                            for request_text, rsid in requests
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            contacts: list[str] = []

            def forbidden_model(_loaded: object) -> object:
                contacts.append("model")
                raise AssertionError("invalid request contacted model")

            def forbidden_source() -> object:
                contacts.append("source")
                raise AssertionError("invalid request contacted source")

            before = journey.path.read_bytes()
            with self.assertRaises(InvestigationRequestRefused) as preflight_refused:
                read_investigation_request(str(request_file))

            self.assertEqual(
                preflight_refused.exception.code,
                "investigation_request_missing_adjacent_rsid",
            )
            self.assertEqual(
                preflight_refused.exception.fields,
                (
                    "request_file.requests[0].request",
                    "request_file.requests[1].request",
                    "request_file.requests[2].request",
                ),
            )
            with (
                patch(
                    "bio_neb.investigation.investigation_commands._read_continuation",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "bio_neb.investigation.investigation_commands."
                    "_validated_continuation_timeline",
                    side_effect=InvestigationApplicationRefused(
                        "invalid_request_crossed_readiness"
                    ),
                ) as validate_continuation,
                self.assertRaises(InvestigationApplicationRefused) as refused,
            ):
                begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(journey.path),
                        scientific_continuation_file=str(continuation_file),
                        request_files=(str(request_file),),
                    ),
                    environ={},
                    model_factory=forbidden_model,
                    source_adapter_factory=forbidden_source,
                )

            self.assertEqual(
                refused.exception.code,
                "investigation_request_missing_adjacent_rsid",
            )
            self.assertEqual(
                refused.exception.fields,
                (
                    "request_file.requests[0].request",
                    "request_file.requests[1].request",
                    "request_file.requests[2].request",
                ),
            )
            self.assertEqual(contacts, [])
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(
                project_investigation(read_evidence(journey)).status,
                "not_started",
            )
            validate_continuation.assert_not_called()

    def test_public_begin_accepts_corrected_three_rsid_request(self) -> None:
        from bio_neb.investigation.investigation_request import (
            read_investigation_request,
        )

        requests = (
            (
                "For rs28503599 on GCF_000001405.40 / GRCh38.p14 in sample "
                "HG001, observe the exact allele NC_000001.11:14574 A>G in the "
                "named immutable variant-only VCF.",
                "rs28503599",
            ),
            (
                "For rs201327123 on GCF_000001405.40 / GRCh38.p14 in sample "
                "HG001, observe the exact allele NC_000001.11:14677 G>A in the "
                "named immutable variant-only VCF.",
                "rs201327123",
            ),
            (
                "For rs11580262 on GCF_000001405.40 / GRCh38.p14 in sample "
                "HG001, observe the exact allele NC_000001.11:15118 A>G in the "
                "named immutable variant-only VCF.",
                "rs11580262",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey = Journey.create(root / "run")
            journey.record("test_anchor", {})
            continuation_file = root / "continuation.json"
            continuation_file.write_text("{}", encoding="utf-8")
            request_file = root / "investigation-request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "schema_version": "bio-neb.investigation-request/2",
                        "asserted_by": "researcher:owner",
                        "requests": [
                            {
                                "request": request_text,
                                "target": {"rsid": rsid},
                                "normalized_question_action": "observe_variant",
                            }
                            for request_text, rsid in requests
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            contacts: list[str] = []

            def forbidden_model(_loaded: object) -> object:
                contacts.append("model")
                raise AssertionError("accepted boundary contacted model")

            def forbidden_source() -> object:
                contacts.append("source")
                raise AssertionError("accepted boundary contacted source")

            before = journey.path.read_bytes()
            ready_request = read_investigation_request(str(request_file))
            self.assertEqual(ready_request.asserted_by, "researcher:owner")
            self.assertEqual(
                tuple((item.request_text, item.rsid) for item in ready_request.items),
                requests,
            )
            with (
                patch(
                    "bio_neb.investigation.investigation_commands._read_continuation",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "bio_neb.investigation.investigation_commands."
                    "_validated_continuation_timeline",
                    side_effect=InvestigationApplicationRefused(
                        "corrected_request_reached_accepted_begin"
                    ),
                ) as validate_continuation,
                self.assertRaises(InvestigationApplicationRefused) as accepted,
            ):
                begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(journey.path),
                        scientific_continuation_file=str(continuation_file),
                        request_files=(str(request_file),),
                    ),
                    environ={},
                    model_factory=forbidden_model,
                    source_adapter_factory=forbidden_source,
                )

            self.assertEqual(
                accepted.exception.code,
                "corrected_request_reached_accepted_begin",
            )
            self.assertEqual(contacts, [])
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(
                project_investigation(read_evidence(journey)).status,
                "not_started",
            )
            validate_continuation.assert_called_once()

    # Its message and Tool order assertions read members in request order.
    @members_one_at_a_time()
    def test_public_begin_accepts_collection_json_at_cardinalities_one_and_three(
        self,
    ) -> None:
        request_texts = (
            (
                "From the already immutable HG001 VCF on GCF_000001405.40, "
                "investigate rs429358; verify it through the declared lawful sources, "
                "propose the exact one-target Investigation for review before any VCF "
                "access, and after both reviews render a deterministic report without "
                "inferring coverage, reference genotype, biological absence, function, "
                "causality, or diagnosis."
            ),
            "Investigate rs7412 in sample HG001 on GRCh38.",
            "Investigate rs4149056 in sample HG001 on GRCh38.",
        )
        rsids = ("rs429358", "rs7412", "rs4149056")

        # One request and several: the begin path has no branch between two and
        # three members, and the compiler owns order across 1, 2 and 3.
        for cardinality in (1, 3):
            with (
                self.subTest(cardinality=cardinality),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                request = {
                    "schema_version": "bio-neb.investigation-request/2",
                    "asserted_by": "researcher:owner",
                    "requests": [
                        {
                            "request": request_text,
                            "target": {"rsid": rsid},
                            "normalized_question_action": "observe_variant",
                        }
                        for request_text, rsid in zip(
                            request_texts[:cardinality],
                            rsids[:cardinality],
                            strict=True,
                        )
                    ],
                }
                journey = forked_child(
                    root,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=request
                    ).journey_file,
                )
                request_file = root / "investigation-request.json"
                request_file.write_text(
                    json.dumps(request, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                model = _SequentialAdaptiveModel()
                adapter = _SequentialSourceResponses(assembly_body=ASSEMBLY_BODY)

                projection = begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    ),
                    environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                    model_factory=model.model,  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                    source_adapter_factory=lambda: adapter,  # noqa: B023 - Existing fixture scenario; model identity change preserves its execution.
                )

                self.assertEqual(projection.status, "review_pending")
                self.assertEqual(model.calls, 3 * cardinality)
                # Members run concurrently: each fetches its own sources once,
                # in whatever order the members interleave.
                self.assertEqual(
                    sorted(adapter.operations),
                    sorted(
                        operation
                        for rsid in rsids[:cardinality]
                        for operation in (
                            f"fetch_refsnp:{rsid}",
                            "fetch_assembly_report:GCF_000001405.40",
                        )
                    ),
                )
                self.assertEqual(
                    tuple(
                        item.target.aliases[0]
                        for item in projection.proposals[0].work_items
                    ),
                    rsids[:cardinality],
                )
                for item_index in range(cardinality):
                    self.assertIn(
                        "refsnp-result:", model.visible_messages[3 * item_index + 1]
                    )
                    self.assertIn(
                        "assembly-report-result:",
                        model.visible_messages[3 * item_index + 2],
                    )

                # Tool starts and terminals are Contact-journal records; the
                # semantic Journey alone holds none of them (ADR 0033). The Fork
                # child's inherited planning contacts keep the parent's identity.
                events = [
                    event
                    for event in read_evidence(Journey.open(journey.path))
                    if event["journey_id"] == journey.journey_id
                ]
                starts = [
                    event
                    for event in events
                    if event["event_type"] == "ObservedToolExecutionStarted"
                ]
                terminals = [
                    event
                    for event in events
                    if event["event_type"] == "ObservedToolExecutionSucceeded"
                ]
                self.assertEqual(len(starts), 2 * cardinality)
                self.assertEqual(len(terminals), 2 * cardinality)
                self.assertEqual(
                    tuple(event["payload"]["tool_name"] for event in starts),
                    tuple(
                        tool_name
                        for _ in range(cardinality)
                        for tool_name in ("fetch_refsnp", "fetch_assembly_report")
                    ),
                )
                self.assertEqual(
                    {event["payload"]["started_event_id"] for event in terminals},
                    {event["event_id"] for event in starts},
                )
                self.assertEqual(
                    sum(
                        event["event_type"] == "RefSnpToolResultRecorded"
                        for event in events
                    ),
                    cardinality,
                )
                self.assertEqual(
                    sum(
                        event["event_type"] == "AssemblyReportToolResultRecorded"
                        for event in events
                    ),
                    cardinality,
                )
                self.assertEqual(
                    sum(
                        event["event_type"] == "InvestigationReviewRequested"
                        for event in events
                    ),
                    1,
                )
                current = observe(ObserveRun(journey_file=str(journey.path)))
                rendered_current = current.model_dump_json()
                self.assertNotIn("variant_target_planning", rendered_current)
                self.assertNotIn("variant_target_review_pending", rendered_current)
                self.assertEqual(current.investigation.status, "review_pending")
                self.assertEqual(
                    current.investigation.investigation_input,
                    projection.investigation_input,
                )

    def test_public_four_item_forked_collection_reaches_authority_and_report(
        self,
    ) -> None:
        requests = (
            ("Investigate rs429358 in sample HG001 on GRCh38.", "rs429358"),
            ("Investigate rs7412 in sample HG001 on GRCh38.", "rs7412"),
            ("Investigate rs4149056 in sample HG001 on GRCh38.", "rs4149056"),
            ("Investigate rs28503599 in sample HG001 on GRCh38.", "rs28503599"),
        )

        class FourItemSourceResponses(_SequentialSourceResponses):
            positions = {
                **_SequentialSourceResponses.positions,
                "rs28503599": 14574,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": request_text,
                        "target": {"rsid": rsid},
                        "normalized_question_action": "observe_variant",
                    }
                    for request_text, rsid in requests
                ],
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(
                json.dumps(request, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            model = _SequentialAdaptiveModel()
            adapter = FourItemSourceResponses(assembly_body=ASSEMBLY_BODY)

            pending = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                model_factory=model.model,
                source_adapter_factory=lambda: adapter,
            )

            self.assertEqual(pending.status, "review_pending")
            assert pending.pending_request is not None
            approved = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=pending.pending_request.proposal.proposal_id,
                    ),
                )
            )

            expected_rsids = tuple(rsid for _, rsid in requests)
            self.assertEqual(approved.status, "approved")
            assert approved.authority is not None
            self.assertEqual(
                approved.authority.authorized_work_items,
                pending.pending_request.proposal.work_items,
            )
            self.assertEqual(
                tuple(
                    item.target.aliases[0]
                    for item in approved.authority.authorized_work_items
                ),
                expected_rsids,
            )
            self.assertEqual(model.calls, 3 * len(requests))
            self.assertEqual(adapter.contact_count, 2 * len(requests))
            reopened = project_investigation(read_evidence(Journey.open(journey.path)))
            self.assertEqual(reopened.authority, approved.authority)
            # The Run's failed turns, else the role default's, from the
            # runtime the Fork child inherits from its parent's registration.
            runtime_file = Path(
                next(
                    event["payload"]["runtime_config_file"]
                    for event in Journey.open(journey.path).read()
                    if event["event_type"] == "RunInputsRegistered"
                )
            )
            from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

            selected = RuntimeConfiguration.model_validate_json(
                runtime_file.read_text()
            ).model_dump(mode="json")
            continuation = reopened.attempts[0].continuation_policy.model_dump(
                mode="json"
            )
            from bio_neb.agentic_journey.role_declaration import ROLE_DEFAULT

            self.assertEqual(
                continuation,
                {
                    "failed_turns": selected["agent_invocation"].get(
                        "failed_turns", ROLE_DEFAULT.failed_turns
                    )
                },
            )
            self.assertNotIn("source_contact_ceiling", continuation)
            investigation_input = approved.investigation_input
            assert investigation_input is not None
            descriptor = VerifiedObject(
                role="VCF",
                bucket=investigation_input.vcf.bucket,
                object_key=investigation_input.vcf.object_key,
                byte_count=investigation_input.vcf.byte_count,
                sha256=investigation_input.vcf.sha256,
            )
            observation_command = InvestigationObservationCommand(
                journey_file=str(journey.path),
                authority_id=approved.authority.authority_id,
                vcf_file=str(root / "variants.vcf"),
                materialization_receipt_file=str(root / "materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )
            observed = investigation_observe(
                observation_command,
                adapter=_SequentialObservationAdapter(
                    observation_command, descriptor, investigation_input.sample_id
                ),
                evidence_reader=_Evidence(descriptor),
            )
            projected = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(projected.investigation_evidence, observed)
            run_table_reading(journey.path)
            finalization = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )
            pending_finalization = finalization.pending_request
            assert pending_finalization is not None
            self.assertEqual(
                pending_finalization.basis.selected_work_item_ids,
                tuple(
                    item.work_item_id
                    for item in approved.authority.authorized_work_items
                ),
            )
            finalization_approved = approve_finalization(
                ApproveFinalizationCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending_finalization.request_event_id,
                    basis_digest=pending_finalization.basis.basis_digest,
                    head_event_id=pending_finalization.request_event_id,
                )
            )
            report_basis = finalization_approved.report_basis
            assert report_basis is not None
            published = publish_grounded_report(
                PublishGroundedReportCommand(
                    journey_file=str(journey.path),
                    report_basis_digest=report_basis.report_basis_digest,
                    head_event_id=Journey.open(journey.path).read()[-1]["event_id"],
                )
            )
            report = published.report
            assert report is not None
            self.assertEqual(len(report.work_items), 4)
            self.assertEqual(len(report.sample_observations), 4)
            # On a forked child the parent's inherited contacts are not the
            # report's; membership and contact-free reopen are pinned by
            # test_collection_finalization_report_preserves_membership_status_and_contact_free_reopen.
            self.assertEqual(report.report_external_contact_count, 0)

    def test_semantic_identity_excludes_sample_source_and_preserves_duplicate_assertions(
        self,
    ) -> None:
        target = _target()
        first = WorkItem.create(
            sample_id="HG001",
            target=target,
            intended_action="observe_variant",
            source_ids=(
                "retrieved-source:" + "3" * 64,
                "retrieved-source:" + "6" * 64,
            ),
        )
        repeated_semantics = WorkItem.create(
            sample_id="HG002",
            target=target,
            intended_action="observe_variant",
            source_ids=(
                "retrieved-source:" + "4" * 64,
                "retrieved-source:" + "7" * 64,
            ),
        )

        self.assertEqual(first.work_item_id, repeated_semantics.work_item_id)
        self.assertNotEqual(
            first.work_item_id,
            WorkItem.create(
                sample_id="HG001",
                target=_target(position=44908685),
                intended_action="observe_variant",
                source_ids=(
                    "retrieved-source:" + "3" * 64,
                    "retrieved-source:" + "6" * 64,
                ),
            ).work_item_id,
        )
        legacy_material = first.model_dump(
            mode="json",
            exclude={"work_item_id", "revision_id", "previous_work_item_id"},
        )
        legacy_payload = first.model_dump(mode="python", exclude={"work_item_id"})
        legacy = WorkItem(
            work_item_id="work-item:"
            + hashlib.sha256(
                canonical_json(legacy_material).encode("utf-8")
            ).hexdigest(),
            **legacy_payload,
        )
        proposal_material = {
            "revision_number": 1,
            "investigation_input_id": "investigation-input:" + "5" * 64,
            "assertion_id": "user-assertion:" + "6" * 64,
            "work_items": (legacy, repeated_semantics),
            "unresolved_conflicts": (),
            "completeness": "complete",
            "supersedes_proposal_id": None,
            "predecessor_ids": (
                "refsnp-result:" + "7" * 64,
                "assembly-report-result:" + "8" * 64,
            ),
        }
        unvalidated = InvestigationProposal.model_construct(
            proposal_id="investigation-proposal:" + "0" * 64,
            **proposal_material,
        )
        proposal_id = (
            "investigation-proposal:"
            + hashlib.sha256(
                canonical_json(
                    unvalidated.model_dump(mode="json", exclude={"proposal_id"})
                ).encode("utf-8")
            ).hexdigest()
        )
        with self.assertRaisesRegex(ValueError, "duplicate semantic Work item"):
            InvestigationProposal(proposal_id=proposal_id, **proposal_material)
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=(
                InvestigationRequestItem(
                    request_text="Investigate rs429358 in sample HG001.",
                    rsid="rs429358",
                ),
                InvestigationRequestItem(
                    request_text="Check rs429358 again for sample HG001.",
                    rsid="rs429358",
                ),
            ),
            predecessor_ids=("investigation-input:" + "5" * 64,),
        )
        self.assertEqual(len(assertion.requests), 2)

    def test_collection_proposal_preserves_one_two_three_request_order(self) -> None:
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/HG001.vcf.gz",
                byte_count=1234,
                sha256="a" * 64,
            ),
            sample_id="HG001",
            assembly_accession="GCF_000001405.40",
        )
        requests = tuple(
            InvestigationRequestItem(
                request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
            )
            for rsid in ("rs429358", "rs7412", "rs4149056")
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=requests,
            predecessor_ids=(investigation_input.input_id,),
        )
        pairs = tuple(
            _source_pair(rsid, position)
            for rsid, position in zip(
                ("rs429358", "rs7412", "rs4149056"),
                (44908684, 44908822, 21331549),
                strict=True,
            )
        )

        for cardinality in (1, 2, 3):
            proposal = compile_investigation_collection_proposal(
                investigation_input=investigation_input,
                assertion=UserAssertion.create_collection(
                    asserted_by=assertion.asserted_by,
                    requests=requests[:cardinality],
                    predecessor_ids=assertion.predecessor_ids,
                ),
                source_results=pairs[:cardinality],
                revision_number=1,
                supersedes_proposal_id=None,
                predecessor_ids=tuple(
                    result.result_id for pair in pairs[:cardinality] for result in pair
                ),
            )
            self.assertEqual(
                tuple(item.target.aliases[0] for item in proposal.work_items),
                tuple(item.rsid for item in requests[:cardinality]),
            )
            self.assertEqual(len(proposal.work_items), len(set(proposal.work_items)))

        collision_pairs = (
            _source_pair("rs429358", 44908684),
            _source_pair("rs7412", 44908684),
        )
        deduplicated = compile_investigation_collection_proposal(
            investigation_input=investigation_input,
            assertion=UserAssertion.create_collection(
                asserted_by=assertion.asserted_by,
                requests=requests[:2],
                predecessor_ids=assertion.predecessor_ids,
            ),
            source_results=collision_pairs,
            revision_number=1,
            supersedes_proposal_id=None,
            predecessor_ids=tuple(
                result.result_id for pair in collision_pairs for result in pair
            ),
        )
        self.assertEqual(len(deduplicated.work_items), 1)
        self.assertEqual(len(deduplicated.request_members), 2)
        self.assertEqual(
            deduplicated.request_members[0].work_item_ids,
            deduplicated.request_members[1].work_item_ids,
        )


if __name__ == "__main__":
    unittest.main()


def _assert_investigation_read_snapshot(test):
    from bio_neb.application import investigation_read
    from bio_neb.application.queries import (
        read_completion,
        read_investigation,
    )
    from tests.finalization_fakes import evidence_journey

    with tempfile.TemporaryDirectory() as temporary:
        journey = evidence_journey(
            Path(temporary).resolve(),
            parent_file=test.prerequisite(
                "executed", investigation_request=EVIDENCE_REQUEST
            ).journey_file,
        )
        before = journey.path.read_bytes()
        with patch.object(
            investigation_read,
            "project_investigation_evidence",
            wraps=investigation_read.project_investigation_evidence,
        ) as assembled:
            first = read_investigation(journey)
            completion = read_completion(journey)
            repeated = read_completion(journey)
            test.assertEqual(assembled.call_count, 1)
            test.assertEqual(completion, repeated)
        candidate = completion.candidate
        test.assertIsNotNone(candidate)
        test.assertIsNone(completion.approved_report)
        test.assertEqual(
            candidate.source_head_event["event_id"], first.prefix_event_ids[-1]
        )
        # The collection finalizes from its item evidence fold, the head (ADR 0047).
        test.assertEqual(
            candidate.source_head_event["event_type"],
            "InvestigationItemEvidenceCollected",
        )
        test.assertEqual(
            candidate.collection.event["event_id"],
            first.collections[-1].event["event_id"],
        )
        member = first.active_members[0].recorded
        test.assertIsNotNone(member)
        test.assertIn(member.event["event_id"], first.prefix_event_ids)
        test.assertEqual(
            member.model_contact_count,
            member.evidence.evidence_state.inherited_model_contact_count,
        )
        test.assertEqual(
            member.source_contact_count,
            member.evidence.evidence_state.inherited_source_contact_count,
        )
        original_authority = member.evidence.authority
        member.event["payload"].clear()
        member.evidence.__dict__["authority"] = None
        candidate.collection.event["payload"].clear()
        fresh = read_completion(journey)
        test.assertEqual(
            fresh.investigation.recorded_members[0].evidence.authority,
            original_authority,
        )
        test.assertTrue(fresh.candidate.collection.event["payload"])
        test.assertEqual(journey.path.read_bytes(), before)
        compiled = compile_finalization(
            CompileFinalizationCommand(journey_file=str(journey.path)),
            journey=journey,
        )
        retained = read_completion(journey)
        test.assertIsNone(retained.candidate)
        test.assertEqual(retained.finalization.bases, compiled.bases)
        test.assertEqual(
            retained.finalization.bases[0].source_head_event_id,
            first.prefix_event_ids[-1],
        )
        # A later unrelated family changes the present head, never the retained basis.
        journey.record(
            "DiagnosticNoteRecorded",
            {"note": "later"},
            [journey.prefix_event_ids[-1]],
        )
        later = read_completion(journey)
        test.assertIsNone(later.candidate)
        test.assertEqual(later.finalization.bases, compiled.bases)
        test.assertNotEqual(
            later.investigation.prefix_event_ids, first.prefix_event_ids
        )
