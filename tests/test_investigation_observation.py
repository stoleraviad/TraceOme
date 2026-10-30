from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
)

from bio_neb.application.fork import own_fork_origin
from bio_neb.application.queries import ObserveRun, observe
from bio_neb.investigation.investigation import (
    ApproveInvestigationResponse,
    InvestigationAuthority,
    VariantTarget,
    WorkItem,
)
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    RespondInvestigationReviewCommand,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationObservationCommand,
    SampleObservation,
    assess_investigation_observation,
    validate_investigation_evidence_fact,
)
from bio_neb.investigation.investigation_observation_adapter import (
    DirectInvestigationObservationEvidence,
)
from bio_neb.investigation.investigation_observation_commands import (
    InvestigationObservationApplicationRefused,
    MaterializationObservationEvidence,
    RuntimeProbeObservationEvidence,
    investigation_observe,
)
from bio_neb.investigation.vcf_materialization import (
    VcfMaterializationFile,
    VcfMaterializationLineage,
    VcfMaterializationReceipt,
)
from bio_neb.investigation.vcf_observation import (
    AmbiguousRecordsObserved,
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
    CompletedProcess,
    ExactAlleleObserved,
    LocalFileObservation,
    LocusIdentityMismatch,
    LocusOverlapped,
    NoRecordObserved,
    ObserveVariantInput,
    TargetAlleleNotCarried,
    TargetGenotypeUnresolved,
    VariantObservationRefused,
    VcfPreparationFailed,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.result import read_completed_parabricks_result
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from bio_neb.verified_artifacts import VerifiedObject
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.prerequisite_builds import measured_build
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import begin_in_pass, source_basis
from tests.terminal_journey_fakes import forked_child
from tests.vcf_lookup_fakes import lookup_command, query_row, vcf_header

REFSNP_BODY = canonical_json(
    {
        "refsnp_id": "429358",
        "primary_snapshot_data": {
            "placements_with_allele": [
                {
                    "seq_id": "NC_000019.10",
                    "is_ptlp": True,
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "assembly_name": "GRCh38.p14",
                            }
                        ]
                    },
                    "alleles": [
                        {
                            "allele": {
                                "spdi": {
                                    "seq_id": "NC_000019.10",
                                    "position": 44908683,
                                    "deleted_sequence": "T",
                                    "inserted_sequence": base,
                                }
                            }
                        }
                        for base in ("T", "C")
                    ],
                }
            ]
        },
    }
).encode()
ASSEMBLY_BODY = canonical_json(
    {
        "reports": [
            {
                "accession": "GCF_000001405.40",
                "assembly_info": {
                    "assembly_name": "GRCh38.p14",
                    "assembly_status": "current",
                },
            }
        ]
    }
).encode()


class _Planner:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        visible = ModelMessagesTypeAdapter.dump_json(messages).decode()
        self.calls += 1
        if '"cite":{"refsnp_event_id"' not in visible:
            name, arguments = "fetch_refsnp", {"rsid": "rs429358"}
        elif '"cite":{"assembly_event_id"' not in visible:
            name, arguments = (
                "fetch_assembly_report",
                {
                    "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                    "assembly_accession": "GCF_000001405.40",
                },
            )
        else:
            name = info.output_tools[0].name
            arguments = {
                **source_basis(info),
                "outcome": "complete",
            }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    canonical_json(arguments),
                    tool_call_id=f"observation-seed-{self.calls}",
                )
            ]
        )


class _Sources:
    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
                + rsid.removeprefix("rs")
            ),
            status_code=200,
            body=REFSNP_BODY,
            observed_at="2026-08-27T00:00:00Z",
            elapsed_time_ms=1,
        )

    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                f"{assembly_accession}/dataset_report"
            ),
            status_code=200,
            body=ASSEMBLY_BODY,
            observed_at="2026-08-27T00:00:01Z",
            elapsed_time_ms=1,
        )


# The approved investigation's request; its executed parent registers it.
OBSERVATION_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "request": "Investigate rs429358 in sample HG001.",
    "target": {"rsid": "rs429358"},
}


@measured_build("investigation-observation:approved-investigation")
def _approved_journey(
    root: Path, *, parent_file: Path | None = None
) -> tuple[Journey, object]:
    journey = forked_child(
        root,
        parent_file=parent_file,
        investigation_request=None if parent_file else OBSERVATION_REQUEST,
    )
    request_file = root / "investigation-request.json"
    request_file.write_text(canonical_json(OBSERVATION_REQUEST), encoding="utf-8")
    planner = _Planner()
    pending = begin_in_pass(
        BeginInvestigationCommand(
            journey_file=str(journey.path),
            request_files=(str(request_file),),
        ),
        environ={"LOCAL_MODEL_KEY": "not-retained"},
        model_factory=planner.model,
        source_adapter_factory=_Sources,
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
    assert approved.authority is not None
    return Journey.open(journey.path), approved


def _file(path: str, sha256: str, *, executable: bool = False, size: int = 10):
    return LocalFileObservation(
        path=path,
        exists=True,
        is_regular=True,
        is_symlink=False,
        is_writable=False,
        is_executable=executable,
        byte_count=size,
        sha256=sha256,
        device="1",
        inode=path,
        modified_at_ns=1,
    )


class _Evidence:
    def __init__(self, descriptor: VerifiedObject) -> None:
        self.descriptor = descriptor
        self.calls: list[str] = []

    def materialization(self, *, receipt_file: str, vcf_file: str):
        self.calls.append("materialization")
        receipt = _file(receipt_file, "6" * 64, size=100)
        return MaterializationObservationEvidence(
            receipt_id="vcf-materialization-receipt:" + "7" * 64,
            descriptor=self.descriptor,
            attested_vcf=AttestedPlainVcf(
                local_path=vcf_file,
                byte_count=self.descriptor.byte_count,
                sha256=self.descriptor.sha256,
                acquisition_receipt_path=receipt_file,
                acquisition_receipt_sha256=receipt.sha256,
            ),
            receipt_file=receipt,
        )

    def runtime(self, *, receipt_file: str, executable: str):
        self.calls.append("runtime")
        receipt = _file(receipt_file, "8" * 64, size=100)
        return RuntimeProbeObservationEvidence(
            runtime=BcftoolsRuntimeBinding(
                executable_path=executable,
                executable_sha256="9" * 64,
                bcftools_version="1.24",
                htslib_version="1.24",
                probe_receipt_path=receipt_file,
                probe_receipt_sha256=receipt.sha256,
            ),
            receipt_file=receipt,
        )


class _Adapter:
    def __init__(
        self, command: InvestigationObservationCommand, descriptor, sample_id: str
    ) -> None:
        self.command = command
        self.descriptor = descriptor
        self.sample_id = sample_id
        self.calls: list[tuple[str, ...]] = []
        self.files = {
            command.vcf_file: _file(
                command.vcf_file, descriptor.sha256, size=descriptor.byte_count
            ),
            command.materialization_receipt_file: _file(
                command.materialization_receipt_file, "6" * 64, size=100
            ),
            command.bcftools_executable: _file(
                command.bcftools_executable, "9" * 64, executable=True, size=100
            ),
            command.bcftools_probe_receipt_file: _file(
                command.bcftools_probe_receipt_file, "8" * 64, size=100
            ),
        }

    def inspect_file(self, path):
        return copy.deepcopy(
            self.files.get(
                path,
                LocalFileObservation(
                    path=path,
                    exists=False,
                    is_regular=False,
                    is_symlink=False,
                    is_writable=False,
                    is_executable=False,
                    byte_count=None,
                    sha256=None,
                    device=None,
                    inode=None,
                    modified_at_ns=None,
                ),
            )
        )

    def path_exists(self, path):
        return path in self.files

    def create_fresh_directory(self, path):
        if self.path_exists(path):
            raise FileExistsError(path)

    def write_fresh_file(self, path, data):
        # A query's regions file: kept as written, never inspected.
        if self.path_exists(path):
            raise FileExistsError(path)
        self.files[path] = data

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        del timeout_seconds
        self.calls.append(tuple(argv))
        bgzf = str(Path(self.command.derivation_root) / "variants.vcf.gz")
        step = lookup_command(argv)
        if step == "bgzf_derivation":
            assert stdout_path == bgzf
            self.files[bgzf] = _file(bgzf, "a" * 64, size=50)
            return CompletedProcess(0, b"", b"")
        if step == "csi_derivation":
            self.files[bgzf + ".csi"] = _file(bgzf + ".csi", "b" * 64, size=20)
            return CompletedProcess(0, b"", b"")
        if step == "header_inventory":
            return CompletedProcess(0, vcf_header(self.sample_id), b"")
        return CompletedProcess(
            0,
            query_row("NC_000019.10", 44908684, "T", "C", self.sample_id, "0/1"),
            b"",
        )


class InvestigationObservationApplicationTests(RunPrerequisites, unittest.TestCase):
    def test_only_a_runtime_audit_selection_records_a_full_read_of_the_runs_vcf(
        self,
    ) -> None:
        # Migrated from the retired investigation-observe CLI test: the
        # verification policy comes from the Run's registered selections, not a
        # flag or its runtime file. Ordinary use reads none of the Run's VCF; a
        # selected audit reads it once and the preparation records that read.
        # Each case Forks a parent that registered its own selection.

        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_investigation
        from tests.observation_input_fakes import (
            RUN_VCF,
            TABLE_READING_ENVIRON,
            RunOutputStore,
        )
        from tests.sequential_investigation_support import (
            approved_collection_observation,
            collection_request,
        )
        from tests.vcf_lookup_fakes import TableReaderModel

        # Each parent registers its own selection; the child resumes on its
        # parent's registered selections.
        for verification, expected_mode in (
            (None, "reuse_creation_proof"),
            (
                {"mode": "independent_full_read", "reason": "explicit runtime audit"},
                "independent_full_read",
            ),
        ):
            parent = self.prerequisite(
                "executed",
                investigation_request=collection_request(3),
                output_payloads={"VCF": RUN_VCF},
                **(
                    {}
                    if verification is None
                    else {"artifact_verification": verification}
                ),
            ).journey_file
            with (
                self.subTest(mode=expected_mode),
                tempfile.TemporaryDirectory() as temporary,
            ):
                _, journey, first, _, _ = approved_collection_observation(
                    Path(temporary).resolve(),
                    cardinality=3,
                    approve_count=1,
                    forked=True,
                    parent_file=parent,
                )
                with (
                    use_run_advancement_dependencies(
                        environ=TABLE_READING_ENVIRON,
                        investigation_model_factory=TableReaderModel().model,
                        investigation_vcf_source_factory=RunOutputStore(),
                    ),
                ):
                    advance(ResumeRun(journey_file=str(journey.path)))
                read = read_investigation(Journey.open(journey.path))
                command = read.collections[-1].evidence.command
                receipt = VcfMaterializationReceipt.model_validate_json(
                    Path(command.materialization_receipt_file).read_bytes()
                )
                self.assertEqual(
                    receipt.verification_policy.effective.mode, expected_mode
                )
                member = next(
                    m
                    for m in read.recorded_members
                    if m.evidence.work_item.work_item_id
                    == first.authorized_work_items[0].work_item_id
                )
                original = member.evidence.sample_observation.preparation.original
                if expected_mode == "reuse_creation_proof":
                    self.assertIsNone(original)
                    continue
                self.assertIsNotNone(original, "the audit recorded no full read")
                self.assertEqual(
                    (original.path, original.byte_count, original.sha256),
                    (
                        command.vcf_file,
                        len(RUN_VCF),
                        hashlib.sha256(RUN_VCF).hexdigest(),
                    ),
                )

    def test_closed_result_union_has_one_explicit_admission_disposition_per_member(
        self,
    ) -> None:
        # This pure admission table does not consume Journey or Execution authority.
        # Public application tests below independently prove exact approval/lineage.
        authority = InvestigationAuthority.model_construct(
            authority_id="investigation-authority:" + "1" * 64,
            proposal_id="investigation-proposal:" + "2" * 64,
            review_response_id="investigation-review-response:" + "3" * 64,
        )
        work_item = WorkItem.model_construct(
            work_item_id="work-item:" + "4" * 64,
            sample_id="HG001",
            target=VariantTarget.model_construct(
                assembly_accession="GCF_000001405.40",
                assembly_name="GRCh38.p14",
                contig="NC_000019.10",
                one_based_position=44908684,
                ref="T",
                alt="C",
                aliases=("rs429358",),
            ),
        )
        descriptor = VerifiedObject(
            role="VCF",
            bucket="accepted-run-bucket",
            object_key="outputs/HG001.vcf",
            byte_count=1234,
            sha256="a" * 64,
        )
        target = ObserveVariantInput(
            assembly="GCF_000001405.40",
            contig="NC_000019.10",
            one_based_position=44908684,
            ref="T",
            alt="C",
            display_label="rs429358",
        )
        from bio_neb.investigation.vcf_observation import (
            ObservedVcfRecord,
            SampleGenotype,
        )

        def record(ref="T", alts=("C",), position=44908684, genotype="0/1"):
            # The claim carries each record's call quality as written.
            return ObservedVcfRecord(
                contig="NC_000019.10",
                one_based_position=position,
                ref=ref,
                alts=alts,
                genotypes=(SampleGenotype(sample_name="HG001", genotype=genotype),),
            )

        cases = (
            (
                ExactAlleleObserved.model_construct(
                    target=target,
                    record=record(),
                    other_records=(),
                    genotype=SimpleNamespace(raw_genotype="0/1", target_alt_index=1),
                ),
                "admitted",
                "supported",
            ),
            (
                TargetAlleleNotCarried.model_construct(
                    target=target,
                    record=record(genotype="0/0"),
                    other_records=(),
                    genotype=SimpleNamespace(raw_genotype="0/0", target_alt_index=1),
                ),
                "admitted",
                "contradicted",
            ),
            (
                # A fully called sibling ALT is the same "not carried".
                TargetAlleleNotCarried.model_construct(
                    target=target,
                    record=record(alts=("A",)),
                    other_records=(),
                    genotype=SimpleNamespace(raw_genotype="0/1", target_alt_index=None),
                ),
                "admitted",
                "contradicted",
            ),
            (
                TargetGenotypeUnresolved.model_construct(
                    target=target,
                    record=record(genotype="./."),
                    other_records=(),
                    genotype=SimpleNamespace(raw_genotype="./.", target_alt_index=1),
                ),
                "not_admitted",
                "unresolved",
            ),
            (
                LocusIdentityMismatch.model_construct(
                    target=target,
                    records=(record(ref="A", alts=("G",)),),
                ),
                "admitted",
                "unresolved",
            ),
            (
                LocusOverlapped.model_construct(
                    target=target,
                    records=(record(ref="AT", alts=("A",), position=44908683),),
                ),
                "admitted",
                "unresolved",
            ),
            (
                NoRecordObserved.model_construct(target=target),
                "admitted",
                "unresolved",
            ),
            (
                AmbiguousRecordsObserved.model_construct(
                    target=target,
                    matching_records=(record(), record()),
                ),
                "not_admitted",
                "partial",
            ),
            (
                VariantObservationRefused.model_construct(
                    target=target, code="tool_failure"
                ),
                "not_admitted",
                "refused",
            ),
        )
        observed = []
        for result, admission_status, disposition in cases:
            observation = SampleObservation.model_construct(
                investigation_authority_id=authority.authority_id,
                proposal_id=authority.proposal_id,
                review_response_id=authority.review_response_id,
                work_item_id=work_item.work_item_id,
                sample_id=work_item.sample_id,
                assembly_accession=work_item.target.assembly_accession,
                assembly_name=work_item.target.assembly_name,
                vcf_descriptor=descriptor,
                materialization_receipt_id=("vcf-materialization-receipt:" + "7" * 64),
                materialization_receipt_sha256="6" * 64,
                preparation=VcfPreparationFailed(code="tool_failure"),
                result=result,
                run_provenance="unestablished",
                grants_scientific_truth=False,
            )
            candidate, assessment, admission, state = assess_investigation_observation(
                authority=authority,
                work_item=work_item,
                observation=observation,
                inherited_model_contact_count=3,
                inherited_source_contact_count=2,
            )
            observed.append(result.outcome)
            self.assertFalse(candidate.scientifically_admitted)
            self.assertFalse(assessment.admits_claim)
            self.assertEqual(admission.status, admission_status)
            self.assertEqual(state.work_item_disposition, disposition)
            self.assertEqual(state.limitations[0], result.claim_scope)
            if result.outcome == "no_record":
                self.assertIn(
                    "reference genotype is not established", result.claim_scope
                )
        self.assertEqual(
            observed,
            [
                "exact_allele_observed",
                "target_allele_not_carried",
                "target_allele_not_carried",
                "target_genotype_unresolved",
                "locus_present_identity_mismatch",
                "locus_overlapped",
                "no_record",
                "ambiguous_records",
                "refused",
            ],
        )

    def test_public_application_derives_only_approved_target_and_appends_atomic_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, projection = _approved_journey(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=OBSERVATION_REQUEST
                ).journey_file,
            )
            authority = projection.authority
            assert authority is not None
            work_item = authority.authorized_work_items[0]
            investigation_input = projection.investigation_input
            assert investigation_input is not None
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
                materialization_receipt_file=str(root / "vcf-materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )
            evidence = _Evidence(descriptor)
            adapter = _Adapter(command, descriptor, investigation_input.sample_id)

            collection = investigation_observe(
                command, adapter=adapter, evidence_reader=evidence
            )

            recorded = validate_investigation_evidence_fact(
                Journey.open(journey.path).read()[-2]
            )
            assert recorded is not None
            self.assertEqual(
                recorded.sample_observation.result.outcome, "exact_allele_observed"
            )
            self.assertEqual(
                recorded.sample_observation.schema_version,
                "bio-neb.sample-observation/2",
            )
            for provenance in (recorded.sample_observation, recorded.evidence_state):
                self.assertEqual(provenance.run_provenance, "causal_child_or_fork")
                self.assertEqual(
                    provenance.provenance_origin_event_id,
                    own_fork_origin(Journey.open(journey.path))[0],
                )
                self.assertNotIn(
                    "non_fresh_development_clone", provenance.model_dump(mode="json")
                )
            self.assertNotIn(
                "This evidence comes from an owner-authorized non-fresh development clone.",
                recorded.evidence_state.limitations,
            )
            self.assertFalse(recorded.candidate_evidence.scientifically_admitted)
            self.assertFalse(recorded.assessment.admits_claim)
            self.assertEqual(recorded.admission.status, "admitted")
            self.assertEqual(
                recorded.sample_observation.materialization_receipt_id,
                "vcf-materialization-receipt:" + "7" * 64,
            )
            self.assertEqual(
                recorded.sample_observation.materialization_receipt_sha256,
                "6" * 64,
            )
            claim = recorded.candidate_evidence.proposed_claim
            self.assertEqual(claim.sample_id, investigation_input.sample_id)
            self.assertEqual(
                claim.assembly_accession, work_item.target.assembly_accession
            )
            self.assertEqual(claim.assembly_name, work_item.target.assembly_name)
            self.assertEqual(claim.target, recorded.sample_observation.result.target)
            self.assertEqual(claim.vcf_descriptor, descriptor)
            self.assertEqual(claim.outcome, "exact_allele_observed")
            self.assertIn(investigation_input.sample_id, claim.observation_detail)
            self.assertIn("0/1", claim.observation_detail)
            self.assertEqual(recorded.admission.admitted_claim, claim)
            self.assertEqual(
                recorded.evidence_state.status, "investigation_evidence_ready"
            )
            self.assertEqual(recorded.evidence_state.inherited_model_contact_count, 3)
            self.assertEqual(recorded.evidence_state.inherited_source_contact_count, 2)
            self.assertEqual(
                len(Journey.open(journey.path).read()), len(journey.read()) + 3
            )
            reopened = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(reopened.status, "investigation_evidence_ready")
            self.assertEqual(reopened.investigation_evidence, collection)

            current = Journey.open(journey.path)
            evidence_event = current.read()[-2]
            current.record(
                "InvestigationEvidenceConsumerRecorded",
                {"evidence_state_id": recorded.evidence_state.evidence_state_id},
                [evidence_event["event_id"]],
            )
            later_reopen = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(later_reopen.status, "investigation_evidence_ready")
            self.assertEqual(later_reopen.investigation_evidence, collection)

            retained = copy.deepcopy(evidence_event)
            retained["payload"]["assessment"]["supports_admission"] = False
            with self.assertRaises(ValidationError):
                validate_investigation_evidence_fact(retained)
            retained = copy.deepcopy(evidence_event)
            retained["payload"]["sample_observation"]["proposal_id"] = (
                "investigation-proposal:" + "0" * 64
            )
            with self.assertRaises(ValidationError):
                validate_investigation_evidence_fact(retained)

    def test_duplicate_replay_returns_before_local_evidence_or_process_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, projection = _approved_journey(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=OBSERVATION_REQUEST
                ).journey_file,
            )
            authority = projection.authority
            investigation_input = projection.investigation_input
            assert authority is not None and investigation_input is not None
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
            adapter = _Adapter(command, descriptor, investigation_input.sample_id)
            collection = investigation_observe(
                command, adapter=adapter, evidence_reader=evidence
            )
            before = journey.path.read_bytes()
            calls = (tuple(evidence.calls), tuple(adapter.calls))

            replay = investigation_observe(
                command,
                adapter=SimpleNamespace(
                    run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("duplicate replay contacted process")
                    )
                ),
                evidence_reader=SimpleNamespace(
                    materialization=lambda **_kwargs: (_ for _ in ()).throw(
                        AssertionError("duplicate replay read local evidence")
                    )
                ),
            )

            self.assertEqual(replay, collection)
            first = validate_investigation_evidence_fact(
                Journey.open(journey.path).read()[-2]
            )
            assert first is not None
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual((tuple(evidence.calls), tuple(adapter.calls)), calls)

            observation = first.sample_observation.model_copy(
                update={
                    "run_provenance": "causal_child_or_fork",
                    "provenance_origin_event_id": journey.read()[1]["event_id"],
                }
            )
            candidate, assessment, admission, state = assess_investigation_observation(
                authority=first.authority,
                work_item=first.work_item,
                observation=observation,
                inherited_model_contact_count=3,
                inherited_source_contact_count=2,
            )
            forged = first.model_copy(
                update={
                    "sample_observation": observation,
                    "candidate_evidence": candidate,
                    "assessment": assessment,
                    "admission": admission,
                    "evidence_state": state,
                }
            ).model_dump(
                mode="json", exclude={"event_type"}, exclude_computed_fields=True
            )
            legacy = first.model_dump(
                mode="json", exclude={"event_type"}, exclude_computed_fields=True
            )
            for name in ("sample_observation", "evidence_state"):
                legacy[name]["schema_version"] = legacy[name]["schema_version"].replace(
                    "/2", "/1"
                )
                legacy[name].pop("run_provenance")
                legacy[name].pop("provenance_origin_event_id")
                legacy[name]["non_fresh_development_clone"] = True
            for payload in (forged, legacy):
                events = [json.loads(line) for line in before.splitlines()]
                events[-2]["payload"] = payload
                hostile = (
                    "\n".join(canonical_json(event) for event in events) + "\n"
                ).encode()
                journey.path.write_bytes(hostile)
                with self.assertRaises(ValueError):
                    investigation_observe(
                        command, adapter=adapter, evidence_reader=evidence
                    )
                with self.assertRaises(ValueError):
                    observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(journey.path.read_bytes(), hostile)
                self.assertEqual((tuple(evidence.calls), tuple(adapter.calls)), calls)

    def test_changed_vcf_preparation_refusal_and_conflicting_replay_are_atomic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, projection = _approved_journey(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=OBSERVATION_REQUEST
                ).journey_file,
            )
            authority = projection.authority
            investigation_input = projection.investigation_input
            assert authority is not None and investigation_input is not None
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
            before = journey.path.read_bytes()
            changed = descriptor.model_copy(update={"sha256": "0" * 64})
            changed_evidence = _Evidence(changed)

            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_vcf_identity_changed",
            ):
                investigation_observe(
                    command,
                    adapter=SimpleNamespace(
                        run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("changed VCF reached process contact")
                        )
                    ),
                    evidence_reader=changed_evidence,
                )
            self.assertEqual(changed_evidence.calls, ["materialization"])
            self.assertEqual(journey.path.read_bytes(), before)

            evidence = _Evidence(descriptor)
            adapter = _Adapter(command, descriptor, "unexpected-sample")
            from bio_neb.application.queries import read_investigation
            from bio_neb.failure import Invariant

            # The sample's identity fails the shared preparation: retained on its
            # item, then the Run is held for the fix (R2).
            with self.assertRaises(Invariant) as held:
                investigation_observe(
                    command, adapter=adapter, evidence_reader=evidence
                )
            self.assertEqual(
                held.exception.invariant_id, "investigation_shared_observation_failed"
            )
            collection = read_investigation(Journey.open(journey.path)).evidence
            refused = validate_investigation_evidence_fact(
                Journey.open(journey.path).read()[-2]
            )
            assert refused is not None
            self.assertEqual(collection.item_states[0].status, "failed")
            self.assertIsInstance(
                refused.sample_observation.preparation, VcfPreparationFailed
            )
            self.assertIsInstance(
                refused.sample_observation.result, VariantObservationRefused
            )
            self.assertEqual(
                refused.sample_observation.result.preparation_failure,
                refused.sample_observation.preparation,
            )
            preparation = refused.sample_observation.preparation
            # The application retains ASCII-escaped JSON for this failure identity.
            # Select the public failure fields independently of either digest owner.
            failure_material = {
                "status": "failed",
                "code": preparation.code,
                "details": list(preparation.details),
                "invocations": [
                    {
                        "purpose": item.purpose,
                        "argv": list(item.argv),
                        "return_code": item.return_code,
                        "stdout_byte_count": item.stdout_byte_count,
                        "stdout_sha256": item.stdout_sha256,
                        "stderr_byte_count": item.stderr_byte_count,
                        "stderr_sha256": item.stderr_sha256,
                    }
                    for item in preparation.invocations
                ],
            }
            expected_failure_id = (
                "vcf-observation-preparation-failure:"
                + hashlib.sha256(
                    json.dumps(
                        failure_material,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            )
            self.assertEqual(
                refused.sample_observation.result.preparation_receipt_id,
                expected_failure_id,
            )
            self.assertIn(expected_failure_id, refused.assessment.basis)
            self.assertEqual(refused.admission.status, "not_admitted")
            self.assertEqual(refused.evidence_state.work_item_disposition, "refused")
            refused_event = Journey.open(journey.path).read()[-2]
            tampered = copy.deepcopy(refused_event)
            tampered["payload"]["sample_observation"]["preparation"]["code"] = (
                "input_unavailable"
            )
            with self.assertRaises(ValidationError):
                validate_investigation_evidence_fact(tampered)
            retained = journey.path.read_bytes()

            conflicting = command.model_copy(
                update={"derivation_root": str(root / "other-derived")}
            )
            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_conflict",
            ):
                investigation_observe(
                    conflicting,
                    evidence_reader=SimpleNamespace(
                        materialization=lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("conflicting replay read local evidence")
                        )
                    ),
                )
            self.assertEqual(journey.path.read_bytes(), retained)

    def test_stale_authority_refuses_before_local_evidence_and_journey_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, projection = _approved_journey(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=OBSERVATION_REQUEST
                ).journey_file,
            )
            authority = projection.authority
            assert authority is not None
            before = journey.path.read_bytes()
            command = InvestigationObservationCommand(
                journey_file=str(journey.path),
                authority_id="investigation-authority:" + "0" * 64,
                vcf_file=str(root / "variants.vcf"),
                materialization_receipt_file=str(root / "materialization.json"),
                bcftools_executable=str(root / "bcftools"),
                bcftools_probe_receipt_file=str(root / "probe.json"),
                derivation_root=str(root / "derived"),
            )

            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_authority_stale",
            ):
                investigation_observe(
                    command,
                    evidence_reader=SimpleNamespace(
                        materialization=lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("stale authority read local evidence")
                        )
                    ),
                )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_concrete_materialization_reader_requires_canonical_read_only_bound_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey_file = self.prerequisite("executed").journey_file
            packet = read_completed_parabricks_result(journey_file)
            vcf_path = root / "variants.vcf"
            vcf_path.write_bytes(
                b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            )
            vcf_path.chmod(0o400)

            def retained_file(
                path: Path, *, read_only: bool = True
            ) -> VcfMaterializationFile:
                body = path.read_bytes()
                return VcfMaterializationFile(
                    path=str(path),
                    byte_count=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    read_only=read_only,
                )

            from bio_neb.application.verification_policy import (
                ArtifactVerificationDefaults,
                ArtifactVerificationOverrides,
                resolve_artifact_verification_policy,
            )
            from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof

            vcf = retained_file(vcf_path).model_copy(
                update={"independently_full_read": False}
            )
            identity = ArtifactIdentity(byte_count=vcf.byte_count, sha256=vcf.sha256)
            topology = packet.selected_context.topology
            receipt = VcfMaterializationReceipt.bind(
                lineage=VcfMaterializationLineage(
                    journey_id=packet.journey_id,
                    journey_file=retained_file(Path(packet.journey_file)),
                    execution_id=packet.execution_id,
                    composition_binding_id=packet.composition_binding_id,
                    authority_id=packet.authority_id,
                    proposal_id=packet.proposal_id,
                    h100_configuration_id=(packet.h100_configuration_id),
                    h100_request_id="h100-execution-request:" + "1" * 64,
                    h100_receipt_id="h100-execution-receipt:" + "2" * 64,
                    h100_checkpoint_event_id="checkpoint-event",
                    provider_job_id="provider-job",
                    selected_context=packet.selected_context,
                    runtime=packet.runtime,
                ),
                descriptor=VerifiedObject(
                    role="VCF",
                    bucket=packet.storage.bucket,
                    object_key="execution/outputs/variants.vcf",
                    byte_count=vcf.byte_count,
                    sha256=vcf.sha256,
                ),
                materialized_vcf=vcf,
                proof=ArtifactProof(
                    identity=identity,
                    location=vcf.path,
                    producer="vcf_object_get",
                    expected_identity=identity,
                    source_proof="h100-execution-receipt:" + "2" * 64,
                ),
                credential_receipt=ObjectStorageCredentialReuseReceipt(
                    service_account_id=topology.service_account_id,
                    access_key_resource_id=topology.access_key_resource_id,
                    access_key_id=topology.access_key_id,
                    exact_topology_match_count=1,
                    access_key_state="ACTIVE",
                    secret_state="ACTIVE",
                    secret_delivery_mode="mystery_box",
                    host_secret_resolved=True,
                    worker_secret_version_bound=True,
                ),
                verification_policy=resolve_artifact_verification_policy(
                    ArtifactVerificationDefaults(), ArtifactVerificationOverrides()
                ),
            )
            receipt_file = root / "vcf-materialization-receipt.json"
            receipt_file.write_text(
                canonical_json(receipt.model_dump(mode="json")), encoding="utf-8"
            )
            receipt_file.chmod(0o400)
            vcf_file = str(vcf_path)
            reader = DirectInvestigationObservationEvidence()

            accepted = reader.materialization(
                receipt_file=str(receipt_file), vcf_file=vcf_file
            )
            self.assertEqual(accepted.receipt_id, receipt.receipt_id)
            self.assertEqual(
                accepted.receipt_file.sha256,
                hashlib.sha256(receipt_file.read_bytes()).hexdigest(),
            )

            receipt_file.chmod(0o600)
            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_receipt_not_immutable",
            ):
                reader.materialization(
                    receipt_file=str(receipt_file), vcf_file=vcf_file
                )
            receipt_file.chmod(0o400)

            noncanonical = root / "noncanonical-receipt.json"
            noncanonical.write_bytes(receipt_file.read_bytes() + b"\n")
            noncanonical.chmod(0o400)
            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_materialization_receipt_invalid",
            ):
                reader.materialization(
                    receipt_file=str(noncanonical), vcf_file=vcf_file
                )

            with self.assertRaisesRegex(
                InvestigationObservationApplicationRefused,
                "investigation_observation_vcf_identity_changed",
            ):
                reader.materialization(
                    receipt_file=str(receipt_file), vcf_file=str(root / "other.vcf")
                )

    def test_runtime_reader_binds_the_bcftools_the_run_probed(self) -> None:
        from bio_neb.agentic_journey.runtime_config import InvestigationTools
        from bio_neb.investigation.observation_tools import probe_observation_tools

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pinned = {"package": "pysam", "package_version": "0.24.1"}
            probe = probe_observation_tools(
                InvestigationTools(bcftools=pinned, samtools=pinned), root / "tools"
            )
            bcftools = probe.receipt.tool("bcftools")
            runtime = DirectInvestigationObservationEvidence().runtime(
                receipt_file=probe.receipt_path, executable=bcftools.executable
            )
        self.assertEqual(runtime.runtime.executable_path, bcftools.executable)
        self.assertEqual(runtime.runtime.executable_sha256, bcftools.executable_sha256)
        self.assertEqual(
            (runtime.runtime.bcftools_version, runtime.runtime.htslib_version),
            ("1.24", "1.24"),
        )
        self.assertEqual(runtime.runtime.probe_receipt_path, probe.receipt_path)
        self.assertEqual(runtime.runtime.probe_receipt_sha256, probe.receipt_sha256)
        self.assertEqual(runtime.receipt_file.sha256, probe.receipt_sha256)

    def test_runtime_reader_refuses_an_executable_the_probe_did_not_name(
        self,
    ) -> None:
        from bio_neb.agentic_journey.runtime_config import InvestigationTools
        from bio_neb.investigation.observation_tools import probe_observation_tools

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pinned = {"package": "pysam", "package_version": "0.24.1"}
            probe = probe_observation_tools(
                InvestigationTools(bcftools=pinned, samtools=pinned), root / "tools"
            )
            for executable, field in (
                (probe.receipt.tool("samtools").executable, "bcftools_executable"),
                (str(root / "bcftools"), "bcftools_executable"),
            ):
                with self.subTest(executable=executable):
                    with self.assertRaises(
                        InvestigationObservationApplicationRefused
                    ) as refused:
                        DirectInvestigationObservationEvidence().runtime(
                            receipt_file=probe.receipt_path, executable=executable
                        )
                    self.assertEqual(
                        (refused.exception.code, refused.exception.fields),
                        ("investigation_observation_probe_receipt_invalid", (field,)),
                    )

    def test_runtime_reader_refuses_a_receipt_that_is_not_a_tool_probe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            probe = root / "probe.json"
            version_output = "bcftools 1.24\nUsing htslib 1.24\n"
            # The retired free-form operator receipt: a descriptor with a size and
            # digest beside version text no longer names a probed tool.
            probe.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "binary": {
                                "path": str(root / "bcftools"),
                                "byte_count": 14,
                                "sha256": "a" * 64,
                            },
                            "version_output": version_output,
                        }
                    }
                ),
                encoding="utf-8",
            )
            probe.chmod(0o444)
            with self.assertRaises(
                InvestigationObservationApplicationRefused
            ) as refused:
                DirectInvestigationObservationEvidence().runtime(
                    receipt_file=str(probe), executable=str(root / "bcftools")
                )
        self.assertEqual(
            (refused.exception.code, refused.exception.fields),
            (
                "investigation_observation_probe_receipt_invalid",
                ("bcftools_probe_receipt_file",),
            ),
        )


if __name__ == "__main__":
    unittest.main()
