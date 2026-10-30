"""Exact upstream Journey evidence for contact-free Finalization tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import ApproveInvestigationResponse
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    RespondInvestigationReviewCommand,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationObservationCommand,
)
from bio_neb.investigation.investigation_observation_commands import (
    MaterializationObservationEvidence,
    RuntimeProbeObservationEvidence,
    investigation_observe,
)
from bio_neb.investigation.vcf_observation import (
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
    CompletedProcess,
    LocalFileObservation,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
)
from bio_neb.verified_artifacts import VerifiedObject
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.prerequisite_builds import measured_build
from tests.sequential_investigation_support import begin_in_pass, source_basis
from tests.terminal_journey_fakes import EVIDENCE_REQUEST, forked_child
from tests.vcf_lookup_fakes import (
    lookup_command,
    run_table_reading,
    table_reading_reply,
    vcf_header,
)

REFSNP_BODY = json.dumps(
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
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()

ASSEMBLY_BODY = json.dumps(
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
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()


class _Planner:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        if (reply := table_reading_reply(info, messages)) is not None:
            return reply
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
                    tool_call_id=f"finalization-seed-{self.calls}",
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

    def materialization(self, *, receipt_file: str, vcf_file: str):
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


class _NoRecordAdapter:
    def __init__(
        self,
        command: InvestigationObservationCommand,
        descriptor: VerifiedObject,
        *,
        sample_id: str,
        query_stdout: bytes = b"",
    ) -> None:
        self.command = command
        self.query_stdout = query_stdout
        self.sample_id = sample_id
        self.files = {
            command.vcf_file: _file(
                command.vcf_file,
                descriptor.sha256,
                size=descriptor.byte_count,
            ),
            command.materialization_receipt_file: _file(
                command.materialization_receipt_file, "6" * 64, size=100
            ),
            command.bcftools_executable: _file(
                command.bcftools_executable,
                "9" * 64,
                executable=True,
                size=100,
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
        return CompletedProcess(0, self.query_stdout, b"")


def _record_observation(
    root: Path, journey: Journey, approved, *, query_stdout: bytes = b""
) -> Journey:
    authority = approved.authority
    selected_input = approved.investigation_input
    assert authority is not None and selected_input is not None
    descriptor = VerifiedObject(
        role="VCF",
        bucket=selected_input.vcf.bucket,
        object_key=selected_input.vcf.object_key,
        byte_count=selected_input.vcf.byte_count,
        sha256=selected_input.vcf.sha256,
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
    adapter = _NoRecordAdapter(
        command,
        descriptor,
        sample_id=selected_input.sample_id,
        query_stdout=query_stdout,
    )
    recorded = investigation_observe(
        command, adapter=adapter, evidence_reader=_Evidence(descriptor)
    )
    assert not recorded.failed, recorded.item_states
    # Every item's window, the chunked table reading, then the fold Finalization
    # reads (ADR 0047).
    run_table_reading(journey.path, adapter)
    return Journey.open(journey.path)


def evidence_journey(
    root: Path, *, query_stdout: bytes = b"", parent_file: Path | None = None
) -> Journey:
    """Create current retained-execution evidence for Finalization tests."""
    return retained_execution_evidence_journey(
        root, query_stdout=query_stdout, parent_file=parent_file
    )


@measured_build("retained-evidence")
def retained_execution_evidence_journey(
    root: Path, *, query_stdout: bytes = b"", parent_file: Path | None = None
) -> Journey:
    """Create evidence in a child admitted from current execution artifacts.

    Without parent_file the parent is executed under root/source for this case.
    """

    journey = forked_child(
        root,
        parent_file=parent_file,
        investigation_request=None if parent_file else EVIDENCE_REQUEST,
    )
    request_file = root / "investigation-request.json"
    request_file.write_text(canonical_json(EVIDENCE_REQUEST), encoding="utf-8")
    planner = _Planner()
    pending = begin_in_pass(
        BeginInvestigationCommand(
            journey_file=str(journey.path),
            request_files=(str(request_file),),
        ),
        environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
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
    return _record_observation(root, journey, approved, query_stdout=query_stdout)


@measured_build("planned-evidence")
def planned_evidence_journey(root: Path) -> Journey:
    """Create evidence in the planned Run itself, after its verified Execution.

    Investigation begins from this Run's own verified Execution, so the Journey
    that Finalization reads holds the proposal plan.
    """

    from tests.terminal_journey_fakes import execute_terminal

    journey = Journey.open(
        Path(execute_terminal(root / "source", investigation_request=EVIDENCE_REQUEST))
    )
    return observe_existing_run(root, journey)


def observe_existing_run(root: Path, journey: Journey) -> Journey:
    """Prepare, approve and observe science on the caller's existing verified Run."""
    request_file = root / "investigation-request.json"
    request_file.write_text(canonical_json(EVIDENCE_REQUEST), encoding="utf-8")
    planner = _Planner()
    pending = begin_in_pass(
        BeginInvestigationCommand(
            journey_file=str(journey.path), request_files=(str(request_file),)
        ),
        environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
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
    return _record_observation(root, journey, approved)


__all__ = [
    "evidence_journey",
    "planned_evidence_journey",
    "retained_execution_evidence_journey",
]
