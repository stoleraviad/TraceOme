"""How a Run's VCF was made, as its succeeded Execution recorded it.

Copied from the approved execution package, the effective plan Execution composed
from it and the verified H100 outputs; read from the Journey's execution-path
facts alone, so a report's basis recompiles it from its exact prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.journey_path import Hit, project_execution
from bio_neb.prebuilt_index.contigs import ExtraContig

# The command the H100 stage runs (``h100_stage``) and what it runs, as NVIDIA
# documents ``pbrun germline``: fq2bam alignment, then HaplotypeCaller.
GERMLINE_COMMAND = ("pbrun", "germline")
_GERMLINE_ALIGNER = "fq2bam (BWA-MEM)"
GERMLINE_CALLER = "HaplotypeCaller"
# The ploidy that command calls at: HaplotypeCaller's default, since the worker
# passes no ploidy option (``worker_request`` fixes its argv). The Execution
# Review offers it as the researcher's one choice until Parabricks ploidy support
# is verified (the parked ploidy proposal).
GERMLINE_PLOIDY = 2


class RunMethod(FrozenContract):
    """The sample, reference, index, recipe, tools and VCF of one Execution."""

    sample_id: StrictStr
    assay: StrictStr
    read_layout: StrictStr
    reference_accession: StrictStr
    # The assembly the researcher asked for, when they chose the Run's, its newer
    # version with a fitting Galaxy index, in its place.
    requested_reference_accession: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference_source: StrictStr
    reference_checksum: StrictStr
    index_origin: Literal["prebuilt", "admitted", "cpu_built"]
    # The publication a prebuilt index came from, and the BWA it states.
    index_source: StrictStr | None = None
    # None when a prebuilt index's publisher states no bwa version.
    index_bwa_version: StrictStr | None
    # The bound prebuilt index's contigs beyond the Run's reference sequences, each
    # with the NCBI role of the row naming it; empty for any other index.
    index_extra_contigs: tuple[ExtraContig, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    workflow_id: StrictStr
    workflow_version: StrictStr
    command: tuple[StrictStr, ...] = Field(min_length=2)
    aligner: StrictStr
    caller: StrictStr
    # The ploidy the approved proposal stated and the researcher answered; absent for
    # an Execution approved before the proposal stated one.
    calling_ploidy: Literal[2] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The planning model's judgment of the organism's ploidy that the approved
    # proposal stated beside it, copied for the report's qualification.
    organism_ploidy: Literal["diploid", "likely_non_diploid", "unknown"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    pbrun_version: StrictStr
    samtools_version: StrictStr
    worker_image: StrictStr
    vcf_content_id: StrictStr
    source_event_ids: tuple[StrictStr, ...] = Field(min_length=2)


def run_method(timeline: Sequence[Mapping[str, Any]]) -> RunMethod | None:
    """The method of the timeline's succeeded Execution; None when no Execution
    succeeded (the Run ended before its VCF existed)."""
    from bio_neb.parabricks.plan import (
        ApprovedParabricksInput,
        RetainedExecution,
        plan_execution,
    )

    recorded = [
        event for event in timeline if event["event_type"] == "ExecutionPathRecorded"
    ]
    if not recorded:
        return None
    state = project_execution([event["payload"] for event in recorded])
    if state.disposition != "succeeded":
        return None
    approved = ApprovedParabricksInput.model_validate(state.bound.approved_input)
    runtimes = plan_execution(
        approved, RetainedExecution(state, approved.authority, approved.proposal)
    ).runtimes
    package = approved.proposal.execution_package
    reference, workflow = package.reference, package.workflow
    prebuilt = reference.prebuilt_index
    (vcf,) = (item for item in state.h100.pointers if item.role == "VCF")
    path_ids = {
        state.latest_event_ids["ExecutionBound"],
        state.latest_event_ids["H100ArtifactsVerified"],
    }
    return RunMethod(
        sample_id=workflow.sample_id,
        assay=workflow.assay,
        read_layout=workflow.read_layout,
        reference_accession=reference.assembly_accession,
        requested_reference_accession=reference.requested_assembly,
        reference_source=reference.source_locator,
        reference_checksum=(
            f"{reference.expected_checksum_algorithm}:{reference.expected_checksum}"
        ),
        index_origin=(
            "cpu_built"
            if not isinstance(state.reference, Hit)
            else "prebuilt"
            if prebuilt is not None
            else "admitted"
        ),
        index_source=(
            prebuilt.candidate.candidate_id
            if prebuilt is not None and isinstance(state.reference, Hit)
            else None
        ),
        index_bwa_version=(
            prebuilt.stated_bwa_version
            if prebuilt is not None and isinstance(state.reference, Hit)
            else runtimes.cpu.bwa_version
        ),
        index_extra_contigs=(
            prebuilt.contigs.extras
            if prebuilt is not None and isinstance(state.reference, Hit)
            else ()
        ),
        workflow_id=workflow.workflow_id,
        workflow_version=workflow.workflow_version,
        command=GERMLINE_COMMAND,
        aligner=_GERMLINE_ALIGNER,
        caller=GERMLINE_CALLER,
        calling_ploidy=(
            approved.proposal.ploidy.calling_ploidy
            if approved.proposal.ploidy is not None
            else None
        ),
        organism_ploidy=(
            approved.proposal.ploidy.organism.judgment
            if approved.proposal.ploidy is not None
            else None
        ),
        pbrun_version=runtimes.h100.pbrun_version,
        samtools_version=runtimes.cpu.samtools_version,
        worker_image=runtimes.h100.worker_runtime_image,
        vcf_content_id=vcf.content_id,
        source_event_ids=tuple(
            event["event_id"]
            for event in recorded
            if event["payload"]["event_id"] in path_ids
        ),
    )


__all__ = [
    "GERMLINE_CALLER",
    "GERMLINE_COMMAND",
    "GERMLINE_PLOIDY",
    "RunMethod",
    "run_method",
]
