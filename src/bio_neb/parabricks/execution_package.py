"""Scientific execution meaning compiled before final Execution HITL."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.application.input_journey import (
    InputMetadataInspected,
    RunInputsRegistered,
    validate_fact,
)
from bio_neb.contracts import FrozenContract
from bio_neb.journey import JourneyEvent, canonical_json_digest
from bio_neb.parabricks.execution_planning_journey import (
    RecipeObservationRecorded,
    validate_execution_planning_fact,
)
from bio_neb.prebuilt_index.choice import PrebuiltIndexCandidate
from bio_neb.prebuilt_index.contigs import RequestedContig
from bio_neb.reference_discovery.reference_source import (
    ReferenceSourceAdmitted,
    SourceDerivationRule,
    validate_reference_source_fact,
)


def _uuid(value: str, field: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError(f"{field} must be canonical")
    return value


class ExecutionPackageEvidence(FrozenContract):
    source_event_id: StrictStr
    event_type: StrictStr
    evidence_identities: tuple[StrictStr, ...] = ()

    @field_validator("source_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _uuid(value, "source_event_id")


class ExecutionPackageSourcePlan(FrozenContract):
    planned_proposal_id: StrictStr = Field(pattern=r"^execution-proposal:[0-9a-f]{64}$")
    plan_event_id: StrictStr
    checkpoint_event_id: StrictStr
    execution_planning_evidence_event_id: StrictStr
    evidence_closure: tuple[ExecutionPackageEvidence, ...] = Field(min_length=1)

    @field_validator(
        "plan_event_id", "checkpoint_event_id", "execution_planning_evidence_event_id"
    )
    @classmethod
    def canonical_events(cls, value: str, info: Any) -> str:
        return _uuid(value, info.field_name)


class ExecutionInputSource(FrozenContract):
    """Selected source metadata; optional SHA-256 is an independent expectation."""

    file_id: StrictStr = Field(pattern=r"^execution-file:[0-9a-f]{64}$")
    role: StrictStr
    registered_path: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    filesystem_device: StrictStr
    filesystem_inode: StrictStr
    modified_at_ns: StrictInt
    registration_event_id: StrictStr
    inspection_event_id: StrictStr

    @field_validator("registration_event_id", "inspection_event_id")
    @classmethod
    def canonical_events(cls, value: str, info: Any) -> str:
        return _uuid(value, info.field_name)

    @model_validator(mode="after")
    def exact_identity(self) -> ExecutionInputSource:
        material = self.model_dump(mode="json", exclude={"file_id"})
        if self.file_id != ("execution-file:" + canonical_json_digest(material)):
            raise ValueError("execution source identity omitted selection or lineage")
        return self


class ExecutionReference(FrozenContract):
    assembly_accession: StrictStr
    reference_asset_id: StrictStr
    representation: StrictStr
    selection_event_id: StrictStr
    evidence_id: StrictStr
    admission_event_id: StrictStr
    source_admission_id: StrictStr = Field(
        pattern=r"^reference-source-admission:[0-9a-f]{64}$"
    )
    authority_locator: StrictStr
    source_mode: Literal["public_url", "local_file", "existing_object"]
    source_locator: StrictStr
    source_representation: Literal["gzip_fasta", "plain_fasta"]
    expected_size_bytes: StrictInt = Field(gt=0)
    expected_checksum_algorithm: Literal["md5", "sha256", "sha512"]
    expected_checksum: StrictStr
    checksum_authority_locator: StrictStr
    derivation_rule: SourceDerivationRule
    # The Run's reference sequences, by their name in NCBI's FASTA and their length
    # (`contigs.REFERENCE_ROLES`), recorded once at the genomic FASTA's
    # description: a prebuilt index's fit is against them.
    reference_sequences: tuple[RequestedContig, ...]
    # The NCBI-FASTA names of the assembly's other sequences (alt loci, patches),
    # dropped from the FASTA the CPU build indexes and input storage measures.
    excluded_sequences: tuple[StrictStr, ...]
    # The prebuilt index the planning model chose and the Execution Review binds:
    # after approval execution downloads, fits and places it, or falls to the CPU
    # Review.
    prebuilt_index: PrebuiltIndexCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The assembly the researcher asked for, when they chose this one, its newer
    # version with a fitting Galaxy index, for the whole Run.
    requested_assembly: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("selection_event_id", "admission_event_id")
    @classmethod
    def canonical_events(cls, value: str, info: Any) -> str:
        return _uuid(value, info.field_name)


class MeaningParameter(FrozenContract):
    name: StrictStr
    value: StrictStr
    evidence_event_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("evidence_event_ids")
    @classmethod
    def canonical_events(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_uuid(item, "evidence_event_id") for item in value)


class ParabricksWorkflow(FrozenContract):
    workflow_id: StrictStr
    workflow_version: StrictStr
    recipe_candidate_id: StrictStr
    recipe_evidence_event_id: StrictStr
    assay: StrictStr
    read_layout: Literal["paired", "single"]
    sample_id: StrictStr
    meaning_parameters: tuple[MeaningParameter, ...] = ()

    @field_validator("recipe_evidence_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _uuid(value, "recipe_evidence_event_id")


class ExecutionResource(FrozenContract):
    profile_id: StrictStr
    candidate_id: StrictStr
    evidence_event_id: StrictStr
    accelerator_count: StrictInt = Field(ge=1)
    cpu_count: StrictInt = Field(ge=1)
    memory_gib: StrictInt = Field(ge=1)
    accelerator_memory_gib: StrictInt = Field(ge=1)

    @field_validator("evidence_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _uuid(value, "evidence_event_id")


class LogicalOutput(FrozenContract):
    role: Literal["aligned_bam", "variants"]
    format: Literal["BAM", "VCF"]
    evidence_event_ids: tuple[StrictStr, ...] = Field(min_length=1)


class ExecutionPackage(FrozenContract):
    package_id: StrictStr = Field(pattern=r"^execution-package:[0-9a-f]{64}$")
    journey_id: StrictStr
    source_plan: ExecutionPackageSourcePlan
    input_set_id: StrictStr
    inputs: tuple[ExecutionInputSource, ...] = Field(min_length=1)
    reference: ExecutionReference
    workflow: ParabricksWorkflow
    resource: ExecutionResource
    outputs: tuple[LogicalOutput, ...] = Field(min_length=1)

    @field_validator("journey_id")
    @classmethod
    def canonical_journey(cls, value: str) -> str:
        return _uuid(value, "journey_id")

    @model_validator(mode="after")
    def exact_identity(self) -> ExecutionPackage:
        if len({item.role for item in self.inputs}) != len(self.inputs):
            raise ValueError("execution package repeats an input role")
        if {(item.role, item.format) for item in self.outputs} != {
            ("aligned_bam", "BAM"),
            ("variants", "VCF"),
        }:
            raise ValueError("execution package changed logical Parabricks outputs")
        material = self.model_dump(mode="json", exclude={"package_id"})
        if self.package_id != ("execution-package:" + canonical_json_digest(material)):
            raise ValueError("execution package identity omitted scientific meaning")
        return self


class ExecutionPackageCompiled(FrozenContract):
    event_type: Literal["ExecutionPackageCompiled"] = "ExecutionPackageCompiled"
    package: ExecutionPackage


def describe_execution_input(
    inspected: Any, registration_event_id: str, inspection_event_id: str
) -> ExecutionInputSource:
    """Bind retained metadata without opening, hashing or re-statting the source.

    A succeeded inspection's input exists, is regular and is non-empty by contract.
    """
    material = {
        "role": inspected.role,
        "registered_path": inspected.registered_path,
        "byte_count": inspected.size_bytes,
        "sha256": None,
        "filesystem_device": inspected.filesystem_identity.device,
        "filesystem_inode": inspected.filesystem_identity.inode,
        "modified_at_ns": inspected.modified_at_ns,
        "registration_event_id": registration_event_id,
        "inspection_event_id": inspection_event_id,
    }
    return ExecutionInputSource(
        file_id=("execution-file:" + canonical_json_digest(material)), **material
    )


def compile_execution_package(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    planned: Any,
) -> ExecutionPackage:
    """Bind selected source descriptions and exact retained scientific evidence."""

    from bio_neb.parabricks.execution_proposal import ProposalCompilationRefused

    timeline = [
        item.model_dump(mode="json") if isinstance(item, JourneyEvent) else dict(item)
        for item in events
    ]
    by_id = {item["event_id"]: item for item in timeline}
    # Actual checked selections establish readiness; no ceremonial Agent terminal is required.
    # Planning readiness admits exactly one recipe and one resource.
    (recipe_binding,), (resource_binding,) = planned.recipes, planned.resources
    readiness_event_id = resource_binding.selection_event_id
    if readiness_event_id not in by_id:
        raise ProposalCompilationRefused("execution_package_planning_changed")
    registrations = [
        (event, fact)
        for event in timeline
        if isinstance((fact := validate_fact(event)), RunInputsRegistered)
        and fact.input_set_id == planned.inspected_inputs.input_set_id
    ]
    inspections = [
        (event, fact)
        for event in timeline
        if isinstance((fact := validate_fact(event)), InputMetadataInspected)
        and fact.result.status == "succeeded"
        and fact.result.input_set_id == planned.inspected_inputs.input_set_id
    ]
    if len(registrations) != 1 or not inspections:
        raise ProposalCompilationRefused("execution_package_input_lineage_invalid")
    registration_event, registration = registrations[0]
    inspection_event, inspection = inspections[-1]
    from bio_neb.application.input_journey import project_input_selection

    selected = project_input_selection(timeline)
    if (
        selected.correction_pending
        or selected.stopped
        or selected.inspections[-1]["event_id"] != inspection_event["event_id"]
    ):
        raise ProposalCompilationRefused("execution_package_input_lineage_invalid")
    inputs = tuple(
        describe_execution_input(
            item, registration_event["event_id"], inspection_event["event_id"]
        )
        for item in inspection.result.inputs
        if item.role in selected.fastq_roles
    )
    source_facts = [
        (event, fact)
        for event in timeline
        if isinstance(
            (fact := validate_reference_source_fact(event)), ReferenceSourceAdmitted
        )
        and (
            fact.reference_event_id == planned.reference_event_id
            or (fact.reference_event_id is None and planned.reference_basis is None)
        )
    ]
    # Preparation admitted the one source before compiling the package. The
    # recorded admission must be the one ``planned`` links; the requirement
    # meaning is ``planned``'s own, compiled from this timeline, and is not redone.
    ((source_event, source_fact),) = source_facts
    source = source_fact.admission
    companions = [item for item in planned.companions if item.role == "genomic_fasta"]
    if len(companions) != 1:
        raise ProposalCompilationRefused("execution_package_reference_invalid")
    companion = companions[0]
    if (
        source != planned.reference_source_admission
        or source.reference_asset_id != companion.candidate.asset_id
        or source.assembly_accession != planned.reference.assembly_accession
        or source.authority_locator != companion.candidate.authority_locator
    ):
        raise ProposalCompilationRefused("execution_package_reference_changed")
    recipe_observations = []
    for event in timeline:
        fact = validate_execution_planning_fact(event)
        if isinstance(fact, RecipeObservationRecorded) and any(
            item.candidate_id == recipe_binding.candidate.candidate_id
            for item in fact.result.candidates
        ):
            recipe_observations.append((event, fact))
    # The selection's basis observed this candidate, whose identity hashes its
    # capability snapshot and so its version.
    (workflow_version,) = {
        fact.result.source.capability_version for _, fact in recipe_observations
    }
    return assemble_execution_package(
        planned=planned,
        inputs=inputs,
        source_plan=ExecutionPackageSourcePlan(
            planned_proposal_id=planned.proposal_id,
            plan_event_id=planned.plan_event_id,
            checkpoint_event_id=readiness_event_id,
            execution_planning_evidence_event_id=readiness_event_id,
            evidence_closure=tuple(
                ExecutionPackageEvidence(**item.model_dump())
                for item in planned.evidence_closure
            ),
        ),
        source_admission_event_id=source_event["event_id"],
        workflow_version=workflow_version,
        recipe_evidence_event_id=recipe_observations[-1][0]["event_id"],
        output_evidence=tuple(
            item.source_event_id
            for item in planned.evidence_closure
            if item.source_event_id in by_id
        ),
    )


def assemble_execution_package(
    *,
    planned: Any,
    inputs: tuple[ExecutionInputSource, ...],
    source_plan: ExecutionPackageSourcePlan,
    source_admission_event_id: str,
    workflow_version: str,
    recipe_evidence_event_id: str,
    output_evidence: tuple[str, ...],
) -> ExecutionPackage:
    """One domain mapping for checked agent information and direct accepted facts."""

    (companion,) = (item for item in planned.companions if item.role == "genomic_fasta")
    (recipe_binding,), (resource_binding,) = planned.recipes, planned.resources
    source = planned.reference_source_admission
    sample_id = planned.requirement_resolution.one_value("sample_identity").value
    material = {
        "journey_id": planned.journey_id,
        "source_plan": source_plan,
        "input_set_id": planned.inspected_inputs.input_set_id,
        "inputs": inputs,
        "reference": ExecutionReference(
            assembly_accession=planned.reference.assembly_accession,
            reference_asset_id=companion.candidate.asset_id,
            representation=companion.candidate.representation,
            selection_event_id=companion.validation_event_id,
            evidence_id=companion.candidate.evidence_id,
            admission_event_id=source_admission_event_id,
            source_admission_id=source.admission_id,
            authority_locator=source.authority_locator,
            source_mode=source.source_mode,
            source_locator=source.source_locator,
            source_representation=source.source_representation,
            expected_size_bytes=source.expected_size_bytes,
            expected_checksum_algorithm=source.expected_checksum_algorithm,
            expected_checksum=source.expected_checksum,
            checksum_authority_locator=source.checksum_authority_locator,
            derivation_rule=source.derivation_rule,
            reference_sequences=companion.candidate.reference_sequences,
            excluded_sequences=companion.candidate.excluded_sequences,
            prebuilt_index=companion.prebuilt_index,
            requested_assembly=None
            if planned.reference_basis is None
            else planned.reference_basis.requested_assembly,
        ),
        "workflow": ParabricksWorkflow(
            workflow_id=recipe_binding.candidate.recipe_id,
            workflow_version=workflow_version,
            recipe_candidate_id=recipe_binding.candidate.candidate_id,
            recipe_evidence_event_id=recipe_evidence_event_id,
            assay=recipe_binding.candidate.assay,
            read_layout=recipe_binding.candidate.read_layout,
            sample_id=sample_id,
            meaning_parameters=(
                MeaningParameter(
                    name="capture_intervals",
                    value="omitted",
                    evidence_event_ids=(recipe_evidence_event_id,),
                ),
                MeaningParameter(
                    name="known_sites",
                    value="omitted",
                    evidence_event_ids=(recipe_evidence_event_id,),
                ),
            ),
        ),
        "resource": ExecutionResource(
            profile_id=resource_binding.candidate.profile_id,
            candidate_id=resource_binding.candidate.candidate_id,
            evidence_event_id=resource_binding.selection_event_id,
            accelerator_count=resource_binding.candidate.accelerator_count,
            cpu_count=resource_binding.candidate.cpu_count,
            memory_gib=resource_binding.candidate.memory_gib,
            accelerator_memory_gib=resource_binding.candidate.accelerator_memory_gib,
        ),
        "outputs": (
            LogicalOutput(
                role="aligned_bam", format="BAM", evidence_event_ids=output_evidence
            ),
            LogicalOutput(
                role="variants", format="VCF", evidence_event_ids=output_evidence
            ),
        ),
    }
    serial = ExecutionPackage.model_construct(
        package_id="execution-package:" + "0" * 64,
        **material,
    ).model_dump(mode="json", exclude={"package_id"})
    return ExecutionPackage(
        package_id=("execution-package:" + canonical_json_digest(serial)), **material
    )


def validate_execution_package_fact(
    event: Mapping[str, Any],
) -> ExecutionPackageCompiled | None:
    if event.get("event_type") != "ExecutionPackageCompiled":
        return None
    return ExecutionPackageCompiled.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


__all__ = [
    "ExecutionPackage",
    "ExecutionPackageCompiled",
    "compile_execution_package",
    "validate_execution_package_fact",
]
