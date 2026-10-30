"""Typed evidence state derived from one approved Investigation observation."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Sequence, Union

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    model_validator,
)

from bio_neb.agentic_journey.continuation_policy import InvestigationWorkPolicy
from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation import (
    InvestigationAuthority,
    RegionTarget,
    WorkItem,
)
from bio_neb.investigation.qualification_origins import observation_origins
from bio_neb.investigation.vcf_observation import (
    AmbiguousRecordsObserved,
    ExactAlleleObserved,
    LocusIdentityMismatch,
    LocusOverlapped,
    NoRecordObserved,
    ObservedVcfRecord,
    ObserveRegionInput,
    ObserveVariantInput,
    PreparedVcfObservationReceipt,
    TargetAlleleNotCarried,
    TargetGenotypeUnresolved,
    VariantObservationRefused,
    VariantObservationResult,
    VcfPreparationFailed,
    VcfRegionObserved,
    allele_text,
)
from bio_neb.journey import (
    FORK_ORIGIN_EVENT_TYPE,
    JourneyFile,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.verified_artifacts import VerifiedObject

RunProvenance = Literal["unestablished", "causal_child_or_fork"]


def validate_run_provenance(
    run_provenance: RunProvenance, provenance_origin_event_id: str | None
) -> None:
    if (run_provenance == "causal_child_or_fork") != (
        provenance_origin_event_id is not None
    ):
        raise ValueError("Run provenance requires its exact causal origin")


def derive_investigation_run_provenance(
    timeline: Sequence[Mapping[str, Any]],
) -> tuple[RunProvenance, str | None]:
    """A Fork child's provenance is its latest origin; a fresh Run's is unestablished.

    A prefix of a child's timeline that ends before its own origin keeps the
    parent's classification, so inherited evidence keeps its original one.
    """
    origins = [
        event["event_id"]
        for event in timeline
        if event.get("event_type") == FORK_ORIGIN_EVENT_TYPE
    ]
    if not origins:
        return "unestablished", None
    return "causal_child_or_fork", origins[-1]


def observation_target(
    work_item: WorkItem,
) -> ObserveVariantInput | ObserveRegionInput:
    """The one VCF lookup input a Work item's target hands over.

    Its label is the target's first rsID alias; a target without one has none. A
    region's input is the region with the Run's reference bases over it (#367).
    """
    target = work_item.target
    if isinstance(target, RegionTarget):
        return ObserveRegionInput(
            assembly=target.assembly_accession,
            contig=target.contig,
            start=target.start,
            end=target.end,
            reference_bases=target.reference_bases,
        )
    return ObserveVariantInput(
        assembly=target.assembly_accession,
        contig=target.contig,
        one_based_position=target.one_based_position,
        ref=target.ref,
        alt=target.alt,
        display_label=target.aliases[0] if target.aliases else None,
    )


class InvestigationObservationCommand(FrozenContract):
    journey_file: JourneyFile
    authority_id: StrictStr = Field(pattern=r"^investigation-authority:[0-9a-f]{64}$")
    vcf_file: StrictStr
    materialization_receipt_file: StrictStr
    bcftools_executable: StrictStr
    bcftools_probe_receipt_file: StrictStr
    derivation_root: StrictStr


class InvestigationWorkItemStarted(FrozenContract):
    event_type: Literal["InvestigationWorkItemStarted"] = "InvestigationWorkItemStarted"
    authority_id: StrictStr = Field(pattern=r"^investigation-authority:[0-9a-f]{64}$")
    work_item_id: StrictStr = Field(pattern=r"^work-item:[0-9a-f]{64}$")
    selected_ordinal: StrictInt = Field(gt=0)
    sample_id: StrictStr
    target: ObserveVariantInput | ObserveRegionInput


class SampleObservation(FrozenContract):
    schema_version: Literal["bio-neb.sample-observation/2"] = (
        "bio-neb.sample-observation/2"
    )
    investigation_authority_id: StrictStr = Field(
        pattern=r"^investigation-authority:[0-9a-f]{64}$"
    )
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    review_response_id: StrictStr = Field(
        pattern=r"^investigation-review-response:[0-9a-f]{64}$"
    )
    work_item_id: StrictStr = Field(pattern=r"^work-item:[0-9a-f]{64}$")
    sample_id: StrictStr
    assembly_accession: StrictStr
    assembly_name: StrictStr
    vcf_descriptor: VerifiedObject
    materialization_receipt_id: StrictStr = Field(
        pattern=r"^vcf-materialization-receipt:[0-9a-f]{64}$"
    )
    materialization_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: PreparedVcfObservationReceipt | VcfPreparationFailed
    result: VariantObservationResult
    run_provenance: RunProvenance = "unestablished"
    provenance_origin_event_id: StrictStr | None = Field(default=None, min_length=1)
    grants_scientific_truth: Literal[False] = False

    @computed_field(return_type=str)
    @property
    def observation_id(self) -> str:
        return "sample-observation:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"observation_id"})
        )

    @model_validator(mode="after")
    def exact_preparation_lineage(self) -> "SampleObservation":
        validate_run_provenance(self.run_provenance, self.provenance_origin_event_id)
        if self.vcf_descriptor.role != "VCF" or self.result.target is None:
            raise ValueError("Sample observation lacks exact VCF or target identity")
        if (
            self.result.authority_kind != "investigation_authority"
            or self.result.continuation_id != self.investigation_authority_id
        ):
            raise ValueError("Sample observation changed Tool authority")
        if isinstance(self.preparation, PreparedVcfObservationReceipt):
            if (
                self.preparation.authority_kind != "investigation_authority"
                or self.preparation.continuation_id != self.investigation_authority_id
                or self.result.preparation_receipt_id != self.preparation.receipt_id
                or self.preparation.assembly != self.assembly_accession
                or self.preparation.continuation_sample_id != self.sample_id
                or self.preparation.vcf_descriptor != self.vcf_descriptor
                or self.preparation.acquisition_receipt_sha256
                != self.materialization_receipt_sha256
                or self.preparation.acquisition_receipt_file.sha256
                != self.materialization_receipt_sha256
                or (
                    self.result.query_invocation is not None
                    and (
                        self.result.query_invocation.argv[0]
                        != self.preparation.runtime.executable_path
                        or self.result.query_invocation.argv[-1]
                        != self.preparation.derived_bgzf.path
                    )
                )
            ):
                raise ValueError("Sample observation changed preparation authority")
        elif (
            not isinstance(self.result, VariantObservationRefused)
            or self.result.preparation_failure != self.preparation
        ):
            raise ValueError("preparation failure must retain one exact refused result")
        return self


ObservationOutcome = Literal[
    "exact_allele_observed",
    "target_allele_not_carried",
    "target_genotype_unresolved",
    "locus_present_identity_mismatch",
    "locus_overlapped",
    "no_record",
    "ambiguous_records",
    "refused",
    "region_observed",
]


class VcfLocalObservationClaim(FrozenContract):
    schema_version: Literal["bio-neb.vcf-local-observation-claim/1"] = (
        "bio-neb.vcf-local-observation-claim/1"
    )
    outcome: ObservationOutcome
    sample_id: StrictStr
    assembly_accession: StrictStr
    assembly_name: StrictStr
    target: ObserveVariantInput | ObserveRegionInput
    vcf_descriptor: VerifiedObject
    observation_detail: StrictStr
    claim_scope: StrictStr
    # The records the claim rests on, each with FILTER, QUAL, GQ, DP and AD as
    # written; recorded for the report, never a gate.
    observed_records: tuple[ObservedVcfRecord, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


def observed_records(result: VariantObservationResult) -> tuple[ObservedVcfRecord, ...]:
    """Every record an observation outcome carries, in its own order."""
    if isinstance(
        result, (ExactAlleleObserved, TargetAlleleNotCarried, TargetGenotypeUnresolved)
    ):
        return (result.record, *result.other_records)
    if isinstance(result, (LocusIdentityMismatch, LocusOverlapped)):
        return result.records
    if isinstance(result, AmbiguousRecordsObserved):
        return result.matching_records
    if isinstance(result, (NoRecordObserved, VariantObservationRefused)):
        return ()
    if isinstance(result, VcfRegionObserved):
        return tuple(row.record for row in result.records)
    raise TypeError("observation result union is not exhaustive")


class CandidateEvidence(FrozenContract):
    schema_version: Literal["bio-neb.candidate-evidence/1"] = (
        "bio-neb.candidate-evidence/1"
    )
    investigation_authority_id: StrictStr
    work_item_id: StrictStr
    observation_id: StrictStr
    outcome: StrictStr
    proposed_claim: VcfLocalObservationClaim
    claim_scope: StrictStr
    limitations: tuple[StrictStr, ...] = Field(min_length=2)
    source_role: Literal["sample_observation"] = "sample_observation"
    scientifically_admitted: Literal[False] = False

    @computed_field(return_type=str)
    @property
    def candidate_id(self) -> str:
        return "candidate-evidence:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"candidate_id"})
        )


WorkItemDisposition = Literal[
    "supported", "contradicted", "unresolved", "refused", "partial"
]


class ObservationEvidenceAssessment(FrozenContract):
    schema_version: Literal["bio-neb.observation-evidence-assessment/1"] = (
        "bio-neb.observation-evidence-assessment/1"
    )
    investigation_authority_id: StrictStr
    work_item_id: StrictStr
    candidate_id: StrictStr
    observation_id: StrictStr
    outcome: StrictStr
    verdict: Literal["supported", "unsupported", "refused"]
    work_item_disposition: WorkItemDisposition
    basis: tuple[StrictStr, ...] = Field(min_length=3)
    claim_scope: StrictStr
    limitations: tuple[StrictStr, ...] = Field(min_length=2)
    contradictions: tuple[StrictStr, ...]
    applicability: tuple[StrictStr, ...] = Field(min_length=1)
    supports_admission: StrictBool
    assessor: Literal["deterministic_investigation_observation_policy"] = (
        "deterministic_investigation_observation_policy"
    )
    admits_claim: Literal[False] = False

    @computed_field(return_type=str)
    @property
    def assessment_id(self) -> str:
        return "observation-evidence-assessment:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"assessment_id"})
        )


class AdmittedObservationEvidence(FrozenContract):
    status: Literal["admitted"] = "admitted"
    investigation_authority_id: StrictStr
    work_item_id: StrictStr
    candidate_id: StrictStr
    assessment_id: StrictStr
    observation_id: StrictStr
    admitted_claim: VcfLocalObservationClaim
    claim_scope: StrictStr
    admission_authority: Literal[
        "investigation_authority_exact_vcf_local_claim_policy"
    ] = "investigation_authority_exact_vcf_local_claim_policy"

    @computed_field(return_type=str)
    @property
    def admission_id(self) -> str:
        return "admitted-observation-evidence:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"admission_id"})
        )


class ObservationEvidenceNotAdmitted(FrozenContract):
    status: Literal["not_admitted"] = "not_admitted"
    investigation_authority_id: StrictStr
    work_item_id: StrictStr
    candidate_id: StrictStr
    assessment_id: StrictStr
    observation_id: StrictStr
    reason: StrictStr
    claim_scope: StrictStr

    @computed_field(return_type=str)
    @property
    def admission_id(self) -> str:
        return "observation-evidence-not-admitted:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"admission_id"})
        )


ObservationAdmissionDecision = Annotated[
    Union[AdmittedObservationEvidence, ObservationEvidenceNotAdmitted],
    Field(discriminator="status"),
]


class InvestigationEvidenceState(FrozenContract):
    schema_version: Literal["bio-neb.investigation-evidence-state/2"] = (
        "bio-neb.investigation-evidence-state/2"
    )
    status: Literal["investigation_evidence_ready"] = "investigation_evidence_ready"
    investigation_authority_id: StrictStr
    proposal_id: StrictStr
    review_response_id: StrictStr
    work_item_id: StrictStr
    observation_id: StrictStr
    candidate_id: StrictStr
    assessment_id: StrictStr
    admission_id: StrictStr
    outcome: StrictStr
    work_item_disposition: WorkItemDisposition
    evidence_admission: Literal["admitted", "not_admitted"]
    limitations: tuple[StrictStr, ...] = Field(min_length=2)
    conflicts: tuple[StrictStr, ...]
    gaps: tuple[StrictStr, ...]
    inherited_model_contact_count: StrictInt = Field(ge=0)
    inherited_source_contact_count: StrictInt = Field(ge=0)
    observation_model_contact_count: Literal[0] = 0
    observation_source_contact_count: Literal[0] = 0
    forbidden_contact_count: Literal[0] = 0
    run_provenance: RunProvenance = "unestablished"
    provenance_origin_event_id: StrictStr | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def exact_provenance(self) -> "InvestigationEvidenceState":
        validate_run_provenance(self.run_provenance, self.provenance_origin_event_id)
        return self

    @computed_field(return_type=str)
    @property
    def evidence_state_id(self) -> str:
        return "investigation-evidence-state:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"evidence_state_id"})
        )


class InvestigationEvidenceRecorded(FrozenContract):
    event_type: Literal["InvestigationEvidenceRecorded"] = (
        "InvestigationEvidenceRecorded"
    )
    command: InvestigationObservationCommand
    authority: InvestigationAuthority
    work_item: WorkItem
    sample_observation: SampleObservation
    candidate_evidence: CandidateEvidence
    assessment: ObservationEvidenceAssessment
    admission: ObservationAdmissionDecision
    evidence_state: InvestigationEvidenceState

    @model_validator(mode="after")
    def exact_aggregate_lineage(self) -> "InvestigationEvidenceRecorded":
        values = (
            self.command.authority_id,
            self.authority.authority_id,
            self.sample_observation.investigation_authority_id,
            self.candidate_evidence.investigation_authority_id,
            self.assessment.investigation_authority_id,
            self.admission.investigation_authority_id,
            self.evidence_state.investigation_authority_id,
        )
        work_items = (
            self.work_item.work_item_id,
            self.sample_observation.work_item_id,
            self.candidate_evidence.work_item_id,
            self.assessment.work_item_id,
            self.admission.work_item_id,
            self.evidence_state.work_item_id,
        )
        if len(set(values)) != 1 or len(set(work_items)) != 1:
            raise ValueError("Investigation evidence changed authority or Work item")
        if (
            self.work_item not in self.authority.authorized_work_items
            or self.sample_observation.proposal_id != self.authority.proposal_id
            or self.sample_observation.review_response_id
            != self.authority.review_response_id
            or self.sample_observation.sample_id != self.work_item.sample_id
            or self.sample_observation.assembly_accession
            != self.work_item.target.assembly_accession
            or self.sample_observation.assembly_name
            != self.work_item.target.assembly_name
            or self.sample_observation.result.target
            != observation_target(self.work_item)
            or self.candidate_evidence.observation_id
            != self.sample_observation.observation_id
            or self.assessment.candidate_id != self.candidate_evidence.candidate_id
            or self.assessment.observation_id != self.sample_observation.observation_id
            or self.admission.candidate_id != self.candidate_evidence.candidate_id
            or self.admission.assessment_id != self.assessment.assessment_id
            or self.admission.observation_id != self.sample_observation.observation_id
            or self.evidence_state.observation_id
            != self.sample_observation.observation_id
            or self.evidence_state.candidate_id != self.candidate_evidence.candidate_id
            or self.evidence_state.assessment_id != self.assessment.assessment_id
            or self.evidence_state.admission_id != self.admission.admission_id
            or self.evidence_state.evidence_admission != self.admission.status
        ):
            raise ValueError("Investigation evidence aggregate changed exact lineage")
        expected = assess_investigation_observation(
            authority=self.authority,
            work_item=self.work_item,
            observation=self.sample_observation,
            inherited_model_contact_count=(
                self.evidence_state.inherited_model_contact_count
            ),
            inherited_source_contact_count=(
                self.evidence_state.inherited_source_contact_count
            ),
        )
        if expected != (
            self.candidate_evidence,
            self.assessment,
            self.admission,
            self.evidence_state,
        ):
            raise ValueError("Investigation evidence changed deterministic policy")
        return self


InvestigationCollectionItemStatus = Literal[
    "approval_pending",
    "observation_pending",
    "not_started",
    "active",
    "completed",
    "negative",
    "unresolved",
    "refused",
    "failed",
    "stopped",
]


class InvestigationCollectionItemState(FrozenContract):
    work_item_id: StrictStr = Field(pattern=r"^work-item:[0-9a-f]{64}$")
    selected_ordinal: StrictInt = Field(gt=0)
    status: InvestigationCollectionItemStatus
    evidence_state_id: StrictStr | None = Field(
        default=None, pattern=r"^investigation-evidence-state:[0-9a-f]{64}$"
    )
    outcome: StrictStr | None = None
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def exact_terminal_shape(self) -> "InvestigationCollectionItemState":
        retained_evidence = self.status in {
            "completed",
            "negative",
            "unresolved",
            "refused",
        } or (self.status == "failed" and self.evidence_state_id is not None)
        if retained_evidence != (
            self.evidence_state_id is not None and self.outcome is not None
        ):
            raise ValueError("collection item changed its retained evidence state")
        if self.status in {"failed", "stopped"} and self.reason is None:
            raise ValueError("failed or stopped collection item requires a reason")
        if retained_evidence and self.reason is not None and self.status != "failed":
            raise ValueError("evidence-terminal collection item cannot carry a reason")
        return self


InvestigationCollectionCombineStatus = Literal[
    "complete",
    "partial",
    "stopped_before_start",
    "stopped_after_partial",
    "failed_to_combine",
]


def _collection_combine_status(
    item_states: tuple[InvestigationCollectionItemState, ...],
) -> InvestigationCollectionCombineStatus:
    statuses = tuple(item.status for item in item_states)
    if any(
        status in {"approval_pending", "observation_pending"} for status in statuses
    ):
        return "partial"
    if all(status in {"completed", "negative", "unresolved"} for status in statuses):
        return "complete"
    if all(status in {"not_started", "stopped"} for status in statuses):
        return "stopped_before_start"
    if "stopped" in statuses or "not_started" in statuses:
        return "stopped_after_partial"
    return "partial"


class InvestigationCollectionEvidenceRecorded(FrozenContract):
    event_type: Literal["InvestigationCollectionEvidenceRecorded"] = (
        "InvestigationCollectionEvidenceRecorded"
    )
    command: InvestigationObservationCommand
    authority: InvestigationAuthority
    selected_work_item_ids: tuple[StrictStr, ...]
    collection_work_item_ids: tuple[StrictStr, ...] = ()
    collection_proposal_id: StrictStr | None = None
    carried_revision_id: StrictStr | None = Field(
        default=None, pattern=r"^investigation-revision:[0-9a-f]{64}$"
    )
    previous_collection_state_id: StrictStr | None = None
    item_states: tuple[InvestigationCollectionItemState, ...] = Field(min_length=1)
    combine_status: InvestigationCollectionCombineStatus
    work_policy: InvestigationWorkPolicy = Field(
        default_factory=InvestigationWorkPolicy
    )
    work_policy_id: StrictStr | None = None

    @property
    def failed(self) -> bool:
        return any(item.status == "failed" for item in self.item_states)

    @computed_field(return_type=str)
    @property
    def collection_state_id(self) -> str:
        return "investigation-collection-evidence:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"collection_state_id"})
        )

    @model_validator(mode="after")
    def exact_collection_fold(self) -> "InvestigationCollectionEvidenceRecorded":
        expected_ids = (
            ()
            if self.carried_revision_id
            else tuple(
                item.work_item_id for item in self.authority.authorized_work_items
            )
        )
        if (
            self.command.authority_id != self.authority.authority_id
            or self.selected_work_item_ids != expected_ids
            or tuple(item.work_item_id for item in self.item_states)
            != (self.collection_work_item_ids or expected_ids)
            or tuple(item.selected_ordinal for item in self.item_states)
            != tuple(range(1, len(self.item_states) + 1))
            or not set(expected_ids)
            <= set(self.collection_work_item_ids or expected_ids)
            or len(set(self.collection_work_item_ids))
            != len(self.collection_work_item_ids)
        ):
            raise ValueError("collection fold changed selected Work-item order")
        if self.combine_status == "failed_to_combine":
            if not all(
                item.status == "failed" and item.reason == "failed_to_combine"
                for item in self.item_states
            ):
                raise ValueError("failed collection combine changed its exact reason")
            return self
        expected_status = _collection_combine_status(self.item_states)
        if self.combine_status != expected_status:
            raise ValueError("collection combine status changed deterministic fold")
        return self


def fold_investigation_collection(
    *,
    command: InvestigationObservationCommand,
    authority: InvestigationAuthority,
    item_states: tuple[InvestigationCollectionItemState, ...],
    collection_work_item_ids: tuple[str, ...] = (),
    collection_proposal_id: str | None = None,
    carried_revision_id: str | None = None,
    previous_collection_state_id: str | None = None,
    work_policy: InvestigationWorkPolicy | None = None,
    work_policy_id: str | None = None,
) -> InvestigationCollectionEvidenceRecorded:
    """Derive the total collection status from exact selected-order item states."""

    selected_ids = (
        ()
        if carried_revision_id
        else tuple(item.work_item_id for item in authority.authorized_work_items)
    )
    accounted_ids = collection_work_item_ids or selected_ids
    exact_order = tuple(
        item.work_item_id for item in item_states
    ) == accounted_ids and tuple(
        item.selected_ordinal for item in item_states
    ) == tuple(range(1, len(accounted_ids) + 1))
    if not exact_order:
        item_states = tuple(
            InvestigationCollectionItemState(
                work_item_id=work_item_id,
                selected_ordinal=ordinal,
                status="failed",
                reason="failed_to_combine",
            )
            for ordinal, work_item_id in enumerate(selected_ids, start=1)
        )
        combine_status = "failed_to_combine"
    else:
        combine_status = _collection_combine_status(item_states)
    return InvestigationCollectionEvidenceRecorded(
        command=command,
        authority=authority,
        selected_work_item_ids=selected_ids,
        collection_work_item_ids=collection_work_item_ids,
        collection_proposal_id=collection_proposal_id,
        carried_revision_id=carried_revision_id,
        previous_collection_state_id=previous_collection_state_id,
        item_states=item_states,
        combine_status=combine_status,
        work_policy=work_policy or InvestigationWorkPolicy(),
        work_policy_id=work_policy_id,
    )


def correlate_investigation_collection_evidence(
    collection: InvestigationCollectionEvidenceRecorded,
    evidence_records: tuple[InvestigationEvidenceRecorded, ...],
) -> tuple[InvestigationEvidenceRecorded, ...]:
    """Bind each retained member state to its exact identity-correlated evidence."""

    retained_by_id: dict[str, InvestigationEvidenceRecorded] = {}
    for evidence in evidence_records:
        work_item_id = evidence.work_item.work_item_id
        if work_item_id in retained_by_id:
            raise ValueError("Investigation collection repeated a Work item")
        retained_by_id[work_item_id] = evidence
    states_by_id = {
        item.work_item_id: item
        for item in collection.item_states
        if item.evidence_state_id is not None
    }
    if set(retained_by_id) != set(states_by_id):
        raise ValueError("Investigation collection evidence membership changed")
    expected_status = {
        "supported": "completed",
        "contradicted": "negative",
        "refused": "refused",
    }
    for work_item_id, item_state in states_by_id.items():
        evidence = retained_by_id[work_item_id]
        state = evidence.evidence_state
        failure_reason = observation_failure_reason(evidence.sample_observation)
        if (
            (
                work_item_id in collection.selected_work_item_ids
                and (
                    evidence.command != collection.command
                    or evidence.authority != collection.authority
                )
            )
            or item_state.evidence_state_id != state.evidence_state_id
            or item_state.outcome != state.outcome
            or item_state.status
            != (
                "failed"
                if failure_reason
                else expected_status.get(state.work_item_disposition, "unresolved")
            )
            or item_state.reason != failure_reason
        ):
            raise ValueError("Investigation collection evidence membership changed")
    return tuple(
        retained_by_id[work_item_id]
        for work_item_id in (
            collection.collection_work_item_ids or collection.selected_work_item_ids
        )
        if work_item_id in retained_by_id
    )


def observation_failure_reason(observation: SampleObservation) -> str | None:
    """Separate shared preparation/integrity failure from item-local refusal."""
    if isinstance(observation.preparation, VcfPreparationFailed):
        return f"shared_preparation:{observation.preparation.code}"
    result = observation.result
    if isinstance(result, VariantObservationRefused) and (
        "continuation_binding" in result.violations
        or "sample_identity" in result.violations
    ):
        return f"shared_observation:{result.code}"
    return None


def _policy(
    result: VariantObservationResult,
) -> tuple[Literal["supported", "unsupported", "refused"], WorkItemDisposition, bool]:
    if isinstance(result, ExactAlleleObserved):
        return "supported", "supported", True
    if isinstance(result, TargetAlleleNotCarried):
        return "supported", "contradicted", True
    if isinstance(result, TargetGenotypeUnresolved):
        return "unsupported", "unresolved", False
    if isinstance(result, (LocusIdentityMismatch, LocusOverlapped)):
        return "supported", "unresolved", True
    if isinstance(result, NoRecordObserved):
        return "supported", "unresolved", True
    if isinstance(result, AmbiguousRecordsObserved):
        return "unsupported", "partial", False
    if isinstance(result, VariantObservationRefused):
        return "refused", "refused", False
    if isinstance(result, VcfRegionObserved):
        # The records a region holds, each its own row: no finding is admitted.
        return "supported", "partial", False
    raise TypeError("observation result union is not exhaustive")


def _observation_detail(result: VariantObservationResult, sample_id: str) -> str:
    target = result.target
    if target is None:
        raise ValueError("Observation claim lacks its exact target")
    if isinstance(result, VcfRegionObserved):
        counted = (
            f", and {result.not_reported} more not reported"
            if result.not_reported
            else ""
        )
        return (
            f"Sample {sample_id}'s VCF holds {len(result.records)} record"
            f"{'' if len(result.records) == 1 else 's'} in {target.assembly}:"
            f"{target.contig}:{target.start}-{target.end}{counted}."
        )
    if isinstance(target, ObserveRegionInput):
        return f"Sample {sample_id}'s region {target.contig}:{target.start}-{target.end} was refused."
    locus = (
        f"{target.assembly}:{target.contig}:{target.one_based_position} "
        f"{allele_text(target.ref, target.alt)}"
    )
    if isinstance(result, ExactAlleleObserved):
        return (
            f"Sample {sample_id} carries the target ALT at {locus} with genotype "
            f"{result.genotype.raw_genotype}."
        )
    if isinstance(result, TargetAlleleNotCarried):
        if result.genotype.target_alt_index is None:
            return (
                f"Sample {sample_id} does not carry the target ALT at {locus}; the "
                f"record at this REF lists {'/'.join(result.record.alts)}; genotype "
                f"{result.genotype.raw_genotype}."
            )
        return (
            f"Sample {sample_id} does not carry the target ALT at {locus} in the "
            f"observed record; genotype {result.genotype.raw_genotype}."
        )
    if isinstance(result, TargetGenotypeUnresolved):
        return (
            f"Sample {sample_id} has unresolved target genotype "
            f"{result.genotype.raw_genotype} at {locus}."
        )
    if isinstance(result, LocusIdentityMismatch):
        identities = ", ".join(
            f"{record.ref}>{'/'.join(record.alts)}" for record in result.records
        )
        return (
            f"Sample {sample_id} has locus record identity {identities}, not {locus}."
        )
    if isinstance(result, LocusOverlapped):
        spans = ", ".join(
            f"{record.one_based_position} {record.ref}>{'/'.join(record.alts)}"
            for record in result.records
        )
        return (
            f"Sample {sample_id} has records starting upstream that span {locus}: "
            f"{spans}."
        )
    if isinstance(result, NoRecordObserved):
        return f"No record was observed for Sample {sample_id} at {locus}."
    if isinstance(result, AmbiguousRecordsObserved):
        return (
            f"Sample {sample_id} has {len(result.matching_records)} exact records at "
            f"{locus}; no unique observation was selected."
        )
    if isinstance(result, VariantObservationRefused):
        return (
            f"Observation for Sample {sample_id} at {locus} was refused: {result.code}."
        )
    raise TypeError("observation result union is not exhaustive")


def assess_investigation_observation(
    *,
    authority: InvestigationAuthority,
    work_item: WorkItem,
    observation: SampleObservation,
    inherited_model_contact_count: int,
    inherited_source_contact_count: int,
) -> tuple[
    CandidateEvidence,
    ObservationEvidenceAssessment,
    ObservationAdmissionDecision,
    InvestigationEvidenceState,
]:
    """Fold one exact Tool result through the closed claim-admission table."""

    result = observation.result
    verdict, disposition, admits = _policy(result)
    claim = VcfLocalObservationClaim(
        outcome=result.outcome,
        sample_id=observation.sample_id,
        assembly_accession=observation.assembly_accession,
        assembly_name=observation.assembly_name,
        target=result.target,
        vcf_descriptor=observation.vcf_descriptor,
        observation_detail=_observation_detail(result, observation.sample_id),
        claim_scope=result.claim_scope,
        observed_records=observed_records(result),
    )
    # Worded by each qualification's typed origin, as Finalization reads it.
    origins = observation_origins(result.outcome, observation.run_provenance, admits)
    limitations = tuple(origin.audit() for origin in origins["limitation"])
    candidate = CandidateEvidence(
        investigation_authority_id=authority.authority_id,
        work_item_id=work_item.work_item_id,
        observation_id=observation.observation_id,
        outcome=result.outcome,
        proposed_claim=claim,
        claim_scope=result.claim_scope,
        limitations=limitations,
    )
    preparation_id = (
        observation.preparation.receipt_id
        if isinstance(observation.preparation, PreparedVcfObservationReceipt)
        else (
            "vcf-observation-preparation-failure:"
            + canonical_json_digest(observation.preparation.model_dump(mode="json"))
        )
    )
    assessment = ObservationEvidenceAssessment(
        investigation_authority_id=authority.authority_id,
        work_item_id=work_item.work_item_id,
        candidate_id=candidate.candidate_id,
        observation_id=observation.observation_id,
        outcome=result.outcome,
        verdict=verdict,
        work_item_disposition=disposition,
        basis=(authority.authority_id, preparation_id, observation.observation_id),
        claim_scope=result.claim_scope,
        limitations=limitations,
        contradictions=tuple(origin.audit() for origin in origins["conflict"]),
        applicability=(
            "Only the approved Work item, sample, assembly, target, and immutable VCF identity.",
        ),
        supports_admission=admits,
    )
    common: dict[str, Any] = {
        "investigation_authority_id": authority.authority_id,
        "work_item_id": work_item.work_item_id,
        "candidate_id": candidate.candidate_id,
        "assessment_id": assessment.assessment_id,
        "observation_id": observation.observation_id,
    }
    if admits:
        admission: ObservationAdmissionDecision = AdmittedObservationEvidence(
            admitted_claim=candidate.proposed_claim,
            claim_scope=candidate.claim_scope,
            **common,
        )
    else:
        admission = ObservationEvidenceNotAdmitted(
            reason=(
                "Refused Tool evidence supports no scientific claim."
                if isinstance(result, VariantObservationRefused)
                else "The exact Tool outcome does not support automatic admission."
            ),
            claim_scope=candidate.claim_scope,
            **common,
        )
    state = InvestigationEvidenceState(
        investigation_authority_id=authority.authority_id,
        proposal_id=authority.proposal_id,
        review_response_id=authority.review_response_id,
        work_item_id=work_item.work_item_id,
        observation_id=observation.observation_id,
        candidate_id=candidate.candidate_id,
        assessment_id=assessment.assessment_id,
        admission_id=admission.admission_id,
        outcome=result.outcome,
        work_item_disposition=disposition,
        evidence_admission=admission.status,
        limitations=limitations,
        conflicts=assessment.contradictions,
        gaps=tuple(origin.audit() for origin in origins["gap"]),
        inherited_model_contact_count=inherited_model_contact_count,
        inherited_source_contact_count=inherited_source_contact_count,
        run_provenance=observation.run_provenance,
        provenance_origin_event_id=observation.provenance_origin_event_id,
    )
    return candidate, assessment, admission, state


def validate_investigation_evidence_fact(
    event: Mapping[str, Any],
) -> InvestigationEvidenceRecorded | None:
    if event.get("event_type") != "InvestigationEvidenceRecorded":
        return None
    return InvestigationEvidenceRecorded.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def validate_investigation_work_item_started_fact(
    event: Mapping[str, Any],
) -> InvestigationWorkItemStarted | None:
    if event.get("event_type") != "InvestigationWorkItemStarted":
        return None
    return InvestigationWorkItemStarted.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def validate_investigation_collection_evidence_fact(
    event: Mapping[str, Any],
) -> InvestigationCollectionEvidenceRecorded | None:
    if event.get("event_type") != "InvestigationCollectionEvidenceRecorded":
        return None
    return InvestigationCollectionEvidenceRecorded.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


__all__ = [
    "AdmittedObservationEvidence",
    "CandidateEvidence",
    "InvestigationCollectionEvidenceRecorded",
    "InvestigationCollectionItemState",
    "InvestigationEvidenceRecorded",
    "InvestigationEvidenceState",
    "InvestigationObservationCommand",
    "InvestigationWorkItemStarted",
    "ObservationAdmissionDecision",
    "ObservationEvidenceAssessment",
    "ObservationEvidenceNotAdmitted",
    "RunProvenance",
    "SampleObservation",
    "VcfLocalObservationClaim",
    "WorkItemDisposition",
    "assess_investigation_observation",
    "correlate_investigation_collection_evidence",
    "derive_investigation_run_provenance",
    "fold_investigation_collection",
    "validate_run_provenance",
    "validate_investigation_evidence_fact",
    "validate_investigation_collection_evidence_fact",
    "validate_investigation_work_item_started_fact",
]
