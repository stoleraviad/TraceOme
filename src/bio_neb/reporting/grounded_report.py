"""Deterministic Grounded-report contract derived from exact Report authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from bio_neb.application.planning_read import PrefixSnapshot
from bio_neb.content import ContentId
from bio_neb.contracts import FrozenContract
from bio_neb.investigation.collection_view import CollectionView
from bio_neb.investigation.investigation import (
    EvidenceAssessment,
    InvestigationIssueItem,
    RetrievedSource,
    UserAssertion,
    WorkItem,
)
from bio_neb.investigation.investigation_observation import (
    CandidateEvidence,
    InvestigationCollectionCombineStatus,
    InvestigationCollectionItemState,
    ObservationAdmissionDecision,
    ObservationEvidenceAssessment,
    RunProvenance,
    validate_run_provenance,
)
from bio_neb.investigation.run_health import RunHealthSection
from bio_neb.investigation.vcf_observation import (
    ObservedVcfRecord,
    ObserveRegionInput,
    ObserveVariantInput,
    RegionRecord,
    VariantObservationResult,
    VcfRegionObserved,
)
from bio_neb.journey import (
    ArtifactReference,
    JourneyEvent,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.paper_reading.report import ReportedPaper
from bio_neb.parabricks.run_method import RunMethod
from bio_neb.reporting.finalization import (
    FinalizationAccountingEntry,
    FinalizationEvidencePointer,
    FinalizationReviewReceipt,
    Qualification,
    ReportBasis,
    ReportedFactSlot,
    ReportMembershipDecision,
)

REPORT_MEDIA_TYPE = "application/vnd.bio-neb.grounded-report+json"
REPORT_RENDERER_VERSION = "bio-neb.grounded-report-renderer/2"


MemberValue = TypeVar("MemberValue", bound=BaseModel)


class GroundedReportMember(FrozenContract, Generic[MemberValue]):
    membership: ReportMembershipDecision
    value: MemberValue


class GroundedTargetGenotypeAssessment(FrozenContract):
    sample_name: StrictStr
    raw_genotype: StrictStr
    target_alt_index: StrictInt | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    state: Literal[
        "target_alt_carried",
        "target_alt_not_carried",
        "no_call",
        "invalid",
    ]
    phased: StrictBool | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    allele_indexes: tuple[StrictInt | None, ...] = Field(min_length=1)
    reason: Literal["missing_allele", "malformed", "out_of_range"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class GroundedExactRecordObservation(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"]
    outcome: Literal[
        "exact_allele_observed",
        "target_allele_not_carried",
        "target_genotype_unresolved",
    ]
    target: ObserveVariantInput
    record: ObservedVcfRecord
    genotype: GroundedTargetGenotypeAssessment
    other_records: tuple[ObservedVcfRecord, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    claim_scope: StrictStr


class GroundedLocusMismatchObservation(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"]
    outcome: Literal["locus_present_identity_mismatch", "locus_overlapped"]
    target: ObserveVariantInput
    records: tuple[ObservedVcfRecord, ...] = Field(min_length=1)
    claim_scope: StrictStr


class GroundedNoRecordObservation(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"]
    outcome: Literal["no_record"]
    target: ObserveVariantInput
    claim_scope: StrictStr


class GroundedAmbiguousObservation(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"]
    outcome: Literal["ambiguous_records"]
    target: ObserveVariantInput
    matching_records: tuple[ObservedVcfRecord, ...] = Field(min_length=2)
    claim_scope: StrictStr


class GroundedRefusedObservation(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"]
    outcome: Literal["refused"]
    target: ObserveVariantInput | ObserveRegionInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    code: Literal[
        "invalid_target",
        "target_unresolved",
        "tool_failure",
    ]
    violations: tuple[StrictStr, ...]
    claim_scope: StrictStr


class GroundedRegionObservation(FrozenContract):
    """A region's records, each with its REF check, as the report carries them."""

    schema_version: Literal["bio-neb.vcf-region-observation/1"]
    outcome: Literal["region_observed"]
    target: ObserveRegionInput
    region_records: tuple[RegionRecord, ...] = ()
    not_reported: StrictInt = Field(default=0, ge=0)
    claim_scope: StrictStr


GroundedObservationResult = Annotated[
    GroundedExactRecordObservation
    | GroundedLocusMismatchObservation
    | GroundedNoRecordObservation
    | GroundedAmbiguousObservation
    | GroundedRefusedObservation
    | GroundedRegionObservation,
    Field(discriminator="outcome"),
]
_GROUNDED_OBSERVATION_RESULT = TypeAdapter(GroundedObservationResult)
_REPORT_SAFE_OBSERVATION_FIELDS = (
    "schema_version",
    "outcome",
    "target",
    "record",
    "other_records",
    "records",
    "matching_records",
    "genotype",
    "region_records",
    "not_reported",
    "code",
    "violations",
    "claim_scope",
)


class GroundedSampleObservation(FrozenContract):
    observation_id: StrictStr
    investigation_authority_id: StrictStr
    proposal_id: StrictStr
    review_response_id: StrictStr
    work_item_id: StrictStr
    sample_id: StrictStr
    assembly_accession: StrictStr
    assembly_name: StrictStr
    result: GroundedObservationResult
    materialization_receipt_id: StrictStr
    materialization_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    run_provenance: RunProvenance
    provenance_origin_event_id: StrictStr | None = Field(min_length=1)

    @model_validator(mode="after")
    def exact_provenance(self) -> "GroundedSampleObservation":
        validate_run_provenance(self.run_provenance, self.provenance_origin_event_id)
        return self


class GroundedReport(FrozenContract):
    schema_version: Literal["bio-neb.grounded-report/7"] = "bio-neb.grounded-report/7"
    renderer_version: Literal["bio-neb.grounded-report-renderer/2"] = (
        REPORT_RENDERER_VERSION
    )
    report_digest: StrictStr = Field(pattern=r"^grounded-report:[0-9a-f]{64}$")
    journey_id: StrictStr
    source_head_event_id: StrictStr
    report_basis_digest: StrictStr = Field(pattern=r"^report-basis:[0-9a-f]{64}$")
    finalization_basis_digest: StrictStr = Field(
        pattern=r"^finalization-basis:[0-9a-f]{64}$"
    )
    outcome: StrictStr = Field(min_length=1)
    accounting: tuple[GroundedReportMember[FinalizationAccountingEntry], ...] = Field(
        min_length=1
    )
    fact_slots: tuple[ReportedFactSlot, ...] = ()
    # The basis's issue SNPs, each listed as such; never a scientific fact.
    issue_items: tuple[InvestigationIssueItem, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # The basis's per-SNP view and cross-SNP cases; never a fact.
    collection_view: CollectionView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The basis's Run health and depth lines; never a fact.
    run_health: RunHealthSection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The basis's method: how the Run's VCF was made; never a fact.
    method: RunMethod | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The basis's papers: what each paper read in the Run found.
    papers: tuple[ReportedPaper, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    evidence_state_id: StrictStr | None
    selected_work_item_ids: tuple[StrictStr, ...]
    collection_item_states: tuple[InvestigationCollectionItemState, ...]
    collection_status: InvestigationCollectionCombineStatus | None
    review_authority: FinalizationReviewReceipt
    question: GroundedReportMember[UserAssertion] | None
    work_items: tuple[GroundedReportMember[WorkItem], ...]
    retrieved_sources: tuple[GroundedReportMember[RetrievedSource], ...]
    source_assessments: tuple[GroundedReportMember[EvidenceAssessment], ...]
    sample_observations: tuple[GroundedReportMember[GroundedSampleObservation], ...]
    candidate_evidence: tuple[GroundedReportMember[CandidateEvidence], ...]
    observation_assessments: tuple[
        GroundedReportMember[ObservationEvidenceAssessment], ...
    ]
    admissions: tuple[GroundedReportMember[ObservationAdmissionDecision], ...]
    report_membership: tuple[ReportMembershipDecision, ...]
    conflicts: tuple[Qualification, ...]
    limitations: tuple[Qualification, ...]
    gaps: tuple[Qualification, ...]
    evidence_pointers: tuple[FinalizationEvidencePointer, ...]
    inherited_model_contact_count: StrictInt = Field(ge=0)
    inherited_source_contact_count: StrictInt = Field(ge=0)
    report_external_contact_count: Literal[0] = 0
    run_provenance: RunProvenance = "unestablished"
    provenance_origin_event_id: StrictStr | None = Field(default=None, min_length=1)

    def all_members(self) -> tuple[GroundedReportMember[Any], ...]:
        return (
            *self.accounting,
            *((self.question,) if self.question is not None else ()),
            *self.work_items,
            *self.retrieved_sources,
            *self.source_assessments,
            *self.sample_observations,
            *self.candidate_evidence,
            *self.observation_assessments,
            *self.admissions,
        )

    def canonical_bytes(self) -> bytes:
        return (
            canonical_json(self.model_dump(mode="json", exclude_computed_fields=True))
            + "\n"
        ).encode("utf-8")

    @model_validator(mode="after")
    def exhaustive_stable_projection(self) -> "GroundedReport":
        validate_run_provenance(self.run_provenance, self.provenance_origin_event_id)
        sections = (
            self.accounting,
            self.work_items,
            self.retrieved_sources,
            self.source_assessments,
            self.sample_observations,
            self.candidate_evidence,
            self.observation_assessments,
            self.admissions,
        )
        if any(
            tuple(item.membership.member_id for item in section)
            != tuple(sorted(item.membership.member_id for item in section))
            for section in sections
        ):
            raise ValueError("Grounded-report member order changed")
        retained = tuple(item.member_id for item in self.report_membership)
        if retained != tuple(sorted(retained)) or len(retained) != len(set(retained)):
            raise ValueError("Grounded-report membership order changed")
        projected = tuple(item.membership.member_id for item in self.all_members())
        if set(projected) != set(retained) or len(projected) != len(retained):
            raise ValueError("Grounded report changed exhaustive Report membership")
        decisions = {item.member_id: item for item in self.report_membership}
        if any(
            item.membership != decisions[item.membership.member_id]
            for item in self.all_members()
        ):
            raise ValueError("Grounded report changed a Report-membership decision")
        if any(
            item.membership.member_role != "accounting"
            or item.membership.admitted
            or item.membership.member_id != item.value.accounting_id
            for item in self.accounting
        ):
            raise ValueError("Grounded report admitted or changed Run accounting")
        if tuple(
            item.work_item_id for item in self.collection_item_states
        ) != self.selected_work_item_ids or set(self.selected_work_item_ids) != {
            item.value.work_item_id for item in self.work_items
        }:
            raise ValueError("Grounded report changed collection membership")
        for kind, values in (
            ("conflict", self.conflicts),
            ("limitation", self.limitations),
            ("gap", self.gaps),
        ):
            texts = tuple(item.text for item in values)
            if texts != tuple(sorted(set(texts))):
                raise ValueError(f"Grounded-report {kind} order changed")
        if tuple(item.pointer_id for item in self.evidence_pointers) != tuple(
            sorted(item.pointer_id for item in self.evidence_pointers)
        ):
            raise ValueError("Grounded-report pointer order changed")
        material = self.model_dump(
            mode="json",
            exclude={"report_digest"},
            exclude_computed_fields=True,
        )
        if self.report_digest != ("grounded-report:" + canonical_json_digest(material)):
            raise ValueError("Grounded-report identity changed")
        return self


class GroundedReportPublished(FrozenContract):
    event_type: Literal["GroundedReportPublished"] = "GroundedReportPublished"
    report: GroundedReport
    report_external_contact_count: Literal[0] = 0


class GroundedReportProjection(FrozenContract):
    status: Literal["not_published", "published"]
    report: GroundedReport | None = None
    artifact: ArtifactReference | None = None
    publication_event_id: StrictStr | None = None
    report_external_contact_count: Literal[0] = 0

    @model_validator(mode="after")
    def exact_shape(self) -> "GroundedReportProjection":
        published = self.status == "published"
        if published != all(
            value is not None
            for value in (self.report, self.artifact, self.publication_event_id)
        ):
            raise ValueError("Grounded-report projection shape changed")
        return self


def _as_dict(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def _member(
    membership: Mapping[str, ReportMembershipDecision],
    member_id: str,
    value: MemberValue,
) -> GroundedReportMember[MemberValue]:
    return GroundedReportMember[MemberValue](
        membership=membership[member_id], value=value
    )


def _grounded_observation_result(
    result: VariantObservationResult,
) -> GroundedObservationResult:
    source = result.model_dump(mode="json", exclude_none=True)
    if isinstance(result, VcfRegionObserved):
        source["region_records"] = source.pop("records", [])
    return _GROUNDED_OBSERVATION_RESULT.validate_python(
        {
            field: source[field]
            for field in _REPORT_SAFE_OBSERVATION_FIELDS
            if field in source
        }
    )


def derive_grounded_report(
    report_basis: ReportBasis,
    *,
    journey_id: str,
    source_head_event_id: str,
) -> GroundedReport:
    """Exhaustively project one exact approved Report basis without contact."""

    basis = report_basis.finalization_basis
    membership = {item.member_id: item for item in basis.report_membership}
    sample_observations = []
    for observation in basis.sample_observations:
        sample_observations.append(
            _member(
                membership,
                observation.observation_id,
                GroundedSampleObservation(
                    observation_id=observation.observation_id,
                    investigation_authority_id=(observation.investigation_authority_id),
                    proposal_id=observation.proposal_id,
                    review_response_id=observation.review_response_id,
                    work_item_id=observation.work_item_id,
                    sample_id=observation.sample_id,
                    assembly_accession=observation.assembly_accession,
                    assembly_name=observation.assembly_name,
                    result=_grounded_observation_result(observation.result),
                    materialization_receipt_id=(observation.materialization_receipt_id),
                    materialization_receipt_sha256=(
                        observation.materialization_receipt_sha256
                    ),
                    run_provenance=observation.run_provenance,
                    provenance_origin_event_id=observation.provenance_origin_event_id,
                ),
            )
        )
    payload: dict[str, Any] = {
        "schema_version": "bio-neb.grounded-report/7",
        "renderer_version": REPORT_RENDERER_VERSION,
        "journey_id": journey_id,
        "source_head_event_id": source_head_event_id,
        "report_basis_digest": report_basis.report_basis_digest,
        "finalization_basis_digest": basis.basis_digest,
        "outcome": basis.outcome,
        "accounting": tuple(
            _member(membership, item.accounting_id, item) for item in basis.accounting
        ),
        "fact_slots": basis.fact_slots,
        "issue_items": basis.issue_items,
        "collection_view": basis.collection_view,
        "run_health": basis.run_health,
        "method": basis.method,
        "papers": basis.papers,
        "evidence_state_id": basis.evidence_state_id,
        "selected_work_item_ids": basis.selected_work_item_ids,
        "collection_item_states": basis.collection_item_states,
        "collection_status": basis.collection_status,
        "review_authority": report_basis.approval,
        "question": (
            _member(membership, basis.question.assertion_id, basis.question)
            if basis.question is not None
            else None
        ),
        "work_items": tuple(
            _member(membership, item.work_item_id, item) for item in basis.work_items
        ),
        "retrieved_sources": tuple(
            _member(membership, item.source_id, item)
            for item in basis.retrieved_sources
        ),
        "source_assessments": tuple(
            _member(membership, item.assessment_id, item)
            for item in basis.source_assessments
        ),
        "sample_observations": tuple(sample_observations),
        "candidate_evidence": tuple(
            _member(membership, item.candidate_id, item)
            for item in basis.candidate_evidence
        ),
        "observation_assessments": tuple(
            _member(membership, item.assessment_id, item)
            for item in basis.observation_assessments
        ),
        "admissions": tuple(
            _member(membership, item.admission_id, item) for item in basis.admissions
        ),
        "report_membership": basis.report_membership,
        "conflicts": basis.conflicts,
        "limitations": basis.limitations,
        "gaps": basis.gaps,
        "evidence_pointers": basis.evidence_pointers,
        "inherited_model_contact_count": basis.inherited_model_contact_count,
        "inherited_source_contact_count": basis.inherited_source_contact_count,
        "report_external_contact_count": 0,
        "run_provenance": basis.run_provenance,
        "provenance_origin_event_id": basis.provenance_origin_event_id,
    }
    material = GroundedReport.model_construct(
        report_digest="grounded-report:" + "0" * 64, **payload
    ).model_dump(mode="json", exclude={"report_digest"}, exclude_computed_fields=True)
    return GroundedReport.model_validate(
        {
            "report_digest": ("grounded-report:" + canonical_json_digest(material)),
            **payload,
        }
    )


def validate_grounded_report_fact(
    event: Mapping[str, Any],
) -> tuple[GroundedReportPublished, ArtifactReference] | None:
    if event.get("event_type") != "GroundedReportPublished":
        return None
    fact = GroundedReportPublished.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )
    references = tuple(event.get("artifact_references", ()))
    if len(references) != 1:
        raise ValueError("Grounded report requires one retained artifact")
    reference = ArtifactReference.model_validate(references[0])
    report_bytes = fact.report.canonical_bytes()
    expected_artifact_id = str(ContentId.of(report_bytes))
    if (
        reference.artifact_id != expected_artifact_id
        or reference.byte_count != len(report_bytes)
        or reference.media_type != REPORT_MEDIA_TYPE
        or reference.journey_id != event.get("journey_id")
        or reference.source_event_id != event.get("event_id")
    ):
        raise ValueError("Grounded-report artifact binding changed")
    return fact, reference


def project_grounded_report(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    report_digest: str | None = None,
    snapshot: PrefixSnapshot | None = None,
) -> GroundedReportProjection:
    """``snapshot`` is the prefix ``events`` already holds, if the caller has one."""
    timeline = [_as_dict(event) for event in events]
    snapshot = snapshot or PrefixSnapshot(timeline)
    publications = [
        (index, event)
        for index, event in enumerate(timeline)
        if event.get("event_type") == "GroundedReportPublished"
    ]
    projections = []
    for index, event in publications:
        # A publication is fixed by the prefix through it: checked once per owner.
        projections.append(
            snapshot.checked_once(
                ("report_publication", event["event_id"]),
                lambda index=index, event=event: _project_report_publication(
                    timeline[: index + 1],
                    publication_event_id=event["event_id"],
                    snapshot=snapshot.through(event["event_id"]),
                ),
            )
        )
    bases = [
        item.report.report_basis_digest
        for item in projections
        if item.report is not None
    ]
    if len(set(bases)) != len(bases):
        raise ValueError("Grounded report was published more than once for a basis")
    if report_digest is not None:
        return next(
            (
                item
                for item in projections
                if item.report is not None
                and item.report.report_digest == report_digest
            ),
            GroundedReportProjection(status="not_published"),
        )
    return (
        projections[-1]
        if projections
        else GroundedReportProjection(status="not_published")
    )


def _project_report_publication(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    publication_event_id: str,
    snapshot: PrefixSnapshot,
) -> GroundedReportProjection:
    timeline = [_as_dict(event) for event in events]
    retained = [
        (event, validated)
        for event in timeline
        if (validated := validate_grounded_report_fact(event)) is not None
        and event.get("event_id") == publication_event_id
    ]
    if not retained:
        return GroundedReportProjection(status="not_published")
    if len(retained) != 1:
        raise ValueError("Grounded report was published more than once")
    event, (fact, reference) = retained[0]
    approved = snapshot.investigation.completion.approved_report
    if approved is None:
        raise ValueError("Grounded report lacks approved Report basis")
    report = fact.report
    if (
        report.source_head_event_id != approved.approval_event_id
        or event.get("parent_ids") != [report.source_head_event_id]
        or event.get("journey_id") != report.journey_id
        or report.report_basis_digest != approved.basis.report_basis_digest
        or report
        != derive_grounded_report(
            approved.basis,
            journey_id=event["journey_id"],
            source_head_event_id=report.source_head_event_id,
        )
    ):
        raise ValueError("Grounded report changed exact Report-basis lineage")
    return GroundedReportProjection(
        status="published",
        report=report,
        artifact=reference,
        publication_event_id=event["event_id"],
    )


__all__ = [
    "GroundedObservationResult",
    "GroundedReport",
    "GroundedReportMember",
    "GroundedReportProjection",
    "GroundedReportPublished",
    "GroundedSampleObservation",
    "REPORT_MEDIA_TYPE",
    "REPORT_RENDERER_VERSION",
    "derive_grounded_report",
    "project_grounded_report",
    "validate_grounded_report_fact",
]
