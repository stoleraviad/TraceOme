"""Immutable Finalization basis and exact Report-basis review authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    model_validator,
)

from bio_neb.agentic_journey.continuation_policy import (
    FinalizationReviewPolicy,
    explicit_review_policy,
)
from bio_neb.application.investigation_read import FinalizationCandidate
from bio_neb.application.planning_read import PrefixSnapshot
from bio_neb.content import ContentId
from bio_neb.contracts import FrozenContract
from bio_neb.investigation.collection_view import (
    CollectionView,
    ended_collection_view,
)
from bio_neb.investigation.investigation import (
    EvidenceAssessment,
    InvestigationIssueItem,
    RetrievedSource,
    ReviseInvestigationResponse,
    UserAssertion,
    WorkItem,
    investigation_issue_items,
    latest_lookup_results,
)
from bio_neb.investigation.investigation_observation import (
    CandidateEvidence,
    InvestigationCollectionCombineStatus,
    InvestigationCollectionItemState,
    InvestigationEvidenceRecorded,
    ObservationAdmissionDecision,
    ObservationEvidenceAssessment,
    RunProvenance,
    SampleObservation,
    derive_investigation_run_provenance,
    validate_run_provenance,
)
from bio_neb.investigation.item_evidence import TABLE_READER_ROLE, ItemEvidence
from bio_neb.investigation.qualification_origins import (
    CollectionItem,
    CoverageNotMeasured,
    DiploidCallingJudgedOtherwise,
    QualificationOrigin,
    observation_origins,
)
from bio_neb.investigation.run_health import RunHealthSection, collection_measurements
from bio_neb.investigation.vcf_observation import PreparedVcfObservationReceipt
from bio_neb.journey import (
    ArtifactReference,
    JourneyEvent,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.paper_reading.report import ReportedPaper, reported_papers
from bio_neb.parabricks.requirement_resolution import (
    FactSlotStatus,
    planned_fact_slots,
)
from bio_neb.parabricks.run_method import RunMethod, run_method
from bio_neb.recording import is_contact_evidence


class FinalizationEvidencePointer(FrozenContract):
    role: StrictStr
    locator: StrictStr
    content_id: StrictStr | None = None
    sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_event_id: StrictStr

    @computed_field(return_type=str)
    @property
    def pointer_id(self) -> str:
        return "finalization-evidence-pointer:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"pointer_id"})
        )


class ReportedFactSlot(FactSlotStatus):
    """One subject's identity slot as the report states it (#339 ``report``).

    The Investigation's slot for a requested rsID carries the planning slots that
    asked for the same rsID, so one subject reads as one reconciled entry with
    both statuses; a slot no later slot reconciles stands alone.
    """

    planning_slots: tuple[FactSlotStatus, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


def reconciled_fact_slots(
    planned: Iterable[FactSlotStatus], investigated: Iterable[FactSlotStatus]
) -> tuple[ReportedFactSlot, ...]:
    """Planning's slots, then the Investigation's, each subject once.

    A planning rsID slot (a scientific investigation's) reconciles into every
    Investigation request slot that requested the same rsID; a shared rsID is not proof
    of one identity (0039), so each request keeps its own entry and its own
    verification, which names the source its lookup asked (#469, owner Q6 a).
    """
    planned = tuple(planned)
    investigated = tuple(investigated)
    reconciled: set[int] = set()
    entries = []
    for slot in investigated:
        matches = tuple(
            index
            for index, earlier in enumerate(planned)
            if slot.kind == "requested_rsid"
            and earlier.kind == "scientific_investigation"
            and earlier.requested_value is not None
            and earlier.requested_value == slot.requested_value
        )
        reconciled.update(matches)
        entries.append(
            ReportedFactSlot(
                **slot.model_dump(),
                planning_slots=tuple(planned[index] for index in matches),
            )
        )
    return (
        *(
            ReportedFactSlot(**slot.model_dump())
            for index, slot in enumerate(planned)
            if index not in reconciled
        ),
        *entries,
    )


class FinalizationAccountingEntry(FrozenContract):
    """An exact retained semantic fact, never scientific admission authority."""

    source_event_id: StrictStr
    event_type: StrictStr
    parent_ids: tuple[StrictStr, ...]
    payload: dict[str, JsonValue]
    artifact_references: tuple[ArtifactReference, ...]

    @computed_field(return_type=str)
    @property
    def accounting_id(self) -> str:
        return "finalization-accounting:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"accounting_id"})
        )


FinalizationMemberRole = Literal[
    "accounting",
    "question",
    "work_item",
    "retrieved_source",
    "source_assessment",
    "sample_observation",
    "candidate_evidence",
    "observation_assessment",
    "admission",
]


class ReportMembershipDecision(FrozenContract):
    member_id: StrictStr
    member_role: FinalizationMemberRole
    admitted: StrictBool
    included: StrictBool = True
    payload_retained: Literal[True] = True


class Qualification(FrozenContract):
    """One limitation, conflict or gap, once per sentence, with its typed origin
    (#474) and what it concerns: the Work items and requests whose own state or
    evidence stated it (#448 D2). A qualification concerning neither is the Run's
    own. ``text`` is the origin's audit sentence for its subject; the readable
    report and the explainer read the origin's plain words instead."""

    text: StrictStr = Field(min_length=1)
    origin: QualificationOrigin
    work_item_ids: tuple[StrictStr, ...] = ()
    request_member_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def ordered_subjects(self) -> "Qualification":
        for values in (self.work_item_ids, self.request_member_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("qualification subjects changed their order")
        return self


# What states a qualification: its origin, the subject its audit sentence names,
# and the Work item or request it concerns, or neither for the Run's own.
Stated = tuple[
    QualificationOrigin, str, Literal["work_item", "request", None], str | None
]


def qualifications(stated: Iterable[Stated]) -> tuple[Qualification, ...]:
    """Each sentence once, sorted, with its origin and every subject that stated
    it (owner Q2 a)."""
    held: dict[str, tuple[QualificationOrigin, dict[str, set[str]]]] = {}
    for origin, named, kind, subject in stated:
        text = origin.audit(named)
        known, subjects = held.setdefault(
            text, (origin, {"work_item": set(), "request": set()})
        )
        if known != origin:
            raise ValueError("one qualification sentence has two origins")
        if kind is not None and subject is not None:
            subjects[kind].add(subject)
    return tuple(
        Qualification(
            text=text,
            origin=origin,
            work_item_ids=tuple(sorted(subjects["work_item"])),
            request_member_ids=tuple(sorted(subjects["request"])),
        )
        for text, (origin, subjects) in sorted(held.items())
    )


class FinalizationBasis(FrozenContract):
    schema_version: Literal["bio-neb.finalization-basis/7"] = (
        "bio-neb.finalization-basis/7"
    )
    journey_id: StrictStr
    source_head_event_id: StrictStr
    outcome: StrictStr = Field(min_length=1)
    accounting: tuple[FinalizationAccountingEntry, ...] = Field(min_length=1)
    evidence_state_id: StrictStr | None = Field(
        pattern=r"^investigation-collection-evidence:[0-9a-f]{64}$"
    )
    selected_work_item_ids: tuple[StrictStr, ...]
    collection_item_states: tuple[InvestigationCollectionItemState, ...]
    collection_status: InvestigationCollectionCombineStatus | None
    question: UserAssertion | None
    work_items: tuple[WorkItem, ...]
    retrieved_sources: tuple[RetrievedSource, ...]
    source_assessments: tuple[EvidenceAssessment, ...]
    sample_observations: tuple[SampleObservation, ...]
    candidate_evidence: tuple[CandidateEvidence, ...]
    observation_assessments: tuple[ObservationEvidenceAssessment, ...]
    admissions: tuple[ObservationAdmissionDecision, ...]
    conflicts: tuple[Qualification, ...]
    limitations: tuple[Qualification, ...]
    gaps: tuple[Qualification, ...]
    evidence_pointers: tuple[FinalizationEvidencePointer, ...]
    report_membership: tuple[ReportMembershipDecision, ...] = Field(min_length=1)
    inherited_model_contact_count: StrictInt = Field(ge=0)
    inherited_source_contact_count: StrictInt = Field(ge=0)
    finalization_external_contact_count: Literal[0] = 0
    run_provenance: RunProvenance = "unestablished"
    provenance_origin_event_id: StrictStr | None = Field(default=None, min_length=1)
    supersedes_basis_digest: StrictStr | None = Field(
        default=None, pattern=r"^finalization-basis:[0-9a-f]{64}$"
    )
    correction_request_ids: tuple[StrictStr, ...] = ()
    # The proposal's fact slots as the report states them: the researcher's value,
    # its status and source, and any requested-vs-verified mismatch; planning and
    # Investigation slots for one requested rsID read as one entry.
    fact_slots: tuple[ReportedFactSlot, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # Each settled Work item's own record after its lookup: its window, its
    # reading and the fields later tools fill (ADR 0047); never a scientific fact.
    item_evidence: tuple[ItemEvidence, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # Each requested SNP whose check ended with an issue, with its verdict, its
    # evidence and the researcher's decision; checked, never verified.
    issue_items: tuple[InvestigationIssueItem, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # Every requested SNP together, with its items' lookup outcomes and the cases
    # across SNPs that are not simple, and each request's and item's real state;
    # it reports, never admits.
    collection_view: CollectionView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The Run's health and each measured item's depth line, cited from their
    # facts; reported, never a decision.
    run_health: RunHealthSection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # How the Run's VCF was made, copied from its succeeded Execution;
    # omitted when the Run ended before one. It reports, never admits.
    method: RunMethod | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # What each paper read in the Run found, copied from its reading's answer with
    # the chunks the reading kept; it reports, never admits.
    papers: tuple[ReportedPaper, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @computed_field(return_type=str)
    @property
    def basis_digest(self) -> str:
        return "finalization-basis:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"basis_digest"})
        )

    @model_validator(mode="after")
    def complete_stably_ordered_inventory(self) -> "FinalizationBasis":
        validate_run_provenance(self.run_provenance, self.provenance_origin_event_id)
        ordered = (
            (self.accounting, "accounting_id"),
            (self.work_items, "work_item_id"),
            (self.retrieved_sources, "source_id"),
            (self.source_assessments, "assessment_id"),
            (self.sample_observations, "observation_id"),
            (self.candidate_evidence, "candidate_id"),
            (self.observation_assessments, "assessment_id"),
            (self.admissions, "admission_id"),
            (self.item_evidence, "work_item_id"),
            (self.issue_items, "request_event_id"),
        )
        if any(
            tuple(getattr(item, field) for item in values)
            != tuple(sorted(getattr(item, field) for item in values))
            for values, field in ordered
        ):
            raise ValueError("Finalization evidence inventory order changed")
        for values, kind in (
            (self.conflicts, "conflict"),
            (self.limitations, "limitation"),
            (self.gaps, "gap"),
        ):
            texts = tuple(item.text for item in values)
            if texts != tuple(sorted(set(texts))):
                raise ValueError("Finalization text inventory order changed")
            if any(item.origin.kind != kind for item in values):
                raise ValueError(f"a Finalization {kind} has another kind's origin")
        if tuple(item.pointer_id for item in self.evidence_pointers) != tuple(
            sorted(item.pointer_id for item in self.evidence_pointers)
        ):
            raise ValueError("Finalization evidence pointer order changed")
        if tuple(item.member_id for item in self.report_membership) != tuple(
            sorted(item.member_id for item in self.report_membership)
        ):
            raise ValueError("Finalization report membership order changed")
        expected_members = {
            **{item.accounting_id: ("accounting", False) for item in self.accounting},
            **(
                {self.question.assertion_id: ("question", False)}
                if self.question is not None
                else {}
            ),
            **{item.work_item_id: ("work_item", False) for item in self.work_items},
            **{
                item.source_id: ("retrieved_source", False)
                for item in self.retrieved_sources
            },
            **{
                item.assessment_id: ("source_assessment", False)
                for item in self.source_assessments
            },
            **{
                item.observation_id: ("sample_observation", False)
                for item in self.sample_observations
            },
            **{
                item.candidate_id: ("candidate_evidence", False)
                for item in self.candidate_evidence
            },
            **{
                item.assessment_id: ("observation_assessment", False)
                for item in self.observation_assessments
            },
            **{
                item.admission_id: ("admission", item.status == "admitted")
                for item in self.admissions
            },
        }
        retained_members = {
            item.member_id: (item.member_role, item.admitted)
            for item in self.report_membership
        }
        if retained_members != expected_members:
            raise ValueError("Finalization report membership changed retained evidence")
        if len(retained_members) != len(self.report_membership):
            raise ValueError("Finalization report membership repeated an identity")
        if len({item.source_event_id for item in self.accounting}) != len(
            self.accounting
        ):
            raise ValueError("Finalization accounting repeated a source event")
        if self.evidence_state_id is None:
            if (
                self.collection_status is not None
                or self.selected_work_item_ids
                or self.collection_item_states
                or self.work_items
                or self.sample_observations
                or self.candidate_evidence
                or self.observation_assessments
                or self.admissions
                or self.item_evidence
            ):
                raise ValueError("Finalization invented scientific collection evidence")
        elif (
            self.collection_status is None
            or not self.work_items
            or self.question is None
        ):
            raise ValueError("Finalization scientific collection is incomplete")
        if (
            tuple(item.work_item_id for item in self.collection_item_states)
            != self.selected_work_item_ids
            or tuple(item.selected_ordinal for item in self.collection_item_states)
            != tuple(range(1, len(self.selected_work_item_ids) + 1))
            or set(self.selected_work_item_ids)
            != {item.work_item_id for item in self.work_items}
        ):
            raise ValueError("Finalization changed selected collection membership")
        return self


class FinalizationReviewRequested(FrozenContract):
    review_policy: Annotated[
        FinalizationReviewPolicy, BeforeValidator(explicit_review_policy)
    ]
    event_type: Literal["FinalizationBasisCompiled"] = "FinalizationBasisCompiled"
    source_head_event_id: StrictStr
    basis: FinalizationBasis

    @model_validator(mode="after")
    def exact_source_binding(self) -> "FinalizationReviewRequested":
        if self.source_head_event_id != self.basis.source_head_event_id:
            raise ValueError("Finalization request changed its evidence-state head")
        return self


class ApproveFinalizationResponse(FrozenContract):
    action: Literal["approve"] = "approve"


class SetReportMembershipChange(FrozenContract):
    action: Literal["set_report_membership"] = "set_report_membership"
    member_id: StrictStr
    included: StrictBool


class InvestigationRequiredChange(FrozenContract):
    action: Literal[
        "add_scientific_fact",
        "change_source",
        "change_observation",
        "change_assessment_or_admission",
        "change_target_or_question",
        "new_research",
    ]
    description: StrictStr = Field(min_length=1)


FinalizationCorrectionChange = Annotated[
    SetReportMembershipChange | InvestigationRequiredChange,
    Field(discriminator="action"),
]


class FinalizationCorrectionAssertion(FrozenContract):
    schema_version: Literal["bio-neb.finalization-correction/1"] = (
        "bio-neb.finalization-correction/1"
    )
    asserted_by: StrictStr = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)
    requested_changes: tuple[FinalizationCorrectionChange, ...] = Field(min_length=1)
    investigation_revision: ReviseInvestigationResponse | None = None

    @computed_field(return_type=str)
    @property
    def correction_id(self) -> str:
        return "finalization-correction:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"correction_id"})
        )

    @model_validator(mode="after")
    def unambiguous_membership_changes(self) -> "FinalizationCorrectionAssertion":
        if (
            self.investigation_revision is not None
            and self.investigation_revision.asserted_by != self.asserted_by
        ):
            raise ValueError("Finalization rework changed its assertion owner")
        member_ids = tuple(
            item.member_id
            for item in self.requested_changes
            if isinstance(item, SetReportMembershipChange)
        )
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Finalization correction repeated a member identity")
        return self


class CorrectFinalizationResponse(FrozenContract):
    action: Literal["correct"] = "correct"
    correction: FinalizationCorrectionAssertion


class FinalizationStopAssertion(FrozenContract):
    schema_version: Literal["bio-neb.finalization-stop/1"] = (
        "bio-neb.finalization-stop/1"
    )
    asserted_by: StrictStr = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)

    @computed_field(return_type=str)
    @property
    def stop_id(self) -> str:
        return "finalization-stop:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"stop_id"})
        )


class StopFinalizationResponse(FrozenContract):
    action: Literal["stop"] = "stop"
    stop: FinalizationStopAssertion


FinalizationResponse = Annotated[
    ApproveFinalizationResponse
    | CorrectFinalizationResponse
    | StopFinalizationResponse,
    Field(discriminator="action"),
]


class FinalizationReviewReceipt(FrozenContract):
    request_event_id: StrictStr
    response_head_event_id: StrictStr
    basis_digest: StrictStr = Field(pattern=r"^finalization-basis:[0-9a-f]{64}$")
    source_head_event_id: StrictStr
    response: FinalizationResponse

    @computed_field(return_type=str)
    @property
    def response_id(self) -> str:
        return "finalization-review-response:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"response_id"})
        )


class ReportBasis(FrozenContract):
    schema_version: Literal["bio-neb.report-basis/1"] = "bio-neb.report-basis/1"
    finalization_basis: FinalizationBasis
    approval: FinalizationReviewReceipt
    finalization_external_contact_count: Literal[0] = 0

    @computed_field(return_type=str)
    @property
    def report_basis_digest(self) -> str:
        return "report-basis:" + canonical_json_digest(
            {
                "finalization_basis_digest": self.finalization_basis.basis_digest,
                "approval_response_id": self.approval.response_id,
            }
        )

    @model_validator(mode="after")
    def exact_approval(self) -> "ReportBasis":
        if (
            self.approval.response.action != "approve"
            or self.approval.basis_digest != self.finalization_basis.basis_digest
            or self.approval.source_head_event_id
            != self.finalization_basis.source_head_event_id
        ):
            raise ValueError("Report basis lacks exact Finalization approval")
        return self


class FinalizationReturnToInvestigation(FrozenContract):
    schema_version: Literal["bio-neb.finalization-return-to-investigation/1"] = (
        "bio-neb.finalization-return-to-investigation/1"
    )
    basis_digest: StrictStr = Field(pattern=r"^finalization-basis:[0-9a-f]{64}$")
    correction_id: StrictStr = Field(pattern=r"^finalization-correction:[0-9a-f]{64}$")
    reason_code: Literal[
        "new_research_required",
        "unknown_member_id",
    ]
    requested_changes: tuple[FinalizationCorrectionChange, ...] = Field(min_length=1)
    finalization_external_contact_count: Literal[0] = 0


class FinalizationStop(FrozenContract):
    schema_version: Literal["bio-neb.finalization-stop-disposition/1"] = (
        "bio-neb.finalization-stop-disposition/1"
    )
    basis_digest: StrictStr = Field(pattern=r"^finalization-basis:[0-9a-f]{64}$")
    stop: FinalizationStopAssertion
    finalization_external_contact_count: Literal[0] = 0


class FinalizationReviewRecorded(FrozenContract):
    review_policy: Annotated[
        FinalizationReviewPolicy, BeforeValidator(explicit_review_policy)
    ]
    event_type: Literal["FinalizationReviewRecorded"] = "FinalizationReviewRecorded"
    receipt: FinalizationReviewReceipt
    report_basis: ReportBasis | None = None
    superseding_basis: FinalizationBasis | None = None
    return_to_investigation: FinalizationReturnToInvestigation | None = None
    stop: FinalizationStop | None = None

    @model_validator(mode="after")
    def exact_disposition(self) -> "FinalizationReviewRecorded":
        shapes = (
            self.report_basis is not None,
            self.superseding_basis is not None,
            self.return_to_investigation is not None,
            self.stop is not None,
        )
        if sum(shapes) != 1:
            raise ValueError("Finalization review requires one exact disposition")
        if self.receipt.response.action == "approve":
            if self.report_basis is None or self.report_basis.approval != self.receipt:
                raise ValueError("Finalization review changed Report-basis approval")
        elif self.report_basis is not None:
            raise ValueError("Finalization correction cannot approve a Report basis")
        if self.superseding_basis is not None and (
            self.receipt.response.action != "correct"
            or self.superseding_basis.supersedes_basis_digest
            != self.receipt.basis_digest
            or self.receipt.response.correction.correction_id
            not in self.superseding_basis.correction_request_ids
        ):
            raise ValueError("Finalization correction changed its superseding basis")
        if self.return_to_investigation is not None and (
            self.receipt.response.action != "correct"
            or self.return_to_investigation.basis_digest != self.receipt.basis_digest
            or self.return_to_investigation.correction_id
            != self.receipt.response.correction.correction_id
        ):
            raise ValueError("Finalization return changed its correction request")
        if self.stop is not None and (
            self.receipt.response.action != "stop"
            or self.stop.basis_digest != self.receipt.basis_digest
            or self.stop.stop != self.receipt.response.stop
        ):
            raise ValueError("Finalization stop changed its review request")
        return self


class FinalizationReviewRequestView(FrozenContract):
    review_policy: FinalizationReviewPolicy
    request_event_id: StrictStr
    source_head_event_id: StrictStr
    basis: FinalizationBasis
    legal_actions: tuple[Literal["approve", "correct", "stop"], ...] = (
        "approve",
        "correct",
        "stop",
    )
    pending: StrictBool


class FinalizationProjection(FrozenContract):
    status: Literal[
        "not_started",
        "review_pending",
        "approved",
        "returned_to_investigation",
        "stopped",
    ]
    bases: tuple[FinalizationBasis, ...] = ()
    requests: tuple[FinalizationReviewRequestView, ...] = ()
    receipts: tuple[FinalizationReviewReceipt, ...] = ()
    pending_request: FinalizationReviewRequestView | None = None
    report_basis: ReportBasis | None = None
    return_to_investigation: FinalizationReturnToInvestigation | None = None
    stop: FinalizationStop | None = None
    finalization_external_contact_count: Literal[0] = 0

    @model_validator(mode="after")
    def exact_status(self) -> "FinalizationProjection":
        pending = tuple(item for item in self.requests if item.pending)
        expected_pending = (
            () if self.pending_request is None else (self.pending_request,)
        )
        if pending != expected_pending:
            raise ValueError("Finalization pending review projection is inconsistent")
        shapes = {
            "not_started": not self.bases
            and not self.requests
            and not self.receipts
            and self.report_basis is None,
            "review_pending": bool(self.bases)
            and self.pending_request is not None
            and self.report_basis is None,
            "approved": bool(self.bases)
            and self.pending_request is None
            and self.report_basis is not None,
            "returned_to_investigation": bool(self.bases)
            and self.pending_request is None
            and self.report_basis is None
            and self.return_to_investigation is not None,
            "stopped": bool(self.bases)
            and self.pending_request is None
            and self.report_basis is None
            and self.stop is not None,
        }
        if not shapes[self.status]:
            raise ValueError("Finalization projection status changed")
        return self


def _as_dict(event: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        event.model_dump(mode="json")
        if isinstance(event, JourneyEvent)
        else dict(event)
    )


def validate_finalization_request_fact(
    event: Mapping[str, Any],
) -> FinalizationReviewRequested | None:
    if event.get("event_type") != "FinalizationBasisCompiled":
        return None
    return FinalizationReviewRequested.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def validate_finalization_review_fact(
    event: Mapping[str, Any],
) -> FinalizationReviewRecorded | None:
    if event.get("event_type") != "FinalizationReviewRecorded":
        return None
    return FinalizationReviewRecorded.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def _pointer(
    *,
    role: str,
    locator: str,
    source_event_id: str,
    content_id: str | None = None,
    sha256: str | None = None,
) -> FinalizationEvidencePointer:
    return FinalizationEvidencePointer(
        role=role,
        locator=locator,
        content_id=content_id,
        sha256=sha256,
        source_event_id=source_event_id,
    )


def _invocation_pointer(
    invocation: BaseModel, *, source_event_id: str
) -> FinalizationEvidencePointer:
    digest = canonical_json_digest(invocation.model_dump(mode="json"))
    return _pointer(
        role=f"process_invocation:{invocation.purpose}",
        locator="argv:" + canonical_json(list(invocation.argv)),
        content_id=f"process-invocation:{digest}",
        sha256=invocation.stdout_sha256,
        source_event_id=source_event_id,
    )


def _evidence_pointers(
    timeline: list[dict[str, Any]],
    evidence_event: dict[str, Any],
    evidence: InvestigationEvidenceRecorded,
    sources: tuple[RetrievedSource, ...],
) -> tuple[FinalizationEvidencePointer, ...]:
    event_id = evidence_event["event_id"]
    pointers = [
        _pointer(
            role="journey_evidence_state",
            locator=f"journey:{evidence_event['journey_id']}#event:{event_id}",
            content_id=evidence.evidence_state.evidence_state_id,
            source_event_id=event_id,
        ),
        _pointer(
            role="investigation_authority",
            locator=f"journey:{evidence_event['journey_id']}#event:{event_id}",
            content_id=evidence.authority.authority_id,
            source_event_id=event_id,
        ),
        _pointer(
            role="materialization_receipt",
            locator=evidence.command.materialization_receipt_file,
            content_id=evidence.sample_observation.materialization_receipt_id,
            sha256=evidence.sample_observation.materialization_receipt_sha256,
            source_event_id=event_id,
        ),
    ]
    for event in timeline:
        for reference in event.get("artifact_references", []):
            artifact_id = reference["artifact_id"]
            pointers.append(
                _pointer(
                    role=f"journey_artifact:{event['event_type']}",
                    locator=f"journey:{event['journey_id']}#artifact:{artifact_id}",
                    content_id=artifact_id,
                    sha256=ContentId.parse(artifact_id).digest,
                    source_event_id=event["event_id"],
                )
            )
    for source in sources:
        source_event = next(
            (
                event
                for event in timeline
                if any(
                    reference.get("artifact_id") == source.raw_artifact_id
                    for reference in event.get("artifact_references", [])
                )
            ),
            None,
        )
        if source_event is None:
            raise ValueError("Finalization source artifact is missing")
        pointers.append(
            _pointer(
                role=f"retrieved_source:{source.operation}",
                locator=source.request_identity,
                content_id=source.source_id,
                sha256=source.raw_response_sha256,
                source_event_id=source_event["event_id"],
            )
        )
    preparation = evidence.sample_observation.preparation
    if isinstance(preparation, PreparedVcfObservationReceipt):
        pointers.extend(
            (
                _pointer(
                    role="vcf_observation_preparation",
                    locator=preparation.derived_bgzf.path,
                    content_id=preparation.receipt_id,
                    sha256=preparation.derived_bgzf.sha256,
                    source_event_id=event_id,
                ),
                _pointer(
                    role="bcftools_runtime",
                    locator=preparation.runtime.executable_path,
                    sha256=preparation.runtime.executable_sha256,
                    source_event_id=event_id,
                ),
                _pointer(
                    role="bcftools_probe_receipt",
                    locator=preparation.runtime.probe_receipt_path,
                    sha256=preparation.runtime.probe_receipt_sha256,
                    source_event_id=event_id,
                ),
            )
        )
        pointers.extend(
            _invocation_pointer(item, source_event_id=event_id)
            for item in preparation.invocations
        )
    query = getattr(evidence.sample_observation.result, "query_invocation", None)
    if query is not None:
        pointers.append(_invocation_pointer(query, source_event_id=event_id))
    return tuple(sorted(set(pointers), key=lambda item: item.pointer_id))


def _member(
    member_id: str, role: FinalizationMemberRole, *, admitted: bool = False
) -> ReportMembershipDecision:
    return ReportMembershipDecision(
        member_id=member_id,
        member_role=role,
        admitted=admitted,
    )


def _collection_item_limitation(
    item: InvestigationCollectionItemState,
) -> CollectionItem | None:
    """An item the collection holds that ended without its observation."""
    if item.status in {
        "failed",
        "stopped",
        "not_started",
        "approval_pending",
        "observation_pending",
    }:
        return CollectionItem(
            status=item.status,
            reason=item.reason if item.status in {"failed", "stopped"} else None,
        )
    return None


# Transport payloads and liveness events explain mechanics, not Run outcomes.
# Keep semantic failures, actions, source facts and human decisions. Reporting
# events are excluded so later bases do not recursively embed earlier bases.
_ACCOUNTING_OMITTED_EVENTS = frozenset(
    {
        "FinalizationBasisCompiled",
        "FinalizationReviewRecorded",
        "GroundedReportPublished",
        "ModelTransportRequestRecorded",
        "ModelTransportResponseRecorded",
        "ModelRequestRecorded",
        "ModelResponseRecorded",
        "PassStarted",
        "PassClosed",
        "activity_started",
        "activity_progress",
        "activity_heartbeat",
        "activity_succeeded",
        "JourneySearchRead",
    }
)


def _accounting_entries(
    timeline: list[dict[str, Any]],
) -> tuple[FinalizationAccountingEntry, ...]:
    entries = (
        FinalizationAccountingEntry.model_validate_json(
            canonical_json(
                {
                    "source_event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "parent_ids": event.get("parent_ids", []),
                    "payload": event.get("payload", {}),
                    "artifact_references": event.get("artifact_references", []),
                }
            ),
            strict=True,
        )
        for event in timeline
        if not is_contact_evidence(event)
        and event["event_type"] not in _ACCOUNTING_OMITTED_EVENTS
    )
    return tuple(sorted(entries, key=lambda item: item.accounting_id))


def _item_windows(timeline, collection_event_id: str) -> dict[str, Any]:
    """Each item's window of one collection, by Work item (ADR 0047)."""
    from bio_neb.investigation.item_evidence import (
        InvestigationItemWindowRecorded,
        validate_item_evidence_fact,
    )

    return {
        fact.work_item_id: fact.result
        for event in timeline
        if isinstance(
            fact := validate_item_evidence_fact(event), InvestigationItemWindowRecorded
        )
        and fact.collection_event_id == collection_event_id
    }


def _state_accounting(
    view: CollectionView | None, collection_item_ids: set[str]
) -> list[Stated]:
    """The gap or limitation each request's and item's real state leaves (#339),
    by the typed origin its state recorded (#474).

    An observed item has its evidence, and an approved or failed item the
    collection holds has its collection limitation. Researcher exclusions and
    targets not checked as stated (#367) are limitations; work that did not
    happen is a gap.
    """
    stated: list[Stated] = []
    for snp in () if view is None else view.snps:
        name = snp.requested_rsid or snp.request_text
        if snp.state != "candidates":
            if snp.state_origin is not None:
                stated.append(
                    (snp.state_origin, name, "request", snp.request_member_id)
                )
            continue
        for item in snp.items:
            held = item.work_item_id in collection_item_ids
            if item.state_origin is None or (
                held and item.state in {"approved_not_observed", "failed"}
            ):
                continue
            stated.append(
                (
                    item.state_origin,
                    f"Work item {item.work_item_id} ({name} {item.target})",
                    "work_item",
                    item.work_item_id,
                )
            )
    return stated


def _table_reader_contact_count(timeline: list[dict[str, Any]]) -> int:
    """The table reader's physical model contacts, which planner Attempts omit."""
    readers = {
        event["payload"]["attempt_id"]
        for event in timeline
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["agent_role"] == TABLE_READER_ROLE
    }
    return sum(
        1
        for event in timeline
        if event["event_type"] == "ModelPhysicalContactStarted"
        and event["payload"]["attempt_id"] in readers
    )


def compile_finalization_basis(candidate: FinalizationCandidate) -> FinalizationBasis:
    """Freeze Run accounting and any independently checked scientific members."""
    read = candidate.investigation
    timeline = read.timeline
    investigation = read.investigation
    source_head_event = candidate.source_head_event
    collection = candidate.collection
    evidence = collection.evidence if collection is not None else None
    members = tuple(
        item.recorded
        for item in (collection.active_members if collection is not None else ())
        if item.recorded is not None
    )
    evidence_events = tuple(item.event for item in members)
    evidence_records = tuple(item.evidence for item in members)
    collection_fields: dict[str, Any] = {
        "selected_work_item_ids": (
            evidence.collection_work_item_ids or evidence.selected_work_item_ids
        )
        if evidence is not None
        else (),
        "collection_item_states": evidence.item_states if evidence is not None else (),
        "collection_status": evidence.combine_status if evidence is not None else None,
    }
    evidence_state_id = evidence.collection_state_id if evidence is not None else None
    question = (
        collection.question
        if collection is not None
        else investigation.assertions[-1]
        if investigation.assertions
        else None
    )
    work_items = tuple(
        sorted(
            collection.proposal.work_items if collection is not None else (),
            key=lambda item: item.work_item_id,
        )
    )
    sources = tuple(
        sorted(
            {item.source_id: item for item in investigation.sources}.values(),
            key=lambda item: item.source_id,
        )
    )
    source_assessments = tuple(
        sorted(
            {item.assessment_id: item for item in investigation.assessments}.values(),
            key=lambda item: item.assessment_id,
        )
    )
    observations = tuple(
        sorted(
            (item.sample_observation for item in evidence_records),
            key=lambda item: item.observation_id,
        )
    )
    candidates = tuple(
        sorted(
            (item.candidate_evidence for item in evidence_records),
            key=lambda item: item.candidate_id,
        )
    )
    assessments = tuple(
        sorted(
            (item.assessment for item in evidence_records),
            key=lambda item: item.assessment_id,
        )
    )
    admissions = tuple(
        sorted(
            (item.admission for item in evidence_records),
            key=lambda item: item.admission_id,
        )
    )
    accounting = _accounting_entries(timeline)
    item_evidence = (
        tuple(sorted(read.item_evidence, key=lambda item: item.work_item_id))
        if collection is not None
        else ()
    )
    measured = {
        item.work_item_id
        for item in item_evidence
        if item.coverage.status == "measured"
    }
    measurements = collection_measurements(timeline, item_evidence)
    membership = [
        *(_member(item.accounting_id, "accounting") for item in accounting),
        *([_member(question.assertion_id, "question")] if question is not None else []),
        *(_member(item.work_item_id, "work_item") for item in work_items),
        *(_member(item.source_id, "retrieved_source") for item in sources),
        *(
            _member(item.assessment_id, "source_assessment")
            for item in source_assessments
        ),
        *(_member(item.observation_id, "sample_observation") for item in observations),
        *(_member(item.candidate_id, "candidate_evidence") for item in candidates),
        *(
            _member(item.assessment_id, "observation_assessment")
            for item in assessments
        ),
        *(
            _member(
                item.admission_id,
                "admission",
                admitted=item.status == "admitted",
            )
            for item in admissions
        ),
    ]
    states = tuple(item.evidence_state for item in evidence_records)
    view = ended_collection_view(
        investigation,
        collection.proposal
        if collection is not None
        else investigation.proposals[-1]
        if investigation.proposals
        else None,
        observations,
        _item_windows(timeline, collection.event["event_id"])
        if collection is not None
        else None,
        item_evidence,
        collection_fields["collection_item_states"],
        latest_lookup_results(timeline),
    )
    method = run_method(timeline)
    papers = reported_papers(timeline)
    stated = [
        # Each observation's own, typed from its outcome, provenance and
        # admission as its Investigation evidence states them.
        *(
            (origin, "", "work_item", item.work_item_id)
            for item in states
            for origins in observation_origins(
                item.outcome,
                item.run_provenance,
                item.evidence_admission == "admitted",
                depth_measured=item.work_item_id in measured,
            ).values()
            for origin in origins
        ),
        *(
            (origin, item.work_item_id, "work_item", item.work_item_id)
            for item in (evidence.item_states if evidence is not None else ())
            if (origin := _collection_item_limitation(item)) is not None
        ),
        *_state_accounting(
            view,
            {item.work_item_id for item in collection_fields["collection_item_states"]},
        ),
        *(
            (origin, work_item_id, "work_item", work_item_id)
            for work_item_id, origin in measurements.limitations
        ),
        # Read from each item's coverage field; an item with no item evidence has
        # no coverage either.
        *(
            (
                measurements.not_measured.get(item.work_item_id, CoverageNotMeasured()),
                item.work_item_id,
                "work_item",
                item.work_item_id,
            )
            for item in work_items
            if item.work_item_id not in measured
        ),
        # Diploid calling the researcher chose on an organism the planning model
        # judged otherwise: the Run's own.
        *(
            [
                (
                    DiploidCallingJudgedOtherwise(judgment=method.organism_ploidy),
                    "",
                    None,
                    None,
                )
            ]
            if method is not None
            and method.organism_ploidy in {"likely_non_diploid", "unknown"}
            else []
        ),
    ]
    run_provenance, provenance_origin_event_id = derive_investigation_run_provenance(
        timeline
    )
    pointers = tuple(
        sorted(
            {
                pointer
                for item_event, item_evidence in zip(
                    evidence_events, evidence_records, strict=True
                )
                for pointer in _evidence_pointers(
                    timeline, item_event, item_evidence, sources
                )
            },
            key=lambda item: item.pointer_id,
        )
    )
    accounting_pointers = tuple(
        _pointer(
            role=f"journey_accounting:{item.event_type}",
            locator=f"journey:{source_head_event['journey_id']}#event:{item.source_event_id}",
            content_id=item.accounting_id,
            source_event_id=item.source_event_id,
        )
        for item in accounting
    )
    return FinalizationBasis(
        journey_id=source_head_event["journey_id"],
        source_head_event_id=source_head_event["event_id"],
        outcome=candidate.outcome,
        accounting=accounting,
        evidence_state_id=evidence_state_id,
        question=question,
        work_items=work_items,
        retrieved_sources=sources,
        source_assessments=source_assessments,
        sample_observations=observations,
        candidate_evidence=candidates,
        observation_assessments=assessments,
        admissions=admissions,
        item_evidence=item_evidence,
        conflicts=qualifications(item for item in stated if item[0].kind == "conflict"),
        limitations=qualifications(
            item for item in stated if item[0].kind == "limitation"
        ),
        gaps=qualifications(item for item in stated if item[0].kind == "gap"),
        evidence_pointers=tuple(
            sorted(
                {*pointers, *accounting_pointers},
                key=lambda item: item.pointer_id,
            )
        ),
        report_membership=tuple(sorted(membership, key=lambda item: item.member_id)),
        inherited_model_contact_count=investigation.model_contact_count
        + _table_reader_contact_count(timeline),
        inherited_source_contact_count=investigation.source_contact_count,
        run_provenance=run_provenance,
        provenance_origin_event_id=provenance_origin_event_id,
        fact_slots=reconciled_fact_slots(
            planned_fact_slots(timeline), investigation.fact_slots
        ),
        issue_items=tuple(
            sorted(
                investigation_issue_items(investigation),
                key=lambda item: item.request_event_id,
            )
        ),
        collection_view=view,
        run_health=measurements.section,
        method=method,
        papers=papers,
        **collection_fields,
    )


def derive_finalization_correction(
    basis: FinalizationBasis,
    correction: FinalizationCorrectionAssertion,
) -> FinalizationBasis | FinalizationReturnToInvestigation:
    """Derive the only legal disposition for an exact retained correction."""

    membership = {item.member_id: item for item in basis.report_membership}
    unknown_ids = tuple(
        item.member_id
        for item in correction.requested_changes
        if isinstance(item, SetReportMembershipChange)
        and item.member_id not in membership
    )
    investigation_changes = tuple(
        item
        for item in correction.requested_changes
        if isinstance(item, InvestigationRequiredChange)
    )
    if (
        investigation_changes
        or unknown_ids
        or correction.investigation_revision is not None
    ):
        return FinalizationReturnToInvestigation(
            basis_digest=basis.basis_digest,
            correction_id=correction.correction_id,
            reason_code=(
                "new_research_required"
                if investigation_changes
                or correction.investigation_revision is not None
                else "unknown_member_id"
            ),
            requested_changes=correction.requested_changes,
        )
    changed = False
    for item in correction.requested_changes:
        if not isinstance(item, SetReportMembershipChange):
            raise AssertionError("unsupported correction escaped classification")
        retained = membership[item.member_id]
        if retained.included != item.included:
            membership[item.member_id] = retained.model_copy(
                update={"included": item.included}
            )
            changed = True
    if not changed:
        raise ValueError("Finalization correction is a no-op")
    basis_payload = basis.model_dump(mode="json", exclude_computed_fields=True)
    basis_payload.update(
        {
            "report_membership": [
                item.model_dump(mode="json")
                for item in sorted(
                    membership.values(), key=lambda value: value.member_id
                )
            ],
            "supersedes_basis_digest": basis.basis_digest,
            "correction_request_ids": sorted(
                {*basis.correction_request_ids, correction.correction_id}
            ),
        }
    )
    return FinalizationBasis.model_validate_json(
        canonical_json(basis_payload), strict=True
    )


def project_finalization(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    snapshot: PrefixSnapshot | None = None,
) -> FinalizationProjection:
    """``snapshot`` is the prefix ``events`` already holds, if the caller has one."""
    timeline = [_as_dict(event) for event in events]
    snapshot = snapshot or PrefixSnapshot(timeline)
    starts = [
        index
        for index, event in enumerate(timeline)
        if event.get("event_type") == "FinalizationBasisCompiled"
    ]
    if len(starts) <= 1:
        return _project_finalization_generation(timeline, snapshot=snapshot)
    generations = []
    for ordinal, start in enumerate(starts):
        end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(timeline)
        previous = generations[-1] if generations else None
        if previous is not None and previous.status not in {
            "approved",
            "returned_to_investigation",
        }:
            raise ValueError(
                "Finalization generation followed a pending or stopped basis"
            )
        generations.append(
            _project_finalization_generation(
                timeline[:end],
                snapshot=snapshot,
                generation_start=start if ordinal else 0,
                previous_basis=previous.bases[-1] if previous else None,
            )
        )
    latest = generations[-1]
    return latest.model_copy(
        update={
            "bases": tuple(
                basis for generation in generations for basis in generation.bases
            ),
            "requests": tuple(
                request for generation in generations for request in generation.requests
            ),
            "receipts": tuple(
                receipt for generation in generations for receipt in generation.receipts
            ),
        }
    )


def held_finalization_fact(snapshot: PrefixSnapshot, parse, event):
    """An admitted event's Finalization fact, parsed once for the snapshot's owner."""
    return snapshot.checked_once(
        (parse.__name__, event["event_id"]), lambda: parse(event)
    )


def _project_finalization_generation(  # noqa: C901 - Existing orchestration complexity; contract-base migration preserves this flow.
    events: Iterable[JourneyEvent | Mapping[str, Any]],
    *,
    snapshot: PrefixSnapshot,
    previous_basis: FinalizationBasis | None = None,
    generation_start: int = 0,
) -> FinalizationProjection:
    timeline = [_as_dict(event) for event in events]
    requests = [
        (event, fact)
        for event in timeline[generation_start:]
        if (
            fact := held_finalization_fact(
                snapshot, validate_finalization_request_fact, event
            )
        )
        is not None
    ]
    reviews = [
        (event, fact)
        for event in timeline[generation_start:]
        if (
            fact := held_finalization_fact(
                snapshot, validate_finalization_review_fact, event
            )
        )
        is not None
    ]
    if not requests:
        if reviews:
            raise ValueError("Finalization review exists without a basis request")
        return FinalizationProjection(status="not_started")
    if len(requests) != 1:
        raise ValueError("Finalization Journey repeats its initial basis")
    request_event, request = requests[0]
    if request_event.get("parent_ids") != [request.source_head_event_id]:
        raise ValueError("Finalization basis lacks exact evidence-head parent")
    request_index = timeline.index(request_event)
    source_matches = [
        (index, event)
        for index, event in enumerate(timeline[:request_index])
        if event.get("event_id") == request.source_head_event_id
    ]
    if len(source_matches) != 1:
        raise ValueError("Finalization evidence-head parent is not retained")
    candidate = snapshot.investigation.through(request.source_head_event_id).candidate
    if candidate is None:
        raise ValueError(
            "Finalization evidence-head parent is not an eligible candidate"
        )
    # The basis its source head gives is fixed by that head: compiled once.
    expected_basis = snapshot.checked_once(
        ("finalization_basis", request.source_head_event_id),
        lambda: compile_finalization_basis(candidate),
    )
    if previous_basis is not None:
        if expected_basis.source_head_event_id == previous_basis.source_head_event_id:
            raise ValueError("Finalization generation reused its previous evidence")
        expected_basis = expected_basis.model_copy(
            update={"supersedes_basis_digest": previous_basis.basis_digest}
        )
    if request.basis != expected_basis:
        raise ValueError("Finalization basis changed retained Journey evidence")
    bases = [request.basis]
    request_views: list[FinalizationReviewRequestView] = []
    receipts: list[FinalizationReviewReceipt] = []
    current_event = request_event
    current_basis = request.basis
    report_basis = None
    returned = None
    stopped = None
    for review_event, review in reviews:
        if report_basis is not None or returned is not None or stopped is not None:
            raise ValueError("Finalization review followed a terminal disposition")
        if (
            review_event.get("parent_ids") != [current_event["event_id"]]
            or review.receipt.request_event_id != current_event["event_id"]
            or review.receipt.response_head_event_id != current_event["event_id"]
            or review.receipt.basis_digest != current_basis.basis_digest
            or review.receipt.source_head_event_id != current_basis.source_head_event_id
        ):
            raise ValueError("Finalization review changed exact request lineage")
        request_views.append(
            FinalizationReviewRequestView(
                request_event_id=current_event["event_id"],
                review_policy=FinalizationReviewPolicy.model_validate(
                    current_event["payload"]["review_policy"]
                ),
                source_head_event_id=current_basis.source_head_event_id,
                basis=current_basis,
                pending=False,
            )
        )
        receipts.append(review.receipt)
        if review.report_basis is not None:
            if review.report_basis.finalization_basis != current_basis:
                raise ValueError("Finalization approval changed its exact basis")
            report_basis = review.report_basis
            continue
        if review.return_to_investigation is not None:
            response = review.receipt.response
            if response.action != "correct" or (
                derive_finalization_correction(current_basis, response.correction)
                != review.return_to_investigation
            ):
                raise ValueError("Finalization return changed its correction request")
            returned = review.return_to_investigation
            continue
        if review.stop is not None:
            stopped = review.stop
            continue
        superseding = review.superseding_basis
        if superseding is None:
            raise ValueError("Finalization correction lacks its next disposition")
        response = review.receipt.response
        if response.action != "correct" or (
            derive_finalization_correction(current_basis, response.correction)
            != superseding
        ):
            raise ValueError("Finalization correction changed its exact disposition")
        bases.append(superseding)
        current_event = review_event
        current_basis = superseding
    if report_basis is not None:
        return FinalizationProjection(
            status="approved",
            bases=tuple(bases),
            requests=tuple(request_views),
            receipts=tuple(receipts),
            report_basis=report_basis,
        )
    if returned is not None:
        return FinalizationProjection(
            status="returned_to_investigation",
            bases=tuple(bases),
            requests=tuple(request_views),
            receipts=tuple(receipts),
            return_to_investigation=returned,
        )
    if stopped is not None:
        return FinalizationProjection(
            status="stopped",
            bases=tuple(bases),
            requests=tuple(request_views),
            receipts=tuple(receipts),
            stop=stopped,
        )
    pending = FinalizationReviewRequestView(
        request_event_id=current_event["event_id"],
        review_policy=FinalizationReviewPolicy.model_validate(
            current_event["payload"]["review_policy"]
        ),
        source_head_event_id=current_basis.source_head_event_id,
        basis=current_basis,
        pending=True,
    )
    return FinalizationProjection(
        status="review_pending",
        bases=tuple(bases),
        requests=(*request_views, pending),
        receipts=tuple(receipts),
        pending_request=pending,
    )


__all__ = [
    "ApproveFinalizationResponse",
    "CorrectFinalizationResponse",
    "FinalizationAccountingEntry",
    "FinalizationBasis",
    "FinalizationCorrectionAssertion",
    "FinalizationEvidencePointer",
    "FinalizationProjection",
    "FinalizationReviewReceipt",
    "FinalizationReviewRecorded",
    "FinalizationReviewRequestView",
    "FinalizationReviewRequested",
    "FinalizationReturnToInvestigation",
    "FinalizationStop",
    "FinalizationStopAssertion",
    "InvestigationRequiredChange",
    "ReportBasis",
    "ReportMembershipDecision",
    "ReportedFactSlot",
    "SetReportMembershipChange",
    "StopFinalizationResponse",
    "compile_finalization_basis",
    "derive_finalization_correction",
    "project_finalization",
    "reconciled_fact_slots",
    "validate_finalization_request_fact",
    "validate_finalization_review_fact",
]
