"""Canonical execution proposals and backend-owned Execution HITL authority."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BeforeValidator,
    Field,
    JsonValue,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.continuation_policy import (
    ExecutionReviewPolicy,
    explicit_review_policy,
)
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.application.input_journey import InputMetadataInspected, validate_fact
from bio_neb.application.planning_read import PlanningRead, PrefixSnapshot
from bio_neb.contracts import FrozenContract
from bio_neb.failure import Invariant
from bio_neb.input_inspection.tool import InputMetadataResult
from bio_neb.journey import JourneyEvent, canonical_json_digest
from bio_neb.parabricks.capability_policy import ExecutionRuntimeSelection
from bio_neb.parabricks.configuration import (
    ParabricksConfiguration,
    current_configuration,
)
from bio_neb.parabricks.execution_package import (
    ExecutionPackage,
    ExecutionPackageCompiled,
    validate_execution_package_fact,
)
from bio_neb.parabricks.execution_planning_journey import (
    ExecutionRevisionContext,
    OrganismPloidyJudgment,
    SelectedRecipeBinding,
    SelectedResourceBinding,
)
from bio_neb.parabricks.execution_runtime import runtime_matches_package
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
)
from bio_neb.parabricks.requirement_resolution import (
    ExecutionRequirementResolution,
    FactSlotStatus,
)
from bio_neb.parabricks.run_method import (
    GERMLINE_CALLER,
    GERMLINE_COMMAND,
    GERMLINE_PLOIDY,
)
from bio_neb.prebuilt_index.choice import PrebuiltIndexCandidate
from bio_neb.reference_discovery.reference_companions_tool import CompanionCandidate
from bio_neb.reference_discovery.reference_identity_tool import ValidatedReference
from bio_neb.reference_discovery.reference_journey import (
    ReferenceDecisionBasis,
    reference_decision_basis,
)
from bio_neb.reference_discovery.reference_source import (
    ReferenceSourceAdmission,
    ReferenceSourceAdmitted,
    validate_reference_source_fact,
)


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


def _canonical_text(value: str, field_name: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


class ProposalEvidenceClosureItem(FrozenContract):
    """One exact Journey source and every source-owned evidence identity it carries."""

    source_event_id: StrictStr
    event_type: StrictStr
    evidence_identities: tuple[StrictStr, ...] = ()

    @field_validator("source_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _canonical_uuid(value, "source_event_id")


class ProposedCompanionAsset(FrozenContract):
    work_item_id: StrictStr
    role: StrictStr
    candidate: CompanionCandidate
    validation_event_id: StrictStr
    # The genomic FASTA's index choice the Review binds: its event, and the Galaxy
    # index it chose, extras shown with NCBI's role (None: the CPU build).
    prebuilt_choice_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    prebuilt_index: PrebuiltIndexCandidate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("validation_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _canonical_uuid(value, "validation_event_id")

    @model_validator(mode="after")
    def exact_opaque_identity(self) -> ProposedCompanionAsset:
        _canonical_text(self.candidate.asset_id, "companion asset_id")
        _canonical_text(self.candidate.evidence_id, "companion evidence_id")
        return self


class ExecutionPloidy(FrozenContract):
    """What the Execution Review states about ploidy (#470): the one ploidy the
    package calls at, which the researcher answers, beside the model's judgment of
    the organism and the record it cites; a question when that is not diploid."""

    calling_ploidy: Literal[2]
    statement: StrictStr
    organism: OrganismPloidyJudgment
    selection_event_id: StrictStr
    question: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


_JUDGED = {"likely_non_diploid": "likely not diploid", "unknown": "of unknown ploidy"}


def _execution_ploidy(planning) -> ExecutionPloidy | None:
    """The proposal's ploidy notice, from the recipe selection's judgment; none for a
    selection recorded before #470, so its proposal compiles as before (owner R2 a)."""
    selections = {item.source_event_id: item for item in planning.selections}
    judged = [
        selections[binding.selection_event_id]
        for binding in planning.selected_recipe_bindings
        if selections[binding.selection_event_id].organism_ploidy is not None
    ]
    if not judged:
        return None
    selection = judged[0]
    organism = selection.organism_ploidy
    return ExecutionPloidy(
        calling_ploidy=GERMLINE_PLOIDY,
        statement=(
            f"Variant calling assumes a diploid sample: `{' '.join(GERMLINE_COMMAND)}` "
            f"runs {GERMLINE_CALLER} at ploidy {GERMLINE_PLOIDY}."
        ),
        organism=organism,
        selection_event_id=selection.source_event_id,
        question=(
            None
            if organism.judgment == "diploid"
            else f"The planning model judged this organism {_JUDGED[organism.judgment]}, "
            f"citing its record's words “{organism.quote}”. Calling is diploid only: "
            f"choose calling_ploidy {GERMLINE_PLOIDY} to call it diploid, or revise or "
            "reject the Run."
        ),
    )


class ExecutionProposal(FrozenContract):
    """Complete immutable consequences, compiled only from exact retained evidence."""

    proposal_id: StrictStr = Field(pattern=r"^execution-proposal:[0-9a-f]{64}$")
    journey_id: StrictStr
    plan_event_id: StrictStr
    requirement_resolution: ExecutionRequirementResolution
    # Each fact slot's requested value, status, source and any mismatch, as the
    # researcher reviews them; a researcher's value is never shown as verified.
    fact_slots: tuple[FactSlotStatus, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    inspection_event_id: StrictStr
    inspected_inputs: InputMetadataResult
    reference_event_id: StrictStr
    reference: ValidatedReference
    reference_basis: ReferenceDecisionBasis | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    companions: tuple[ProposedCompanionAsset, ...] = Field(min_length=1)
    reference_source_admission: ReferenceSourceAdmission | None = None
    recipes: tuple[SelectedRecipeBinding, ...] = Field(min_length=1)
    resources: tuple[SelectedResourceBinding, ...] = Field(min_length=1)
    unresolved_execution_constraints: tuple[StrictStr, ...] = Field(min_length=1)
    evidence_closure: tuple[ProposalEvidenceClosureItem, ...] = Field(min_length=1)
    supersedes_proposal_id: StrictStr | None = None
    revision_event_id: StrictStr | None = None
    revision_direction: ExecutionRevisionContext | None = None
    execution_package: ExecutionPackage | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    configuration: ParabricksConfiguration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    runtime_selection: ExecutionRuntimeSelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # States diploid calling with the model's judgment of the organism; absent only
    # on a proposal whose recipe selection predates the judgment.
    ploidy: ExecutionPloidy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def configured_runtime(self):
        if (
            self.configuration is not None
            and self.runtime_selection != self.configuration.runtime_selection
        ):
            raise ValueError("proposal changed its configured runtime")
        return self

    @model_validator(mode="after")
    def exact_identity_and_lineage(self) -> ExecutionProposal:
        _canonical_uuid(self.journey_id, "journey_id")
        _canonical_uuid(self.plan_event_id, "plan_event_id")
        _canonical_uuid(self.inspection_event_id, "inspection_event_id")
        _canonical_uuid(self.reference_event_id, "reference_event_id")
        if self.reference_basis is not None:
            basis = self.reference_basis
            if (
                basis.validation_event_id != self.reference_event_id
                or basis.validation.references != (self.reference,)
                or basis.validation.selected.selection_event_id
                != basis.choice.source_event_id
            ):
                raise ValueError("proposal changed its exact reference decision basis")
            if not {
                basis.choice.source_event_id,
                basis.validation_event_id,
                *basis.choice.grounding.source_event_ids,
            } <= {e.source_event_id for e in self.evidence_closure}:
                raise ValueError("proposal omitted its reference decision evidence")
        if self.runtime_selection is not None:
            runtime = self.runtime_selection.resolve().h100
            package = self.execution_package
            if not runtime_matches_package(runtime, package):
                raise ValueError(
                    "runtime selection changed approved package resources or recipe"
                )
        if not (
            bool(self.supersedes_proposal_id)
            == bool(self.revision_event_id)
            == bool(self.revision_direction)
        ):
            raise ValueError("proposal supersession requires its exact revision event")
        if self.execution_package is not None and (
            self.execution_package.journey_id != self.journey_id
            or self.execution_package.source_plan.plan_event_id != self.plan_event_id
        ):
            raise ValueError("execution package changed proposal lineage")
        if self.execution_package is not None:
            admission = self.reference_source_admission
            packaged = self.execution_package.reference
            if admission is None or (
                packaged.source_admission_id != admission.admission_id
                or packaged.reference_asset_id != admission.reference_asset_id
                or packaged.assembly_accession != admission.assembly_accession
                or packaged.authority_locator != admission.authority_locator
                or packaged.source_mode != admission.source_mode
                or packaged.source_locator != admission.source_locator
                or packaged.source_representation != admission.source_representation
                or packaged.expected_size_bytes != admission.expected_size_bytes
                or packaged.expected_checksum_algorithm
                != admission.expected_checksum_algorithm
                or packaged.expected_checksum != admission.expected_checksum
                or packaged.checksum_authority_locator
                != admission.checksum_authority_locator
                or packaged.derivation_rule != admission.derivation_rule
            ):
                raise ValueError("execution package changed admitted reference source")
        if self.revision_event_id is not None:
            _canonical_uuid(self.revision_event_id, "revision_event_id")
            if self.revision_direction.revision_event_id != self.revision_event_id:
                raise ValueError("proposal revision direction changed its exact event")
        if (
            self.requirement_resolution.journey_id != self.journey_id
            or self.requirement_resolution.plan_event_id != self.plan_event_id
        ):
            raise ValueError("proposal changed its requirement resolution lineage")
        if (
            self.requirement_resolution.input_set_id
            != self.inspected_inputs.input_set_id
        ):
            raise ValueError("proposal changed the inspected input-set identity")
        if any(
            self.reference.assembly_accession != companion.candidate.assembly_accession
            for companion in self.companions
        ):
            raise ValueError(
                "proposal reference and companion assembly identities disagree"
            )
        if self.reference_source_admission is not None:
            fasta = [
                item.candidate
                for item in self.companions
                if item.role == "genomic_fasta"
            ]
            admission = self.reference_source_admission
            if (
                len(fasta) != 1
                or admission.reference_asset_id != fasta[0].asset_id
                or admission.assembly_accession != self.reference.assembly_accession
                or admission.authority_locator != fasta[0].authority_locator
            ):
                raise ValueError(
                    "proposal source admission changed reference authority"
                )
        if any(
            item.candidate.recipe_candidate_id
            not in {recipe.candidate.candidate_id for recipe in self.recipes}
            for item in self.resources
        ):
            raise ValueError("proposal resource changed its selected recipe dependency")
        event_ids = [item.source_event_id for item in self.evidence_closure]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("proposal evidence closure repeats an event identity")
        material = self.model_dump(mode="json", exclude={"proposal_id"})
        if self.proposal_id != (
            "execution-proposal:" + canonical_json_digest(material)
        ):
            raise ValueError("proposal identity omitted consequence-bearing material")
        return self


class ApproveExecutionResponse(FrozenContract):
    action: Literal["approve"]
    proposal_id: StrictStr
    # The researcher's answer to the proposal's calling ploidy: required exactly
    # when the proposal states one, and only its value is offered.
    calling_ploidy: Literal[2] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ReviseExecutionResponse(FrozenContract):
    action: Literal["revise"]
    proposal_id: StrictStr
    instruction: StrictStr
    revalidation_scope: Literal["execution_planning", "reference"]

    @field_validator("instruction")
    @classmethod
    def canonical_instruction(cls, value: str) -> str:
        return _canonical_text(value, "instruction")


class RejectExecutionResponse(FrozenContract):
    action: Literal["reject"]
    proposal_id: StrictStr
    reason: StrictStr | None = None

    @field_validator("reason")
    @classmethod
    def canonical_reason(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_text(value, "reason")


ExecutionHitlResponse = Annotated[
    ApproveExecutionResponse | ReviseExecutionResponse | RejectExecutionResponse,
    Field(discriminator="action"),
]
_RESPONSE_ADAPTER = TypeAdapter(ExecutionHitlResponse)


class ExecutionProposalSubmitted(FrozenContract):
    review_policy: Annotated[
        ExecutionReviewPolicy, BeforeValidator(explicit_review_policy)
    ]
    """One proposal recording and its inseparable backend-owned review request."""

    event_type: Literal["ExecutionProposalSubmitted"] = "ExecutionProposalSubmitted"
    proposal: ExecutionProposal


class ExecutionHitlResponded(FrozenContract):
    event_type: Literal["ExecutionHitlResponded"] = "ExecutionHitlResponded"
    request_event_id: StrictStr
    response: ExecutionHitlResponse
    authority: Literal["user_decision"] = "user_decision"
    scientifically_verified: Literal[False] = False
    authorizes_execution: bool = False

    @model_validator(mode="after")
    def approval_only(self):
        if self.authorizes_execution and not isinstance(
            self.response, ApproveExecutionResponse
        ):
            raise ValueError("only exact approval grants execution")
        return self

    @field_validator("request_event_id")
    @classmethod
    def canonical_request(cls, value: str) -> str:
        return _canonical_uuid(value, "request_event_id")


class ExecutionAuthority(FrozenContract):
    """The exact approving answer an Execution rests on; never a Journey event."""

    authority_id: StrictStr = Field(pattern=r"^execution-authority:[0-9a-f]{64}$")
    proposal_id: StrictStr
    request_event_id: StrictStr
    approval_event_id: StrictStr

    @model_validator(mode="after")
    def exact_authority(self) -> ExecutionAuthority:
        _canonical_uuid(self.request_event_id, "request_event_id")
        _canonical_uuid(self.approval_event_id, "approval_event_id")
        expected = execution_authority_id(
            self.proposal_id, self.request_event_id, self.approval_event_id
        )
        if self.authority_id != expected:
            raise ValueError("Execution authority changed its exact approval binding")
        return self


def execution_authority_id(
    proposal_id: str, request_event_id: str, approval_event_id: str
) -> str:
    return "execution-authority:" + canonical_json_digest(
        {
            "proposal_id": proposal_id,
            "request_event_id": request_event_id,
            "approval_event_id": approval_event_id,
        }
    )


class ExecutionHitlLegalActionView(FrozenContract):
    action: Literal["approve", "revise", "reject"]
    response_schema: dict[str, JsonValue]


class ExecutionHitlRequestView(FrozenContract):
    review_policy: ExecutionReviewPolicy
    request_event_id: StrictStr
    proposal: ExecutionProposal
    legal_actions: tuple[ExecutionHitlLegalActionView, ...]
    pending: bool


class ExecutionHitlReceipt(FrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response: ExecutionHitlResponse
    failure_event_id: StrictStr | None = None


class ExecutionHitlProjection(FrozenContract):
    requests: tuple[ExecutionHitlRequestView, ...]
    receipts: tuple[ExecutionHitlReceipt, ...]
    pending_request: ExecutionHitlRequestView | None
    authorities: tuple[ExecutionAuthority, ...]
    status: Literal[
        "not_requested",
        "execution_hitl_pending",
        "execution_revision_pending",
        "execution_rejected",
        "execution_authorized",
    ]


ExecutionHitlRefusalCode = Literal[
    "execution_hitl_response_illegal",
    "execution_hitl_response_stale",
    "execution_hitl_response_conflict",
]


class ExecutionHitlResponseRefused(ValueError):
    def __init__(self, code: ExecutionHitlRefusalCode, request_event_id: str):
        super().__init__(code)
        self.code = code
        self.request_event_id = request_event_id
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class ProposalCompilationRefused(Invariant, ValueError):
    """The compiler refuses a Journey planning readiness found complete (#217).

    Completeness belongs to planning readiness, so the refusal is an integrity
    fault: wherever it escapes, the failure module holds the Run under one id,
    with the refusal's code as the cause.
    """

    def __init__(self, code: str, details: tuple[str, ...] = ()):
        super().__init__(
            "proposal_compilation_refused",
            code,
            cause=code,
            evidence={"details": list(details)},
        )
        self.code = code
        self.details = details


def _payload(fact: FrozenContract) -> dict[str, Any]:
    value = fact.model_dump(mode="json")
    value.pop("event_type", None)
    return value


def validate_execution_fact(event: Mapping[str, Any]) -> FrozenContract | None:
    owners = {
        "ExecutionProposalSubmitted": ExecutionProposalSubmitted,
        "ExecutionHitlResponded": ExecutionHitlResponded,
    }
    owner = owners.get(event.get("event_type"))
    return (
        None
        if owner is None
        else owner.model_validate(
            {"event_type": event["event_type"], **event.get("payload", {})}
        )
    )


def _response_schema(
    owner: type[FrozenContract], proposal: ExecutionProposal
) -> dict[str, JsonValue]:
    schema = copy.deepcopy(owner.model_json_schema(mode="validation"))
    schema.pop("title", None)
    schema["properties"]["proposal_id"] = {
        "const": proposal.proposal_id,
        "type": "string",
    }
    if owner is ApproveExecutionResponse:
        if proposal.ploidy is None:
            schema["properties"].pop("calling_ploidy")
        else:
            schema["properties"]["calling_ploidy"] = {
                "enum": [proposal.ploidy.calling_ploidy]
            }
            schema["required"] = [*schema["required"], "calling_ploidy"]
    return schema


def execution_hitl_request_view(
    request_event_id: str,
    proposal: ExecutionProposal,
    *,
    pending: bool,
    review_policy: ExecutionReviewPolicy,
) -> ExecutionHitlRequestView:
    _canonical_uuid(request_event_id, "request_event_id")
    owners: tuple[tuple[str, type[FrozenContract]], ...] = (
        ("approve", ApproveExecutionResponse),
        ("revise", ReviseExecutionResponse),
        ("reject", RejectExecutionResponse),
    )
    return ExecutionHitlRequestView(
        review_policy=review_policy,
        request_event_id=request_event_id,
        proposal=proposal,
        legal_actions=tuple(
            ExecutionHitlLegalActionView(
                action=action,
                response_schema=_response_schema(owner, proposal),
            )
            for action, owner in owners
        ),
        pending=pending,
    )


def validate_execution_hitl_response(
    request: ExecutionHitlRequestView, response: ExecutionHitlResponse
) -> None:
    ploidy = request.proposal.ploidy
    if response.proposal_id != request.proposal.proposal_id or (
        isinstance(response, ApproveExecutionResponse)
        and response.calling_ploidy
        != (None if ploidy is None else ploidy.calling_ploidy)
    ):
        raise ExecutionHitlResponseRefused(
            "execution_hitl_response_illegal", request.request_event_id
        )


def evaluate_execution_hitl_response(
    projection: ExecutionHitlProjection,
    request_event_id: str,
    response: ExecutionHitlResponse,
) -> Literal["accept", "identical"]:
    from bio_neb.junction import evaluate_scoped_answer

    _canonical_uuid(request_event_id, "request_event_id")
    _, retained = evaluate_scoped_answer(
        projection.requests,
        projection.receipts,
        request_event_id,
        response,
        validate=validate_execution_hitl_response,
        refusal=lambda reason, identity: ExecutionHitlResponseRefused(
            f"execution_hitl_response_{reason}", identity
        ),
    )
    return "identical" if retained is not None else "accept"


def approval_binding(
    request_event_id: str, response_event_id: str, proposal_id: str
) -> ExecutionAuthority:
    """An execution reference to the exact answer, never another authority grant."""
    return ExecutionAuthority(
        authority_id=execution_authority_id(
            proposal_id, request_event_id, response_event_id
        ),
        proposal_id=proposal_id,
        request_event_id=request_event_id,
        approval_event_id=response_event_id,
    )


def _collect_named_values(value: object, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    collected: list[str] = []

    def walk(selected: object, key: str = "") -> None:
        if isinstance(selected, Mapping):
            for name, child in selected.items():
                walk(child, str(name))
        elif isinstance(selected, (list, tuple)):
            for child in selected:
                walk(child, key)
        elif (
            isinstance(selected, str)
            and any(key.endswith(suffix) for suffix in suffixes)
            and selected not in collected
        ):
            collected.append(selected)

    walk(value)
    return tuple(collected)


def compile_execution_proposal(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ExecutionProposal:
    """Re-derive one complete proposal without accepting caller-supplied consequences."""

    return _compile_execution_proposal(
        PrefixSnapshot(events), bind_execution_package=True
    )


def _compile_agent_planned_execution_proposal(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ExecutionProposal:
    """Compile the current Tool-grounded plan before its exact Execution package.

    Preparation meaning reads no model transport or contact evidence, so the
    writer's semantic record compiles it without the journal-only contacts;
    readers on merged evidence keep transport integrity.
    """

    return _compile_execution_proposal(
        PrefixSnapshot(events, check_model_transport=False),
        bind_execution_package=False,
    )


def _compile_execution_proposal(
    context: PrefixSnapshot,
    *,
    bind_execution_package: bool,
) -> ExecutionProposal:
    timeline = context.timeline
    packages = [
        (event, fact)
        for event in timeline
        if isinstance(
            (fact := validate_execution_package_fact(event)),
            ExecutionPackageCompiled,
        )
    ]
    if packages and bind_execution_package:
        package_event, package_fact = packages[-1]
        index = timeline.index(package_event)
        if index == 0:
            raise ProposalCompilationRefused("execution_package_ancestry_invalid")
        planned = _compile_execution_proposal(
            context.before(package_event["event_id"]), bind_execution_package=False
        )
        package = package_fact.package
        planned_sources = tuple(
            item.source_event_id for item in planned.evidence_closure
        )
        expected_parents = {
            package.source_plan.checkpoint_event_id,
            package.reference.admission_event_id,
            *planned_sources,
        }
        if (
            package.journey_id != planned.journey_id
            or package.source_plan.planned_proposal_id != planned.proposal_id
            or package.source_plan.plan_event_id != planned.plan_event_id
            or tuple(
                item.model_dump(mode="json")
                for item in package.source_plan.evidence_closure
            )
            != tuple(item.model_dump(mode="json") for item in planned.evidence_closure)
            or not expected_parents <= set(package_event.get("parent_ids", ()))
        ):
            raise ProposalCompilationRefused("execution_package_evidence_changed")
        closure_item = ProposalEvidenceClosureItem(
            source_event_id=package_event["event_id"],
            event_type=package_event["event_type"],
            evidence_identities=(package.package_id,),
        )
        material = planned.model_dump(mode="json", exclude={"proposal_id"})
        material["execution_package"] = package.model_dump(mode="json")
        material["evidence_closure"] = [
            *material["evidence_closure"],
            closure_item.model_dump(mode="json"),
        ]
        configuration_event, configuration = current_configuration(timeline)
        if configuration is not None:
            material["configuration"] = configuration.model_dump(mode="json")
            material["runtime_selection"] = configuration.runtime_selection.model_dump(
                mode="json"
            )
            material["evidence_closure"].append(
                ProposalEvidenceClosureItem(
                    source_event_id=configuration_event["event_id"],
                    event_type=configuration_event["event_type"],
                    evidence_identities=(),
                ).model_dump(mode="json")
            )
        return ExecutionProposal(
            proposal_id=("execution-proposal:" + canonical_json_digest(material)),
            **material,
        )
    return _compile_from_planning(context.read, context)


def _compile_from_planning(
    meaning: PlanningRead, evidence: PrefixSnapshot
) -> ExecutionProposal:
    timeline, by_id = evidence.timeline, evidence.by_id
    planning = meaning.execution
    if planning is None:
        raise ProposalCompilationRefused("proposal_compiled_before_planning_ready")
    if planning.status != "execution_planning_ready":
        raise ProposalCompilationRefused("proposal_compiled_before_planning_ready")
    plan_fact = meaning.plan
    if not isinstance(plan_fact, ProposalRequirementsIdentified):
        raise ProposalCompilationRefused("proposal_information_plan_invalid")
    # A recorded plan always has its resolution; once ready, its execution part.
    resolution = meaning.requirements
    if (
        resolution.plan_event_id != planning.plan_event_id
        or resolution.plan_id != plan_fact.plan.plan_id
    ):
        raise ProposalCompilationRefused("proposal_requirement_lineage_changed")
    if not resolution.ready_for_execution_compilation:
        raise ProposalCompilationRefused(
            "proposal_compiled_before_evidence_resolved",
            resolution.unresolved_execution_requirement_ids,
        )
    from bio_neb.application.input_journey import project_input_selection

    selected = project_input_selection(timeline)
    if selected.correction_pending or selected.stopped or not selected.inspections:
        raise ProposalCompilationRefused("proposal_input_evidence_invalid")
    inspection_event = selected.inspections[-1]
    inspection = validate_fact(inspection_event)
    if (
        not isinstance(inspection, InputMetadataInspected)
        or inspection.result.status != "succeeded"
    ):
        raise ProposalCompilationRefused("proposal_input_evidence_invalid")
    reference = meaning.reference
    if (
        reference is None
        or reference.verified_reference is None
        or reference.description_event_id is None
    ):
        raise ProposalCompilationRefused("proposal_reference_evidence_invalid")
    companions = tuple(
        ProposedCompanionAsset(**binding.model_dump())
        for binding in meaning.verified_companion_bindings
    )
    source_admissions = [
        (event, fact)
        for event in timeline
        if isinstance(
            (fact := validate_reference_source_fact(event)), ReferenceSourceAdmitted
        )
        and (
            fact.reference_event_id == reference.description_event_id
            or (fact.reference_event_id is None and not reference.choices)
        )
    ]
    # The writer admits one source per described reference.
    ((source_admission_event, source_admission_fact),) = source_admissions or [
        (None, None)
    ]

    submitted = [
        (event, fact)
        for event in timeline
        if isinstance(
            (fact := validate_execution_fact(event)), ExecutionProposalSubmitted
        )
    ]
    supersedes: str | None = None
    revision_event_id: str | None = None
    revision_direction: ExecutionRevisionContext | None = None
    if submitted:
        # Planning is ready again only after the researcher's revise answer: its
        # direction is the planning view's revision context, and every selection
        # descends from it.
        supersedes = submitted[-1][1].proposal.proposal_id
        revision_direction = planning.revision_context
        revision_event_id = revision_direction.revision_event_id

    base_material: dict[str, Any] = {
        "journey_id": planning.journey_id,
        "plan_event_id": planning.plan_event_id,
        "requirement_resolution": resolution.execution_resolution.model_dump(
            mode="json"
        ),
        "inspection_event_id": inspection_event["event_id"],
        "inspected_inputs": inspection.result.model_dump(mode="json"),
        "reference_event_id": reference.description_event_id,
        "reference": reference.verified_reference.model_dump(mode="json"),
        "companions": [item.model_dump(mode="json") for item in companions],
        "reference_source_admission": (
            source_admission_fact.admission.model_dump(mode="json")
            if source_admission_fact is not None
            else None
        ),
        "recipes": [
            item.model_dump(mode="json") for item in planning.selected_recipe_bindings
        ],
        "resources": [
            item.model_dump(mode="json") for item in planning.selected_resource_bindings
        ],
        "unresolved_execution_constraints": list(
            planning.unresolved_execution_constraints
        ),
        "supersedes_proposal_id": supersedes,
        "revision_event_id": revision_event_id,
        "revision_direction": (
            revision_direction.model_dump(mode="json")
            if revision_direction is not None
            else None
        ),
    }
    ploidy = _execution_ploidy(planning)
    if ploidy is not None:
        base_material["ploidy"] = ploidy.model_dump(mode="json")
    if resolution.fact_slots:
        base_material["fact_slots"] = [
            item.model_dump(mode="json") for item in resolution.fact_slots
        ]
    basis = reference_decision_basis(reference)
    if basis is not None:
        base_material["reference_basis"] = basis.model_dump(mode="json")
    source_event_ids = tuple(
        dict.fromkeys(
            [
                planning.plan_event_id,
                inspection_event["event_id"],
                reference.description_event_id,
                *(
                    [
                        basis.choice.source_event_id,
                        *basis.choice.grounding.source_event_ids,
                    ]
                    if basis is not None
                    else []
                ),
                *[item.validation_event_id for item in companions],
                *[
                    item.prebuilt_choice_event_id
                    for item in companions
                    if item.prebuilt_choice_event_id is not None
                ],
                *(
                    [source_admission_event["event_id"]]
                    if source_admission_event is not None
                    else []
                ),
                *[
                    item.selection_event_id
                    for item in planning.selected_recipe_bindings
                ],
                *[
                    item.selection_event_id
                    for item in planning.selected_resource_bindings
                ],
                *[
                    event_id
                    for item in planning.requirements
                    for event_id in item.evidence_event_ids
                ],
                *[
                    event_id
                    for item in resolution.execution_resolution.requirements
                    for event_id in item.requirement_value.lineage_event_ids
                ],
                *([ploidy.organism.record_event_id] if ploidy is not None else []),
            ]
        )
    )
    closure = tuple(
        ProposalEvidenceClosureItem(
            source_event_id=event_id,
            event_type=by_id[event_id]["event_type"],
            evidence_identities=_collect_named_values(
                by_id[event_id].get("payload", {}), ("evidence_id", "evidence_ids")
            ),
        )
        for event_id in source_event_ids
    )
    material = {
        **base_material,
        "evidence_closure": [item.model_dump(mode="json") for item in closure],
    }
    return ExecutionProposal(
        proposal_id=("execution-proposal:" + canonical_json_digest(material)),
        **material,
    )


def project_execution_hitl(  # noqa: C901 - Existing orchestration complexity; contract-base migration preserves this flow.
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ExecutionHitlProjection:
    """Replay proposal review, supersession, decisions, and exact approval contact-free."""

    context = PrefixSnapshot(events)
    timeline, ancestors = context.timeline, context.ancestry
    requests: list[tuple[dict[str, Any], ExecutionProposalSubmitted]] = []
    responses: dict[str, tuple[dict[str, Any], ExecutionHitlResponded]] = {}
    authorities: list[ExecutionAuthority] = []
    rejections: dict[str, str] = {}
    pending: str | None = None
    for event in timeline:
        fact = validate_execution_fact(event)
        if isinstance(fact, ExecutionProposalSubmitted):
            if pending is not None:
                raise ValueError(
                    "Execution proposal was submitted while review was pending"
                )
            expected = _compile_execution_proposal(
                context.before(event["event_id"]), bind_execution_package=True
            )
            if fact.proposal != expected:
                raise ValueError(
                    "submitted Execution proposal was not deterministically compiled"
                )
            if (
                not {item.source_event_id for item in fact.proposal.evidence_closure}
                <= ancestors[event["event_id"]]
            ):
                raise ValueError(
                    "submitted Execution proposal evidence is outside request ancestry"
                )
            requests.append((event, fact))
            pending = event["event_id"]
        elif isinstance(fact, ExecutionHitlResponded):
            if fact.request_event_id != pending or fact.request_event_id not in {
                item[0]["event_id"] for item in requests
            }:
                raise ValueError(
                    "Execution HITL response did not target the pending request"
                )
            if event.get("parent_ids") != [fact.request_event_id]:
                raise ValueError("Execution HITL response is outside request ancestry")
            request = next(item for item in requests if item[0]["event_id"] == pending)
            view = execution_hitl_request_view(
                pending,
                request[1].proposal,
                pending=True,
                review_policy=request[1].review_policy,
            )
            validate_execution_hitl_response(view, fact.response)
            responses[pending] = (event, fact)
            if fact.authorizes_execution:
                authorities.append(
                    approval_binding(
                        pending, event["event_id"], fact.response.proposal_id
                    )
                )
            pending = None
        elif (
            event.get("event_type") == "RunFailed"
            and event.get("payload", {}).get("code") == "rejected_by_user"
        ):
            source_id = event.get("payload", {}).get("source_event_id")
            if any(
                prior["event_id"] == source_id
                and prior["event_type"] == "InvestigationReviewResponded"
                for prior in timeline
            ):
                # An issue rejection ends the Run after Execution; the Run
                # reader checks that decision, not this Review.
                continue
            rejected = next(
                (
                    response
                    for response in responses.values()
                    if response[0]["event_id"] == source_id
                    and isinstance(response[1].response, RejectExecutionResponse)
                    and event.get("parent_ids") == [source_id]
                ),
                None,
            )
            if rejected is None:
                raise ValueError(
                    "user rejection terminal lacks exact decision ancestry"
                )
            rejections[rejected[1].response.proposal_id] = event["event_id"]

    receipts: list[ExecutionHitlReceipt] = []
    for request_event, request in requests:
        retained = responses.get(request_event["event_id"])
        if retained is None:
            continue
        response_event, response = retained
        receipts.append(
            ExecutionHitlReceipt(
                request_event_id=request_event["event_id"],
                response_event_id=response_event["event_id"],
                response=response.response,
                failure_event_id=rejections.get(request.proposal.proposal_id),
            )
        )
    request_views = tuple(
        execution_hitl_request_view(
            event["event_id"],
            fact.proposal,
            pending=event["event_id"] == pending,
            review_policy=fact.review_policy,
        )
        for event, fact in requests
    )
    last_response = receipts[-1].response if receipts else None
    status: str
    if pending is not None:
        status = "execution_hitl_pending"
    elif isinstance(last_response, ReviseExecutionResponse):
        status = "execution_revision_pending"
    elif isinstance(last_response, RejectExecutionResponse):
        if requests[-1][1].proposal.proposal_id not in rejections:
            raise ValueError("rejected proposal lacks typed user terminal")
        status = "execution_rejected"
    elif isinstance(last_response, ApproveExecutionResponse):
        last_proposal = requests[-1][1].proposal.proposal_id
        if not any(item.proposal_id == last_proposal for item in authorities):
            raise ValueError("approved proposal lacks its exact answer authority")
        status = "execution_authorized"
    else:
        status = "not_requested"
    return ExecutionHitlProjection(
        requests=request_views,
        receipts=tuple(receipts),
        pending_request=next((item for item in request_views if item.pending), None),
        authorities=tuple(authorities),
        status=status,
    )


def current_execution_approval(
    events,
) -> tuple[ExecutionAuthority | None, ExecutionProposal | None]:
    """Select the current approval from a checked planning prefix.

    The last Execution review fact must be the approving answer to the latest
    submitted proposal; a later submission or other answer leaves none.
    """
    approvals = [
        event
        for event in events
        if event["event_type"]
        in {"ExecutionProposalSubmitted", "ExecutionHitlResponded"}
    ]
    if not approvals:
        return None, None
    last = approvals[-1]
    fact = validate_execution_fact(last)
    if not (isinstance(fact, ExecutionHitlResponded) and fact.authorizes_execution):
        return None, None
    candidate = approval_binding(
        fact.request_event_id, last["event_id"], fact.response.proposal_id
    )
    submissions = [
        e for e in approvals if e["event_type"] == "ExecutionProposalSubmitted"
    ]
    if not submissions or submissions[-1]["event_id"] != candidate.request_event_id:
        return None, None
    selected = validate_execution_fact(submissions[-1]).proposal
    if candidate.proposal_id != selected.proposal_id:
        return None, None
    return candidate, selected


__all__ = [
    "ApproveExecutionResponse",
    "ExecutionAuthority",
    "ExecutionHitlProjection",
    "ExecutionHitlResponse",
    "ExecutionHitlResponseRefused",
    "ExecutionProposal",
    "ExecutionProposalSubmitted",
    "ProposalCompilationRefused",
    "RejectExecutionResponse",
    "ReviseExecutionResponse",
    "compile_execution_proposal",
    "evaluate_execution_hitl_response",
    "execution_authority_id",
    "project_execution_hitl",
]
