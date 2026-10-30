"""Compile the exact Journey frontier into descriptor-only scientific authority."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from bio_neb.application.queries import (
    JourneyReview,
    project_journey_events,
)
from bio_neb.content import ContentId
from bio_neb.contracts import StrictFrozenContract
from bio_neb.journey import (
    JourneyEvent,
    JourneyFile,
    JourneyProjection,
    JourneyValidationError,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.journey_path import JourneyPath
from bio_neb.parabricks.execution_package import ExecutionInputSource
from bio_neb.parabricks.requirement_resolution import (
    ScientificInvestigationHandoff,
)
from bio_neb.parabricks.result import (
    ParabricksExecutionOutcome,
    read_completed_parabricks_result,
)
from bio_neb.reference_discovery.reference_source import SourceDerivationRule
from bio_neb.verified_artifacts import VerifiedObject

ScientificContinuationRefusalCode = Literal[
    "authority_mismatch",
    "descriptor_drift",
    "evidence_inactive",
    "evidence_superseded",
    "frontier_ambiguous",
    "frontier_missing",
    "frontier_not_terminal",
    "input_unacquired",
    "journey_mismatch",
    "journey_projection_invalid",
    "lineage_invalid",
    "outcome_mismatch",
    "proposal_mismatch",
]


class ScientificContinuationRefused(ValueError):
    """Stable contact-free refusal of an inconsistent continuation frontier."""

    def __init__(self, code: ScientificContinuationRefusalCode, *details: str) -> None:
        validated = TypeAdapter(ScientificContinuationRefusalCode).validate_python(
            code, strict=True
        )
        super().__init__(validated)
        object.__setattr__(self, "_code", validated)
        object.__setattr__(self, "_details", tuple(details))
        object.__setattr__(self, "_sealed", True)

    @property
    def code(self) -> ScientificContinuationRefusalCode:
        return self._code

    @property
    def details(self) -> tuple[str, ...]:
        return self._details

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("scientific continuation refusal is immutable")
        object.__setattr__(self, name, value)


ScientificContinuationPublicationRefusalCode = Literal[
    "scientific_continuation_output_exists",
    "scientific_continuation_output_invalid",
    "scientific_continuation_publication_failed",
    "scientific_continuation_reopen_mismatch",
]


class ScientificContinuationPublicationRefused(ValueError):
    """Typed refusal owned by publication, separate from scientific compilation."""

    def __init__(
        self,
        code: ScientificContinuationPublicationRefusalCode,
        *details: str,
    ) -> None:
        validated = TypeAdapter(
            ScientificContinuationPublicationRefusalCode
        ).validate_python(code, strict=True)
        super().__init__(validated)
        self.code = validated
        self.details = tuple(details)


class PublishScientificContinuationCommand(StrictFrozenContract):
    """The two public locators accepted by scientific-continuation publication."""

    execution_journey_file: JourneyFile
    output: StrictStr

    @field_validator("output")
    @classmethod
    def absolute_canonical_path(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError(
                "scientific continuation paths must be absolute and canonical"
            )
        return value


class ScientificContinuationPublicationReceipt(StrictFrozenContract):
    schema_version: Literal["bio-neb.scientific-continuation-publication/1"] = (
        "bio-neb.scientific-continuation-publication/1"
    )
    status: Literal["scientific_continuation_published"] = (
        "scientific_continuation_published"
    )
    output: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    continuation_id: StrictStr
    journey_id: StrictStr
    frontier_event_id: StrictStr


class CausalJourneyLocator(StrictFrozenContract):
    journey_id: StrictStr
    event_id: StrictStr
    event_type: StrictStr
    parent_event_ids: tuple[StrictStr, ...]

    @classmethod
    def from_event(
        cls, event: JourneyEvent, *, journey_id: str | None = None
    ) -> "CausalJourneyLocator":
        """Retain typed event lineage under the identity selected by its owner."""
        return cls(
            journey_id=event.journey_id if journey_id is None else journey_id,
            event_id=event.event_id,
            event_type=event.event_type,
            parent_event_ids=tuple(event.parent_ids),
        )

    @field_validator("journey_id", "event_id")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        if str(uuid.UUID(value)) != value:
            raise ValueError("causal Journey identities must be canonical UUIDs")
        return value

    @field_validator("parent_event_ids")
    @classmethod
    def canonical_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(str(uuid.UUID(item)) != item for item in value):
            raise ValueError("causal Journey parents must be canonical UUIDs")
        return value


def iter_causal_locators(value: object) -> Iterable[CausalJourneyLocator]:
    if isinstance(value, CausalJourneyLocator):
        yield value
        return
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from iter_causal_locators(getattr(value, name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_causal_locators(item)


class ContinuationUserAssertion(StrictFrozenContract):
    role: Literal["user_assertion"] = "user_assertion"
    subject_key: StrictStr
    asserted_value: StrictStr
    assertion_kind: Literal["research_request", "planning_hitl"]
    locator: CausalJourneyLocator
    evidence_locators: tuple[CausalJourneyLocator, ...] = ()
    scientifically_verified: Literal[False] = False


class ContinuationRetrievedSource(StrictFrozenContract):
    role: Literal["retrieved_source"] = "retrieved_source"
    subject_key: StrictStr
    source_kind: StrictStr
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    locator: CausalJourneyLocator
    evidence_locators: tuple[CausalJourneyLocator, ...] = ()
    scientifically_admitted: Literal[False] = False


class ContinuationCandidateEvidence(StrictFrozenContract):
    role: Literal["candidate_evidence"] = "candidate_evidence"
    subject_key: StrictStr
    candidate_id: StrictStr
    candidate_kind: StrictStr
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["selected", "not_selected", "superseded", "unresolved"]
    locator: CausalJourneyLocator
    evidence_locators: tuple[CausalJourneyLocator, ...] = ()
    scientifically_admitted: Literal[False] = False


class ContinuationOpenItem(StrictFrozenContract):
    subject_key: StrictStr
    state: Literal["unresolved", "conflicted", "rejected", "deferred", "superseded"]
    reason: StrictStr
    locators: tuple[CausalJourneyLocator, ...] = Field(min_length=1)


class ContinuationPathReceipt(StrictFrozenContract):
    review_kind: Literal["planning", "execution"]
    action: StrictStr
    request_locator: CausalJourneyLocator
    response_locator: CausalJourneyLocator
    evidence_locators: tuple[CausalJourneyLocator, ...] = ()
    authority: Literal["user_assertion", "user_decision"]
    scientifically_verified: Literal[False] = False


class ContinuationInput(StrictFrozenContract):
    role: StrictStr
    file_id: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    registration_locator: CausalJourneyLocator
    inspection_locator: CausalJourneyLocator


class ContinuationSample(StrictFrozenContract):
    sample_id: StrictStr
    requirement_id: StrictStr
    evidence_locators: tuple[CausalJourneyLocator, ...] = Field(min_length=1)


class ContinuationSourceAdmission(StrictFrozenContract):
    """Execution source admission, explicitly not a scientific admitted fact."""

    admission_id: StrictStr
    reference_asset_id: StrictStr
    assembly_accession: StrictStr
    source_mode: Literal["public_url", "local_file", "existing_object"]
    source_representation: Literal["gzip_fasta", "plain_fasta"]
    expected_size_bytes: StrictInt = Field(gt=0)
    expected_checksum_algorithm: Literal["md5", "sha256", "sha512"]
    expected_checksum: StrictStr
    derivation_rule: SourceDerivationRule
    authority_locator_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    checksum_authority_locator_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    locator: CausalJourneyLocator
    scientifically_admitted: Literal[False] = False

    @model_validator(mode="after")
    def exact_content(self) -> ContinuationSourceAdmission:
        content = self.model_dump(
            mode="json",
            exclude={"content_sha256", "locator", "scientifically_admitted"},
        )
        if self.content_sha256 != canonical_json_digest(content):
            raise ValueError("source admission content digest changed")
        return self


class ScientificAdmittedFact(StrictFrozenContract):
    """Reserved role; Ticket 123 admits no scientific fact."""

    subject_key: StrictStr
    fact_id: StrictStr
    locator: CausalJourneyLocator


class ContinuationReference(StrictFrozenContract):
    assembly_accession: StrictStr
    reference_asset_id: StrictStr
    representation: StrictStr
    evidence_id: StrictStr
    source_admission_id: StrictStr
    selection_locator: CausalJourneyLocator
    admission_locator: CausalJourneyLocator


class ContinuationWorkflow(StrictFrozenContract):
    workflow_id: StrictStr
    workflow_version: StrictStr
    recipe_candidate_id: StrictStr
    assay: StrictStr
    read_layout: Literal["paired", "single"]
    meaning_parameters: tuple[tuple[StrictStr, StrictStr], ...]
    evidence_locator: CausalJourneyLocator


class ContinuationResource(StrictFrozenContract):
    profile_id: StrictStr
    candidate_id: StrictStr
    accelerator_count: StrictInt = Field(ge=1)
    cpu_count: StrictInt = Field(ge=1)
    memory_gib: StrictInt = Field(ge=1)
    accelerator_memory_gib: StrictInt = Field(ge=1)
    evidence_locator: CausalJourneyLocator


class ContinuationExecutionIdentity(StrictFrozenContract):
    execution_package_id: StrictStr
    proposal_id: StrictStr
    authority_id: StrictStr
    prepared_reference_receipt_id: StrictStr
    h100_receipt_id: StrictStr
    h100_configuration_id: StrictStr
    h100_request_id: StrictStr
    proposal_locator: CausalJourneyLocator
    approval_locator: CausalJourneyLocator
    receipt_locator: CausalJourneyLocator


class ScientificContinuationPacket(StrictFrozenContract):
    schema_version: Literal["bio-neb.scientific-continuation/1"] = (
        "bio-neb.scientific-continuation/1"
    )
    continuation_id: StrictStr = Field(
        pattern=r"^scientific-continuation:[0-9a-f]{64}$"
    )
    journey_id: StrictStr
    frontier: CausalJourneyLocator
    original_prompt: StrictStr
    prompt_locator: CausalJourneyLocator
    user_assertions: tuple[ContinuationUserAssertion, ...]
    retrieved_sources: tuple[ContinuationRetrievedSource, ...]
    candidate_evidence: tuple[ContinuationCandidateEvidence, ...]
    admitted_facts: tuple[ScientificAdmittedFact, ...] = ()
    scientific_investigation: ScientificInvestigationHandoff
    source_admission: ContinuationSourceAdmission
    open_items: tuple[ContinuationOpenItem, ...]
    path_receipts: tuple[ContinuationPathReceipt, ...]
    input_set_id: StrictStr
    inputs: tuple[ContinuationInput, ...] = Field(min_length=1)
    sample: ContinuationSample
    reference: ContinuationReference
    workflow: ContinuationWorkflow
    resource: ContinuationResource
    execution: ContinuationExecutionIdentity
    verified_objects: tuple[VerifiedObject, ...] = Field(min_length=4, max_length=4)
    vcf: VerifiedObject
    grants_live_access: Literal[False] = False

    @model_validator(mode="after")
    def exact_packet(self) -> ScientificContinuationPacket:
        if self.admitted_facts:
            raise ValueError("Ticket 123 cannot promote a scientific admitted fact")
        if tuple(item.role for item in self.verified_objects) != (
            "aligned_bam",
            "VCF",
            "worker_receipt",
            "execution_manifest",
        ):
            raise ValueError("continuation changed verified-object order")
        if self.vcf != self.verified_objects[1]:
            raise ValueError("continuation VCF descriptor changed")
        if any(
            locator.journey_id != self.journey_id
            for locator in iter_causal_locators(self)
        ):
            raise ValueError("continuation mixed causal Journey identities")
        material = self.model_dump(mode="json", exclude={"continuation_id"})
        if self.continuation_id != (
            "scientific-continuation:" + canonical_json_digest(material)
        ):
            raise ValueError("continuation identity omitted retained authority")
        return self


def _event(value: JourneyEvent | Mapping[str, Any]) -> dict[str, Any]:
    return (
        value.model_dump(mode="json")
        if isinstance(value, JourneyEvent)
        else dict(value)
    )


def _public_json(value: object) -> JsonValue:
    """Remove operational path fields while retaining semantic JSON content."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        omitted = {
            "journey_file",
            "prompt_file",
            "registered_path",
            "run_root",
            "runtime_config_file",
            "source_locator",
        }
        return {
            str(key): _public_json(item)
            for key, item in value.items()
            if str(key) not in omitted and not str(key).endswith("_path")
        }
    if isinstance(value, (tuple, list)):
        return [_public_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("continuation content is not JSON")


def _locator(
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    event_id: str,
    expected_type: str | None = None,
) -> CausalJourneyLocator:
    event = by_id.get(event_id)
    if event is None or (
        expected_type is not None and event.get("event_type") != expected_type
    ):
        raise ScientificContinuationRefused("lineage_invalid", event_id)
    return CausalJourneyLocator(
        journey_id=journey_id,
        event_id=event_id,
        event_type=str(event["event_type"]),
        parent_event_ids=tuple(event.get("parent_ids", ())),
    )


def _ancestors(by_id: Mapping[str, Mapping[str, Any]], event_id: str) -> set[str]:
    pending = [event_id]
    found: set[str] = set()
    while pending:
        selected = pending.pop()
        if selected in found:
            continue
        event = by_id.get(selected)
        if event is None:
            raise ScientificContinuationRefused("lineage_invalid", selected)
        found.add(selected)
        pending.extend(event.get("parent_ids", ()))
    return found


def _nested_event_ids(value: object) -> set[str]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        selected: set[str] = set()
        for key, item in value.items():
            if key.endswith("event_id") and isinstance(item, str):
                selected.add(item)
            elif key.endswith("event_ids") and isinstance(item, (tuple, list)):
                selected.update(value for value in item if isinstance(value, str))
            selected.update(_nested_event_ids(item))
        return selected
    if isinstance(value, (tuple, list)):
        selected: set[str] = set()
        for item in value:
            selected.update(_nested_event_ids(item))
        return selected
    return set()


def _locators(
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    event_ids: Iterable[str],
) -> tuple[CausalJourneyLocator, ...]:
    return tuple(_locator(journey_id, by_id, event_id) for event_id in event_ids)


def _active_execution(
    *,
    outcome: ParabricksExecutionOutcome,
    review: JourneyReview,
    by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Any, Any]:
    hitl = review.execution_hitl
    if (
        hitl is None
        or hitl.status != "execution_authorized"
        or hitl.pending_request is not None
        or not hitl.requests
    ):
        raise ScientificContinuationRefused("authority_mismatch", "not_authorized")
    active = hitl.requests[-1]
    proposal = active.proposal
    if proposal.proposal_id != outcome.proposal_id:
        raise ScientificContinuationRefused("proposal_mismatch", proposal.proposal_id)
    receipt = next(
        (
            item
            for item in hitl.receipts
            if item.request_event_id == active.request_event_id
        ),
        None,
    )
    authority = next(
        (item for item in hitl.authorities if item.proposal_id == proposal.proposal_id),
        None,
    )
    if (
        receipt is None
        or receipt.response.action != "approve"
        or authority is None
        or authority.authority_id != outcome.authority_id
        or authority.approval_event_id != receipt.response_event_id
        or authority.request_event_id != active.request_event_id
    ):
        raise ScientificContinuationRefused("authority_mismatch")
    previous = None
    receipt_by_request = {item.request_event_id: item for item in hitl.receipts}
    for request in hitl.requests:
        selected = request.proposal
        if previous is None:
            if selected.supersedes_proposal_id is not None:
                raise ScientificContinuationRefused(
                    "evidence_superseded", selected.proposal_id
                )
        else:
            prior_receipt = receipt_by_request.get(previous.request_event_id)
            if (
                selected.supersedes_proposal_id != previous.proposal.proposal_id
                or prior_receipt is None
                or prior_receipt.response.action != "revise"
                or selected.revision_event_id != prior_receipt.response_event_id
            ):
                raise ScientificContinuationRefused(
                    "evidence_superseded", selected.proposal_id
                )
        previous = request
    return active, receipt, authority


def _acquired_input_checksums(
    outcome: ParabricksExecutionOutcome, inputs: Iterable[ExecutionInputSource]
) -> dict[str, str]:
    """Each input's identity is what the Run acquired, never its expectation."""

    acquired: dict[str, str] = {}
    for item in inputs:
        proofs = [proof for proof in outcome.input_proofs if proof.role == item.role]
        if len(proofs) != 1:
            raise ScientificContinuationRefused("input_unacquired", item.role)
        acquired[item.role] = ContentId.parse(proofs[0].content_id).digest
    return acquired


def _path_receipts(
    *,
    review: JourneyReview,
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    positions: Mapping[str, int],
    active_ancestors: set[str],
) -> tuple[ContinuationPathReceipt, ...]:
    retained: list[ContinuationPathReceipt] = []
    for receipt in review.planning_hitl.receipts:
        if receipt.response_event_id not in active_ancestors:
            continue
        retained.append(
            ContinuationPathReceipt(
                review_kind="planning",
                action=receipt.response.action,
                request_locator=_locator(journey_id, by_id, receipt.request_event_id),
                response_locator=_locator(journey_id, by_id, receipt.response_event_id),
                evidence_locators=_locators(
                    journey_id, by_id, receipt.assertion.evidence_event_ids
                ),
                authority="user_assertion",
            )
        )
    if review.execution_hitl is not None:
        for receipt in review.execution_hitl.receipts:
            if receipt.response_event_id not in active_ancestors:
                continue
            retained.append(
                ContinuationPathReceipt(
                    review_kind="execution",
                    action=receipt.response.action,
                    request_locator=_locator(
                        journey_id, by_id, receipt.request_event_id
                    ),
                    response_locator=_locator(
                        journey_id, by_id, receipt.response_event_id
                    ),
                    authority="user_decision",
                )
            )
    return tuple(
        sorted(retained, key=lambda item: positions[item.response_locator.event_id])
    )


def _assertions(
    *,
    review: JourneyReview,
    proposal: Any,
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    active_ancestors: set[str],
) -> tuple[tuple[ContinuationUserAssertion, ...], list[ContinuationOpenItem]]:
    assertions = [
        ContinuationUserAssertion(
            subject_key="research_request",
            asserted_value=review.request.prompt,
            assertion_kind="research_request",
            locator=_locator(journey_id, by_id, review.request.source_event_id),
        )
    ]
    open_items: list[ContinuationOpenItem] = []
    resolution = review.requirement_resolution
    if resolution is None:
        raise ScientificContinuationRefused(
            "descriptor_drift", "requirement_resolution"
        )
    for requirement in resolution.requirements:
        for assertion in requirement.user_assertions:
            if assertion.assertion_kind != "research_request":
                continue
            assertions.append(
                ContinuationUserAssertion(
                    subject_key=requirement.subject_key,
                    asserted_value=assertion.asserted_value,
                    assertion_kind="research_request",
                    locator=_locator(journey_id, by_id, assertion.source_event_id),
                    evidence_locators=_locators(
                        journey_id, by_id, assertion.evidence_event_ids
                    ),
                )
            )
    receipts: dict[tuple[str, str, str], list[Any]] = {}
    for receipt in review.planning_hitl.receipts:
        if receipt.response_event_id in active_ancestors:
            key = (receipt.plan_id, receipt.requirement_id, receipt.subject_key)
            receipts.setdefault(key, []).append(receipt)
    for selected in receipts.values():
        for prior in selected[:-1]:
            open_items.append(
                ContinuationOpenItem(
                    subject_key=prior.subject_key,
                    state="superseded",
                    reason="later_planning_assertion",
                    locators=(_locator(journey_id, by_id, prior.response_event_id),),
                )
            )
        receipt = selected[-1]
        assertions.append(
            ContinuationUserAssertion(
                subject_key=receipt.subject_key,
                asserted_value=receipt.assertion.asserted_value,
                assertion_kind="planning_hitl",
                locator=_locator(journey_id, by_id, receipt.response_event_id),
                evidence_locators=_locators(
                    journey_id, by_id, receipt.assertion.evidence_event_ids
                ),
            )
        )
    return tuple(assertions), open_items


def _retrieved_sources(
    *,
    review: JourneyReview,
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    active_ids: set[str],
    active_ancestors: set[str],
) -> tuple[ContinuationRetrievedSource, ...]:
    sources: list[ContinuationRetrievedSource] = []
    groups = (
        getattr(review.reference_identity, "facts", ())
        if review.reference_identity is not None
        else (),
        getattr(review.reference_facts, "facts", ())
        if review.reference_facts is not None
        else (),
    )
    for facts in groups:
        for fact in facts:
            source_event_id = fact.source_event_id
            if (
                source_event_id not in active_ids
                or source_event_id not in active_ancestors
            ):
                continue
            content = _public_json(getattr(fact, "result", {}))
            if (
                not isinstance(content, dict)
                or content.get("status") != "observed"
                or not isinstance(content.get("source"), dict)
            ):
                # Assertions, failures, and tool-local attestations retain their
                # original roles; only source-bearing observations are Retrieved.
                continue
            if not content["source"]:
                raise ScientificContinuationRefused("descriptor_drift", source_event_id)
            subject = str(
                getattr(fact, "work_item_id", None)
                or getattr(fact, "event_type", None)
                or "retrieved_source"
            )
            source_kind = str(
                getattr(fact, "event_type", None)
                or getattr(fact, "kind", None)
                or "tool_observation"
            )
            sources.append(
                ContinuationRetrievedSource(
                    subject_key=subject,
                    source_kind=source_kind,
                    content_sha256=canonical_json_digest(content),
                    locator=_locator(journey_id, by_id, source_event_id),
                )
            )
    return tuple(sources)


def _candidate_evidence(
    *,
    review: JourneyReview,
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    active_ids: set[str],
    active_ancestors: set[str],
) -> tuple[tuple[ContinuationCandidateEvidence, ...], list[ContinuationOpenItem]]:
    candidates: list[ContinuationCandidateEvidence] = []
    open_items: list[ContinuationOpenItem] = []
    receipt_by_request = {
        item.request_event_id: item for item in review.planning_hitl.receipts
    }
    for request in review.planning_hitl.requests:
        if request.request_event_id not in active_ancestors:
            continue
        receipt = receipt_by_request.get(request.request_event_id)
        if request.request_event_id not in active_ids and (
            receipt is None or receipt.response_event_id not in active_ids
        ):
            continue
        selected_id = (
            receipt.assertion.selected_candidate_id if receipt is not None else None
        )
        for candidate in request.candidates:
            disposition: Literal["selected", "not_selected", "superseded", "unresolved"]
            disposition = (
                "selected"
                if candidate.candidate_id == selected_id
                else "not_selected"
                if receipt is not None
                else "unresolved"
            )
            content = {
                "asserted_value": candidate.asserted_value,
                "display_label": candidate.display_label,
            }
            candidates.append(
                ContinuationCandidateEvidence(
                    subject_key=request.subject_key,
                    candidate_id=candidate.candidate_id,
                    candidate_kind="planning_hitl",
                    content_sha256=canonical_json_digest(content),
                    disposition=disposition,
                    locator=_locator(journey_id, by_id, request.request_event_id),
                    evidence_locators=_locators(
                        journey_id, by_id, candidate.evidence_event_ids
                    ),
                )
            )
            if disposition == "unresolved":
                open_items.append(
                    ContinuationOpenItem(
                        subject_key=request.subject_key,
                        state="unresolved",
                        reason=f"candidate_{disposition}",
                        locators=(
                            _locator(journey_id, by_id, request.request_event_id),
                        ),
                    )
                )
    return tuple(candidates), open_items


def _open_projection_items(
    *,
    review: JourneyReview,
    proposal: Any,
    proposal_event_id: str,
    journey_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
) -> list[ContinuationOpenItem]:
    items: list[ContinuationOpenItem] = []
    planning = review.execution_planning
    if planning is not None:
        for requirement in planning.requirements:
            if requirement.state == "ready":
                continue
            state = "conflicted" if requirement.state == "conflicted" else "unresolved"
            items.append(
                ContinuationOpenItem(
                    subject_key=requirement.subject_key,
                    state=state,
                    reason=requirement.state,
                    locators=_locators(
                        journey_id,
                        by_id,
                        requirement.evidence_event_ids or (proposal_event_id,),
                    ),
                )
            )
    facts = review.reference_facts
    if facts is not None:
        for work_item_id in facts.conflicted_work_item_ids:
            event_ids = tuple(
                item.source_event_id
                for item in facts.facts
                if item.work_item_id == work_item_id
            ) or (facts.plan_event_id,)
            items.append(
                ContinuationOpenItem(
                    subject_key=work_item_id,
                    state="conflicted",
                    reason="companion_evidence_conflicted",
                    locators=_locators(journey_id, by_id, event_ids),
                )
            )
        incomplete_work = (
            *facts.pending_work_item_ids,
            *facts.no_progress_work_item_ids,
        )
        for work_item_id in incomplete_work:
            event_ids = tuple(
                item.source_event_id
                for item in facts.facts
                if item.work_item_id == work_item_id
            ) or (facts.plan_event_id,)
            items.append(
                ContinuationOpenItem(
                    subject_key=work_item_id,
                    state="unresolved",
                    reason="companion_evidence_incomplete",
                    locators=_locators(journey_id, by_id, event_ids),
                )
            )
    for constraint in proposal.unresolved_execution_constraints:
        if constraint == "prepared_reference_not_verified":
            continue
        items.append(
            ContinuationOpenItem(
                subject_key="execution_constraint",
                state="deferred",
                reason=constraint,
                locators=(
                    _locator(
                        journey_id,
                        by_id,
                        proposal_event_id,
                        "ExecutionProposalSubmitted",
                    ),
                ),
            )
        )
    return items


def _path_scientific_basis(outcome, projection):
    from bio_neb.parabricks.result import parabricks_result

    retained = parabricks_result(projection)
    if retained.disposition != "succeeded":
        raise ScientificContinuationRefused("frontier_not_terminal")
    if retained != outcome:
        raise ScientificContinuationRefused("outcome_mismatch")
    planning = projection.planning_evidence()
    review = project_journey_events(planning, journey_file=outcome.journey_file)
    timeline = list(planning)
    # G1 causality is file order plus rests_on; these are locator edges,
    # not fabricated historical execution facts.
    for event in projection.read():
        parents = tuple(dict.fromkeys((*event.rests_on, timeline[-1]["event_id"])))
        timeline.append(
            {
                **event.model_dump(mode="json", by_alias=True),
                "journey_id": outcome.journey_id,
                "parent_ids": list(parents),
            }
        )
    by_id = {event["event_id"]: event for event in timeline}
    positions = {event["event_id"]: index for index, event in enumerate(timeline)}
    frontier = _locator(outcome.journey_id, by_id, outcome.h100_checkpoint_event_id)
    receipt_locator = frontier
    return review, timeline, by_id, positions, frontier, receipt_locator


def compile_scientific_continuation(
    outcome: ParabricksExecutionOutcome,
    projection: JourneyProjection | JourneyPath,
) -> ScientificContinuationPacket:
    """Compile one immutable scientific handoff without bytes, paths, or contact."""

    if not isinstance(projection, JourneyPath):
        raise ScientificContinuationRefused(
            "journey_projection_invalid", "legacy_execution_unsupported"
        )
    review, timeline, by_id, positions, frontier, receipt_locator = (
        _path_scientific_basis(outcome, projection)
    )
    active, approval_receipt, authority = _active_execution(
        outcome=outcome, review=review, by_id=by_id
    )
    proposal = active.proposal
    package = proposal.execution_package
    if package is None or package.journey_id != outcome.journey_id:
        raise ScientificContinuationRefused("proposal_mismatch", "execution_package")
    if package.input_set_id != review.registered_input_set.input_set_id:
        raise ScientificContinuationRefused("descriptor_drift", "input_set")
    verified_objects = outcome.verified_objects
    prepared_receipt_id = outcome.prepared_reference_receipt_id

    active_ancestors = _ancestors(by_id, frontier.event_id)
    required_authority_lineage = {
        active.request_event_id,
        approval_receipt.response_event_id,
    }
    if not required_authority_lineage <= active_ancestors:
        raise ScientificContinuationRefused("lineage_invalid", "active_proposal")
    active_ids = _nested_event_ids(proposal)
    active_ids.update(item.source_event_id for item in proposal.evidence_closure)
    active_ids.add(active.request_event_id)
    missing = sorted(event_id for event_id in active_ids if event_id not in by_id)
    if missing:
        raise ScientificContinuationRefused("evidence_inactive", *missing)
    if not active_ids <= active_ancestors:
        raise ScientificContinuationRefused(
            "evidence_inactive", *sorted(active_ids - active_ancestors)
        )

    resolution = review.requirement_resolution
    if (
        resolution is None
        or resolution.plan_event_id != proposal.plan_event_id
        or not resolution.ready_for_execution_compilation
        or resolution.execution_resolution is None
    ):
        raise ScientificContinuationRefused(
            "descriptor_drift", "requirement_resolution"
        )
    if resolution.execution_resolution != proposal.requirement_resolution:
        raise ScientificContinuationRefused(
            "descriptor_drift", "requirement_resolution"
        )
    try:
        sample_fact = proposal.requirement_resolution.one_value("sample_identity")
    except ValueError as error:
        raise ScientificContinuationRefused("descriptor_drift", "sample") from error
    if package.workflow.sample_id != sample_fact.value:
        raise ScientificContinuationRefused("descriptor_drift", "sample")
    sample_event_ids = sample_fact.lineage_event_ids

    reference_projection = review.reference_identity
    admission = proposal.reference_source_admission
    projected_reference_matches = bool(
        reference_projection is not None
        and reference_projection.verified_reference is not None
        and package.reference.assembly_accession
        == reference_projection.verified_reference.assembly_accession
    )
    if (
        admission is None
        or not projected_reference_matches
        or package.reference.assembly_accession != admission.assembly_accession
        or package.reference.source_admission_id != admission.admission_id
        or package.reference.reference_asset_id != admission.reference_asset_id
    ):
        raise ScientificContinuationRefused("descriptor_drift", "reference")
    source_locator = _locator(
        outcome.journey_id,
        by_id,
        package.reference.admission_event_id,
        "ReferenceSourceAdmitted",
    )
    source_content = {
        "admission_id": admission.admission_id,
        "reference_asset_id": admission.reference_asset_id,
        "assembly_accession": admission.assembly_accession,
        "source_mode": admission.source_mode,
        "source_representation": admission.source_representation,
        "expected_size_bytes": admission.expected_size_bytes,
        "expected_checksum_algorithm": admission.expected_checksum_algorithm,
        "expected_checksum": admission.expected_checksum,
        "derivation_rule": admission.derivation_rule,
        "authority_locator_sha256": canonical_json_digest(admission.authority_locator),
        "checksum_authority_locator_sha256": canonical_json_digest(
            admission.checksum_authority_locator
        ),
    }

    assertions, assertion_open = _assertions(
        review=review,
        proposal=proposal,
        journey_id=outcome.journey_id,
        by_id=by_id,
        active_ancestors=active_ancestors,
    )
    candidates, candidate_open = _candidate_evidence(
        review=review,
        journey_id=outcome.journey_id,
        by_id=by_id,
        active_ids=active_ids,
        active_ancestors=active_ancestors,
    )
    open_items = [
        *assertion_open,
        *candidate_open,
        *_open_projection_items(
            review=review,
            proposal=proposal,
            proposal_event_id=active.request_event_id,
            journey_id=outcome.journey_id,
            by_id=by_id,
        ),
    ]
    acquired = _acquired_input_checksums(outcome, package.inputs)
    input_values = tuple(
        ContinuationInput(
            role=item.role,
            file_id=item.file_id,
            byte_count=item.byte_count,
            sha256=acquired[item.role],
            registration_locator=_locator(
                outcome.journey_id, by_id, item.registration_event_id
            ),
            inspection_locator=_locator(
                outcome.journey_id, by_id, item.inspection_event_id
            ),
        )
        for item in package.inputs
    )
    h100_receipt_id = outcome.h100_receipt_id
    h100_configuration_id = outcome.h100_configuration_id
    h100_request_id = outcome.h100_request_id
    if (
        h100_receipt_id is None
        or h100_configuration_id is None
        or h100_request_id is None
    ):
        raise ScientificContinuationRefused("descriptor_drift", "h100_identity")
    material = {
        "schema_version": "bio-neb.scientific-continuation/1",
        "journey_id": outcome.journey_id,
        "frontier": frontier,
        "original_prompt": review.request.prompt,
        "prompt_locator": _locator(
            outcome.journey_id, by_id, review.request.source_event_id
        ),
        "user_assertions": assertions,
        "retrieved_sources": _retrieved_sources(
            review=review,
            journey_id=outcome.journey_id,
            by_id=by_id,
            active_ids=active_ids,
            active_ancestors=active_ancestors,
        ),
        "candidate_evidence": candidates,
        "admitted_facts": (),
        "scientific_investigation": resolution.scientific_investigation,
        "source_admission": ContinuationSourceAdmission(
            **source_content,
            content_sha256=canonical_json_digest(source_content),
            locator=source_locator,
        ),
        "open_items": tuple(open_items),
        "path_receipts": _path_receipts(
            review=review,
            journey_id=outcome.journey_id,
            by_id=by_id,
            positions=positions,
            active_ancestors=active_ancestors,
        ),
        "input_set_id": package.input_set_id,
        "inputs": input_values,
        "sample": ContinuationSample(
            sample_id=package.workflow.sample_id,
            requirement_id=sample_fact.requirement_id,
            evidence_locators=_locators(outcome.journey_id, by_id, sample_event_ids),
        ),
        "reference": ContinuationReference(
            assembly_accession=package.reference.assembly_accession,
            reference_asset_id=package.reference.reference_asset_id,
            representation=package.reference.representation,
            evidence_id=package.reference.evidence_id,
            source_admission_id=package.reference.source_admission_id,
            selection_locator=_locator(
                outcome.journey_id, by_id, package.reference.selection_event_id
            ),
            admission_locator=source_locator,
        ),
        "workflow": ContinuationWorkflow(
            workflow_id=package.workflow.workflow_id,
            workflow_version=package.workflow.workflow_version,
            recipe_candidate_id=package.workflow.recipe_candidate_id,
            assay=package.workflow.assay,
            read_layout=package.workflow.read_layout,
            meaning_parameters=tuple(
                (item.name, item.value) for item in package.workflow.meaning_parameters
            ),
            evidence_locator=_locator(
                outcome.journey_id,
                by_id,
                package.workflow.recipe_evidence_event_id,
            ),
        ),
        "resource": ContinuationResource(
            profile_id=package.resource.profile_id,
            candidate_id=package.resource.candidate_id,
            accelerator_count=package.resource.accelerator_count,
            cpu_count=package.resource.cpu_count,
            memory_gib=package.resource.memory_gib,
            accelerator_memory_gib=package.resource.accelerator_memory_gib,
            evidence_locator=_locator(
                outcome.journey_id, by_id, package.resource.evidence_event_id
            ),
        ),
        "execution": ContinuationExecutionIdentity(
            execution_package_id=package.package_id,
            proposal_id=proposal.proposal_id,
            authority_id=authority.authority_id,
            prepared_reference_receipt_id=prepared_receipt_id,
            h100_receipt_id=h100_receipt_id,
            h100_configuration_id=h100_configuration_id,
            h100_request_id=h100_request_id,
            proposal_locator=_locator(
                outcome.journey_id,
                by_id,
                active.request_event_id,
                "ExecutionProposalSubmitted",
            ),
            approval_locator=_locator(
                outcome.journey_id,
                by_id,
                approval_receipt.response_event_id,
                "ExecutionHitlResponded",
            ),
            receipt_locator=receipt_locator,
        ),
        "verified_objects": verified_objects,
        "vcf": verified_objects[1],
        "grants_live_access": False,
    }
    packet_material = _public_json(material)
    if not isinstance(packet_material, dict):
        raise AssertionError("continuation packet material must be a JSON object")
    return ScientificContinuationPacket(
        continuation_id=(
            "scientific-continuation:" + canonical_json_digest(packet_material)
        ),
        **material,
    )


def _packet_bytes(packet: ScientificContinuationPacket) -> bytes:
    return canonical_json(packet.model_dump(mode="json")).encode("utf-8")


def _strict_published_packet(
    target: Path,
) -> tuple[ScientificContinuationPacket, bytes]:
    try:
        payload = target.read_bytes()
        packet = ScientificContinuationPacket.model_validate_json(payload, strict=True)
    except (OSError, UnicodeError, ValidationError) as error:
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_reopen_mismatch", "output"
        ) from error
    if payload != _packet_bytes(packet):
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_reopen_mismatch", "noncanonical_bytes"
        )
    return packet, payload


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_linked_target(target: Path) -> None:
    try:
        os.unlink(target)
        _fsync_directory(target.parent)
    except OSError as error:
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_publication_failed", "output_cleanup"
        ) from error


def _publish_packet(
    packet: ScientificContinuationPacket,
    target: Path,
) -> tuple[ScientificContinuationPacket, bytes]:
    payload = _packet_bytes(packet)
    descriptor: int | None = None
    staging: str | None = None
    linked = False
    try:
        descriptor, staging = tempfile.mkstemp(
            prefix=".bio-neb-scientific-continuation-", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, target)
        linked = True
        os.unlink(staging)
        staging = None
        _fsync_directory(target.parent)
        reopened, reopened_bytes = _strict_published_packet(target)
        if reopened != packet or reopened_bytes != payload:
            raise ScientificContinuationPublicationRefused(
                "scientific_continuation_reopen_mismatch", "packet"
            )
        return reopened, reopened_bytes
    except FileExistsError as error:
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_output_exists", "output"
        ) from error
    except ScientificContinuationPublicationRefused:
        if linked:
            _remove_linked_target(target)
        raise
    except (OSError, UnicodeError) as error:
        if linked:
            _remove_linked_target(target)
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_publication_failed", "output"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staging is not None:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass


def publish_scientific_continuation(
    command: PublishScientificContinuationCommand,
) -> ScientificContinuationPublicationReceipt:
    """Compile, publish once, and strictly reopen one terminal continuation."""

    target = Path(command.output)
    if os.path.lexists(target):
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_output_exists", "output"
        )
    if not target.parent.is_dir():
        raise ScientificContinuationPublicationRefused(
            "scientific_continuation_output_invalid", "output"
        )

    try:
        outcome = read_completed_parabricks_result(command.execution_journey_file)
        projection = JourneyPath.open(Path(command.execution_journey_file))
    except (OSError, UnicodeError, JourneyValidationError, ValueError) as error:
        raise ScientificContinuationRefused(
            "journey_projection_invalid", type(error).__name__
        ) from error
    packet = compile_scientific_continuation(outcome, projection)
    reopened, payload = _publish_packet(packet, target)
    return ScientificContinuationPublicationReceipt(
        output=str(target),
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        continuation_id=reopened.continuation_id,
        journey_id=reopened.journey_id,
        frontier_event_id=reopened.frontier.event_id,
    )


__all__ = [
    "CausalJourneyLocator",
    "ContinuationCandidateEvidence",
    "ContinuationExecutionIdentity",
    "ContinuationInput",
    "ContinuationOpenItem",
    "ContinuationPathReceipt",
    "ContinuationReference",
    "ContinuationResource",
    "ContinuationRetrievedSource",
    "ContinuationSample",
    "ContinuationSourceAdmission",
    "ContinuationUserAssertion",
    "ContinuationWorkflow",
    "ScientificAdmittedFact",
    "ScientificContinuationPacket",
    "PublishScientificContinuationCommand",
    "ScientificContinuationPublicationReceipt",
    "ScientificContinuationPublicationRefusalCode",
    "ScientificContinuationPublicationRefused",
    "ScientificContinuationRefusalCode",
    "ScientificContinuationRefused",
    "compile_scientific_continuation",
    "publish_scientific_continuation",
]
