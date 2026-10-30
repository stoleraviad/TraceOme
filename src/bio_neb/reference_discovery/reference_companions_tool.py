"""Assembly- and role-bound reference companion Tool contracts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, Tuple, Union, runtime_checkable

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.contracts import FrozenContract
from bio_neb.prebuilt_index.contigs import RequestedContig, reference_partition
from bio_neb.providers.ncbi_assembly_report import SequenceRecord
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    ReferenceAdapterError,
    ReferenceAdapterFailure,
    SourceObservation,
)

_ASSEMBLY_ACCESSION = re.compile(ASSEMBLY_ACCESSION_PATTERN)
_PLAN_ID = re.compile(r"^plan_[0-9a-f]{64}$")
_REQUIREMENT_ID = re.compile(r"^requirement_[0-9a-f]{32}$")
_SUBJECT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_CHECKSUMS = {"md5": 32, "sha256": 64, "sha512": 128}

CompanionRole = Literal["genomic_fasta", "annotation"]


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


def _evidence_id(namespace: str, value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(encoded).hexdigest()}"


class CompanionRequirementGrounding(FrozenContract):
    """The unchanged ticket-18 requirement and its Journey ancestry."""

    journey_id: StrictStr
    plan_id: StrictStr
    requirement_id: StrictStr
    kind: Literal["companion_asset"] = "companion_asset"
    subject_key: StrictStr
    source_event_ids: Tuple[StrictStr, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def canonical_grounding(self) -> "CompanionRequirementGrounding":
        _canonical_uuid(self.journey_id, "journey_id")
        if not _PLAN_ID.fullmatch(self.plan_id):
            raise ValueError("plan_id must be a canonical grounded plan identity")
        if not _REQUIREMENT_ID.fullmatch(self.requirement_id):
            raise ValueError("requirement_id must be canonical")
        if not _SUBJECT_KEY.fullmatch(self.subject_key):
            raise ValueError("subject_key must be lowercase snake case")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("source_event_ids must be unique")
        for event_id in self.source_event_ids:
            _canonical_uuid(event_id, "source_event_id")
        return self


class VerifiedReferenceBinding(FrozenContract):
    reference_event_id: StrictStr
    assembly_accession: StrictStr
    reference_evidence_id: StrictStr

    @field_validator("reference_event_id")
    @classmethod
    def canonical_event(cls, value: str) -> str:
        return _canonical_uuid(value, "reference_event_id")

    @field_validator("assembly_accession")
    @classmethod
    def exact_accession(cls, value: str) -> str:
        if not _ASSEMBLY_ACCESSION.fullmatch(value):
            raise ValueError("reference binding requires accession.version")
        return value


class CompanionWorkItem(FrozenContract):
    """One stable child of the existing collection-shaped companion requirement."""

    work_item_id: StrictStr = Field(pattern=r"^companion-work:[0-9a-f]{64}$")
    grounding: CompanionRequirementGrounding
    reference: VerifiedReferenceBinding
    role: CompanionRole

    @model_validator(mode="after")
    def stable_identity(self) -> "CompanionWorkItem":
        if self.reference.reference_event_id not in self.grounding.source_event_ids:
            raise ValueError("companion work omitted verified-reference ancestry")
        expected = companion_work_item_id(
            grounding=self.grounding, reference=self.reference, role=self.role
        )
        if self.work_item_id != expected:
            raise ValueError("companion work identity changed its exact subject")
        return self


def companion_work_item_id(
    *,
    grounding: CompanionRequirementGrounding,
    reference: VerifiedReferenceBinding,
    role: CompanionRole,
) -> str:
    """Derive identity only from retained plan/reference identity plus the lawful role."""

    return _evidence_id(
        "companion-work",
        {
            "plan_id": grounding.plan_id,
            "requirement_id": grounding.requirement_id,
            "subject_key": grounding.subject_key,
            "reference_event_id": reference.reference_event_id,
            "assembly_accession": reference.assembly_accession,
            "role": role,
        },
    )


def derive_companion_work_items(
    *,
    grounding: CompanionRequirementGrounding,
    reference: VerifiedReferenceBinding,
    selected_roles: Tuple[CompanionRole, ...],
    role_selection_event_ids: Tuple[StrictStr, ...],
) -> Tuple[CompanionWorkItem, ...]:
    """Derive the minimal lawful role collection selected through Planning HITL.

    The current plan says only to determine required companions.  NCBI metadata supports the
    assembly-qualified ``genomic_fasta`` and ``annotation`` vocabulary, but neither role is product
    authority.  A generic Planning HITL disposition supplies the minimal roles for this proposal;
    each selected role must still pass the strict Tool before readiness.
    """

    if not selected_roles or len(selected_roles) != len(set(selected_roles)):
        raise ValueError("companion roles must be a non-empty unique collection")
    if not role_selection_event_ids:
        raise ValueError("companion roles require Planning HITL disposition evidence")
    for event_id in role_selection_event_ids:
        _canonical_uuid(event_id, "role_selection_event_id")
    if not set(role_selection_event_ids) <= set(grounding.source_event_ids):
        raise ValueError("companion role disposition is outside requirement ancestry")
    return tuple(
        CompanionWorkItem(
            work_item_id=companion_work_item_id(
                grounding=grounding, reference=reference, role=role
            ),
            grounding=grounding,
            reference=reference,
            role=role,
        )
        for role in selected_roles
    )


class CompanionSourceRecord(FrozenContract):
    """Source-owned asset metadata; ``asset_id`` is retained verbatim."""

    asset_id: StrictStr = Field(min_length=1, max_length=1000)
    assembly_accession: StrictStr
    role: CompanionRole
    publisher: StrictStr = Field(min_length=1, max_length=300)
    release_version: StrictStr | None = Field(
        default=None, min_length=1, max_length=300
    )
    representation: StrictStr = Field(min_length=1, max_length=300)
    byte_size: StrictInt | None = Field(default=None, ge=0)
    checksum_algorithm: Literal["md5", "sha256", "sha512"] | None = None
    checksum: StrictStr | None = None
    authority_locator: StrictStr | None = Field(
        default=None, pattern=r"^https://", max_length=1000
    )
    # The publisher's exact file and the listing that publishes its checksum, when the
    # source names one; omitted from the dump when absent.
    source_locator: StrictStr | None = Field(
        default=None,
        pattern=r"^https://",
        max_length=1000,
        exclude_if=lambda value: value is None,
    )
    checksum_authority_locator: StrictStr | None = Field(
        default=None,
        pattern=r"^https://",
        max_length=1000,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def coherent_authority(self) -> "CompanionSourceRecord":
        _canonical_text(self.asset_id, "asset_id")
        if not _ASSEMBLY_ACCESSION.fullmatch(self.assembly_accession):
            raise ValueError("companion source requires accession.version")
        if bool(self.checksum_algorithm) != bool(self.checksum):
            raise ValueError("companion checksum algorithm and value are inseparable")
        if self.checksum_algorithm is not None:
            checksum = self.checksum or ""
            if len(checksum) != _CHECKSUMS[self.checksum_algorithm] or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError("companion checksum is not canonical")
        if self.checksum_authority_locator is not None and self.checksum is None:
            raise ValueError("companion checksum authority requires a checksum")
        return self

    @property
    def authority_complete(self) -> bool:
        return bool(
            (self.checksum_algorithm and self.checksum and self.release_version)
            or (self.authority_locator and self.release_version)
        )


class CompanionAdapterObservation(FrozenContract):
    records: Tuple[CompanionSourceRecord, ...] = Field(max_length=100)
    source: SourceObservation
    source_truncated: StrictBool = False

    @model_validator(mode="after")
    def ncbi_source(self) -> "CompanionAdapterObservation":
        if self.source.authority != "ncbi_datasets":
            raise ValueError("companion observation requires NCBI Datasets authority")
        return self


@runtime_checkable
class ReferenceCompanionAdapter(Protocol):
    """NCBI's companion source: a role's companions at an assembly, and the
    assembly's sequences that a verified genomic FASTA's description reads (#464)."""

    contact_count: int

    def describe_companions(
        self, assembly_accession: str, role: CompanionRole
    ) -> CompanionAdapterObservation: ...

    def fetch_assembly_sequences(
        self, genomic_fasta_locator: str
    ) -> Sequence[SequenceRecord]:
        """Every row of the assembly report beside ``genomic_fasta_locator``, each
        one's names, role and length copied as NCBI writes it (#467, #464)."""
        ...

    def fetch_accession_sequences(
        self, assembly_accession: str
    ) -> Sequence[SequenceRecord]:
        """Every row of ``assembly_accession``'s assembly report, found where NCBI
        Datasets' ``links`` answer names its FTP directory: an index read before any
        companion is described (owner, #459 D1)."""
        ...


class DescribeCompanionInput(FrozenContract):
    work_item: CompanionWorkItem
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    selected_asset_id: StrictStr | None = Field(
        default=None, min_length=1, max_length=1000
    )
    selection_event_id: StrictStr | None = None

    @model_validator(mode="after")
    def coherent_selection(self) -> "DescribeCompanionInput":
        if bool(self.selected_asset_id) != bool(self.selection_event_id):
            raise ValueError("companion selection identity and event are inseparable")
        if self.selection_event_id is not None:
            _canonical_uuid(self.selection_event_id, "selection_event_id")
        return self


def companion_asset_disposition_id(work_item_id: str, asset_id: str) -> str:
    """Identify a review disposition without replacing the opaque source asset ID."""

    return _evidence_id(
        "companion-asset-disposition",
        {"work_item_id": work_item_id, "asset_id": asset_id},
    )


class CompanionCandidate(FrozenContract):
    asset_id: StrictStr
    assembly_accession: StrictStr
    role: CompanionRole
    publisher: StrictStr
    release_version: StrictStr | None
    representation: StrictStr
    byte_size: StrictInt | None
    checksum_algorithm: Literal["md5", "sha256", "sha512"] | None
    checksum: StrictStr | None
    authority_locator: StrictStr | None
    source_locator: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    checksum_authority_locator: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    compatibility: Literal["compatible", "incompatible", "unverified"]
    source: SourceObservation
    evidence_id: StrictStr
    # The Run's reference sequences (`contigs.REFERENCE_ROLES`) by their names in
    # NCBI's FASTA, read once from NCBI's assembly report beside the verified genomic
    # FASTA; omitted for another role or outcome.
    reference_sequences: Tuple[RequestedContig, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # The NCBI-FASTA names of the report's other sequences (alt loci, patches), which
    # the CPU build and input storage's measure drop; usually none.
    excluded_sequences: Tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


CompanionOutcome = Literal[
    "verified", "absent", "partial", "multiple", "ambiguous", "incompatible"
]


class CompanionObserved(FrozenContract):
    status: Literal["observed"] = "observed"
    outcome: CompanionOutcome
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    work_item: CompanionWorkItem
    candidates: Tuple[CompanionCandidate, ...] = Field(max_length=100)
    source: SourceObservation
    evidence_id: StrictStr
    summary: StrictStr

    @model_validator(mode="after")
    def coherent_outcome(self) -> "CompanionObserved":
        compatible = [
            item for item in self.candidates if item.compatibility == "compatible"
        ]
        if self.outcome == "verified" and (
            len(self.candidates) != 1
            or len(compatible) != 1
            or self.assembly_accession != self.work_item.reference.assembly_accession
            or compatible[0].assembly_accession
            != self.work_item.reference.assembly_accession
            or compatible[0].role != self.work_item.role
        ):
            raise ValueError("verified companion requires one compatible candidate")
        if self.outcome == "absent" and self.candidates:
            raise ValueError("absent companion cannot contain candidates")
        if self.outcome == "multiple" and len(compatible) < 2:
            raise ValueError(
                "multiple companion outcome requires multiple compatible candidates"
            )
        if self.outcome == "incompatible" and compatible:
            raise ValueError(
                "incompatible outcome cannot contain a compatible candidate"
            )
        return self


class CompanionToolFailed(FrozenContract):
    status: Literal["failed"] = "failed"
    work_item: CompanionWorkItem
    failure: ReferenceAdapterFailure
    evidence_id: StrictStr
    summary: StrictStr


DescribeCompanionResult = Annotated[
    Union[CompanionObserved, CompanionToolFailed], Field(discriminator="status")
]


class CompanionToolCallRejected(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class DescribeCompanionTool:
    name = "describe_reference_companion"

    def describe(
        self,
        tool_input: DescribeCompanionInput,
        *,
        expected_work_item: CompanionWorkItem,
        adapter: ReferenceCompanionAdapter,
    ) -> DescribeCompanionResult:
        if tool_input.work_item != expected_work_item:
            raise CompanionToolCallRejected("companion_work_item_ungrounded")
        from bio_neb.agentic_journey.source_contacts import call_source

        try:
            raw = call_source(
                "describe_companion",
                lambda: adapter.describe_companions(
                    tool_input.assembly_accession,
                    tool_input.work_item.role,
                ),
            )
        except ReferenceAdapterError as error:
            return self._failed(tool_input.work_item, error.failure)
        try:
            observation = CompanionAdapterObservation.model_validate(raw)
        except (ValueError, TypeError):
            return self._failed(
                tool_input.work_item,
                ReferenceAdapterFailure(
                    code="adapter_output_invalid",
                    public_message="Companion adapter returned invalid typed output.",
                    retryable=False,
                ),
            )

        records = tuple(
            record
            for record in observation.records
            if tool_input.selected_asset_id is None
            or record.asset_id == tool_input.selected_asset_id
        )
        candidates = tuple(
            self._candidate(
                record,
                source=observation.source,
                work_item=tool_input.work_item,
                assembly_accession=tool_input.assembly_accession,
            )
            for record in records
        )
        exact = tuple(item for item in candidates if item.compatibility == "compatible")
        mismatched = tuple(
            item for item in candidates if item.compatibility == "incompatible"
        )
        unresolved = tuple(
            item for item in candidates if item.compatibility == "unverified"
        )
        if not candidates:
            outcome: CompanionOutcome = "absent"
        elif exact and (mismatched or unresolved):
            outcome = "ambiguous"
        elif len(exact) > 1:
            outcome = "multiple"
        elif len(exact) == 1:
            outcome = "verified"
        elif mismatched and not unresolved:
            outcome = "incompatible"
        else:
            outcome = "partial"
        payload = {
            "work_item": tool_input.work_item.model_dump(mode="json"),
            "assembly_accession": tool_input.assembly_accession,
            "selected_asset_id": tool_input.selected_asset_id,
            "selection_event_id": tool_input.selection_event_id,
            "outcome": outcome,
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "source": observation.source.model_dump(mode="json"),
            "source_truncated": observation.source_truncated,
        }
        if observation.source_truncated and outcome in {"verified", "absent"}:
            outcome = "partial"
            payload["outcome"] = outcome
        if (
            outcome == "verified"
            and tool_input.work_item.role == "genomic_fasta"
            and candidates[0].source_locator is not None
        ):
            # The Run's reference sequences, from NCBI's assembly report beside the
            # one verified FASTA, read once in this Tool's scope.
            locator = candidates[0].source_locator
            try:
                sequences = call_source(
                    "describe_companion",
                    lambda: adapter.fetch_assembly_sequences(locator),
                )
            except ReferenceAdapterError as error:
                return self._failed(tool_input.work_item, error.failure)
            requested, excluded = reference_partition(
                tool_input.assembly_accession, sequences
            )
            candidates = (
                self._candidate(
                    records[0],
                    source=observation.source,
                    work_item=tool_input.work_item,
                    assembly_accession=tool_input.assembly_accession,
                    reference_sequences=tuple(
                        RequestedContig(name=name, length=sequence.length)
                        for name, sequence in requested
                    ),
                    excluded_sequences=excluded,
                ),
            )
            payload["candidates"] = [
                item.model_dump(mode="json") for item in candidates
            ]
        return CompanionObserved(
            outcome=outcome,
            assembly_accession=tool_input.assembly_accession,
            work_item=tool_input.work_item,
            candidates=candidates,
            source=observation.source,
            evidence_id=_evidence_id("companion-result", payload),
            summary=(
                f"NCBI Datasets observed {len(candidates)} {tool_input.work_item.role} "
                f"asset(s) queried for {tool_input.assembly_accession} against "
                f"selected {tool_input.work_item.reference.assembly_accession}; "
                f"outcome={outcome}."
            ),
        )

    @staticmethod
    def _candidate(
        record: CompanionSourceRecord,
        *,
        source: SourceObservation,
        work_item: CompanionWorkItem,
        assembly_accession: str,
        reference_sequences: Tuple[RequestedContig, ...] = (),
        excluded_sequences: Tuple[str, ...] = (),
    ) -> CompanionCandidate:
        if (
            assembly_accession != work_item.reference.assembly_accession
            or record.assembly_accession != work_item.reference.assembly_accession
            or record.role != work_item.role
        ):
            compatibility: Literal["compatible", "incompatible", "unverified"] = (
                "incompatible"
            )
        elif not record.authority_complete:
            compatibility = "unverified"
        else:
            compatibility = "compatible"
        payload = {
            "asset": record.model_dump(mode="json"),
            "source": source.model_dump(mode="json"),
            "work_item_id": work_item.work_item_id,
            "compatibility": compatibility,
        }
        if reference_sequences:
            payload["reference_sequences"] = [
                item.model_dump(mode="json") for item in reference_sequences
            ]
        if excluded_sequences:
            payload["excluded_sequences"] = list(excluded_sequences)
        return CompanionCandidate(
            **record.model_dump(mode="python"),
            compatibility=compatibility,
            source=source,
            evidence_id=_evidence_id("companion-candidate", payload),
            reference_sequences=reference_sequences,
            excluded_sequences=excluded_sequences,
        )

    @staticmethod
    def _failed(
        work_item: CompanionWorkItem, failure: ReferenceAdapterFailure
    ) -> CompanionToolFailed:
        return CompanionToolFailed(
            work_item=work_item,
            failure=failure,
            evidence_id=_evidence_id(
                "companion-failure",
                {
                    "work_item_id": work_item.work_item_id,
                    "failure": failure.model_dump(mode="json"),
                },
            ),
            summary=failure.public_message,
        )


__all__ = [
    "CompanionAdapterObservation",
    "CompanionCandidate",
    "CompanionObserved",
    "CompanionRequirementGrounding",
    "CompanionRole",
    "CompanionSourceRecord",
    "CompanionToolCallRejected",
    "CompanionToolFailed",
    "CompanionWorkItem",
    "DescribeCompanionInput",
    "DescribeCompanionResult",
    "DescribeCompanionTool",
    "ReferenceCompanionAdapter",
    "VerifiedReferenceBinding",
    "companion_work_item_id",
    "companion_asset_disposition_id",
    "derive_companion_work_items",
]
