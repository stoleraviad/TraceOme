"""Product-owned Investigation contracts and deterministic proposal compilation."""

from __future__ import annotations

import json
import re
import urllib.parse
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Discriminator,
    Field,
    StrictInt,
    StrictStr,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)

from bio_neb.agentic_journey.attempt_journey import (
    MODEL_REQUEST_TERMINALS,
    AttemptCancelled,
    AttemptFailed,
    AttemptStopped,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.continuation_policy import (
    AgentInvocationPolicy,
    ContinuationPolicy,
    InvestigationReviewPolicy,
    explicit_review_policy,
    validate_continuation,
)
from bio_neb.agentic_journey.instruction_evidence import retained_instruction_reference
from bio_neb.agentic_journey.model_availability import (
    prior_model_attempt_ids,
    project_model_availability_review,
)
from bio_neb.agentic_journey.runtime_config import (
    RetainedRuntimeSelections,
    retained_endpoint_policy,
    retained_invocation_policy,
    retained_runtime_selections,
    validate_retained_model_request,
)
from bio_neb.agentic_journey.source_answers import SourceAnswer
from bio_neb.agentic_journey.source_contacts import source_semantic_parent
from bio_neb.agentic_journey.structured_output import (
    StructuredOutputLineage,
    project_structured_output_lineages,
)
from bio_neb.agentic_journey.transcript import check_stall_notes
from bio_neb.content import CONTENT_ID_PATTERN, ContentId, shown_to_model
from bio_neb.contracts import StrictFrozenContract
from bio_neb.failure import (
    ITEM_ISSUE_ACTIONS,
    ExpectedValue,
    ItemIssue,
    OmitFields,
    RefusalDetail,
    item_issue_key,
    refusal_detail_text,
)
from bio_neb.identifier_rule import site_in_quote, word_aligned_spans
from bio_neb.investigation.collection_view import CollectionView, collection_view
from bio_neb.investigation.investigation_request import (
    RESEARCHER_INSTRUCTION_ACTIONS,
    RSID_CANONICAL_PATTERN,
    RSID_RULE,
    InvestigationRequest,
    InvestigationRequestItem,
    NotCheckedReason,
    PlacedPosition,
    QuestionAction,
    RequestedVariantIdentity,
    StatedAssembly,
    StatedChange,
    StatedPosition,
    StatedRegion,
    StatedTarget,
    TargetLocation,
    effective_location,
    effective_requested_rsid,
    not_checked_reason,
    researcher_words,
    typed_answers,
    untyped,
)
from bio_neb.investigation.lookup_origins import (
    LookupOrigin,
    ModelMatchedName,
    PlacementUnnamed,
    PositionWithoutAlleles,
    SequenceUnnamed,
    SourceAnswered,
)
from bio_neb.investigation.member_context import (
    InvestigationMemberContextRecorded,
    member_context_fact,
    member_request,
    rendered_text,
)
from bio_neb.investigation.vcf_observation import (
    allele_text,
    command_safe_contig,
    trimmed_allele,
)
from bio_neb.journey import ArtifactReference, canonical_json, canonical_json_digest
from bio_neb.parabricks.requirement_resolution import (
    FactSlotStatus,
    RequirementValue,
    fact_slot_status,
)
from bio_neb.providers.variant_placements import VariantPlacement
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    AssemblyQuery,
    AssemblySourceRecord,
)
from bio_neb.target_request.records import GeneChangeAnswer

if TYPE_CHECKING:
    from bio_neb.agentic_journey.tool_execution import CheckedOnce

_RSID = re.compile(RSID_CANONICAL_PATTERN)
_CanonicalRsid = Annotated[StrictStr, Field(pattern=RSID_CANONICAL_PATTERN)]
# A target allele as the source states it (SPDI): no single-base cap; the two
# alleles differ, so they are never both empty.
_ALLELE = r"^[ACGTN]*$"


def _digest(prefix: str, value: object) -> str:
    digest = canonical_json_digest(_json_material(value))
    return f"{prefix}:{digest}"


def _json_material(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: _json_material(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_material(item) for item in value)
    return value


def _canonical_text(value: str, name: str) -> str:
    if not value or value.strip() != value or any(char in value for char in "\r\n\0"):
        raise ValueError(f"{name} must be one non-empty canonical line")
    return value


def investigation_output_member_id(
    request: InvestigationRequestItem, ordinal: int = 0
) -> str:
    """Return the stable structured-output member identity for one exact request."""

    return _digest("request-member", {"request": request, "ordinal": ordinal})


def request_member_id(
    assertion_id: str, request: InvestigationRequestItem, ordinal: int
) -> str:
    """The member identity one request of an assertion has, planned or not."""

    return _digest(
        "request-member",
        {
            "assertion_id": assertion_id,
            "output_member_id": investigation_output_member_id(request, ordinal),
        },
    )


class InvestigationVcfIdentity(StrictFrozenContract):
    vcf_id: StrictStr = Field(pattern=r"^investigation-vcf:[0-9a-f]{64}$")
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("bucket", "object_key")
    @classmethod
    def canonical_locator(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @model_validator(mode="after")
    def exact_identity(self) -> "InvestigationVcfIdentity":
        material = self.model_dump(mode="json", exclude={"vcf_id"})
        if self.vcf_id != _digest("investigation-vcf", material):
            raise ValueError("VCF identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        bucket: str,
        object_key: str,
        byte_count: int,
        sha256: str,
    ) -> "InvestigationVcfIdentity":
        material = {
            "bucket": bucket,
            "object_key": object_key,
            "byte_count": byte_count,
            "sha256": sha256,
        }
        return cls(vcf_id=_digest("investigation-vcf", material), **material)


class InvestigationInput(StrictFrozenContract):
    input_id: StrictStr = Field(pattern=r"^investigation-input:[0-9a-f]{64}$")
    vcf: InvestigationVcfIdentity
    sample_id: StrictStr
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)

    @field_validator("sample_id")
    @classmethod
    def canonical_sample(cls, value: str) -> str:
        return _canonical_text(value, "sample_id")

    @model_validator(mode="after")
    def exact_identity(self) -> "InvestigationInput":
        material = self.model_dump(mode="json", exclude={"input_id"})
        if self.input_id != _digest("investigation-input", material):
            raise ValueError("Investigation input identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        vcf: InvestigationVcfIdentity,
        sample_id: str,
        assembly_accession: str,
    ) -> "InvestigationInput":
        material = {
            "vcf": vcf.model_dump(mode="json"),
            "sample_id": sample_id,
            "assembly_accession": assembly_accession,
        }
        return cls(input_id=_digest("investigation-input", material), **material)


class UserAssertion(StrictFrozenContract):
    assertion_id: StrictStr = Field(pattern=r"^user-assertion:[0-9a-f]{64}$")
    asserted_by: StrictStr
    request_text: StrictStr = Field(max_length=4_000)
    rsid: StrictStr | None = Field(default=None, pattern=RSID_CANONICAL_PATTERN)
    requests: tuple[InvestigationRequestItem, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # The research goal the request states, in the researcher's words: every
    # member reads it as the original intent and keeps its own request.
    research_goals: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    predecessor_ids: tuple[StrictStr, ...] = Field(min_length=1)
    scientifically_verified: Literal[False] = False

    @field_validator("asserted_by", "request_text")
    @classmethod
    def canonical_text(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @model_validator(mode="after")
    def exact_identity(self) -> "UserAssertion":
        if self.requests:
            if self.rsid != self.requests[0].rsid:
                raise ValueError("collection assertion changed its first request")
        elif self.rsid is not None and not RSID_RULE.matches(
            self.rsid, self.request_text
        ):
            raise ValueError("explicit request must contain its exact rsID")
        if len(self.predecessor_ids) != len(set(self.predecessor_ids)):
            raise ValueError("assertion predecessors must be unique")
        material = self.model_dump(mode="json", exclude={"assertion_id"})
        if self.assertion_id != _digest("user-assertion", material):
            raise ValueError("User assertion identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        asserted_by: str,
        request_text: str,
        rsid: str | None,
        predecessor_ids: tuple[str, ...],
        research_goals: tuple[str, ...] = (),
    ) -> "UserAssertion":
        material = {
            "asserted_by": asserted_by,
            "request_text": request_text,
            "rsid": rsid,
            **({"research_goals": research_goals} if research_goals else {}),
            "predecessor_ids": predecessor_ids,
            "scientifically_verified": False,
        }
        return cls(assertion_id=_digest("user-assertion", material), **material)

    @classmethod
    def create_collection(
        cls,
        *,
        asserted_by: str,
        requests: tuple[InvestigationRequestItem, ...],
        predecessor_ids: tuple[str, ...],
        research_goals: tuple[str, ...] = (),
    ) -> "UserAssertion":
        if not requests:
            raise ValueError("Investigation requires at least one request")
        material = {
            "asserted_by": asserted_by,
            "request_text": requests[0].request_text,
            "rsid": requests[0].rsid,
            "requests": tuple(item.model_dump(mode="json") for item in requests),
            **({"research_goals": research_goals} if research_goals else {}),
            "predecessor_ids": predecessor_ids,
            "scientifically_verified": False,
        }
        return cls(assertion_id=_digest("user-assertion", material), **material)


SourceOperation = Literal[
    "fetch_refsnp",
    "fetch_assembly_report",
    "eva_clustered_variant",
    "eva_submitted_variants",
    "eva_clustered_variants_at",
    "eva_contig_alias",
    "ncbi_sequence_names",
    "ncbi_reference_span",
    "ncbi_spdi_rsids",
    # A stated version's placement inside the position lookup.
    "ncbi_taxon_assemblies",
    "ncbi_spdi_equivalents",
    # A gene change's placement: the Run assembly's annotation and NCBI's HGVS
    # reading.
    "ncbi_genome_annotation",
    "ncbi_hgvs_contextuals",
]
SourceAuthority = Literal["ncbi", "eva"]
SOURCE_AUTHORITY: dict[SourceOperation, SourceAuthority] = {
    "fetch_refsnp": "ncbi",
    "fetch_assembly_report": "ncbi",
    "ncbi_sequence_names": "ncbi",
    "ncbi_reference_span": "ncbi",
    "ncbi_spdi_rsids": "ncbi",
    "ncbi_taxon_assemblies": "ncbi",
    "ncbi_spdi_equivalents": "ncbi",
    "ncbi_genome_annotation": "ncbi",
    "ncbi_hgvs_contextuals": "ncbi",
    "eva_clustered_variant": "eva",
    "eva_submitted_variants": "eva",
    "eva_clustered_variants_at": "eva",
    "eva_contig_alias": "eva",
}
"""The source each operation reads: NCBI, or the European Variation Archive (#352)."""


class RetrievedSource(StrictFrozenContract):
    source_id: StrictStr = Field(pattern=r"^retrieved-source:[0-9a-f]{64}$")
    operation: SourceOperation
    authority: SourceAuthority = "ncbi"
    request_identity: StrictStr
    status_code: StrictInt = Field(ge=100, le=599)
    observed_at: StrictStr
    elapsed_time_ms: StrictInt = Field(ge=0)
    raw_artifact_id: StrictStr = Field(pattern=CONTENT_ID_PATTERN)
    raw_response_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response_byte_count: StrictInt = Field(ge=0)

    @field_validator("request_identity", "observed_at")
    @classmethod
    def canonical_text(cls, value: str, info: Any) -> str:
        canonical = _canonical_text(value, info.field_name)
        if info.field_name == "observed_at":
            parsed = datetime.fromisoformat(canonical.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("observed_at must be timezone-aware")
        return canonical

    @model_validator(mode="after")
    def exact_identity(self) -> "RetrievedSource":
        if self.authority != SOURCE_AUTHORITY[self.operation]:
            raise ValueError(
                f"{self.operation} is an {SOURCE_AUTHORITY[self.operation]} operation"
            )
        if ContentId.parse(self.raw_artifact_id).digest != self.raw_response_sha256:
            raise ValueError("Retrieved-source artifact identity changed")
        material = self.model_dump(mode="json", exclude={"source_id"})
        if self.source_id != _digest("retrieved-source", material):
            raise ValueError("Retrieved-source identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        operation: SourceOperation,
        request_identity: str,
        status_code: int,
        observed_at: str,
        elapsed_time_ms: int,
        raw_response: bytes,
    ) -> "RetrievedSource":
        raw_id = ContentId.of(raw_response)
        material = {
            "operation": operation,
            "authority": SOURCE_AUTHORITY[operation],
            "request_identity": request_identity,
            "status_code": status_code,
            "observed_at": observed_at,
            "elapsed_time_ms": elapsed_time_ms,
            "raw_artifact_id": str(raw_id),
            "raw_response_sha256": raw_id.digest,
            "raw_response_byte_count": len(raw_response),
        }
        return cls(source_id=_digest("retrieved-source", material), **material)


AssessmentVerdict = Literal[
    "accepted",
    "source_answered",
    "merged",
    "ambiguous",
    "conflicted",
    "incompatible",
    "not_found",
    "malformed",
]


class EvidenceAssessment(StrictFrozenContract):
    assessment_id: StrictStr = Field(pattern=r"^evidence-assessment:[0-9a-f]{64}$")
    source_id: StrictStr = Field(pattern=r"^retrieved-source:[0-9a-f]{64}$")
    verdict: AssessmentVerdict
    rationale: StrictStr

    @field_validator("rationale")
    @classmethod
    def canonical_rationale(cls, value: str) -> str:
        return _canonical_text(value, "rationale")

    @model_validator(mode="after")
    def exact_identity(self) -> "EvidenceAssessment":
        material = self.model_dump(mode="json", exclude={"assessment_id"})
        if self.assessment_id != _digest("evidence-assessment", material):
            raise ValueError("Evidence-assessment identity changed")
        return self

    @classmethod
    def create(
        cls, *, source_id: str, verdict: AssessmentVerdict, rationale: str
    ) -> "EvidenceAssessment":
        material = {
            "source_id": source_id,
            "verdict": verdict,
            "rationale": rationale,
        }
        return cls(assessment_id=_digest("evidence-assessment", material), **material)


class ResolvedVariantCandidate(StrictFrozenContract):
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: StrictStr
    contig: StrictStr
    one_based_position: StrictInt = Field(gt=0)
    ref: StrictStr = Field(pattern=_ALLELE)
    alt: StrictStr = Field(pattern=_ALLELE)
    aliases: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @field_validator("assembly_name", "contig")
    @classmethod
    def canonical_text(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @model_validator(mode="after")
    def exact_variant(self) -> "ResolvedVariantCandidate":
        if not command_safe_contig(self.contig):
            raise ValueError("variant contig is unsafe")
        if self.ref == self.alt:
            raise ValueError("variant alleles must differ")
        if len(self.aliases) != len(set(self.aliases)) or any(
            not _RSID.fullmatch(alias) for alias in self.aliases
        ):
            raise ValueError("variant aliases are not exact canonical rsIDs")
        return self

    @property
    def target_identity(self) -> RequestedVariantIdentity:
        """The allele a Work item's target holds: trimmed and left-aligned (#368)."""
        position, ref, alt = trimmed_allele(self.one_based_position, self.ref, self.alt)
        return RequestedVariantIdentity(self.contig, position, ref, alt)

    @property
    def locus(self) -> tuple[str, str, str, int, str]:
        """Where the candidate sits: every alternate allele of one locus shares it."""
        return (
            self.assembly_accession,
            self.assembly_name,
            self.contig,
            self.one_based_position,
            self.ref,
        )


VariantSource = Literal["ncbi_dbsnp", "eva"]
"""A variant source a lookup Tool reads, the model's choice on every Run (#469)."""
LookupSource = Literal[VariantSource, "ncbi"]
"""The source a lookup read: the variant source the model chose, NCBI dbSNP or the
European Variation Archive (#469: both are offered on every Run), or, for a region,
which lists no rsID, NCBI's own sequence and reference services only."""
EVA_SUPPORTING_OPERATIONS = frozenset(
    {"fetch_assembly_report", "eva_contig_alias", "eva_submitted_variants"}
)
"""An EVA lookup's GETs after its clustered variant: the Run's assembly report (its
paired accession, owner #468 R2 a), contig aliases and submitted variants."""


def eva_record_key(rsid: str, contig_naming: str | None) -> str:
    """The key an EVA record GET is asked with: the rsID, with the naming convention the
    lookup asked as EVA's own query option (#468, owner R3)."""
    if contig_naming is None:
        return rsid
    return f"{rsid}?{urllib.parse.urlencode({'contigNamingConvention': contig_naming})}"


class EvaLookupDetail(StrictFrozenContract):
    """What an EVA lookup adds to the shared result: the Run's INSDC accession as NCBI
    pairs it, its naming convention and its supporting GETs. Every field is always
    written, `None` or empty when it does not apply, so an omitted field is never a
    skipped one."""

    run_insdc_assembly_accession: StrictStr | None
    # EVA's `contigNamingConvention` the model asked for, `None` for EVA's own
    # default (INSDC).
    contig_naming: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    supporting_sources: tuple[RetrievedSource, ...]

    @model_validator(mode="after")
    def eva_sources(self) -> "EvaLookupDetail":
        if any(
            source.operation not in EVA_SUPPORTING_OPERATIONS
            for source in self.supporting_sources
        ):
            raise ValueError(
                "EVA supporting sources are the Run's assembly report, contig names "
                "or submitted variants"
            )
        return self


_RETAINED_MEDIA = {
    # An assembly's sequence names: NCBI's `links` answer (JSON), then its FTP
    # assembly report (text).
    "ncbi_sequence_names": {"application/json", "text/plain"},
    # An assembly's annotation, NCBI Datasets' zip package.
    "ncbi_genome_annotation": {"application/zip"},
}
"""The media a retained source answer may have, by operation; JSON otherwise."""


LOCATION_OPERATIONS = frozenset(
    {
        "fetch_assembly_report",
        # A merged record a position lookup followed to its current rsID.
        "fetch_refsnp",
        "ncbi_sequence_names",
        "ncbi_reference_span",
        "ncbi_spdi_rsids",
        "eva_contig_alias",
        "eva_clustered_variants_at",
        # A position stated on another version, moved onto the Run's.
        "ncbi_taxon_assemblies",
        "ncbi_spdi_equivalents",
    }
)
"""The GETs that place a stated position on the Run's assembly (#369)."""


class StatedVersionPlacement(StrictFrozenContract):
    """Where the source places the looked-up variant on the version a request
    states, when that is not the Run's (#369): the paper's own coordinates."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: StrictStr
    contig: StrictStr
    one_based_position: StrictInt = Field(gt=0)
    ref: StrictStr = Field(pattern=_ALLELE)
    alt: StrictStr = Field(pattern=_ALLELE)


class VersionListing(StrictFrozenContract):
    """The rsIDs dbSNP lists at a stated allele read on another version than the
    Run's, for a position whose request states no version (#369)."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: StrictStr
    contig: StrictStr | None
    listed_rsids: tuple[_CanonicalRsid, ...]


class SequenceChoice(StrictFrozenContract):
    """The model's naming of the sequence a written contig means on one assembly
    (#467, ADR 0049; owner D1 a): one record of that assembly's report, by its GenBank
    or RefSeq accession, with the researcher's words that write the contig. Code
    checks only membership: the quote stands in the researcher's words and holds the
    written contig, and the assembly's report has the record."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    record: StrictStr = Field(min_length=1, max_length=200)
    quote: StrictStr = Field(min_length=1, max_length=2000)


class SequenceNaming(StrictFrozenContract):
    """A named sequence the lookup read, as checked (#467): the contig as written, the
    record the model named for it and its quote, and whether the written contig is
    literally one of the record's names or the match is the model's
    (``model_matched``, owner RC5 1a). A position a placement moved names its Run
    sequence by the accession the placement's source answer gave it, with no quote."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    written: StrictStr = Field(min_length=1)
    record: StrictStr = Field(min_length=1)
    sequence_name: StrictStr = Field(min_length=1)
    quote: StrictStr | None
    match: Literal["literal", "model_matched"]


class VersionChoice(StrictFrozenContract):
    """The model's naming of the genome version a request states (#471 Q3 a, ADR
    0049): one assembly by its accession, with the researcher's words that state the
    version. Code checks only membership: the assembly is a row of an NCBI listing this
    member read (`search_assemblies`) or an assembly a dbSNP record it read places the
    variant on, and the quote stands in the line that states the version."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    quote: StrictStr = Field(min_length=1, max_length=2000)


class VersionNaming(StrictFrozenContract):
    """The stated version a lookup read, as checked (#471 Q3 a): the version as
    written, the assembly named for it and its quote, and whether the written version
    is literally one of that assembly's names (its name, UCSC name, accession or
    paired accession) or the match is the model's (``model_matched``)."""

    written: StrictStr = Field(min_length=1)
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    quote: StrictStr | None
    match: Literal["literal", "model_matched"]


AssemblyStatus = Literal["current", "previous", "suppressed", "retired"]
"""NCBI's status of an assembly version: current, or replaced (``previous``),
suppressed or retired, which only its any-version report describes."""


class BodylessAnswer(StrictFrozenContract):
    """A listing's answer that carries no body: only its status, which the model's
    ``listed_none`` decision cites when there are no words to quote (owner #480 Q3 a).
    It establishes no rsID listed there and nothing about the variant's existence."""

    status_code: StrictInt = Field(ge=100, le=599)


class VariantLocationDetail(StrictFrozenContract):
    """What a position lookup adds to the shared result (#369): the request's stated
    position, the Run's name for its sequence, the Run's reference bases at its REF,
    the rsIDs the source lists at it, and every GET that placed it. Every field is
    always written, ``None`` or empty when it does not apply. A placed target (#367)
    is looked up at its ``placed`` position, kept beside the stated one; ``stated``
    is empty only for a placed gene and change, which states no position."""

    stated: StatedPosition | None
    stated_assembly: StatedAssembly | None
    placed: PlacedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # A region target's own lookup: its sequence named on the Run's assembly and
    # ``reference_bases`` over the whole region.
    region: StatedRegion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    run_assembly_name: StrictStr | None
    run_assembly_synonym: StrictStr | None
    # The Run's assembly's NCBI status and paired GenBank/RefSeq accession, written
    # whenever its report was read; a result stored before they were kept has neither.
    run_assembly_status: AssemblyStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    run_assembly_paired_accession: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    run_contig: StrictStr | None
    reference_sequence: StrictStr | None
    reference_bases: StrictStr | None
    listed_rsids: tuple[_CanonicalRsid, ...]
    # The record's placements on the version a request states, when that is not the
    # Run's, with the stated sequence's RefSeq name there; with no version stated,
    # its placements on every version but the Run's, cited by the intent check.
    stated_version_placements: tuple[StatedVersionPlacement, ...]
    stated_version_contig: StrictStr | None
    version_listings: tuple[VersionListing, ...]
    # Every sequence the lookup named, on the Run's assembly and any other.
    sequence_namings: tuple[SequenceNaming, ...]
    # The assembly the stated version was named as; none when the request states no
    # version or a placement put the position on the Run's assembly.
    version_naming: VersionNaming | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The model's words, quoted from a listing service's own answer, that it lists no
    # rsID at the stated allele: checked by membership in each listing answer the
    # lookup read that was not 2xx; or, for an answer with no body, its status,
    # checked against that answer.
    listed_none: StrictStr | BodylessAnswer | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # EVA's own variantType the model named for EVA's listing at the stated position,
    # passed to EVA as is; none when it took the allele's EVA form.
    variant_type: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # EVA's own contigNamingConvention the model named for EVA's GETs; none when it
    # named none (EVA's INSDC).
    contig_naming: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    supporting_sources: tuple[RetrievedSource, ...]
    # The GET that names the stated sequence on another version, made after the
    # record it needs (its placements name that version's accession).
    following_sources: tuple[RetrievedSource, ...]

    @property
    def looked_up(self) -> StatedPosition | PlacedPosition | None:
        """The position the lookup read: the placed one, else the stated one; none
        for a region."""
        return self.placed or self.stated

    @model_validator(mode="after")
    def location_sources(self) -> "VariantLocationDetail":
        if (self.stated is None and self.placed is None) == (self.region is None):
            raise ValueError("a location reads a position or a region")
        if any(
            source.operation not in LOCATION_OPERATIONS
            for source in (*self.supporting_sources, *self.following_sources)
        ):
            raise ValueError("a location keeps only the GETs that place a position")
        return self


ADMISSIBLE_LOOKUP_STATUSES = frozenset({"resolved", "unlisted"})
"""A lookup whose candidates may become Work items: a source's resolved record, or a
stated position no source lists, its REF on the Run's reference (#369)."""
RefSnpStatus = Literal[
    "resolved",
    "unlisted",
    "merged",
    "ambiguous",
    "conflicted",
    "incompatible",
    "not_found",
    "malformed",
    "region_located",
    "sequence_unnamed",
    "version_unnamed",
    "placement_unnamed",
    "source_answered",
]
"""``region_located`` (#367, owner option b): a region's sequence named on the Run's
assembly and its reference bases read, with no rsID listed and no candidate.
``sequence_unnamed`` (#467, owner D1 a): the lookup reached an assembly on which no
member sequence is named for the written contig; its conflict names the assembly, and
the model names one (`search_sequence_names`) and calls again.
``version_unnamed`` (#471, owner Q3 a, R3 a): the request states a genome version
that is not literally the Run's and the call named no assembly for it; the model
searches NCBI's assemblies (`search_assemblies`), or reads the dbSNP record's
placements the result copies, and calls again naming one (``version``).
``placement_unnamed`` (#468, owner Q1 a): the record places the variant at more than one
locus on the Run's assembly and the call named none of its placements; every placement
is copied, and the model names one (``placement``) and calls again.
``source_answered`` (#468, owner Q2 a, R1 b, R4 a, ADR 0049): a source answered something
other than a record the lookup reads, any non-2xx status among them or a record with no
placement snapshot (``record_state``); its answer is in the Tool's ``source_answers``,
for the model to read, with no verdict on it."""


# The lookup endings that carry a typed origin (``lookup_origins``).
TYPED_ENDINGS = frozenset(
    {
        "source_answered",
        "merged",
        "sequence_unnamed",
        "placement_unnamed",
        "version_unnamed",
        "malformed",
    }
)


class RefSnpToolResult(StrictFrozenContract):
    result_id: StrictStr = Field(pattern=r"^refsnp-result:[0-9a-f]{64}$")
    # The rsID the researcher stated; a position-only request states none.
    requested_rsid: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        exclude_if=lambda value: value is None,
    )
    verified_rsid: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        exclude_if=lambda value: value is None,
    )
    expected_assembly_accession: StrictStr
    status: RefSnpStatus
    candidates: tuple[ResolvedVariantCandidate, ...]
    conflicts: tuple[StrictStr, ...]
    # A merged record's current rsIDs (dbSNP ``merged_snapshot_data.merged_into``):
    # the lookups this member's admission now also allows.
    merged_into: tuple[_CanonicalRsid, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # The rsIDs a current record says dbSNP merged into it (``dbsnp1_merges``):
    # the source's own record of a rename the intent check accepts.
    merged_rsids: tuple[_CanonicalRsid, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # Every placement the record states, copied, each marked on or off the Run's
    # assembly (ADR 0049): no top-level filter, no cap.
    placements: tuple[VariantPlacement, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    # The snapshot a dbSNP record carries, by dbSNP's own key
    # (``primary_snapshot_data``, ``merged_snapshot_data``, ``withdrawn_snapshot_data``,
    # ...), copied.
    record_state: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The placement the model named, by its sequence, when the record has more than one
    # locus on the Run's assembly; checked by membership.
    named_placement: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The ending's typed origin: its audit sentence is the conflict and the
    # rationale, stating only what the source answered.
    origin: LookupOrigin | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source: RetrievedSource
    assessment: EvidenceAssessment
    # The lookup's own source and, for EVA, what its answer adds. Both are always
    # written: a dbSNP result says "not_applicable" rather than leaving EVA out.
    lookup_source: LookupSource
    eva: EvaLookupDetail | Literal["not_applicable"]
    # A position lookup's placement; an rsID lookup says "not_applicable".
    location: VariantLocationDetail | Literal["not_applicable"]

    @model_validator(mode="after")
    def exact_result(self) -> "RefSnpToolResult":
        if self.assessment.source_id != self.source.source_id:
            raise ValueError("RefSNP assessment changed its source")
        if self.status in TYPED_ENDINGS and self.origin is None:
            # Each of these endings is typed by its origin's owner.
            raise ValueError("RefSNP ending has no typed origin")
        if self.origin is not None and (
            # A record with no placement snapshot is the source's answer.
            {
                "record_state": "source_answered",
                # Not looked up: no source lists rsIDs without an allele.
                "position_without_alleles": "not_found",
            }.get(self.origin.code, self.origin.code)
            != self.status
            or self.conflicts != (self.origin.audit(),)
            or self.assessment.rationale != self.origin.audit()
        ):
            raise ValueError("RefSNP ending changed its origin's sentence")
        located = self.location != "not_applicable"
        # A position lookup that reached no record ends on its own last GET.
        record = self.source.operation in {"fetch_refsnp", "eva_clustered_variant"}
        if self.lookup_source == "ncbi":
            if (
                not located
                or self.location.region is None
                or self.eva != "not_applicable"
                or self.source.operation
                not in LOCATION_OPERATIONS
                - {"eva_contig_alias", "eva_clustered_variants_at", "fetch_refsnp"}
            ):
                raise ValueError(
                    "only a region's lookup reads NCBI alone, with no variant source"
                )
        elif self.lookup_source == "ncbi_dbsnp":
            if self.eva != "not_applicable" or not (
                self.source.operation == "fetch_refsnp"
                or located
                and self.source.operation
                in LOCATION_OPERATIONS
                - {"eva_contig_alias", "eva_clustered_variants_at"}
            ):
                raise ValueError(
                    "a dbSNP result reads fetch_refsnp and has no EVA detail"
                )
        elif self.status == "merged" or not (
            record
            and self.source.operation == "eva_clustered_variant"
            and self.eva != "not_applicable"
            or located
            and not record
            and self.source.operation in LOCATION_OPERATIONS
            and self.eva == "not_applicable"
        ):
            raise ValueError(
                "an EVA result reads its clustered variant and keeps its detail"
            )
        if not located and self.requested_rsid is None:
            raise ValueError("an rsID lookup states the rsID it was asked")
        if self.status == "unlisted" and (
            record or len(self.candidates) != 1 or self.candidates[0].aliases
        ):
            raise ValueError(
                "only a located position no source lists is unlisted, as its one "
                "stated allele"
            )
        if self.status == "resolved" and not record:
            raise ValueError("a resolved lookup reads its source's record")
        region = located and self.location.region is not None
        if self.status == "region_located" and (
            not region
            or self.candidates
            or self.conflicts
            or self.verified_rsid is not None
            or record
        ):
            raise ValueError(
                "a located region is a region lookup with no rsID, candidate or "
                "conflict"
            )
        if region and self.status not in {
            "region_located",
            "incompatible",
            "malformed",
            "sequence_unnamed",
            "version_unnamed",
            "source_answered",
        }:
            raise ValueError(
                "a region lookup is located, incompatible, malformed, unnamed or "
                "answered by its source"
            )
        if (self.status == "merged") != bool(self.merged_into):
            raise ValueError("only a merged RefSNP result names its current rsIDs")
        if self.status == "resolved" and (
            len({candidate.locus for candidate in self.candidates}) != 1
            or len({candidate.alt for candidate in self.candidates})
            != len(self.candidates)
        ):
            raise ValueError(
                "resolved RefSNP result requires one candidate per ALT at one locus"
            )
        if self.candidates and self.status not in {
            "resolved",
            "unlisted",
            "ambiguous",
            "conflicted",
            "placement_unnamed",
        }:
            raise ValueError("unresolved source status cannot carry candidate targets")
        if self.status == "resolved" and self.conflicts:
            raise ValueError("resolved RefSNP result cannot retain conflicts")
        material = self.model_dump(mode="json", exclude={"result_id"})
        if self.result_id != _digest("refsnp-result", material):
            raise ValueError("RefSNP result identity changed")
        return self

    @classmethod
    def create(cls, **values: Any) -> "RefSnpToolResult":
        # An absent rsID is omitted, as its dump omits it.
        values = {
            key: value
            for key, value in values.items()
            if not (key in {"requested_rsid", "verified_rsid"} and value is None)
        }
        material = cls._canonical_material(values)
        # Nested contracts are passed as built: a dumped one would carry lists where
        # its strict tuples belong.
        return cls(result_id=_digest("refsnp-result", material), **values)

    @classmethod
    def _canonical_material(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
            for key, value in values.items()
        }


AssemblyReportStatus = Literal[
    "compatible", "conflicted", "incompatible", "not_found", "malformed"
]


class AssemblyReportToolResult(StrictFrozenContract):
    result_id: StrictStr = Field(pattern=r"^assembly-report-result:[0-9a-f]{64}$")
    expected_assembly_accession: StrictStr
    expected_assembly_name: StrictStr
    status: AssemblyReportStatus
    observed_assembly_accession: StrictStr | None = None
    observed_assembly_name: StrictStr | None = None
    # NCBI's status of the exact report's assembly, written whenever one was read; a
    # result stored before it was kept has none.
    observed_assembly_status: AssemblyStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    conflicts: tuple[StrictStr, ...]
    source: RetrievedSource
    assessment: EvidenceAssessment

    @model_validator(mode="after")
    def exact_result(self) -> "AssemblyReportToolResult":
        if self.assessment.source_id != self.source.source_id:
            raise ValueError("assembly assessment changed its source")
        if self.status == "compatible":
            if (
                self.observed_assembly_accession != self.expected_assembly_accession
                or self.observed_assembly_name != self.expected_assembly_name
                or self.conflicts
            ):
                raise ValueError(
                    "compatible assembly result changed its exact identity"
                )
        material = self.model_dump(mode="json", exclude={"result_id"})
        if self.result_id != _digest("assembly-report-result", material):
            raise ValueError("assembly-report result identity changed")
        return self

    @classmethod
    def create(cls, **values: Any) -> "AssemblyReportToolResult":
        # An unread status is omitted, as its dump omits it.
        material = {
            key: value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
            for key, value in values.items()
            if not (key == "observed_assembly_status" and value is None)
        }
        return cls(result_id=_digest("assembly-report-result", material), **material)


class VariantTarget(StrictFrozenContract):
    """One Work item's variant, its allele trimmed and left-aligned (#368)."""

    target_id: StrictStr = Field(pattern=r"^variant-target:[0-9a-f]{64}$")
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: StrictStr
    contig: StrictStr
    one_based_position: StrictInt = Field(gt=0)
    ref: StrictStr = Field(pattern=_ALLELE)
    alt: StrictStr = Field(pattern=_ALLELE)
    aliases: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    assessment_ids: tuple[StrictStr, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def exact_identity(self) -> "VariantTarget":
        if not command_safe_contig(self.contig) or self.ref == self.alt:
            raise ValueError("Variant target is not one safe variant")
        if len(self.aliases) != len(set(self.aliases)) or any(
            not _RSID.fullmatch(alias) for alias in self.aliases
        ):
            raise ValueError("Variant target aliases are not exact canonical rsIDs")
        if len(self.assessment_ids) != len(set(self.assessment_ids)):
            raise ValueError("Variant target assessments must be unique")
        material = self.model_dump(mode="json", exclude={"target_id"})
        if self.target_id != _digest("variant-target", material):
            raise ValueError("Variant-target identity changed")
        return self

    @property
    def variant_text(self) -> str:
        """``contig:position REF>ALT``, with ``-`` for an empty allele."""
        return (
            f"{self.contig}:{self.one_based_position} {allele_text(self.ref, self.alt)}"
        )


class RegionTarget(StrictFrozenContract):
    """One region Work item's stretch of the Run's sequence (#367, owner option b):
    every VCF record in it is its own row, checked by the VCF and REF only."""

    target_id: StrictStr = Field(pattern=r"^region-target:[0-9a-f]{64}$")
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: StrictStr
    contig: StrictStr
    start: StrictInt = Field(gt=0)
    end: StrictInt = Field(gt=0)
    # The Run's reference bases over the region, each record's REF is checked by.
    reference_bases: StrictStr = Field(pattern=r"^[ACGTN]+$")
    assessment_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_identity(self) -> "RegionTarget":
        if (
            not command_safe_contig(self.contig)
            or self.start > self.end
            or len(self.reference_bases) != self.end - self.start + 1
        ):
            raise ValueError("Region target is not one safe region")
        material = self.model_dump(mode="json", exclude={"target_id"})
        if self.target_id != _digest("region-target", material):
            raise ValueError("Region-target identity changed")
        return self

    @property
    def variant_text(self) -> str:
        """``contig:start-end``, as a region's rows name it."""
        return f"{self.contig}:{self.start}-{self.end}"


def _work_item_semantic_id(
    target: "VariantTarget | RegionTarget", intended_action: QuestionAction
) -> str:
    if isinstance(target, RegionTarget):
        return _digest(
            "work-item",
            {
                "assembly_accession": target.assembly_accession,
                "contig": target.contig,
                "start": target.start,
                "end": target.end,
                "normalized_question_action": intended_action,
            },
        )
    return _digest(
        "work-item",
        {
            "assembly_accession": target.assembly_accession,
            "contig": target.contig,
            "one_based_position": target.one_based_position,
            "ref": target.ref,
            "alt": target.alt,
            "normalized_question_action": intended_action,
        },
    )


class WorkItem(StrictFrozenContract):
    work_item_id: StrictStr = Field(pattern=r"^work-item:[0-9a-f]{64}$")
    sample_id: StrictStr
    target: VariantTarget | RegionTarget
    intended_action: QuestionAction = "observe_variant"
    source_ids: tuple[StrictStr, ...] = Field(min_length=2)
    revision_id: StrictStr | None = Field(
        default=None, pattern=r"^investigation-revision:[0-9a-f]{64}$"
    )
    previous_work_item_id: StrictStr | None = Field(
        default=None, pattern=r"^work-item:[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def exact_identity(self) -> "WorkItem":
        if isinstance(self.target, RegionTarget) != (
            self.intended_action == "observe_region"
        ):
            raise ValueError("a region Work item, and only one, observes its region")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("Work-item sources must be unique")
        legacy_material = self.model_dump(
            mode="json",
            exclude={"work_item_id", "revision_id", "previous_work_item_id"},
        )
        allowed = (
            {
                _digest("work-item", legacy_material),
                _work_item_semantic_id(self.target, self.intended_action),
            }
            if self.revision_id is None
            else {
                _digest(
                    "work-item", self.model_dump(mode="json", exclude={"work_item_id"})
                )
            }
        )
        if self.previous_work_item_id is not None and self.revision_id is None:
            raise ValueError("Work-item predecessor requires its revision")
        if self.work_item_id not in allowed:
            raise ValueError("Work-item identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        sample_id: str,
        target: "VariantTarget | RegionTarget",
        intended_action: QuestionAction = "observe_variant",
        source_ids: tuple[str, ...],
    ) -> "WorkItem":
        return cls(
            work_item_id=_work_item_semantic_id(target, intended_action),
            sample_id=sample_id,
            target=target,
            intended_action=intended_action,
            source_ids=source_ids,
        )


class InvestigationRequestNotChecked(StrictFrozenContract):
    """A target the Run cannot check by what it states, recorded by the coordinator
    before any model request (#367, owner 2026-10-22): no step placed it, or it states
    no rsID, position or region. It asks nothing; its reason is its outcome."""

    event_type: Literal["InvestigationRequestNotChecked"] = (
        "InvestigationRequestNotChecked"
    )
    collection_member_id: StrictStr
    reason_code: NotCheckedReason
    reason: StrictStr = Field(min_length=1, max_length=1_500)


class InvestigationNotChecked(StrictFrozenContract):
    """A member's recorded not-checked outcome, as its proposal accounts for it."""

    event_id: StrictStr
    reason_code: NotCheckedReason
    reason: StrictStr


class InvestigationRequestRejection(StrictFrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    reason: StrictStr
    # Who ended the member, where its answer names one; an effort or availability
    # stop names nobody, and is left out so earlier proposals keep their bytes.
    asserted_by: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The issue the researcher removed the item over, when the answer was to an
    # issue Review: the report keeps its source and verdict.
    issue: ItemIssue | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class InvestigationIssueApproval(StrictFrozenContract):
    """The researcher's approval of a member's issue finding: checked, not verified."""

    request_event_id: StrictStr
    response_event_id: StrictStr
    asserted_by: StrictStr


class InvestigationRequestMember(StrictFrozenContract):
    """One attributed request and its source-backed resolution, before Work exists."""

    request_member_id: StrictStr = Field(pattern=r"^request-member:[0-9a-f]{64}$")
    request: InvestigationRequestItem
    refsnp: RefSnpToolResult | None
    assembly_report: AssemblyReportToolResult | None
    rejection: InvestigationRequestRejection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # An issue finding the researcher approved: it no longer blocks work.
    issue_approval: InvestigationIssueApproval | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # A target the Run could not check by what it states: no source, no Work.
    not_checked: InvestigationNotChecked | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Who asserts the target a researcher's answer supplied; the request item
    # and its requested rsID stay the original's.
    target_asserted_by: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # One candidate Work item per source-backed ALT; approval selects among them.
    work_item_ids: tuple[
        Annotated[StrictStr, Field(pattern=r"^work-item:[0-9a-f]{64}$")], ...
    ] = ()
    previous_request_member_id: StrictStr | None = Field(
        default=None, pattern=r"^request-member:[0-9a-f]{64}$"
    )

    @property
    def conflicts(self) -> tuple[str, ...]:
        return (() if self.refsnp is None else self.refsnp.conflicts) + (
            () if self.assembly_report is None else self.assembly_report.conflicts
        )

    @model_validator(mode="after")
    def exact_disposition(self):
        if self.not_checked is not None:
            if (
                self.rejection is not None
                or self.issue_approval is not None
                or self.refsnp is not None
                or self.assembly_report is not None
                or self.work_item_ids
            ):
                raise ValueError("a target not checked has no source or Work")
            return self
        if self.rejection is not None:
            if (
                self.refsnp is not None
                or self.assembly_report is not None
                or self.work_item_ids
            ):
                raise ValueError(
                    "rejected request accounting cannot grant scientific evidence"
                )
        elif self.refsnp is None:
            raise ValueError("unrejected request requires its selected source")
        if self.issue_approval is not None and (
            self.rejection is not None or self.work_item_ids
        ):
            raise ValueError("an approved issue finding is a member without Work")
        return self


class InvestigationProposal(StrictFrozenContract):
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    revision_number: StrictInt = Field(gt=0)
    investigation_input_id: StrictStr = Field(
        pattern=r"^investigation-input:[0-9a-f]{64}$"
    )
    assertion_id: StrictStr = Field(pattern=r"^user-assertion:[0-9a-f]{64}$")
    work_items: tuple[WorkItem, ...]
    request_members: tuple[InvestigationRequestMember, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    unresolved_conflicts: tuple[StrictStr, ...]
    completeness: Literal["complete"] = "complete"
    supersedes_proposal_id: StrictStr | None = Field(
        default=None, pattern=r"^investigation-proposal:[0-9a-f]{64}$"
    )
    predecessor_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @property
    def unresolved_request_member_ids(self) -> tuple[str, ...]:
        return tuple(
            member.request_member_id
            for member in self.request_members
            if not member.work_item_ids
            and member.rejection is None
            and member.issue_approval is None
            and member.not_checked is None
        )

    @model_validator(mode="after")
    def exact_identity(self) -> "InvestigationProposal":
        if (self.revision_number == 1) != (self.supersedes_proposal_id is None):
            raise ValueError("proposal revision does not match supersession")
        if self.unresolved_conflicts:
            raise ValueError("complete proposal cannot retain unresolved conflicts")
        if not self.work_items and not self.request_members:
            raise ValueError("proposal requires request membership")
        if len({member.request_member_id for member in self.request_members}) != len(
            self.request_members
        ):
            raise ValueError("proposal repeated request membership")
        work_ids = {item.work_item_id for item in self.work_items}
        if any(
            item not in work_ids
            for member in self.request_members
            for item in member.work_item_ids
        ):
            raise ValueError("request member changed its normalized Work link")
        semantic_ids = tuple(
            _work_item_semantic_id(item.target, item.intended_action)
            for item in self.work_items
        )
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("duplicate semantic Work item")
        if len(self.predecessor_ids) != len(set(self.predecessor_ids)):
            raise ValueError("proposal predecessors must be unique")
        material = self.model_dump(mode="json", exclude={"proposal_id"})
        if self.proposal_id != _digest("investigation-proposal", material):
            raise ValueError("Investigation proposal identity changed")
        return self


class InvestigationAuthority(StrictFrozenContract):
    authority_id: StrictStr = Field(pattern=r"^investigation-authority:[0-9a-f]{64}$")
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    review_response_id: StrictStr = Field(
        pattern=r"^investigation-review-response:[0-9a-f]{64}$"
    )
    authorized_work_items: tuple[WorkItem, ...] = Field(min_length=1)
    predecessor_ids: tuple[StrictStr, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def exact_identity(self) -> "InvestigationAuthority":
        if len(self.predecessor_ids) != len(set(self.predecessor_ids)):
            raise ValueError("Investigation authority predecessors must be unique")
        material = self.model_dump(mode="json", exclude={"authority_id"})
        if self.authority_id != _digest("investigation-authority", material):
            raise ValueError("Investigation authority identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        proposal_id: str,
        review_response_id: str,
        authorized_work_items: tuple[WorkItem, ...],
        predecessor_ids: tuple[str, ...],
    ) -> "InvestigationAuthority":
        material = {
            "proposal_id": proposal_id,
            "review_response_id": review_response_id,
            "authorized_work_items": tuple(
                item.model_dump(mode="python") for item in authorized_work_items
            ),
            "predecessor_ids": predecessor_ids,
        }
        return cls(
            authority_id=_digest("investigation-authority", material), **material
        )


class InvestigationStarted(StrictFrozenContract):
    event_type: Literal["InvestigationStarted"] = "InvestigationStarted"
    continuation_id: StrictStr = Field(
        pattern=r"^scientific-continuation:[0-9a-f]{64}$"
    )
    investigation_input: InvestigationInput
    assertion: UserAssertion
    # Planning's retained GETs that describe the Run's assembly, which the lookups
    # read in place of a contact, and the Run organism's taxon at species
    # level, which `search_assemblies` is bound to. Outside
    # `InvestigationInput`, whose digest is its identity.
    run_reference_sources: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    run_taxid: StrictInt | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )


class InvestigationAttemptBound(StrictFrozenContract):
    event_type: Literal["InvestigationAttemptBound"] = "InvestigationAttemptBound"
    attempt_id: StrictStr
    continued_attempt_id: StrictStr | None = None
    investigation_input_id: StrictStr
    assertion_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    supersedes_proposal_id: StrictStr | None = None
    revision_id: StrictStr | None = Field(
        default=None, pattern=r"^investigation-revision:[0-9a-f]{64}$"
    )
    work_requests: tuple[InvestigationRequestItem, ...] = Field(min_length=1)
    continuation_policy: ContinuationPolicy
    runtime_selections: RetainedRuntimeSelections
    continuation_policy_id: StrictStr
    # The one collection member this Attempt plans; its continuations keep
    # it. Additive: absent when unset.
    collection_member_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def exact_work_binding(self) -> "InvestigationAttemptBound":
        if self.continuation_policy_id != self.continuation_policy.policy_id:
            raise ValueError("Investigation Attempt changed its continuation policy")
        return self


class InvestigationSourceContactStarted(StrictFrozenContract):
    event_type: Literal["InvestigationSourceContactStarted"] = (
        "InvestigationSourceContactStarted"
    )
    attempt_id: StrictStr
    operation: SourceOperation
    request_key: StrictStr


class InvestigationSourceContactFailed(StrictFrozenContract):
    event_type: Literal["InvestigationSourceContactFailed"] = (
        "InvestigationSourceContactFailed"
    )
    attempt_id: StrictStr
    operation: SourceOperation
    request_key: StrictStr
    error_type: StrictStr
    detail: StrictStr


class InvestigationSourceRetrieved(StrictFrozenContract):
    event_type: Literal["InvestigationSourceRetrieved"] = "InvestigationSourceRetrieved"
    attempt_id: StrictStr
    source: RetrievedSource


class InvestigationSourceCarried(StrictFrozenContract):
    """A carried planning GET a Tool read in place of its own contact (#479 G): its
    bytes retained again on this fact, on the Tool's causal chain where the GET would
    be; ``cited_event_id`` is planning's retained GET, one the Investigation's start
    carries. No source was contacted."""

    event_type: Literal["InvestigationSourceCarried"] = "InvestigationSourceCarried"
    attempt_id: StrictStr
    cited_event_id: StrictStr
    source: RetrievedSource


class RefSnpToolResultRecorded(StrictFrozenContract):
    event_type: Literal["RefSnpToolResultRecorded"] = "RefSnpToolResultRecorded"
    attempt_id: StrictStr
    source_event_id: StrictStr
    collection_member_id: StrictStr
    tool_call_id: StrictStr
    query: StrictStr
    result: RefSnpToolResult
    # What each of the Tool's GETs' sources answered, as the model is shown it.
    source_answers: tuple[SourceAnswer, ...] = Field(min_length=1)
    # The item's lookup decisions the lookup read, in the order it read them; left
    # out when it read none.
    decision_event_ids: tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class SequenceDecision(StrictFrozenContract):
    """The sequence the written contig means on one assembly a ``sequence_unnamed``
    result named (#467, #479 G2.2): one record of that assembly's report, with the
    researcher's words that write the contig."""

    kind: Literal["sequence"] = "sequence"
    choice: SequenceChoice


class VersionDecision(StrictFrozenContract):
    """The assembly the stated genome version means, for a ``version_unnamed`` result
    (#471, #479 G2.2), with the researcher's words that state it."""

    kind: Literal["version"] = "version"
    choice: VersionChoice


class PlacementDecision(StrictFrozenContract):
    """The placement a ``placement_unnamed`` result's record means, by its Run
    sequence (#468, #479 G2.2): one of the sequences the result lists."""

    kind: Literal["placement"] = "placement"
    placement: StrictStr = Field(min_length=1, max_length=200)


class ListedNoneDecision(StrictFrozenContract):
    """A listing service's own words saying it lists no rsID at the stated allele, for
    a ``source_answered`` position result (#468 R5 b, #479 G2.2); or, when that answer
    has no body, its status (owner #480 Q3 a). Exactly one of the two."""

    kind: Literal["listed_none"] = "listed_none"
    quote: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        exclude_if=lambda value: value is None,
    )
    status_code: StrictInt | None = Field(
        default=None, ge=100, le=599, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def one_basis(self) -> "ListedNoneDecision":
        if (self.quote is None) == (self.status_code is None):
            raise ValueError("a listed-none decision quotes words or cites a status")
        return self

    @property
    def listing(self) -> "str | BodylessAnswer":
        """What a later lookup reads the listing's answer against."""
        if self.quote is not None:
            return self.quote
        return BodylessAnswer(status_code=self.status_code)


LookupDecision = Annotated[
    SequenceDecision | VersionDecision | PlacementDecision | ListedNoneDecision,
    Field(discriminator="kind"),
]


class InvestigationLookupDecided(StrictFrozenContract):
    """One decision a lookup result asked for, made once and recorded as its own fact
    (owner, #476 alignment G2.2 B; ADR 0049): ``refsnp_event_id`` is the result that
    asked, and the decision is checked by membership in that result's options. A later
    lookup of the same item reads it; nothing re-sends it."""

    event_type: Literal["InvestigationLookupDecided"] = "InvestigationLookupDecided"
    attempt_id: StrictStr
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    tool_call_id: StrictStr
    refsnp_event_id: StrictStr
    decision: LookupDecision


class AssemblyReportToolResultRecorded(StrictFrozenContract):
    event_type: Literal["AssemblyReportToolResultRecorded"] = (
        "AssemblyReportToolResultRecorded"
    )
    attempt_id: StrictStr
    source_event_id: StrictStr
    collection_member_id: StrictStr
    tool_call_id: StrictStr
    query: StrictStr
    refsnp_event_id: StrictStr
    result: AssemblyReportToolResult
    source_answers: tuple[SourceAnswer, ...] = Field(min_length=1)


class SequenceNamesPage(StrictFrozenContract):
    """One page of an assembly's retained sequence names (#467, owner Q1 a): the rows
    whose columns hold the searched text, each copied under NCBI's column names, in the
    report's order, from ``offset``, with how many match in all, so the rest stays
    reachable (ADR 0049, bounded views)."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    report_source_event_id: StrictStr
    text: StrictStr
    offset: StrictInt = Field(ge=0)
    total_matches: StrictInt = Field(ge=0)
    rows: tuple[dict[StrictStr, StrictStr | StrictInt | None], ...]


class AssemblySearchPage(StrictFrozenContract):
    """One page of NCBI Datasets' genome reports for a taxon under the model's query
    (#471 R3 a): every row copied in NCBI's order, NCBI's count of the whole listing
    and its next page token (ADR 0049, bounded views). The same copied rows as
    planning's discovery."""

    ncbi_tax_id: StrictInt = Field(gt=0)
    query: AssemblyQuery
    # The source's status: only a 2xx answer that reads as a page has rows; any other
    # answered GET is NCBI's answer for the model to read in the Tool's source
    # answers (ADR 0049), ``unread`` naming why a 2xx answer was no page.
    status_code: StrictInt
    rows: tuple[AssemblySourceRecord, ...] = ()
    total_count: StrictInt | None = Field(default=None, ge=0)
    next_page_token: StrictStr | None = Field(default=None, min_length=1)
    unread: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def rows_of_a_page(self) -> "AssemblySearchPage":
        if self.rows and (not 200 <= self.status_code < 300 or self.unread):
            raise ValueError("only a 2xx page that reads has copied rows")
        return self


class AssembliesSearched(StrictFrozenContract):
    """A search of NCBI's genome reports the model made for a stated version (#471):
    one GET, retained, and its copied page; the stated version the model then names
    is checked against these rows."""

    event_type: Literal["AssembliesSearched"] = "AssembliesSearched"
    attempt_id: StrictStr
    source_event_id: StrictStr
    collection_member_id: StrictStr
    tool_call_id: StrictStr
    result: AssemblySearchPage
    source_answers: tuple[SourceAnswer, ...] = Field(min_length=1)


class InvestigationGeneChangePlaced(StrictFrozenContract):
    """G15's answer for a member's typed gene change on the Run's assembly (#473 Q1 a,
    Q3 a, ADR 0049): every transcript reading and candidate the Run's own annotation
    gives, copied, and NCBI's HGVS answer, status and body, when a transcript is
    named; every GET retained and shown. Code gives no verdict: the model picks a
    candidate (`InvestigationGeneChangePicked`) or opens the member's Review."""

    event_type: Literal["InvestigationGeneChangePlaced"] = (
        "InvestigationGeneChangePlaced"
    )
    attempt_id: StrictStr
    source_event_id: StrictStr
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    tool_call_id: StrictStr
    gene_change: StatedChange
    result: GeneChangeAnswer
    source_answers: tuple[SourceAnswer, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def its_own_change(self) -> "InvestigationGeneChangePlaced":
        """It places the member's typed gene change; the transcript the model named,
        if any, is the claim's (#473 Q4 a)."""
        claimed = self.result.claim.change.model_copy(
            update={"transcript": self.gene_change.transcript}
        )
        if claimed != self.gene_change or not self.gene_change.typed:
            raise ValueError("a placement answers the typed gene change it placed")
        return self


class InvestigationGeneChangePicked(StrictFrozenContract):
    """The model's pick of one candidate a placement of the member's gene change
    copied (#473 Q11, ADR 0049): checked by membership in that placement's
    candidates. It places the member at that candidate, a placed position on the
    Run's assembly, which the member's lookup then reads as any SNV; the model's
    reason cites what it rests on (a transcript's tags, the researcher's words)."""

    event_type: Literal["InvestigationGeneChangePicked"] = (
        "InvestigationGeneChangePicked"
    )
    attempt_id: StrictStr
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    placement_event_id: StrictStr
    gene_change: StatedChange
    placed: PlacedPosition
    reason: StrictStr = Field(min_length=1, max_length=2000)
    # The member's answers this pick was made after, by event id: a newer answer
    # outranks it.
    after_answer_event_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def an_allele(self) -> "InvestigationGeneChangePicked":
        if self.placed.ref is None:
            raise ValueError("a picked candidate is one allele: its REF and ALT")
        return self


class InvestigationTargetRead(StrictFrozenContract):
    """The target the Investigation's model typed from a member's words (#472 Q3 a,
    R1 b): a researcher answer's instruction (``answer``, by its response event) or an
    untyped request item's own words (``request``, by the Investigation's start). A
    target cites the quote that holds it, checked by membership; a reading that finds
    none says why, and the member's Review asks (owner R1 b, 2026-10-28). It replaces
    the member's target as a typed amend does (#369); the words stay a User assertion."""

    event_type: Literal["InvestigationTargetRead"] = "InvestigationTargetRead"
    attempt_id: StrictStr
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    source: Literal["request", "answer"]
    source_event_id: StrictStr
    words: StrictStr = Field(min_length=1)
    quote: StrictStr | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    target: StatedTarget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stated_assembly: StatedAssembly | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reason: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def quoted_target(self) -> "InvestigationTargetRead":
        if (self.target is None) != (self.quote is None) or (
            (self.target is None) != (self.reason is not None)
        ):
            raise ValueError("a reading types a quoted target, or says why it has none")
        if self.target is not None:
            if self.target.region is not None:
                raise ValueError("a reading types an rsID, a site or a gene change")
            if not word_aligned_spans(self.words, self.quote):
                raise ValueError("a reading's quote stands in the words it reads")
            if self.target.rsid is not None and not RSID_RULE.matches(
                self.target.rsid, self.quote
            ):
                raise ValueError("a reading's rsID is one its quote states")
            site = self.target.position
            if site is not None and not site_in_quote(
                self.quote, site.contig, site.position, site.ref, site.alt
            ):
                raise ValueError("a reading's site is one its quote writes")
            change = self.target.gene_change
            if change is not None and not (
                change.typed and change.in_quote(self.quote)
            ):
                raise ValueError("a reading's gene change is one its quote writes")
        if self.stated_assembly is not None and (
            self.target is None
            or self.target.position is None
            or self.stated_assembly.cited_line != self.quote
        ):
            raise ValueError("a reading's version belongs to its quoted site")
        return self


class SequenceNamesRead(StrictFrozenContract):
    """A search of the sequence names a lookup retained (#467): a read receipt over
    retained bytes, never scientific evidence or authority."""

    event_type: Literal["SequenceNamesRead"] = "SequenceNamesRead"
    attempt_id: StrictStr
    collection_member_id: StrictStr
    tool_call_id: StrictStr
    result: SequenceNamesPage


class InvestigationProposalRecorded(StrictFrozenContract):
    event_type: Literal["InvestigationProposalRecorded"] = (
        "InvestigationProposalRecorded"
    )
    attempt_id: StrictStr
    output_lineages: tuple[StructuredOutputLineage, ...]
    question_response_event_id: StrictStr | None = None
    proposal: InvestigationProposal


QuestionKind = Literal["target", "version", "intent"]
"""What a member question asks the researcher (#472 Q4 a), typed by its asker, never
read from its words: which target (an rsID, a site), the genome version a site is on,
or other intent. The sample's own result is no kind: the Run reads it, after approval."""


class InvestigationQuestion(StrictFrozenContract):
    """Missing researcher intent, scoped before scientific normalization."""

    assertion_id: StrictStr
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    request: InvestigationRequestItem
    question: StrictStr = Field(min_length=1, max_length=2000)
    missing_information: StrictStr = Field(min_length=1, max_length=2000)
    asks_about: QuestionKind
    # Set when the intent check, not the Agent, asks: the refused output whose
    # retrieved target still differed from the request past the repair allowance.
    refused_output_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Set when the member's accepted finding ended with an issue: the failure
    # owner keys it, and the researcher approves, amends or rejects it.
    issue: ItemIssue | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The lookup result the question rests on and that result's statement, which
    # code copies: every trigger's question shows the source's words.
    cited_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    statement: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def one_asker(self) -> "InvestigationQuestion":
        if self.issue is not None and self.refused_output_event_id is not None:
            raise ValueError("a member question has one asker")
        if (self.cited_event_id is None) != (self.statement is None):
            raise ValueError("a member question cites a result with its statement")
        return self


def lookup_statement(payload: Mapping[str, Any]) -> str:
    """What a lookup fact says the source answered: its observation, no next step
    (#478; the next step is the Tool's model view's)."""
    return payload["result"]["assessment"]["rationale"]


def item_question(
    *,
    attempt_id: str,
    assertion_id: str,
    collection_member_id: str,
    revision_number: int,
    request: InvestigationRequestItem,
    question: str,
    missing_information: str,
    asks_about: QuestionKind,
    cited: Mapping[str, Any] | None = None,
    refused_output_event_id: str | None = None,
    issue: ItemIssue | None = None,
) -> InvestigationReviewRequested:
    """One item's question Review, the one shape every trigger asks (#479 C): the
    model's ask, the spent-allowance intent check and an unresolved finding's issue.
    ``cited`` is the lookup fact it rests on, whose statement code copies; the legal
    answers follow whether a finding is asked about (owner Q3 A)."""
    return InvestigationReviewRequested(
        review_policy=InvestigationReviewPolicy(),
        attempt_id=attempt_id,
        question=InvestigationQuestion(
            assertion_id=assertion_id,
            collection_member_id=collection_member_id,
            revision_number=revision_number,
            request=request,
            question=question,
            missing_information=missing_information,
            asks_about=asks_about,
            refused_output_event_id=refused_output_event_id,
            issue=issue,
            cited_event_id=None if cited is None else cited["event_id"],
            statement=None if cited is None else lookup_statement(cited["payload"]),
        ),
        legal_actions=QUESTION_ACTIONS if issue is None else ITEM_ISSUE_ACTIONS,
    )


QUESTION_ACTIONS = ("clarify", "reject")
"""A question with no finding: the researcher clarifies or rejects the item."""


RETRIEVED_TARGET_NOT_REQUESTED = "retrieved_target_not_requested"


# The intent refusal's next action, after its difference; the question the
# product asks itself reads the difference alone.
_ASK_INTENDED_TARGET = (
    "if the request means another target, ask the researcher which one with "
    "request_investigation_information"
)


def retrieved_target_question(errors: tuple[str, ...]) -> tuple[str, str]:
    """The question and missing information for one intent-check refusal, chosen by
    what its typed difference is in (#472 Q4 a): an rsID difference asks for the
    rsID, a location one (#369) for the stated position."""
    differs_in, difference = errors[1], errors[2]
    question = (
        f"The source returned a different rsID than this request names ({difference}), "
        "and no merge record of the source links them. Clarify with the rsID you "
        "mean, or reject this request."
        if differs_in == "rsid"
        else "The source places the retrieved target differently than this request "
        f"states it ({difference}). Clarify with the position, alleles or genome "
        "version you mean, as a typed target, or reject this request."
    )
    return (
        question,
        f"The retrieved target differs from the request ({difference}); only the "
        "researcher can say whether it is the intended target.",
    )


def answered_without_body(result: RefSnpToolResult, status_code: int) -> bool:
    """Whether a ``source_answered`` lookup result ended on an answer with
    ``status_code`` and zero body bytes, the one basis of a status-only
    ``listed_none`` decision (owner #480 Q3 a): its answering GET's retained byte
    count, never the audit sentence's trimmed words (review r1 ST1)."""
    origin = result.origin
    # The GET that ended the lookup: the last one after a record already read
    # (`following_sources`, review r2 SP1), else the result's own source; never any
    # other zero-byte GET of the result.
    location = result.location
    following = () if location == "not_applicable" else location.following_sources
    ending = following[-1] if following else result.source
    return (
        isinstance(origin, SourceAnswered)
        and origin.status_code == status_code == ending.status_code
        and ending.raw_response_byte_count == 0
    )


def deferred_finding(
    refsnp: RefSnpToolResult, refsnp_event_id: str
) -> InvestigationNotChecked | None:
    """A finding whose lookup found a position stated without alleles or an rsID: the
    item is deferred out of the Run, with that reason, and asks no question (owner
    #480 Q1, "if we cant find it its out"), in a first collection and a revision alike.
    ``None`` for any other finding."""
    origin = refsnp.origin
    if not isinstance(origin, PositionWithoutAlleles):
        return None
    return InvestigationNotChecked(
        event_id=refsnp_event_id,
        reason_code="position_without_alleles",
        reason=origin.words(),
    )


def deferred_findings(lineages, pairs):
    """Each member of a collection whose accepted finding is deferred
    (`deferred_finding`), by member id."""
    deferred = {}
    for lineage in lineages:
        pair = pairs.get(lineage.collection_member_id)
        if pair is None or lineage.accepted_output is None:
            continue
        outcome = deferred_finding(pair[0], lineage.accepted_output["refsnp_event_id"])
        if outcome is not None:
            deferred[lineage.collection_member_id] = outcome
    return deferred


def member_issue(
    refsnp: RefSnpToolResult,
    assembly: AssemblyReportToolResult | None,
    *,
    finding_event_id: str,
    refsnp_event_id: str,
    assembly_event_id: str | None,
) -> ItemIssue:
    """The issue an unresolved finding names: the source and verdict that decided it.

    The RefSNP result decides unless it resolved; then the assembly report does,
    as the output validator reads the finding's conflicts.
    """
    deciding_assembly = (
        refsnp.status in ADMISSIBLE_LOOKUP_STATUSES and assembly is not None
    )
    source, result = (
        ("assembly_report", assembly) if deciding_assembly else ("refsnp", refsnp)
    )
    return ItemIssue(
        key=item_issue_key(source, result.status),
        source=source,
        verdict=result.status,
        conflicts=result.conflicts,
        finding_event_id=finding_event_id,
        evidence_event_ids=(
            refsnp_event_id,
            *((assembly_event_id,) if assembly_event_id is not None else ()),
        ),
    )


def issue_question(
    issue: ItemIssue, request: InvestigationRequestItem
) -> tuple[str, str]:
    """The question and missing information of one member's issue Review: the
    finding first, then the request to approve removing the item (owner #479 Q1 A,
    C7), with what each answer does."""
    target = request.rsid or request.request_text
    return (
        f"The check of {target} ended with an issue ({issue.key}): "
        f"{' '.join(issue.conflicts)} Approve removing this item from the Run "
        "(reject): the other items go on. Or keep it as checked but not verified "
        "(approve), or amend the request, for example with a corrected rsID, to "
        "check it again (amend).",
        " ".join(issue.conflicts),
    )


class InvestigationIssueItem(StrictFrozenContract):
    """One requested SNP whose check ended with an issue, as the report lists it.

    Its finding was checked, never verified (#359): the issue keeps the verdict
    and evidence the failure owner keyed, and ``decision`` the researcher's answer,
    or ``unanswered`` when the Run ended first.
    """

    collection_member_id: StrictStr
    request: InvestigationRequestItem
    issue: ItemIssue
    request_event_id: StrictStr
    decision: Literal["approved", "rejected", "unanswered"]
    decided_by: StrictStr | None = None
    response_event_id: StrictStr | None = None

    @model_validator(mode="after")
    def exact_decision(self) -> "InvestigationIssueItem":
        if (self.decision == "unanswered") != (self.response_event_id is None) or (
            self.response_event_id is None
        ) != (self.decided_by is None):
            raise ValueError("an issue item's decision names its answer")
        return self


def investigation_issue_items(
    projection: "InvestigationProjection",
) -> tuple[InvestigationIssueItem, ...]:
    """Each issue finding not sent back by an amend, in Review order."""
    receipts = {item.request_event_id: item for item in projection.review_receipts}
    items = []
    for request in projection.review_requests:
        if request.question is None or request.question.issue is None:
            continue
        receipt = receipts.get(request.request_event_id)
        if receipt is not None and receipt.response.action == "amend":
            continue
        items.append(
            InvestigationIssueItem(
                collection_member_id=request.question.collection_member_id,
                request=request.question.request,
                issue=request.question.issue,
                request_event_id=request.request_event_id,
                decision="unanswered"
                if receipt is None
                else "approved"
                if receipt.response.action == "approve"
                else "rejected",
                decided_by=None if receipt is None else receipt.response.asserted_by,
                response_event_id=None
                if receipt is None
                else receipt.response_event_id,
            )
        )
    return tuple(items)


class InvestigationAttemptCompleted(StrictFrozenContract):
    event_type: Literal["InvestigationAttemptCompleted"] = (
        "InvestigationAttemptCompleted"
    )
    attempt_id: StrictStr
    question_request_event_id: StrictStr | None = None
    held_review_event_ids: tuple[StrictStr, ...] = ()


class InvestigationReviewRequested(StrictFrozenContract):
    review_policy: Annotated[
        InvestigationReviewPolicy, BeforeValidator(explicit_review_policy)
    ]
    event_type: Literal["InvestigationReviewRequested"] = "InvestigationReviewRequested"
    attempt_id: StrictStr | None = None
    proposal_event_id: StrictStr | None = None
    proposal_id: StrictStr | None = None
    question: InvestigationQuestion | None = None
    follows_response_event_id: StrictStr | None = None
    remaining_work_item_ids: tuple[StrictStr, ...] | None = None
    legal_actions: tuple[
        Literal["approve", "revise", "stop", "clarify", "reject", "amend"], ...
    ] = (
        "approve",
        "revise",
        "stop",
    )

    @model_validator(mode="after")
    def exact_scope(self):
        if self.question is None:
            if (
                self.proposal_event_id is None
                or self.proposal_id is None
                or self.legal_actions
                not in (("approve", "revise", "stop"), ("revise", "stop"))
            ):
                raise ValueError("Investigation proposal Review changed its scope")
        elif (
            self.proposal_event_id is not None
            or self.proposal_id is not None
            or self.attempt_id is None
            or self.legal_actions
            != (QUESTION_ACTIONS if self.question.issue is None else ITEM_ISSUE_ACTIONS)
        ):
            raise ValueError("Investigation question changed its scope")
        if (self.follows_response_event_id is None) != (
            self.remaining_work_item_ids is None
        ):
            raise ValueError(
                "Follow-up Review requires an exact answer and remaining scope"
            )
        if self.question is not None and self.follows_response_event_id is not None:
            raise ValueError("A member question cannot be a proposal follow-up")
        return self


def investigation_proposal_actions(proposal: InvestigationProposal) -> tuple[str, ...]:
    """An accounting-only collection cannot grant scientific Work approval."""
    if not proposal.work_items and all(
        member.rejection is not None
        or member.issue_approval is not None
        or member.not_checked is not None
        for member in proposal.request_members
    ):
        return ("revise", "stop")
    return ("approve", "revise", "stop")


class _TypedTargetAnswer(StrictFrozenContract):
    """A clarify or amend answer may state the corrected target, typed: it replaces
    the member's rsID and position, and its stated version, for that member's next
    lookup (#369). Its instruction stays a User assertion."""

    target: StatedTarget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stated_assembly: StatedAssembly | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def typed_version(self) -> "_TypedTargetAnswer":
        if self.target is not None and self.target.region is not None:
            raise ValueError("an answer states an rsID, a position or a gene change")
        change = None if self.target is None else self.target.gene_change
        if change is not None and not change.typed:
            # Code reads no change text: a gene change is typed.
            raise ValueError("an answer's gene change is typed")
        if self.stated_assembly is not None and (
            self.target is None or self.target.position is None
        ):
            raise ValueError("a stated version belongs to a stated position")
        return self


class ClarifyInvestigationResponse(_TypedTargetAnswer):
    action: Literal["clarify"]
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    instruction: StrictStr = Field(min_length=1, max_length=2000)
    asserted_by: StrictStr = Field(min_length=1)


class RejectInvestigationResponse(StrictFrozenContract):
    action: Literal["reject"]
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    reason: StrictStr = Field(min_length=1, max_length=2000)
    asserted_by: StrictStr = Field(min_length=1)


class ApproveInvestigationFindingResponse(StrictFrozenContract):
    """Approve one member's issue finding: checked, not verified (#359)."""

    action: Literal["approve"]
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    asserted_by: StrictStr = Field(min_length=1)


class AmendInvestigationFindingResponse(_TypedTargetAnswer):
    """Amend one member's issue finding: its instruction sends that member back."""

    action: Literal["amend"]
    collection_member_id: StrictStr
    revision_number: StrictInt = Field(gt=0)
    instruction: StrictStr = Field(min_length=1, max_length=2000)
    asserted_by: StrictStr = Field(min_length=1)


class ApproveInvestigationResponse(StrictFrozenContract):
    action: Literal["approve"]
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    work_item_ids: tuple[StrictStr, ...] | None = Field(default=None, min_length=1)

    @field_validator("work_item_ids")
    @classmethod
    def exact_selection(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Investigation approval repeated a Work item")
        return None if value is None else tuple(sorted(value))

    def selected_items(self, proposal: InvestigationProposal) -> tuple[WorkItem, ...]:
        selected = self.work_item_ids
        if selected is None:
            return proposal.work_items
        by_id = {item.work_item_id: item for item in proposal.work_items}
        if not set(selected) <= by_id.keys():
            raise ValueError("Investigation approval selected an unknown Work item")
        return tuple(
            item for item in proposal.work_items if item.work_item_id in selected
        )


class _ExistingInvestigationMemberOperation(StrictFrozenContract):
    work_item_id: StrictStr = Field(pattern=r"^work-item:[0-9a-f]{64}$")


class KeepInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["keep"]


class RetryInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["retry"]


class AddInvestigationMember(StrictFrozenContract):
    operation: Literal["add"]
    request: InvestigationRequestItem


class AmendInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["amend"]
    request: InvestigationRequestItem


class RemoveInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["remove"]


class DeferInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["defer"]


class ClarifyInvestigationMember(_ExistingInvestigationMemberOperation):
    operation: Literal["clarify"]
    request: InvestigationRequestItem


class AdjustInvestigationSearch(_ExistingInvestigationMemberOperation):
    operation: Literal["search_adjustment"]
    request: InvestigationRequestItem


class ClarifyInvestigationRequest(StrictFrozenContract):
    operation: Literal["clarify_request", "amend_request"]
    request_member_id: StrictStr = Field(pattern=r"^request-member:[0-9a-f]{64}$")
    request: InvestigationRequestItem


InvestigationRevisionOperation = Annotated[
    KeepInvestigationMember
    | RetryInvestigationMember
    | AddInvestigationMember
    | AmendInvestigationMember
    | RemoveInvestigationMember
    | DeferInvestigationMember
    | ClarifyInvestigationMember
    | AdjustInvestigationSearch
    | ClarifyInvestigationRequest,
    Field(discriminator="operation"),
]


class ReviseInvestigationResponse(StrictFrozenContract):
    action: Literal["revise"]
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    asserted_by: StrictStr
    operations: tuple[InvestigationRevisionOperation, ...] = Field(min_length=1)

    @field_validator("operations", mode="before")
    @classmethod
    def json_operations(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("asserted_by")
    @classmethod
    def canonical_asserted_by(cls, value: str) -> str:
        return _canonical_text(value, "asserted_by")


class InvestigationRevisionMember(StrictFrozenContract):
    member_id: StrictStr = Field(
        pattern=r"^investigation-revision-member:[0-9a-f]{64}$"
    )
    operation: Literal[
        "keep",
        "retry",
        "add",
        "amend",
        "clarify",
        "search_adjustment",
        "clarify_request",
        "amend_request",
    ]
    previous_request_member_id: StrictStr | None = Field(
        default=None,
        pattern=r"^request-member:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    previous_work_item_id: StrictStr | None = Field(
        default=None, pattern=r"^work-item:[0-9a-f]{64}$"
    )
    carried_work_item: WorkItem | None = None
    request: InvestigationRequestItem | None = None

    @model_validator(mode="after")
    def exact_member(self) -> "InvestigationRevisionMember":
        if self.operation == "keep":
            valid_shape = (
                self.carried_work_item is not None
                and self.previous_work_item_id == self.carried_work_item.work_item_id
                and self.request is None
            )
        elif self.operation in {"add", "clarify_request", "amend_request"}:
            valid_shape = (
                self.previous_work_item_id is None
                and self.carried_work_item is None
                and self.request is not None
            )
        else:
            valid_shape = (
                self.previous_work_item_id is not None
                and self.carried_work_item is None
                and self.request is not None
            )
        if not valid_shape:
            raise ValueError(
                "Investigation revision member changed its operation shape"
            )
        if (self.operation in {"clarify_request", "amend_request"}) != (
            self.previous_request_member_id is not None
        ):
            raise ValueError("request clarification changed its original member")
        material = self.model_dump(mode="json", exclude={"member_id"})
        if self.member_id != _digest("investigation-revision-member", material):
            raise ValueError("Investigation revision member identity changed")
        return self

    @classmethod
    def create(
        cls,
        *,
        operation: Literal[
            "keep",
            "retry",
            "add",
            "amend",
            "clarify",
            "search_adjustment",
            "clarify_request",
            "amend_request",
        ],
        previous_work_item_id: str | None = None,
        carried_work_item: WorkItem | None = None,
        request: InvestigationRequestItem | None = None,
        previous_request_member_id: str | None = None,
    ) -> "InvestigationRevisionMember":
        material = {
            "operation": operation,
            "previous_work_item_id": previous_work_item_id,
            "carried_work_item": carried_work_item,
            "request": request,
        }
        if previous_request_member_id is not None:
            material["previous_request_member_id"] = previous_request_member_id
        return cls(
            member_id=_digest("investigation-revision-member", material), **material
        )


class InvestigationCollectionRevision(StrictFrozenContract):
    revision_id: StrictStr = Field(pattern=r"^investigation-revision:[0-9a-f]{64}$")
    revision_number: StrictInt = Field(gt=1)
    supersedes_proposal_id: StrictStr = Field(
        pattern=r"^investigation-proposal:[0-9a-f]{64}$"
    )
    response_id: StrictStr = Field(
        pattern=r"^investigation-review-response:[0-9a-f]{64}$"
    )
    members: tuple[InvestigationRevisionMember, ...] = Field(min_length=1)
    removed_work_item_ids: tuple[StrictStr, ...]
    deferred_work_item_ids: tuple[StrictStr, ...]
    carried_request_members: tuple[InvestigationRequestMember, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def exact_revision(self) -> "InvestigationCollectionRevision":
        previous_ids = tuple(
            item.previous_work_item_id
            for item in self.members
            if item.previous_work_item_id is not None
        )
        disposition_ids = (*self.removed_work_item_ids, *self.deferred_work_item_ids)
        if (
            len(previous_ids) != len(set(previous_ids))
            or len(disposition_ids) != len(set(disposition_ids))
            or set(previous_ids).intersection(disposition_ids)
        ):
            raise ValueError("Investigation revision repeated contradictory membership")
        material = self.model_dump(mode="json", exclude={"revision_id"})
        if self.revision_id != _digest("investigation-revision", material):
            raise ValueError("Investigation revision identity changed")
        return self

    @property
    def work_requests(self) -> tuple[InvestigationRequestItem, ...]:
        return tuple(item.request for item in self.members if item.request is not None)

    @property
    def carried_work_items(self) -> tuple[WorkItem, ...]:
        return tuple(
            item.carried_work_item
            for item in self.members
            if item.carried_work_item is not None
        )


class InvestigationRevisionRefused(ValueError):
    def __init__(self, code: str, *work_item_ids: str) -> None:
        super().__init__(code)
        self.code = code
        self.work_item_ids = tuple(work_item_ids)


def _retry_request(work_item: WorkItem) -> InvestigationRequestItem:
    """The internal request that re-runs one Work item and re-selects its allele, or
    re-reads its region on the Run's sequence (#367)."""
    target = work_item.target
    if isinstance(target, RegionTarget):
        return InvestigationRequestItem(
            request_text=(
                f"Retry {work_item.intended_action} for {target.variant_text} "
                f"in sample {work_item.sample_id}."
            ),
            region=StatedRegion(
                contig=target.contig, start=target.start, end=target.end
            ),
            normalized_question_action=work_item.intended_action,
        )
    rsid = target.aliases[0] if target.aliases else None
    return InvestigationRequestItem(
        rsid=rsid,
        request_text=(
            f"Retry {work_item.intended_action} for "
            f"{rsid or 'the target'} in sample {work_item.sample_id}; "
            f"allele {target.variant_text}."
        ),
        # The Work item's own allele, typed: approval, not prose, re-selects it.
        position=StatedPosition(
            contig=target.contig,
            position=target.one_based_position,
            ref=target.ref,
            alt=target.alt,
        ),
        normalized_question_action=work_item.intended_action,
    )


def compile_investigation_collection_revision(
    *,
    proposal: InvestigationProposal,
    response_id: str,
    response: ReviseInvestigationResponse,
) -> InvestigationCollectionRevision:
    """Apply one complete patch while retaining every unchanged Work item exactly."""

    if response.proposal_id != proposal.proposal_id:
        raise InvestigationRevisionRefused("stale_proposal", response.proposal_id)
    request_operations = tuple(
        operation
        for operation in response.operations
        if isinstance(operation, ClarifyInvestigationRequest)
    )
    selected_request_ids = tuple(
        operation.request_member_id for operation in request_operations
    )
    if len(selected_request_ids) != len(set(selected_request_ids)):
        raise InvestigationRevisionRefused(
            "duplicate_request_membership", *selected_request_ids
        )
    # A request without Work is open to rework: unresolved, or an approved issue
    # finding, which a revision checks again.
    reworkable = {
        member.request_member_id
        for member in proposal.request_members
        if not member.work_item_ids and member.rejection is None
    }
    if not set(selected_request_ids) <= reworkable:
        raise InvestigationRevisionRefused(
            "unknown_unresolved_request", *selected_request_ids
        )
    existing = {item.work_item_id: item for item in proposal.work_items}
    referenced = tuple(
        operation.work_item_id
        for operation in response.operations
        if isinstance(operation, _ExistingInvestigationMemberOperation)
    )
    if len(referenced) != len(set(referenced)):
        raise InvestigationRevisionRefused("duplicate_membership", *referenced)
    unknown = tuple(item for item in referenced if item not in existing)
    if unknown:
        raise InvestigationRevisionRefused("unknown_membership", *unknown)
    missing = tuple(item for item in existing if item not in referenced)
    if missing and not request_operations:
        raise InvestigationRevisionRefused("missing_membership", *missing)
    by_id = {
        operation.work_item_id: operation
        for operation in response.operations
        if isinstance(operation, _ExistingInvestigationMemberOperation)
    }
    for work_item_id in missing:
        by_id[work_item_id] = KeepInvestigationMember(
            operation="keep", work_item_id=work_item_id
        )
    members: list[InvestigationRevisionMember] = []
    removed: list[str] = []
    deferred: list[str] = []
    for work_item in proposal.work_items:
        operation = by_id[work_item.work_item_id]
        if isinstance(operation, KeepInvestigationMember):
            members.append(
                InvestigationRevisionMember.create(
                    operation="keep",
                    previous_work_item_id=work_item.work_item_id,
                    carried_work_item=work_item,
                )
            )
        elif isinstance(operation, RemoveInvestigationMember):
            removed.append(work_item.work_item_id)
        elif isinstance(operation, DeferInvestigationMember):
            deferred.append(work_item.work_item_id)
        else:
            members.append(
                InvestigationRevisionMember.create(
                    operation=operation.operation,
                    previous_work_item_id=work_item.work_item_id,
                    request=(
                        _retry_request(work_item)
                        if isinstance(operation, RetryInvestigationMember)
                        else operation.request
                    ),
                )
            )
    for operation in response.operations:
        if isinstance(operation, AddInvestigationMember):
            members.append(
                InvestigationRevisionMember.create(
                    operation="add", request=operation.request
                )
            )
        elif isinstance(operation, ClarifyInvestigationRequest):
            members.append(
                InvestigationRevisionMember.create(
                    operation=operation.operation,
                    request=operation.request,
                    previous_request_member_id=operation.request_member_id,
                )
            )
    if not members:
        raise InvestigationRevisionRefused("empty_collection")
    material = {
        "revision_number": proposal.revision_number + 1,
        "supersedes_proposal_id": proposal.proposal_id,
        "response_id": response_id,
        "members": tuple(item.model_dump(mode="python") for item in members),
        "removed_work_item_ids": tuple(removed),
        "deferred_work_item_ids": tuple(deferred),
    }
    kept_work_ids = {
        item.previous_work_item_id for item in members if item.operation == "keep"
    }
    # A request carries forward with the Work items the patch keeps; a removed,
    # deferred or reworked candidate is accounted by the revision instead.
    carried_requests = tuple(
        member.model_copy(
            update={
                "work_item_ids": tuple(
                    item for item in member.work_item_ids if item in kept_work_ids
                )
            }
        )
        for member in proposal.request_members
        if member.request_member_id not in selected_request_ids
        and (
            not member.work_item_ids or kept_work_ids.intersection(member.work_item_ids)
        )
    )
    if carried_requests:
        material["carried_request_members"] = carried_requests
    try:
        return InvestigationCollectionRevision(
            revision_id=_digest("investigation-revision", material), **material
        )
    except ValidationError as error:
        raise InvestigationRevisionRefused("contradictory_membership") from error


class StopInvestigationResponse(StrictFrozenContract):
    action: Literal["stop"]
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    reason: StrictStr | None = None

    @field_validator("reason")
    @classmethod
    def canonical_reason(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_text(value, "reason")


def _response_kind(value: Any) -> str | None:
    """An approval names a proposal, or one member's issue finding (#359)."""
    if isinstance(value, Mapping):
        action, member = value.get("action"), "collection_member_id" in value
    else:
        action = getattr(value, "action", None)
        member = isinstance(value, ApproveInvestigationFindingResponse)
    return "approve_finding" if action == "approve" and member else action


InvestigationReviewResponse = Annotated[
    Annotated[ApproveInvestigationResponse, Tag("approve")]
    | Annotated[ApproveInvestigationFindingResponse, Tag("approve_finding")]
    | Annotated[AmendInvestigationFindingResponse, Tag("amend")]
    | Annotated[ReviseInvestigationResponse, Tag("revise")]
    | Annotated[StopInvestigationResponse, Tag("stop")]
    | Annotated[ClarifyInvestigationResponse, Tag("clarify")]
    | Annotated[RejectInvestigationResponse, Tag("reject")],
    Discriminator(_response_kind),
]


class InvestigationReviewResponded(StrictFrozenContract):
    event_type: Literal["InvestigationReviewResponded"] = "InvestigationReviewResponded"
    request_event_id: StrictStr
    response_id: StrictStr = Field(
        pattern=r"^investigation-review-response:[0-9a-f]{64}$"
    )
    response: InvestigationReviewResponse
    finalization_response_event_id: StrictStr | None = None

    @model_validator(mode="after")
    def exact_identity(self) -> "InvestigationReviewResponded":
        material = {
            "request_event_id": self.request_event_id,
            "response": self.response.model_dump(mode="json"),
        }
        if self.response_id != _digest("investigation-review-response", material):
            raise ValueError("Investigation Review response identity changed")
        return self

    @classmethod
    def create(
        cls, *, request_event_id: str, response: InvestigationReviewResponse
    ) -> "InvestigationReviewResponded":
        material = {
            "request_event_id": request_event_id,
            "response": response.model_dump(mode="json"),
        }
        return cls(
            request_event_id=request_event_id,
            response_id=_digest("investigation-review-response", material),
            response=response,
        )


class InvestigationCollectionRevisionRecorded(StrictFrozenContract):
    event_type: Literal["InvestigationCollectionRevisionRecorded"] = (
        "InvestigationCollectionRevisionRecorded"
    )
    response_event_id: StrictStr
    revision: InvestigationCollectionRevision


class InvestigationProposalCarriedForward(StrictFrozenContract):
    event_type: Literal["InvestigationProposalCarriedForward"] = (
        "InvestigationProposalCarriedForward"
    )
    revision_event_id: StrictStr
    proposal: InvestigationProposal


class InvestigationAssertionRecorded(StrictFrozenContract):
    event_type: Literal["InvestigationAssertionRecorded"] = (
        "InvestigationAssertionRecorded"
    )
    revision_event_id: StrictStr
    reason: Literal["revise"] = "revise"
    assertion: UserAssertion


class InvestigationAuthorityGranted(StrictFrozenContract):
    event_type: Literal["InvestigationAuthorityGranted"] = (
        "InvestigationAuthorityGranted"
    )
    request_event_id: StrictStr
    response_event_id: StrictStr
    authority: InvestigationAuthority


class InvestigationStopped(StrictFrozenContract):
    event_type: Literal["InvestigationStopped"] = "InvestigationStopped"
    request_event_id: StrictStr
    response_event_id: StrictStr
    proposal_id: StrictStr
    reason: StrictStr | None = None


class InvestigationAttemptFailed(StrictFrozenContract):
    event_type: Literal["InvestigationAttemptFailed"] = "InvestigationAttemptFailed"
    attempt_id: StrictStr
    code: Literal[
        "source_contact_failed",
        "model_contact_failed",
        "agent_run_failed",
    ]
    source_event_id: StrictStr
    reason: StrictStr | None = Field(
        default=None,
        pattern=(
            r"^[A-Za-z_][A-Za-z0-9_]*"
            r"(?:@[A-Za-z0-9_][A-Za-z0-9_.-]*:[1-9][0-9]*)?$"
        ),
    )

    @model_validator(mode="after")
    def exact_reason_scope(self) -> "InvestigationAttemptFailed":
        if self.code == "agent_run_failed" and self.reason is None:
            raise ValueError("agent_run_failed requires a safe initiating cause")
        if self.code != "agent_run_failed" and self.reason is not None:
            raise ValueError("only agent_run_failed may retain an initiating cause")
        return self


class InvestigationAttemptView(StrictFrozenContract):
    attempt_id: StrictStr
    instruction_identity: StrictStr
    instruction_artifact: ArtifactReference
    revision_id: StrictStr | None = None
    work_requests: tuple[InvestigationRequestItem, ...] = Field(min_length=1)
    continuation_policy: ContinuationPolicy
    effective_invocation_policy: AgentInvocationPolicy
    invocation_policy_id: StrictStr
    invocation_overrides: dict[str, Any]
    continuation_policy_id: StrictStr
    outcome: Literal[
        "question_pending",
        "work_completed",
        "review_pending",
        "failed",
        "model_availability_review_pending",
        "operational_review_pending",
        "cancelled",
        "effort_review_pending",
        "agent_work_review_pending",
    ]
    model_request_event_ids: tuple[StrictStr, ...]
    model_contact_event_ids: tuple[StrictStr, ...]
    model_response_event_ids: tuple[StrictStr, ...]
    model_failure_event_ids: tuple[StrictStr, ...]
    source_contact_event_ids: tuple[StrictStr, ...]
    source_failure_event_ids: tuple[StrictStr, ...]
    source_result_event_ids: tuple[StrictStr, ...]
    structured_output_lineages: tuple[StructuredOutputLineage, ...] = ()
    # The one member this Attempt planned, when it planned one.
    collection_member_id: StrictStr | None = None
    stop_reason: StrictStr | None = None
    failure_reason: StrictStr | None = None


class InvestigationReviewRequestView(StrictFrozenContract):
    review_policy: InvestigationReviewPolicy
    request_event_id: StrictStr
    proposal: InvestigationProposal | None = None
    question: InvestigationQuestion | None = None
    legal_actions: tuple[
        Literal["approve", "revise", "stop", "clarify", "reject", "amend"], ...
    ]
    pending: bool
    remaining_work_item_ids: tuple[StrictStr, ...] | None = None
    fact_slots: tuple[FactSlotStatus, ...] = ()
    # A proposal Review shows every requested SNP together, with the cases across
    # them that are not simple; a view, never authority.
    collection_view: CollectionView | None = None
    # A question or issue Review shows each sequence the model matched for its
    # member's written contig in the lookup the question cites (r2 SP2); a view,
    # never authority.
    model_matched: tuple[ModelMatchedName, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class InvestigationReviewReceipt(StrictFrozenContract):
    request_event_id: StrictStr
    response_event_id: StrictStr
    response_id: StrictStr
    response: InvestigationReviewResponse


class InvestigationProposalOutputLineageView(StrictFrozenContract):
    proposal_id: StrictStr = Field(pattern=r"^investigation-proposal:[0-9a-f]{64}$")
    output_lineages: tuple[StructuredOutputLineage, ...]


class InvestigationNotCheckedView(StrictFrozenContract):
    """One member recorded not checked, with its outcome (#367)."""

    collection_member_id: StrictStr
    outcome: InvestigationNotChecked


class InvestigationProjection(StrictFrozenContract):
    status: Literal[
        "not_started",
        "investigation_work_ready",
        "model_availability_review_pending",
        "operational_review_pending",
        "operational_review_direction_recorded",
        "operational_stopped",
        "effort_review_pending",
        "effort_review_direction_recorded",
        "effort_stopped",
        "agent_work_review_pending",
        "agent_work_review_direction_recorded",
        "agent_work_rejected",
        "model_availability_review_direction_recorded",
        "model_availability_stopped",
        "incomplete",
        "review_pending",
        "approved",
        "stopped",
        "failed",
    ]
    investigation_input: InvestigationInput | None = None
    assertions: tuple[UserAssertion, ...] = ()
    sources: tuple[RetrievedSource, ...] = ()
    source_failures: tuple[InvestigationSourceContactFailed, ...] = ()
    assessments: tuple[EvidenceAssessment, ...] = ()
    attempts: tuple[InvestigationAttemptView, ...] = ()
    proposals: tuple[InvestigationProposal, ...] = ()
    requirement_values: tuple[RequirementValue, ...] = ()
    fact_slots: tuple[FactSlotStatus, ...] = ()
    proposal_output_lineages: tuple[InvestigationProposalOutputLineageView, ...] = ()
    revisions: tuple[InvestigationCollectionRevision, ...] = ()
    review_requests: tuple[InvestigationReviewRequestView, ...] = ()
    review_receipts: tuple[InvestigationReviewReceipt, ...] = ()
    pending_request: InvestigationReviewRequestView | None = None
    authority: InvestigationAuthority | None = None
    authorities: tuple[InvestigationAuthority, ...] = ()
    pending_work_item_ids: tuple[StrictStr, ...] = ()
    eligible_member_ids: tuple[StrictStr, ...] = ()
    # The current revision's collection: each member's latest output lineage
    # among its own Attempts, in request order.
    collection_lineages: tuple[StructuredOutputLineage, ...] = ()
    # Each original member recorded not checked before any model request.
    not_checked: tuple["InvestigationNotCheckedView", ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    stop_reason: StrictStr | None = None
    model_contact_count: StrictInt = Field(ge=0)
    source_contact_count: StrictInt = Field(ge=0)


_EVENT_TYPES = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        InvestigationStarted,
        InvestigationAttemptCompleted,
        InvestigationAttemptBound,
        InvestigationSourceContactStarted,
        InvestigationSourceContactFailed,
        InvestigationSourceRetrieved,
        InvestigationSourceCarried,
        RefSnpToolResultRecorded,
        AssemblyReportToolResultRecorded,
        SequenceNamesRead,
        AssembliesSearched,
        InvestigationGeneChangePlaced,
        InvestigationGeneChangePicked,
        InvestigationTargetRead,
        InvestigationLookupDecided,
        InvestigationProposalRecorded,
        InvestigationReviewRequested,
        InvestigationReviewResponded,
        InvestigationCollectionRevisionRecorded,
        InvestigationProposalCarriedForward,
        InvestigationAssertionRecorded,
        InvestigationAuthorityGranted,
        InvestigationStopped,
        InvestigationAttemptFailed,
        InvestigationRequestNotChecked,
        InvestigationMemberContextRecorded,
    )
}


def validate_investigation_fact(event: Mapping[str, Any]) -> BaseModel | None:
    owner = _EVENT_TYPES.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def _uuid(value: str, name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be canonical")
    return value


def _contains(value: object, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains(item, expected) for item in value)
    return False


def asking_member_by_attempt(timeline) -> dict[str, str]:
    """Each Investigation Attempt a question closed, with the member that asked."""
    askers = {
        event["event_id"]: event["payload"]["question"]["collection_member_id"]
        for event in timeline
        if event.get("event_type") == "InvestigationReviewRequested"
        and event["payload"].get("question") is not None
    }
    return {
        event["payload"]["attempt_id"]: askers[
            event["payload"]["question_request_event_id"]
        ]
        for event in timeline
        if event.get("event_type") == "InvestigationAttemptCompleted"
        and event["payload"].get("question_request_event_id") in askers
    }


def without_unfinished_repair(lineages, member_id: str) -> tuple:
    """An answered question is new intent for its member: that member's unfinished
    output repair restarts; every other member keeps its lineage."""
    return tuple(
        lineage
        for lineage in lineages
        if lineage.collection_member_id != member_id
        or lineage.accepted_output is not None
    )


def _restarted_attempts(timeline, work_review) -> dict[str, str]:
    """Map each Investigation Attempt a continue answer started to the one it restarts."""
    requests = {
        r.request_event_id: r.attempt_id
        for r in work_review.requests
        if r.agent_role == "investigation_planner"
    }
    answers = {
        receipt.response_event_id: requests[receipt.request_event_id]
        for receipt in work_review.receipts
        if receipt.response_event_id in work_review.continued_response_event_ids
        and receipt.request_event_id in requests
    }
    return {
        event["payload"]["attempt_id"]: answers[answer_id]
        for event in timeline
        if event.get("event_type") == "AttemptStarted"
        and (answer_id := event["payload"].get("continuation_work_review_event_id"))
        in answers
    }


def _validate_retrieved_target_question(event, fact, timeline, policy) -> None:
    """The intent check's question cites the refusal whose failed turn reached its
    limit."""
    from bio_neb.agentic_journey.continuation_policy import (
        OUTPUT_KEY,
        failed_turns_spent,
    )

    question = fact.question
    prior = timeline[: timeline.index(event)]
    refused = next(
        (
            item
            for item in prior
            if item["event_id"] == question.refused_output_event_id
            and item["event_type"] == "AgentOutputValidated"
        ),
        None,
    )
    # The cited refusal is the one that spent the allowance: this member's latest
    # validation in this Attempt, never an earlier one.
    linked = next(
        (
            item["payload"]
            for item in reversed(prior)
            if item["event_type"] == "StructuredOutputValidationLinked"
            and item["payload"]["attempt_id"] == fact.attempt_id
            and item["payload"]["collection_member_id"] == question.collection_member_id
        ),
        None,
    )
    if (
        refused is None
        or linked is None
        or not failed_turns_spent(
            prior, attempt_id=fact.attempt_id, policy=policy, key=OUTPUT_KEY
        )
        or linked["validation_event_id"] != question.refused_output_event_id
        or refused["payload"]["attempt_id"] != fact.attempt_id
        or refused["payload"]["status"] != "invalid"
        or refused["payload"]["errors"][:1] != [RETRIEVED_TARGET_NOT_REQUESTED]
        or linked["collection_member_id"] != question.collection_member_id
        or (question.question, question.missing_information)
        != retrieved_target_question(tuple(refused["payload"]["errors"]))
        or question.asks_about != "target"
        or question.cited_event_id != refused["payload"]["output"]["refsnp_event_id"]
    ):
        raise ValueError("Investigation question changed its refused output")


def _amended_findings(timeline, parse=None) -> frozenset[str]:
    """The issue findings an amend answer in ``timeline`` sent back to the model.

    ``parse`` reads one event's Investigation fact; the fold passes its own.
    """
    parse = parse or validate_investigation_fact
    findings = {
        event["event_id"]: fact.question.issue.finding_event_id
        for event in timeline
        if event["event_type"] == "InvestigationReviewRequested"
        and (fact := parse(event)).question is not None
        and fact.question.issue is not None
    }
    return frozenset(
        findings[event["payload"]["request_event_id"]]
        for event in timeline
        if event["event_type"] == "InvestigationReviewResponded"
        and event["payload"]["response"]["action"] == "amend"
        and event["payload"]["request_event_id"] in findings
    )


def _validate_issue_question(event, fact, timeline) -> None:
    """An issue Review keys the member's latest accepted finding in its Attempt."""
    question = fact.question
    prior = timeline[: timeline.index(event)]
    linked = next(
        (
            item["payload"]
            for item in reversed(prior)
            if item["event_type"] == "StructuredOutputValidationLinked"
            and item["payload"]["attempt_id"] == fact.attempt_id
            and item["payload"]["collection_member_id"] == question.collection_member_id
        ),
        None,
    )
    finding = next(
        (
            item["payload"]
            for item in prior
            if item["event_id"] == question.issue.finding_event_id
            and item["event_type"] == "AgentOutputValidated"
        ),
        None,
    )
    if (
        linked is None
        or finding is None
        or linked["validation_event_id"] != question.issue.finding_event_id
        or finding["attempt_id"] != fact.attempt_id
        or finding["status"] != "valid"
        or finding["output"]["outcome"] != "unresolved"
    ):
        raise ValueError("Investigation issue changed its accepted finding")
    try:
        refsnp, assembly = select_investigation_sources(
            prior, question.collection_member_id, finding["output"]
        )
    except InvestigationProposalRefused as error:
        raise ValueError("Investigation issue changed its finding's sources") from error
    expected = member_issue(
        refsnp,
        assembly,
        finding_event_id=question.issue.finding_event_id,
        refsnp_event_id=finding["output"]["refsnp_event_id"],
        assembly_event_id=finding["output"]["assembly_event_id"],
    )
    if (
        expected != question.issue
        or (question.question, question.missing_information)
        != issue_question(expected, question.request)
        or question.asks_about != "intent"
        or question.cited_event_id != finding["output"]["refsnp_event_id"]
    ):
        raise ValueError("Investigation issue changed its verdict or evidence")


def _validate_question_citation(event, question, timeline) -> None:
    """A question's cited lookup is a lookup fact of its own member recorded before
    it, and its statement is that fact's own (#479 A + C, ADR 0049 membership)."""
    if question.cited_event_id is None:
        return
    cited = next(
        (
            item
            for item in timeline[: timeline.index(event)]
            if item["event_id"] == question.cited_event_id
        ),
        None,
    )
    if (
        cited is None
        or cited["event_type"] != "RefSnpToolResultRecorded"
        or cited["payload"]["collection_member_id"] != question.collection_member_id
        or question.statement != lookup_statement(cited["payload"])
    ):
        raise ValueError("Investigation question changed its cited statement")


def _result_contact_operations(fact) -> tuple[str, ...]:
    """The source contacts one lookup or assembly result made, in order: a carried
    planning GET it read is no contact of its own (#479 G)."""
    carried = {
        (answer.request_identity, answer.sha256)
        for answer in fact.source_answers
        if answer.cited_event_id is not None
    }
    if isinstance(fact, AssemblyReportToolResultRecorded):
        return () if carried else ("fetch_assembly_report",)
    if isinstance(fact, AssembliesSearched):
        return ("ncbi_taxon_assemblies",)
    if isinstance(fact, InvestigationGeneChangePlaced):
        # Each GET the placement made, in call order.
        return tuple(
            answer.operation
            for answer in fact.source_answers
            if answer.cited_event_id is None
        )
    return tuple(
        source.operation
        for source in _lookup_sources(fact.result)
        if (source.request_identity, source.raw_response_sha256) not in carried
    )


def _lookup_sources(result: RefSnpToolResult) -> tuple[RetrievedSource, ...]:
    """Every GET one lookup made, in order: a position's placement GETs (#369), the
    result's own source, then EVA's supporting GETs."""
    location = result.location
    return (
        *(() if location == "not_applicable" else location.supporting_sources),
        result.source,
        *(() if result.eva == "not_applicable" else result.eva.supporting_sources),
        *(() if location == "not_applicable" else location.following_sources),
    )


def _last_lookup_source_event(
    fact, source_events, timeline, result_event
) -> str | None:
    """The retained source a lookup result follows: the last of its GETs
    (`_lookup_sources`), each retained in order in the same Attempt before the result.
    The chain is read back from the result, so a lookup called again in one Attempt
    (#467: the model names a sequence and calls again) follows its own GETs, never an
    earlier call's equal ones. ``None`` when a GET is not retained so."""
    if not isinstance(fact, RefSnpToolResultRecorded):
        return fact.source_event_id
    sources = _lookup_sources(fact.result)
    if len(sources) == 1:
        return fact.source_event_id
    position = {event["event_id"]: index for index, event in enumerate(timeline)}
    bound = position[result_event["event_id"]]
    chain: list[str] = []
    for source in reversed(sources):
        retained = [
            event_id
            for event_id, (event, kept) in source_events.items()
            if kept == source
            and event["payload"]["attempt_id"] == fact.attempt_id
            and position[event_id] < bound
        ]
        if not retained:
            return None
        bound = position[max(retained, key=position.__getitem__)]
        chain.append(timeline[bound]["event_id"])
    if sources[0] == fact.result.source and chain[-1] != fact.source_event_id:
        return None
    return chain[0]


def _observed_member_call(event, attempt_id, timeline, by_id):
    """The loop's admitted call behind a Tool result."""
    from bio_neb.agentic_journey.tool_execution import tool_result_lineage

    return _member_call(
        tool_result_lineage(event["event_id"], timeline, by_id), attempt_id
    )


def _member_call(lineage, attempt_id):
    """A Tool result's call lineage, checked as its own Attempt's admitted request."""
    if lineage is None:
        raise ValueError("Investigation observation lacks one exact Tool receipt")
    if (
        lineage.started["attempt_id"] != attempt_id
        or lineage.request["event_type"] != "ModelRequestRecorded"
    ):
        raise ValueError("Investigation observation changed its exact admitted request")
    return lineage


def _validate_member_contexts(  # noqa: PLR0913 - one Attempt's recorded relationships
    owned, started, timeline, by_id, position, checked_once, parse
) -> None:
    """Each member's requests hold a recorded context, and each context points back
    into its own Attempt's basis and renders to exactly the part its first
    request sent.

    A member's first request in the Attempt reads the context this Attempt
    recorded, or one an Attempt it exactly continues recorded (a resumed
    transcript; ``member_context_fact``). Its bound work is its own Attempt's,
    and the Investigation, source results, answers and instruction it names were
    recorded before it. The pointers are rendered again with the Attempt's own
    instructions (owner, #444 Q3), once per fact for the reading owner
    (``checked_once``).
    """
    from bio_neb.agentic_journey.role_instructions import role_instructions
    from bio_neb.investigation.member_context import SOURCE_RESULTS

    members: set[str] = set()
    for event in owned:
        if (
            event["event_type"] == "ModelRequestRecorded"
            and event["payload"]["member_id"] not in members
        ):
            members.add(event["payload"]["member_id"])
            member_context_fact(event, timeline, parse)
    role, version, form = started["payload"]["instruction_identity"].split(":")
    for index, event in enumerate(owned):
        if event["event_type"] != "InvestigationMemberContextRecorded":
            continue
        fact = parse(event)
        here = position[event["event_id"]]

        def earlier(event_id, kinds, here=here):
            pointed = by_id.get(event_id)
            if (
                pointed is None
                or position[event_id] >= here
                or pointed["event_type"] not in kinds
            ):
                raise ValueError("member context points outside its Attempt's basis")
            return pointed

        bound = earlier(fact.bound_event_id, {"InvestigationAttemptBound"})
        if bound["payload"]["attempt_id"] != fact.attempt_id:
            raise ValueError("member context points outside its Attempt's basis")
        earlier(fact.started_event_id, {"InvestigationStarted"})
        for event_id in fact.source_result_event_ids:
            earlier(event_id, set(SOURCE_RESULTS))
        for event_id in fact.answer_event_ids:
            earlier(event_id, {"InvestigationReviewResponded"})
        if fact.instruction_event_id is not None:
            earlier(
                fact.instruction_event_id,
                {
                    "AgentWorkReviewResponded",
                    "ModelAvailabilityReviewResponded",
                    "AgentFailureReviewResponded",
                },
            )
        _validate_replaced_target(fact, bound, timeline[: position[event["event_id"]]])
        request = next(
            (
                later
                for later in owned[index + 1 :]
                if later["event_type"] == "ModelRequestRecorded"
                and later["payload"]["member_id"] == fact.collection_member_id
            ),
            None,
        )
        if request is None:
            # The member stopped before its first request: nothing was sent.
            continue

        def rendered(fact=fact):
            return rendered_text(
                fact, by_id, role_instructions(role, version=version, form=form), parse
            )

        text = (
            rendered()
            if checked_once is None
            else checked_once(("member_context", event["event_id"]), rendered)
        )
        messages = request["payload"]["compatible_adapter_evidence"]["request"][
            "messages"
        ]
        if len(messages) != 1 or not any(
            part["part_kind"] == "user-prompt" and part["content"] == text
            for part in messages[0]["parts"]
        ):
            raise ValueError("member context changed its rendered request")


def _validate_replaced_target(fact, bound, prefix) -> None:
    """A member context marks replaced exactly the item's last lookup before its
    target changed, and the event that changed it, as the one item history reads
    them (#479 Q2 A, G2.3)."""
    from bio_neb.investigation.item_history import item_history

    history = item_history(
        prefix,
        attempt_id=fact.attempt_id,
        member_id=fact.collection_member_id,
        revision_number=bound["payload"]["revision_number"],
        read_artifact=lambda _event: b"",
    )
    replaced = history.replaced_result
    if (fact.replaced_result_event_id, fact.replaced_by_event_id) != (
        (None, None)
        if replaced is None
        else (replaced["event_id"], history.target_event_id)
    ):
        raise ValueError("member context changed its replaced target")


def _same_location(location: "VariantLocationDetail", expected) -> bool:
    """A lookup read the member's own location. A position stated on another version
    and placed by no step is moved inside its lookup (#462 R2): its placed position
    rests on the lookup's own retained equivalence GET."""
    moved = (
        expected.placed is None
        and location.placed is not None
        and any(
            source.operation == "ncbi_spdi_equivalents"
            for source in location.supporting_sources
        )
    )
    return (
        location.stated,
        location.stated_assembly,
        expected.placed if moved else location.placed,
        location.region,
    ) == expected


def model_matched_names(location) -> tuple[ModelMatchedName, ...]:
    """Each sequence the model named for a contig that is not literally one of its
    names (#467, owner RC5 1a), typed (#475): every view that shows the match reads
    these."""
    if location == "not_applicable":
        return ()
    return tuple(
        ModelMatchedName(
            written=naming.written,
            sequence_name=naming.sequence_name,
            record=naming.record,
            assembly_accession=naming.assembly_accession,
        )
        for naming in location.sequence_namings
        if naming.match == "model_matched"
    )


def _cited_model_matched(timeline, question) -> tuple[ModelMatchedName, ...]:
    """The model-matched namings of the lookup a Review's question cites (#467 Spec
    r2 SP2): an issue's first evidence event, or the refused output's
    ``refsnp_event_id``, or the lookup the model's question cites (#479 A). A later exploration's
    match is not this finding's."""
    by_id = {event["event_id"]: event for event in timeline}
    if question.issue is not None:
        cited = question.issue.evidence_event_ids[0]
    elif question.refused_output_event_id is not None:
        refused = by_id.get(question.refused_output_event_id)
        output = None if refused is None else refused["payload"].get("output")
        cited = output.get("refsnp_event_id") if isinstance(output, dict) else None
    elif question.cited_event_id is not None:
        cited = question.cited_event_id
    else:
        return ()
    lookup = by_id.get(cited)
    if lookup is None or lookup["event_type"] != "RefSnpToolResultRecorded":
        return ()
    result = RefSnpToolResult.model_validate_json(
        json.dumps(lookup["payload"]["result"])
    )
    return model_matched_names(result.location)


def _options_as_called(arguments, fact) -> bool:
    """A lookup's call named its result's placement and EVA naming convention (#468,
    owner Q1 a and R3): the result records exactly what the call asked, and the call is
    recorded as called, a blank option absent (#476 G3)."""
    if not isinstance(fact, RefSnpToolResultRecorded):
        return True
    eva = fact.result.eva
    location = fact.result.location
    # A position lookup records the convention its call named, listing or not.
    naming = (
        location.contig_naming
        if location != "not_applicable"
        else eva.contig_naming
        if isinstance(eva, EvaLookupDetail)
        else None
    )
    return (
        arguments.get("placement") == fact.result.named_placement
        and arguments.get("contig_naming") == naming
    )


def _query_key(fact) -> str:
    """The key the result's own GET was asked with: the query, with the EVA naming
    convention the call asked as EVA's own option (#468, owner R3)."""
    if (
        isinstance(fact, RefSnpToolResultRecorded)
        and isinstance(fact.result.eva, EvaLookupDetail)
        and fact.result.source.operation == "eva_clustered_variant"
    ):
        return eva_record_key(fact.query, fact.result.eva.contig_naming)
    return fact.query


def _named_as_called(arguments, result) -> bool:
    """A position lookup's call named its result's sequences (#467): it takes only
    ``sequences``, ``version`` (#471), ``placement`` (bound by `_options_as_called`),
    ``listed_none``,
    ``source`` and ``variant_type`` (#469), which its result records as called (a
    region names no source and reads NCBI alone); each quoted naming is one of them."""
    location = result.location
    if set(arguments) - {
        "sequences",
        "version",
        "placement",
        "listed_none",
        "source",
        "variant_type",
        "contig_naming",
    }:
        return False
    # The version the call named is the one the lookup recorded; a literal naming of
    # the Run's own assembly is the lookup's, with no quote and no argument.
    named = location.version_naming
    version = arguments.get("version")
    if named is not None and (named.quote is not None) != (version is not None):
        return False
    if (
        version is not None
        and named is not None
        and (
            version["assembly_accession"],
            version["quote"],
        )
        != (named.assembly_accession, named.quote)
    ):
        return False
    if arguments.get("listed_none") != location.model_dump(mode="json").get(
        "listed_none"
    ):
        return False
    if arguments.get("variant_type") != location.variant_type:
        return False
    # A region's call names no source, omitted or null alike.
    if (arguments.get("source") or "ncbi") != result.lookup_source:
        return False
    called = {
        (item["assembly_accession"], item["record"], item["quote"])
        for item in arguments.get("sequences", ())
    }
    return all(
        (naming.assembly_accession, naming.record, naming.quote) in called
        for naming in location.sequence_namings
        if naming.quote is not None
    )


def _validate_assembly_search(event, fact, timeline, bound, by_id, parse) -> None:
    """Bind an assembly search to its exact admitted call, member and GET (#471, ADR
    0034): its Attempt and call, the member context its request held, the query the
    call named (taxid and NCBI's options) and its contact's request key."""
    from bio_neb.reference_discovery.reference_identity_tool import assembly_query

    lineage = _observed_member_call(event, fact.attempt_id, timeline, by_id)
    started, arguments = lineage.started, lineage.validation["arguments"]
    context = member_context_fact(lineage.request, timeline, parse)
    members = {
        investigation_output_member_id(item, ordinal)
        for ordinal, item in enumerate(bound.work_requests)
    }
    page = fact.result
    called = assembly_query(
        **{name: value for name, value in arguments.items() if name != "taxid"}
    )
    source = by_id[fact.source_event_id]
    contact = source_semantic_parent(source, by_id)
    while (
        contact is not None
        and contact["event_type"] == "InvestigationSourceRetrieved"
        and contact["payload"]["source"]["operation"]
        == source["payload"]["source"]["operation"]
    ):
        contact = source_semantic_parent(contact, by_id)
    run_taxid = next(
        item["payload"].get("run_taxid")
        for item in timeline
        if item["event_type"] == "InvestigationStarted"
    )
    if (
        fact.collection_member_id not in members
        # The Run organism's TaxId, bound by code.
        or page.ncbi_tax_id != run_taxid
        or context.collection_member_id != fact.collection_member_id
        or started["attempt_id"] != fact.attempt_id
        or started["tool_call_id"] != fact.tool_call_id
        or started["tool_name"] != "search_assemblies"
        or arguments.get("taxid") != page.ncbi_tax_id
        or called != page.query
        or source["payload"]["source"]["status_code"] != page.status_code
        or contact is None
        or contact["payload"].get("request_key")
        != f"{page.ncbi_tax_id}?{urllib.parse.urlencode(page.query.parameters())}"
    ):
        raise ValueError("assembly search changed its member, query or call identity")


def named_version(
    stated: StatedAssembly | None,
    searches: Iterable[AssemblySearchPage],
    lookups: Iterable[Mapping[str, Any]],
    run_accession: str,
    choice: VersionChoice,
) -> VersionNaming | str:
    """A naming of the stated version, checked by membership only (#471 Q3 a): the
    assembly is the Run's own, a row of a page the item's `search_assemblies` read, or
    an assembly a dbSNP record it read places the variant on, and the quote stands in
    the line that states the version; else the refusal's code. The naming is
    ``literal`` when the written version is one of that assembly's own names."""
    if stated is None:
        return "version_not_stated"
    names: dict[str, set[str]] = {}
    for page in searches:
        for row in page.rows:
            names.setdefault(row.assembly_accession, set()).update(
                name
                for name in (
                    row.assembly_name,
                    row.synonym,
                    row.assembly_accession,
                    row.paired_accession,
                )
                if name
            )
    for event in lookups:
        for placement in event["payload"]["result"].get("placements", ()):
            for assembly in (placement.get("dbsnp") or {}).get("assemblies", ()):
                names.setdefault(assembly["assembly_accession"], set()).update(
                    (assembly["assembly_name"], assembly["assembly_accession"])
                )
    # The Run's own assembly is one the lookup read: the model may name it directly.
    names.setdefault(run_accession, set()).add(run_accession)
    if choice.assembly_accession not in names:
        return "version_not_listed"
    # The quote stands in the line that states the version and holds the version
    # as written (ADR 0049: the quote holds the value's token).
    if not word_aligned_spans(stated.cited_line, choice.quote) or not (
        word_aligned_spans(choice.quote, stated.name)
    ):
        return "version_quote_ungrounded"
    return VersionNaming(
        written=stated.name,
        assembly_accession=choice.assembly_accession,
        quote=choice.quote,
        match=(
            "literal"
            if stated.name in names[choice.assembly_accession]
            else "model_matched"
        ),
    )


DECISION_ASKS = {
    "sequence": "sequence_unnamed",
    "version": "version_unnamed",
    "placement": "placement_unnamed",
    "listed_none": "source_answered",
}
"""The result status that asks for each lookup decision (#479 G2.2)."""

_LOOKUP_SOURCES = {"fetch_refsnp": "ncbi_dbsnp", "fetch_eva_variant": "eva"}


def applied_decisions(
    tool_name: str,
    rsid: str | None,
    decisions: Iterable[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """The item's decisions a lookup call reads, in record order (#479 G2.2): for
    ``locate_variant``, the latest sequence per assembly, version, placement and listed
    none a position result asked for; for an rsID lookup, the latest placement a result
    of the same Tool's source asked for that rsID. ``decisions`` are the item's
    decision events for its current target, ``results`` its lookup events by id."""
    latest: dict[Any, Mapping[str, Any]] = {}
    for event in decisions:
        decision = event["payload"]["decision"]
        asked = results.get(event["payload"]["refsnp_event_id"])
        if asked is None:
            continue
        asking = asked["payload"]["result"]
        positioned = asking["location"] != "not_applicable"
        if tool_name == "locate_variant" and positioned:
            key = (
                ("sequence", decision["choice"]["assembly_accession"])
                if decision["kind"] == "sequence"
                else decision["kind"]
            )
        elif (
            tool_name in _LOOKUP_SOURCES
            and not positioned
            and decision["kind"] == "placement"
            and asked["payload"]["query"] == rsid
            and asking["lookup_source"] == _LOOKUP_SOURCES[tool_name]
        ):
            key = "placement"
        else:
            continue
        latest[key] = event
    return tuple(
        sorted(latest.values(), key=lambda event: list(decisions).index(event))
    )


def decided_arguments(applied: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The lookup options the applied decisions supply, as a call once sent them."""
    arguments: dict[str, Any] = {}
    for event in applied:
        decision = event["payload"]["decision"]
        if decision["kind"] == "sequence":
            arguments.setdefault("sequences", []).append(decision["choice"])
        elif decision["kind"] == "version":
            arguments["version"] = decision["choice"]
        elif decision["kind"] == "placement":
            arguments["placement"] = decision["placement"]
        else:
            # Words quoted, or a body-less answer's status.
            arguments["listed_none"] = (
                decision["quote"]
                if "quote" in decision
                else {"status_code": decision["status_code"]}
            )
    return arguments


def _item_decisions(prefix, fact) -> tuple[list, dict]:
    """The item's decisions for its current target and its lookups, as of ``prefix``
    (#479 G2.3: the one history's rule)."""
    from bio_neb.investigation.item_history import item_read_attempts, target_change

    bound = next(
        event["payload"]
        for event in prefix
        if event["event_type"] == "InvestigationAttemptBound"
        and event["payload"]["attempt_id"] == fact.attempt_id
    )
    owned = set(item_read_attempts(prefix, fact.attempt_id))
    change = target_change(prefix, fact.collection_member_id, bound["revision_number"])
    start = -1 if change is None else prefix.index(change)
    decisions, results = [], {}
    for index, event in enumerate(prefix):
        payload = event["payload"]
        if (
            payload.get("attempt_id") not in owned
            or payload.get("collection_member_id") != fact.collection_member_id
        ):
            continue
        if event["event_type"] == "RefSnpToolResultRecorded":
            results[event["event_id"]] = event
        elif event["event_type"] == "InvestigationLookupDecided" and index > start:
            decisions.append(event)
    return decisions, results


def _effective_arguments(arguments, fact, timeline, event) -> dict:
    """A lookup call's arguments with the decisions its result read (#479 G2.2): the
    result names exactly the item's applicable decisions recorded before it."""
    if not isinstance(fact, RefSnpToolResultRecorded):
        return arguments
    decisions, results = _item_decisions(timeline[: timeline.index(event)], fact)
    applied = applied_decisions(
        _lookup_tool_name(fact, arguments), fact.query, decisions, results
    )
    if tuple(e["event_id"] for e in applied) != fact.decision_event_ids:
        raise ValueError("lookup result changed the decisions it read")
    return {**arguments, **decided_arguments(applied)}


def _lookup_tool_name(fact, arguments) -> str:
    """The Tool a lookup result answers: a position's ``locate_variant``, else the
    rsID lookup of its source."""
    if fact.result.location != "not_applicable":
        return "locate_variant"
    return "fetch_eva_variant" if fact.result.lookup_source == "eva" else "fetch_refsnp"


def _called_as_recorded(arguments, fact) -> bool:
    """A lookup result names exactly the arguments of the call it answers."""
    if (
        isinstance(fact, RefSnpToolResultRecorded)
        and fact.result.location != "not_applicable"
    ):
        # A position lookup's one argument names its sequences; its query is its
        # result's own GET.
        named = _named_as_called(arguments, fact.result)
    else:
        query_key = (
            "rsid"
            if isinstance(fact, RefSnpToolResultRecorded)
            else "assembly_accession"
        )
        named = arguments.get(query_key) == fact.query
    return named and _options_as_called(arguments, fact)


def _names_its_source(fact, source_entry) -> bool:
    """A lookup result names a retained source it copied: ``source_entry`` is the
    named event and its source, or None when no retained source has that id."""
    return (
        source_entry is not None
        and source_entry[0]["event_id"] == fact.source_event_id
        and source_entry[1] == fact.result.source
    )


def check_source_observation(prefix, candidate, owner) -> None:
    """Before a lookup result is appended, its own reopen check runs on it (owner,
    #476 C8): member, call, source and researcher request, read from the call's
    start, since the receipt naming the result exists only after it."""
    from bio_neb.agentic_journey.tool_execution import tool_call_lineage
    from bio_neb.investigation.item_history import item_ancestry

    # Not yet appended, so it holds no event id; one of its own for this read.
    candidate = {**candidate, "event_id": str(uuid.uuid4())}
    timeline = [*prefix, candidate]
    by_id = {event["event_id"]: event for event in timeline}
    fact = validate_investigation_fact(candidate)

    def parse(event):
        # The owner's one parse of each admitted fact, which its reopen shares.
        if event is candidate:
            return fact
        return owner.derived_once(
            ("investigation_fact", event["event_id"]),
            lambda: validate_investigation_fact(event),
        )

    bound = [
        parsed
        for event in prefix
        if event["event_type"] == "InvestigationAttemptBound"
        and isinstance(parsed := parse(event), InvestigationAttemptBound)
        and parsed.attempt_id == fact.attempt_id
    ]
    if len(bound) != 1:
        raise ValueError("Investigation observation lacks its Attempt's binding")
    source = by_id.get(fact.source_event_id)
    retrieved = None if source is None else parse(source)
    if not _names_its_source(
        fact,
        (source, retrieved.source)
        # A carried planning GET is the Tool's source as a retrieved one is.
        if isinstance(
            retrieved, (InvestigationSourceRetrieved, InvestigationSourceCarried)
        )
        else None,
    ):
        raise ValueError("typed Tool result changed its exact source lineage")
    _validate_source_observation(
        candidate,
        fact,
        timeline,
        bound[0],
        item_ancestry(timeline, fact.attempt_id),
        by_id,
        parse,
        lineage=tool_call_lineage(fact.attempt_id, fact.tool_call_id, prefix, by_id),
    )


def _validate_source_observation(  # noqa: PLR0913 - one observation's recorded relationships
    event, fact, timeline, bound, prior_ids, by_id, parse, *, lineage=None
):
    """Bind a domain observation to its exact admitted call and member prompt.

    ``lineage`` is the call's, when the caller read it before the observation's
    receipt exists (``check_source_observation``)."""
    lineage = (
        _observed_member_call(event, fact.attempt_id, timeline, by_id)
        if lineage is None
        else _member_call(lineage, fact.attempt_id)
    )
    started, validation = lineage.started, lineage.validation
    # The member context its request held, as recorded.
    context = member_context_fact(lineage.request, timeline, parse)
    members = {
        investigation_output_member_id(item, ordinal): item
        for ordinal, item in enumerate(bound.work_requests)
    }
    member = members.get(fact.collection_member_id)
    located = (
        isinstance(fact, RefSnpToolResultRecorded)
        and fact.result.location != "not_applicable"
    )
    source = by_id[fact.source_event_id]
    contact = source_semantic_parent(source, by_id)
    # The operation's earlier GETs sit between its contact and its last source.
    while (
        contact is not None
        and contact["event_type"] == "InvestigationSourceRetrieved"
        and contact["payload"]["source"]["operation"]
        == source["payload"]["source"]["operation"]
    ):
        contact = source_semantic_parent(contact, by_id)
    if (
        member is None
        or context.collection_member_id != fact.collection_member_id
        or member_request(context, by_id, parse).model_dump(mode="json")
        != member.model_dump(mode="json")
        or started["attempt_id"] != fact.attempt_id
        or started["tool_call_id"] != fact.tool_call_id
        or not _called_as_recorded(
            _effective_arguments(validation["arguments"], fact, timeline, event), fact
        )
        or (
            # A carried planning GET is the Run assembly's, asked by no contact of
            # this Tool's.
            _query_key(fact) != _run_accession(timeline)
            if source["event_type"] == "InvestigationSourceCarried"
            else contact is None
            or contact["payload"].get("request_key") != _query_key(fact)
        )
    ):
        raise ValueError(
            "source observation changed its member, query or call identity"
        )
    if isinstance(fact, RefSnpToolResultRecorded):
        answers = member_answers(
            timeline[: timeline.index(event)],
            member,
            fact.collection_member_id,
            bound.revision_number,
        )
        if fact.result.requested_rsid != effective_requested_rsid(member, answers):
            raise ValueError("RefSNP observation changed its researcher request")
        if located and not _same_location(
            fact.result.location, effective_location(member, answers)
        ):
            raise ValueError("position observation changed its stated position")
    else:
        basis = by_id.get(fact.refsnp_event_id)
        if (
            basis is None
            or basis["event_type"] != "RefSnpToolResultRecorded"
            or timeline.index(basis) >= timeline.index(event)
            or basis["payload"]["attempt_id"] not in {*prior_ids, fact.attempt_id}
            or basis["payload"]["collection_member_id"] != fact.collection_member_id
            or basis["payload"]["result"]["status"] not in ADMISSIBLE_LOOKUP_STATUSES
            or validation["arguments"].get("refsnp_event_id") != fact.refsnp_event_id
        ):
            raise ValueError("assembly observation changed its RefSNP basis")
        candidate = basis["payload"]["result"]["candidates"][0]
        if (
            fact.result.expected_assembly_accession != candidate["assembly_accession"]
            or fact.result.expected_assembly_name != candidate["assembly_name"]
        ):
            raise ValueError("assembly observation changed its expected candidate")


def _validate_member_scope(attempt, started, owned, lineages) -> None:
    """A member's Attempt holds only that member's work (#105).

    Its start and binding name the same member of its own requests, and every
    model request, source result and output lineage it retains is that member's.
    An Attempt that names no member is a whole-collection Attempt.
    """
    member = attempt.collection_member_id
    started_member = started["payload"].get("member_id")
    if member is None and started_member is None:
        return
    requested = {
        investigation_output_member_id(request, ordinal)
        for ordinal, request in enumerate(attempt.work_requests)
    }
    if (
        started_member != member
        or member not in requested
        or any(
            event.get("event_type") == "ModelRequestRecorded"
            and event["payload"].get("member_id") not in {None, member}
            for event in owned
        )
        or any(
            event.get("event_type")
            in {
                "RefSnpToolResultRecorded",
                "AssemblyReportToolResultRecorded",
                "AssembliesSearched",
                "InvestigationGeneChangePlaced",
                "InvestigationGeneChangePicked",
                "InvestigationLookupDecided",
            }
            and event["payload"].get("collection_member_id") != member
            for event in owned
        )
        or any(lineage.collection_member_id != member for lineage in lineages)
    ):
        raise ValueError("Investigation Attempt changed its member scope")


def _validate_carried_reference(prefix, start, investigation_input) -> None:
    """The start carries exactly planning's latest retained GETs that describe the
    Run's assembly, and the Run organism's species-level taxon, as read from the
    Journey before it (#479 G, owner Q4 A)."""
    from bio_neb.investigation.run_reference import (
        carried_reference_sources,
        run_species_taxid,
    )

    accession = investigation_input.assembly_accession
    if start.run_reference_sources != carried_reference_sources(
        prefix, accession
    ) or start.run_taxid != run_species_taxid(prefix, accession):
        raise ValueError("Investigation start changed its carried reference facts")


def _validate_carried_source(event, fact, start, by_id) -> None:
    """A carried GET is one the start carries, for the operation it stands in for,
    its bytes and answer exactly planning's (#479 G: a cited prior source)."""
    from bio_neb.investigation.run_reference import carried_operation

    cited = by_id.get(fact.cited_event_id)
    references = event.get("artifact_references", ())
    payload = None if cited is None else cited["payload"]
    if (
        fact.cited_event_id not in start.run_reference_sources
        or cited is None
        or cited["event_type"] != "ReferenceSourceRetrieved"
        or carried_operation(payload, start.investigation_input.assembly_accession)
        != fact.source.operation
        or (
            payload["request_identity"],
            payload["status_code"],
            payload["byte_count"],
            payload["sha256"],
        )
        != (
            fact.source.request_identity,
            fact.source.status_code,
            fact.source.raw_response_byte_count,
            fact.source.raw_response_sha256,
        )
        or not isinstance(references, list)
        or len(references) != 1
        or references[0].get("artifact_id") != fact.source.raw_artifact_id
    ):
        raise ValueError("Investigation carried source changed its cited GET")


def _answer_identities(answers) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            answer.operation,
            answer.request_identity,
            answer.status_code,
            answer.byte_count,
            answer.sha256,
            answer.cited_event_id,
        )
        for answer in answers
    )


def _call_sources(
    event, timeline_by_id, fact_by_event_id
) -> tuple[tuple[Any, ...], ...]:
    """A source Tool fact's own retained GETs, in call order: each retained source on
    its causal chain back to the call's first contact."""
    sources = []
    parent = source_semantic_parent(event, timeline_by_id)
    while parent is not None:
        fact = fact_by_event_id.get(parent["event_id"])
        if isinstance(fact, (InvestigationSourceRetrieved, InvestigationSourceCarried)):
            # A carried planning GET is shown as the prior source it cites.
            sources.append(
                (
                    fact.source.operation,
                    fact.source.request_identity,
                    fact.source.status_code,
                    fact.source.raw_response_byte_count,
                    fact.source.raw_response_sha256,
                    getattr(fact, "cited_event_id", None),
                )
            )
        elif not isinstance(fact, InvestigationSourceContactStarted):
            break
        parent = source_semantic_parent(parent, timeline_by_id)
    return tuple(reversed(sources))


def _prefix_reference(prefix):
    """The reference owner's checked view of the Investigation's planning prefix, the
    one interpretation the live request reader uses too (#459, ADR 0032); None for a
    Run whose prefix holds no verified reference work."""
    from bio_neb.reference_discovery.reference_journey import (
        project_reference_identity,
    )

    if not any(e["event_type"] == "ReferenceDescriptionRecorded" for e in prefix):
        return None
    return project_reference_identity(prefix)


def project_investigation(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    events: Iterable[Mapping[str, Any] | Any],
    *,
    checked_once: CheckedOnce | None = None,
) -> InvestigationProjection:
    """Fold the Investigation; ``checked_once`` is the reading owner's, if any."""
    timeline = [
        event.model_dump(mode="json") if isinstance(event, BaseModel) else dict(event)
        for event in events
    ]
    if any(
        event.get("event_type") == "SourceContactScopeStarted" for event in timeline
    ):
        from bio_neb.agentic_journey.source_contacts import source_contact_units

        source_contact_units(timeline)
    from bio_neb.agentic_journey.tool_execution import (
        model_view_of,
        tool_model_view,
        validate_attempt_closure,
        validate_observed_function_tool_role,
        validate_tool_failure_source,
    )

    validate_observed_function_tool_role(
        timeline, agent_role="investigation_planner", checked_once=checked_once
    )

    def parse(event):
        # An admitted event's fact never changes: its owner parses it once.
        if checked_once is None:
            return validate_investigation_fact(event)
        return checked_once(
            ("investigation_fact", event["event_id"]),
            lambda: validate_investigation_fact(event),
        )

    facts = [(event, fact) for event in timeline if (fact := parse(event)) is not None]
    for event, _fact in facts:
        _uuid(event.get("event_id"), "Investigation event_id")
    fact_by_event_id = {event["event_id"]: fact for event, fact in facts}
    bound = [fact for _, fact in facts if isinstance(fact, InvestigationAttemptBound)]
    # Each member context belongs to a bound Investigation Attempt.
    bound_ids = {attempt.attempt_id for attempt in bound}
    if any(
        isinstance(fact, InvestigationMemberContextRecorded)
        and fact.attempt_id not in bound_ids
        for _, fact in facts
    ):
        raise ValueError("member context names no bound Investigation Attempt")
    starts = [
        event for event in timeline if event.get("event_type") == "InvestigationStarted"
    ]
    if not bound and not starts:
        return InvestigationProjection(
            status="not_started", model_contact_count=0, source_contact_count=0
        )
    if len(starts) != 1:
        raise ValueError("Investigation requires one exact start")
    start_payload = starts[0].get("payload", {})
    start = InvestigationStarted.model_validate_json(
        canonical_json({"event_type": starts[0]["event_type"], **start_payload}),
        strict=True,
    )
    registered_selections = [
        retained_runtime_selections(event["payload"]["runtime_selections"])
        for event in timeline
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registered_selections) > 1:
        raise ValueError("Investigation has ambiguous configured policy authority")
    configured_selections = registered_selections[0] if registered_selections else None
    for attempt in bound:
        if (
            configured_selections is not None
            and attempt.runtime_selections != configured_selections
        ):
            raise ValueError("Investigation Attempt changed its configured policy")
    investigation_input = InvestigationInput.model_validate(
        start_payload.get("investigation_input")
    )
    _validate_carried_reference(
        timeline[: timeline.index(starts[0])], start, investigation_input
    )
    first_assertion = UserAssertion.model_validate_json(
        canonical_json(start_payload.get("assertion")), strict=True
    )
    from bio_neb.investigation.investigation_request import (
        chosen_suggestion_targets,
        project_captured_investigation_requests,
        run_investigation_request,
    )
    from bio_neb.parabricks.requirement_resolution import (
        project_requirement_resolution,
    )

    prefix = timeline[: timeline.index(starts[0])]
    origin = run_investigation_request(
        project_requirement_resolution(prefix)
        if any(
            event["event_type"] == "ProposalRequirementsIdentified" for event in prefix
        )
        else None,
        project_captured_investigation_requests(prefix),
        suggested=chosen_suggestion_targets(_prefix_reference(prefix)),
    )
    if origin is not None or len(first_assertion.predecessor_ids) > 2:
        requests = first_assertion.requests or (
            InvestigationRequestItem(
                request_text=first_assertion.request_text,
                rsid=first_assertion.rsid,
            ),
        )
        if (
            origin is None
            or origin.request
            != InvestigationRequest(
                asserted_by=first_assertion.asserted_by, items=requests
            )
            or first_assertion.research_goals != origin.research_goals
            or first_assertion.predecessor_ids
            != tuple(
                dict.fromkeys(
                    (
                        investigation_input.input_id,
                        start.continuation_id,
                        origin.source_event_id,
                        *origin.source_event_ids,
                    )
                )
            )
        ):
            raise ValueError("Investigation start changed its request origin")
    not_checked = _not_checked_members(timeline, facts, first_assertion, bound)
    if not bound:
        every = len(not_checked) == len(_assertion_requests(first_assertion))
        return InvestigationProjection(
            # Every target not checked: nothing is planned and nothing asks.
            status="incomplete" if every else "failed",
            investigation_input=investigation_input,
            assertions=(first_assertion,),
            not_checked=tuple(not_checked.values()),
            stop_reason=None if every else "attempt_not_bound",
            model_contact_count=0,
            source_contact_count=0,
        )
    first = bound[0]
    if (
        first.investigation_input_id != investigation_input.input_id
        or first.assertion_id != first_assertion.assertion_id
    ):
        raise ValueError("Investigation Attempt changed its input basis")
    assertions = [first_assertion]
    assertions.extend(
        fact.assertion
        for _, fact in facts
        if isinstance(fact, InvestigationAssertionRecorded)
    )
    assertion_by_id = {item.assertion_id: item for item in assertions}
    if len(assertion_by_id) != len(assertions):
        raise ValueError("Investigation repeated a User assertion identity")
    sources = [
        fact.source
        for _, fact in facts
        if isinstance(fact, InvestigationSourceRetrieved)
    ]
    source_events: dict[str, tuple[Mapping[str, Any], RetrievedSource]] = {}
    source_timeline = {event["event_id"]: event for event in timeline}
    for event, fact in facts:
        if not isinstance(fact, InvestigationSourceRetrieved):
            continue
        references = event.get("artifact_references", ())
        semantic_parent = source_semantic_parent(event, source_timeline)
        parent = (
            fact_by_event_id.get(semantic_parent["event_id"])
            if semantic_parent
            else None
        )
        if (
            # A retained source follows its contact, or the same operation's earlier
            # GET (an assembly's plain report before its any-version one).
            not isinstance(
                parent,
                (InvestigationSourceContactStarted, InvestigationSourceRetrieved),
            )
            or parent.attempt_id != fact.attempt_id
            or (
                parent.operation
                if isinstance(parent, InvestigationSourceContactStarted)
                else parent.source.operation
            )
            != fact.source.operation
            or not isinstance(references, list)
            or len(references) != 1
            or references[0].get("artifact_id") != fact.source.raw_artifact_id
            or references[0].get("byte_count") != fact.source.raw_response_byte_count
            or references[0].get("source_event_id") != event["event_id"]
            or references[0].get("media_type")
            not in _RETAINED_MEDIA.get(fact.source.operation, {"application/json"})
        ):
            raise ValueError("Investigation source changed its raw evidence artifact")
        source_events[event["event_id"]] = (event, fact.source)
    for event, fact in facts:
        if isinstance(fact, InvestigationSourceCarried):
            _validate_carried_source(event, fact, start, source_timeline)
            source_events[event["event_id"]] = (event, fact.source)
    for event, fact in facts:
        if isinstance(
            fact,
            (
                RefSnpToolResultRecorded,
                AssemblyReportToolResultRecorded,
                AssembliesSearched,
                InvestigationGeneChangePlaced,
            ),
        ) and _answer_identities(fact.source_answers) != _call_sources(
            event, source_timeline, fact_by_event_id
        ):
            # A Tool shows exactly its own GETs: each retained source on its causal
            # chain back to the call's first contact, in call order.
            raise ValueError("Investigation Tool result changed its source answers")
    for event, fact in facts:
        if not isinstance(fact, InvestigationSourceContactFailed):
            continue
        semantic_parent = source_semantic_parent(event, source_timeline)
        # A fault reading an answer the contact retained follows that answer (the
        # answer's bytes stay beside the fault).
        while (
            semantic_parent is not None
            and isinstance(
                retrieved := fact_by_event_id.get(semantic_parent["event_id"]),
                InvestigationSourceRetrieved,
            )
            and retrieved.attempt_id == fact.attempt_id
            and retrieved.source.operation == fact.operation
        ):
            semantic_parent = source_semantic_parent(semantic_parent, source_timeline)
        started = (
            fact_by_event_id.get(semantic_parent["event_id"])
            if semantic_parent
            else None
        )
        if (
            not isinstance(started, InvestigationSourceContactStarted)
            or started.attempt_id != fact.attempt_id
            or started.operation != fact.operation
            or started.request_key != fact.request_key
        ):
            raise ValueError("Investigation source failure changed its contact")
    assessments = [
        fact.result.assessment
        for _, fact in facts
        if isinstance(
            fact, (RefSnpToolResultRecorded, AssemblyReportToolResultRecorded)
        )
    ]
    proposals = [
        fact.proposal
        for _, fact in facts
        if isinstance(
            fact, (InvestigationProposalRecorded, InvestigationProposalCarriedForward)
        )
    ]
    revisions = [
        fact.revision
        for _, fact in facts
        if isinstance(fact, InvestigationCollectionRevisionRecorded)
    ]
    retained_revision_by_id = {item.revision_id: item for item in revisions}
    if len(retained_revision_by_id) != len(revisions):
        raise ValueError("Investigation repeated a collection revision identity")
    bound_by_attempt: dict[str, InvestigationAttemptBound] = {}
    for attempt in bound:
        if attempt.attempt_id in bound_by_attempt:
            raise ValueError("Investigation repeated an Attempt binding")
        bound_by_attempt[attempt.attempt_id] = attempt
    from bio_neb.agentic_journey.agent_work_review import (
        agent_work_status,
        validate_agent_work_history,
        validate_agent_work_terminal,
    )
    from bio_neb.agentic_journey.effort import (
        effort_status,
        validate_effort_history,
        validate_effort_terminal,
    )

    validate_effort_history(timeline, checked_once=checked_once)
    work_review = validate_agent_work_history(timeline, checked_once=checked_once)

    availability = project_model_availability_review(
        timeline, checked_once=checked_once
    )
    prior_ids = {
        a.attempt_id: prior_model_attempt_ids(timeline, a.attempt_id) for a in bound
    }
    for attempt in bound:
        for prior_id in prior_ids[attempt.attempt_id]:
            prior = bound_by_attempt.get(prior_id)
            if prior is None or prior.model_dump(
                exclude={"attempt_id", "continued_attempt_id"}
            ) != attempt.model_dump(exclude={"attempt_id", "continued_attempt_id"}):
                raise ValueError(
                    "provider continuation changed Investigation work binding"
                )
    from bio_neb.investigation.item_history import item_ancestry

    for attempt in bound:
        if attempt.continued_attempt_id is not None:
            prior = bound_by_attempt.get(attempt.continued_attempt_id)
            if (
                prior is None
                or prior == attempt
                or bound.index(prior) >= bound.index(attempt)
                or prior.model_dump(exclude={"attempt_id", "continued_attempt_id"})
                != attempt.model_dump(exclude={"attempt_id", "continued_attempt_id"})
                or not (
                    any(
                        isinstance(fact, InvestigationAttemptCompleted)
                        and fact.attempt_id == prior.attempt_id
                        for _, fact in facts
                    )
                    # An interrupted Attempt's completed work continues too.
                    or any(
                        event["event_type"] == "AttemptCancelled"
                        and event["payload"]["attempt_id"] == prior.attempt_id
                        for event in timeline
                    )
                )
            ):
                raise ValueError(
                    "Investigation scoped continuation changed its prior work"
                )
    from bio_neb.investigation.investigation_questions import latest_collection_frontier

    by_id = {event["event_id"]: event for event in timeline}
    for attempt in bound:
        start = next(
            event
            for event in timeline
            if event["event_type"] == "AttemptStarted"
            and event["payload"]["attempt_id"] == attempt.attempt_id
        )
        direction_id = start["payload"].get(
            "continuation_availability_review_event_id"
        ) or start["payload"].get("continuation_operational_review_event_id")
        if direction_id is not None:
            direction = by_id[direction_id]
            request = by_id[direction["payload"]["request_event_id"]]["payload"]
            frontier = latest_collection_frontier(
                timeline[: timeline.index(start)], request["attempt_id"]
            )
            if attempt.continued_attempt_id != frontier["attempt_id"]:
                raise ValueError(
                    "operational recovery lost the latest collection frontier"
                )
    prior_ids = {a.attempt_id: item_ancestry(timeline, a.attempt_id) for a in bound}
    restarted = _restarted_attempts(timeline, work_review)
    asked = asking_member_by_attempt(timeline)
    structured_lineages_by_attempt = {}
    for attempt in bound:
        previous_ids = prior_ids[attempt.attempt_id]
        inherited = (
            structured_lineages_by_attempt[previous_ids[-1]] if previous_ids else ()
        )
        if previous_ids and previous_ids[-1] in asked:
            inherited = without_unfinished_repair(inherited, asked[previous_ids[-1]])
        if attempt.attempt_id in restarted:
            inherited = tuple(
                lineage for lineage in inherited if lineage.accepted_output is not None
            )
        start_event = next(
            e
            for e in timeline
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["attempt_id"] == attempt.attempt_id
        )
        started = start_event["payload"]
        # An amended issue finding restarts its member's whole lineage.
        amended = _amended_findings(timeline[: timeline.index(start_event)], parse)
        inherited = tuple(
            lineage
            for lineage in inherited
            if lineage.accepted_output is None
            or lineage.validation_attempts[-1].validation_event_id not in amended
        )
        direction_id = started.get(
            "continuation_availability_review_event_id"
        ) or started.get("continuation_operational_review_event_id")
        if direction_id is not None:
            direction = next(
                e["payload"] for e in timeline if e["event_id"] == direction_id
            )
            if direction["response"]["action"] == "amend":
                request = next(
                    e["payload"]
                    for e in timeline
                    if e["event_id"] == direction["request_event_id"]
                )
                inherited = tuple(
                    lineage
                    for lineage in inherited
                    if lineage.collection_member_id != request["member_id"]
                    or lineage.accepted_output is not None
                )
        lineages = project_structured_output_lineages(
            timeline,
            attempt_id=attempt.attempt_id,
            agent_role="investigation_planner",
            tool_name="compile_proposal",
            max_retries=attempt.continuation_policy.most_failed_turns,
            retained_lineages=inherited,
        )
        structured_lineages_by_attempt[attempt.attempt_id] = tuple(
            lineage
            for ordinal, request in enumerate(attempt.work_requests)
            for lineage in lineages
            if lineage.collection_member_id
            == investigation_output_member_id(request, ordinal)
        )
    bound_index = {
        fact.attempt_id: timeline.index(event)
        for event, fact in facts
        if isinstance(fact, InvestigationAttemptBound)
    }

    def revision_key(attempt: InvestigationAttemptBound):
        return (attempt.assertion_id, attempt.revision_number, attempt.revision_id)

    def collection_lineages(key, before: int | None = None):
        """Each member's latest lineage among the revision's Attempts (#105).

        Members are planned in their own Attempts; the collection is the latest
        lineage each member's own work recorded, in request order.
        """
        latest = {}
        requests_of_key = ()
        for attempt in bound:
            if revision_key(attempt) != key or (
                before is not None and bound_index[attempt.attempt_id] >= before
            ):
                continue
            requests_of_key = attempt.work_requests
            for lineage in structured_lineages_by_attempt[attempt.attempt_id]:
                latest[lineage.collection_member_id] = lineage
        return tuple(
            latest[member]
            for ordinal, request in enumerate(requests_of_key)
            if (member := investigation_output_member_id(request, ordinal)) in latest
        )

    for attempt_id, stopped_id in restarted.items():
        attempt, stopped = (
            bound_by_attempt.get(attempt_id),
            bound_by_attempt.get(stopped_id),
        )
        if (
            attempt is None
            or stopped is None
            or attempt.model_dump(exclude={"attempt_id"})
            != stopped.model_dump(exclude={"attempt_id"})
        ):
            raise ValueError("Agent work continuation changed Investigation work")
    roots = [
        a
        for a in bound
        if not prior_ids[a.attempt_id] and a.attempt_id not in restarted
    ]
    # Each member's first Attempt in a revision is a root; revisions never
    # go back.
    if [item.revision_number for item in roots] != sorted(
        item.revision_number for item in roots
    ):
        raise ValueError("Investigation Attempt revisions are not increasing")

    def origin(attempt_id: str) -> str:
        # Provider/effort continuations and Agent work restarts keep their origin's work.
        while True:
            if prior_ids[attempt_id]:
                attempt_id = prior_ids[attempt_id][0]
            elif attempt_id in restarted:
                attempt_id = restarted[attempt_id]
            else:
                return attempt_id

    for attempt in bound:
        assertion = assertion_by_id.get(attempt.assertion_id)
        revision = (
            retained_revision_by_id.get(attempt.revision_id)
            if attempt.revision_id is not None
            else None
        )
        # Every member's first-revision Attempt plans the original request.
        initial = attempt.revision_id is None and (
            origin(attempt.attempt_id) == bound[0].attempt_id
            or attempt.assertion_id == bound[0].assertion_id
        )
        if (
            assertion is None
            or attempt.investigation_input_id != investigation_input.input_id
            or attempt.work_requests
            != (
                assertion.requests
                if assertion.requests
                else (
                    InvestigationRequestItem(
                        request_text=assertion.request_text, rsid=assertion.rsid
                    ),
                )
            )
            or (
                initial
                and (
                    attempt.revision_number != 1
                    or attempt.supersedes_proposal_id is not None
                    or assertion != first_assertion
                )
            )
            or (
                not initial
                and (
                    revision is None
                    or attempt.revision_number != revision.revision_number
                    or attempt.supersedes_proposal_id != revision.supersedes_proposal_id
                    or assertion.requests != revision.work_requests
                )
            )
        ):
            raise ValueError("Investigation Attempt changed its revision basis")
    requests: list[InvestigationReviewRequestView] = []
    request_events: dict[str, InvestigationReviewRequested] = {}
    responses: dict[str, tuple[str, InvestigationReviewResponded]] = {}
    approved_by_request: dict[str, set[str]] = {}
    closed_requests: set[str] = set()
    revised_requests: set[str] = set()
    retained_approved_ids: set[str] = set()
    proposal_by_id = {item.proposal_id: item for item in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("Investigation repeated a proposal identity")
    for event, fact in facts:
        if isinstance(fact, InvestigationReviewRequested):
            if fact.question is not None:
                question = fact.question
                attempt = bound_by_attempt.get(fact.attempt_id)
                members = (
                    {}
                    if attempt is None
                    else {
                        investigation_output_member_id(item, ordinal): item
                        for ordinal, item in enumerate(attempt.work_requests)
                    }
                )
                if (
                    attempt is None
                    or question.assertion_id != attempt.assertion_id
                    or question.revision_number != attempt.revision_number
                    or members.get(question.collection_member_id) != question.request
                    # A member's Attempt asks only about its own member.
                    or attempt.collection_member_id
                    not in {None, question.collection_member_id}
                ):
                    raise ValueError(
                        "Investigation question changed its exact member or revision"
                    )
                _validate_question_citation(event, question, timeline)
                if question.refused_output_event_id is not None:
                    _validate_retrieved_target_question(
                        event, fact, timeline, attempt.continuation_policy
                    )
                    request_events[event["event_id"]] = fact
                    continue
                if question.issue is not None:
                    _validate_issue_question(event, fact, timeline)
                    request_events[event["event_id"]] = fact
                    continue
                lineage = _observed_member_call(
                    event, fact.attempt_id, timeline, source_timeline
                )
                started, validation = lineage.started, lineage.validation
                context = member_context_fact(lineage.request, timeline, parse)
                if (
                    context.collection_member_id != question.collection_member_id
                    or member_request(context, source_timeline, parse).model_dump(
                        mode="json"
                    )
                    != question.request.model_dump(mode="json")
                    or started["attempt_id"] != fact.attempt_id
                    or validation["tool_name"] != "request_investigation_information"
                    or validation["arguments"]
                    != {
                        "question": question.question,
                        "missing_information": question.missing_information,
                        "asks_about": question.asks_about,
                        # A cited lookup is the call's own.
                        **(
                            {}
                            if question.cited_event_id is None
                            else {"refsnp_event_id": question.cited_event_id}
                        ),
                    }
                ):
                    raise ValueError(
                        "Investigation question changed its exact admitted call"
                    )
                request_events[event["event_id"]] = fact
                continue
            proposal = proposal_by_id.get(fact.proposal_id)
            if proposal is None:
                raise ValueError("Investigation Review changed its proposal")
            expected_actions = investigation_proposal_actions(proposal)
            if fact.follows_response_event_id is not None:
                prior = responses.get(fact.follows_response_event_id)
                remaining = tuple(
                    sorted(
                        item.work_item_id
                        for item in proposal.work_items
                        if item.work_item_id not in retained_approved_ids
                    )
                )
                if (
                    prior is None
                    or not isinstance(prior[1].response, ApproveInvestigationResponse)
                    or prior[1].response.proposal_id != proposal.proposal_id
                    or fact.remaining_work_item_ids != remaining
                    or any(
                        r.follows_response_event_id == fact.follows_response_event_id
                        for r in request_events.values()
                    )
                ):
                    raise ValueError(
                        "Follow-up Investigation Review changed its exact remaining scope"
                    )
                expected_actions = (
                    ("approve", "revise", "stop") if remaining else ("revise", "stop")
                )
            if fact.legal_actions != expected_actions:
                raise ValueError("Investigation Review changed its legal actions")
            request_events[event["event_id"]] = fact
            carried = retained_approved_ids & {
                item.work_item_id for item in proposal.work_items
            }
            approved_by_request[event["event_id"]] = carried
            if (
                proposal.work_items
                and not proposal.unresolved_request_member_ids
                and carried == {item.work_item_id for item in proposal.work_items}
            ):
                closed_requests.add(event["event_id"])
        elif isinstance(fact, InvestigationReviewResponded):
            request_id = fact.request_event_id
            request = request_events.get(request_id)
            if request is not None and request.question is not None:
                from bio_neb.investigation.investigation_questions import (
                    question_answer,
                )

                question_views = tuple(
                    InvestigationReviewRequestView(
                        request_event_id=identity,
                        review_policy=prior.review_policy,
                        question=prior.question,
                        legal_actions=prior.legal_actions,
                        pending=identity not in closed_requests,
                    )
                    for identity, prior in request_events.items()
                    if prior.question is not None
                )
                prior_receipts = tuple(
                    InvestigationReviewReceipt(
                        request_event_id=answer.request_event_id,
                        response_event_id=identity,
                        response_id=answer.response_id,
                        response=answer.response,
                    )
                    for identity, answer in responses.values()
                )
                _, retained = question_answer(
                    question_views, prior_receipts, request_id, fact.response
                )
                if retained is not None:
                    raise ValueError("Investigation question repeated its answer")
                closed_requests.add(request_id)
                responses[event["event_id"]] = (event["event_id"], fact)
                continue
            revising = isinstance(fact.response, ReviseInvestigationResponse)
            if (
                request is None
                or request_id in revised_requests
                or any(
                    answer.request_event_id == request_id
                    for _, answer in responses.values()
                )
                or (request_id in closed_requests and not revising)
            ):
                raise ValueError("Investigation Review has repeated responses")
            selected = approved_by_request.setdefault(request_id, set())
            proposal = proposal_by_id[request.proposal_id]
            if isinstance(fact.response, ApproveInvestigationResponse):
                ids = {
                    item.work_item_id for item in fact.response.selected_items(proposal)
                }
                if request.remaining_work_item_ids is not None and not ids <= set(
                    request.remaining_work_item_ids
                ):
                    raise ValueError(
                        "Investigation Review widened its remaining selection"
                    )
                if selected & ids:
                    raise ValueError(
                        "Investigation Review repeated an approved selection"
                    )
                selected.update(ids)
                retained_approved_ids.update(ids)
                closed_requests.add(request_id)
            else:
                closed_requests.add(request_id)
                if revising:
                    if request_id != next(reversed(request_events)):
                        raise ValueError("Investigation revision targets stale Review")
                    revised_requests.add(request_id)
            responses[event["event_id"]] = (event["event_id"], fact)
    for event_id, request in request_events.items():
        if request.question is not None:
            requests.append(
                InvestigationReviewRequestView(
                    request_event_id=event_id,
                    review_policy=request.review_policy,
                    question=request.question,
                    legal_actions=request.legal_actions,
                    pending=event_id not in closed_requests,
                    model_matched=_cited_model_matched(timeline, request.question),
                )
            )
            continue
        proposal = proposal_by_id[request.proposal_id]
        requests.append(
            InvestigationReviewRequestView(
                request_event_id=event_id,
                review_policy=request.review_policy,
                proposal=proposal,
                legal_actions=request.legal_actions,
                pending=event_id not in closed_requests,
                remaining_work_item_ids=request.remaining_work_item_ids,
                fact_slots=_investigation_identity_values(proposal, timeline)[1],
                collection_view=collection_view(proposal),
            )
        )
    pending = [request for request in requests if request.pending]
    if len([r for r in pending if r.question is None]) > 1:
        raise ValueError("Investigation has more than one pending proposal Review")
    receipts = tuple(
        InvestigationReviewReceipt(
            request_event_id=fact.request_event_id,
            response_event_id=event_id,
            response_id=fact.response_id,
            response=fact.response,
        )
        for request_id, (event_id, fact) in responses.items()
    )
    response_events_by_id: dict[
        str, tuple[Mapping[str, Any], InvestigationReviewResponded]
    ] = {}
    for response_event_id, response in responses.values():
        request_id = response.request_event_id
        request = request_events.get(request_id)
        response_event = next(
            event for event in timeline if event.get("event_id") == response_event_id
        )
        response_parents = [request_id]
        if response.finalization_response_event_id is not None:
            from bio_neb.reporting.finalization import (
                validate_finalization_review_fact,
            )

            source = next(
                (
                    event
                    for event in timeline[: timeline.index(response_event)]
                    if event.get("event_id") == response.finalization_response_event_id
                ),
                None,
            )
            final_review = (
                validate_finalization_review_fact(source)
                if source is not None
                else None
            )
            if (
                final_review is None
                or final_review.return_to_investigation is None
                or final_review.receipt.response.action != "correct"
                or final_review.receipt.response.correction.investigation_revision
                != response.response
            ):
                raise ValueError(
                    "Investigation rework changed Finalization correction lineage"
                )
            response_parents.append(response.finalization_response_event_id)
        if (
            request is None
            or (
                request.question is None
                and response.response.proposal_id != request.proposal_id
            )
            or response_event.get("parent_ids") != response_parents
        ):
            raise ValueError("Investigation Review response changed its request")
        response_events_by_id[response_event_id] = (response_event, response)
    revision_by_id: dict[str, InvestigationCollectionRevision] = {}
    for event, fact in facts:
        if not isinstance(fact, InvestigationCollectionRevisionRecorded):
            continue
        response_entry = response_events_by_id.get(fact.response_event_id)
        response = response_entry[1].response if response_entry is not None else None
        request = (
            request_events.get(response_entry[1].request_event_id)
            if response_entry is not None
            else None
        )
        previous = proposal_by_id.get(fact.revision.supersedes_proposal_id)
        try:
            expected_revision = (
                compile_investigation_collection_revision(
                    proposal=previous,
                    response_id=response_entry[1].response_id,
                    response=response,
                )
                if previous is not None
                and isinstance(response, ReviseInvestigationResponse)
                else None
            )
        except InvestigationRevisionRefused as error:
            raise ValueError(
                "Investigation revision changed its collection patch"
            ) from error
        if (
            expected_revision is None
            or fact.revision != expected_revision
            or request is None
            or request.proposal_id != fact.revision.supersedes_proposal_id
            or event.get("parent_ids") != [fact.response_event_id]
            or fact.revision.revision_id in revision_by_id
        ):
            raise ValueError("Investigation revision changed its Review lineage")
        revision_by_id[fact.revision.revision_id] = fact.revision
    assertion_records = [
        (event, fact)
        for event, fact in facts
        if isinstance(fact, InvestigationAssertionRecorded)
    ]
    assertion_revision_event_ids = tuple(
        fact.revision_event_id for _, fact in assertion_records
    )
    if len(assertion_revision_event_ids) != len(set(assertion_revision_event_ids)):
        raise ValueError("Investigation revision repeated its User assertion")
    for event, fact in assertion_records:
        revision_entry = next(
            (
                (revision_event, recorded)
                for revision_event, recorded in facts
                if isinstance(recorded, InvestigationCollectionRevisionRecorded)
                and revision_event.get("event_id") == fact.revision_event_id
            ),
            None,
        )
        response_entry = (
            response_events_by_id.get(revision_entry[1].response_event_id)
            if revision_entry is not None
            else None
        )
        response = response_entry[1].response if response_entry is not None else None
        request_id = (
            response_entry[1].request_event_id if response_entry is not None else None
        )
        revision = revision_entry[1].revision if revision_entry is not None else None
        proposal_id = response.proposal_id if response is not None else None
        if (
            not isinstance(response, ReviseInvestigationResponse)
            or revision is None
            or fact.reason != "revise"
            or event.get("parent_ids") != [fact.revision_event_id]
            or fact.assertion.predecessor_ids
            != (proposal_id, request_id, revision.revision_id)
            or fact.assertion.asserted_by != response.asserted_by
            or fact.assertion.requests != revision.work_requests
        ):
            raise ValueError("Investigation assertion changed its Review lineage")
    assertion_revision_event_id_set = set(assertion_revision_event_ids)
    for revision_event, recorded in facts:
        if not isinstance(recorded, InvestigationCollectionRevisionRecorded):
            continue
        has_assertion = revision_event["event_id"] in assertion_revision_event_id_set
        if has_assertion != bool(recorded.revision.work_requests):
            raise ValueError("Investigation revision changed its assertion cardinality")
    authorities = [
        fact.authority
        for _, fact in facts
        if isinstance(fact, InvestigationAuthorityGranted)
    ]
    granted_responses: set[str] = set()
    for event, fact in facts:
        if isinstance(fact, InvestigationAuthorityGranted):
            response_entry = response_events_by_id.get(fact.response_event_id)
            response = response_entry[1] if response_entry is not None else None
            proposal = (
                proposal_by_id.get(response.response.proposal_id)
                if response is not None
                and isinstance(response.response, ApproveInvestigationResponse)
                else None
            )
            if (
                response is None
                or response.request_event_id != fact.request_event_id
                or proposal is None
                or event.get("parent_ids") != [fact.response_event_id]
                or fact.authority.proposal_id != proposal.proposal_id
                or fact.authority.review_response_id != response.response_id
                or fact.authority.authorized_work_items
                != response.response.selected_items(proposal)
                or fact.response_event_id in granted_responses
                or fact.authority.predecessor_ids
                != (proposal.proposal_id, response.response_id)
            ):
                raise ValueError("Investigation authority changed its exact approval")
            granted_responses.add(fact.response_event_id)
        elif isinstance(fact, InvestigationStopped):
            response_entry = response_events_by_id.get(fact.response_event_id)
            response = response_entry[1] if response_entry is not None else None
            if (
                response is None
                or response.request_event_id != fact.request_event_id
                or not isinstance(response.response, StopInvestigationResponse)
                or response.response.proposal_id != fact.proposal_id
                or response.response.reason != fact.reason
                or event.get("parent_ids") != [fact.response_event_id]
            ):
                raise ValueError("Investigation stop changed its exact Review response")
    if granted_responses != {
        event_id
        for event_id, response in responses.values()
        if isinstance(response.response, ApproveInvestigationResponse)
    }:
        raise ValueError("Investigation approval is missing its exact grant")
    source_results = [
        (event, fact)
        for event, fact in facts
        if isinstance(
            fact, (RefSnpToolResultRecorded, AssemblyReportToolResultRecorded)
        )
    ]
    _validate_target_reads(timeline, facts, bound_by_attempt)
    _validate_lookup_decisions(timeline, facts, source_timeline)
    _validate_gene_changes(
        timeline, facts, bound_by_attempt, source_events, source_timeline
    )
    for search_event, fact in facts:
        if not isinstance(fact, AssembliesSearched):
            continue
        source_entry = source_events.get(fact.source_event_id)
        if (
            fact.attempt_id not in bound_by_attempt
            or source_entry is None
            or source_entry[0]["event_id"] != fact.source_event_id
            or search_event.get("parent_ids") != [fact.source_event_id]
        ):
            raise ValueError("assembly search changed its exact source lineage")
        _validate_assembly_search(
            search_event,
            fact,
            timeline,
            bound_by_attempt[fact.attempt_id],
            source_timeline,
            parse,
        )
    result_by_event_id: dict[str, RefSnpToolResult | AssemblyReportToolResult] = {}
    for result_event, fact in source_results:
        if (
            fact.attempt_id not in bound_by_attempt
            or not _names_its_source(fact, source_events.get(fact.source_event_id))
            or result_event.get("parent_ids")
            != [_last_lookup_source_event(fact, source_events, timeline, result_event)]
        ):
            raise ValueError("typed Tool result changed its exact source lineage")
        _validate_source_observation(
            result_event,
            fact,
            timeline,
            bound_by_attempt[fact.attempt_id],
            prior_ids[fact.attempt_id],
            source_timeline,
            parse,
        )
        result_by_event_id[result_event["event_id"]] = fact.result
        later_requests = [
            event
            for event in timeline[timeline.index(result_event) + 1 :]
            if event.get("event_type") == "ModelRequestRecorded"
            and event.get("payload", {}).get("attempt_id") == fact.attempt_id
        ]
        stops = [
            event
            for event in timeline[timeline.index(result_event) + 1 :]
            if event.get("event_type") == "AttemptStopped"
            and event.get("payload", {}).get("attempt_id") == fact.attempt_id
        ]
        questions = [
            event
            for event in timeline[timeline.index(result_event) + 1 :]
            if event.get("event_type") == "InvestigationAttemptCompleted"
            and event["payload"]["attempt_id"] == fact.attempt_id
            and event["payload"].get("question_request_event_id") is not None
        ]
        cancellations = [
            event
            for event in timeline[timeline.index(result_event) + 1 :]
            if event.get("event_type") == "AttemptCancelled"
            and event["payload"]["attempt_id"] == fact.attempt_id
        ]
        # A question, or an interrupt at the turn boundary, closes this response
        # without another model request. Its ordered Tool receipts and Attempt
        # closure retain the preceding result.
        if not later_requests and (questions or cancellations):
            continue
        if not later_requests and len(stops) != 1:
            raise ValueError("typed Tool result was not visible before the next action")
        if not later_requests and stops[0]["payload"]["code"] != "effort_exhausted":
            # The response that ran the Tool also ended the Attempt for the Agent
            # work review; no continuation transcript retains the result.
            validate_agent_work_terminal(timeline, stops[0])
        elif not later_requests:
            validate_effort_terminal(timeline, stops[0])
            review = next(
                event
                for event in timeline
                if event["event_id"] == stops[0]["payload"]["source_event_id"]
            )
            # The Tool the result's own admitted call named (fetch_refsnp,
            # fetch_eva_variant, locate_variant or fetch_assembly_report).
            tool_name = _observed_member_call(
                result_event, fact.attempt_id, timeline, by_id
            ).started["tool_name"]
            pending_returns = [
                part
                for message in review["payload"]["messages"]
                for part in message.get("parts", ())
                if part.get("part_kind") == "tool-return"
                and part.get("tool_name") == tool_name
                and part.get("tool_call_id") == fact.tool_call_id
                and part.get("content")
                == shown_to_model(
                    tool_model_view(model_view_of(tool_name), result_event)
                )
            ]
            if (
                timeline.index(review) <= timeline.index(result_event)
                or len(pending_returns) != 1
            ):
                raise ValueError(
                    "typed Tool result was not retained in its effort continuation"
                )
        if later_requests and not _contains(
            later_requests[0].get("payload", {}), result_event["event_id"]
        ):
            raise ValueError("typed Tool result was not visible to the Agent")
    proposal_event_by_id: dict[str, Mapping[str, Any]] = {}
    proposal_output_lineages: list[InvestigationProposalOutputLineageView] = []
    for event, fact in facts:
        if not isinstance(fact, InvestigationProposalRecorded):
            continue
        attempt = bound_by_attempt.get(fact.attempt_id)
        from bio_neb.investigation.investigation_questions import (
            answer_target_asserters,
            issue_approvals,
            member_stop_rejections,
            request_rejections,
        )

        prior_receipts = tuple(
            r
            for r in receipts
            if next(
                i
                for i, e in enumerate(timeline)
                if e["event_id"] == r.response_event_id
            )
            < timeline.index(event)
        )
        rejections = request_rejections(
            requests,
            prior_receipts,
            fact.proposal.assertion_id,
            fact.proposal.revision_number,
        ) | member_stop_rejections(
            timeline[: timeline.index(event)],
            fact.proposal.assertion_id,
            fact.proposal.revision_number,
            checked_once,
        )
        approvals = issue_approvals(
            requests,
            prior_receipts,
            fact.proposal.assertion_id,
            fact.proposal.revision_number,
        )
        # A revision carries its not-checked members; the first collection
        # accounts for them itself.
        accounted = (
            {view.collection_member_id: view.outcome for view in not_checked.values()}
            if fact.proposal.revision_number == 1
            else {}
        )
        # The collection the proposal closes: each member's latest lineage among
        # its revision's Attempts before the proposal.
        revision_attempts = (
            [
                a.attempt_id
                for a in bound
                if revision_key(a) == revision_key(attempt)
                and bound_index[a.attempt_id] < timeline.index(event)
            ]
            if attempt is not None
            else []
        )
        output_lineages = tuple(
            lineage
            for lineage in (
                collection_lineages(revision_key(attempt), timeline.index(event))
                if attempt is not None
                else ()
            )
            if lineage.collection_member_id not in rejections
        )
        owned_attempts = {
            identity
            for attempt_id in revision_attempts
            for identity in (*prior_ids[attempt_id], attempt_id)
        }
        owned_events = [
            result_event
            for result_event, _ in source_results
            if result_event["payload"]["attempt_id"] in owned_attempts
            and timeline.index(result_event) < timeline.index(event)
        ]
        source_pairs = {
            lineage.collection_member_id: select_investigation_sources(
                owned_events, lineage.collection_member_id, lineage.accepted_output
            )
            for lineage in output_lineages
            if lineage.accepted_output is not None
        }
        paired_results = (
            tuple(
                source_pairs.get(investigation_output_member_id(request, ordinal))
                for ordinal, request in enumerate(attempt.work_requests)
            )
            if attempt is not None
            else ()
        )
        owned_results = [
            result
            for pair in paired_results
            if pair is not None
            for result in pair
            if result is not None
        ]
        attempt_revision = (
            retained_revision_by_id.get(attempt.revision_id)
            if attempt is not None and attempt.revision_id is not None
            else None
        )
        assertion = assertion_by_id.get(fact.proposal.assertion_id)
        previous = (
            proposal_by_id.get(fact.proposal.supersedes_proposal_id)
            if fact.proposal.supersedes_proposal_id is not None
            else None
        )
        expected_proposal = None
        if (
            paired_results
            and all(
                pair is None
                or (
                    isinstance(pair[0], RefSnpToolResult)
                    and (
                        pair[1] is None or isinstance(pair[1], AssemblyReportToolResult)
                    )
                )
                for pair in paired_results
            )
            and assertion is not None
        ):
            answered = answer_target_asserters(
                requests,
                prior_receipts,
                _assertion_requests(assertion),
                fact.proposal.assertion_id,
                fact.proposal.revision_number,
                tuple(
                    item["payload"]
                    for item in timeline[: timeline.index(event)]
                    if item["event_type"] == "InvestigationTargetRead"
                ),
            )
            # A finding deferred out of the Run is accounted like a member not
            # checked, its lookup kept as its outcome's event.
            deferred = deferred_findings(output_lineages, source_pairs)
            expected_proposal = (
                compile_investigation_collection_proposal(
                    investigation_input=investigation_input,
                    assertion=assertion,
                    source_results=paired_results,
                    rejections=rejections,
                    issue_approvals=approvals,
                    not_checked={**accounted, **deferred},
                    target_asserters=answered,
                    revision_number=fact.proposal.revision_number,
                    supersedes_proposal_id=fact.proposal.supersedes_proposal_id,
                    predecessor_ids=tuple(
                        dict.fromkeys(result.result_id for result in owned_results)
                    )
                    + tuple(r.response_event_id for r in rejections.values())
                    + tuple(a.response_event_id for a in approvals.values())
                    + tuple(n.event_id for n in accounted.values())
                    + tuple(n.event_id for n in deferred.values()),
                )
                if attempt_revision is None
                else compile_contact_investigation_revision_proposal(
                    investigation_input=investigation_input,
                    assertion=assertion,
                    previous_proposal=previous,
                    revision=attempt_revision,
                    source_results=paired_results,
                    rejections=rejections,
                    issue_approvals=approvals,
                    deferred=deferred,
                    target_asserters=answered,
                )
                if previous is not None
                else None
            )
        if fact.proposal.unresolved_request_member_ids:
            # Every unresolved finding asks its own issue Review, and the proposal
            # waits until each is answered.
            raise ValueError(
                "Investigation proposal was recorded before its issues were answered"
            )
        proposal_index = proposals.index(fact.proposal)
        expected_superseded = (
            proposals[proposal_index - 1].proposal_id if proposal_index else None
        )
        if (
            attempt is None
            or (not output_lineages and not rejections)
            or fact.output_lineages != output_lineages
            or len(output_lineages) + len(rejections) + len(accounted)
            != len(attempt.work_requests)
            or tuple(item.collection_member_id for item in output_lineages)
            != tuple(
                investigation_output_member_id(item, ordinal)
                for ordinal, item in enumerate(attempt.work_requests)
                if investigation_output_member_id(item, ordinal)
                not in {*rejections, *accounted}
            )
            or any(item.accepted_output is None for item in output_lineages)
            or event.get("parent_ids")
            != (
                [fact.question_response_event_id]
                if fact.question_response_event_id is not None
                else [
                    max(
                        (
                            item.validation_attempts[-1].validation_link_event_id
                            for item in output_lineages
                        ),
                        key=lambda identity: next(
                            i
                            for i, e in enumerate(timeline)
                            if e["event_id"] == identity
                        ),
                    )
                ]
            )
            or (
                fact.question_response_event_id is not None
                and fact.question_response_event_id
                not in {
                    *(r.response_event_id for r in rejections.values()),
                    *(a.response_event_id for a in approvals.values()),
                }
            )
            or fact.proposal.investigation_input_id != attempt.investigation_input_id
            or fact.proposal.assertion_id != attempt.assertion_id
            or fact.proposal.revision_number != attempt.revision_number
            or fact.proposal.supersedes_proposal_id != attempt.supersedes_proposal_id
            or fact.proposal != expected_proposal
            or fact.proposal.supersedes_proposal_id != expected_superseded
        ):
            raise ValueError("Investigation proposal changed its exact Attempt lineage")
        proposal_event_by_id[fact.proposal.proposal_id] = event
        proposal_output_lineages.append(
            InvestigationProposalOutputLineageView(
                proposal_id=fact.proposal.proposal_id,
                output_lineages=fact.output_lineages,
            )
        )
    for event, fact in facts:
        if not isinstance(fact, InvestigationProposalCarriedForward):
            continue
        revision = revision_by_id.get(
            next(
                (
                    item.revision.revision_id
                    for revision_event, item in facts
                    if isinstance(item, InvestigationCollectionRevisionRecorded)
                    and revision_event.get("event_id") == fact.revision_event_id
                ),
                "",
            )
        )
        previous = (
            proposal_by_id.get(revision.supersedes_proposal_id)
            if revision is not None
            else None
        )
        expected = (
            compile_contact_free_investigation_revision_proposal(
                investigation_input=investigation_input,
                previous_proposal=previous,
                revision=revision,
            )
            if previous is not None and revision is not None
            else None
        )
        if (
            expected is None
            or fact.proposal != expected
            or event.get("parent_ids") != [fact.revision_event_id]
            or fact.proposal.proposal_id in proposal_event_by_id
        ):
            raise ValueError("contact-free Investigation proposal changed its revision")
        proposal_event_by_id[fact.proposal.proposal_id] = event
    if len(proposals) != len(proposal_event_by_id):
        raise ValueError("Investigation proposal event identity is not exact")
    for event, fact in facts:
        if (
            not isinstance(fact, InvestigationReviewRequested)
            or fact.question is not None
        ):
            continue
        proposal_event = proposal_event_by_id.get(fact.proposal_id)
        if (
            proposal_event is None
            or proposal_event["event_id"] != fact.proposal_event_id
            or event.get("parent_ids")
            != [
                fact.proposal_event_id,
                *(
                    [fact.follows_response_event_id]
                    if fact.follows_response_event_id
                    else []
                ),
            ]
        ):
            raise ValueError("Investigation Review changed its proposal lineage")
    positions = {event["event_id"]: index for index, event in enumerate(timeline)}
    attempt_views = []
    for attempt in bound:
        _uuid(attempt.attempt_id, "attempt_id")
        validate_continuation(
            timeline, attempt_id=attempt.attempt_id, policy=attempt.continuation_policy
        )
        owned = [
            event
            for event in timeline
            if event.get("payload", {}).get("attempt_id") == attempt.attempt_id
        ]
        attempt_started = [
            event
            for event in owned
            if event.get("event_type") == "AttemptStarted"
            and event.get("payload", {}).get("agent_role") == "investigation_planner"
        ]
        if len(attempt_started) != 1:
            raise ValueError("Investigation requires one exact Attempt start")
        _validate_member_scope(
            attempt,
            attempt_started[0],
            owned,
            structured_lineages_by_attempt[attempt.attempt_id],
        )
        instruction_reference = retained_instruction_reference(attempt_started[0])
        effective_policy = retained_invocation_policy(
            attempt_started[0]["payload"],
            attempt.runtime_selections,
            agent_role="investigation_planner",
        )
        if effective_policy.continuation != attempt.continuation_policy:
            raise ValueError(
                "Investigation continuation policy changed its invocation binding"
            )
        for event in owned:
            if event.get("event_type") == "ModelRequestRecorded":
                validate_retained_model_request(event["payload"], effective_policy)
        endpoint = retained_endpoint_policy(
            effective_policy, attempt.runtime_selections
        )
        check_stall_notes(owned, endpoint.correction_wording)
        _validate_member_contexts(
            owned,
            attempt_started[0],
            timeline,
            source_timeline,
            positions,
            checked_once,
            parse,
        )
        terminal_candidates = [
            (event, fact)
            for event, fact in facts
            if getattr(fact, "attempt_id", None) == attempt.attempt_id
            and isinstance(
                fact,
                (
                    InvestigationReviewRequested,
                    InvestigationAttemptCompleted,
                    InvestigationAttemptFailed,
                ),
            )
        ]
        terminal_candidates = [
            (event, fact)
            for event, fact in terminal_candidates
            if not isinstance(fact, InvestigationReviewRequested)
            or fact.question is None
        ]
        terminal_candidates.extend(
            (event, validate_attempt_fact(event))
            for event in owned
            if event["event_type"] in {"AttemptCancelled", "AttemptStopped"}
            or (
                event["event_type"] == "AttemptFailed"
                and event["payload"].get("code") == "model_availability_exhausted"
            )
        )
        if len(terminal_candidates) != 1:
            raise ValueError("Investigation Attempt lacks one exact terminal")
        terminal_event, terminal = terminal_candidates[0]
        # The loop owns the Attempt's closure after its terminal (ADR 0044).
        try:
            validate_attempt_closure(timeline, attempt_started[0], terminal_event)
        except ValueError as error:
            raise ValueError(
                "Investigation Attempt closure changed its exact terminal"
            ) from error
        # Every result that made a contact, in call order: lookups, assembly reports,
        # the assembly searches a stated version's naming read and gene-change
        # placements.
        owned_results = [
            fact
            for event, fact in facts
            if isinstance(
                fact,
                (
                    RefSnpToolResultRecorded,
                    AssemblyReportToolResultRecorded,
                    AssembliesSearched,
                    InvestigationGeneChangePlaced,
                ),
            )
            and event.get("payload", {}).get("attempt_id") == attempt.attempt_id
        ]
        assertion = assertion_by_id[attempt.assertion_id]
        if (
            not owned_results
            and not prior_ids[attempt.attempt_id]
            and isinstance(terminal, InvestigationReviewRequested)
        ):
            raise ValueError("Investigation stopped without one typed Tool result")
        if isinstance(terminal, InvestigationAttemptFailed):
            source_event = next(
                (
                    event
                    for event in owned
                    if event.get("event_id") == terminal.source_event_id
                ),
                None,
            )
            expected_types = {
                "source_contact_failed": {"InvestigationSourceContactFailed"},
                "model_contact_failed": {"ModelRequestFailed"},
                "agent_run_failed": {
                    "InvestigationAttemptBound",
                    "InvestigationMemberContextRecorded",
                    "ObservedToolExecutionFailed",
                    "ToolCallValidated",
                    "ModelResponseRecorded",
                },
            }[terminal.code]
            if source_event is None or (
                terminal.source_event_id
                != next(
                    (
                        event["event_id"]
                        for event in reversed(owned)
                        if event.get("event_type") in expected_types
                        and (
                            event.get("event_type")
                            not in {"AgentOutputValidated", "ToolCallValidated"}
                            or event.get("payload", {}).get("status") == "invalid"
                        )
                    ),
                    None,
                )
            ):
                raise ValueError("Investigation failure changed its exact source")
            if (
                terminal.code == "agent_run_failed"
                and source_event.get("event_type") == "InvestigationAttemptBound"
                and any(
                    event.get("event_type")
                    in {
                        "ModelRequestRecorded",
                        "ModelPhysicalContactStarted",
                        "ModelResponseRecorded",
                        "ModelRequestFailed",
                        "ModelRequestStalled",
                        "ModelRequestRejected",
                        "InvestigationSourceContactStarted",
                        "InvestigationSourceContactFailed",
                        "RefSnpToolResultRecorded",
                        "AssemblyReportToolResultRecorded",
                        "AgentOutputValidated",
                    }
                    for event in owned
                )
            ):
                raise ValueError("pre-contact Agent failure changed its evidence scope")
            if (
                terminal.code == "agent_run_failed"
                and source_event.get("event_type") == "ObservedToolExecutionFailed"
            ):
                validate_tool_failure_source(
                    timeline,
                    source=source_event,
                    terminal=terminal_event,
                    attempt_id=attempt.attempt_id,
                )
            elif terminal.code == "agent_run_failed":
                expected_parent = terminal.source_event_id
                if source_event.get("event_type") == "ToolCallValidated":
                    dispositions = [
                        event
                        for event in owned
                        if event.get("event_type")
                        == "AgentActionTurnDispositionRecorded"
                        and event["payload"].get("validation_event_ids")
                        == [terminal.source_event_id]
                        and event["payload"].get("disposition")
                        in {"singleton_refused", "multi_call_refused"}
                        and event.get("parent_ids") == [terminal.source_event_id]
                    ]
                    if len(dispositions) != 1:
                        raise ValueError(
                            "Investigation action exhaustion changed its refusal"
                        )
                    expected_parent = dispositions[0]["event_id"]
                elif source_event.get("event_type") == "ModelResponseRecorded":
                    closures = [
                        event
                        for event in timeline
                        if event.get("event_type") == "activity_succeeded"
                        and event.get("payload", {}).get("result_event_id")
                        == terminal.source_event_id
                        and event.get("payload", {}).get("stage") == "model_contact"
                        and event.get("parent_ids") == [terminal.source_event_id]
                    ]
                    if len(closures) != 1:
                        raise ValueError(
                            "Investigation failure changed its response closure"
                        )
                    expected_parent = closures[0]["event_id"]
                if terminal_event.get("parent_ids") != [expected_parent]:
                    raise ValueError("Investigation failure changed its exact parent")
        contacts = [
            fact
            for _, fact in facts
            if isinstance(fact, InvestigationSourceContactStarted)
            and fact.attempt_id == attempt.attempt_id
        ]
        model_requests = [
            event
            for event in owned
            if event.get("event_type") == "ModelRequestRecorded"
        ]
        model_contacts = [
            event
            for event in owned
            if event.get("event_type") == "ModelPhysicalContactStarted"
        ]
        selection = attempt.runtime_selections.model_for_policy(effective_policy)
        expected_endpoint = {
            "provider": selection.provider,
            "model": selection.model,
            "contacted_url": selection.endpoint,
        }
        endpoint_evidence = [
            event.get("payload", {}).get("nebius_model_endpoint")
            for event in owned
            if event.get("event_type")
            in {
                "ModelRequestRecorded",
                "ModelResponseRecorded",
                "ModelRequestFailed",
                "ModelRequestStalled",
                "ModelRequestRejected",
            }
        ]
        if any(endpoint != expected_endpoint for endpoint in endpoint_evidence):
            raise ValueError("Investigation exceeded or changed its contact binding")
        from bio_neb.agentic_journey.model_availability import (
            validate_attempt_cancellation,
            validate_model_contact_window,
        )

        model_terminals = [
            e for e in owned if e["event_type"] in MODEL_REQUEST_TERMINALS
        ]
        cancelled = isinstance(terminal, AttemptCancelled)
        # The Attempt's own facts: its siblings may interleave.
        validate_attempt_cancellation(
            [
                event
                for event in timeline[
                    timeline.index(attempt_started[0]) : timeline.index(terminal_event)
                    + 1
                ]
                if event.get("payload", {}).get("attempt_id") == attempt.attempt_id
            ],
            terminal_event if cancelled else None,
            model_requests,
            model_terminals,
        )
        completed_requests = (
            model_requests[: len(model_terminals)] if cancelled else model_requests
        )
        for model_request, model_terminal in zip(
            completed_requests, model_terminals, strict=True
        ):
            validate_model_contact_window(
                timeline[
                    timeline.index(model_request) : timeline.index(model_terminal) + 1
                ],
                model_request,
                model_terminal,
            )
        if tuple(item.operation for item in contacts) != tuple(
            operation
            for item in owned_results
            for operation in _result_contact_operations(item)
        ):
            source_failure = any(
                isinstance(fact, InvestigationSourceContactFailed)
                and fact.attempt_id == attempt.attempt_id
                for _, fact in facts
            )
            mechanical_failure = any(
                event.get("event_type") == "ObservedToolExecutionFailed"
                and event.get("payload", {}).get("attempt_id") == attempt.attempt_id
                and event.get("payload", {}).get("stage_failure_event_id") is None
                for event in owned
            )
            if not source_failure and not mechanical_failure:
                raise ValueError("Investigation source contacts changed Tool order")
        if isinstance(terminal, InvestigationAttemptCompleted):
            question = request_events.get(terminal.question_request_event_id)
            if terminal.question_request_event_id is not None:
                if (
                    question is None
                    or question.question is None
                    or question.attempt_id != attempt.attempt_id
                ):
                    raise ValueError(
                        "Investigation question completion changed its request"
                    )
            elif not terminal.held_review_event_ids and not any(
                lineage.accepted_output is not None
                for lineage in structured_lineages_by_attempt[attempt.attempt_id]
            ):
                raise ValueError(
                    "Investigation member completion lacks accepted output"
                )
            from bio_neb.agentic_journey.operational_review import (
                project_operational_review,
            )

            before_terminal = timeline[: timeline.index(terminal_event)]
            # Fixed by the prefix before the terminal: one piece per owner.
            held = (
                project_operational_review(before_terminal)
                if checked_once is None
                else checked_once(
                    ("operational_review_before", terminal_event["event_id"]),
                    lambda prefix=before_terminal: project_operational_review(
                        prefix, checked_once=checked_once
                    ),
                )
            )
            held_ids = {
                r.request_event_id
                for r in (*held.requests, *availability.requests)
                if r.attempt_id == attempt.attempt_id
            }
            expected_holds = tuple(
                e["event_id"]
                for e in timeline[: timeline.index(terminal_event)]
                if e["event_id"] in held_ids
            )
            if terminal.held_review_event_ids != expected_holds:
                raise ValueError(
                    "Investigation completion changed its held operational Reviews"
                )
            outcome = (
                "question_pending"
                if question is not None
                else "operational_review_pending"
                if expected_holds
                else "work_completed"
            )
            stop_reason = None
        elif isinstance(terminal, InvestigationReviewRequested):
            outcome = "review_pending"
            stop_reason = None
        elif isinstance(terminal, AttemptFailed):
            if not any(
                r.attempt_id == attempt.attempt_id
                and r.logical_failure_event_id == terminal.source_event_id
                for r in availability.requests
            ):
                raise ValueError(
                    "Investigation availability terminal lacks its exact review"
                )
            outcome = "model_availability_review_pending"
            stop_reason = terminal.code
        elif (
            isinstance(terminal, AttemptStopped) and terminal.code == "effort_exhausted"
        ):
            validate_effort_terminal(timeline, terminal_event)
            outcome = "effort_review_pending"
            stop_reason = terminal.code
        elif isinstance(terminal, AttemptStopped):
            validate_agent_work_terminal(timeline, terminal_event)
            outcome = "agent_work_review_pending"
            stop_reason = terminal.code
        elif isinstance(terminal, AttemptCancelled):
            outcome = "cancelled"
            stop_reason = "cancelled"
        elif isinstance(terminal, InvestigationAttemptFailed):
            outcome = "failed"
            stop_reason = terminal.code
        else:
            raise AssertionError("unreachable Investigation terminal")
        attempt_views.append(
            InvestigationAttemptView(
                attempt_id=attempt.attempt_id,
                instruction_identity=attempt_started[0]["payload"][
                    "instruction_identity"
                ],
                instruction_artifact=instruction_reference,
                revision_id=attempt.revision_id,
                work_requests=attempt.work_requests,
                continuation_policy=attempt.continuation_policy,
                effective_invocation_policy=effective_policy,
                invocation_policy_id=effective_policy.policy_id,
                invocation_overrides=attempt_started[0]["payload"][
                    "invocation_overrides"
                ],
                continuation_policy_id=attempt.continuation_policy_id,
                outcome=outcome,
                model_request_event_ids=tuple(
                    event["event_id"] for event in model_requests
                ),
                model_contact_event_ids=tuple(
                    event["event_id"] for event in model_contacts
                ),
                model_response_event_ids=tuple(
                    event["event_id"]
                    for event in owned
                    if event.get("event_type") == "ModelResponseRecorded"
                ),
                model_failure_event_ids=tuple(
                    event["event_id"]
                    for event in owned
                    if event.get("event_type") == "ModelRequestFailed"
                ),
                source_contact_event_ids=tuple(
                    event["event_id"]
                    for event in owned
                    if event.get("event_type") == "InvestigationSourceContactStarted"
                ),
                source_failure_event_ids=tuple(
                    event["event_id"]
                    for event in owned
                    if event.get("event_type") == "InvestigationSourceContactFailed"
                ),
                source_result_event_ids=tuple(
                    event["event_id"]
                    for event in owned
                    if event.get("event_type")
                    in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
                ),
                structured_output_lineages=structured_lineages_by_attempt.get(
                    attempt.attempt_id, ()
                ),
                collection_member_id=attempt.collection_member_id,
                stop_reason=stop_reason,
                failure_reason=(
                    terminal.reason
                    if isinstance(terminal, InvestigationAttemptFailed)
                    else None
                ),
            )
        )
    stopped = [fact for _, fact in facts if isinstance(fact, InvestigationStopped)]
    # A member Attempt's fault holds the Run through the failure module; after
    # `resume` that member is planned again (R2). A fault no later Attempt
    # of its member followed leaves the collection's work ready, never ended.
    member_of = {a.attempt_id: (revision_key(a), a.collection_member_id) for a in bound}
    unrecovered = [
        fact
        for _, fact in facts
        if isinstance(fact, InvestigationAttemptFailed)
        and not any(
            member_of[later.attempt_id] == member_of[fact.attempt_id]
            and bound_index[later.attempt_id] > bound_index[fact.attempt_id]
            for later in bound
        )
    ]
    latest_assertion_used = bound[-1].assertion_id == assertions[-1].assertion_id
    status = (
        "stopped"
        if stopped
        else "review_pending"
        if pending
        else "investigation_work_ready"
        if receipts
        and receipts[-1].response.action == "revise"
        and not latest_assertion_used
        else "investigation_work_ready"
        if unrecovered
        else "approved"
        if authorities
        else "investigation_work_ready"
    )
    from bio_neb.investigation.investigation_questions import member_stop_rejections

    # A member's stop answered stop settles that member alone; only a
    # whole-collection stop holds the Investigation.
    settled_members = member_stop_rejections(
        timeline, bound[-1].assertion_id, bound[-1].revision_number, checked_once
    )
    settled_requests = {r.request_event_id for r in settled_members.values()}
    if (
        availability.pending_request is not None
        and availability.pending_request.agent_role == "investigation_planner"
    ):
        status = "model_availability_review_pending"
    elif (
        availability.receipts
        and availability.requests[-1].agent_role == "investigation_planner"
        and availability.requests[-1].request_event_id not in settled_requests
    ):
        last = availability.receipts[-1]
        answer_index = next(
            i for i, e in enumerate(timeline) if e["event_id"] == last.response_event_id
        )
        if last.response.action == "stop":
            status = "model_availability_stopped"
        elif not any(
            e["event_type"] == "AttemptStarted" for e in timeline[answer_index + 1 :]
        ):
            status = "model_availability_review_direction_recorded"
    if stopped_effort := effort_status(
        timeline, agent_role="investigation_planner", checked_once=checked_once
    ):
        status = stopped_effort
    if stopped_work := agent_work_status(
        timeline, agent_role="investigation_planner", checked_once=checked_once
    ):
        status = stopped_work
    eligible = ()
    from bio_neb.agentic_journey.operational_review import project_operational_review

    operational = project_operational_review(timeline, checked_once=checked_once)
    latest = bound[-1]
    current_lineages = collection_lineages(revision_key(latest))
    if status in {
        "review_pending",
        "investigation_work_ready",
        "model_availability_review_pending",
        "model_availability_stopped",
        "model_availability_review_direction_recorded",
        "effort_review_pending",
        "effort_review_direction_recorded",
        "effort_stopped",
        "agent_work_review_pending",
        "agent_work_review_direction_recorded",
        "agent_work_rejected",
    } and not any(p.revision_number == latest.revision_number for p in proposals):
        from bio_neb.investigation.investigation_questions import eligible_members

        eligible = eligible_members(
            latest.work_requests,
            latest.assertion_id,
            latest.revision_number,
            current_lineages,
            requests,
            receipts,
        )
        revision_attempts = {
            identity
            for attempt in bound
            if revision_key(attempt) == revision_key(latest)
            for identity in (*prior_ids[attempt.attempt_id], attempt.attempt_id)
        }
        scoped_operations = [
            r
            for r in (*operational.requests, *availability.requests)
            if r.agent_role == "investigation_planner"
            and r.attempt_id in revision_attempts
            and r.request_event_id not in settled_requests
        ]
        # A member's own stop holds only that member; a whole-collection
        # Attempt's stop holds every member, as before.
        eligible = tuple(
            member
            for member in eligible
            if member not in settled_members
            and member not in {r.member_id for r in scoped_operations}
            and effort_status(
                timeline,
                agent_role="investigation_planner",
                member_id=member,
                checked_once=checked_once,
            )
            is None
            and agent_work_status(
                timeline,
                agent_role="investigation_planner",
                member_id=member,
                checked_once=checked_once,
            )
            is None
        )
        collection_statuses = {
            "review_pending",
            "investigation_work_ready",
            "model_availability_review_pending",
            "model_availability_stopped",
        }
        if eligible:
            status = "investigation_work_ready"
        elif status in collection_statuses and any(
            r.pending for r in scoped_operations
        ):
            status = "operational_review_pending"
        elif status in collection_statuses and any(
            receipt.response.action != "stop"
            and receipt.request_event_id
            in {r.request_event_id for r in scoped_operations}
            and not any(
                e["event_type"] == "AttemptStarted"
                and e["payload"].get("continuation_operational_review_event_id")
                == receipt.response_event_id
                for e in timeline
            )
            for receipt in operational.receipts
        ):
            status = "operational_review_direction_recorded"
        elif status in collection_statuses and any(
            receipt.response.action == "stop"
            and receipt.request_event_id
            in {r.request_event_id for r in scoped_operations}
            for receipt in operational.receipts
        ):
            status = "operational_stopped"
    return InvestigationProjection(
        status=status,
        eligible_member_ids=tuple(m for m in eligible if m not in not_checked),
        not_checked=tuple(not_checked.values()),
        collection_lineages=current_lineages,
        investigation_input=investigation_input,
        assertions=tuple(assertions),
        sources=tuple(sources),
        source_failures=tuple(
            fact
            for _, fact in facts
            if isinstance(fact, InvestigationSourceContactFailed)
        ),
        assessments=tuple(assessments),
        attempts=tuple(attempt_views),
        proposals=tuple(proposals),
        requirement_values=_investigation_identity_values(proposals[-1], timeline)[0]
        if proposals
        else (),
        fact_slots=_investigation_identity_values(proposals[-1], timeline)[1]
        if proposals
        else (),
        proposal_output_lineages=tuple(proposal_output_lineages),
        revisions=tuple(revisions),
        review_requests=tuple(requests),
        review_receipts=receipts,
        pending_request=pending[0] if pending else None,
        authority=authorities[-1] if authorities else None,
        authorities=tuple(authorities),
        pending_work_item_ids=tuple(
            item.work_item_id
            for request in pending
            if request.proposal is not None
            for item in request.proposal.work_items
            if item.work_item_id
            not in approved_by_request.get(request.request_event_id, set())
        ),
        stop_reason=(
            stopped[-1].reason
            if stopped
            else unrecovered[-1].code
            if unrecovered
            else None
        ),
        model_contact_count=sum(
            len(attempt.model_contact_event_ids) for attempt in attempt_views
        ),
        source_contact_count=sum(
            len(attempt.source_contact_event_ids) for attempt in attempt_views
        ),
    )


def project_snapshot_investigation(snapshot) -> InvestigationProjection:
    """One evidence snapshot's Investigation, the reader its ``derived`` caches.

    Pass metadata and a Fork origin change no Investigation fact, so a snapshot
    ending in them shares the projection of its prefix before them. The prefix
    through an admitted event never changes, so its owner folds it once, for
    every snapshot of that prefix across the owner's appends.
    """
    from bio_neb.recording import REVIEW_METADATA

    head = next(
        (
            event["event_id"]
            for event in reversed(snapshot.timeline)
            if event["event_type"] not in REVIEW_METADATA
        ),
        None,
    )
    if head is not None and head != snapshot.prefix_event_ids[-1]:
        return snapshot.through(head).derived(project_snapshot_investigation)

    def fold():
        return project_investigation(
            snapshot.timeline, checked_once=snapshot.checked_once
        )

    if head is None:
        return fold()
    return snapshot.checked_once(("investigation_projection", head), fold)


class InvestigationProposalRefused(ValueError):
    """A refused output's code and its details: typed facts (the source's
    values, the fields to omit) kept until the writer renders them (#372 D2)."""

    def __init__(self, code: str, *details: RefusalDetail) -> None:
        super().__init__(code)
        self.code = code
        self.facts = tuple(details)
        self.details = tuple(refusal_detail_text(detail) for detail in details)


def compile_investigation_proposal(
    *,
    investigation_input: InvestigationInput,
    assertion: UserAssertion,
    refsnp: RefSnpToolResult,
    assembly_report: AssemblyReportToolResult,
    revision_number: int,
    supersedes_proposal_id: str | None,
    predecessor_ids: tuple[str, ...],
) -> InvestigationProposal:
    """Compile only facts admitted by both exact deterministic source outcomes."""

    return compile_investigation_collection_proposal(
        investigation_input=investigation_input,
        assertion=assertion,
        source_results=((refsnp, assembly_report),),
        revision_number=revision_number,
        supersedes_proposal_id=supersedes_proposal_id,
        predecessor_ids=predecessor_ids,
    )


def compile_contact_free_investigation_revision_proposal(
    *,
    investigation_input: InvestigationInput,
    previous_proposal: InvestigationProposal,
    revision: InvestigationCollectionRevision,
) -> InvestigationProposal:
    """Compile a superseding proposal when the patch carries only retained evidence."""

    if (
        revision.supersedes_proposal_id != previous_proposal.proposal_id
        or revision.work_requests
        or not revision.carried_work_items
    ):
        raise InvestigationProposalRefused("revision_requires_contact")
    material = {
        "revision_number": revision.revision_number,
        "investigation_input_id": investigation_input.input_id,
        "assertion_id": previous_proposal.assertion_id,
        "work_items": tuple(
            item.model_dump(mode="python") for item in revision.carried_work_items
        ),
        "unresolved_conflicts": (),
        "completeness": "complete",
        "supersedes_proposal_id": previous_proposal.proposal_id,
        "predecessor_ids": (previous_proposal.proposal_id, revision.revision_id),
    }
    if revision.carried_request_members:
        material["request_members"] = revision.carried_request_members
    return InvestigationProposal(
        proposal_id=_digest("investigation-proposal", material), **material
    )


def compile_contact_investigation_revision_proposal(
    *,
    investigation_input: InvestigationInput,
    assertion: UserAssertion,
    previous_proposal: InvestigationProposal,
    revision: InvestigationCollectionRevision,
    source_results: tuple[
        tuple[RefSnpToolResult, AssemblyReportToolResult | None] | None, ...
    ],
    rejections: Mapping[str, InvestigationRequestRejection] | None = None,
    issue_approvals: Mapping[str, InvestigationIssueApproval] | None = None,
    deferred: Mapping[str, InvestigationNotChecked] | None = None,
    target_asserters: Mapping[str, str] | None = None,
) -> InvestigationProposal:
    """Merge exact carried Work items with only this revision's contacted work; a
    carried request member keeps its own attribution. A revision member whose finding
    is deferred (`deferred_finding`, owner #480 Q1) is accounted not checked, with no
    Work."""

    if (
        revision.supersedes_proposal_id != previous_proposal.proposal_id
        or not revision.work_requests
        or tuple(assertion.requests) != revision.work_requests
        or len(source_results) != len(revision.work_requests)
    ):
        raise InvestigationProposalRefused("revision_work_set_mismatch")
    source_pairs = iter(enumerate(source_results))
    work_items: list[WorkItem] = []
    by_semantics = {
        _work_item_semantic_id(item.target, item.intended_action): item
        for item in revision.carried_work_items
    }
    request_members = list(revision.carried_request_members)
    for member in revision.members:
        if member.request is None:
            if (
                member.carried_work_item is not None
                and member.carried_work_item not in work_items
            ):
                work_items.append(member.carried_work_item)
            continue
        ordinal, pair = next(source_pairs)
        rejection = (rejections or {}).get(
            investigation_output_member_id(member.request, ordinal)
        )
        outcome = (deferred or {}).get(
            investigation_output_member_id(member.request, ordinal)
        )
        if outcome is not None and rejection is None:
            resolved, items = (
                InvestigationRequestMember(
                    request_member_id=request_member_id(
                        assertion.assertion_id, member.request, ordinal
                    ),
                    request=member.request,
                    refsnp=None,
                    assembly_report=None,
                    not_checked=outcome,
                ),
                (),
            )
        elif rejection is not None:
            resolved, items = (
                _rejected_request_member(
                    member.request, ordinal, assertion.assertion_id, rejection
                ),
                (),
            )
        else:
            resolved, items = _resolve_request_member(
                investigation_input,
                member.request,
                pair,
                ordinal,
                assertion.assertion_id,
                (issue_approvals or {}).get(
                    investigation_output_member_id(member.request, ordinal)
                ),
                (target_asserters or {}).get(
                    investigation_output_member_id(member.request, ordinal)
                ),
            )
        revised = []
        for item in items:
            item = _revised_work_item(item, revision, member)
            item = by_semantics.setdefault(
                _work_item_semantic_id(item.target, item.intended_action), item
            )
            if item not in work_items:
                work_items.append(item)
            revised.append(item.work_item_id)
        request_members.append(
            resolved.model_copy(
                update={
                    "work_item_ids": tuple(revised),
                    "previous_request_member_id": member.previous_request_member_id,
                }
            )
        )
    result_ids = tuple(
        dict.fromkeys(
            result.result_id
            for pair in source_results
            if pair is not None
            for result in pair
            if result is not None
        )
    )
    material = {
        "revision_number": revision.revision_number,
        "investigation_input_id": investigation_input.input_id,
        "assertion_id": assertion.assertion_id,
        "work_items": tuple(item.model_dump(mode="python") for item in work_items),
        "request_members": tuple(request_members),
        "unresolved_conflicts": (),
        "completeness": "complete",
        "supersedes_proposal_id": previous_proposal.proposal_id,
        "predecessor_ids": (
            previous_proposal.proposal_id,
            revision.revision_id,
            *result_ids,
            *(r.response_event_id for r in (rejections or {}).values()),
            *(a.response_event_id for a in (issue_approvals or {}).values()),
            *(n.event_id for n in (deferred or {}).values()),
        ),
    }
    return InvestigationProposal(
        proposal_id=_digest("investigation-proposal", material), **material
    )


def _retried_target_changed(
    request: InvestigationRequestItem | None, item: WorkItem
) -> bool:
    """Whether a retry's new Work item names another allele or region than it re-ran."""
    target = item.target
    if request is None:
        return True
    if isinstance(target, RegionTarget):
        return request.region != StatedRegion(
            contig=target.contig, start=target.start, end=target.end
        ) or (item.intended_action != request.normalized_question_action)
    return request.position is None or request.position.identity != (
        target.contig,
        target.one_based_position,
        target.ref,
        target.alt,
    )


def _revised_work_item(
    item: WorkItem,
    revision: InvestigationCollectionRevision,
    member: InvestigationRevisionMember,
) -> WorkItem:
    if member.operation == "retry" and _retried_target_changed(member.request, item):
        raise InvestigationProposalRefused("retry_target_changed")
    material = item.model_dump(mode="python", exclude={"work_item_id"})
    material.update(
        revision_id=revision.revision_id,
        previous_work_item_id=member.previous_work_item_id,
    )
    return WorkItem(work_item_id=_digest("work-item", material), **material)


def compile_investigation_collection_proposal(
    *,
    investigation_input: InvestigationInput,
    assertion: UserAssertion,
    source_results: tuple[
        tuple[RefSnpToolResult, AssemblyReportToolResult | None] | None, ...
    ],
    revision_number: int,
    supersedes_proposal_id: str | None,
    predecessor_ids: tuple[str, ...],
    rejections: Mapping[str, InvestigationRequestRejection] | None = None,
    issue_approvals: Mapping[str, InvestigationIssueApproval] | None = None,
    not_checked: Mapping[str, InvestigationNotChecked] | None = None,
    target_asserters: Mapping[str, str] | None = None,
) -> InvestigationProposal:
    """Compile one ordered Work item for each admitted request and source pair; a
    rejected or not-checked request is its own accounting (#367)."""

    requested_rsids = (
        tuple(request.rsid for request in assertion.requests)
        if assertion.requests
        else (assertion.rsid,)
    )
    if len(source_results) != len(requested_rsids):
        raise InvestigationProposalRefused("request_source_cardinality_mismatch")

    requests = assertion.requests or (
        InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        ),
    )
    rejections = rejections or {}
    issue_approvals = issue_approvals or {}
    not_checked = not_checked or {}
    resolved = tuple(
        (
            InvestigationRequestMember(
                request_member_id=request_member_id(
                    assertion.assertion_id, request, ordinal
                ),
                request=request,
                refsnp=None,
                assembly_report=None,
                not_checked=not_checked[
                    investigation_output_member_id(request, ordinal)
                ],
            ),
            (),
        )
        if investigation_output_member_id(request, ordinal) in not_checked
        else (
            _rejected_request_member(
                request,
                ordinal,
                assertion.assertion_id,
                rejections[investigation_output_member_id(request, ordinal)],
            ),
            (),
        )
        if investigation_output_member_id(request, ordinal) in rejections
        else _resolve_request_member(
            investigation_input,
            request,
            pair,
            ordinal,
            assertion.assertion_id,
            issue_approvals.get(investigation_output_member_id(request, ordinal)),
            (target_asserters or {}).get(
                investigation_output_member_id(request, ordinal)
            ),
        )
        for ordinal, (request, pair) in enumerate(
            zip(requests, source_results, strict=True)
        )
    )
    request_members = tuple(member for member, _ in resolved)
    work_items = tuple(item for _, items in resolved for item in items)
    unique_work: dict[str, WorkItem] = {}
    for item in work_items:
        unique_work.setdefault(item.work_item_id, item)
    work_items = tuple(unique_work.values())
    proposal_material = {
        "revision_number": revision_number,
        "investigation_input_id": investigation_input.input_id,
        "assertion_id": assertion.assertion_id,
        "work_items": tuple(item.model_dump(mode="python") for item in work_items),
        "request_members": request_members,
        "unresolved_conflicts": (),
        "completeness": "complete",
        "supersedes_proposal_id": supersedes_proposal_id,
        "predecessor_ids": predecessor_ids,
    }
    return InvestigationProposal(
        proposal_id=_digest("investigation-proposal", proposal_material),
        **proposal_material,
    )


def _investigation_identity_values(
    proposal: InvestigationProposal,
    timeline: list[dict[str, Any]],
) -> tuple[tuple[RequirementValue, ...], tuple[FactSlotStatus, ...]]:
    """The original requests and verified source identities, using the shared slots."""
    values = []
    slots = []
    for member in proposal.request_members:
        identity = dict(
            requirement_id=member.request_member_id,
            kind="requested_rsid",
            subject_key=member.request_member_id,
        )
        # A revision's proposal names its new assertion, but unchanged members
        # retain their original identity. Bind each member to the first checked
        # proposal that introduced it, not to a later collection amendment.
        introducing_proposal = next(
            event
            for event in timeline
            if event["event_type"] == "InvestigationProposalRecorded"
            and any(
                item["request_member_id"] == member.request_member_id
                for item in event["payload"]["proposal"].get("request_members", ())
            )
        )
        assertion_id = introducing_proposal["payload"]["proposal"]["assertion_id"]
        revision_number = introducing_proposal["payload"]["proposal"]["revision_number"]
        assertion_event = next(
            event
            for event in timeline
            if event["payload"].get("assertion", {}).get("assertion_id") == assertion_id
        )
        requested_rsid = effective_requested_rsid(member.request)
        request_origin = assertion_event["event_id"]
        request_source = "research_request"
        if requested_rsid is None:
            assertions = assertion_event["payload"]["assertion"].get("requests") or [
                {
                    "request_text": assertion_event["payload"]["assertion"][
                        "request_text"
                    ],
                    "rsid": None,
                }
            ]
            output_id = next(
                candidate
                for ordinal, item in enumerate(assertions)
                if (
                    candidate := investigation_output_member_id(
                        InvestigationRequestItem.model_validate(item, strict=False),
                        ordinal,
                    )
                )
                and _digest(
                    "request-member",
                    {
                        "assertion_id": assertion_id,
                        "output_member_id": candidate,
                    },
                )
                == member.request_member_id
            )
            prefix = timeline[: timeline.index(introducing_proposal)]
            questions = {
                event["event_id"]: event["payload"].get("question")
                for event in prefix
                if event["event_type"] == "InvestigationReviewRequested"
            }
            answers = []
            for event in prefix:
                if event["event_type"] != "InvestigationReviewResponded":
                    continue
                answer = event["payload"]["response"]
                question = questions.get(event["payload"]["request_event_id"])
                if (
                    answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS
                    and question
                    and question["assertion_id"] == assertion_id
                    and question["revision_number"] == revision_number
                    and question["collection_member_id"] == output_id
                ):
                    answers.append(event)
            # The rsID the model typed from an untyped item's own words, else the
            # latest answer whose typed target, the researcher's or the model's,
            # names one (R1 b).
            reads = target_reads(prefix, output_id, revision_number)
            requested_rsid = effective_requested_rsid(
                member.request, typed_answers(member.request, (), reads)
            )
            for answer in () if requested_rsid is not None else reversed(answers):
                (typed,) = typed_answers(
                    member.request,
                    (
                        {
                            **answer["payload"]["response"],
                            "response_event_id": answer["event_id"],
                        },
                    ),
                    [read for read in reads if read["source"] == "answer"],
                )
                if (typed.get("target") or {}).get("rsid") is not None:
                    requested_rsid = typed["target"]["rsid"]
                    request_origin = answer["event_id"]
                    request_source = "review_answer"
                    break
        event_ids = {event["event_id"] for event in timeline}
        request_lineage = tuple(
            dict.fromkeys(
                (
                    request_origin,
                    *(
                        identity
                        for identity in assertion_event["payload"]["assertion"][
                            "predecessor_ids"
                        ]
                        if assertion_event["event_type"] == "InvestigationStarted"
                        and identity in event_ids
                    ),
                )
            )
        )
        member_values = (
            [
                RequirementValue.create(
                    **identity,
                    value=requested_rsid,
                    source=request_source,
                    lineage_event_ids=request_lineage,
                )
            ]
            if requested_rsid is not None
            else []
        )
        if member.refsnp is not None and member.refsnp.verified_rsid is not None:
            selected_event_id = next(
                (
                    lineage["accepted_output"]["refsnp_event_id"]
                    for proposal_event in timeline
                    if proposal_event["event_type"] == "InvestigationProposalRecorded"
                    and proposal_event["payload"]["proposal"]["assertion_id"]
                    == assertion_id
                    for lineage in proposal_event["payload"]["output_lineages"]
                    if _digest(
                        "request-member",
                        {
                            "assertion_id": assertion_id,
                            "output_member_id": lineage["collection_member_id"],
                        },
                    )
                    == member.request_member_id
                ),
                None,
            )
            if selected_event_id is None:
                raise ValueError(
                    "Investigation proposal changed its exact Attempt lineage"
                )
            source_event = next(
                (event for event in timeline if event["event_id"] == selected_event_id),
                None,
            )
            if source_event is None:
                raise ValueError(
                    "Investigation proposal lost its selected source observation"
                )
            member_values.append(
                RequirementValue.create(
                    **identity,
                    value=member.refsnp.verified_rsid,
                    source="tool_result",
                    lineage_event_ids=(source_event["event_id"],),
                )
            )
        values.extend(member_values)
        slots.append(
            fact_slot_status(
                **identity,
                # The source its lookup asked, from the result; none before a
                # lookup.
                source=None
                if member.refsnp is None
                else {"ncbi_dbsnp": "NCBI RefSNP", "eva": "EVA", "ncbi": "NCBI"}[
                    member.refsnp.lookup_source
                ],
                requested=requested_rsid,
                values=tuple(member_values),
                # A checked rsID with no verified record says what its source answered.
                source_outcome=None
                if member.refsnp is None or member.refsnp.verified_rsid is not None
                else member.refsnp.status,
            )
        )
    return tuple(values), tuple(slots)


# The model's choice and its citations only (ADR 0049): the outcome, the lookup
# result and the assembly check it rests on. Code fills every copied field (rsID,
# assembly, contig, position, alleles, conflicts) from the cited facts when it
# compiles the proposal, so the model re-types nothing code holds; a draft that sends
# one is refused as a field the output does not take.
class InvestigationPlanDraft(StrictFrozenContract):
    outcome: Literal["complete", "unresolved"]
    refsnp_event_id: StrictStr
    assembly_event_id: StrictStr | None = None


def validate_investigation_output(
    events: Iterable[Mapping[str, Any]],
    member_id: str,
    output: InvestigationPlanDraft,
    *,
    requested_rsids: tuple[str, ...],
    stated_position: TargetLocation,
) -> tuple[RefSnpToolResult, AssemblyReportToolResult | None]:
    """Check the selected observations and every scientific output field.

    A complete output's retrieved target is checked against the member's request:
    its verified rsID is one the researcher stated, or its dbSNP record lists one
    of those as merged into it (a documented rename).
    """
    refsnp, assembly = select_investigation_sources(
        events,
        member_id,
        output.model_dump(mode="json"),
    )
    if refsnp.status == "region_located":
        return _validated_region_output(refsnp, output, stated_position), None
    if output.outcome == "unresolved":
        if refsnp.status in ADMISSIBLE_LOOKUP_STATUSES:
            if assembly is None:
                raise InvestigationProposalRefused(
                    "assembly_report_missing",
                    _assembly_detail(events, member_id, output.refsnp_event_id, refsnp),
                )
            if assembly.status == "compatible":
                raise InvestigationProposalRefused(
                    "complete_sources_marked_unresolved",
                    ExpectedValue("outcome", "complete"),
                )
        return refsnp, assembly
    if refsnp.status not in ADMISSIBLE_LOOKUP_STATUSES:
        raise _marked_complete(refsnp.conflicts)
    if assembly is None:
        raise InvestigationProposalRefused(
            "assembly_report_missing",
            _assembly_detail(events, member_id, output.refsnp_event_id, refsnp),
        )
    if assembly.status != "compatible":
        raise _marked_complete(assembly.conflicts)
    if stated_position.position is not None and (
        refsnp.location == "not_applicable"
        or not _same_location(refsnp.location, stated_position)
    ):
        # A finding answers the member's current position, never an earlier one.
        raise InvestigationProposalRefused(
            RETRIEVED_TARGET_NOT_REQUESTED,
            "location",
            "the selected lookup did not place the position this request now "
            "states; look it up with locate_variant",
        )
    if (difference := _intent_difference(refsnp, requested_rsids)) is not None:
        raise InvestigationProposalRefused(
            RETRIEVED_TARGET_NOT_REQUESTED, *difference, _ASK_INTENDED_TARGET
        )
    return refsnp, assembly


def _validated_region_output(
    refsnp: RefSnpToolResult, output: InvestigationPlanDraft, current: TargetLocation
) -> RefSnpToolResult:
    """A located region's output is complete and cites no assembly report (#367):
    code copies the Run's assembly, sequence name and the region's positions from
    the lookup."""
    if output.outcome != "complete":
        raise InvestigationProposalRefused(
            "complete_sources_marked_unresolved", ExpectedValue("outcome", "complete")
        )
    if output.assembly_event_id is not None:
        raise InvestigationProposalRefused(
            "region_cites_assembly_report", OmitFields(("assembly_event_id",))
        )
    location = refsnp.location
    if (
        location.stated,
        location.stated_assembly,
        location.placed,
        location.region,
    ) != current:
        # A finding answers the member's current target, never an earlier one.
        raise InvestigationProposalRefused(
            RETRIEVED_TARGET_NOT_REQUESTED,
            "location",
            "the selected lookup did not locate the target this request now states",
        )
    return refsnp


DifferenceKind = Literal["rsid", "location"]
"""What an intent refusal's retrieved target differs in (#472 Q4 a): its rsID, or where
it is (position, alleles, version). The refusal carries it typed, after its code, and
the question it asks is chosen by it, never by the difference's wording."""


def _intent_difference(
    refsnp: RefSnpToolResult, requested_rsids: tuple[str, ...]
) -> tuple[DifferenceKind, str] | None:
    """How a complete finding's retrieved target differs from what its request
    states, or ``None``: the rsID it names, and for a stated position (#369) the
    position and alleles, compared on the Run's sequence name the lookup found.

    A verified rsID other than the stated one passes only through the source's own
    merge record. With no rsID stated, a position's lookup may read the rsID the
    source lists at it.
    """
    location = refsnp.location
    located = location != "not_applicable"
    verified = refsnp.verified_rsid
    allowed = set(requested_rsids) or set(location.listed_rsids if located else ())
    if verified is None:
        if requested_rsids:
            return (
                "rsid",
                f"requested {', '.join(requested_rsids)}; the source lists no "
                "variant at the stated position",
            )
    elif verified not in allowed and not allowed.intersection(refsnp.merged_rsids):
        listed = located and verified in location.listed_rsids
        return (
            "rsid",
            f"requested {', '.join(requested_rsids)}, retrieved {verified}"
            + (" at the stated position" if listed else ""),
        )
    if not located:
        return None
    placed = _location_difference(refsnp, verified)
    return None if placed is None else ("location", placed)


def _location_difference(refsnp: RefSnpToolResult, verified: str | None) -> str | None:
    """Where a located finding's retrieved target is, unlike its request states."""
    location = refsnp.location
    stated, source = location.looked_up, _LOOKUP_NAMES[refsnp.lookup_source]
    # A placed position is on the Run's assembly by its placement; a stated version
    # is the Run's only when it was named as the Run's own assembly.
    naming = location.version_naming
    if (
        location.placed is None
        and location.stated_assembly is not None
        and (
            naming is None
            or naming.assembly_accession
            not in {
                refsnp.expected_assembly_accession,
                location.run_assembly_paired_accession,
            }
        )
    ):
        return _stated_version_difference(refsnp, source)
    if location.run_contig is None:
        return None
    identity = stated.identity
    if identity is not None:
        run_identity = identity._replace(contig=location.run_contig)
        held = tuple(dict.fromkeys(c.target_identity for c in refsnp.candidates))
        if run_identity in held:
            return None
        return (
            f"the request states {location.run_contig}:{run_identity.one_based_position}"
            f" {allele_text(run_identity.ref, run_identity.alt)}; {source} holds "
            + ", ".join(
                f"{item.contig}:{item.one_based_position} "
                f"{allele_text(item.ref, item.alt)}"
                for item in held
            )
            + _other_versions(location, source)
        )
    places = {
        (candidate.contig, position)
        for candidate in refsnp.candidates
        for position in (
            candidate.one_based_position,
            candidate.target_identity.one_based_position,
        )
    }
    if (location.run_contig, stated.position) in places or not refsnp.candidates:
        return None
    candidate = refsnp.candidates[0]
    return (
        f"the request places {verified} at {location.run_contig}:{stated.position}; "
        f"{source} places it at {candidate.contig}:{candidate.one_based_position}"
        + _other_versions(location, source)
    )


_LOOKUP_NAMES = {"ncbi_dbsnp": "dbSNP", "eva": "EVA", "ncbi": "NCBI"}


def _other_versions(location: VariantLocationDetail, source: str) -> str:
    """With no version stated, where the record places the variant on every other
    version, so the researcher's answer can name one (owner's rule, steps 2-4)."""
    if location.stated_assembly is not None:
        return ""
    places = dict.fromkeys(
        (item.contig, item.one_based_position, item.assembly_name)
        for item in location.stated_version_placements
    )
    return "".join(
        f"; with no version stated, {source} also places it at {contig}:{position} "
        f"on {name}"
        for contig, position, name in places
    )


def _stated_version_difference(refsnp: RefSnpToolResult, source: str) -> str | None:
    """Where a request states an rsID on another version than the Run's, compared
    with the record's own placement there, on that version's sequence name; no
    position is mapped between versions (#369)."""
    location = refsnp.location
    stated, name = location.stated, location.stated_assembly.name
    placed = location.stated_version_placements
    if not placed:
        return (
            f"the request places {refsnp.verified_rsid} on {name}; {source} places "
            "it on no assembly by that name"
        )
    identity = stated.identity
    for placement in placed:
        position, ref, alt = trimmed_allele(
            placement.one_based_position, placement.ref, placement.alt
        )
        if placement.contig != location.stated_version_contig:
            continue
        if identity is None and stated.position in {
            placement.one_based_position,
            position,
        }:
            return None
        if identity is not None and (
            identity.one_based_position,
            identity.ref,
            identity.alt,
        ) == (position, ref, alt):
            return None
    first = placed[0]
    return (
        f"the request places {refsnp.verified_rsid} at {stated.contig}:"
        f"{stated.position} on {name}; {source} places it at {first.contig}:"
        f"{first.one_based_position} on {first.assembly_name}"
    )


def _marked_complete(conflicts: tuple[str, ...]) -> InvestigationProposalRefused:
    """A complete draft of sources that did not resolve: the outcome to send, and
    what the cited source observed."""
    return InvestigationProposalRefused(
        "unresolved_sources_marked_complete",
        ExpectedValue("outcome", "unresolved"),
        *conflicts,
    )


def _assembly_detail(
    events: Iterable[Mapping[str, Any]],
    member_id: str,
    refsnp_event_id: str,
    refsnp: RefSnpToolResult,
) -> str:
    """The assembly report a draft cites: the member's compatible report that
    checked the lookup, or the call that checks it."""
    checks = [
        event["event_id"]
        for event in events
        if event["event_type"] == "AssemblyReportToolResultRecorded"
        and event["payload"]["collection_member_id"] == member_id
        and event["payload"]["refsnp_event_id"] == refsnp_event_id
        and event["payload"]["result"]["status"] == "compatible"
    ]
    if len(checks) == 1:
        return f"assembly_event_id: expected={json.dumps(checks[0])}"
    if checks:
        return f"assembly_event_id: expected one of {json.dumps(checks)}"
    accession = refsnp.candidates[0].assembly_accession if refsnp.candidates else None
    return (
        f"assembly_event_id: no compatible assembly report checked {refsnp_event_id}; "
        "call fetch_assembly_report("
        f"refsnp_event_id={json.dumps(refsnp_event_id)}, "
        f"assembly_accession={json.dumps(accession)})"
    )


def select_investigation_sources(
    events: Iterable[Mapping[str, Any]],
    member_id: str,
    output: Mapping[str, Any],
) -> tuple[RefSnpToolResult, AssemblyReportToolResult | None]:
    """Resolve explicit member-owned observation citations, never positional results."""
    events = tuple(events)
    by_id = {
        event["event_id"]: event
        for event in events
        if event["payload"].get("collection_member_id") == member_id
    }
    refsnp_event = by_id.get(output.get("refsnp_event_id"))
    assembly_event = by_id.get(output.get("assembly_event_id"))
    if refsnp_event is None or refsnp_event["event_type"] != "RefSnpToolResultRecorded":
        # The refusal names the member's own lookup results to cite.
        own = [
            event["event_id"]
            for event in by_id.values()
            if event["event_type"] == "RefSnpToolResultRecorded"
        ]
        raise InvestigationProposalRefused(
            "refsnp_observation_not_owned",
            f"refsnp_event_id: expected={json.dumps(own[0])}"
            if len(own) == 1
            else f"refsnp_event_id: expected one of {json.dumps(own)}"
            if own
            else "refsnp_event_id: this member has no lookup result yet; look the "
            "variant up first",
        )
    refsnp = RefSnpToolResult.model_validate_json(
        canonical_json(refsnp_event["payload"]["result"])
    )
    if output.get("assembly_event_id") is not None and (
        assembly_event is None
        or assembly_event["event_type"] != "AssemblyReportToolResultRecorded"
        or assembly_event["payload"]["refsnp_event_id"] != refsnp_event["event_id"]
    ):
        raise InvestigationProposalRefused(
            "assembly_observation_basis_mismatch",
            _assembly_detail(events, member_id, refsnp_event["event_id"], refsnp),
        )
    assembly = (
        None
        if assembly_event is None
        else AssemblyReportToolResult.model_validate_json(
            canonical_json(assembly_event["payload"]["result"])
        )
    )
    return refsnp, assembly


def latest_lookup_results(
    timeline: Iterable[Mapping[str, Any]],
) -> dict[str, RefSnpToolResult]:
    """Each output member's latest recorded lookup result, by its member id: what
    its check last read from a source, proposed or not (#460)."""
    return {
        event["payload"]["collection_member_id"]: RefSnpToolResult.model_validate_json(
            canonical_json(event["payload"]["result"])
        )
        for event in timeline
        if event["event_type"] == "RefSnpToolResultRecorded"
    }


def gene_change_events(
    events, event_type: str, member_id: str, revision_number: int
) -> tuple[dict, ...]:
    """One member's gene-change placements or picks in one revision, in order (#473)."""
    return tuple(
        event
        for event in events
        if event["event_type"] == event_type
        and event["payload"]["collection_member_id"] == member_id
        and event["payload"]["revision_number"] == revision_number
    )


def target_reads(events, member_id: str, revision_number: int) -> tuple[dict, ...]:
    """The payloads of one member's target readings in one revision, in order."""
    return tuple(
        event["payload"]
        for event in events
        if event["event_type"] == "InvestigationTargetRead"
        and event["payload"]["collection_member_id"] == member_id
        and event["payload"]["revision_number"] == revision_number
    )


def member_answers(
    events, request: InvestigationRequestItem, member_id: str, revision_number: int
) -> tuple[dict, ...]:
    """One member's instruction answers in one revision, each with the target the
    researcher or the model typed for it (`typed_answers`, #472 Q3 a)."""
    answers = [
        {**event["payload"]["response"], "response_event_id": event["event_id"]}
        for event in events
        if event["event_type"] == "InvestigationReviewResponded"
        and event["payload"]["response"]["action"] in RESEARCHER_INSTRUCTION_ACTIONS
        and event["payload"]["response"]["collection_member_id"] == member_id
        and event["payload"]["response"]["revision_number"] == revision_number
    ]
    return typed_answers(
        request,
        answers,
        target_reads(events, member_id, revision_number),
        tuple(
            event["payload"]
            for event in gene_change_events(
                events, "InvestigationGeneChangePicked", member_id, revision_number
            )
        ),
    )


_DECISION_TOOLS = {
    "sequence": "name_sequence",
    "version": "name_version",
    "placement": "name_placement",
    "listed_none": "quote_listed_none",
}


def _decision_arguments(fact: InvestigationLookupDecided) -> dict[str, Any]:
    """The call that records a decision, as admitted: its cite and its decision."""
    decision = fact.decision
    if isinstance(decision, (SequenceDecision, VersionDecision)):
        values = decision.choice.model_dump(mode="json")
    elif isinstance(decision, PlacementDecision):
        values = {"placement": decision.placement}
    else:
        values = decision.model_dump(mode="json", exclude={"kind"})
    return {"refsnp_event_id": fact.refsnp_event_id, **values}


def check_lookup_decision(prefix, candidate, owner) -> None:
    """Before a decision is appended, its own reopen check runs on it (#479 G2.2,
    #476 C8), its call read from the call's start."""
    from bio_neb.agentic_journey.tool_execution import tool_call_lineage

    candidate = {**candidate, "event_id": str(uuid.uuid4())}
    timeline = [*prefix, candidate]
    by_id = {event["event_id"]: event for event in timeline}
    fact = validate_investigation_fact(candidate)
    _validate_lookup_decision(
        candidate,
        fact,
        timeline,
        by_id,
        _member_call(
            tool_call_lineage(fact.attempt_id, fact.tool_call_id, prefix, by_id),
            fact.attempt_id,
        ),
    )


def _validate_lookup_decisions(timeline, facts, by_id) -> None:
    for event, fact in facts:
        if isinstance(fact, InvestigationLookupDecided):
            _validate_lookup_decision(
                event,
                fact,
                timeline,
                by_id,
                _observed_member_call(event, fact.attempt_id, timeline, by_id),
            )


def _validate_lookup_decision(event, fact, timeline, by_id, call) -> None:
    """A decision is its own admitted call's, made for its bound item, and names one
    of the options of the item's lookup result it cites, which asks for it (#479
    G2.2; ADR 0034 call binding, ADR 0049 membership)."""
    from bio_neb.investigation.item_history import item_lineage, item_read_attempts

    prefix = timeline[: timeline.index(event)]
    bound = next(
        (
            parse
            for item in prefix
            if item["event_type"] == "InvestigationAttemptBound"
            and (parse := validate_investigation_fact(item)).attempt_id
            == fact.attempt_id
        ),
        None,
    )
    cited = by_id.get(fact.refsnp_event_id)
    kind = fact.decision.kind
    if (
        bound is None
        or bound.collection_member_id != fact.collection_member_id
        or bound.revision_number != fact.revision_number
        or call.validation["tool_name"] != _DECISION_TOOLS[kind]
        or call.validation["tool_call_id"] != fact.tool_call_id
        or call.validation["arguments"] != _decision_arguments(fact)
        or cited is None
        or cited not in prefix
        or cited["event_type"] != "RefSnpToolResultRecorded"
        or cited["payload"]["collection_member_id"] != fact.collection_member_id
        or cited["payload"]["attempt_id"]
        not in item_read_attempts(prefix, fact.attempt_id)
    ):
        raise ValueError("lookup decision changed its call or cited result")
    result = RefSnpToolResult.model_validate_json(
        json.dumps(cited["payload"]["result"])
    )
    origin = result.origin
    decision = fact.decision
    if result.status != DECISION_ASKS[kind] or (
        kind == "listed_none" and result.location == "not_applicable"
    ):
        raise ValueError("lookup decision cites a result that does not ask for it")
    if isinstance(decision, PlacementDecision):
        listed = isinstance(origin, PlacementUnnamed) and (
            decision.placement in origin.sequences
        )
    elif isinstance(decision, ListedNoneDecision):
        listed = isinstance(origin, SourceAnswered) and (
            decision.quote in origin.words_sent
            if decision.quote is not None
            else answered_without_body(result, decision.status_code)
        )
    elif isinstance(decision, SequenceDecision):
        member = _bound_member(bound, fact.collection_member_id)
        answers = member_answers(
            prefix, member, fact.collection_member_id, fact.revision_number
        )
        choice = decision.choice
        listed = (
            isinstance(origin, SequenceUnnamed)
            and choice.assembly_accession in origin.assemblies
            and origin.contig in choice.quote
            and any(
                choice.quote in words for words in researcher_words(member, answers)
            )
        )
    else:
        member = _bound_member(bound, fact.collection_member_id)
        answers = member_answers(
            prefix, member, fact.collection_member_id, fact.revision_number
        )
        owned, members = item_lineage(prefix, fact.attempt_id)
        mine = [
            item
            for item in prefix
            if item["payload"].get("attempt_id") in owned
            and item["payload"].get("collection_member_id") in members
        ]
        listed = not isinstance(
            named_version(
                effective_location(member, answers).stated_assembly,
                (
                    AssemblySearchPage.model_validate_json(
                        json.dumps(item["payload"]["result"])
                    )
                    for item in mine
                    if item["event_type"] == "AssembliesSearched"
                ),
                [i for i in mine if i["event_type"] == "RefSnpToolResultRecorded"],
                _run_accession(prefix),
                decision.choice,
            ),
            str,
        )
    if not listed:
        raise ValueError("lookup decision names no option of its cited result")


def _bound_member(bound, member_id):
    return next(
        item
        for ordinal, item in enumerate(bound.work_requests)
        if investigation_output_member_id(item, ordinal) == member_id
    )


def _run_accession(prefix) -> str:
    return next(
        item["payload"]["investigation_input"]["assembly_accession"]
        for item in prefix
        if item["event_type"] == "InvestigationStarted"
    )


def _validate_target_reads(timeline, facts, bound_by_attempt) -> None:
    """Each reading reads its member's own words, unaltered (#472 Q3 a, R1 b): an
    earlier instruction answer to that member, in its revision, the researcher typed
    no target for, or the Investigation's start for an untyped request item's words."""
    started = {
        event["event_id"]: fact
        for event, fact in facts
        if isinstance(fact, InvestigationStarted)
    }
    for event, fact in facts:
        if not isinstance(fact, InvestigationTargetRead):
            continue
        bound = bound_by_attempt.get(fact.attempt_id)
        prefix = timeline[: timeline.index(event)]
        source = next(
            (item for item in prefix if item["event_id"] == fact.source_event_id), None
        )
        if fact.source == "answer":
            response = (
                None
                if source is None
                or source["event_type"] != "InvestigationReviewResponded"
                else source["payload"]["response"]
            )
            reads_its_words = (
                response is not None
                and response["action"] in RESEARCHER_INSTRUCTION_ACTIONS
                and response["collection_member_id"] == fact.collection_member_id
                and response["revision_number"] == fact.revision_number
                and "target" not in response
                and response["instruction"] == fact.words
            )
        else:
            start = started.get(fact.source_event_id)
            reads_its_words = (
                source is not None
                and start is not None
                and any(
                    investigation_output_member_id(item, ordinal)
                    == fact.collection_member_id
                    and untyped(item)
                    and item.request_text == fact.words
                    for ordinal, item in enumerate(_assertion_requests(start.assertion))
                )
            )
        if (
            bound is None
            or bound.collection_member_id != fact.collection_member_id
            or bound.revision_number != fact.revision_number
            or not reads_its_words
        ):
            raise ValueError("Investigation target reading changed the words it reads")


def _validate_gene_changes(
    timeline, facts, bound_by_attempt, source_events, by_id
) -> None:
    """Each placement is its bound member's, made by its own admitted
    place_gene_change call on its last retained GET, its claim the member's change on
    the transcript that call named; each pick is its own admitted call's arguments,
    names a candidate an earlier placement of that member, of the same gene change,
    copied, and follows exactly the member's answers before it (#473 Q4 a, Q11; ADR
    0034 call binding, ADR 0049 membership)."""
    placements = {}
    for event, fact in facts:
        if isinstance(fact, InvestigationGeneChangePlaced):
            bound = bound_by_attempt.get(fact.attempt_id)
            call = _observed_member_call(event, fact.attempt_id, timeline, by_id)
            named = call.validation["arguments"].get("transcript")
            claimed = fact.result.claim.change
            if (
                bound is None
                or bound.collection_member_id != fact.collection_member_id
                or bound.revision_number != fact.revision_number
                or fact.source_event_id not in source_events
                or call.started["tool_call_id"] != fact.tool_call_id
                or call.started["tool_name"] != "place_gene_change"
                or claimed
                != (
                    fact.gene_change
                    if named is None
                    else fact.gene_change.model_copy(update={"transcript": named})
                )
            ):
                raise ValueError("gene-change placement changed its member or call")
            placements[event["event_id"]] = fact
        elif isinstance(fact, InvestigationGeneChangePicked):
            bound = bound_by_attempt.get(fact.attempt_id)
            placed = placements.get(fact.placement_event_id)
            call = _observed_member_call(event, fact.attempt_id, timeline, by_id)
            arguments = call.validation["arguments"]
            candidates = (
                set()
                if placed is None
                else {
                    (item.contig, item.position, item.ref, item.alt)
                    for item in placed.result.candidates
                }
            )
            picked = (
                fact.placed.contig,
                fact.placed.position,
                fact.placed.ref,
                fact.placed.alt,
            )
            if (
                bound is None
                or placed is None
                or call.started["tool_name"] != "pick_gene_change_candidate"
                or (
                    arguments.get("placement_event_id"),
                    tuple(
                        arguments.get(name)
                        for name in ("contig", "position", "ref", "alt")
                    ),
                    arguments.get("reason"),
                )
                != (fact.placement_event_id, picked, fact.reason)
                or bound.collection_member_id != fact.collection_member_id
                or bound.revision_number != fact.revision_number
                or placed.collection_member_id != fact.collection_member_id
                or placed.revision_number != fact.revision_number
                or placed.gene_change != fact.gene_change
                or picked not in candidates
                or fact.after_answer_event_ids
                != _member_answer_ids(
                    timeline[: timeline.index(event)],
                    fact.collection_member_id,
                    fact.revision_number,
                )
            ):
                raise ValueError(
                    "gene-change pick names no candidate its placement lists"
                )


def _member_answer_ids(events, member_id: str, revision_number: int):
    """The event ids of one member's instruction answers in one revision, sorted."""
    return tuple(
        sorted(
            event["event_id"]
            for event in events
            if event["event_type"] == "InvestigationReviewResponded"
            and event["payload"]["response"]["action"] in RESEARCHER_INSTRUCTION_ACTIONS
            and event["payload"]["response"]["collection_member_id"] == member_id
            and event["payload"]["response"]["revision_number"] == revision_number
        )
    )


def _assertion_requests(
    assertion: UserAssertion,
) -> tuple[InvestigationRequestItem, ...]:
    return assertion.requests or (
        InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        ),
    )


def _not_checked_members(
    timeline, facts, assertion: UserAssertion, bound
) -> dict[str, "InvestigationNotCheckedView"]:
    """Each original member recorded not checked (#367): once, on its start, with the
    reason its own request gives, and never bound into an Attempt."""
    requests = {
        investigation_output_member_id(request, ordinal): request
        for ordinal, request in enumerate(_assertion_requests(assertion))
    }
    start = next(e for e in timeline if e["event_type"] == "InvestigationStarted")
    attempted = {attempt.collection_member_id for attempt in bound}
    found: dict[str, InvestigationNotCheckedView] = {}
    for event, fact in facts:
        if not isinstance(fact, InvestigationRequestNotChecked):
            continue
        request = requests.get(fact.collection_member_id)
        if (
            request is None
            or not_checked_reason(request) != (fact.reason_code, fact.reason)
            or fact.collection_member_id in found
            or fact.collection_member_id in attempted
            or event.get("parent_ids") != [start["event_id"]]
        ):
            raise ValueError("Investigation recorded a checkable member as not checked")
        found[fact.collection_member_id] = InvestigationNotCheckedView(
            collection_member_id=fact.collection_member_id,
            outcome=InvestigationNotChecked(
                event_id=event["event_id"],
                reason_code=fact.reason_code,
                reason=fact.reason,
            ),
        )
    return found


def _rejected_request_member(request, ordinal, assertion_id, rejection):
    return InvestigationRequestMember(
        request_member_id=request_member_id(assertion_id, request, ordinal),
        request=request,
        refsnp=None,
        assembly_report=None,
        rejection=rejection,
    )


def _resolve_request_member(
    investigation_input: InvestigationInput,
    request: InvestigationRequestItem,
    pair: tuple[RefSnpToolResult, AssemblyReportToolResult | None],
    ordinal: int,
    assertion_id: str,
    issue_approval: InvestigationIssueApproval | None = None,
    target_asserted_by: str | None = None,
) -> tuple[InvestigationRequestMember, tuple[WorkItem, ...]]:
    refsnp, assembly = pair
    if (
        request.rsid is not None and refsnp.requested_rsid != request.rsid
    ) or refsnp.expected_assembly_accession != investigation_input.assembly_accession:
        raise InvestigationProposalRefused("assertion_source_mismatch")
    if refsnp.status == "region_located":
        # A located region is its one Work item, with no assembly report.
        if refsnp.location.region != request.region:
            raise InvestigationProposalRefused("assertion_target_mismatch")
        item = _region_work_item(investigation_input, refsnp)
        return InvestigationRequestMember(
            request_member_id=request_member_id(assertion_id, request, ordinal),
            request=request,
            refsnp=refsnp,
            assembly_report=None,
            work_item_ids=(item.work_item_id,),
            target_asserted_by=target_asserted_by,
        ), (item,)
    if refsnp.status in ADMISSIBLE_LOOKUP_STATUSES and assembly is None:
        raise InvestigationProposalRefused("assembly_report_missing")
    requested = _run_identity(refsnp)
    if refsnp.status == "resolved" and requested is not None:
        # A position that states its alleles keeps that allele's one candidate.
        if tuple(candidate.target_identity for candidate in refsnp.candidates) != (
            requested,
        ):
            raise InvestigationProposalRefused("assertion_target_mismatch")
    items = (
        _compile_investigation_work_items(
            investigation_input=investigation_input,
            requested_rsid=refsnp.requested_rsid,
            refsnp=refsnp,
            assembly_report=assembly,
        )
        if refsnp.status in ADMISSIBLE_LOOKUP_STATUSES
        and assembly is not None
        and assembly.status == "compatible"
        else ()
    )
    return InvestigationRequestMember(
        request_member_id=request_member_id(assertion_id, request, ordinal),
        request=request,
        refsnp=refsnp,
        assembly_report=assembly,
        issue_approval=issue_approval,
        work_item_ids=tuple(item.work_item_id for item in items),
        target_asserted_by=target_asserted_by,
    ), items


def _region_work_item(
    investigation_input: InvestigationInput, refsnp: RefSnpToolResult
) -> WorkItem:
    """A located region's one ``observe_region`` Work item (#367)."""
    location = refsnp.location
    material = {
        "assembly_accession": refsnp.expected_assembly_accession,
        "assembly_name": location.run_assembly_name,
        "contig": location.run_contig,
        "start": location.region.start,
        "end": location.region.end,
        "reference_bases": location.reference_bases,
        "assessment_ids": (refsnp.assessment.assessment_id,),
    }
    return WorkItem.create(
        sample_id=investigation_input.sample_id,
        target=RegionTarget(target_id=_digest("region-target", material), **material),
        intended_action="observe_region",
        source_ids=tuple(
            dict.fromkeys(source.source_id for source in _lookup_sources(refsnp))
        ),
    )


def _run_identity(refsnp: RefSnpToolResult) -> RequestedVariantIdentity | None:
    """The one allele a request's typed position names, on the Run's sequence name
    its position lookup found (#369); ``None`` when it names no allele or was not
    placed."""
    # The position the lookup placed: the member's, or its amend's typed one.
    location = refsnp.location
    if location == "not_applicable" or location.run_contig is None:
        return None
    identity = location.looked_up.identity
    return None if identity is None else identity._replace(contig=location.run_contig)


def _compile_investigation_work_items(
    *,
    investigation_input: InvestigationInput,
    requested_rsid: str | None,
    refsnp: RefSnpToolResult,
    assembly_report: AssemblyReportToolResult,
) -> tuple[WorkItem, ...]:
    """Admit one source pair into one candidate Work item per source-backed ALT."""

    if refsnp.requested_rsid != requested_rsid:
        raise InvestigationProposalRefused("assertion_source_mismatch")
    if (
        refsnp.expected_assembly_accession != investigation_input.assembly_accession
        or assembly_report.expected_assembly_accession
        != investigation_input.assembly_accession
    ):
        raise InvestigationProposalRefused("assembly_input_mismatch")
    if refsnp.status not in ADMISSIBLE_LOOKUP_STATUSES:
        raise InvestigationProposalRefused("refsnp_unresolved", refsnp.status)
    if assembly_report.status != "compatible":
        raise InvestigationProposalRefused(
            "assembly_report_unresolved", assembly_report.status
        )
    candidate = refsnp.candidates[0]
    if (
        candidate.assembly_accession != assembly_report.observed_assembly_accession
        or candidate.assembly_name != assembly_report.observed_assembly_name
    ):
        raise InvestigationProposalRefused("assembly_source_conflict")
    assessment_ids = (
        refsnp.assessment.assessment_id,
        assembly_report.assessment.assessment_id,
    )
    items = []
    for candidate in refsnp.candidates:
        identity = candidate.target_identity
        target_material = {
            **candidate.model_dump(mode="python"),
            "one_based_position": identity.one_based_position,
            "ref": identity.ref,
            "alt": identity.alt,
            "assessment_ids": assessment_ids,
        }
        items.append(
            WorkItem.create(
                sample_id=investigation_input.sample_id,
                target=VariantTarget(
                    target_id=_digest("variant-target", target_material),
                    **target_material,
                ),
                intended_action="observe_variant",
                # Every GET behind the target, a position's placement among them.
                source_ids=tuple(
                    dict.fromkeys(
                        (
                            *(source.source_id for source in _lookup_sources(refsnp)),
                            assembly_report.source.source_id,
                        )
                    )
                ),
            )
        )
    return tuple(items)


__all__ = [
    "AddInvestigationMember",
    "AdjustInvestigationSearch",
    "AmendInvestigationFindingResponse",
    "AmendInvestigationMember",
    "RetryInvestigationMember",
    "ApproveInvestigationFindingResponse",
    "ApproveInvestigationResponse",
    "AssemblyReportToolResult",
    "AssemblyStatus",
    "BodylessAnswer",
    "AssemblyReportToolResultRecorded",
    "ClarifyInvestigationMember",
    "DeferInvestigationMember",
    "EvidenceAssessment",
    "InvestigationAuthority",
    "InvestigationAuthorityGranted",
    "InvestigationAssertionRecorded",
    "InvestigationAttemptBound",
    "InvestigationAttemptFailed",
    "InvestigationCollectionRevision",
    "InvestigationCollectionRevisionRecorded",
    "InvestigationInput",
    "InvestigationIssueApproval",
    "InvestigationIssueItem",
    "InvestigationProposal",
    "InvestigationProposalCarriedForward",
    "InvestigationRequest",
    "InvestigationRequestItem",
    "InvestigationProposalRecorded",
    "InvestigationProposalRefused",
    "InvestigationProjection",
    "InvestigationReviewRequested",
    "InvestigationReviewResponded",
    "InvestigationReviewResponse",
    "InvestigationRevisionMember",
    "InvestigationRevisionOperation",
    "InvestigationSourceContactFailed",
    "InvestigationSourceContactStarted",
    "InvestigationSourceRetrieved",
    "InvestigationStarted",
    "InvestigationStopped",
    "InvestigationVcfIdentity",
    "KeepInvestigationMember",
    "RETRIEVED_TARGET_NOT_REQUESTED",
    "RefSnpToolResult",
    "RefSnpToolResultRecorded",
    "ResolvedVariantCandidate",
    "RemoveInvestigationMember",
    "RetrievedSource",
    "ReviseInvestigationResponse",
    "StopInvestigationResponse",
    "UserAssertion",
    "VariantTarget",
    "WorkItem",
    "compile_investigation_collection_proposal",
    "compile_investigation_collection_revision",
    "compile_investigation_proposal",
    "investigation_issue_items",
    "project_investigation",
    "project_snapshot_investigation",
    "request_member_id",
    "validate_investigation_fact",
]
