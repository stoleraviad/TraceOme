"""Grounded sample, taxon, and reference-assembly Tool contracts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol, Tuple, Union, runtime_checkable

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.contracts import FrozenContract
from bio_neb.failure import ProviderError
from bio_neb.identifier_rule import IdentifierRule

_SUBJECT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_PLAN_ID = re.compile(r"^plan_[0-9a-f]{64}$")
_REQUIREMENT_ID = re.compile(r"^requirement_[0-9a-f]{32}$")
_SAMPLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TAXON_CANDIDATE = re.compile(r"^taxon:ncbitaxon:([1-9][0-9]*)$")
# The one assembly accession grammar: a versioned GCA_ or GCF_ accession.
_ACCESSION = r"GC[AF]_[0-9]{9}\.[1-9][0-9]*"
ASSEMBLY_ACCESSION_PATTERN = rf"^{_ACCESSION}$"
_ASSEMBLY_ACCESSION = re.compile(ASSEMBLY_ACCESSION_PATTERN)
_STATED_ASSEMBLY_ACCESSION = re.compile(rf"\b{_ACCESSION}\b")

_SAMPLE_TOKEN = re.compile(r"[A-Za-z0-9._-]+")
# A whole number: not part of a word, an identifier or a decimal.
_WHOLE_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[1-9][0-9]*(?![A-Za-z0-9_]|\.[0-9])")


def _stated_sample_identifiers(text: str) -> Tuple[str, ...]:
    """Every whole token in the sample grammar, exactly as written."""

    tokens = (match.group() for match in _SAMPLE_TOKEN.finditer(text))
    return tuple(
        dict.fromkeys(token for token in tokens if _SAMPLE_IDENTIFIER.fullmatch(token))
    )


def _stated_tax_ids(text: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(_WHOLE_NUMBER.findall(text)))


def _stated_accessions(text: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(_STATED_ASSEMBLY_ACCESSION.findall(text)))


def candidate_value(candidate_id: str) -> str | None:
    """The typed value a Review candidate this system minted carries (an assembly's
    accession, a taxon's TaxId), or None for any other candidate: a selected
    candidate's value is copied from its id, never read from its label (#472 Q2 a)."""
    if candidate_id.startswith("assembly:"):
        return candidate_id.removeprefix("assembly:")
    if (match := _TAXON_CANDIDATE.fullmatch(candidate_id)) is not None:
        return match.group(1)
    return None


# A sample name has no source that can verify it: it stays the researcher's.
SAMPLE_IDENTIFIER_RULE = IdentifierRule(
    name="sample_identifier", stated=_stated_sample_identifiers, closing="."
)
# Format only: a name such as "human" and merged or secondary ids are NCBI
# Taxonomy's to resolve.
NCBI_TAXON_ID_RULE = IdentifierRule(
    name="ncbi_taxon_id", stated=_stated_tax_ids, source="NCBI Taxonomy"
)
# Versioned and exact: a file name, an unversioned or a lowercase form states none.
ASSEMBLY_ACCESSION_RULE = IdentifierRule(
    name="assembly_accession", stated=_stated_accessions, source="NCBI Datasets"
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


RequirementKind = Literal[
    "sample_identity", "taxon_identity", "reference_assembly_identity"
]


class RequirementGrounding(FrozenContract):
    """Exact Journey-derived subject against which a Tool call is checked."""

    journey_id: StrictStr
    plan_id: StrictStr
    requirement_id: StrictStr
    kind: RequirementKind
    subject_key: StrictStr
    source_event_ids: Tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("journey_id")
    @classmethod
    def canonical_journey_id(cls, value: str) -> str:
        return _canonical_uuid(value, "journey_id")

    @field_validator("source_event_ids")
    @classmethod
    def canonical_source_event_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("source_event_ids must be unique")
        return tuple(_canonical_uuid(item, "source_event_id") for item in value)

    @field_validator("plan_id")
    @classmethod
    def canonical_plan_id(cls, value: str) -> str:
        if not _PLAN_ID.fullmatch(value):
            raise ValueError("plan_id must be a canonical grounded plan identity")
        return value

    @field_validator("requirement_id")
    @classmethod
    def canonical_requirement_id(cls, value: str) -> str:
        if not _REQUIREMENT_ID.fullmatch(value):
            raise ValueError("requirement_id must be canonical")
        return value

    @field_validator("subject_key")
    @classmethod
    def canonical_subject_key(cls, value: str) -> str:
        if not _SUBJECT_KEY.fullmatch(value):
            raise ValueError("subject_key must be lowercase snake case")
        return value


class ReferenceToolCallRejected(ValueError):
    """A schema-valid call disagreed with the current Journey-derived contract."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SourceObservation(FrozenContract):
    authority: Literal["ncbi_taxonomy", "ncbi_datasets"]
    adapter_version: StrictStr
    source_version: StrictStr
    request_identity: StrictStr = Field(pattern=r"^https://", max_length=1000)
    response_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: datetime
    citation_url: StrictStr = Field(pattern=r"^https://", max_length=1000)
    elapsed_time_ms: StrictInt = Field(ge=0)
    response_bytes: StrictInt = Field(ge=0)

    @field_validator("adapter_version", "source_version")
    @classmethod
    def canonical_versions(cls, value: str, info) -> str:
        return _canonical_text(value, info.field_name)

    @field_validator("observed_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value

    @property
    def observation_id(self) -> str:
        return _evidence_id(
            "source-observation",
            self.model_dump(mode="json"),
        )


class ReferenceAdapterFailure(FrozenContract):
    failure_class: Literal["transient", "exhausted", "rejected", "semantic"] = (
        "semantic"
    )
    original_failure_class: Literal["transient", "rejected", "semantic"] | None = None
    contact_attempt: int = Field(default=1, ge=1)
    # ``source_answered``: the source's own answer about the request, a client
    # status with its body, which the model reads.
    code: Literal[
        "transport_error",
        "http_error",
        "source_answered",
        "response_too_large",
        "invalid_response",
        "source_disagreement",
        "adapter_output_invalid",
    ]
    public_message: StrictStr
    retryable: StrictBool

    @field_validator("public_message")
    @classmethod
    def canonical_message(cls, value: str) -> str:
        value = _canonical_text(value, "public_message")
        if len(value) > 500:
            raise ValueError("public_message exceeds its bounded contract")
        return value


class ReferenceAdapterError(ProviderError):
    def __init__(
        self,
        code: Literal[
            "transport_error",
            "http_error",
            "source_answered",
            "response_too_large",
            "invalid_response",
            "source_disagreement",
        ],
        public_message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(
            "transient"
            if retryable
            else "semantic"
            if code in {"invalid_response", "source_disagreement"}
            else "rejected",
            public_message,
            evidence={"source_code": code},
        )
        self._failure = ReferenceAdapterFailure(
            failure_class=self.failure_class,
            code=code,
            public_message=public_message[:500],
            retryable=retryable,
        )

    @property
    def failure(self):
        return self._failure.model_copy(
            update={
                "failure_class": self.failure_class,
                "original_failure_class": self.evidence.get(
                    "original_failure_class", self.failure_class
                ),
                "contact_attempt": self.evidence.get("contact_attempt", 1),
            }
        )


class ReferenceQueryRefused(Exception):
    """The port refused a query its source cannot read as one request, before contact.

    Not an operational fault: admission asks the port first and returns the
    refusal to the model, so a refused query never reaches the source.
    """

    def __init__(self, code: str, term: str) -> None:
        super().__init__(code)
        self.code = code
        self.term = term


class TaxonomySourceRecord(FrozenContract):
    ncbi_tax_id: StrictInt = Field(gt=0)
    scientific_name: StrictStr = Field(min_length=1, max_length=300)
    rank: StrictStr | None = Field(default=None, min_length=1, max_length=100)
    lineage: Tuple[StrictStr, ...] = ()
    lineage_tax_ids: Tuple[StrictInt, ...] = ()
    merged_or_secondary_tax_ids: Tuple[StrictInt, ...] = ()
    provisional: StrictBool = False
    deleted: StrictBool = False

    @model_validator(mode="after")
    def coherent_status(self) -> "TaxonomySourceRecord":
        if self.provisional and self.deleted:
            raise ValueError("taxonomy record cannot be provisional and deleted")
        if any(
            value <= 0
            for value in (*self.lineage_tax_ids, *self.merged_or_secondary_tax_ids)
        ):
            raise ValueError("taxonomy identifiers must be positive")
        return self


class TaxonomyAdapterObservation(FrozenContract):
    records: Tuple[TaxonomySourceRecord, ...] = Field(max_length=10)
    source: SourceObservation

    @model_validator(mode="after")
    def taxonomy_source(self) -> "TaxonomyAdapterObservation":
        if self.source.authority != "ncbi_taxonomy":
            raise ValueError("taxonomy observation requires NCBI Taxonomy authority")
        tax_ids = [item.ncbi_tax_id for item in self.records]
        if len(tax_ids) != len(set(tax_ids)):
            raise ValueError("taxonomy observation contains duplicate TaxIds")
        return self


_SHORT_TEXT = Field(default=None, min_length=1, max_length=300)


class InfraspecificNames(FrozenContract):
    """NCBI's ``organism.infraspecific_names``, every field copied as NCBI writes it."""

    strain: StrictStr | None = _SHORT_TEXT
    breed: StrictStr | None = _SHORT_TEXT
    cultivar: StrictStr | None = _SHORT_TEXT
    ecotype: StrictStr | None = _SHORT_TEXT
    isolate: StrictStr | None = _SHORT_TEXT
    sex: StrictStr | None = _SHORT_TEXT


class AssemblySourceRecord(FrozenContract):
    """One row of an NCBI Datasets genome report, copied: every field NCBI writes
    for it, as it writes it, with no reading of its status, category or level
    (ADR 0049). Only the accession is required; a field NCBI leaves out stays
    out."""

    assembly_accession: StrictStr
    current_accession: StrictStr | None = None
    paired_accession: StrictStr | None = None
    source_database: StrictStr | None = _SHORT_TEXT
    organism_name: StrictStr | None = _SHORT_TEXT
    common_name: StrictStr | None = _SHORT_TEXT
    ncbi_tax_id: StrictInt | None = Field(default=None, gt=0)
    infraspecific_names: InfraspecificNames = InfraspecificNames()
    assembly_name: StrictStr | None = _SHORT_TEXT
    synonym: StrictStr | None = _SHORT_TEXT
    assembly_status: StrictStr | None = _SHORT_TEXT
    paired_assembly_status: StrictStr | None = _SHORT_TEXT
    suppression_reason: StrictStr | None = Field(
        default=None, min_length=1, max_length=2000
    )
    refseq_category: StrictStr | None = _SHORT_TEXT
    assembly_level: StrictStr | None = _SHORT_TEXT
    assembly_type: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description=(
            "How the assembly represents the genome (NCBI writes haploid for diploid "
            "Arabidopsis TAIR10), not the organism's ploidy."
        ),
    )
    release_date: StrictStr | None = _SHORT_TEXT
    submitter: StrictStr | None = _SHORT_TEXT
    genome_notes: Tuple[StrictStr, ...] = ()

    @field_validator("assembly_accession", "current_accession", "paired_accession")
    @classmethod
    def versioned_accession(cls, value: str | None) -> str | None:
        if value is not None and not _ASSEMBLY_ACCESSION.fullmatch(value):
            raise ValueError("assembly identity must be an exact accession.version")
        return value


# NCBI Datasets' documented values for its genome-report query options
# (`/genome/taxon/{taxons}/dataset_report`, OpenAPI v2), sent as NCBI spells them.
AssemblySource = Literal["all", "refseq", "genbank"]
AssemblyVersion = Literal["current", "all_assemblies"]
AssemblyLevel = Literal["chromosome", "scaffold", "contig", "complete_genome"]
MetagenomeDerived = Literal[
    "METAGENOME_DERIVED_UNSET", "metagenome_derived_only", "metagenome_derived_exclude"
]
TypeMaterialCategory = Literal[
    "NONE",
    "TYPE_MATERIAL",
    "TYPE_MATERIAL_CLADE",
    "TYPE_MATERIAL_NEOTYPE",
    "TYPE_MATERIAL_REFTYPE",
    "PATHOVAR_TYPE",
    "TYPE_MATERIAL_SYN",
]
SortDirection = Literal[
    "SORT_DIRECTION_UNSPECIFIED",
    "SORT_DIRECTION_ASCENDING",
    "SORT_DIRECTION_DESCENDING",
]
# One page shows at most this many rows: the context-safety bound on what one Tool
# result puts before the model; NCBI's own maximum is 1000, and the rest of a
# listing stays reachable by its page token.
ASSEMBLY_PAGE_ROWS = 100
_ISO_DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"


class AssemblyQuery(FrozenContract):
    """The model's NCBI Datasets genome-report query: NCBI's own options, each
    with its documented values and sent verbatim (owner #471 R1 a, #469 SP4). An
    option left out is NCBI's default; code adds no filter of its own."""

    reference_only: StrictBool | None = None
    assembly_source: AssemblySource | None = None
    has_annotation: StrictBool | None = None
    exclude_paired_reports: StrictBool | None = None
    exclude_atypical: StrictBool | None = None
    assembly_version: AssemblyVersion | None = None
    assembly_level: Tuple[AssemblyLevel, ...] = ()
    first_release_date: StrictStr | None = Field(default=None, pattern=_ISO_DATE)
    last_release_date: StrictStr | None = Field(default=None, pattern=_ISO_DATE)
    search_text: Tuple[StrictStr, ...] = ()
    is_metagenome_derived: MetagenomeDerived | None = None
    is_type_material: StrictBool | None = None
    is_ictv_exemplar: StrictBool | None = None
    exclude_multi_isolate: StrictBool | None = None
    type_material_category: TypeMaterialCategory | None = None
    tax_exact_match: StrictBool | None = None
    sort_field: StrictStr | None = Field(default=None, min_length=1, max_length=100)
    sort_direction: SortDirection | None = None
    page_size: StrictInt = Field(
        default=ASSEMBLY_PAGE_ROWS, ge=1, le=ASSEMBLY_PAGE_ROWS
    )
    page_token: StrictStr | None = Field(default=None, min_length=1, max_length=2000)

    @field_validator("search_text")
    @classmethod
    def plain_search_text(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        for text in value:
            _canonical_text(text, "search_text")
        return value

    def parameters(self) -> tuple[tuple[str, str], ...]:
        """The query's parameters in NCBI's documented order and spelling, each list
        option repeated per value (OpenAPI form style)."""
        names = (
            ("reference_only", "filters.reference_only"),
            ("assembly_source", "filters.assembly_source"),
            ("has_annotation", "filters.has_annotation"),
            ("exclude_paired_reports", "filters.exclude_paired_reports"),
            ("exclude_atypical", "filters.exclude_atypical"),
            ("assembly_version", "filters.assembly_version"),
            ("assembly_level", "filters.assembly_level"),
            ("first_release_date", "filters.first_release_date"),
            ("last_release_date", "filters.last_release_date"),
            ("search_text", "filters.search_text"),
            ("is_metagenome_derived", "filters.is_metagenome_derived"),
            ("is_type_material", "filters.is_type_material"),
            ("is_ictv_exemplar", "filters.is_ictv_exemplar"),
            ("exclude_multi_isolate", "filters.exclude_multi_isolate"),
            ("type_material_category", "filters.type_material_category"),
            ("tax_exact_match", "tax_exact_match"),
            ("sort_field", "sort.field"),
            ("sort_direction", "sort.direction"),
            ("page_size", "page_size"),
            ("page_token", "page_token"),
        )
        pairs: list[tuple[str, str]] = []
        for field, name in names:
            value = getattr(self, field)
            values = value if isinstance(value, tuple) else (value,)
            pairs.extend(
                (name, str(item).lower() if isinstance(item, bool) else str(item))
                for item in values
                if item is not None
            )
        return tuple(pairs)


# Each query option as a Tool argument, with NCBI's documented description and
# values (OpenAPI v2), shared by planning's `discover_assemblies` and the
# Investigation's `search_assemblies` so both offer the model the same query.
def _option(description: str) -> Any:
    return Field(default=None, description=description)


ReferenceOnlyOption = Annotated[
    bool | None,
    _option("filters.reference_only: if true, limit to reference genome assemblies."),
]
AssemblySourceOption = Annotated[
    AssemblySource | None,
    _option(
        "filters.assembly_source: limit to RefSeq (GCF_) or GenBank (GCA_) genome "
        "assemblies; one of all, refseq, genbank (NCBI's default: all)."
    ),
]
HasAnnotationOption = Annotated[
    bool | None,
    _option("filters.has_annotation: limit to annotated genome assemblies."),
]
ExcludePairedReportsOption = Annotated[
    bool | None,
    _option(
        "filters.exclude_paired_reports: if true, for GenBank (GCA_)/RefSeq (GCF_) "
        "pairs, return the RefSeq copy; when no RefSeq copy exists, the GenBank one."
    ),
]
ExcludeAtypicalOption = Annotated[
    bool | None,
    _option("filters.exclude_atypical: if true, exclude atypical genome assemblies."),
]
AssemblyVersionOption = Annotated[
    AssemblyVersion | None,
    _option(
        "filters.assembly_version: current (NCBI's default) limits to the most recent "
        "version of each assembly; all_assemblies lists every version. An earlier "
        "build or patch that NCBI replaced under the same accession is listed only "
        "with all_assemblies, so a search_text that finds nothing under current may "
        "find it there."
    ),
]
AssemblyLevelOption = Annotated[
    list[AssemblyLevel] | None,
    _option(
        "filters.assembly_level: limit to genomes at these assembly levels; any of "
        "chromosome, scaffold, contig, complete_genome."
    ),
]
FirstReleaseDateOption = Annotated[
    str | None,
    Field(
        default=None,
        pattern=_ISO_DATE,
        description="filters.first_release_date: genomes released on or after this date (YYYY-MM-DD).",
    ),
]
LastReleaseDateOption = Annotated[
    str | None,
    Field(
        default=None,
        pattern=_ISO_DATE,
        description="filters.last_release_date: genomes released on or before this date (YYYY-MM-DD).",
    ),
]
SearchTextOption = Annotated[
    list[str] | None,
    _option(
        "filters.search_text: limit to genomes that match this text in the submitter "
        "name, assembly name, infraspecific name (or modifier), or organism name; NCBI "
        "does not search UCSC names such as hg19."
    ),
]
MetagenomeDerivedOption = Annotated[
    MetagenomeDerived | None,
    _option(
        "filters.is_metagenome_derived: exclude or limit to metagenome-assembled "
        "genomes; one of METAGENOME_DERIVED_UNSET (NCBI's default), "
        "metagenome_derived_only, metagenome_derived_exclude."
    ),
]
TypeMaterialOption = Annotated[
    bool | None,
    _option(
        "filters.is_type_material: if true, limit to genomes derived from type material."
    ),
]
IctvExemplarOption = Annotated[
    bool | None,
    _option(
        "filters.is_ictv_exemplar: if true, limit to genomes derived from an ICTV "
        "exemplar (virus genomes only)."
    ),
]
ExcludeMultiIsolateOption = Annotated[
    bool | None,
    _option(
        "filters.exclude_multi_isolate: if true, exclude genomes sequenced as part "
        "of large multi-isolate projects."
    ),
]
TypeMaterialCategoryOption = Annotated[
    TypeMaterialCategory | None,
    _option(
        "filters.type_material_category: limit to genomes of one type material "
        "category; one of NONE, TYPE_MATERIAL, TYPE_MATERIAL_CLADE, "
        "TYPE_MATERIAL_NEOTYPE, TYPE_MATERIAL_REFTYPE, PATHOVAR_TYPE, TYPE_MATERIAL_SYN."
    ),
]
TaxExactMatchOption = Annotated[
    bool | None,
    _option(
        "tax_exact_match: if true, only assemblies of the taxon itself, not of its "
        "child taxa (NCBI's default lists child taxa too)."
    ),
]
SortFieldOption = Annotated[
    str | None,
    Field(
        default=None,
        min_length=1,
        max_length=100,
        description="sort.field: the report field NCBI sorts by.",
    ),
]
SortDirectionOption = Annotated[
    SortDirection | None,
    _option(
        "sort.direction: one of SORT_DIRECTION_UNSPECIFIED, SORT_DIRECTION_ASCENDING, "
        "SORT_DIRECTION_DESCENDING."
    ),
]
PageSizeOption = Annotated[
    int,
    Field(
        default=ASSEMBLY_PAGE_ROWS,
        ge=1,
        le=ASSEMBLY_PAGE_ROWS,
        description=(
            "page_size: how many reports this page shows, 1 to 100 (default 100). "
            "NCBI allows 1000; 100 is the bound on what one result puts before you, "
            "and the rest stays reachable by page_token."
        ),
    ),
]
PageTokenOption = Annotated[
    str | None,
    Field(
        default=None,
        min_length=1,
        max_length=2000,
        description=(
            "page_token: the next_page_token of an earlier page; send it with the "
            "same options to read the next page."
        ),
    ),
]


def assembly_query(**options: Any) -> AssemblyQuery:
    """The typed query of a Tool call's NCBI options (list options as tuples)."""
    return AssemblyQuery(
        **{
            name: tuple(value) if isinstance(value, list) else value
            for name, value in options.items()
            if value is not None
        }
    )


class AssemblyAdapterObservation(FrozenContract):
    """One page of NCBI's genome reports, every row copied, with NCBI's own count of
    the whole listing and the token of its next page (ADR 0049 bounded views)."""

    records: Tuple[AssemblySourceRecord, ...]
    source: SourceObservation
    total_count: StrictInt | None = Field(default=None, ge=0)
    next_page_token: StrictStr | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def assembly_source(self) -> "AssemblyAdapterObservation":
        if self.source.authority != "ncbi_datasets":
            raise ValueError("assembly observation requires NCBI Datasets authority")
        return self


@runtime_checkable
class ReferenceIdentityAdapter(Protocol):
    contact_count: int

    def taxon_term_refusal(self, term: str) -> str | None:
        """The port's reason it cannot send ``term`` as one taxon query, if any."""
        ...

    def resolve_taxon(self, query: str) -> TaxonomyAdapterObservation: ...

    def discover_assemblies(
        self, ncbi_tax_id: int, query: AssemblyQuery
    ) -> AssemblyAdapterObservation: ...

    def describe_assembly(
        self, assembly_accession: str
    ) -> AssemblyAdapterObservation: ...


class SampleAttestationInput(FrozenContract):
    """The sample name the model typed from a researcher statement, with the words it
    quotes there (#472 Q2 a); the statement's event is the assertion."""

    grounding: RequirementGrounding
    input_set_id: StrictStr
    sample_identifier: StrictStr = Field(min_length=1, max_length=128)
    assertion_event_id: StrictStr
    quote: StrictStr = Field(min_length=1, max_length=300)

    @field_validator("input_set_id", "assertion_event_id")
    @classmethod
    def canonical_identities(cls, value: str, info) -> str:
        return _canonical_uuid(value, info.field_name)

    @model_validator(mode="after")
    def assertion_is_grounded(self) -> "SampleAttestationInput":
        if self.grounding.kind != "sample_identity":
            raise ValueError("sample Tool requires a sample_identity requirement")
        if self.assertion_event_id not in self.grounding.source_event_ids:
            raise ValueError("sample assertion event is outside requirement grounding")
        return self


class SampleAttestationResult(FrozenContract):
    status: Literal["verified", "negative"]
    resolution: Literal["bound", "invalid_format"]
    grounding: RequirementGrounding
    input_set_id: StrictStr
    sample_identifier: StrictStr
    assertion_event_id: StrictStr
    evidence_id: StrictStr
    summary: StrictStr

    @model_validator(mode="after")
    def coherent_result(self) -> "SampleAttestationResult":
        if (self.status, self.resolution) not in {
            ("verified", "bound"),
            ("negative", "invalid_format"),
        }:
            raise ValueError("sample result status and resolution disagree")
        return self


class VerifySampleAttestationTool:
    name = "verify_sample_attestation"

    def verify(
        self,
        tool_input: SampleAttestationInput,
        *,
        expected_grounding: RequirementGrounding,
        expected_input_set_id: str,
        expected_sample_identifier: str,
    ) -> SampleAttestationResult:
        _require_exact_grounding(tool_input.grounding, expected_grounding)
        if tool_input.input_set_id != expected_input_set_id:
            raise ReferenceToolCallRejected("sample_input_set_ungrounded")
        if tool_input.sample_identifier != expected_sample_identifier:
            raise ReferenceToolCallRejected("sample_identifier_ungrounded")
        valid = _SAMPLE_IDENTIFIER.fullmatch(tool_input.sample_identifier) is not None
        resolution = "bound" if valid else "invalid_format"
        payload = {
            "grounding": tool_input.grounding.model_dump(mode="json"),
            "input_set_id": tool_input.input_set_id,
            "sample_identifier": tool_input.sample_identifier,
            "assertion_event_id": tool_input.assertion_event_id,
            "resolution": resolution,
        }
        return SampleAttestationResult(
            status="verified" if valid else "negative",
            resolution=resolution,
            grounding=tool_input.grounding,
            input_set_id=tool_input.input_set_id,
            sample_identifier=tool_input.sample_identifier,
            assertion_event_id=tool_input.assertion_event_id,
            evidence_id=_evidence_id("sample-attestation", payload),
            summary=(
                "The exact researcher-attested sample identifier is bound to the registered input set."
                if valid
                else "The researcher-attested sample identifier does not satisfy the safe format."
            ),
        )


class ResolveTaxonInput(FrozenContract):
    grounding: RequirementGrounding
    query: StrictStr = Field(min_length=1, max_length=300)
    assertion_event_id: StrictStr

    @field_validator("query")
    @classmethod
    def canonical_query(cls, value: str) -> str:
        return _canonical_text(value, "query")

    @field_validator("assertion_event_id")
    @classmethod
    def canonical_assertion_event(cls, value: str) -> str:
        return _canonical_uuid(value, "assertion_event_id")

    @model_validator(mode="after")
    def assertion_is_grounded(self) -> "ResolveTaxonInput":
        if self.grounding.kind != "taxon_identity":
            raise ValueError("taxon Tool requires a taxon_identity requirement")
        if self.assertion_event_id not in self.grounding.source_event_ids:
            raise ValueError("taxon assertion event is outside requirement grounding")
        return self


class TaxonCandidate(FrozenContract):
    candidate_id: StrictStr = Field(pattern=r"^taxon:ncbitaxon:[1-9][0-9]*$")
    ncbi_tax_id: StrictInt = Field(gt=0)
    submitted_query: StrictStr
    scientific_name: StrictStr
    rank: StrictStr | None = None
    lineage: Tuple[StrictStr, ...] = ()
    merged_or_secondary_tax_ids: Tuple[StrictInt, ...] = ()
    identity_status: Literal["resolved", "provisional", "deleted"]
    source: SourceObservation
    evidence_id: StrictStr

    @model_validator(mode="after")
    def identity_agrees(self) -> "TaxonCandidate":
        if self.candidate_id != f"taxon:ncbitaxon:{self.ncbi_tax_id}":
            raise ValueError("taxon candidate identity disagrees with TaxId")
        return self


class ResolveTaxonObserved(FrozenContract):
    status: Literal["observed"] = "observed"
    # NCBI's provisional or deleted status rides on the copied candidate, never a
    # verdict of no taxon; two taxa stay ``ambiguous`` under the owner's conflict
    # rule: a real ambiguity goes to the Review.
    resolution: Literal["resolved", "ambiguous", "not_found"]
    grounding: RequirementGrounding
    query: StrictStr
    candidates: Tuple[TaxonCandidate, ...] = Field(max_length=10)
    source: SourceObservation
    evidence_id: StrictStr
    summary: StrictStr
    elapsed_time_ms: StrictInt = Field(ge=0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)

    @model_validator(mode="after")
    def coherent_resolution(self) -> "ResolveTaxonObserved":
        count = len(self.candidates)
        if self.resolution == "not_found" and count:
            raise ValueError("not-found taxon observation cannot contain candidates")
        if self.resolution != "not_found" and not count:
            raise ValueError("taxon observation requires candidates")
        if self.resolution == "resolved" and count != 1:
            raise ValueError("resolved taxon observation requires one candidate")
        if self.resolution == "ambiguous" and count < 2:
            raise ValueError("ambiguous taxon observation requires multiple candidates")
        return self


class ReferenceToolFailed(FrozenContract):
    status: Literal["failed"] = "failed"
    grounding: RequirementGrounding
    failure: ReferenceAdapterFailure
    evidence_id: StrictStr
    summary: StrictStr


ResolveTaxonResult = Annotated[
    Union[ResolveTaxonObserved, ReferenceToolFailed],
    Field(discriminator="status"),
]


class ResolveTaxonTool:
    name = "resolve_taxon"

    def resolve(
        self,
        tool_input: ResolveTaxonInput,
        *,
        expected_grounding: RequirementGrounding,
        expected_query: str,
        adapter: ReferenceIdentityAdapter,
    ) -> ResolveTaxonResult:
        _require_exact_grounding(tool_input.grounding, expected_grounding)
        if tool_input.query != expected_query:
            raise ReferenceToolCallRejected("taxon_query_ungrounded")
        try:
            from bio_neb.agentic_journey.source_contacts import call_source

            raw = call_source(
                "resolve_taxon", lambda: adapter.resolve_taxon(tool_input.query)
            )
        except ReferenceAdapterError as error:
            return _failed(tool_input.grounding, self.name, error.failure)
        try:
            observation = _validate_taxonomy_observation(raw)
        except (ValueError, TypeError):
            return _failed(
                tool_input.grounding,
                self.name,
                ReferenceAdapterFailure(
                    code="adapter_output_invalid",
                    public_message="NCBI Taxonomy adapter returned invalid typed output.",
                    retryable=False,
                ),
            )

        candidates = tuple(
            _taxon_candidate(record, query=tool_input.query, source=observation.source)
            for record in observation.records
        )
        resolution: Literal["resolved", "ambiguous", "not_found"] = (
            "not_found"
            if not candidates
            else "ambiguous"
            if len(candidates) > 1
            else "resolved"
        )
        payload = {
            "grounding": tool_input.grounding.model_dump(mode="json"),
            "query": tool_input.query,
            "resolution": resolution,
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "source": observation.source.model_dump(mode="json"),
        }
        return ResolveTaxonObserved(
            resolution=resolution,
            grounding=tool_input.grounding,
            query=tool_input.query,
            candidates=candidates,
            source=observation.source,
            evidence_id=_evidence_id("taxon-result", payload),
            summary=f"NCBI Taxonomy returned {len(candidates)} exact candidate(s).",
            elapsed_time_ms=observation.source.elapsed_time_ms,
        )


class ResolvedTaxonEvidence(FrozenContract):
    candidate_id: StrictStr = Field(pattern=r"^taxon:ncbitaxon:[1-9][0-9]*$")
    ncbi_tax_id: StrictInt = Field(gt=0)
    candidate_evidence_id: StrictStr
    result_evidence_id: StrictStr
    source_event_id: StrictStr

    @field_validator("source_event_id")
    @classmethod
    def canonical_source_event(cls, value: str) -> str:
        return _canonical_uuid(value, "source_event_id")

    @model_validator(mode="after")
    def identity_agrees(self) -> "ResolvedTaxonEvidence":
        if self.candidate_id != f"taxon:ncbitaxon:{self.ncbi_tax_id}":
            raise ValueError("resolved taxon candidate and TaxId disagree")
        return self


class DiscoverReferenceAssembliesInput(FrozenContract):
    grounding: RequirementGrounding
    taxon: ResolvedTaxonEvidence
    query: AssemblyQuery = AssemblyQuery()

    @model_validator(mode="after")
    def reference_grounding(self) -> "DiscoverReferenceAssembliesInput":
        if self.grounding.kind != "reference_assembly_identity":
            raise ValueError("discovery Tool requires a reference requirement")
        return self


class DiscoverAssembliesObserved(FrozenContract):
    """One page of NCBI's genome reports for the resolved taxon under the model's
    query: every row copied in NCBI's order, NCBI's count of the whole listing and
    its next page token (#471, ADR 0049). Code ranks, cuts and judges none of them;
    which row the reference is, is the model's choice."""

    status: Literal["observed"] = "observed"
    grounding: RequirementGrounding
    taxon: ResolvedTaxonEvidence
    query: AssemblyQuery
    rows: Tuple[AssemblySourceRecord, ...] = Field(max_length=ASSEMBLY_PAGE_ROWS)
    total_count: StrictInt | None = Field(default=None, ge=0)
    next_page_token: StrictStr | None = Field(default=None, min_length=1)
    source: SourceObservation
    evidence_id: StrictStr
    summary: StrictStr
    elapsed_time_ms: StrictInt = Field(ge=0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)

    def row(self, accession: str) -> AssemblySourceRecord | None:
        """The page's row for ``accession``: the membership check of a choice."""
        return next(
            (row for row in self.rows if row.assembly_accession == accession), None
        )


DiscoverReferenceAssembliesResult = Annotated[
    Union[DiscoverAssembliesObserved, ReferenceToolFailed],
    Field(discriminator="status"),
]


class DiscoverReferenceAssembliesTool:
    name = "discover_reference_assemblies"

    def discover(
        self,
        tool_input: DiscoverReferenceAssembliesInput,
        *,
        expected_grounding: RequirementGrounding,
        expected_taxon: ResolvedTaxonEvidence,
        adapter: ReferenceIdentityAdapter,
    ) -> DiscoverReferenceAssembliesResult:
        _require_exact_grounding(tool_input.grounding, expected_grounding)
        if tool_input.taxon != expected_taxon:
            raise ReferenceToolCallRejected("resolved_taxon_ungrounded")
        try:
            from bio_neb.agentic_journey.source_contacts import call_source

            raw = call_source(
                "discover_assemblies",
                lambda: adapter.discover_assemblies(
                    tool_input.taxon.ncbi_tax_id, tool_input.query
                ),
            )
        except ReferenceAdapterError as error:
            return _failed(tool_input.grounding, self.name, error.failure)
        try:
            observation = _validate_assembly_observation(raw)
            if len(observation.records) > tool_input.query.page_size:
                raise ValueError("NCBI returned more rows than the page asked for")
        except (ValueError, TypeError):
            return _failed(
                tool_input.grounding,
                self.name,
                ReferenceAdapterFailure(
                    code="adapter_output_invalid",
                    public_message="NCBI Datasets adapter returned invalid typed output.",
                    retryable=False,
                ),
            )
        payload = {
            "grounding": tool_input.grounding.model_dump(mode="json"),
            "taxon": tool_input.taxon.model_dump(mode="json"),
            "query": tool_input.query.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in observation.records],
            "total_count": observation.total_count,
            "next_page_token": observation.next_page_token,
            "source": observation.source.model_dump(mode="json"),
        }
        listed = (
            f" of {observation.total_count}"
            if observation.total_count is not None
            else ""
        )
        return DiscoverAssembliesObserved(
            grounding=tool_input.grounding,
            taxon=tool_input.taxon,
            query=tool_input.query,
            rows=observation.records,
            total_count=observation.total_count,
            next_page_token=observation.next_page_token,
            source=observation.source,
            evidence_id=_evidence_id("assembly-discovery-result", payload),
            summary=(
                f"NCBI Datasets returned {len(observation.records)}{listed} "
                "genome reports"
                + (
                    "; its next page token reads the rest."
                    if observation.next_page_token
                    else "."
                )
            ),
            elapsed_time_ms=observation.source.elapsed_time_ms,
        )


class SelectedAssemblyEvidence(FrozenContract):
    """A discovered row the model or the researcher chose, with the discovery page
    it is a member of."""

    row: AssemblySourceRecord
    discovery_evidence_id: StrictStr
    discovery_event_id: StrictStr
    selection_event_id: StrictStr

    @field_validator("discovery_event_id", "selection_event_id")
    @classmethod
    def canonical_event_ids(cls, value: str, info) -> str:
        return _canonical_uuid(value, info.field_name)


class SuppliedAssemblyEvidence(FrozenContract):
    """An exact researcher assertion, with no invented discovery metadata."""

    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assertion_event_id: StrictStr
    selection_event_id: StrictStr
    expected_taxon: ResolvedTaxonEvidence | None = None

    @field_validator("assertion_event_id", "selection_event_id")
    @classmethod
    def canonical_ids(cls, value, info):
        return _canonical_uuid(value, info.field_name)


ReferenceSelection = SelectedAssemblyEvidence | SuppliedAssemblyEvidence


def selected_accession(selected: ReferenceSelection) -> str:
    return (
        selected.row.assembly_accession
        if isinstance(selected, SelectedAssemblyEvidence)
        else selected.assembly_accession
    )


class DescribeReferenceInput(FrozenContract):
    grounding: RequirementGrounding
    selected: ReferenceSelection
    assembly_accession: StrictStr

    @field_validator("assembly_accession")
    @classmethod
    def exact_accession(cls, value: str) -> str:
        if not _ASSEMBLY_ACCESSION.fullmatch(value):
            raise ValueError("reference validation requires exact accession.version")
        return value

    @model_validator(mode="after")
    def selected_identity_agrees(self) -> "DescribeReferenceInput":
        if self.grounding.kind != "reference_assembly_identity":
            raise ValueError("reference validation requires a reference requirement")
        return self


class ValidatedReference(FrozenContract):
    """The exact accession's own NCBI row, copied with its status (#471 Q5 a)."""

    candidate_id: StrictStr
    assembly_accession: StrictStr
    record: AssemblySourceRecord
    source: SourceObservation
    evidence_id: StrictStr

    @model_validator(mode="after")
    def exact_record(self) -> "ValidatedReference":
        if self.record.assembly_accession != self.assembly_accession:
            raise ValueError("validated reference changed its record's accession")
        return self

    @property
    def ncbi_tax_id(self) -> int | None:
        return self.record.ncbi_tax_id


class DescribeReferenceObserved(FrozenContract):
    status: Literal["observed"] = "observed"
    # NCBI's status rides on the copied row; code maps no status to a verdict.
    # Only identity is checked: absent, or a row that disagrees with the
    # selection's own accession or taxon.
    resolution: Literal["resolved", "absent", "source_disagreement"]
    grounding: RequirementGrounding
    selected: ReferenceSelection
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    references: Tuple[ValidatedReference, ...] = Field(max_length=1)
    source: SourceObservation
    evidence_id: StrictStr
    summary: StrictStr
    elapsed_time_ms: StrictInt = Field(ge=0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)

    @model_validator(mode="after")
    def coherent_resolution(self) -> "DescribeReferenceObserved":
        if (
            self.references
            and self.references[0].assembly_accession != self.assembly_accession
        ):
            raise ValueError("reference validation changed its exact queried identity")
        if (
            self.resolution == "resolved"
            and self.assembly_accession != selected_accession(self.selected)
        ):
            raise ValueError("resolved reference differs from the selected identity")
        if self.resolution == "resolved" and self.references:
            expected_taxon = _expected_taxon(self.selected)
            if (
                expected_taxon is not None
                and self.references[0].ncbi_tax_id != expected_taxon
            ):
                raise ValueError(
                    "reference validation changed its exact selected identity taxon"
                )
        if self.resolution == "absent" and self.references:
            raise ValueError("absent exact validation cannot contain a reference")
        if self.resolution != "absent" and len(self.references) != 1:
            raise ValueError("exact validation requires one exact reference record")
        return self


def _expected_taxon(selected: ReferenceSelection) -> int | None:
    """The TaxId the selection itself states: its discovered row's, or the resolved
    taxon a supplied accession is expected under."""
    if isinstance(selected, SelectedAssemblyEvidence):
        return selected.row.ncbi_tax_id
    return selected.expected_taxon.ncbi_tax_id if selected.expected_taxon else None


DescribeReferenceResult = Annotated[
    Union[DescribeReferenceObserved, ReferenceToolFailed],
    Field(discriminator="status"),
]


class DescribeReferenceTool:
    name = "describe_reference"

    def describe(
        self,
        tool_input: DescribeReferenceInput,
        *,
        expected_grounding: RequirementGrounding,
        expected_selection: ReferenceSelection,
        adapter: ReferenceIdentityAdapter,
    ) -> DescribeReferenceResult:
        _require_exact_grounding(tool_input.grounding, expected_grounding)
        if tool_input.selected != expected_selection:
            raise ReferenceToolCallRejected("selected_reference_ungrounded")
        try:
            from bio_neb.agentic_journey.source_contacts import call_source

            raw = call_source(
                "inspect_reference",
                lambda: adapter.describe_assembly(tool_input.assembly_accession),
            )
        except ReferenceAdapterError as error:
            return _failed(tool_input.grounding, self.name, error.failure)
        try:
            observation = _validate_assembly_observation(raw)
        except (ValueError, TypeError):
            return _failed(
                tool_input.grounding,
                self.name,
                ReferenceAdapterFailure(
                    code="adapter_output_invalid",
                    public_message="NCBI Datasets adapter returned invalid typed output.",
                    retryable=False,
                ),
            )
        return _derive_exact_reference(tool_input, observation)


def _require_exact_grounding(
    supplied: RequirementGrounding,
    expected: RequirementGrounding,
) -> None:
    if supplied != expected:
        raise ReferenceToolCallRejected("requirement_subject_ungrounded")


def _validate_taxonomy_observation(value: object) -> TaxonomyAdapterObservation:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return TaxonomyAdapterObservation.model_validate(value)


def _validate_assembly_observation(value: object) -> AssemblyAdapterObservation:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return AssemblyAdapterObservation.model_validate(value)


def _failed(
    grounding: RequirementGrounding,
    tool_name: str,
    failure: ReferenceAdapterFailure,
) -> ReferenceToolFailed:
    return ReferenceToolFailed(
        grounding=grounding,
        failure=failure,
        evidence_id=_evidence_id(
            "reference-tool-failure",
            {
                "tool_name": tool_name,
                "grounding": grounding.model_dump(mode="json"),
                "failure": failure.model_dump(mode="json"),
            },
        ),
        summary=f"{tool_name} failed before producing scientific source evidence.",
    )


def _taxon_candidate(
    record: TaxonomySourceRecord,
    *,
    query: str,
    source: SourceObservation,
) -> TaxonCandidate:
    identity_status: Literal["resolved", "provisional", "deleted"] = (
        "deleted"
        if record.deleted
        else "provisional"
        if record.provisional
        else "resolved"
    )
    payload = {
        "record": record.model_dump(mode="json"),
        "query": query,
        "source": source.model_dump(mode="json"),
    }
    return TaxonCandidate(
        candidate_id=f"taxon:ncbitaxon:{record.ncbi_tax_id}",
        ncbi_tax_id=record.ncbi_tax_id,
        submitted_query=query,
        scientific_name=record.scientific_name,
        rank=record.rank,
        lineage=record.lineage,
        merged_or_secondary_tax_ids=record.merged_or_secondary_tax_ids,
        identity_status=identity_status,
        source=source,
        evidence_id=_evidence_id("taxon-candidate", payload),
    )


def _derive_exact_reference(
    tool_input: DescribeReferenceInput,
    observation: AssemblyAdapterObservation,
) -> DescribeReferenceObserved:
    exact = tuple(
        record
        for record in observation.records
        if record.assembly_accession == tool_input.assembly_accession
    )
    resolution: Literal["resolved", "absent", "source_disagreement"]
    references: tuple[ValidatedReference, ...] = ()
    if not exact:
        resolution = "absent"
    else:
        record = exact[0]
        references = (_validated_reference(record, observation.source),)
        expected_taxon = _expected_taxon(tool_input.selected)
        if (
            len({item.model_dump_json() for item in exact}) > 1
            or tool_input.assembly_accession != selected_accession(tool_input.selected)
            or (expected_taxon is not None and record.ncbi_tax_id != expected_taxon)
        ):
            resolution = "source_disagreement"
        else:
            resolution = "resolved"
    payload = {
        "grounding": tool_input.grounding.model_dump(mode="json"),
        "selected": tool_input.selected.model_dump(mode="json"),
        "assembly_accession": tool_input.assembly_accession,
        "resolution": resolution,
        "references": [item.model_dump(mode="json") for item in references],
        "source": observation.source.model_dump(mode="json"),
    }
    return DescribeReferenceObserved(
        resolution=resolution,
        grounding=tool_input.grounding,
        selected=tool_input.selected,
        assembly_accession=tool_input.assembly_accession,
        references=references,
        source=observation.source,
        evidence_id=_evidence_id("reference-validation-result", payload),
        summary=f"NCBI Datasets exact assembly validation resolved as {resolution}.",
        elapsed_time_ms=observation.source.elapsed_time_ms,
    )


def _validated_reference(
    record: AssemblySourceRecord,
    source: SourceObservation,
) -> ValidatedReference:
    payload = {
        "record": record.model_dump(mode="json"),
        "source": source.model_dump(mode="json"),
    }
    return ValidatedReference(
        candidate_id=f"assembly:{record.assembly_accession}",
        assembly_accession=record.assembly_accession,
        record=record,
        source=source,
        evidence_id=_evidence_id("validated-reference", payload),
    )


__all__ = [
    "AssemblyAdapterObservation",
    "AssemblyQuery",
    "AssemblySourceRecord",
    "DescribeReferenceInput",
    "DescribeReferenceResult",
    "DescribeReferenceTool",
    "DiscoverReferenceAssembliesInput",
    "DiscoverReferenceAssembliesResult",
    "DiscoverReferenceAssembliesTool",
    "ReferenceAdapterError",
    "ReferenceIdentityAdapter",
    "ReferenceToolCallRejected",
    "RequirementGrounding",
    "ResolveTaxonInput",
    "ResolveTaxonResult",
    "ResolveTaxonTool",
    "ResolvedTaxonEvidence",
    "SampleAttestationInput",
    "SampleAttestationResult",
    "SelectedAssemblyEvidence",
    "SourceObservation",
    "TaxonomyAdapterObservation",
    "TaxonomySourceRecord",
    "VerifySampleAttestationTool",
]
