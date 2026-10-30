"""Deep, authority-free ingestion for one Investigation request document."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.contracts import FrozenContract, StrictFrozenContract
from bio_neb.identifier_rule import (
    IdentifierRule,
    gene_change_in_quote,
    site_in_quote,
)
from bio_neb.investigation.vcf_observation import trimmed_allele
from bio_neb.journey import ArtifactReference

_MISSING_ADJACENT_RSID = "investigation_request_item_missing_adjacent_rsid"
# One rsID token: ``rs`` or ``rsID`` in any case, an optional separator, digits.
_RSID_TOKEN = re.compile(
    r"(?<!\w)rs(?:id)?\s*[:#-]?\s*([1-9][0-9]*)(?!\w)", re.IGNORECASE
)
RSID_CANONICAL_PATTERN = r"^rs[1-9][0-9]*$"


def _stated_rsids(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(f"rs{match.group(1)}" for match in _RSID_TOKEN.finditer(text))
    )


# Format only: a merged or withdrawn rsID is its source's to resolve. No source is named:
# the Investigation's model chooses dbSNP or EVA, and the slot reports the one its lookup
# asked.
RSID_RULE = IdentifierRule(name="rsid", stated=_stated_rsids)


def request_target_key(request_text: str, rsid: str | None) -> str:
    """One target's identity across the request, its answers and a request file: its
    typed rsID, else its exact words (#472: no code reads an rsID out of words)."""
    return rsid or request_text


def _canonical_text(value: str, name: str) -> str:
    if not value or value.strip() != value or any(char in value for char in "\r\n\0"):
        raise ValueError(f"{name} must be one non-empty canonical line")
    return value


class RequestedVariantIdentity(NamedTuple):
    contig: str
    one_based_position: int
    ref: str
    alt: str


_STATED_ALLELE = r"^[ACGTN]*$"


class _GenomePosition(StrictFrozenContract):
    """A sequence as it is named, a one-based position and, when given, the alleles in
    VCF form. An empty allele is lawful only in an already trimmed form, as a retry
    restates a Work item's allele."""

    contig: StrictStr = Field(min_length=1, max_length=200)
    position: StrictInt = Field(gt=0)
    ref: StrictStr | None = Field(
        default=None, pattern=_STATED_ALLELE, exclude_if=lambda value: value is None
    )
    alt: StrictStr | None = Field(
        default=None, pattern=_STATED_ALLELE, exclude_if=lambda value: value is None
    )

    @field_validator("contig")
    @classmethod
    def canonical_contig(cls, value: str) -> str:
        return _canonical_text(value, "contig")

    @model_validator(mode="after")
    def stated_alleles(self) -> "_GenomePosition":
        if (self.ref is None) != (self.alt is None):
            raise ValueError("a position states both alleles or neither")
        if self.ref is not None and self.ref == self.alt:
            raise ValueError("REF and ALT must differ")
        return self

    @property
    def identity(self) -> RequestedVariantIdentity | None:
        """The stated allele trimmed of its shared bases (#368), on the stated name."""
        if self.ref is None or self.alt is None:
            return None
        position, ref, alt = trimmed_allele(self.position, self.ref, self.alt)
        return RequestedVariantIdentity(self.contig, position, ref, alt)


class StatedPosition(_GenomePosition):
    """Where a request's source places its target, on the version it states: a
    sequence as the request names it, a one-based position and, when stated, the
    alleles in VCF form (#369).

    The sequence keeps the request's own name ("chr11", "NC_000011.10"); naming it
    on the Run's assembly is the location lookup's. No coordinates are mapped
    between versions here: a position placement, or its lookup (#462 R2), moved is a
    `PlacedPosition`.
    """


class SiteDraft(_GenomePosition):
    """A site the model types from the researcher's words (#472 Q1 a): the sequence
    as the quote writes it, the position, and the alleles when the quote writes a
    directed pair. Code checks only that the quote holds them
    (`identifier_rule.site_in_quote`); naming the sequence is the location lookup's.
    It holds a stated position's rules: both alleles or neither, and they differ."""

    contig: StrictStr = Field(
        min_length=1,
        max_length=200,
        description="The sequence exactly as the quote writes it, e.g. V, "
        "Chromosome IV or a sequence accession.",
    )
    position: StrictInt = Field(
        gt=0,
        description="The one-based position the quote writes, as a number (207469 for "
        "207,469). For an HGVS g. range or deletion, its first position.",
    )
    ref: StrictStr | None = Field(
        default=None,
        min_length=1,
        pattern=_STATED_ALLELE,
        exclude_if=lambda value: value is None,
        description="Optional. The REF bases of a directed pair the quote writes (C for "
        "C>T), in upper case; with alt, or neither.",
    )
    alt: StrictStr | None = Field(
        default=None,
        min_length=1,
        pattern=_STATED_ALLELE,
        exclude_if=lambda value: value is None,
        description="Optional. The ALT bases of that pair (T for C>T), in upper case.",
    )

    def stated(self) -> StatedPosition:
        return StatedPosition(
            contig=self.contig, position=self.position, ref=self.ref, alt=self.alt
        )

    def in_quote(self, quote: str) -> bool:
        return site_in_quote(quote, self.contig, self.position, self.ref, self.alt)


class PlacedPosition(_GenomePosition):
    """Where placement put a target on the Run's assembly (#367): a position no source
    states on the Run's version, kept apart from the stated one it came from."""


class StatedRegion(StrictFrozenContract):
    """A stretch of one sequence a request names (#367, from #377): the sequence as
    named and its first and last one-based positions. Every VCF record in it is its
    own outcome."""

    contig: StrictStr = Field(min_length=1, max_length=200)
    start: StrictInt = Field(gt=0)
    end: StrictInt = Field(gt=0)

    @field_validator("contig")
    @classmethod
    def canonical_contig(cls, value: str) -> str:
        return _canonical_text(value, "contig")

    @model_validator(mode="after")
    def ordered(self) -> "StatedRegion":
        if self.start > self.end:
            raise ValueError("a region starts at or before its end")
        return self


Residue = Literal[
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*",
]  # fmt: skip
"""One amino acid by its one-letter code, or ``*`` for a stop, as the model types it."""
Base = Literal["A", "C", "G", "T"]


class ProteinChangeDraft(StrictFrozenContract):
    """A residue change the model types from the words that state it (#473 Q2 a):
    the reference residue, its codon number and the residue it becomes, one-letter,
    a stop ``*``."""

    ref: Residue = Field(description="The reference residue, one-letter.")
    codon: StrictInt = Field(gt=0, description="The codon number the change names.")
    alt: Residue = Field(
        description="The residue it becomes, one-letter; * for a stop (X, Ter, *)."
    )

    @model_validator(mode="after")
    def a_change(self) -> "ProteinChangeDraft":
        if self.ref == self.alt:
            raise ValueError("a protein change states another residue")
        return self


class CodingChangeDraft(StrictFrozenContract):
    """A coding-strand substitution the model types (#473 Q2 a): its CDS position
    (``c.N``), the reference base and the base it becomes."""

    cds_position: StrictInt = Field(gt=0, description="The CDS position, N of c.N.")
    ref: Base = Field(description="The reference base on the coding strand.")
    alt: Base = Field(description="The base it becomes.")

    @model_validator(mode="after")
    def a_change(self) -> "CodingChangeDraft":
        if self.ref == self.alt:
            raise ValueError("a coding change states another base")
        return self


class StatedChange(StrictFrozenContract):
    """A gene and the change a source writes for it: ``PBP2 (Y127X)`` is gene ``PBP2``
    and change ``Y127X``, as written (G15's claim, #376). The model types the change
    from the words (#473 Q2 a): its transcript when one is written, and a protein
    change, a coding change or both. Code reads nothing out of ``change``; a change
    with neither typed part is untyped, as a paper's change stays until #383."""

    gene: StrictStr = Field(
        min_length=1,
        max_length=4_000,
        description="The gene symbol exactly as the words write it.",
    )
    change: StrictStr = Field(
        min_length=1,
        max_length=4_000,
        description="The change exactly as the words write it, e.g. V600E or "
        "c.1799T>A.",
    )
    transcript: StrictStr | None = Field(
        default=None,
        max_length=200,
        exclude_if=lambda value: value is None,
        description="Optional. The transcript accession the words write, as written.",
    )
    protein: ProteinChangeDraft | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The protein change the words write, typed.",
    )
    coding: CodingChangeDraft | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The coding change the words write, typed.",
    )

    @field_validator("gene", "change", "transcript")
    @classmethod
    def canonical(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _canonical_text(value, info.field_name)

    @model_validator(mode="after")
    def one_codon(self) -> "StatedChange":
        if self.protein is not None and self.coding is not None:
            codon = (self.coding.cds_position + 2) // 3
            if codon != self.protein.codon:
                raise ValueError(
                    f"c.{self.coding.cds_position} is codon {codon}, and the protein "
                    f"change names codon {self.protein.codon}"
                )
        return self

    @property
    def typed(self) -> bool:
        """Whether the model typed the change: a protein or a coding part."""
        return self.protein is not None or self.coding is not None

    def in_quote(self, quote: str) -> bool:
        """Whether ``quote`` writes this change as the model typed it (#473 Q2 a)."""
        return gene_change_in_quote(
            quote,
            self.gene,
            self.change,
            self.transcript,
            tuple(
                number
                for number in (
                    None if self.protein is None else self.protein.codon,
                    None if self.coding is None else self.coding.cds_position,
                )
                if number is not None
            ),
        )


PlacementStep = Literal[
    "same_version",
    "table",
    "database_equivalence",
    "flank_alignment",
    "gene_change",
    "none",
]
"""The step that placed a target on the Run's assembly (#367): its stated version is
the Run's (``same_version``), a flank move from the Run's stated-sequence table row
(``table``), NCBI's Variation service (``database_equivalence``), a flank move from the
sequence NCBI names (``flank_alignment``), or G15 on its gene and change
(``gene_change``). ``none``: no step placed it."""
_POSITION_STEPS = frozenset(
    {"same_version", "table", "database_equivalence", "flank_alignment"}
)


class RequestPlacement(StrictFrozenContract):
    """How placement answered one target (#367): the step that placed it, the written
    version name that matched (the source's own, or another name it writes for that
    version), the placed position on the Run's assembly and why. A target no step
    placed has step ``none``, no position, and its reason."""

    step: PlacementStep
    matched_name: StrictStr | None = Field(
        default=None, max_length=200, exclude_if=lambda value: value is None
    )
    position: PlacedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reason: StrictStr = Field(min_length=1, max_length=1_500)

    @field_validator("reason")
    @classmethod
    def canonical_reason(cls, value: str) -> str:
        return _canonical_text(value, "reason")

    @model_validator(mode="after")
    def placed_or_not(self) -> "RequestPlacement":
        if (self.position is None) != (self.step == "none"):
            raise ValueError("a placed target has its position, an unplaced one none")
        return self


QuestionAction = Literal["observe_variant", "observe_region"]


VersionForm = Literal["assembly_name", "synonym", "accession", "paired_accession"]


def literal_version_form(
    stated: str,
    name: str | None,
    synonym: str | None,
    accession: str | None,
    paired: str | None,
) -> VersionForm | None:
    """Which of an assembly's own names a stated version literally is: its assembly
    name, its UCSC synonym, its versioned accession or its paired GenBank/RefSeq
    accession, exactly as written (#471 Q4 a). Code strips, folds and matches no other
    spelling; a version written any other way is the model's to name (#383)."""
    forms: tuple[tuple[VersionForm, str | None], ...] = (
        ("assembly_name", name),
        ("synonym", synonym),
        ("accession", accession),
        ("paired_accession", paired),
    )
    return next((form for form, value in forms if value and value == stated), None)


class StatedAssembly(StrictFrozenContract):
    """The genome version a request's source states, and the line that states it."""

    name: StrictStr = Field(min_length=1, max_length=200)
    cited_line: StrictStr = Field(min_length=1, max_length=4_000)

    @field_validator("name", "cited_line")
    @classmethod
    def canonical(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)


class InvestigationRequestItem(StrictFrozenContract):
    """One target as its request states it: its words, the typed target (an rsID, a
    position, a region or a gene and change), the version its source states, how
    placement answered it, and who asserts it when that is not the request's own
    asserter (#367). Every optional field is left out when empty."""

    request_text: StrictStr = Field(max_length=4_000)
    rsid: StrictStr | None = Field(default=None, pattern=RSID_CANONICAL_PATTERN)
    position: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    region: StatedRegion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    gene_change: StatedChange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stated_assembly: StatedAssembly | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    placement: RequestPlacement | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    asserted_by: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    normalized_question_action: QuestionAction = "observe_variant"

    @field_validator("request_text")
    @classmethod
    def canonical_request(cls, value: str) -> str:
        return _canonical_text(value, "request_text")

    @model_validator(mode="after")
    def exact_request(self) -> "InvestigationRequestItem":
        if self.rsid is not None and not RSID_RULE.matches(
            self.rsid, self.request_text
        ):
            raise PydanticCustomError(
                _MISSING_ADJACENT_RSID,
                "explicit request must contain its exact rsID",
            )
        if self.asserted_by is not None:
            _canonical_text(self.asserted_by, "asserted_by")
        if (self.normalized_question_action == "observe_region") != (
            self.region is not None
        ):
            raise ValueError("a region target, and only one, observes its region")
        if self.region is not None and (
            self.rsid is not None
            or self.position is not None
            or self.gene_change is not None
            or self.placement is not None
        ):
            raise ValueError("a region target states its region alone")
        if self.placement is not None:
            _check_placement(self)
        return self

    @property
    def placed(self) -> PlacedPosition | None:
        """The position placement put this target at on the Run's assembly, if any."""
        return None if self.placement is None else self.placement.position


def _check_placement(item: "InvestigationRequestItem") -> None:
    """A placement answers what its target states: a position step moves or keeps a
    stated position on a stated version, G15 places a stated gene and change."""
    placement = item.placement
    if placement.step in _POSITION_STEPS and (
        item.position is None or item.stated_assembly is None
    ):
        raise ValueError("a position step places a stated position on its version")
    if placement.step == "gene_change" and item.gene_change is None:
        raise ValueError("G15 places a stated gene and change")
    if placement.step == "same_version":
        kept = placement.position.model_dump()
        if kept != item.position.model_dump():
            raise ValueError("a position kept on its version is the stated one")


class InvestigationRequest(StrictFrozenContract):
    asserted_by: StrictStr
    items: tuple[InvestigationRequestItem, ...] = Field(min_length=1)

    @field_validator("asserted_by")
    @classmethod
    def canonical_asserted_by(cls, value: str) -> str:
        return _canonical_text(value, "asserted_by")


class StatedTarget(StrictFrozenContract):
    """A typed target: what a request or an amend states (#369): an rsID, a position, or
    both; in a ``/3`` request also a region or a gene and change (#367). Prose never
    states one."""

    rsid: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        exclude_if=lambda value: value is None,
    )
    position: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    region: StatedRegion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    gene_change: StatedChange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def stated_target(self) -> "StatedTarget":
        if (
            self.rsid is None
            and self.position is None
            and self.region is None
            and self.gene_change is None
        ):
            raise ValueError(
                "a target states an rsID, a position, a region or a gene and change"
            )
        return self


REQUEST_SCHEMA = "bio-neb.investigation-request/3"
"""The request document the target-request module writes (#367); ``/2`` still reads."""
_EARLIER_REQUEST_SCHEMA = "bio-neb.investigation-request/2"


class _InvestigationRequestDocumentItem(StrictFrozenContract):
    request: StrictStr
    target: StatedTarget | None = None
    stated_assembly: StatedAssembly | None = None
    placement: RequestPlacement | None = None
    normalized_question_action: QuestionAction = "observe_variant"


class _InvestigationRequestDocument(StrictFrozenContract):
    schema_version: Literal[
        "bio-neb.investigation-request/2", "bio-neb.investigation-request/3"
    ] = REQUEST_SCHEMA
    asserted_by: StrictStr
    requests: tuple[_InvestigationRequestDocumentItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def earlier_shape(self) -> "_InvestigationRequestDocument":
        # A /2 document states rsIDs and positions only, with no placement.
        if self.schema_version == _EARLIER_REQUEST_SCHEMA and any(
            item.placement is not None
            or item.normalized_question_action != "observe_variant"
            or (
                item.target is not None
                and (item.target.region is not None or item.target.gene_change)
            )
            for item in self.requests
        ):
            raise ValueError("a /2 request states rsIDs and positions only")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_cardinality_one_document(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        if "requests" in value:
            if not isinstance(value["requests"], list):
                return value
            normalized = dict(value)
            normalized["requests"] = tuple(value["requests"])
            return normalized
        if "request" not in value:
            return value
        normalized = dict(value)
        normalized["requests"] = (
            {
                key: normalized.pop(key)
                for key in ("request", "target", "stated_assembly")
                if key in normalized
            },
        )
        return normalized


class InvestigationRequestRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


def _request_file(value: str) -> Path:
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise InvestigationRequestRefused(
            "investigation_input_path_invalid", "request_file"
        )
    return Path(value)


def _contains_error(error: ValidationError, error_type: str) -> bool:
    return any(detail.get("type") == error_type for detail in error.errors())


def read_investigation_request(path: str) -> InvestigationRequest:
    """Read one fully domain-validated request without activating any authority."""

    return read_investigation_request_document(path)[1]


def read_investigation_request_document(
    path: str,
) -> tuple[bytes, InvestigationRequest]:
    """Return the exact bytes read and the request they validate as, read once."""

    try:
        content = _request_file(path).read_bytes()
    except InvestigationRequestRefused:
        raise
    except OSError as error:
        raise InvestigationRequestRefused(
            "investigation_request_invalid", "request_file"
        ) from error
    return content, _parse_investigation_request(content)


def _parse_investigation_request(content: bytes) -> InvestigationRequest:
    try:
        document = content.decode("utf-8")
        value = json.loads(document)
        if not isinstance(value, dict):
            raise ValueError("request must be an object")
        request_document = _InvestigationRequestDocument.model_validate_json(
            document, strict=True
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as error:
        raise InvestigationRequestRefused(
            "investigation_request_invalid", "request_file"
        ) from error

    request_items: list[InvestigationRequestItem] = []
    missing_adjacent_rsid_fields: list[str] = []
    for index, item in enumerate(request_document.requests):
        try:
            request_items.append(
                InvestigationRequestItem(
                    request_text=item.request,
                    rsid=None if item.target is None else item.target.rsid,
                    position=None if item.target is None else item.target.position,
                    region=None if item.target is None else item.target.region,
                    gene_change=None
                    if item.target is None
                    else item.target.gene_change,
                    stated_assembly=item.stated_assembly,
                    placement=item.placement,
                    normalized_question_action=item.normalized_question_action,
                )
            )
        except ValidationError as error:
            if _contains_error(error, _MISSING_ADJACENT_RSID):
                missing_adjacent_rsid_fields.append(
                    f"request_file.requests[{index}].request"
                )
                continue
            raise InvestigationRequestRefused(
                "investigation_request_invalid",
                f"request_file.requests[{index}]",
            ) from error
    if missing_adjacent_rsid_fields:
        raise InvestigationRequestRefused(
            "investigation_request_missing_adjacent_rsid",
            *missing_adjacent_rsid_fields,
        )
    try:
        return InvestigationRequest(
            asserted_by=request_document.asserted_by,
            items=tuple(request_items),
        )
    except ValidationError as error:
        raise InvestigationRequestRefused(
            "investigation_request_invalid", "request_file"
        ) from error


INVESTIGATION_REQUEST_MEDIA_TYPE = "application/json"


class InvestigationRequestRegistered(FrozenContract):
    """The researcher's request as supplied at registration; a User assertion only."""

    event_type: Literal["InvestigationRequestRegistered"] = (
        "InvestigationRequestRegistered"
    )
    request_file: StrictStr
    request: InvestigationRequest

    @field_validator("request", mode="before")
    @classmethod
    def retained_request(cls, value: Any) -> Any:
        # Retained payloads are JSON; the request's own strict contract still applies.
        if isinstance(value, Mapping):
            return InvestigationRequest.model_validate(value, strict=False)
        return value


class CapturedInvestigationRequest(StrictFrozenContract):
    """One Run's captured request, its retained bytes and the fact that names them."""

    request: InvestigationRequest
    request_file: StrictStr | None
    source_event_id: StrictStr
    artifact: ArtifactReference | None
    source_event_ids: tuple[StrictStr, ...] = ()
    # The research goal the request states, in the researcher's words.
    research_goals: tuple[StrictStr, ...] = ()


# A clarify answer to a member question and an amend answer to its issue both
# state the researcher's intent for that member: shared mechanics, each with its
# own consequence.
RESEARCHER_INSTRUCTION_ACTIONS = frozenset({"clarify", "amend"})


def researcher_answers(receipts) -> tuple:
    """The Review receipts whose instruction states researcher intent for a member."""
    return tuple(
        receipt
        for receipt in receipts
        if receipt.response.action in RESEARCHER_INSTRUCTION_ACTIONS
    )


def _amended_target(answers=()) -> dict | None:
    """The typed target of a member's latest answer that states one (#369)."""
    for answer in reversed(answers):
        if answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS and answer.get(
            "target"
        ):
            return answer
    return None


class TargetLocation(NamedTuple):
    """Where a member's lookup reads: its stated position and version, the position
    placement put it at on the Run's assembly, or the region it states (#367)."""

    stated: StatedPosition | None
    stated_assembly: StatedAssembly | None
    placed: PlacedPosition | None = None
    region: StatedRegion | None = None

    @property
    def position(self) -> StatedPosition | PlacedPosition | None:
        """The position the lookup reads: the placed one, else the stated one."""
        return self.placed or self.stated


def effective_location(request: InvestigationRequestItem, answers=()) -> TargetLocation:
    """Where a member's lookup reads: an answer's typed target replaces the request's,
    a region's too (#369, #367 r1 D2); a placed request is read at its placed
    position, and one no step placed is not looked up by position (#367). A gene
    change the model placed and picked a candidate of is read at that candidate
    (#473 Q11), a placed position on the Run's assembly."""
    amended = _amended_target(answers)
    if amended is not None and "placed" in amended:
        return TargetLocation(
            None, None, PlacedPosition.model_validate(amended["placed"])
        )
    if amended is not None:
        position = amended["target"].get("position")
        assembly = amended.get("stated_assembly")
        return TargetLocation(
            None if position is None else StatedPosition.model_validate(position),
            None if assembly is None else StatedAssembly.model_validate(assembly),
        )
    if request.placement is not None:
        if request.placed is None:
            return TargetLocation(None, None)
        return TargetLocation(request.position, request.stated_assembly, request.placed)
    return TargetLocation(
        request.position, request.stated_assembly, region=request.region
    )


def effective_requested_rsid(
    request: InvestigationRequestItem, answers=()
) -> str | None:
    """One asserted target: the request's typed rsID, else the rsID of the latest
    answer's typed target, the researcher's or the one the model typed from its words
    (`typed_answers`, #472 Q3 a). No code reads an rsID out of words.

    The request's own rsID stays its requested one through any answer, as a typed
    amend's target only changes what is looked up (#369)."""
    if request.rsid is not None:
        return request.rsid
    for answer in reversed(answers):
        if answer.get("action") not in RESEARCHER_INSTRUCTION_ACTIONS:
            continue
        typed = (answer.get("target") or {}).get("rsid")
        if typed is not None:
            return typed
    return None


NotCheckedReason = Literal["not_placed", "position_without_alleles"]
"""Why the Run cannot check a target by what it states (#367): no step placed it on the
Run's assembly. A target only words state is read by the Investigation's model
(#472 R1 b), and one its words name none of opens its member's Review.
``position_without_alleles`` (owner #480 Q1): its lookup found a position stated
without alleles or an rsID, which no source lists, so the item is deferred out of the
Run after that lookup, with no question."""


def untyped(request: InvestigationRequestItem) -> bool:
    """Whether the item states its target only in words: no typed rsID, position,
    region, gene change, or placement (#472 R1 b). A gene and change no model typed
    is still words (#473 Q2 a)."""
    return all(
        part is None
        for part in (
            request.rsid,
            request.position,
            request.region,
            request.placement,
        )
    ) and (request.gene_change is None or not request.gene_change.typed)


def effective_gene_change(
    request: InvestigationRequestItem, answers=()
) -> StatedChange | None:
    """The gene change a member's lookup places (#473 Q1 a): the latest answer's typed
    target's, when one states a target, else the request's own, when typed."""
    amended = _amended_target(answers)
    if amended is not None:
        change = amended["target"].get("gene_change")
        return None if change is None else StatedChange.model_validate(change)
    change = request.gene_change
    return change if change is not None and change.typed else None


def not_checked_reason(
    request: InvestigationRequestItem,
) -> tuple[NotCheckedReason, str] | None:
    """Why a target is not checked, or None when the Run can check it: a typed rsID,
    position or region, or words the Investigation's model reads (#472 R1 b). Nothing
    is guessed into an rsID (#367, readiness Q2 b)."""
    location = effective_location(request)
    if (
        location.region is None
        and request.rsid is None
        and location.position is None
        and request.placement is not None
    ):
        return "not_placed", request.placement.reason
    return None


def researcher_rsids(request: InvestigationRequestItem, answers=()) -> tuple[str, ...]:
    """Every rsID the researcher stated for one member, typed: its request's own, or
    an answer's typed target, which replaces it (#369). A lookup or retrieved target
    outside these needs a dbSNP merge record. An answer's words state an rsID only
    through the target the model typed from them (`typed_answers`, #472 Q3 a)."""
    amended = _amended_target(answers)
    if amended is not None:
        rsid = amended["target"].get("rsid")
        return () if rsid is None else (rsid,)
    rsid = effective_requested_rsid(request)
    return () if rsid is None else (rsid,)


def typed_answers(
    request: InvestigationRequestItem, answers, reads=(), picks=()
) -> tuple[dict, ...]:
    """A member's answers with the target the Investigation's model typed from each
    one's words (`InvestigationTargetRead`, #472 Q3 a), where the researcher typed
    none: a reading replaces the member's target as a typed amend does (#369). A
    reading of an untyped item's own words comes first, as its typed target (R1 b).
    ``answers`` carry their ``response_event_id``; ``reads`` are the readings'
    payloads. ``picks`` are the candidates the model picked from its placements of a
    gene change (`InvestigationGeneChangePicked`, #473 Q11): the latest pick of the
    gene change the member now states places it, as a typed answer would; a pick of
    a change an answer has since replaced places nothing."""
    by_source = {read["source_event_id"]: read for read in reads}
    typed = [
        {
            "action": "clarify",
            "instruction": request.request_text,
            "target": read["target"],
            **(
                {"stated_assembly": read["stated_assembly"]}
                if "stated_assembly" in read
                else {}
            ),
        }
        for read in reads
        if read["source"] == "request" and "target" in read
    ]
    for answer in answers:
        read = by_source.get(answer["response_event_id"])
        if answer.get("target") or read is None or "target" not in read:
            typed.append(answer)
            continue
        typed.append(
            {
                **answer,
                "target": read["target"],
                **(
                    {"stated_assembly": read["stated_assembly"]}
                    if "stated_assembly" in read
                    else {}
                ),
            }
        )
    current = effective_gene_change(request, typed)
    answered = {
        answer["response_event_id"]
        for answer in answers
        if answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS
        and "response_event_id" in answer
    }
    stated = effective_location(request, typed)
    if stated.position is not None or stated.region is not None:
        # A site the researcher states, or a placement the request carries, is the
        # member's location; no pick replaces it.
        return tuple(typed)
    amended = _amended_target(typed)
    target = (
        amended["target"]
        if amended is not None
        else StatedTarget(
            rsid=request.rsid, gene_change=request.gene_change
        ).model_dump(mode="json")
        if current is not None
        else None
    )
    for pick in picks:
        # A pick places the change the member states now, made after every answer
        # it has: a newer answer outranks it (SP1). It keeps the target's own typed
        # fields, an rsID among them, beside the placed position.
        if (
            current is not None
            and pick["gene_change"] == current.model_dump(mode="json")
            and answered <= set(pick["after_answer_event_ids"])
        ):
            typed.append(
                {
                    "action": "clarify",
                    "instruction": request.request_text,
                    "target": target,
                    "placed": pick["placed"],
                }
            )
    return tuple(typed)


def unread_target_words(
    request: InvestigationRequestItem, answers, reads=(), *, started_event_id: str
) -> tuple[str, ...]:
    """The member's words its model has yet to type a target from (#472 Q3 a, R1 b):
    an untyped item's own words (keyed by the Investigation's start) and each
    instruction answer the researcher typed no target for, by event id."""
    read = {item["source_event_id"] for item in reads}
    pending = [started_event_id] if untyped(request) else []
    pending.extend(
        answer["response_event_id"]
        for answer in answers
        if answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS
        and not answer.get("target")
    )
    return tuple(item for item in pending if item not in read)


def current_investigation_request(journey) -> CapturedInvestigationRequest | None:
    """The Run's request: once begun, the started Investigation's; before, planning's.

    Planning owns the targets (ADR 0010): its handoff carries the request's own
    targets and the registered request file it cites, so the file never overrides
    the plan unseen. A Run without a plan reads its registered file alone.
    """
    # The first native request remains the entry identity after scoped revisions.
    # A revised assertion is scientific work, not a replacement begin command.
    from bio_neb.recording import read_evidence

    events = read_evidence(journey)
    started = next(
        (event for event in events if event["event_type"] == "InvestigationStarted"),
        None,
    )
    if started is not None:
        from bio_neb.investigation.investigation import UserAssertion
        from bio_neb.journey import canonical_json

        assertion = UserAssertion.model_validate_json(
            canonical_json(started["payload"]["assertion"]), strict=True
        )
        event_ids = {event["event_id"] for event in events}
        return CapturedInvestigationRequest(
            request=InvestigationRequest(
                asserted_by=assertion.asserted_by,
                items=assertion.requests
                or (
                    InvestigationRequestItem(
                        request_text=assertion.request_text, rsid=assertion.rsid
                    ),
                ),
            ),
            request_file=None,
            artifact=None,
            source_event_id=started["event_id"],
            source_event_ids=tuple(
                identity
                for identity in assertion.predecessor_ids
                if identity in event_ids
            ),
            research_goals=assertion.research_goals,
        )
    from bio_neb.application.planning_read import current_planning

    planning = current_planning(journey)
    return run_investigation_request(
        planning.requirements,
        journey.checked_view("investigation_request"),
        suggested=chosen_suggestion_targets(planning.reference),
    )


def run_investigation_request(
    resolution,
    registered: Sequence[CapturedInvestigationRequest],
    *,
    suggested: Sequence[Any] = (),
) -> CapturedInvestigationRequest | None:
    """Planning's request when the Run has a plan; a planless Run's registered files.
    ``suggested`` is the known targets as a suggested assembly the researcher chose
    checked them (`chosen_suggestion_targets`, #459 D3)."""
    if resolution is None:
        return _registered_request(registered)
    captured = planned_investigation_request(resolution, registered)
    if captured is None or not suggested:
        return captured
    return captured.model_copy(
        update={
            "request": captured.request.model_copy(
                update={
                    "items": tuple(
                        _on_suggested_assembly(item, suggested)
                        for item in captured.request.items
                    )
                }
            )
        }
    )


def _on_suggested_assembly(item: "InvestigationRequestItem", suggested):
    """A known target as the Run on the chosen suggested assembly reads it: a position
    that stated no version is stated on the requested assembly, and a position the
    check moved keeps its move as its placement (#459 D3); any other target is read
    by the Investigation's lookup as it arrives (D5)."""
    target = next((t for t in suggested if t.answers(item)), None)
    if target is None or item.position is None:
        return item
    update: dict[str, Any] = {}
    if item.stated_assembly is None:
        update["stated_assembly"] = target.stated_assembly
    if target.placement is not None and item.placement is None:
        update["placement"] = target.placement
    return item.model_copy(update=update) if update else item


def chosen_suggestion_targets(reference) -> tuple:
    """The known targets of the suggested assembly the researcher chose for the whole
    Run, as the reference owner's checked view holds them (`standing_suggestion`, #459
    D4); () when the Run has no reference view or no standing suggestion."""
    standing = None if reference is None else reference.standing_suggestion
    return () if standing is None else standing.targets


RESEARCHER = "researcher"
"""Who asserts a target the prompt names or a researcher's answer states (#367)."""


def _collection_identity(item: InvestigationRequestItem) -> tuple:
    """One target's identity within a collection (#367 r1 D1): its full typed meaning.
    Two alleles of one rsID, or one rsID stated on two versions, are two targets; a
    target that types nothing beyond an rsID is its rsID, else its words."""
    typed = (
        item.position,
        item.stated_assembly,
        item.placed,
        item.region,
        item.gene_change,
    )
    if all(part is None for part in typed):
        return (request_target_key(item.request_text, item.rsid),)
    return (
        item.rsid,
        *(None if part is None else part.model_dump_json() for part in typed),
    )


def combined_request(
    files: Sequence[InvestigationRequest],
    prompt_targets: Sequence[InvestigationRequestItem] = (),
) -> InvestigationRequest | None:
    """One collection from several request files and the prompt's own targets: the
    files' targets in the files' order, then the prompt's, each once (#367). Each
    target keeps its own asserter: the collection's is the first file's (else the
    researcher's), and a target asserted by anyone else names its own."""
    asserted_by = files[0].asserted_by if files else RESEARCHER
    items: dict[str, InvestigationRequestItem] = {}
    for owner, item in (
        *((file.asserted_by, item) for file in files for item in file.items),
        # A prompt's target is the researcher's; a cited paper's names its paper.
        *((item.asserted_by or RESEARCHER, item) for item in prompt_targets),
    ):
        items.setdefault(
            _collection_identity(item),
            item
            if owner == asserted_by
            else item.model_copy(update={"asserted_by": owner}),
        )
    if not items:
        return None
    return InvestigationRequest(asserted_by=asserted_by, items=tuple(items.values()))


def _registered_request(
    registered: Sequence[CapturedInvestigationRequest],
) -> CapturedInvestigationRequest | None:
    """A planless Run's registered files as one request, citing each registration."""
    request = combined_request([file.request for file in registered])
    if request is None:
        return None
    first = registered[0]
    return CapturedInvestigationRequest(
        request=request,
        request_file=first.request_file if len(registered) == 1 else None,
        source_event_id=first.source_event_id,
        artifact=first.artifact if len(registered) == 1 else None,
        source_event_ids=tuple(file.source_event_id for file in registered[1:]),
    )


def planned_investigation_request(
    resolution, registered: Sequence[CapturedInvestigationRequest]
) -> CapturedInvestigationRequest | None:
    """Translate the planning owner's typed handoff and preserve its exact origin.

    Items keep the researcher's order and words: each cited registered file's targets
    in the files' registration order, then the request's own targets in the order it
    names them, each once, each with its own asserter (#367). The researcher's goal
    travels with them."""
    handoff = resolution.scientific_investigation
    cited_ids = {
        item.registered_request_event_id
        for item in handoff.items
        if item.registered_request_event_id
    }
    by_id = {file.source_event_id: file for file in registered}
    if not cited_ids <= by_id.keys():
        raise ValueError("planning cites a registered request the Run lacks")
    cited = [file for file in registered if file.source_event_id in cited_ids]
    sources: list[str] = []
    for item in handoff.items:
        if item.registered_request_event_id:
            sources.extend(item.lineage_event_ids)
    prompt_targets = []
    for deferred in handoff.items:
        for target in deferred.targets:
            prompt_targets.append(
                InvestigationRequestItem(
                    request_text=target.words,
                    rsid=target.rsid,
                    position=target.position,
                    gene_change=target.gene_change,
                    stated_assembly=target.stated_assembly,
                    asserted_by=target.asserted_by,
                )
            )
            sources.extend((*deferred.lineage_event_ids, target.source_event_id))
    request = combined_request([file.request for file in cited], prompt_targets)
    if request is None:
        return None
    return CapturedInvestigationRequest(
        request=request,
        request_file=cited[0].request_file if len(cited) == 1 else None,
        artifact=cited[0].artifact if len(cited) == 1 else None,
        source_event_id=resolution.plan_event_id,
        source_event_ids=tuple(dict.fromkeys(sources)),
        research_goals=tuple(goal.words for goal in handoff.research_goals),
    )


def investigation_request_document_schema() -> dict[str, Any]:
    """The document shape a researcher supplies; validation stays with the reader."""

    return _InvestigationRequestDocument.model_json_schema()


def project_captured_investigation_requests(
    events: Sequence[Mapping[str, Any]],
) -> tuple[CapturedInvestigationRequest, ...]:
    """Fold each captured request file, in registration order, bound to its exact
    registration: a child of the Run's one registration with its own retained bytes
    (#367: a Run takes several files)."""

    captured = [
        e for e in events if e["event_type"] == "InvestigationRequestRegistered"
    ]
    if not captured:
        return ()
    registrations = [e for e in events if e["event_type"] == "RunInputsRegistered"]
    if len(registrations) != 1:
        raise ValueError("captured Investigation request lacks its exact registration")
    files = []
    for event in captured:
        if (
            event["parent_ids"] != [registrations[0]["event_id"]]
            or len(event["artifact_references"]) != 1
        ):
            raise ValueError(
                "captured Investigation request lacks its exact registration"
            )
        fact = InvestigationRequestRegistered.model_validate(
            {"event_type": event["event_type"], **event["payload"]}
        )
        artifact = ArtifactReference.model_validate(
            event["artifact_references"][0], strict=False
        )
        if (
            artifact.source_event_id != event["event_id"]
            or artifact.media_type != INVESTIGATION_REQUEST_MEDIA_TYPE
        ):
            raise ValueError("captured Investigation request artifact is not its own")
        files.append(
            CapturedInvestigationRequest(
                request=fact.request,
                request_file=fact.request_file,
                source_event_id=event["event_id"],
                artifact=artifact,
            )
        )
    return tuple(files)


def researcher_words(
    request: InvestigationRequestItem, answers: tuple[dict, ...]
) -> tuple[str, ...]:
    """The researcher's own words a named sequence's quote must stand in (#467): the
    request's text, the contig its typed (or amended) position or region writes, the
    contig its placement put it at (#367), and each retained instruction answer."""
    location = effective_location(request, answers)
    return (
        request.request_text,
        *(
            item.contig
            for item in (location.stated, location.placed, location.region)
            if item is not None
        ),
        *(
            answer["instruction"]
            for answer in answers
            if answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS
        ),
    )


__all__ = [
    "INVESTIGATION_REQUEST_MEDIA_TYPE",
    "CapturedInvestigationRequest",
    "InvestigationRequest",
    "InvestigationRequestItem",
    "InvestigationRequestRefused",
    "InvestigationRequestRegistered",
    "RequestedVariantIdentity",
    "StatedAssembly",
    "StatedPosition",
    "StatedTarget",
    "TargetLocation",
    "effective_location",
    "VersionForm",
    "literal_version_form",
    "investigation_request_document_schema",
    "NotCheckedReason",
    "typed_answers",
    "unread_target_words",
    "untyped",
    "RESEARCHER",
    "not_checked_reason",
    "combined_request",
    "project_captured_investigation_requests",
    "read_investigation_request",
    "read_investigation_request_document",
    "request_target_key",
    "researcher_words",
    "run_investigation_request",
]
