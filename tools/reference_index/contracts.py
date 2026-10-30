"""Typed records of the reference-index spike.

Every model is frozen and rejects extra fields, like the product contracts in
``bio_neb.contracts``. They are kept in the tool's local cache and shaped as later
Journey facts; nothing here is a Journey event or grants authority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation_request import RSID_CANONICAL_PATTERN
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblySourceRecord,
    SourceObservation,
)
from tools.reference_index.fit import FileDigest
from tools.variant_sources.contracts import ClaimAnswer, VariantClaim

CANDIDATES_SCHEMA = "bio-neb.reference-index-candidates/1"

NamingStyle = Literal["refseq", "genbank", "ucsc", "ensembl", "mixed", "unknown"]
ContentPart = Literal["primary", "alts", "patches", "decoys", "hla", "ebv", "unknown"]
Relation = Literal["exact", "same_assembly_series"]


class Fetched(FrozenContract):
    """A whole file fetched into the cache, kept beside it as `<file>.fetched.json`: the
    URL of the request that answered it and the whole file's size and digests."""

    url: str
    destination: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class ToolIdentity(FrozenContract):
    """A local tool by its executable's digest and version line."""

    executable: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_line: str


class FitSummary(FrozenContract):
    """The fit report a candidate was checked with, by its saved digest."""

    verdict: Literal["fits", "fits_after_rename", "refused"]
    sequence_match: Literal["all", "partial", "none"]
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def only_an_all_contig_match_fits(self) -> FitSummary:
        if self.verdict != "refused" and self.sequence_match != "all":
            raise ValueError("only an all-contig M5 match can fit")
        return self


class Candidate(FrozenContract):
    """One prebuilt BWA index offered for the requested assembly."""

    candidate_id: str
    source: Literal["galaxy"]
    catalogue: str
    assembly_accession: str
    relation: Relation
    display_name: str
    location: str
    """The index's datacache URL (its five BWA files are this plus their suffixes)."""
    naming_style: NamingStyle
    contig_count: int | None = Field(ge=1)
    contig_count_basis: Literal["observed_fai", "not_measured"]
    content_class: tuple[ContentPart, ...] = Field(min_length=1)
    content_basis: Literal["observed_fai", "not_measured"]
    index_format: str
    fit: FitSummary | None = None
    auto_pick: bool = False

    @model_validator(mode="after")
    def auto_pick_needs_an_all_contig_match(self) -> Candidate:
        if self.auto_pick != (self.fit is not None and self.fit.verdict != "refused"):
            raise ValueError(
                "auto_pick is set exactly when the fit matched every contig's M5"
            )
        return self


class CandidateList(FrozenContract):
    """Every prebuilt candidate found for one requested assembly accession."""

    schema_version: Literal["bio-neb.reference-index-candidates/1"] = CANDIDATES_SCHEMA
    requested_accession: str
    candidates: tuple[Candidate, ...]


CARD_SCHEMA = "bio-neb.reference-index-card/1"
CHOICE_SCHEMA = "bio-neb.reference-index-choice/1"
BUNDLE_SCHEMA = "bio-neb.reference-index-bundle/1"

ChoiceAction = Literal[
    "build_requested",
    "use_prebuilt",
    "use_newer_prebuilt",
    "build_newer",
    "use_older_prebuilt",
]
_BUILDS = ("build_requested", "build_newer")


class Cost(FrozenContract):
    """What taking an option costs: a CPU build, or a fetch of a prebuilt bundle."""

    work: Literal["cpu_build", "fetch_prebuilt", "not_fetchable"]
    download_bytes: int | None = Field(ge=0)
    basis: Literal["fetched", "estimated_from_fai", "unknown"]
    detail: str

    @model_validator(mode="after")
    def bytes_follow_their_basis(self) -> Cost:
        if (self.download_bytes is None) != (self.basis == "unknown"):
            raise ValueError("download_bytes is given exactly when its basis is known")
        return self


class OptionFit(FrozenContract):
    """How an option's assembly relates to the requested one. `snv_chain` names the
    variant-chain steps a paper's SNVs need when the option is another assembly."""

    sequence: Literal[
        "requested_assembly",
        "exact_sequence",
        "refused",
        "newer_assembly",
        "older_assembly",
    ]
    detail: str
    reasons: tuple[str, ...] = ()
    fit_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    snv_chain: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_follows_the_relation(self) -> OptionFit:
        if (self.sequence == "refused") != bool(self.reasons):
            raise ValueError("reasons are given exactly for a refused fit")
        if (
            self.sequence in ("exact_sequence", "refused")
            and not self.fit_report_sha256
        ):
            raise ValueError("an exact or refused fit names its fit report")
        moved = self.sequence in ("newer_assembly", "older_assembly")
        if moved != bool(self.snv_chain):
            raise ValueError("the SNV chain is named exactly for another assembly")
        return self


class ChoiceOption(FrozenContract):
    """One way to get the reference for the request, with its cost and fit."""

    option_id: str
    action: ChoiceAction
    assembly_accession: str
    assembly_name: str | None
    candidate_id: str | None
    cost: Cost
    fit: OptionFit

    @model_validator(mode="after")
    def builds_have_no_candidate(self) -> ChoiceOption:
        if (self.action in _BUILDS) != (self.candidate_id is None):
            raise ValueError(
                "a build has no prebuilt candidate; a prebuilt option has one"
            )
        if (self.action in _BUILDS) != (self.cost.work == "cpu_build"):
            raise ValueError("a build costs a CPU build; a prebuilt option a fetch")
        return self


class StoredBundle(FrozenContract):
    """A fitted bundle's content in the local store, under its content digest, in the
    product's eight prepared roles (`bio_neb.parabricks.worker_exchange.PREPARED_ROLES`).
    The digest covers the fetched content (the FASTA and the five BWA files); the
    product's assembly (`tools.reference_index.bundle`) placed them, renamed to NCBI's
    names after a `fits_after_rename` fit, and made the `.dict` (and a renamed FASTA's
    `.fai`) with pysam's samtools, `made_with`. Which assembly and candidate a fit bound
    it to is each `Choice`'s, with its fit report (ADR 0029)."""

    schema_version: Literal["bio-neb.reference-index-bundle/1"] = BUNDLE_SCHEMA
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FileDigest, ...]
    made_with: ToolIdentity


class SnvClaim(FrozenContract):
    """One SNV as the request states it, on the requested assembly (1-based)."""

    label: str = Field(min_length=1)
    assembly_accession: str
    contig: str = Field(min_length=1)
    position: int = Field(ge=1)
    ref: str = Field(pattern=r"^[ACGTN]+$")
    alt: str = Field(pattern=r"^[ACGTN]+$")


class Choice(FrozenContract):
    """The reference chosen for one request: picked without asking on an all-contig M5
    match, or an option taken from the choice card. Written once per request. When the
    chosen assembly is another one, the request's SNVs are handed to the variant chain
    (`option.fit.snv_chain`) as `snvs_to_move`."""

    schema_version: Literal["bio-neb.reference-index-choice/1"] = CHOICE_SCHEMA
    requested: AssemblySourceRecord
    decision: Literal["auto_pick", "taken"]
    option: ChoiceOption
    card_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The fit that bound this choice's assembly and candidate to the stored bundle."""
    snvs_to_move: tuple[SnvClaim, ...] = ()

    @model_validator(mode="after")
    def the_decision_follows_its_evidence(self) -> Choice:
        requested = self.requested.assembly_accession
        exact = (
            self.option.action == "use_prebuilt"
            and self.option.fit.sequence == "exact_sequence"
            and self.option.assembly_accession == requested
        )
        if self.decision == "auto_pick" and not exact:
            raise ValueError(
                "only an all-contig M5 match to the request is auto-picked"
            )
        if self.decision == "auto_pick" and (
            self.option.fit.fit_report_sha256 != self.fit_report_sha256
        ):
            raise ValueError("an auto-pick is bound by the fit its option names")
        if (self.decision == "taken") != (self.card_sha256 is not None):
            raise ValueError("a taken option names its card; an auto-pick has none")
        if self.snvs_to_move and not self.option.fit.snv_chain:
            raise ValueError("SNVs are handed over only for another assembly")
        if any(snv.assembly_accession != requested for snv in self.snvs_to_move):
            raise ValueError("SNVs to move are stated on the requested assembly")
        return self


class SpeciesListing(FrozenContract):
    """The species' NCBI assemblies as the product adapter listed them (at most one
    page of 100; `truncated` says when NCBI holds more)."""

    ncbi_tax_id: int = Field(gt=0)
    assemblies: tuple[AssemblySourceRecord, ...] = Field(max_length=100)
    total_count: int | None = Field(ge=0)
    truncated: bool
    reference_genome: str | None
    """The assembly NCBI marks as the species' reference genome."""
    reference_basis: Literal["listing", "described_galaxy_row", "not_found"]
    """Where NCBI said so: in the listing, or (when the listing is truncated without
    one) in the description of one of the species' Galaxy rows."""

    @model_validator(mode="after")
    def a_reference_has_its_basis(self) -> SpeciesListing:
        if (self.reference_genome is None) != (self.reference_basis == "not_found"):
            raise ValueError("a reference genome is given exactly with its basis")
        return self


class ChoiceCard(FrozenContract):
    """The question asked when nothing can be picked without asking: build the requested
    assembly, use a prebuilt one, or use a newer assembly, each with its cost and fit.
    The default is the paper's own reference (plan decision 8)."""

    schema_version: Literal["bio-neb.reference-index-card/1"] = CARD_SCHEMA
    requested: AssemblySourceRecord
    species: SpeciesListing
    options: tuple[ChoiceOption, ...] = Field(min_length=1)
    default_option_id: str
    not_offered: tuple[str, ...] = ()
    sources: tuple[SourceObservation, ...]

    @model_validator(mode="after")
    def options_follow_the_request(self) -> ChoiceCard:
        ids = [option.option_id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids are unique")
        requested = self.requested.assembly_accession
        defaults = [o for o in self.options if o.option_id == self.default_option_id]
        if [(o.action, o.assembly_accession) for o in defaults] != [
            ("build_requested", requested)
        ]:
            raise ValueError("the default is building the requested assembly")
        for option in self.options:
            same = option.fit.sequence in (
                "requested_assembly",
                "exact_sequence",
                "refused",
            )
            if same != (option.assembly_accession == requested):
                raise ValueError(f"{option.option_id}: fit and assembly disagree")
        return self


class AskAnswer(FrozenContract):
    """One request's answer: a pick, a taken option, a choice card, or a cache hit, and
    the requests this answer sent."""

    requested_accession: str
    outcome: Literal["picked", "taken", "card", "cache_hit"]
    choice: Choice | None = None
    bundle: StoredBundle | None = None
    card: ChoiceCard | None = None
    contacts: int = Field(ge=0)

    @model_validator(mode="after")
    def records_follow_the_outcome(self) -> AskAnswer:
        carded = self.outcome == "card"
        chosen = (self.choice is not None, self.bundle is not None)
        if (self.card is not None, chosen) != (carded, (not carded, not carded)):
            raise ValueError(
                "a card answers with the card alone; any other outcome with its "
                "choice and stored bundle"
            )
        return self


MOVE_SCHEMA = "bio-neb.reference-index-move/2"

Base = Literal["A", "C", "G", "T"]
MoveResult = Literal[
    "unique", "several_places", "not_found", "ref_changed", "strand_flipped"
]
MoveMethod = Literal["database_placement", "flank_realignment"]


class SnvToMove(FrozenContract):
    """One SNV as stated on the source assembly A (1-based, one-base REF and ALT), to be
    placed on another assembly B.

    `assembly` is A's NCBI accession when NCBI names one, else the name the source states
    (UCSC sacCer1 has no NCBI assembly). `contig` is the sequence's name in A's FASTA. An rs
    ID, when the SNV has one, is looked up on B first (`move.py`)."""

    label: str = Field(min_length=1)
    assembly: str = Field(min_length=1)
    contig: str = Field(min_length=1)
    position: int = Field(ge=1)
    ref: Base
    alt: Base
    rsid: str | None = Field(default=None, pattern=RSID_CANONICAL_PATTERN)

    @model_validator(mode="after")
    def _a_change(self) -> SnvToMove:
        if self.ref == self.alt:
            raise ValueError("REF and ALT differ")
        return self

    @classmethod
    def of_claim(cls, claim: VariantClaim) -> SnvToMove:
        """The shared variant claim (#331) of one SNV at a position."""
        if claim.position is None or claim.ref is None:
            raise ValueError(f"{claim.label}: only a claim with a position moves")
        if len(claim.ref) != 1 or len(claim.alt) != 1:
            raise ValueError(f"{claim.label}: only an SNV moves")
        return cls(
            label=claim.label,
            assembly=claim.assembly_accession,
            contig=claim.contig,
            position=claim.position,
            ref=claim.ref,
            alt=claim.alt,
            rsid=claim.rsid,
        )

    @classmethod
    def of_snv(cls, snv: SnvClaim) -> SnvToMove:
        """#330's choice record keeps the SNVs to move as `SnvClaim`s."""
        return cls(
            label=snv.label,
            assembly=snv.assembly_accession,
            contig=snv.contig,
            position=snv.position,
            ref=snv.ref,
            alt=snv.alt,
        )


class Place(FrozenContract):
    """Where the SNV's base lands on B: in one alignment of its flanks (`cigar` is bwa's;
    `mapq` and `score` only for the primary alignment, not for an alternative bwa lists in
    `XA`), or in a database placement (on B's forward strand, no alignment). `ref` is B's
    base there, forward strand, as B's FASTA has it."""

    contig: str
    position: int = Field(ge=1)
    strand: Literal["+", "-"]
    ref: str | None
    mapq: int | None = Field(default=None, ge=0)
    cigar: str | None = None
    score: int | None = None


class MovedSnv(FrozenContract):
    """One SNV's result on B and the method that gave it, with A's coordinate kept.

    `place` is the accepted place (`unique`, `ref_changed`, `strand_flipped`); `places`
    is every place the method reported, alternatives included. `target_ref` and
    `target_alt` are the SNV's alleles on B's forward strand when B's base is A's REF on
    the aligned strand. `database` is the variant source's answer for an SNV with an rs ID
    (#331), kept when its placement was used and when it was not."""

    snv: SnvToMove
    target_assembly: str
    result: MoveResult
    method: MoveMethod
    place: Place | None
    places: tuple[Place, ...] = ()
    target_ref: Base | None = None
    target_alt: Base | None = None
    database: ClaimAnswer | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _the_result_follows_the_place(self) -> MovedSnv:
        placed = self.result in {"unique", "ref_changed", "strand_flipped"}
        if placed != (self.place is not None):
            raise ValueError(f"a {self.result} result has a place only when placed")
        alleles = self.result in {"unique", "strand_flipped"}
        if alleles != (self.target_ref is not None and self.target_alt is not None):
            raise ValueError(
                "alleles on B are given for unique and strand_flipped only"
            )
        if self.method == "database_placement" and self.database is None:
            raise ValueError("a database placement keeps the database's answer")
        return self


class FastaRead(FrozenContract):
    """A FASTA a move read, by its size and `.fai` digest."""

    path: str
    byte_count: int = Field(ge=0)
    fai_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MoveRecord(FrozenContract):
    """One `move` run: SNVs on one source assembly placed on one target's stored index.
    The query FASTA and `bwa mem`'s SAM are kept by digest beside the record."""

    schema_version: Literal["bio-neb.reference-index-move/2"] = MOVE_SCHEMA
    made_at: str
    source_assembly: str
    source_fasta: FastaRead
    target_assembly: str
    target_bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_fasta: FastaRead
    flank_bases: int = Field(ge=1)
    min_mapq: int = Field(ge=0)
    min_score_fraction: float = Field(gt=0, le=1)
    bwa: ToolIdentity
    samtools: ToolIdentity
    query_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    alignment_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    moves: tuple[MovedSnv, ...]
    contact_count: int = Field(default=0, ge=0)
