"""A paper's proposed targets (`bio-neb.paper-targets/1`): the one record paper reading
writes and placement reads (#367), with the lines and values it keeps from the paper.

Every model is frozen and rejects extra fields. Nothing here is a Journey event or grants
authority: the record is the input placement turns into a Run's request file.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation_request import (
    CodingChangeDraft,
    InvestigationRequestItem,
    ProteinChangeDraft,
)


class PaperIdentity(FrozenContract):
    """What resolution established about the paper. Absent values are unknown.

    A paper the researcher supplied as a file, with no DOI, PMID or PMCID, is named by
    the Run's ``paper_id`` and its file's name (owner, #383 A1b)."""

    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str | None = None
    journal: str | None = None
    year: str | None = None
    open_access: bool | None = None
    resolved_by: Literal["europe_pmc", "web_search", "user_supplied"]
    landing_url: str | None = None
    paper_id: str | None = Field(default=None, pattern=r"^paper-[0-9a-f]{64}$")
    file_name: str | None = None


# ---------------------------------------------------------------------------
# What a paper states: its lines, and the reference and variant values read in them
# ---------------------------------------------------------------------------


class SourceLine(FrozenContract):
    """One line or sentence of a retrieved document: ``text`` is exactly ``start:end`` of it."""

    document_id: str
    section: str  # the heading above the line, as the document gives it
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


def candidate_id(kind: Literal["r", "v"], key: str) -> str:
    return f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def reference_key(name: str) -> str:
    """CanFam 3.1, CanFam3.1 and canfam-3.1 are one name."""

    return re.sub(r"[\s._-]+", "", name).lower()


class ReferenceCandidate(FrozenContract):
    """A reference assembly as the paper names it, with every line that names it."""

    candidate_id: str
    name: str  # the first spelling, in document order
    spellings: tuple[str, ...] = Field(min_length=1)
    sources: tuple[SourceLine, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _derived_identity(self) -> ReferenceCandidate:
        keys = {reference_key(spelling) for spelling in self.spellings}
        if self.name != self.spellings[0] or len(keys) != 1:
            raise ValueError("a reference candidate is one name in its spellings")
        if self.candidate_id != candidate_id("r", keys.pop()):
            raise ValueError("the candidate id is not derived from the name")
        return self


def is_protein_name(name: str) -> bool:
    return bool(re.search(r"(?:^|:)p\.", name))


class VariantCandidate(FrozenContract):
    """One variant the paper reports, with every line that names it.

    Values are copied from the paper: the contig as written (``10``, ``X``, ``NC_006592.3``),
    the 1-based position, REF and ALT, HGVS names and rsIDs. Absent values are not stated.
    The id is derived from the values, so the same paper text gives the same ids.
    """

    candidate_id: str
    contig: str | None = None
    position: int | None = Field(default=None, gt=0)
    ref: str | None = None
    alt: str | None = None
    hgvs: tuple[str, ...] = ()
    rsids: tuple[str, ...] = ()
    sources: tuple[SourceLine, ...] = Field(min_length=1)

    @property
    def snv(self) -> bool:
        """REF and ALT are one base each, and differ."""

        bases = ("A", "C", "G", "T")
        return self.ref in bases and self.alt in bases and self.ref != self.alt

    @property
    def key(self) -> str:
        """The locus when stated, else the rsID, else the first DNA name, else the protein name."""

        if self.position is not None:
            alleles = f":{self.ref}>{self.alt}" if self.ref and self.alt else ""
            return f"{self.contig or '?'}:{self.position}{alleles}"
        dna = [name for name in self.hgvs if not is_protein_name(name)]
        return (self.rsids or dna or self.hgvs)[0]

    @model_validator(mode="after")
    def _derived_identity(self) -> VariantCandidate:
        if self.position is None and not (self.hgvs or self.rsids):
            raise ValueError(
                "a variant candidate names a position, an HGVS name or an rsID"
            )
        if self.candidate_id != candidate_id("v", self.key):
            raise ValueError("the candidate id is not derived from the values")
        return self


class StatedReference(FrozenContract):
    """The reference as the paper states it; lane Rf resolves it, by its name and then by
    each other name the paper writes for it (``aliases``, as written; #411 LF2)."""

    candidate: ReferenceCandidate
    reason: str
    aliases: tuple[str, ...] = ()


class Finding(FrozenContract):
    """One finding's evidence: what the model read, kept by code where the paper writes it.

    ``passages`` are exact spans of its documents, each ``start:end``: first the quote (the
    words that name the variant), then the paragraphs or table rows the finding rests on, the
    quote's own paragraph or row among them. ``variants`` holds the one variant its stated
    values make, if it states a position, a change name or an rsID: each value as the paper
    writes it, with the passages the general parser read it in (``sources``). The record
    keeps the field names of ``paper-targets/1`` (owner, #395 decision 3); ``unlocated`` is
    no longer filled.
    """

    variants: tuple[VariantCandidate, ...] = ()
    passages: tuple[SourceLine, ...] = ()
    reason: str
    gene: str | None = None
    unlocated: str | None = None  # why no located candidate is this variant

    @model_validator(mode="after")
    def _evidence(self) -> Finding:
        if not (self.variants or self.passages):
            raise ValueError("a finding holds a candidate or a passage")
        return self


# ---------------------------------------------------------------------------
# Findings as proposed targets: main's request shape, asserted by the paper
# ---------------------------------------------------------------------------

TargetKind = Literal["snv", "indel", "mnv", "not_stated"]
# Why main cannot check a target by what it states: it states no rsID and no position
# (``not_located``), or a position without its alleles and no rsID (``no_alleles``).
# Main asks the researcher in both cases.
NotChecked = Literal["not_located", "no_alleles"]


def not_checked(request: InvestigationRequestItem) -> NotChecked | None:
    """Why main cannot check ``request`` by its typed target, or None when it can: an rsID,
    or a position with REF and ALT."""

    if request.rsid is not None:
        return None
    if request.position is None:
        return "not_located"
    return "no_alleles" if request.position.ref is None else None


class TargetChange(FrozenContract):
    """The gene and its change as the target's line writes them (``PBP2 (Y127X)``,
    ``RAPGEF5 (c.2624C>A p.Ser875*)``), for lane Rf to place (G15); nothing is parsed or
    normalized. ``transcript`` is the one transcript id the line states, if it states one.
    ``protein`` and ``coding`` are the change as a model types it (#473 Q2 a); no reader
    types them before #383, and G15 answers a change with neither not checked (#473
    owner Q12 a)."""

    gene: str
    text: str
    transcript: str | None = None
    protein: ProteinChangeDraft | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coding: CodingChangeDraft | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PaperTarget(FrozenContract):
    """One proposed target: main's request item, asserted by the paper, and what it states.

    ``request`` is main's ``InvestigationRequestItem``: the paper's line (``quote``), the one
    rsID that line states, the position its located candidates state and, with a position, the
    paper's stated genome version and one line citing it. ``findings`` are every finding the
    target stands for, joined by main's target identity, with every value and line the paper
    states. ``kind`` is the variant kind the paper states; ``not_checked`` says why main cannot
    check the target by its typed fields, if it cannot (main asks the researcher). ``change``
    is the gene and change the quote writes, if it writes one.
    """

    request: InvestigationRequestItem
    quote: SourceLine
    kind: TargetKind
    not_checked: NotChecked | None = None
    change: TargetChange | None = None
    findings: tuple[Finding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _the_paper_line(self) -> PaperTarget:
        if self.request.request_text != self.quote.text:
            raise ValueError("a target's request is its quote")
        if self.not_checked != not_checked(self.request):
            raise ValueError("a target is not checked only when main cannot check it")
        if self.request.stated_assembly is not None and self.request.position is None:
            raise ValueError("a stated version belongs to a stated position")
        return self


class PaperTargets(FrozenContract):
    """One answer's findings as proposed targets, asserted by its paper (``paper:doi:…``).

    ``reference`` is the answer's stated reference, whole: its name, every spelling and every
    line that names it, and the reason. A positioned target's request cites one of those lines.
    ``schema_version`` names this record's shape: lane Rf reads it as data, never as code.
    """

    schema_version: Literal["bio-neb.paper-targets/1"] = "bio-neb.paper-targets/1"
    question: str
    paper: PaperIdentity
    asserted_by: str
    reference: StatedReference | None = None
    targets: tuple[PaperTarget, ...] = ()
    not_answered: str | None = None


CITED_LINE_LIMIT = 4_000
"""The longest line a request cites for its version (`StatedAssembly.cited_line`)."""


def cited_window(text: str, name: str) -> str:
    """A paper's line cut to the cited-line limit around ``name`` (the version it
    states), trimmed: the one rule for citing a line too long for a request."""

    start = max(0, max(text.find(name), 0) - CITED_LINE_LIMIT // 2)
    return text[start : start + CITED_LINE_LIMIT].strip()
