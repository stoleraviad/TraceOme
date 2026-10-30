"""Typed records of the variant-source spike.

Every model is frozen and rejects extra fields, like the product contracts in
``bio_neb.contracts``. They are kept in the tool's local cache and shaped as later
Journey facts; nothing here is a Journey event or grants authority.

`VariantClaim` is the shared variant claim (#331): one variant as a paper, a researcher or
another tool states it, tied to the assembly its coordinates are on. It follows the paper
prototype's `ReportedVariant` (`c8e9edaa:tools/paper_retrieval/contracts.py`), keyed by an
assembly accession instead of a name. Its source is the caller's choice (#469).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.investigation.investigation import RefSnpToolResult
from bio_neb.investigation.investigation_request import RSID_CANONICAL_PATTERN
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)
from bio_neb.target_request.records import ALLELE_PATTERN, LookupStop, SourceFailure

LOOKUP_SCHEMA = "bio-neb.variant-lookup/7"

Route = Literal["dbsnp", "eva"]


class VariantClaim(FrozenContract):
    """One variant as stated, on the assembly its coordinates belong to.

    A claim names an rs ID, a position (contig as written, 1-based position, and REF and ALT
    in VCF form when stated, of any length), HGVS names, or several of these. `request_text`
    is the item's own words, the product request item's field (#342): a researcher's request
    or the paper line that states the claim. Nothing here is checked yet.
    """

    label: str = Field(min_length=1, max_length=200)
    assembly_accession: str = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    assembly_name: str | None = None
    rsid: str | None = Field(default=None, pattern=RSID_CANONICAL_PATTERN)
    contig: str | None = Field(default=None, min_length=1)
    position: int | None = Field(default=None, gt=0)
    ref: str | None = Field(default=None, pattern=ALLELE_PATTERN)
    alt: str | None = Field(default=None, pattern=ALLELE_PATTERN)
    hgvs: tuple[str, ...] = ()
    request_text: str | None = Field(default=None, min_length=1)
    stated_in: str | None = None

    @model_validator(mode="after")
    def _one_variant(self) -> VariantClaim:
        if self.position is not None and self.contig is None:
            raise ValueError("a position claim names its contig")
        if (self.ref is None) != (self.alt is None):
            raise ValueError("REF and ALT are stated together")
        if self.ref is not None and self.position is None:
            raise ValueError("REF and ALT need a position")
        if self.ref is not None and self.ref == self.alt:
            raise ValueError("REF and ALT differ")
        if self.rsid is None and self.position is None and not self.hgvs:
            raise ValueError("a claim names an rs ID, a position or an HGVS name")
        return self


def _refsnp_query(result: RefSnpToolResult) -> str | None:
    """The rsID a RefSNP lookup asked for: the product adapter's request names its number
    last (`NcbiInvestigationSourceAdapter.fetch_refsnp`); `None` for another operation."""
    if result.source.operation != "fetch_refsnp":
        return None
    return "rs" + result.source.request_identity.rsplit("/", 1)[-1]


class ClaimAnswer(FrozenContract):
    """One claim, the source its caller named, and what came back: the source's
    answer, a `not_checked` reason, or the failure its source contact ended in.

    `merge` is the claim's rsID's own dbSNP record when dbSNP merged it into one current
    rsID; the lookup then follows it, as the product's Agent does, and `dbsnp` is the current
    record's result, with the claim's rsID as the requested one. `eva` is an rs claim's
    product EVA lookup result (#366); `position` a position claim's product lookup result
    (`look_up_stated_position`, #369) on either route."""

    claim: VariantClaim
    route: Route
    route_reason: str
    merge: RefSnpToolResult | None = None
    dbsnp: RefSnpToolResult | None = None
    eva: RefSnpToolResult | None = None
    position: RefSnpToolResult | None = None
    not_checked: str | None = None
    failure: SourceFailure | None = None

    @model_validator(mode="after")
    def _one_source_answer(self) -> ClaimAnswer:
        answers = [
            self.dbsnp,
            self.eva,
            self.position,
            self.not_checked,
            self.failure,
        ]
        if sum(answer is not None for answer in answers) != 1:
            raise ValueError(
                "a claim has one source answer, one not_checked reason or one failure"
            )
        if self.dbsnp is not None and self.route != "dbsnp":
            raise ValueError("a dbSNP answer comes from the dbsnp route")
        if self.eva is not None and self.route != "eva":
            raise ValueError("an EVA answer comes from the eva route")
        if self.eva is not None and self.eva.lookup_source != "eva":
            raise ValueError("an rs claim's EVA answer is the product's EVA lookup")
        if self.position is not None and (
            self.claim.position is None
            or self.position.lookup_source
            != ("ncbi_dbsnp" if self.route == "dbsnp" else "eva")
        ):
            raise ValueError("a position answer is the claim's route's position lookup")
        if self.merge is not None and (
            self.route != "dbsnp"
            or self.merge.status != "merged"
            or len(self.merge.merged_into) != 1
            or self.merge.requested_rsid != self.claim.rsid
            or _refsnp_query(self.merge) != self.claim.rsid
            or self.merge.expected_assembly_accession != self.claim.assembly_accession
        ):
            raise ValueError(
                "a merge is the claim's dbSNP record merged into one current rsID"
            )
        # The follow asks for the merge's current rsID and keeps the claim's rsID as the
        # requested one; what dbSNP returns for it is `normalize_refsnp`'s to type
        # (`conflicted` for a record of another rsID, `not_found` with no record).
        if (
            self.merge is not None
            and self.dbsnp is not None
            and (
                self.dbsnp.requested_rsid != self.claim.rsid
                or _refsnp_query(self.dbsnp) != self.merge.merged_into[0]
                or self.dbsnp.expected_assembly_accession
                != self.claim.assembly_accession
            )
        ):
            raise ValueError(
                "a followed lookup asks for the merge's current rsID for the claim's rsID"
            )
        return self


class LookupRecord(FrozenContract):
    """One `look` run: its claims' answers, where it stopped if it did, and how many
    source GETs it made (each one a physical contact of the product's transports)."""

    schema_version: Literal["bio-neb.variant-lookup/7"] = LOOKUP_SCHEMA
    made_at: str
    assembly_accession: str = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    answers: tuple[ClaimAnswer, ...]
    stopped: LookupStop | None
    contact_count: int = Field(ge=0)
