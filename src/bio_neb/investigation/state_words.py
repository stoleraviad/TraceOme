"""Every recorded state a request or Work item ends in, in words, from one place (#475).

One definition per state: the readable report's results cell and its short
table words, whether the VCF was searched and whether the cell says the target allele was carried, the claim word
the report's explainer records, and the CLI's words. The readable report, the CLI
and the explainer read these and keep no table of their own. Why a state was
reached is its typed origin's (``qualification_origins``); these words name only
the state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bio_neb.investigation.qualification_origins import DEFERRED_WITHOUT_ALLELES

# An item's state once the Run has ended.
ItemState = Literal[
    "observed",
    "failed",
    "approved_not_observed",
    "not_approved",
    "unselected_candidate",
    "removed",
    "deferred",
]
# A request's state once the Run has ended: its candidates carry their own
# states; a request with none says why.
RequestState = Literal[
    "candidates",
    "unresolved",
    "rejected",
    "not_checked",
    "removed",
    "deferred",
    "not_proposed",
    "never_run",
]
NOT_RECORDED = "state_not_recorded"


@dataclass(frozen=True)
class ItemWords:
    """A Work item's lookup outcome or ended state in words (#460 N4 A)."""

    cell: str
    searched: bool
    carries: bool
    claim: str
    # The CLI's words for an ended state; an outcome shows its own code.
    cli: str | None = None
    # The readable report's results table: a short phrase per outcome or
    # state, beside the VCF's own cells.
    table: str = ""


@dataclass(frozen=True)
class RequestWords:
    """A request's ended state in words."""

    cell: str
    claim: str
    cli: str
    # The explainer's row words when no origin says more; the cell's words unless
    # the state names positions of its own.
    row_words: str | None = None

    @property
    def row(self) -> str:
        return self.row_words or self.cell


ITEM_WORDS = {
    "exact_allele_observed": ItemWords(
        "Found: the sample carries the target allele",
        True,
        True,
        "carries_target_allele",
        table="called: carries the target allele",
    ),
    "target_allele_not_carried": ItemWords(
        "Called here: the target allele is not carried",
        True,
        False,
        "target_allele_not_carried",
        table="called: target allele not carried",
    ),
    "target_genotype_unresolved": ItemWords(
        "Called here: the genotype is unresolved",
        True,
        False,
        "genotype_unresolved",
        table="called: genotype unresolved",
    ),
    "locus_present_identity_mismatch": ItemWords(
        "A different variant was called here",
        True,
        False,
        "different_variant_at_locus",
        table="a different variant is called here",
    ),
    "locus_overlapped": ItemWords(
        "A nearby variant call overlaps this position",
        True,
        False,
        "locus_overlapped",
        table="an overlapping call",
    ),
    "no_record": ItemWords(
        "Searched the VCF: no variant call here; genotype not established",
        True,
        False,
        "searched_no_record",
        table="no call",
    ),
    "ambiguous_records": ItemWords(
        "Several variant calls here; not decided",
        True,
        False,
        "ambiguous_records",
        table="several calls; not decided",
    ),
    "refused": ItemWords(
        "The VCF search was refused",
        True,
        False,
        "lookup_refused",
        table="search refused",
    ),
    # A region's search has no claim word of its own yet.
    "region_observed": ItemWords(
        "Searched the VCF across this region",
        True,
        False,
        NOT_RECORDED,
        table="region searched",
    ),
    "failed": ItemWords(
        "The VCF search failed",
        False,
        False,
        "lookup_failed",
        "failed",
        table="search failed",
    ),
    "approved_not_observed": ItemWords(
        "Approved; the Run ended before the VCF was searched",
        False,
        False,
        "approved_not_looked_up",
        "approved, not observed",
        table="approved; not searched",
    ),
    "not_approved": ItemWords(
        "Not approved for a VCF search",
        False,
        False,
        "not_approved",
        "proposed, not approved",
        table="not approved",
    ),
    "unselected_candidate": ItemWords(
        "Another allele was chosen instead",
        False,
        False,
        "unselected_candidate",
        "a candidate the approval did not select",
        table="another allele chosen",
    ),
    "removed": ItemWords(
        "Removed by the researcher", False, False, "removed", "removed", table="removed"
    ),
    "deferred": ItemWords(
        "Deferred by the researcher",
        False,
        False,
        "deferred",
        "deferred",
        table="deferred",
    ),
}
_ITEM_NOT_RECORDED = ItemWords(
    "State not recorded", False, False, NOT_RECORDED, table="state not recorded"
)

_REQUEST_WORDS: dict[str, RequestWords] = {
    "unresolved": RequestWords(
        "not placed on this reference; the VCF was not searched",
        "unresolved",
        "unresolved",
    ),
    "rejected": RequestWords(
        "rejected; the VCF was not searched", "rejected", "rejected"
    ),
    "not_checked": RequestWords(
        "could not be checked as stated; the VCF was not searched",
        "not_checked",
        "not checked",
    ),
    "removed": RequestWords(
        "removed by the researcher; the VCF was not searched", "removed", "removed"
    ),
    "deferred": RequestWords(
        "deferred by the researcher; the VCF was not searched", "deferred", "deferred"
    ),
    "not_proposed": RequestWords(
        "checked, but the Run ended before a search was proposed",
        "not_proposed",
        "checked, not proposed",
    ),
    "never_run": RequestWords(
        "its check did not finish before the Run ended", "never_run", "never run"
    ),
    "candidates": RequestWords(
        "no candidate was searched",
        "candidates",
        "candidates",
        "each position proposed for it is its own target",
    ),
}
# A position stated without alleles or an rsID, deferred out of the Run after its
# lookup: its typed origin's own words.
_DEFERRED_WITHOUT_ALLELES = RequestWords(
    DEFERRED_WITHOUT_ALLELES,
    "not_checked",
    "deferred out of the Run",
)
_REQUEST_NOT_RECORDED = RequestWords("state not recorded", NOT_RECORDED, "not recorded")


def item_words(key: str | None) -> ItemWords:
    """An item's words by its lookup outcome, or by its state when it has none."""
    return ITEM_WORDS.get(key or "", _ITEM_NOT_RECORDED)


def request_words(state: str | None, reason_code: str | None = None) -> RequestWords:
    """A request's words by its state and, for a request not checked, its origin's
    reason code."""
    if state == "not_checked" and reason_code == "position_without_alleles":
        return _DEFERRED_WITHOUT_ALLELES
    return _REQUEST_WORDS.get(state or "", _REQUEST_NOT_RECORDED)
