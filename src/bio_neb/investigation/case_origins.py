"""Every case across a collection view, by its typed origin (#475, ADR 0049).

One place defines each case on #474's pattern (``qualification_origins``): its
code, the typed fields it copies from the requests and lookups that show it, its
audit sentence and its plain words. The audit sentence is what the CLI shows; no
reader parses it. A case reports only: it changes no admission, claim scope or
authority (``collection_view``).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.contracts import StrictFrozenContract
from bio_neb.investigation.qualification_origins import (
    RequestIssueRejected,
    RequestNotChecked,
    RequestRejected,
    RequestUnresolved,
)


class _Case(StrictFrozenContract):
    def audit(self) -> str:
        """The case as the view's audit sentence states it."""
        raise NotImplementedError

    def words(self) -> str:
        """The same case in plain words, naming no code or field."""
        raise NotImplementedError


class MergedIdentity(_Case):
    code: Literal["merged_identity"] = "merged_identity"
    requested: StrictStr
    verified: StrictStr

    def audit(self) -> str:
        return f"dbSNP merged {self.requested} into {self.verified}."

    def words(self) -> str:
        return (
            f"dbSNP merged {self.requested} into {self.verified}, so the request "
            f"reads as {self.verified}"
        )


class SharedIdentity(_Case):
    code: Literal["shared_identity"] = "shared_identity"
    requests: tuple[StrictStr, ...] = Field(min_length=2)
    rsid: StrictStr

    def audit(self) -> str:
        return (
            f"The requests for {', '.join(self.requests)} all resolve to {self.rsid}."
        )

    def words(self) -> str:
        return f"{', '.join(self.requests)} all name the same variant, {self.rsid}"


class ContradictorySources(_Case):
    """Retained records that disagree: one rsID's records on its locus or alleles,
    or the records at one locus on its reference allele."""

    code: Literal["contradictory_sources"] = "contradictory_sources"
    rsid: StrictStr | None = None
    contig: StrictStr | None = None
    position: StrictInt | None = None

    @model_validator(mode="after")
    def one_record_or_locus(self) -> "ContradictorySources":
        locus = (self.contig is not None, self.position is not None)
        if locus not in {(True, True), (False, False)} or (self.rsid is None) != all(
            locus
        ):
            raise ValueError("a contradiction names one rsID or one locus")
        return self

    def audit(self) -> str:
        if self.rsid is not None:
            return (
                f"The retained records of {self.rsid} disagree on its locus or alleles."
            )
        return (
            f"The records at {self.contig}:{self.position} state different reference "
            "alleles."
        )

    def words(self) -> str:
        if self.rsid is not None:
            return (
                f"the sources' records of {self.rsid} disagree on its position or "
                "alleles"
            )
        return (
            f"the records at {self.contig}:{self.position} disagree on the reference "
            "allele"
        )


class SharedLocus(_Case):
    code: Literal["shared_locus"] = "shared_locus"
    identities: tuple[StrictStr, ...] = Field(min_length=2)
    contig: StrictStr
    position: StrictInt

    def audit(self) -> str:
        return (
            f"{', '.join(self.identities)} share the locus "
            f"{self.contig}:{self.position}."
        )

    def words(self) -> str:
        return (
            f"{', '.join(self.identities)} are at the same position, "
            f"{self.contig}:{self.position}"
        )


class MultiAlt(_Case):
    code: Literal["multi_alt"] = "multi_alt"
    request: StrictStr
    alts: tuple[StrictStr, ...] = Field(min_length=2)

    def audit(self) -> str:
        return (
            f"{self.request} lists {len(self.alts)} alternate alleles "
            f"({', '.join(self.alts)}), each its own Work item."
        )

    def words(self) -> str:
        return (
            f"{self.request} lists {len(self.alts)} alternate alleles "
            f"({', '.join(self.alts)}), each proposed as its own position"
        )


NoWorkReason = Annotated[
    Union[RequestRejected, RequestIssueRejected, RequestNotChecked, RequestUnresolved],
    Field(discriminator="code"),
]


def no_work_note(reason: NoWorkReason) -> str:
    """Why a request has no Work item, in its origin's audit note."""
    if isinstance(reason, (RequestRejected, RequestIssueRejected)):
        return f"rejected {reason.note()}"
    if isinstance(reason, RequestNotChecked):
        # Deferred out of the Run after its lookup.
        deferred = reason.reason_code == "position_without_alleles"
        return f"{'deferred' if deferred else 'not checked'}: {reason.note()}"
    return reason.note()


class NoWorkItem(_Case):
    """A request its check left without a Work item, with the origin of why."""

    code: Literal["no_work_item"] = "no_work_item"
    request: StrictStr
    reason: NoWorkReason

    def audit(self) -> str:
        sentence = f"{self.request} has no Work item: {no_work_note(self.reason)}"
        return sentence if sentence.endswith((".", "?", "!")) else sentence + "."

    def words(self) -> str:
        return f"no position was searched for {self.request}: {self.reason.words()}"


class OverlappingRecord(_Case):
    code: Literal["overlapping_record"] = "overlapping_record"
    request: StrictStr

    def audit(self) -> str:
        return f"A record that overlaps {self.request}'s locus sits beside its target."

    def words(self) -> str:
        return f"a variant call that overlaps {self.request}'s position sits beside it"


class MixedAltOutcomes(_Case):
    code: Literal["mixed_alt_outcomes"] = "mixed_alt_outcomes"
    request: StrictStr

    def audit(self) -> str:
        return f"{self.request}'s alternate alleles have different lookup outcomes."

    def words(self) -> str:
        return f"{self.request}'s alternate alleles had different results in the VCF"


class TrimmedMatch(_Case):
    code: Literal["trimmed_match"] = "trimmed_match"
    request: StrictStr

    def audit(self) -> str:
        return (
            f"{self.request} matched a multi-base record only once its alleles were "
            "trimmed."
        )

    def words(self) -> str:
        return (
            f"{self.request} matched a longer variant call only once its alleles were "
            "trimmed"
        )


CaseOrigin = Annotated[
    Union[
        MergedIdentity,
        SharedIdentity,
        ContradictorySources,
        SharedLocus,
        MultiAlt,
        NoWorkItem,
        OverlappingRecord,
        MixedAltOutcomes,
        TrimmedMatch,
    ],
    Field(discriminator="code"),
]

__all__ = [
    "CaseOrigin",
    "ContradictorySources",
    "MergedIdentity",
    "MixedAltOutcomes",
    "MultiAlt",
    "NoWorkItem",
    "NoWorkReason",
    "OverlappingRecord",
    "SharedIdentity",
    "SharedLocus",
    "TrimmedMatch",
    "no_work_note",
]
