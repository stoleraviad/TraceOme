"""Per-request explanation members and their combination (#408).

Each request of a published report is explained by its own member Attempt over
that request's own rows, and one combination Attempt then answers over the
admitted members. Each row's claim word is its recorded state in the
researcher's words (owner, #408 part 3), filled by code (#448 D1) from the state's
one owner (``state_words``, #475): the model
explains the claims and never writes them. Qualifications are placed by the
rows their ids concern (#448 D2, owner Q5 b).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.contracts import StrictFrozenContract
from bio_neb.investigation import state_words
from bio_neb.journey import canonical_json
from bio_neb.reporting.explanation_rows import ExplanationRow, ReportQualification

# The combination's typed outcome Tool; each member records its own with
# ``record_explanation``.
COMBINATION_OUTPUT_TOOL = "record_combined_answer"

ClaimWord = Literal[
    "carries_target_allele",
    "target_allele_not_carried",
    "genotype_unresolved",
    "different_variant_at_locus",
    "locus_overlapped",
    "searched_no_record",
    "ambiguous_records",
    "lookup_refused",
    "lookup_failed",
    "approved_not_looked_up",
    "not_approved",
    "unselected_candidate",
    "removed",
    "deferred",
    "candidates",
    "unresolved",
    "rejected",
    "not_proposed",
    "never_run",
    "not_checked",
    "state_not_recorded",
]


def row_claim(row: ExplanationRow) -> str:
    """The claim word a row's recorded state reads as; never the model's choice."""
    if row.kind == "request":
        return state_words.request_words(row.state).claim
    if row.state == "observed":
        return state_words.item_words(row.outcome).claim
    return state_words.item_words(row.state).claim


def member_rows(
    rows: Sequence[ExplanationRow],
) -> tuple[tuple[str, tuple[ExplanationRow, ...]], ...]:
    """Each request's member and its own rows (its request row, then its items),
    in the report's order and numbering."""
    grouped: dict[str, list[ExplanationRow]] = {}
    for row in rows:
        grouped.setdefault(row.request_member_id, []).append(row)
    return tuple((member, tuple(own)) for member, own in grouped.items())


def _names(rows: Sequence[ExplanationRow]) -> re.Pattern[str]:
    """What names one request in report text or a retained payload: its member
    id, its items' Work item ids and its own subject, each as a whole word."""
    request = next(row for row in rows if row.kind == "request")
    tokens = {
        request.request_member_id,
        request.subject,
        *(row.work_item_id for row in rows if row.work_item_id is not None),
    }
    return re.compile(
        "|".join(
            rf"(?<!\w){re.escape(token)}(?!\w)"
            for token in sorted(tokens, key=lambda token: (-len(token), token))
        )
    )


def _run_wide(
    rows: Sequence[ExplanationRow], qualification: ReportQualification
) -> bool:
    """A qualification of the Run as a whole: it concerns no row, or every row
    with a lookup in the report (owner Q5 b, #448)."""
    looked_up = {row.number for row in rows if row.kind == "position" and row.outcome}
    return not qualification.targets or (
        bool(looked_up) and looked_up <= set(qualification.targets)
    )


def member_qualifications(
    rows: Sequence[ExplanationRow],
    member_id: str,
    qualifications: Sequence[ReportQualification],
) -> tuple[ReportQualification, ...]:
    """The qualifications whose ids concern some of this request's rows and are
    not the Run's own, each with only this request's rows."""
    own = {row.number for row in rows if row.request_member_id == member_id}
    return tuple(
        item.model_copy(
            update={"targets": tuple(row for row in item.targets if row in own)}
        )
        for item in qualifications
        if own & set(item.targets) and not _run_wide(rows, item)
    )


def combination_qualifications(
    rows: Sequence[ExplanationRow], qualifications: Sequence[ReportQualification]
) -> tuple[ReportQualification, ...]:
    """The qualifications no request's member is given: the Run's own."""
    return tuple(item for item in qualifications if _run_wide(rows, item))


def member_scope(
    events: Sequence[Mapping[str, Any]],
    scope: Iterable[str],
    rows: Sequence[ExplanationRow],
    publication_event_id: str,
) -> tuple[str, ...]:
    """The report's evidence events whose payload names this request, and the
    report's publication: the only events the member's search may read."""
    names = _names(rows)
    allowed = set(scope)
    return tuple(
        sorted(
            {
                publication_event_id,
                *(
                    event["event_id"]
                    for event in events
                    if event["event_id"] in allowed
                    and names.search(canonical_json(event.get("payload", {})))
                ),
            }
        )
    )


class RowClaim(StrictFrozenContract):
    """One target's claim as the explanation retains it: its number, the word its
    recorded state reads as and the genotype it carries."""

    target: StrictInt = Field(ge=1)
    claim: ClaimWord
    genotype: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def row_claims(rows: Sequence[ExplanationRow]) -> tuple[RowClaim, ...]:
    """Every row's claim, filled from its recorded state (#448 D1)."""
    return tuple(
        RowClaim(target=row.number, claim=row_claim(row), genotype=row.genotype)
        for row in rows
    )


NOT_EXPLAINED = "not_explained"


class CombinationSection(StrictFrozenContract):
    """One request as the combination is bound to it: its admitted member's
    claim words, or the decision that settled its member as not explained."""

    request_member_id: StrictStr
    request: StrictInt = Field(ge=1)
    subject: StrictStr
    status: Literal["explained", "not_explained"]
    # The member Attempt that was admitted, or the one its decision stopped.
    attempt_id: StrictStr
    validation_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The member's claim words, sorted and each once.
    claims: tuple[StrictStr, ...] = ()
    review_response_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reason: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def one_settlement(self) -> "CombinationSection":
        explained = self.status == "explained"
        if (
            explained != (self.validation_event_id is not None)
            or explained != bool(self.claims)
            or explained == (self.review_response_event_id is not None)
            or explained == (self.reason is not None)
            or self.claims != tuple(sorted(set(self.claims)))
        ):
            raise ValueError("a request section is explained or settled, not both")
        return self
