"""The targets, qualifications and evidence scope one report explanation reads.

Shared by the explanation's facts (``reporting.explanation``) and its per-request
members (``reporting.explanation_members``): one numbered target for every request
and every position searched for it (each Work item) the approved basis lists, in
its recorded state, the report's numbered qualifications in their origins' plain
words (#474), and the events the basis names. The model reads "the request" and
"the position searched", never rows or Work items (#460 live a3 H1, #474 Q3 a).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.contracts import StrictFrozenContract
from bio_neb.identifier_rule import word_aligned_spans
from bio_neb.investigation.qualification_origins import plain_words
from bio_neb.investigation.state_words import item_words, request_words
from bio_neb.paper_reading.report import asserter_words

_TEXT = 2000
_REFERENCES = 20


class ExplanationRow(StrictFrozenContract):
    """One numbered target the approved basis lists: a request, or a position
    searched for it (its Work item), as the Agent is sent it.

    ``subject`` is a request's rsID (or its words) and a position's target; the
    state, outcome and genotype are the basis's own, never the Agent's. The
    position's Work item id links it to its qualifications and evidence; the
    Agent is not sent it.
    """

    number: StrictInt = Field(ge=1)
    kind: Literal["request", "position"]
    request_member_id: StrictStr
    subject: StrictStr
    state: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    work_item_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    outcome: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    genotype: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # What happened to the target, in its typed owners' plain words: the
    # lookup's ending, then its state's origin, else the state's own words.
    words: StrictStr = Field(min_length=1)
    # Who proposed a request when it is not the researcher: the paper whose
    # reading proposed it, so the text never calls it the researcher's.
    asserted_by: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


References = Annotated[
    tuple[StrictStr, ...],
    Field(
        min_length=1,
        max_length=_REFERENCES,
        description="Event ids from the evidence index that support this text.",
    ),
]
Text = Annotated[StrictStr, Field(min_length=1, max_length=_TEXT)]


class ReportQualification(StrictFrozenContract):
    """One limitation, conflict or gap the approved report records, numbered for
    the explanation: its origin's code and plain words, never the report's audit
    sentence (#474 Q3 a), and the targets its ids concern (#448 D2), linked by
    code, never by the model."""

    qualification: StrictInt = Field(ge=1)
    kind: Literal["limitation", "conflict", "gap"]
    origin: StrictStr = Field(min_length=1)
    words: StrictStr = Field(min_length=1)
    targets: tuple[StrictInt, ...] = ()

    @model_validator(mode="after")
    def ordered_targets(self) -> "ReportQualification":
        if self.targets != tuple(sorted(set(self.targets))):
            raise ValueError("qualification targets changed their order")
        return self


class LimitationDraft(StrictFrozenContract):
    qualifications: tuple[Annotated[StrictInt, Field(ge=1)], ...] = Field(
        min_length=1,
        description="The numbers of the report's qualifications this text explains.",
    )
    text: Text
    references: References


def explanation_rows(collection_view) -> tuple[ExplanationRow, ...]:
    """Every request of the basis's view, then each position searched for it,
    numbered in order."""
    rows: list[ExplanationRow] = []
    for snp in () if collection_view is None else collection_view.snps:
        rows.append(
            ExplanationRow(
                number=len(rows) + 1,
                kind="request",
                request_member_id=snp.request_member_id,
                subject=snp.requested_rsid or snp.request_text,
                state=snp.state,
                words=_request_words(snp),
                asserted_by=asserter_words(snp.asserted_by),
            )
        )
        for item in snp.items:
            rows.append(
                ExplanationRow(
                    number=len(rows) + 1,
                    kind="position",
                    request_member_id=snp.request_member_id,
                    subject=item.target,
                    state=item.state,
                    work_item_id=item.work_item_id,
                    outcome=item.outcome,
                    genotype=item.genotype,
                    words=_item_words(item),
                )
            )
    return tuple(rows)


def _request_words(snp) -> str:
    """What the request's check found and what became of it: its lookup ending's
    words, then its state origin's, else its state's."""
    words = []
    if snp.lookup is not None and snp.lookup.origin is not None:
        words.append(snp.lookup.origin.words())
    if snp.state_origin is not None:
        words.append(snp.state_origin.words())
    else:
        words.append(request_words(snp.state).row)
    return "; ".join(words)


def _item_words(item) -> str:
    """A searched position's outcome, else its state origin's words."""
    if item.outcome is None and item.state_origin is not None:
        return item.state_origin.words()
    return item_words(item.outcome or item.state).cell


def explanation_scope(report, publication_event_id: str) -> tuple[str, ...]:
    """The events an explanation may read: those its approved basis names.

    The Grounded report projects that basis exhaustively: its accounting and
    evidence pointers name their source events, and its publication holds it.
    """
    return tuple(
        sorted(
            {
                *(member.value.source_event_id for member in report.accounting),
                *(pointer.source_event_id for pointer in report.evidence_pointers),
                publication_event_id,
            }
        )
    )


def explanation_qualifications(report) -> tuple[ReportQualification, ...]:
    """The report's limitations, conflicts and gaps, numbered in that order,
    each with the targets of its collection view its ids concern."""
    return numbered_qualifications(explanation_rows(report.collection_view), report)


def numbered_report_qualifications(report) -> tuple:
    """Each of the report's qualifications with its number and kind:
    limitations, then conflicts, then gaps."""
    stated = (
        *(("limitation", item) for item in report.limitations),
        *(("conflict", item) for item in report.conflicts),
        *(("gap", item) for item in report.gaps),
    )
    return tuple(
        (number, kind, item) for number, (kind, item) in enumerate(stated, start=1)
    )


def numbered_qualifications(
    rows: Sequence[ExplanationRow], report
) -> tuple[ReportQualification, ...]:
    """Each qualification of ``report`` numbered, in its origin's plain words, with
    the targets it concerns: the positions of its Work items and its requests."""
    return tuple(
        ReportQualification(
            qualification=number,
            kind=kind,
            origin=item.origin.code,
            words=plain_words(item.origin),
            targets=tuple(
                row.number
                for row in rows
                if (row.kind == "position" and row.work_item_id in item.work_item_ids)
                or (
                    row.kind == "request"
                    and row.request_member_id in item.request_member_ids
                )
            ),
        )
        for number, kind, item in numbered_report_qualifications(report)
    )


# The structural checks every explanation text keeps: each returns its
# refusal code naming what it found and what the model reads to repair it,
# or nothing. They do not establish entailment.


def places_naming(
    places: Sequence[tuple[str, Sequence[int]]], number: int
) -> list[str]:
    """Each written place that names the number, once per time it names it."""
    return [place for place, numbers in places for n in numbers if n == number]


def numbered_coverage(
    code: str, expected: Iterable[int], named: Sequence[int]
) -> list[str]:
    """Which expected numbers were named none, several times, or not expected."""
    known = set(expected)
    errors = []
    for suffix, values in (
        ("missing", [n for n in sorted(known) if n not in named]),
        ("repeated", sorted({n for n in named if named.count(n) > 1})),
        ("unknown", sorted({n for n in named if n not in known})),
    ):
        if values:
            errors.append(f"{code}_{suffix}:{','.join(str(n) for n in values)}")
    return errors


def qualification_issues(
    qualifications: Sequence[ReportQualification],
    limitations: Sequence[tuple[str, Sequence[int]]],
) -> list[tuple[str, tuple[str, ...]]]:
    """Every qualification named once by the limitations that own it."""
    kinds = {item.qualification: item.kind for item in qualifications}
    issues = []
    for code in numbered_coverage(
        "explanation_qualifications",
        kinds,
        [number for _place, numbers in limitations for number in numbers],
    ):
        numbers = list(map(int, code.partition(":")[2].split(",")))
        details: tuple[str, ...] = ()
        if code.startswith("explanation_qualifications_missing:"):
            details = tuple(
                f"Qualification {n} ({kinds[n]}) is in no limitation: name it by its "
                "number in a limitation."
                for n in numbers
            )
        elif code.startswith("explanation_qualifications_repeated:"):
            details = tuple(
                f"Qualification {n} is named {len(places)} times, in "
                f"{' and '.join(places)}: keep it once, in one limitation."
                for n in numbers
                for places in [places_naming(limitations, n)]
            )
        elif code.startswith("explanation_qualifications_unknown:"):
            details = tuple(
                f"Qualification {n} is no qualification of this report"
                + (
                    f" (1 to {len(kinds)})"
                    if tuple(kinds) == tuple(range(1, len(kinds) + 1))
                    else " this explanation names"
                    + f" ({', '.join(str(k) for k in kinds) or 'none'})"
                )
                + ": remove it."
                for n in numbers
            )
        issues.append((code, details))
    return issues


def reference_issues(
    referenced: Sequence[tuple[str, Sequence[str]]], allowed: Iterable[str]
) -> list[tuple[str, tuple[str, ...]]]:
    """Every cited event in the evidence index; a shortened id is named whole."""
    allowed = frozenset(allowed)
    outside = sorted(
        {ref for _place, refs in referenced for ref in refs if ref not in allowed}
    )
    if not outside:
        return []
    details = []
    for place, refs in referenced:
        for ref in refs:
            if ref in allowed:
                continue
            whole = [item for item in sorted(allowed) if item.startswith(ref)]
            details.append(
                f"{place} cites {ref}, which is not in the evidence index; "
                + (
                    f"cite the whole id {whole[0]}."
                    if len(whole) == 1
                    else "cite only ids the evidence index lists."
                )
            )
    return [
        ("explanation_reference_outside_scope:" + ",".join(outside), tuple(details))
    ]


class GenotypeCitation(StrictFrozenContract):
    """One genotype the model's text writes: the target whose call it is, the
    genotype as that target carries it and the text's own words that write it
    (#473 owner Q5 a). Code checks membership only (ADR 0049)."""

    target: StrictInt = Field(
        ge=1, description="The number of the target whose call the text writes."
    )
    genotype: StrictStr = Field(
        min_length=1,
        max_length=50,
        description="The genotype as that target carries it.",
    )
    quote: StrictStr = Field(
        min_length=1,
        max_length=300,
        description="Your text's exact words that write the genotype.",
    )


def genotype_issues(
    targets: Sequence[ExplanationRow],
    cited: Sequence[tuple[str, Any]],
    texts: Sequence[tuple[str, str]],
) -> list[tuple[str, tuple[str, ...]]]:
    """Each listed genotype that is not a bound target's call, or whose quote is
    not the text's words writing it: a membership check over what the model
    listed, never a reading of the prose (ADR 0049, #473 owner Q5 a)."""
    calls = {row.number: row.genotype for row in targets}
    issues: list[tuple[str, tuple[str, ...]]] = []
    for place, item in cited:
        if not isinstance(item, Mapping):
            continue
        number, genotype, quote = (
            item.get("target"),
            item.get("genotype"),
            item.get("quote"),
        )
        if not (
            isinstance(number, int)
            and not isinstance(number, bool)
            and isinstance(genotype, str)
            and isinstance(quote, str)
        ):
            continue
        if number not in calls:
            issues.append(
                (
                    f"explanation_genotype_target_unknown:{place}",
                    (
                        f"{place} names target {number}, which this explanation "
                        "does not list; name a listed target.",
                    ),
                )
            )
        elif calls[number] != genotype:
            carried = calls[number]
            issues.append(
                (
                    f"explanation_genotype_unsupported:{place}",
                    (
                        f"{place} gives target {number} the genotype {genotype}; "
                        + (
                            f"that target carries {carried}: write it as it carries it."
                            if carried is not None
                            else "that target carries no genotype: write none for it."
                        ),
                    ),
                )
            )
        if not any(word_aligned_spans(text, quote) for _place, text in texts):
            issues.append(
                (
                    f"explanation_genotype_quote_outside_text:{place}",
                    (
                        f"{place}'s quote is not words of your text; quote the exact "
                        "words that write the genotype.",
                    ),
                )
            )
        elif not word_aligned_spans(quote, genotype):
            issues.append(
                (
                    f"explanation_genotype_not_in_quote:{place}",
                    (
                        f"{place}'s quote does not write {genotype}; quote the words "
                        "that write it.",
                    ),
                )
            )
    return issues
