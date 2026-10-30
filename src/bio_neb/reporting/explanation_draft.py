"""The explanation draft the model writes and its one check (#448 D1, #460).

A request's member writes why its request ended as it did (one cell of the
readable report's results), the why's references and limitations over its
qualifications; the combination writes the report's conclusion, its references,
what would change the answer, limitations, what was not done and its title. The shape is
flat and described once, here, by its fields;
the instruction and the correction name it by reading it. Code fills every
per-target fact the targets decide (each target's claim word and genotype), so the model
writes none; each genotype its text writes it lists with its target and quote, checked by
membership (#473 owner Q5 a). One check reads the raw arguments and returns the admitted draft or
its faults, covering the shape and the text's structural checks together; the
failure package words the one correction from them (ADR 0043).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field, StrictStr, ValidationError

from bio_neb.contracts import StrictFrozenContract
from bio_neb.failure import OutputFaults
from bio_neb.journey import canonical_json
from bio_neb.reporting.explanation_rows import (
    ExplanationRow,
    GenotypeCitation,
    LimitationDraft,
    References,
    ReportQualification,
    Text,
    genotype_issues,
    qualification_issues,
    reference_issues,
)

# The readable report keeps each text short: a why is one cell of the results
# table, the conclusion tops the report.
WhyText = Annotated[StrictStr, Field(min_length=1, max_length=500)]
ConclusionText = Annotated[StrictStr, Field(min_length=1, max_length=1500)]
# The page's title under its fixed heading.
TitleText = Annotated[StrictStr, Field(min_length=1, max_length=120)]

_LIMITATIONS = (
    "Limitations that together name every qualification in qualifications exactly "
    "once, each saying which part of the answer it limits; empty only when there "
    "are none."
)
_GENOTYPES = (
    "Every genotype your text writes, each with the target whose call it is and "
    "the exact words that write it; empty when the text writes none."
)


class MemberDraft(StrictFrozenContract):
    """One request's explanation as its member writes it."""

    why: WhyText = Field(
        description=(
            "Why this request ended as its targets record, in plain words: what was "
            "looked up, what the source or the VCF answered."
        )
    )
    why_references: References = Field(
        description="Event ids from the evidence index that support the why."
    )
    limitations: tuple[LimitationDraft, ...] = Field(description=_LIMITATIONS)
    genotypes: tuple[GenotypeCitation, ...] = Field(description=_GENOTYPES)


class CombinationDraft(StrictFrozenContract):
    """The report's conclusion over every request, as the model writes it."""

    conclusion: ConclusionText = Field(
        description=(
            "One or two plain sentences answering the researcher's question over "
            "every request."
        )
    )
    conclusion_references: References = Field(
        description="Event ids from the evidence index that support the conclusion."
    )
    would_change: Text = Field(
        description="What new evidence would change the answer, in plain words."
    )
    limitations: tuple[LimitationDraft, ...] = Field(description=_LIMITATIONS)
    not_done: Text = Field(
        description="What was excluded, rejected, removed, deferred, not done or not explained."
    )
    genotypes: tuple[GenotypeCitation, ...] = Field(description=_GENOTYPES)
    title: TitleText = Field(
        description=(
            "A short plain title for the report: what was asked and what was found."
        )
    )


Draft = MemberDraft | CombinationDraft


@dataclass(frozen=True)
class DraftBinding:
    """What one draft is checked against: its step's targets, the qualifications
    it names and the events it may cite."""

    stage: Literal["member", "combination"]
    targets: tuple[ExplanationRow, ...]
    qualifications: tuple[ReportQualification, ...]
    evidence_event_ids: tuple[str, ...]

    @property
    def output_type(self) -> type[Draft]:
        return MemberDraft if self.stage == "member" else CombinationDraft


@dataclass(frozen=True)
class Refused:
    """A refused draft: its faults as the failure package words them (ADR 0043),
    and whether its shape is refused (the framework then refuses it first)."""

    faults: OutputFaults
    shape: bool

    @property
    def codes(self) -> tuple[str, ...]:
        return self.faults.codes


Issue = tuple[str, tuple[str, ...]]


def check(binding: DraftBinding, arguments: Mapping[str, Any]) -> Draft | Refused:
    """The admitted draft of ``arguments``, or one correction for every fault."""
    output_type = binding.output_type
    shape = _shape_issues(output_type, arguments)
    issues = [*shape, *_text_issues(binding, arguments)]
    if not issues:
        return output_type.model_validate_json(canonical_json(arguments), strict=True)
    return Refused(
        faults=OutputFaults(
            codes=tuple(code for code, _details in issues),
            sentences=tuple(detail for _code, details in issues for detail in details),
            fields=tuple(output_type.model_fields),
        ),
        shape=bool(shape),
    )


def _shape_issues(
    output_type: type[Draft], arguments: Mapping[str, Any]
) -> list[Issue]:
    try:
        output_type.model_validate_json(canonical_json(arguments), strict=True)
    except ValidationError as error:
        fields = ", ".join(output_type.model_fields)
        issues = []
        for detail in error.errors(include_url=False, include_input=False):
            place = ".".join(str(part) for part in detail["loc"])
            if detail["type"] == "extra_forbidden":
                text = f"{place} is no field of this output ({fields}): remove it."
            elif detail["type"] == "missing":
                described = (
                    output_type.model_fields[place].description
                    if place in output_type.model_fields
                    else None
                )
                text = f"{place} is missing: add it" + (
                    f" ({described})" if described else "."
                )
                text = text if text.endswith(".") else text + "."
            else:
                text = f"{place}: {detail['msg']}."
            issues.append((f"{place}:{detail['type']}", (text,)))
        return issues
    return []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _numbers(value: Any) -> tuple[int, ...]:
    """The qualification numbers a limitation carries, whatever JSON it sent."""
    if not isinstance(value, list):
        return ()
    return tuple(
        item for item in value if isinstance(item, int) and not isinstance(item, bool)
    )


def _references(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text_issues(binding: DraftBinding, arguments: Mapping[str, Any]) -> list[Issue]:
    """The structural checks the text keeps, over what the arguments carry even
    when their shape is refused; they do not establish entailment."""
    limitations = arguments.get("limitations")
    written = [
        (f"limitations.{index}", item)
        for index, item in enumerate(
            limitations if isinstance(limitations, list) else [], start=1
        )
        if isinstance(item, Mapping)
    ]
    lead = "why" if binding.stage == "member" else "conclusion"
    texts = [
        (lead, _text(arguments.get(lead))),
        *((place, _text(item.get("text"))) for place, item in written),
    ]
    if binding.stage == "combination":
        texts.extend(
            (place, _text(arguments.get(place)))
            for place in ("title", "would_change", "not_done")
        )
    texts = [(place, text) for place, text in texts if text is not None]
    genotypes = arguments.get("genotypes")
    return [
        *(
            qualification_issues(
                binding.qualifications,
                [
                    (place, _numbers(item.get("qualifications")))
                    for place, item in written
                ],
            )
            if isinstance(limitations, list)
            else ()
        ),
        *reference_issues(
            [
                (
                    f"{lead}_references",
                    _references(arguments.get(f"{lead}_references")),
                ),
                *(
                    (f"{place}.references", _references(item.get("references")))
                    for place, item in written
                ),
            ],
            binding.evidence_event_ids,
        ),
        *genotype_issues(
            binding.targets,
            [
                (f"genotypes.{index}", item)
                for index, item in enumerate(
                    genotypes if isinstance(genotypes, list) else [], start=1
                )
            ],
            texts,
        ),
        *_internal_term_issues(binding, texts),
    ]


def _internal_terms(binding: DraftBinding) -> tuple[str, ...]:
    """The exact identifiers and recorded state words of the binding (#460 live
    a1 F1): its event ids, Work item and request ids, every snake_case outcome,
    state or claim word its targets read as, and its qualifications' origin codes
    (#474 Q2 a). An English word stays the model's."""
    from bio_neb.reporting.explanation_members import NOT_EXPLAINED, row_claim

    words = {
        word
        for row in binding.targets
        for word in (row.outcome, row.state, row_claim(row))
        if word is not None and "_" in word
    } | {item.origin for item in binding.qualifications}
    if binding.stage == "combination":
        # A settled request's status the combination reads (live a1).
        words.add(NOT_EXPLAINED)
    return tuple(
        sorted(
            {
                *binding.evidence_event_ids,
                *(row.request_member_id for row in binding.targets),
                *(row.work_item_id for row in binding.targets if row.work_item_id),
                *words,
            },
            key=lambda term: (-len(term), term),
        )
    )


def _internal_term_issues(
    binding: DraftBinding, texts: list[tuple[str, str]]
) -> list[Issue]:
    """Each text that names an exact identifier or recorded state word: ids belong
    in the references, and a state is said in plain words (owner N4 item 8). An
    exact membership of the binding's own tokens, never a judged word (#465)."""
    terms = _internal_terms(binding)
    issues = []
    for place, text in texts:
        found = [
            term
            for term in terms
            # A token's edges are any character an id or label is not made of:
            # label punctuation after it (``e2:``) still names it (live r2 LSP4).
            if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text)
        ]
        if found:
            issues.append(
                (
                    f"explanation_internal_term:{place}",
                    (
                        f"{place} names {', '.join(found)}: write for the researcher in "
                        "plain words, naming each target as the researcher wrote it and "
                        "each source by its name; event ids belong in the references.",
                    ),
                )
            )
    return issues


__all__ = [
    "CombinationDraft",
    "Draft",
    "DraftBinding",
    "MemberDraft",
    "Refused",
    "check",
]
