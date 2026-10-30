"""Every qualification a report can carry, by its typed origin (#474, ADR 0049).

One place defines each origin: its code, its kind (limitation, conflict or gap),
the typed fields it copies from the record that produced it, its audit sentence
and its plain words. The audit sentence is the report basis's own text, which the
CLI and the Investigation's stored evidence read; the plain words are what the
readable report and the report's explainer read. No code reads a
sentence back for its meaning: a qualification's meaning is its origin.

The subject an audit sentence names is passed by its caller: a request's name for
a request origin, a Work item's label for an item origin, and a Work item's id
for a collection, coverage or depth origin. Observation and ploidy origins name
none.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Annotated, ClassVar, Literal, Union, get_args

from pydantic import Field, StrictInt, StrictStr

from bio_neb.contracts import StrictFrozenContract

QualificationKind = Literal["limitation", "conflict", "gap"]

# Which source decided a request's issue: dbSNP's RefSNP lookup or NCBI's
# assembly report (the failure owner's ``ItemIssue.source``).
IssueSource = Literal["refsnp", "assembly_report"]
_SOURCE_WORDS = {
    "refsnp": "the variant lookup could not place it on the Run's reference",
    "assembly_report": (
        "the assembly report could not confirm its place on the Run's reference"
    ),
}


# A position stated without its alleles or an rsID, deferred out of the Run: no
# source is asked about it, and what it does not establish. Its state words and the
# readable report's cell read this one sentence.
DEFERRED_WITHOUT_ALLELES = (
    "deferred out of the Run: it was stated without its alleles or an rsID, so no "
    "source was asked about it and the VCF was not searched; this does not say "
    "whether the sample carries a variant there"
)


class _Origin(StrictFrozenContract):
    kind: ClassVar[QualificationKind]

    def audit(self, subject: str = "") -> str:
        """The basis's own sentence, naming ``subject`` as the module docstring says."""
        raise NotImplementedError

    def words(self) -> str:
        """The plain words, naming no subject, id, state code or stage."""
        raise NotImplementedError


class _StateOrigin(_Origin):
    """An ended request's or Work item's state, with the note its view shows."""

    def note(self) -> str | None:
        return None


# Request states (``collection_view.RequestState``).


class RequestRejected(_StateOrigin):
    """The researcher's rejection, or a stop that names nobody (an effort or
    availability stop), with its reason."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["request_rejected"] = "request_rejected"
    by: StrictStr | None
    reason: StrictStr

    def note(self) -> str:
        return f"by {self.by}: {self.reason}" if self.by else self.reason

    def audit(self, subject: str = "") -> str:
        return _sentence(f"Request {subject} was rejected {self.note()}")

    def words(self) -> str:
        # Only a researcher's own reason is quoted, as theirs.
        if self.by is None:
            return "stopped before it was checked; the VCF was not searched"
        return f'rejected by {self.by}, who wrote: "{self.reason}"'


class RequestIssueRejected(_StateOrigin):
    """A rejection of the issue a request's source check ended with."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["request_issue_rejected"] = "request_issue_rejected"
    by: StrictStr | None
    source: IssueSource
    verdict: StrictStr

    def note(self) -> str:
        return f"by {self.by}: {self.source}: {self.verdict} finding"

    def audit(self, subject: str = "") -> str:
        return _sentence(f"Request {subject} was rejected {self.note()}")

    def words(self) -> str:
        return f"{_SOURCE_WORDS[self.source]}, and {self.by} rejected it"


class RequestSetAside(_StateOrigin):
    """A request one answer removed or deferred whole."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["request_removed", "request_deferred"]
    by: StrictStr

    def note(self) -> str:
        return f"by {self.by}"

    def audit(self, subject: str = "") -> str:
        return _sentence(f"Request {subject} was {self._state} {self.note()}")

    def words(self) -> str:
        return f"{self._state} by {self.by}; the VCF was not searched"

    @property
    def _state(self) -> str:
        return self.code.removeprefix("request_")


class RequestNotChecked(_StateOrigin):
    """A target the Run cannot check by what it states (#367), or a position its
    lookup found stated without alleles or an rsID, deferred out of the Run (owner
    #480 Q1)."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["request_not_checked"] = "request_not_checked"
    reason_code: Literal["not_placed", "position_without_alleles"]
    # The placement step's or the lookup's own words: its audit sentence
    # only.
    reason: StrictStr

    def note(self) -> str:
        return self.reason

    def audit(self, subject: str = "") -> str:
        if self.reason_code == "position_without_alleles":
            return _sentence(
                f"Request {subject} was deferred out of the Run: {self.reason}"
            )
        return _sentence(f"Request {subject} was not checked: {self.reason}")

    def words(self) -> str:
        if self.reason_code == "position_without_alleles":
            return DEFERRED_WITHOUT_ALLELES
        return (
            "no step could place it on the Run's reference as stated, so it was "
            "not checked"
        )


class RequestUnresolved(_StateOrigin):
    """A request whose source check left no Work item: the source and verdict that
    decided it, and the researcher's answer to its issue, if any."""

    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["request_unresolved"] = "request_unresolved"
    source: IssueSource | None = None
    verdict: StrictStr | None = None
    decision: Literal["approved", "unanswered"] | None = None

    def note(self) -> str:
        verdict = (
            "unresolved" if self.source is None else f"{self.source}: {self.verdict}"
        )
        return (
            verdict
            + {
                "approved": " (approved, checked not verified)",
                "unanswered": ", unanswered when the Run ended",
                None: "",
            }[self.decision]
        )

    def audit(self, subject: str = "") -> str:
        return _sentence(f"Request {subject} has no Work item: {self.note()}")

    def words(self) -> str:
        found = (
            "it was not placed on the Run's reference"
            if self.source is None
            else _SOURCE_WORDS[self.source]
        )
        return (
            found
            + {
                "approved": "; you accepted that finding as checked but not verified",
                "unanswered": "; its question was unanswered when the Run ended",
                None: "",
            }[self.decision]
        )


class RequestNotProposed(_StateOrigin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["request_not_proposed"] = "request_not_proposed"

    def note(self) -> str:
        return "the Run ended before the collection was proposed"

    def audit(self, subject: str = "") -> str:
        return _sentence(
            f"Request {subject} was checked but not proposed: {self.note()}"
        )

    def words(self) -> str:
        return "checked, but the Run ended before a VCF search was proposed"


class RequestNeverRun(_StateOrigin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["request_never_run"] = "request_never_run"

    def note(self) -> str:
        return "its check did not finish before Finalization"

    def audit(self, subject: str = "") -> str:
        return _sentence(f"Request {subject} was never checked: {self.note()}")

    def words(self) -> str:
        return "its check did not finish before the Run ended"


# Work item states (``collection_view.ItemState``); an observed item has none.

CollectionItemEnd = Literal[
    "approval_pending", "observation_pending", "not_started", "failed", "stopped"
]


class ItemApprovedNotObserved(_StateOrigin):
    """An approved item never looked up, with its collection status when the
    collection holds it."""

    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["item_approved_not_observed"] = "item_approved_not_observed"
    status: StrictStr | None = None
    reason: StrictStr | None = None

    def note(self) -> str | None:
        if self.status is None or self.reason is None:
            return self.status
        return f"{self.status}: {self.reason}"

    def audit(self, subject: str = "") -> str:
        return f"{subject} was approved but not observed before Finalization."

    def words(self) -> str:
        return "approved, but the Run ended before the VCF was searched here"


class ItemNotApproved(_StateOrigin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["item_not_approved"] = "item_not_approved"

    def audit(self, subject: str = "") -> str:
        return f"{subject} was proposed but not approved before Finalization."

    def words(self) -> str:
        return "proposed, but not approved for a VCF search"


class ItemUnselected(_StateOrigin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["item_unselected_candidate"] = "item_unselected_candidate"

    def audit(self, subject: str = "") -> str:
        return f"{subject} is a candidate the approval did not select."

    def words(self) -> str:
        return "another allele was chosen instead, so this one was not searched"


class ItemSetAside(_StateOrigin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["item_removed", "item_deferred"]
    by: StrictStr

    def note(self) -> str:
        return f"by {self.by}"

    def audit(self, subject: str = "") -> str:
        return _sentence(f"{subject} was {self._state} {self.note()}")

    def words(self) -> str:
        return f"{self._state} by {self.by}; the VCF was not searched here"

    @property
    def _state(self) -> str:
        return self.code.removeprefix("item_")


class ItemFailed(_StateOrigin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["item_failed"] = "item_failed"
    # The failure's own detail: its audit sentence only.
    reason: StrictStr | None = None

    def note(self) -> str | None:
        return self.reason

    def audit(self, subject: str = "") -> str:
        return _sentence(f"{subject} failed: {self.reason}")

    def words(self) -> str:
        return "the VCF search failed here"


class CollectionItem(_Origin):
    """An item the collection holds that ended without its observation."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["collection_item"] = "collection_item"
    status: CollectionItemEnd
    reason: StrictStr | None = None

    def audit(self, subject: str = "") -> str:
        if self.status in {"failed", "stopped"}:
            return f"Collection Work item {subject} is {self.status}: {self.reason}."
        return f"Collection Work item {subject} is {self.status}."

    def words(self) -> str:
        return {
            "failed": "the VCF search failed here",
            "stopped": "the VCF search was stopped here",
            "not_started": "the VCF search here did not start",
            "approval_pending": "its approval was still pending when the Run ended",
            "observation_pending": (
                "its VCF search was still pending when the Run ended"
            ),
        }[self.status]


# Coverage and depth, fed by the Run's own measurement (``run_health``).

# Why an item's coverage was not measured: declared once, typed by the coverage
# fact (``item_evidence``) and by the report origin below.
CoverageNotMeasuredCode = Literal[
    "contig_unknown", "span_beyond_contig", "bam_unavailable", "measurement_failed"
]


class CoverageNotMeasured(_Origin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["coverage_not_measured"] = "coverage_not_measured"
    reason_code: CoverageNotMeasuredCode | None = None
    detail: StrictStr | None = None

    def audit(self, subject: str = "") -> str:
        reason = (
            "" if self.reason_code is None else f": {self.reason_code} ({self.detail})"
        )
        return f"Coverage was not measured for Work item {subject}{reason}."

    def words(self) -> str:
        return "coverage was not measured here"


class DepthBelowMinimum(_Origin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["depth_below_minimum"] = "depth_below_minimum"
    depth: StrictInt = Field(ge=0)
    minimum: StrictInt = Field(ge=0)

    def audit(self, subject: str = "") -> str:
        return (
            f"Work item {subject}: read depth {self.depth} at the least-covered base "
            f"of its span, below the minimum of {self.minimum}."
        )

    def words(self) -> str:
        return (
            f"read depth here was {self.depth} at its lowest, below the minimum "
            f"of {self.minimum}"
        )


# One VCF observation's own qualifications, which the Investigation's evidence
# stores as these audit sentences (``investigation_observation``).


class ObservationScope(_Origin):
    """The claim scope of the observation's outcome, as its result type states it."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["observation_scope"] = "observation_scope"
    outcome: StrictStr

    def audit(self, subject: str = "") -> str:
        return _claim_scopes()[self.outcome]

    def words(self) -> str:
        return "the result holds for this sample's VCF at this target only"


class ObservationNotInferred(_Origin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["observation_not_inferred"] = "observation_not_inferred"

    def audit(self, subject: str = "") -> str:
        return (
            "Coverage, callability, biological absence, function, causality, "
            "and diagnosis are not inferred."
        )

    def words(self) -> str:
        return (
            "coverage, callability, biological absence, function, causality and "
            "diagnosis are not inferred from it"
        )


class ObservationNotInferredDepthMeasured(_Origin):
    """L1 for an item whose read depth the Run measured (#414): depth is stated
    by its own measurement, so it leaves the not-inferred list."""

    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["observation_not_inferred_depth_measured"] = (
        "observation_not_inferred_depth_measured"
    )

    def audit(self, subject: str = "") -> str:
        return (
            "Callability, biological absence, function, causality, and diagnosis "
            "are not inferred; read depth is the Run's measured value."
        )

    def words(self) -> str:
        return (
            "callability, biological absence, function, causality and diagnosis "
            "are not inferred from it; its read depth was measured"
        )


class ObservationProvenance(_Origin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["observation_provenance"] = "observation_provenance"
    run_provenance: Literal["unestablished", "causal_child_or_fork"]

    def audit(self, subject: str = "") -> str:
        if self.run_provenance == "causal_child_or_fork":
            return (
                "This observation belongs to a Run a Fork created; retained "
                "execution is not fresh child work."
            )
        return (
            "Run provenance is unestablished; freshness and development-clone "
            "status are not inferred."
        )

    def words(self) -> str:
        if self.run_provenance == "causal_child_or_fork":
            return "it comes from a Run forked from earlier work, not a fresh analysis"
        return "whether the Run's analysis was freshly made is not established"


class TargetAltNotCarried(_Origin):
    kind: ClassVar[QualificationKind] = "conflict"
    code: Literal["observation_target_alt_not_carried"] = (
        "observation_target_alt_not_carried"
    )

    def audit(self, subject: str = "") -> str:
        return "The sample does not carry the requested target ALT in this VCF."

    def words(self) -> str:
        return "the sample does not carry the target allele here"


class ObservationNotAdmitted(_Origin):
    kind: ClassVar[QualificationKind] = "gap"
    code: Literal["observation_not_admitted"] = "observation_not_admitted"

    def audit(self, subject: str = "") -> str:
        return "Automatic claim admission is unavailable for this outcome."

    def words(self) -> str:
        return "this result is not admitted as a finding automatically"


# The Run's own: diploid calling on an organism the planning model judged
# otherwise, which the researcher chose at Execution Review.

_JUDGED = {"likely_non_diploid": "likely not diploid", "unknown": "of unknown ploidy"}


class DiploidCallingJudgedOtherwise(_Origin):
    kind: ClassVar[QualificationKind] = "limitation"
    code: Literal["diploid_calling_judged_otherwise"] = (
        "diploid_calling_judged_otherwise"
    )
    judgment: Literal["likely_non_diploid", "unknown"]

    def audit(self, subject: str = "") -> str:
        return (
            "Variant calling assumed a diploid sample (ploidy 2); the planning model "
            f"judged this organism {_JUDGED[self.judgment]}, and the researcher "
            "chose diploid calling."
        )

    def words(self) -> str:
        return (
            "the variants were called as diploid, though the planning model judged "
            f"this organism {_JUDGED[self.judgment]}; you chose diploid calling"
        )


StateOrigin = Annotated[
    Union[
        RequestRejected,
        RequestIssueRejected,
        RequestSetAside,
        RequestNotChecked,
        RequestUnresolved,
        RequestNotProposed,
        RequestNeverRun,
        ItemApprovedNotObserved,
        ItemNotApproved,
        ItemUnselected,
        ItemSetAside,
        ItemFailed,
    ],
    Field(discriminator="code"),
]
QualificationOrigin = Annotated[
    Union[
        StateOrigin,
        CollectionItem,
        CoverageNotMeasured,
        DepthBelowMinimum,
        ObservationScope,
        ObservationNotInferred,
        ObservationNotInferredDepthMeasured,
        ObservationProvenance,
        TargetAltNotCarried,
        ObservationNotAdmitted,
        DiploidCallingJudgedOtherwise,
    ],
    Field(discriminator="code"),
]


def origin_codes() -> tuple[str, ...]:
    """Every origin's code, sorted."""
    classes = [
        *get_args(get_args(StateOrigin)[0]),
        *get_args(get_args(QualificationOrigin)[0])[1:],
    ]
    return tuple(
        sorted(
            code
            for cls in classes
            for code in get_args(cls.model_fields["code"].annotation)
        )
    )


def observation_origins(
    outcome: str, run_provenance: str, admitted: bool, *, depth_measured: bool = False
) -> dict[QualificationKind, tuple[_Origin, ...]]:
    """One observation's limitations, conflicts and gaps, by its typed outcome,
    provenance and admission; an item whose depth the Run measured states it so
    (#414, L1). The lookup's own evidence predates any measurement."""
    return {
        "limitation": (
            ObservationScope(outcome=outcome),
            ObservationNotInferredDepthMeasured()
            if depth_measured
            else ObservationNotInferred(),
            ObservationProvenance(run_provenance=run_provenance),
        ),
        "conflict": (
            (TargetAltNotCarried(),) if outcome == "target_allele_not_carried" else ()
        ),
        "gap": () if admitted else (ObservationNotAdmitted(),),
    }


def plain_words(origin: _Origin, targets: Sequence[str] = ()) -> str:
    """The origin's plain words as one sentence, led by the targets it concerns
    as the researcher wrote them, or alone for the Run's own."""
    words = origin.words()
    if targets:
        return f"{', '.join(targets)}: {words}."
    return words[0].upper() + words[1:] + "."


@cache
def _claim_scopes() -> dict[str, str]:
    """Each observation outcome's claim scope, as its result type fixes it."""
    from bio_neb.investigation.vcf_observation import VariantObservationResult

    return {
        cls.model_fields["outcome"].default: cls.model_fields["claim_scope"].default
        for cls in get_args(get_args(VariantObservationResult)[0])
    }


def _sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


__all__ = [
    "CoverageNotMeasuredCode",
    "DEFERRED_WITHOUT_ALLELES",
    "CollectionItem",
    "CoverageNotMeasured",
    "DepthBelowMinimum",
    "DiploidCallingJudgedOtherwise",
    "ItemApprovedNotObserved",
    "ItemFailed",
    "ItemNotApproved",
    "ItemSetAside",
    "ItemUnselected",
    "ObservationNotAdmitted",
    "ObservationNotInferred",
    "ObservationNotInferredDepthMeasured",
    "ObservationProvenance",
    "ObservationScope",
    "QualificationKind",
    "QualificationOrigin",
    "RequestIssueRejected",
    "RequestNeverRun",
    "RequestNotChecked",
    "RequestNotProposed",
    "RequestRejected",
    "RequestSetAside",
    "RequestUnresolved",
    "StateOrigin",
    "TargetAltNotCarried",
    "observation_origins",
    "origin_codes",
    "plain_words",
]
