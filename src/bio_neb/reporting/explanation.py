"""An optional Agent explanation of one approved Grounded report (ADR 0038, #109).

The Grounded report is the deterministic projection of the approved Report
basis. A researcher may ask for a separate, Agent-written explanation of it:
``ReportExplanationRequested`` names the published report. Each request of the
report is explained by its own member Attempt, bound to that request's numbered
rows and the evidence scope it may read (``ReportExplanationMemberBound``), and
one combination Attempt answers over the admitted members
(``ReportExplanationCombinationBound``, #408). The rows are every request and
every Work item the basis's collection view lists, each in its recorded state;
the scope is the basis's own accounting and evidence-pointer events plus the
report's publication. Typed outputs are checked deterministically and published
as one immutable artifact with its own identity (``ReportExplanationPublished``),
or an Attempt fails (``ReportExplanationFailed``) or stops at a shared Review.
The one-Attempt ``/1`` explanation and its ``ReportExplanationBound`` are retired
facts a Journey cannot carry (#441, owner Q6).

An explanation grants no scientific authority and changes no report. The checks
are structural: every row accounted once, rows explained together share one
state, every reference inside the scope, no genotype a row does not carry and no
coverage or read-depth claim. They do not establish that each sentence follows
from its evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.agentic_journey.continuation_policy import ContinuationPolicy
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.agentic_journey.structured_output import (
    project_structured_output_lineages,
)
from bio_neb.content import ContentId
from bio_neb.contracts import FrozenContract, StrictFrozenContract
from bio_neb.journey import (
    ArtifactReference,
    Journey,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.recording import review_head
from bio_neb.reporting.explanation_draft import (
    CombinationDraft,
    DraftBinding,
    MemberDraft,
    Refused,
    check,
)
from bio_neb.reporting.explanation_members import (
    COMBINATION_OUTPUT_TOOL,
    NOT_EXPLAINED,
    CombinationSection,
    RowClaim,
    combination_qualifications,
    member_qualifications,
    member_rows,
    member_scope,
    row_claims,
)
from bio_neb.reporting.explanation_rows import (
    ExplanationRow,
    GenotypeCitation,
    ReportQualification,
    explanation_qualifications,
    explanation_rows,
    explanation_scope,
)

if TYPE_CHECKING:
    from bio_neb.application.planning_read import PrefixSnapshot

EXPLANATION_ROLE = "report_explainer"
EXPLANATION_OUTPUT_TOOL = "record_explanation"
EXPLANATION_MEDIA_TYPE = "application/vnd.bio-neb.report-explanation+json"
_REPORT = r"^grounded-report:[0-9a-f]{64}$"
_REPORT_BASIS = r"^report-basis:[0-9a-f]{64}$"
_EXPLANATION = r"^report-explanation:[0-9a-f]{64}$"


class ExplainedText(StrictFrozenContract):
    text: StrictStr = Field(min_length=1)
    references: tuple[StrictStr, ...] = Field(min_length=1)


class ExplainedLimitation(StrictFrozenContract):
    qualifications: tuple[StrictInt, ...] = Field(min_length=1)
    text: StrictStr = Field(min_length=1)
    references: tuple[StrictStr, ...] = Field(min_length=1)


class _IdentifiedExplanation(FrozenContract):
    """An explanation artifact whose identity is the digest of its own content."""

    def canonical_bytes(self) -> bytes:
        return (
            canonical_json(self.model_dump(mode="json", exclude_computed_fields=True))
            + "\n"
        ).encode("utf-8")

    @model_validator(mode="after")
    def exact_identity(self):
        material = self.model_dump(
            mode="json", exclude={"explanation_id"}, exclude_computed_fields=True
        )
        if self.explanation_id != "report-explanation:" + canonical_json_digest(
            material
        ):
            raise ValueError("report explanation identity changed")
        return self


class ExplainedRequest(StrictFrozenContract):
    """One request as its admitted member explained it (#408)."""

    status: Literal["explained"] = "explained"
    request_member_id: StrictStr
    # The request's own target number.
    request: StrictInt
    subject: StrictStr
    attempt_id: StrictStr
    validation_event_id: StrictStr
    instruction_identity: StrictStr
    # The member's own targets and qualifications, by their report numbers.
    targets: tuple[StrictInt, ...] = Field(min_length=1)
    qualifications: tuple[StrictInt, ...]
    claims: tuple[RowClaim, ...] = Field(min_length=1)
    why: ExplainedText
    limitations: tuple[ExplainedLimitation, ...]
    # Each genotype the member's text writes, by target and quote.
    genotypes: tuple[GenotypeCitation, ...]


class UnexplainedRequest(StrictFrozenContract):
    """One request whose member its decision stopped: not explained, and why."""

    status: Literal["not_explained"] = "not_explained"
    request_member_id: StrictStr
    request: StrictInt
    subject: StrictStr
    attempt_id: StrictStr
    review_response_event_id: StrictStr
    reason: StrictStr = Field(min_length=1)


RequestSection = Annotated[
    ExplainedRequest | UnexplainedRequest, Field(discriminator="status")
]


class RequestClaims(StrictFrozenContract):
    request: StrictInt
    claims: tuple[StrictStr, ...] = Field(min_length=1)


class ReportExplanationByRequest(_IdentifiedExplanation):
    """One immutable explanation of one exact published report, each request by
    its own member and one combined answer over them (#408); never a finding."""

    schema_version: Literal["bio-neb.report-explanation/5"] = (
        "bio-neb.report-explanation/5"
    )
    explanation_id: StrictStr = Field(pattern=_EXPLANATION)
    journey_id: StrictStr
    request_event_id: StrictStr
    # The combination's Attempt, its output validation and its instructions.
    attempt_id: StrictStr
    report_digest: StrictStr = Field(pattern=_REPORT)
    report_basis_digest: StrictStr = Field(pattern=_REPORT_BASIS)
    publication_event_id: StrictStr
    supersedes_explanation_id: StrictStr | None = Field(
        default=None, pattern=_EXPLANATION, exclude_if=lambda value: value is None
    )
    agent_written: Literal[True] = True
    scientific_authority: Literal["none"] = "none"
    instruction_identity: StrictStr
    validation_event_id: StrictStr
    targets: tuple[ExplanationRow, ...]
    # The report's own qualifications in their origins' plain words, retained
    # beside the Agent's text.
    qualifications: tuple[ReportQualification, ...]
    requests: tuple[RequestSection, ...]
    # The readable report's title, the combination's; an explanation
    # written before it has none and keeps its bytes.
    title: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        exclude_if=lambda value: value is None,
    )
    conclusion: ExplainedText
    would_change: StrictStr = Field(min_length=1)
    request_claims: tuple[RequestClaims, ...]
    # The qualifications no request's member was given, which the combination names.
    combination_qualifications: tuple[StrictInt, ...]
    limitations: tuple[ExplainedLimitation, ...]
    not_done: StrictStr = Field(min_length=1)
    # Each genotype the combination's text writes, by target and quote.
    genotypes: tuple[GenotypeCitation, ...]


# Retired explanation shapes, each refused by name: the one-Attempt one, the
# answer-per-request one, the one that kept the report's audit sentences and
# rows and the one whose genotypes a prose pattern read.
RETIRED_EXPLANATION_SCHEMAS = (
    "bio-neb.report-explanation/1",
    "bio-neb.report-explanation/2",
    "bio-neb.report-explanation/3",
    "bio-neb.report-explanation/4",
)


class ReportExplanationRequested(FrozenContract):
    """The researcher's request to explain one published report."""

    event_type: Literal["ReportExplanationRequested"] = "ReportExplanationRequested"
    schema_version: Literal["bio-neb.report-explanation-request/1"] = (
        "bio-neb.report-explanation-request/1"
    )
    report_digest: StrictStr = Field(pattern=_REPORT)
    report_basis_digest: StrictStr = Field(pattern=_REPORT_BASIS)
    publication_event_id: StrictStr
    # A rewrite names the published explanation it supersedes; the prior stays.
    supersedes_explanation_id: StrictStr | None = Field(
        default=None, pattern=_EXPLANATION, exclude_if=lambda value: value is None
    )
    requested_by: StrictStr = Field(min_length=1)


class _MemberScoped(FrozenContract):
    """What a member or combination binding shares: its request and policy."""

    attempt_id: StrictStr
    request_event_id: StrictStr
    continued_attempt_id: StrictStr | None = None
    report_digest: StrictStr = Field(pattern=_REPORT)
    report_basis_digest: StrictStr = Field(pattern=_REPORT_BASIS)
    publication_event_id: StrictStr
    # The only events the Attempt's Journey search may read (its scope fact).
    evidence_event_ids: tuple[StrictStr, ...] = Field(min_length=1)
    runtime_selections: RuntimeSelections
    continuation_policy: ContinuationPolicy
    continuation_policy_id: StrictStr
    researcher_instruction: StrictStr | None = None

    @model_validator(mode="after")
    def ordered_scope(self):
        if self.evidence_event_ids != tuple(sorted(set(self.evidence_event_ids))):
            raise ValueError("explanation evidence scope changed its order")
        return self


class ReportExplanationMemberBound(_MemberScoped):
    """One request's member Attempt: its targets, its qualifications and its scope (#408)."""

    event_type: Literal["ReportExplanationMemberBound"] = "ReportExplanationMemberBound"
    schema_version: Literal["bio-neb.report-explanation-member-bound/2"] = (
        "bio-neb.report-explanation-member-bound/2"
    )
    request_member_id: StrictStr
    # The request's own target, then each position searched for it, by their
    # report numbers.
    targets: tuple[ExplanationRow, ...] = Field(min_length=1)
    # The report qualifications whose ids concern this request.
    qualifications: tuple[ReportQualification, ...]

    @model_validator(mode="after")
    def one_request(self) -> "ReportExplanationMemberBound":
        numbers = [row.number for row in self.targets]
        if (
            self.targets[0].kind != "request"
            or any(row.kind != "position" for row in self.targets[1:])
            or any(
                row.request_member_id != self.request_member_id for row in self.targets
            )
            or numbers != sorted(set(numbers))
        ):
            raise ValueError("an explanation member binds one request's own targets")
        numbers = [item.qualification for item in self.qualifications]
        if numbers != sorted(set(numbers)):
            raise ValueError("explanation qualifications changed their order")
        return self


class ReportExplanationMemberAdmitted(FrozenContract):
    """One member's admitted section: its claim per target, why and limitations."""

    event_type: Literal["ReportExplanationMemberAdmitted"] = (
        "ReportExplanationMemberAdmitted"
    )
    attempt_id: StrictStr
    validation_event_id: StrictStr
    request_member_id: StrictStr
    claims: tuple[RowClaim, ...] = Field(min_length=1)
    why: ExplainedText
    limitations: tuple[ExplainedLimitation, ...]
    genotypes: tuple[GenotypeCitation, ...]


class ReportExplanationCombinationBound(_MemberScoped):
    """The combination Attempt: every request's settled section and the report."""

    event_type: Literal["ReportExplanationCombinationBound"] = (
        "ReportExplanationCombinationBound"
    )
    schema_version: Literal["bio-neb.report-explanation-combination-bound/2"] = (
        "bio-neb.report-explanation-combination-bound/2"
    )
    targets: tuple[ExplanationRow, ...]
    qualifications: tuple[ReportQualification, ...]
    combination_qualifications: tuple[StrictInt, ...]
    sections: tuple[CombinationSection, ...]

    @model_validator(mode="after")
    def numbered_report(self) -> "ReportExplanationCombinationBound":
        if tuple(row.number for row in self.targets) != tuple(
            range(1, len(self.targets) + 1)
        ):
            raise ValueError("explanation targets changed their numbering")
        if tuple(item.qualification for item in self.qualifications) != tuple(
            range(1, len(self.qualifications) + 1)
        ):
            raise ValueError("explanation qualifications changed their numbering")
        return self


class ReportExplanationPublished(FrozenContract):
    event_type: Literal["ReportExplanationPublished"] = "ReportExplanationPublished"
    attempt_id: StrictStr
    validation_event_id: StrictStr
    explanation: ReportExplanationByRequest


class ReportExplanationFailed(FrozenContract):
    event_type: Literal["ReportExplanationFailed"] = "ReportExplanationFailed"
    schema_version: Literal["bio-neb.report-explanation-failed/1"] = (
        "bio-neb.report-explanation-failed/1"
    )
    attempt_id: StrictStr
    code: Literal["model_contact_failed", "agent_run_failed"]
    source_event_id: StrictStr
    reason: StrictStr | None = None


READABLE_REPORT_FILE = "report.md"


class ReadableReportWritten(StrictFrozenContract):
    """The readable page of one published explanation, written into the Run root
    as ``report.md`` with its bytes' identity (owner, #476 G5/C13): the Run's own
    final report, which a researcher reads without a command or digest."""

    event_type: Literal["ReadableReportWritten"] = "ReadableReportWritten"
    report_digest: StrictStr
    explanation_id: StrictStr
    file_name: Literal["report.md"] = READABLE_REPORT_FILE
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: StrictInt = Field(gt=0)


_FACTS = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        ReportExplanationRequested,
        ReportExplanationMemberBound,
        ReportExplanationMemberAdmitted,
        ReportExplanationCombinationBound,
        ReportExplanationPublished,
        ReportExplanationFailed,
        ReadableReportWritten,
    )
}
EXPLANATION_EVENT_TYPES = frozenset(_FACTS)


def validate_explanation_fact(event: Mapping[str, Any]) -> FrozenContract | None:
    """The explanation fact an event records, or None for any other event.

    A retired one-Attempt fact is refused by name before it is read (#441 Q6).
    """
    from bio_neb.application.planning_read import RETIRED_JOURNEY_FACTS

    if event.get("event_type") in RETIRED_JOURNEY_FACTS:
        raise ValueError(f"retired_journey_fact:{event['event_type']}")
    owner = _FACTS.get(event.get("event_type"))
    if owner is None:
        return None
    # Only an object can name a schema; any other shape meets strict validation.
    explanation = event["payload"].get("explanation")
    if (
        owner is ReportExplanationPublished
        and isinstance(explanation, Mapping)
        and explanation.get("schema_version") in RETIRED_EXPLANATION_SCHEMAS
    ):
        raise ValueError(f"retired_journey_fact:{explanation['schema_version']}")
    return owner.model_validate_json(
        canonical_json({"event_type": event["event_type"], **event.get("payload", {})}),
        strict=True,
    )


def validate_explanation_artifact(
    event: Mapping[str, Any], fact: ReportExplanationPublished
) -> ArtifactReference:
    """The published explanation's one retained artifact is its own bytes."""
    references = tuple(event.get("artifact_references", ()))
    if len(references) != 1:
        raise ValueError("report explanation requires one retained artifact")
    reference = ArtifactReference.model_validate(references[0])
    content = fact.explanation.canonical_bytes()
    if (
        reference.artifact_id != str(ContentId.of(content))
        or reference.byte_count != len(content)
        or reference.media_type != EXPLANATION_MEDIA_TYPE
        or reference.journey_id != event.get("journey_id")
        or reference.source_event_id != event.get("event_id")
        or fact.explanation.journey_id != event.get("journey_id")
        or fact.explanation.attempt_id != fact.attempt_id
        or fact.explanation.validation_event_id != fact.validation_event_id
    ):
        raise ValueError("report explanation artifact binding changed")
    return reference


_BINDINGS = (ReportExplanationMemberBound, ReportExplanationCombinationBound)
ExplanationBinding = ReportExplanationMemberBound | ReportExplanationCombinationBound


def bound_member(bound: ExplanationBinding) -> str | None:
    """The request member a binding's Attempt explains: a member's own, and none
    for the combination, which answers the whole request."""
    if isinstance(bound, ReportExplanationMemberBound):
        return bound.request_member_id
    return None


def bound_fact(events, attempt_id: str) -> ExplanationBinding:
    """The one binding an explanation Attempt recorded: a request member's or the
    combination's (#408)."""
    bound = [
        fact
        for event in events
        if isinstance(fact := validate_explanation_fact(event), _BINDINGS)
        and fact.attempt_id == attempt_id
    ]
    if len(bound) != 1:
        raise ValueError("explanation Attempt lacks its exact request")
    return bound[0]


class _RequestPrefixes:
    """A timeline whose prefix before each request is one snapshot.

    Only those prefixes are interpreted, as one report read each did before;
    every Attempt of a request then reads that request's derived report. The
    Journey ``owner`` of the timeline, when given, holds each prefix for every
    reader across its appends.
    """

    def __init__(self, timeline: list[dict], owner: Journey | None = None):
        self.timeline = timeline
        self._owner = owner
        self._prefixes: dict[str, Any] = {}

    def before(self, event_id: str):
        from bio_neb.application.planning_read import PrefixSnapshot

        if event_id not in self._prefixes:
            index = [event["event_id"] for event in self.timeline].index(event_id)
            self._prefixes[event_id] = (
                PrefixSnapshot.held(
                    self._owner, self.timeline, self.timeline[index - 1]["event_id"]
                )
                if self._owner is not None and index > 0
                else PrefixSnapshot(self.timeline[:index])
            )
        return self._prefixes[event_id]


class _ReportsBefore:
    """The reports one prefix publishes, each projected once by digest."""

    def __init__(self, prefix):
        self._prefix = prefix
        self._reports: dict[str, Any] = {}

    def report(self, report_digest: str):
        from bio_neb.reporting.grounded_report import project_grounded_report

        if report_digest not in self._reports:
            self._reports[report_digest] = project_grounded_report(
                self._prefix.timeline,
                report_digest=report_digest,
                snapshot=self._prefix,
            )
        return self._reports[report_digest]


def report_to_explain(owner: Journey, events: list[dict], report_digest: str):
    """The report ``report_digest`` names in ``owner``'s merged evidence
    ``events``: the prefix a request recorded now is read before, so the
    request's later readers share this one derivation."""
    from bio_neb.application.planning_read import PrefixSnapshot

    return (
        PrefixSnapshot.held(owner, events, events[-1]["event_id"])
        .derived(_ReportsBefore)
        .report(report_digest)
    )


def _report_before(snapshot, request_event_id: str, report_digest: str):
    """The report a request names, read from the snapshot's prefix before it.

    The prefix and its report are derived once for every reader of the
    snapshot, such as each Attempt of one request: every member's and the
    combination's reopen read the one projection (ADR 0032, #408).
    """
    return (
        snapshot.before(request_event_id).derived(_ReportsBefore).report(report_digest)
    )


def _bound_request(snapshot, bound: ExplanationBinding):
    """The retained request a binding names and the report it names."""
    timeline = snapshot.timeline
    ids = [event["event_id"] for event in timeline]
    if bound.request_event_id not in ids:
        raise ValueError("explanation Attempt names no retained request")
    index = ids.index(bound.request_event_id)
    request = validate_explanation_fact(timeline[index])
    if not isinstance(request, ReportExplanationRequested):
        raise ValueError("explanation Attempt names no retained request")
    report = _report_before(
        snapshot, bound.request_event_id, request.report_digest
    ).report
    if report is None:
        raise ValueError("explanation request names no published report")
    return index, request, report


def member_binding(
    events: Sequence[Mapping[str, Any]],
    report,
    publication_event_id: str,
    member_id: str,
) -> tuple[
    tuple[ExplanationRow, ...], tuple[ReportQualification, ...], tuple[str, ...]
]:
    """One request member's rows, qualifications and evidence scope, as its
    request's report and the events before the request give them."""
    every = explanation_rows(report.collection_view)
    rows = dict(member_rows(every))
    if member_id not in rows:
        raise ValueError("explanation member names no request of its report")
    own = rows[member_id]
    return (
        own,
        member_qualifications(every, member_id, explanation_qualifications(report)),
        member_scope(
            events,
            explanation_scope(report, publication_event_id),
            own,
            publication_event_id,
        ),
    )


def _check_binding(snapshot, bound: ExplanationBinding, bound_event_id: str) -> None:
    """A binding is exactly what its request's approved report gives; a
    combination's sections are its members' settlements before it bound."""
    index, request, report = _bound_request(snapshot, bound)
    if (
        bound.report_digest != request.report_digest
        or bound.report_basis_digest != request.report_basis_digest
        or bound.publication_event_id != request.publication_event_id
        or report.report_digest != request.report_digest
        or report.report_basis_digest != request.report_basis_digest
    ):
        raise ValueError("explanation Attempt changed its approved report binding")
    timeline = snapshot.timeline
    if isinstance(bound, ReportExplanationMemberBound):
        given = member_binding(
            timeline[:index],
            report,
            request.publication_event_id,
            bound.request_member_id,
        )
        if (bound.targets, bound.qualifications, bound.evidence_event_ids) != given:
            raise ValueError("explanation member changed its approved report binding")
        return
    rows = explanation_rows(report.collection_view)
    qualifications = explanation_qualifications(report)
    position = [event["event_id"] for event in timeline].index(bound_event_id)
    if (
        bound.targets != rows
        or bound.qualifications != qualifications
        or bound.combination_qualifications
        != tuple(
            item.qualification
            for item in combination_qualifications(rows, qualifications)
        )
        or bound.evidence_event_ids
        != explanation_scope(report, request.publication_event_id)
        or bound.sections
        != settled_sections(timeline[:position], bound.request_event_id, rows)
    ):
        raise ValueError("explanation combination changed its approved report binding")


def _check_hold_scope(started: Mapping[str, Any], bound: ExplanationBinding) -> None:
    """The scope the shared Reviews read from the Attempt's start is the work its
    binding answers: a member's own request member, or the whole request for the
    combination, held only by stops since that request (#408)."""
    expected = (bound_member(bound), bound.request_event_id)
    if (started.get("member_id"), started.get("holds_from_event_id")) != expected:
        raise ValueError("explanation Attempt changed its hold scope")


def _searches_in_scope(selected, bound_event_id: str) -> None:
    if any(
        event["event_type"] == "JourneySearchRead"
        and event["payload"]["query"].get("scope_event_id") != bound_event_id
        for event in selected
    ):
        raise ValueError("explanation search read outside its bound scope")


def _accepted_output(timeline, attempt_id: str, bound, validation_event_id: str, tool):
    lineages = project_structured_output_lineages(
        timeline,
        attempt_id=attempt_id,
        agent_role=EXPLANATION_ROLE,
        tool_name=tool,
        max_retries=bound.continuation_policy.most_failed_turns,
    )
    accepted = [
        attempt
        for lineage in lineages
        for attempt in lineage.validation_attempts
        if attempt.status == "valid"
    ]
    if len(accepted) != 1 or accepted[0].validation_event_id != validation_event_id:
        raise ValueError("explanation changed its accepted output")
    return accepted[0].output


def derive_member_admitted(
    bound: ReportExplanationMemberBound,
    draft: MemberDraft,
    *,
    validation_event_id: str,
) -> ReportExplanationMemberAdmitted:
    """The admitted section of one member's accepted draft: each target's claim
    filled from its recorded state, beside the member's why and limitations."""
    return ReportExplanationMemberAdmitted(
        attempt_id=bound.attempt_id,
        validation_event_id=validation_event_id,
        request_member_id=bound.request_member_id,
        claims=row_claims(bound.targets),
        why=ExplainedText(text=draft.why, references=draft.why_references),
        limitations=_limitations(draft),
        genotypes=draft.genotypes,
    )


def _limitations(draft: MemberDraft | CombinationDraft):
    return tuple(
        ExplainedLimitation(
            qualifications=item.qualifications,
            text=item.text,
            references=item.references,
        )
        for item in draft.limitations
    )


def draft_binding(bound: ExplanationBinding) -> DraftBinding:
    """What one step's draft is checked against: a member's targets and
    qualifications, or the combination's targets and its own qualifications."""
    if isinstance(bound, ReportExplanationMemberBound):
        return DraftBinding(
            stage="member",
            targets=bound.targets,
            qualifications=bound.qualifications,
            evidence_event_ids=bound.evidence_event_ids,
        )
    owned = set(bound.combination_qualifications)
    return DraftBinding(
        stage="combination",
        targets=bound.targets,
        qualifications=tuple(
            item for item in bound.qualifications if item.qualification in owned
        ),
        evidence_event_ids=bound.evidence_event_ids,
    )


def admitted_members(timeline) -> dict[str, tuple]:
    """Each admitted member Attempt: its binding, its section and its
    instructions, as the Journey retains them."""
    starts = {
        event["payload"]["attempt_id"]: event["payload"].get("instruction_identity")
        for event in timeline
        if event["event_type"] == "AttemptStarted"
    }
    bound = {
        fact.attempt_id: fact
        for event in timeline
        if isinstance(
            fact := validate_explanation_fact(event), ReportExplanationMemberBound
        )
    }
    return {
        fact.attempt_id: (bound[fact.attempt_id], fact, starts.get(fact.attempt_id))
        for event in timeline
        if isinstance(
            fact := validate_explanation_fact(event), ReportExplanationMemberAdmitted
        )
        and fact.attempt_id in bound
    }


def derive_explanation_by_request(
    bound: ReportExplanationCombinationBound,
    members: Mapping[str, tuple],
    draft: CombinationDraft,
    *,
    journey_id: str,
    supersedes_explanation_id: str | None,
    instruction_identity: str,
    validation_event_id: str,
) -> ReportExplanationByRequest:
    """The immutable explanation of the combination's accepted draft over the
    sections it was bound to."""
    requests: list[ExplainedRequest | UnexplainedRequest] = []
    for section in bound.sections:
        if section.status == "not_explained":
            requests.append(
                UnexplainedRequest(
                    request_member_id=section.request_member_id,
                    request=section.request,
                    subject=section.subject,
                    attempt_id=section.attempt_id,
                    review_response_event_id=section.review_response_event_id,
                    reason=section.reason,
                )
            )
            continue
        member_bound, admitted, member_instructions = members[section.attempt_id]
        requests.append(
            ExplainedRequest(
                request_member_id=section.request_member_id,
                request=section.request,
                subject=section.subject,
                attempt_id=section.attempt_id,
                validation_event_id=admitted.validation_event_id,
                instruction_identity=member_instructions,
                targets=tuple(row.number for row in member_bound.targets),
                qualifications=tuple(
                    item.qualification for item in member_bound.qualifications
                ),
                claims=admitted.claims,
                why=admitted.why,
                limitations=admitted.limitations,
                genotypes=admitted.genotypes,
            )
        )
    payload: dict[str, Any] = {
        "schema_version": "bio-neb.report-explanation/5",
        "journey_id": journey_id,
        "request_event_id": bound.request_event_id,
        "attempt_id": bound.attempt_id,
        "report_digest": bound.report_digest,
        "report_basis_digest": bound.report_basis_digest,
        "publication_event_id": bound.publication_event_id,
        "supersedes_explanation_id": supersedes_explanation_id,
        "agent_written": True,
        "scientific_authority": "none",
        "instruction_identity": instruction_identity,
        "validation_event_id": validation_event_id,
        "targets": bound.targets,
        "qualifications": bound.qualifications,
        "requests": tuple(requests),
        "title": draft.title,
        "conclusion": ExplainedText(
            text=draft.conclusion, references=draft.conclusion_references
        ),
        "would_change": draft.would_change,
        # Each request's claim words as its member's section holds them.
        "request_claims": tuple(
            RequestClaims(
                request=section.request, claims=section.claims or (NOT_EXPLAINED,)
            )
            for section in bound.sections
        ),
        "combination_qualifications": bound.combination_qualifications,
        "limitations": _limitations(draft),
        "not_done": draft.not_done,
        "genotypes": draft.genotypes,
    }
    material = ReportExplanationByRequest.model_construct(
        explanation_id="report-explanation:" + "0" * 64, **payload
    ).model_dump(mode="json", exclude={"explanation_id"}, exclude_computed_fields=True)
    return ReportExplanationByRequest.model_validate(
        {
            "explanation_id": "report-explanation:" + canonical_json_digest(material),
            **payload,
        }
    )


def validate_explanation_attempt_facts(snapshot, selected) -> None:
    """An explanation Attempt's facts are its accepted output over its own
    binding: a member's section, or the combination's publication over the
    sections it was bound to. Every search the Attempt made
    read its own bound scope.
    """
    timeline = snapshot.timeline
    (started,) = [
        event for event in selected if event["event_type"] == "AttemptStarted"
    ]
    attempt_id = started["payload"]["attempt_id"]
    bound_events = [
        event
        for event in selected
        if event["event_type"]
        in {binding.model_fields["event_type"].default for binding in _BINDINGS}
    ]
    if len(bound_events) != 1:
        raise ValueError("explanation Attempt lacks its exact request")
    (bound_event,) = bound_events
    bound = bound_fact(selected, attempt_id)
    _check_binding(snapshot, bound, bound_event["event_id"])
    _check_hold_scope(started["payload"], bound)
    _searches_in_scope(selected, bound_event["event_id"])
    facts = [
        fact
        for event in selected
        if isinstance(
            fact := validate_explanation_fact(event),
            ReportExplanationPublished | ReportExplanationMemberAdmitted,
        )
    ]
    if len(facts) > 1:
        raise ValueError("explanation Attempt published more than once")
    for fact in facts:
        tool = (
            EXPLANATION_OUTPUT_TOOL
            if isinstance(bound, ReportExplanationMemberBound)
            else COMBINATION_OUTPUT_TOOL
        )
        draft = check(
            draft_binding(bound),
            _accepted_output(
                timeline, attempt_id, bound, fact.validation_event_id, tool
            ),
        )
        if isinstance(draft, Refused):
            raise ValueError("explanation changed its accepted output")
        if isinstance(bound, ReportExplanationMemberBound):
            if not isinstance(fact, ReportExplanationMemberAdmitted):
                raise ValueError("explanation member published a whole explanation")
            if fact != derive_member_admitted(
                bound, draft, validation_event_id=fact.validation_event_id
            ):
                raise ValueError("explanation changed its accepted output")
            continue
        if not isinstance(fact, ReportExplanationPublished):
            raise ValueError("explanation Attempt admitted a member section")
        if not isinstance(fact.explanation, ReportExplanationByRequest):
            raise ValueError("explanation changed its accepted output")
        expected = derive_explanation_by_request(
            bound,
            admitted_members(timeline),
            draft,
            journey_id=fact.explanation.journey_id,
            supersedes_explanation_id=fact.explanation.supersedes_explanation_id,
            instruction_identity=started["payload"]["instruction_identity"],
            validation_event_id=fact.validation_event_id,
        )
        if fact.explanation != expected:
            raise ValueError("explanation changed its accepted output")


ExplanationStatus = Literal[
    "requested", "review_pending", "answered", "stopped", "failed", "published"
]
# One member's or the combination's state inside a request.
ExplanationStepStatus = Literal[
    "waiting",
    "running",
    "review_pending",
    "answered",
    "admitted",
    "not_explained",
    "stopped",
    "failed",
    "published",
]
_RUN_ARGUMENTS = {
    "continuation_availability_review_event_id": "availability_review_response_event_id",
    "continuation_effort_review_event_id": "effort_review_response_event_id",
    "continuation_work_review_event_id": "work_review_response_event_id",
    "continuation_operational_review_event_id": "operational_review_response_event_id",
}
_SHARED_TERMINALS = frozenset({"AttemptStopped", "AttemptCancelled", "AttemptFailed"})


class PublishedExplanation(FrozenContract):
    explanation: ReportExplanationByRequest
    artifact: ArtifactReference
    publication_event_id: StrictStr


class ExplanationStepView(FrozenContract):
    """One request member's work, or the combination's, read from its Attempts.

    ``answer`` names the Attempt argument that consumes an unused Review answer;
    ``section`` is a settled member's, as the combination is bound to it.
    """

    member_id: StrictStr | None
    status: ExplanationStepStatus
    attempt_ids: tuple[StrictStr, ...] = ()
    pending_review_event_ids: tuple[StrictStr, ...] = ()
    answer: dict[str, StrictStr] | None = None
    section: CombinationSection | None = None


class ExplanationRequestView(FrozenContract):
    """One request's state, read from its members' and combination's Attempts
    and their shared Reviews (#408)."""

    request_event_id: StrictStr
    request: ReportExplanationRequested
    status: ExplanationStatus
    attempt_ids: tuple[StrictStr, ...] = ()
    pending_review_event_ids: tuple[StrictStr, ...] = ()
    explanation: PublishedExplanation | None = None
    failure: ReportExplanationFailed | None = None
    members: tuple[ExplanationStepView, ...] = ()
    combination: ExplanationStepView | None = None
    # The readable page written from this request's published explanation.
    readable_report: ReadableReportWritten | None = None


class ReportExplanationView(FrozenContract):
    requests: tuple[ExplanationRequestView, ...] = ()

    @property
    def current(self) -> ExplanationRequestView | None:
        return self.requests[-1] if self.requests else None

    @property
    def explanations(self) -> tuple[PublishedExplanation, ...]:
        return tuple(
            item.explanation for item in self.requests if item.explanation is not None
        )


def explanation_attempt_ids(events: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(
        event["payload"]["attempt_id"]
        for event in events
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["agent_role"] == EXPLANATION_ROLE
    )


def explanation_event_ids(events: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Every event the report's explanation wrote: its facts, its Attempts' own
    events and its Attempts' Reviews with their answers.

    They are presentation after the report: no reporting status, Finalization
    candidate or Run ending reads them.
    """
    from bio_neb.agentic_journey.continuation import REVIEW_CONTINUATIONS

    attempts = explanation_attempt_ids(events)
    if not attempts and not any(
        event["event_type"] in EXPLANATION_EVENT_TYPES for event in events
    ):
        return frozenset()
    requests = {kind.request_kind for kind in REVIEW_CONTINUATIONS}
    responses = {kind.response_kind for kind in REVIEW_CONTINUATIONS}
    owned: set[str] = set()
    activities: set[str] = set()
    for event in events:
        payload = event.get("payload", {})
        event_type = event["event_type"]
        if "attempt_id" in payload:
            mine = payload["attempt_id"] in attempts
        elif event_type in responses:
            mine = payload["request_event_id"] in owned
        elif event_type.startswith("activity_"):
            mine = payload.get("activity_id") in activities or any(
                parent in owned for parent in event["parent_ids"]
            )
        else:
            mine = event_type in EXPLANATION_EVENT_TYPES or (
                event_type not in {"PassStarted", "PassClosed"}
                and bool(event["parent_ids"])
                and all(parent in owned for parent in event["parent_ids"])
            )
        if mine or (event_type in requests and payload.get("attempt_id") in attempts):
            owned.add(event["event_id"])
            if event_type == "activity_started":
                activities.add(payload["activity_id"])
    return frozenset(owned)


def without_explanation(
    events: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The Run's events with the report's explanation left out."""
    owned = explanation_event_ids(events)
    return [event for event in events if event["event_id"] not in owned]


def _step_reviews(events, attempt_ids):
    """One step's Reviews: those still pending, the latest stopping answer and
    an answer no Attempt consumed yet, as the Attempt argument that consumes it."""
    from bio_neb.agentic_journey.continuation import REVIEW_CONTINUATIONS

    consumed = {
        event["payload"].get(kind.field)
        for event in events
        if event["event_type"] == "AttemptStarted"
        for kind in REVIEW_CONTINUATIONS
    }
    positions = {event["event_id"]: index for index, event in enumerate(events)}
    pending: list[str] = []
    stops: list[dict] = []
    answer = None
    for kind in REVIEW_CONTINUATIONS:
        answers = {
            event["payload"]["request_event_id"]: event
            for event in events
            if event["event_type"] == kind.response_kind
        }
        for request in events:
            if (
                request["event_type"] != kind.request_kind
                or request["payload"]["attempt_id"] not in attempt_ids
            ):
                continue
            response = answers.get(request["event_id"])
            if response is None:
                pending.append(request["event_id"])
            elif response["payload"]["response"]["action"] not in kind.actions:
                stops.append(response)
            elif response["event_id"] not in consumed:
                answer = {_RUN_ARGUMENTS[kind.field]: response["event_id"]}
    stop = max(stops, key=lambda event: positions[event["event_id"]], default=None)
    return tuple(pending), stop, answer


def _step(
    timeline, member_id, attempts, facts, *, published, admitted, failed, ended
) -> ExplanationStepView:
    """One member's or the combination's state from its Attempts, in order."""
    pending, stop, answer = _step_reviews(timeline, frozenset(attempts))
    status: ExplanationStepStatus
    if any(item in published for item in attempts):
        status = "published"
    elif any(item in admitted for item in attempts):
        status = "admitted"
    elif answer is not None:
        status = "answered"
    elif pending:
        status = "review_pending"
    elif stop is not None:
        status = "stopped" if member_id is None else "not_explained"
    elif any(item in failed or item in ended for item in attempts):
        status = "failed"
    else:
        status = "running" if attempts else "waiting"
    section = None
    if member_id is not None and status in {"admitted", "not_explained"}:
        bound = facts[attempts[0]]
        own = bound.targets[0]
        if status == "admitted":
            fact = next(admitted[item] for item in attempts if item in admitted)
            section = CombinationSection(
                request_member_id=member_id,
                request=own.number,
                subject=own.subject,
                status="explained",
                attempt_id=fact.attempt_id,
                validation_event_id=fact.validation_event_id,
                claims=tuple(sorted({claim.claim for claim in fact.claims})),
            )
        else:
            response = stop["payload"]["response"]
            section = CombinationSection(
                request_member_id=member_id,
                request=own.number,
                subject=own.subject,
                status="not_explained",
                attempt_id=next(
                    event["payload"]["attempt_id"]
                    for event in timeline
                    if event["event_id"] == stop["payload"]["request_event_id"]
                ),
                review_response_event_id=stop["event_id"],
                reason=response.get("reason")
                or f"Its Review was answered {response['action']}.",
            )
    return ExplanationStepView(
        member_id=member_id,
        status=status,
        attempt_ids=tuple(attempts),
        pending_review_event_ids=pending,
        answer=answer,
        section=section,
    )


def _request_steps(timeline, request_event_id, rows, facts, outcomes):
    """Each member of one request and its combination, from the bindings the
    timeline holds for that request."""
    members = []
    for member_id, _own in member_rows(rows):
        attempts = [
            attempt_id
            for attempt_id, fact in facts.items()
            if isinstance(fact, ReportExplanationMemberBound)
            and fact.request_event_id == request_event_id
            and fact.request_member_id == member_id
        ]
        members.append(_step(timeline, member_id, attempts, facts, **outcomes))
    combination = _step(
        timeline,
        None,
        [
            attempt_id
            for attempt_id, fact in facts.items()
            if isinstance(fact, ReportExplanationCombinationBound)
            and fact.request_event_id == request_event_id
        ],
        facts,
        **outcomes,
    )
    return tuple(members), combination


def _outcomes(timeline):
    admitted, published, failed, ended = {}, {}, {}, set()
    for event in timeline:
        fact = validate_explanation_fact(event)
        if isinstance(fact, ReportExplanationMemberAdmitted):
            admitted[fact.attempt_id] = fact
        elif isinstance(fact, ReportExplanationPublished):
            published[fact.attempt_id] = (event, fact)
        elif isinstance(fact, ReportExplanationFailed):
            failed[fact.attempt_id] = fact
        elif event["event_type"] in _SHARED_TERMINALS:
            ended.add(event["payload"]["attempt_id"])
    return {
        "admitted": admitted,
        "published": published,
        "failed": failed,
        "ended": ended,
    }


def _bindings(timeline) -> dict[str, ExplanationBinding]:
    return {
        fact.attempt_id: fact
        for event in timeline
        if isinstance(fact := validate_explanation_fact(event), _BINDINGS)
    }


def settled_sections(
    timeline, request_event_id: str, rows: Sequence[ExplanationRow]
) -> tuple[CombinationSection, ...] | None:
    """Every request's settled section, in the report's order, once every member
    is admitted or not explained; None while any member's work is open."""
    members, _combination = _request_steps(
        timeline, request_event_id, rows, _bindings(timeline), _outcomes(timeline)
    )
    if any(member.section is None for member in members):
        return None
    return tuple(member.section for member in members)


def _attempt_selection(timeline, attempt_id: str) -> list[dict]:
    """One explanation Attempt's own events, also while it is still open.

    Its lifecycle is the shared Attempt's: an invalid start or terminal is
    refused, never read as an open Attempt.
    """
    from bio_neb.agentic_journey.attempt_journey import attempt_events_so_far

    starts = [
        event
        for event in timeline
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == attempt_id
    ]
    if len(starts) != 1 or starts[0]["payload"]["agent_role"] != EXPLANATION_ROLE:
        raise ValueError("explanation fact has no explanation Attempt")
    return attempt_events_so_far(timeline, starts[0])


def _same_step(earlier: ExplanationBinding, later: ExplanationBinding) -> bool:
    return type(earlier) is type(later) and bound_member(earlier) == bound_member(later)


def _checked_attempts(
    snapshot, index: int, request: ReportExplanationRequested, bound, earlier
) -> tuple[str, ...]:
    """A request's Attempts, each checked against the report the request names.

    The request names a report this Run published before it and, for a
    rewrite, an earlier explanation of that report; each Attempt's binding is
    what that report gives, its facts are its own accepted output (ADR 0034)
    and a continued Attempt resumes an earlier one of the same request's step:
    the same member, or the combination.
    """
    timeline = snapshot.timeline
    event = timeline[index]
    report = _report_before(snapshot, event["event_id"], request.report_digest)
    if (
        report.report is None
        or report.publication_event_id != request.publication_event_id
        or report.report.report_basis_digest != request.report_basis_digest
        # Its parent is the event before it, or, for a request the coordinator
        # records after its own Pass's start, the Run's head before it.
        or event["parent_ids"]
        not in (
            [timeline[index - 1]["event_id"]],
            [review_head(timeline[:index])["event_id"]],
        )
    ):
        raise ValueError("explanation request names no report this Run published")
    if (
        request.supersedes_explanation_id is not None
        and request.supersedes_explanation_id not in earlier
    ):
        raise ValueError("explanation request supersedes no earlier explanation")
    attempts = tuple(
        attempt_id
        for attempt_id, fact in bound.items()
        if fact.request_event_id == event["event_id"]
    )
    for position, attempt_id in enumerate(attempts):
        validate_explanation_attempt_facts(
            snapshot, _attempt_selection(timeline, attempt_id)
        )
        continued = bound[attempt_id].continued_attempt_id
        if continued is not None and not (
            continued in attempts[:position]
            and _same_step(bound[continued], bound[attempt_id])
        ):
            raise ValueError("explanation Attempt continued another request")
    return attempts


def _request_status(view_members, combination) -> ExplanationStatus:
    steps = (*view_members, combination)
    statuses = {step.status for step in steps}
    if combination.status == "published":
        return "published"
    if "failed" in statuses:
        return "failed"
    if combination.status == "stopped":
        return "stopped"
    if "answered" in statuses:
        return "answered"
    if any(member.status == "waiting" for member in view_members) or (
        combination.status == "waiting"
        and all(member.section is not None for member in view_members)
    ):
        return "requested"
    if "review_pending" in statuses:
        return "review_pending"
    if "running" in statuses:
        return "requested"
    raise ValueError("explanation request ended without its outcome")


def project_report_explanations(
    events: Sequence[Mapping[str, Any]],
    *,
    owner: Journey | None = None,
    snapshot: PrefixSnapshot | None = None,
) -> ReportExplanationView:
    """Each explanation request with its state and published explanation.

    A request names a report this Run published before it; a published
    explanation is its Attempt's accepted output (checked by the Attempt view)
    and its artifact is its own bytes. A request explained by members (#408)
    reads each member's state and its combination's. Each request's report is
    read from the prefix before it: the ``snapshot`` of ``events`` when the
    caller holds one, else the ``owner`` whose merged evidence ``events`` are
    holds that prefix, else it is derived here.
    """
    timeline = [dict(event) for event in events]
    requests: list[tuple[int, dict, ReportExplanationRequested]] = []
    for index, event in enumerate(timeline):
        fact = validate_explanation_fact(event)
        if isinstance(fact, ReportExplanationRequested):
            requests.append((index, event, fact))
    bound = _bindings(timeline)
    outcomes = _outcomes(timeline)
    published = outcomes["published"]
    failed = outcomes["failed"]
    positions = {event["event_id"]: index for index, event in enumerate(timeline)}
    request_ids = {event["event_id"] for _index, event, _request in requests}
    if any(fact.request_event_id not in request_ids for fact in bound.values()) or any(
        attempt_id not in bound
        for attempt_id in (*published, *failed, *outcomes["admitted"])
    ):
        raise ValueError("explanation fact names no explanation request or Attempt")
    snapshot = snapshot or _RequestPrefixes(timeline, owner)
    views: list[ExplanationRequestView] = []
    for index, event, request in requests:
        earlier = {
            fact.explanation.explanation_id
            for published_event, fact in published.values()
            if positions[published_event["event_id"]] < index
            and fact.explanation.report_digest == request.report_digest
        }
        attempts = _checked_attempts(snapshot, index, request, bound, earlier)
        pending, _stop, _answer = _step_reviews(timeline, frozenset(attempts))
        explanation = next(
            (
                PublishedExplanation(
                    explanation=fact.explanation,
                    artifact=validate_explanation_artifact(published_event, fact),
                    publication_event_id=published_event["event_id"],
                )
                for attempt_id in attempts
                if attempt_id in published
                for published_event, fact in (published[attempt_id],)
            ),
            None,
        )
        failure = next(
            (failed[attempt_id] for attempt_id in attempts if attempt_id in failed),
            None,
        )
        if explanation is not None and (
            explanation.explanation.request_event_id != event["event_id"]
            or explanation.explanation.supersedes_explanation_id
            != request.supersedes_explanation_id
        ):
            raise ValueError("report explanation changed its request")
        report = _report_before(snapshot, event["event_id"], request.report_digest)
        members, combination = _request_steps(
            timeline,
            event["event_id"],
            explanation_rows(report.report.collection_view),
            bound,
            outcomes,
        )
        views.append(
            ExplanationRequestView(
                request_event_id=event["event_id"],
                request=request,
                status=_request_status(members, combination),
                attempt_ids=attempts,
                pending_review_event_ids=pending,
                explanation=explanation,
                failure=failure,
                members=members,
                combination=combination,
            )
        )
    open_requests = [item for item in views[:-1] if item.status in _OPEN]
    if open_requests:
        raise ValueError("explanation request opened beside an open request")
    return ReportExplanationView(
        requests=tuple(_with_readable_reports(timeline, positions, views))
    )


def _with_readable_reports(timeline, positions, views):
    """Each written page names a published explanation of its report, written
    after it and once."""
    written = [
        (event, fact)
        for event in timeline
        if isinstance(fact := validate_explanation_fact(event), ReadableReportWritten)
    ]
    pages = {}
    for event, fact in written:
        (view,) = [
            item
            for item in views
            if item.explanation is not None
            and item.explanation.explanation.explanation_id == fact.explanation_id
            and item.request.report_digest == fact.report_digest
            and positions[item.explanation.publication_event_id]
            < positions[event["event_id"]]
            and event.get("parent_ids") == [item.explanation.publication_event_id]
        ] or (None,)
        if view is None or view.request_event_id in pages:
            raise ValueError(
                "readable report names no published explanation of its own"
            )
        pages[view.request_event_id] = fact
    return [
        item.model_copy(update={"readable_report": pages.get(item.request_event_id)})
        for item in views
    ]


_OPEN = frozenset({"requested", "review_pending", "answered"})


__all__ = [
    "ExplanationStepView",
    "ExplainedRequest",
    "UnexplainedRequest",
    "RequestClaims",
    "ReportExplanationByRequest",
    "ReportExplanationMemberBound",
    "ReportExplanationMemberAdmitted",
    "ReportExplanationCombinationBound",
    "admitted_members",
    "derive_explanation_by_request",
    "derive_member_admitted",
    "draft_binding",
    "member_binding",
    "settled_sections",
    "EXPLANATION_EVENT_TYPES",
    "RETIRED_EXPLANATION_SCHEMAS",
    "EXPLANATION_OUTPUT_TOOL",
    "bound_fact",
    "bound_member",
    "validate_explanation_attempt_facts",
    "ExplanationRequestView",
    "PublishedExplanation",
    "ReportExplanationView",
    "explanation_attempt_ids",
    "explanation_event_ids",
    "project_report_explanations",
    "without_explanation",
    "EXPLANATION_MEDIA_TYPE",
    "EXPLANATION_ROLE",
    "ExplainedLimitation",
    "ReportExplanationFailed",
    "ReportExplanationPublished",
    "ReportExplanationRequested",
    "READABLE_REPORT_FILE",
    "ReadableReportWritten",
    "validate_explanation_artifact",
    "validate_explanation_fact",
]
