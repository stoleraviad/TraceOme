"""Reference-identity Tools the planning assignment offers through its one Agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, Any

from pydantic import Field

from bio_neb.agentic_journey.action_policy import ChoiceReason
from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.planning_preconditions import (
    CORRECTION_PENDING,
    QUESTION_PENDING,
    REFERENCE_NOT_OPEN,
)
from bio_neb.agentic_journey.tool_execution import (
    QUESTION_OPENED,
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolPrecondition,
    asked_view,
    function_tool_contract,
    recorded_view,
)
from bio_neb.journey import Journey
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
)
from bio_neb.reference_discovery.reference_action_policy import (
    answered_choice_receipt,
    applicable_reference_receipt,
    question_choice,
    question_page,
    reference_discovery,
    reference_grounding,
    reference_projection,
    reference_requirement,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    DescribeReferenceTool,
    DiscoverReferenceAssembliesTool,
    ReferenceIdentityAdapter,
    ResolveTaxonTool,
    VerifySampleAttestationTool,
)
from bio_neb.reference_discovery.reference_tools import (
    REASON_INVALID,
    REFERENCE_CHOICE_STANDS,
)


@dataclass
class ReferenceIdentityPlannerDependencies:
    parent_activity: Activity
    attempt_id: str
    journey: Journey
    reference_adapter: ReferenceIdentityAdapter
    sample_tool: VerifySampleAttestationTool = field(
        default_factory=VerifySampleAttestationTool
    )
    taxon_tool: ResolveTaxonTool = field(default_factory=ResolveTaxonTool)
    discovery_tool: DiscoverReferenceAssembliesTool = field(
        default_factory=DiscoverReferenceAssembliesTool
    )
    description_tool: DescribeReferenceTool = field(
        default_factory=DescribeReferenceTool
    )
    observed_tool_session: ObservedToolSession | None = None

    def __post_init__(self) -> None:
        if str(uuid.UUID(self.attempt_id)) != self.attempt_id:
            raise ValueError("attempt_id must be canonical")


def _planning_request(
    dependencies: ReferenceIdentityPlannerDependencies,
) -> PlanningHitlRequested:
    from bio_neb.reference_discovery.reference_tools import question_subject

    projection = reference_projection(dependencies)
    kind, evidence_event_id = question_subject(dependencies, projection)
    if kind == "paper":
        return _paper_request(dependencies, projection, evidence_event_id)
    if kind != "reference_assembly_identity":
        requirement = reference_requirement(dependencies, projection, kind)
        label = "sample" if kind == "sample_identity" else "taxon"
        if evidence_event_id == projection.plan_event_id:
            issue = f"The proposal does not contain an exact {label} assertion."
            question = f"Provide the exact {label} identity for Tool verification."
        else:
            issue = _taxon_outcome_issue(projection, evidence_event_id)
            question = "Clarify the exact taxon identity for further Tool verification."
        return PlanningHitlRequested(
            attempt_id=dependencies.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            issue=issue,
            question=question,
            evidence_event_ids=(evidence_event_id,),
            candidates=(),
            recommendation=None,
        )
    requirement = reference_requirement(
        dependencies, projection, "reference_assembly_identity"
    )
    if evidence_event_id == projection.description_event_id:
        return _supplied_reference_request(
            dependencies, projection, requirement, evidence_event_id
        )
    # The question shows the page the model's open choice cites, else the latest
    # page, every row as NCBI listed it.
    page_event_id = question_page(projection)
    assert page_event_id is not None
    discovery = reference_discovery(projection, page_event_id)
    issue, cited, evidence = _reference_choice_basis(projection, discovery)
    question = (
        "Select one presented assembly candidate or clarify the reference identity "
        "requirement. Selection remains a user assertion until exact validation succeeds."
    )
    if cited:
        question = _bounded(f"{question} {cited}", _QUESTION_LIMIT)
    return PlanningHitlRequested(
        attempt_id=dependencies.attempt_id,
        plan_event_id=projection.plan_event_id,
        plan_id=projection.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        issue=issue,
        question=question,
        evidence_event_ids=(page_event_id, *evidence),
        candidates=tuple(
            PlanningHitlCandidate(
                candidate_id=f"assembly:{row.assembly_accession}",
                asserted_value=row.assembly_accession,
                display_label=" — ".join(
                    value
                    for value in (
                        row.assembly_accession,
                        row.assembly_name,
                        row.organism_name,
                        row.assembly_status,
                    )
                    if value
                ),
                evidence_event_ids=(page_event_id,),
            )
            for row in discovery.rows
        ),
        recommendation=None,
    )


def assembly_question(journey, suggestion_event_id) -> PlanningHitlRequested:
    """The researcher's question the coordinator opens on a suggestion, with no model
    turn (owner, #459 D4): use the suggested assembly for the whole Run, its Galaxy
    index bound and the known targets as the suggestion checked them, or keep the
    verified assembly and build its index on CPU. The suggested assembly is labelled
    from the discovery page of this Run that lists it, the verified one from its own
    description; selecting either answers the reference question as any selection
    does."""
    from bio_neb.application.planning_read import planning_evidence
    from bio_neb.parabricks.planning_hitl import assembly_question_ids
    from bio_neb.reference_discovery.reference_facts_journey import (
        AssemblyWithIndexSuggested,
    )

    dependencies = SimpleNamespace(journey=journey)
    event = next(
        e for e in planning_evidence(journey) if e["event_id"] == suggestion_event_id
    )
    fact = AssemblyWithIndexSuggested.model_validate(
        {"event_type": event["event_type"], **event["payload"]}
    )
    tool_input, suggested = fact.tool_input, fact.result
    projection = reference_projection(dependencies)
    requirement = reference_requirement(
        dependencies, projection, "reference_assembly_identity"
    )
    verified = projection.verified_reference
    requested, other = verified.assembly_accession, tool_input.assembly_accession
    page_event_id = next(
        event_id
        for event_id in reversed(projection.discovery_event_ids)
        if reference_discovery(projection, event_id).row(other) is not None
    )
    row = reference_discovery(projection, page_event_id).row(other)
    choice_event_id = projection.choices[-1].source_event_id
    lines = [
        f"- {target.request_text}: {target.outcome.replace('_', ' ')}. {target.reason}"
        for target in suggested.targets
    ]
    index = tool_input.candidate.candidate.candidate_id

    def label(accession, name, status, words):
        return " — ".join(value for value in (accession, name, status, words) if value)

    attempt_id, _ = assembly_question_ids(journey.journey_id, suggestion_event_id)
    return PlanningHitlRequested(
        attempt_id=attempt_id,
        owner="assembly_suggestion",
        plan_event_id=projection.plan_event_id,
        plan_id=projection.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        issue=_bounded(
            f"Galaxy has no fitting index of the requested {requested}; {other}, "
            f"another assembly of the organism, has one ({index}). The planning "
            f"Agent's reason: {tool_input.reason}",
            _ISSUE_LIMIT,
        ),
        question=_bounded(
            f"Run on {other} with its Galaxy index, or keep {requested} and build its "
            f"index on CPU? The known targets, checked on {other} with "
            f"{tool_input.source}: "
            + (" ".join(lines) if lines else "the Run knows no target yet."),
            _QUESTION_LIMIT,
        ),
        evidence_event_ids=(
            page_event_id,
            suggestion_event_id,
            choice_event_id,
            projection.description_event_id,
        ),
        candidates=(
            PlanningHitlCandidate(
                candidate_id=f"assembly:{other}",
                asserted_value=other,
                display_label=label(
                    other,
                    row.assembly_name,
                    row.assembly_status,
                    f"Galaxy index {index}; {suggested.summary}",
                ),
                evidence_event_ids=(page_event_id, suggestion_event_id),
            ),
            PlanningHitlCandidate(
                candidate_id=f"assembly:{requested}",
                asserted_value=requested,
                display_label=label(
                    requested,
                    verified.record.assembly_name,
                    verified.record.assembly_status,
                    "its index built on CPU",
                ),
                evidence_event_ids=(projection.description_event_id,),
            ),
        ),
        recommendation=None,
    )


def _supplied_reference_request(
    dependencies, projection, requirement, description_event_id
) -> PlanningHitlRequested:
    """The question after a supplied accession's inspection did not verify it."""
    choice = projection.choices[-1]
    result = next(
        f.result for f in projection.facts if f.source_event_id == description_event_id
    )
    return PlanningHitlRequested(
        attempt_id=dependencies.attempt_id,
        plan_event_id=projection.plan_event_id,
        plan_id=projection.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        issue=(
            f"Exact inspection did not verify the supplied accession "
            f"{choice.assembly_accession} (inspection result: "
            f"{result.get('resolution', result['status'])})."
        ),
        question=(
            "Clarify the exact reference assembly accession to use. The clarified "
            "accession remains a user assertion until exact inspection verifies it."
        ),
        evidence_event_ids=(choice.source_event_id, description_event_id),
        candidates=(),
        recommendation=None,
    )


def _paper_request(dependencies, projection, search_event_id) -> PlanningHitlRequested:
    """The question before a cited paper is downloaded (owner, #420 F2 b): every row
    of the search page as a candidate, as Europe PMC listed it, on the cited paper's
    own requirement (#420 D1 b)."""
    from bio_neb.paper_intake.tools import PAPER_CANDIDATE, paper_question

    requirement, search = paper_question(dependencies)
    if search.source_event_id != search_event_id:
        raise ValueError("the paper question changed its search")
    page = search.result
    return PlanningHitlRequested(
        attempt_id=dependencies.attempt_id,
        plan_event_id=projection.plan_event_id,
        plan_id=projection.plan_id,
        requirement_id=requirement.requirement_id,
        subject_key=requirement.subject_key,
        issue=_bounded(
            f'Europe PMC lists {page.hit_count or 0} papers for the search "'
            f'{search.tool_input.query}"; this page shows {len(page.rows)}. Which '
            "one your words cite needs your confirmation before it is downloaded.",
            _ISSUE_LIMIT,
        ),
        question=(
            "Select the paper you cite, or clarify which paper it is. It is "
            "downloaded only after your answer."
        ),
        evidence_event_ids=(search_event_id,),
        candidates=tuple(
            PlanningHitlCandidate(
                candidate_id=f"{PAPER_CANDIDATE}{row.candidate_id}",
                asserted_value=row.candidate_id,
                display_label=_bounded(
                    " — ".join(
                        value.strip()
                        for value in (
                            row.title,
                            row.authors,
                            row.journal,
                            row.year,
                            row.doi or row.pmcid or row.pmid,
                        )
                        if value and value.strip()
                    )
                    or row.candidate_id,
                    _LABEL_LIMIT,
                ),
                evidence_event_ids=(search_event_id,),
            )
            for row in page.rows
        ),
        recommendation=None,
    )


_QUESTION_LIMIT = PlanningHitlRequested.model_fields["question"].metadata[0].max_length
_LABEL_LIMIT = (
    PlanningHitlCandidate.model_fields["display_label"].metadata[0].max_length
)
_ISSUE_LIMIT = PlanningHitlRequested.model_fields["issue"].metadata[0].max_length


def _taxon_outcome_issue(projection, evidence_event_id: str) -> str:
    """The latest taxonomy result in plain words: its query and actual outcome."""
    result = next(
        f.result for f in projection.facts if f.source_event_id == evidence_event_id
    )
    outcome = result.get("resolution", result["status"])
    names = ", ".join(
        f"{c['scientific_name']} (TaxId {c['ncbi_tax_id']})"
        for c in result.get("candidates", ())
    )
    return _bounded(
        f'NCBI Taxonomy returned {outcome} for the query "{result.get("query")}"'
        + (f": {names}." if names else "."),
        _ISSUE_LIMIT,
    )


def _bounded(text: str, limit: int) -> str:
    """The text within the field limit; the cited event retains the whole reason."""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _reference_choice_basis(projection, discovery) -> tuple[str, str, tuple[str, ...]]:
    """The issue, the cited Agent reason and the evidence the reference question adds.

    The question opens after a choice only for explicit ambiguity or an unresolved
    inspection of the concrete choice (``reference_tools.reference_question``).
    """
    choice = question_choice(projection)
    if choice is None:
        return (
            "Reference assembly selection or refinement requires researcher direction.",
            "",
            (),
        )
    cited = f"The planning Agent's reason: {choice.reason}"
    if choice.assembly_accession is None:
        return (
            "The planning Agent found no listed assembly that fits the reference "
            "requirement.",
            cited,
            (choice.source_event_id,),
        )
    row = discovery.row(choice.assembly_accession)
    result = next(
        (
            f.result.get("resolution", f.result["status"])
            for f in projection.facts
            if f.source_event_id == projection.description_event_id
        ),
        None,
    )
    return (
        f"Exact inspection did not verify the chosen assembly "
        f"{choice.assembly_accession}"
        + (f" ({row.assembly_name})" if row and row.assembly_name else "")
        + (f" (inspection result: {result})." if result else "."),
        cited,
        tuple(
            event_id
            for event_id in (choice.source_event_id, projection.description_event_id)
            if event_id
        ),
    )


def request_planning_hitl(
    context: ToolContext[ReferenceIdentityPlannerDependencies],
    requirement_id: str,
    evidence_event_id: str,
) -> dict[str, Any]:
    """It covers a missing exact sample or taxon assertion, a latest taxonomy result
    that resolved no taxon, a discovery page that conflicts with the researcher's
    words (another name, a suppressed record, another strain) or a "none" choice,
    and an unresolved inspection of your choice.
    """
    dependencies = context.deps
    request_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal request_event
        request = _planning_request(dependencies)
        accepted = dependencies.parent_activity.record(request)
        request_event = accepted
        return accepted

    return dependencies.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="planning_hitl",
            stage_result_event=lambda _result: (
                request_event if request_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
        ),
    )


def _paper_quote(deps, quote, assembly_accession) -> str | None:
    """The reading fact a quote of a cited paper's words stands in, recorded with the
    choice; None for the researcher's words or the named none (#383 A2)."""
    from bio_neb.application.planning_read import planning_evidence
    from bio_neb.reference_discovery.reference_journey import (
        paper_quote_event,
        quoted,
        researcher_words,
    )
    from bio_neb.reference_discovery.reference_tools import NO_FITTING_CANDIDATE

    if quote is None or assembly_accession == NO_FITTING_CANDIDATE:
        return None
    events = planning_evidence(deps.journey)
    if quoted(researcher_words(events), quote):
        return None
    return paper_quote_event(events, quote)


def record_reference_choice(
    context: ToolContext[ReferenceIdentityPlannerDependencies],
    discovery_event_id: Annotated[
        str,
        Field(
            description=(
                "The ReferenceAssembliesRecorded event id of the discovery page that "
                "lists the assembly you choose."
            )
        ),
    ],
    assembly_accession: Annotated[
        str,
        Field(
            description=(
                'The accession of one row of that page, or "none" when no listed '
                "assembly fits the researcher's words."
            )
        ),
    ],
    reason: ChoiceReason,
    quote: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            max_length=300,
            description=(
                "The researcher's exact words (in the request or an answer) that name "
                "the assembly, strain or organism this row answers: that span, not the "
                "whole request; or, for a value to find in a cited paper, the paper's "
                'words its reading kept; omitted only with "none".'
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    from bio_neb.reference_discovery.reference_journey import ReferenceChoiceRecorded
    from bio_neb.reference_discovery.reference_tools import (
        NO_FITTING_CANDIDATE,
        reference_choice_candidate,
    )

    deps = context.deps
    event = None

    def execute():
        nonlocal event
        view = reference_projection(deps)
        receipt = applicable_reference_receipt(view) or answered_choice_receipt(view)
        event = deps.parent_activity.record(
            ReferenceChoiceRecorded(
                attempt_id=deps.attempt_id,
                grounding=reference_grounding(
                    deps,
                    reference_projection(deps),
                    "reference_assembly_identity",
                    discovery_event_id,
                    receipt.response_event_id if receipt else None,
                ),
                discovery_event_id=discovery_event_id,
                assembly_accession=reference_choice_candidate(assembly_accession),
                quote=quote if assembly_accession != NO_FITTING_CANDIDATE else None,
                quote_event_id=_paper_quote(deps, quote, assembly_accession),
                reason=reason,
            )
        )
        return event

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        execute,
        ToolExecutionAdapter(
            identity="deterministic",
            stage_result_event=lambda result: event,
            artifact_digests=lambda result: (),
        ),
    )


_REFERENCE_RESOLVED = ToolPrecondition(
    "reference_identity_resolved",
    "reference status",
    "The reference identity is already resolved; no reference action remains.",
)


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "request_planning_hitl",
        "planning_agent",
        ("requirement_id: str", "evidence_event_id: str"),
        "PlanningHitlRequested",
        "planning_hitl",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=(
            "input_correction",
            "target_answer",
            "proposal_requirements",
            "paper",
            "reference",
            "companions",
            "execution_planning",
        ),
        examples=("{} (the server supplies the requirement and its evidence)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            _REFERENCE_RESOLVED,
            ToolPrecondition(
                "reference_question_not_needed",
                "reference status",
                "No reference requirement needs the researcher now. {next}",
            ),
            ToolPrecondition(
                "paper_question_not_needed",
                "paper search",
                "No cited paper's search waits for the researcher: search for it with `find_paper`, and ask only when its page lists several rows that fit their words or a row disagrees with them.",
            ),
            ToolPrecondition(
                "reference_choice_uninspected",
                "reference choice",
                "Inspect your concrete choice first with `inspect_reference`; ask only if its result is unresolved.",
            ),
        ),
    ),
    function_tool_contract(
        "record_reference_choice",
        "planning_agent",
        (
            "discovery_event_id: str",
            "assembly_accession: str",
            "reason: str",
            "quote: str | None",
        ),
        "ReferenceChoiceRecorded",
        "deterministic",
        "ReferenceChoiceRecorded",
        None,
        None,
        recorded_view(
            {
                "*": "Inspect the chosen reference with `inspect_reference`; a "
                "choice of none asks the researcher next."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("reference",),
        examples=(
            '{"discovery_event_id": "<a discovery page>", "assembly_accession": "<one row\'s accession>", "quote": "<the researcher\'s exact words>", "reason": "The page lists this RefSeq reference genome of the strain the researcher names."}',
            '{"discovery_event_id": "<a discovery page>", "assembly_accession": "none", "reason": "No listed assembly is of the strain the researcher names."}',
        ),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            _REFERENCE_RESOLVED,
            ToolPrecondition(
                "reference_discovery_not_open",
                "reference discovery",
                "A choice names one discovered candidate; discover the candidate references first with `discover_assemblies`.",
                met_by=("discover_assemblies",),
            ),
            REFERENCE_CHOICE_STANDS,
            ToolPrecondition(
                "assembly_not_on_discovery_page",
                "assembly_accession",
                'Name an accession that is a row of the discovery page you cite, or "none"; read another page with `discover_assemblies` and its page_token, or another query.',
            ),
            ToolPrecondition(
                "researcher_selected_other",
                "assembly_accession",
                "The researcher selected {selected}; record that row, or ask them with `request_planning_hitl` when it conflicts with what NCBI lists.",
            ),
            ToolPrecondition(
                "choice_quote_missing",
                "quote",
                "Quote the researcher's exact words that the chosen assembly answers.",
            ),
            ToolPrecondition(
                "choice_quote_ungrounded",
                "quote",
                "The quote must be the researcher's exact words, copied from the request or one of their answers.",
            ),
            REASON_INVALID,
        ),
    ),
)


__all__ = [
    "TOOL_ROWS",
    "ReferenceIdentityPlannerDependencies",
    "record_reference_choice",
    "request_planning_hitl",
]
