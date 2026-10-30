"""Scientific assignment: source Tools, offers, admission and output validation."""

from __future__ import annotations

import json
import re
import tempfile
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any

from pydantic import Field

from bio_neb.agentic_journey.action_admission import ProposedAgentAction
from bio_neb.agentic_journey.action_policy import (
    CANONICAL_TEXT_PATTERN,
    PreconditionUnmet,
)
from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
from bio_neb.agentic_journey.continuation_policy import AgentInvocationPolicy
from bio_neb.agentic_journey.invocation import (
    ActionAdmissionSpec,
    AssembledAgent,
    AssignmentOutput,
    OutputRepair,
    ToolContext,
    assemble_agent,
)
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.model_settings import ModelEndpointPolicy
from bio_neb.agentic_journey.role_instructions import RoleInstructions
from bio_neb.agentic_journey.source_answers import AnsweredGet, shown_answers
from bio_neb.agentic_journey.source_contacts import observed_source_operation
from bio_neb.agentic_journey.stage_findings import (
    StageFinding,
    StageFindings,
    StageOutput,
    verified_output,
    verify_stage_output,
)
from bio_neb.agentic_journey.structured_output import StructuredOutputSession
from bio_neb.agentic_journey.tool_execution import (
    QUESTION_OPENED,
    ModelView,
    NegativeResult,
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolPrecondition,
    asked_view,
    function_tool_contract,
    recorded_view,
)
from bio_neb.identifier_rule import word_aligned_spans
from bio_neb.investigation.investigation import (
    ADMISSIBLE_LOOKUP_STATUSES,
    DECISION_ASKS,
    RETRIEVED_TARGET_NOT_REQUESTED,
    AssembliesSearched,
    AssemblyReportToolResult,
    AssemblyReportToolResultRecorded,
    AssemblySearchPage,
    BodylessAnswer,
    InvestigationGeneChangePicked,
    InvestigationGeneChangePlaced,
    InvestigationInput,
    InvestigationLookupDecided,
    InvestigationPlanDraft,
    InvestigationProposalRefused,
    InvestigationRequestItem,
    InvestigationSourceCarried,
    InvestigationSourceContactFailed,
    InvestigationSourceContactStarted,
    InvestigationSourceRetrieved,
    InvestigationTargetRead,
    ListedNoneDecision,
    PlacementDecision,
    QuestionKind,
    RefSnpToolResult,
    RefSnpToolResultRecorded,
    RetrievedSource,
    SequenceChoice,
    SequenceDecision,
    SequenceNamesPage,
    SequenceNamesRead,
    UserAssertion,
    VariantSource,
    VersionChoice,
    VersionDecision,
    VersionNaming,
    answered_without_body,
    applied_decisions,
    decided_arguments,
    deferred_finding,
    issue_question,
    item_question,
    lookup_statement,
    member_issue,
    named_version,
    retrieved_target_question,
    validate_investigation_output,
)
from bio_neb.investigation.investigation_request import (
    RESEARCHER_INSTRUCTION_ACTIONS,
    RSID_CANONICAL_PATTERN,
    RSID_RULE,
    PlacedPosition,
    SiteDraft,
    StatedAssembly,
    StatedChange,
    StatedPosition,
    StatedTarget,
    effective_gene_change,
    effective_location,
    effective_requested_rsid,
    researcher_rsids,
    researcher_words,
    typed_answers,
    unread_target_words,
)
from bio_neb.investigation.investigation_sources import (
    EvaContigNaming,
    EvaVariantType,
    look_up_eva_rsid,
    look_up_region,
    look_up_stated_position,
    normalize_assembly_report,
    normalize_refsnp,
)
from bio_neb.investigation.item_history import item_call_history, item_history
from bio_neb.investigation.observation_defaults import RegionDefaults
from bio_neb.investigation.vcf_observation import allele_text
from bio_neb.paper_reading.reader import Paper
from bio_neb.paper_reading.tools import (
    READING_TOOLS,
    filter_paper_rows,
    grep_paper,
    outline_paper,
    read_paper,
    reading_admission_errors,
    reading_bound_values,
    reading_tool_rows,
)
from bio_neb.providers.eva_investigation import fetch_source
from bio_neb.providers.ncbi_assembly_report import (
    ReportUnreadable,
    record_of,
    search_rows,
)
from bio_neb.providers.ncbi_datasets import genome_report_page
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceAdapter,
    InvestigationSourceResponse,
)
from bio_neb.providers.source_operations import SourceSurface
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    AssemblyLevelOption,
    AssemblyQuery,
    AssemblySourceOption,
    AssemblyVersionOption,
    ExcludeAtypicalOption,
    ExcludeMultiIsolateOption,
    ExcludePairedReportsOption,
    FirstReleaseDateOption,
    HasAnnotationOption,
    IctvExemplarOption,
    LastReleaseDateOption,
    MetagenomeDerivedOption,
    PageSizeOption,
    PageTokenOption,
    ReferenceOnlyOption,
    SearchTextOption,
    SortDirectionOption,
    SortFieldOption,
    TaxExactMatchOption,
    TypeMaterialCategoryOption,
    TypeMaterialOption,
    assembly_query,
)


@dataclass
class InvestigationPlannerDependencies:
    parent_activity: Activity
    attempt_id: str
    investigation_input: InvestigationInput
    assertion: UserAssertion
    request: InvestigationRequestItem
    collection_member_id: str
    revision_number: int
    supersedes_proposal_id: str | None
    adapter: InvestigationSourceAdapter
    failure_source: AttemptFailureSource
    observed_tool_session: ObservedToolSession
    # The member's answers with the target the researcher or the model typed for each
    # (`typed_answers`), and as recorded, with their response event ids.
    researcher_answers: tuple[dict, ...] = ()
    raw_answers: tuple[dict, ...] = ()
    # The model's target readings of this member's words, and where an untyped request
    # item's own words start, the Investigation's start (R1 b).
    target_reads: list[dict[str, Any]] = field(default_factory=list)
    started_event_id: str | None = None
    journey: Any = None
    source_events: list[dict[str, Any]] = field(default_factory=list)
    selected_sources: (
        tuple[RefSnpToolResult, AssemblyReportToolResult | None] | None
    ) = None
    validation_event_id: str | None = None
    question_event: dict[str, Any] | None = None
    structured_output_session: StructuredOutputSession | None = None
    # Each assembly's sequence names this member's lookups read, by accession: the
    # retaining event's id and its bytes, which `search_sequence_names` reads.
    sequence_reports: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    # Each page of NCBI's genome reports this member's `search_assemblies` read: a
    # stated version the model names is checked against their rows.
    assembly_searches: list[AssemblySearchPage] = field(default_factory=list)
    # This member's gene-change placements, as recorded events, and the candidates
    # the model picked from them, as payloads (Q11).
    gene_change_placements: list[dict[str, Any]] = field(default_factory=list)
    gene_change_picks: list[dict[str, Any]] = field(default_factory=list)
    # The item's lookup decisions for its current target, as recorded events, which
    # its lookups read (G2.3).
    decisions: list[dict[str, Any]] = field(default_factory=list)
    # Planning's retained GETs of the Run's assembly the Investigation's start carries,
    # by the Investigation operation and key each stands in for: the planning event,
    # its bytes and media type. A lookup reads them in place of a contact.
    carried_sources: dict[tuple[str, str], tuple[dict[str, Any], bytes, str]] = field(
        default_factory=dict
    )
    # The Run organism's species-level taxon `search_assemblies` is bound to;
    # ``None`` when planning recorded none.
    run_taxid: int | None = None
    # The item's lookup results for a target a researcher's answer or a new revision
    # has since replaced: evidence still, but no decision cites them.
    replaced_result_ids: frozenset[str] = frozenset()
    # The paper that proposed this member's target, which the reading Tools
    # read; ``None`` for a target no paper proposed.
    paper: Paper | None = None


class SourceReadingOutput(StageOutput):
    stage = "source_reading"
    evidence_types = {
        "refsnp_event_id": frozenset({"RefSnpToolResultRecorded"}),
        "assembly_event_ids": frozenset({"AssemblyReportToolResultRecorded"}),
    }
    agrees_with = {"refsnp_result_id": "refsnp_event_id"}
    refsnp_event_id: str
    refsnp_result_id: str
    assembly_event_ids: tuple[str, ...]


def investigation_stage_findings(
    deps: InvestigationPlannerDependencies,
) -> StageFindings:
    """Show this member's checked source results, including lawful negatives."""
    events = tuple(deps.source_events)
    outputs = []
    for event in events:
        if event["event_type"] != "RefSnpToolResultRecorded":
            continue
        assemblies = tuple(
            item["event_id"]
            for item in events
            if item["event_type"] == "AssemblyReportToolResultRecorded"
            and item["payload"]["refsnp_event_id"] == event["event_id"]
        )
        if (
            event["payload"]["result"]["status"] in ADMISSIBLE_LOOKUP_STATUSES
            and not assemblies
        ):
            continue
        output = SourceReadingOutput(
            refsnp_event_id=event["event_id"],
            refsnp_result_id=event["payload"]["result"]["result_id"],
            assembly_event_ids=assemblies,
        )
        if verify_stage_output(output, events) is None:
            outputs.append(verified_output(output))
    return StageFindings(
        stage="source_reading",
        findings=tuple(
            StageFinding(
                kind=event["event_type"],
                source_event_id=event["event_id"],
                status=event["payload"]["result"]["status"],
                subject_key=deps.request.rsid,
                value=event["payload"]["result"]["result_id"],
            )
            for event in events
        ),
        output="verified" if outputs else "open",
        verified_outputs=tuple(outputs),
    )


def investigation_action_errors(
    dependencies: InvestigationPlannerDependencies,
    action: ProposedAgentAction,
) -> tuple[str, ...]:
    """A lookup is bound to the member's request; assembly admission requires the
    exact resolved observation it checks. Every Run is offered both lookup Tools, and
    which source to ask is the model's choice (#469, ADR 0049). Words the model has
    yet to type a target from are read first (#472 Q3 a, R1 b)."""
    if action.tool_name in READING_TOOLS:
        # A read of the member's paper, refused only where the paper has no such
        # view; it may come before the member's words are read.
        return reading_admission_errors(dependencies, action)
    unread = _unread_words(dependencies)
    if action.tool_name == "record_target_reading":
        return _reading_errors(dependencies, unread, action.arguments)
    if unread:
        return ("target_words_unread",)
    if action.tool_name in ("place_gene_change", "pick_gene_change_candidate"):
        return _gene_change_errors(dependencies, action.tool_name, action.arguments)
    merge_targets = {
        rsid
        for event in dependencies.source_events
        if event["event_type"] == "RefSnpToolResultRecorded"
        for rsid in event["payload"]["result"].get("merged_into", ())
    }
    position = effective_location(
        dependencies.request, dependencies.researcher_answers
    ).position
    region = effective_location(
        dependencies.request, dependencies.researcher_answers
    ).region
    if action.tool_name == "locate_variant":
        if position is None and region is None:
            return ("investigation_position_missing",)
        source = action.arguments.get("source")
        eva_options = [
            name
            for name in ("variant_type", "contig_naming")
            if action.arguments.get(name) is not None
        ]
        if region is not None and (source is not None or eva_options):
            # A region lists no rsID and reads NCBI alone.
            return ("region_names_no_source",)
        if region is None and source is None:
            # A position's rsIDs are listed by the source the model names.
            return ("locate_variant_source_missing",)
        if eva_options and source != "eva":
            # EVA's own options go only with EVA (SP4 a).
            return ("eva_option_without_eva",)
        return ()
    if action.tool_name in DECISION_TOOLS:
        return _decision_errors(dependencies, action)
    if action.tool_name == "search_assemblies":
        # NCBI's own query, the model's choice; a version it names is checked by
        # membership in the rows it returns.
        return ()
    if action.tool_name == "search_sequence_names":
        return (
            ()
            if action.arguments.get("assembly_accession")
            in dependencies.sequence_reports
            else ("sequence_names_unread",)
        )
    if action.tool_name in LOOKUP_TOOLS and (
        position is not None or region is not None
    ):
        # A stated position is placed first; its lookup reads the rsID, and
        # follows a dbSNP merge, too.
        return ("stated_position_needs_location",)
    if action.tool_name in LOOKUP_TOOLS:
        # A member stating no rsID and no position is recorded not checked before any
        # Attempt, so every member that reaches a lookup states one. A follow-up is an
        # rsID a source named: a merge record's copied current rsID, or one a retained
        # answer of this member's lookups names.
        stated = researcher_rsids(dependencies.request, dependencies.researcher_answers)
        rsid = action.arguments.get("rsid")
        if rsid in {*stated, *merge_targets} or named_in_answers(
            rsid, _member_lookups(dependencies)
        ):
            return ()
        return (
            ("eva_query_not_requested",)
            if action.tool_name == "fetch_eva_variant"
            else ("fetch_refsnp_query_not_requested",)
        )
    if action.tool_name == "request_investigation_information":
        return _question_errors(dependencies, action)
    basis = _cited_lookup(dependencies, action)
    if basis is None:
        # A lookup this member already has is named, never "call the lookup
        # first" (the false text sent members into repeated calls).
        return (
            ("fetch_assembly_report_refsnp_event_unknown",)
            if _member_lookups(dependencies)
            else ("fetch_assembly_report_requires_typed_refsnp_result",)
        )
    if basis["payload"]["result"]["status"] not in ADMISSIBLE_LOOKUP_STATUSES:
        return ("fetch_assembly_report_requires_resolved_refsnp_result",)
    return ()


def _question_errors(
    dependencies: InvestigationPlannerDependencies, action: ProposedAgentAction
) -> tuple[str, ...]:
    """A member asks once, before its output is accepted; a cited result is one of
    its own lookups (#479 A, ADR 0049 membership)."""
    if (
        dependencies.question_event is not None
        or dependencies.validation_event_id is not None
    ):
        return ("investigation_question_not_needed",)
    if "refsnp_event_id" in action.arguments and (
        _cited_lookup(dependencies, action) is None
    ):
        return ("question_cite_unknown",)
    return ()


def named_in_answers(rsid: Any, lookups: list[Mapping[str, Any]]) -> bool:
    """Whether a retained source answer of these lookups names ``rsid`` (owner #468 Q3 a,
    ADR 0049: membership only): the rsID's number in one of the fields its source writes
    an rsID in (`_RSID_FIELDS`, by the answer's operation: EVA writes a clustered
    variant's accession as a bare number), or the rsID as one whole token in any of the
    answer's text. Another kind of id (a taxon, a submitted variant's ss) names none."""
    if not isinstance(rsid, str) or re.fullmatch(RSID_CANONICAL_PATTERN, rsid) is None:
        return False
    number = rsid[2:]
    token = re.compile(rf"(?<![A-Za-z0-9]){rsid}(?![A-Za-z0-9])")

    def at(value: Any, path: tuple[str, ...]) -> list[Any]:
        if isinstance(value, list):
            return [item for element in value for item in at(element, path)]
        if not path:
            return [value]
        if isinstance(value, Mapping) and path[0] in value:
            return at(value[path[0]], path[1:])
        return []

    def in_text(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(in_text(item) for item in value.values())
        if isinstance(value, list):
            return any(in_text(item) for item in value)
        return isinstance(value, str) and bool(token.search(value))

    def names(answer: Mapping[str, Any]) -> bool:
        body = answer["body"]
        fields = _RSID_FIELDS.get(answer["operation"], ())
        return in_text(body) or any(
            (
                isinstance(item, int)
                and not isinstance(item, bool)
                and str(item) == number
            )
            or item == number
            for path in fields
            for item in at(body, path)
        )

    return any(
        names(answer)
        for event in lookups
        for answer in event["payload"]["source_answers"]
    )


_RSID_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    # dbSNP's record: its own id and the rsIDs a merge names.
    "fetch_refsnp": (
        ("refsnp_id",),
        ("merged_snapshot_data", "merged_into"),
        ("dbsnp1_merges", "merged_rsid"),
    ),
    # EVA's clustered variants: each record's accession is its rs number, and a merge
    # names the clustered variant it went to.
    "eva_clustered_variant": (
        ("accession",),
        ("mergedInto",),
        ("destinationAccession",),
    ),
    "eva_clustered_variants_at": (("accession",),),
    # EVA's submitted variants: an ss's own accession is no rsID; its cluster's is.
    "eva_submitted_variants": (("data", "clusteredVariantAccession"),),
    "ncbi_spdi_rsids": (("data", "rsids"),),
}
"""Where each source writes an rsID's number, by the operation that answered: the fields a
follow-up may cite (owner #468 Q3 a; Spec r2 SP1: by the source's own record, never by a
key's spelling)."""


def version_naming(
    dependencies: InvestigationPlannerDependencies, choice: VersionChoice
) -> VersionNaming | str:
    """The model's naming of the stated version, checked by membership only (#471 Q3
    a; `investigation.named_version`) over this member's searches and lookups."""
    return named_version(
        effective_location(
            dependencies.request, dependencies.researcher_answers
        ).stated_assembly,
        dependencies.assembly_searches,
        _member_lookups(dependencies),
        dependencies.investigation_input.assembly_accession,
        choice,
    )


def _member_lookups(dependencies) -> list[Mapping[str, Any]]:
    """This member's lookup results, in the order they were recorded."""
    return [
        event
        for event in dependencies.source_events
        if event["event_type"] == "RefSnpToolResultRecorded"
    ]


def _cited_lookup(
    dependencies, action, *, replaced: bool = False
) -> Mapping[str, Any] | None:
    """The member's lookup a call cites: one for its current target, or with
    ``replaced`` any of the item's (a replaced target's lookup is evidence only, #479
    G2.3)."""
    event_id = action.arguments.get("refsnp_event_id")
    return next(
        (
            event
            for event in _member_lookups(dependencies)
            if event["event_id"] == event_id
            and (
                replaced
                or event_id not in getattr(dependencies, "replaced_result_ids", ())
            )
        ),
        None,
    )


def investigation_refusal_facts(
    dependencies: InvestigationPlannerDependencies,
    action: ProposedAgentAction,
    code: str,
) -> dict[str, Any]:
    """The typed facts a member refusal's sentence names (#372 D2): the
    member's lookup results, the cited lookup's outcome and the rsIDs it may
    look up."""
    if code in {"target_words_unread", "target_words_read"}:
        return {
            "unread": ", ".join(
                f"`{event_id}`" for event_id in _unread_words(dependencies)
            )
            or "none"
        }
    if code == "decision_not_asked":
        return {
            "status": _cited_lookup(dependencies, action)["payload"]["result"]["status"]
        }
    if code in {
        "fetch_assembly_report_refsnp_event_unknown",
        "question_cite_unknown",
        "decision_result_unknown",
    }:
        return {
            "lookups": ", ".join(
                f"`{event['event_id']}` ({event['payload']['result']['status']})"
                for event in _member_lookups(dependencies)
            )
        }
    if code == "fetch_assembly_report_requires_resolved_refsnp_result":
        result = _cited_lookup(dependencies, action)["payload"]["result"]
        merged = tuple(result.get("merged_into", ()))
        return {
            "status": result["status"],
            "next": (
                f"It is merged into {', '.join(merged)}: look that rsID up with "
                "fetch_refsnp, then check the assembly of its result."
                if merged
                else 'Return compile_proposal with outcome "unresolved", citing '
                "this refsnp_event_id."
            ),
        }
    stated = researcher_rsids(dependencies.request, dependencies.researcher_answers)
    if code == "stated_position_needs_location":
        location = effective_location(
            dependencies.request, dependencies.researcher_answers
        )
        if location.position is not None:
            return {"value": _position_text(location.position)}
        region = location.region
        return {
            "value": None
            if region is None
            else f"{region.contig}:{region.start}-{region.end}"
        }
    if code == "locate_variant_source_missing":
        # The source the call left out, never the argument dict it sent.
        return {"value": None}
    if code == "eva_option_without_eva":
        return {
            "value": ", ".join(
                f"{name}={action.arguments[name]}"
                for name in ("variant_type", "contig_naming")
                if action.arguments.get(name) is not None
            )
        }
    if code == "region_names_no_source":
        return {
            "value": action.arguments.get("source")
            or action.arguments.get("variant_type")
            or action.arguments.get("contig_naming")
        }
    if code == "investigation_gene_change_missing":
        return {
            "value": None,
            "next": "look up its rsID or its stated position, or ask the researcher "
            "with `request_investigation_information`.",
        }
    changed = effective_gene_change(
        dependencies.request, dependencies.researcher_answers
    )
    if code == "investigation_position_missing" and changed is not None:
        # A gene change is placed first; its picked candidate is the
        # position.
        return {
            "value": None,
            "next": "place its gene change with `place_gene_change`, then pick a "
            "candidate with `pick_gene_change_candidate`.",
        }
    if code == "investigation_position_missing":
        return {
            "value": None,
            "next": (
                f"look up {', '.join(f'`{rsid}`' for rsid in stated)} with "
                "`fetch_refsnp` or `fetch_eva_variant`, the source that suits the "
                "Run's organism."
                if stated
                else "ask the researcher for the variant with "
                "`request_investigation_information`."
            ),
        }
    if code == "eva_query_not_requested":
        code = "fetch_refsnp_query_not_requested"
    if code == "investigation_question_not_needed":
        if dependencies.question_event is not None:
            return {
                "value": "question pending",
                "reason": "this member's question is pending; its answer comes "
                "back on a later request.",
            }
        return {
            "value": "output accepted",
            "reason": "this member's output is already accepted.",
        }
    if code == "fetch_refsnp_query_not_requested":
        merged = {
            rsid
            for event in _member_lookups(dependencies)
            for rsid in event["payload"]["result"].get("merged_into", ())
        }
        return {"allowed": tuple(dict.fromkeys((*stated, *sorted(merged))))}
    return {}


def _position_text(position: StatedPosition | PlacedPosition) -> str:
    """A position as the request reads: sequence, position and alleles."""
    place = f"{position.contig}:{position.position}"
    if position.ref is None:
        return place
    return f"{place} {allele_text(position.ref, position.alt)}"


def investigation_tool_values(deps: Any, name: str) -> SimpleNamespace:
    """Each planner Tool's server-bound values: none for the lookups (#469: no lookup
    Tool is bound to the Run's assembly), the words a target reading may read (#472),
    and the Run organism's TaxId for an assembly search (#479 Q4 A)."""
    if name in READING_TOOLS:
        # Only a member whose target a paper proposed has a paper to read.
        if deps.paper is None:
            raise PreconditionUnmet(MEMBER_HAS_NO_PAPER.code)
        return reading_bound_values(deps, name)
    if name == "record_target_reading":
        unread = tuple(_unread_words(deps))
        if not unread:
            raise PreconditionUnmet("target_words_all_read")
        return SimpleNamespace(constants={}, enums={"source_event_id": unread})
    if name in DECISION_TOOLS and not any(
        event["payload"]["result"]["status"] == DECISION_ASKS[DECISION_TOOLS[name]]
        and event["event_id"] not in deps.replaced_result_ids
        for event in _member_lookups(deps)
    ):
        # A decision Tool waits until a result of the current target asks for it
        # (callable now only then).
        raise PreconditionUnmet("decision_none_asked")
    if name == "search_assemblies":
        # The Run's organism, never a typed TaxId.
        if deps.run_taxid is None:
            raise PreconditionUnmet("run_taxon_unknown")
        return SimpleNamespace(constants={"taxid": deps.run_taxid}, enums={})
    return SimpleNamespace(constants={}, enums={})


def _unread_words(deps: InvestigationPlannerDependencies) -> dict[str, str]:
    """The member's words the model has yet to read, by the event that holds them: an
    untyped item's own words (the Investigation's start) and each instruction answer
    the researcher typed no target for (#472 Q3 a, R1 b)."""
    pending = set(
        unread_target_words(
            deps.request,
            deps.raw_answers,
            deps.target_reads,
            started_event_id=deps.started_event_id,
        )
    )
    words = {deps.started_event_id: deps.request.request_text}
    words.update(
        (answer["response_event_id"], answer["instruction"])
        for answer in deps.raw_answers
        if "instruction" in answer
    )
    return {event_id: words[event_id] for event_id in pending}


def _reading_errors(
    deps: InvestigationPlannerDependencies,
    unread: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    """A target reading reads words not yet read, and types a target only from a quote
    of them that holds it (#472 Q3 a, ADR 0049 membership), or says why none."""
    words = unread.get(arguments.get("source_event_id"))
    if words is None:
        return ("target_words_read",)
    quote, rsid, site, change = (
        arguments.get("quote"),
        arguments.get("rsid"),
        arguments.get("site"),
        arguments.get("gene_change"),
    )
    if rsid is None and site is None and change is None:
        return () if arguments.get("reason") else ("target_reading_reason_missing",)
    if not quote or not word_aligned_spans(words, quote):
        return ("target_quote_ungrounded",)
    if rsid is not None and not RSID_RULE.matches(rsid, quote):
        return ("target_rsid_not_in_quote",)
    if site is not None and not SiteDraft.model_validate(site).in_quote(quote):
        return ("target_site_not_in_quote",)
    if change is not None:
        typed = StatedChange.model_validate(change)
        if not typed.typed:
            # Code reads no change text: the model types it.
            return ("target_gene_change_untyped",)
        if not typed.in_quote(quote):
            return ("target_gene_change_not_in_quote",)
    if arguments.get("stated_version") is not None and site is None:
        return ("target_version_without_site",)
    return ()


def _record_source(
    dependencies: InvestigationPlannerDependencies,
    *,
    operation: str,
    contact: InvestigationSourceResponse,
    answers: list[AnsweredGet],
) -> tuple[RetrievedSource, str]:
    """Retain each of one operation's GETs, its earlier ones first, and keep each in
    call order for the Tool's return; the last is the operation's source."""
    for answer in (*contact.preceding, contact):
        answers.append(
            AnsweredGet(
                operation=operation,
                request_identity=answer.request_identity,
                status_code=answer.status_code,
                body=answer.body,
            )
        )
        source = RetrievedSource.create(
            operation=operation,
            request_identity=answer.request_identity,
            status_code=answer.status_code,
            observed_at=answer.observed_at,
            elapsed_time_ms=answer.elapsed_time_ms,
            raw_response=answer.body,
        )
        retained = dependencies.parent_activity.record_with_artifact(
            InvestigationSourceRetrieved(
                attempt_id=dependencies.attempt_id,
                source=source,
            ),
            answer.body,
            media_type=answer.media_type,
        )
    return source, retained["event_id"]


def _carried(
    dependencies: InvestigationPlannerDependencies,
    operation: str,
    key: str,
    answers: list[AnsweredGet],
) -> tuple[RetrievedSource, str, InvestigationSourceResponse] | None:
    """Planning's retained GET this one stands in for, read in place of a contact
    (#479 G): its bytes retained on an ``InvestigationSourceCarried`` where the GET
    would be, and shown as the cited prior source; ``None`` when none is carried."""
    carried = dependencies.carried_sources.get((operation, key))
    if carried is None:
        return None
    cited, body, media_type = carried
    payload = cited["payload"]
    source = RetrievedSource.create(
        operation=operation,
        request_identity=payload["request_identity"],
        status_code=payload["status_code"],
        observed_at=cited["occurred_at"],
        # No contact of this Tool's: the GET is planning's.
        elapsed_time_ms=0,
        raw_response=body,
    )
    retained = dependencies.parent_activity.record_with_artifact(
        InvestigationSourceCarried(
            attempt_id=dependencies.attempt_id,
            cited_event_id=cited["event_id"],
            source=source,
        ),
        body,
        media_type=media_type,
    )
    answers.append(
        AnsweredGet(
            operation=operation,
            request_identity=payload["request_identity"],
            status_code=payload["status_code"],
            body=body,
            cited_event_id=cited["event_id"],
        )
    )
    return (
        source,
        retained["event_id"],
        InvestigationSourceResponse(
            request_identity=payload["request_identity"],
            status_code=payload["status_code"],
            body=body,
            observed_at=cited["occurred_at"],
            elapsed_time_ms=0,
            media_type=media_type,
        ),
    )


def _requested_variant(dependencies: InvestigationPlannerDependencies):
    """The one allele a member's typed position names, on the Run's sequence name:
    the name its position lookup found (#369). ``None`` before one is found."""
    stated = effective_location(
        dependencies.request, dependencies.researcher_answers
    ).position
    if stated is None or stated.identity is None:
        return None
    for event in reversed(dependencies.source_events):
        location = (
            event["payload"]["result"].get("location")
            if event["event_type"] == "RefSnpToolResultRecorded"
            else None
        )
        if isinstance(location, dict) and location.get("run_contig"):
            # A position its lookup moved onto the Run's assembly names the allele
            # at its placed position.
            placed = location.get("placed")
            identity = (
                stated.identity
                if placed is None
                else PlacedPosition.model_validate(placed).identity
            )
            return (
                None
                if identity is None
                else identity._replace(contig=location["run_contig"])
            )
    return None


def _applied(deps: InvestigationPlannerDependencies, tool_name: str, rsid):
    """The item's decisions this lookup call reads (#479 G2.2)."""
    return applied_decisions(
        tool_name,
        rsid,
        deps.decisions,
        {event["event_id"]: event for event in _member_lookups(deps)},
    )


def _ids(events) -> tuple[str, ...]:
    return tuple(event["event_id"] for event in events)


def fetch_refsnp(
    context: ToolContext[InvestigationPlannerDependencies],
    rsid: Annotated[
        str,
        Field(
            pattern=RSID_CANONICAL_PATTERN,
            description="Canonical rsID lookup term (rs and its number); preserve the researcher request separately.",
        ),
    ],
) -> dict[str, Any]:
    """Read a chosen query; retain every typed result for this member. A placement the
    item named for this rsID (`name_placement`) is read, never re-sent (#479 G2.2)."""
    dependencies = context.deps
    session = dependencies.observed_tool_session
    applied = _applied(dependencies, "fetch_refsnp", rsid)
    placement = decided_arguments(applied).get("placement")

    def typed_operation() -> dict[str, Any]:
        dependencies.parent_activity.record(
            InvestigationSourceContactStarted(
                attempt_id=dependencies.attempt_id,
                operation="fetch_refsnp",
                request_key=rsid,
            )
        )
        try:
            from bio_neb.agentic_journey.source_contacts import call_source

            contact = call_source(
                "fetch_refsnp",
                lambda: fetch_source(dependencies.adapter, "fetch_refsnp", rsid),
            )
        except Exception as error:
            failed = dependencies.parent_activity.record(
                InvestigationSourceContactFailed(
                    attempt_id=dependencies.attempt_id,
                    operation="fetch_refsnp",
                    request_key=rsid,
                    error_type=type(error).__name__,
                    detail="NCBI RefSNP contact ended without a source result.",
                )
            )
            dependencies.failure_source.capture(failed)
            raise
        answers: list[AnsweredGet] = []
        source, source_event_id = _record_source(
            dependencies,
            operation="fetch_refsnp",
            contact=contact,
            answers=answers,
        )
        result = normalize_refsnp(
            rsid=rsid,
            requested_rsid=effective_requested_rsid(
                dependencies.request, dependencies.researcher_answers
            ),
            assembly_accession=(dependencies.investigation_input.assembly_accession),
            source=source,
            raw_response=contact.body,
            requested_variant=_requested_variant(dependencies),
            placement=placement,
        )
        event = dependencies.parent_activity.record(
            RefSnpToolResultRecorded(
                attempt_id=dependencies.attempt_id,
                source_event_id=source_event_id,
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                query=rsid,
                result=result,
                source_answers=shown_answers(answers),
                decision_event_ids=_ids(applied),
            )
        )
        dependencies.source_events.append(event)
        return event

    return session.execute(
        context.tool_call_id,
        observed_source_operation(
            dependencies.parent_activity,
            dependencies.attempt_id,
            context.tool_call_id,
            "fetch_refsnp",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="ncbi_investigation",
            stage_result_event=lambda _result: dependencies.source_events[-1],
            artifact_digests=lambda returned: (
                returned["payload"]["result"]["source"]["raw_response_sha256"],
            ),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type
                == "InvestigationSourceContactFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


def fetch_eva_variant(
    context: ToolContext[InvestigationPlannerDependencies],
    rsid: Annotated[
        str,
        Field(
            pattern=RSID_CANONICAL_PATTERN,
            description="Canonical rsID lookup term (rs and its number); preserve the researcher request separately.",
        ),
    ],
    contig_naming: Annotated[
        EvaContigNaming | None,
        Field(
            description="EVA's own contigNamingConvention for the placements and "
            "submitted variants it answers, passed to EVA as is (default: EVA's "
            "INSDC). EVA answers 404 when it cannot name every placement in it.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Read a chosen query from EVA; retain every GET and the typed result. A
    placement the item named for this rsID is read, never re-sent (#479 G2.2)."""
    dependencies = context.deps
    session = dependencies.observed_tool_session
    applied = _applied(dependencies, "fetch_eva_variant", rsid)
    placement = decided_arguments(applied).get("placement")

    def typed_operation() -> dict[str, Any]:
        from bio_neb.agentic_journey.source_contacts import call_source

        source_event_ids: dict[str, str] = {}
        answers: list[AnsweredGet] = []

        def fetch(operation: str, key: str):
            if (carried := _carried(dependencies, operation, key, answers)) is not None:
                # Planning's report of the Run's assembly, read again.
                source, event_id, contact = carried
                source_event_ids.setdefault(operation, event_id)
                return source, contact.body
            # Each GET is its own contact on its own operation's failure surface.
            dependencies.parent_activity.record(
                InvestigationSourceContactStarted(
                    attempt_id=dependencies.attempt_id,
                    operation=operation,
                    request_key=key,
                )
            )
            try:
                contact = call_source(
                    operation,
                    lambda: fetch_source(dependencies.adapter, operation, key),
                )
            except Exception as error:
                failed = dependencies.parent_activity.record(
                    InvestigationSourceContactFailed(
                        attempt_id=dependencies.attempt_id,
                        operation=operation,
                        request_key=key,
                        error_type=type(error).__name__,
                        detail="EVA contact ended without a source result.",
                    )
                )
                dependencies.failure_source.capture(failed)
                raise
            source, event_id = _record_source(
                dependencies, operation=operation, contact=contact, answers=answers
            )
            source_event_ids.setdefault(operation, event_id)
            return source, contact.body

        result = look_up_eva_rsid(
            rsid=rsid,
            requested_rsid=effective_requested_rsid(
                dependencies.request, dependencies.researcher_answers
            ),
            assembly_accession=dependencies.investigation_input.assembly_accession,
            fetch=fetch,
            requested_variant=_requested_variant(dependencies),
            placement=placement,
            contig_naming=contig_naming,
        )
        event = dependencies.parent_activity.record(
            RefSnpToolResultRecorded(
                attempt_id=dependencies.attempt_id,
                source_event_id=source_event_ids["eva_clustered_variant"],
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                query=rsid,
                result=result,
                source_answers=shown_answers(answers),
                decision_event_ids=_ids(applied),
            )
        )
        dependencies.source_events.append(event)
        return event

    return session.execute(
        context.tool_call_id,
        observed_source_operation(
            dependencies.parent_activity,
            dependencies.attempt_id,
            context.tool_call_id,
            "fetch_eva_variant",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="eva_investigation",
            stage_result_event=lambda _result: dependencies.source_events[-1],
            # The result binds its clustered-variant bytes; each supporting GET
            # keeps its own retained source event.
            artifact_digests=lambda returned: (
                returned["payload"]["result"]["source"]["raw_response_sha256"],
            ),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type
                == "InvestigationSourceContactFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


def locate_variant(
    context: ToolContext[InvestigationPlannerDependencies],
    source: Annotated[
        VariantSource | None,
        Field(
            description="The variant source that lists and reads a stated position's "
            "rsIDs, your choice: ncbi_dbsnp (dbSNP: human variants) or eva (EVA: "
            "non-human variants). A source with no record there is its answer; you may "
            "ask the other. A region lists no rsID: omit it.",
        ),
    ] = None,
    variant_type: Annotated[
        EvaVariantType | None,
        Field(
            description="With source eva: EVA's own variantType for its variants at "
            "the stated position, passed to EVA as is (default: the stated allele's "
            "type in EVA's form). SNV, MNV, INS and DEL are one, several, inserted or "
            "deleted bases; INDEL both inserts and deletes; TANDEM_REPEAT, "
            "SEQUENCE_ALTERATION, NO_SEQUENCE_ALTERATION, NO_ALTERNATE, SV and CNV "
            "are EVA's other classes. EVA answers a type it does not file the "
            "variant under with an empty 404.",
        ),
    ] = None,
    contig_naming: Annotated[
        EvaContigNaming | None,
        Field(
            description="With source eva: EVA's own contigNamingConvention for its "
            "GETs, passed to EVA as is (default: EVA's INSDC). The listing sends the "
            "Run sequence's name in it, from NCBI's report: INSDC its GenBank "
            "accession, REFSEQ its RefSeq accession, UCSC its UCSC-style name, "
            "GENBANK_SEQUENCE_NAME its sequence name; EVA's ENA name, MD5_CHECKSUM, "
            "TRUNC512_CHECKSUM and NO_REPLACEMENT have no name there. EVA names its "
            "answers in it too. EVA answers a convention it does not hold for the "
            "assembly with a 404.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Place this member's stated position and read what the chosen variant source
    keeps there; retain every GET and the typed result (#369, #469). The sequences,
    version, placement and listed none the item decided (#467, #471, #468) are read
    from its decision facts, never re-sent (#479 G2.2)."""
    dependencies = context.deps
    session = dependencies.observed_tool_session
    applied = _applied(dependencies, "locate_variant", None)
    decided = decided_arguments(applied)
    sequences = tuple(
        SequenceChoice.model_validate(item) for item in decided.get("sequences", ())
    )
    version = (
        None
        if "version" not in decided
        else VersionChoice.model_validate(decided["version"])
    )
    placement = decided.get("placement")
    listed_none = decided.get("listed_none")
    if isinstance(listed_none, dict):
        listed_none = BodylessAnswer.model_validate(listed_none)

    def typed_operation() -> dict[str, Any]:
        from bio_neb.agentic_journey.source_contacts import call_source

        source_event_ids: dict[str, str] = {}
        keys: dict[str, str] = {}
        answers: list[AnsweredGet] = []

        def fetch(operation: str, key: str):
            if (carried := _carried(dependencies, operation, key, answers)) is not None:
                # Planning's GET of the Run's assembly, read again.
                source, event_id, contact = carried
                source_event_ids[source.source_id] = event_id
                keys[source.source_id] = key
                return source, contact.body
            # Each GET is its own contact on its own operation's failure surface.
            dependencies.parent_activity.record(
                InvestigationSourceContactStarted(
                    attempt_id=dependencies.attempt_id,
                    operation=operation,
                    request_key=key,
                )
            )
            try:
                contact = call_source(
                    operation,
                    lambda: fetch_source(dependencies.adapter, operation, key),
                )
            except Exception as error:
                failed = dependencies.parent_activity.record(
                    InvestigationSourceContactFailed(
                        attempt_id=dependencies.attempt_id,
                        operation=operation,
                        request_key=key,
                        error_type=type(error).__name__,
                        detail="A position lookup contact ended without a source "
                        "result.",
                    )
                )
                dependencies.failure_source.capture(failed)
                raise
            source, event_id = _record_source(
                dependencies, operation=operation, contact=contact, answers=answers
            )
            source_event_ids[source.source_id] = event_id
            keys[source.source_id] = key
            if operation == "ncbi_sequence_names":
                dependencies.sequence_reports[key] = (event_id, contact.body)
            return source, contact.body

        location = effective_location(
            dependencies.request, dependencies.researcher_answers
        )
        words = researcher_words(dependencies.request, dependencies.researcher_answers)
        # Admission checked the naming; it is read again here as the lookup's own.
        named_version = (
            None if version is None else version_naming(dependencies, version)
        )
        if location.region is not None:
            # A region is located on the Run's assembly, its bases read.
            result = look_up_region(
                region=location.region,
                stated_assembly=location.stated_assembly,
                assembly_accession=dependencies.investigation_input.assembly_accession,
                fetch=fetch,
                max_bases=RegionDefaults().max_bases,
                choices=sequences,
                words=words,
                version=named_version,
            )
        else:
            result = look_up_stated_position(
                position=location.stated,
                placed=location.placed,
                stated_assembly=location.stated_assembly,
                rsid=next(
                    iter(
                        researcher_rsids(
                            dependencies.request, dependencies.researcher_answers
                        )
                    ),
                    None,
                ),
                requested_rsid=effective_requested_rsid(
                    dependencies.request, dependencies.researcher_answers
                ),
                assembly_accession=dependencies.investigation_input.assembly_accession,
                fetch=fetch,
                choices=sequences,
                words=words,
                version=named_version,
                placement=placement,
                listed_none=listed_none,
                source=source,
                variant_type=variant_type,
                contig_naming=contig_naming,
            )
        event = dependencies.parent_activity.record(
            RefSnpToolResultRecorded(
                attempt_id=dependencies.attempt_id,
                source_event_id=source_event_ids[result.source.source_id],
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                query=keys[result.source.source_id],
                result=result,
                source_answers=shown_answers(answers),
                decision_event_ids=_ids(applied),
            )
        )
        dependencies.source_events.append(event)
        return event

    return session.execute(
        context.tool_call_id,
        observed_source_operation(
            dependencies.parent_activity,
            dependencies.attempt_id,
            context.tool_call_id,
            "locate_variant",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="variant_location",
            stage_result_event=lambda _result: dependencies.source_events[-1],
            artifact_digests=lambda returned: (
                returned["payload"]["result"]["source"]["raw_response_sha256"],
            ),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type
                == "InvestigationSourceContactFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


SEQUENCE_NAMES_PAGE_ROWS = 25
"""Rows one search returns; ``offset`` reaches the rest (ADR 0049, bounded views)."""


def search_sequence_names(
    context: ToolContext[InvestigationPlannerDependencies],
    assembly_accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="An assembly whose sequence names a locate_variant result of "
            "this member read.",
        ),
    ],
    text: Annotated[
        str,
        Field(
            max_length=200,
            description="Text any column of a row holds, case aside (a name, an "
            "accession, a role, a unit); empty lists every row.",
        ),
    ] = "",
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """Search an assembly's sequence names as NCBI's assembly report copies them
    (#467, owner Q1 a): its retained bytes are read, no source is asked."""
    dependencies = context.deps
    event = None

    def read() -> dict[str, Any]:
        nonlocal event
        report_event_id, body = dependencies.sequence_reports[assembly_accession]
        try:
            rows, total = search_rows(
                body, text, offset=offset, limit=SEQUENCE_NAMES_PAGE_ROWS
            )
        except ReportUnreadable as unreadable:
            raise RuntimeError(str(unreadable)) from unreadable
        event = dependencies.parent_activity.record(
            SequenceNamesRead(
                attempt_id=dependencies.attempt_id,
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                result=SequenceNamesPage(
                    assembly_accession=assembly_accession,
                    report_source_event_id=report_event_id,
                    text=text,
                    offset=offset,
                    total_matches=total,
                    rows=tuple(row.copied() for row in rows),
                ),
            )
        )
        return event

    return dependencies.observed_tool_session.execute(
        context.tool_call_id,
        read,
        ToolExecutionAdapter(
            identity="sequence_names_read",
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (),
        ),
    )


_DECISION_CITE = Field(
    description="The cite (refsnp_event_id) of this member's lookup result that asks "
    "for this decision."
)


def _record_decision(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: str,
    decision: SequenceDecision
    | VersionDecision
    | PlacementDecision
    | ListedNoneDecision,
    identity: str,
) -> dict[str, Any]:
    """The one constructor every decision Tool writes through (#479 G2.2): the item's
    ``InvestigationLookupDecided`` fact citing the result that asked, which its later
    lookups read."""
    deps = context.deps
    event = None

    def record():
        nonlocal event
        event = deps.parent_activity.record(
            InvestigationLookupDecided(
                attempt_id=deps.attempt_id,
                collection_member_id=deps.collection_member_id,
                revision_number=deps.revision_number,
                tool_call_id=context.tool_call_id,
                refsnp_event_id=refsnp_event_id,
                decision=decision,
            )
        )
        deps.decisions.append(event)
        return event

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        record,
        ToolExecutionAdapter(
            identity=identity,
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (),
        ),
    )


def name_sequence(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: Annotated[str, _DECISION_CITE],
    assembly_accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="An assembly the sequence_unnamed result names.",
        ),
    ],
    record: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description="The GenBank or RefSeq accession of the record of that "
            "assembly's sequence names you judge the written contig means.",
        ),
    ],
    quote: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="The researcher's exact words that write the contig.",
        ),
    ],
) -> dict[str, Any]:
    """Name the sequence a written contig means on one assembly a sequence_unnamed
    result asks about (#467, #479 G2.2); later locate_variant calls read it."""
    return _record_decision(
        context,
        refsnp_event_id,
        SequenceDecision(
            choice=SequenceChoice(
                assembly_accession=assembly_accession, record=record, quote=quote
            )
        ),
        "lookup_decision",
    )


def name_version(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: Annotated[str, _DECISION_CITE],
    assembly_accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="The assembly the stated genome version means: the Run's own, "
            "a row of a search_assemblies page of this member, or one a dbSNP "
            "record's placements name.",
        ),
    ],
    quote: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="The researcher's exact words that state the version.",
        ),
    ],
) -> dict[str, Any]:
    """Name the assembly a version_unnamed result's stated version means (#471,
    #479 G2.2); later locate_variant calls read it."""
    return _record_decision(
        context,
        refsnp_event_id,
        VersionDecision(
            choice=VersionChoice(assembly_accession=assembly_accession, quote=quote)
        ),
        "lookup_decision",
    )


def name_placement(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: Annotated[str, _DECISION_CITE],
    placement: Annotated[
        str,
        Field(
            pattern=CANONICAL_TEXT_PATTERN,
            description="The run_contig of the placement the request means, one of "
            "the placement_unnamed result's placements on the Run's assembly.",
        ),
    ],
) -> dict[str, Any]:
    """Name the placement a placement_unnamed result's record means (#468, #479
    G2.2); the same lookup called again reads it."""
    return _record_decision(
        context,
        refsnp_event_id,
        PlacementDecision(placement=placement),
        "lookup_decision",
    )


def quote_listed_none(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: Annotated[str, _DECISION_CITE],
    quote: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=500,
            pattern=CANONICAL_TEXT_PATTERN,
            description="The listing service's own words saying no rsID is listed at "
            "the stated allele, quoted from the result's source_answers.",
        ),
    ] = None,
    status_code: Annotated[
        int | None,
        Field(
            ge=100,
            le=599,
            description="Only when the listing's answer has no body: its HTTP status, "
            "as source_answers shows it, in place of a quote.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Record a listing's own words that it lists no rsID at the stated allele, or the
    status of its answer when that answer has no body, for a source_answered position
    result (#468 R5 b, #479 G2.2, owner #480 Q3 a); later locate_variant calls read the
    listing as empty when its answer holds the words, or has that status and no body."""
    return _record_decision(
        context,
        refsnp_event_id,
        ListedNoneDecision(quote=quote, status_code=status_code),
        "lookup_decision",
    )


DECISION_TOOLS = {
    "name_sequence": "sequence",
    "name_version": "version",
    "name_placement": "placement",
    "quote_listed_none": "listed_none",
}
"""Each decision Tool and the decision kind it records (#479 G2.2)."""


def _decision_errors(
    deps: InvestigationPlannerDependencies, action: ProposedAgentAction
) -> tuple[str, ...]:
    """A decision cites this item's lookup result that asks for it, for its current
    target, and names one of that result's options (ADR 0049 membership)."""
    kind = DECISION_TOOLS[action.tool_name]
    cited = _cited_lookup(deps, action, replaced=True)
    if cited is None:
        return ("decision_result_unknown",)
    if cited["event_id"] in deps.replaced_result_ids:
        # A decision is the current target's.
        return ("decision_result_replaced",)
    result = cited["payload"]["result"]
    if result["status"] != DECISION_ASKS[kind] or (
        kind == "listed_none" and result["location"] == "not_applicable"
    ):
        return ("decision_not_asked",)
    origin = result.get("origin") or {}
    arguments = action.arguments
    if kind == "placement":
        return (
            ()
            if arguments.get("placement") in origin.get("sequences", ())
            else ("placement_not_listed",)
        )
    if kind == "listed_none":
        quote, status = arguments.get("quote"), arguments.get("status_code")
        if (quote is None) == (status is None):
            return ("listed_none_basis_missing",)
        if status is not None:
            # A body-less answer is cited by its status: the cited result's own
            # answering GET has that status and zero bytes (review r1 ST1), not
            # merely no words once trimmed.
            return (
                ()
                if answered_without_body(
                    RefSnpToolResult.model_validate_json(json.dumps(result)), status
                )
                else ("listed_none_status_not_bodyless",)
            )
        return (
            ()
            if quote and quote in origin.get("words_sent", "")
            else ("listed_none_quote_ungrounded",)
        )
    if kind == "version":
        naming = version_naming(
            deps,
            VersionChoice(
                assembly_accession=arguments["assembly_accession"],
                quote=arguments["quote"],
            ),
        )
        return (naming,) if isinstance(naming, str) else ()
    accession = arguments.get("assembly_accession")
    if accession not in origin.get("assemblies", ()):
        return ("sequence_assembly_not_asked",)
    if accession not in deps.sequence_reports:
        return ("sequence_names_unread",)
    quote = arguments.get("quote") or ""
    if origin.get("contig") not in quote or not any(
        quote in words
        for words in researcher_words(deps.request, deps.researcher_answers)
    ):
        return ("sequence_quote_ungrounded",)
    try:
        listed = record_of(deps.sequence_reports[accession][1], arguments["record"])
    except ReportUnreadable:
        listed = None
    return () if listed is not None else ("sequence_record_not_listed",)


def search_assemblies(
    context: ToolContext[InvestigationPlannerDependencies],
    taxid: Annotated[
        int,
        Field(
            gt=0,
            description="Bound by the Run: the Run organism's TaxId at species "
            "level; never typed.",
        ),
    ],
    reference_only: ReferenceOnlyOption = None,
    assembly_source: AssemblySourceOption = None,
    has_annotation: HasAnnotationOption = None,
    exclude_paired_reports: ExcludePairedReportsOption = None,
    exclude_atypical: ExcludeAtypicalOption = None,
    assembly_version: AssemblyVersionOption = None,
    assembly_level: AssemblyLevelOption = None,
    first_release_date: FirstReleaseDateOption = None,
    last_release_date: LastReleaseDateOption = None,
    search_text: SearchTextOption = None,
    is_metagenome_derived: MetagenomeDerivedOption = None,
    is_type_material: TypeMaterialOption = None,
    is_ictv_exemplar: IctvExemplarOption = None,
    exclude_multi_isolate: ExcludeMultiIsolateOption = None,
    type_material_category: TypeMaterialCategoryOption = None,
    tax_exact_match: TaxExactMatchOption = None,
    sort_field: SortFieldOption = None,
    sort_direction: SortDirectionOption = None,
    page_size: PageSizeOption = 100,
    page_token: PageTokenOption = None,
) -> dict[str, Any]:
    """List one page of NCBI Datasets genome assembly reports for the Run's organism
    with NCBI's own query options, to name the assembly a stated genome version means
    (#471 R3 a): every row copied as NCBI lists it, with NCBI's total_count and next
    page token. Nothing is ranked, cut or matched by code. The TaxId is the Run
    organism's at species level, bound by code (owner #479 Q4 A): NCBI's default
    includes its descendant taxa, so the same organism's other assemblies stay
    reachable."""
    dependencies = context.deps
    query = assembly_query(
        reference_only=reference_only,
        assembly_source=assembly_source,
        has_annotation=has_annotation,
        exclude_paired_reports=exclude_paired_reports,
        exclude_atypical=exclude_atypical,
        assembly_version=assembly_version,
        assembly_level=assembly_level,
        first_release_date=first_release_date,
        last_release_date=last_release_date,
        search_text=search_text,
        is_metagenome_derived=is_metagenome_derived,
        is_type_material=is_type_material,
        is_ictv_exemplar=is_ictv_exemplar,
        exclude_multi_isolate=exclude_multi_isolate,
        type_material_category=type_material_category,
        tax_exact_match=tax_exact_match,
        sort_field=sort_field,
        sort_direction=sort_direction,
        page_size=page_size,
        page_token=page_token,
    )
    key = f"{taxid}?{urllib.parse.urlencode(query.parameters())}"
    operation = "ncbi_taxon_assemblies"
    event: dict[str, Any] | None = None
    digest: str | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal event, digest
        from bio_neb.agentic_journey.source_contacts import call_source

        dependencies.parent_activity.record(
            InvestigationSourceContactStarted(
                attempt_id=dependencies.attempt_id,
                operation=operation,
                request_key=key,
            )
        )
        try:
            contact = call_source(
                operation, lambda: fetch_source(dependencies.adapter, operation, key)
            )
        except Exception as error:
            failed = dependencies.parent_activity.record(
                InvestigationSourceContactFailed(
                    attempt_id=dependencies.attempt_id,
                    operation=operation,
                    request_key=key,
                    error_type=type(error).__name__,
                    detail="An assembly search contact ended without a source result.",
                )
            )
            dependencies.failure_source.capture(failed)
            raise
        answers: list[AnsweredGet] = []
        source, source_event_id = _record_source(
            dependencies, operation=operation, contact=contact, answers=answers
        )
        digest = source.raw_response_sha256
        page = _search_page(taxid, query, contact)
        event = dependencies.parent_activity.record(
            AssembliesSearched(
                attempt_id=dependencies.attempt_id,
                source_event_id=source_event_id,
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                result=page,
                source_answers=shown_answers(answers),
            )
        )
        dependencies.assembly_searches.append(page)
        return event

    return dependencies.observed_tool_session.execute(
        context.tool_call_id,
        observed_source_operation(
            dependencies.parent_activity,
            dependencies.attempt_id,
            context.tool_call_id,
            "search_assemblies",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="ncbi_investigation",
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (digest,),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type
                == "InvestigationSourceContactFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


def _search_page(
    taxid: int, query: AssemblyQuery, contact: InvestigationSourceResponse
) -> AssemblySearchPage:
    """NCBI's answer to a search as the Tool records it (#471 R3 a; ADR 0049): a 2xx
    genome-report page copied row by row; any other answered GET with no rows and
    its status, its body reaching the model in the Tool's source answers."""
    answered = {
        "ncbi_tax_id": taxid,
        "query": query,
        "status_code": contact.status_code,
    }
    if not 200 <= contact.status_code < 300:
        return AssemblySearchPage(**answered)
    try:
        rows, total_count, next_page_token = genome_report_page(contact.body)
    except ValueError as unread:
        return AssemblySearchPage(**answered, unread=str(unread))
    return AssemblySearchPage(
        **answered,
        rows=rows,
        total_count=total_count,
        next_page_token=next_page_token,
    )


def fetch_assembly_report(
    context: ToolContext[InvestigationPlannerDependencies],
    refsnp_event_id: Annotated[
        str,
        Field(
            description="Exact resolved RefSNP source_event_id from this member stage findings."
        ),
    ],
    assembly_accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="Versioned NCBI assembly accession to look up, as your lookup result's candidate names it; source validation checks agreement with the candidate.",
        ),
    ],
) -> dict[str, Any]:
    """Check a chosen assembly query against an explicit resolved observation."""
    dependencies = context.deps
    session = dependencies.observed_tool_session

    def typed_operation() -> dict[str, Any]:
        event = next(
            event
            for event in dependencies.source_events
            if event["event_id"] == refsnp_event_id
        )
        refsnp = RefSnpToolResult.model_validate_json(
            json.dumps(event["payload"]["result"])
        )
        candidate = refsnp.candidates[0]
        answers: list[AnsweredGet] = []
        carried = _carried(
            dependencies, "fetch_assembly_report", assembly_accession, answers
        )
        if carried is not None:
            # Planning's report of the Run's assembly, read again.
            source, source_event_id, contact = carried
        else:
            source, source_event_id, contact = _assembly_report_contact(
                dependencies, assembly_accession, answers
            )
        result = normalize_assembly_report(
            expected_assembly_accession=candidate.assembly_accession,
            expected_assembly_name=candidate.assembly_name,
            source=source,
            raw_response=contact.body,
        )
        event = dependencies.parent_activity.record(
            AssemblyReportToolResultRecorded(
                attempt_id=dependencies.attempt_id,
                source_event_id=source_event_id,
                collection_member_id=dependencies.collection_member_id,
                tool_call_id=context.tool_call_id,
                query=assembly_accession,
                refsnp_event_id=refsnp_event_id,
                result=result,
                source_answers=shown_answers(answers),
            )
        )
        dependencies.source_events.append(event)
        return event

    return session.execute(
        context.tool_call_id,
        observed_source_operation(
            dependencies.parent_activity,
            dependencies.attempt_id,
            context.tool_call_id,
            "fetch_assembly_report",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="ncbi_investigation",
            stage_result_event=lambda _result: dependencies.source_events[-1],
            artifact_digests=lambda returned: (
                returned["payload"]["result"]["source"]["raw_response_sha256"],
            ),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type
                == "InvestigationSourceContactFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


def _assembly_report_contact(
    dependencies: InvestigationPlannerDependencies,
    assembly_accession: str,
    answers: list[AnsweredGet],
):
    """One NCBI assembly-report GET on its own failure surface, retained."""
    from bio_neb.agentic_journey.source_contacts import call_source

    dependencies.parent_activity.record(
        InvestigationSourceContactStarted(
            attempt_id=dependencies.attempt_id,
            operation="fetch_assembly_report",
            request_key=assembly_accession,
        )
    )
    try:
        contact = call_source(
            "fetch_assembly_report",
            lambda: fetch_source(
                dependencies.adapter, "fetch_assembly_report", assembly_accession
            ),
        )
    except Exception as error:
        failed = dependencies.parent_activity.record(
            InvestigationSourceContactFailed(
                attempt_id=dependencies.attempt_id,
                operation="fetch_assembly_report",
                request_key=assembly_accession,
                error_type=type(error).__name__,
                detail="NCBI assembly-report contact ended without a source result.",
            )
        )
        dependencies.failure_source.capture(failed)
        raise
    source, source_event_id = _record_source(
        dependencies,
        operation="fetch_assembly_report",
        contact=contact,
        answers=answers,
    )
    return source, source_event_id, contact


def request_investigation_information(
    context: ToolContext[InvestigationPlannerDependencies],
    question: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="The exact missing intent to ask the researcher; first use the request and retained answers.",
        ),
    ],
    missing_information: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="Explain what remains missing and why the checked request and evidence cannot supply it.",
        ),
    ],
    asks_about: Annotated[
        QuestionKind,
        Field(
            description="What you ask: target (which rsID or site the researcher "
            "means), version (the genome version a site is on) or intent (anything "
            "else they mean). The sample's genotype, call or VCF is the Run's own "
            "result, read after approval: never ask for it."
        ),
    ],
    refsnp_event_id: Annotated[
        str | None,
        Field(
            description="The cite of the lookup result this question rests on (its "
            "refsnp_event_id), such as a source placing the requested rsID or "
            "position on another organism or assembly: code copies that result's "
            "statement into the question. Omit it when no result prompts it.",
        ),
    ] = None,
) -> dict[str, Any]:
    deps = context.deps

    def record():
        cited = (
            None
            if refsnp_event_id is None
            else next(
                e for e in _member_lookups(deps) if e["event_id"] == refsnp_event_id
            )
        )
        deps.question_event = deps.parent_activity.record(
            _item_question(
                deps,
                question=question,
                missing_information=missing_information,
                asks_about=asks_about,
                cited=cited,
            )
        )
        return deps.question_event

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        record,
        ToolExecutionAdapter(
            identity="investigation_review",
            stage_result_event=lambda _: deps.question_event,
            artifact_digests=lambda _: (),
        ),
    )


def ask_about_retrieved_target(deps: InvestigationPlannerDependencies) -> bool:
    """Ask the researcher when the output repair allowance is spent and the refusal
    that spent it is the intent check's: the retrieved target still differs from the
    request. The caller establishes that the allowance is spent; any other stop keeps
    its own Review. The question holds only this member."""
    refused = (
        None
        if deps.structured_output_session is None
        else deps.structured_output_session.latest_validation(deps.collection_member_id)
    )
    if (
        refused is None
        or refused.status != "invalid"
        or refused.errors[:1] != (RETRIEVED_TARGET_NOT_REQUESTED,)
    ):
        return False
    question, missing_information = retrieved_target_question(refused.errors)
    deps.question_event = deps.parent_activity.record(
        _item_question(
            deps,
            question=question,
            missing_information=missing_information,
            asks_about="target",
            cited=_lookup_event(deps, refused.output["refsnp_event_id"]),
            refused_output_event_id=refused.validation_event_id,
        )
    )
    return True


def ask_about_finding_issue(deps: InvestigationPlannerDependencies) -> bool:
    """Key an accepted finding that ended with an issue and ask its own Review (#359).

    The failure owner keys the issue by its source and verdict; with no
    correction allowance the member's Review opens at once and holds only this
    member. A complete finding asks nothing.
    """
    accepted = (
        None
        if deps.structured_output_session is None
        else deps.structured_output_session.latest_validation(deps.collection_member_id)
    )
    if (
        accepted is None
        or accepted.status != "valid"
        or accepted.validation_event_id != deps.validation_event_id
        or accepted.output["outcome"] != "unresolved"
        or deps.selected_sources is None
    ):
        return False
    refsnp, assembly = deps.selected_sources
    if deferred_finding(refsnp, accepted.output["refsnp_event_id"]) is not None:
        # Deferred out of the Run with its reason: it asks nothing.
        return False
    issue = member_issue(
        refsnp,
        assembly,
        finding_event_id=accepted.validation_event_id,
        refsnp_event_id=accepted.output["refsnp_event_id"],
        assembly_event_id=accepted.output["assembly_event_id"],
    )
    question, missing_information = issue_question(issue, deps.request)
    deps.question_event = deps.parent_activity.record(
        _item_question(
            deps,
            question=question,
            missing_information=missing_information,
            asks_about="intent",
            cited=_lookup_event(deps, accepted.output["refsnp_event_id"]),
            issue=issue,
        )
    )
    return True


def refresh_target_facts(deps: InvestigationPlannerDependencies) -> None:
    """Set the item's target-tied facts from its one history as the Journey holds it
    now: its decisions, gene-change placements and picks for the current target, and
    the lookups a replaced target holds (#479 G2.3)."""
    from bio_neb.recording import read_evidence

    history = item_history(
        read_evidence(deps.journey),
        attempt_id=deps.attempt_id,
        member_id=deps.collection_member_id,
        revision_number=deps.revision_number,
        # Target-tied facts read no retained bytes.
        read_artifact=lambda _event: b"",
    )
    deps.decisions = list(history.decisions)
    deps.gene_change_placements = list(history.gene_change_placements)
    deps.gene_change_picks = [event["payload"] for event in history.gene_change_picks]
    deps.replaced_result_ids = history.replaced_result_ids


def _item_question(deps: InvestigationPlannerDependencies, **question: Any):
    """This member's question in the one item-question shape (#479 C)."""
    return item_question(
        attempt_id=deps.attempt_id,
        assertion_id=deps.assertion.assertion_id,
        collection_member_id=deps.collection_member_id,
        revision_number=deps.revision_number,
        request=deps.request,
        **question,
    )


def _lookup_event(deps: InvestigationPlannerDependencies, event_id: str):
    """This member's lookup fact by its id."""
    return next(e for e in _member_lookups(deps) if e["event_id"] == event_id)


def record_target_reading(
    context: ToolContext[InvestigationPlannerDependencies],
    source_event_id: Annotated[
        str,
        Field(
            description="The event whose words you read: a researcher answer to this "
            "member, or the Investigation's start for the request's own words."
        ),
    ],
    quote: Annotated[
        str | None,
        Field(
            max_length=300,
            description="The words' exact span that states the target, on whole "
            "words; leave it out when they state none.",
        ),
    ] = None,
    rsid: Annotated[
        str | None,
        Field(
            pattern=RSID_CANONICAL_PATTERN,
            description="The rsID the quote states, typed as rs and digits, whatever "
            "the quote's case.",
        ),
    ] = None,
    site: Annotated[
        SiteDraft | None,
        Field(description="The site the quote writes by coordinates, typed."),
    ] = None,
    stated_version: Annotated[
        str | None,
        Field(
            max_length=200,
            pattern=CANONICAL_TEXT_PATTERN,
            description="The genome version the quote writes for the site, as written.",
        ),
    ] = None,
    gene_change: Annotated[
        StatedChange | None,
        Field(
            description="The gene and change the quote writes, typed: the protein "
            "change, the coding change or both, and the transcript when written."
        ),
    ] = None,
    reason: Annotated[
        str | None,
        Field(
            max_length=2000,
            description="When the words state no target: why not, citing them.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Type the target this member's words state, or say why they state none (#472
    Q3 a, R1 b): the reading replaces the member's target for its lookups, as a typed
    answer does. Words that state none are asked about with
    request_investigation_information."""
    deps = context.deps
    words = _unread_words(deps)[source_event_id]
    typed = rsid is not None or site is not None or gene_change is not None
    fact = InvestigationTargetRead(
        attempt_id=deps.attempt_id,
        collection_member_id=deps.collection_member_id,
        revision_number=deps.revision_number,
        source="request" if source_event_id == deps.started_event_id else "answer",
        source_event_id=source_event_id,
        words=words,
        quote=quote if typed else None,
        target=(
            StatedTarget(
                rsid=rsid,
                position=None if site is None else site.stated(),
                gene_change=gene_change,
            )
            if typed
            else None
        ),
        stated_assembly=(
            StatedAssembly(name=stated_version, cited_line=quote)
            if typed and site is not None and stated_version is not None
            else None
        ),
        reason=None if typed else reason,
    )
    event = None

    def record():
        nonlocal event
        event = deps.parent_activity.record(fact)
        deps.target_reads.append(event["payload"])
        # A reading may change the target: the target-tied facts are read again from
        # the item's history before the next call.
        refresh_target_facts(deps)
        deps.researcher_answers = typed_answers(
            deps.request, deps.raw_answers, deps.target_reads, deps.gene_change_picks
        )
        return event

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        record,
        ToolExecutionAdapter(
            identity="target_reading",
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (),
        ),
    )


def place_gene_change(
    context: ToolContext[InvestigationPlannerDependencies],
    transcript: Annotated[
        str | None,
        Field(
            max_length=200,
            pattern=CANONICAL_TEXT_PATTERN,
            description="A transcript accession a previous place_gene_change result "
            "of this member lists, to read the change on it alone; omit it to read "
            "every transcript of the gene.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Place this member's typed gene change on the Run's assembly (#473 Q1 a, Q3 a):
    the gene's transcripts from the assembly's own annotation, each codon read and
    translated with the transcript's copied genetic code, every candidate SNV that
    gives the stated change, and NCBI's HGVS answer when a transcript is named, all
    copied. Pick a candidate with pick_gene_change_candidate."""
    deps = context.deps
    session = deps.observed_tool_session

    def typed_operation() -> dict[str, Any]:
        from bio_neb.agentic_journey.source_contacts import call_source
        from bio_neb.target_request.ncbi_sources import NcbiSources, answer_claim
        from bio_neb.target_request.records import GeneChangeClaim

        answers: list[AnsweredGet] = []
        last: list[str] = []

        def contact(operation: str, key: str, read):
            if (carried := _carried(deps, operation, key, answers)) is not None:
                # Planning's GET of the Run's assembly, read again.
                last.append(carried[1])
                return read(carried[2])
            # Each GET is its own contact on its own operation's surface, its policy
            # applied once, here (ADR 0031): a fault the policy gives up on, at the
            # GET or reading its answer, is the Tool's failure and its Review's.
            deps.parent_activity.record(
                InvestigationSourceContactStarted(
                    attempt_id=deps.attempt_id, operation=operation, request_key=key
                )
            )

            def once():
                response = fetch_source(deps.adapter, operation, key)
                _source, event_id = _record_source(
                    deps, operation=operation, contact=response, answers=answers
                )
                last.append(event_id)
                return read(response)

            try:
                return call_source(operation, once)
            except Exception as error:
                event = deps.parent_activity.record(
                    InvestigationSourceContactFailed(
                        attempt_id=deps.attempt_id,
                        operation=operation,
                        request_key=key,
                        error_type=type(error).__name__,
                        detail="A gene-change placement contact ended without a "
                        "readable source result.",
                    )
                )
                deps.failure_source.capture(event)
                raise

        change = effective_gene_change(deps.request, deps.researcher_answers)
        # The transcript the model named reads the change on it alone (Q4 a).
        claimed = (
            change
            if transcript is None
            else change.model_copy(update={"transcript": transcript})
        )
        with tempfile.TemporaryDirectory() as cache:
            result = answer_claim(
                NcbiSources(deps.adapter, Path(cache), contact=contact),
                deps.investigation_input.assembly_accession,
                GeneChangeClaim(
                    label=deps.collection_member_id,
                    change=claimed,
                    request_text=deps.request.request_text,
                ),
            )
        event = deps.parent_activity.record(
            InvestigationGeneChangePlaced(
                attempt_id=deps.attempt_id,
                source_event_id=last[-1],
                collection_member_id=deps.collection_member_id,
                revision_number=deps.revision_number,
                tool_call_id=context.tool_call_id,
                gene_change=change,
                result=result,
                source_answers=shown_answers(answers),
            )
        )
        deps.gene_change_placements.append(event)
        return event

    return session.execute(
        context.tool_call_id,
        observed_source_operation(
            deps.parent_activity,
            deps.attempt_id,
            context.tool_call_id,
            "place_gene_change",
            typed_operation,
        ),
        ToolExecutionAdapter(
            identity="gene_change_placement",
            stage_result_event=lambda _result: deps.gene_change_placements[-1],
            # The bytes of the GET its result names, its last (as a lookup's).
            artifact_digests=lambda returned: (
                returned["payload"]["source_answers"][-1]["sha256"],
            ),
            stage_failure_event=lambda _error: (
                {
                    "event_id": deps.failure_source.event_id,
                    "event_type": deps.failure_source.event_type,
                }
                if deps.failure_source.event_type == "InvestigationSourceContactFailed"
                and deps.failure_source.event_id is not None
                else None
            ),
        ),
    )


def pick_gene_change_candidate(
    context: ToolContext[InvestigationPlannerDependencies],
    placement_event_id: Annotated[
        str,
        Field(
            description="The cite (placement_event_id) of the place_gene_change "
            "result of this member whose candidate you pick."
        ),
    ],
    contig: Annotated[str, Field(description="The candidate's contig, as listed.")],
    position: Annotated[int, Field(gt=0, description="The candidate's position.")],
    ref: Annotated[str, Field(description="The candidate's REF, as listed.")],
    alt: Annotated[str, Field(description="The candidate's ALT, as listed.")],
    reason: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="Why this candidate: what you read in the result (a "
            "transcript and its tags, the bases changed, NCBI's answer) or the "
            "researcher's words.",
        ),
    ],
) -> dict[str, Any]:
    """Pick one candidate SNV of this member's gene-change placement (#473 Q11): it
    becomes the member's placed position, read by locate_variant as any SNV."""
    deps = context.deps
    placement = _placement(deps, placement_event_id)
    fact = InvestigationGeneChangePicked(
        attempt_id=deps.attempt_id,
        collection_member_id=deps.collection_member_id,
        revision_number=deps.revision_number,
        placement_event_id=placement_event_id,
        gene_change=placement["payload"]["gene_change"],
        placed=PlacedPosition(contig=contig, position=position, ref=ref, alt=alt),
        reason=reason,
        after_answer_event_ids=tuple(
            sorted(
                answer["response_event_id"]
                for answer in deps.raw_answers
                if answer.get("action") in RESEARCHER_INSTRUCTION_ACTIONS
            )
        ),
    )
    event = None

    def record():
        nonlocal event
        event = deps.parent_activity.record(fact)
        deps.gene_change_picks.append(event["payload"])
        deps.researcher_answers = typed_answers(
            deps.request, deps.raw_answers, deps.target_reads, deps.gene_change_picks
        )
        return event

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        record,
        ToolExecutionAdapter(
            identity="gene_change_pick",
            stage_result_event=lambda _result: event,
            artifact_digests=lambda _result: (),
        ),
    )


def _placement(deps, event_id: str | None) -> dict[str, Any] | None:
    """This member's placement of the gene change it states now, by event id."""
    current = effective_gene_change(deps.request, deps.researcher_answers)
    return next(
        (
            event
            for event in deps.gene_change_placements
            if event["event_id"] == event_id
            and current is not None
            and event["payload"]["gene_change"] == current.model_dump(mode="json")
        ),
        None,
    )


def _gene_change_errors(
    deps: InvestigationPlannerDependencies,
    tool: str,
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    """A placement places the gene change the member states, when it states no site
    of its own, on a transcript a result of this member lists; a pick names a
    candidate that placement copied (#473 Q1 a, Q4 a, Q11; ADR 0049 membership)."""
    current = effective_gene_change(deps.request, deps.researcher_answers)
    if current is None:
        return ("investigation_gene_change_missing",)
    stated = effective_location(
        deps.request,
        typed_answers(deps.request, deps.raw_answers, deps.target_reads),
    )
    if stated.position is not None or stated.region is not None:
        # The member's own site is its location: a placement would not replace
        # it.
        return ("gene_change_site_stated",)
    if tool == "place_gene_change":
        transcript = arguments.get("transcript")
        listed = {
            read["accession"]
            for event in deps.gene_change_placements
            for read in event["payload"]["result"]["transcripts"]
        }
        if transcript is not None and transcript not in listed:
            return ("gene_change_transcript_not_listed",)
        return ()
    placement = _placement(deps, arguments.get("placement_event_id"))
    if placement is None:
        return ("gene_change_placement_unknown",)
    named = tuple(
        arguments.get(field) for field in ("contig", "position", "ref", "alt")
    )
    if named not in {
        (item["contig"], item["position"], item["ref"], item["alt"])
        for item in placement["payload"]["result"]["candidates"]
    }:
        return ("gene_change_candidate_not_listed",)
    return ()


# Every other Tool waits until the member's words are read (R1 b).
TARGET_WORDS_UNREAD = ToolPrecondition(
    "target_words_unread",
    "researcher words",
    "This member's words still need their target typed: call record_target_reading for each of {unread} first.",
    met_by=("record_target_reading",),
)

LOOKUP_TOOLS = frozenset({"fetch_refsnp", "fetch_eva_variant"})
"""The rsID lookup Tools, each its source's; every Run is offered both (#469)."""
INVESTIGATION_TOOLS = {
    "fetch_refsnp": fetch_refsnp,
    "fetch_eva_variant": fetch_eva_variant,
    "locate_variant": locate_variant,
    "search_sequence_names": search_sequence_names,
    "name_sequence": name_sequence,
    "name_version": name_version,
    "name_placement": name_placement,
    "quote_listed_none": quote_listed_none,
    "search_assemblies": search_assemblies,
    "fetch_assembly_report": fetch_assembly_report,
    "request_investigation_information": request_investigation_information,
    "record_target_reading": record_target_reading,
    "place_gene_change": place_gene_change,
    "pick_gene_change_candidate": pick_gene_change_candidate,
    # A member whose target a paper proposed reads that paper.
    "outline_paper": outline_paper,
    "grep_paper": grep_paper,
    "read_paper": read_paper,
    "filter_paper_rows": filter_paper_rows,
}
"""The role's whole Tool table, offered on every Run (#469)."""

MEMBER_HAS_NO_PAPER = ToolPrecondition(
    "member_has_no_paper",
    "call",
    "No paper proposed this member's target, so there is no paper to read; go on "
    "with this member's lookup.",
)


def validate_plan(
    context: ToolContext[InvestigationPlannerDependencies],
    output: InvestigationPlanDraft,
) -> InvestigationPlanDraft:
    dependencies = context.deps
    if unread := _unread_words(dependencies):
        # The member's words are read before its proposal (R1 b).
        if dependencies.structured_output_session is None:
            raise RuntimeError("Investigation output session is missing")
        message = (
            "Investigation output waits on the member's words: call "
            "record_target_reading for " + ", ".join(sorted(unread)) + " first"
        )
        dependencies.structured_output_session.validate(
            output.model_dump(mode="json"), ("target_words_unread",), repair=message
        )
        raise OutputRepair(message)
    try:
        refsnp, assembly = validate_investigation_output(
            dependencies.source_events,
            dependencies.collection_member_id,
            output,
            requested_rsids=researcher_rsids(
                dependencies.request, dependencies.researcher_answers
            ),
            stated_position=effective_location(
                dependencies.request, dependencies.researcher_answers
            ),
        )
    except InvestigationProposalRefused as error:
        if dependencies.structured_output_session is None:
            raise RuntimeError("Investigation output session is missing") from error
        message = "Investigation output is not source-grounded: " + "; ".join(
            (error.code, *error.details)
        )
        dependencies.structured_output_session.validate(
            output.model_dump(mode="json"),
            (error.code, *error.details),
            repair=message,
        )
        raise OutputRepair(message) from error
    if dependencies.structured_output_session is None:
        raise RuntimeError("Investigation output session is missing")
    valid = dependencies.structured_output_session.validate(
        output.model_dump(mode="json"), ()
    )
    dependencies.selected_sources = (refsnp, assembly)
    dependencies.validation_event_id = valid["event_id"]
    return output


INVESTIGATION_OUTPUT = AssignmentOutput(
    output_type=InvestigationPlanDraft,
    name="compile_proposal",
    description=(
        "Compile one exact source-backed Investigation proposal or one exact "
        "typed unresolved result."
    ),
    validator=validate_plan,
)


def assemble_investigation_planner(
    model: Any,
    *,
    instructions: RoleInstructions,
    policy: AgentInvocationPolicy,
    model_policy: ModelEndpointPolicy,
    journey: Any,
    attempt_id: str,
    parent_activity: Activity,
    failure_source: AttemptFailureSource,
    dependencies: Callable[[ObservedToolSession], InvestigationPlannerDependencies],
) -> AssembledAgent[InvestigationPlannerDependencies]:
    """Assemble one member's source-grounded planner through the shared Agent module,
    offering both lookup Tools on every Run (#469)."""
    assembled = None

    def request_context():
        return {
            "stage_findings": investigation_stage_findings(assembled.deps).model_dump(
                mode="json"
            )
        }

    assembled = assemble_agent(
        model,
        assignment="investigation_planner",
        instructions=instructions.text,
        context=request_context,
        tools=INVESTIGATION_TOOLS,
        stage=lambda _deps: "source_reading",
        bound_values=investigation_tool_values,
        admission=ActionAdmissionSpec(
            validator=investigation_action_errors,
            refusal_facts=investigation_refusal_facts,
            # The loop guard reads the item's one history.
            call_history=item_call_history,
        ),
        output=INVESTIGATION_OUTPUT,
        policy=policy,
        model_policy=model_policy,
        role_text=instructions,
        journey=journey,
        attempt_id=attempt_id,
        parent_activity=parent_activity,
        failure_source=failure_source,
        dependencies=lambda _admission, observed: dependencies(observed),
    )
    return assembled


# A member's lookup refusal about its own request: it names the call that places
# a stated position or region. A member whose words state no target is read and
# asked about first, so no lookup refuses for one.
_POSITION_NEEDS_LOCATION = ToolPrecondition(
    "stated_position_needs_location",
    "stated position",
    "A member whose request states a position or a region is looked up with "
    "`locate_variant()`, which reads a position's rsID and follows a dbSNP merge "
    "too.",
)


# What each source Tool shows the model of its recorded fact: the result's status,
# the source's statement, what it answered, the id a later call cites it by, and
# one next step. The next step lives only here, never in the fact.
_UNRESOLVED_OR_FOLLOW = (
    "Follow up on what the source's answer names with another lookup, or compile an "
    "unresolved proposal citing this lookup."
)
LOOKUP_VIEW = ModelView(
    next_step={
        "resolved": "Check its assembly with `fetch_assembly_report`, citing this "
        "lookup; then compile the proposal citing both.",
        "unlisted": "Check its assembly with `fetch_assembly_report`, citing this "
        "lookup; then compile the proposal citing both.",
        "region_located": "Compile the proposal citing this lookup; a region needs "
        "no assembly report.",
        "merged": "Look up the current rsID it was merged into with the lookup Tool "
        "you used.",
        "placement_unnamed": "Name one of its placements with `name_placement`, "
        "citing this lookup; then call the same lookup again.",
        "sequence_unnamed": "Name the sequence: find its record with "
        "`search_sequence_names`, then call `name_sequence` for each assembly the "
        "result names, citing this lookup and quoting the researcher's words that "
        "write the contig; then call `locate_variant` again.",
        "version_unnamed": "Name the assembly the stated version means (the Run's "
        "own, a row of `search_assemblies`, or one the record's placements name) "
        "with `name_version`, citing this lookup and quoting the researcher's words "
        "that state it; then call `locate_variant` again.",
        "source_answered": "Read the source's answer in source_answers. When it "
        "names a current record, look that up; when it places the requested target "
        "on another organism or assembly, ask the researcher with "
        "`request_investigation_information` (asks_about target), citing this "
        "lookup; when it says no rsID is listed at the position, quote its words "
        "with `quote_listed_none`, or, when the listing's answer has no body, cite "
        "its status_code there, then call `locate_variant` again; otherwise "
        "compile an unresolved proposal citing this lookup.",
        "*": _UNRESOLVED_OR_FOLLOW,
    },
    statement=lookup_statement,
    result=lambda payload: {
        key: value
        for key, value in payload["result"].items()
        # The origin's audit sentence is the statement.
        if key not in ("status", "conflicts", "origin")
    },
    cite="refsnp_event_id",
)
ASSEMBLY_REPORT_VIEW = ModelView(
    next_step={
        "compatible": "Compile the proposal citing this assembly check and the "
        "lookup it checked.",
        "*": "Compile an unresolved proposal citing this assembly check and the "
        "lookup it checked, or look up the assembly its answer names.",
    },
    statement=lookup_statement,
    result=lambda payload: {
        key: value
        for key, value in payload["result"].items()
        # The origin's audit sentence is the statement.
        if key not in ("status", "conflicts", "origin")
    },
    cite="assembly_event_id",
)
SEQUENCE_NAMES_VIEW = ModelView(
    next_step={
        "*": "Choose the record you judge the written contig means and call "
        "`locate_variant` again with `sequences`; read more rows with a later "
        "offset, or other text."
    },
    status=lambda payload: "read",
)
ASSEMBLY_SEARCH_VIEW = ModelView(
    next_step={
        "*": "Name the assembly the researcher means: call `locate_variant` again "
        "with `version` naming a listed row and quoting their words; read the next "
        "page with its next_page_token, or ask with "
        "`request_investigation_information` when NCBI's rows conflict with their "
        "words."
    },
    status=lambda payload: "answered",
)
TARGET_READING_VIEW = recorded_view(
    lambda _status, payload: (
        "Ask the researcher with `request_investigation_information` (asks_about "
        "target), citing the words you read."
        # A reading that states no target records none (the field is left out).
        if "target" not in payload
        else "The reading is this member's target now; look it up with the lookup "
        "Tool its target needs."
    )
)
DECISION_VIEW = recorded_view(
    lambda _status, payload: (
        "Call the same lookup again: it reads this placement."
        if payload["decision"]["kind"] == "placement"
        else "Call `locate_variant` again: it reads this decision."
    ),
    result=lambda payload: payload["decision"],
)
GENE_CHANGE_VIEW = ModelView(
    next_step={
        "candidates": "Pick one candidate with `pick_gene_change_candidate`, "
        "citing this placement and what you rest on.",
        "ambiguous": "Pick by the transcript the researcher's words or the copied "
        "tags support with `pick_gene_change_candidate`, or call "
        "`place_gene_change` again naming that transcript.",
        "*": "Ask the researcher with `request_investigation_information` "
        "(asks_about target), naming what the result shows.",
    },
    status=lambda payload: payload["result"]["outcome"] or "answered",
    cite="placement_event_id",
)


_DECISION_PRECONDITIONS = (
    ToolPrecondition(
        "decision_none_asked",
        "refsnp_event_id",
        "No lookup result of this member's current target asks for this decision.",
    ),
    ToolPrecondition(
        "decision_result_unknown",
        "refsnp_event_id",
        "No lookup result of this member has that cite. Cite the result that asks for "
        "this decision by its refsnp_event_id: {lookups}.",
    ),
    ToolPrecondition(
        "decision_result_replaced",
        "refsnp_event_id",
        "That result was for the target a researcher's answer has since replaced; "
        "look up the current target first.",
    ),
    ToolPrecondition(
        "decision_not_asked",
        "refsnp_event_id",
        "That result is {status}, which does not ask for this decision.",
    ),
)
_SEQUENCE_PRECONDITIONS = (
    ToolPrecondition(
        "sequence_assembly_not_asked",
        "assembly_accession",
        "Name an assembly that sequence_unnamed result names.",
    ),
    ToolPrecondition(
        "sequence_names_unread",
        "assembly_accession",
        "No lookup of this member read that assembly's sequence names.",
    ),
    ToolPrecondition(
        "sequence_quote_ungrounded",
        "quote",
        "The quote must be the researcher's exact words, holding the written contig.",
    ),
    ToolPrecondition(
        "sequence_record_not_listed",
        "record",
        "Name a record that assembly's sequence names list, by its GenBank or RefSeq "
        "accession; search them with search_sequence_names.",
        met_by=("search_sequence_names",),
    ),
)
_VERSION_PRECONDITIONS = (
    ToolPrecondition(
        "version_not_stated",
        "assembly_accession",
        "This member's request states no genome version, so there is none to name.",
    ),
    ToolPrecondition(
        "version_not_listed",
        "assembly_accession",
        "Name the Run's own assembly, an assembly a search_assemblies page of "
        "this member lists, or one a dbSNP record's placements name; search "
        "NCBI's genome reports first.",
        met_by=("search_assemblies",),
    ),
    ToolPrecondition(
        "version_quote_ungrounded",
        "quote",
        "The version's quote must be the researcher's exact words in the line "
        "that states the genome version.",
    ),
)


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "fetch_refsnp",
        "investigation_planner",
        ("rsid: str",),
        "RefSnpToolResult",
        "ncbi_investigation",
        "RefSnpToolResultRecorded",
        "InvestigationSourceContactFailed",
        "error_type",
        LOOKUP_VIEW,
        "investigation_source_failure_retained",
        source_surface=SourceSurface(),
        stages=("source_reading",),
        # Steering metadata, not a route: the model chooses.
        source_tag="source: NCBI dbSNP · human variants; dbSNP keeps no non-human record",
        # A placement_unnamed result's placement is named with name_placement and
        # read by the next call.
        examples=('fetch_refsnp(rsid="<the rsID the researcher stated>")',),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "fetch_refsnp_query_not_requested",
                "rsid",
                "Look up only an rsID this member's researcher stated (in the "
                "request or a retained answer), a current rsID one of this member's "
                "merged results names in merged_into: {allowed}, or an rsID one of "
                "this member's lookup results' source_answers names as an id.",
                met_by=("fetch_refsnp",),
            ),
            _POSITION_NEEDS_LOCATION,
        ),
        negative_results=(
            NegativeResult(
                "RefSnpToolResultRecorded",
                "status",
                (
                    "merged",
                    "ambiguous",
                    "conflicted",
                    "incompatible",
                    "not_found",
                    "malformed",
                    "placement_unnamed",
                    "source_answered",
                ),
            ),
        ),
    ),
    function_tool_contract(
        "fetch_eva_variant",
        "investigation_planner",
        ("rsid: str", "contig_naming: str | None"),
        "RefSnpToolResult",
        "eva_investigation",
        "RefSnpToolResultRecorded",
        "InvestigationSourceContactFailed",
        "error_type",
        LOOKUP_VIEW,
        "investigation_source_failure_retained",
        # A non-human Run's rsID lookup: one Tool whose EVA GETs each run on their
        # own operation's surface.
        source_surface=SourceSurface(
            frozenset(
                {
                    "eva_clustered_variant",
                    "fetch_assembly_report",
                    "eva_contig_alias",
                    "eva_submitted_variants",
                }
            )
        ),
        stages=("source_reading",),
        source_tag="source: EVA (European Variation Archive) · non-human variants; "
        "EVA keeps no human record",
        examples=('fetch_eva_variant(rsid="<the rsID the researcher stated>")',),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "eva_query_not_requested",
                "rsid",
                "Look up only an rsID this member's researcher stated (in the "
                "request or a retained answer), a current rsID one of this member's "
                "merged results names in merged_into: {allowed}, or an rsID one of "
                "this member's lookup results' source_answers names as an id.",
            ),
            _POSITION_NEEDS_LOCATION,
        ),
        negative_results=(
            NegativeResult(
                "RefSnpToolResultRecorded",
                "status",
                (
                    "ambiguous",
                    "conflicted",
                    "incompatible",
                    "not_found",
                    "malformed",
                    "placement_unnamed",
                    "source_answered",
                ),
            ),
        ),
    ),
    function_tool_contract(
        "locate_variant",
        "investigation_planner",
        (
            "source: VariantSource | None",
            "variant_type: EvaVariantType | None",
            "contig_naming: EvaContigNaming | None",
        ),
        "RefSnpToolResult",
        "variant_location",
        "RefSnpToolResultRecorded",
        "InvestigationSourceContactFailed",
        "error_type",
        LOOKUP_VIEW,
        "investigation_source_failure_retained",
        # A stated position's lookup: one Tool whose GETs each run on their own
        # operation's surface, NCBI's three placement GETs among them, and the two that
        # move a position stated on another version onto the Run's.
        source_surface=SourceSurface(
            frozenset(
                {
                    "fetch_assembly_report",
                    "ncbi_sequence_names",
                    "ncbi_reference_span",
                    "ncbi_spdi_rsids",
                    "ncbi_taxon_assemblies",
                    "ncbi_spdi_equivalents",
                    "fetch_refsnp",
                    "eva_clustered_variants_at",
                    "eva_clustered_variant",
                    "eva_contig_alias",
                    "eva_submitted_variants",
                }
            )
        ),
        stages=("source_reading",),
        source_tag="source: NCBI sequence names and reference bases · the variant "
        "source you name (dbSNP: human variants; EVA: non-human variants)",
        # The decisions a result asks for are their own Tools' facts, read by the
        # next call.
        examples=('locate_variant(source="<ncbi_dbsnp or eva>")',),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "investigation_position_missing",
                "stated position",
                "This member's request states no position or region, so "
                "locate_variant does not apply; {next}",
            ),
            ToolPrecondition(
                "locate_variant_source_missing",
                "source",
                "A stated position's rsIDs are listed by the variant source you name: "
                'source="ncbi_dbsnp" (dbSNP: human variants) or source="eva" (EVA: '
                "non-human variants).",
            ),
            ToolPrecondition(
                "region_names_no_source",
                "source",
                "A region lists no rsID: locate_variant reads only NCBI's sequence "
                "names and reference bases over it, so send no source, variant_type "
                "or contig_naming.",
            ),
            ToolPrecondition(
                "eva_option_without_eva",
                "EVA option",
                "variant_type and contig_naming are EVA's own options for its GETs; "
                'send them only with source="eva".',
            ),
        ),
        negative_results=(
            NegativeResult(
                "RefSnpToolResultRecorded",
                "status",
                (
                    "ambiguous",
                    "conflicted",
                    "incompatible",
                    "not_found",
                    "malformed",
                    "sequence_unnamed",
                    "version_unnamed",
                    "source_answered",
                ),
            ),
        ),
    ),
    # The decisions a lookup result asks for, each its own Tool and the one
    # fact: no GET; the next lookup reads it.
    *(
        function_tool_contract(
            name,
            "investigation_planner",
            arguments,
            "InvestigationLookupDecided",
            "lookup_decision",
            "InvestigationLookupDecided",
            None,
            None,
            DECISION_VIEW,
            "generic_mechanical_failure_only",
            stages=("source_reading",),
            source_tag="no source: records your decision for the lookup result you cite",
            examples=(example,),
            preconditions=(TARGET_WORDS_UNREAD, *_DECISION_PRECONDITIONS, *own),
        )
        for name, arguments, example, own in (
            (
                "name_sequence",
                (
                    "refsnp_event_id: str",
                    "assembly_accession: str",
                    "record: str",
                    "quote: str",
                ),
                'name_sequence(refsnp_event_id="<the sequence_unnamed result\'s cite>", '
                'assembly_accession="<an assembly it names>", record="<a GenBank or '
                'RefSeq accession search_sequence_names lists>", quote="<the '
                "researcher's exact words that write the contig>\")",
                _SEQUENCE_PRECONDITIONS,
            ),
            (
                "name_version",
                ("refsnp_event_id: str", "assembly_accession: str", "quote: str"),
                'name_version(refsnp_event_id="<the version_unnamed result\'s cite>", '
                'assembly_accession="<the assembly the stated version means>", '
                'quote="<the researcher\'s exact words that state the version>")',
                _VERSION_PRECONDITIONS,
            ),
            (
                "name_placement",
                ("refsnp_event_id: str", "placement: str"),
                "name_placement(refsnp_event_id=\"<the placement_unnamed result's "
                'cite>", placement="<the run_contig of the placement you name>")',
                (
                    ToolPrecondition(
                        "placement_not_listed",
                        "placement",
                        "Name the run_contig of one of that result's placements on "
                        "the Run's assembly.",
                    ),
                ),
            ),
            (
                "quote_listed_none",
                ("refsnp_event_id: str", "quote: str", "status_code: int"),
                'quote_listed_none(refsnp_event_id="<the source_answered position '
                "result's cite>\", quote=\"<the listing's own words saying no rsID "
                'is listed there>")',
                (
                    ToolPrecondition(
                        "listed_none_quote_ungrounded",
                        "quote",
                        "Quote the listing service's own words, as its answer in that "
                        "result's source_answers writes them, saying no rsID is listed "
                        "there.",
                    ),
                    ToolPrecondition(
                        "listed_none_basis_missing",
                        "quote",
                        "Send either quote (the listing's own words) or, only when its "
                        "answer has no body, status_code; never both.",
                    ),
                    ToolPrecondition(
                        "listed_none_status_not_bodyless",
                        "status_code",
                        "Cite status_code only for a listing answer with no body, as "
                        "the HTTP status that result's source_answers shows; an answer "
                        "with words is quoted.",
                    ),
                ),
            ),
        )
    ),
    # A search of the sequence names a lookup retained: no GET.
    function_tool_contract(
        "search_sequence_names",
        "investigation_planner",
        ("assembly_accession: str", "text: str", "offset: int"),
        "SequenceNamesPage",
        "sequence_names_read",
        "SequenceNamesRead",
        None,
        None,
        SEQUENCE_NAMES_VIEW,
        "generic_mechanical_failure_only",
        stages=("source_reading",),
        source_tag="source: NCBI's assembly report, as a locate_variant result retained it",
        examples=(
            'search_sequence_names(assembly_accession="<an assembly a locate_variant '
            'result read>", text="<part of the written contig>")',
        ),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "sequence_names_unread",
                "assembly_accession",
                "No locate_variant result of this member read that assembly's sequence "
                "names; search an assembly a sequence_unnamed result names.",
            ),
        ),
    ),
    # NCBI's genome reports for a stated version: one GET of NCBI's own query,
    # every row copied, for the model to name the version it means.
    function_tool_contract(
        "search_assemblies",
        "investigation_planner",
        (
            "reference_only, assembly_source, has_annotation, exclude_paired_reports, "
            "exclude_atypical, assembly_version, assembly_level, first_release_date, "
            "last_release_date, search_text, is_metagenome_derived, is_type_material, "
            "is_ictv_exemplar, exclude_multi_isolate, type_material_category, "
            "tax_exact_match, sort_field, sort_direction: NCBI's own options",
            "page_size: int (1-100, default 100)",
            "page_token: str | None",
        ),
        "AssemblySearchPage",
        "ncbi_investigation",
        "AssembliesSearched",
        "InvestigationSourceContactFailed",
        "error_type",
        ASSEMBLY_SEARCH_VIEW,
        "investigation_source_failure_retained",
        source_surface=SourceSurface(frozenset({"ncbi_taxon_assemblies"})),
        stages=("source_reading",),
        source_tag="source: NCBI Datasets · a taxon's genome reports",
        examples=(
            'search_assemblies(search_text=["<the stated version as written>"], '
            'assembly_version="all_assemblies")',
        ),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "run_taxon_unknown",
                "taxid",
                "This Run's Journey records no taxon for its reference, so NCBI's "
                "genome reports cannot be listed for its organism.",
            ),
        ),
    ),
    function_tool_contract(
        "request_investigation_information",
        "investigation_planner",
        (
            "question: str",
            "missing_information: str",
            "asks_about: str",
            "refsnp_event_id: str | None",
        ),
        "InvestigationReviewRequested",
        "investigation_review",
        "InvestigationReviewRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("source_reading",),
        examples=(
            'request_investigation_information(question="Which allele?", missing_information="The request does not specify an allele.", asks_about="target")',
            # A source placing the requested rsID elsewhere is asked, citing its
            # result; the value is a shape, never another organism's.
            'request_investigation_information(question="<which target the researcher means>", missing_information="<what the source\'s answer leaves open>", asks_about="target", refsnp_event_id="<the cite of the lookup that placed it elsewhere>")',
        ),
        opens_question=True,
        preconditions=(
            TARGET_WORDS_UNREAD,
            # The calls after an opened question are refused with this reason, as
            # planning's are.
            QUESTION_OPENED,
            ToolPrecondition(
                "investigation_question_not_needed",
                "member state",
                "No researcher question is needed: {reason}",
            ),
            ToolPrecondition(
                "question_cite_unknown",
                "refsnp_event_id",
                "No lookup result of this member has that cite. Cite one of this "
                "member's lookup results by its refsnp_event_id, or omit it: {lookups}.",
            ),
        ),
    ),
    # The model types the target a member's words state (R1 b); no GET.
    function_tool_contract(
        "record_target_reading",
        "investigation_planner",
        (
            "source_event_id: str",
            "quote: str | None",
            "rsid: str | None",
            "site: SiteDraft | None",
            "stated_version: str | None",
            "reason: str | None",
        ),
        "InvestigationTargetRead",
        "target_reading",
        "InvestigationTargetRead",
        None,
        None,
        TARGET_READING_VIEW,
        "generic_mechanical_failure_only",
        stages=("source_reading",),
        examples=(
            'record_target_reading(source_event_id="<the answer\'s event id>", '
            'quote="<the words that state the rsID>", rsid="<that rsID>")',
            'record_target_reading(source_event_id="<the answer\'s event id>", '
            'quote="V:2129317 C>T on WBcel235", site={"contig": "V", '
            '"position": 2129317, "ref": "C", "alt": "T"}, stated_version="WBcel235")',
            'record_target_reading(source_event_id="<the answer\'s event id>", '
            'reason="The answer asks to check the gene again and names no variant.")',
        ),
        preconditions=(
            QUESTION_OPENED,
            ToolPrecondition(
                "target_words_all_read",
                "words to read",
                "Every word of this member's request and of the researcher's answers is read already; no words wait for a reading.",
            ),
            ToolPrecondition(
                "target_words_read",
                "source_event_id",
                "Those words are read already, or are not this member's; read the words still to read: {unread}.",
            ),
            ToolPrecondition(
                "target_reading_reason_missing",
                "reason",
                "A reading with no rsID or site says why the words state no target, citing them.",
            ),
            ToolPrecondition(
                "target_quote_ungrounded",
                "quote",
                "The quote must copy the words you read exactly, on whole words: the span that states the target.",
            ),
            ToolPrecondition(
                "target_rsid_not_in_quote",
                "rsid",
                "The rsID must be one the quote states, typed as rs and digits.",
            ),
            ToolPrecondition(
                "target_site_not_in_quote",
                "site",
                "The site must be one the quote writes: its sequence as written, its position as a number it writes, and REF and ALT as written bases.",
            ),
            ToolPrecondition(
                "target_version_without_site",
                "stated_version",
                "A genome version belongs to a site the quote writes; type the site, or leave the version out.",
            ),
        ),
    ),
    # G15 on the member's typed gene change (Q3 a): the Run's own annotation,
    # the codon's bases and NCBI's HGVS answer, every GET on its own operation's
    # surface, every answer copied.
    function_tool_contract(
        "place_gene_change",
        "investigation_planner",
        ("transcript: str | None",),
        "GeneChangeAnswer",
        "gene_change_placement",
        "InvestigationGeneChangePlaced",
        "InvestigationSourceContactFailed",
        "error_type",
        GENE_CHANGE_VIEW,
        "investigation_source_failure_retained",
        source_surface=SourceSurface(
            frozenset(
                {
                    "fetch_assembly_report",
                    "ncbi_sequence_names",
                    "ncbi_genome_annotation",
                    "ncbi_reference_span",
                    "ncbi_hgvs_contextuals",
                    "ncbi_spdi_equivalents",
                }
            )
        ),
        stages=("source_reading",),
        source_tag="source: the Run assembly's own annotation and reference bases · "
        "NCBI's HGVS service when a transcript is named (it answers human sequences "
        "only)",
        examples=(
            "place_gene_change()",
            'place_gene_change(transcript="<a transcript accession a previous result '
            'lists>")',
        ),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "investigation_gene_change_missing",
                "gene change",
                "This member states no typed gene change, so place_gene_change does "
                "not apply; {next}",
            ),
            ToolPrecondition(
                "gene_change_site_stated",
                "gene change",
                "This member states its own site, which locate_variant reads; a "
                "gene change is placed only for a member that states no site.",
            ),
            ToolPrecondition(
                "gene_change_transcript_not_listed",
                "transcript",
                "Name a transcript a place_gene_change result of this member lists, "
                "or omit transcript to read every transcript of the gene.",
                met_by=("place_gene_change",),
            ),
        ),
    ),
    # The model's pick of a placement's candidate: no GET.
    function_tool_contract(
        "pick_gene_change_candidate",
        "investigation_planner",
        (
            "placement_event_id: str",
            "contig: str",
            "position: int",
            "ref: str",
            "alt: str",
            "reason: str",
        ),
        "PlacedPosition",
        "gene_change_pick",
        "InvestigationGeneChangePicked",
        None,
        None,
        recorded_view({"*": "Look the picked position up with `locate_variant`."}),
        "generic_mechanical_failure_only",
        stages=("source_reading",),
        examples=(
            'pick_gene_change_candidate(placement_event_id="<a place_gene_change '
            'result\'s cite>", contig="<a candidate\'s contig>", '
            'position=<its position>, ref="<its REF>", alt="<its ALT>", '
            'reason="<what in the result or the researcher\'s words you rest on>")',
        ),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "investigation_gene_change_missing",
                "gene change",
                "This member states no typed gene change, so there is no candidate "
                "to pick; {next}",
            ),
            ToolPrecondition(
                "gene_change_site_stated",
                "gene change",
                "This member states its own site, which locate_variant reads; a "
                "gene change is placed only for a member that states no site.",
            ),
            ToolPrecondition(
                "gene_change_placement_unknown",
                "placement_event_id",
                "Cite a place_gene_change result of this member for the gene change "
                "it states now: its cite's placement_event_id.",
                met_by=("place_gene_change",),
            ),
            ToolPrecondition(
                "gene_change_candidate_not_listed",
                "candidate",
                "Name one candidate that placement lists, with its contig, position, "
                "REF and ALT exactly as listed.",
            ),
        ),
    ),
    function_tool_contract(
        "fetch_assembly_report",
        "investigation_planner",
        ("refsnp_event_id: str", "assembly_accession: str"),
        "AssemblyReportToolResult",
        "ncbi_investigation",
        "AssemblyReportToolResultRecorded",
        "InvestigationSourceContactFailed",
        "error_type",
        ASSEMBLY_REPORT_VIEW,
        "investigation_source_failure_retained",
        source_surface=SourceSurface(),
        stages=("source_reading",),
        source_tag="source: NCBI Datasets · any assembly",
        # The example cites its lookup's event id, which the call requires, and
        # names where the accession comes from: no organism's value, so no Run is
        # shown another's.
        examples=(
            'fetch_assembly_report(refsnp_event_id="<the event id of your lookup result>", assembly_accession="<the versioned assembly accession of that result\'s candidate>")',
        ),
        preconditions=(
            TARGET_WORDS_UNREAD,
            ToolPrecondition(
                "fetch_assembly_report_requires_typed_refsnp_result",
                "refsnp_event_id",
                "Call fetch_refsnp first (fetch_eva_variant on a non-human Run, locate_variant for a stated position) and read its typed result before looking up the assembly.",
                met_by=("fetch_refsnp", "fetch_eva_variant", "locate_variant"),
            ),
            ToolPrecondition(
                "fetch_assembly_report_refsnp_event_unknown",
                "refsnp_event_id",
                "No lookup result of this member has that event id. Send the event id of one of this member's lookup results, not its result_id: {lookups}.",
            ),
            ToolPrecondition(
                "fetch_assembly_report_requires_resolved_refsnp_result",
                "refsnp_event_id",
                "That lookup result is {status}, so no assembly report checks it. {next}",
            ),
        ),
        negative_results=(
            NegativeResult(
                "AssemblyReportToolResultRecorded",
                "status",
                ("conflicted", "incompatible", "not_found", "malformed"),
            ),
        ),
    ),
    # The member's paper, read with the Paper Reader's four Tools.
    *reading_tool_rows(
        "investigation_planner",
        "source_reading",
        then="go on with this member's lookup",
        preconditions=(MEMBER_HAS_NO_PAPER,),
    ),
)


__all__ = [
    "TOOL_ROWS",
    "InvestigationPlannerDependencies",
    "ask_about_finding_issue",
    "ask_about_retrieved_target",
    "assemble_investigation_planner",
]
