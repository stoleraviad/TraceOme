"""The reference and companion stages' Tools: one recorded observation per call.

The model chooses each taxonomy term, TaxId, filter, accession and companion asset
(ADR 0040 "Lookup arguments"). Admission checks their shape, asks the source port
whether it can send a term as one query (ADR 0020), and refuses a premature call
with its registry reason before contact. Every result, negative or not, returns to
the model as the Tool's result; an operational source fault keeps its terminal
route (ADR 0031). Replay reads each call against its own recorded fact (ADR 0034).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from pydantic import Field

from bio_neb.agentic_journey.action_policy import ChoiceReason, PreconditionUnmet
from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.planning_preconditions import (
    COMPANION_WORK_COMPLETE,
    COMPANIONS_NOT_OPEN,
    CORRECTION_PENDING,
    QUESTION_PENDING,
    REFERENCE_NOT_OPEN,
)
from bio_neb.agentic_journey.source_answers import collected_answers, shown_answers
from bio_neb.agentic_journey.source_contacts import (
    observed_source_operation,
    source_contact_stage,
)
from bio_neb.agentic_journey.tool_execution import (
    FunctionToolContract,
    ModelView,
    NegativeResult,
    ToolExecutionAdapter,
    ToolPrecondition,
    function_tool_contract,
    recorded_view,
)
from bio_neb.application.planning_read import planning_evidence
from bio_neb.providers.source_operations import SourceSurface
from bio_neb.reference_discovery.prebuilt_index_tool import (
    GALAXY_OPERATION,
    NCBI_OPERATION,
    PrebuiltIndexesObserved,
    PrebuiltRepository,
    ReadPrebuiltIndexInput,
    SearchPrebuiltIndexesInput,
    read_candidate,
    search_catalogues,
)
from bio_neb.reference_discovery.reference_action_policy import (
    answered_choice_receipt,
    applicable_reference_receipt,
    current_choices,
    question_choice,
    question_page,
    reference_assertion,
    reference_discovery,
    reference_grounding,
    reference_projection,
    reference_requirement,
    selected_reference,
)
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionRole,
    DescribeCompanionInput,
)
from bio_neb.reference_discovery.reference_facts_journey import (
    AssemblyWithIndexSuggested,
    CompanionObservationRecorded,
    PrebuiltIndexChosen,
    PrebuiltIndexesListed,
    PrebuiltIndexRead,
    ReferenceFactsProjection,
    SuggestAssemblyInput,
    SuggestedAssembly,
    project_reference_facts,
    validate_prebuilt_index_fact,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    ASSEMBLY_ACCESSION_RULE,
    SAMPLE_IDENTIFIER_RULE,
    AssemblyLevelOption,
    AssemblyQuery,
    AssemblySourceOption,
    AssemblyVersionOption,
    DescribeReferenceInput,
    DiscoverReferenceAssembliesInput,
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
    ResolveTaxonInput,
    SampleAttestationInput,
    SearchTextOption,
    SortDirectionOption,
    SortFieldOption,
    SuppliedAssemblyEvidence,
    TaxExactMatchOption,
    TypeMaterialCategoryOption,
    TypeMaterialOption,
    assembly_query,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceAssembliesRecorded,
    ReferenceChoiceRecorded,
    ReferenceDescriptionRecorded,
    ReferenceSourceRetrieved,
    SampleAttestationRecorded,
    TaxonResolutionRecorded,
    paper_quote_event,
    quoted,
    researcher_text,
    researcher_words,
    statement_texts,
)

# The named choice that no discovered candidate fits.
NO_FITTING_CANDIDATE = "none"

REFERENCE_TOOLS = (
    "attest_sample",
    "resolve_taxon",
    "discover_assemblies",
    "record_supplied_reference",
    "inspect_reference",
)
# The genomic FASTA's Galaxy index Tools: the model searches, reads and chooses.
PREBUILT_INDEX_TOOLS = (
    "search_prebuilt_indexes",
    "read_prebuilt_index",
    "record_prebuilt_index_choice",
    "suggest_assembly_with_index",
)
COMPANION_TOOLS = (
    "describe_companion",
    "request_companion_asset_selection",
    "request_companion_refinement",
    *PREBUILT_INDEX_TOOLS,
)
# The Tools whose one recorded fact replay reads back against the admitted call.
RECORDED_TOOLS = (
    "attest_sample",
    "resolve_taxon",
    "discover_assemblies",
    "inspect_reference",
    "describe_companion",
    "search_prebuilt_indexes",
    "read_prebuilt_index",
)

_BOUND_NOTHING = SimpleNamespace(constants={}, enums={})


def _bound(**constants: Any) -> SimpleNamespace:
    return SimpleNamespace(constants=constants, enums={})


def _standing_choice(view):
    """The current concrete reference choice, until a later researcher answer."""
    choices = current_choices(view)
    if not choices:
        return None
    choice = choices[-1]
    if choice.assembly_accession is None:
        return None
    receipt = applicable_reference_receipt(view) or answered_choice_receipt(view)
    if (
        receipt is not None
        and receipt.response_event_id not in choice.grounding.source_event_ids
    ):
        return None
    return choice


def _index_work_open(deps) -> bool:
    """Whether the verified assembly's index work waits for a choice."""
    work = reference_facts_projection(deps).prebuilt_index
    return work is not None and work.choice is None


def _no_choice_yet(view) -> None:
    if view.verified_reference is not None:
        raise PreconditionUnmet(
            "reference_already_verified", view.verified_reference.assembly_accession
        )
    choice = _standing_choice(view)
    if choice is not None:
        raise PreconditionUnmet("reference_choice_stands", choice.assembly_accession)


def _taxon_source(deps, view) -> str | None:
    """The statement the taxon is resolved from: the researcher's latest assertion
    (their words come first), else the reading fact that keeps the organism a cited
    paper writes for the taxon value (#383 A2); None when neither states one, and the
    stage's taxon question opens."""
    assertion = reference_assertion(deps, view, "taxon_identity")
    if assertion is not None:
        return assertion.source_event_id
    return _paper_organism(deps, view)


def _paper_organism(deps, view) -> str | None:
    """The reading fact that keeps the organism a cited paper writes for the taxon
    value, if a reading of a paper that value cites kept one (#383 A2)."""
    from bio_neb.application.planning_read import current_planning
    from bio_neb.paper_reading.planning import answered_readings

    requirement = reference_requirement(deps, view, "taxon_identity")
    return next(
        (
            reading.plan_event_id
            for reading in answered_readings(
                current_planning(deps.journey), requirement
            )
            if reading.plan is not None and reading.plan.organism is not None
        ),
        None,
    )


def _paper_versions(deps, view) -> tuple[str, ...]:
    """The reading facts that keep the genome version a cited paper states for the
    reference value, which a supplied accession may cite (#383 A2)."""
    from bio_neb.application.planning_read import current_planning
    from bio_neb.paper_reading.planning import answered_readings

    requirement = reference_requirement(deps, view, "reference_assembly_identity")
    return tuple(
        reading.plan_event_id
        for reading in answered_readings(current_planning(deps.journey), requirement)
        if reading.plan is not None and reading.plan.reference is not None
    )


def _taxon_facts(view):
    return tuple(f for f in view.facts if f.event_type == "TaxonResolutionRecorded")


def _taxon_outcome(view) -> str | None:
    facts = _taxon_facts(view)
    if not facts:
        return None
    result = facts[-1].result
    return result.get("resolution", result.get("status"))


def _inspectable(view):
    """The reference an inspection checks now: a recorded choice or answer."""
    selected = selected_reference(view)
    if selected is None or selected.selection_event_id is None:
        raise PreconditionUnmet("reference_choice_absent")
    if isinstance(selected, SuppliedAssemblyEvidence):
        if view.sample_event_id is None:
            raise PreconditionUnmet("sample_unattested")
        if view.resolved_taxon is None:
            raise PreconditionUnmet("taxon_unresolved", _taxon_outcome(view))
    return selected


def reference_question(deps, view) -> tuple[str, str]:
    """The reference requirement that needs the researcher now, with its evidence.

    Raises the reason no reference question is needed. The taxon question opens
    on the latest taxonomy result when it resolved no taxon; the model decides
    when its own terms are exhausted.
    """
    if view.status == "reference_identity":
        raise PreconditionUnmet("reference_identity_resolved", view.status)
    if _sample_open(deps, view):
        return "sample_identity", view.plan_event_id
    if _taxon_source(deps, view) is None:
        return "taxon_identity", view.plan_event_id
    if view.resolved_taxon is None:
        facts = _taxon_facts(view)
        if not facts:
            raise _question_not_needed(deps, view)
        return "taxon_identity", facts[-1].source_event_id
    choice = _standing_choice(view)
    if (
        choice is not None
        and choice.discovery_event_id is None
        and view.description_event_id is not None
    ):
        # A supplied accession's inspection did not verify it; the researcher
        # clarifies the accession (there is no discovery to choose from).
        return "reference_assembly_identity", view.description_event_id
    if view.status != "reference_candidates":
        raise _question_not_needed(deps, view)
    choice = question_choice(view)
    if choice is None and applicable_reference_receipt(view):
        # The researcher answered a question on the latest page; another page or
        # a new choice opens the next question.
        raise _question_not_needed(deps, view)
    if (
        choice is not None
        and choice.assembly_accession is not None
        and view.description_event_id is None
    ):
        # A concrete choice is inspected first; only an unresolved result asks.
        raise PreconditionUnmet(
            "reference_choice_uninspected", choice.assembly_accession
        )
    # The model opens the question on a discovery page whenever what it read
    # conflicts with the researcher's words: code judges no page.
    return "reference_assembly_identity", question_page(view)


def unasked_suggestion(deps, view) -> str | None:
    """The event id of the verified assembly's latest suggestion of another assembly
    that no question presents yet (#459 D4), else None."""
    if view.verified_reference is None:
        return None
    work = reference_facts_projection(deps).prebuilt_index
    if work is None or not work.suggestion_event_ids:
        return None
    latest = work.suggestion_event_ids[-1]
    asked = any(
        event["event_type"] == "PlanningHitlRequested"
        and latest in event["payload"]["evidence_event_ids"]
        for event in planning_evidence(deps.journey)
    )
    return None if asked else latest


def _sample_open(deps, view) -> bool:
    """Whether the sample may still be asked about: no sample is attested, and the
    researcher stated none, answered a question (the model reads whether the answer
    names one), or attestation refused what they stated (#472 Q2 a, ADR 0031: a dead
    end opens the stage's Review). No code reads the words."""
    if view.sample_event_id is not None:
        return False
    sample = reference_assertion(deps, view, "sample_identity")
    return (
        sample is None
        or sample.assertion_kind == "planning_hitl"
        or any(fact.event_type == "SampleAttestationRecorded" for fact in view.facts)
    )


def _question_not_needed(deps, view) -> PreconditionUnmet:
    return PreconditionUnmet(
        "reference_question_not_needed",
        view.status,
        lambda: {"next": _next_reference_step(deps, view, "request_planning_hitl")},
    )


def _next_reference_step(deps, view, refused: str) -> str:
    """The reference Tools callable now, read by the same checks admission
    runs (#409); none once the reference is verified."""
    if view.verified_reference is not None:
        return "The reference is verified; the reference work is done."
    tools = []
    for name, values in _REFERENCE_VALUES.items():
        if name in {refused, "request_planning_hitl"}:
            continue
        try:
            values(deps, view)
        except PreconditionUnmet:
            continue
        tools.append(f"`{name}`")
    if not tools:
        return "Continue the reference work."
    return f"Continue the reference work with {', '.join(tools)}."


def question_subject(deps, view) -> tuple[str, str]:
    """What the planning question asks about now, with its evidence: a cited paper's
    open search first, as the researcher confirms the paper before it is downloaded
    (owner, #420 F2 b; the paper stage, #420 D1 b), else the reference question."""
    from bio_neb.paper_intake.tools import paper_question

    paper = paper_question(deps)
    if paper is not None:
        return "paper", paper[1].source_event_id
    return reference_question(deps, view)


def _question_values(deps, view):
    from bio_neb.paper_intake.tools import paper_question

    paper = paper_question(deps)
    if paper is not None:
        # The paper question is the cited paper's own.
        requirement, search = paper
        return _bound(
            requirement_id=requirement.requirement_id,
            evidence_event_id=search.source_event_id,
        )
    kind, evidence_event_id = reference_question(deps, view)
    return _bound(
        requirement_id=reference_requirement(deps, view, kind).requirement_id,
        evidence_event_id=evidence_event_id,
    )


def _attest_values(deps, view):
    if view.sample_event_id is not None:
        raise PreconditionUnmet("sample_attested", view.sample_event_id)
    if reference_assertion(deps, view, "sample_identity") is None:
        raise PreconditionUnmet("sample_assertion_missing")
    return _BOUND_NOTHING


def _sample_errors(deps, arguments) -> tuple[str, ...]:
    """The sample name the model typed stands in the researcher's latest statement
    about the sample (#472 Q2 a)."""
    sample = reference_assertion(deps, reference_projection(deps), "sample_identity")
    return _statement_errors(
        deps,
        sample.source_event_id,
        arguments["quote"],
        SAMPLE_IDENTIFIER_RULE,
        arguments["sample_identifier"],
        ("sample_quote_ungrounded", "sample_not_in_quote"),
    )


def _resolve_values(deps, view):
    choice = _standing_choice(view)
    # A supplied accession still needs its taxon resolved before inspection.
    if not (
        choice is not None
        and choice.discovery_event_id is None
        and view.resolved_taxon is None
    ):
        _no_choice_yet(view)
    if view.sample_event_id is None:
        raise PreconditionUnmet("sample_unattested")
    if _taxon_source(deps, view) is None:
        raise PreconditionUnmet("taxon_assertion_missing")
    return _BOUND_NOTHING


def _discover_values(deps, view):
    if view.verified_reference is not None and _index_work_open(deps):
        # The organism's other assemblies, listed during the index work while no
        # index is chosen: one may hold the fitting Galaxy index; the verified
        # reference stays.
        return _BOUND_NOTHING
    _no_choice_yet(view)
    if view.resolved_taxon is None:
        raise PreconditionUnmet("taxon_unresolved", _taxon_outcome(view))
    receipt = applicable_reference_receipt(view)
    if receipt is not None and receipt.assertion.selected_candidate_id is not None:
        raise PreconditionUnmet(
            "researcher_selected_candidate", receipt.assertion.selected_candidate_id
        )
    return _BOUND_NOTHING


def _choice_values(deps, view):
    if view.status == "reference_identity":
        raise PreconditionUnmet("reference_identity_resolved", view.status)
    if view.status != "reference_candidates":
        # Before a discovery the status names earlier work, not a discovery.
        raise PreconditionUnmet("reference_discovery_not_open")
    choices = current_choices(view)
    receipt = applicable_reference_receipt(view) or answered_choice_receipt(view)
    if choices and (
        receipt is None
        or receipt.response_event_id in choices[-1].grounding.source_event_ids
    ):
        raise PreconditionUnmet(
            "reference_choice_stands",
            choices[-1].assembly_accession or NO_FITTING_CANDIDATE,
        )
    if not view.discovery_event_ids:
        raise PreconditionUnmet("reference_discovery_not_open")
    return SimpleNamespace(
        constants={}, enums={"discovery_event_id": view.discovery_event_ids}
    )


def _choice_errors(deps, arguments) -> tuple[str, ...]:
    """A choice names a row of the page it cites and the researcher's words it rests
    on (#471 Q2 a); a researcher's own selection binds the row."""
    view = reference_projection(deps)
    accession = reference_choice_candidate(arguments["assembly_accession"])
    if accession is None:
        return ()
    page = reference_discovery(view, arguments["discovery_event_id"])
    if page.row(accession) is None:
        return ("assembly_not_on_discovery_page",)
    receipt = applicable_reference_receipt(view) or answered_choice_receipt(view)
    selected = receipt.assertion.selected_candidate_id if receipt else None
    if selected is not None and selected != f"assembly:{accession}":
        return ("researcher_selected_other",)
    quote = arguments.get("quote")
    if not quote:
        return ("choice_quote_missing",)
    events = planning_evidence(deps.journey)
    # The researcher's words, or a cited paper's as its reading kept them.
    if not quoted(researcher_words(events), quote) and (
        paper_quote_event(events, quote) is None
    ):
        return ("choice_quote_ungrounded",)
    return ()


def _supplied_values(deps, view):
    choices = current_choices(view)
    # A researcher's answer about the current choice may supply a new accession.
    receipt = answered_choice_receipt(view)
    if (choices or view.description_event_id) and receipt is None:
        raise PreconditionUnmet(
            "reference_choice_already_recorded",
            choices[-1].source_event_id if choices else view.description_event_id,
            lambda: {
                "next": _next_reference_step(deps, view, "record_supplied_reference")
            },
        )
    # The model types the accession from a researcher statement it cites; only the
    # answer's own words replace a choice, never a stale statement. A statement about
    # the reference is the request, an answer to a reference question (one asked on a
    # page before any choice too) or an Execution Review's reference revision: another
    # Review's answer (a target, the sample) states no reference, and citing it would
    # change the offered call under a pending retry (review r1 W1, r2 SP3).
    events = planning_evidence(deps.journey)
    reference = reference_requirement(
        deps, view, "reference_assembly_identity"
    ).requirement_id
    asked = {
        event["event_id"]: event["payload"]["requirement_id"]
        for event in events
        if event["event_type"] == "PlanningHitlRequested"
    }
    statements = (
        (receipt.response_event_id,)
        if receipt is not None
        else tuple(
            event["event_id"]
            for event in events
            if researcher_text(event) is not None
            and (
                event["event_type"] != "PlanningHitlResponded"
                or asked.get(event["payload"]["request_event_id"]) == reference
            )
        )
    )
    if receipt is None:
        # A cited paper's reading that keeps the genome version it states.
        statements = (*statements, *_paper_versions(deps, view))
    if not statements:
        raise PreconditionUnmet("researcher_statement_missing")
    return SimpleNamespace(constants={}, enums={"assertion_event_id": statements})


def _statement_errors(
    deps, event_id, quote, rule, value, codes, *, paper=False
) -> tuple[str, ...]:
    """A value the model typed from a researcher statement, or with ``paper`` from a
    reading fact of a cited paper (#383 A2): the quote stands in the words it cites
    and states the value (#472 Q2 a, ADR 0049 membership)."""
    events = planning_evidence(deps.journey)
    texts = (
        statement_texts(events, event_id)
        if paper
        else tuple(
            text
            for event in events
            if event["event_id"] == event_id
            and (text := researcher_text(event)) is not None
        )
    )
    if not quoted(texts, quote):
        return (codes[0],)
    if not rule.matches(value, quote):
        return (codes[1],)
    return ()


def _inspect_values(deps, view):
    if view.verified_reference is not None:
        raise PreconditionUnmet(
            "reference_already_verified", view.verified_reference.assembly_accession
        )
    _inspectable(view)
    return _BOUND_NOTHING


_REFERENCE_VALUES = {
    "request_planning_hitl": _question_values,
    "attest_sample": _attest_values,
    "resolve_taxon": _resolve_values,
    "discover_assemblies": _discover_values,
    "record_reference_choice": _choice_values,
    "record_supplied_reference": _supplied_values,
    "inspect_reference": _inspect_values,
}


def _reference_values(deps, name):
    view = reference_projection(deps)
    if view.planning_hitl.pending_request is not None:
        raise PreconditionUnmet(
            "planning_question_pending",
            view.planning_hitl.pending_request.request_event_id,
        )
    return _REFERENCE_VALUES[name](deps, view)


def reference_facts_projection(deps) -> ReferenceFactsProjection:
    """The checked companion view; historical replay supplies its exact prefix."""
    if hasattr(deps.journey, "read_view"):
        from bio_neb.application.planning_read import read_planning

        projected = read_planning(deps.journey).companions
        if projected is None:
            raise ValueError("companions capability lacks its checked prerequisites")
        return projected
    return project_reference_facts(deps.journey.read())


def _companion_due(projection):
    """Each actionable companion work item with its latest fact and disposition."""
    actionable = [
        work_item_id
        for work_item_id in projection.pending_work_item_ids
        if work_item_id not in projection.no_progress_work_item_ids
        or work_item_id in projection.conflicted_work_item_ids
    ]
    for work in projection.work_items:
        if work.work_item_id not in actionable:
            continue
        facts = [f for f in projection.facts if f.work_item_id == work.work_item_id]
        latest = facts[-1] if facts else None
        outcome = (
            latest.result.get("outcome")
            if latest is not None and latest.result.get("status") == "observed"
            else None
        )
        disposition = next(
            (
                item
                for item in reversed(projection.asset_dispositions)
                if item.work_item_id == work.work_item_id
            ),
            None,
        )
        selection_basis = next(
            (
                fact
                for fact in reversed(facts)
                if fact.result.get("outcome") in {"multiple", "ambiguous"}
            ),
            latest
            if work.work_item_id in projection.conflicted_work_item_ids
            else None,
        )
        selection_needed = selection_basis is not None
        yield SimpleNamespace(
            work=work,
            latest=latest,
            outcome=outcome,
            disposition=disposition,
            selection_needed=selection_needed,
            selection_basis=selection_basis,
            describe=latest is None
            or outcome in {"absent", "partial", "incompatible", "multiple", "ambiguous"}
            or (selection_needed and disposition is not None),
        )


def _prebuilt_work(deps):
    """The verified assembly's index work (#464), open with companion work (owner,
    #459 D1)."""
    return reference_facts_projection(deps).prebuilt_index


def _prebuilt_values(deps, name):
    """The index Tools stay callable once the FASTA is verified: after a choice (a
    later one replaces it) and after an Execution Review revise (owner #464 I2)."""
    work = _prebuilt_work(deps)
    if name != "search_prebuilt_indexes" and not work.listing_event_ids:
        raise PreconditionUnmet("prebuilt_index_unsearched")
    if name == "record_prebuilt_index_choice" and _suggestion_unanswered(deps, work):
        raise PreconditionUnmet("suggestion_unanswered")
    return _BOUND_NOTHING


def _suggestion_unanswered(deps, work) -> bool:
    """Whether the latest suggestion of another assembly waits for the researcher:
    no answered question presents it (owner, #459 D4: the researcher chooses)."""
    if not work.suggestion_event_ids:
        return False
    latest = work.suggestion_event_ids[-1]
    events = planning_evidence(deps.journey)
    asked = {
        event["event_id"]
        for event in events
        if event["event_type"] == "PlanningHitlRequested"
        and latest in event["payload"]["evidence_event_ids"]
    }
    return not any(
        event["event_type"] == "PlanningHitlResponded"
        and event["payload"]["request_event_id"] in asked
        for event in events
    )


def _prebuilt_fact(deps, event_id):
    """The index fact recorded as ``event_id`` in this Run, or None."""
    event = next(
        (e for e in planning_evidence(deps.journey) if e["event_id"] == event_id),
        None,
    )
    return None if event is None else validate_prebuilt_index_fact(event)


def _prebuilt_errors(deps, name, arguments) -> tuple[str, ...]:
    """A read names a candidate row of a listing of this FASTA; a choice cites a read of
    it that holds every requested sequence, or none (ADR 0049 membership, #464)."""
    work = _prebuilt_work(deps)
    if name == "read_prebuilt_index":
        if arguments["listing_event_id"] not in work.listing_event_ids:
            return ("index_listing_unknown",)
        listing = _prebuilt_fact(deps, arguments["listing_event_id"])
        if (
            not isinstance(listing.result, PrebuiltIndexesObserved)
            or listing.result.candidate(arguments["value"]) is None
        ):
            return ("index_row_not_on_listing",)
        other = arguments.get("assembly_accession")
        if other is not None and other not in _discovered_accessions(deps):
            return ("index_assembly_not_discovered",)
        return ()
    if name == "suggest_assembly_with_index":
        return _suggestion_errors(deps, work, arguments)
    if name == "record_prebuilt_index_choice":
        if _reason_invalid(arguments["reason"]):
            return ("prebuilt_choice_reason_invalid",)
        # The researcher chose a suggested assembly with its index: while that
        # answer stands, the Run's index is that one, until an Execution Review
        # revise reopens the index work (review r2 SP3-R2).
        standing = (
            reference_projection(deps).standing_suggestion
            if work.reopened_by_event_id is None
            else None
        )
        # A null read is an absent one.
        if arguments["index"] == "none":
            if arguments.get("read_event_id") is not None:
                return ("prebuilt_choice_read_unexpected",)
            if standing is not None:
                return ("choice_not_the_researchers_index",)
            return ()
        if arguments.get("read_event_id") is None:
            return ("prebuilt_choice_read_missing",)
        if arguments["read_event_id"] not in work.read_event_ids:
            return ("index_read_unknown",)
        read = _prebuilt_fact(deps, arguments["read_event_id"])
        if read.tool_input.other_assembly is not None:
            return ("index_of_another_assembly",)
        if read.bound_candidate() is None:
            return ("index_lacks_requested_sequences",)
        if (
            standing is not None
            and read.tool_input.candidate.candidate_id != standing.candidate_id
        ):
            return ("choice_not_the_researchers_index",)
    return ()


def _suggestion_errors(deps, work, arguments) -> tuple[str, ...]:
    """A suggestion cites a read of this work that pairs another assembly's index
    holding that assembly's main sequences, not yet answered (#459 D3)."""
    if _reason_invalid(arguments["reason"]):
        return ("prebuilt_choice_reason_invalid",)
    if arguments["read_event_id"] not in work.read_event_ids:
        return ("index_read_unknown",)
    read = _prebuilt_fact(deps, arguments["read_event_id"])
    if read.tool_input.other_assembly is None:
        return ("suggestion_of_the_verified_assembly",)
    if read.bound_candidate() is None:
        return ("index_lacks_requested_sequences",)
    if any(
        _prebuilt_fact(deps, event_id).tool_input.read_event_id
        == arguments["read_event_id"]
        for event_id in work.suggestion_event_ids
    ):
        return ("suggestion_already_answered",)
    return ()


def _discovered_accessions(deps) -> frozenset[str]:
    """Every assembly a discovery page of this Run lists (ADR 0049 membership): the
    assemblies of the Run's organism an index read may pair against (#459)."""
    view = reference_projection(deps)
    return frozenset(
        row["assembly_accession"]
        for fact in view.facts
        if fact.event_type == "ReferenceAssembliesRecorded"
        for row in fact.result.get("rows", ())
    )


def _companion_values(deps, name):
    projection = reference_facts_projection(deps)
    if projection.status == "planning_hitl_pending":
        raise PreconditionUnmet("planning_question_pending", projection.status)
    if name in PREBUILT_INDEX_TOOLS:
        return _prebuilt_values(deps, name)
    due = list(_companion_due(projection))
    if not due:
        raise PreconditionUnmet("companion_work_complete")
    item = due[0]
    if name == "describe_companion":
        if not any(each.describe for each in due):
            raise PreconditionUnmet("companion_description_not_due", item.outcome)
        return _BOUND_NOTHING
    if name == "request_companion_asset_selection":
        if not item.selection_needed or item.disposition is not None:
            raise PreconditionUnmet("companion_selection_not_needed", item.outcome)
        return _bound(
            work_item_id=item.work.work_item_id,
            basis_event_id=item.selection_basis.source_event_id,
        )
    if name == "request_companion_refinement":
        if item.outcome not in {"partial", "absent", "incompatible"}:
            raise PreconditionUnmet("companion_refinement_not_needed", item.outcome)
        return _bound(
            work_item_id=item.work.work_item_id,
            evidence_event_id=item.latest.source_event_id,
        )
    raise ValueError(f"no companion Tool {name!r}")


def reference_tool_values(deps, name):
    """A reference or companion Tool's server-bound values now.

    Raises ``PreconditionUnmet`` with the registry's reason when the Tool cannot
    run now; admission refuses the call before contact.
    """
    if name in COMPANION_TOOLS:
        return _companion_values(deps, name)
    return _reference_values(deps, name)


def _reason_invalid(reason: str) -> bool:
    return not reason.strip() or reason.strip() != reason or len(reason) > 2000


def reference_tool_errors(deps, action) -> tuple[str, ...]:
    """The domain refusals of a call's agent-chosen arguments, before contact.

    Admission already refused an unmet precondition and validated the argument
    shapes; these read the arguments against the checked state and the port.
    """
    name, arguments = action.tool_name, action.arguments
    view = None if name in COMPANION_TOOLS else reference_projection(deps)
    if name == "resolve_taxon":
        refusal = deps.reference.reference_adapter.taxon_term_refusal(arguments["term"])
        return () if refusal is None else (refusal,)
    if name == "discover_assemblies":
        if arguments["taxid"] != view.resolved_taxon.ncbi_tax_id:
            return ("taxid_unresolved",)
        return ()
    if name in PREBUILT_INDEX_TOOLS:
        return _prebuilt_errors(deps, name, arguments)
    if name == "describe_companion":
        due = {
            item.work.role: item
            for item in _companion_due(reference_facts_projection(deps))
        }
        if arguments["asset"] not in due or not due[arguments["asset"]].describe:
            return ("companion_asset_not_due",)
        return ()
    if name == "attest_sample":
        return _sample_errors(deps, arguments)
    if name in {"record_reference_choice", "record_supplied_reference"}:
        if _reason_invalid(arguments["reason"]):
            return ("reference_choice_reason_invalid",)
        if name == "record_reference_choice":
            return _choice_errors(deps, arguments)
        return _statement_errors(
            deps,
            arguments["assertion_event_id"],
            arguments["quote"],
            ASSEMBLY_ACCESSION_RULE,
            arguments["assembly_accession"],
            ("supplied_quote_ungrounded", "supplied_accession_not_in_quote"),
            paper=True,
        )
    return ()


def reference_refusal_facts(deps, action, code) -> dict:
    """The typed facts a reference refusal's sentence names (#409)."""
    if code == "researcher_selected_other":
        view = reference_projection(deps)
        receipt = applicable_reference_receipt(view) or answered_choice_receipt(view)
        selected = receipt.assertion.selected_candidate_id
        return {"selected": selected.removeprefix("assembly:")}
    return {}


def reference_choice_candidate(assembly_accession: str) -> str | None:
    """The recorded accession of a ``record_reference_choice`` argument.

    ``"none"`` is the named choice that no listed row fits (explicit ambiguity); a
    null argument is not that choice.
    """
    if not isinstance(assembly_accession, str):
        raise ValueError("reference choice must name an accession or none")
    return None if assembly_accession == NO_FITTING_CANDIDATE else assembly_accession


def _operational_result(result):
    from bio_neb.failure import ProviderError
    from bio_neb.reference_discovery.reference_identity_tool import (
        ReferenceAdapterFailure,
    )

    value = result["payload"]["result"]
    if value.get("status") != "failed":
        return None
    failure = ReferenceAdapterFailure.model_validate(value["failure"])
    if failure.code == "source_answered":
        # The source's own answer about the request: the model reads it.
        return None
    return ProviderError(
        failure.failure_class,
        failure.public_message,
        evidence={
            "original_failure_class": failure.original_failure_class
            or failure.failure_class,
            "contact_attempt": failure.contact_attempt,
        },
    )


_MEDIA = {"json": "application/json", "text": "text/plain"}


def record_source_fact(
    ctx,
    tool_name,
    child_stage,
    fact_type,
    tool_input,
    operation,
    *,
    adapter="ncbi_datasets",
):
    """Run one planning source Tool: its child stage contacts the source, retains
    each GET with its bytes, then records the fact that shows them (#458). The paper
    search runs through it too (#420); ``adapter`` names the stage's source."""
    deps = ctx.deps
    event = None

    def stage():
        nonlocal event
        activity = deps.context.activity.start_child(
            adapter=adapter,
            stage=child_stage,
        )
        try:
            with source_contact_stage(activity), collected_answers(tool_name) as got:
                result = operation()
            shown = shown_answers(got)
            # Each GET is its own retained record, in call order, before the fact
            # that shows it.
            for answer, view in zip(got, shown, strict=True):
                activity.record_with_artifact(
                    ReferenceSourceRetrieved(
                        attempt_id=deps.context.attempt_id,
                        operation=answer.operation,
                        request_identity=answer.request_identity,
                        status_code=answer.status_code,
                        byte_count=view.byte_count,
                        sha256=view.sha256,
                    ),
                    answer.body,
                    media_type=_MEDIA[view.media],
                )
            event = activity.record(
                fact_type(
                    attempt_id=deps.context.attempt_id,
                    tool_input=tool_input,
                    result=result,
                    source_answers=shown,
                )
            )
            activity.succeeded(event)
        except Exception as error:
            from bio_neb.agentic_journey.attempt_journey import ToolExecutionFailed

            failed = activity.record(
                ToolExecutionFailed(
                    attempt_id=deps.context.attempt_id,
                    tool_name=child_stage,
                    error_type=type(error).__name__,
                )
            )
            deps.context.failure_source.capture(failed)
            activity.failed(failed)
            raise

    def execute():
        observed_source_operation(
            deps.context.activity,
            deps.context.attempt_id,
            ctx.tool_call_id,
            tool_name,
            stage,
        )()
        return event

    return deps.observed_tool_session.execute(
        ctx.tool_call_id,
        execute,
        ToolExecutionAdapter(
            identity=adapter,
            operational_failure=_operational_result,
            stage_result_event=lambda result: event,
            artifact_digests=lambda result: (),
        ),
    )


SampleIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        description="The sample name exactly as the researcher's words write it, e.g. "
        "MM-04.",
    ),
]
StatementQuote = Annotated[
    str,
    Field(
        min_length=1,
        max_length=300,
        description="The researcher's exact words that state the value, copied on whole "
        "words from their request or answer.",
    ),
]


def attest_sample(
    ctx: ToolContext, sample_identifier: SampleIdentifier, quote: StatementQuote
) -> dict[str, Any]:
    """Bind the sample name you typed from the researcher's latest statement about
    the sample, quoting their words; the result says whether its format is safe. It
    is still the researcher's assertion: no source verifies it.
    """
    deps = ctx.deps
    view = reference_projection(deps)
    sample = reference_assertion(deps, view, "sample_identity")
    tool_input = SampleAttestationInput(
        grounding=reference_grounding(
            deps, view, "sample_identity", sample.source_event_id
        ),
        input_set_id=view.input_set_id,
        sample_identifier=sample_identifier,
        assertion_event_id=sample.source_event_id,
        quote=quote,
    )
    event = None

    def execute():
        nonlocal event
        activity = deps.context.activity.start_child(
            adapter="deterministic",
            stage="verify_sample_attestation",
        )
        event = activity.record(
            SampleAttestationRecorded(
                attempt_id=deps.context.attempt_id,
                tool_input=tool_input,
                result=deps.reference.sample_tool.verify(
                    tool_input,
                    expected_grounding=tool_input.grounding,
                    expected_input_set_id=view.input_set_id,
                    expected_sample_identifier=sample_identifier,
                ),
            )
        )
        activity.succeeded(event)
        return event

    return deps.observed_tool_session.execute(
        ctx.tool_call_id,
        execute,
        ToolExecutionAdapter(
            identity="deterministic",
            stage_result_event=lambda result: event,
            artifact_digests=lambda result: (),
        ),
    )


TaxonTerm = Annotated[
    str,
    Field(
        min_length=1,
        max_length=300,
        pattern=r"^\S(.*\S)?$",
        description="One organism name or NCBI TaxId, as you would search NCBI Taxonomy for it.",
    ),
]


def resolve_taxon(ctx: ToolContext, term: TaxonTerm) -> dict[str, Any]:
    """Look one taxonomy term up in NCBI Taxonomy; you choose the term. The result
    says resolved, not_found, ambiguous, provisional or deleted, with candidates.
    """
    deps = ctx.deps
    view = reference_projection(deps)
    # The researcher's statement, or the reading fact keeping the paper's organism.
    assertion_id = _taxon_source(deps, view)
    tool_input = ResolveTaxonInput(
        grounding=reference_grounding(
            deps, view, "taxon_identity", view.sample_event_id, assertion_id
        ),
        query=term,
        assertion_event_id=assertion_id,
    )
    return record_source_fact(
        ctx,
        "resolve_taxon",
        "resolve_taxon",
        TaxonResolutionRecorded,
        tool_input,
        lambda: deps.reference.taxon_tool.resolve(
            tool_input,
            expected_grounding=tool_input.grounding,
            expected_query=term,
            adapter=deps.reference.reference_adapter,
        ),
    )


def discover_assemblies(
    ctx: ToolContext,
    taxid: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "The NCBI TaxId `resolve_taxon` resolved; a strain is listed under its "
                "own TaxId once `resolve_taxon` resolves it, or found by search_text."
            ),
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
    """List one page of NCBI Datasets genome assembly reports for a taxon, with
    NCBI's own query options: every row copied as NCBI lists it, NCBI's total_count
    and the next page token. Nothing is ranked or cut: which assembly is the
    reference is your choice, recorded with `record_reference_choice`.
    """
    deps = ctx.deps
    view = reference_projection(deps)
    taxon = view.resolved_taxon
    tool_input = DiscoverReferenceAssembliesInput(
        grounding=reference_grounding(
            deps, view, "reference_assembly_identity", taxon.source_event_id
        ),
        taxon=taxon,
        query=assembly_query(
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
        ),
    )
    return record_source_fact(
        ctx,
        "discover_assemblies",
        "discover_reference_assemblies",
        ReferenceAssembliesRecorded,
        tool_input,
        lambda: deps.reference.discovery_tool.discover(
            tool_input,
            expected_grounding=tool_input.grounding,
            expected_taxon=tool_input.taxon,
            adapter=deps.reference.reference_adapter,
        ),
    )


def inspect_reference(
    ctx: ToolContext,
    accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="A versioned accession to look up; only the selected reference can verify the basis.",
        ),
    ],
) -> dict[str, Any]:
    """Inspect an exact accession in NCBI Datasets against the standing choice.
    A mismatched or negative result permits a different lookup, not a new selection.
    """
    deps = ctx.deps
    view = reference_projection(deps)
    selected = _inspectable(view)
    basis_id = (
        selected.assertion_event_id
        if isinstance(selected, SuppliedAssemblyEvidence)
        else selected.discovery_event_id
    )
    tool_input = DescribeReferenceInput(
        grounding=reference_grounding(
            deps,
            view,
            "reference_assembly_identity",
            basis_id,
            selected.selection_event_id,
        ),
        selected=selected,
        assembly_accession=accession,
    )
    return record_source_fact(
        ctx,
        "inspect_reference",
        "describe_reference",
        ReferenceDescriptionRecorded,
        tool_input,
        lambda: deps.reference.description_tool.describe(
            tool_input,
            expected_grounding=tool_input.grounding,
            expected_selection=selected,
            adapter=deps.reference.reference_adapter,
        ),
    )


def describe_companion(
    ctx: ToolContext,
    asset: Annotated[
        CompanionRole,
        Field(
            description="The companion asset role to describe, such as genomic_fasta."
        ),
    ],
    accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="Versioned assembly accession to query; this does not change the verified reference.",
        ),
    ],
) -> dict[str, Any]:
    """Describe the verified reference's companion asset of this role in NCBI
    Datasets. A researcher's asset selection, if any, is bound by the server.
    """
    deps = ctx.deps
    item = next(
        item
        for item in _companion_due(reference_facts_projection(deps))
        if item.work.role == asset
    )
    tool_input = DescribeCompanionInput(
        work_item=item.work,
        assembly_accession=accession,
        selected_asset_id=None
        if item.disposition is None
        else item.disposition.asset_id,
        selection_event_id=None
        if item.disposition is None
        else item.disposition.response_event_id,
    )
    return record_source_fact(
        ctx,
        "describe_companion",
        "describe_reference_companion",
        CompanionObservationRecorded,
        tool_input,
        lambda: deps.companions.companion_tool.describe(
            tool_input,
            expected_work_item=item.work,
            adapter=deps.companions.companion_adapter,
        ),
    )


def search_prebuilt_indexes(
    ctx: ToolContext,
    repository: Annotated[
        PrebuiltRepository,
        Field(
            description="The Galaxy reference-data repository to search, by Galaxy's "
            "name: indexes (hand-built genomes), managed (data-manager builds), brc or "
            "vgp."
        ),
    ],
    search_text: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=200,
            description="Text to find in a row's value, dbkey or name, as Galaxy "
            "writes them (an accession, an assembly name such as hg38, or an "
            "organism); omitted lists every row.",
        ),
    ] = None,
    offset: Annotated[
        int,
        Field(ge=0, description="The first matching row to show; next_offset pages."),
    ] = 0,
) -> dict[str, Any]:
    """Search one Galaxy repository's BWA-MEM index catalogue for the verified genomic
    FASTA: every row copied as Galaxy publishes it, one page of 25 matching rows, with
    the catalogue's row count, the matches' count and the next offset. A row whose index
    and FASTA Galaxy serves names its candidate; read one with `read_prebuilt_index`.
    Nothing is ranked: which index fits is your choice."""
    from bio_neb.agentic_journey.source_contacts import call_source

    deps = ctx.deps
    tool_input = SearchPrebuiltIndexesInput(
        grounding=_prebuilt_work(deps).grounding,
        repository=repository,
        search_text=search_text,
        offset=offset,
    )
    return record_source_fact(
        ctx,
        "search_prebuilt_indexes",
        "search_prebuilt_indexes",
        PrebuiltIndexesListed,
        tool_input,
        lambda: search_catalogues(
            tool_input, galaxy=deps.companions.galaxy_adapter, contact=call_source
        ),
        adapter="galaxy_datacache",
    )


def read_prebuilt_index(
    ctx: ToolContext,
    listing_event_id: Annotated[
        str,
        Field(
            description="The PrebuiltIndexesListed event id of the search page that "
            "lists the row."
        ),
    ],
    value: Annotated[
        str,
        Field(
            min_length=1,
            max_length=300,
            description="The row's value, exactly as that page lists it.",
        ),
    ],
    assembly_accession: Annotated[
        str | None,
        Field(
            description="Another assembly of the Run's organism, as a discovery page "
            "of this Run lists it, to pair the index against its own NCBI report: "
            "the verified assembly's newer version when Galaxy has no index of the "
            "verified one. Left out, the index is paired against the verified assembly."
        ),
    ] = None,
) -> dict[str, Any]:
    """Read a listed index's contigs (its `.ann`) and each of its seven files' size;
    nothing large is downloaded. Each sequence of the assembly's reference comes back
    paired, missing or of another length, with its NCBI role, and each extra contig
    with the NCBI role of the row naming it."""
    from bio_neb.agentic_journey.source_contacts import call_source

    deps = ctx.deps
    work = _prebuilt_work(deps)
    listing = _prebuilt_fact(deps, listing_event_id)
    tool_input = ReadPrebuiltIndexInput(
        grounding=work.grounding,
        listing_event_id=listing_event_id,
        value=value,
        candidate=listing.result.candidate(value),
        assembly_accession=assembly_accession,
    )

    def sequences():
        from bio_neb.providers.ncbi_datasets import parse_assembly_report

        accession = tool_input.pairing_accession
        retained = _retained_report(deps, accession)
        if retained is not None:
            event_id, body = retained
            return parse_assembly_report(body), event_id
        return (
            call_source(
                NCBI_OPERATION,
                lambda: deps.companions.companion_adapter.fetch_accession_sequences(
                    accession
                ),
            ),
            None,
        )

    return record_source_fact(
        ctx,
        "read_prebuilt_index",
        "read_prebuilt_index",
        PrebuiltIndexRead,
        tool_input,
        lambda: read_candidate(
            tool_input,
            galaxy=deps.companions.galaxy_adapter,
            contact=call_source,
            sequences=sequences,
        ),
        adapter="galaxy_datacache",
    )


def _retained_report(deps, assembly_accession):
    """NCBI's assembly report of ``assembly_accession`` that a genomic FASTA's
    description retained (#464): its event id and bytes, read through the Journey's
    artifact reader, so a read pairs against the same answer with no second GET; None
    when no description of the assembly's FASTA retained a 2xx answer of it (standards
    r1 ST4), as before any companion is described (owner, #459 D1)."""
    events = planning_evidence(deps.journey)
    for position in range(len(events) - 1, -1, -1):
        event = events[position]
        if event["event_type"] != "CompanionObservationRecorded":
            continue
        payload = event["payload"]
        candidates = payload["result"].get("candidates") or ()
        if (
            payload["tool_input"]["assembly_accession"] != assembly_accession
            or payload["tool_input"]["work_item"]["role"] != "genomic_fasta"
            or not candidates
            or not candidates[0].get("source_locator")
        ):
            continue
        url = (
            candidates[0]["source_locator"].removesuffix("_genomic.fna.gz")
            + "_assembly_report.txt"
        )
        retrieved = next(
            (
                e
                for e in reversed(events[:position])
                if e["event_type"] == "ReferenceSourceRetrieved"
                and e["payload"]["request_identity"] == url
                and 200 <= e["payload"]["status_code"] < 300
            ),
            None,
        )
        if retrieved is not None:
            return retrieved["event_id"], deps.journey.read_artifact(
                retrieved["artifact_references"][0]
            )
    return None


def record_prebuilt_index_choice(
    ctx: ToolContext,
    index: Annotated[
        Literal["chosen", "none"],
        Field(
            description='"chosen" for the index of the read you cite, or "none" when '
            "no Galaxy index fits: the Run then builds its index on CPU."
        ),
    ],
    reason: ChoiceReason,
    read_event_id: Annotated[
        str | None,
        Field(
            description="The PrebuiltIndexRead event id of the index you choose; "
            'left out when index is "none".'
        ),
    ] = None,
) -> dict[str, Any]:
    """Record the Run's index: a read that holds every requested sequence at its
    length (its extra contigs are shown in the Execution Review), or none for the CPU
    build. A later choice replaces it."""
    from bio_neb.agentic_journey.planning_assignment import _record_result

    deps = ctx.deps
    work = _prebuilt_work(deps)
    read = None if index == "none" else _prebuilt_fact(deps, read_event_id)
    return _record_result(
        ctx,
        "record_prebuilt_index_choice",
        lambda: PrebuiltIndexChosen(
            attempt_id=deps.context.attempt_id,
            grounding=work.grounding,
            read_event_id=None if read is None else read_event_id,
            candidate=None if read is None else read.bound_candidate(),
            reason=reason,
        ),
    )


def suggest_assembly_with_index(
    ctx: ToolContext,
    read_event_id: Annotated[
        str,
        Field(
            description="The PrebuiltIndexRead event id of another assembly's index "
            "that holds that assembly's main sequences."
        ),
    ],
    source: Annotated[
        Literal["ncbi_dbsnp", "eva"],
        Field(
            description="The variant source the known targets are checked in: "
            "ncbi_dbsnp (human variants) or eva (non-human variants)."
        ),
    ],
    reason: ChoiceReason,
) -> dict[str, Any]:
    """Suggest another assembly of the Run's organism, whose Galaxy index fits it,
    when the verified assembly has none: the targets the Run knows are checked there
    in the source you name, the Attempt ends, and the researcher is asked to choose
    that assembly for the whole Run or the CPU build of the verified one."""
    from bio_neb.agentic_journey.source_contacts import call_source

    deps = ctx.deps
    work = _prebuilt_work(deps)
    read = _prebuilt_fact(deps, read_event_id)
    tool_input = SuggestAssemblyInput(
        grounding=work.grounding,
        read_event_id=read_event_id,
        assembly_accession=read.tool_input.other_assembly,
        candidate=read.bound_candidate(),
        source=source,
        reason=reason,
    )

    def suggestion() -> SuggestedAssembly:
        targets = _checked_targets(
            deps, tool_input.assembly_accession, source, call_source
        )
        moved = sum(1 for target in targets if target.outcome in {"moved", "looked_up"})
        return SuggestedAssembly(
            assembly_accession=tool_input.assembly_accession,
            targets=targets,
            summary=(
                f"{len(targets)} known target(s) checked on "
                f"{tool_input.assembly_accession}: {moved} placed there."
            ),
        )

    return record_source_fact(
        ctx,
        "suggest_assembly_with_index",
        "suggest_assembly_with_index",
        AssemblyWithIndexSuggested,
        tool_input,
        suggestion,
        adapter="ncbi_investigation",
    )


def _checked_targets(deps, assembly_accession, source, call_source):
    """The Run's known targets checked on ``assembly_accession`` with the
    Investigation's lookup, each GET retained once (#459 D3; live a2 F1)."""
    from bio_neb.investigation.investigation import RetrievedSource
    from bio_neb.investigation.investigation_request import (
        StatedAssembly,
        current_investigation_request,
    )
    from bio_neb.providers.eva_investigation import (
        PublicInvestigationSources,
        fetch_source,
    )
    from bio_neb.reference_discovery.assembly_suggestion import check_known_targets

    adapter = deps.companions.variant_sources or PublicInvestigationSources()

    def fetch(operation, key):
        # Each GET is reported to the collecting Tool by its source's transport.
        contact = call_source(operation, lambda: fetch_source(adapter, operation, key))
        return (
            RetrievedSource.create(
                operation=operation,
                request_identity=contact.request_identity,
                status_code=contact.status_code,
                observed_at=contact.observed_at,
                elapsed_time_ms=contact.elapsed_time_ms,
                raw_response=contact.body,
            ),
            contact.body,
        )

    request = current_investigation_request(deps.journey)
    reference = reference_projection(deps).verified_reference
    record = reference.record
    requested = reference.assembly_accession
    return check_known_targets(
        () if request is None else request.request.items,
        requested=StatedAssembly(
            name=requested, cited_line=f"{requested}, the Run's requested reference"
        ),
        requested_names=frozenset(
            name
            for name in (
                requested,
                record.assembly_name,
                record.synonym,
                record.paired_accession,
            )
            if name
        ),
        requested_accession=requested,
        assembly_accession=assembly_accession,
        fetch=fetch,
        source=source,
    )


def record_supplied_reference(
    ctx: ToolContext,
    assembly_accession: Annotated[
        str,
        Field(
            pattern=ASSEMBLY_ACCESSION_PATTERN,
            description="The versioned GCA_/GCF_ assembly accession the researcher "
            "supplied, exactly as their words write it.",
        ),
    ],
    assertion_event_id: Annotated[
        str,
        Field(
            description="The event id of the researcher's request or answer that "
            "supplies it."
        ),
    ],
    quote: StatementQuote,
    reason: ChoiceReason,
) -> dict[str, Any]:
    """Record the accession you typed from the researcher's words as the reference
    choice, citing the statement and quoting it; inspect it next."""
    from bio_neb.agentic_journey.planning_assignment import _record_result

    return _record_result(
        ctx,
        "record_supplied_reference",
        lambda: ReferenceChoiceRecorded(
            attempt_id=ctx.deps.context.attempt_id,
            grounding=reference_grounding(
                ctx.deps,
                reference_projection(ctx.deps),
                "reference_assembly_identity",
                assertion_event_id,
            ),
            assembly_accession=assembly_accession,
            assertion_event_id=assertion_event_id,
            quote=quote,
            reason=reason,
        ),
    )


def recorded_arguments(tool_name: str, fact: dict[str, Any]) -> dict[str, Any]:
    """The arguments a call's recorded fact was produced from (replay, ADR 0034)."""
    tool_input = fact["payload"]["tool_input"]
    if tool_name == "attest_sample":
        return {
            "sample_identifier": tool_input["sample_identifier"],
            "quote": tool_input["quote"],
        }
    if tool_name == "resolve_taxon":
        return {"term": tool_input["query"]}
    if tool_name == "discover_assemblies":
        return _discovery_arguments(
            tool_input["taxon"]["ncbi_tax_id"],
            AssemblyQuery.model_validate(tool_input["query"]),
        )
    if tool_name == "inspect_reference":
        return {"accession": tool_input["assembly_accession"]}
    if tool_name == "describe_companion":
        return {
            "asset": tool_input["work_item"]["role"],
            "accession": tool_input["assembly_accession"],
        }
    if tool_name == "search_prebuilt_indexes":
        return {key: tool_input[key] for key in ("repository", "search_text", "offset")}
    if tool_name == "read_prebuilt_index":
        return {
            key: tool_input[key]
            for key in ("listing_event_id", "value", "assembly_accession")
            if key in tool_input
        }
    raise ValueError(f"no recorded reference Tool {tool_name!r}")


def admitted_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The admitted call's arguments with each omitted option at its default."""
    if tool_name == "discover_assemblies":
        options = {name: value for name, value in arguments.items() if name != "taxid"}
        return _discovery_arguments(arguments["taxid"], assembly_query(**options))
    if tool_name == "search_prebuilt_indexes":
        return {"search_text": None, "offset": 0, **arguments}
    return dict(arguments)


def _discovery_arguments(taxid: int, query: AssemblyQuery) -> dict[str, Any]:
    """A discovery call's arguments in one form, recorded or admitted alike."""
    return {"taxid": taxid, **query.model_dump(mode="json")}


_SAMPLE_UNATTESTED = ToolPrecondition(
    "sample_unattested",
    "sample attestation",
    "The researcher's sample is not attested yet; attest it first with `attest_sample`.",
    met_by=("attest_sample",),
)


_REFERENCE_CHOICES = ("record_reference_choice", "record_supplied_reference")


REFERENCE_CHOICE_STANDS = ToolPrecondition(
    "reference_choice_stands",
    "reference choice",
    "Your concrete choice stands until a later researcher answer; inspect it with `inspect_reference`.",
    unmet_by=_REFERENCE_CHOICES,
)


_REFERENCE_ALREADY_VERIFIED = ToolPrecondition(
    "reference_already_verified",
    "verified reference",
    "The reference is already verified; describe its companion assets with `describe_companion`.",
    unmet_by=("inspect_reference",),
)


_TAXON_UNRESOLVED = ToolPrecondition(
    "taxon_unresolved",
    "latest taxonomy result",
    "No taxon is resolved yet; look the organism up with `resolve_taxon`, or ask the researcher with `request_planning_hitl` when your terms are exhausted.",
    met_by=("resolve_taxon",),
)


_PREBUILT_PRECONDITIONS = (COMPANIONS_NOT_OPEN, QUESTION_PENDING)


_PREBUILT_UNSEARCHED = ToolPrecondition(
    "prebuilt_index_unsearched",
    "index search",
    "Search Galaxy's catalogues first with `search_prebuilt_indexes`; a read and a choice, none included, follow a search.",
    met_by=("search_prebuilt_indexes",),
)


REASON_INVALID = ToolPrecondition(
    "reference_choice_reason_invalid",
    "reason",
    "State the reason in at most 2000 characters, not blank and without surrounding whitespace.",
)


def _source_lookup(
    name: str,
    arguments: tuple[str, ...],
    fact_type: str,
    view: ModelView,
    *,
    adapter_identity: str = "ncbi_datasets",
    **use: Any,
) -> FunctionToolContract:
    """A reference-stage lookup: one call records one fact and shows its view; a
    source lookup's view carries its sources' answers (#458, #478 G1.3)."""
    return function_tool_contract(
        name,
        "planning_agent",
        arguments,
        fact_type,
        adapter_identity,
        fact_type,
        None,
        None,
        view,
        "generic_mechanical_failure_only",
        **use,
    )


def _observed_status(payload: Mapping[str, Any]) -> str:
    """A source lookup's status: its observed resolution or outcome, else
    ``failed``."""
    result = payload["result"]
    if result["status"] != "observed":
        return result["status"]
    for field in ("resolution", "outcome"):
        if field in result:
            return result[field]
    return "observed"


def _summary(payload: Mapping[str, Any]) -> str:
    return payload["result"]["summary"]


_SOURCE_FAILED = (
    "The source gave no usable answer, as the statement says; call it again, or "
    "continue with another Tool callable now."
)


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    _source_lookup(
        "attest_sample",
        ("sample_identifier: str", "quote: str"),
        "SampleAttestationRecorded",
        ModelView(
            next_step={
                "verified": "The sample is attested; resolve the organism with "
                "`resolve_taxon`.",
                "*": "Quote the researcher's sample identifier exactly as they "
                "wrote it, or ask them with `request_planning_hitl`.",
            },
            statement=_summary,
        ),
        adapter_identity="deterministic",
        stages=("reference",),
        examples=('{"sample_identifier": "MM-04", "quote": "sample MM-04"}',),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            ToolPrecondition(
                "sample_attested",
                "sample attestation",
                "The sample is already attested; continue with `resolve_taxon`.",
            ),
            ToolPrecondition(
                "sample_assertion_missing",
                "sample assertion",
                "The researcher has not stated the sample name; ask for it with `request_planning_hitl`.",
            ),
            ToolPrecondition(
                "sample_quote_ungrounded",
                "quote",
                "The quote must copy the researcher's latest words about the sample (their request, or their answer about it) exactly, on whole words.",
            ),
            ToolPrecondition(
                "sample_not_in_quote",
                "sample_identifier",
                "The sample name must be one the quote writes, exactly as it writes it.",
            ),
        ),
    ),
    _source_lookup(
        "resolve_taxon",
        ("term: str",),
        "TaxonResolutionRecorded",
        ModelView(
            next_step={
                "resolved": "Discover the candidate references of the resolved "
                "TaxId with `discover_assemblies`.",
                "ambiguous": "Call `resolve_taxon` again with a term that names one "
                "candidate, or ask the researcher with `request_planning_hitl`.",
                "not_found": "Call `resolve_taxon` with another term, or ask the "
                "researcher with `request_planning_hitl` when your terms are "
                "exhausted.",
                "*": _SOURCE_FAILED,
            },
            status=_observed_status,
            statement=_summary,
        ),
        source_surface=SourceSurface(),
        stages=("reference",),
        examples=(
            '{"term": "9606"}',
            '{"term": "Homo sapiens"}',
            '{"term": "Mus musculus"}',
        ),
        negative_results=(
            NegativeResult(
                "TaxonResolutionRecorded",
                "resolution",
                ("not_found", "ambiguous"),
            ),
        ),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            _REFERENCE_ALREADY_VERIFIED,
            REFERENCE_CHOICE_STANDS,
            _SAMPLE_UNATTESTED,
            ToolPrecondition(
                "taxon_assertion_missing",
                "taxon assertion",
                "The researcher has not stated the organism; ask for it with `request_planning_hitl`.",
            ),
            ToolPrecondition(
                "taxon_term_not_one_taxon",
                "term",
                "NCBI Taxonomy reads a comma as a list of several taxa; send one organism name or one TaxId per call, without a comma.",
            ),
        ),
    ),
    _source_lookup(
        "discover_assemblies",
        (
            "taxid: int",
            "reference_only, assembly_source, has_annotation, exclude_paired_reports, "
            "exclude_atypical, assembly_version, assembly_level, first_release_date, "
            "last_release_date, search_text, is_metagenome_derived, is_type_material, "
            "is_ictv_exemplar, exclude_multi_isolate, type_material_category, "
            "tax_exact_match, sort_field, sort_direction: NCBI's own options",
            "page_size: int (1-100, default 100)",
            "page_token: str | None",
        ),
        "ReferenceAssembliesRecorded",
        ModelView(
            next_step={
                "observed": "Choose one row with `record_reference_choice`, citing "
                "this discovery, or read the next page with its next_page_token; "
                "during the index work, search Galaxy for a listed assembly and read "
                "its index with its assembly_accession.",
                "*": _SOURCE_FAILED,
            },
            statement=_summary,
            cite="discovery_event_id",
        ),
        source_surface=SourceSurface(),
        # Also during the index work, beside the verified reference.
        stages=("reference", "companions"),
        examples=(
            '{"taxid": <the resolved TaxId>, "reference_only": true}',
            '{"taxid": <the resolved TaxId>, "search_text": ["<the strain or isolate '
            'the researcher names>"], "assembly_source": "refseq"}',
            '{"taxid": <the resolved TaxId>, "search_text": ["<the assembly or version '
            'the researcher names, as written>"], "assembly_version": "all_assemblies"}',
        ),
        # Every page is NCBI's answer; no count of rows is a negative result.
        negative_results=(),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            _REFERENCE_ALREADY_VERIFIED,
            REFERENCE_CHOICE_STANDS,
            _TAXON_UNRESOLVED,
            ToolPrecondition(
                "researcher_selected_candidate",
                "researcher answer",
                "The researcher already selected a candidate; record that choice with `record_reference_choice`.",
            ),
            ToolPrecondition(
                "taxid_unresolved",
                "taxid",
                "Discovery searches the TaxId `resolve_taxon` resolved; send that TaxId, or resolve another organism first.",
            ),
        ),
    ),
    function_tool_contract(
        "record_supplied_reference",
        "planning_agent",
        (
            "assembly_accession: str",
            "assertion_event_id: str",
            "quote: str",
            "reason: str",
        ),
        "ReferenceChoiceRecorded",
        "deterministic",
        "ReferenceChoiceRecorded",
        None,
        None,
        recorded_view(
            {"*": "Inspect the supplied reference with `inspect_reference`."}
        ),
        "generic_mechanical_failure_only",
        stages=("reference",),
        examples=(
            '{"assembly_accession": "<the accession the researcher wrote>", "assertion_event_id": "<their request or answer event id>", "quote": "<their words that write it>", "reason": "The request names this exact assembly."}',
        ),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            ToolPrecondition(
                "reference_choice_already_recorded",
                "reference choice",
                "A reference choice is already recorded; a supplied accession is recorded before any choice, or after the researcher answers a question about the current choice. {next}",
            ),
            ToolPrecondition(
                "researcher_statement_missing",
                "researcher statement",
                "No request or answer of the researcher's states a reference to cite yet; continue from what the researcher wrote, for example with `discover_assemblies`, or ask.",
            ),
            ToolPrecondition(
                "supplied_quote_ungrounded",
                "quote",
                "The quote must copy the cited statement's words exactly, on whole words: the researcher's request or answer whose event id is assertion_event_id. After a question about the current choice, only that answer replaces it.",
            ),
            ToolPrecondition(
                "supplied_accession_not_in_quote",
                "assembly_accession",
                "The accession must be one the quote writes, versioned, exactly as it writes it; when the researcher's words name no accession, continue from them, for example with `discover_assemblies`, or ask.",
            ),
            REASON_INVALID,
        ),
    ),
    _source_lookup(
        "inspect_reference",
        ("accession: str",),
        "ReferenceDescriptionRecorded",
        ModelView(
            next_step={
                "resolved": "The reference is described; continue with the Tools "
                "callable now.",
                "absent": "NCBI described no such assembly; choose another "
                "candidate, or ask the researcher with `request_planning_hitl`.",
                "source_disagreement": "The sources disagree, as their answers "
                "show; choose another candidate, or ask the researcher with "
                "`request_planning_hitl`.",
                "*": _SOURCE_FAILED,
            },
            status=_observed_status,
            statement=_summary,
        ),
        source_surface=SourceSurface(),
        stages=("reference",),
        # The example names where its value comes from, not an organism's.
        examples=(
            '{"accession": "<the versioned accession of the chosen or supplied reference>"}',
        ),
        negative_results=(
            NegativeResult(
                "ReferenceDescriptionRecorded",
                "resolution",
                ("absent", "source_disagreement"),
            ),
        ),
        preconditions=(
            CORRECTION_PENDING,
            REFERENCE_NOT_OPEN,
            QUESTION_PENDING,
            _REFERENCE_ALREADY_VERIFIED,
            ToolPrecondition(
                "reference_choice_absent",
                "reference choice",
                "Inspect a reference after it is chosen with `record_reference_choice` or supplied with `record_supplied_reference`.",
                met_by=_REFERENCE_CHOICES,
            ),
            _SAMPLE_UNATTESTED,
            _TAXON_UNRESOLVED,
        ),
    ),
    _source_lookup(
        "describe_companion",
        ("asset: str", "accession: str"),
        "CompanionObservationRecorded",
        ModelView(
            next_step={
                "verified": "The companion is verified; describe the next companion "
                "asset, or continue with the Tools callable now.",
                "multiple": "Ask the researcher to choose with "
                "`request_companion_asset_selection`.",
                "ambiguous": "Ask the researcher to choose with "
                "`request_companion_asset_selection`.",
                "partial": "Ask the researcher how to refine it with "
                "`request_companion_refinement`.",
                "incompatible": "Ask the researcher how to refine it with "
                "`request_companion_refinement`.",
                "absent": "Ask the researcher how to refine it with "
                "`request_companion_refinement`.",
                "*": _SOURCE_FAILED,
            },
            status=_observed_status,
            statement=_summary,
        ),
        # NCBI's GETs keep their own surface and bound (256 MiB, for the whole
        # assembly report the Run's reference sequences are read from).
        source_surface=SourceSurface(operations=frozenset({"describe_companion"})),
        stages=("companions",),
        examples=(
            '{"asset": "genomic_fasta", "accession": "<the verified reference\'s versioned accession>"}',
        ),
        negative_results=(
            NegativeResult(
                "CompanionObservationRecorded",
                "outcome",
                ("partial", "absent", "incompatible", "multiple", "ambiguous"),
            ),
        ),
        preconditions=(
            CORRECTION_PENDING,
            COMPANIONS_NOT_OPEN,
            QUESTION_PENDING,
            COMPANION_WORK_COMPLETE,
            ToolPrecondition(
                "companion_description_not_due",
                "companion observation",
                "The companion asset was already described; act on its result, or ask the researcher.",
            ),
            ToolPrecondition(
                "companion_asset_not_due",
                "asset",
                "Describe a companion role that the reference still needs and that is not already described.",
            ),
        ),
    ),
    _source_lookup(
        "search_prebuilt_indexes",
        (
            "repository: indexes | managed | brc | vgp",
            "search_text: str | None",
            "offset: int (default 0)",
        ),
        "PrebuiltIndexesListed",
        ModelView(
            next_step={
                "observed": "Read a listed row's contigs with `read_prebuilt_index`, "
                "citing this search; read the next page with next_offset, or search "
                "another repository or text.",
                "*": _SOURCE_FAILED,
            },
            statement=_summary,
            cite="listing_event_id",
        ),
        adapter_identity="galaxy_datacache",
        source_surface=SourceSurface(operations=frozenset({GALAXY_OPERATION})),
        stages=("companions", "execution_planning"),
        examples=(
            '{"repository": "brc", "search_text": "<the verified reference\'s accession>"}',
            '{"repository": "indexes", "search_text": "<the assembly\'s name, as Galaxy may write it>"}',
        ),
        # Every page is Galaxy's answer; no count of rows is a negative result.
        negative_results=(),
        preconditions=_PREBUILT_PRECONDITIONS,
    ),
    _source_lookup(
        "read_prebuilt_index",
        ("listing_event_id: str", "value: str", "assembly_accession: str | None"),
        "PrebuiltIndexRead",
        ModelView(
            next_step={
                "observed": "Choose it with `record_prebuilt_index_choice`, citing "
                "this read, when it holds every requested sequence; otherwise read "
                "another row or search again, or choose none for the CPU build.",
                "*": _SOURCE_FAILED,
            },
            statement=_summary,
            cite="read_event_id",
        ),
        adapter_identity="galaxy_datacache",
        source_surface=SourceSurface(
            operations=frozenset({GALAXY_OPERATION, NCBI_OPERATION})
        ),
        stages=("companions", "execution_planning"),
        examples=(
            '{"listing_event_id": "<a search page>", "value": "<one listed row\'s value>"}',
        ),
        negative_results=(),
        preconditions=(
            *_PREBUILT_PRECONDITIONS,
            _PREBUILT_UNSEARCHED,
            ToolPrecondition(
                "index_listing_unknown",
                "listing_event_id",
                "Cite a search page of this Run's genomic FASTA, a PrebuiltIndexesListed event id that `search_prebuilt_indexes` returned.",
            ),
            ToolPrecondition(
                "index_row_not_on_listing",
                "value",
                "Name the value of a row that the cited page lists with a candidate (its index and FASTA on Galaxy's datacache); read another page with next_offset, or search again.",
            ),
            ToolPrecondition(
                "index_assembly_not_discovered",
                "assembly_accession",
                "Pair against an assembly a discovery page of this Run lists (its organism's assemblies); list it with `discover_assemblies` first, or omit assembly_accession for the verified assembly.",
            ),
        ),
    ),
    function_tool_contract(
        "record_prebuilt_index_choice",
        "planning_agent",
        ("index: chosen | none", "reason: str", "read_event_id: str | None"),
        "PrebuiltIndexChosen",
        "deterministic",
        "PrebuiltIndexChosen",
        None,
        None,
        recorded_view(
            {
                "*": "The Run's index is recorded; continue with the Tools callable "
                "now. A later choice replaces it."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("companions", "execution_planning"),
        examples=(
            '{"index": "chosen", "read_event_id": "<a read that holds every requested sequence>", "reason": "Galaxy\'s index of the verified assembly holds every requested sequence; its extra contigs are alt loci."}',
            '{"index": "none", "reason": "No listed Galaxy index holds every requested sequence."}',
        ),
        preconditions=(
            *_PREBUILT_PRECONDITIONS,
            _PREBUILT_UNSEARCHED,
            ToolPrecondition(
                "suggestion_unanswered",
                "suggestion",
                "Another assembly's index is suggested and the researcher has not answered; the answer comes back as your next instruction.",
            ),
            ToolPrecondition(
                "prebuilt_choice_read_missing",
                "read_event_id",
                'A chosen index cites its read: give the PrebuiltIndexRead event id of the index you choose, or set index to "none" for the CPU build.',
            ),
            ToolPrecondition(
                "prebuilt_choice_read_unexpected",
                "read_event_id",
                'No index cites no read: leave read_event_id out when index is "none", or set index to "chosen" to choose the read you cite.',
            ),
            ToolPrecondition(
                "index_read_unknown",
                "read_event_id",
                "Cite a read of this Run's genomic FASTA, a PrebuiltIndexRead event id that `read_prebuilt_index` returned.",
            ),
            ToolPrecondition(
                "index_of_another_assembly",
                "read_event_id",
                "The cited read pairs another assembly's index: it is the Run's only when the researcher chooses that assembly; suggest it with `suggest_assembly_with_index`, or choose an index of the verified assembly or none.",
            ),
            ToolPrecondition(
                "choice_not_the_researchers_index",
                "read_event_id",
                "The researcher chose this assembly with its Galaxy index for the whole Run: read that index and choose it; a revise of the Execution Review reopens the choice.",
            ),
            ToolPrecondition(
                "index_lacks_requested_sequences",
                "read_event_id",
                'The cited read does not hold the main assembly: every assembled molecule paired at its length (owner #459 D2; missing minor scaffolds and extra contigs are fine and shown in the Execution Review); read another index, or set index to "none" for the CPU build.',
            ),
            ToolPrecondition(
                "prebuilt_choice_reason_invalid",
                "reason",
                "State the reason in at most 2000 characters, not blank and without surrounding whitespace.",
            ),
        ),
    ),
    function_tool_contract(
        "suggest_assembly_with_index",
        "planning_agent",
        ("read_event_id: str", "source: ncbi_dbsnp | eva", "reason: str"),
        "AssemblyWithIndexSuggested",
        "ncbi_investigation",
        "AssemblyWithIndexSuggested",
        None,
        None,
        recorded_view(
            {
                "*": "The suggestion and its checked targets are recorded; the "
                "Attempt ends and the researcher is asked to choose."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("companions",),
        examples=(
            '{"read_event_id": "<a read of the newer version\'s index against its own report>", "source": "ncbi_dbsnp", "reason": "Galaxy has no index of the requested GRCh37; GRCh38.p14, its current version, has one that holds its main assembly."}',
        ),
        # The Investigation's position and rsID lookups, run on the suggested
        # assembly before the researcher answers.
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
        preconditions=(
            *_PREBUILT_PRECONDITIONS,
            _PREBUILT_UNSEARCHED,
            ToolPrecondition(
                "index_read_unknown",
                "read_event_id",
                "Cite a read of this Run's index work, a PrebuiltIndexRead event id that `read_prebuilt_index` returned.",
            ),
            ToolPrecondition(
                "suggestion_of_the_verified_assembly",
                "read_event_id",
                "The cited read pairs an index of the verified assembly: choose it with `record_prebuilt_index_choice`; suggest only another assembly's index, read with its assembly_accession.",
            ),
            ToolPrecondition(
                "index_lacks_requested_sequences",
                "read_event_id",
                "The cited read does not hold its assembly's main sequences at their lengths; read another index, or choose none for the CPU build.",
            ),
            ToolPrecondition(
                "suggestion_already_answered",
                "read_event_id",
                "This index was suggested and the researcher answered; follow the answer: choose an index of the verified assembly, or none for the CPU build.",
            ),
            ToolPrecondition(
                "prebuilt_choice_reason_invalid",
                "reason",
                "State the reason in at most 2000 characters, not blank and without surrounding whitespace.",
            ),
        ),
    ),
)


__all__ = [
    "TOOL_ROWS",
    "COMPANION_TOOLS",
    "NO_FITTING_CANDIDATE",
    "PREBUILT_INDEX_TOOLS",
    "RECORDED_TOOLS",
    "REFERENCE_TOOLS",
    "admitted_arguments",
    "attest_sample",
    "describe_companion",
    "discover_assemblies",
    "inspect_reference",
    "read_prebuilt_index",
    "record_prebuilt_index_choice",
    "record_supplied_reference",
    "recorded_arguments",
    "reference_choice_candidate",
    "reference_facts_projection",
    "question_subject",
    "reference_question",
    "reference_tool_errors",
    "reference_tool_values",
    "record_source_fact",
    "resolve_taxon",
    "search_prebuilt_indexes",
    "suggest_assembly_with_index",
]
