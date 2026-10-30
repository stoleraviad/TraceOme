"""One pre-execution assignment using independently grounded domain capabilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from types import SimpleNamespace
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, StrictStr

from bio_neb.agentic_journey.action_policy import ChoiceReason, PreconditionUnmet
from bio_neb.agentic_journey.agent_roles import (
    AttemptAdmission,
    AttemptAssignment,
    AttemptReopen,
    BoundInvocation,
    RuntimeBinding,
)
from bio_neb.agentic_journey.agent_work_review import AgentResultMissing
from bio_neb.agentic_journey.attempt_terminal import (
    AttemptDirection,
    ContinueAttempt,
    PlanningHitlTerminal,
    SliceReachedTerminal,
    prepare_attempt_direction,
    project_attempt_terminal,
    record_attempt_terminal,
)
from bio_neb.agentic_journey.continuation_policy import validate_continuation
from bio_neb.agentic_journey.invocation import (
    ActionAdmissionSpec,
    ToolContext,
    assemble_agent,
)
from bio_neb.agentic_journey.planning_preconditions import (
    CORRECTION_PENDING,
    EXECUTION_NOT_OPEN,
    NO_COMPATIBLE_RECIPE,
    QUESTION_PENDING,
)
from bio_neb.agentic_journey.role_declaration import role
from bio_neb.agentic_journey.stage_findings import (
    StageFinding,
    StageFindings,
    StageOutput,
    verified_output,
    verify_stage_output,
)
from bio_neb.agentic_journey.tool_execution import (
    ModelView,
    ToolExecutionAdapter,
    ToolPrecondition,
    function_tool_contract,
    recorded_view,
)
from bio_neb.application.input_journey import InputMetadataInspected
from bio_neb.application.planning_read import (
    current_planning,
    planning_evidence,
    read_planning_events,
)
from bio_neb.input_inspection.binding import (
    InputInspectionDependencies,
    inspect_input_metadata,
)
from bio_neb.input_inspection.staged_objects import registered_inputs_adapter
from bio_neb.journey import Journey, canonical_json
from bio_neb.paper_intake import tools as paper_tools
from bio_neb.parabricks import execution_planning_agent as execution
from bio_neb.parabricks.execution_planning_action_policy import (
    EXECUTION_PLANNING_TOOLS,
    execution_refusal_facts,
    execution_tool_errors,
    execution_tool_values,
)
from bio_neb.parabricks.execution_planning_journey import PlanningCandidateSelected
from bio_neb.parabricks.proposal_information import (
    PROPOSAL_REQUIREMENT_REPAIR_GUIDANCE,
    PROPOSAL_REQUIREMENT_RULES,
    AnsweredTargetsDraft,
    ProposalInformationPlanDraft,
    ProposalPlanGroundingError,
    ground_answered_targets,
    ground_proposal_information_plan,
)
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
    TargetAnswerRead,
)
from bio_neb.parabricks.recipe_resources import RecipeResourceAdapter
from bio_neb.reference_discovery import reference_facts_agent as companions
from bio_neb.reference_discovery import reference_identity_agent as reference
from bio_neb.reference_discovery import reference_tools

ROLE = "planning_agent"
DOMAIN_TOOLS = {
    **dict.fromkeys(("record_reference_choice", "request_planning_hitl"), "reference"),
    **dict.fromkeys(
        (
            *reference_tools.REFERENCE_TOOLS,
            "describe_companion",
            *reference_tools.PREBUILT_INDEX_TOOLS,
        ),
        "reference_tools",
    ),
    **dict.fromkeys(
        ("request_companion_asset_selection", "request_companion_refinement"),
        "companions",
    ),
    **dict.fromkeys(EXECUTION_PLANNING_TOOLS, "execution"),
    **dict.fromkeys(paper_tools.PAPER_TOOLS, "paper_tools"),
}


def _current_inputs(deps):
    if isinstance(deps.journey, Journey):
        return deps.journey.checked_view("input_selection")
    # Detached evidence is validated by the same owner during historical replay.
    from bio_neb.application.input_journey import project_input_selection

    return project_input_selection(planning_evidence(deps.journey))


def _registration(events):
    return next(e for e in events if e["event_type"] == "RunInputsRegistered")


def _inspection(events):
    return next(
        (e for e in reversed(events) if e["event_type"] == "InputMetadataInspected"),
        None,
    )


def _ground_draft(events, draft):
    registration = _registration(events)
    request = next(e for e in events if e["event_type"] == "ResearchRequestRegistered")
    inspection = _inspection(events)
    if inspection is None:
        raise ValueError("proposal requirements need inspected inputs")
    fact = InputMetadataInspected.model_validate(
        {"event_type": inspection["event_type"], **inspection["payload"]}
    )
    registered = tuple(
        e["event_id"]
        for e in events
        if e["event_type"] == "InvestigationRequestRegistered"
    )
    return ground_proposal_information_plan(
        draft,
        journey_id=registration["journey_id"],
        research_request=request["payload"]["prompt"],
        request_event_id=request["event_id"],
        registration_event_id=registration["event_id"],
        inspection_event_id=inspection["event_id"],
        input_set_id=registration["payload"]["input_set_id"],
        inspection_result=fact.result,
        registered_request_event_ids=registered,
        # A cited paper's quote stands in the paper Tools' own words.
        citation_words=paper_tools.paper_statements(events),
        # A paper retrieved before the plan: what a requirement may cite.
        recorded_papers={
            e["event_id"]: e["payload"]["origin"]
            for e in events
            if e["event_type"] == "PaperRecorded"
        },
    )


def pending_target_answer(journey, events=None, view=None):
    """The researcher's target answer planning has not read yet, or None (#462,
    owner 2026-10-26: an answer goes back to planning's target reading, as an
    input-correction amend goes back to its stage). ``events`` and ``view`` are the
    caller's own reads, when it has them."""
    events = planning_evidence(journey) if events is None else events
    read = {
        e["payload"]["answer_event_id"]
        for e in events
        if e["event_type"] == "TargetAnswerRead"
    }
    if not any(
        e["event_type"] == "PlanningHitlResponded"
        and e["payload"]["response"]["action"] == "clarify"
        and e["event_id"] not in read
        for e in events
    ):
        return None
    view = current_planning(journey) if view is None else view
    if view.plan_event_id is None or view.requirements is None:
        return None
    return next(
        (
            unsettled
            for unsettled in view.requirements.unsettled_targets
            if unsettled.reason == "answer_unread"
        ),
        None,
    )


_CONTINUATION_KEYS = (
    "continuation_request_event_id",
    "continuation_effort_review_event_id",
    "continuation_availability_review_event_id",
    "continuation_work_review_event_id",
    "continuation_operational_review_event_id",
)


def _work_origin(events, attempt_id):
    """The index of the AttemptStarted that began this Attempt's work: its own, or,
    for an Attempt that continues a stopped one, that Attempt's origin. None when a
    continuation's source names no Attempt."""
    by_id = {e["event_id"]: e for e in events}
    starts = {
        e["payload"]["attempt_id"]: i
        for i, e in enumerate(events)
        if e["event_type"] == "AttemptStarted"
    }
    seen = set()
    while attempt_id in starts and attempt_id not in seen:
        seen.add(attempt_id)
        payload = events[starts[attempt_id]]["payload"]
        source = next((payload[k] for k in _CONTINUATION_KEYS if payload.get(k)), None)
        if source is None:
            return starts[attempt_id]
        event = by_id.get(source, {})
        prior = event.get("payload", {}).get("attempt_id")
        if prior is None:
            request = by_id.get(event.get("payload", {}).get("request_event_id"), {})
            prior = request.get("payload", {}).get("attempt_id")
        if prior is None:
            return None
        attempt_id = prior
    return None


def _answer_to_read(journey, attempt_id, events=None, view=None):
    """The unread target answer this Attempt reads: one recorded before the Attempt's
    work began. A fresh Attempt started after the answer reads it, and so does one
    that continues it (a provider retry resends the reading's exact request, Spec r3
    SP6). An Attempt whose work began before the answer, running or continued,
    finishes that work, and the next fresh Attempt reads the answer."""
    events = planning_evidence(journey) if events is None else events
    pending = pending_target_answer(journey, events, view)
    if pending is None or attempt_id is None:
        return pending
    if not any(
        e["event_type"] == "AttemptStarted" and e["payload"]["attempt_id"] == attempt_id
        for e in events
    ):
        return pending
    origin = _work_origin(events, attempt_id)
    answer = next(
        i for i, e in enumerate(events) if e["event_id"] == pending.answer_event_id
    )
    return pending if origin is not None and answer < origin else None


def _attempt_id(deps):
    return getattr(getattr(deps, "context", None), "attempt_id", None)


def _draft_arguments(arguments):
    # Pydantic AI exposes a single model argument as the model schema itself and
    # accepts its explicit one-key envelope too. Admission uses the same native
    # contract, retaining the original call arguments unchanged.
    return arguments["draft"] if set(arguments) == {"draft"} else arguments


_REFERENCE_STAGE_TOOLS = frozenset(
    {
        *reference_tools.REFERENCE_TOOLS,
        "record_reference_choice",
        "request_planning_hitl",
    }
)
_COMPANION_STAGE_TOOLS = frozenset(reference_tools.COMPANION_TOOLS)
_PAPER_TOOLS = frozenset(paper_tools.PAPER_TOOLS)


def assignment_action_errors(deps, action):
    """Domain admission over effective arguments, before contact.

    Admission already refused an unmet precondition, filled and checked each
    Tool's bound values and validated its arguments; domains keep their choices,
    free text and science.
    """
    name = action.tool_name
    events = planning_evidence(deps.journey)
    if name == "inspect_input_metadata":
        return ()
    if name == "record_proposal_requirements":
        draft = _draft_arguments(action.arguments)
        try:
            _ground_draft(
                events,
                ProposalInformationPlanDraft.model_validate(draft),
            )
        except ProposalPlanGroundingError as error:
            return error.errors
        except ValueError:
            return ("proposal_inputs_ineligible",)
        return ()
    if name == "record_answered_targets":
        pending = _answer_to_read(deps.journey, _attempt_id(deps), events)
        if pending is None:
            return ("target_answer_not_pending",)
        try:
            ground_answered_targets(
                AnsweredTargetsDraft.model_validate(_draft_arguments(action.arguments)),
                pending.answer,
            )
        except ProposalPlanGroundingError as error:
            return error.errors
        return ()
    if name == "reconsider_recipe_choice":
        reason = action.arguments["reason"]
        return (
            ()
            if isinstance(reason, str)
            and reason.strip() == reason
            and 1 <= len(reason) <= 2000
            else ("reconsideration_reason_invalid",)
        )
    if name in _REFERENCE_STAGE_TOOLS or name in _COMPANION_STAGE_TOOLS:
        return reference_tools.reference_tool_errors(deps, action)
    if name in _PAPER_TOOLS:
        return paper_tools.paper_tool_errors(deps, action)
    if name in EXECUTION_PLANNING_TOOLS:
        return execution_tool_errors(deps, action)
    return ("tool_not_eligible",)


class InputCorrectionStageOutput(StageOutput):
    """The re-inspection a pending input correction asked for."""

    stage: ClassVar[str] = "input_correction"
    evidence_types: ClassVar = {
        "inspection_event_id": frozenset({"InputMetadataInspected"})
    }
    inspection_event_id: StrictStr


class ProposalRequirementsStageOutput(StageOutput):
    """The recorded proposal plan the reference work opened on."""

    stage: ClassVar[str] = "proposal_requirements"
    evidence_types: ClassVar = {
        "plan_event_id": frozenset({"ProposalRequirementsIdentified"})
    }
    plan_event_id: StrictStr


class ReferenceStageOutput(StageOutput):
    """The verified reference companion work opened on."""

    stage: ClassVar[str] = "reference"
    evidence_types: ClassVar = {
        "reference_event_id": frozenset({"ReferenceDescriptionRecorded"})
    }
    agrees_with: ClassVar = {"assembly_accession": "reference_event_id"}
    assembly_accession: StrictStr
    reference_event_id: StrictStr


class CompanionsStageOutput(StageOutput):
    """The companion observations execution planning opened on."""

    stage: ClassVar[str] = "companions"
    evidence_types: ClassVar = {
        "companion_event_ids": frozenset({"CompanionObservationRecorded"})
    }
    companion_event_ids: tuple[StrictStr, ...]


class ExecutionPlanningStageOutput(StageOutput):
    """The recipe and resource selections of a ready execution plan."""

    stage: ClassVar[str] = "execution_planning"
    evidence_types: ClassVar = {
        "selection_event_ids": frozenset({"PlanningCandidateSelected"})
    }
    selection_event_ids: tuple[StrictStr, ...]


def _status(result):
    """A recorded result's own outcome word, as the findings view shows it."""
    return next(
        (
            result[key]
            for key in ("resolution", "outcome", "status")
            if isinstance(result.get(key), str)
        ),
        None,
    )


def _stage_facts(stage, view, events):
    """The recorded facts the stage's Tools answered so far, in order."""
    if stage == "target_answer":
        # The answer to read, by its item: the researcher's words as written.
        pending = next(
            u
            for u in view.requirements.unsettled_targets
            if u.reason == "answer_unread"
        )
        subject = next(
            item.subject_key
            for item in view.requirements.scientific_investigation.items
            if item.requirement_id == pending.requirement_id
        )
        return [
            StageFinding(
                kind="PlanningHitlResponded",
                source_event_id=pending.answer_event_id,
                subject_key=subject,
                value=pending.answer,
            )
        ]
    if stage == "paper":
        # Each cited paper with its state, and its searches, retrievals and
        # re-entered citations.
        return list(paper_tools.paper_findings(view))
    if stage in {"input_correction", "proposal_requirements"}:
        # The plan stage also shows a cited paper found and retrieved before the
        # plan; none on a Run that cites none.
        papers = (
            list(paper_tools.paper_findings(view))
            if stage == "proposal_requirements"
            else []
        )
        return [
            StageFinding(
                kind=e["event_type"],
                source_event_id=e["event_id"],
                status=e["payload"]["result"]["status"]
                if e["event_type"] == "InputMetadataInspected"
                else None,
            )
            for e in events
            if e["event_type"]
            in {"InputMetadataInspected", "ProposalRequirementsIdentified"}
        ] + papers
    facts = {
        "reference": lambda: view.reference.facts,
        "companions": lambda: view.companions.facts,
        "execution_planning": lambda: (
            *view.execution.facts,
            *view.execution.selections,
        ),
    }[stage]()
    kinds = {e["event_id"]: e["event_type"] for e in events}
    recorded = [
        StageFinding(
            kind=kinds[fact.source_event_id],
            source_event_id=fact.source_event_id,
            status=_status(fact.result)
            if hasattr(fact, "result")
            else getattr(fact, "authority", None),
        )
        for fact in facts
    ]
    if stage == "companions":
        return [*recorded, *_index_findings(view, events)]
    if stage != "reference" or view.requirements is None:
        return recorded
    # The reference stage verifies the plan's fact slots; each shows its state, and
    # what a cited paper's reading kept for them, by its reading fact.
    from bio_neb.paper_reading.planning import paper_findings as reading_findings

    return [
        *recorded,
        *reading_findings(view),
        *(
            StageFinding(
                kind="fact_slot",
                source_event_id=view.plan_event_id,
                status=slot.summary,
                subject_key=slot.subject_key,
                value=slot.verified_value or slot.requested_value,
            )
            for slot in view.requirements.fact_slots
            if slot.kind != "scientific_investigation"
        ),
    ]


def _index_findings(view, events):
    """The genomic FASTA's index work (#464): its searches, reads and current choice,
    and, while a revise has reopened it, the researcher's revise with their words."""
    work = view.companions.prebuilt_index
    if work is None:
        return []
    by_id = {e["event_id"]: e for e in events}
    findings = [
        StageFinding(
            kind=by_id[event_id]["event_type"],
            source_event_id=event_id,
            status=_status(by_id[event_id]["payload"]["result"]),
        )
        for event_id in (*work.listing_event_ids, *work.read_event_ids)
    ]
    if work.choice is not None:
        findings.append(
            StageFinding(
                kind="PrebuiltIndexChosen",
                source_event_id=work.choice.source_event_id,
                status="none" if work.choice.candidate is None else "chosen",
            )
        )
    elif work.reopened_by_event_id is not None:
        revision = by_id[work.reopened_by_event_id]
        findings.append(
            StageFinding(
                kind=revision["event_type"],
                source_event_id=revision["event_id"],
                status="revise",
                value=revision["payload"]["response"]["instruction"],
            )
        )
    return findings


def _stage_outputs(view, events):
    """Each stage of the Run with its typed output once its closing facts are
    recorded, in order. The paper stage is the Run's only when its plan cites a
    paper (owner, #420 D1 b): a Run that cites none never enters it."""
    plan = (
        None
        if view.reference is None
        else ProposalRequirementsStageOutput(plan_event_id=view.plan_event_id)
    )
    paper = (
        (("paper", paper_tools.paper_stage_output(view)),)
        if paper_tools.cited_papers(view)
        else ()
    )
    binding = None if view.companions is None else view.companions.verified_reference
    reference = (
        None
        if binding is None
        else ReferenceStageOutput(
            assembly_accession=binding.assembly_accession,
            reference_event_id=binding.reference_event_id,
        )
    )
    companions = (
        None
        if view.execution is None
        else CompanionsStageOutput(
            companion_event_ids=tuple(f.source_event_id for f in view.companions.facts)
        )
    )
    execution = (
        None
        if view.execution is None or view.execution.status != "execution_planning_ready"
        else ExecutionPlanningStageOutput(
            selection_event_ids=tuple(
                s.source_event_id for s in view.execution.selections
            )
        )
    )
    return (
        ("proposal_requirements", plan),
        *paper,
        ("reference", reference),
        ("companions", companions),
        ("execution_planning", execution),
    )


def planning_stage_findings(deps):
    """The current planning stage, its combined findings and its typed output.

    A pending input correction (the input inspection Review's retry or amendment)
    comes first. Otherwise each stage closes on its typed output once the server
    verifies it; the first stage without a verified output is current, and a
    refused output keeps its stage open with the reason.
    """
    events = planning_evidence(deps.journey)
    view = current_planning(deps.journey)
    if _current_inputs(deps).correction_pending:
        return _findings("input_correction", view, events, None, ())
    if _answer_to_read(deps.journey, _attempt_id(deps), events, view) is not None:
        return _findings("target_answer", view, events, None, ())
    verified = []
    outputs = _stage_outputs(view, events)
    for stage, output in outputs:
        if output is None or stage == outputs[-1][0]:
            return _findings(stage, view, events, output, tuple(verified))
        if refusal := verify_stage_output(output, events):
            return _findings(stage, view, events, output, tuple(verified), refusal)
        verified.append(verified_output(output))
    raise AssertionError("planning has no stage")


def _findings(stage, view, events, output, verified, refusal=None):
    if output is not None and refusal is None:
        refusal = verify_stage_output(output, events)
    return StageFindings(
        stage=stage,
        findings=tuple(_stage_facts(stage, view, events)),
        output="open" if output is None else "refused" if refusal else "verified",
        refusal=refusal,
        verified_outputs=verified,
    )


def planning_stage(deps):
    """The coordinator's current planning stage, re-read per request (ADR 0040)."""
    return planning_stage_findings(deps).stage


def planning_bound_values(deps, name):
    """A planning Tool's server-bound values now.

    Raises ``PreconditionUnmet`` with the registry's reason when the Tool's
    preconditions do not hold; admission then refuses the call before contact.
    """
    events = planning_evidence(deps.journey)
    if name == "reconsider_recipe_choice":
        return reconsider_recipe_offer(deps)
    if name == "inspect_input_metadata":
        if not _current_inputs(deps).correction_pending:
            raise PreconditionUnmet("input_set_id_ineligible")
        return SimpleNamespace(
            constants={
                "input_set_id": _registration(events)["payload"]["input_set_id"]
            },
            enums={},
        )
    if name == "record_answered_targets":
        if _answer_to_read(deps.journey, _attempt_id(deps), events) is None:
            raise PreconditionUnmet("target_answer_not_pending")
        return SimpleNamespace(constants={}, enums={})
    if name == "record_proposal_requirements":
        plan = next(
            (e for e in events if e["event_type"] == "ProposalRequirementsIdentified"),
            None,
        )
        if plan is not None:
            raise PreconditionUnmet("proposal_plan_already_recorded", plan["event_id"])
        inspection = _inspection(events)
        status = (
            None if inspection is None else inspection["payload"]["result"]["status"]
        )
        if status != "succeeded":
            raise PreconditionUnmet("proposal_inputs_uninspected", status)
        return SimpleNamespace(constants={}, enums={})
    view = current_planning(deps.journey)
    if name in _REFERENCE_STAGE_TOOLS:
        if view.reference is None:
            raise PreconditionUnmet("reference_work_not_open")
        return reference_tools.reference_tool_values(deps, name)
    if name in _COMPANION_STAGE_TOOLS:
        if view.companions is None:
            raise PreconditionUnmet("companion_work_not_open")
        return reference_tools.reference_tool_values(deps, name)
    if name in _PAPER_TOOLS:
        return paper_tools.paper_tool_values(deps, name)
    if name in EXECUTION_PLANNING_TOOLS:
        if view.execution is None:
            raise execution_planning_unopened(view)
        return execution_tool_values(deps, name)
    raise ValueError(f"planning has no Tool {name!r}")


def execution_planning_unopened(view):
    """Why the execution-planning stage is not open: its upstream work."""
    return PreconditionUnmet("execution_planning_not_open")


def planning_precondition(deps, name):
    """The planning Tool's unmet precondition now, or None when it may run."""
    try:
        planning_bound_values(deps, name)
    except PreconditionUnmet as unmet:
        return unmet
    return None


@dataclass
class PlanningAssignmentDependencies:
    journey: Any
    context: Any
    input: Any
    reference: Any
    companions: Any
    execution: Any
    observed_tool_session: Any
    action_admission_session: Any
    # The paper Tools' Europe PMC and PMC open-data adapter.
    paper_adapter: Any


def _delegate(function, dependency_name):
    @wraps(function)
    def delegated(ctx, *args, **kwargs):
        return function(
            replace(ctx, deps=getattr(ctx.deps, dependency_name)), *args, **kwargs
        )

    return delegated


def _record_result(ctx, name, operation, *, adapter="deterministic"):
    event = None

    def execute():
        nonlocal event
        event = ctx.deps.context.activity.record(operation())
        return event

    return ctx.deps.observed_tool_session.execute(
        ctx.tool_call_id,
        execute,
        ToolExecutionAdapter(
            identity=adapter,
            stage_result_event=lambda result: event,
            artifact_digests=lambda result: (),
        ),
    )


def record_proposal_requirements(
    ctx: ToolContext[PlanningAssignmentDependencies],
    draft: ProposalInformationPlanDraft,
) -> dict[str, Any]:
    """Call it once; reference, companion and recipe work open only after it is recorded."""
    plan = _ground_draft(planning_evidence(ctx.deps.journey), draft)
    return _record_result(
        ctx,
        "record_proposal_requirements",
        lambda: ProposalRequirementsIdentified(
            attempt_id=ctx.deps.context.attempt_id,
            input_set_id=plan.input_set_id,
            plan=plan,
        ),
    )


def record_answered_targets(
    ctx: ToolContext[PlanningAssignmentDependencies],
    draft: AnsweredTargetsDraft,
) -> dict[str, Any]:
    """Record the targets the researcher's answer names, in the answer's own words."""
    pending = _answer_to_read(ctx.deps.journey, ctx.deps.context.attempt_id)
    targets = ground_answered_targets(draft, pending.answer)
    return _record_result(
        ctx,
        "record_answered_targets",
        lambda: TargetAnswerRead(
            attempt_id=ctx.deps.context.attempt_id,
            requirement_id=pending.requirement_id,
            answer_event_id=pending.answer_event_id,
            targets=targets,
        ),
    )


# The Tool's description is its docstring; the cross-field rules join it here,
# so the description and this Tool's refusal guidance share one text.
record_proposal_requirements.__doc__ += "\n    " + PROPOSAL_REQUIREMENT_RULES + "\n"


def reconsider_recipe_choice(
    ctx: ToolContext[PlanningAssignmentDependencies],
    prior_selection_event_id: str,
    conflicting_evidence_event_id: str,
    reason: ChoiceReason,
) -> dict[str, Any]:
    """This lists the compatible recipes again for the same work item; a researcher's
    selection is never reopened here.
    """
    deps = ctx.deps
    try:
        reconsider_recipe_offer(deps)
    except PreconditionUnmet as unmet:
        raise ValueError("recipe reconsideration lost its checked frontier") from unmet
    prior = next(
        e
        for e in planning_evidence(deps.journey)
        if e["event_id"] == prior_selection_event_id
    )
    view = current_planning(deps.journey).execution
    tool_input = next(
        item
        for item in view.recipe_tool_inputs
        if item.work_item.work_item_id == prior["payload"]["work_item_id"]
    )
    deps.context.activity.record(
        PlanningChoiceReconsidered(
            attempt_id=deps.context.attempt_id,
            prior_selection_event_id=prior_selection_event_id,
            conflicting_evidence_event_id=conflicting_evidence_event_id,
            reason=reason,
            authority_event_id=deps.action_admission_session.admitted_validation_event_id(
                ctx.tool_call_id
            ),
        )
    )
    assertion = tool_input.eligibility_assertion
    return execution.record_recipe_observation(
        replace(ctx, deps=deps.execution),
        tool_input.work_item.work_item_id,
        tool_input.work_item.input_set_id,
        tool_input.work_item.assembly_accession,
        assertion.assay if assertion else None,
        assertion.read_layout if assertion else None,
        assertion.assertion_event_id if assertion else None,
        operation_name="reconsider_recipe_choice",
    )


def _planning_tools():
    families = {
        "reference_tools": reference_tools,
        "reference": reference,
        "companions": companions,
        "execution": execution,
        "paper_tools": paper_tools,
    }
    return {
        "inspect_input_metadata": _delegate(inspect_input_metadata, "input"),
        **{
            name: getattr(families[family], name)
            if family in {"reference_tools", "paper_tools"}
            else _delegate(getattr(families[family], name), family)
            for name, family in DOMAIN_TOOLS.items()
        },
        "record_proposal_requirements": record_proposal_requirements,
        "record_answered_targets": record_answered_targets,
        "reconsider_recipe_choice": reconsider_recipe_choice,
    }


def assignment_refusal_facts(deps, action, code):
    """The typed facts a planning refusal's sentence names (#372 D2).

    A proposal draft's grounding computes them beside its codes: the missing
    kinds, the unrepresented input roles and the request file to cite; a stage
    Tool's domain names the values its choice may take (#409).
    """
    name = action.tool_name
    if name in _REFERENCE_STAGE_TOOLS or name in _COMPANION_STAGE_TOOLS:
        return reference_tools.reference_refusal_facts(deps, action, code)
    if name in _PAPER_TOOLS:
        return paper_tools.paper_refusal_facts(deps, action, code)
    if name in EXECUTION_PLANNING_TOOLS:
        return execution_refusal_facts(deps, action, code)
    if name != "record_proposal_requirements":
        return {}
    try:
        _ground_draft(
            planning_evidence(deps.journey),
            ProposalInformationPlanDraft.model_validate(
                _draft_arguments(action.arguments)
            ),
        )
    except ProposalPlanGroundingError as error:
        return error.facts.get(code, {})
    except ValueError:
        return {}
    return {}


PLANNING_ADMISSION = ActionAdmissionSpec(
    validator=assignment_action_errors,
    repair_guidance={
        "record_proposal_requirements": PROPOSAL_REQUIREMENT_REPAIR_GUIDANCE
    },
    repeat_error="planning_action_repeated",
    refusal_facts=assignment_refusal_facts,
)


def assignment_terminal(journey_events, attempt_id):
    """Complete the assignment at a checked useful frontier, never a role stage."""
    if isinstance(journey_events, Journey):
        deps = SimpleNamespace(journey=journey_events)
    else:
        # Historical terminal validation has no live owner or effect capability.
        events = list(journey_events)
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))
    events = planning_evidence(deps.journey)
    view = current_planning(deps.journey)
    inspections = [e for e in events if e["event_type"] == "InputMetadataInspected"]
    current = _current_inputs(deps)
    if current.correction_pending:
        return ContinueAttempt()
    attempt_events = events[
        next(
            i
            for i, e in enumerate(events)
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["attempt_id"] == attempt_id
        ) :
    ]
    inspected_here = any(
        e["event_type"] == "InputMetadataInspected" for e in attempt_events
    )
    read_here = next(
        (e for e in attempt_events if e["event_type"] == "TargetAnswerRead"), None
    )
    if read_here is not None:
        # The answer is read; the coordinator settles or asks again.
        return SliceReachedTerminal(
            checkpoint="target_answer_read", source_event_id=read_here["event_id"]
        )
    if _answer_to_read(deps.journey, attempt_id, events, view) is not None:
        return ContinueAttempt()
    if inspected_here:
        # The stage closes on its typed output once the server verifies it.
        output = InputCorrectionStageOutput(
            inspection_event_id=inspections[-1]["event_id"]
        )
        if verify_stage_output(output, events) is not None:
            return ContinueAttempt()
        return SliceReachedTerminal(
            checkpoint="registered_inputs_inspected",
            source_event_id=output.inspection_event_id,
        )
    if _execution_planning_closed(view, events):
        sources = [*view.execution.facts, *view.execution.selections]
        ids = {s.source_event_id for s in sources}
        source = next(e for e in reversed(events) if e["event_id"] in ids)
        return SliceReachedTerminal(
            checkpoint="execution_planning_ready", source_event_id=source["event_id"]
        )
    suggested = next(
        (
            e
            for e in attempt_events
            if e["event_type"] == "AssemblyWithIndexSuggested"
            and e["event_id"]
            == reference_tools.unasked_suggestion(deps, view.reference)
        ),
        None,
    )
    if suggested is not None:
        # The coordinator asks the researcher, with no model turn.
        return SliceReachedTerminal(
            checkpoint="assembly_suggested", source_event_id=suggested["event_id"]
        )
    asked_here = {e["event_id"] for e in attempt_events}
    paper_review = next(
        (
            r
            for r in paper_tools.run_papers(deps).requests
            if r.pending and r.request_event_id in asked_here
        ),
        None,
    )
    if paper_review is not None:
        # A retrieval's dead end asked the paper Review: the Attempt ends.
        return SliceReachedTerminal(
            checkpoint="paper_review_requested",
            source_event_id=paper_review.request_event_id,
        )
    from bio_neb.paper_reading.planning import reading_due

    if view.plan_event_id is not None and reading_due(view) is not None:
        # The plan's values cite a recorded paper still to read: the Attempt ends,
        # and the coordinator reads the paper before the reference work.
        return SliceReachedTerminal(
            checkpoint="paper_reading_due", source_event_id=view.plan_event_id
        )
    pending = next(
        (r for r in view.hitl.pending_requests if r.attempt_id == attempt_id), None
    )
    if pending is not None:
        # The Attempt ends at its own pending researcher question (ADR 0040).
        return PlanningHitlTerminal(source_event_id=pending.request_event_id)
    return ContinueAttempt()


def _execution_planning_closed(view, events):
    """Execution planning holds nothing more for the Agent, and its output is verified.

    Readiness may still lack what only the researcher supplies, through a question
    the coordinator opens with no model turn: preparation information (#301) or the
    targets (#342).
    """
    from bio_neb.parabricks.planning_readiness import readiness_of

    gap = readiness_of(view, events)
    return (gap is None or gap.asked_by_coordinator) and verify_stage_output(
        _stage_outputs(view, events)[-1][1], events
    ) is None


def planning_closes_before_contact(journey):
    """Whether a fresh planning Attempt would close at execution planning before
    model contact, recording no new fact (its terminal rule in ``assignment_terminal``).
    """
    deps = SimpleNamespace(journey=journey)
    if _current_inputs(deps).correction_pending:
        return False
    view, events = current_planning(journey), planning_evidence(journey)
    return (
        _execution_planning_closed(view, events)
        and pending_target_answer(journey, events, view) is None
    )


_INCOMPLETE = "planning assignment ended without useful typed closure"


# A model or Tool fault keeps its code; any other fault is our bug (D4).
def _review(journey):
    from bio_neb.application.queries import ReviewJourneyQuery, review_journey

    return review_journey(ReviewJourneyQuery(journey_file=str(journey.path)))


@dataclass(frozen=True)
class PlanningWork:
    """The admitted direction and the recipe capability bound before the Attempt."""

    direction: AttemptDirection
    recipe_adapter: RecipeResourceAdapter


def _bound_recipe_adapter(journey):
    """Bind the configured or accepted capability; drift raises before any write."""
    from bio_neb.parabricks.accepted_pre_execution import (
        _accepted_capability_snapshot,
    )
    from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
    from bio_neb.parabricks.configuration import current_configuration

    _, configured = current_configuration(planning_evidence(journey))
    return CapabilitySnapshotAdapter(
        configured.runtime_selection.capability
        if configured is not None
        else _accepted_capability_snapshot(journey)
    )


def _admit(journey, reentry, recipe_adapter=None):
    direction = prepare_attempt_direction(
        journey,
        agent_role=ROLE,
        availability_review_response_event_id=reentry.availability_review_response_event_id,
        effort_review_response_event_id=reentry.effort_review_response_event_id,
        work_review_response_event_id=reentry.work_review_response_event_id,
        operational_review_response_event_id=reentry.operational_review_response_event_id,
    )
    before = direction.before
    fallback = reentry.effort_review_response_event_id or _eligible_head(before)
    causal_head = direction.causal_head or fallback
    registered = before.registered_input_set
    return AttemptAdmission(
        PlanningWork(
            direction,
            _bound_recipe_adapter(journey)
            if recipe_adapter is None
            else recipe_adapter,
        ),
        RuntimeBinding(registered.runtime_selections, registered.input_set_id),
        causal_head,
        availability_retry_frontier=direction.availability_retry_frontier
        or causal_head,
    )


def _eligible_head(before):
    if before.status in {"failed", "execution_approved", "completed"} or (
        before.execution_hitl is not None and before.execution_hitl.authorities
    ):
        raise ValueError("planning assignment is not eligible")
    return before.lineage[-1].event_id


async def _drive(context, invocation, invoke_member, message_history):
    def completed_work():
        disposition = project_attempt_terminal(
            ROLE, context.attempt_id, context.journey
        )
        if isinstance(disposition, ContinueAttempt):
            return None
        # The calls a multi-call response still holds are refused before the
        # Attempt's terminal is recorded; none of them runs.
        invocation.deps.observed_tool_session.close_response(disposition.checkpoint)
        return disposition

    result = await invoke_member(
        invocation.agent,
        None if message_history is not None else invocation.prompt(),
        deps=invocation.deps,
        member_id=context.input_set_id,
        message_history=message_history,
        completed_work=completed_work,
    )
    if not hasattr(result, "checkpoint"):
        # Admission corrects every text ending before the framework sees it, so a
        # run that still ends without the typed result stops for review.
        raise AgentResultMissing("model ended without a typed planning result")
    return result


def _complete(context, disposition):
    if disposition is None or isinstance(disposition, ContinueAttempt):
        raise RuntimeError(_INCOMPLETE)
    record_attempt_terminal(
        context.activity,
        disposition,
        attempt_id=context.attempt_id,
        input_set_id=context.input_set_id,
        proposal_plan_event_id=current_planning(context.journey).plan_event_id,
    )


def _fail(context, _invocation, error):
    """Retain the Attempt's failure by its true class, then hold the Run (#216).

    The failure module holds the Run; only the researcher's ``resume`` starts
    the next Attempt.
    """
    from bio_neb.agentic_journey.attempt_journey import record_attempt_failure
    from bio_neb.agentic_journey.failure_cause import cause_identity
    from bio_neb.failure import Invariant

    failed = record_attempt_failure(context, error)
    code = failed["payload"]["code"]
    raise Invariant(
        "planning_attempt_failed",
        "planning Attempt failed",
        cause=cause_identity(error),
        evidence={
            "attempt_id": context.attempt_id,
            "attempt_failed_event_id": failed["event_id"],
            "code": code,
        },
    ) from error


def planning_assignment(
    *,
    metadata_adapter=None,
    reference_adapter=None,
    companion_adapter=None,
    galaxy_adapter=None,
    recipe_adapter=None,
    model_factory=None,
    variant_source_factory=None,
    paper_adapter=None,
):
    from bio_neb.agentic_journey.adapters import create_nebius_model_endpoint
    from bio_neb.providers.galaxy_datacache import GalaxyDatacache
    from bio_neb.providers.ncbi_datasets import NcbiDatasetsReferenceAdapter
    from bio_neb.providers.paper_sources import PaperSourceAdapter

    # Galaxy is searched beside NCBI's own adapter; a supplied companion adapter (a
    # seam test's) searches only the Galaxy it is given.
    if galaxy_adapter is None and companion_adapter is None:
        galaxy_adapter = GalaxyDatacache()

    @contextmanager
    def open_invocation(context, observe):
        direction = context.admitted.direction

        def dependencies(admission, observed):
            return PlanningAssignmentDependencies(
                journey=context.journey,
                context=context,
                input=InputInspectionDependencies(
                    parent_activity=context.activity,
                    attempt_id=context.attempt_id,
                    journey=context.journey,
                    metadata_adapter=metadata_adapter
                    or registered_inputs_adapter(context.journey),
                    failure_source=context.failure_source,
                    observed_tool_session=observed,
                ),
                reference=reference.ReferenceIdentityPlannerDependencies(
                    parent_activity=context.activity,
                    attempt_id=context.attempt_id,
                    journey=context.journey,
                    reference_adapter=reference_adapter
                    or NcbiDatasetsReferenceAdapter(),
                    observed_tool_session=observed,
                ),
                companions=companions.ReferenceFactsPlannerDependencies(
                    parent_activity=context.activity,
                    attempt_id=context.attempt_id,
                    journey=context.journey,
                    companion_adapter=companion_adapter
                    or NcbiDatasetsReferenceAdapter(),
                    galaxy_adapter=galaxy_adapter,
                    variant_sources=None
                    if variant_source_factory is None
                    else variant_source_factory(),
                    observed_tool_session=observed,
                ),
                execution=execution.ExecutionPlanningDependencies(
                    parent_activity=context.activity,
                    attempt_id=context.attempt_id,
                    journey=context.journey,
                    adapter=context.admitted.recipe_adapter,
                    failure_source=context.failure_source,
                    action_admission_session=admission,
                    observed_tool_session=observed,
                ),
                observed_tool_session=observed,
                action_admission_session=admission,
                paper_adapter=paper_adapter or PaperSourceAdapter(),
            )

        def findings():
            return planning_stage_findings(
                SimpleNamespace(journey=context.journey, context=context)
            )

        def instructions():
            from bio_neb.agentic_journey.role_instructions import stage_text

            # The text is the request's stage's (ADR 0044).
            return stage_text(context.instructions.text, findings().stage)

        def request_context():
            from bio_neb.agentic_journey.assignment_context import (
                current_assignment_context,
            )

            # The stage's combined findings fill as its Tools answer (ADR 0040);
            # the assembly adds the Tools callable now and those waiting.
            return {
                **current_assignment_context(
                    context.journey, context.instructions.message("objective")
                ),
                "stage_findings": findings().model_dump(mode="json"),
            }

        agent, deps = assemble_agent(
            observe(),
            assignment=ROLE,
            instructions=instructions,
            context=request_context,
            tools=_planning_tools(),
            stage=planning_stage,
            bound_values=planning_bound_values,
            admission=PLANNING_ADMISSION,
            policy=context.effective_policy,
            model_policy=context.model,
            role_text=context.instructions,
            journey=context.journey,
            attempt_id=context.attempt_id,
            parent_activity=context.activity,
            failure_source=context.failure_source,
            dependencies=dependencies,
        )
        evidence = planning_evidence(context.journey)
        prompt = canonical_json(
            {
                # Nemotron does the job its last message states: the request's
                # own sentence is the overall question, never a target. Each
                # form's words are its own.
                "task": context.instructions.message(
                    "target_answer_task"
                    if _answer_to_read(context.journey, context.attempt_id, evidence)
                    is not None
                    else role(ROLE).job
                ),
                "researcher_instruction": direction.researcher_instruction,
            }
        )
        yield BoundInvocation(agent, deps, lambda: prompt)

    return AttemptAssignment(
        agent_role=ROLE,
        activity_stage="planning_attempt",
        admit=lambda journey, reentry: _admit(journey, reentry, recipe_adapter),
        validate_continuation=validate_continuation,
        open=open_invocation,
        drive=_drive,
        complete=_complete,
        fail=_fail,
        read_result=_review,
        model_factory=model_factory or create_nebius_model_endpoint,
        reopen=AttemptReopen(
            admission=PLANNING_ADMISSION,
            validate_facts=lambda snapshot, selected: validate_assignment_facts(
                snapshot.timeline, selected
            ),
            compiled_checkpoint=True,
        ),
    )


def _reference_choice_arguments(tool, fact, args):
    payload = fact["payload"]
    recorded = {key: payload[key] for key in args}
    if (
        tool == "record_reference_choice"
        and reference_tools.reference_choice_candidate(args["assembly_accession"])
        == payload["assembly_accession"]
    ):
        # The fact records the named "none" choice as no accession and no quote.
        recorded["assembly_accession"] = args["assembly_accession"]
        if args["assembly_accession"] == reference_tools.NO_FITTING_CANDIDATE:
            recorded.update({key: args[key] for key in ("quote",) if key in args})
    return recorded


def validate_assignment_facts(timeline, selected):
    """Reopen each domain result against its exact admitted arguments and prefix."""

    by_id = {e["event_id"]: e for e in timeline}
    _validate_reconsiderations(timeline, selected)
    for terminal in selected:
        if terminal["event_type"] != "ObservedToolExecutionSucceeded":
            continue
        receipt = terminal["payload"]
        start = by_id[receipt["started_event_id"]]["payload"]
        validated = by_id[start["validation_event_id"]]["payload"]
        args = validated["arguments"]
        fact = by_id[receipt["stage_result_event_id"]]
        payload = fact["payload"]
        prefix = timeline[: timeline.index(fact)]
        tool = start["tool_name"]
        expected = None
        tool_input = payload.get("tool_input", {})
        work = tool_input.get("work_item", {})
        if tool == "inspect_input_metadata":
            expected = {"input_set_id": payload["input_set_id"]}
        elif tool in {
            "record_proposal_requirements",
            "record_answered_targets",
            "search_journey",
        }:
            # The shared Tool reader links the fact to its admitted call; the plan's
            # projector validates its content. The admitted draft is not re-grounded.
            pass
        elif tool in reference_tools.RECORDED_TOOLS:
            # Each call's own fact names the arguments it was produced from.
            args = reference_tools.admitted_arguments(tool, args)
            expected = reference_tools.recorded_arguments(tool, fact)
        elif tool in paper_tools.PAPER_TOOLS:
            args = paper_tools.admitted_arguments(tool, args)
            expected = paper_tools.recorded_arguments(tool, fact)
        elif tool in {"record_supplied_reference", "record_reference_choice"}:
            expected = _reference_choice_arguments(tool, fact, args)
        elif tool == "record_prebuilt_index_choice":
            # A no-index choice cites no read: its null is an absent one.
            expected = {
                "index": "none" if payload["read_event_id"] is None else "chosen",
                "reason": payload["reason"],
                **(
                    {}
                    if payload["read_event_id"] is None
                    else {"read_event_id": payload["read_event_id"]}
                ),
            }
        elif tool == "suggest_assembly_with_index":
            expected = {
                key: tool_input[key] for key in ("read_event_id", "source", "reason")
            }
        elif tool == "reconsider_recipe_choice":
            prior = by_id[args["prior_selection_event_id"]]
            if (
                tool_input["work_item"]["work_item_id"]
                != prior["payload"]["work_item_id"]
            ):
                raise ValueError(
                    "reconsideration refreshed a different recipe work item"
                )
            reconsidered = [
                e
                for e in prefix
                if e["event_type"] == "PlanningChoiceReconsidered"
                and e["payload"]["authority_event_id"] == start["validation_event_id"]
            ]
            if len(reconsidered) != 1:
                raise ValueError("recipe refresh lost its exact retained reason")
        elif tool == "list_compatible_analysis_recipes":
            assertion = tool_input.get("eligibility_assertion") or {}
            expected = {
                "work_item_id": work["work_item_id"],
                "input_set_id": work["input_set_id"],
                "assembly_accession": work["assembly_accession"],
                "input_assay": assertion.get("assay"),
                "input_read_layout": assertion.get("read_layout"),
                "input_assertion_event_id": assertion.get("assertion_event_id"),
            }
        elif tool == "lookup_compatible_resource_profiles":
            expected = {
                "work_item_id": work["work_item_id"],
                "recipe_candidate_id": work["recipe_candidate"]["candidate_id"],
            }
        elif tool in {"select_recipe_candidate", "select_resource_candidate"}:
            expected = PlanningCandidateSelected.model_validate(
                {"event_type": fact["event_type"], **payload}
            ).admitted_arguments()
        elif fact["event_type"] == "PlanningHitlRequested":
            if args.get("requirement_id", payload["requirement_id"]) != payload[
                "requirement_id"
            ] or any(
                args[key] not in payload["evidence_event_ids"]
                for key in ("basis_event_id", "evidence_event_id")
                if key in args
            ):
                raise ValueError("assignment review changed its exact evidence basis")
        if expected is not None and args != expected:
            raise ValueError("assignment domain result changed its admitted arguments")
    # Domain projectors independently validate all science, selections and review
    # meaning; generic execution success never establishes scientific truth.
    read_planning_events(timeline)


class PlanningChoiceReconsidered(BaseModel):
    """Intent to revalidate an Agent choice; the new observation supplies science."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_type: Literal["PlanningChoiceReconsidered"] = "PlanningChoiceReconsidered"
    attempt_id: str
    prior_selection_event_id: str
    conflicting_evidence_event_id: str
    reason: str
    authority_event_id: str


def reconsider_recipe_offer(deps):
    """The reconsideration's bound selection and conflict, or its unmet reason."""
    events = planning_evidence(deps.journey)
    view = current_planning(deps.journey)
    planning = view.execution
    if planning is None:
        raise execution_planning_unopened(view)
    # The coordinator's target question bears on no recipe choice.
    pending = view.hitl.for_requirements(
        {
            item.requirement_id
            for item in view.requirements.requirements
            if item.kind != "scientific_investigation"
        }
    ).pending_requests
    if pending:
        raise PreconditionUnmet(
            "planning_question_pending", pending[0].request_event_id
        )
    if len(planning.selected_recipe_bindings) != 1:
        raise PreconditionUnmet(
            "recipe_selection_absent", len(planning.selected_recipe_bindings)
        )
    submitted = next(
        (e for e in events if e["event_type"] == "ExecutionProposalSubmitted"), None
    )
    if submitted is not None:
        raise PreconditionUnmet("execution_proposal_submitted", submitted["event_id"])
    selection_id = planning.selected_recipe_bindings[0].selection_event_id
    selected = next(e for e in events if e["event_id"] == selection_id)
    if selected["payload"]["authority"] != "agent_ranking":
        raise PreconditionUnmet(
            "recipe_selected_by_researcher", selected["payload"]["authority"]
        )
    conflicts = [
        e
        for e in events
        if e["event_type"] == "ResourceObservationRecorded"
        and e["payload"]["tool_input"]["work_item"]["recipe_event_id"] == selection_id
        and e["payload"]["result"]["outcome"] in {"missing", "observed"}
        and not any(
            c["compatibility"] == "compatible"
            for c in e["payload"]["result"]["candidates"]
        )
    ]
    if not conflicts:
        raise PreconditionUnmet("recipe_conflict_absent")
    conflict_id = conflicts[-1]["event_id"]
    if any(
        e["event_type"] == "PlanningChoiceReconsidered"
        and e["payload"]["conflicting_evidence_event_id"] == conflict_id
        for e in events
    ):
        raise PreconditionUnmet("recipe_conflict_already_reconsidered", conflict_id)
    return SimpleNamespace(
        constants={
            "prior_selection_event_id": selection_id,
            "conflicting_evidence_event_id": conflict_id,
        },
        enums={},
    )


def _validate_reconsiderations(timeline, selected):
    by_id = {e["event_id"]: e for e in timeline}
    owning_attempt_id = next(
        e["payload"]["attempt_id"]
        for e in selected
        if e["event_type"] == "AttemptStarted"
    )
    reconsideration_authorities = set()
    for fact in selected:
        if fact["event_type"] != "PlanningChoiceReconsidered":
            continue
        recorded = PlanningChoiceReconsidered.model_validate(
            {"event_type": fact["event_type"], **fact["payload"]}
        )
        authority = by_id.get(recorded.authority_event_id)
        if (
            recorded.attempt_id != owning_attempt_id
            or recorded.authority_event_id in reconsideration_authorities
            or authority not in timeline[: timeline.index(fact)]
        ):
            raise ValueError(
                "recipe reconsideration changed its owning admission ancestry"
            )
        reconsideration_authorities.add(recorded.authority_event_id)
        if (
            authority is None
            or authority["event_type"] != "ToolCallValidated"
            or authority["payload"]["attempt_id"] != recorded.attempt_id
            or authority["payload"]["tool_name"] != "reconsider_recipe_choice"
            or authority["payload"]["status"] != "valid"
            or authority["payload"]["arguments"]
            != {
                "prior_selection_event_id": recorded.prior_selection_event_id,
                "conflicting_evidence_event_id": recorded.conflicting_evidence_event_id,
                "reason": recorded.reason,
            }
        ):
            raise ValueError("recipe reconsideration changed its exact admitted reason")
        # Recorded relationships, not a re-derived offer: an Agent-ranked recipe
        # selection and retained resource evidence, compatible with none of its
        # candidates, against that exact selection; one reason per conflict.
        prefix = timeline[: timeline.index(fact)]
        prior = by_id.get(recorded.prior_selection_event_id)
        conflict = by_id.get(recorded.conflicting_evidence_event_id)
        if (
            prior not in prefix
            or conflict not in prefix
            or prior["event_type"] != "PlanningCandidateSelected"
            or prior["payload"].get("kind") != "recipe"
            or prior["payload"].get("authority") != "agent_ranking"
            or conflict["event_type"] != "ResourceObservationRecorded"
            or conflict["payload"]["tool_input"]["work_item"]["recipe_event_id"]
            != recorded.prior_selection_event_id
            or conflict["payload"]["result"]["outcome"] not in {"missing", "observed"}
            or any(
                candidate["compatibility"] == "compatible"
                for candidate in conflict["payload"]["result"]["candidates"]
            )
            or any(
                e["event_type"] == "PlanningChoiceReconsidered"
                and e["payload"]["conflicting_evidence_event_id"]
                == conflict["event_id"]
                for e in prefix
            )
        ):
            raise ValueError(
                "recipe reconsideration lacks its recorded conflict with an Agent choice"
            )


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "reconsider_recipe_choice",
        "planning_agent",
        (
            "prior_selection_event_id: str",
            "conflicting_evidence_event_id: str",
            "reason: str",
        ),
        "RecipeObservationRecorded",
        "compiled_capability_snapshot",
        "RecipeObservationRecorded",
        "ToolExecutionFailed",
        "error_type",
        ModelView(
            next_step={
                "observed": "Select the recipe again with `select_recipe_candidate`, "
                "or ask the researcher with `request_recipe_selection` when several "
                "fit.",
                "missing": "No other recipe fits; ask the researcher with "
                "`request_recipe_eligibility_assertion`.",
                "*": "The capability snapshot could not be read; the Attempt stops "
                "for the researcher.",
            },
            status=lambda payload: payload["result"]["outcome"],
        ),
        "legacy_tool_execution_failed_retained",
        stages=("execution_planning",),
        examples=(
            '{"reason": "The ranked recipe has no compatible resource profile on this capability."}',
        ),
        negative_results=NO_COMPATIBLE_RECIPE,
        preconditions=(
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            ToolPrecondition(
                "recipe_selection_absent",
                "selected recipes",
                "Reconsideration needs exactly one selected recipe.",
            ),
            ToolPrecondition(
                "execution_proposal_submitted",
                "execution proposal",
                "The execution proposal is already submitted; its recipe is not reopened here.",
            ),
            ToolPrecondition(
                "recipe_selected_by_researcher",
                "recipe selection",
                "The researcher selected this recipe; a researcher's selection is never reopened here.",
            ),
            ToolPrecondition(
                "recipe_conflict_absent",
                "resource evidence",
                "No resource lookup found the selected recipe without a compatible profile.",
            ),
            ToolPrecondition(
                "recipe_conflict_already_reconsidered",
                "resource evidence",
                "This conflict was already reconsidered once.",
            ),
            ToolPrecondition(
                "reconsideration_reason_invalid",
                "reason",
                "State the reason in 1 to 2000 characters without surrounding whitespace.",
            ),
        ),
    ),
    function_tool_contract(
        "record_proposal_requirements",
        "planning_agent",
        ("draft: ProposalInformationPlanDraft",),
        "ProposalRequirementsIdentified",
        "deterministic",
        "ProposalRequirementsIdentified",
        None,
        None,
        recorded_view(
            {
                "*": "The requirements are recorded; continue with the stage's "
                "Tools callable now."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("proposal_requirements",),
        examples=(
            '{"requirements": [{"kind": "sample_identity", "knowledge": {"state": "asserted", "value": "HG001", "quote": "HG001"}, "acquisition_route": "use_supplied_assertion", ...}, {"kind": "taxon_identity", "knowledge": {"state": "asserted", "value": 9606, "quote": "human (NCBI Taxonomy 9606)"}, ...}, ...]}: a typed value the request states, with the words that state it',
            '{"requirements": [{"kind": "sample_identity", "knowledge": {"state": "asserted", "value": "MM-04", "quote": "MM-04"}, ...}, {"kind": "taxon_identity", "knowledge": {"state": "asserted", "value": 10090, "quote": "mouse (NCBI Taxonomy 10090)"}, ...}, ...]}: the same for a mouse sample',
            '{"kind": "reference_assembly_identity", "knowledge": {"state": "to_find", "evidence_event_ids": ["<request event id>"]}, "acquisition_route": "validate_with_tool", ...}: a value the request does not state is found by a Tool',
            '{"kind": "scientific_investigation", "knowledge": {"state": "to_find", "evidence_event_ids": ["<request event id>"]}, "acquisition_route": "ask_planning_hitl", "validation": {"kind": "researcher_decision", ...}, ...}: a request that names no target and lists no registered request file; the researcher names the targets before execution. A registered request file is cited the same way, by its event id, and its item stands for its targets: no other to_find scientific_investigation is recorded beside it, except a target the researcher says a cited paper gives (to_find citing that paper\'s PaperRecorded, or the request while the paper is still to find, validate_with_tool)',
            '{"kind": "scientific_investigation", "knowledge": {"state": "asserted", "value": "rs429358", "quote": "rs429358"}, "investigation_purpose": "known_variant", ...}: one item per site the request asks about, with the rsID it gives',
            '{"kind": "scientific_investigation", "knowledge": {"state": "asserted", "quote": "V:2129317 C>T", "site": {"contig": "V", "position": 2129317, "ref": "C", "alt": "T"}}, "investigation_purpose": "known_variant", ...}: a site the request gives only by coordinates is asserted with its site typed from the quote; an rsID the request does not give is to_find, never a placeholder',
        ),
        preconditions=(
            CORRECTION_PENDING,
            ToolPrecondition(
                "proposal_inputs_uninspected",
                "input inspection",
                "The proposal requirements need a successful inspection of the registered inputs.",
            ),
            ToolPrecondition(
                "proposal_plan_already_recorded",
                "proposal plan",
                "The proposal requirements are already recorded; they are recorded once.",
            ),
            ToolPrecondition(
                "proposal_inputs_ineligible",
                "input inspection",
                "The proposal requirements need a successful inspection of the registered inputs.",
            ),
            ToolPrecondition(
                "context_evidence_ungrounded",
                "knowledge",
                "to_find knowledge may cite only the research request, the registered request file, the input registration, the inspection or a recorded paper's events: {allowed}.",
            ),
            ToolPrecondition(
                "request_assertion_ungrounded",
                "knowledge",
                "Its quote must copy words of the research request exactly, starting and ending on whole words.{file}",
            ),
            ToolPrecondition(
                "cited_paper_quote_ungrounded",
                "knowledge",
                "A cited_paper's quote must copy the researcher's exact words that cite the paper, on whole words, from their request, one of their answers or a registered request file's item.",
            ),
            ToolPrecondition(
                "cited_paper_evidence_ungrounded",
                "knowledge",
                "A cited_paper's evidence_event_ids may name only a PaperRecorded event of a paper retrieved with retrieve_paper: {allowed}; omit them when the paper is still to find.",
            ),
            ToolPrecondition(
                "asserted_value_not_in_quote",
                "knowledge",
                'Its typed value must be one its quote states: the quote may spell it differently (an rsID in capitals) but must not state another value. A site the request gives only by coordinates is asserted with its site typed, {"state": "asserted", "quote": "V:2129317 C>T", "site": {"contig": "V", "position": 2129317, "ref": "C", "alt": "T"}}; a value the request does not state is to_find: an rsID the request does not give is to_find, never a placeholder.',
            ),
            ToolPrecondition(
                "site_not_in_quote",
                "knowledge",
                "Its site must be one its quote writes: the sequence exactly as the quote writes it, the position as a number the quote writes, and REF and ALT as bases it writes.",
            ),
            ToolPrecondition(
                "input_observation_ungrounded",
                "knowledge",
                "The observed input role and fact must name a value the inspection recorded: input_role one of {roles}; fact one of {facts}.",
            ),
            ToolPrecondition(
                "registered_inputs_not_fully_represented",
                "requirements",
                "Every registered input needs its own observed registered_input requirement; still missing for: {roles}.",
            ),
            ToolPrecondition(
                "proposal_requirement_kinds_incomplete",
                "requirements",
                "The requirements must include one of each required kind (a scientific_investigation at least once); still missing: {missing}.",
            ),
            ToolPrecondition(
                "registered_request_uncited",
                "requirements",
                "Each registered request file lists targets: record one scientific_investigation to_find whose evidence_event_ids cite that file's event id; still uncited: {event_ids}.",
            ),
            ToolPrecondition(
                "registered_requests_cited_together",
                "knowledge",
                "One scientific_investigation item stands for one registered request file, and this item cites {event_ids}: cite each file by its own item.",
            ),
            ToolPrecondition(
                "registered_request_cited_twice",
                "requirements",
                "One scientific_investigation item stands for each registered request file, and more than one item cites {event_ids}: cite each file by one item.",
            ),
            ToolPrecondition(
                "target_question_beside_registered_request",
                "knowledge",
                "A registered request file lists this request's targets ({event_ids}), so no scientific_investigation asks the researcher to name them: the file's own to_find item stands for its targets, and each target the request itself names is its own asserted scientific_investigation; a target the researcher says a cited paper gives is to_find from that paper, validate_with_tool.",
            ),
            ToolPrecondition(
                "registered_request_not_a_target",
                "knowledge",
                "Only a scientific_investigation to_find item cites the registered request file; every other requirement cites the research request, the input registration or the inspection events.",
            ),
            ToolPrecondition(
                "registered_request_quoted",
                "requirements",
                "The item citing a registered request file stands for the file's targets and takes no quote: record each target the request itself names as its own asserted scientific_investigation.",
            ),
        ),
    ),
    function_tool_contract(
        "record_answered_targets",
        "planning_agent",
        ("draft: AnsweredTargetsDraft",),
        "TargetAnswerRead",
        "deterministic",
        "TargetAnswerRead",
        None,
        None,
        recorded_view(
            {
                "*": "The answer's targets are read; continue with the stage's "
                "Tools callable now."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("target_answer",),
        examples=(
            '{"targets": [{"state": "asserted", "quote": "V:2129317 C>T on WBcel235", "site": {"contig": "V", "position": 2129317, "ref": "C", "alt": "T"}, "stated_version": "WBcel235"}]}: a site the answer names, typed, with the version it states',
            '{"targets": [{"state": "asserted", "value": "rs429358", "quote": "rs429358"}]}: an rsID the answer names',
            '{"targets": []}: the answer names no target; the researcher is asked again',
        ),
        preconditions=(
            ToolPrecondition(
                "target_answer_not_pending",
                "target answer",
                "No researcher answer to a target question waits to be read.",
            ),
            ToolPrecondition(
                "answer_quote_ungrounded",
                "targets",
                "Each quote must copy words of the researcher's answer exactly, starting and ending on whole words.",
            ),
            ToolPrecondition(
                "asserted_value_not_in_quote",
                "targets",
                "An rsID must be one its quote states; a site the answer gives only by coordinates has no rsID value.",
            ),
            ToolPrecondition(
                "site_not_in_quote",
                "targets",
                "A site must be one its quote writes: the sequence exactly as the quote writes it, the position as a number the quote writes, and REF and ALT as bases it writes.",
            ),
        ),
    ),
)
