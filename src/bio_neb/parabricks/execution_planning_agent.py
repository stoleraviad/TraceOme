"""Recipe/resource Tools the planning assignment offers through its one Agent."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal

from pydantic import Field

from bio_neb.agentic_journey.action_admission import ActionAdmissionSession
from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    ToolExecutionFailed,
)
from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.planning_preconditions import (
    CORRECTION_PENDING,
    EXECUTION_NOT_OPEN,
    NO_COMPATIBLE_RECIPE,
    QUESTION_PENDING,
)
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
from bio_neb.failure import ProviderError
from bio_neb.journey import Journey, canonical_json
from bio_neb.parabricks.execution_planning_action_policy import (
    ExecutionPlanningActionName as ActionName,
)
from bio_neb.parabricks.execution_planning_action_policy import (
    execution_planning_selected_disposition as _selected_disposition,
)
from bio_neb.parabricks.execution_planning_journey import (
    ExecutionPlanningProjection,
    OrganismPloidyJudgment,
    PlanningCandidateSelected,
    PloidyJudgment,
    RecipeObservationRecorded,
    ResourceObservationRecorded,
)
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlCandidate,
    PlanningHitlRequested,
)
from bio_neb.parabricks.recipe_resources import (
    ListRecipesInput,
    ListRecipesTool,
    LookupResourcesInput,
    LookupResourcesTool,
    RecipeCandidate,
    RecipeEligibilityAssertion,
    RecipeResourceAdapter,
    candidate_disposition_id,
    recipe_eligibility_assertion_id,
    resource_recipe_dependency_id,
)


@dataclass
class ExecutionPlanningDependencies:
    parent_activity: Activity
    attempt_id: str
    journey: Journey
    adapter: RecipeResourceAdapter
    failure_source: AttemptFailureSource
    recipe_tool: ListRecipesTool = field(default_factory=ListRecipesTool)
    resource_tool: LookupResourcesTool = field(default_factory=LookupResourcesTool)
    action_admission_session: ActionAdmissionSession | None = None
    observed_tool_session: ObservedToolSession | None = None

    def __post_init__(self) -> None:
        if str(uuid.UUID(self.attempt_id)) != self.attempt_id:
            raise ValueError("attempt_id must be canonical")


def _outcome(payload):
    return payload["result"]["outcome"]


def _operational_capability_result(returned):
    result = returned["payload"]["result"]
    if result["outcome"] != "operational_failure":
        return None
    return ProviderError(
        result["failure_class"],
        result["failure_code"],
        evidence={
            "original_failure_class": result["original_failure_class"],
            "contact_attempt": result["contact_attempt"],
        },
    )


def _projection(deps: ExecutionPlanningDependencies) -> ExecutionPlanningProjection:
    from bio_neb.parabricks.execution_planning_action_policy import (
        execution_planning_projection,
    )

    return execution_planning_projection(deps)


def list_compatible_analysis_recipes(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    input_set_id: str,
    assembly_accession: str,
    input_assay: str | None,
    input_read_layout: Literal["paired", "single"] | None,
    input_assertion_event_id: str | None,
) -> dict[str, Any]:
    return record_recipe_observation(
        context,
        work_item_id,
        input_set_id,
        assembly_accession,
        input_assay,
        input_read_layout,
        input_assertion_event_id,
    )


def record_recipe_observation(
    context,
    work_item_id,
    input_set_id,
    assembly_accession,
    input_assay,
    input_read_layout,
    input_assertion_event_id,
    *,
    operation_name="list_compatible_analysis_recipes",
):
    """One exact deterministic operation, also used for supported reconsideration."""
    deps = context.deps
    stage_result_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal stage_result_event
        projection = _projection(deps)
        work = projection.recipe_work_item
        assertion = (
            RecipeEligibilityAssertion(
                assay=input_assay,
                read_layout=input_read_layout,
                assertion_event_id=input_assertion_event_id,
            )
            if input_assay is not None
            and input_read_layout is not None
            and input_assertion_event_id is not None
            else None
        )
        tool_input = ListRecipesInput(work_item=work, eligibility_assertion=assertion)
        if tool_input != projection.recipe_tool_input:
            raise ValueError("recipe eligibility assertion changed after validation")
        activity = deps.parent_activity.start_child(
            adapter="compiled_capability_snapshot",
            stage="list_compatible_analysis_recipes",
        )
        try:
            result = deps.recipe_tool.list(
                tool_input,
                expected_input=projection.recipe_tool_input,
                adapter=deps.adapter,
            )
            fact = RecipeObservationRecorded(
                attempt_id=deps.attempt_id, tool_input=tool_input, result=result
            )
            accepted = activity.record(fact)
            stage_result_event = accepted
            activity.succeeded(accepted)
        except Exception as error:
            failed = activity.record(
                ToolExecutionFailed(
                    attempt_id=deps.attempt_id,
                    tool_name=operation_name,
                    error_type=type(error).__name__,
                )
            )
            deps.failure_source.capture(failed)
            activity.failed(failed)
            raise
        return accepted

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="compiled_capability_snapshot",
            operational_failure=_operational_capability_result,
            stage_result_event=lambda _result: (
                stage_result_event if stage_result_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
            stage_failure_event=lambda _error: (
                {
                    "event_id": deps.failure_source.event_id,
                    "event_type": deps.failure_source.event_type,
                }
                if deps.failure_source.event_type == "ToolExecutionFailed"
                and deps.failure_source.event_id is not None
                else None
            ),
        ),
    )


def lookup_compatible_resource_profiles(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    recipe_candidate_id: str,
) -> dict[str, Any]:
    deps = context.deps
    stage_result_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal stage_result_event
        projection = _projection(deps)
        work = projection.resource_work_item
        assert work is not None
        tool_input = LookupResourcesInput(work_item=work)
        activity = deps.parent_activity.start_child(
            adapter="compiled_capability_snapshot",
            stage="lookup_compatible_resource_profiles",
        )
        try:
            result = deps.resource_tool.lookup(
                tool_input, expected_work_item=work, adapter=deps.adapter
            )
            fact = ResourceObservationRecorded(
                attempt_id=deps.attempt_id, tool_input=tool_input, result=result
            )
            accepted = activity.record(fact)
            stage_result_event = accepted
            activity.succeeded(accepted)
        except Exception as error:
            failed = activity.record(
                ToolExecutionFailed(
                    attempt_id=deps.attempt_id,
                    tool_name="lookup_compatible_resource_profiles",
                    error_type=type(error).__name__,
                )
            )
            deps.failure_source.capture(failed)
            activity.failed(failed)
            raise
        return accepted

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="compiled_capability_snapshot",
            operational_failure=_operational_capability_result,
            stage_result_event=lambda _result: (
                stage_result_event if stage_result_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
            stage_failure_event=lambda _error: (
                {
                    "event_id": deps.failure_source.event_id,
                    "event_type": deps.failure_source.event_type,
                }
                if deps.failure_source.event_type == "ToolExecutionFailed"
                and deps.failure_source.event_id is not None
                else None
            ),
        ),
    )


def _selection_candidates(
    projection: ExecutionPlanningProjection, kind: Literal["recipe", "resource"]
):
    fact = projection.facts[-1]
    result = fact.result
    return tuple(
        PlanningHitlCandidate(
            candidate_id=candidate_disposition_id(
                kind, fact.work_item_id, item["candidate_id"]
            ),
            asserted_value=item["candidate_id"],
            display_label=item["candidate_id"][:300],
            evidence_event_ids=(fact.source_event_id,),
        )
        for item in result["candidates"]
        if item["compatibility"] == "compatible"
    )


def _execute_planning_request(
    context: ToolContext[ExecutionPlanningDependencies],
    make_fact: Callable[[], PlanningHitlRequested],
    *,
    stage: Literal[
        "request_recipe_eligibility_assertion",
        "request_recipe_selection",
        "request_resource_selection",
        "request_resource_recipe_dependency",
    ],
) -> dict[str, Any]:
    request_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal request_event
        deps = context.deps
        fact = make_fact()
        activity = deps.parent_activity.start_child(
            adapter="deterministic",
            stage=stage,
        )
        accepted = activity.record(fact)
        request_event = accepted
        activity.succeeded(accepted)
        return accepted

    return context.deps.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="deterministic",
            stage_result_event=lambda _result: (
                request_event if request_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
        ),
    )


def request_recipe_eligibility_assertion(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    basis_event_id: str,
):
    """The answer is retained as a researcher assertion for the next recipe listing."""
    deps = context.deps

    def make_fact() -> PlanningHitlRequested:
        projection = _projection(deps)
        work = projection.recipe_work_item
        requirement = next(
            item
            for item in projection.requirements
            if item.requirement_id == work.grounding.requirement_id
        )
        fact_view = next(
            item for item in projection.facts if item.source_event_id == basis_event_id
        )
        candidates = tuple(
            {
                recipe_eligibility_assertion_id(
                    work_item_id, RecipeCandidate.model_validate(item)
                ): PlanningHitlCandidate(
                    candidate_id=recipe_eligibility_assertion_id(
                        work_item_id, RecipeCandidate.model_validate(item)
                    ),
                    asserted_value=canonical_json(
                        {"assay": item["assay"], "read_layout": item["read_layout"]}
                    ),
                    display_label=f"{item['assay']} / {item['read_layout']}",
                    evidence_event_ids=(basis_event_id,),
                )
                for item in fact_view.result["candidates"]
                if item["compatibility"] == "ambiguous"
            }.values()
        )
        return PlanningHitlRequested(
            attempt_id=deps.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            issue="Input assay and read layout are not verified by registered file metadata.",
            question=(
                "Which exact assay and read layout should be retained as a user assertion?"
            ),
            evidence_event_ids=(basis_event_id,),
            candidates=candidates,
        )

    return _execute_planning_request(
        context,
        make_fact,
        stage="request_recipe_eligibility_assertion",
    )


def _request_selection(
    context, kind: Literal["recipe", "resource"], work_item_id: str, basis_event_id: str
):
    deps = context.deps
    stage = (
        "request_recipe_selection" if kind == "recipe" else "request_resource_selection"
    )

    def make_fact() -> PlanningHitlRequested:
        projection = _projection(deps)
        works = (
            projection.recipe_work_items
            if kind == "recipe"
            else projection.resource_work_items
        )
        requirement_id = next(
            item.grounding.requirement_id
            for item in works
            if item.work_item_id == work_item_id
        )
        requirement = next(
            item
            for item in projection.requirements
            if item.requirement_id == requirement_id
        )
        return PlanningHitlRequested(
            attempt_id=deps.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=requirement.requirement_id,
            subject_key=requirement.subject_key,
            issue=f"Multiple compatible {kind} candidates require an exact disposition.",
            question=(
                f"Which exact {kind} candidate should be revalidated? "
                "Selection is not compatibility proof."
            ),
            evidence_event_ids=(basis_event_id,),
            candidates=_selection_candidates(projection, kind),
        )

    return _execute_planning_request(context, make_fact, stage=stage)


def request_recipe_selection(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    basis_event_id: str,
):
    return _request_selection(context, "recipe", work_item_id, basis_event_id)


def request_resource_selection(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    basis_event_id: str,
):
    return _request_selection(context, "resource", work_item_id, basis_event_id)


def request_resource_recipe_dependency(
    context: ToolContext[ExecutionPlanningDependencies],
    requirement_id: str,
):
    deps = context.deps

    def make_fact() -> PlanningHitlRequested:
        projection = _projection(deps)
        requirement = next(
            item
            for item in projection.requirements
            if item.requirement_id == requirement_id
        )
        candidates = tuple(
            PlanningHitlCandidate(
                candidate_id=resource_recipe_dependency_id(
                    requirement_id,
                    binding.requirement_id,
                    binding.selection_event_id,
                ),
                asserted_value=binding.requirement_id,
                display_label=f"{binding.subject_key}: {binding.candidate.recipe_id}"[
                    :300
                ],
                evidence_event_ids=(binding.selection_event_id,),
            )
            for binding in projection.selected_recipe_bindings
        )
        return PlanningHitlRequested(
            attempt_id=deps.attempt_id,
            plan_event_id=projection.plan_event_id,
            plan_id=projection.plan_id,
            requirement_id=requirement_id,
            subject_key=requirement.subject_key,
            issue=(
                "Multiple selected recipe subjects can own this resource requirement."
            ),
            question=(
                "Which exact selected recipe requirement owns this resource profile?"
            ),
            evidence_event_ids=tuple(
                item.selection_event_id for item in projection.selected_recipe_bindings
            ),
            candidates=candidates,
        )

    return _execute_planning_request(
        context,
        make_fact,
        stage="request_resource_recipe_dependency",
    )


def _select_candidate(
    context,
    kind: Literal["recipe", "resource"],
    work_item_id: str,
    candidate_id: str,
    basis_event_id: str,
    organism_ploidy: OrganismPloidyJudgment | None = None,
):
    action: ActionName = (
        "select_recipe_candidate" if kind == "recipe" else "select_resource_candidate"
    )
    deps = context.deps
    selection_event: dict[str, Any] | None = None

    def typed_operation() -> dict[str, Any]:
        nonlocal selection_event
        activity = deps.parent_activity.start_child(
            adapter="deterministic",
            stage=action,
        )
        disposition = _selected_disposition(deps, kind, work_item_id)
        researcher = disposition is not None
        if researcher and disposition.assertion.asserted_value != candidate_id:
            raise ValueError("researcher selection changed its exact candidate")
        fact = PlanningCandidateSelected(
            attempt_id=deps.attempt_id,
            kind=kind,
            work_item_id=work_item_id,
            candidate_id=candidate_id,
            basis_event_id=basis_event_id,
            authority="researcher_selection" if researcher else "agent_ranking",
            authority_event_id=(
                disposition.response_event_id
                if researcher
                else deps.action_admission_session.admitted_validation_event_id(
                    context.tool_call_id
                )
            ),
            organism_ploidy=organism_ploidy,
        )
        accepted = activity.record(fact)
        selection_event = accepted
        activity.succeeded(accepted)
        return accepted

    return deps.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="deterministic",
            stage_result_event=lambda _result: (
                selection_event if selection_event is not None else {}
            ),
            artifact_digests=lambda _result: (),
        ),
    )


def select_recipe_candidate(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    candidate_id: str,
    basis_event_id: str,
    organism_ploidy: Annotated[
        PloidyJudgment,
        Field(
            description=(
                "Your judgment of the Run organism's ploidy, from the organism record "
                "you cite: diploid, likely_non_diploid (a haploid bacterium or yeast, "
                "a polyploid plant) or unknown. An assembly's assembly_type describes "
                "how the assembly represents the genome, not the organism's ploidy."
            )
        ),
    ],
    organism_record_event_id: Annotated[
        str,
        Field(description="The organism record (its event id) the judgment rests on."),
    ],
    organism_quote: Annotated[
        str,
        Field(
            min_length=1,
            max_length=300,
            description=(
                "The words of that record you read, copied exactly: the organism's "
                "name or a lineage name."
            ),
        ),
    ],
):
    """Rank among several only when the evidence supports one; otherwise ask the
    researcher. Calling is diploid; your ploidy judgment is shown in the Execution
    Review, which asks the researcher before a likely non-diploid Run.
    """
    return _select_candidate(
        context,
        "recipe",
        work_item_id,
        candidate_id,
        basis_event_id,
        OrganismPloidyJudgment(
            judgment=organism_ploidy,
            record_event_id=organism_record_event_id,
            quote=organism_quote,
        ),
    )


def select_resource_candidate(
    context: ToolContext[ExecutionPlanningDependencies],
    work_item_id: str,
    candidate_id: str,
    basis_event_id: str,
):
    return _select_candidate(
        context, "resource", work_item_id, candidate_id, basis_event_id
    )


_NO_COMPATIBLE_RESOURCE = (
    NegativeResult("ResourceObservationRecorded", "outcome", ("missing",)),
    NegativeResult("ResourceObservationRecorded", "compatibility", ("incompatible",)),
)


_EXECUTION_STEP_NOT_CURRENT = ToolPrecondition(
    "execution_step_not_current",
    "current execution-planning step",
    "This Tool does not act on that step; the Tool for it is {tools}.",
)


def _step_after(*tools: str) -> ToolPrecondition:
    """The step precondition of a Tool whose step follows those ``tools``' step."""
    return replace(_EXECUTION_STEP_NOT_CURRENT, met_by=tools)


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "list_compatible_analysis_recipes",
        "planning_agent",
        (
            "work_item_id: str",
            "input_set_id: str",
            "assembly_accession: str",
            "input_assay: str | None",
            "input_read_layout: Literal['paired', 'single'] | None",
            "input_assertion_event_id: str | None",
        ),
        "RecipeObservation",
        "compiled_capability_snapshot",
        "RecipeObservationRecorded",
        "ToolExecutionFailed",
        "error_type",
        ModelView(
            next_step={
                "observed": "Select the recipe with `select_recipe_candidate`, or ask the researcher with `request_recipe_selection` when several fit.",
                "missing": "No recipe fits these inputs; ask the researcher with `request_recipe_eligibility_assertion`.",
                "*": "The capability snapshot could not be read; the Attempt stops "
                "for the researcher.",
            },
            status=_outcome,
        ),
        "legacy_tool_execution_failed_retained",
        stages=("execution_planning",),
        examples=("{} (the server supplies the work item)",),
        negative_results=NO_COMPATIBLE_RECIPE,
        preconditions=(
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _EXECUTION_STEP_NOT_CURRENT,
        ),
    ),
    function_tool_contract(
        "lookup_compatible_resource_profiles",
        "planning_agent",
        ("work_item_id: str", "recipe_candidate_id: str"),
        "ResourceObservation",
        "compiled_capability_snapshot",
        "ResourceObservationRecorded",
        "ToolExecutionFailed",
        "error_type",
        ModelView(
            next_step={
                "observed": "Select the resource with `select_resource_candidate`, or ask the researcher with `request_resource_selection` when several fit.",
                "missing": "No resource profile fits the selected recipe; ask the researcher with `request_resource_recipe_dependency`.",
                "*": "The capability snapshot could not be read; the Attempt stops "
                "for the researcher.",
            },
            status=_outcome,
        ),
        "legacy_tool_execution_failed_retained",
        stages=("execution_planning",),
        examples=("{} (the server supplies the work item and recipe)",),
        negative_results=_NO_COMPATIBLE_RESOURCE,
        preconditions=(
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("select_recipe_candidate"),
        ),
    ),
    function_tool_contract(
        "request_recipe_eligibility_assertion",
        "planning_agent",
        ("work_item_id: str", "basis_event_id: str"),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=("{} (the server supplies the work item and its basis)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("list_compatible_analysis_recipes"),
        ),
    ),
    function_tool_contract(
        "request_recipe_selection",
        "planning_agent",
        ("work_item_id: str", "basis_event_id: str"),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=("{} (the server supplies the work item and its basis)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("list_compatible_analysis_recipes"),
        ),
    ),
    function_tool_contract(
        "request_resource_selection",
        "planning_agent",
        ("work_item_id: str", "basis_event_id: str"),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=("{} (the server supplies the work item and its basis)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("lookup_compatible_resource_profiles"),
        ),
    ),
    function_tool_contract(
        "request_resource_recipe_dependency",
        "planning_agent",
        ("requirement_id: str",),
        "PlanningHitlRequested",
        "deterministic",
        "PlanningHitlRequested",
        None,
        None,
        asked_view(),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=("{} (the server supplies the requirement)",),
        opens_question=True,
        preconditions=(
            QUESTION_OPENED,
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("lookup_compatible_resource_profiles"),
        ),
    ),
    function_tool_contract(
        "select_recipe_candidate",
        "planning_agent",
        (
            "work_item_id: str",
            "candidate_id: str",
            "basis_event_id: str",
            "organism_ploidy: str",
            "organism_record_event_id: str",
            "organism_quote: str",
        ),
        "PlanningCandidateSelected",
        "deterministic",
        "PlanningCandidateSelected",
        None,
        None,
        recorded_view(
            {
                "*": "Look up the compatible resources with `lookup_compatible_resource_profiles`."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=(
            '{"organism_ploidy": "diploid", "organism_record_event_id": "<an offered organism record>", "organism_quote": "Homo sapiens"} when the server binds the only recipe; add "candidate_id": "<one offered recipe candidate>" when several are offered',
        ),
        preconditions=(
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("list_compatible_analysis_recipes"),
            ToolPrecondition(
                "candidate_id_ungrounded",
                "candidate_id",
                "Choose one of the offered compatible recipe candidates: {offered}.",
            ),
            ToolPrecondition(
                "organism_record_missing",
                "organism record",
                "This Run has kept no organism record to cite for the ploidy judgment; the reference stage's taxon and assembly records are cited here.",
            ),
            ToolPrecondition(
                "organism_record_event_id_ungrounded",
                "organism_record_event_id",
                "Cite one of this Run's organism records: {offered}.",
            ),
            ToolPrecondition(
                "organism_quote_not_in_record",
                "organism_quote",
                "The quote is not words of the cited organism record; copy the organism's name or a lineage name exactly as that record writes it.",
            ),
        ),
    ),
    function_tool_contract(
        "select_resource_candidate",
        "planning_agent",
        ("work_item_id: str", "candidate_id: str", "basis_event_id: str"),
        "PlanningCandidateSelected",
        "deterministic",
        "PlanningCandidateSelected",
        None,
        None,
        recorded_view(
            {
                "*": "Execution planning is complete once every selection is recorded; continue with the Tools callable now."
            }
        ),
        "generic_mechanical_failure_only",
        stages=("execution_planning",),
        examples=("{} (the server binds the compatible profile)",),
        preconditions=(
            CORRECTION_PENDING,
            EXECUTION_NOT_OPEN,
            QUESTION_PENDING,
            _step_after("lookup_compatible_resource_profiles"),
        ),
    ),
)


__all__ = [
    "TOOL_ROWS",
    "ExecutionPlanningDependencies",
    "list_compatible_analysis_recipes",
    "lookup_compatible_resource_profiles",
    "record_recipe_observation",
    "request_recipe_eligibility_assertion",
    "request_recipe_selection",
    "request_resource_recipe_dependency",
    "request_resource_selection",
    "select_recipe_candidate",
    "select_resource_candidate",
]
