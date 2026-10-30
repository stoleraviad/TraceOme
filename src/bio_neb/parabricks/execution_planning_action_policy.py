"""Execution-planning Tools' server-bound values and argument refusals, per Tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, Protocol, get_args

from pydantic import JsonValue

from bio_neb.agentic_journey.action_policy import PreconditionUnmet
from bio_neb.journey import Journey
from bio_neb.parabricks.execution_planning_journey import (
    ORGANISM_RECORD_EVENTS,
    ExecutionPlanningProjection,
    project_execution_planning,
    quoted_in_record,
)
from bio_neb.parabricks.recipe_resources import candidate_disposition_id

ExecutionPlanningActionName = Literal[
    "list_compatible_analysis_recipes",
    "request_recipe_selection",
    "select_recipe_candidate",
    "lookup_compatible_resource_profiles",
    "request_resource_selection",
    "select_resource_candidate",
    "request_resource_recipe_dependency",
    "request_recipe_eligibility_assertion",
]


class ExecutionPlanningActionPolicyDependencies(Protocol):
    journey: Journey


class ExecutionPlanningAction(Protocol):
    tool_name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionToolValues:
    constants: dict[str, JsonValue]
    enums: dict[str, tuple[JsonValue, ...]] = field(default_factory=dict)


# Argument structure is each Tool's own Pydantic model; only the names are shared.
EXECUTION_PLANNING_TOOLS: tuple[ExecutionPlanningActionName, ...] = get_args(
    ExecutionPlanningActionName
)


def execution_planning_projection(
    dependencies: ExecutionPlanningActionPolicyDependencies,
) -> ExecutionPlanningProjection:
    if hasattr(dependencies.journey, "read_view"):
        from bio_neb.application.planning_read import read_planning

        projected = read_planning(dependencies.journey).execution
        if projected is None:
            raise ValueError("execution capability lacks its checked prerequisites")
        return projected
    return project_execution_planning(dependencies.journey.read())


def execution_planning_selected_disposition(
    dependencies: ExecutionPlanningActionPolicyDependencies,
    kind: Literal["recipe", "resource"],
    work_item_id: str,
):
    from bio_neb.application.planning_read import current_planning

    hitl = current_planning(dependencies.journey).hitl
    projection = execution_planning_projection(dependencies)
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
    prefix = f"{kind}-candidate-disposition:"
    return next(
        (
            receipt
            for receipt in reversed(hitl.receipts)
            if receipt.plan_id == projection.plan_id
            and receipt.requirement_id == requirement_id
            and receipt.assertion.selected_candidate_id is not None
            and receipt.assertion.selected_candidate_id.startswith(prefix)
            and candidate_disposition_id(
                kind, work_item_id, receipt.assertion.asserted_value or ""
            )
            == receipt.assertion.selected_candidate_id
        ),
        None,
    )


# The current step as the model reads it, never the projection's status name.
_STEPS = {
    "recipe_validation_pending": "recipe validation",
    "recipe_selection_pending": "recipe selection",
    "resource_validation_pending": "resource validation",
    "resource_selection_pending": "resource selection",
    "planning_hitl_pending": "a pending researcher question",
    "typed_failure": "a failed planning step",
    "execution_planning_ready": "execution planning complete",
}


def _step_not_current(projection):
    return PreconditionUnmet("execution_step_not_current", _STEPS[projection.status])


def _bound(constants, enums=None) -> ExecutionToolValues:
    return ExecutionToolValues(constants, enums or {})


def _recipe_step(deps, projection):
    """The current recipe selection step: its work item, basis and choices."""
    if projection.status != "recipe_selection_pending":
        raise _step_not_current(projection)
    work = projection.recipe_work_item
    basis = projection.facts[-1]
    return SimpleNamespace(
        work=work,
        basis_event_id=basis.source_event_id,
        requirement=next(
            item
            for item in projection.requirements
            if item.requirement_id == work.grounding.requirement_id
        ),
        disposition=execution_planning_selected_disposition(
            deps, "recipe", work.work_item_id
        ),
        compatible=[
            item
            for item in basis.result["candidates"]
            if item["compatibility"] == "compatible"
        ],
    )


def _resource_step(deps, projection):
    """The current resource selection step for its owning recipe's work item."""
    if (
        projection.status != "resource_selection_pending"
        or projection.resource_work_item is None
    ):
        raise _step_not_current(projection)
    work = projection.resource_work_item
    basis = projection.facts[-1]
    return SimpleNamespace(
        work=work,
        basis_event_id=basis.source_event_id,
        disposition=execution_planning_selected_disposition(
            deps, "resource", work.work_item_id
        ),
        compatible=[
            item
            for item in basis.result["candidates"]
            if item["compatibility"] == "compatible"
        ],
    )


def _adaptive(deps) -> bool:
    """Whether the planning Agent itself may rank several compatible recipes."""
    from bio_neb.application.planning_read import planning_evidence

    return (
        next(
            (
                e["payload"].get("agent_role")
                for e in reversed(planning_evidence(deps.journey))
                if e["event_type"] == "AttemptStarted"
            ),
            None,
        )
        == "planning_agent"
    )


def _list_recipes_values(deps, projection):
    if projection.status != "recipe_validation_pending":
        raise _step_not_current(projection)
    work = projection.recipe_work_item
    assertion = projection.recipe_tool_input.eligibility_assertion
    return _bound(
        {
            "work_item_id": work.work_item_id,
            "input_set_id": work.input_set_id,
            "assembly_accession": work.assembly_accession,
            "input_assay": assertion.assay if assertion is not None else None,
            "input_read_layout": (
                assertion.read_layout if assertion is not None else None
            ),
            "input_assertion_event_id": (
                assertion.assertion_event_id if assertion is not None else None
            ),
        }
    )


def _eligibility_question_values(deps, projection):
    step = _recipe_step(deps, projection)
    if "request_recipe_eligibility_assertion" not in step.requirement.next_actions:
        raise _step_not_current(projection)
    return _bound(
        {"work_item_id": step.work.work_item_id, "basis_event_id": step.basis_event_id}
    )


def _organism_records(deps) -> dict[str, dict[str, Any]]:
    """The Run's organism records the reference stage kept, by event id (#469 Q2)."""
    from bio_neb.application.planning_read import planning_evidence

    return {
        event["event_id"]: event
        for event in planning_evidence(deps.journey)
        if event["event_type"] in ORGANISM_RECORD_EVENTS
    }


def _select_recipe_values(deps, projection):
    step = _recipe_step(deps, projection)
    bound = {
        "work_item_id": step.work.work_item_id,
        "basis_event_id": step.basis_event_id,
    }
    # The ploidy judgment cites an organism record by membership.
    records = {"organism_record_event_id": tuple(_organism_records(deps))}
    if not records["organism_record_event_id"]:
        raise PreconditionUnmet("organism_record_missing")
    if step.disposition is not None:
        return _bound(
            {**bound, "candidate_id": step.disposition.assertion.asserted_value},
            records,
        )
    if len(step.compatible) == 1:
        return _bound(
            {**bound, "candidate_id": step.compatible[0]["candidate_id"]}, records
        )
    if len(step.compatible) > 1 and _adaptive(deps):
        return _bound(
            bound,
            {
                "candidate_id": tuple(c["candidate_id"] for c in step.compatible),
                **records,
            },
        )
    raise _step_not_current(projection)


def _recipe_question_values(deps, projection):
    step = _recipe_step(deps, projection)
    if step.disposition is not None or len(step.compatible) <= 1:
        raise _step_not_current(projection)
    return _bound(
        {"work_item_id": step.work.work_item_id, "basis_event_id": step.basis_event_id}
    )


def _lookup_resources_values(deps, projection):
    if projection.status != "resource_validation_pending":
        raise _step_not_current(projection)
    work = projection.resource_work_item
    assert work is not None
    return _bound(
        {
            "work_item_id": work.work_item_id,
            "recipe_candidate_id": work.recipe_candidate.candidate_id,
        }
    )


def _resource_dependency_values(deps, projection):
    if (
        projection.status != "resource_selection_pending"
        or projection.resource_work_item is not None
    ):
        raise _step_not_current(projection)
    requirement = next(
        item
        for item in projection.requirements
        if item.kind == "resource_profile"
        and "request_resource_recipe_dependency" in item.next_actions
    )
    return _bound({"requirement_id": requirement.requirement_id})


def _select_resource_values(deps, projection):
    step = _resource_step(deps, projection)
    bound = {
        "work_item_id": step.work.work_item_id,
        "basis_event_id": step.basis_event_id,
    }
    if step.disposition is not None:
        return _bound(
            {**bound, "candidate_id": step.disposition.assertion.asserted_value}
        )
    if len(step.compatible) == 1:
        return _bound({**bound, "candidate_id": step.compatible[0]["candidate_id"]})
    raise _step_not_current(projection)


def _resource_question_values(deps, projection):
    step = _resource_step(deps, projection)
    if step.disposition is not None or len(step.compatible) <= 1:
        raise _step_not_current(projection)
    return _bound(
        {"work_item_id": step.work.work_item_id, "basis_event_id": step.basis_event_id}
    )


_EXECUTION_VALUES = {
    "list_compatible_analysis_recipes": _list_recipes_values,
    "request_recipe_eligibility_assertion": _eligibility_question_values,
    "select_recipe_candidate": _select_recipe_values,
    "request_recipe_selection": _recipe_question_values,
    "lookup_compatible_resource_profiles": _lookup_resources_values,
    "request_resource_recipe_dependency": _resource_dependency_values,
    "select_resource_candidate": _select_resource_values,
    "request_resource_selection": _resource_question_values,
}


def execution_tool_values(
    dependencies: ExecutionPlanningActionPolicyDependencies,
    action: ExecutionPlanningActionName,
) -> ExecutionToolValues:
    """An execution-planning Tool's server-bound values now.

    Raises ``PreconditionUnmet`` with the registry's reason when the Tool does not
    act on the current step; admission refuses the call before contact.
    """

    projection = execution_planning_projection(dependencies)
    if projection.status == "planning_hitl_pending":
        raise PreconditionUnmet("planning_question_pending", projection.status)
    try:
        return _EXECUTION_VALUES[action](dependencies, projection)
    except PreconditionUnmet as unmet:
        if unmet.code != "execution_step_not_current":
            raise
        # The refusal names the Tools that do act on the current step.
        raise PreconditionUnmet(
            unmet.code,
            unmet.value,
            lambda: {"tools": _current_step_tools(dependencies, projection)},
        ) from None


def _current_step_tools(dependencies, projection) -> tuple[str, ...]:
    """The execution-planning Tools whose own step guard admits them now."""
    tools = []
    for name in EXECUTION_PLANNING_TOOLS:
        try:
            _EXECUTION_VALUES[name](dependencies, projection)
        except PreconditionUnmet:
            continue
        tools.append(name)
    return tuple(tools)


def execution_tool_errors(
    dependencies: ExecutionPlanningActionPolicyDependencies,
    action: ExecutionPlanningAction,
) -> tuple[str, ...]:
    """The refusals of a call's agent-chosen arguments against the current step.

    Admission already filled and checked the bound values and the Tool's
    argument structure.
    """

    try:
        offer = execution_tool_values(
            dependencies,
            action.tool_name,  # type: ignore[arg-type]
        )
    except PreconditionUnmet as unmet:
        return (unmet.code,)
    errors = tuple(
        f"{name}_ungrounded"
        for name, choices in offer.enums.items()
        if action.arguments.get(name) not in choices
    )
    if action.tool_name == "select_recipe_candidate" and not errors:
        record = _organism_records(dependencies)[
            action.arguments["organism_record_event_id"]
        ]
        quote = action.arguments.get("organism_quote")
        if (
            not isinstance(quote, str)
            or not quote
            or not quoted_in_record(record, quote)
        ):
            errors = ("organism_quote_not_in_record",)
    return errors


def execution_refusal_facts(
    dependencies: ExecutionPlanningActionPolicyDependencies,
    action: ExecutionPlanningAction,
    code: str,
) -> dict[str, Any]:
    """The typed facts an execution-planning refusal's sentence names (#409): the
    values a choice argument may take, which the Tool's schema enum offers."""
    field = code.removesuffix("_ungrounded")
    if field == code:
        return {}
    offer = execution_tool_values(dependencies, action.tool_name)  # type: ignore[arg-type]
    return {"offered": offer.enums[field]}


__all__ = [
    "EXECUTION_PLANNING_TOOLS",
    "ExecutionPlanningActionName",
    "ExecutionToolValues",
    "execution_planning_projection",
    "execution_planning_selected_disposition",
    "execution_refusal_facts",
    "execution_tool_errors",
    "execution_tool_values",
]
