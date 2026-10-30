"""Reusable execution-proposal Journey builders."""

from __future__ import annotations

from bio_neb.agentic_journey.attempt_journey import ToolCallValidated
from bio_neb.agentic_journey.continuation_policy import ExecutionReviewPolicy
from bio_neb.parabricks.capability_policy import CapabilitySnapshotAdapter
from bio_neb.parabricks.execution_planning_journey import (
    OrganismPloidyJudgment,
    PlanningCandidateSelected,
    ResourceObservationRecorded,
    project_execution_planning,
)
from bio_neb.parabricks.execution_proposal import (
    ExecutionProposalSubmitted,
    compile_execution_proposal,
)
from bio_neb.parabricks.recipe_resources import (
    LookupResourcesInput,
    LookupResourcesTool,
)
from tests.planning_fakes import (
    add_inspection,
    add_requirement,
    add_target_requirement,
    append_asserted_recipe_observation,
    policy,
)
from tests.reference_fakes import (
    append,
    append_observation,
    attempt_start,
    companion_timeline,
    compatible_record,
    payload,
)


def ready_timeline(*, asset_id: str = "opaque-current", ploidy: str | None = None):
    """A Journey ready to compile its proposal. ``ploidy`` is the model's judgment on
    its recipe selection, citing the Run's taxonomy record (#470); None is a
    selection recorded before #470."""
    timeline, companion_work, _ = companion_timeline()
    add_requirement(timeline, "analysis_recipe", "analysis_recipe", "1")
    add_requirement(timeline, "resource_profile", "resource_profile", "2")
    add_target_requirement(timeline)
    add_inspection(timeline)
    append_observation(
        timeline,
        companion_work,
        label="proposal-companion",
        records=(compatible_record(companion_work, asset_id=asset_id),),
    )
    adapter = CapabilitySnapshotAdapter(policy())
    work, recipe_result, recipe_event = append_asserted_recipe_observation(
        timeline, adapter, "proposal-recipe"
    )
    recipe_attempt = attempt_start(timeline, "proposal-recipe-selection")
    recipe = recipe_result.candidates[0]
    judged = (
        None
        if ploidy is None
        else OrganismPloidyJudgment(
            judgment=ploidy,
            record_event_id=next(
                e["event_id"]
                for e in timeline
                if e["event_type"] == "TaxonResolutionRecorded"
            ),
            quote="Homo sapiens",
        )
    )
    recipe_selection = PlanningCandidateSelected(
        attempt_id=recipe_attempt,
        kind="recipe",
        work_item_id=work.work_item_id,
        candidate_id=recipe.candidate_id,
        basis_event_id=recipe_event["event_id"],
        authority="agent_ranking",
        authority_event_id="pending",
        organism_ploidy=judged,
    )
    validated = append(
        timeline,
        "proposal-recipe-selection-validated",
        "ToolCallValidated",
        payload(
            ToolCallValidated(
                attempt_id=recipe_attempt,
                tool_name="select_recipe_candidate",
                tool_call_id="select-recipe",
                arguments=recipe_selection.admitted_arguments(),
                status="valid",
                errors=(),
            )
        ),
    )
    append(
        timeline,
        "proposal-recipe-selected",
        "PlanningCandidateSelected",
        payload(
            recipe_selection.model_copy(
                update={"authority_event_id": validated["event_id"]}
            )
        ),
    )
    recipe_ready = project_execution_planning(timeline)
    resource_input = LookupResourcesInput(work_item=recipe_ready.resource_work_item)
    resource_result = LookupResourcesTool().lookup(
        resource_input,
        expected_work_item=recipe_ready.resource_work_item,
        adapter=adapter,
    )
    resource_attempt = attempt_start(timeline, "proposal-resource-observation")
    resource_event = append(
        timeline,
        "proposal-resource-observed",
        "ResourceObservationRecorded",
        payload(
            ResourceObservationRecorded(
                attempt_id=resource_attempt,
                tool_input=resource_input,
                result=resource_result,
            )
        ),
    )
    resource_attempt = attempt_start(timeline, "proposal-resource-selection")
    resource = resource_result.candidates[0]
    validated = append(
        timeline,
        "proposal-resource-selection-validated",
        "ToolCallValidated",
        payload(
            ToolCallValidated(
                attempt_id=resource_attempt,
                tool_name="select_resource_candidate",
                tool_call_id="select-resource",
                arguments={
                    "work_item_id": resource_input.work_item.work_item_id,
                    "candidate_id": resource.candidate_id,
                    "basis_event_id": resource_event["event_id"],
                },
                status="valid",
                errors=(),
            )
        ),
    )
    append(
        timeline,
        "proposal-resource-selected",
        "PlanningCandidateSelected",
        payload(
            PlanningCandidateSelected(
                attempt_id=resource_attempt,
                kind="resource",
                work_item_id=resource_input.work_item.work_item_id,
                candidate_id=resource.candidate_id,
                basis_event_id=resource_event["event_id"],
                authority="agent_ranking",
                authority_event_id=validated["event_id"],
            )
        ),
    )
    return timeline


def submit(timeline):
    proposal = compile_execution_proposal(timeline)
    event = append(
        timeline,
        f"submit-{len(timeline)}",
        "ExecutionProposalSubmitted",
        payload(
            ExecutionProposalSubmitted(
                review_policy=ExecutionReviewPolicy(), proposal=proposal
            )
        ),
    )
    return proposal, event
