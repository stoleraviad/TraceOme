"""The planning stages' shared refusals: each planning Tool row declares the
stage gate or open question it waits on (ADR 0040). The planning domains' rows
(reference, companions, execution planning, input correction) read them."""

from __future__ import annotations

from bio_neb.agentic_journey.tool_execution import (
    NegativeResult,
    ToolPrecondition,
)

QUESTION_PENDING = ToolPrecondition(
    "planning_question_pending",
    "pending researcher question",
    "A researcher question is open; this Attempt ends there and resumes after the answer.",
    unmet_by=(
        "request_planning_hitl",
        "request_companion_asset_selection",
        "request_companion_refinement",
        "request_recipe_eligibility_assertion",
        "request_recipe_selection",
        "request_resource_selection",
        "request_resource_recipe_dependency",
    ),
)


CORRECTION_PENDING = ToolPrecondition(
    "execution_correction_pending",
    "execution correction",
    "An execution input correction is in progress; only the input re-inspection runs until it is prepared.",
)


REFERENCE_NOT_OPEN = ToolPrecondition(
    "reference_work_not_open",
    "proposal plan",
    "Reference work opens once the proposal requirements are recorded.",
)


COMPANIONS_NOT_OPEN = ToolPrecondition(
    "companion_work_not_open",
    "verified reference",
    "Companion work opens once the reference is verified.",
)


EXECUTION_NOT_OPEN = ToolPrecondition(
    "execution_planning_not_open",
    "reference work",
    "Execution planning opens once the reference and its companions are verified.",
)


# No compatible recipe or resource is a lawful domain answer the model reads
# (ADR 0040); an adapter exception stays an operational fault (ADR 0031).
NO_COMPATIBLE_RECIPE = (
    NegativeResult("RecipeObservationRecorded", "outcome", ("missing",)),
    NegativeResult("RecipeObservationRecorded", "compatibility", ("incompatible",)),
)


COMPANION_WORK_COMPLETE = ToolPrecondition(
    "companion_work_complete",
    "companion work items",
    "Companion work is complete; the next request opens execution planning.",
)
