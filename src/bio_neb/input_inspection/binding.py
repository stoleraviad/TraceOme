"""Thin metadata correction Tool binding for the shared planning Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    ToolExecutionFailed,
)
from bio_neb.agentic_journey.invocation import ToolContext
from bio_neb.agentic_journey.lifecycle import Activity
from bio_neb.agentic_journey.tool_execution import (
    ModelView,
    ObservedToolSession,
    ToolExecutionAdapter,
    ToolPrecondition,
    function_tool_contract,
)
from bio_neb.input_inspection.tool import (
    InputMetadataAdapter,
    InputMetadataResult,
)
from bio_neb.journey import Journey


@dataclass
class InputInspectionDependencies:
    parent_activity: Activity
    attempt_id: str
    journey: Journey
    metadata_adapter: InputMetadataAdapter
    failure_source: AttemptFailureSource
    observed_tool_session: ObservedToolSession
    inspection_result: InputMetadataResult | None = None
    inspection_event_id: str | None = None


def inspect_input_metadata(
    context: ToolContext[InputInspectionDependencies], input_set_id: str
) -> dict[str, Any]:
    """The correction is named in
    `supplied_inputs.current_selection.correction_direction`; the new observation closes
    this assignment.
    """
    dependencies = context.deps

    def typed_operation() -> InputMetadataResult:
        nonlocal inspection
        activity = dependencies.parent_activity.start_child(
            adapter="local_files",
            stage="inspect_input_metadata",
        )
        try:
            from bio_neb.input_inspection.commands import inspect_inputs

            result, inspected = inspect_inputs(
                dependencies.journey,
                adapter=dependencies.metadata_adapter,
                record=activity.record,
            )
            activity.succeeded(inspected)
        except Exception as error:
            if not activity.closed:
                failed = activity.record(
                    ToolExecutionFailed(
                        attempt_id=dependencies.attempt_id,
                        error_type=type(error).__name__,
                    ),
                )
                dependencies.failure_source.capture(failed)
                activity.failed(failed)
            raise RuntimeError("registered input inspection failed") from error
        dependencies.inspection_result = result
        dependencies.inspection_event_id = inspected["event_id"]
        inspection = inspected
        return result

    inspection = None
    return dependencies.observed_tool_session.execute(
        context.tool_call_id,
        typed_operation,
        ToolExecutionAdapter(
            identity="local_files",
            stage_result_event=lambda _result: (
                inspection if inspection is not None else {}
            ),
            artifact_digests=lambda _result: (),
            stage_failure_event=lambda _error: (
                {
                    "event_id": dependencies.failure_source.event_id,
                    "event_type": dependencies.failure_source.event_type,
                }
                if dependencies.failure_source.event_type == "ToolExecutionFailed"
                and dependencies.failure_source.event_id is not None
                else None
            ),
        ),
    )


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "inspect_input_metadata",
        "planning_agent",
        ("input_set_id: str",),
        "InputMetadataResult",
        "local_files",
        "InputMetadataInspected",
        "ToolExecutionFailed",
        "error_type",
        ModelView(
            next_step={
                "succeeded": "The new observation closes the input correction; "
                "continue with the Tools callable now.",
                "failed": "The inputs still fail as the result says; the "
                "researcher corrects them.",
            },
            statement=lambda payload: payload["result"]["summary"],
        ),
        "legacy_tool_execution_failed_retained",
        "stage_error_preserved_then_wrapped_as_RuntimeError",
        stages=("input_correction",),
        examples=("{} (the server supplies the input set)",),
        preconditions=(
            ToolPrecondition(
                "input_set_id_ineligible",
                "input correction",
                "The registered inputs have no pending correction, so they are not re-inspected.",
            ),
        ),
    ),
)
