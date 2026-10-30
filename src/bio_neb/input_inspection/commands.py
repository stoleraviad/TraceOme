"""Direct deterministic inspection of registered inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import StrictStr

from bio_neb.application.commands import RegistrationError
from bio_neb.application.input_journey import record_fact
from bio_neb.contracts import FrozenContract
from bio_neb.journey import Journey, JourneyFile

if TYPE_CHECKING:
    from bio_neb.application.queries import JourneyReview
    from bio_neb.input_inspection.tool import InputMetadataAdapter


class InspectRegisteredInputsCommand(FrozenContract):
    journey_file: JourneyFile
    parent_event_id: StrictStr | None = None


def inspect_registered_inputs(
    command: InspectRegisteredInputsCommand,
    *,
    adapter: InputMetadataAdapter | None = None,
    journey: Journey | None = None,
) -> JourneyReview:
    """Inspect the current supplied selection and return its checked Journey view."""
    from bio_neb.application.queries import ReviewJourneyQuery, review_journey

    selected_journey = journey or Journey.open(Path(command.journey_file))
    if str(selected_journey.path) != command.journey_file:
        raise RegistrationError("journey_identity_invalid", ("journey_file",))
    before = review_journey(ReviewJourneyQuery(journey_file=command.journey_file))
    if before.status != "inputs_registered":
        raise RegistrationError("input_inspection_ineligible", ("journey_file",))
    inspect_inputs(
        selected_journey, adapter=adapter, parent_event_id=command.parent_event_id
    )
    return review_journey(ReviewJourneyQuery(journey_file=command.journey_file))


def inspect_inputs(journey, *, adapter=None, parent_event_id=None, record=None):
    """One selection, observation and semantic write for direct and Tool callers."""
    import uuid

    from bio_neb.application.input_journey import (
        InputInspectionReviewRequested,
        InputMetadataInspected,
    )
    from bio_neb.failure import Ask
    from bio_neb.input_inspection.staged_objects import registered_inputs_adapter
    from bio_neb.input_inspection.tool import (
        InputMetadataInspectionTool,
        InspectInputMetadataInput,
    )

    selected = journey.checked_view("input_selection")
    if selected.stopped or (selected.inspections and not selected.correction_pending):
        raise RegistrationError("input_inspection_ineligible", ("journey_file",))

    def retain(fact, parent):
        return (
            record(fact)
            if record is not None
            else record_fact(journey, fact, (parent,))
        )

    try:
        result = InputMetadataInspectionTool().inspect(
            InspectInputMetadataInput(
                input_set_id=selected.input_set_id, inputs=selected.inputs
            ),
            adapter if adapter is not None else registered_inputs_adapter(journey),
        )
    except Exception as error:
        from bio_neb.application.input_journey import InputInspectionErrored

        retain(
            InputInspectionErrored(
                selection_event_id=selected.source_event_id,
                error_type=type(error).__name__,
            ),
            parent_event_id or selected.source_event_id,
        )
        raise
    event = retain(
        InputMetadataInspected(
            input_set_id=selected.input_set_id,
            selection_event_id=selected.source_event_id,
            result=result,
        ),
        parent_event_id or selected.source_event_id,
    )
    if result.status == "failed":
        retain(
            InputInspectionReviewRequested(
                request=Ask(
                    stage="input_metadata",
                    failure_event_id=event["event_id"],
                    request_id=str(uuid.uuid4()),
                    reason=result.summary,
                    legal_actions=("retry", "amend", "abort"),
                    amendable=tuple(i.role for i in selected.inputs),
                )
            ),
            event["event_id"],
        )
    return result, event
