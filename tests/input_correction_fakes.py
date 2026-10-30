"""Explicit current correction setup for shared Agent mechanics tests.

The missing-path observation is synthetic; the exact review and retry answer use
product commands. These helpers grant no live authority and do not replace the
Agent runner, metadata Tool or Journey writer under test.
"""

from bio_neb.application.input_journey import answer_input_review
from bio_neb.failure import ReviewAnswer
from bio_neb.input_inspection.commands import inspect_inputs
from bio_neb.input_inspection.local_files import LocalFilesAdapter


def prepare_input_correction(journey):
    state = journey.checked_view("input_selection")
    if state.inspections:
        return journey

    def missing(path, **kwargs):
        raise FileNotFoundError(path)

    inspect_inputs(journey, adapter=LocalFilesAdapter(stat_provider=missing))
    pending = journey.checked_view("input_selection").requests[-1]
    answer_input_review(
        journey,
        pending.request_event_id,
        ReviewAnswer(
            request_id=pending.request.request_id,
            action="retry",
            by="test researcher",
        ),
    )
    return journey
