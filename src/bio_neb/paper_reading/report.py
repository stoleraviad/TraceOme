"""What the Run's papers found, as a report's basis copies it (#485, owner Q3 a).

Each answered reading of a Run paper is one reported paper: its question, the paper as
intake resolved it, who its targets are asserted by, and its answer as the reading's
fold recorded it (``PaperAnswerRecorded``). Every finding keeps the chunks code copied
from the stored paper (its quote first, then each paragraph or table row it rests on),
so the report cites exactly what the reading kept; nothing here is re-read or judged.
A reading whose answer is not yet recorded is not reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import StrictStr

from bio_neb.contracts import FrozenContract
from bio_neb.paper_reading.contracts import PaperAnswer
from bio_neb.paper_reading.journey import PaperAnswerRecorded, project_paper_readings
from bio_neb.target_request.paper_targets import PaperIdentity


class ReportedPaper(FrozenContract):
    """One Run paper's answered reading, copied from its answer fact."""

    answer_event_id: StrictStr
    paper_event_id: StrictStr
    question: StrictStr
    paper: PaperIdentity
    # The id the paper's proposed targets are asserted by (``paper:doi:…``); None when
    # the paper names no id to assert them by, so it proposed no target.
    asserted_by: StrictStr | None
    answer: PaperAnswer


def asserter_words(asserted_by: str | None) -> str | None:
    """A paper asserter (``paper:…``) as the report names it: "the paper doi:…" (#485);
    None for any other asserter, so the researcher's prompt-named targets read as
    the researcher's (M smoke 028a063a7 F1)."""

    if asserted_by is None or not asserted_by.startswith("paper:"):
        return None
    return "the paper " + asserted_by.removeprefix("paper:")


def reported_papers(timeline: Sequence[Mapping[str, Any]]) -> tuple[ReportedPaper, ...]:
    """Every answered reading in the timeline, in the order its answer was recorded."""

    readings = project_paper_readings(timeline)
    by_id = {event["event_id"]: event for event in timeline}
    answered = [
        reading for reading in readings.readings if reading.answer_event_id is not None
    ]
    reported = []
    for reading in answered:
        fact = PaperAnswerRecorded.model_validate(
            by_id[reading.answer_event_id]["payload"]
        )
        reported.append(
            ReportedPaper(
                answer_event_id=reading.answer_event_id,
                paper_event_id=reading.paper_event_id,
                question=reading.question,
                paper=reading.paper,
                asserted_by=None if fact.targets is None else fact.targets.asserted_by,
                answer=fact.answer,
            )
        )
    order = {event_id: position for position, event_id in enumerate(by_id)}
    return tuple(sorted(reported, key=lambda paper: order[paper.answer_event_id]))


__all__ = ["ReportedPaper", "asserter_words", "reported_papers"]
