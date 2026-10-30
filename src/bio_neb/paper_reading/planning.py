"""A Run paper read in planning (#383 A2): the values a plan finds in a cited paper.

A plan whose ``to_find`` values cite a recorded Run paper (their evidence names its
``PaperRecorded``, #420), or that cites a paper its paper stage recorded, has that
paper read before its reference work, for the plan's
research goal as the researcher wrote it (owner, #383 Q3 a). The coordinator drives the
reading between planning Attempts, one step at a time (``advance_planned_reading``), as
it sequences every assignment (ADR 0030); the paper stage stays open until the reading
records its answer, and a stopped reading item holds the Run at its Review: that item
alone, its siblings read meanwhile (``held_only_by_its_items``).

Planning then cites what the reading kept by membership only (ADR 0049): the organism
the paper writes for the taxon, the genome version it states for the reference choice
(``kept_statements``), and the paper's proposed targets as the Run's targets. Code
decides none of these values; the planning model names each one, and code checks that
its words stand in a span the reading kept.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bio_neb.paper_reading.journey import PaperReading


@dataclass(frozen=True)
class PlannedReading:
    """One reading the plan needs: a recorded paper its ``to_find`` values cite, the
    question it is read for, and its latest reading of that question, if any."""

    paper_event_id: str
    question: str
    reading: PaperReading | None

    @property
    def answered(self) -> bool:
        return self.reading is not None and self.reading.answer_event_id is not None


def cited_paper_events(requirement, recorded: frozenset[str]) -> tuple[str, ...]:
    """The recorded papers a ``to_find`` value cites, by their ``PaperRecorded``."""

    # The cited_paper requirement's own record is the paper, not a value found in it.
    if requirement.kind == "cited_paper" or requirement.knowledge_state != "to_find":
        return ()
    return tuple(
        dict.fromkeys(
            evidence.source_event_id
            for evidence in requirement.evidence
            if evidence.source_kind == "planning_context"
            and evidence.source_event_id in recorded
        )
    )


def value_papers(view, requirement) -> tuple[str, ...]:
    """The recorded papers a ``to_find`` value is found in: those its evidence cites,
    else each paper the plan's own ``cited_paper`` requirements recorded. A plan is
    recorded before its paper stage retrieves the paper, so its values cannot cite
    the ``PaperRecorded`` yet (live, #383 A2 probe); the plan's citation names it."""

    explicit = cited_paper_events(requirement, _recorded(view))
    if explicit or requirement.kind == "cited_paper":
        return explicit
    if requirement.knowledge_state != "to_find":
        return ()
    return _plan_papers(view)


def _plan_papers(view) -> tuple[str, ...]:
    """Each paper the plan's ``cited_paper`` requirements recorded, in its order."""

    from bio_neb.paper_intake.tools import citation_intake, cited_papers

    return tuple(
        dict.fromkeys(
            intake.paper_event_id
            for requirement in cited_papers(view)
            if (intake := citation_intake(view, requirement)) is not None
            and intake.state == "recorded"
            and intake.paper_event_id is not None
        )
    )


def _recorded(view) -> frozenset[str]:
    return frozenset(
        intake.paper_event_id
        for intake in view.papers.intakes
        if intake.state == "recorded" and intake.paper_event_id is not None
    )


def research_question(plan) -> str | None:
    """The plan's research goal as the researcher wrote it: each research_goal quote,
    in the plan's order, untouched; None when the plan quotes none."""

    quotes = [
        evidence.exact_text
        for requirement in plan.requirements
        if requirement.kind == "research_goal"
        for evidence in requirement.evidence
        if evidence.source_kind == "research_request"
    ]
    return "\n".join(dict.fromkeys(quotes)) or None


def planned_readings(view) -> tuple[PlannedReading, ...]:
    """Each reading the plan needs, in the order its values cite the papers: none
    before the plan, for a plan whose values cite no recorded paper, or for one that
    quotes no research goal (the stage's own questions open as before)."""

    if view.plan is None:
        return ()
    plan = view.plan.plan
    question = research_question(plan)
    if question is None:
        return ()
    papers = dict.fromkeys(
        paper
        for requirement in plan.requirements
        for paper in value_papers(view, requirement)
    )
    return tuple(
        PlannedReading(
            paper_event_id=paper,
            question=question,
            reading=next(
                (
                    reading
                    for reading in reversed(view.paper_readings.readings)
                    if reading.paper_event_id == paper and reading.question == question
                ),
                None,
            ),
        )
        for paper in papers
    )


def reading_due(view) -> PlannedReading | None:
    """The first reading the plan needs that has recorded no answer yet."""

    return next((p for p in planned_readings(view) if not p.answered), None)


def answered_readings(view, requirement) -> tuple[PaperReading, ...]:
    """The answered readings of the papers one ``to_find`` value cites."""

    cited = value_papers(view, requirement)
    return tuple(
        planned.reading
        for planned in planned_readings(view)
        if planned.answered and planned.paper_event_id in cited
    )


def held_only_by_its_items(journey) -> bool:
    """Whether the Run's pending Reviews are all ones the reading the plan needs asked
    for its own items: each holds only its item, so the reading's other items go on
    (ADR 0039; owner, #383 Q23) while the reading's answer waits for it."""

    from bio_neb.application.planning_read import current_planning
    from bio_neb.paper_reading.reading import asked_only_by_its_items

    due = reading_due(current_planning(journey))
    return (
        due is not None
        and due.reading is not None
        and asked_only_by_its_items(journey, due.reading.reading_id)
    )


def advance_planned_reading(
    journey,
    *,
    environ: Mapping[str, str],
    model_factory: Any = None,
) -> bool:
    """Take one step of the reading the plan needs; True when it took one.

    The reading opens on the paper and the plan's research goal, as written; each
    later step is the reading's own (``advance_reading``): an item its Review's
    terminal answer settled, a stopped item its continue answer re-enters, the next
    open item no stop holds, then the answer. Nothing runs once every reading is
    answered, or while only items held by their Reviews remain."""

    from bio_neb.application.planning_read import current_planning
    from bio_neb.paper_reading.reading import (
        _registered_runtime,
        advance_reading,
        start_reading,
    )

    due = reading_due(current_planning(journey))
    if due is None:
        return False
    runtime = _registered_runtime(journey)
    if due.reading is None:
        start_reading(
            journey,
            paper_event_id=due.paper_event_id,
            question=due.question,
            runtime=runtime,
            environ=environ,
            model_factory=model_factory,
        )
        return True
    return advance_reading(
        journey,
        due.reading.reading_id,
        runtime=runtime,
        environ=environ,
        model_factory=model_factory,
    )


def paper_findings(view) -> tuple:
    """The reference stage's rows for each answered reading the plan needs: the
    organism and the genome version as the paper writes them, each by the reading
    fact that keeps its span (``ReadingPlanKept``), or ``not_written`` when the
    paper item kept none, so the planning model cites them or asks (#383 A2)."""

    from bio_neb.agentic_journey.stage_findings import StageFinding

    rows = []
    for planned in planned_readings(view):
        reading = planned.reading
        if not planned.answered or reading.plan is None:
            continue
        plan = reading.plan
        for subject, value in (
            ("organism", plan.organism.written if plan.organism else None),
            (
                "genome_version",
                plan.reference.candidate.name if plan.reference else None,
            ),
        ):
            rows.append(
                StageFinding(
                    kind="ReadingPlanKept",
                    source_event_id=reading.plan_event_id,
                    status="kept" if value is not None else "not_written",
                    subject_key=subject,
                    value=value,
                )
            )
    return tuple(rows)


def kept_statements(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Each reading fact that keeps the paper's words, with the spans it keeps, each
    copied by code from the stored text (ADR 0046): a paper item's plan (the lines
    that name its genome version, the organism as written and its paragraph or row)
    and an accepted finding (its quote and its lines). A planning quote stands in the
    paper's words when it stands in one of these (ADR 0049 membership)."""

    from bio_neb.paper_reading.journey import ReadingFindingJudged, ReadingPlanKept

    statements = []
    for event in events:
        kind = event["event_type"]
        if kind == "ReadingPlanKept":
            plan = ReadingPlanKept.model_validate(
                {"event_type": kind, **event["payload"]}
            )
            lines = [
                *(plan.reference.candidate.sources if plan.reference else ()),
                *(
                    (plan.organism.source, plan.organism.passage)
                    if plan.organism
                    else ()
                ),
            ]
        elif kind == "ReadingFindingJudged":
            judged = ReadingFindingJudged.model_validate(
                {"event_type": kind, **event["payload"]}
            )
            if judged.outcome != "accepted":
                continue
            lines = [judged.quote, *judged.lines]
        else:
            continue
        if lines:
            statements.append(
                (event["event_id"], tuple(dict.fromkeys(line.text for line in lines)))
            )
    return tuple(statements)


def kept_texts(events: Sequence[Mapping[str, Any]], event_id: str) -> tuple[str, ...]:
    """The spans one reading fact keeps (``kept_statements``); none for another event."""

    return next(
        (texts for eid, texts in kept_statements(events) if eid == event_id), ()
    )


__all__ = [
    "PlannedReading",
    "advance_planned_reading",
    "answered_readings",
    "cited_paper_events",
    "held_only_by_its_items",
    "kept_statements",
    "kept_texts",
    "paper_findings",
    "planned_readings",
    "reading_due",
    "research_question",
    "value_papers",
]
