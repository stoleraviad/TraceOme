"""A Run paper as the Investigation receives it for one member (#485, owner Q2 a).

A member whose target a paper's reading proposed (its request ``asserted_by`` the
paper) receives that paper: the reading's answer it came from, and what planning's
reading already derived from it, each with the chunks the reading kept (its target's
quote and passages, the genome version and organism as written). The member reads the
paper itself with the reading Tools (``tools.reading_tool_rows``); nothing here is
re-read or judged, and a member whose target no paper proposed receives none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bio_neb.contracts import FrozenContract
from bio_neb.paper_reading.journey import PaperAnswerRecorded
from bio_neb.target_request.paper_targets import PaperTarget, PaperTargets


class TargetPaper(FrozenContract):
    """The reading answer that proposed one member's target, and that target."""

    answer_event_id: str
    paper_event_id: str
    targets: PaperTargets
    target: PaperTarget


def _proposes(target: PaperTarget, request: Any) -> bool:
    """Whether ``target`` is the member's own: the same words and the same typed
    target, so two targets one shared line states stay two members (ADR 0039)."""

    proposed = target.request
    return (
        proposed.request_text,
        proposed.rsid,
        proposed.position,
        proposed.gene_change,
    ) == (
        request.request_text,
        request.rsid,
        request.position,
        request.gene_change,
    )


def target_paper(
    timeline: Sequence[Mapping[str, Any]], request: Any
) -> TargetPaper | None:
    """The answer whose proposed targets hold this member's request, asserted by its
    paper; None for a request no paper's reading proposed."""

    if request.asserted_by is None:
        return None
    from bio_neb.paper_reading.journey import project_paper_readings

    readings = {
        reading.answer_event_id: reading
        for reading in project_paper_readings(timeline).readings
        if reading.answer_event_id is not None
    }
    for event in timeline:
        if event["event_id"] not in readings:
            continue
        targets = PaperAnswerRecorded.model_validate(event["payload"]).targets
        if targets is None or targets.asserted_by != request.asserted_by:
            continue
        for target in targets.targets:
            if _proposes(target, request):
                return TargetPaper(
                    answer_event_id=event["event_id"],
                    paper_event_id=readings[event["event_id"]].paper_event_id,
                    targets=targets,
                    target=target,
                )
    return None


def _chunk(line) -> dict[str, Any]:
    return {
        "text": line.text,
        "section": line.section,
        "document_id": line.document_id,
        "start": line.start,
        "end": line.end,
    }


def paper_context(
    by_id: Mapping[str, Mapping[str, Any]], answer_event_id: str, request: Any
) -> dict[str, Any]:
    """What the member reads of its paper, rendered from the answer event: the paper,
    the question it was read for, and what the reading kept for this target."""

    event = by_id.get(answer_event_id)
    if event is None or event["event_type"] != "PaperAnswerRecorded":
        raise ValueError("member context names no paper answer")
    fact = PaperAnswerRecorded.model_validate(event["payload"])
    targets = fact.targets
    target = next(
        (
            item
            for item in (() if targets is None else targets.targets)
            if _proposes(item, request)
        ),
        None,
    )
    if target is None or targets.asserted_by != request.asserted_by:
        raise ValueError("member context names a paper answer without its target")
    paper = targets.paper
    context: dict[str, Any] = {
        "answer_event_id": answer_event_id,
        "asserted_by": targets.asserted_by,
        "title": paper.title,
        "ids": {
            key: value
            for key, value in (
                ("doi", paper.doi),
                ("pmid", paper.pmid),
                ("pmcid", paper.pmcid),
            )
            if value
        },
        "read_for": fact.answer.question,
        "target_quote": _chunk(target.quote),
        "target_findings": [
            {
                "passages": [_chunk(line) for line in finding.passages],
                "reason": finding.reason,
            }
            for finding in target.findings
        ],
    }
    if fact.answer.reference is not None:
        candidate = fact.answer.reference.candidate
        context["genome_version_as_written"] = {
            "name": candidate.name,
            "spellings": list(candidate.spellings),
            "lines": [_chunk(line) for line in candidate.sources],
        }
    if fact.answer.organism is not None:
        context["organism_as_written"] = {
            "written": fact.answer.organism.written,
            "passage": _chunk(fact.answer.organism.passage),
        }
    return context


__all__ = ["TargetPaper", "paper_context", "target_paper"]
