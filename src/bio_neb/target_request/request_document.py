"""The one request writer (#367): a placement record as a Run's ``/3`` request file.

Each answer of a `PaperTargetsPlaced` record becomes the request items it stands for, typed:
the target holds only what the paper states (its rsID, its position on the version it
states, its gene and change), ``stated_assembly`` the paper's version and line, and
``placement`` how placement answered it. A placed position is written only as the
placement's position on the Run's assembly, never as a stated one (CONTEXT **Placed
position**). A target no step placed is written with step ``none`` and the reason, so the
Run records it as not checked unless it states an rsID. Nothing is written while the run
stopped or waits for you.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bio_neb.investigation.investigation_request import (
    REQUEST_SCHEMA,
    InvestigationRequestItem,
    PlacedPosition,
    RequestPlacement,
    StatedAssembly,
    StatedChange,
    StatedTarget,
)

from .paper_targets import CITED_LINE_LIMIT, PaperTarget, PaperTargets, cited_window
from .placement import PlacementRefused, TargetsFile
from .records import (
    GeneCandidate,
    PaperTargetsPlaced,
    SourceFailure,
    TargetPlacement,
)

_TEXT_LIMIT = 4_000
"""The longest request line (`InvestigationRequestItem.request_text`)."""
_NOTE_LIMIT = 1_500
"""A placement note keeps at most this much of a long reason, so the paper's line keeps
room; it is also the longest `RequestPlacement.reason`."""


def write_request_file(
    record: PaperTargetsPlaced, targets: TargetsFile, request_file: Path
) -> PaperTargetsPlaced:
    """Write the record's ``/3`` request file and return the record with each answer's
    item indexes and the file. Nothing is written while the run stopped or waits."""
    if record.stopped is not None or record.waiting:
        return record
    if (
        targets.content is None
        or hashlib.sha256(targets.content).hexdigest() != record.paper_targets_sha256
    ):
        raise PlacementRefused(
            "targets_mismatch",
            "the request writer takes only the targets the record placed (its digest)",
        )
    paper = PaperTargets.model_validate_json(targets.content)
    items: list[dict[str, Any]] = []
    answers = []
    for answer, target in zip(record.answers, paper.targets, strict=True):
        start = len(items)
        items.extend(request_items(answer, target, paper))
        answers.append(
            answer.model_copy(update={"requests": tuple(range(start, len(items)))})
        )
    request_sha256 = None
    if items:
        document = {
            "schema_version": REQUEST_SCHEMA,
            "asserted_by": record.asserted_by,
            "requests": items,
        }
        written = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(written)
        request_sha256 = hashlib.sha256(written.encode()).hexdigest()
    return record.model_copy(
        update={
            "answers": tuple(answers),
            "request_file": str(request_file) if request_sha256 else None,
            "request_sha256": request_sha256,
        }
    )


def request_items(
    answer: TargetPlacement, target: PaperTarget, paper: PaperTargets
) -> list[dict[str, Any]]:
    """The ``/3`` items of one answer, from the typed answer and the paper's target."""
    request = target.request
    stated = _stated(target)
    if answer.failure is not None:
        failure = answer.failure
        attempts = (
            f" after {failure.contact_attempts} attempts"
            if isinstance(failure, SourceFailure)
            else ""
        )
        note = (
            f"accepted failure: {failure.failure_class}{attempts} on {failure.tool}: "
            f"{failure.detail}"
        )
        return [_unplaced(target, stated, note)]
    if answer.position is not None:
        at = answer.position
        placement = RequestPlacement(
            step=answer.step,
            matched_name=answer.matched_name,
            position=PlacedPosition(
                contig=at.contig, position=at.position, ref=at.ref, alt=at.alt
            ),
            reason=_reason(answer.reason),
        )
        return [
            _item(
                _text(
                    request.request_text,
                    request.rsid,
                    f"{answer.answer}: {answer.reason}",
                ),
                stated,
                request.stated_assembly,
                placement,
            )
        ]
    found = answer.gene_change
    if found is not None and answer.answer in ("placed", "ambiguous"):
        return [
            _candidate(answer, target, paper, stated, index, candidate)
            for index, candidate in enumerate(found.candidates, start=1)
        ] or [_unplaced(target, stated, f"{answer.answer}: {answer.reason}")]
    if answer.answer == "passed" and request.position is None and target.change is None:
        # An rsID alone: the Run looks it up, and nothing is placed.
        return [_item(request.request_text, stated)]
    return [_unplaced(target, stated, f"{answer.answer}: {answer.reason}")]


def _stated(target: PaperTarget) -> dict[str, Any]:
    """What the paper states for the target, typed: its rsID, its position, its change."""
    request = target.request
    stated: dict[str, Any] = {"rsid": request.rsid}
    if request.position is not None:
        stated["position"] = request.position.model_dump(mode="json")
    if target.change is not None:
        stated["gene_change"] = StatedChange(
            gene=_one_line(target.change.gene), change=_one_line(target.change.text)
        ).model_dump(mode="json")
    return {key: value for key, value in stated.items() if value is not None}


def _unplaced(target: PaperTarget, stated: dict[str, Any], note: str) -> dict[str, Any]:
    """A target no step placed: what the paper states, with step ``none`` and why. The
    Run looks up its rsID if it states one, else records it as not checked."""
    request = target.request
    return _item(
        _text(request.request_text, request.rsid, note),
        stated,
        request.stated_assembly,
        RequestPlacement(step="none", reason=_reason(note)),
    )


def _candidate(
    answer: TargetPlacement,
    target: PaperTarget,
    paper: PaperTargets,
    stated: dict[str, Any],
    index: int,
    candidate: GeneCandidate,
) -> dict[str, Any]:
    """One G15 candidate: the paper's gene and change, placed at the candidate."""
    found = answer.gene_change
    claim = found.claim
    count = len(found.candidates)
    changed = candidate.bases_changed
    status = f"ambiguous ({found.reason}), " if found.outcome == "ambiguous" else ""
    source = "that assembly's annotation"
    reason = (
        f"{status}candidate {index} of {count}: {candidate.contig}:{candidate.position} "
        f"{candidate.ref}>{candidate.alt} gives codon {candidate.reference_codon}>"
        f"{candidate.alt_codon} on {', '.join(candidate.transcripts)} ({changed} "
        f"base{'s' if changed > 1 else ''} changed), from {source}"
    )
    version = None
    if claim.stated_assembly is not None:
        version = StatedAssembly(
            name=claim.stated_assembly.name,
            cited_line=_cited_line(
                claim.stated_assembly.cited_line, claim.stated_assembly.name
            ),
        )
    placement = RequestPlacement(
        step="gene_change",
        matched_name=found.version.stated
        if found.version is not None and found.version.matched is not None
        else None,
        position=PlacedPosition(
            contig=candidate.contig,
            position=candidate.position,
            ref=candidate.ref,
            alt=candidate.alt,
        ),
        reason=_reason(reason),
    )
    request = target.request
    return _item(
        _text(request.request_text, request.rsid, f"placed: {reason}"),
        {key: value for key, value in stated.items() if key != "position"},
        version,
        placement,
    )


def _item(
    text: str,
    stated: dict[str, Any],
    assembly: StatedAssembly | None = None,
    placement: RequestPlacement | None = None,
) -> dict[str, Any]:
    """One ``/3`` item, checked as the Run reads it."""
    item: dict[str, Any] = {"request": text}
    if stated:
        item["target"] = StatedTarget.model_validate(stated).model_dump(mode="json")
    if assembly is not None:
        item["stated_assembly"] = assembly.model_dump(mode="json")
    if placement is not None:
        item["placement"] = placement.model_dump(mode="json")
    InvestigationRequestItem(
        request_text=item["request"],
        rsid=stated.get("rsid"),
        position=stated.get("position"),
        gene_change=stated.get("gene_change"),
        stated_assembly=assembly,
        placement=placement,
    )
    return item


def _text(line: str, rsid: str | None, note: str) -> str:
    """The paper's line, then the placement's note, within the request limit. A line too
    long for both is cut, never the note's answer and reason; a cut line's typed rsID is
    then restated in the note, as the request requires its line to state it."""
    note = _one_line(note)
    text = f"{line} (placement: {note})"
    if len(text) <= _TEXT_LIMIT:
        return text
    if len(note) > _NOTE_LIMIT:
        note = f"{note[: _NOTE_LIMIT - 1]}…"
    restated = f"; the paper's line states {rsid}" if rsid else ""
    tail = f" (placement: {note}{restated})"
    cut = line[: _TEXT_LIMIT - len(tail) - 1].rstrip()
    return f"{cut}…{tail}"


def _reason(text: str) -> str:
    line = _one_line(text)
    return line if len(line) <= _NOTE_LIMIT else f"{line[: _NOTE_LIMIT - 1]}…"


def _cited_line(line: str, name: str) -> str:
    """The paper's version line as one line: whole when it fits, else cut around the
    version's name."""
    one = _one_line(line)
    return one if len(one) <= CITED_LINE_LIMIT else cited_window(one, name)


def _one_line(text: str) -> str:
    return " ".join(text.split())
