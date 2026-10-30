"""A paper's targets onto the Run's assembly: placement as one typed callable (#393, #367).

Paper reading writes a paper's targets (`bio-neb.paper-targets/1`, `paper_targets.PaperTargets`)
and resolves nothing; placement answers each one on the Run's assembly (owner, #351 q7).
`place(targets, run, sources)` returns the typed record, `PaperTargetsPlaced`; the request
writer (`request_document`) and the developer entry are adapters over it, and the request text
is not the interface (#393 Q3, Q4). Every target gets one answer:

- a position goes through the resolver order (`resolver.py`), and the answer names its step;
  the flank move is development-only, so a Run passes no mover, and a target only a flank
  move would place fails with `FlankMoveUnavailable` and waits (owner, #449 Q1 a);
- a gene and change with no position goes through G15 (`placed`, or its own answer);
- an rsID with nothing else is `passed`: the Run looks it up.

A position no step places passes its rsID instead when it has one. Each target fails on its
own (#393 Q16): a source contact the failure module gave up on is that target's `failure`,
the other targets are still answered and the record counts the failures. A failure routed to
`ask` waits for the owner (Q18), who answers each target on a later run: retry (rerun it, the
other answers reused) or accept (written as unplaced, with the failure named). Only a
plan-level problem stops the run (Q17): a request the product's transport refuses, an
unreadable record or our bug. Outside a Run nothing is journaled (owner, #417 Q4); the record's
`contact_count` is what a live plan watches (#367 readiness Q5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from bio_neb.failure import FailurePolicy
from bio_neb.investigation.investigation_request import StatedChange
from bio_neb.investigation.investigation_sources import AssemblyReport

from .ncbi_sources import NcbiSources, answer_claim
from .paper_targets import PaperTarget, PaperTargets
from .records import (
    FlankMoveUnavailable,
    GeneChangeClaim,
    LookupStop,
    PaperTargetsPlaced,
    StatedSequences,
    StatedVersion,
    TableUsed,
    TargetPlacement,
)
from .resolver import FlankMover, resolve_position
from .source_failure import SourceFailed


class PlacementRefused(Exception):
    """A run's own inputs cannot be read as given (a stated-sequence table, a waiting
    record or its answers); nothing is contacted."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def read_stated_sequences(content: bytes) -> tuple[StatedSequences, TableUsed]:
    """A Run's stated-sequence table and its name and digest, as the record names them."""
    try:
        table = StatedSequences.model_validate_json(content)
    except ValidationError as error:
        raise PlacementRefused(
            "stated_sequences_unreadable",
            f"not a bio-neb.stated-sequences/1 table: {error}",
        ) from error
    return table, TableUsed(name=table.name, sha256=hashlib.sha256(content).hexdigest())


@dataclass(frozen=True)
class TargetsFile:
    """A paper's targets as one run read them: the file's bytes, or why it could not be
    read. `place` and the request writer both use this one reading (Spec r2 F3)."""

    path: Path
    content: bytes | None
    unread: str | None

    @classmethod
    def read(cls, path: Path) -> TargetsFile:
        try:
            return cls(path, path.read_bytes(), None)
        except OSError as error:
            return cls(path, None, f"{type(error).__name__}: {error}")


@dataclass(frozen=True)
class Resume:
    """The owner's answers to a waiting run (#393 Q18): its record, that record's digest,
    and the targets to retry and to accept."""

    record: PaperTargetsPlaced
    record_sha256: str
    retry: frozenset[str]
    accept: frozenset[str]


@dataclass(frozen=True)
class PlacementRun:
    """What a run places onto: the Run's assembly, its stated-sequence table if it gives
    one, and the owner's answers when it resumes a waiting run."""

    assembly: str
    table: StatedSequences | None = None
    table_used: TableUsed | None = None
    resume: Resume | None = None


@dataclass(frozen=True)
class PlacementSources:
    """The sources a run reads: NCBI's product operations and the flank-move port, which
    only a developer entry fills; a Run passes None."""

    ncbi: NcbiSources
    mover: FlankMover | None


def place(
    targets: TargetsFile, run: PlacementRun, sources: PlacementSources
) -> PaperTargetsPlaced:
    """Every target's answer on the Run's assembly, in the record's order, as the typed
    record. No request file is written here (`request_document` writes it). A targets file
    that cannot be read, as a file or as a `paper-targets/1` record, stops the run before
    any contact."""
    already = sources.ncbi.contact_count
    answers: list[TargetPlacement] = []
    stopped = None
    asserted_by = None
    digest = None
    paper = None
    if targets.content is None:
        stopped = LookupStop(claim_label="-", error=targets.unread, unattempted=())
    else:
        digest = hashlib.sha256(targets.content).hexdigest()
        try:
            paper = PaperTargets.model_validate_json(targets.content)
        except ValidationError as error:
            stopped = LookupStop(
                claim_label="-",
                error=f"not a bio-neb.paper-targets/1 record: {error}",
                unattempted=(),
            )
    if paper is not None:
        asserted_by = paper.asserted_by
        answers, stopped = _answers(paper, digest, run, sources)
    return PaperTargetsPlaced(
        made_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        assembly_accession=run.assembly,
        paper_targets_sha256=digest,
        asserted_by=asserted_by,
        table=run.table_used,
        resumed_from=run.resume.record_sha256 if run.resume else None,
        answers=tuple(answers),
        failure_count=sum(answer.failure is not None for answer in answers),
        waiting=tuple(
            answer.label
            for answer in answers
            if answer.failure is not None
            and answer.failure.route == "ask"
            and not answer.accepted
        ),
        stopped=stopped,
        contact_count=sources.ncbi.contact_count - already,
    )


def _answers(
    paper: PaperTargets, digest: str, run: PlacementRun, sources: PlacementSources
) -> tuple[list[TargetPlacement], LookupStop | None]:
    labels = [f"t{number}" for number in range(1, len(paper.targets) + 1)]
    earlier = _earlier_answers(run, digest, labels)
    answers: list[TargetPlacement] = []
    for index, (label, target) in enumerate(zip(labels, paper.targets, strict=True)):
        try:
            if label in earlier:
                answers.append(earlier[label])
            else:
                answers.append(_answer(label, target, paper, run, sources))
        except Exception as error:  # a plan stop or our bug: kept with the answers
            return answers, LookupStop(
                claim_label=label,
                error=f"{type(error).__name__}: {error}",
                unattempted=tuple(labels[index + 1 :]),
            )
        failure = answers[-1].failure
        if failure is not None and failure.route != "ask":
            # The failure module holds the run: our bug or the runtime's, no target's.
            return answers, LookupStop(
                claim_label=label,
                error=f"the failure module holds the run ({failure.failure_class} on "
                f"{failure.tool}): {failure.detail}",
                unattempted=tuple(labels[index + 1 :]),
            )
    return answers, None


def _earlier_answers(
    run: PlacementRun, digest: str, labels: list[str]
) -> dict[str, TargetPlacement]:
    """The waiting record's answers this run keeps: every answered target, and each
    accepted failure; a retried target is answered again. The owner's answers must match
    the record's waiting targets, one answer each."""
    resume = run.resume
    if resume is None:
        return {}
    record = resume.record
    if (record.paper_targets_sha256, record.assembly_accession) != (
        digest,
        run.assembly,
    ):
        raise PlacementRefused(
            "resume_mismatch",
            "the waiting record is another run's: its targets or assembly differ",
        )
    if record.table != run.table_used:
        raise PlacementRefused(
            "resume_mismatch", "the waiting record read another stated-sequence table"
        )
    answered = resume.retry | resume.accept
    if (
        record.stopped is not None
        or not record.waiting
        or resume.retry & resume.accept
        or answered != set(record.waiting)
    ):
        raise PlacementRefused(
            "resume_answers",
            f"answer each waiting target once, by retry or accept: {', '.join(record.waiting) or 'none waits'}",
        )
    if [answer.label for answer in record.answers] != labels:
        raise PlacementRefused(
            "resume_mismatch", "the waiting record does not answer every target"
        )
    return {
        answer.label: answer.model_copy(
            update={
                "requests": (),
                "accepted": answer.accepted or answer.label in resume.accept,
            }
        )
        for answer in record.answers
        if answer.label not in resume.retry
    }


def _answer(
    label: str,
    target: PaperTarget,
    paper: PaperTargets,
    run: PlacementRun,
    sources: PlacementSources,
) -> TargetPlacement:
    request = target.request
    quote = {"label": label, "quote": target.quote.text}
    if request.position is None and target.change is None:
        if request.rsid is not None:
            reason = "an rsID only: the Run looks it up"
            return TargetPlacement(
                **quote, answer="passed", step="rsid_only", reason=reason
            )
        reason = "the paper states no rsID, position or gene change for it"
        return TargetPlacement(
            **quote, answer="not_checked", step="none", reason=reason
        )
    step = "same_version" if request.position is not None else "g15"
    try:
        # Read once per run (`NcbiSources`); a failure is each dependent target's.
        assembly = sources.ncbi.run_assembly(run.assembly)
        if request.position is None:
            return _placed_change(quote, target, paper, assembly, sources)
        # The paper's other names for its stated version, as written.
        reference = paper.reference
        aliases = (
            reference.aliases
            if reference is not None
            and request.stated_assembly is not None
            and reference.candidate.name == request.stated_assembly.name
            else ()
        )
        resolved = resolve_position(
            request, assembly, run.table, sources.ncbi, sources.mover, aliases
        )
    except SourceFailed as failed:
        return TargetPlacement(
            **quote,
            answer=None,
            step=step,
            reason=failed.failure.detail,
            failure=failed.failure,
        )
    answer, step, reason = resolved.answer, resolved.step, resolved.reason
    if answer == "not_checked" and request.rsid is not None:
        answer, step = "passed", "rsid_only"
        reason = f"its position is not used: {reason}"
    elif answer == "not_checked" and resolved.move_needed is not None:
        # The target's own typed failure, never a plain miss (Q2 a).
        needed = resolved.move_needed
        return TargetPlacement(
            **quote,
            answer=None,
            step=needed.step,
            reason=reason,
            version=resolved.version,
            matched_name=resolved.matched_name,
            stated_sequence=needed.stated_sequence,
            failure=FlankMoveUnavailable(
                step=needed.step,
                stated_sequence=needed.stated_sequence,
                failure_class="rejected",
                route=FailurePolicy().route("rejected"),
                detail=needed.detail,
            ),
        )
    return TargetPlacement(
        **quote,
        answer=answer,
        step=step,
        reason=reason,
        version=resolved.version,
        matched_name=resolved.matched_name,
        stated_sequence=resolved.stated_sequence,
        equivalence=resolved.equivalence,
        move=resolved.move,
        position=resolved.position,
    )


def _placed_change(
    quote: dict[str, str],
    target: PaperTarget,
    paper: PaperTargets,
    assembly: AssemblyReport,
    sources: PlacementSources,
) -> TargetPlacement:
    """G15 on the target's gene and change as typed, with the paper's stated reference.
    No reader types a paper's change before #383, so G15 answers an untyped one not
    checked (#473 owner Q12 a)."""
    change = target.change
    reference = paper.reference
    try:
        claim = GeneChangeClaim(
            label=f"{quote['label']}-{change.gene}",
            change=StatedChange(
                gene=" ".join(change.gene.split()),
                change=" ".join(change.text.split()),
                transcript=change.transcript,
                protein=change.protein,
                coding=change.coding,
            ),
            stated_assembly=StatedVersion(
                name=reference.candidate.name,
                cited_line=reference.candidate.sources[0].text,
            )
            if reference
            else None,
            request_text=target.quote.text,
        )
    except ValidationError as error:
        # The target's own answer, never the run's stop.
        problems = "; ".join(
            f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}"
            for problem in error.errors(include_url=False)
        )
        reason = f"G15 cannot read {change.gene}'s change as a claim: {problems}"
        return TargetPlacement(**quote, answer="not_checked", step="g15", reason=reason)
    found = answer_claim(sources.ncbi, assembly.accession, claim)
    fields = {"step": "g15", "gene_change": found, "version": found.version}
    if found.failure is not None:
        return TargetPlacement(
            **quote,
            answer=None,
            reason=found.failure.detail,
            failure=found.failure,
            **fields,
        )
    if found.outcome == "candidates":
        count = len(found.candidates)
        reason = (
            f"G15 placed {change.gene} ({change.text}) at one codon on the Run's "
            f"{assembly.name}: {count} candidate change{'s' if count > 1 else ''}"
        )
        return TargetPlacement(**quote, answer="placed", reason=reason, **fields)
    answer = found.outcome
    reason = found.reason
    if answer == "not_checked" and target.request.rsid is not None:
        answer, reason = "passed", f"its position is not used: {reason}"
        fields["step"] = "rsid_only"
    return TargetPlacement(**quote, answer=answer, reason=reason, **fields)
