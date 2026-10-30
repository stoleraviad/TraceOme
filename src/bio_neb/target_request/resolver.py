"""One positioned target onto the Run's assembly: the resolver order (#393 Q6, Q8).

The first step that answers wins, and the answer keeps its step:

1. `same_version`: the stated version is the Run's (G15's `resolve_version`) → `kept`.
   Here and in steps 2 and 4 the stated version is the paper's name, then each other name the
   paper writes for it (the record's `reference.aliases`, #411), then each version name the
   general parser (`paper_text.read`) reads inside an alias written as a phrase.
2. `table`: the Run's stated-sequence table has a row for the stated version and contig → a
   flank move from that sequence onto the Run's stored index (the `FlankMover` port, #332;
   development-only, so a Run passes no mover).
3. `database_equivalence`: NCBI's Variation service gives the stated allele, on the stated
   sequence, one equivalent on the Run's sequences → `moved` (`all_equivalent_contextual`),
   for any assembly the service covers.
4. `flank_alignment`: a flank move from the sequence NCBI names onto the Run's stored index
   → `moved`, `ambiguous` or `not_found`.
5. Otherwise `not_checked`, naming why each step did not answer. When the Run passes no mover
   and a flank step would have run, the answer keeps that step's `FlankMoveNeeded`, which
   placement turns into the target's typed failure (owner, #449 Q1 a).

The stated sequence comes from the table row, else from NCBI: the one assembly NCBI's search
for the stated words lists literally under them, any source and every version, every page
(#471 R4 a), then that assembly's sequence report. Placement has no model turn, so a version
written any other way (``hg19``, which NCBI's search does not cover, or ``GRCh37`` for a patch
release) stays unplaced with its reason until #383 brings it to the model. A step that cannot
run says why, and the next step runs. A source contact the failure module gives up on raises
`SourceFailed` to the caller, which records it as the target's failure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
)
from bio_neb.investigation.investigation_sources import (
    AssemblyReport,
    literal_version,
    run_equivalents,
    stated_version_sequence,
)

from .ncbi_sources import NcbiSources, resolve_version
from .paper_text import read
from .records import (
    Equivalence,
    FlankMove,
    FlankMoveRequest,
    RunPosition,
    StatedSequence,
    StatedSequences,
    VersionResolution,
)

MOVE_ANSWERS = {
    "unique": "moved",
    "strand_flipped": "moved",
    "several_places": "ambiguous",
    "not_found": "not_found",
    "ref_changed": "not_checked",
}
"""A move's result as the target's answer."""


@dataclass(frozen=True)
class FlankMoveNeeded:
    """A flank step that would have moved the target, with no mover to run it."""

    step: str
    stated_sequence: StatedSequence
    detail: str


@dataclass(frozen=True)
class Resolved:
    """A positioned target's answer from the step that gave it, with what that step read."""

    answer: str
    step: str
    reason: str
    version: VersionResolution
    matched_name: str | None = None
    stated_sequence: StatedSequence | None = None
    equivalence: Equivalence | None = None
    move: FlankMove | None = None
    position: RunPosition | None = None
    move_needed: FlankMoveNeeded | None = None


class FlankMoveRefused(Exception):
    """The flank move cannot run as asked; the resolver names why and goes on."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class FlankMover(Protocol):
    """The placement port for a flank move (ADR 0020), development-only: its adapters are
    the developer entries' (`tools.reference_index.stored_move.StoredMove`, the bwa move,
    and `devtools.target_placement.NoFlankMover`), and a Run passes no mover (owner, #439
    C3, #449). It gives a stated sequence's FASTA when the mover holds it (else how to get
    it), and the move of one SNV from it onto the Run's stored index. A move it cannot run
    raises `FlankMoveRefused`."""

    def sequence(self, accession: str) -> Path | None: ...

    def missing(self, accession: str) -> str: ...

    def __call__(
        self, snv: FlankMoveRequest, source: Path, run_assembly: str
    ) -> FlankMove: ...


def resolve_position(
    request: InvestigationRequestItem,
    run: AssemblyReport,
    table: StatedSequences | None,
    sources: NcbiSources,
    mover: FlankMover | None,
    aliases: tuple[str, ...],
) -> Resolved:
    """The target's answer by the resolver order. Each step that reads the stated version's
    name tries the paper's name, then each other name the paper writes for it (``aliases``,
    #411 LF2: c3's Nipponbare 1.0 is IRGSP-1.0). `SourceFailed` passes through."""
    stated = request.stated_assembly
    # Each written name, then each version name the general parser reads inside an alias
    # written as a phrase ("IRGSP-1.0, release 52": IRGSP-1.0).
    said = {}
    if stated:
        said[stated.name] = stated.name
        for alias in aliases:
            said.setdefault(alias, f"{alias} (its alias of {stated.name})")
        for alias in aliases:
            for inside in sorted(read(alias).versions):
                said.setdefault(
                    inside, f"{inside} (in its alias {alias} of {stated.name})"
                )
    names = tuple(said)

    versions = [resolve_version(name, run) for name in names]
    matched = next((v for v in versions if v.matched is not None), None)
    version = matched or (versions[0] if versions else resolve_version(None, run))
    position = request.position
    if matched is not None:
        reason = (
            f"the paper's {said[matched.stated]} is the Run's {run.name} "
            f"({run.accession}) by its {matched.matched.replace('_', ' ')}"
        )
        kept = RunPosition(
            contig=position.contig,
            position=position.position,
            ref=position.ref,
            alt=position.alt,
        )
        return Resolved(
            "kept",
            "same_version",
            reason,
            version,
            matched_name=matched.stated,
            position=kept,
        )
    if stated is None:
        reason = "the paper states no genome version for its position"
        return Resolved("not_checked", "none", reason, version)
    misses = [
        f"{' or '.join(names)} is not a name NCBI's report gives the Run's {run.name} "
        f"({run.accession})"
    ]
    needed = None
    rows = [(n, table.row(n, position.contig)) for n in names] if table else []
    name, row = next(((n, r) for n, r in rows if r is not None), (stated.name, None))
    if row is not None:
        sequence = StatedSequence(accession=row.accession, named_by="table")
        moved = _flank_move(request, name, sequence, run, mover, version, "table")
        if isinstance(moved, Resolved):
            return replace(moved, matched_name=name)
        if isinstance(moved, FlankMoveNeeded):
            needed, moved = moved, moved.detail
        misses.append(f"table: {moved}")
    else:
        sequence = None
        named_rows = (
            "no table row" if table else "no table row (the Run gives no table)"
        )
        misses.append(f"table: {named_rows} for {' or '.join(names)} {position.contig}")
    ncbi_named = False
    if sequence is None:
        found = [(n, _ncbi_sequence(n, position.contig, run, sources)) for n in names]
        name, sequence = next(
            ((n, s) for n, s in found if not isinstance(s, str)), (None, None)
        )
        if sequence is None:
            misses += [f"NCBI: {s}" for _, s in found]
            return Resolved("not_checked", "none", "; ".join(misses), version)
        ncbi_named = True
    equivalent = _equivalence(request, sequence, run, sources, version)
    if isinstance(equivalent, Resolved):
        return replace(equivalent, matched_name=name)
    misses.append(f"database equivalence: {equivalent}")
    if ncbi_named:
        moved = _flank_move(
            request, name, sequence, run, mover, version, "flank_alignment"
        )
        if isinstance(moved, Resolved):
            return replace(moved, matched_name=name)
        if isinstance(moved, FlankMoveNeeded):
            needed, moved = moved, moved.detail
        misses.append(f"flank alignment: {moved}")
    return Resolved(
        "not_checked",
        "none",
        "; ".join(misses),
        version,
        stated_sequence=sequence,
        move_needed=needed,
    )


def _ncbi_sequence(
    stated: str, contig: str, run: AssemblyReport, sources: NcbiSources
) -> StatedSequence | str:
    """The stated contig's RefSeq sequence on the stated version as NCBI names it, or why
    NCBI names none (the product's one reading, `stated_version_sequence`). Paper
    placement has no model turn, so its version is the one assembly NCBI's search lists
    literally under the stated words (#471 R4 a) and its sequence the one record that
    literally holds the written contig among its names (owner #467 R4 a); #383 brings
    any other version or contig to the model."""

    def literal(accession: str, name: str):
        held = [
            record for record in sources.sequences(accession) if contig in record.names
        ]
        if len(held) != 1:
            count = "no" if not held else "more than one"
            return (
                f"NCBI's {name} ({accession}) has {count} sequence literally named "
                f"{contig}"
            )
        return held[0]

    if run.taxid is None:
        return f"NCBI's report of the Run's {run.name} names no taxon"
    version = literal_version(stated, run.taxid, sources.genome_reports)
    if isinstance(version, str):
        return version
    named = stated_version_sequence(
        version.assembly_accession, version.assembly_name or stated, literal
    )
    if isinstance(named, str):
        return named
    accession, assembly = named
    return StatedSequence(
        accession=accession, named_by="ncbi", assembly_accession=assembly
    )


def _equivalence(
    request: InvestigationRequestItem,
    sequence: StatedSequence,
    run: AssemblyReport,
    sources: NcbiSources,
    version: VersionResolution,
) -> Resolved | str:
    """Step 3: the stated allele's equivalents on the Run's sequences, or why none
    answers (the product's one reading, `run_equivalents`)."""
    position = request.position
    moved = run_equivalents(position, sequence.accession, run.accession, sources)
    if isinstance(moved, str):
        return moved
    spdi, on_run = moved
    equivalence = Equivalence(spdi=spdi, on_run=on_run)
    fields = {"stated_sequence": sequence, "equivalence": equivalence}
    if len(on_run) > 1:
        reason = (
            f"NCBI's Variation service gives {spdi} {len(on_run)} equivalents on the "
            f"Run's sequences: {', '.join(on_run)}"
        )
        return Resolved("ambiguous", "database_equivalence", reason, version, **fields)
    contig, start, deleted, inserted = on_run[0].split(":")
    moved = RunPosition(
        contig=contig, position=int(start) + 1, ref=deleted, alt=inserted
    )
    reason = (
        f"moved from {request.stated_assembly.name} {position.contig}:"
        f"{position.position} {position.ref}>{position.alt} ({sequence.accession}) to "
        f"the Run's {run.name} {contig}:{moved.position} {deleted}>{inserted} by NCBI's "
        f"Variation service ({spdi} is {on_run[0]})"
    )
    return Resolved(
        "moved", "database_equivalence", reason, version, position=moved, **fields
    )


def _flank_move(
    request: InvestigationRequestItem,
    version_name: str,
    sequence: StatedSequence,
    run: AssemblyReport,
    mover: FlankMover | None,
    version: VersionResolution,
    step: str,
) -> Resolved | FlankMoveNeeded | str:
    """Steps 2 and 4: #332's flank move from the stated sequence onto the Run's stored
    index, or why it cannot run; with no mover, the move this step needs."""
    position = request.position
    if position.ref is None:
        return "a position without its alleles cannot be moved"
    if len(position.ref) != 1 or len(position.alt) != 1:
        return f"only an SNV moves; the paper states {position.ref}>{position.alt}"
    if mover is None:
        snv = _snv(request, version_name, sequence)
        if isinstance(snv, str):
            return snv
        return FlankMoveNeeded(
            step,
            sequence,
            f"{version_name}'s {position.contig} is {sequence.accession}, which only a "
            "flank move places, and a Run holds no flank mover (the move is "
            "development-only)",
        )
    source = mover.sequence(sequence.accession)
    if source is None:
        return (
            f"{version_name}'s {position.contig} is {sequence.accession}, "
            f"{mover.missing(sequence.accession)}"
        )
    snv = _snv(request, version_name, sequence)
    if isinstance(snv, str):
        return snv
    try:
        moved = mover(snv, source, run.accession)
    except FlankMoveRefused as refused:
        return f"the move refused it ({refused.reason}: {refused.detail})"
    fields = {"stated_sequence": sequence, "move": moved}
    answer = MOVE_ANSWERS[moved.result]
    if answer != "moved":
        return Resolved(answer, step, moved.reason, version, **fields)
    place = moved.place
    reason = (
        f"moved from {version_name} {position.contig}:{snv.position} {snv.ref}>{snv.alt} "
        f"({sequence.accession}) to the Run's {run.name} {place.contig}:{place.position} "
        f"{moved.target_ref}>{moved.target_alt}: {moved.reason}"
    )
    at = RunPosition(
        contig=place.contig,
        position=place.position,
        ref=moved.target_ref,
        alt=moved.target_alt,
    )
    return Resolved("moved", step, reason, version, position=at, **fields)


def _snv(
    request: InvestigationRequestItem, version_name: str, sequence: StatedSequence
) -> FlankMoveRequest | str:
    """The target's SNV as the move takes it, or why it is not one."""
    position = request.position
    try:
        return FlankMoveRequest(
            label=f"{version_name} {position.contig}:{position.position}",
            assembly=version_name,
            sequence=sequence.accession,
            position=position.position,
            ref=position.ref,
            alt=position.alt,
        )
    except ValidationError:
        return (
            "only an SNV of A, C, G and T moves; the paper states "
            f"{position.ref}>{position.alt}"
        )
